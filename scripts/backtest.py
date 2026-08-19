#!/usr/bin/env python
"""
Entry point: walk-forward backtesting over historical seasons.

Usage:
    uv run python scripts/backtest.py --sport NFL --start-season 2018 --end-season 2023

Walk-forward approach:
  - For each test season N, train on seasons [start..N-1]
  - Simulate the strategy on season N
  - Track bankroll, ROI, win rate, max drawdown per season
  - Output equity curve PNG and season-by-season CSV
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import joblib

import matplotlib.pyplot as plt
import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.ev import (
    american_to_implied_prob,
    calculate_edge_fair,
    calculate_spread_edge,
    calculate_total_edge,
    remove_vig,
)
from betting_agent.intelligence.kelly import recommended_bet, scaled_kelly
from betting_agent.intelligence.picks import _passes_guardrails
from betting_agent.sports.nfl import market
from betting_agent.models.classification import train_calibrated_classifier, save_classifier
from betting_agent.models.engine import PredictionEngine
from betting_agent.models.regression import (
    train_final_regressors,
    train_regressors,
    save_regressors,
    compute_residual_sigma,
)
from betting_agent.sports.registry import get_sport_config, available_sports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Vig: sportsbooks typically charge ~4.5% overround on a two-way market
BOOK_VIG = 0.045


def _prob_to_american(prob: float) -> int:
    """Convert a probability to American odds with vig baked in."""
    if prob <= 0.0 or prob >= 1.0:
        prob = max(0.01, min(0.99, prob))
    # Add half the vig to each side (simulate bookmaker margin)
    vigged = prob + BOOK_VIG / 2.0
    vigged = min(vigged, 0.98)
    if vigged >= 0.5:
        return int(-100.0 * vigged / (1.0 - vigged))
    else:
        return int(100.0 * (1.0 - vigged) / vigged)


def _generate_market_odds(
    model_prob: float = 0.57,
    noise_std: float = 0.06,
) -> tuple[int, int]:
    """
    Generate realistic synthetic moneyline odds for a game.

    Uses the model's own prediction as the market center (with noise + vig)
    to create realistic, game-specific edges rather than phantom edges
    against pure noise. noise_std controls market disagreement.
    Returns (home_odds, away_odds).
    """
    market_home = model_prob + random.gauss(0, noise_std)
    market_home = max(0.15, min(0.85, market_home))
    market_away = 1.0 - market_home

    home_odds = _prob_to_american(market_home)
    away_odds = _prob_to_american(market_away)
    return home_odds, away_odds


def _generate_ou_line(
    train_totals: pd.Series,
    hist_avg_total: float,
    total_stdev: float,
) -> float:
    """
    Generate an O/U line from training data season averages + noise.
    No access to the actual game's score.
    """
    if len(train_totals) > 0:
        avg = float(train_totals.mean())
        std = float(train_totals.std()) if len(train_totals) > 1 else total_stdev
    else:
        avg = hist_avg_total
        std = total_stdev

    # Round to nearest 0.5 like real sportsbooks
    line = avg + random.gauss(0, std * 0.3)
    return round(line * 2) / 2.0


def _simulate_season_real_lines(
    test_df: pd.DataFrame,
    engine: PredictionEngine,
    feature_names: list[str],
    bankroll: float,
    build_features,
    flat_stake: float | None = None,
    total_sigma: float = 14.0,
    margin_sigma: float = 14.0,
) -> tuple[list[dict], float]:
    """
    Simulate one season against the real closing lines on the schedule rows.

    Every bet is priced at the number the market closed at, which is the
    hardest test available: no early-line advantage, no shopping, no synthetic
    market built from the model's own opinion. Uses the same edge maths,
    guardrails, and Kelly sizing as live pick generation, so what comes out is
    a measurement of the strategy rather than of a simulation.
    """
    test_df = test_df.copy().reset_index(drop=True)
    test_df["_bt_row_id"] = range(len(test_df))

    features = build_features(test_df)
    complete = features[
        features["home_score"].notna() & features["away_score"].notna()
    ].copy()
    if complete.empty:
        return [], bankroll

    row_ids = complete["_bt_row_id"].astype(int).tolist()
    X = complete.drop(
        columns=["home_team_wins", "home_score", "away_score", "_bt_row_id"],
        errors="ignore",
    )
    X = X.reindex(columns=feature_names, fill_value=0)

    win_probs = engine.predict_win_prob(X)
    home_scores, away_scores = engine.predict_scores(X)

    lines = test_df.set_index("_bt_row_id")
    bet_log: list[dict] = []

    def _place(bet: dict) -> None:
        """Size, settle, and record one bet."""
        nonlocal bankroll
        if flat_stake is not None:
            # A bankroll that cannot cover the stake is busted; betting past
            # that point reports losses no real bettor could have taken.
            if bankroll < flat_stake:
                return
            amount = flat_stake
            kf = 0.0
        else:
            kf, amount = recommended_bet(
                bet["model_prob"], bet["odds"], bet["edge"], bankroll
            )
        if amount <= 0:
            return
        pnl = market.payout(amount, bet["odds"], bet["result"])
        bankroll += pnl
        bet_log.append({**bet, "kelly": kf, "amount": amount,
                        "pnl": pnl, "bankroll": bankroll})

    for i, row_id in enumerate(row_ids):
        game = lines.loc[row_id]
        win_prob = float(win_probs[i])
        pred_margin = float(home_scores[i] - away_scores[i])
        pred_total = float(home_scores[i] + away_scores[i])
        home_actual = float(game["home_score"])
        away_actual = float(game["away_score"])

        base = {
            "game_id": game.get("game_id") or game.get("external_id"),
            "game_date": game.get("game_date"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
        }

        # ---- Moneyline ----
        ml_home = game.get("home_moneyline")
        ml_away = game.get("away_moneyline")
        if pd.notna(ml_home) and pd.notna(ml_away):
            for side, prob, odds in (
                ("home", win_prob, ml_home),
                ("away", 1.0 - win_prob, ml_away),
            ):
                edge = calculate_edge_fair(
                    prob, ml_home, ml_away, pick_home=(side == "home")
                )
                if edge < settings.min_edge_pct:
                    continue
                if not _passes_guardrails(edge, int(odds), prob, "moneyline"):
                    continue
                _place({
                    **base, "bet_type": "moneyline", "side": side, "line": None,
                    "odds": int(odds), "model_prob": prob, "edge": edge,
                    "result": market.grade_moneyline(home_actual, away_actual, side),
                })

        # ---- Spread ----
        spread_line = game.get("spread_line")
        spread_home_odds = game.get("home_spread_odds")
        spread_away_odds = game.get("away_spread_odds")
        if pd.notna(spread_line) and pd.notna(spread_home_odds) and pd.notna(spread_away_odds):
            cover_prob, edge = calculate_spread_edge(
                pred_margin, float(spread_line), spread_home_odds,
                away_odds=spread_away_odds, sigma=margin_sigma,
            )
            side, prob, odds, side_edge = (
                ("home", cover_prob, spread_home_odds, edge) if edge >= 0
                else ("away", 1.0 - cover_prob, spread_away_odds,
                      _away_edge(1.0 - cover_prob, spread_away_odds, spread_home_odds))
            )
            if side_edge >= settings.min_edge_pct and _passes_guardrails(
                side_edge, int(odds), prob, "spread"
            ):
                _place({
                    **base, "bet_type": "spread", "side": side,
                    "line": float(spread_line), "odds": int(odds),
                    "model_prob": prob, "edge": side_edge,
                    "result": market.grade_spread(
                        home_actual, away_actual, float(spread_line), side
                    ),
                })

        # ---- Total ----
        total_line = game.get("total_line")
        over_odds = game.get("over_odds")
        under_odds = game.get("under_odds")
        if pd.notna(total_line) and pd.notna(over_odds) and pd.notna(under_odds):
            for side, odds, other in (
                ("over", over_odds, under_odds),
                ("under", under_odds, over_odds),
            ):
                prob, edge = calculate_total_edge(
                    pred_total, float(total_line), odds, side,
                    other_odds=other, sigma=total_sigma,
                )
                if edge < settings.min_edge_pct:
                    continue
                if not _passes_guardrails(edge, int(odds), prob, "total"):
                    continue
                _place({
                    **base, "bet_type": "total", "side": side,
                    "line": float(total_line), "odds": int(odds),
                    "model_prob": prob, "edge": edge,
                    "result": market.grade_total(
                        home_actual, away_actual, float(total_line), side
                    ),
                })

    return bet_log, bankroll


def _away_edge(away_prob: float, away_odds, home_odds) -> float:
    """Edge on the away spread side, de-vigged within the same market."""
    home_imp = american_to_implied_prob(home_odds)
    away_imp = american_to_implied_prob(away_odds)
    _, fair_away = remove_vig(home_imp, away_imp)
    return away_prob - fair_away


def _simulate_season(
    test_df: pd.DataFrame,
    engine: PredictionEngine,
    feature_names: list[str],
    bankroll: float,
    train_totals: pd.Series,
    build_features,
    hist_avg_total: float,
    total_stdev: float,
    flat_stake: float | None = None,
    total_sigma: float = 14.0,
) -> tuple[list[dict], float]:
    """
    Simulate one season. Returns (bet_log, final_bankroll).

    flat_stake: if set, use flat dollar amount per bet instead of Kelly sizing.
                If None, uses Kelly sizing with no min-bet floor.
    """
    test_features = build_features(test_df)
    # Only rows with actual scores (for grading)
    complete = test_features[
        test_features["home_score"].notna() & test_features["away_score"].notna()
    ].copy()

    if complete.empty:
        return [], bankroll

    # Extract targets before dropping
    actual_home = complete["home_score"].values
    actual_away = complete["away_score"].values

    # Feature columns only
    X = complete.drop(
        columns=["home_team_wins", "home_score", "away_score"], errors="ignore"
    )
    X = X.reindex(columns=feature_names, fill_value=0)

    # Predict
    win_probs = engine.predict_win_prob(X)
    home_scores, away_scores = engine.predict_scores(X)

    bet_log = []
    for i in range(len(X)):
        win_prob = float(win_probs[i])
        h_pred = float(home_scores[i])
        a_pred = float(away_scores[i])
        pred_total = h_pred + a_pred

        h_actual = float(actual_home[i])
        a_actual = float(actual_away[i])
        actual_total = h_actual + a_actual

        # ---- Moneyline ----
        home_ml, away_ml = _generate_market_odds(model_prob=win_prob)

        bet_side = "home" if win_prob >= 0.5 else "away"
        bet_prob = win_prob if bet_side == "home" else 1.0 - win_prob
        bet_odds = home_ml if bet_side == "home" else away_ml
        # Vig-removed fair edge, matching how live picks compute it — the
        # vig-included calculate_edge() would size Kelly off a different
        # number than the live pipeline bets on.
        edge = calculate_edge_fair(
            bet_prob, home_ml, away_ml, pick_home=(bet_side == "home")
        )

        if edge >= settings.min_edge_pct:
            kf = scaled_kelly(bet_prob, bet_odds, edge)
            if flat_stake is not None:
                bet_amount = flat_stake
            else:
                bet_amount = kf * bankroll
                bet_amount = min(bet_amount, settings.max_bet_pct * bankroll)

            if bet_amount > 0 and kf > 0:
                won = (h_actual > a_actual) if bet_side == "home" else (a_actual > h_actual)
                result = "win" if won else "loss"
                if result == "win":
                    payout = bet_amount * (
                        bet_odds / 100.0 if bet_odds > 0
                        else 100.0 / abs(bet_odds)
                    )
                    bankroll += payout
                else:
                    bankroll -= bet_amount

                bet_log.append({
                    "bet_type": "moneyline",
                    "side": bet_side,
                    "result": result,
                    "edge": edge,
                    "kelly": kf,
                    "amount": bet_amount,
                    "odds": bet_odds,
                    "model_prob": bet_prob,
                    "bankroll": bankroll,
                })

        # ---- Over/Under ----
        ou_line = _generate_ou_line(train_totals, hist_avg_total, total_stdev)
        ou_home_odds, ou_away_odds = _generate_market_odds(model_prob=0.50, noise_std=0.02)

        if pred_total > ou_line:
            side = "over"
            over_prob, o_edge = calculate_total_edge(
                pred_total, ou_line, ou_home_odds, "over",
                other_odds=ou_away_odds, sigma=total_sigma,
            )
        else:
            side = "under"
            over_prob, o_edge = calculate_total_edge(
                pred_total, ou_line, ou_away_odds, "under",
                other_odds=ou_home_odds, sigma=total_sigma,
            )

        ou_bet_odds = ou_home_odds if side == "over" else ou_away_odds
        cover_prob = over_prob

        if o_edge >= settings.min_edge_pct:
            kf = scaled_kelly(cover_prob, ou_bet_odds, o_edge)
            if flat_stake is not None:
                bet_amount = flat_stake
            else:
                bet_amount = kf * bankroll
                bet_amount = min(bet_amount, settings.max_bet_pct * bankroll)

            if bet_amount > 0 and kf > 0:
                won = (actual_total > ou_line) if side == "over" else (actual_total < ou_line)
                result = "win" if won else "loss"
                if result == "win":
                    payout = bet_amount * (
                        ou_bet_odds / 100.0 if ou_bet_odds > 0
                        else 100.0 / abs(ou_bet_odds)
                    )
                    bankroll += payout
                else:
                    bankroll -= bet_amount

                bet_log.append({
                    "bet_type": f"total_{side}",
                    "side": side,
                    "result": result,
                    "edge": o_edge,
                    "kelly": kf,
                    "amount": bet_amount,
                    "odds": ou_bet_odds,
                    "model_prob": cover_prob,
                    "ou_line": ou_line,
                    "bankroll": bankroll,
                })

    return bet_log, bankroll


def _print_market_breakdown(log_df: pd.DataFrame) -> None:
    """
    Per-market results against the real closing lines, with the win rate each
    market needed just to break even at the prices paid. The gap between
    actual and required is the whole question.
    """
    print(f"\n  {'-'*56}")
    print("  By market (vs. closing lines)")
    print(f"  {'market':<12}{'bets':>6}{'win%':>8}{'needed':>9}{'ROI':>9}")
    for bet_type, group in log_df.groupby("bet_type"):
        decided = group[group["result"] != "push"]
        if decided.empty:
            continue
        wr = (decided["result"] == "win").mean() * 100.0
        needed = decided["odds"].map(market.breakeven_rate).mean() * 100.0
        wagered = group["amount"].sum()
        roi = (group["pnl"].sum() / wagered * 100.0) if wagered else 0.0
        print(f"  {bet_type:<12}{len(group):>6}{wr:>7.1f}%{needed:>8.1f}%{roi:>+8.2f}%")

    edges = log_df["edge"]
    print(
        f"\n  Claimed edge: median {edges.median()*100:.1f}%, "
        f"max {edges.max()*100:.1f}% over {len(log_df)} bets"
    )
    print(
        "  A model that cannot beat the close is not necessarily useless —\n"
        "  but it should not be bet at the close."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument(
        "--sport",
        type=str,
        default="NFL",
        help=f"Sport ({', '.join(available_sports())})",
    )
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--bankroll", type=float, default=settings.starting_bankroll)
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=3,
        help="Minimum training seasons before first test season",
    )
    parser.add_argument(
        "--flat-stake",
        type=float,
        default=None,
        help="Use flat dollar stake per bet instead of Kelly sizing",
    )
    parser.add_argument(
        "--reset-bankroll",
        action="store_true",
        help="Reset bankroll to starting amount at the beginning of each season",
    )
    parser.add_argument(
        "--synthetic-odds",
        action="store_true",
        help=(
            "Force the synthetic-odds simulation even when real closing lines "
            "are available. Measures model skill, not profitability."
        ),
    )
    args = parser.parse_args()

    sport = args.sport.upper()
    config = get_sport_config(sport)

    all_seasons = list(range(args.start_season, args.end_season + 1))
    if len(all_seasons) < args.min_train_seasons + 1:
        print(f"Need at least {args.min_train_seasons + 1} seasons for walk-forward.")
        return

    print(f"\nWalk-forward backtest ({sport}): {args.start_season}-{args.end_season}")
    print(f"Starting bankroll: ${args.bankroll:,.2f}")
    if args.flat_stake:
        print(f"Flat stake: ${args.flat_stake:,.2f}")
    if args.reset_bankroll:
        print("Bankroll resets each season")
    print()

    # Load all raw data once
    logger.info("Loading all %s schedule data...", sport)
    loader = config.loader_cls()
    all_raw = loader.load_schedules(all_seasons).to_pandas()

    # Ensure game_date is parsed for season filtering
    if "game_date" not in all_raw.columns and "gameday" in all_raw.columns:
        all_raw = all_raw.rename(columns={"gameday": "game_date"})
    all_raw["game_date"] = pd.to_datetime(all_raw["game_date"])

    # For NFL, apply normalisation (NFL features expect normalised input)
    if sport == "NFL":
        from betting_agent.sports.nfl.features import normalise_raw_schedules
        all_raw = normalise_raw_schedules(all_raw)
        all_raw["game_date"] = pd.to_datetime(all_raw["game_date"])

    use_real_lines = market.has_closing_lines(all_raw) and not args.synthetic_odds
    if use_real_lines:
        coverage = market.line_coverage(all_raw)
        print("  Pricing against REAL closing lines from the schedule data.")
        print(
            "  Line coverage — moneyline {moneyline:.0%}, spread {spread:.0%}, "
            "total {total:.0%}".format(**coverage)
        )
        print(
            "  Betting at the close is the hardest test there is: no early-line\n"
            "  advantage and no shopping. Clearing it is real edge; falling a\n"
            "  few percent short of break-even is the expected outcome."
        )
    else:
        print(f"\n{'!'*60}")
        print("  WARNING: Results use synthetic odds — estimates model skill,")
        print("  not real profitability against actual sportsbook lines.")
        if args.synthetic_odds and market.has_closing_lines(all_raw):
            print("  (Real closing lines ARE available — drop --synthetic-odds.)")
        print(f"{'!'*60}")
    print()

    fold_models_dir = tempfile.mkdtemp(prefix="betting_agent_backtest_")

    bankroll = args.bankroll
    starting_bankroll = bankroll
    all_logs: list[dict] = []
    season_results: list[dict] = []
    cumulative_pnl = 0.0

    for test_idx in range(args.min_train_seasons, len(all_seasons)):
        train_seasons = all_seasons[:test_idx]
        test_season = all_seasons[test_idx]

        logger.info("Training on %s | Testing on %s", train_seasons, test_season)

        # Train
        train_raw = all_raw[all_raw["season"].isin(train_seasons)].copy()
        train_features = config.build_features(train_raw)
        X_tr, y_tr = config.split_features_targets(train_features)

        if len(X_tr) < 100:
            logger.warning("Insufficient training data for season %d", test_season)
            continue

        feature_names = list(X_tr.columns)

        clf, calibrator = train_calibrated_classifier(X_tr, y_tr["home_team_wins"].astype(int), verbose=False)
        y_home_tr = y_tr["home_score"].astype(float)
        y_away_tr = y_tr["away_score"].astype(float)
        home_reg, away_reg = train_final_regressors(X_tr, y_home_tr, y_away_tr)

        # Compute empirical sigma from regression residuals on this fold
        sigma_split = int(len(X_tr) * 0.8)
        X_sig_tr, X_sig_te = X_tr.iloc[:sigma_split], X_tr.iloc[sigma_split:]
        yh_sig_tr, yh_sig_te = y_home_tr.iloc[:sigma_split], y_home_tr.iloc[sigma_split:]
        ya_sig_tr, ya_sig_te = y_away_tr.iloc[:sigma_split], y_away_tr.iloc[sigma_split:]
        h_sig_model, a_sig_model = train_regressors(
            X_sig_tr, yh_sig_tr, ya_sig_tr, eval_split=0.2, verbose=False,
        )
        fold_sigma = compute_residual_sigma(h_sig_model, a_sig_model, X_sig_te, yh_sig_te, ya_sig_te)
        fold_total_sigma = fold_sigma["total_sigma"]
        logger.info("  Fold sigma — total: %.2f, margin: %.2f", fold_sigma["total_sigma"], fold_sigma["margin_sigma"])

        # Compute historical total points for O/U line generation
        train_complete = train_raw[
            train_raw["home_score"].notna() & train_raw["away_score"].notna()
        ]
        train_totals = (
            train_complete["home_score"].astype(float)
            + train_complete["away_score"].astype(float)
        )

        # Save fold models to a scratch dir and load them through the engine.
        # Deliberately outside saved_models/ so a backtest can never be
        # mistaken for a trained model.
        temp_dir = Path(fold_models_dir) / f"{sport}_fold_{test_season}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        save_classifier(clf, temp_dir / "classifier.json")
        save_regressors(home_reg, away_reg, temp_dir)
        with open(temp_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        joblib.dump(calibrator, temp_dir / "calibrator.joblib")

        engine = PredictionEngine(sport=f"{sport}_backtest_tmp", models_dir=temp_dir.parent)
        engine.models_dir = temp_dir
        engine.load()

        # Reset bankroll per season if requested
        if args.reset_bankroll:
            bankroll = starting_bankroll

        # Test
        test_raw = all_raw[all_raw["season"] == test_season].copy()
        season_start_bankroll = bankroll
        if use_real_lines:
            logs, bankroll = _simulate_season_real_lines(
                test_raw, engine, feature_names, bankroll,
                build_features=config.build_features,
                flat_stake=args.flat_stake,
                total_sigma=fold_total_sigma,
                margin_sigma=fold_sigma["margin_sigma"],
            )
        else:
            logs, bankroll = _simulate_season(
                test_raw, engine, feature_names, bankroll,
                train_totals=train_totals,
                build_features=config.build_features,
                hist_avg_total=config.hist_avg_total,
                total_stdev=config.total_stdev,
                flat_stake=args.flat_stake,
                total_sigma=fold_total_sigma,
            )

        for log in logs:
            log["season"] = test_season
        all_logs.extend(logs)

        season_bets = len(logs)
        season_wins = sum(1 for entry in logs if entry["result"] == "win")
        season_pushes = sum(1 for entry in logs if entry["result"] == "push")
        season_pnl = bankroll - season_start_bankroll
        cumulative_pnl += season_pnl
        # ROI on money risked, not on opening bankroll — with flat stakes the
        # latter reports a profit on a losing season once the bankroll dips.
        season_wagered = sum(entry["amount"] for entry in logs)
        season_roi = (season_pnl / season_wagered * 100.0) if season_wagered else 0.0
        decided = season_bets - season_pushes
        win_rate = (season_wins / decided * 100.0) if decided else 0.0

        season_results.append({
            "season": test_season,
            "bets": season_bets,
            "wins": season_wins,
            "pushes": season_pushes,
            "win_rate_pct": round(win_rate, 1),
            "wagered": round(season_wagered, 2),
            "pnl": round(season_pnl, 2),
            "roi_pct": round(season_roi, 2),
            "end_bankroll": round(bankroll, 2),
        })

        print(
            f"  {test_season}: {season_bets} bets  "
            f"WR={win_rate:.1f}%  "
            f"P&L=${season_pnl:+,.2f}  "
            f"ROI={season_roi:+.1f}%"
        )

    shutil.rmtree(fold_models_dir, ignore_errors=True)

    if not all_logs:
        print("\nNo bets were placed.")
        return

    log_df = pd.DataFrame(all_logs)
    csv_path = Path("backtest_results.csv")
    season_df = pd.DataFrame(season_results)
    season_df.to_csv(csv_path, index=False)
    bets_path = Path("backtest_bets.csv")
    log_df.to_csv(bets_path, index=False)

    # Summary
    total_bets = len(log_df)
    total_wins = int((log_df["result"] == "win").sum())
    total_pushes = int((log_df["result"] == "push").sum())
    total_decided = total_bets - total_pushes
    total_wagered = log_df["amount"].sum()
    final_roi = (cumulative_pnl / total_wagered * 100.0) if total_wagered else 0.0
    win_rate = total_wins / total_decided * 100.0 if total_decided else 0.0

    # Max drawdown
    equity = log_df["bankroll"]
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min() * 100.0

    print(f"\n{'='*60}")
    print(f"  BACKTEST SUMMARY ({sport} {args.start_season}-{args.end_season})")
    print(f"{'='*60}")
    print(f"  Starting bankroll: ${starting_bankroll:,.2f}")
    print(f"  Final bankroll:    ${bankroll:,.2f}")
    print(f"  Cumulative P&L:    ${cumulative_pnl:+,.2f}")
    print(f"  Total bets:        {total_bets}  ({total_pushes} pushes)")
    print(f"  Total wagered:     ${total_wagered:,.2f}")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  ROI (on wagered):  {final_roi:+.2f}%")
    print(f"  Max drawdown:      {max_dd:.2f}%")

    if use_real_lines:
        _print_market_breakdown(log_df)

    print(f"  Results saved to:  {csv_path}")
    print(f"  Per-bet log:       {bets_path}")

    # Equity curve
    plt.figure(figsize=(14, 6))
    plt.plot(equity.values, linewidth=1.5, label="Equity Curve")
    plt.axhline(y=starting_bankroll, color="red", linestyle="--", alpha=0.6, label="Starting bankroll")
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll ($)")
    plt.title(f"Walk-Forward Backtest Equity Curve ({sport} {args.start_season}-{args.end_season})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_path = Path("equity_curve.png")
    plt.savefig(img_path, dpi=150)
    print(f"  Equity curve:      {img_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
