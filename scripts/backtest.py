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
import math
import pickle
import random
from pathlib import Path

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.ev import (
    american_to_implied_prob,
    calculate_edge,
    calculate_total_edge,
)
from betting_agent.intelligence.kelly import kelly_fraction, scaled_kelly
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
        edge = calculate_edge(bet_prob, bet_odds)

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
    args = parser.parse_args()

    sport = args.sport.upper()
    config = get_sport_config(sport)

    all_seasons = list(range(args.start_season, args.end_season + 1))
    if len(all_seasons) < args.min_train_seasons + 1:
        print(f"Need at least {args.min_train_seasons + 1} seasons for walk-forward.")
        return

    print(f"\n{'!'*60}")
    print("  WARNING: Results use synthetic odds — estimates model skill,")
    print("  not real profitability against actual sportsbook lines.")
    print(f"{'!'*60}")
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

        # Save to temp location and load via engine
        temp_dir = Path(settings.saved_models_dir) / f"{sport}_backtest_tmp"
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
        season_pnl = bankroll - season_start_bankroll
        cumulative_pnl += season_pnl
        season_roi = (season_pnl / season_start_bankroll * 100.0) if season_start_bankroll else 0.0
        win_rate = (season_wins / season_bets * 100.0) if season_bets else 0.0

        season_results.append({
            "season": test_season,
            "bets": season_bets,
            "wins": season_wins,
            "win_rate_pct": round(win_rate, 1),
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

    if not all_logs:
        print("\nNo bets were placed.")
        return

    log_df = pd.DataFrame(all_logs)
    csv_path = Path("backtest_results.csv")
    season_df = pd.DataFrame(season_results)
    season_df.to_csv(csv_path, index=False)

    # Summary
    total_bets = len(log_df)
    total_wins = (log_df["result"] == "win").sum()
    total_wagered = log_df["amount"].sum()
    final_roi = (cumulative_pnl / total_wagered * 100.0) if total_wagered else 0.0
    win_rate = total_wins / total_bets * 100.0 if total_bets else 0.0

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
    print(f"  Total bets:        {total_bets}")
    print(f"  Total wagered:     ${total_wagered:,.2f}")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  ROI (on wagered):  {final_roi:+.2f}%")
    print(f"  Max drawdown:      {max_dd:.2f}%")
    print(f"  Results saved to:  {csv_path}")

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
