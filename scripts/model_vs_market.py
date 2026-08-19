#!/usr/bin/env python
"""
Walk-forward comparison of model probabilities against the closing line.

The backtest answers "did this betting rule profit". This answers the prior
question: does the model know anything the market does not? It scores both
against the same outcomes, and — the part that matters — checks who is right
when the two disagree.

A model with real edge is right more often as its disagreement with the
market grows. A model that is merely noisy is right less often, because large
disagreements are where its errors live.

Usage:
    uv run python scripts/model_vs_market.py --sport NFL \
        --start-season 2016 --end-season 2025
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from betting_agent.intelligence.ev import american_to_implied_prob, remove_vig
from betting_agent.models.classification import train_calibrated_classifier
from betting_agent.sports.nfl.market import has_closing_lines
from betting_agent.sports.registry import available_sports, get_sport_config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def brier(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((pred - actual) ** 2))


def log_loss(pred: np.ndarray, actual: np.ndarray) -> float:
    p = np.clip(pred, 1e-9, 1 - 1e-9)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


def collect(sport: str, seasons: list[int], min_train: int) -> pd.DataFrame:
    """Walk forward, recording model and market probabilities per game."""
    config = get_sport_config(sport)
    raw = config.loader_cls().load_schedules(seasons).to_pandas()

    if sport == "NFL":
        from betting_agent.sports.nfl.features import normalise_raw_schedules
        raw = normalise_raw_schedules(raw)

    if not has_closing_lines(raw):
        raise SystemExit(
            f"{sport} schedule data carries no closing lines — nothing to "
            "compare against."
        )

    rows: list[dict] = []
    for i in range(min_train, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        logger.warning("Training on %s, testing %s", train_seasons, test_season)

        train = raw[raw["season"].isin(train_seasons)]
        X_tr, y_tr = config.split_features_targets(config.build_features(train))
        clf, calibrator = train_calibrated_classifier(
            X_tr, y_tr["home_team_wins"].astype(int), verbose=False
        )

        test = raw[raw["season"] == test_season].copy().reset_index(drop=True)
        test["_row_id"] = range(len(test))
        features = config.build_features(test)
        features = features[features["home_score"].notna()]
        row_ids = features["_row_id"].astype(int).tolist()

        X_te = features.drop(
            columns=["home_team_wins", "home_score", "away_score", "_row_id"],
            errors="ignore",
        ).reindex(columns=list(X_tr.columns), fill_value=0)

        raw_probs = clf.predict(xgb.DMatrix(X_te, feature_names=list(X_te.columns)))
        probs = np.clip(calibrator.predict(raw_probs), 0.05, 0.95)

        source = test.set_index("_row_id")
        for j, row_id in enumerate(row_ids):
            game = source.loc[row_id]
            if pd.isna(game.get("home_moneyline")) or pd.isna(game.get("away_moneyline")):
                continue
            fair_home, _ = remove_vig(
                american_to_implied_prob(game["home_moneyline"]),
                american_to_implied_prob(game["away_moneyline"]),
            )
            rows.append({
                "season": test_season,
                "model": float(probs[j]),
                "market": fair_home,
                "home_won": int(game["home_score"] > game["away_score"]),
            })

    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    actual = df["home_won"].to_numpy()
    print(f"\nGames compared: {len(df)}\n")

    print(f"{'':18}{'Brier':>9}{'LogLoss':>10}{'Accuracy':>10}")
    for name, col in (("model", df["model"]), ("market (close)", df["market"])):
        pred = col.to_numpy()
        acc = ((pred > 0.5).astype(int) == actual).mean()
        print(f"{name:18}{brier(pred, actual):>9.4f}"
              f"{log_loss(pred, actual):>10.4f}{acc:>9.1%}")
    print("\n  Lower Brier and log loss are better.")

    print(f"\nCorrelation, model vs market: {df['model'].corr(df['market']):.3f}")
    print(f"Mean |model - market|:        {(df['model'] - df['market']).abs().mean():.3f}")

    print("\nWhen they pick opposite sides, who is right?")
    print(f"{'disagreement':>14}{'games':>8}{'model right':>13}{'market right':>14}")
    gap = (df["model"] - df["market"]).abs()
    for lo, hi in ((0.02, 0.05), (0.05, 0.10), (0.10, 1.01)):
        band = df[gap.between(lo, hi)]
        opposed = band[
            (band["model"] > 0.5).astype(int) != (band["market"] > 0.5).astype(int)
        ]
        label = f"{lo:.0%}-{min(hi, 1.0):.0%}".rjust(14)
        if opposed.empty:
            print(f"{label}{len(band):>8}   (never opposed)")
            continue
        model_right = (
            (opposed["model"] > 0.5).astype(int) == opposed["home_won"]
        ).mean()
        print(f"{label}{len(opposed):>8}{model_right:>12.1%}{1 - model_right:>14.1%}")

    print(
        "\n  Edge looks like the model getting MORE right as disagreement grows.\n"
        "  The opposite pattern means large disagreements are where the model's\n"
        "  errors live, and no threshold tuning will turn them into profit."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs closing line")
    parser.add_argument(
        "--sport", type=str, default="NFL",
        help=f"Sport ({', '.join(available_sports())})",
    )
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument(
        "--min-train-seasons", type=int, default=4,
        help="Seasons to train on before the first test season",
    )
    args = parser.parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    if len(seasons) <= args.min_train_seasons:
        raise SystemExit(
            f"Need more than {args.min_train_seasons} seasons for walk-forward."
        )

    df = collect(args.sport.upper(), seasons, args.min_train_seasons)
    if df.empty:
        raise SystemExit("No games with closing lines to compare.")
    report(df)


if __name__ == "__main__":
    main()
