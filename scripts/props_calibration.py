#!/usr/bin/env python
"""
Walk-forward calibration check for the prop projection models.

There are no free historical prop lines, so this validates the projections
themselves: for every player-week in the evaluation seasons, project the
distribution using only earlier games, then test it against what actually
happened, two ways:

1. Reliability at pseudo-lines: quote lines around the projection (the
   neighbourhood real books quote in), bin the model's P(over) and compare
   with how often the over actually hit. A calibrated model's bins sit on
   the diagonal.
2. Central interval coverage: how often the actual lands inside the model's
   50% and 80% intervals.

Both must pass before paper-trading real lines means anything.

Usage:
    uv run python scripts/props_calibration.py --train-seasons 2020 2021 2022 \
        --eval-seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    ReceivingPropsModel,
    build_receiving_history,
    load_player_stats,
    pseudo_lines,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate_market(market: str, history: pd.DataFrame,
                    train_seasons: list[int], eval_seasons: list[int]) -> pd.DataFrame:
    model = ReceivingPropsModel(market)
    train = history[history["season"].isin(train_seasons)]
    model.fit(train)
    # Tune interval width on the later training seasons only — never on eval.
    model.tune_dispersion(sorted(train_seasons)[-2:])
    model.extend_history(history[history["season"].isin(eval_seasons)])

    rows = []
    eval_rows = history[history["season"].isin(eval_seasons)]
    for (season, week), games in eval_rows.groupby(["season", "week"]):
        for _, g in games.iterrows():
            proj = model.project(
                g["player_key"], int(season), int(week),
                opponent=g["opponent_team"],
            )
            if proj is None:
                continue
            actual = float(g[model.stat_col])

            lines = pseudo_lines(market, proj.mean)

            entry = {
                "season": season, "week": week, "player": g["player_key"],
                "mean": proj.mean, "actual": actual,
                "in50": proj._dist.ppf(0.25) <= actual <= proj._dist.ppf(0.75),
                "in80": proj._dist.ppf(0.10) <= actual <= proj._dist.ppf(0.90),
            }
            for i, line in enumerate(lines):
                if line <= 0:
                    continue
                entry[f"p_over_{i}"] = proj.prob_over(line)
                entry[f"hit_over_{i}"] = float(actual > line)
            rows.append(entry)
    return pd.DataFrame(rows)


def report(market: str, df: pd.DataFrame) -> None:
    print(f"\n=== {market} — {len(df)} projected player-weeks ===")
    mae = (df["mean"] - df["actual"]).abs().mean()
    bias = (df["mean"] - df["actual"]).mean()
    print(f"Projection MAE {mae:.2f}, bias {bias:+.2f} "
          f"(positive = projections run high)")
    print(f"50% interval coverage: {df['in50'].mean():.1%}  (want ~50%)")
    print(f"80% interval coverage: {df['in80'].mean():.1%}  (want ~80%)")

    # Pool all pseudo-lines into one reliability table.
    p, hit = [], []
    for i in range(6):
        pc, hc = f"p_over_{i}", f"hit_over_{i}"
        if pc in df.columns:
            mask = df[pc].notna()
            p.append(df.loc[mask, pc])
            hit.append(df.loc[mask, hc])
    pooled = pd.DataFrame({"p": pd.concat(p), "hit": pd.concat(hit)})
    pooled["bin"] = pd.cut(pooled["p"], np.arange(0, 1.05, 0.1))
    print(f"\n{'P(over) bin':>14}{'n':>8}{'predicted':>11}{'actual':>9}{'gap':>8}")
    for interval, grp in pooled.groupby("bin", observed=True):
        if len(grp) < 30:
            continue
        pred, act = grp["p"].mean(), grp["hit"].mean()
        print(f"{str(interval):>14}{len(grp):>8}{pred:>10.1%}{act:>9.1%}"
              f"{act - pred:>+8.1%}")
    overall_brier = ((pooled["p"] - pooled["hit"]) ** 2).mean()
    base = pooled["hit"].mean()
    naive_brier = ((base - pooled["hit"]) ** 2).mean()
    print(f"\nBrier {overall_brier:.4f} vs naive-constant {naive_brier:.4f} "
          f"({'beats' if overall_brier < naive_brier else 'LOSES TO'} the naive baseline)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prop projection calibration")
    parser.add_argument("--train-seasons", type=int, nargs="+", required=True)
    parser.add_argument("--eval-seasons", type=int, nargs="+", required=True)
    parser.add_argument("--markets", type=str, nargs="+", default=list(MODELED_MARKETS))
    args = parser.parse_args()

    seasons = sorted(set(args.train_seasons) | set(args.eval_seasons))
    print(f"Loading player stats for {seasons}...")
    history = build_receiving_history(load_player_stats(seasons))
    if history.empty:
        raise SystemExit("No player stats loaded.")

    for market in args.markets:
        df = evaluate_market(market, history, args.train_seasons, args.eval_seasons)
        if df.empty:
            print(f"{market}: nothing to evaluate")
            continue
        report(market, df)


if __name__ == "__main__":
    main()
