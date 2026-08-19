#!/usr/bin/env python
"""
Walk-forward diagnostic for prop PICKS (not just projections).

props_calibration.py answers "is P(over) calibrated?". This answers the
decision-relevant question: if we take every pick clearing an edge floor
against a book-proxy line, what actually happens — realized hit rate and ROI
by side, by claimed-edge bucket, and how concentrated the exposure is.

This is what set the per-market edge floors and the per-player dedup rule in
props.py. Re-run it after any change to the projection or calibration code:
the floors are only right while the numbers underneath them hold.

Lines here are trailing-median proxies, softer than a real book, so absolute
ROI is not a forecast. The comparisons between buckets and policies are what
the output is for.

Usage:
    uv run python scripts/props_diagnostic.py [out.csv]
"""
from __future__ import annotations

import bisect
import sys
from collections import defaultdict

import pandas as pd

from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    ReceivingPropsModel,
    book_proxy_line,
    build_receiving_history,
    load_player_stats,
)

TRAIN = [2020, 2021, 2022, 2023]
EVAL = [2024, 2025]
EDGE_FLOOR = 0.05   # report floor: deliberately below the production floors
PRICE_DEVIG = 0.5   # -110/-110 both sides de-vigs to an even-money market


def run(market: str, history: pd.DataFrame) -> pd.DataFrame:
    model = ReceivingPropsModel(market)
    model.fit(history[history["season"].isin(TRAIN)])
    model.tune_dispersion(TRAIN[-2:])
    model.extend_history(history[history["season"].isin(EVAL)])
    stat_col = model.stat_col

    # Per-player time series for the book proxy + recency filter.
    series: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for pk, t, val in history[["player_key", "t", stat_col]].itertuples(index=False):
        ts, vs = series[pk]
        ts.append(int(t))
        vs.append(float(max(0.0, val)) if pd.notna(val) else 0.0)

    rows = []
    ev = history[history["season"].isin(EVAL)]
    for (season, week), games in ev.groupby(["season", "week"]):
        asof_t = int(season) * 100 + int(week)
        for r in games.itertuples(index=False):
            pk = r.player_key
            ts, vs = series[pk]
            cut = bisect.bisect_left(ts, asof_t)
            if cut == 0:
                continue
            # Quotable filter, mirroring replay.py: seen in the last 3 weeks.
            if ts[cut - 1] < asof_t - 3:
                continue
            line = book_proxy_line(vs[:cut], market)
            if line is None:
                continue
            proj = model.project(pk, int(season), int(week), opponent=r.opponent_team)
            if proj is None:
                continue
            p_over = proj.prob_over(line)
            if not 0.15 <= p_over <= 0.85:
                continue
            actual = float(max(0.0, getattr(r, stat_col)))
            for side, p in (("over", p_over), ("under", proj.prob_under(line))):
                edge = p - PRICE_DEVIG
                if edge < EDGE_FLOOR:
                    continue
                if actual == line:
                    result = "push"
                elif (actual > line) == (side == "over"):
                    result = "win"
                else:
                    result = "loss"
                rows.append({
                    "market": market, "season": season, "week": week,
                    "player": pk, "game": "|".join(sorted([r.team, r.opponent_team])),
                    "side": side, "line": line, "p": p, "edge": edge,
                    "mean": proj.mean, "actual": actual, "result": result,
                })
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame) -> tuple[float, float, int]:
    d = df[df["result"] != "push"]
    if d.empty:
        return 0.0, 0.0, 0
    wins = (d["result"] == "win").sum()
    losses = (d["result"] == "loss").sum()
    pnl = wins * (100 / 110) - losses
    return wins / len(d) * 100, pnl / len(d) * 100, len(d)


def report(df: pd.DataFrame) -> None:
    print(f"\n{'='*78}\nPICKS AT >={EDGE_FLOOR:.0%} EDGE — {len(df)} picks "
          f"({EVAL[0]}-{EVAL[-1]} walk-forward, breakeven 52.4%)\n{'='*78}")
    for label, sub in [("ALL", df)] + [(m, df[df["market"] == m]) for m in df["market"].unique()]:
        hit, r, n = roi(sub)
        print(f"{label:<24} n={n:<6} hit={hit:5.1f}%  ROI={r:+6.2f}%")

    print("\n-- by side --")
    for side, sub in df.groupby("side"):
        hit, r, n = roi(sub)
        print(f"  {side:<8} n={n:<6} hit={hit:5.1f}%  ROI={r:+6.2f}%  "
              f"claimed={sub['p'].mean():.1%}  gap={hit/100 - sub['p'].mean():+.1%}")

    print("\n-- by claimed edge --")
    bins = [0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
    df = df.copy()
    df["bucket"] = pd.cut(df["edge"], bins, right=False)
    for b, sub in df.groupby("bucket", observed=True):
        hit, r, n = roi(sub)
        if n < 30:
            continue
        print(f"  {str(b):<16} n={n:<6} hit={hit:5.1f}%  ROI={r:+6.2f}%  "
              f"claimed={sub['p'].mean():.1%}")

    print("\n-- side split by market --")
    for m, sub in df.groupby("market"):
        u = (sub["side"] == "under").mean()
        print(f"  {m:<24} under share = {u:.1%}")

    print("\n-- exposure concentration --")
    per_game = df.groupby(["season", "week", "game"]).size()
    per_player = df.groupby(["season", "week", "player"]).size()
    print(f"  picks per game:   mean {per_game.mean():.2f}, "
          f"max {per_game.max()}, share in games with >=3 picks: "
          f"{per_game[per_game >= 3].sum() / len(df):.1%}")
    print(f"  picks per player: mean {per_player.mean():.2f}, "
          f"share of picks where the SAME player is bet twice "
          f"(both markets): {per_player[per_player >= 2].sum() / len(df):.1%}")
    # Do same-player double bets agree in direction (near-duplicate risk)?
    both = df.groupby(["season", "week", "player"]).filter(lambda g: len(g) >= 2)
    if not both.empty:
        agree = both.groupby(["season", "week", "player"])["side"].nunique().eq(1).mean()
        print(f"  of doubled players, {agree:.1%} are the SAME side in both markets")


if __name__ == "__main__":
    seasons = TRAIN + EVAL
    print(f"Loading {seasons[0]}-{seasons[-1]}...", flush=True)
    hist = build_receiving_history(load_player_stats(seasons))
    out = pd.concat([run(m, hist) for m in MODELED_MARKETS], ignore_index=True)
    out.to_csv(sys.argv[1] if len(sys.argv) > 1 else "prop_picks_eval.csv", index=False)
    report(out)
