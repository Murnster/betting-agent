#!/usr/bin/env python
"""
Walk-forward diagnostic for LADDER hits — sets LADDER_EDGE_FLOORS /
LADDER_EDGE_CAPS, LADDER_MIN_RUNG and the LADDER_MIN/MAX_FAIR_PROB window in
sports/nfl/props.py, the way props_diagnostic.py does for the main card.
Re-run after any change to the distributions, the ladder projection
(LADDER_PRIOR_GAMES) or the tail calibrator.

A ladder hit is the player reaching a milestone: Over at a rung of the
alternate board (60+ receiving yards, 6+ receptions, 40+ rushing). Two
questions decide whether the section can exist:

1. Is P(hit) calibrated in the tails? The main isotonic layer only saw lines
   near the projection; `Projection.prob_hit` uses a second layer fit on the
   rungs the books hang. Section "tail calibration" reports claimed vs
   realised by probability bucket for every rung the proxy book would quote.
2. Where is the edge real? The proxy book prices each rung from the same
   distribution family located at the player's trailing 8-game mean (books
   price recent central tendency), with no shrinkage, defense factor or
   calibration — a soft book, so absolute ROI is optimistic. The claimed
   edge is P(hit) minus that proxy fair price; picks are bucketed by claimed
   edge and by fair probability, with flat-stake ROI at several holds.

Usage:
    uv run python scripts/ladder_diagnostic.py [out.csv]
"""
from __future__ import annotations

import bisect
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from betting_agent.sports.nfl.props import (
    LADDER_MIN_RUNG,
    LADDER_RUNGS,
    Projection,
    ReceivingPropsModel,
    active_player_keys,
    build_receiving_history,
    build_rushing_history,
    load_player_stats,
)

TRAIN = [2020, 2021, 2022, 2023]
EVAL = [2024, 2025]
REPORT_FLOOR = 0.03
HOLDS = (0.05, 0.10, 0.15)
PROXY_WINDOW = 8
PROXY_RANGE = (0.05, 0.95)     # rungs a book would bother to quote


def run(market: str, history: pd.DataFrame) -> pd.DataFrame:
    model = ReceivingPropsModel(market)
    model.fit(history[history["season"].isin(TRAIN)])
    model.tune_dispersion(TRAIN[-2:])
    model.extend_history(history[history["season"].isin(EVAL)])
    col = model.stat_col

    series: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    ordered = history.sort_values("t")
    for pk, t, val in ordered[["player_key", "t", col]].itertuples(index=False):
        ts, vs = series[pk]
        ts.append(int(t))
        vs.append(max(0.0, float(val)) if pd.notna(val) else 0.0)

    rows = []
    ev = history[history["season"].isin(EVAL)]
    for (season, week), games in ev.groupby(["season", "week"]):
        asof_t = int(season) * 100 + int(week)
        active = active_player_keys(history, asof_t)
        for r in games.itertuples(index=False):
            pk = r.player_key
            if pk not in active:
                continue
            ts, vs = series[pk]
            cut = bisect.bisect_left(ts, asof_t)
            if cut < 4:
                continue
            proxy_mean = float(np.mean(vs[max(0, cut - PROXY_WINDOW):cut]))
            if proxy_mean <= 0:
                continue
            proj = model.project_ladder(pk, int(season), int(week), opponent=r.opponent_team)
            if proj is None:
                continue
            book = Projection(player=pk, market=market, mean=proxy_mean, games=0,
                              _dist=model._make_dist(proxy_mean, r.position, model.dispersion_scale))
            actual = float(max(0.0, getattr(r, col)))
            for rung in LADDER_RUNGS[market]:
                p_fair = book._raw_over(rung)
                if not PROXY_RANGE[0] <= p_fair <= PROXY_RANGE[1]:
                    continue
                p_hit = proj.prob_hit(rung)
                rows.append({
                    "market": market, "season": season, "week": week, "player": pk,
                    "position": r.position,
                    "game": "|".join(sorted([r.team, r.opponent_team])),
                    "rung": rung, "p_model": p_hit, "p_fair": p_fair,
                    "edge": p_hit - p_fair, "mean": proj.mean, "proxy_mean": proxy_mean,
                    "actual": actual, "hit": float(actual > rung),
                })
    return pd.DataFrame(rows)


def roi_at(df: pd.DataFrame, hold: float) -> float:
    """Flat-stake ROI% if the book sells each rung at fair × (1 + hold)."""
    if df.empty:
        return 0.0
    p_book = (df["p_fair"] * (1 + hold)).clip(upper=0.98)
    pnl = np.where(df["hit"] == 1, 1 / p_book - 1, -1.0)
    return float(pnl.mean() * 100)


def _bucket_table(df: pd.DataFrame, by: str, bins: list[float], min_n: int = 40) -> None:
    d = df.copy()
    d["b"] = pd.cut(d[by], bins, right=False)
    for b, sub in d.groupby("b", observed=True):
        if len(sub) < min_n:
            continue
        rois = " / ".join(f"{roi_at(sub, h):+6.1f}%" for h in HOLDS)
        print(f"  {str(b):<14} n={len(sub):<6} claimed={sub['p_model'].mean():.3f} "
              f"realised={sub['hit'].mean():.3f} gap={sub['hit'].mean() - sub['p_model'].mean():+.3f}"
              f"  ROI {rois}")


def report(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}\nLADDER RUNGS — {len(df)} quotable player-rungs "
          f"({EVAL[0]}-{EVAL[-1]} walk-forward)\n{'=' * 78}")
    for m, sub in df.groupby("market"):
        print(f"  {m:<24} n={len(sub):<6} claimed={sub['p_model'].mean():.3f} "
              f"realised={sub['hit'].mean():.3f}  proxy={sub['p_fair'].mean():.3f}")

    print("\n-- tail calibration: P(hit) buckets, every quotable rung --")
    for m, sub in df.groupby("market"):
        print(f"  [{m}]")
        _bucket_table(sub, "p_model", [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0], min_n=100)

    # From here on only milestone rungs (LADDER_MIN_RUNG): the low rungs are
    # 10+/15+ yards for backups — depth-chart guesses, not what books hang.
    df = df[df["rung"] >= df["market"].map(LADDER_MIN_RUNG).fillna(0.0)]
    print(f"\n-- milestone rungs only ({LADDER_MIN_RUNG}): {len(df)} player-rungs --")

    print(f"\n-- picks by claimed edge (>= {REPORT_FLOOR:.0%}); ROI at book hold "
          + " / ".join(f"{h:.0%}" for h in HOLDS) + " --")
    picks = df[df["edge"] >= REPORT_FLOOR]
    for m, sub in picks.groupby("market"):
        print(f"  [{m}]")
        _bucket_table(sub, "edge", [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0])

    print("\n-- picks by proxy fair probability (edge >= 8%) --")
    for m, sub in df[df["edge"] >= 0.08].groupby("market"):
        print(f"  [{m}]")
        _bucket_table(sub, "p_fair", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.96])

    print("\n-- cumulative floors, fair in [0.20, 0.65], edge < 20%, best rung per player --")
    window = df[(df["p_fair"] >= 0.20) & (df["p_fair"] <= 0.65) & (df["edge"] < 0.20)]
    for m, sub in window.groupby("market"):
        print(f"  [{m}]")
        for floor in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
            s = sub[sub["edge"] >= floor]
            if len(s) < 20:
                continue
            best = s.sort_values("edge", ascending=False).groupby(["season", "week", "player"]).head(1)
            rois = " / ".join(f"{roi_at(best, h):+6.1f}%" for h in HOLDS)
            per_game = len(best) / max(1, best["game"].nunique())
            print(f"    >= {floor:.0%}  rungs={len(s):<5} best-per-player={len(best):<5} "
                  f"hit={best['hit'].mean():.3f} claimed={best['p_model'].mean():.3f}  "
                  f"ROI {rois}  per game={per_game:.2f}")

    print("\n-- by position (best rung per player, edge 8-20%, fair window) --")
    pk = window[window["edge"] >= 0.08].sort_values("edge", ascending=False)
    pk = pk.groupby(["market", "season", "week", "player"]).head(1)
    for (m, pos), sub in pk.groupby(["market", "position"]):
        if len(sub) < 20:
            continue
        print(f"  {m:<24} {pos:<3} n={len(sub):<5} hit={sub['hit'].mean():.3f} "
              f"claimed={sub['p_model'].mean():.3f}  ROI@10%={roi_at(sub, 0.10):+.1f}%")


if __name__ == "__main__":
    seasons = TRAIN + EVAL
    print(f"Loading {seasons[0]}-{seasons[-1]}...", flush=True)
    stats = load_player_stats(seasons)
    frames = []
    for market, hist in (("player_receptions", build_receiving_history(stats)),
                         ("player_reception_yds", build_receiving_history(stats)),
                         ("player_rush_yds", build_rushing_history(stats))):
        print(f"Running {market}...", flush=True)
        frames.append(run(market, hist))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(sys.argv[1] if len(sys.argv) > 1 else "ladder_eval.csv", index=False)
    report(out)
