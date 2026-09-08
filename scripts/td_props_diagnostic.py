#!/usr/bin/env python
"""
Walk-forward diagnostic for anytime-TD PICKS — sets the edge floor for
`player_anytime_td` in PROP_EDGE_FLOORS, the way props_diagnostic.py does
for the receiving markets. Re-run after any change to td_props.py.

Per eval game the board is every skill player seen in the last three
regular-season slates. The proxy book prices each player from a naive
trailing TD rate shrunk toward a usage-scaled prior (a real book knows who
plays), de-vigged against the market's expected TDs for
the game (real closing spread/total from nflreadpy) exactly as production
de-vigs the real board. Our model's P(any TD) against that fair price is
the claimed edge; a pick's payout assumes the book sells the fair price
plus a hold, reported at several holds because that is the number we do
not know until the live boards have been logged for a while.

The proxy is softer than a real book, so absolute ROI is optimistic; the
bucket-to-bucket comparisons and the claimed-vs-realised gap are what set
the floor.

Usage:
    uv run python scripts/td_props_diagnostic.py [out.csv]
"""
from __future__ import annotations

import bisect
import sys

import numpy as np
import pandas as pd

from betting_agent.sports.nfl.props import active_player_keys, load_player_stats, load_season_schedule
from betting_agent.sports.nfl.td_props import (
    BOARD_COVERAGE,
    MIN_FAIR_PROB,
    TD_STAT_COL,
    TouchdownPropsModel,
    build_td_history,
    expected_game_tds,
    fair_yes_probabilities,
)

TRAIN = [2020, 2021, 2022, 2023]
EVAL = [2024, 2025]
REPORT_FLOOR = 0.03
HOLDS = (0.15, 0.25, 0.35)
NAIVE_WINDOW = 16
NAIVE_PRIOR = 8.0


def _prob_to_american(p: float) -> int:
    p = min(max(p, 0.02), 0.98)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def run(history: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    model = TouchdownPropsModel().fit(history[history["season"].isin(TRAIN)])
    model.calibrate(TRAIN[-2:], schedule=schedules)
    model.extend_history(history[history["season"].isin(EVAL)])

    ordered = history.sort_values("t")
    ordered = ordered.assign(_touches=ordered[["carries", "targets"]].fillna(0).sum(axis=1))
    series: dict[str, tuple[list[int], np.ndarray, np.ndarray, list[str], str]] = {}
    for pk, g in ordered.groupby("player_key", sort=False):
        series[pk] = (g["t"].astype(int).tolist(), g[TD_STAT_COL].to_numpy(dtype=float),
                      g["_touches"].to_numpy(dtype=float), g["team"].tolist(),
                      str(g["position"].iloc[-1]))
    train = ordered[ordered["season"].isin(TRAIN)]
    per_touch = (train.groupby("position")[TD_STAT_COL].sum()
                 / train.groupby("position")["_touches"].sum().clip(lower=1))
    actual_by = {(int(r.season), int(r.week), r.player_key): float(getattr(r, TD_STAT_COL))
                 for r in ordered.itertuples(index=False)}

    sched = schedules[schedules["season"].isin(EVAL) & (schedules["game_type"] == "REG")]
    rows = []
    for (season, week), games in sched.groupby(["season", "week"]):
        asof_t = int(season) * 100 + int(week)
        active = active_player_keys(history, asof_t)
        for g in games.itertuples(index=False):
            exp_home, exp_away = expected_game_tds(g.spread_line, g.total_line)
            teams = {g.home_team: (exp_home, g.away_team), g.away_team: (exp_away, g.home_team)}
            board: dict[str, tuple[str, float, float, str]] = {}   # pk → (team, naive λ, model p, pos)
            for pk in active:
                ts, tds, touches, tms, pos = series[pk]
                cut = bisect.bisect_left(ts, asof_t)
                if cut < 4:
                    continue
                team = tms[cut - 1]
                if team not in teams:
                    continue
                exp_tds, opp = teams[team]
                recent = tds[max(0, cut - NAIVE_WINDOW):cut]
                usage_prior = per_touch.get(pos, 0.03) * touches[max(0, cut - NAIVE_WINDOW):cut].mean()
                naive = (recent.sum() + NAIVE_PRIOR * usage_prior) / (len(recent) + NAIVE_PRIOR)
                proj = model.project(pk, int(season), int(week), opponent=opp, team=team,
                                     team_expected_tds=exp_tds)
                if proj is None:
                    continue
                board[pk] = (team, naive, proj.prob, pos)
            if len(board) < 10:
                continue
            target = BOARD_COVERAGE * (exp_home + exp_away)
            # Proxy book: naive rates sold with a uniform hold; the de-vig
            # cancels the hold, leaving the naive board scaled to the market.
            prices = {pk: _prob_to_american(min(0.95, (1 - np.exp(-lam)) * 1.25))
                      for pk, (_, lam, _, _) in board.items()}
            fair, _ = fair_yes_probabilities(prices, target)
            for pk, (team, naive, p_model, pos) in board.items():
                p_fair = fair[pk]
                if p_fair < MIN_FAIR_PROB:
                    continue      # production never bets these
                edge = p_model - p_fair
                actual = actual_by.get((int(season), int(week), pk))
                rows.append({
                    "season": season, "week": week, "game": f"{g.away_team}@{g.home_team}",
                    "player": pk, "team": team, "position": pos, "p_model": p_model,
                    "p_fair": p_fair, "edge": edge, "played": actual is not None,
                    "scored": None if actual is None else float(actual >= 1),
                })
    return pd.DataFrame(rows)


def roi_at(df: pd.DataFrame, hold: float) -> float:
    """Flat-stake ROI% if the book sells each pick at fair × (1 + hold)."""
    d = df[df["played"]]
    if d.empty:
        return 0.0
    p_book = (d["p_fair"] * (1 + hold)).clip(upper=0.98)
    pnl = np.where(d["scored"] == 1, 1 / p_book - 1, -1.0)
    return float(pnl.mean() * 100)


def report(df: pd.DataFrame) -> None:
    played = df[df["played"]]
    print(f"\n{'=' * 78}\nANYTIME TD — {len(played)} board player-games "
          f"({EVAL[0]}-{EVAL[-1]} walk-forward), {df['game'].nunique()} games\n{'=' * 78}")
    print(f"board calibration: claimed {played['p_model'].mean():.3f} vs realised "
          f"{played['scored'].mean():.3f}; fair-proxy mean {played['p_fair'].mean():.3f}")

    print("\n-- model probability buckets (all board players) --")
    played = played.copy()
    played["pb"] = pd.cut(played["p_model"], [0, .1, .2, .3, .4, .5, 1.0])
    for b, sub in played.groupby("pb", observed=True):
        print(f"  {str(b):<12} n={len(sub):<6} claimed={sub['p_model'].mean():.3f} "
              f"realised={sub['scored'].mean():.3f}")

    print(f"\n-- picks by claimed edge (floor {REPORT_FLOOR:.0%}); ROI at book hold "
          + " / ".join(f"{h:.0%}" for h in HOLDS) + " --")
    picks = played[played["edge"] >= REPORT_FLOOR].copy()
    picks["bucket"] = pd.cut(picks["edge"], [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 1.0], right=False)
    for b, sub in picks.groupby("bucket", observed=True):
        if len(sub) < 30:
            continue
        rois = " / ".join(f"{roi_at(sub, h):+6.1f}%" for h in HOLDS)
        print(f"  {str(b):<14} n={len(sub):<5} claimed={sub['p_model'].mean():.3f} "
              f"realised={sub['scored'].mean():.3f} gap={sub['scored'].mean() - sub['p_model'].mean():+.3f}"
              f"  ROI {rois}")

    print("\n-- cumulative: every pick at or above a floor --")
    for floor in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
        sub = played[played["edge"] >= floor]
        if len(sub) < 30:
            continue
        rois = " / ".join(f"{roi_at(sub, h):+6.1f}%" for h in HOLDS)
        per_game = len(sub) / max(1, sub["game"].nunique())
        print(f"  >= {floor:.0%}  n={len(sub):<5} hit={sub['scored'].mean():.3f} "
              f"claimed={sub['p_model'].mean():.3f}  ROI {rois}  picks/game={per_game:.2f}")

    print("\n-- by position (picks >= 8% edge) --")
    for pos, sub in played[played["edge"] >= 0.08].groupby("position"):
        if len(sub) < 20:
            continue
        print(f"  {pos:<4} n={len(sub):<5} hit={sub['scored'].mean():.3f} "
              f"claimed={sub['p_model'].mean():.3f}  ROI@25%={roi_at(sub, 0.25):+.1f}%")


if __name__ == "__main__":
    seasons = TRAIN + EVAL
    print(f"Loading {seasons[0]}-{seasons[-1]}...", flush=True)
    hist = build_td_history(load_player_stats(seasons))
    scheds = pd.concat([load_season_schedule(s) for s in seasons], ignore_index=True)
    out = run(hist, scheds)
    out.to_csv(sys.argv[1] if len(sys.argv) > 1 else "td_picks_eval.csv", index=False)
    report(out)
