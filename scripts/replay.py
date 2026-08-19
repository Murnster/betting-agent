#!/usr/bin/env python
"""
Replay past NFL weeks — a dress rehearsal for the props workflow.

Pick any completed week (or let --random pick one), choose games
interactively, and see exactly what a props run will look like in season:
projections, pick cards, instant grading against what actually happened,
and session P&L.

The prop lines here are SYNTHETIC (each player's trailing median, priced
-110/-110) because historical prop odds aren't available on the free tier.
A naive median line is softer than a real book's, so treat the P&L as a
feel for the workflow and the models — the real test is the in-season
paper trade against bet365 lines.

Usage:
    uv run python scripts/replay.py --season 2025 --week 7
    uv run python scripts/replay.py --random
    uv run python scripts/replay.py --random --seed 42 --all-games
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    ReceivingPropsModel,
    build_receiving_history,
    load_player_stats,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MARKET_LABELS = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving yds",
}


def synthetic_line(recent: pd.Series, market: str) -> float | None:
    """
    A book-like line: the player's trailing median, forced to a half point
    the way real prop lines are quoted. Independent of our model's center.
    """
    if recent.empty:
        return None
    med = float(recent.median())
    if market == "player_receptions":
        line = float(int(med)) + 0.5
    else:
        line = round(med * 2) / 2
        if line == int(line):
            line += 0.5
    return line if line > 0 else None


def week_games(history: pd.DataFrame, season: int, week: int) -> list[tuple[str, str]]:
    """Matchups for the week, derived from player rows (team, opponent)."""
    rows = history[(history["season"] == season) & (history["week"] == week)]
    pairs: set[frozenset] = set()
    for team, opp in rows[["team", "opponent_team"]].drop_duplicates().itertuples(index=False):
        pairs.add(frozenset((team, opp)))
    return sorted(tuple(sorted(p)) for p in pairs if len(p) == 2)


def choose_games(games: list[tuple[str, str]], take_all: bool) -> list[tuple[str, str]]:
    print("\nGames that week:")
    for i, (a, b) in enumerate(games, 1):
        print(f"  {i:>2}. {a} vs {b}")
    if take_all or not sys.stdin.isatty():
        return games
    raw = input("\nSelect games (e.g. 1,3,5 — or 'all'): ").strip().lower()
    if raw in ("", "all", "a"):
        return games
    picked = []
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit() and 1 <= int(tok) <= len(games):
            picked.append(games[int(tok) - 1])
    return picked or games


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a past NFL week (props dress rehearsal)")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--random", action="store_true", help="Pick a random completed week")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--all-games", action="store_true", help="Skip the interactive picker")
    parser.add_argument("--max-picks-per-game", type=int, default=4)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    if not args.random and (args.season is None or args.week is None):
        parser.error("Pass --season and --week, or --random")

    # The replayed season plus three before it for warm-up and tuning.
    season = args.season or random.choice([2023, 2024, 2025])
    print(f"Loading player stats ({season - 3}-{season})...")
    history = build_receiving_history(load_player_stats(list(range(season - 3, season + 1))))
    if history.empty:
        raise SystemExit("No player stats available.")

    completed_weeks = sorted(
        history.loc[history["season"] == season, "week"].unique().tolist()
    )
    if not completed_weeks:
        raise SystemExit(f"No completed games found for season {season}.")
    week = args.week or random.choice([w for w in completed_weeks if w >= 4])
    if week not in completed_weeks:
        raise SystemExit(
            f"Season {season} week {week} has no stats "
            f"(completed weeks: {completed_weeks[0]}-{completed_weeks[-1]})."
        )
    asof_t = season * 100 + week
    bankroll = args.bankroll or settings.starting_bankroll

    print(f"\n=== Replaying {season} week {week} "
          f"(models see nothing from this week on) ===")

    # Fit on strictly earlier data — same leak hygiene as the calibration gate.
    train = history[history["t"] < asof_t]
    models: dict[str, ReceivingPropsModel] = {}
    for m in MODELED_MARKETS:
        model = ReceivingPropsModel(m).fit(train)
        model.tune_dispersion([season - 1])
        models[m] = model

    actuals = history[(history["season"] == season) & (history["week"] == week)]
    games = week_games(history, season, week)
    if not games:
        raise SystemExit("No games found for that week.")
    selected = choose_games(games, args.all_games)

    # Most recent team per player, as of the replayed week — and only
    # players a book would actually hang a line on: active this season and
    # seen within the last three weeks. Without this the replay fills up
    # with practice-squad names that DNP.
    before = train.sort_values("t")
    recent_t = before.groupby("player_key")["t"].max()
    active = recent_t[recent_t >= season * 100 + max(1, week - 3)].index
    last_team = before.groupby("player_key")["team"].last().loc[active]

    staked = won = 0.0
    record = {"win": 0, "loss": 0, "push": 0, "void": 0}
    print(f"\n{'player':<24} {'market':<14} {'side':<6} {'line':>6} "
          f"{'model':>7} {'edge':>7} {'stake':>7}  {'actual':>7} {'result':<6} {'pnl':>8}")
    print("-" * 108)

    for team_a, team_b in selected:
        game_players = last_team[last_team.isin([team_a, team_b])].index
        candidates = []
        for market, model in models.items():
            stat_col = model.stat_col
            for player_key in game_players:
                player_team = last_team[player_key]
                opponent = team_b if player_team == team_a else team_a
                proj = model.project(player_key, season, week, opponent=opponent)
                if proj is None:
                    continue
                recent = (
                    before[before["player_key"] == player_key][stat_col].tail(8)
                )
                line = synthetic_line(recent, market)
                if line is None:
                    continue
                # Books only quote meaningful lines — skip bench-depth numbers.
                if market == "player_receptions" and line < 1.5:
                    continue
                if market == "player_reception_yds" and line < 10.0:
                    continue
                p_over = proj.prob_over(line)
                # -110/-110 synthetic prices de-vig to 0.5 each side.
                for side, model_p in (("over", p_over), ("under", proj.prob_under(line))):
                    # A book wouldn't hang -110/-110 on a line this lopsided;
                    # these are pricing artifacts, not edges.
                    if not 0.15 <= model_p <= 0.85:
                        continue
                    edge = model_p - 0.5
                    if edge < args.min_edge:
                        continue
                    kf, stake = recommended_bet(model_p, -110, edge, bankroll)
                    if stake <= 0:
                        continue
                    candidates.append((edge, player_key, market, side, line, model_p, stake))

        candidates.sort(reverse=True)
        print(f"\n  -- {team_a} vs {team_b} --")
        if not candidates:
            print("     no edges clear the threshold")
            continue
        for edge, player_key, market, side, line, model_p, stake in \
                candidates[: args.max_picks_per_game]:
            row = actuals[actuals["player_key"] == player_key]
            stat_col = models[market].stat_col
            if row.empty:
                actual, result, pnl = None, "void", 0.0
            else:
                actual = float(row.iloc[0][stat_col])
                if actual == line:
                    result, pnl = "push", 0.0
                elif (actual > line) == (side == "over"):
                    result, pnl = "win", stake * 100 / 110
                else:
                    result, pnl = "loss", -stake
            record[result] += 1
            if result in ("win", "loss"):
                staked += stake
            won += pnl
            actual_str = f"{actual:>7.1f}" if actual is not None else "    DNP"
            print(f"{player_key:<24} {MARKET_LABELS[market]:<14} {side:<6} {line:>6.1f} "
                  f"{model_p:>6.1%} {edge:>+6.1%} {stake:>7.2f}  {actual_str} "
                  f"{result:<6} {pnl:>+8.2f}")

    print("\n" + "=" * 108)
    roi = (won / staked * 100) if staked else 0.0
    print(f"Session: {record['win']}-{record['loss']}-{record['push']} "
          f"({record['void']} void) | staked ${staked:,.2f} | "
          f"P&L ${won:+,.2f} | ROI {roi:+.1f}%")
    print("Synthetic -110 lines at each player's trailing median — softer than a "
          "real book. This rehearses the workflow, not the edge.")


if __name__ == "__main__":
    main()
