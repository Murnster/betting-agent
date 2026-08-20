#!/usr/bin/env python
"""
Replay past NFL weeks or weekends — a dress rehearsal for the props workflow.

Pick any completed week, calendar day, or a random Sunday, optionally run the
same heat pre-screen the live pipeline uses (`--suggest N` mirrors
`props.py --suggest N`), and see exactly what a props run will look like in
season: the slate, the heat board, projections, pick cards, instant grading
against what actually happened, and session P&L.

The prop lines here are SYNTHETIC (each player's trailing median, priced
-110/-110) because historical prop odds aren't available on the free tier.
A naive median line is softer than a real book's, so treat the P&L as a
feel for the workflow and the models — the real test is the in-season
paper trade against bet365 lines.

Usage:
    uv run python scripts/replay.py --season 2025 --week 7
    uv run python scripts/replay.py --random
    uv run python scripts/replay.py --date 2025-12-07 --suggest 4
    uv run python scripts/replay.py --random-sunday --suggest 4   # full weekend rehearsal
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from datetime import date

import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    PROP_EDGE_FLOORS,
    ReceivingPropsModel,
    book_proxy_line,
    build_receiving_history,
    edge_floor,
    load_player_stats,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MARKET_LABELS = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving yds",
}
SHORT_LABELS = {
    "player_receptions": "rec",
    "player_reception_yds": "yds",
}


def week_games(history: pd.DataFrame, season: int, week: int) -> list[tuple[str, str]]:
    """Matchups for the week, derived from player rows (team, opponent)."""
    rows = history[(history["season"] == season) & (history["week"] == week)]
    pairs: set[frozenset] = set()
    for team, opp in rows[["team", "opponent_team"]].drop_duplicates().itertuples(index=False):
        pairs.add(frozenset((team, opp)))
    return sorted(tuple(sorted(p)) for p in pairs if len(p) == 2)


def day_slate(
    season: int, day: date | None, pick_random_sunday: bool
) -> tuple[date, int, list[tuple[str, str]]]:
    """
    (day, week, matchups) for one calendar day, from published schedules.
    With pick_random_sunday, choose a random Sunday carrying a full slate.
    """
    from betting_agent.sports.nfl.features import normalise_raw_schedules
    from betting_agent.sports.nfl.loader import NFLLoader

    sched = normalise_raw_schedules(NFLLoader().load_schedules([season]).to_pandas())
    sched = sched[sched["home_score"].notna()].copy()
    sched["game_date"] = pd.to_datetime(sched["game_date"])

    if pick_random_sunday:
        sundays = sched[sched["game_date"].dt.dayofweek == 6]
        full = [d for d, g in sundays.groupby("game_date") if len(g) >= 6]
        if not full:
            raise SystemExit(f"No full Sunday slates found in season {season}.")
        stamp = random.choice(sorted(full))
    else:
        stamp = pd.Timestamp(day)

    rows = sched[sched["game_date"] == stamp]
    if rows.empty:
        raise SystemExit(f"No completed NFL games on {stamp.date()} in season {season}.")
    week = int(rows["week"].iloc[0])
    matchups = sorted(
        tuple(sorted((r.home_team, r.away_team))) for r in rows.itertuples(index=False)
    )
    return stamp.date(), week, matchups


def choose_games(games: list[tuple[str, str]], take_all: bool) -> list[tuple[str, str]]:
    print("\nGames:")
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


def game_edges(
    models: dict[str, ReceivingPropsModel],
    before: pd.DataFrame,
    last_team: pd.Series,
    season: int,
    week: int,
    team_a: str,
    team_b: str,
    bankroll: float,
    min_edge: float | None,
) -> list[tuple]:
    """
    Every prop edge clearing the floors for one game, deduped to one pick
    per player (best market), sorted best-first. Same policy as production.
    Tuples: (edge, player_key, market, side, line, model_p, stake).
    """
    candidates = []
    game_players = last_team[last_team.isin([team_a, team_b])].index
    for market, model in models.items():
        stat_col = model.stat_col
        for player_key in game_players:
            player_team = last_team[player_key]
            opponent = team_b if player_team == team_a else team_a
            proj = model.project(player_key, season, week, opponent=opponent)
            if proj is None:
                continue
            recent = before[before["player_key"] == player_key][stat_col].tail(8)
            line = book_proxy_line([max(0.0, v) for v in recent.tolist()], market)
            if line is None:
                continue
            p_over = proj.prob_over(line)
            # -110/-110 synthetic prices de-vig to 0.5 each side.
            for side, model_p in (("over", p_over), ("under", proj.prob_under(line))):
                # A book wouldn't hang -110/-110 on a line this lopsided;
                # these are pricing artifacts, not edges.
                if not 0.15 <= model_p <= 0.85:
                    continue
                edge = model_p - 0.5
                if edge < edge_floor(market, min_edge):
                    continue
                kf, stake = recommended_bet(model_p, -110, edge, bankroll)
                if stake <= 0:
                    continue
                candidates.append((edge, player_key, market, side, line, model_p, stake))

    best_per_player: dict[str, tuple] = {}
    for cand in sorted(candidates, reverse=True):
        best_per_player.setdefault(cand[1], cand)
    return sorted(best_per_player.values(), reverse=True)


def heat_board(
    games: list[tuple[str, str]],
    edges_by_game: dict[tuple[str, str], list[tuple]],
) -> list[tuple[float, int, tuple[str, str]]]:
    """Rank games by summed best-per-player edges — the free pre-screen."""
    board = []
    for game in games:
        edges = edges_by_game[game]
        board.append((sum(e[0] for e in edges), len(edges), game))
    board.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return board


def print_heat_board(board: list[tuple[float, int, tuple[str, str]]],
                     edges_by_game: dict[tuple[str, str], list[tuple]]) -> None:
    print("\n=== HEAT PRE-SCREEN (in season this is free — no odds calls) ===")
    print(f"{'rank':<5} {'game':<12} {'edges':>5} {'heat':>7}   top drivers")
    for i, (heat, n, game) in enumerate(board, 1):
        drivers = ", ".join(
            f"{pk} {SHORT_LABELS[mk]} {side[0].upper()}{line:g} ({p:.0%})"
            for _, pk, mk, side, line, p, _ in edges_by_game[game][:3]
        ) or "-"
        print(f"{i:<5} {game[0]:>3} vs {game[1]:<6} {n:>5} {heat:>+7.2f}   {drivers}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a past NFL week or weekend (props dress rehearsal)")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--random", action="store_true", help="Pick a random completed week")
    parser.add_argument("--date", type=str, default=None,
                        help="Replay one calendar day (YYYY-MM-DD), e.g. a Sunday")
    parser.add_argument("--random-sunday", action="store_true",
                        help="Pick a random Sunday with a full slate")
    parser.add_argument("--suggest", type=int, default=None, metavar="N",
                        help="Heat-rank the games (prints the board) and replay "
                             "only the top N — mirrors props.py --suggest")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=None,
                        help=f"Override the per-market floors {PROP_EDGE_FLOORS}")
    parser.add_argument("--all-games", action="store_true", help="Skip the interactive picker")
    parser.add_argument("--max-picks-per-game", type=int, default=4)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    day = date.fromisoformat(args.date) if args.date else None
    day_mode = day is not None or args.random_sunday
    if not day_mode and not args.random and (args.season is None or args.week is None):
        parser.error("Pass --season and --week, --date, --random-sunday, or --random")

    if day is not None:
        season = args.season or (day.year if day.month >= 8 else day.year - 1)
    else:
        season = args.season or random.choice([2023, 2024, 2025])

    day_games = None
    week = args.week
    if day_mode:
        day, week, day_games = day_slate(season, day, args.random_sunday)

    # The replayed season plus three before it for warm-up and tuning.
    print(f"Loading player stats ({season - 3}-{season})...")
    history = build_receiving_history(load_player_stats(list(range(season - 3, season + 1))))
    if history.empty:
        raise SystemExit("No player stats available.")

    completed_weeks = sorted(
        history.loc[history["season"] == season, "week"].unique().tolist()
    )
    if not completed_weeks:
        raise SystemExit(f"No completed games found for season {season}.")
    week = week or random.choice([w for w in completed_weeks if w >= 4])
    if week not in completed_weeks:
        raise SystemExit(
            f"Season {season} week {week} has no stats "
            f"(completed weeks: {completed_weeks[0]}-{completed_weeks[-1]})."
        )
    asof_t = season * 100 + week
    bankroll = args.bankroll or settings.starting_bankroll

    when = f"{day} (week {week})" if day_mode else f"week {week}"
    print(f"\n=== Replaying {season} {when} "
          f"(models see nothing from this week on) ===")

    # Fit on strictly earlier data — same leak hygiene as the calibration gate.
    train = history[history["t"] < asof_t]
    models: dict[str, ReceivingPropsModel] = {}
    for m in MODELED_MARKETS:
        model = ReceivingPropsModel(m).fit(train)
        model.tune_dispersion([season - 1])
        models[m] = model

    actuals = history[(history["season"] == season) & (history["week"] == week)]
    games = day_games if day_games is not None else week_games(history, season, week)
    if not games:
        raise SystemExit("No games found.")

    # Most recent team per player, as of the replayed week — and only
    # players a book would actually hang a line on: active this season and
    # seen within the last three weeks. Without this the replay fills up
    # with practice-squad names that DNP.
    before = train.sort_values("t")
    recent_t = before.groupby("player_key")["t"].max()
    active = recent_t[recent_t >= season * 100 + max(1, week - 3)].index
    last_team = before.groupby("player_key")["team"].last().loc[active]

    if args.suggest:
        edges_by_game = {
            g: game_edges(models, before, last_team, season, week, *g, bankroll, args.min_edge)
            for g in games
        }
        board = heat_board(games, edges_by_game)
        print_heat_board(board, edges_by_game)
        selected = [g for _, _, g in board[: args.suggest]]
    else:
        edges_by_game = {}
        selected = choose_games(games, args.all_games)

    staked = won = 0.0
    record = {"win": 0, "loss": 0, "push": 0, "void": 0}
    print(f"\n{'player':<24} {'market':<14} {'side':<6} {'line':>6} "
          f"{'model':>7} {'edge':>7} {'stake':>7}  {'actual':>7} {'result':<6} {'pnl':>8}")
    print("-" * 108)

    for team_a, team_b in selected:
        candidates = edges_by_game.get((team_a, team_b)) or game_edges(
            models, before, last_team, season, week, team_a, team_b, bankroll, args.min_edge
        )
        print(f"\n  -- {team_a} vs {team_b} --")
        if not candidates:
            print("     no edges clear the threshold")
            continue
        # Same-game correlation: props in one game share a quarterback and a
        # game script, so production scales their stakes by 1/sqrt(n). Mirror
        # it here or the rehearsal overstates what gets risked per game.
        slate = candidates[: args.max_picks_per_game]
        corr = 1.0 / math.sqrt(len(slate)) if len(slate) > 1 else 1.0
        for edge, player_key, market, side, line, model_p, raw_stake in slate:
            stake = raw_stake * corr
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
