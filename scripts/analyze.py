#!/usr/bin/env python
"""
Standalone matchup analysis using Ollama.

Usage:
    # Analyze all today's games
    uv run python scripts/analyze.py [--sport NFL]

    # Analyze a specific matchup
    uv run python scripts/analyze.py --home "Kansas City Chiefs" --away "Buffalo Bills"

    # Include line movement analysis (requires odds in DB)
    uv run python scripts/analyze.py --line-movement
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from betting_agent.config import settings
from betting_agent.intelligence.sentiment import (
    analyze_game,
    analyze_line_movement,
    fetch_team_context,
    is_ollama_available,
    line_moved_significantly,
)
from betting_agent.sports.registry import get_sport_config, available_sports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _format_analysis(home: str, away: str, result: dict) -> str:
    """Format a single game analysis for terminal output."""
    lines = [
        "",
        f"  {away} @ {home}",
        "  " + "=" * 55,
        f"  Home health/momentum: {result['home_score']:+.2f}",
        f"  Away health/momentum: {result['away_score']:+.2f}",
        f"  Matchup edge: {result['matchup_edge'].upper()}",
    ]
    if result.get("key_factors"):
        lines.append("  Key factors:")
        for f in result["key_factors"]:
            lines.append(f"    - {f}")
    if result.get("summary"):
        lines.append(f"  Summary: {result['summary']}")
    lines.append("")
    return "\n".join(lines)


def analyze_specific_matchup(
    home: str, away: str, sport: str, include_line_movement: bool = False
) -> None:
    """Analyze a single specified matchup."""
    current_year = date.today().year
    depth = settings.sentiment_depth

    logger.info("Building context for %s vs %s...", away, home)
    home_ctx = fetch_team_context(home, current_year, depth=depth, sport=sport)
    away_ctx = fetch_team_context(away, current_year, depth=depth, sport=sport)

    logger.info("Running game analysis...")
    result = analyze_game(home, away, home_ctx, away_ctx, sport=sport)
    print(_format_analysis(home, away, result))

    if include_line_movement:
        _run_line_movement_for_matchup(home, away, sport)


def analyze_todays_games(sport: str, include_line_movement: bool = False) -> None:
    """Fetch today's games and analyze each."""
    from betting_agent.api.odds import OddsAPIClient

    config = get_sport_config(sport)
    client = OddsAPIClient()
    sport_key = config.sport_key

    logger.info("Fetching today's %s games from Odds API...", sport)
    raw_games = client.fetch_odds(sport_key)

    if not raw_games:
        # Fall back to DB
        try:
            from betting_agent.db.queries import get_scheduled_games
            from betting_agent.db.session import get_session

            with get_session() as session:
                db_games = get_scheduled_games(session, sport, date.today())
                raw_games = [
                    {"home_team": g.home_team, "away_team": g.away_team}
                    for g in db_games
                ]
        except Exception:
            pass

    if not raw_games:
        print(f"\nNo {sport} games found for today. Check your ODDS_API_KEY or DB.\n")
        return

    current_year = date.today().year
    depth = settings.sentiment_depth

    print("\n" + "=" * 60)
    print(f"  {sport} MATCHUP ANALYSIS — {date.today()}")
    print("=" * 60)

    for g in raw_games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue

        logger.info("Analyzing %s @ %s...", away, home)
        home_ctx = fetch_team_context(home, current_year, depth=depth, sport=sport)
        away_ctx = fetch_team_context(away, current_year, depth=depth, sport=sport)

        result = analyze_game(home, away, home_ctx, away_ctx, sport=sport)
        print(_format_analysis(home, away, result))

    if include_line_movement:
        _run_line_movement_all(sport)

    print("=" * 60 + "\n")


def _run_line_movement_for_matchup(home: str, away: str, sport: str) -> None:
    """Run line movement analysis for a specific matchup from DB odds."""
    try:
        from betting_agent.db.models import Game, Odds
        from betting_agent.db.session import get_session

        with get_session() as session:
            game = (
                session.query(Game)
                .filter(Game.home_team == home, Game.away_team == away)
                .order_by(Game.game_date.desc())
                .first()
            )
            if not game:
                print("  No game found in DB for line movement analysis.\n")
                return

            opening = (
                session.query(Odds)
                .filter(Odds.game_id == game.id, Odds.is_closing == False, Odds.bet_type == "moneyline")
                .first()
            )
            closing = (
                session.query(Odds)
                .filter(Odds.game_id == game.id, Odds.is_closing == True, Odds.bet_type == "moneyline")
                .first()
            )
            if not opening or not closing:
                print("  No opening/closing odds found for line movement analysis.\n")
                return

            open_dict = {"home_price": opening.home_price, "away_price": opening.away_price, "spread_home": opening.spread_home}
            close_dict = {"home_price": closing.home_price, "away_price": closing.away_price, "spread_home": closing.spread_home}

            if not line_moved_significantly(open_dict, close_dict):
                print("  No significant line movement detected.\n")
                return

            ctx = fetch_team_context(home, date.today().year, depth="basic", sport=sport)
            ctx += "\n" + fetch_team_context(game.away_team, date.today().year, depth="basic", sport=sport)

            result = analyze_line_movement(home, away, open_dict, close_dict, ctx)
            print(f"  LINE MOVEMENT: {result['explanation']}")
            if result["sharp_side"]:
                print(f"  Sharp side: {result['sharp_side']}")
            print()
    except Exception as exc:
        logger.warning("Line movement analysis failed: %s", exc)


def _run_line_movement_all(sport: str) -> None:
    """Run line movement analysis for all today's games in DB."""
    try:
        from betting_agent.db.models import Game, Odds
        from betting_agent.db.session import get_session

        print("\n  LINE MOVEMENT ANALYSIS")
        print("  " + "-" * 40)

        analyzed = 0
        with get_session() as session:
            today_games = (
                session.query(Game)
                .filter(Game.sport == sport, Game.game_date == date.today())
                .all()
            )
            for game in today_games:
                opening = (
                    session.query(Odds)
                    .filter(Odds.game_id == game.id, Odds.is_closing == False, Odds.bet_type == "moneyline")
                    .first()
                )
                closing = (
                    session.query(Odds)
                    .filter(Odds.game_id == game.id, Odds.is_closing == True, Odds.bet_type == "moneyline")
                    .first()
                )
                if not opening or not closing:
                    continue

                open_dict = {"home_price": opening.home_price, "away_price": opening.away_price, "spread_home": opening.spread_home}
                close_dict = {"home_price": closing.home_price, "away_price": closing.away_price, "spread_home": closing.spread_home}

                if not line_moved_significantly(open_dict, close_dict):
                    continue

                ctx = fetch_team_context(game.home_team, date.today().year, depth="basic", sport=sport)
                ctx += "\n" + fetch_team_context(game.away_team, date.today().year, depth="basic", sport=sport)

                result = analyze_line_movement(
                    game.home_team, game.away_team, open_dict, close_dict, ctx
                )
                print(f"\n  {game.away_team} @ {game.home_team}")
                print(f"  {result['explanation']}")
                if result["sharp_side"]:
                    print(f"  Sharp side: {result['sharp_side']}")
                analyzed += 1

        if not analyzed:
            print("  No significant line movements detected.")
    except Exception as exc:
        logger.warning("Line movement analysis failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone matchup analysis")
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name")
    parser.add_argument(
        "--sport",
        type=str,
        default="NFL",
        help=f"Sport ({', '.join(available_sports())})",
    )
    parser.add_argument("--line-movement", action="store_true", help="Include line movement analysis")
    args = parser.parse_args()

    sport = args.sport.upper()

    if not is_ollama_available():
        print("\nOllama is not running. Start it with: ollama serve\n")
        return

    if args.home and args.away:
        analyze_specific_matchup(args.home, args.away, sport, args.line_movement)
    elif args.home or args.away:
        print("\nBoth --home and --away are required for specific matchup analysis.\n")
        return
    else:
        analyze_todays_games(sport, args.line_movement)


if __name__ == "__main__":
    main()
