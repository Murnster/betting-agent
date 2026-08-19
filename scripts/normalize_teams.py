#!/usr/bin/env python
"""
One-off backfill: rewrite games.home_team / games.away_team to the canonical
abbreviation.

Rows created from Odds API events stored full club names ("Kansas City
Chiefs") while loader-seeded rows stored abbreviations ("KC"). Grading and CLV
compare a pick's team against these strings, so mixed rows graded against the
wrong side. New writes are canonical; this fixes what is already there.

Usage:
    uv run python scripts/normalize_teams.py            # report only
    uv run python scripts/normalize_teams.py --apply    # write changes
    uv run python scripts/normalize_teams.py --apply --regrade
"""

from __future__ import annotations

import argparse
import logging

from betting_agent.db.models import Game, Pick
from betting_agent.db.session import get_session
from betting_agent.sports.teams import canonical_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalize_games(apply: bool) -> int:
    """Rewrite non-canonical team names. Returns the number of rows changed."""
    changed = 0
    with get_session() as session:
        for game in session.query(Game).all():
            home = canonical_team(game.sport, game.home_team)
            away = canonical_team(game.sport, game.away_team)
            if home == game.home_team and away == game.away_team:
                continue

            changed += 1
            print(
                f"  game {game.id} ({game.sport} {game.game_date}): "
                f"{game.away_team} @ {game.home_team}  →  {away} @ {home}"
            )
            if apply:
                game.home_team = home
                game.away_team = away

        if not apply:
            session.rollback()
    return changed


def reset_affected_picks(apply: bool) -> int:
    """
    Clear results on graded picks so they can be re-graded against the
    corrected game rows. Anything graded before the fix may be wrong.
    """
    with get_session() as session:
        picks = session.query(Pick).filter(Pick.result.isnot(None)).all()
        if apply:
            for pick in picks:
                pick.result = None
                pick.pnl = None
                pick.graded_at = None
                pick.clv = None
                pick.closing_odds = None
        else:
            session.rollback()
        return len(picks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalise team names in the DB")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Also clear existing pick results so grade.py can re-grade them",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\n=== Canonicalising team names ({mode}) ===\n")

    changed = normalize_games(args.apply)
    print(f"\n{changed} game rows {'updated' if args.apply else 'would be updated'}.")

    if args.regrade:
        n = reset_affected_picks(args.apply)
        print(f"{n} graded picks {'reset' if args.apply else 'would be reset'}.")
        if args.apply:
            print("\nRun: uv run python scripts/grade.py")

    if not args.apply:
        print("\nNothing was written. Re-run with --apply to commit.")


if __name__ == "__main__":
    main()
