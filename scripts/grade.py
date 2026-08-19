#!/usr/bin/env python
"""
Entry point: grade yesterday's picks and update CLV/ROI.

Usage:
    uv run python scripts/grade.py [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from betting_agent.accounting.clv import update_clv_for_picks
from betting_agent.accounting.grader import grade_picks
from betting_agent.accounting.roi import format_roi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _grade_nfl_props(target_date: date | None) -> int:
    """Grade ungraded NFL prop picks, loading player stats only when needed."""
    from betting_agent.db.models import Pick
    from betting_agent.db.session import get_session

    with get_session() as session:
        pending = (
            session.query(Pick)
            .filter(Pick.result.is_(None), Pick.bet_type == "prop", Pick.sport == "NFL")
            .count()
        )
    if not pending:
        return 0

    from betting_agent.accounting.grader import grade_prop_picks
    from betting_agent.sports.nfl.props import make_stat_lookup

    year = date.today().year
    season = year if date.today().month >= 8 else year - 1
    lookup = make_stat_lookup([season])
    return grade_prop_picks(lookup, sport="NFL", target_date=target_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade picks and print ROI report")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport")
    parser.add_argument("--season", type=int, default=None, help="Filter by season")
    parser.add_argument("--date", type=str, default=None, help="Re-grade a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None

    try:
        n_graded = grade_picks(target_date=target_date)
        print(f"\nGraded {n_graded} picks.")
    except Exception as exc:
        logger.error("Grading failed (DB may not be running): %s", exc)
        n_graded = 0

    # NFL prop picks grade from player stats, not game scores.
    try:
        n_props = _grade_nfl_props(target_date)
        if n_props:
            print(f"Graded {n_props} prop picks.")
    except Exception as exc:
        logger.warning("Prop grading failed: %s", exc)

    try:
        n_clv = update_clv_for_picks()
        print(f"Updated CLV for {n_clv} picks.")
    except Exception as exc:
        logger.warning("CLV update failed: %s", exc)

    try:
        report = format_roi_report(sport=args.sport, season=args.season, since=target_date)
        print(report)
    except Exception as exc:
        logger.error("ROI report failed: %s", exc)

    # ---- Discord results notification ----
    from betting_agent.config import settings
    if settings.discord_enabled:
        try:
            from betting_agent.notifications.discord import send_results_to_discord, is_discord_configured
            from betting_agent.accounting.roi import get_summary, get_breakdown_by_bet_type, get_graded_picks_detail
            from betting_agent.sports.registry import available_sports
            graded_date = target_date or (date.today() - timedelta(days=1))
            sport_list = [args.sport.upper()] if args.sport else available_sports()
            # When re-grading a past date, cap all queries at that date
            # so all-time/per-sport results don't include future days
            until_date = target_date  # None for default (today) runs
            sports_with_results = []
            for sport_name in sport_list:
                if is_discord_configured(sport_name, "RESULTS"):
                    summary = get_summary(sport=sport_name, since=graded_date, until=until_date)
                    if "total_bets" not in summary:
                        continue  # no graded picks for this sport on that date
                    sports_with_results.append(sport_name)
                    logger.info("Sending results to Discord (%s)...", sport_name)
                    breakdown = get_breakdown_by_bet_type(sport=sport_name, since=graded_date, until=until_date)
                    pick_details = get_graded_picks_detail(sport=sport_name, since=graded_date, until=until_date)
                    send_results_to_discord(summary, sport_name, breakdown, graded_date, pick_details=pick_details)

            # All-time results channel — only post if at least one sport had graded picks
            from betting_agent.notifications.discord import send_alltime_to_discord
            if sports_with_results and is_discord_configured("ALLTIME", "RESULTS"):
                logger.info("Sending all-time results to Discord...")
                alltime = {s: get_summary(sport=s, until=until_date) for s in sport_list}
                send_alltime_to_discord(alltime, settings.starting_bankroll, as_of_date=graded_date)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
