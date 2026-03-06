#!/usr/bin/env python
"""
Entry point: grade yesterday's picks and update CLV/ROI.

Usage:
    uv run python scripts/grade.py [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from betting_agent.accounting.clv import update_clv_for_picks
from betting_agent.accounting.grader import grade_picks
from betting_agent.accounting.roi import format_roi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade picks and print ROI report")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport")
    parser.add_argument("--season", type=int, default=None, help="Filter by season")
    args = parser.parse_args()

    try:
        n_graded = grade_picks()
        print(f"\nGraded {n_graded} picks.")
    except Exception as exc:
        logger.error("Grading failed (DB may not be running): %s", exc)
        n_graded = 0

    try:
        n_clv = update_clv_for_picks()
        print(f"Updated CLV for {n_clv} picks.")
    except Exception as exc:
        logger.warning("CLV update failed: %s", exc)

    try:
        report = format_roi_report(sport=args.sport, season=args.season)
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
            from datetime import timedelta

            graded_date = date.today() - timedelta(days=1)
            sport_list = [args.sport.upper()] if args.sport else available_sports()
            sports_with_results = []
            for sport_name in sport_list:
                if is_discord_configured(sport_name, "RESULTS"):
                    summary = get_summary(sport=sport_name, since=graded_date)
                    if "total_bets" not in summary:
                        continue  # no graded picks for this sport yesterday
                    sports_with_results.append(sport_name)
                    logger.info("Sending results to Discord (%s)...", sport_name)
                    breakdown = get_breakdown_by_bet_type(sport=sport_name, since=graded_date)
                    pick_details = get_graded_picks_detail(sport=sport_name, since=graded_date)
                    send_results_to_discord(summary, sport_name, breakdown, graded_date, pick_details=pick_details)

            # All-time results channel — only post if at least one sport had graded picks
            from betting_agent.notifications.discord import send_alltime_to_discord
            if sports_with_results and is_discord_configured("ALLTIME", "RESULTS"):
                logger.info("Sending all-time results to Discord...")
                alltime = {s: get_summary(sport=s) for s in sport_list}
                send_alltime_to_discord(alltime, settings.starting_bankroll)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
