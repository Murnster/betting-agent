#!/usr/bin/env python
"""
Entry point: grade yesterday's picks and update CLV/ROI.

Usage:
    uv run python scripts/grade.py [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

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

    # props.py writes its Game rows as "scheduled" (the Odds API event carries
    # no score), and grading skips any pick whose game is not final. Fill the
    # scores in from published schedules first, or nothing ever settles.
    from betting_agent.sports.nfl.results import finalize_nfl_games

    finalized = finalize_nfl_games(target_date=target_date)
    if finalized:
        print(f"Finalized {finalized} NFL games from published schedules.")

    from betting_agent.accounting.grader import grade_prop_picks
    from betting_agent.sports.nfl.props import make_stat_lookup
    from betting_agent.sports.registry import get_sport_config

    season = get_sport_config("NFL").season_for_date(date.today())
    lookup = make_stat_lookup([season])
    return grade_prop_picks(lookup, sport="NFL", target_date=target_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade picks and print ROI report")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport")
    parser.add_argument("--season", type=int, default=None, help="Filter by season")
    parser.add_argument("--date", type=str, default=None, help="Re-grade a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    # Picks graded from here on are what today's results post reports —
    # an NFL pick is made days before its game and graded days after, so a
    # pick_date window would miss them.
    run_started = datetime.utcnow()

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
            from betting_agent.accounting.roi import (
                LEAN_BET_TYPES,
                get_breakdown_by_bet_type,
                get_graded_picks_detail,
                get_summary,
            )
            from betting_agent.notifications.discord import (
                is_discord_configured,
                send_alltime_to_discord,
                send_results_to_discord,
            )
            from betting_agent.sports.registry import available_sports

            graded_date = target_date or date.today()
            sport_list = [args.sport.upper()] if args.sport else available_sports()
            until_date = target_date  # re-grades cap all-time queries at that date
            # Sports whose all-time recap rides inside the daily results post
            # instead of the shared all-time channel (NFL is the live strategy).
            inline_alltime = {"NFL"}
            sports_with_results = []
            for sport_name in sport_list:
                if not is_discord_configured(sport_name, "RESULTS"):
                    continue
                window = {"graded_since": run_started}
                summary = get_summary(sport=sport_name, **window)
                if "total_bets" not in summary:
                    continue  # nothing settled for this sport in this run
                sports_with_results.append(sport_name)
                lean_summary = None
                if sport_name == "NFL":
                    # Props are the picks; game markets are paper leans.
                    prop_summary = get_summary(sport=sport_name, bet_type="prop", **window)
                    lean_summary = get_summary(sport=sport_name, bet_type=list(LEAN_BET_TYPES), **window)
                    summary = prop_summary if "total_bets" in prop_summary else summary
                breakdown = get_breakdown_by_bet_type(sport=sport_name, **window)
                pick_details = get_graded_picks_detail(sport=sport_name, **window)
                kwargs = {}
                if sport_name in inline_alltime:
                    kwargs = {
                        "alltime_summary": get_summary(sport=sport_name, bet_type="prop", until=until_date)
                        if sport_name == "NFL" else get_summary(sport=sport_name, until=until_date),
                        "alltime_lean_summary": get_summary(sport=sport_name, bet_type=list(LEAN_BET_TYPES),
                                                            until=until_date) if sport_name == "NFL" else None,
                        "starting_bankroll": settings.starting_bankroll,
                    }
                logger.info("Sending results to Discord (%s)...", sport_name)
                send_results_to_discord(summary, sport_name, breakdown, graded_date,
                                        pick_details=pick_details, lean_summary=lean_summary, **kwargs)

            # Shared all-time channel: everything except the inline sports.
            shared = [s for s in sports_with_results if s not in inline_alltime]
            if shared and is_discord_configured("ALLTIME", "RESULTS"):
                logger.info("Sending all-time results to Discord...")
                alltime = {s: get_summary(sport=s, until=until_date) for s in shared}
                send_alltime_to_discord(alltime, settings.starting_bankroll, as_of_date=graded_date)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
