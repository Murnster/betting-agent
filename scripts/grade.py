#!/usr/bin/env python
"""
Entry point: grade yesterday's picks and update CLV/ROI.

Usage:
    uv run python scripts/grade.py [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time

from betting_agent.accounting.clv import update_clv_for_picks
from betting_agent.accounting.grader import grade_picks
from betting_agent.accounting.roi import carded_only, format_roi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _pending_nfl(bet_type: str | None = None) -> int:
    """Ungraded NFL picks, optionally of one bet type."""
    from betting_agent.db.models import Pick
    from betting_agent.db.session import get_session

    with get_session() as session:
        q = session.query(Pick).filter(Pick.result.is_(None), Pick.sport == "NFL")
        if bet_type:
            q = q.filter(Pick.bet_type == bet_type)
        return q.count()


def _finalize_nfl(target_date: date | None) -> int:
    """
    Fill in scores for NFL games that still say "scheduled".

    props.py writes its Game rows as "scheduled" (the Odds API event carries no
    score) and BOTH graders skip a pick whose game is not final, so this has to
    run before either of them. It used to sit inside the prop grader, which
    left the game leans a full day behind: grade_picks() ran first against a
    still-"scheduled" game, skipped the lean, and only then did the prop path
    finalize it. The Sep 9 2026 Seahawks lean settled a day late for exactly
    that reason. Free — published schedules via nflreadpy.
    """
    if not _pending_nfl():
        return 0
    from betting_agent.sports.nfl.results import finalize_nfl_games

    finalized = finalize_nfl_games(target_date=target_date)
    if finalized:
        print(f"Finalized {finalized} NFL games from published schedules.")
    return finalized


def _grade_nfl_props(target_date: date | None) -> int:
    """Grade ungraded NFL prop picks, loading player stats only when needed."""
    if not _pending_nfl("prop"):
        return 0

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
    parser.add_argument("--repost", type=str, default=None, metavar="YYYY-MM-DD",
                        help="Re-post the results for a day already graded, without "
                             "re-grading anything. Use it after correcting which picks "
                             "count (Pick.on_card) so the channel matches the ledger.")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    repost_date = date.fromisoformat(args.repost) if args.repost else None
    # Picks graded from here on are what today's results post reports —
    # an NFL pick is made days before its game and graded days after, so a
    # pick_date window would miss them. A re-post instead bounds graded_at to
    # a day that has already been graded, so it reports exactly the same picks
    # the original post covered — re-read through the current on_card flags.
    run_started = datetime.utcnow()
    grade_window: dict = (
        {"graded_since": datetime.combine(repost_date, time.min),
         "graded_until": datetime.combine(repost_date, time.max)}
        if repost_date else {"graded_since": run_started}
    )

    if repost_date:
        print(f"Re-posting results graded on {repost_date} (no grading, no API calls).")
    else:
        # Scores first — both graders gate on Game.status == "final".
        try:
            _finalize_nfl(target_date)
        except Exception as exc:
            logger.warning("NFL finalization failed: %s", exc)

        try:
            n_graded = grade_picks(target_date=target_date)
            print(f"\nGraded {n_graded} picks.")
        except Exception as exc:
            logger.error("Grading failed (DB may not be running): %s", exc)

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
        report = format_roi_report(sport=args.sport, season=args.season, since=target_date,
                                   on_card=carded_only(args.sport))
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
                send_extras_results_to_discord,
                send_results_to_discord,
            )
            from betting_agent.accounting.ledger import LADDER_STRATEGY, OVERS_STRATEGY, SIDE_BOOKS
            from betting_agent.sports.nfl.td_props import TD_MARKET
            from betting_agent.sports.registry import available_sports

            graded_date = repost_date or target_date or date.today()
            sport_list = [args.sport.upper()] if args.sport else available_sports()
            until_date = target_date  # re-grades cap all-time queries at that date
            # Sports whose all-time recap rides inside the daily results post
            # instead of the shared all-time channel (NFL is the live strategy).
            inline_alltime = {"NFL"}
            sports_with_results = []
            for sport_name in sport_list:
                if not is_discord_configured(sport_name, "RESULTS"):
                    continue
                # The card is the offer: for NFL the results post and the
                # all-time recap count only the picks that made a card.
                # props.py also saves everything else that clears the floors,
                # and those keep grading, but they were never bettable so they
                # stay out of the record, the P&L and the bankroll.
                window = {**grade_window, "on_card": carded_only(sport_name)}
                alltime = {"until": until_date, "on_card": carded_only(sport_name)}
                summary = get_summary(sport=sport_name, **window)
                if "total_bets" not in summary:
                    continue  # nothing settled for this sport in this run
                sports_with_results.append(sport_name)
                lean_summary = td_summary = ladder_summary = overs_summary = None
                if sport_name == "NFL":
                    # Receiving props are the picks; game markets are paper
                    # leans; anytime-TD scorers, the ladder hits and the
                    # straight overs (own bankrolls) get their own lines.
                    main = {"exclude_market": TD_MARKET, "exclude_strategy": SIDE_BOOKS}
                    prop_summary = get_summary(sport=sport_name, bet_type="prop", **main, **window)
                    lean_summary = get_summary(sport=sport_name, bet_type=list(LEAN_BET_TYPES), **window)
                    td_summary = get_summary(sport=sport_name, bet_type="prop", market=TD_MARKET, **window)
                    ladder_summary = get_summary(sport=sport_name, bet_type="prop",
                                                 strategy=LADDER_STRATEGY, **window)
                    overs_summary = get_summary(sport=sport_name, bet_type="prop",
                                                strategy=OVERS_STRATEGY, **window)
                    summary = prop_summary if "total_bets" in prop_summary else summary
                breakdown = get_breakdown_by_bet_type(sport=sport_name, **window)
                pick_details = get_graded_picks_detail(sport=sport_name, **window)
                kwargs = {}
                if sport_name in inline_alltime:
                    kwargs = {
                        "alltime_summary": get_summary(sport=sport_name, bet_type="prop", **main,
                                                       **alltime)
                        if sport_name == "NFL" else get_summary(sport=sport_name, **alltime),
                        "alltime_lean_summary": get_summary(sport=sport_name, bet_type=list(LEAN_BET_TYPES),
                                                            **alltime) if sport_name == "NFL" else None,
                        "alltime_td_summary": get_summary(sport=sport_name, bet_type="prop", market=TD_MARKET,
                                                          **alltime) if sport_name == "NFL" else None,
                        "alltime_ladder_summary": get_summary(sport=sport_name, bet_type="prop",
                                                              strategy=LADDER_STRATEGY, **alltime)
                        if sport_name == "NFL" else None,
                        "alltime_overs_summary": get_summary(sport=sport_name, bet_type="prop",
                                                             strategy=OVERS_STRATEGY, **alltime)
                        if sport_name == "NFL" else None,
                        "starting_bankroll": settings.starting_bankroll,
                        "ladder_bankroll": settings.ladder_bankroll,
                        "overs_bankroll": settings.overs_bankroll,
                    }
                logger.info("Sending results to Discord (%s)...", sport_name)
                send_results_to_discord(summary, sport_name, breakdown, graded_date,
                                        pick_details=pick_details, lean_summary=lean_summary,
                                        td_summary=td_summary, ladder_summary=ladder_summary,
                                        overs_summary=overs_summary, **kwargs)

                # The off-card props, in their own channel and their own
                # bankroll — the same filters as the main prop summary above,
                # inverted on on_card. Never posted to the results channel.
                if (sport_name == "NFL" and settings.extras_enabled
                        and is_discord_configured(sport_name, "EXTRAS_RESULTS")):
                    ex = {**main, "bet_type": "prop", "on_card": False}
                    extras_summary = get_summary(sport=sport_name, **ex, **grade_window)
                    if "total_bets" in extras_summary:
                        # The detail query has no market/strategy filters, so
                        # narrow the list the same way the summary is narrowed
                        # — the side books' own off-card leftovers stay in
                        # their own books, not in the extras headline.
                        extras_detail = [
                            d for d in get_graded_picks_detail(
                                sport=sport_name, on_card=False, **grade_window)
                            if d.get("bet_type") == "prop"
                            and d.get("market") != TD_MARKET
                            and d.get("strategy") not in SIDE_BOOKS
                        ]
                        logger.info("Sending extras results to Discord (%s)...", sport_name)
                        send_extras_results_to_discord(
                            extras_summary, sport_name, graded_date,
                            pick_details=extras_detail,
                            alltime_summary=get_summary(sport=sport_name, **ex, until=until_date),
                            starting_bankroll=settings.extras_bankroll,
                        )

            # Shared all-time channel: everything except the inline sports.
            shared = [s for s in sports_with_results if s not in inline_alltime]
            if shared and is_discord_configured("ALLTIME", "RESULTS"):
                logger.info("Sending all-time results to Discord...")
                alltime = {s: get_summary(sport=s, until=until_date, on_card=carded_only(s))
                           for s in shared}
                send_alltime_to_discord(alltime, settings.starting_bankroll, as_of_date=graded_date)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
