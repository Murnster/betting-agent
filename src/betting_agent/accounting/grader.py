"""
Grade previous picks against final scores.
Sets result (win/loss/push) and pnl on Pick rows.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from betting_agent.db.models import Game, Pick
from betting_agent.db.queries import get_ungraded_picks
from betting_agent.db.session import get_session
from betting_agent.sports.teams import same_team

logger = logging.getLogger(__name__)


def _side_of_game(pick: Pick, game: Game, team: str) -> str | None:
    """
    Return "home" or "away" for the team a pick is on, or None when it matches
    neither. Picks store Odds API club names while loader-seeded games store
    abbreviations, so the comparison has to go through the canonical form.
    """
    sport = pick.sport or game.sport
    if same_team(sport, team, game.home_team):
        return "home"
    if same_team(sport, team, game.away_team):
        return "away"
    logger.warning(
        "Pick %s is on '%s', which matches neither %s nor %s (game %s) — "
        "cannot grade",
        pick.id, team, game.home_team, game.away_team, game.id,
    )
    return None


def _grade_moneyline(pick: Pick, game: Game) -> str | None:
    """Win if pick_side team won outright."""
    if game.home_score is None or game.away_score is None:
        return "push"
    if game.home_score == game.away_score:
        return "push"
    side = _side_of_game(pick, game, pick.pick_side)
    if side is None:
        return None
    won = "home" if game.home_score > game.away_score else "away"
    return "win" if side == won else "loss"


def _grade_spread(pick: Pick, game: Game) -> str | None:
    """pick_side is like 'Kansas City Chiefs +3.5' — parse the line and evaluate."""
    if game.home_score is None or game.away_score is None:
        return "push"
    try:
        parts = pick.pick_side.rsplit(" ", 1)
        team = parts[0]
        spread = float(parts[1])
    except (ValueError, IndexError):
        logger.warning("Could not parse spread pick_side: %s", pick.pick_side)
        return None

    side = _side_of_game(pick, game, team)
    if side is None:
        return None

    if side == "home":
        margin = game.home_score - game.away_score
    else:
        margin = game.away_score - game.home_score

    adjusted = margin + spread
    if adjusted > 0:
        return "win"
    elif adjusted < 0:
        return "loss"
    else:
        return "push"


def _grade_total(pick: Pick, game: Game) -> str | None:
    """pick_side is 'over 45.5' or 'under 45.5' — the line is stored inline."""
    if game.home_score is None or game.away_score is None:
        return "push"
    total = game.home_score + game.away_score

    parts = pick.pick_side.split()
    side = parts[0].lower() if parts else ""
    if side not in ("over", "under") or len(parts) != 2:
        logger.warning("Could not parse total pick_side: %s", pick.pick_side)
        return None
    try:
        line = float(parts[1])
    except ValueError:
        logger.warning("Could not parse total line from: %s", pick.pick_side)
        return None

    if total == line:
        return "push"
    if side == "over":
        return "win" if total > line else "loss"
    return "win" if total < line else "loss"


def _calculate_pnl(pick: Pick, result: str) -> float:
    """Calculate P&L for a graded pick."""
    if result == "push":
        return 0.0
    bet = pick.recommended_bet or 0.0
    if result == "win":
        odds = pick.odds
        if odds > 0:
            return bet * (odds / 100.0)
        else:
            return bet * (100.0 / abs(odds))
    else:  # loss
        return -bet


def grade_picks(target_date: date | None = None) -> int:
    """
    Grade all ungraded picks for games with final scores.
    When target_date is provided, reset and re-grade only that date's picks.
    Returns the number of picks graded.
    """
    graded = 0
    ungradable = 0
    with get_session() as session:
        if target_date is not None:
            # Reset all picks for the target date so they can be re-graded
            picks_to_reset = (
                session.query(Pick)
                .filter(Pick.pick_date == target_date)
                .all()
            )
            for pick in picks_to_reset:
                pick.result = None
                pick.pnl = None
                pick.graded_at = None
                pick.clv = None
                pick.closing_odds = None
            session.flush()
            logger.info("Reset %d picks for %s", len(picks_to_reset), target_date)

        picks = get_ungraded_picks(session, pick_date=target_date)
        for pick in picks:
            game: Game = pick.game
            if game is None or game.status != "final":
                continue

            if pick.bet_type == "moneyline":
                result = _grade_moneyline(pick, game)
            elif pick.bet_type == "spread":
                result = _grade_spread(pick, game)
            elif pick.bet_type == "total":
                result = _grade_total(pick, game)
            else:
                logger.warning(
                    "No grader for bet_type '%s' (pick %s) — left ungraded",
                    pick.bet_type, pick.id,
                )
                result = None

            # None means "could not be determined". Leave the pick ungraded
            # rather than recording a push, which would book a real bet as a
            # break-even and quietly corrupt the ledger.
            if result is None:
                ungradable += 1
                continue

            pick.result = result
            pick.pnl = _calculate_pnl(pick, result)
            pick.graded_at = datetime.utcnow()
            graded += 1

    if ungradable:
        logger.warning(
            "%d picks could not be graded — see warnings above", ungradable
        )
    logger.info("Graded %d picks", graded)
    return graded
