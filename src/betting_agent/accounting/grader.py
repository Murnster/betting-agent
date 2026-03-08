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

logger = logging.getLogger(__name__)


def _grade_moneyline(pick: Pick, game: Game) -> str:
    """Win if pick_side team won outright."""
    if game.home_score is None or game.away_score is None:
        return "push"
    if game.home_score == game.away_score:
        return "push"
    winner = game.home_team if game.home_score > game.away_score else game.away_team
    return "win" if pick.pick_side == winner else "loss"


def _grade_spread(pick: Pick, game: Game) -> str:
    """pick_side is like 'KC Chiefs +3.5' — parse the line and evaluate."""
    if game.home_score is None or game.away_score is None:
        return "push"
    try:
        parts = pick.pick_side.rsplit(" ", 1)
        team = parts[0]
        spread = float(parts[1])
    except (ValueError, IndexError):
        logger.warning("Could not parse spread pick_side: %s", pick.pick_side)
        return "push"

    if team == game.home_team:
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


def _grade_total(pick: Pick, game: Game) -> str:
    """pick_side is 'over' or 'under'."""
    if game.home_score is None or game.away_score is None:
        return "push"
    total = game.home_score + game.away_score
    # Find the total_line from the associated odds row
    # Fallback: compare against model-implied from pick data
    # We store total_line as part of the Odds table; for now use a heuristic
    if pick.pick_side.startswith("over"):
        # The line is encoded in the pick description — check if total > odds implied
        # Since we can't easily retrieve line here without extra join,
        # store line in pick.pick_side if we can later: "over 45.5"
        parts = pick.pick_side.split()
        if len(parts) == 2:
            try:
                line = float(parts[1])
                return "win" if total > line else ("push" if total == line else "loss")
            except ValueError:
                pass
        return "push"
    elif pick.pick_side.startswith("under"):
        parts = pick.pick_side.split()
        if len(parts) == 2:
            try:
                line = float(parts[1])
                return "win" if total < line else ("push" if total == line else "loss")
            except ValueError:
                pass
        return "push"
    return "push"


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
                result = "push"

            pick.result = result
            pick.pnl = _calculate_pnl(pick, result)
            pick.graded_at = datetime.utcnow()
            graded += 1

    logger.info("Graded %d picks", graded)
    return graded
