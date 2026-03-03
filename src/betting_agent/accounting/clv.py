"""
Closing Line Value (CLV) tracking.
CLV = odds we got at pick time vs closing line.
Positive CLV means we got better odds than the closing line.
"""

from __future__ import annotations

import logging

from betting_agent.db.models import Odds, Pick
from betting_agent.db.queries import get_closing_odds
from betting_agent.db.session import get_session
from betting_agent.intelligence.ev import american_to_implied_prob

logger = logging.getLogger(__name__)


def calculate_clv(pick_odds: int, closing_odds: int) -> float:
    """
    CLV in implied probability points.
    Positive = we got a better price than closing.

    Example: pick at +120, closes at +100
    Closing implied = 0.5, pick implied = 0.455 → CLV = 0.5 - 0.455 = +0.045
    """
    closing_implied = american_to_implied_prob(closing_odds)
    pick_implied = american_to_implied_prob(pick_odds)
    # Lower implied probability = better odds, so positive CLV = closing_implied > pick_implied
    return closing_implied - pick_implied


def update_clv_for_picks() -> int:
    """
    For all picks with no CLV yet that have closing odds in the DB,
    calculate and store CLV.
    Returns the number of picks updated.
    """
    updated = 0
    with get_session() as session:
        picks = (
            session.query(Pick)
            .filter(Pick.clv.is_(None))
            .filter(Pick.result.isnot(None))
            .all()
        )
        for pick in picks:
            closing = get_closing_odds(session, pick.game_id, pick.bet_type)
            if closing is None:
                continue

            # Determine which price to use (home or away based on pick_side)
            if pick.pick_side.startswith("over"):
                closing_price = closing.home_price
            elif pick.pick_side.startswith("under"):
                closing_price = closing.away_price
            elif closing.home_price and pick.pick_side.startswith(pick.game.home_team):
                closing_price = closing.home_price
            elif closing.away_price and pick.pick_side.startswith(pick.game.away_team):
                closing_price = closing.away_price
            else:
                closing_price = closing.away_price

            if closing_price is None:
                continue

            pick.closing_odds = closing_price
            pick.clv = calculate_clv(pick.odds, closing_price)
            updated += 1

    logger.info("Updated CLV for %d picks", updated)
    return updated
