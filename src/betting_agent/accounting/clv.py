"""
Closing Line Value (CLV) tracking.
CLV = odds we got at pick time vs closing line.
Positive CLV means we got better odds than the closing line.
"""

from __future__ import annotations

import logging

from betting_agent.db.models import Pick
from betting_agent.db.queries import get_closing_odds
from betting_agent.db.session import get_session
from betting_agent.intelligence.ev import american_to_implied_prob
from betting_agent.sports.teams import same_team

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


def _closing_price_for_side(pick: Pick, closing) -> int | None:
    """
    The closing price for the side a pick is actually on.

    Totals store the over price in home_price and the under price in
    away_price. For sides, the team has to be matched through the canonical
    name: the old code compared the pick's Odds API spelling against whatever
    the game row happened to store and fell through to the away price on a
    mismatch, so a home pick could be scored against the opponent's line.
    """
    side_text = (pick.pick_side or "").strip()
    lowered = side_text.lower()
    if lowered.startswith("over"):
        return closing.home_price
    if lowered.startswith("under"):
        return closing.away_price

    # Spread picks carry a trailing line ("Kansas City Chiefs +3.5").
    team = side_text.rsplit(" ", 1)[0] if pick.bet_type == "spread" else side_text
    game = pick.game
    if game is None:
        return None
    sport = pick.sport or game.sport

    if same_team(sport, team, game.home_team):
        return closing.home_price
    if same_team(sport, team, game.away_team):
        return closing.away_price

    logger.warning(
        "Pick %s is on '%s', which matches neither %s nor %s — skipping CLV",
        pick.id, team, game.home_team, game.away_team,
    )
    return None


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

            closing_price = _closing_price_for_side(pick, closing)
            if closing_price is None:
                continue

            pick.closing_odds = closing_price
            pick.clv = calculate_clv(pick.odds, closing_price)
            updated += 1

    logger.info("Updated CLV for %d picks", updated)
    return updated
