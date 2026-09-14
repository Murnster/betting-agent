"""
Settling parlays: the parent is the AND of its legs.

Legs are ordinary Pick rows (strategy "parlay_leg", stake 0) that the normal
graders settle from player stats and final scores. Once every leg of a
parlay is graded the parent settles by the book convention: any losing leg
loses the ticket; a pushed or voided leg drops out and the ticket pays on
the legs that won; a ticket whose legs all dropped out is void. A ticket
with an ungraded leg stays open — week N's stats parquet can lag a day.
"""

from __future__ import annotations

import logging
from datetime import datetime

from betting_agent.db.models import Pick
from betting_agent.db.session import get_session
from betting_agent.intelligence.parlay import (
    PARLAY_LEG_STRATEGY,
    PARLAY_STRATEGY,
    american_to_decimal,
)

logger = logging.getLogger(__name__)


def settle(stake: float, legs: list[tuple[str, int]]) -> tuple[str, float] | None:
    """
    (result, pnl) for a parlay from its legs' (result, american odds), or
    None while a leg is still ungraded.
    """
    if any(r is None for r, _ in legs):
        return None
    if any(r == "loss" for r, _ in legs):
        return "loss", -stake
    winners = [odds for r, odds in legs if r == "win"]
    if not winners:
        return "void", 0.0
    dec = 1.0
    for odds in winners:
        dec *= american_to_decimal(odds)
    return "win", round(stake * (dec - 1), 2)


def settle_parlays(sport: str = "NFL") -> int:
    """Settle every open parlay whose legs are all graded. Returns the count."""
    settled = 0
    with get_session() as session:
        parents = (
            session.query(Pick)
            .filter(Pick.sport == sport, Pick.strategy == PARLAY_STRATEGY,
                    Pick.result.is_(None))
            .all()
        )
        for parent in parents:
            legs = (
                session.query(Pick)
                .filter(Pick.parlay_id == parent.id, Pick.strategy == PARLAY_LEG_STRATEGY)
                .all()
            )
            if not legs:
                logger.warning("Parlay %s has no legs — left open", parent.id)
                continue
            outcome = settle(parent.stake, [(leg.result, leg.price) for leg in legs])
            if outcome is None:
                continue
            parent.result, parent.pnl = outcome
            parent.graded_at = datetime.utcnow()
            settled += 1
    if settled:
        logger.info("Settled %d parlay(s)", settled)
    return settled


def graded_parlays(sport: str = "NFL", graded_since=None, graded_until=None,
                   since=None, until=None) -> list[dict]:
    """Settled parlays with their legs, for the results post."""
    with get_session() as session:
        q = (session.query(Pick)
             .filter(Pick.sport == sport, Pick.strategy == PARLAY_STRATEGY,
                     Pick.result.isnot(None)))
        if graded_since is not None:
            q = q.filter(Pick.graded_at >= graded_since)
        if graded_until is not None:
            q = q.filter(Pick.graded_at <= graded_until)
        if since is not None:
            q = q.filter(Pick.pick_date >= since)
        if until is not None:
            q = q.filter(Pick.pick_date <= until)
        out = []
        for parent in q.order_by(Pick.pick_date, Pick.id).all():
            legs = (session.query(Pick).filter(Pick.parlay_id == parent.id)
                    .order_by(Pick.id).all())
            out.append({
                "id": parent.id, "pick_date": parent.pick_date, "label": parent.pick_side,
                "market": parent.market, "odds": parent.price, "stake": parent.stake,
                "result": parent.result, "pnl": parent.pnl or 0.0,
                "model_prob": parent.model_prob, "implied_prob": parent.implied_prob,
                "home_team": parent.game.home_team if parent.game else "",
                "away_team": parent.game.away_team if parent.game else "",
                "legs": [{
                    "bet_type": leg.bet_type, "pick_side": leg.pick_side, "player": leg.player,
                    "market": leg.market, "line": leg.line, "odds": leg.price,
                    "result": leg.result, "model_prob": leg.model_prob,
                    "home_team": leg.game.home_team if leg.game else "",
                    "away_team": leg.game.away_team if leg.game else "",
                } for leg in legs],
            })
        return out
