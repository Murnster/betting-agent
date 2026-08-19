"""
Bankroll ledger: running equity reconstructed from graded picks.

The ledger is a view over the picks table, not a separate store — equity at
any point is starting_bankroll plus the cumulative P&L of every graded pick
up to it. Stakes and prices come from Pick.stake / Pick.price, so recording
actual_bet / actual_odds on a pick makes the ledger reflect the bet actually
placed rather than the recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from betting_agent.config import settings
from betting_agent.db.models import Game, Pick
from betting_agent.db.session import get_session


@dataclass
class LedgerEntry:
    pick_id: int
    event_date: date
    sport: str
    bet_type: str
    pick_side: str
    stake: float
    price: int
    result: str
    pnl: float
    equity: float           # bankroll after this pick settled
    clv: float | None


def equity_curve(
    sport: str | None = None,
    starting_bankroll: float | None = None,
) -> list[LedgerEntry]:
    """
    Every graded pick in settlement order with running equity.
    Settlement order is game date, then graded_at, then pick id — stable and
    reproducible even when several picks grade in one run.
    """
    if starting_bankroll is None:
        starting_bankroll = settings.starting_bankroll

    with get_session() as session:
        q = (
            session.query(Pick, Game)
            .join(Game)
            .filter(Pick.result.isnot(None))
        )
        if sport:
            q = q.filter(Pick.sport == sport)
        rows = q.all()
        entries_raw = [
            {
                "pick_id": pick.id,
                "event_date": game.game_date,
                "sport": pick.sport,
                "bet_type": pick.bet_type,
                "pick_side": pick.pick_side,
                "stake": pick.stake,
                "price": pick.price,
                "result": pick.result,
                "pnl": pick.pnl or 0.0,
                "graded_at": pick.graded_at,
                "clv": pick.clv,
            }
            for pick, game in rows
        ]

    entries_raw.sort(
        key=lambda e: (e["event_date"], e["graded_at"] or date.min, e["pick_id"])
    )

    equity = starting_bankroll
    entries: list[LedgerEntry] = []
    for e in entries_raw:
        equity += e["pnl"]
        entries.append(LedgerEntry(
            pick_id=e["pick_id"],
            event_date=e["event_date"],
            sport=e["sport"],
            bet_type=e["bet_type"],
            pick_side=e["pick_side"],
            stake=e["stake"],
            price=e["price"],
            result=e["result"],
            pnl=e["pnl"],
            equity=round(equity, 2),
            clv=e["clv"],
        ))
    return entries


def current_bankroll(sport: str | None = None) -> float:
    """Starting bankroll plus all graded P&L."""
    curve = equity_curve(sport=sport)
    return curve[-1].equity if curve else settings.starting_bankroll


def ledger_summary(sport: str | None = None) -> dict[str, Any]:
    """Headline equity numbers for reports."""
    curve = equity_curve(sport=sport)
    start = settings.starting_bankroll
    if not curve:
        return {
            "starting_bankroll": start,
            "current_bankroll": start,
            "total_pnl": 0.0,
            "peak_equity": start,
            "max_drawdown": 0.0,
            "settled_picks": 0,
        }
    peak = start
    max_dd = 0.0
    for entry in curve:
        peak = max(peak, entry.equity)
        max_dd = max(max_dd, peak - entry.equity)
    return {
        "starting_bankroll": start,
        "current_bankroll": curve[-1].equity,
        "total_pnl": round(curve[-1].equity - start, 2),
        "peak_equity": round(peak, 2),
        "max_drawdown": round(max_dd, 2),
        "settled_picks": len(curve),
    }
