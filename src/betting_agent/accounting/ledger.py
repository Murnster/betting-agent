"""
Bankroll ledger: running equity reconstructed from graded picks.

The ledger is a view over the picks table, not a separate store — equity at
any point is starting_bankroll plus the cumulative P&L of every graded pick
up to it. Stakes and prices come from Pick.stake / Pick.price, so recording
actual_bet / actual_odds on a pick makes the ledger reflect the bet actually
placed rather than the recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from betting_agent.config import settings
from betting_agent.db.models import Game, Pick
from betting_agent.db.session import get_session

#: Pick.strategy of the ladder-hits section — a separate paper book with its
#: own bankroll (settings.ladder_bankroll). NULL strategy = the default book.
LADDER_STRATEGY = "ladder"
#: Pick.strategy of the straight-overs section (main-line Overs, own bankroll).
OVERS_STRATEGY = "overs"
#: Every side book; the main prop summaries exclude all of them.
SIDE_BOOKS: tuple[str, ...] = (LADDER_STRATEGY, OVERS_STRATEGY)


def starting_bankroll_for(strategy: str | None) -> float:
    if strategy == LADDER_STRATEGY:
        return settings.ladder_bankroll
    if strategy == OVERS_STRATEGY:
        return settings.overs_bankroll
    return settings.starting_bankroll


def strategy_exclusion(column, exclude: str | Sequence[str] | None):
    """SQLAlchemy clause keeping NULL-strategy rows and dropping `exclude`
    (one strategy or several); None when nothing is excluded."""
    if not exclude:
        return None
    names = [exclude] if isinstance(exclude, str) else list(exclude)
    return (column.is_(None)) | (column.notin_(names))


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
    strategy: str | None = None,
    exclude_strategy: str | Sequence[str] | None = None,
) -> list[LedgerEntry]:
    """
    Every graded pick in settlement order with running equity.
    Settlement order is game date, then graded_at, then pick id — stable and
    reproducible even when several picks grade in one run.

    `strategy` restricts the curve to one paper book (the ladder section
    starts from settings.ladder_bankroll); `exclude_strategy` drops it.
    """
    if starting_bankroll is None:
        starting_bankroll = starting_bankroll_for(strategy)

    with get_session() as session:
        q = (
            session.query(Pick, Game)
            .join(Game)
            .filter(Pick.result.isnot(None))
        )
        if sport:
            q = q.filter(Pick.sport == sport)
        if strategy:
            q = q.filter(Pick.strategy == strategy)
        clause = strategy_exclusion(Pick.strategy, exclude_strategy)
        if clause is not None:
            q = q.filter(clause)
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


def current_bankroll(sport: str | None = None, strategy: str | None = None,
                     exclude_strategy: str | Sequence[str] | None = None) -> float:
    """Starting bankroll plus all graded P&L."""
    curve = equity_curve(sport=sport, strategy=strategy, exclude_strategy=exclude_strategy)
    return curve[-1].equity if curve else starting_bankroll_for(strategy)


def ledger_summary(sport: str | None = None, strategy: str | None = None,
                   exclude_strategy: str | Sequence[str] | None = None) -> dict[str, Any]:
    """Headline equity numbers for reports."""
    curve = equity_curve(sport=sport, strategy=strategy, exclude_strategy=exclude_strategy)
    start = starting_bankroll_for(strategy)
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
