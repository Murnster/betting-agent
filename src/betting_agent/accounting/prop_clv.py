"""
Closing-line capture for NFL player props.

Game-market CLV comes from the `odds` table (`accounting/clv.py`). Props are
only served by the per-event endpoint and are never stored there, so the
props loop snapshots the closing price itself: shortly before kickoff,
re-fetch the event's prop odds (two credits per game we hold picks in) and
store, per pick, the book's price and line for the same player/market/side.

Books move the NUMBER as well as the price. CLV is only comparable when the
line held, so `clv` is set when `closing_line == pick.line` and left NULL
otherwise; the stored `closing_line` still lets the report count moves for
and against the pick, which is the cheaper but still informative signal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from betting_agent.accounting.clv import calculate_clv
from betting_agent.sports.nfl.props import (
    books_in_preference,
    normalize_player,
    pair_outcomes,
)

logger = logging.getLogger(__name__)


def _commence(event: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(
            str(event.get("commence_time", "")).replace("Z", "+00:00")
        )
    except ValueError:
        return None


def events_kicking_off_within(
    events: list[dict], window_minutes: int, now: datetime | None = None
) -> list[dict]:
    """Events with kickoff in (now, now + window]. Started games are excluded."""
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(minutes=window_minutes)
    out = []
    for e in events:
        kick = _commence(e)
        if kick is not None and now < kick <= horizon:
            out.append(e)
    return out


def line_moved_for_pick(pick_side: str, pick_line: float, closing_line: float) -> bool | None:
    """
    True if the line moved in the pick's favour (over: number went down;
    under: number went up), False if against, None if it held.
    """
    if closing_line == pick_line:
        return None
    if (pick_side or "").lower().startswith("over"):
        return closing_line < pick_line
    return closing_line > pick_line


def _closing_quote(event: dict, player_key: str, market: str, side: str,
                   pick_line: float, book_order: list[str] | None = None,
                   ) -> tuple[float, int] | None:
    """
    (line, price) at close for this player/market/side, from the pair whose
    line is closest to the pick's (books list alternate lines). Never
    matched on the line itself — it may have moved. Reads the same book the
    pick was priced against (first of `book_order` that posted), so CLV
    compares like with like.
    """
    want = "Over" if side.lower().startswith("over") else "Under"
    best: tuple[float, float, int] | None = None   # (gap, line, price)
    for book in books_in_preference(event, book_order):
        for mkt in book.get("markets", []):
            if mkt.get("key") != market:
                continue
            for (player, line), pair in pair_outcomes(mkt).items():
                if normalize_player(player) != player_key:
                    continue
                outcome = pair.get(want)
                if not outcome or outcome.get("price") is None:
                    continue
                gap = abs(float(line) - float(pick_line))
                if best is None or gap < best[0]:
                    best = (gap, float(line), int(outcome["price"]))
    return None if best is None else (best[1], best[2])


def capture_prop_closing_lines(events: list[dict], picks: list,
                               book_order: list[str] | None = None) -> int:
    """
    Store closing_line / closing_odds (and clv when the line held) on each
    pick whose game appears in `events` (per-event odds responses). Picks are
    ORM rows or any object with game.external_id, player, market, pick_side,
    line, odds, closing_odds, closing_line, clv. Returns the count updated.
    """
    by_event = {e.get("id"): e for e in events if e.get("id")}
    updated = 0
    for pick in picks:
        game = getattr(pick, "game", None)
        event = by_event.get(getattr(game, "external_id", None))
        if event is None or pick.line is None:
            continue
        quote = _closing_quote(
            event, normalize_player(pick.player), pick.market or "", pick.pick_side or "",
            float(pick.line), book_order,
        )
        if quote is None:
            logger.info("No closing quote for %s %s %s", pick.player, pick.market, pick.pick_side)
            continue
        closing_line, closing_price = quote
        pick.closing_line = closing_line
        pick.closing_odds = closing_price
        if closing_line == float(pick.line):
            pick.clv = calculate_clv(int(pick.odds), closing_price)
        else:
            pick.clv = None
        updated += 1
    return updated


def capture_closing_lines_for_upcoming(
    window_minutes: int = 90,
    bookmakers: list[str] | None = None,
    sport_key: str = "americanfootball_nfl",
) -> int:
    """
    The `props.py --closing` code path. Free events call → keep games
    kicking off inside the window → load our ungraded prop picks for those
    games that have no closing price yet → per-event odds fetch ONLY for
    those games → store. Zero credits when there is nothing to capture.
    """
    from betting_agent.api.odds import OddsAPIClient
    from betting_agent.db.models import Game, Pick
    from betting_agent.db.session import get_session
    from betting_agent.sports.nfl.props import fetch_prop_odds

    client = OddsAPIClient()
    upcoming = events_kicking_off_within(client.fetch_events(sport_key), window_minutes)
    if not upcoming:
        logger.info("No NFL kickoffs in the next %d minutes", window_minutes)
        return 0
    event_ids = [e["id"] for e in upcoming if e.get("id")]

    with get_session() as session:
        picks = (
            session.query(Pick)
            .join(Game)
            .filter(Pick.bet_type == "prop", Pick.sport == "NFL")
            .filter(Pick.result.is_(None), Pick.closing_odds.is_(None))
            .filter(Game.external_id.in_(event_ids))
            .all()
        )
        if not picks:
            logger.info("No open prop picks in the next %d minutes — nothing fetched",
                        window_minutes)
            return 0
        held = {p.game.external_id for p in picks}
        targets = [e for e in upcoming if e.get("id") in held]
        markets = sorted({p.market for p in picks if p.market})
        logger.info("Capturing closing lines for %d picks across %d games (%d credits)",
                    len(picks), len(targets), len(targets) * len(markets))
        odds = fetch_prop_odds(sport_key, markets=markets, bookmakers=bookmakers, events=targets)
        updated = capture_prop_closing_lines(odds, picks, bookmakers)
    logger.info("Stored closing lines for %d prop picks", updated)
    return updated
