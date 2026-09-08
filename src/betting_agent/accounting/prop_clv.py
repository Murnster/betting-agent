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
from betting_agent.sports.nfl.td_props import TD_MARKET, yes_outcomes

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
            if market == TD_MARKET:
                # Yes-only board, no line: the closing quote is the Yes price.
                for player, outcome in yes_outcomes(mkt).items():
                    if normalize_player(player) == player_key:
                        return float(pick_line), int(outcome["price"])
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


def _game_closing_quote(event: dict, pick, book_order: list[str] | None) -> tuple[float | None, int] | None:
    """
    (closing line, closing price) for a moneyline/spread/total pick from a
    sport-level odds response, read at the first book in `book_order` that
    quotes the game (the same rule the lean was priced with).
    """
    from betting_agent.sports.teams import same_team

    books = books_in_preference(event, book_order)
    if not books:
        return None
    book = books[0]
    home, away = event.get("home_team", ""), event.get("away_team", "")
    side = (pick.pick_side or "").strip()
    for mkt in book.get("markets", []):
        key = mkt.get("key")
        if pick.bet_type == "moneyline" and key == "h2h":
            for o in mkt.get("outcomes", []):
                if same_team(pick.sport, side, o.get("name", "")) and o.get("price") is not None:
                    return None, int(o["price"])
        elif pick.bet_type == "spread" and key == "spreads":
            team = side.rsplit(" ", 1)[0]
            for o in mkt.get("outcomes", []):
                if same_team(pick.sport, team, o.get("name", "")) and o.get("price") is not None:
                    return float(o.get("point")), int(o["price"])
        elif pick.bet_type == "total" and key == "totals":
            want = "Over" if side.lower().startswith("over") else "Under"
            for o in mkt.get("outcomes", []):
                if o.get("name") == want and o.get("price") is not None:
                    return float(o.get("point")), int(o["price"])
    _ = (home, away)
    return None


def capture_game_closing_lines(events: list[dict], picks: list,
                               book_order: list[str] | None = None) -> int:
    """Store closing line/price (and clv when the line held) on moneyline,
    spread and total picks whose game appears in `events`."""
    by_event = {e.get("id"): e for e in events if e.get("id")}
    updated = 0
    for pick in picks:
        game = getattr(pick, "game", None)
        event = by_event.get(getattr(game, "external_id", None))
        if event is None:
            continue
        quote = _game_closing_quote(event, pick, book_order)
        if quote is None:
            logger.info("No closing quote for %s %s", pick.bet_type, pick.pick_side)
            continue
        closing_line, closing_price = quote
        pick.closing_line = closing_line
        pick.closing_odds = closing_price
        line_held = pick.line is None or closing_line is None or float(closing_line) == float(pick.line)
        pick.clv = calculate_clv(int(pick.odds), closing_price) if line_held else None
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

    updated = 0
    with get_session() as session:
        open_picks = (
            session.query(Pick)
            .join(Game)
            .filter(Pick.sport == "NFL")
            .filter(Pick.result.is_(None), Pick.closing_odds.is_(None))
            .filter(Game.external_id.in_(event_ids))
            .all()
        )
        props = [p for p in open_picks if p.bet_type == "prop"]
        games = [p for p in open_picks if p.bet_type in ("moneyline", "spread", "total")]
        if not open_picks:
            logger.info("No open picks in the next %d minutes — nothing fetched", window_minutes)
            return 0
        if props:
            held = {p.game.external_id for p in props}
            targets = [e for e in upcoming if e.get("id") in held]
            markets = sorted({p.market for p in props if p.market})
            logger.info("Capturing closing lines for %d prop picks across %d games (%d credits)",
                        len(props), len(targets), len(targets) * len(markets))
            odds = fetch_prop_odds(sport_key, markets=markets, bookmakers=bookmakers, events=targets)
            updated += capture_prop_closing_lines(odds, props, bookmakers)
        if games:
            from betting_agent.intelligence.game_lean import REFERENCE_BOOK, fetch_game_lines

            held = {p.game.external_id for p in games}
            targets = [e for e in upcoming if e.get("id") in held]
            books = list(bookmakers or []) + [REFERENCE_BOOK]
            logger.info("Capturing closing lines for %d game leans across %d games (3 credits)",
                        len(games), len(targets))
            odds = fetch_game_lines(targets, books, sport_key=sport_key)
            updated += capture_game_closing_lines(odds, games, bookmakers)
    logger.info("Stored closing lines for %d picks", updated)
    return updated
