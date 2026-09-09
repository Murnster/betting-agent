"""
The Odds API client.
Fetches moneyline, spread, totals and player props for any sport/market.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import requests

from betting_agent.config import settings
from betting_agent.db.models import Game, Odds
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session
from betting_agent.sports.teams import canonical_team, same_team

logger = logging.getLogger(__name__)


def _redact(text: str) -> str:
    """Strip apiKey values out of anything headed for the logs."""
    return re.sub(r"([?&]apiKey=)[^&\s]+", r"\1***", text)

# Supported markets per sport
STANDARD_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
]

# Mapping from Odds API market key → canonical bet_type
MARKET_TO_BET_TYPE: dict[str, str] = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
}


def american_to_float(price: Any) -> int | None:
    """Return price as int American odds, or None."""
    try:
        return int(price)
    except (TypeError, ValueError):
        return None


# Which configured key the process is currently spending. Each cron run is a
# fresh process, so this only saves repeat rotations within one run; the cost
# of starting over is a single 401, which does not consume a credit.
_key_index = 0


def _reset_key_rotation() -> None:
    """Start again from the primary key (tests, and anything long-running)."""
    global _key_index
    _key_index = 0


def key_failure_reason(resp: requests.Response) -> str | None:
    """
    Why this key cannot serve the request, or None if the response is not the
    key's fault. The Odds API answers an exhausted key with 401 and a message
    naming the usage quota, and 429 is its documented rate/quota status; a 401
    for any other reason means the key is revoked or mistyped. Both are worth
    rotating past — the next key may still work, and asking a dead key costs
    no credits — but they are logged differently so a config mistake does not
    hide behind an expected end-of-month rotation.
    """
    if resp.status_code == 429:
        return "quota"
    if resp.status_code != 401:
        return None
    try:
        body = resp.text.lower()
    except Exception:  # pragma: no cover - requests always gives text
        return "rejected"
    if "quota" in body or "usage limit" in body:
        return "quota"
    return "rejected"


class OddsAPIClient:
    def __init__(self):
        self.api_keys = settings.odds_api_keys
        self.base_url = settings.odds_api_base

    @property
    def api_key(self) -> str:
        """The key currently being spent (empty when none is configured)."""
        if not self.api_keys:
            return ""
        return self.api_keys[min(_key_index, len(self.api_keys) - 1)]

    def _get(self, path: str, params: dict) -> list[dict] | None:
        global _key_index
        if not self.api_keys:
            logger.warning("ODDS_API_KEY not set — skipping Odds API call")
            return None
        url = f"{self.base_url}/{path}"

        # Walk the remaining keys. A key that is out of credits costs nothing
        # to ask, so falling through the exhausted ones is free.
        while _key_index < len(self.api_keys):
            params["apiKey"] = self.api_keys[_key_index]
            n = _key_index + 1
            try:
                resp = requests.get(url, params=params, timeout=15)
            except requests.exceptions.RequestException as exc:
                logger.error("Odds API request error: %s", _redact(str(exc)))
                return None

            if resp.status_code == 422:
                logger.warning("Odds API 422 for %s — market not available", path)
                return []

            reason = key_failure_reason(resp)
            if reason is not None:
                detail = ("is out of credits" if reason == "quota"
                          else "was rejected (revoked or mistyped)")
                if _key_index + 1 < len(self.api_keys):
                    log = logger.warning if reason == "quota" else logger.error
                    log("Odds API key %d of %d %s (HTTP %d) — falling back to key %d",
                        n, len(self.api_keys), detail, resp.status_code, n + 1)
                    _key_index += 1
                    continue
                logger.error(
                    "Odds API key %d of %d %s (HTTP %d) and no fallback key is left",
                    n, len(self.api_keys), detail, resp.status_code,
                )
                return None

            if not resp.ok:
                logger.error("Odds API HTTP error: %s", _redact(str(resp.status_code)))
                return None

            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                # INFO, not DEBUG: this is the only visibility into the
                # monthly pool an unattended box has (docs/NFL_SETUP.md §7).
                logger.info("Odds API key %d of %d: %s credits remaining",
                            n, len(self.api_keys), remaining)
                # Retire a key the moment it hits zero, so the next call starts
                # on the fallback instead of spending a round trip on a 401.
                try:
                    if float(remaining) <= 0 and _key_index + 1 < len(self.api_keys):
                        logger.warning(
                            "Odds API key %d of %d hit zero credits — next call uses key %d",
                            n, len(self.api_keys), n + 1,
                        )
                        _key_index += 1
                except ValueError:
                    pass
            try:
                return resp.json()
            except ValueError:
                logger.error("Odds API returned non-JSON for %s", path)
                return None

        logger.error("Every configured Odds API key is exhausted or rejected")
        return None

    def fetch_events(self, sport_key: str) -> list[dict]:
        """Fetch all upcoming events for a sport (no odds, cheap call)."""
        data = self._get(f"{sport_key}/events", {"regions": "us"})
        return data or []

    def fetch_odds(
        self,
        sport_key: str,
        markets: list[str] | None = None,
        event_ids: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch odds for a sport.
        markets: list of market keys (h2h, spreads, totals, player_*)
        event_ids: optional list of specific event IDs to fetch
        bookmakers: optional book keys; replaces the region (≤10 books = 1 region)
        """
        if markets is None:
            markets = STANDARD_MARKETS
        params: dict[str, Any] = {
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = "us"
        if event_ids:
            params["eventIds"] = ",".join(event_ids)

        data = self._get(f"{sport_key}/odds", params)
        return data or []

    def fetch_event_odds(
        self,
        sport_key: str,
        event_id: str,
        markets: list[str],
        bookmakers: list[str] | None = None,
    ) -> dict | None:
        """
        Fetch odds for one event. Player prop markets are only available
        here, not on the sport-level /odds endpoint. Costs one API request
        per market region; passing bookmakers narrows the response (and the
        bill) to the books actually bet at.
        """
        params: dict[str, Any] = {
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = "us"
        data = self._get(f"{sport_key}/events/{event_id}/odds", params)
        # _get returns a list for collection endpoints; this one returns a dict.
        if isinstance(data, list):
            return data[0] if data else None
        return data

    def fetch_scores(self, sport_key: str, days_from: int = 3) -> list[dict]:
        """Fetch recent scores (for grading). days_from: how many days back."""
        data = self._get(f"{sport_key}/scores", {"daysFrom": days_from, "dateFormat": "iso"})
        return data or []

    def parse_and_store_odds(
        self,
        raw_games: list[dict],
        is_closing: bool = False,
        sport: str = "NFL",
    ) -> int:
        """
        Parse odds response and upsert into the odds table.
        Returns number of odds rows inserted.
        """
        count = 0
        with get_session() as session:
            for raw in raw_games:
                external_id = raw.get("id")
                game = get_game_by_external_id(session, external_id)
                if game is None:
                    # Upsert a minimal game row so FK constraint passes
                    dt_str = raw.get("commence_time", "")
                    game_date = None
                    try:
                        game_date = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).date()
                    except (ValueError, AttributeError):
                        pass

                    # Store the canonical abbreviation, not the Odds API club
                    # name — loaders seed abbreviations, and grading compares
                    # this against the pick's team.
                    game = Game(
                        sport=sport,
                        season=game_date.year if game_date else 0,
                        game_date=game_date,
                        home_team=canonical_team(sport, raw.get("home_team", "")),
                        away_team=canonical_team(sport, raw.get("away_team", "")),
                        status="scheduled",
                        external_id=external_id,
                    )
                    session.add(game)
                    session.flush()

                for bookmaker in raw.get("bookmakers", []):
                    bk_name = bookmaker.get("title", "")
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key", "")
                        bet_type = MARKET_TO_BET_TYPE.get(market_key, market_key)
                        outcomes = market.get("outcomes", [])

                        home_price = away_price = spread_home = total_line = None
                        description = None

                        if market_key == "h2h":
                            for o in outcomes:
                                p = american_to_float(o.get("price"))
                                if o.get("name") == raw.get("home_team"):
                                    home_price = p
                                elif o.get("name") == raw.get("away_team"):
                                    away_price = p

                        elif market_key == "spreads":
                            for o in outcomes:
                                p = american_to_float(o.get("price"))
                                pt = o.get("point")
                                if o.get("name") == raw.get("home_team"):
                                    home_price = p
                                    spread_home = float(pt) if pt is not None else None
                                elif o.get("name") == raw.get("away_team"):
                                    away_price = p

                        elif market_key == "totals":
                            for o in outcomes:
                                p = american_to_float(o.get("price"))
                                pt = o.get("point")
                                if pt is not None:
                                    total_line = float(pt)
                                if o.get("name") == "Over":
                                    home_price = p
                                elif o.get("name") == "Under":
                                    away_price = p

                        else:
                            # Player prop — one outcome per player
                            for o in outcomes:
                                description = o.get("description") or o.get("name")
                                home_price = american_to_float(o.get("price"))
                                total_line = float(o["point"]) if o.get("point") is not None else None
                                odds_row = Odds(
                                    game_id=game.id,
                                    bookmaker=bk_name,
                                    bet_type="prop",
                                    market_key=market_key,
                                    description=description,
                                    home_price=home_price,
                                    total_line=total_line,
                                    is_closing=is_closing,
                                )
                                session.add(odds_row)
                                count += 1
                            continue

                        odds_row = Odds(
                            game_id=game.id,
                            bookmaker=bk_name,
                            bet_type=bet_type,
                            market_key=market_key,
                            description=description,
                            home_price=home_price,
                            away_price=away_price,
                            spread_home=spread_home,
                            total_line=total_line,
                            is_closing=is_closing,
                        )
                        session.add(odds_row)
                        count += 1

        logger.info("Stored %d odds rows (is_closing=%s)", count, is_closing)
        return count

    def _quotes_for_game(self, raw: dict, bookmakers: list[str] | None) -> list[dict]:
        """
        Flatten one event's bookmakers into per-book quotes.

        Each quote holds one book's view of every market, so a line never gets
        separated from the price it was offered at.
        """
        allowed = {b.lower() for b in bookmakers} if bookmakers else None
        home_name = raw.get("home_team")
        away_name = raw.get("away_team")
        quotes = []

        for bookmaker in raw.get("bookmakers", []):
            key = (bookmaker.get("key") or "").lower()
            title = bookmaker.get("title") or key
            if allowed and key not in allowed and title.lower() not in allowed:
                continue

            q: dict[str, Any] = {"book": title, "book_key": key}
            for market in bookmaker.get("markets", []):
                mk = market.get("key")
                for o in market.get("outcomes", []):
                    name = o.get("name")
                    price = american_to_float(o.get("price"))
                    point = o.get("point")
                    point = float(point) if point is not None else None

                    if mk == "h2h":
                        if name == home_name:
                            q["ml_home"] = price
                        elif name == away_name:
                            q["ml_away"] = price
                    elif mk == "spreads":
                        if name == home_name:
                            q["spread_home_line"] = point
                            q["spread_home_price"] = price
                        elif name == away_name:
                            q["spread_away_line"] = point
                            q["spread_away_price"] = price
                    elif mk == "totals":
                        if name == "Over":
                            q["total_over_line"] = point
                            q["over_price"] = price
                        elif name == "Under":
                            q["total_under_line"] = point
                            q["under_price"] = price
            quotes.append(q)

        if allowed and not quotes:
            logger.warning(
                "No quotes from %s for %s @ %s — falling back to all books",
                ", ".join(sorted(allowed)), away_name, home_name,
            )
            return self._quotes_for_game(raw, None)
        return quotes

    @staticmethod
    def _best_quote(quotes: list[dict], price_key: str, *required: str) -> dict | None:
        """The quote offering the best price on one side, among books that also
        quote everything in `required` — so the paired price and line used for
        vig removal come from the same market."""
        eligible = [
            q for q in quotes
            if q.get(price_key) is not None
            and all(q.get(k) is not None for k in required)
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda q: q[price_key])

    def get_best_odds(
        self,
        raw_games: list[dict],
        home_team: str,
        bookmakers: list[str] | None = None,
        sport: str = "",
    ) -> dict:
        """
        Best available price on each side, with its line and the opposing price
        from the *same* book.

        Taking the best price on one side from book A and the opposing price
        from book B produces a two-sided market that never existed: vig removal
        against it understates the fair probability and inflates every edge.
        Worse, the old version paired a best price with whichever line happened
        to be seen last, so a bet could be evaluated at -3.5 and priced at -2.5.

        `bookmakers` restricts to specific books (keys or titles, e.g.
        ["bet365"]); defaults to settings.preferred_bookmakers.
        """
        if bookmakers is None:
            bookmakers = settings.preferred_bookmaker_list

        quotes: list[dict] = []
        for raw in raw_games:
            event_home = raw.get("home_team")
            matches = (
                same_team(sport, event_home, home_team) if sport
                else event_home == home_team
            )
            if not matches:
                continue
            quotes.extend(self._quotes_for_game(raw, bookmakers))

        best: dict[str, Any] = {
            "home_price": None, "away_price": None,
            "home_pair_away_price": None, "away_pair_home_price": None,
            "spread_home": None, "spread_home_price": None,
            "spread_home_pair_away_price": None,
            "spread_away": None, "spread_away_price": None,
            "total_line": None, "over_price": None, "over_pair_under_price": None,
            "total_under_line": None, "under_price": None,
            "under_pair_over_price": None,
            "books": {},
        }
        if not quotes:
            return best

        # Moneyline: each side priced at its best book, paired with that same
        # book's other side for vig removal.
        if (q := self._best_quote(quotes, "ml_home", "ml_away")):
            best["home_price"] = q["ml_home"]
            best["home_pair_away_price"] = q["ml_away"]
            best["books"]["moneyline_home"] = q["book"]
        if (q := self._best_quote(quotes, "ml_away", "ml_home")):
            best["away_price"] = q["ml_away"]
            best["away_pair_home_price"] = q["ml_home"]
            best["books"]["moneyline_away"] = q["book"]

        # Spread: line and both prices from one book.
        if (q := self._best_quote(
            quotes, "spread_home_price", "spread_home_line", "spread_away_price"
        )):
            best["spread_home"] = q["spread_home_line"]
            best["spread_home_price"] = q["spread_home_price"]
            best["spread_home_pair_away_price"] = q["spread_away_price"]
            best["books"]["spread"] = q["book"]
        if (q := self._best_quote(quotes, "spread_away_price", "spread_away_line")):
            best["spread_away"] = q["spread_away_line"]
            best["spread_away_price"] = q["spread_away_price"]

        # Totals: same, per side — books hang different numbers, so the over
        # and under evaluated here may sit on different lines.
        if (q := self._best_quote(quotes, "over_price", "total_over_line", "under_price")):
            best["total_line"] = q["total_over_line"]
            best["over_price"] = q["over_price"]
            best["over_pair_under_price"] = q["under_price"]
            best["books"]["total_over"] = q["book"]
        if (q := self._best_quote(quotes, "under_price", "total_under_line", "over_price")):
            best["total_under_line"] = q["total_under_line"]
            best["under_price"] = q["under_price"]
            best["under_pair_over_price"] = q["over_price"]
            best["books"]["total_under"] = q["book"]

        return best
