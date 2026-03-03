"""
The Odds API client.
Fetches moneyline, spread, totals and player props for any sport/market.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests

from betting_agent.config import settings
from betting_agent.db.models import Game, Odds
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session

logger = logging.getLogger(__name__)

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


class OddsAPIClient:
    def __init__(self):
        self.api_key = settings.odds_api_key
        self.base_url = settings.odds_api_base

    def _get(self, path: str, params: dict) -> list[dict] | None:
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set — skipping Odds API call")
            return None
        url = f"{self.base_url}/{path}"
        params["apiKey"] = self.api_key
        try:
            resp = requests.get(url, params=params, timeout=15)
            remaining = resp.headers.get("x-requests-remaining")
            if remaining:
                logger.debug("Odds API requests remaining: %s", remaining)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 422:
                logger.warning("Odds API 422 for %s — market not available", path)
                return []
            logger.error("Odds API HTTP error: %s", exc)
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Odds API request error: %s", exc)
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
    ) -> list[dict]:
        """
        Fetch odds for a sport.
        markets: list of market keys (h2h, spreads, totals, player_*)
        event_ids: optional list of specific event IDs to fetch
        """
        if markets is None:
            markets = STANDARD_MARKETS
        params: dict[str, Any] = {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if event_ids:
            params["eventIds"] = ",".join(event_ids)

        data = self._get(f"{sport_key}/odds", params)
        return data or []

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

                    game = Game(
                        sport=sport,
                        season=game_date.year if game_date else 0,
                        game_date=game_date,
                        home_team=raw.get("home_team", ""),
                        away_team=raw.get("away_team", ""),
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

    def get_best_odds(self, raw_games: list[dict], home_team: str) -> dict:
        """
        Extract the best (highest home_price) moneyline odds for a team
        from a raw Odds API response. Returns dict with home/away prices.
        """
        best = {"home_price": None, "away_price": None, "spread_home": None, "total_line": None,
                "over_price": None, "under_price": None,
                "spread_home_price": None, "spread_away_price": None,
                "fair_home_price": None, "fair_away_price": None}

        for raw in raw_games:
            if raw.get("home_team") != home_team:
                continue
            for bookmaker in raw.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    mk = market.get("key")
                    outcomes = market.get("outcomes", [])
                    if mk == "h2h":
                        bk_home = None
                        bk_away = None
                        for o in outcomes:
                            p = american_to_float(o.get("price"))
                            if o.get("name") == raw.get("home_team"):
                                bk_home = p
                                if best["home_price"] is None or (p and p > best["home_price"]):
                                    best["home_price"] = p
                            elif o.get("name") == raw.get("away_team"):
                                bk_away = p
                                if best["away_price"] is None or (p and p > best["away_price"]):
                                    best["away_price"] = p
                        # Track same-bookmaker pair for accurate vig removal
                        if bk_home is not None and bk_away is not None:
                            if best["fair_home_price"] is None or bk_home > best["fair_home_price"]:
                                best["fair_home_price"] = bk_home
                                best["fair_away_price"] = bk_away
                    elif mk == "spreads":
                        for o in outcomes:
                            p = american_to_float(o.get("price"))
                            if o.get("name") == raw.get("home_team"):
                                pt = o.get("point")
                                if pt is not None:
                                    best["spread_home"] = float(pt)
                                if best["spread_home_price"] is None or (p and p > best["spread_home_price"]):
                                    best["spread_home_price"] = p
                            elif o.get("name") == raw.get("away_team"):
                                if best["spread_away_price"] is None or (p and p > best["spread_away_price"]):
                                    best["spread_away_price"] = p
                    elif mk == "totals":
                        for o in outcomes:
                            pt = o.get("point")
                            if pt is not None:
                                best["total_line"] = float(pt)
                            p = american_to_float(o.get("price"))
                            if o.get("name") == "Over":
                                best["over_price"] = p
                            elif o.get("name") == "Under":
                                best["under_price"] = p
        return best
