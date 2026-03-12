from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from betting_agent.config import settings
from betting_agent.intelligence.validator.schemas import SearchFinding

logger = logging.getLogger(__name__)


class TavilySearchClient:
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.tavily_api_key

    def search(self, query: str, team_hint: str | None = None) -> list[SearchFinding]:
        if not self.api_key:
            return []

        try:
            resp = requests.post(
                self.BASE_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=settings.agent_request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Tavily search failed for '%s': %s", query, exc)
            return []

        findings: list[SearchFinding] = []
        for result in data.get("results", []):
            published_at = _parse_datetime(result.get("published_date"))
            freshness_hours = None
            if published_at is not None:
                freshness_hours = (
                    datetime.now(timezone.utc) - published_at.astimezone(timezone.utc)
                ).total_seconds() / 3600

            findings.append(
                SearchFinding(
                    category="news",
                    team=team_hint,
                    headline=str(result.get("title") or query),
                    detail=str(result.get("content") or "")[:400],
                    source_url=str(result.get("url") or ""),
                    published_at=published_at,
                    freshness_hours=freshness_hours,
                    relevance_score=result.get("score"),
                )
            )

        return findings


def build_queries(sport: str, home_team: str, away_team: str) -> list[tuple[str, str | None]]:
    shared = f"{away_team} {home_team} injury report lineup starter today"
    sport_query = {
        "NHL": f"{away_team} {home_team} starting goalie today",
        "NBA": f"{away_team} {home_team} starting lineup today",
        "NFL": f"{away_team} {home_team} starting quarterback injury report today",
        "MLB": f"{away_team} {home_team} probable pitcher today",
    }.get(sport, shared)
    return [(shared, None), (sport_query, None)]


def _parse_datetime(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
