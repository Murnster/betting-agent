"""
NFL player props data.
Wraps Odds API prop markets and links to player stats from nflreadpy.
"""

from __future__ import annotations

import logging

import nflreadpy as nfl
import pandas as pd

from betting_agent.api.odds import OddsAPIClient, PROP_MARKETS

logger = logging.getLogger(__name__)


def fetch_prop_odds(sport_key: str = "americanfootball_nfl") -> list[dict]:
    """Fetch player prop odds from The Odds API."""
    client = OddsAPIClient()
    # First get event IDs
    events = client.fetch_events(sport_key)
    event_ids = [e.get("id") for e in events if e.get("id")]

    if not event_ids:
        return []

    all_props = []
    # Fetch props in chunks (API may limit per request)
    chunk_size = 10
    for i in range(0, len(event_ids), chunk_size):
        chunk = event_ids[i:i + chunk_size]
        props = client.fetch_odds(sport_key, markets=PROP_MARKETS, event_ids=chunk)
        all_props.extend(props)

    return all_props


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    """Load weekly player stats from nflreadpy."""
    try:
        df = nfl.load_player_stats(seasons)
        return df.to_pandas()
    except Exception as exc:
        logger.warning("Could not load player stats: %s", exc)
        return pd.DataFrame()
