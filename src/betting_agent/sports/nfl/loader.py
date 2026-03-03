"""
NFL data loader wrapping nflreadpy.
Converts raw API data to a standardised Polars DataFrame and seeds the DB.
"""

from __future__ import annotations

import logging
from datetime import date

import nflreadpy as nfl
import polars as pl

from betting_agent.db.models import Game
from betting_agent.db.session import get_session
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.sports.base import SportLoader

logger = logging.getLogger(__name__)

# Playoff week identifiers used by nfl_data_py
PLAYOFF_WEEKS = {"WildCard", "Division", "ConfChamp", "SuperBowl", "POST"}

# Map nfl_data_py schedule columns → Game ORM columns
_WEEK_MAP = {
    "WildCard": 19,
    "Division": 20,
    "ConfChamp": 21,
    "SuperBowl": 22,
}


def _week_to_int(week_val) -> int | None:
    """Convert playoff week label or numeric string to int."""
    if week_val is None:
        return None
    try:
        return int(week_val)
    except (TypeError, ValueError):
        return _WEEK_MAP.get(str(week_val), 99)


class NFLLoader(SportLoader):
    def sport_key(self) -> str:
        return "americanfootball_nfl"

    def load_schedules(self, seasons: list[int]) -> pl.DataFrame:
        """
        Load NFL schedules for given seasons.
        Returns standardised Polars DataFrame.
        """
        logger.info("Loading NFL schedules for seasons: %s", seasons)
        df = nfl.load_schedules(seasons)
        return self._normalise_schedules(df)

    def load_injuries(self, seasons: list[int]) -> pl.DataFrame:
        logger.info("Loading NFL injuries for seasons: %s", seasons)
        try:
            df = nfl.load_injuries(seasons)
            return df
        except Exception as exc:
            logger.warning("Could not load injury data: %s", exc)
            return pl.DataFrame()

    def _normalise_schedules(self, raw) -> pl.DataFrame:
        """Normalise nflreadpy Polars DataFrame to standardised Polars schema."""
        df = raw

        # Only completed or future games (exclude cancelled)
        if "game_type" in df.columns:
            pass  # keep all — filtering done by caller

        # Rename to canonical column names
        rename_map = {
            "game_id": "external_id",
            "home_team": "home_team",
            "away_team": "away_team",
            "season": "season",
            "week": "week_raw",
            "gameday": "game_date",
            "home_score": "home_score",
            "away_score": "away_score",
            "location": "neutral_site_raw",
            "game_type": "game_type",
            "temp": "temperature_f",
            "wind": "wind_mph",
            "weather": "weather_desc",
        }
        # Only rename columns that exist
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(actual_rename)

        # Derive week as integer
        if "week_raw" in df.columns:
            df = df.with_columns(
                pl.col("week_raw").map_elements(_week_to_int, return_dtype=pl.Int32).alias("week")
            )
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Int32).alias("week"))

        # Derive is_playoff from game_type
        if "game_type" in df.columns:
            df = df.with_columns(
                pl.col("game_type").is_in(["WC", "DIV", "CON", "SB"]).alias("is_playoff")
            )
        else:
            df = df.with_columns(pl.lit(False).alias("is_playoff"))

        # Neutral site
        if "neutral_site_raw" in df.columns:
            df = df.with_columns(
                (pl.col("neutral_site_raw") == "Neutral").alias("neutral_site")
            )
        else:
            df = df.with_columns(pl.lit(False).alias("neutral_site"))

        # Cast game_date to Date
        if "game_date" in df.columns:
            df = df.with_columns(
                pl.col("game_date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False)
            )

        # Add sport column
        df = df.with_columns(pl.lit("NFL").alias("sport"))

        # Derive status
        if "home_score" in df.columns and "away_score" in df.columns:
            df = df.with_columns(
                pl.when(
                    pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
                )
                .then(pl.lit("final"))
                .otherwise(pl.lit("scheduled"))
                .alias("status")
            )

        return df

    def seed_games_table(self, seasons: list[int]) -> int:
        """
        Load schedules and upsert into the games table.
        Returns the number of rows inserted/updated.
        """
        df = self.load_schedules(seasons)
        count = 0

        with get_session() as session:
            for row in df.iter_rows(named=True):
                ext_id = row.get("external_id")
                if not ext_id:
                    continue

                existing = get_game_by_external_id(session, ext_id)

                game_data = {
                    "sport": "NFL",
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "game_date": row.get("game_date"),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "home_score": row.get("home_score"),
                    "away_score": row.get("away_score"),
                    "neutral_site": bool(row.get("neutral_site", False)),
                    "weather_desc": row.get("weather_desc"),
                    "temperature_f": row.get("temperature_f"),
                    "wind_mph": row.get("wind_mph"),
                    "is_playoff": bool(row.get("is_playoff", False)),
                    "status": row.get("status", "scheduled"),
                    "external_id": ext_id,
                }

                if existing:
                    for k, v in game_data.items():
                        setattr(existing, k, v)
                else:
                    session.add(Game(**game_data))
                    count += 1

        logger.info("Seeded %d new games into DB for seasons %s", count, seasons)
        return count
