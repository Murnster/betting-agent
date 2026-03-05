"""
NBA data loader wrapping nba_api.
Converts raw API data to a standardised Polars DataFrame and seeds the DB.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

import polars as pl

from betting_agent.config import settings
from betting_agent.db.models import Game
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session
from betting_agent.sports.base import SportLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Box score columns preserved from nba_api LeagueGameLog
# ---------------------------------------------------------------------------
BOX_SCORE_COLS = ["FGM", "FGA", "FG3M", "FTA", "OREB", "AST", "TOV"]

# ---------------------------------------------------------------------------
# Team name mappings: nba_api abbreviation ↔ Odds API full name
# ---------------------------------------------------------------------------
NBA_ABBREV_TO_FULL: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

NBA_FULL_TO_ABBREV: dict[str, str] = {v: k for k, v in NBA_ABBREV_TO_FULL.items()}


def int_to_nba_season(year: int) -> str:
    """Convert integer year to NBA season string: 2024 → '2024-25'."""
    next_yr = (year + 1) % 100
    return f"{year}-{next_yr:02d}"


def _parse_nba_date(date_str: str) -> date:
    """Parse nba_api date format. Handles both 'OCT 28, 2024' and '2024-10-28'."""
    # Try ISO format first (common in newer nba_api versions)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass
    # Fallback to 'MON DD, YYYY' format
    return datetime.strptime(date_str, "%b %d, %Y").date()


def _pivot_game_log(df, season: int, is_playoff: bool) -> list[dict]:
    """
    Pivot nba_api LeagueGameLog DataFrame (2 rows per game) into
    1 row per game with home/away fields.

    Home team identified by 'vs.' in MATCHUP, away by '@'.
    """
    import pandas as pd

    if df is None or df.empty:
        return []

    rows = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) < 2:
            continue

        home_row = away_row = None
        for _, r in group.iterrows():
            matchup = str(r.get("MATCHUP", ""))
            if " vs. " in matchup:
                home_row = r
            elif " @ " in matchup:
                away_row = r

        if home_row is None or away_row is None:
            continue

        try:
            game_date = _parse_nba_date(str(home_row["GAME_DATE"]))
        except (ValueError, KeyError):
            continue

        home_abbrev = str(home_row.get("TEAM_ABBREVIATION", ""))
        away_abbrev = str(away_row.get("TEAM_ABBREVIATION", ""))

        home_score = home_row.get("PTS")
        away_score = away_row.get("PTS")

        # Determine status
        status = "final" if home_score is not None and away_score is not None else "scheduled"

        row_dict = {
            "external_id": str(game_id),
            "game_date": game_date,
            "season": season,
            "week": None,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_score": int(home_score) if home_score is not None else None,
            "away_score": int(away_score) if away_score is not None else None,
            "is_playoff": is_playoff,
            "neutral_site": False,
            "sport": "NBA",
            "status": status,
        }

        # Preserve box score columns when available
        for col in BOX_SCORE_COLS:
            row_dict[f"home_{col.lower()}"] = home_row.get(col)
            row_dict[f"away_{col.lower()}"] = away_row.get(col)

        rows.append(row_dict)

    return rows


class NBALoader(SportLoader):
    def sport_key(self) -> str:
        return "basketball_nba"

    def load_schedules(self, seasons: list[int]) -> pl.DataFrame:
        """
        Load NBA schedules for given seasons via nba_api.
        Returns standardised Polars DataFrame.
        """
        from nba_api.stats.endpoints import LeagueGameLog

        logger.info("Loading NBA schedules for seasons: %s", seasons)
        all_rows: list[dict] = []
        rate_limit = settings.nba_api_rate_limit

        for year in seasons:
            nba_season = int_to_nba_season(year)

            for season_type, is_playoff in [
                ("Regular Season", False),
                ("Playoffs", True),
            ]:
                logger.info(
                    "  Fetching %s %s...", nba_season, season_type
                )
                try:
                    log = LeagueGameLog(
                        season=nba_season,
                        season_type_all_star=season_type,
                        player_or_team_abbreviation="T",
                    )
                    df = log.get_data_frames()[0]
                    rows = _pivot_game_log(df, year, is_playoff)
                    all_rows.extend(rows)
                    logger.info("    → %d games", len(rows))
                except Exception as exc:
                    logger.warning(
                        "Failed to load %s %s: %s", nba_season, season_type, exc
                    )

                time.sleep(rate_limit)

        if not all_rows:
            return pl.DataFrame()

        return pl.DataFrame(all_rows).with_columns(
            pl.col("game_date").cast(pl.Date)
        )

    def load_injuries(self, seasons: list[int]) -> pl.DataFrame:
        """NBA has no public injury endpoint — returns empty DataFrame."""
        logger.info("NBA injury data not available via nba_api")
        return pl.DataFrame()

    def seed_games_table(self, seasons: list[int], df: pl.DataFrame | None = None) -> int:
        """
        Load schedules and upsert into the games table.
        Returns the number of rows inserted/updated.
        """
        if df is None:
            df = self.load_schedules(seasons)
        count = 0

        with get_session() as session:
            for row in df.iter_rows(named=True):
                ext_id = row.get("external_id")
                if not ext_id:
                    continue

                existing = get_game_by_external_id(session, ext_id)

                game_data = {
                    "sport": "NBA",
                    "season": row.get("season"),
                    "week": None,
                    "game_date": row.get("game_date"),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "home_score": row.get("home_score"),
                    "away_score": row.get("away_score"),
                    "neutral_site": False,
                    "weather_desc": None,
                    "temperature_f": None,
                    "wind_mph": None,
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
