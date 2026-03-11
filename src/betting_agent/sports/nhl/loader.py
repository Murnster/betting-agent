"""
NHL data loader wrapping nhl-api-py (nhlpy).
Converts raw API data to a standardised Polars DataFrame and seeds the DB.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl

from betting_agent.config import settings
from betting_agent.db.models import Game
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session
from betting_agent.sports.base import SportLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Box score columns preserved from nhlpy boxscores
# ---------------------------------------------------------------------------
BOX_SCORE_COLS = ["sog", "ppg", "ppo", "pim", "hits", "blocks", "faceoff_pct"]

# ---------------------------------------------------------------------------
# Team name mappings: NHL abbreviation <-> Odds API full name
# ---------------------------------------------------------------------------
NHL_ABBREV_TO_FULL: dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}

NHL_FULL_TO_ABBREV: dict[str, str] = {v: k for k, v in NHL_ABBREV_TO_FULL.items()}

# All 32 current team abbreviations (ARI is historical, UTA is current)
NHL_TEAM_ABBREVS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# ---------------------------------------------------------------------------
# Thread-local NHLClient instances (requests sessions aren't thread-safe)
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_client():
    """Return a thread-local NHLClient instance."""
    if not hasattr(_thread_local, "client"):
        from nhlpy import NHLClient
        _thread_local.client = NHLClient()
    return _thread_local.client


def _get_cache_dir() -> Path:
    """Return the NHL parquet cache directory, creating it if needed."""
    cache_dir = Path(settings.saved_models_dir) / ".cache" / "NHL"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _fetch_team_schedule_worker(team_abbr: str, nhl_season: str, season: int) -> list[dict]:
    """Fetch one team's schedule with rate-limit sleep. Thread-safe."""
    client = _get_client()
    try:
        schedule = client.schedule.team_season_schedule(
            team_abbr=team_abbr, season=nhl_season
        )
        time.sleep(settings.nhl_api_rate_limit)
        return _parse_schedule_games(schedule, season)
    except Exception as exc:
        logger.warning("Failed to load schedule for %s %s: %s", team_abbr, nhl_season, exc)
        time.sleep(settings.nhl_api_rate_limit)
        return []


def _fetch_game_story_worker(game_id: str) -> dict[str, float | None] | None:
    """Fetch one game story with rate-limit sleep, return parsed box score dict."""
    client = _get_client()
    try:
        story = client.game_center.game_story(game_id=game_id)
        team_stats = story.get("summary", {}).get("teamGameStats", [])
        home_stats = _parse_team_game_stats(team_stats, "home")
        away_stats = _parse_team_game_stats(team_stats, "away")
        result: dict[str, float | None] = {"external_id": game_id}
        for col in BOX_SCORE_COLS:
            result[f"home_{col}"] = home_stats.get(col)
            result[f"away_{col}"] = away_stats.get(col)
        time.sleep(settings.nhl_api_rate_limit)
        return result
    except Exception as exc:
        logger.debug("Failed to fetch game story for game %s: %s", game_id, exc)
        time.sleep(settings.nhl_api_rate_limit)
        return None


def int_to_nhl_season(year: int) -> str:
    """Convert integer year to NHL season string: 2023 -> '20232024'."""
    return f"{year}{year + 1}"


def _extract_team_boxscore_stats(team_data: dict) -> dict[str, float | None]:
    """Extract box score stats from a team's boxscore data."""
    stats: dict[str, float | None] = {}

    # Shots on goal
    stats["sog"] = team_data.get("sog")

    # Power play
    pp_conv = team_data.get("powerPlayConversion", "0/0")
    if isinstance(pp_conv, str) and "/" in pp_conv:
        parts = pp_conv.split("/")
        try:
            stats["ppg"] = int(parts[0])
            stats["ppo"] = int(parts[1])
        except (ValueError, IndexError):
            stats["ppg"] = None
            stats["ppo"] = None
    else:
        stats["ppg"] = None
        stats["ppo"] = None

    # PIM
    stats["pim"] = team_data.get("pim")

    # Hits
    stats["hits"] = team_data.get("hits")

    # Blocked shots
    stats["blocks"] = team_data.get("blocks")

    # Faceoff %
    fo_pct = team_data.get("faceoffWinningPctg")
    if fo_pct is not None:
        try:
            stats["faceoff_pct"] = float(fo_pct)
        except (ValueError, TypeError):
            stats["faceoff_pct"] = None
    else:
        stats["faceoff_pct"] = None

    return stats


def _parse_team_game_stats(team_game_stats: list[dict], side: str) -> dict[str, float | None]:
    """Parse the teamGameStats list from game_story into our column format.

    Args:
        team_game_stats: list of {"category": str, "awayValue": ..., "homeValue": ...}
        side: "home" or "away"
    """
    value_key = "homeValue" if side == "home" else "awayValue"
    stats: dict[str, float | None] = {
        "sog": None, "ppg": None, "ppo": None,
        "pim": None, "hits": None, "blocks": None, "faceoff_pct": None,
    }

    for entry in team_game_stats:
        cat = entry.get("category", "")
        val = entry.get(value_key)

        if cat == "sog":
            stats["sog"] = val
        elif cat == "pim":
            stats["pim"] = val
        elif cat == "hits":
            stats["hits"] = val
        elif cat == "blockedShots":
            stats["blocks"] = val
        elif cat == "faceoffWinningPctg":
            if val is not None:
                try:
                    stats["faceoff_pct"] = float(val)
                except (ValueError, TypeError):
                    pass
        elif cat == "powerPlay":
            if isinstance(val, str) and "/" in val:
                parts = val.split("/")
                try:
                    stats["ppg"] = int(parts[0])
                    stats["ppo"] = int(parts[1])
                except (ValueError, IndexError):
                    pass

    return stats


def _parse_schedule_games(schedule_data: dict, season: int) -> list[dict]:
    """Parse team season schedule response into game rows."""
    rows = []
    games = schedule_data.get("games", [])

    for game in games:
        game_id = game.get("id")
        if game_id is None:
            continue

        game_type = game.get("gameType")
        # gameType 2 = regular season, 3 = playoffs
        if game_type not in (2, 3):
            continue

        game_state = game.get("gameState", "")
        # OFF = final, LIVE = in progress, FUT/PRE = future
        if game_state not in ("OFF", "FINAL"):
            continue

        home_team_data = game.get("homeTeam", {})
        away_team_data = game.get("awayTeam", {})

        home_abbrev = home_team_data.get("abbrev", "")
        away_abbrev = away_team_data.get("abbrev", "")
        home_score = home_team_data.get("score")
        away_score = away_team_data.get("score")

        game_date_str = game.get("gameDate", "")
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        status = "final" if home_score is not None and away_score is not None else "scheduled"

        rows.append({
            "external_id": str(game_id),
            "game_date": game_date,
            "season": season,
            "week": None,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_score": int(home_score) if home_score is not None else None,
            "away_score": int(away_score) if away_score is not None else None,
            "is_playoff": game_type == 3,
            "neutral_site": False,
            "sport": "NHL",
            "status": status,
        })

    return rows


class NHLLoader(SportLoader):
    def sport_key(self) -> str:
        return "icehockey_nhl"

    def load_schedules_from_db(self, seasons: list[int]) -> pl.DataFrame:
        """Load NHL games from local DB for given seasons."""
        from betting_agent.db.queries import get_games_by_season
        from betting_agent.db.session import get_session

        with get_session() as session:
            games = get_games_by_season(session, "NHL", seasons)
            if not games:
                return pl.DataFrame()

            rows = []
            for g in games:
                row = {
                    "external_id": g.external_id,
                    "game_date": g.game_date,
                    "season": g.season,
                    "week": g.week,
                    "home_team": g.home_team,
                    "away_team": g.away_team,
                    "home_score": g.home_score,
                    "away_score": g.away_score,
                    "is_playoff": g.is_playoff,
                    "neutral_site": g.neutral_site,
                    "sport": g.sport,
                    "status": g.status,
                }
                # Box score stats if available in DB (if stored as attributes)
                for col in BOX_SCORE_COLS:
                    row[f"home_{col}"] = getattr(g, f"home_{col}", None)
                    row[f"away_{col}"] = getattr(g, f"away_{col}", None)
                rows.append(row)

            return pl.DataFrame(rows).with_columns(
                pl.col("game_date").cast(pl.Date)
            )

    def load_schedules(self, seasons: list[int]) -> pl.DataFrame:
        """
        Load NHL schedules for given seasons. Prefers DB, falls back to API.
        Uses concurrent fetching and parquet caching for performance.
        """
        # Try DB first
        df_db = self.load_schedules_from_db(seasons)
        if not df_db.is_empty():
            logger.info("Loaded %d NHL games from local DB", len(df_db))
            return df_db

        logger.info("Loading NHL schedules for seasons: %s", seasons)
        cache_dir = _get_cache_dir()
        current_year = datetime.now().year
        max_workers = settings.nhl_api_concurrency
        all_rows: dict[str, dict] = {}  # game_id -> row (dedup)

        # ----- Phase 1: Schedule fetching -----
        for year in seasons:
            cache_path = cache_dir / f"schedules_{year}.parquet"
            is_completed_season = year < current_year

            if is_completed_season and cache_path.exists():
                logger.info("  Loading cached schedule for %d", year)
                cached_df = pl.read_parquet(cache_path)
                for row in cached_df.iter_rows(named=True):
                    all_rows[row["external_id"]] = row
                continue

            nhl_season = int_to_nhl_season(year)
            logger.info("  Fetching %s schedule concurrently (%d workers)...", nhl_season, max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_fetch_team_schedule_worker, abbr, nhl_season, year): abbr
                    for abbr in NHL_TEAM_ABBREVS
                }
                for future in as_completed(futures):
                    rows = future.result()
                    for row in rows:
                        ext_id = row["external_id"]
                        if ext_id not in all_rows:
                            all_rows[ext_id] = row

            season_rows = [r for r in all_rows.values() if r.get("season") == year]
            logger.info("    -> %d unique games for %s", len(season_rows), nhl_season)

            # Cache completed seasons
            if is_completed_season and season_rows:
                pl.DataFrame(season_rows).with_columns(
                    pl.col("game_date").cast(pl.Date)
                ).write_parquet(cache_path)
                logger.info("    -> Cached schedule to %s", cache_path)

        if not all_rows:
            return pl.DataFrame()

        # ----- Phase 2: Game story fetching -----
        for year in seasons:
            stories_cache_path = cache_dir / f"game_stories_{year}.parquet"
            is_completed_season = year < current_year

            # Load cached game stories if available
            cached_stories: dict[str, dict] = {}
            if stories_cache_path.exists():
                logger.info("  Loading cached game stories for %d", year)
                cached_stories_df = pl.read_parquet(stories_cache_path)
                for row in cached_stories_df.iter_rows(named=True):
                    cached_stories[row["external_id"]] = row

            # Apply cached stories to all_rows
            for ext_id, story_data in cached_stories.items():
                if ext_id in all_rows:
                    for col in BOX_SCORE_COLS:
                        all_rows[ext_id][f"home_{col}"] = story_data.get(f"home_{col}")
                        all_rows[ext_id][f"away_{col}"] = story_data.get(f"away_{col}")

            # Find games that need stories fetched
            season_games = [
                r for r in all_rows.values()
                if r.get("season") == year and r.get("status") == "final"
            ]
            missing_ids = [
                r["external_id"] for r in season_games
                if r["external_id"] not in cached_stories
            ]

            if not missing_ids:
                logger.info("  All game stories cached for %d", year)
                continue

            logger.info("  Fetching %d game stories for %d concurrently (%d workers)...",
                        len(missing_ids), year, max_workers)
            completed = 0
            new_stories: dict[str, dict] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_fetch_game_story_worker, gid): gid
                    for gid in missing_ids
                }
                for future in as_completed(futures):
                    game_id = futures[future]
                    result = future.result()
                    completed += 1

                    if result is not None:
                        new_stories[result["external_id"]] = result
                        ext_id = result["external_id"]
                        if ext_id in all_rows:
                            for col in BOX_SCORE_COLS:
                                all_rows[ext_id][f"home_{col}"] = result.get(f"home_{col}")
                                all_rows[ext_id][f"away_{col}"] = result.get(f"away_{col}")
                    else:
                        # Set nulls for failed fetches
                        if game_id in all_rows:
                            for col in BOX_SCORE_COLS:
                                all_rows[game_id][f"home_{col}"] = None
                                all_rows[game_id][f"away_{col}"] = None

                    if completed % 200 == 0:
                        logger.info("    -> %d/%d game stories fetched", completed, len(missing_ids))

            logger.info("    -> Finished: %d/%d game stories fetched for %d",
                        completed, len(missing_ids), year)

            # Merge new stories with cached and save
            all_season_stories = {**cached_stories}
            for ext_id, story in new_stories.items():
                all_season_stories[ext_id] = story
            # Also include null entries for failed fetches so we don't re-fetch
            for gid in missing_ids:
                if gid not in all_season_stories:
                    null_entry: dict[str, float | None] = {"external_id": gid}
                    for col in BOX_SCORE_COLS:
                        null_entry[f"home_{col}"] = None
                        null_entry[f"away_{col}"] = None
                    all_season_stories[gid] = null_entry

            if is_completed_season and all_season_stories:
                pl.DataFrame(list(all_season_stories.values())).write_parquet(stories_cache_path)
                logger.info("    -> Cached game stories to %s", stories_cache_path)

        # Ensure all rows have box score columns (fill missing with None)
        game_rows = list(all_rows.values())
        for row in game_rows:
            for col in BOX_SCORE_COLS:
                row.setdefault(f"home_{col}", None)
                row.setdefault(f"away_{col}", None)

        return pl.DataFrame(game_rows).with_columns(
            pl.col("game_date").cast(pl.Date)
        )

    def load_injuries(self, seasons: list[int]) -> pl.DataFrame:
        """NHL has no public injury endpoint - returns empty DataFrame."""
        logger.info("NHL injury data not available via nhlpy")
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
                    "sport": "NHL",
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

                # Add box score stats to game_data
                for col in BOX_SCORE_COLS:
                    game_data[f"home_{col}"] = row.get(f"home_{col}")
                    game_data[f"away_{col}"] = row.get(f"away_{col}")

                if existing:
                    for k, v in game_data.items():
                        setattr(existing, k, v)
                else:
                    session.add(Game(**game_data))
                    count += 1

        logger.info("Seeded %d new games into DB for seasons %s", count, seasons)
        return count
