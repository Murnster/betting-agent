"""
MLB data loader wrapping python-mlb-statsapi (mlbstatsapi).
Converts raw API data to a standardised Polars DataFrame and seeds the DB.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import polars as pl

from betting_agent.config import settings
from betting_agent.db.models import Game
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session
from betting_agent.sports.base import SportLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Box score columns preserved from mlbstatsapi boxscores
# ---------------------------------------------------------------------------
# Batting: hits, runs, home_runs, rbi, strikeouts, walks, doubles, triples, stolen_bases
# Pitching: earned_runs, pitching_strikeouts, pitching_walks, hits_allowed, innings_pitched
BOX_SCORE_COLS = [
    "hits", "runs_scored", "home_runs", "rbi", "strikeouts", "walks",
    "doubles", "triples", "stolen_bases",
    "earned_runs", "pitching_strikeouts", "pitching_walks",
    "hits_allowed", "innings_pitched",
]

# ---------------------------------------------------------------------------
# Team name mappings: MLB abbreviation <-> Odds API full name
# ---------------------------------------------------------------------------
MLB_ABBREV_TO_FULL: dict[str, str] = {
    "AZ": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

MLB_FULL_TO_ABBREV: dict[str, str] = {v: k for k, v in MLB_ABBREV_TO_FULL.items()}
# Historical Oakland Athletics mapping
MLB_FULL_TO_ABBREV["Oakland Athletics"] = "ATH"

# All 30 current team abbreviations
MLB_TEAM_ABBREVS = sorted(MLB_ABBREV_TO_FULL.keys())

# MLB API team IDs for schedule queries
_MLB_TEAM_IDS: dict[str, int] = {
    "LAA": 108, "AZ": 109, "BAL": 110, "BOS": 111, "CHC": 112,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117,
    "KC": 118, "LAD": 119, "WSH": 120, "NYM": 121, "ATH": 133,
    "PIT": 134, "SD": 135, "SEA": 136, "SF": 137, "STL": 138,
    "TB": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143,
    "ATL": 144, "CWS": 145, "MIA": 146, "NYY": 147, "MIL": 158,
}

# Reverse: full name → team ID
_MLB_NAME_TO_ID: dict[str, int] = {}
for _abbr, _tid in _MLB_TEAM_IDS.items():
    _full = MLB_ABBREV_TO_FULL.get(_abbr)
    if _full:
        _MLB_NAME_TO_ID[_full] = _tid

# Valid game types: R=Regular, F=Wild Card, D=Division, L=League, W=World Series
_VALID_GAME_TYPES = {"R", "F", "D", "L", "W"}


def _extract_team_boxscore_stats(box_team_stats: dict) -> dict[str, float | None]:
    """Extract batting + pitching stats from a team's boxscore team_stats dict."""
    stats: dict[str, float | None] = {}
    batting = box_team_stats.get("batting", {})
    pitching = box_team_stats.get("pitching", {})

    # Batting stats
    stats["hits"] = batting.get("hits")
    stats["runs_scored"] = batting.get("runs")
    stats["home_runs"] = batting.get("homeRuns")
    stats["rbi"] = batting.get("rbi")
    stats["strikeouts"] = batting.get("strikeOuts")
    stats["walks"] = batting.get("baseOnBalls")
    stats["doubles"] = batting.get("doubles")
    stats["triples"] = batting.get("triples")
    stats["stolen_bases"] = batting.get("stolenBases")

    # Pitching stats
    stats["earned_runs"] = pitching.get("earnedRuns")
    stats["pitching_strikeouts"] = pitching.get("strikeOuts")
    stats["pitching_walks"] = pitching.get("baseOnBalls")
    stats["hits_allowed"] = pitching.get("hits")

    # Innings pitched comes as a string like "9.0"
    ip_raw = pitching.get("inningsPitched")
    if ip_raw is not None:
        try:
            stats["innings_pitched"] = float(ip_raw)
        except (ValueError, TypeError):
            stats["innings_pitched"] = None
    else:
        stats["innings_pitched"] = None

    return stats


def _name_to_abbrev(team_name: str) -> str:
    """Convert full team name to abbreviation."""
    return MLB_FULL_TO_ABBREV.get(team_name, team_name)


def _parse_schedule_games(schedule, season: int) -> list[dict]:
    """Parse mlbstatsapi Schedule object into game rows."""
    rows = []

    for sched_date in schedule.dates:
        for game in sched_date.games:
            game_type = game.game_type
            if game_type not in _VALID_GAME_TYPES:
                continue

            status = game.status.detailed_state if game.status else ""
            if status != "Final":
                continue

            game_pk = game.game_pk
            if game_pk is None:
                continue

            home_team_data = game.teams.home if game.teams else None
            away_team_data = game.teams.away if game.teams else None
            if not home_team_data or not away_team_data:
                continue

            home_name = home_team_data.team.name if home_team_data.team else ""
            away_name = away_team_data.team.name if away_team_data.team else ""
            home_abbrev = _name_to_abbrev(home_name)
            away_abbrev = _name_to_abbrev(away_name)

            home_score = home_team_data.score
            away_score = away_team_data.score

            game_date_str = game.official_date or ""
            try:
                game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            is_playoff = game_type in ("F", "D", "L", "W")

            rows.append({
                "external_id": str(game_pk),
                "game_date": game_date,
                "season": season,
                "week": None,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_score": int(home_score) if home_score is not None else None,
                "away_score": int(away_score) if away_score is not None else None,
                "is_playoff": is_playoff,
                "neutral_site": False,
                "sport": "MLB",
                "status": "final",
            })

    return rows


class MLBLoader(SportLoader):
    def sport_key(self) -> str:
        return "baseball_mlb"

    def load_schedules_from_db(self, seasons: list[int]) -> pl.DataFrame:
        """Load MLB games from local DB for given seasons."""
        from betting_agent.db.queries import get_games_by_season
        from betting_agent.db.session import get_session

        with get_session() as session:
            games = get_games_by_season(session, "MLB", seasons)
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
                for col in BOX_SCORE_COLS:
                    row[f"home_{col}"] = getattr(g, f"home_{col}", None)
                    row[f"away_{col}"] = getattr(g, f"away_{col}", None)
                rows.append(row)

            return pl.DataFrame(rows).with_columns(
                pl.col("game_date").cast(pl.Date)
            )

    def load_schedules(self, seasons: list[int]) -> pl.DataFrame:
        """Load MLB schedules for given seasons. Prefers DB, falls back to API."""
        df_db = self.load_schedules_from_db(seasons)
        if not df_db.is_empty():
            logger.info("Loaded %d MLB games from local DB", len(df_db))
            return df_db

        import mlbstatsapi

        mlb = mlbstatsapi.Mlb()
        logger.info("Loading MLB schedules for seasons: %s", seasons)
        all_rows: dict[str, dict] = {}
        rate_limit = settings.mlb_api_rate_limit

        for year in seasons:
            logger.info("  Fetching %d schedule...", year)

            # Query by month chunks (March through November)
            for month in range(3, 12):
                start = f"{year}-{month:02d}-01"
                if month == 11:
                    end = f"{year}-11-15"
                else:
                    # Last day of month
                    next_month = month + 1
                    end = f"{year}-{next_month:02d}-01"

                try:
                    schedule = mlb.get_schedule(
                        start_date=start, end_date=end, sport_id=1,
                    )
                    rows = _parse_schedule_games(schedule, year)
                    for row in rows:
                        ext_id = row["external_id"]
                        if ext_id not in all_rows:
                            all_rows[ext_id] = row
                except Exception as exc:
                    logger.warning(
                        "Failed to load schedule for %d month %d: %s", year, month, exc
                    )
                time.sleep(rate_limit)

            logger.info("    -> %d unique games for %d so far", len(all_rows), year)

        if not all_rows:
            return pl.DataFrame()

        game_rows = list(all_rows.values())

        # Fetch box scores for completed games
        logger.info("Fetching box scores for %d completed games...", len(game_rows))
        for row in game_rows:
            if row["status"] != "final":
                continue

            try:
                boxscore = mlb.get_game_box_score(int(row["external_id"]))
                for side, team_key in [("home", "home"), ("away", "away")]:
                    team_data = getattr(boxscore.teams, team_key, None)
                    if team_data and hasattr(team_data, "team_stats"):
                        stats = _extract_team_boxscore_stats(team_data.team_stats)
                        for col in BOX_SCORE_COLS:
                            row[f"{side}_{col}"] = stats.get(col)
                    else:
                        for col in BOX_SCORE_COLS:
                            row[f"{side}_{col}"] = None
            except Exception as exc:
                logger.debug(
                    "Failed to fetch box score for game %s: %s", row["external_id"], exc
                )
                for col in BOX_SCORE_COLS:
                    row[f"home_{col}"] = None
                    row[f"away_{col}"] = None

            time.sleep(rate_limit)

        return pl.DataFrame(game_rows).with_columns(
            pl.col("game_date").cast(pl.Date)
        )

    def load_injuries(self, seasons: list[int]) -> pl.DataFrame:
        """MLB has no public injury endpoint - returns empty DataFrame."""
        logger.info("MLB injury data not available via mlbstatsapi")
        return pl.DataFrame()

    def seed_games_table(self, seasons: list[int], df: pl.DataFrame | None = None) -> int:
        """Load schedules and upsert into the games table."""
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
                    "sport": "MLB",
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
