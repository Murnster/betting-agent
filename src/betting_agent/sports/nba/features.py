"""
NBA-specific final feature assembly.
Takes schedules DataFrame, runs shared pipeline steps, and returns
a model-ready feature matrix.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from betting_agent.pipeline.elo import calculate_elo_ratings
from betting_agent.pipeline.features import (
    add_rest_days,
    calculate_head_to_head,
    compute_rolling_averages,
    compute_streaks_and_last5,
)
from betting_agent.sports.nba.loader import BOX_SCORE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conference / Division constants (all 30 NBA teams by abbreviation)
# ---------------------------------------------------------------------------
EASTERN_CONFERENCE = {
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DET", "IND", "MIA", "MIL",
    "NYK", "ORL", "PHI", "TOR", "WAS",
}

WESTERN_CONFERENCE = {
    "DAL", "DEN", "GSW", "HOU", "LAC",
    "LAL", "MEM", "MIN", "NOP", "OKC",
    "PHX", "POR", "SAC", "SAS", "UTA",
}

DIVISIONS: dict[str, set[str]] = {
    "Atlantic": {"BOS", "BKN", "NYK", "PHI", "TOR"},
    "Central": {"CHI", "CLE", "DET", "IND", "MIL"},
    "Southeast": {"ATL", "CHA", "MIA", "ORL", "WAS"},
    "Northwest": {"DEN", "MIN", "OKC", "POR", "UTA"},
    "Pacific": {"GSW", "LAC", "LAL", "PHX", "SAC"},
    "Southwest": {"DAL", "HOU", "MEM", "NOP", "SAS"},
}

# Reverse lookup: team → division name
_TEAM_TO_DIVISION: dict[str, str] = {}
for div, teams in DIVISIONS.items():
    for t in teams:
        _TEAM_TO_DIVISION[t] = div


# ---------------------------------------------------------------------------
# Advanced stats names and rolling windows
# ---------------------------------------------------------------------------
_ADVANCED_STATS = [
    "ts_pct", "efg_pct", "ortg", "drtg", "pace",
    "ast_rate", "tov_rate", "oreb",
]


def _compute_game_advanced(
    pts: float, opp_pts: float,
    fgm: float, fga: float, fg3m: float, fta: float,
    oreb: float, ast: float, tov: float,
) -> dict[str, float]:
    """Compute advanced stats for a single team-game."""
    denom_ts = 2 * (fga + 0.44 * fta)
    ts_pct = pts / denom_ts if denom_ts > 0 else np.nan
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else np.nan
    possessions = max(fga - oreb + tov + 0.44 * fta, 1)
    ortg = pts / possessions * 100
    drtg = opp_pts / possessions * 100
    ast_rate = ast / fgm if fgm > 0 else np.nan
    tov_rate = tov / possessions
    return {
        "ts_pct": ts_pct,
        "efg_pct": efg_pct,
        "ortg": ortg,
        "drtg": drtg,
        "pace": possessions,
        "ast_rate": ast_rate,
        "tov_rate": tov_rate,
        "oreb": oreb,
    }


def compute_nba_advanced_rolling(
    df: pd.DataFrame, windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling advanced basketball stats for home and away teams.

    Requires box score columns (home_fga, away_fga, etc.) from the NBA loader.
    If they are absent (NFL data or upcoming-only frames), returns df unchanged.

    Produces 8 stats × 2 sides × len(windows) = columns.
    """
    if windows is None:
        windows = [3, 5, 10]

    # Guard: need box score columns to compute anything
    if "home_fga" not in df.columns:
        return df

    df = df.copy()

    # Initialize output columns
    for side in ("home", "away"):
        for stat in _ADVANCED_STATS:
            for w in windows:
                df[f"{side}_{stat}_{w}g"] = np.nan

    # Team history: team → list of per-game stat dicts
    team_history: dict[str, list[dict[str, float]]] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        team_history.setdefault(home, [])
        team_history.setdefault(away, [])

        # Write rolling averages from existing history
        for side, team in [("home", home), ("away", away)]:
            hist = team_history[team]
            if hist:
                for w in windows:
                    recent = hist[-w:]
                    for stat in _ADVANCED_STATS:
                        vals = [g[stat] for g in recent if not np.isnan(g[stat])]
                        if vals:
                            df.at[idx, f"{side}_{stat}_{w}g"] = np.mean(vals)

        # Append this game's stats to history (only if box data present)
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        # Check that box stats are not NaN (upcoming games won't have them)
        home_fga = row.get("home_fga")
        if pd.isna(home_fga):
            continue

        # Home team stats
        home_stats = _compute_game_advanced(
            pts=float(home_score), opp_pts=float(away_score),
            fgm=float(row["home_fgm"]), fga=float(row["home_fga"]),
            fg3m=float(row["home_fg3m"]), fta=float(row["home_fta"]),
            oreb=float(row["home_oreb"]), ast=float(row["home_ast"]),
            tov=float(row["home_tov"]),
        )
        team_history[home].append(home_stats)

        # Away team stats
        away_stats = _compute_game_advanced(
            pts=float(away_score), opp_pts=float(home_score),
            fgm=float(row["away_fgm"]), fga=float(row["away_fga"]),
            fg3m=float(row["away_fg3m"]), fta=float(row["away_fta"]),
            oreb=float(row["away_oreb"]), ast=float(row["away_ast"]),
            tov=float(row["away_tov"]),
        )
        team_history[away].append(away_stats)

    return df


def compute_nba_team_quality(
    df: pd.DataFrame, windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute win% and strength-of-schedule rolling features for NBA teams.

    Produces win% over season-to-date, last-N games, and rolling opponent Elo.
    """
    if windows is None:
        windows = [10, 20]

    df = df.copy().sort_values("game_date").reset_index(drop=True)

    for side in ("home", "away"):
        df[f"{side}_win_pct_season"] = np.nan
        for w in windows:
            df[f"{side}_win_pct_{w}g"] = np.nan
            df[f"{side}_opp_elo_{w}g"] = np.nan

    # Per-team state (reset by season)
    results: dict[str, list[float]] = {}
    opp_elos: dict[str, list[float]] = {}
    season_wins: dict[str, int] = {}
    season_games: dict[str, int] = {}
    season_tracker: dict[str, int] = {}

    for idx, row in df.iterrows():
        season = int(row.get("season", 0)) if not pd.isna(row.get("season")) else 0

        for side, team, opp_elo in [
            ("home", row.get("home_team"), row.get("away_elo")),
            ("away", row.get("away_team"), row.get("home_elo")),
        ]:
            if team is None:
                continue
            team = str(team)
            if season_tracker.get(team) != season:
                season_tracker[team] = season
                results[team] = []
                opp_elos[team] = []
                season_wins[team] = 0
                season_games[team] = 0

            if season_games[team] > 0:
                df.at[idx, f"{side}_win_pct_season"] = season_wins[team] / max(1, season_games[team])

            for w in windows:
                recent = results[team][-w:]
                if recent:
                    df.at[idx, f"{side}_win_pct_{w}g"] = float(np.mean(recent))

                opp_recent = opp_elos[team][-w:]
                if opp_recent:
                    df.at[idx, f"{side}_opp_elo_{w}g"] = float(np.mean(opp_recent))

        # Update histories after outcomes are known
        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if pd.isna(h_score) or pd.isna(a_score):
            continue

        home = str(row.get("home_team"))
        away = str(row.get("away_team"))

        if float(h_score) > float(a_score):
            home_res, away_res = 1.0, 0.0
        elif float(h_score) < float(a_score):
            home_res, away_res = 0.0, 1.0
        else:
            home_res = away_res = 0.5

        results.setdefault(home, []).append(home_res)
        results.setdefault(away, []).append(away_res)
        season_wins[home] = season_wins.get(home, 0) + int(home_res == 1.0)
        season_wins[away] = season_wins.get(away, 0) + int(away_res == 1.0)
        season_games[home] = season_games.get(home, 0) + 1
        season_games[away] = season_games.get(away, 0) + 1

        home_opp_elo = row.get("away_elo")
        away_opp_elo = row.get("home_elo")
        if home_opp_elo is not None and not pd.isna(home_opp_elo):
            opp_elos.setdefault(home, []).append(float(home_opp_elo))
        if away_opp_elo is not None and not pd.isna(away_opp_elo):
            opp_elos.setdefault(away, []).append(float(away_opp_elo))

    return df


def add_nba_differentials(
    df: pd.DataFrame, windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add home-away differential features for team strength signals."""
    if windows is None:
        windows = [3, 5, 10]

    df = df.copy()

    # Elo-based differentials
    for col, name in [
        ("elo", "elo_diff"),
        ("recent_elo", "recent_elo_diff"),
        ("playoff_elo", "playoff_elo_diff"),
    ]:
        h = f"home_{col}"
        a = f"away_{col}"
        if h in df.columns and a in df.columns:
            df[name] = df[h] - df[a]

    # Rolling score differentials
    for w in windows:
        h_scored = f"home_avg_scored_{w}g"
        a_scored = f"away_avg_scored_{w}g"
        h_allowed = f"home_avg_allowed_{w}g"
        a_allowed = f"away_avg_allowed_{w}g"
        if h_scored in df.columns and a_scored in df.columns:
            df[f"avg_scored_diff_{w}g"] = df[h_scored] - df[a_scored]
        if h_allowed in df.columns and a_allowed in df.columns:
            df[f"avg_allowed_diff_{w}g"] = df[h_allowed] - df[a_allowed]

    # Advanced stats differentials
    for w in windows:
        for stat in _ADVANCED_STATS:
            h_col = f"home_{stat}_{w}g"
            a_col = f"away_{stat}_{w}g"
            if h_col in df.columns and a_col in df.columns:
                df[f"{stat}_diff_{w}g"] = df[h_col] - df[a_col]

    return df


def normalise_nba_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical columns exist for build_nba_features().
    NBA data from NBALoader already uses canonical names, but this
    fills any missing optional columns so downstream code is safe.
    """
    df = df.copy()

    if "is_playoff" not in df.columns:
        df["is_playoff"] = False

    if "neutral_site" not in df.columns:
        df["neutral_site"] = False

    # NBA is indoor — no weather
    for col in ("weather_desc", "temperature_f", "wind_mph"):
        if col not in df.columns:
            df[col] = None

    return df


def build_nba_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full NBA feature engineering pipeline.

    Returns feature matrix ready for XGBoost (all numeric).
    """
    df = normalise_nba_schedules(df)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # ---- Basic derived columns ----
    if "home_team_wins" not in df.columns:
        df["home_team_wins"] = np.where(
            df["home_score"] > df["away_score"], 1,
            np.where(df["home_score"] < df["away_score"], 0, np.nan)
        )

    df["schedule_playoff"] = df["is_playoff"].fillna(False).astype(int)
    df["is_weekend"] = pd.to_datetime(df["game_date"]).dt.dayofweek.apply(
        lambda x: 1 if x >= 5 else 0
    )

    # ---- Shared feature steps ----
    df = compute_rolling_averages(df, windows=[3, 5, 10])
    df = compute_nba_advanced_rolling(df, windows=[3, 5, 10])
    df = compute_streaks_and_last5(df)
    df["head_to_head"] = calculate_head_to_head(df)
    df = add_rest_days(df)
    df = calculate_elo_ratings(df)
    df = compute_nba_team_quality(df, windows=[10, 20])
    df = add_nba_differentials(df, windows=[3, 5, 10])

    # ---- Spread-derived home_is_favorite ----
    if "home_is_favorite" not in df.columns:
        df["home_is_favorite"] = -1

    # ---- NBA-specific features ----
    # Back-to-back detection (rest_days == 1 is a huge factor in NBA)
    if "home_rest_days" in df.columns:
        df["home_back_to_back"] = (df["home_rest_days"] == 1).astype(int)
    else:
        df["home_back_to_back"] = 0

    if "away_rest_days" in df.columns:
        df["away_back_to_back"] = (df["away_rest_days"] == 1).astype(int)
    else:
        df["away_back_to_back"] = 0

    # Conference features
    df["home_is_eastern"] = df["home_team"].isin(EASTERN_CONFERENCE).astype(int)
    df["away_is_eastern"] = df["away_team"].isin(EASTERN_CONFERENCE).astype(int)
    df["cross_conference"] = (df["home_is_eastern"] != df["away_is_eastern"]).astype(int)

    # Division feature
    home_div = df["home_team"].map(_TEAM_TO_DIVISION)
    away_div = df["away_team"].map(_TEAM_TO_DIVISION)
    df["same_division"] = (home_div == away_div).fillna(False).astype(int)

    # ---- Drop team name strings (redundant with Elo, too sparse for reliable learning) ----
    df = df.drop(columns=["home_team", "away_team"], errors="ignore")

    # ---- Drop non-feature columns ----
    # Raw box score columns (used to derive rolling advanced stats, not features themselves)
    box_drop = [f"{side}_{col.lower()}" for col in BOX_SCORE_COLS for side in ("home", "away")]
    drop_cols = [
        "game_date", "external_id", "sport", "status",
        "neutral_site", "is_playoff",
        "week", "game_id",
        # Weather (always None for NBA)
        "weather_desc", "temperature_f", "wind_mph",
    ] + box_drop
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Convert booleans to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Coerce any remaining object/string columns to numeric
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into complete (scored) rows for training.
    Returns (X, y_combined) where y_combined includes home_team_wins, home_score, away_score.
    """
    complete = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    complete = complete.dropna(subset=["home_team_wins"])
    target_cols = ["home_team_wins", "home_score", "away_score"]
    y = complete[target_cols]
    X = complete.drop(columns=target_cols, errors="ignore")
    return X, y
