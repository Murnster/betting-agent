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
    df = compute_streaks_and_last5(df)
    df["head_to_head"] = calculate_head_to_head(df)
    df = add_rest_days(df)
    df = calculate_elo_ratings(df)

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
    drop_cols = [
        "game_date", "external_id", "sport", "status",
        "neutral_site", "is_playoff",
        "week", "game_id",
        # Weather (always None for NBA)
        "weather_desc", "temperature_f", "wind_mph",
    ]
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
