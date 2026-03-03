"""
NFL-specific final feature assembly.
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

PLAYOFF_WEEK_MAP = {
    "wildcard": 19, "wc": 19,
    "division": 20, "div": 20,
    "conference": 21, "con": 21,
    "superbowl": 22, "sb": 22,
}


def _week_to_int(val) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        key = str(val).strip().lower()
        return PLAYOFF_WEEK_MAP.get(key)


def normalise_raw_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw nflreadpy.load_schedules() DataFrame to the
    canonical column names expected by build_nfl_features().

    nflreadpy uses:   gameday, location, game_type  (no weather columns)
    canonical names:  game_date, neutral_site, is_playoff, weather_desc
    """
    df = df.copy()

    # Rename date column
    if "gameday" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"gameday": "game_date"})

    # Derive is_playoff from game_type
    if "is_playoff" not in df.columns:
        if "game_type" in df.columns:
            df["is_playoff"] = df["game_type"].isin(["WC", "DIV", "CON", "SB"])
        else:
            df["is_playoff"] = False

    # Derive neutral_site from location
    if "neutral_site" not in df.columns:
        if "location" in df.columns:
            df["neutral_site"] = df["location"] == "Neutral"
        else:
            df["neutral_site"] = False

    # Add empty weather columns (nfl_data_py schedules have no weather)
    for col in ("weather_desc", "temperature_f", "wind_mph"):
        if col not in df.columns:
            df[col] = None

    return df


def build_nfl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full NFL feature engineering pipeline.

    Accepts either a raw nfl_data_py.import_schedules() DataFrame OR one
    already normalised by normalise_raw_schedules(). Column normalisation
    is applied automatically.

    Returns feature matrix ready for XGBoost (all numeric).
    """
    df = normalise_raw_schedules(df)
    df = df.sort_values("game_date").reset_index(drop=True)

    # ---- Basic derived columns ----
    if "home_team_wins" not in df.columns:
        df["home_team_wins"] = np.where(
            df["home_score"] > df["away_score"], 1,
            np.where(df["home_score"] < df["away_score"], 0, np.nan)
        )

    # Numeric week
    df["week"] = df["week"].apply(_week_to_int)
    df["schedule_playoff"] = df["is_playoff"].fillna(False).astype(int)
    df["stadium_neutral"] = df["neutral_site"].fillna(False).astype(int)
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

    # ---- One-hot encode categoricals ----
    # Team columns: drop as strings (redundant with Elo, too sparse for reliable learning)
    df = df.drop(columns=["home_team", "away_team"], errors="ignore")
    # Keep weather/roof/surface OHE for NFL
    df["weather_desc"] = df["weather_desc"].fillna("Unknown")
    cat_cols = [c for c in ["weather_desc", "roof", "surface"] if c in df.columns]
    cat_prefix_map = {"weather_desc": "weather", "roof": "roof", "surface": "surface"}
    if cat_cols:
        ohe_df = pd.get_dummies(df[cat_cols], prefix=[cat_prefix_map[c] for c in cat_cols])
        df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)

    # ---- Drop non-feature columns ----
    drop_cols = [
        # Canonical internal columns
        "game_date", "external_id", "sport", "status",
        "week_raw", "game_type", "neutral_site_raw", "neutral_site", "is_playoff",
        # Direct outcome leakage
        "result", "total", "overtime",
        # Duplicate rest (pipeline computes home_rest_days/away_rest_days)
        "away_rest", "home_rest",
        # External IDs — pure noise for model
        "game_id", "old_game_id", "gsis", "nfl_detail_id",
        "pfr", "pff", "espn", "ftn",
        # High-cardinality strings that degrade model
        "stadium_id", "stadium", "referee",
        "home_qb_id", "away_qb_id", "home_coach", "away_coach",
        "home_qb_name", "away_qb_name",
        # Temporal strings — low-signal or derivable from game_date already
        "weekday", "gametime",
        # Already encoded as stadium_neutral
        "location",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Convert booleans to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Coerce any remaining object/string columns to numeric (e.g. temperature_f, wind_mph)
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
