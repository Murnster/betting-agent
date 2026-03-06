"""
NHL-specific final feature assembly.
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
from betting_agent.sports.nhl.loader import BOX_SCORE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conference / Division constants (all 32 NHL teams by abbreviation)
# ---------------------------------------------------------------------------
EASTERN_CONFERENCE = {
    # Atlantic
    "BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR",
    # Metropolitan
    "CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH",
}

WESTERN_CONFERENCE = {
    # Central
    "ARI", "CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG",
    # Pacific
    "ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK",
}

DIVISIONS: dict[str, set[str]] = {
    "Atlantic": {"BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"},
    "Metropolitan": {"CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"},
    "Central": {"ARI", "CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"},
    "Pacific": {"ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"},
}

# Reverse lookup: team -> division name
_TEAM_TO_DIVISION: dict[str, str] = {}
for div, teams in DIVISIONS.items():
    for t in teams:
        _TEAM_TO_DIVISION[t] = div

# ---------------------------------------------------------------------------
# Advanced stats names
# ---------------------------------------------------------------------------
_ADVANCED_STATS = [
    "corsi_proxy", "shot_diff", "pp_pct", "pk_pct",
    "save_pct", "shooting_pct", "pdo", "faceoff_win_pct",
]


def _compute_game_advanced(
    goals_for: float,
    goals_against: float,
    sog: float,
    opp_sog: float,
    ppg: float,
    ppo: float,
    opp_ppg: float,
    opp_ppo: float,
    blocks: float,
    opp_blocks: float,
    faceoff_pct: float | None,
) -> dict[str, float]:
    """Compute advanced NHL stats for a single team-game."""
    # Corsi proxy: shots on goal + blocked shots
    corsi_proxy = sog + blocks
    shot_diff = sog - opp_sog

    # Power play %
    pp_pct = ppg / ppo if ppo > 0 else np.nan

    # Penalty kill %: 1 - (opp_ppg / opp_ppo)
    pk_pct = 1.0 - (opp_ppg / opp_ppo) if opp_ppo > 0 else np.nan

    # Save %: 1 - (goals_against / opp_sog)
    save_pct = 1.0 - (goals_against / opp_sog) if opp_sog > 0 else np.nan

    # Shooting %: goals / sog
    shooting_pct = goals_for / sog if sog > 0 else np.nan

    # PDO: save% + shooting%
    pdo = (save_pct if not np.isnan(save_pct) else 0) + (
        shooting_pct if not np.isnan(shooting_pct) else 0
    )
    if np.isnan(save_pct) and np.isnan(shooting_pct):
        pdo = np.nan

    # Faceoff win %
    faceoff_win_pct = faceoff_pct if faceoff_pct is not None else np.nan

    return {
        "corsi_proxy": corsi_proxy,
        "shot_diff": shot_diff,
        "pp_pct": pp_pct,
        "pk_pct": pk_pct,
        "save_pct": save_pct,
        "shooting_pct": shooting_pct,
        "pdo": pdo,
        "faceoff_win_pct": faceoff_win_pct,
    }


def compute_nhl_advanced_rolling(
    df: pd.DataFrame, windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling advanced hockey stats for home and away teams.

    Requires box score columns (home_sog, away_sog, etc.) from the NHL loader.
    If absent, returns df unchanged.

    Produces 8 stats x 2 sides x len(windows) columns.
    """
    if windows is None:
        windows = [3, 5, 10]

    # Guard: need box score columns to compute anything
    if "home_sog" not in df.columns:
        return df

    df = df.copy()

    # Initialize output columns
    for side in ("home", "away"):
        for stat in _ADVANCED_STATS:
            for w in windows:
                df[f"{side}_{stat}_{w}g"] = np.nan

    # Team history: team -> list of per-game stat dicts
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

        home_sog = row.get("home_sog")
        if home_sog is None or pd.isna(home_sog):
            continue

        def _f(val: object) -> float:
            """Coerce to float, treating None/NaN as 0."""
            if val is None:
                return 0.0
            try:
                v = float(val)
                return 0.0 if np.isnan(v) else v
            except (ValueError, TypeError):
                return 0.0

        # Home team stats
        home_stats = _compute_game_advanced(
            goals_for=float(home_score),
            goals_against=float(away_score),
            sog=_f(row.get("home_sog")),
            opp_sog=_f(row.get("away_sog")),
            ppg=_f(row.get("home_ppg")),
            ppo=_f(row.get("home_ppo")),
            opp_ppg=_f(row.get("away_ppg")),
            opp_ppo=_f(row.get("away_ppo")),
            blocks=_f(row.get("home_blocks")),
            opp_blocks=_f(row.get("away_blocks")),
            faceoff_pct=row.get("home_faceoff_pct"),
        )
        team_history[home].append(home_stats)

        # Away team stats
        away_stats = _compute_game_advanced(
            goals_for=float(away_score),
            goals_against=float(home_score),
            sog=_f(row.get("away_sog")),
            opp_sog=_f(row.get("home_sog")),
            ppg=_f(row.get("away_ppg")),
            ppo=_f(row.get("away_ppo")),
            opp_ppg=_f(row.get("home_ppg")),
            opp_ppo=_f(row.get("home_ppo")),
            blocks=_f(row.get("away_blocks")),
            opp_blocks=_f(row.get("home_blocks")),
            faceoff_pct=row.get("away_faceoff_pct"),
        )
        team_history[away].append(away_stats)

    return df


def normalise_nhl_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical columns exist for build_nhl_features().
    NHL data from NHLLoader already uses canonical names, but this
    fills any missing optional columns so downstream code is safe.
    """
    df = df.copy()

    if "is_playoff" not in df.columns:
        df["is_playoff"] = False

    if "neutral_site" not in df.columns:
        df["neutral_site"] = False

    # NHL is indoor - no weather
    for col in ("weather_desc", "temperature_f", "wind_mph"):
        if col not in df.columns:
            df[col] = None

    return df


def build_nhl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full NHL feature engineering pipeline.

    Returns feature matrix ready for XGBoost (all numeric).
    """
    df = normalise_nhl_schedules(df)
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
    df = compute_nhl_advanced_rolling(df, windows=[3, 5, 10])
    df = compute_streaks_and_last5(df)
    df["head_to_head"] = calculate_head_to_head(df)
    df = add_rest_days(df)
    df = calculate_elo_ratings(df)

    # ---- Spread-derived home_is_favorite ----
    if "home_is_favorite" not in df.columns:
        df["home_is_favorite"] = -1

    # ---- NHL-specific features ----
    # Back-to-back detection
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

    # ---- Drop team name strings ----
    df = df.drop(columns=["home_team", "away_team"], errors="ignore")

    # ---- Drop non-feature columns ----
    box_drop = [f"{side}_{col}" for col in BOX_SCORE_COLS for side in ("home", "away")]
    drop_cols = [
        "game_date", "external_id", "sport", "status",
        "neutral_site", "is_playoff",
        "week", "game_id",
        # Weather (always None for NHL)
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
