"""
MLB-specific final feature assembly.
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
from betting_agent.sports.mlb.loader import BOX_SCORE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League / Division constants (all 30 MLB teams by abbreviation)
# ---------------------------------------------------------------------------
AMERICAN_LEAGUE = {
    # AL East
    "BAL", "BOS", "NYY", "TB", "TOR",
    # AL Central
    "CLE", "CWS", "DET", "KC", "MIN",
    # AL West
    "HOU", "LAA", "ATH", "SEA", "TEX",
}

NATIONAL_LEAGUE = {
    # NL East
    "ATL", "MIA", "NYM", "PHI", "WSH",
    # NL Central
    "CHC", "CIN", "MIL", "PIT", "STL",
    # NL West
    "AZ", "COL", "LAD", "SD", "SF",
}

DIVISIONS: dict[str, set[str]] = {
    "AL East": {"BAL", "BOS", "NYY", "TB", "TOR"},
    "AL Central": {"CLE", "CWS", "DET", "KC", "MIN"},
    "AL West": {"HOU", "LAA", "ATH", "SEA", "TEX"},
    "NL East": {"ATL", "MIA", "NYM", "PHI", "WSH"},
    "NL Central": {"CHC", "CIN", "MIL", "PIT", "STL"},
    "NL West": {"AZ", "COL", "LAD", "SD", "SF"},
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
    "ops", "whip", "k_per_9", "bb_per_9",
    "batting_avg", "obp", "slg", "era", "run_diff",
]


def _compute_game_advanced(
    runs_scored: float,
    runs_allowed: float,
    hits: float,
    walks: float,
    home_runs: float,
    doubles: float,
    triples: float,
    at_bats: float,
    stolen_bases: float,
    strikeouts: float,
    p_earned_runs: float,
    p_innings: float,
    p_strikeouts: float,
    p_walks: float,
    p_hits_allowed: float,
) -> dict[str, float]:
    """Compute advanced MLB stats for a single team-game."""
    # Batting avg
    batting_avg = hits / at_bats if at_bats > 0 else np.nan

    # OBP: (H + BB) / (AB + BB) — simplified, ignoring HBP/SF
    obp = (hits + walks) / (at_bats + walks) if (at_bats + walks) > 0 else np.nan

    # SLG: total bases / AB
    total_bases = hits + doubles + 2 * triples + 3 * home_runs
    slg = total_bases / at_bats if at_bats > 0 else np.nan

    # OPS
    ops = (obp if not np.isnan(obp) else 0) + (slg if not np.isnan(slg) else 0)
    if np.isnan(obp) and np.isnan(slg):
        ops = np.nan

    # ERA: (earned_runs / innings) * 9
    era = (p_earned_runs / p_innings) * 9 if p_innings > 0 else np.nan

    # WHIP: (walks + hits) / innings
    whip = (p_walks + p_hits_allowed) / p_innings if p_innings > 0 else np.nan

    # K/9: strikeouts / innings * 9
    k_per_9 = (p_strikeouts / p_innings) * 9 if p_innings > 0 else np.nan

    # BB/9: walks / innings * 9
    bb_per_9 = (p_walks / p_innings) * 9 if p_innings > 0 else np.nan

    # Run differential
    run_diff = runs_scored - runs_allowed

    return {
        "ops": ops,
        "whip": whip,
        "k_per_9": k_per_9,
        "bb_per_9": bb_per_9,
        "batting_avg": batting_avg,
        "obp": obp,
        "slg": slg,
        "era": era,
        "run_diff": run_diff,
    }


def compute_mlb_advanced_rolling(
    df: pd.DataFrame, windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling advanced baseball stats for home and away teams.

    Requires box score columns (home_hits, away_hits, etc.) from the MLB loader.
    If absent, returns df unchanged.

    Produces 9 stats x 2 sides x len(windows) columns.
    """
    if windows is None:
        windows = [3, 5, 10]

    if "home_hits" not in df.columns:
        return df

    df = df.copy()

    for side in ("home", "away"):
        for stat in _ADVANCED_STATS:
            for w in windows:
                df[f"{side}_{stat}_{w}g"] = np.nan

    team_history: dict[str, list[dict[str, float]]] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        team_history.setdefault(home, [])
        team_history.setdefault(away, [])

        for side, team in [("home", home), ("away", away)]:
            hist = team_history[team]
            if hist:
                for w in windows:
                    recent = hist[-w:]
                    for stat in _ADVANCED_STATS:
                        vals = [g[stat] for g in recent if not np.isnan(g[stat])]
                        if vals:
                            df.at[idx, f"{side}_{stat}_{w}g"] = np.mean(vals)

        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_hits = row.get("home_hits")
        if home_hits is None or pd.isna(home_hits):
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

        # Estimate at-bats from runs + hits (rough: AB ≈ hits + strikeouts + outs)
        # Better: use hits, runs_scored, walks, strikeouts to estimate
        # For simplicity: AB ≈ (innings_pitched * 3) for the opposing pitcher = batters retired
        # Actually just use a reasonable estimate: AB ≈ IP*3 + H
        home_ip = _f(row.get("away_innings_pitched"))  # opponent pitched to home batters
        home_ab = max(home_ip * 3, _f(row.get("home_hits")) + _f(row.get("home_strikeouts")))
        away_ip = _f(row.get("home_innings_pitched"))
        away_ab = max(away_ip * 3, _f(row.get("away_hits")) + _f(row.get("away_strikeouts")))

        home_stats = _compute_game_advanced(
            runs_scored=float(home_score),
            runs_allowed=float(away_score),
            hits=_f(row.get("home_hits")),
            walks=_f(row.get("home_walks")),
            home_runs=_f(row.get("home_home_runs")),
            doubles=_f(row.get("home_doubles")),
            triples=_f(row.get("home_triples")),
            at_bats=home_ab,
            stolen_bases=_f(row.get("home_stolen_bases")),
            strikeouts=_f(row.get("home_strikeouts")),
            p_earned_runs=_f(row.get("home_earned_runs")),
            p_innings=_f(row.get("home_innings_pitched")),
            p_strikeouts=_f(row.get("home_pitching_strikeouts")),
            p_walks=_f(row.get("home_pitching_walks")),
            p_hits_allowed=_f(row.get("home_hits_allowed")),
        )
        team_history[home].append(home_stats)

        away_stats = _compute_game_advanced(
            runs_scored=float(away_score),
            runs_allowed=float(home_score),
            hits=_f(row.get("away_hits")),
            walks=_f(row.get("away_walks")),
            home_runs=_f(row.get("away_home_runs")),
            doubles=_f(row.get("away_doubles")),
            triples=_f(row.get("away_triples")),
            at_bats=away_ab,
            stolen_bases=_f(row.get("away_stolen_bases")),
            strikeouts=_f(row.get("away_strikeouts")),
            p_earned_runs=_f(row.get("away_earned_runs")),
            p_innings=_f(row.get("away_innings_pitched")),
            p_strikeouts=_f(row.get("away_pitching_strikeouts")),
            p_walks=_f(row.get("away_pitching_walks")),
            p_hits_allowed=_f(row.get("away_hits_allowed")),
        )
        team_history[away].append(away_stats)

    return df


def normalise_mlb_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical columns exist for build_mlb_features().
    MLB data from MLBLoader already uses canonical names, but this
    fills any missing optional columns so downstream code is safe.
    """
    df = df.copy()

    if "is_playoff" not in df.columns:
        df["is_playoff"] = False

    if "neutral_site" not in df.columns:
        df["neutral_site"] = False

    for col in ("weather_desc", "temperature_f", "wind_mph"):
        if col not in df.columns:
            df[col] = None

    return df


def build_mlb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full MLB feature engineering pipeline.

    Returns feature matrix ready for XGBoost (all numeric).
    """
    df = normalise_mlb_schedules(df)
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
    df = compute_mlb_advanced_rolling(df, windows=[3, 5, 10])
    df = compute_streaks_and_last5(df)
    df["head_to_head"] = calculate_head_to_head(df)
    df = add_rest_days(df)
    df = calculate_elo_ratings(df)

    # ---- Spread-derived home_is_favorite ----
    if "home_is_favorite" not in df.columns:
        df["home_is_favorite"] = -1

    # ---- MLB-specific features ----
    # Interleague flag
    home_is_al = df["home_team"].isin(AMERICAN_LEAGUE)
    away_is_al = df["away_team"].isin(AMERICAN_LEAGUE)
    df["interleague"] = (home_is_al != away_is_al).astype(int)

    # League features
    df["home_is_al"] = home_is_al.astype(int)
    df["away_is_al"] = away_is_al.astype(int)

    # Division feature
    home_div = df["home_team"].map(_TEAM_TO_DIVISION)
    away_div = df["away_team"].map(_TEAM_TO_DIVISION)
    df["same_division"] = (home_div == away_div).fillna(False).astype(int)

    # Day game detection (from game_date hour if available, otherwise default)
    if "day_night" in df.columns:
        df["day_game"] = (df["day_night"] == "day").astype(int)
    else:
        df["day_game"] = 0

    # ---- Drop team name strings ----
    df = df.drop(columns=["home_team", "away_team"], errors="ignore")

    # ---- Drop non-feature columns ----
    box_drop = [f"{side}_{col}" for col in BOX_SCORE_COLS for side in ("home", "away")]
    drop_cols = [
        "game_date", "external_id", "sport", "status",
        "neutral_site", "is_playoff",
        "week", "game_id",
        "weather_desc", "temperature_f", "wind_mph",
        "day_night",
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
