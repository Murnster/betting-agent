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

# Closing lines and prices that nflreadpy ships alongside each schedule row.
# Never features — see the drop list in build_nfl_features().
MARKET_COLS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
    "home_is_favorite",
)

UNKNOWN_VENUE = "unknown"

# Fixed one-hot vocabularies. Deriving levels from whatever happens to be in
# the frame gives training and pick time different column sets — a slate with
# no open-roof stadium would silently drop roof_open. Anything unrecognised
# (including nflreadpy's stray "grass " with a trailing space, normalised
# below) collapses into the unknown level.
ROOF_LEVELS = ("closed", "dome", "open", "outdoors", UNKNOWN_VENUE)
SURFACE_LEVELS = (
    "a_turf", "astroplay", "astroturf", "fieldturf",
    "grass", "matrixturf", "sportturf", UNKNOWN_VENUE,
)

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

    nflreadpy uses:   gameday, location, game_type, temp, wind
    canonical names:  game_date, neutral_site, is_playoff, temperature_f, wind_mph

    Must produce the same column names as NFLLoader._normalise_schedules(),
    which is the path train.py takes — any divergence between the two is
    train/serve skew.
    """
    df = df.copy()

    # Rename date column
    if "gameday" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"gameday": "game_date"})

    # Coerce to datetime. Loader-sourced history carries real dates while
    # odds-derived upcoming rows carry strings; concatenating the two gives an
    # object column that cannot be sorted.
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Weather: nflreadpy ships temp/wind for outdoor games (NaN for domes and
    # for games where it was never recorded — XGBoost reads that as missing).
    for src, dest in (("temp", "temperature_f"), ("wind", "wind_mph")):
        if src in df.columns:
            if dest in df.columns:
                df[dest] = df[dest].fillna(df[src])
                df = df.drop(columns=[src])
            else:
                df = df.rename(columns={src: dest})

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

    # Ensure the numeric weather columns exist even when the source lacks them
    # (e.g. odds-derived upcoming rows before a forecast is attached).
    for col in ("temperature_f", "wind_mph"):
        if col not in df.columns:
            df[col] = np.nan

    return df


#: Schedule facts that are published before kickoff and therefore legitimate
#: features — but only if the pick-time path actually supplies them.
SCHEDULE_CONTEXT_COLS = ("week", "roof", "surface", "div_game", "is_playoff", "game_type")


def attach_schedule_context(
    upcoming: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill venue and scheduling context on upcoming (odds-derived) rows from the
    published nflreadpy schedule, matching on home/away abbreviation.

    Odds API events carry only teams and a kickoff time. Without this join,
    roof/surface/div_game/week arrive empty at pick time while the model was
    trained on them — 15 of 63 features silently zero-filled.

    `upcoming` must already use nflreadpy abbreviations for home_team/away_team.
    Returns a copy; unmatched rows keep whatever they had.
    """
    if upcoming.empty or schedule is None or schedule.empty:
        return upcoming

    sched = schedule.copy()
    if "gameday" in sched.columns and "game_date" not in sched.columns:
        sched = sched.rename(columns={"gameday": "game_date"})
    if "game_id" in sched.columns and "external_id" not in sched.columns:
        sched["external_id"] = sched["game_id"]

    cols = [c for c in (*SCHEDULE_CONTEXT_COLS, "external_id") if c in sched.columns]
    if not cols:
        return upcoming

    sched["game_date"] = pd.to_datetime(sched.get("game_date"), errors="coerce")

    out = upcoming.copy()
    out_dates = pd.to_datetime(out["game_date"], errors="coerce")

    matches: list[pd.Series | None] = []
    for idx, row in out.iterrows():
        same_fixture = sched[
            (sched["home_team"] == row["home_team"])
            & (sched["away_team"] == row["away_team"])
        ]
        if same_fixture.empty:
            matches.append(None)
            continue
        # Teams can meet twice a season, so pick the scheduled kickoff nearest
        # the event's date rather than whichever row happens to come last.
        target = out_dates.loc[idx]
        if pd.notna(target) and same_fixture["game_date"].notna().any():
            gap = (same_fixture["game_date"] - target).abs()
            matches.append(same_fixture.loc[gap.idxmin()])
        else:
            matches.append(same_fixture.iloc[-1])

    for col in cols:
        values = [None if m is None else m.get(col) for m in matches]
        series = pd.Series(values, index=out.index)
        out[col] = series.combine_first(out[col]) if col in out.columns else series

    if "game_type" in cols:
        derived = out["game_type"].isin(["WC", "DIV", "CON", "SB"])
        out["is_playoff"] = derived.where(out["game_type"].notna(), out.get("is_playoff", False))

    hits = sum(m is not None for m in matches)
    if hits < len(matches):
        logger.warning(
            "Schedule context matched %d of %d upcoming games — unmatched rows "
            "will run on partial features. Check team-name mapping.",
            hits, len(matches),
        )
    return out


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

    # ---- One-hot encode categoricals ----
    # Team columns: drop as strings (redundant with Elo, too sparse for reliable learning)
    df = df.drop(columns=["home_team", "away_team"], errors="ignore")
    # roof/surface are venue facts published with the schedule, so they are
    # available for upcoming games too. weather_desc is not OHE'd: nflreadpy
    # never populates it, so training would only ever see one level.
    for col, levels in (("roof", ROOF_LEVELS), ("surface", SURFACE_LEVELS)):
        values = df[col] if col in df.columns else pd.Series(index=df.index, dtype=object)
        values = values.astype("string").str.strip().str.lower()
        values = values.where(values.isin(levels), UNKNOWN_VENUE)
        df[col] = pd.Categorical(values, categories=levels)

    cat_cols = ["roof", "surface"]
    ohe_df = pd.get_dummies(df[cat_cols], prefix=cat_cols)
    df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)

    # ---- Drop non-feature columns ----
    drop_cols = [
        # Canonical internal columns
        "game_date", "external_id", "sport", "status",
        "week_raw", "game_type", "neutral_site_raw", "neutral_site", "is_playoff",
        "weather_desc",
        # Direct outcome leakage
        "result", "total", "overtime",
        # Market lines and prices. These are the market's opinion, not ours:
        # training on them teaches the model to echo the closing line, and the
        # Odds API path can't supply them at pick time anyway, so reindex
        # zero-fills them and the live model runs on inputs it never saw.
        # Phase 1's backtest reads them straight off the raw schedule instead.
        *MARKET_COLS,
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
