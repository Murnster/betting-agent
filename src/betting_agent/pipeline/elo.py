"""
5-variant Elo rating system adapted from the reference project.
Variants: general, home, away, playoff, recent (L10).
Mean-reverts 1/3 toward base at each season boundary. Base=1000.

Operates on a pandas DataFrame for compatibility with the ML pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START_ELO = 1000.0
BASE_CHANGE = 20.0
MEAN_REVERSION_FACTOR = 1 / 3  # revert 1/3 toward base each season


def _margin_multiplier(margin: float) -> float:
    return 1.0 + min(margin, 30.0) / 30.0


def calculate_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 5 Elo variants for each game row (in chronological order).

    Required columns:
      season, home_team, away_team, home_score (nullable), away_score (nullable),
      is_playoff (bool)

    Appended columns:
      home_elo, away_elo,
      home_home_elo, home_away_elo, away_home_elo, away_away_elo,
      home_playoff_elo, away_playoff_elo,
      home_recent_elo, away_recent_elo
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)

    variants = [
        "general", "home", "away", "playoff", "recent",
    ]

    elo_dict: dict[str, dict[str, float]] = {v: {} for v in variants}
    recent_changes: dict[str, list[float]] = {}
    last_season: int | None = None

    elo_cols = [
        "home_elo", "away_elo",
        "home_home_elo", "home_away_elo", "away_home_elo", "away_away_elo",
        "home_playoff_elo", "away_playoff_elo",
        "home_recent_elo", "away_recent_elo",
    ]
    for col in elo_cols:
        df[col] = np.nan

    def _init_team(team: str) -> None:
        for v in variants:
            elo_dict[v].setdefault(team, START_ELO)
        recent_changes.setdefault(team, [])

    all_teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())

    for idx, row in df.iterrows():
        year: int = int(row["season"])
        home: str = str(row["home_team"])
        away: str = str(row["away_team"])

        # Mean-revert 1/3 toward base at each new season boundary
        if last_season is not None and year != last_season:
            for v in variants:
                for t in list(elo_dict[v].keys()):
                    elo_dict[v][t] = elo_dict[v][t] + MEAN_REVERSION_FACTOR * (START_ELO - elo_dict[v][t])
            recent_changes = {t: [] for t in all_teams}
        last_season = year

        _init_team(home)
        _init_team(away)

        # Snapshot pre-game Elos
        df.at[idx, "home_elo"] = elo_dict["general"][home]
        df.at[idx, "away_elo"] = elo_dict["general"][away]
        df.at[idx, "home_home_elo"] = elo_dict["home"][home]
        df.at[idx, "home_away_elo"] = elo_dict["away"][home]
        df.at[idx, "away_home_elo"] = elo_dict["home"][away]
        df.at[idx, "away_away_elo"] = elo_dict["away"][away]
        df.at[idx, "home_playoff_elo"] = elo_dict["playoff"][home]
        df.at[idx, "away_playoff_elo"] = elo_dict["playoff"][away]
        df.at[idx, "home_recent_elo"] = elo_dict["recent"][home]
        df.at[idx, "away_recent_elo"] = elo_dict["recent"][away]

        # Skip update if no final score
        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if pd.isna(h_score) or pd.isna(a_score):
            continue

        h_score = float(h_score)
        a_score = float(a_score)
        if h_score == a_score:
            continue  # skip ties

        home_result = 1.0 if h_score > a_score else 0.0
        away_result = 1.0 - home_result
        margin = abs(h_score - a_score)
        multiplier = _margin_multiplier(margin)

        home_elo = elo_dict["general"][home]
        away_elo = elo_dict["general"][away]
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
        expected_away = 1.0 - expected_home

        home_change = BASE_CHANGE * (home_result - expected_home) * multiplier
        away_change = BASE_CHANGE * (away_result - expected_away) * multiplier

        # Update general
        elo_dict["general"][home] += home_change
        elo_dict["general"][away] += away_change

        # Home/Away venue-specific
        elo_dict["home"][home] += home_change
        elo_dict["away"][away] += away_change

        # Playoff
        is_playoff = bool(row.get("is_playoff", False))
        if is_playoff:
            elo_dict["playoff"][home] += home_change
            elo_dict["playoff"][away] += away_change

        # Recent form (rolling last 10 changes)
        for team, change in [(home, home_change), (away, away_change)]:
            recent_changes[team].append(change)
            if len(recent_changes[team]) > 10:
                recent_changes[team].pop(0)
            elo_dict["recent"][team] = START_ELO + sum(recent_changes[team])

    return df
