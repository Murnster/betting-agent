"""
Shared feature engineering:
  - 5-game rolling averages (points scored/allowed)
  - Win/loss streaks and last-5 records
  - Head-to-head historical records (last 5 matchups)
  - Rest days between games
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_averages(
    df: pd.DataFrame, windows: list[int] | int = 5,
) -> pd.DataFrame:
    """
    For each row, compute rolling averages (last N games before this game)
    for both home and away teams: points scored and points allowed.

    windows: single int or list of ints for multi-scale rolling windows.
    Required columns: game_date, home_team, away_team, home_score, away_score
    Appended columns per window W: home_avg_scored_Wg, home_avg_allowed_Wg, etc.
    For backwards compatibility, window=5 also creates un-suffixed columns.
    """
    if isinstance(windows, int):
        windows = [windows]

    df = df.copy().sort_values("game_date").reset_index(drop=True)

    # Initialize columns for each window
    for w in windows:
        suffix = f"_{w}g"
        for col in ["home_avg_scored", "home_avg_allowed", "away_avg_scored", "away_avg_allowed"]:
            df[f"{col}{suffix}"] = np.nan

    # Also create un-suffixed columns pointing to the primary (first) window
    primary_suffix = f"_{windows[0]}g"

    team_history: dict[str, list[tuple[float, float]]] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        team_history.setdefault(home, [])
        team_history.setdefault(away, [])

        for w in windows:
            suffix = f"_{w}g"
            home_past = team_history[home][-w:]
            away_past = team_history[away][-w:]

            if home_past:
                df.at[idx, f"home_avg_scored{suffix}"] = np.mean([g[0] for g in home_past])
                df.at[idx, f"home_avg_allowed{suffix}"] = np.mean([g[1] for g in home_past])

            if away_past:
                df.at[idx, f"away_avg_scored{suffix}"] = np.mean([g[0] for g in away_past])
                df.at[idx, f"away_avg_allowed{suffix}"] = np.mean([g[1] for g in away_past])

        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if not pd.isna(h_score) and not pd.isna(a_score):
            team_history[home].append((float(h_score), float(a_score)))
            team_history[away].append((float(a_score), float(h_score)))

    # Create un-suffixed aliases for backwards compatibility
    for col in ["home_avg_scored", "home_avg_allowed", "away_avg_scored", "away_avg_allowed"]:
        df[col] = df[f"{col}{primary_suffix}"]

    return df


def compute_streaks_and_last5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win/loss streaks and last-5 record for home and away teams.

    Streak: positive = win streak length, negative = loss streak length.
    Last5: sum of results over last 5 games (+1 win, -1 loss).

    Appended columns: home_streak, away_streak, home_last5, away_last5
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)

    df["home_result"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] < df["away_score"], -1, 0)
    )
    df["away_result"] = -df["home_result"]

    team_results: dict[str, list[int]] = {}

    home_streaks, away_streaks, home_last5s, away_last5s = [], [], [], []

    def _streak(history: list[int]) -> int | None:
        if not history:
            return None
        streak = 0
        last = history[-1]
        for r in reversed(history):
            if r == last:
                streak += r
            else:
                break
        return streak

    def _last5(history: list[int]) -> int | None:
        if not history:
            return None
        return sum(history[-5:])

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_streaks.append(_streak(team_results.get(home, [])))
        away_streaks.append(_streak(team_results.get(away, [])))
        home_last5s.append(_last5(team_results.get(home, [])))
        away_last5s.append(_last5(team_results.get(away, [])))

        h_res = int(row["home_result"])
        a_res = int(row["away_result"])
        team_results.setdefault(home, []).append(h_res)
        team_results.setdefault(away, []).append(a_res)

    df["home_streak"] = home_streaks
    df["away_streak"] = away_streaks
    df["home_last5"] = home_last5s
    df["away_last5"] = away_last5s
    df = df.drop(columns=["home_result", "away_result"])
    return df


def calculate_head_to_head(df: pd.DataFrame, window: int = 5) -> list[int | None]:
    """
    For each game, calculate home team's win/loss record vs away team
    in their last `window` matchups. Positive = home team winning, negative = away winning.
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    results = []

    for idx, row in df.iterrows():
        d = row["game_date"]
        home = row["home_team"]
        away = row["away_team"]

        past = df[
            (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
            & (df["game_date"] < d)
        ].sort_values("game_date").tail(window)

        score = 0
        for _, m in past.iterrows():
            if pd.isna(m["home_score"]) or pd.isna(m["away_score"]):
                continue
            if float(m["home_score"]) > float(m["away_score"]):
                winner = m["home_team"]
            elif float(m["home_score"]) < float(m["away_score"]):
                winner = m["away_team"]
            else:
                continue
            score += 1 if winner == home else -1

        results.append(score)

    return results


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for home and away team between games.
    Appended columns: home_rest_days, away_rest_days
    """
    df = df.copy().sort_values("game_date").reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])

    home_rest_map: list[float | None] = []
    away_rest_map: list[float | None] = []
    last_game: dict[str, pd.Timestamp] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        gd: pd.Timestamp = row["game_date"]

        home_rest_map.append(
            (gd - last_game[home]).days if home in last_game else None
        )
        away_rest_map.append(
            (gd - last_game[away]).days if away in last_game else None
        )

        last_game[home] = gd
        last_game[away] = gd

    df["home_rest_days"] = home_rest_map
    df["away_rest_days"] = away_rest_map
    return df
