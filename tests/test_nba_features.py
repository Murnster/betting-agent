"""Unit tests for NBA feature pipeline (no network calls)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from betting_agent.sports.nba.features import (
    DIVISIONS,
    EASTERN_CONFERENCE,
    WESTERN_CONFERENCE,
    build_nba_features,
    normalise_nba_schedules,
    split_features_targets,
)


def _make_synthetic_schedule(n_games: int = 50) -> pd.DataFrame:
    """Create a synthetic NBA schedule for testing."""
    teams = ["BOS", "MIL", "NYK", "PHI", "TOR",
             "LAL", "GSW", "DEN", "PHX", "LAC"]
    rows = []
    base_date = date(2024, 10, 22)

    for i in range(n_games):
        home_idx = i % len(teams)
        away_idx = (i + 3) % len(teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(teams)

        game_date = base_date + timedelta(days=i // 3)
        rows.append({
            "external_id": f"002240{i:04d}",
            "game_date": game_date,
            "season": 2024,
            "week": None,
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "home_score": 100 + (i * 7) % 30,
            "away_score": 95 + (i * 11) % 30,
            "is_playoff": False,
            "neutral_site": False,
            "sport": "NBA",
            "status": "final",
        })
    return pd.DataFrame(rows)


class TestNormaliseNbaSchedules:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"home_team": ["BOS"], "away_team": ["LAL"]})
        result = normalise_nba_schedules(df)
        assert "is_playoff" in result.columns
        assert "neutral_site" in result.columns
        assert "weather_desc" in result.columns
        assert "temperature_f" in result.columns
        assert "wind_mph" in result.columns

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({
            "home_team": ["BOS"],
            "is_playoff": [True],
        })
        result = normalise_nba_schedules(df)
        assert result["is_playoff"].iloc[0] == True


class TestBuildNbaFeatures:
    def test_returns_all_numeric(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        for col in result.columns:
            assert result[col].dtype in [np.float64, np.int64, np.int32, np.float32,
                                         np.uint8, np.bool_], \
                f"Column {col} has non-numeric dtype {result[col].dtype}"

    def test_nba_specific_columns_exist(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        for col in ["home_back_to_back", "away_back_to_back",
                     "home_is_eastern", "away_is_eastern",
                     "cross_conference", "same_division"]:
            assert col in result.columns, f"Missing NBA column: {col}"

    def test_no_weather_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        for col in ["weather_desc", "temperature_f", "wind_mph"]:
            assert col not in result.columns

    def test_no_game_date_column(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        assert "game_date" not in result.columns

    def test_no_team_ohe_columns(self):
        """Team OHE columns were removed (too sparse, redundant with Elo)."""
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        team_ohe = [c for c in result.columns if c.startswith("home_BOS") or c.startswith("away_BOS")]
        assert len(team_ohe) == 0, "Team OHE columns should not be present"

    def test_no_team_string_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        assert "home_team" not in result.columns
        assert "away_team" not in result.columns

    def test_preserves_upcoming_tag(self):
        """_is_upcoming tag should survive through the pipeline."""
        df = _make_synthetic_schedule(50)
        df["_is_upcoming"] = False
        df.loc[df.index[-3:], "_is_upcoming"] = True
        df.loc[df.index[-3:], "home_score"] = None
        df.loc[df.index[-3:], "away_score"] = None
        result = build_nba_features(df)
        assert "_is_upcoming" in result.columns
        assert result["_is_upcoming"].sum() == 3


class TestSplitFeaturesTargets:
    def test_split_basic(self):
        df = _make_synthetic_schedule(50)
        features = build_nba_features(df)
        X, y = split_features_targets(features)
        assert "home_team_wins" not in X.columns
        assert "home_score" not in X.columns
        assert "away_score" not in X.columns
        assert "home_team_wins" in y.columns
        assert len(X) == len(y)
        assert len(X) > 0

    def test_excludes_unscored_games(self):
        df = _make_synthetic_schedule(50)
        df.loc[df.index[-5:], "home_score"] = None
        df.loc[df.index[-5:], "away_score"] = None
        features = build_nba_features(df)
        X, y = split_features_targets(features)
        # Should have fewer rows than total due to unscored games
        assert len(X) < 50


class TestConferenceConstants:
    def test_15_teams_per_conference(self):
        assert len(EASTERN_CONFERENCE) == 15
        assert len(WESTERN_CONFERENCE) == 15

    def test_no_overlap(self):
        assert EASTERN_CONFERENCE & WESTERN_CONFERENCE == set()

    def test_30_total(self):
        assert len(EASTERN_CONFERENCE | WESTERN_CONFERENCE) == 30

    def test_6_divisions(self):
        assert len(DIVISIONS) == 6
        all_teams = set()
        for teams in DIVISIONS.values():
            assert len(teams) == 5
            all_teams |= teams
        assert len(all_teams) == 30
