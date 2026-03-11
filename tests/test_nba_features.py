"""Unit tests for NBA feature pipeline (no network calls)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from betting_agent.sports.nba.features import (
    DIVISIONS,
    EASTERN_CONFERENCE,
    WESTERN_CONFERENCE,
    _ADVANCED_STATS,
    build_nba_features,
    compute_nba_advanced_rolling,
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
            # Box score columns
            "home_fgm": 38 + i % 5,
            "home_fga": 85 + i % 8,
            "home_fg3m": 10 + i % 6,
            "home_fta": 18 + i % 7,
            "home_oreb": 8 + i % 5,
            "home_ast": 22 + i % 6,
            "home_tov": 12 + i % 4,
            "away_fgm": 36 + i % 5,
            "away_fga": 83 + i % 8,
            "away_fg3m": 9 + i % 6,
            "away_fta": 17 + i % 7,
            "away_oreb": 7 + i % 5,
            "away_ast": 20 + i % 6,
            "away_tov": 13 + i % 4,
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
        assert result["is_playoff"].iloc[0]


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

    def test_nba_quality_and_diff_columns_exist(self):
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        expected = [
            "home_win_pct_season",
            "away_win_pct_season",
            "home_win_pct_10g",
            "away_win_pct_10g",
            "home_opp_elo_10g",
            "away_opp_elo_10g",
            "elo_diff",
            "recent_elo_diff",
            "playoff_elo_diff",
            "avg_scored_diff_3g",
            "avg_allowed_diff_3g",
            "ortg_diff_5g",
            "drtg_diff_10g",
        ]
        for col in expected:
            assert col in result.columns, f"Missing NBA quality/diff column: {col}"

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

    def test_advanced_rolling_columns_in_output(self):
        """Advanced rolling columns should be present, raw box cols dropped."""
        df = _make_synthetic_schedule(50)
        result = build_nba_features(df)
        # At least some advanced rolling columns should exist
        adv_cols = [c for c in result.columns if "ts_pct" in c or "ortg" in c]
        assert len(adv_cols) > 0, "No advanced rolling columns found"
        # Raw box score columns should be dropped
        for col in ["home_fgm", "away_fga", "home_oreb", "away_tov"]:
            assert col not in result.columns, f"Raw box col {col} should be dropped"


class TestNbaAdvancedRolling:
    def test_rolling_columns_created(self):
        """All 54 columns (8 stats × 2 sides × 3 windows) should be created."""
        df = _make_synthetic_schedule(50)
        df = normalise_nba_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nba_advanced_rolling(df, windows=[3, 5, 10])
        expected_count = len(_ADVANCED_STATS) * 2 * 3  # 8 * 2 * 3 = 48
        adv_cols = [
            c for c in result.columns
            if any(c.endswith(f"_{w}g") and any(s in c for s in _ADVANCED_STATS)
                   for w in [3, 5, 10])
        ]
        assert len(adv_cols) == expected_count, (
            f"Expected {expected_count} advanced columns, got {len(adv_cols)}: {adv_cols}"
        )

    def test_first_game_has_nan_rolling(self):
        """First game for a team has no history → NaN rolling stats."""
        df = _make_synthetic_schedule(50)
        df = normalise_nba_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nba_advanced_rolling(df, windows=[3, 5, 10])
        assert pd.isna(result.at[0, "home_ts_pct_3g"])

    def test_rolling_values_reasonable(self):
        """TS% should be in [0, 1], ORTG in [50, 200] for reasonable data."""
        df = _make_synthetic_schedule(50)
        df = normalise_nba_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nba_advanced_rolling(df, windows=[5])
        ts = result["home_ts_pct_5g"].dropna()
        assert (ts >= 0).all() and (ts <= 1).all(), "TS% out of [0, 1]"
        ortg = result["home_ortg_5g"].dropna()
        assert (ortg >= 50).all() and (ortg <= 200).all(), "ORTG out of [50, 200]"

    def test_no_box_cols_returns_unchanged(self):
        """If box score columns are absent, return df unchanged."""
        df = _make_synthetic_schedule(10)
        # Remove box score columns
        box_cols = [c for c in df.columns if c.startswith(("home_fg", "away_fg",
                    "home_fta", "away_fta", "home_oreb", "away_oreb",
                    "home_ast", "away_ast", "home_tov", "away_tov"))]
        df = df.drop(columns=box_cols)
        original_cols = set(df.columns)
        result = compute_nba_advanced_rolling(df, windows=[3, 5, 10])
        assert set(result.columns) == original_cols

    def test_division_by_zero_safe(self):
        """FGA=0 shouldn't crash — should produce NaN for that game's stats."""
        df = _make_synthetic_schedule(10)
        df = normalise_nba_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        # Set FGA=0 for first game
        df.at[0, "home_fga"] = 0
        df.at[0, "home_fgm"] = 0
        # Should not raise
        result = compute_nba_advanced_rolling(df, windows=[3])
        assert result is not None


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
