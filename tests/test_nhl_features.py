"""Unit tests for NHL feature pipeline (no network calls)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from betting_agent.sports.nhl.features import (
    DIVISIONS,
    EASTERN_CONFERENCE,
    WESTERN_CONFERENCE,
    _ADVANCED_STATS,
    build_nhl_features,
    compute_nhl_advanced_rolling,
    normalise_nhl_schedules,
    split_features_targets,
)


def _make_synthetic_schedule(n_games: int = 50) -> pd.DataFrame:
    """Create a synthetic NHL schedule for testing."""
    teams = ["TOR", "BOS", "MTL", "NYR", "PIT",
             "COL", "EDM", "VGK", "DAL", "SEA"]
    rows = []
    base_date = date(2024, 10, 10)

    for i in range(n_games):
        home_idx = i % len(teams)
        away_idx = (i + 3) % len(teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(teams)

        game_date = base_date + timedelta(days=i // 3)
        rows.append({
            "external_id": f"202302{i:04d}",
            "game_date": game_date,
            "season": 2024,
            "week": None,
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "home_score": 2 + (i * 3) % 5,
            "away_score": 1 + (i * 7) % 5,
            "is_playoff": False,
            "neutral_site": False,
            "sport": "NHL",
            "status": "final",
            # Box score columns
            "home_sog": 28 + i % 10,
            "home_ppg": i % 3,
            "home_ppo": 2 + i % 4,
            "home_pim": 6 + i % 8,
            "home_hits": 20 + i % 15,
            "home_blocks": 12 + i % 8,
            "home_faceoff_pct": 0.45 + (i % 10) * 0.01,
            "away_sog": 25 + i % 10,
            "away_ppg": i % 2,
            "away_ppo": 2 + i % 3,
            "away_pim": 8 + i % 6,
            "away_hits": 18 + i % 12,
            "away_blocks": 10 + i % 8,
            "away_faceoff_pct": 0.48 + (i % 8) * 0.01,
        })
    return pd.DataFrame(rows)


class TestNormaliseNhlSchedules:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"home_team": ["TOR"], "away_team": ["MTL"]})
        result = normalise_nhl_schedules(df)
        assert "is_playoff" in result.columns
        assert "neutral_site" in result.columns
        assert "weather_desc" in result.columns
        assert "temperature_f" in result.columns
        assert "wind_mph" in result.columns

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({
            "home_team": ["TOR"],
            "is_playoff": [True],
        })
        result = normalise_nhl_schedules(df)
        assert result["is_playoff"].iloc[0] == True  # noqa: E712


class TestBuildNhlFeatures:
    def test_returns_all_numeric(self):
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        for col in result.columns:
            assert result[col].dtype in [np.float64, np.int64, np.int32, np.float32,
                                         np.uint8, np.bool_], \
                f"Column {col} has non-numeric dtype {result[col].dtype}"

    def test_nhl_specific_columns_exist(self):
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        for col in ["home_back_to_back", "away_back_to_back",
                     "home_is_eastern", "away_is_eastern",
                     "cross_conference", "same_division"]:
            assert col in result.columns, f"Missing NHL column: {col}"

    def test_no_weather_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        for col in ["weather_desc", "temperature_f", "wind_mph"]:
            assert col not in result.columns

    def test_no_game_date_column(self):
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        assert "game_date" not in result.columns

    def test_no_team_string_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        assert "home_team" not in result.columns
        assert "away_team" not in result.columns

    def test_no_leakage_columns(self):
        """Raw box score columns and identifiers should be dropped."""
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        leakage_cols = [
            "home_sog", "away_sog", "home_ppg", "away_ppg",
            "home_ppo", "away_ppo", "home_pim", "away_pim",
            "home_hits", "away_hits", "home_blocks", "away_blocks",
            "home_faceoff_pct", "away_faceoff_pct",
            "external_id", "sport", "status",
        ]
        for col in leakage_cols:
            assert col not in result.columns, f"Leakage column {col} should be dropped"

    def test_preserves_upcoming_tag(self):
        """_is_upcoming tag should survive through the pipeline."""
        df = _make_synthetic_schedule(50)
        df["_is_upcoming"] = False
        df.loc[df.index[-3:], "_is_upcoming"] = True
        df.loc[df.index[-3:], "home_score"] = None
        df.loc[df.index[-3:], "away_score"] = None
        result = build_nhl_features(df)
        assert "_is_upcoming" in result.columns
        assert result["_is_upcoming"].sum() == 3

    def test_advanced_rolling_columns_in_output(self):
        """Advanced rolling columns should be present, raw box cols dropped."""
        df = _make_synthetic_schedule(50)
        result = build_nhl_features(df)
        adv_cols = [c for c in result.columns if "corsi_proxy" in c or "save_pct" in c]
        assert len(adv_cols) > 0, "No advanced rolling columns found"
        for col in ["home_sog", "away_blocks", "home_ppg", "away_ppo"]:
            assert col not in result.columns, f"Raw box col {col} should be dropped"


class TestNhlAdvancedRolling:
    def test_rolling_columns_created(self):
        """All columns (8 stats x 2 sides x 3 windows) should be created."""
        df = _make_synthetic_schedule(50)
        df = normalise_nhl_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nhl_advanced_rolling(df, windows=[3, 5, 10])
        expected_count = len(_ADVANCED_STATS) * 2 * 3  # 8 * 2 * 3 = 48
        adv_cols = [
            c for c in result.columns
            if any(c.endswith(f"_{w}g") and any(s in c for s in _ADVANCED_STATS)
                   for w in [3, 5, 10])
        ]
        assert len(adv_cols) == expected_count, (
            f"Expected {expected_count} advanced columns, got {len(adv_cols)}"
        )

    def test_first_game_has_nan_rolling(self):
        """First game for a team has no history -> NaN rolling stats."""
        df = _make_synthetic_schedule(50)
        df = normalise_nhl_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nhl_advanced_rolling(df, windows=[3, 5, 10])
        assert pd.isna(result.at[0, "home_corsi_proxy_3g"])

    def test_rolling_values_reasonable(self):
        """Save% should be in [0, 1] for reasonable data."""
        df = _make_synthetic_schedule(50)
        df = normalise_nhl_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_nhl_advanced_rolling(df, windows=[5])
        sv = result["home_save_pct_5g"].dropna()
        assert (sv >= 0).all() and (sv <= 1).all(), "Save% out of [0, 1]"

    def test_no_box_cols_returns_unchanged(self):
        """If box score columns are absent, return df unchanged."""
        df = _make_synthetic_schedule(10)
        box_cols = [c for c in df.columns if c.startswith(("home_sog", "away_sog",
                    "home_ppg", "away_ppg", "home_ppo", "away_ppo",
                    "home_pim", "away_pim", "home_hits", "away_hits",
                    "home_blocks", "away_blocks",
                    "home_faceoff", "away_faceoff"))]
        df = df.drop(columns=box_cols)
        original_cols = set(df.columns)
        result = compute_nhl_advanced_rolling(df, windows=[3, 5, 10])
        assert set(result.columns) == original_cols


class TestSplitFeaturesTargets:
    def test_split_basic(self):
        df = _make_synthetic_schedule(50)
        features = build_nhl_features(df)
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
        features = build_nhl_features(df)
        X, y = split_features_targets(features)
        assert len(X) < 50


class TestConferenceConstants:
    def test_16_teams_eastern(self):
        assert len(EASTERN_CONFERENCE) == 16

    def test_17_teams_western(self):
        # 17 because ARI (historical) and UTA (current) are both in Western
        assert len(WESTERN_CONFERENCE) == 17

    def test_no_overlap(self):
        assert EASTERN_CONFERENCE & WESTERN_CONFERENCE == set()

    def test_4_divisions(self):
        assert len(DIVISIONS) == 4
        all_teams = set()
        for teams in DIVISIONS.values():
            all_teams |= teams
        assert len(all_teams) == len(EASTERN_CONFERENCE | WESTERN_CONFERENCE)

    def test_division_sizes(self):
        assert len(DIVISIONS["Atlantic"]) == 8
        assert len(DIVISIONS["Metropolitan"]) == 8
        assert len(DIVISIONS["Pacific"]) == 8
        # Central has ARI + UTA
        assert len(DIVISIONS["Central"]) == 9
