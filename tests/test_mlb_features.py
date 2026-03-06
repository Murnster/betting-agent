"""Unit tests for MLB feature pipeline (no network calls)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from betting_agent.sports.mlb.features import (
    DIVISIONS,
    AMERICAN_LEAGUE,
    NATIONAL_LEAGUE,
    _ADVANCED_STATS,
    build_mlb_features,
    compute_mlb_advanced_rolling,
    normalise_mlb_schedules,
    split_features_targets,
)


def _make_synthetic_schedule(n_games: int = 50) -> pd.DataFrame:
    """Create a synthetic MLB schedule for testing."""
    teams = ["NYY", "BOS", "LAD", "CHC", "HOU",
             "ATL", "SF", "SD", "CLE", "SEA"]
    rows = []
    base_date = date(2024, 4, 1)

    for i in range(n_games):
        home_idx = i % len(teams)
        away_idx = (i + 3) % len(teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(teams)

        game_date = base_date + timedelta(days=i // 3)
        rows.append({
            "external_id": f"mlb{i:06d}",
            "game_date": game_date,
            "season": 2024,
            "week": None,
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "home_score": 3 + (i * 3) % 7,
            "away_score": 2 + (i * 5) % 7,
            "is_playoff": False,
            "neutral_site": False,
            "sport": "MLB",
            "status": "final",
            # Box score columns — batting
            "home_hits": 7 + i % 6,
            "home_runs_scored": 3 + (i * 3) % 7,
            "home_home_runs": i % 3,
            "home_rbi": 2 + i % 5,
            "home_strikeouts": 5 + i % 8,
            "home_walks": 2 + i % 4,
            "home_doubles": i % 3,
            "home_triples": 0,
            "home_stolen_bases": i % 2,
            # Box score columns — pitching
            "home_earned_runs": 2 + i % 4,
            "home_pitching_strikeouts": 5 + i % 6,
            "home_pitching_walks": 1 + i % 3,
            "home_hits_allowed": 5 + i % 5,
            "home_innings_pitched": 9.0,
            # Away side
            "away_hits": 6 + i % 5,
            "away_runs_scored": 2 + (i * 5) % 7,
            "away_home_runs": i % 2,
            "away_rbi": 1 + i % 4,
            "away_strikeouts": 6 + i % 7,
            "away_walks": 1 + i % 3,
            "away_doubles": i % 2,
            "away_triples": 0,
            "away_stolen_bases": i % 3,
            "away_earned_runs": 1 + i % 5,
            "away_pitching_strikeouts": 4 + i % 7,
            "away_pitching_walks": 2 + i % 3,
            "away_hits_allowed": 6 + i % 4,
            "away_innings_pitched": 9.0,
        })
    return pd.DataFrame(rows)


class TestNormaliseMlbSchedules:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"home_team": ["NYY"], "away_team": ["BOS"]})
        result = normalise_mlb_schedules(df)
        assert "is_playoff" in result.columns
        assert "neutral_site" in result.columns
        assert "weather_desc" in result.columns
        assert "temperature_f" in result.columns
        assert "wind_mph" in result.columns

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({
            "home_team": ["NYY"],
            "is_playoff": [True],
        })
        result = normalise_mlb_schedules(df)
        assert result["is_playoff"].iloc[0] == True  # noqa: E712


class TestBuildMlbFeatures:
    def test_returns_all_numeric(self):
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        for col in result.columns:
            assert result[col].dtype in [np.float64, np.int64, np.int32, np.float32,
                                         np.uint8, np.bool_], \
                f"Column {col} has non-numeric dtype {result[col].dtype}"

    def test_mlb_specific_columns_exist(self):
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        for col in ["interleague", "home_is_al", "away_is_al",
                     "same_division", "day_game"]:
            assert col in result.columns, f"Missing MLB column: {col}"

    def test_no_weather_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        for col in ["weather_desc", "temperature_f", "wind_mph"]:
            assert col not in result.columns

    def test_no_game_date_column(self):
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        assert "game_date" not in result.columns

    def test_no_team_string_columns(self):
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        assert "home_team" not in result.columns
        assert "away_team" not in result.columns

    def test_no_leakage_columns(self):
        """Raw box score columns and identifiers should be dropped."""
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        from betting_agent.sports.mlb.loader import BOX_SCORE_COLS
        leakage_cols = (
            [f"{side}_{col}" for col in BOX_SCORE_COLS for side in ("home", "away")]
            + ["external_id", "sport", "status"]
        )
        for col in leakage_cols:
            assert col not in result.columns, f"Leakage column {col} should be dropped"

    def test_preserves_upcoming_tag(self):
        """_is_upcoming tag should survive through the pipeline."""
        df = _make_synthetic_schedule(50)
        df["_is_upcoming"] = False
        df.loc[df.index[-3:], "_is_upcoming"] = True
        df.loc[df.index[-3:], "home_score"] = None
        df.loc[df.index[-3:], "away_score"] = None
        result = build_mlb_features(df)
        assert "_is_upcoming" in result.columns
        assert result["_is_upcoming"].sum() == 3

    def test_advanced_rolling_columns_in_output(self):
        """Advanced rolling columns should be present, raw box cols dropped."""
        df = _make_synthetic_schedule(50)
        result = build_mlb_features(df)
        adv_cols = [c for c in result.columns if "ops" in c or "era" in c or "whip" in c]
        assert len(adv_cols) > 0, "No advanced rolling columns found"
        for col in ["home_hits", "away_hits", "home_earned_runs", "away_innings_pitched"]:
            assert col not in result.columns, f"Raw box col {col} should be dropped"


class TestMlbAdvancedRolling:
    def test_rolling_columns_created(self):
        """All columns (9 stats x 2 sides x 3 windows) should be created."""
        df = _make_synthetic_schedule(50)
        df = normalise_mlb_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_mlb_advanced_rolling(df, windows=[3, 5, 10])
        expected_count = len(_ADVANCED_STATS) * 2 * 3  # 9 * 2 * 3 = 54
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
        df = normalise_mlb_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_mlb_advanced_rolling(df, windows=[3, 5, 10])
        assert pd.isna(result.at[0, "home_ops_3g"])

    def test_rolling_values_reasonable(self):
        """Batting avg should be in [0, 1] for reasonable data."""
        df = _make_synthetic_schedule(50)
        df = normalise_mlb_schedules(df)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        result = compute_mlb_advanced_rolling(df, windows=[5])
        ba = result["home_batting_avg_5g"].dropna()
        assert (ba >= 0).all() and (ba <= 1).all(), "Batting avg out of [0, 1]"

    def test_no_box_cols_returns_unchanged(self):
        """If box score columns are absent, return df unchanged."""
        df = _make_synthetic_schedule(10)
        box_cols = [c for c in df.columns if c.startswith(("home_hits", "away_hits",
                    "home_runs_scored", "away_runs_scored",
                    "home_home_runs", "away_home_runs",
                    "home_earned", "away_earned",
                    "home_innings", "away_innings",
                    "home_pitching", "away_pitching",
                    "home_rbi", "away_rbi",
                    "home_strikeouts", "away_strikeouts",
                    "home_walks", "away_walks",
                    "home_doubles", "away_doubles",
                    "home_triples", "away_triples",
                    "home_stolen", "away_stolen",
                    "home_hits_allowed", "away_hits_allowed"))]
        df = df.drop(columns=box_cols)
        original_cols = set(df.columns)
        result = compute_mlb_advanced_rolling(df, windows=[3, 5, 10])
        assert set(result.columns) == original_cols


class TestSplitFeaturesTargets:
    def test_split_basic(self):
        df = _make_synthetic_schedule(50)
        features = build_mlb_features(df)
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
        features = build_mlb_features(df)
        X, y = split_features_targets(features)
        assert len(X) < 50


class TestLeagueConstants:
    def test_15_teams_al(self):
        assert len(AMERICAN_LEAGUE) == 15

    def test_15_teams_nl(self):
        assert len(NATIONAL_LEAGUE) == 15

    def test_no_overlap(self):
        assert AMERICAN_LEAGUE & NATIONAL_LEAGUE == set()

    def test_6_divisions(self):
        assert len(DIVISIONS) == 6
        all_teams = set()
        for teams in DIVISIONS.values():
            all_teams |= teams
        assert len(all_teams) == 30

    def test_division_sizes(self):
        for div_name, teams in DIVISIONS.items():
            assert len(teams) == 5, f"{div_name} should have 5 teams, got {len(teams)}"
