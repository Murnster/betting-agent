"""Unit tests for the NFL feature pipeline (no network calls)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from betting_agent.sports.nfl.features import (
    MARKET_COLS,
    attach_schedule_context,
    build_nfl_features,
    normalise_raw_schedules,
    split_features_targets,
)
from betting_agent.sports.nfl.loader import (
    NFL_ABBREV_TO_FULL,
    NFL_FULL_TO_ABBREV,
    nfl_team_to_abbrev,
)

TEAMS = ["KC", "BUF", "PHI", "SF", "DAL", "GB", "BAL", "CIN"]


def _make_schedule(n_games: int = 64, season: int = 2024) -> pd.DataFrame:
    """Synthetic frame shaped like a raw nflreadpy.load_schedules() result."""
    rows = []
    base = date(season, 9, 8)
    for i in range(n_games):
        home = TEAMS[i % len(TEAMS)]
        away = TEAMS[(i + 3) % len(TEAMS)]
        rows.append({
            "game_id": f"{season}_{i // 8 + 1:02d}_{away}_{home}",
            "season": season,
            "game_type": "REG",
            "week": i // 8 + 1,
            "gameday": (base + timedelta(days=7 * (i // 8))).isoformat(),
            "weekday": "Sunday",
            "gametime": "13:00",
            "home_team": home,
            "away_team": away,
            "home_score": 20 + i % 14,
            "away_score": 17 + i % 11,
            "location": "Home",
            "result": (20 + i % 14) - (17 + i % 11),
            "total": (20 + i % 14) + (17 + i % 11),
            "overtime": 0,
            "away_rest": 7,
            "home_rest": 7,
            "away_moneyline": 130.0,
            "home_moneyline": -150.0,
            "spread_line": 3.0,
            "away_spread_odds": -110.0,
            "home_spread_odds": -110.0,
            "total_line": 44.5,
            "under_odds": -110.0,
            "over_odds": -110.0,
            "div_game": i % 4 == 0,
            "roof": "outdoors" if i % 3 else "dome",
            "surface": "grass" if i % 2 else "fieldturf",
            "temp": 60.0 if i % 3 else None,
            "wind": 8.0 if i % 3 else None,
            "stadium": "Test Field",
            "referee": "Someone",
        })
    return pd.DataFrame(rows)


class TestNormaliseRawSchedules:
    def test_renames_nflreadpy_columns(self):
        result = normalise_raw_schedules(_make_schedule(8))
        assert "game_date" in result.columns
        assert "gameday" not in result.columns
        assert "is_playoff" in result.columns
        assert "neutral_site" in result.columns

    def test_maps_real_weather_columns(self):
        result = normalise_raw_schedules(_make_schedule(8))
        assert "temperature_f" in result.columns
        assert "wind_mph" in result.columns
        assert "temp" not in result.columns
        assert result["temperature_f"].notna().any(), "real temps must survive"

    def test_adds_weather_columns_when_source_lacks_them(self):
        df = pd.DataFrame({"home_team": ["KC"], "away_team": ["BUF"]})
        result = normalise_raw_schedules(df)
        assert result["temperature_f"].isna().all()
        assert result["wind_mph"].isna().all()

    def test_coerces_mixed_date_types(self):
        """History carries real dates, odds-derived rows carry strings."""
        df = pd.DataFrame({
            "game_date": [pd.Timestamp("2024-09-08"), "2024-09-15"],
            "home_team": ["KC", "BUF"],
            "away_team": ["BUF", "KC"],
        })
        result = normalise_raw_schedules(df)
        assert pd.api.types.is_datetime64_any_dtype(result["game_date"])

    def test_playoff_flag_from_game_type(self):
        df = _make_schedule(8)
        df.loc[0, "game_type"] = "SB"
        result = normalise_raw_schedules(df)
        assert bool(result.loc[0, "is_playoff"]) is True
        assert bool(result.loc[1, "is_playoff"]) is False


class TestMarketColumnsExcluded:
    def test_no_market_columns_in_features(self):
        """Closing lines are the market's opinion — training on them teaches
        the model to echo the line, and they are absent at pick time."""
        features = build_nfl_features(_make_schedule())
        leaked = [c for c in MARKET_COLS if c in features.columns]
        assert leaked == [], f"market columns leaked into features: {leaked}"

    def test_no_direct_outcome_columns(self):
        features = build_nfl_features(_make_schedule())
        for col in ("result", "total", "overtime"):
            assert col not in features.columns

    def test_venue_context_kept(self):
        """roof/surface/div_game are published pre-kickoff, so they stay."""
        features = build_nfl_features(_make_schedule())
        assert "div_game" in features.columns
        assert any(c.startswith("roof_") for c in features.columns)
        assert any(c.startswith("surface_") for c in features.columns)

    def test_all_numeric(self):
        features = build_nfl_features(_make_schedule())
        non_numeric = [
            c for c in features.columns
            if not pd.api.types.is_numeric_dtype(features[c])
        ]
        assert non_numeric == []


class TestAttachScheduleContext:
    def test_fills_venue_and_scheduling_columns(self):
        schedule = _make_schedule()
        upcoming = pd.DataFrame([{
            "game_id": "evt1",
            "game_date": "2024-09-08",
            "home_team": "KC",
            "away_team": "SF",
            "week": None,
            "is_playoff": False,
        }])
        result = attach_schedule_context(upcoming, schedule)
        assert result.loc[0, "roof"] in ("outdoors", "dome")
        assert result.loc[0, "surface"] in ("grass", "fieldturf")
        assert not pd.isna(result.loc[0, "div_game"])
        assert result.loc[0, "week"] == 1

    def test_picks_nearest_kickoff_for_repeat_fixtures(self):
        schedule = _make_schedule()
        # KC hosts SF in week 1; add a later rematch.
        rematch = schedule.iloc[0].copy()
        rematch["week"] = 14
        rematch["gameday"] = "2024-12-08"
        rematch["roof"] = "dome"
        schedule = pd.concat([schedule, rematch.to_frame().T], ignore_index=True)

        upcoming = pd.DataFrame([{
            "game_id": "evt1",
            "game_date": "2024-12-07",
            "home_team": schedule.iloc[0]["home_team"],
            "away_team": schedule.iloc[0]["away_team"],
        }])
        result = attach_schedule_context(upcoming, schedule)
        assert result.loc[0, "week"] == 14

    def test_unmatched_rows_pass_through(self):
        upcoming = pd.DataFrame([{
            "game_id": "evt1",
            "game_date": "2024-09-08",
            "home_team": "XXX",
            "away_team": "YYY",
            "week": None,
        }])
        result = attach_schedule_context(upcoming, _make_schedule())
        assert len(result) == 1
        assert pd.isna(result.loc[0, "week"])

    def test_empty_inputs_are_safe(self):
        empty = pd.DataFrame()
        assert attach_schedule_context(empty, _make_schedule()).empty
        upcoming = pd.DataFrame([{"home_team": "KC", "away_team": "SF"}])
        assert len(attach_schedule_context(upcoming, empty)) == 1


class TestServeTimeParity:
    """The regression guard for the bug that made live NFL picks meaningless:
    a third of the trained features were absent at pick time and silently
    zero-filled."""

    def _serve_frame(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Replicate what scripts/picks.py assembles: an odds-derived upcoming
        row enriched from the schedule, concatenated with trimmed history."""
        upcoming = pd.DataFrame([{
            "game_id": "evt1",
            "game_date": "2024-12-29",
            "season": 2024,
            "week": None,
            "home_team": nfl_team_to_abbrev("Kansas City Chiefs"),
            "away_team": nfl_team_to_abbrev("Buffalo Bills"),
            "home_team_odds": "Kansas City Chiefs",
            "away_team_odds": "Buffalo Bills",
            "home_score": None,
            "away_score": None,
            "is_playoff": False,
            "neutral_site": False,
            "weather_desc": None,
            "temperature_f": None,
            "wind_mph": None,
            "external_id": "evt1",
        }])
        upcoming = attach_schedule_context(upcoming, schedule)
        upcoming["_is_upcoming"] = True

        history = schedule.rename(columns={"gameday": "game_date"}).copy()
        history["external_id"] = history["game_id"]
        keep = [c for c in list(upcoming.columns) + ["game_type", "location"]
                if c in history.columns]
        history = history[list(dict.fromkeys(keep))].copy()
        history["_is_upcoming"] = False
        return pd.concat([history, upcoming], ignore_index=True)

    def test_no_features_missing_at_serve_time(self):
        schedule = _make_schedule()
        train_features = build_nfl_features(schedule)
        trained_cols = [
            c for c in train_features.columns
            if c not in ("home_team_wins", "home_score", "away_score")
        ]

        serve_features = build_nfl_features(self._serve_frame(schedule))
        missing = [c for c in trained_cols if c not in serve_features.columns]
        assert missing == [], (
            f"{len(missing)} trained features would be zero-filled at pick "
            f"time: {missing}"
        )

    def test_upcoming_row_gets_real_team_history(self):
        """Without the name bridge, upcoming rows match no history and score
        on base Elo with empty rolling averages."""
        schedule = _make_schedule()
        serve_features = build_nfl_features(self._serve_frame(schedule))
        row = serve_features[serve_features["_is_upcoming"].astype(bool)].iloc[0]
        assert row["home_elo"] != 1000.0
        assert row["away_elo"] != 1000.0
        assert not pd.isna(row["home_rest_days"])


class TestSplitFeaturesTargets:
    def test_drops_unplayed_games(self):
        schedule = _make_schedule()
        schedule.loc[schedule.index[-4:], ["home_score", "away_score"]] = np.nan
        features = build_nfl_features(schedule)
        X, y = split_features_targets(features)
        # Unplayed games and ties (no binary target) both drop out.
        playable = features["home_score"].notna() & (
            features["home_score"] != features["away_score"]
        )
        assert len(X) == int(playable.sum())
        assert list(y.columns) == ["home_team_wins", "home_score", "away_score"]
        assert "home_team_wins" not in X.columns
        assert y["home_team_wins"].notna().all()


class TestTeamNameBridge:
    def test_covers_all_32_clubs(self):
        assert len(NFL_FULL_TO_ABBREV) == 32
        assert len(NFL_ABBREV_TO_FULL) == 32

    @pytest.mark.parametrize("full,abbrev", [
        ("Kansas City Chiefs", "KC"),
        ("Los Angeles Rams", "LA"),
        ("Los Angeles Chargers", "LAC"),
        ("San Francisco 49ers", "SF"),
        ("Washington Commanders", "WAS"),
    ])
    def test_maps_odds_api_names(self, full, abbrev):
        assert nfl_team_to_abbrev(full) == abbrev

    def test_abbreviations_pass_through(self):
        assert nfl_team_to_abbrev("KC") == "KC"

    def test_legacy_names_map_to_era_abbreviation(self):
        assert nfl_team_to_abbrev("Oakland Raiders") == "OAK"
        assert nfl_team_to_abbrev("San Diego Chargers") == "SD"

    def test_unknown_name_returned_unchanged(self):
        assert nfl_team_to_abbrev("Toronto Argonauts") == "Toronto Argonauts"

    def test_none_is_safe(self):
        assert nfl_team_to_abbrev(None) is None
