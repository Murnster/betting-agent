"""Unit tests for sport registry."""

from datetime import date

import pytest

from betting_agent.sports.registry import available_sports, get_sport_config


class TestGetSportConfig:
    def test_nfl_config(self):
        config = get_sport_config("NFL")
        assert config.sport_key == "americanfootball_nfl"
        assert config.hist_avg_total == 45.0
        assert config.total_stdev == 7.0
        assert callable(config.build_features)
        assert callable(config.split_features_targets)

    def test_nba_config(self):
        config = get_sport_config("NBA")
        assert config.sport_key == "basketball_nba"
        assert config.hist_avg_total == 220.0
        assert config.total_stdev == 15.0
        assert callable(config.build_features)
        assert callable(config.split_features_targets)

    def test_case_insensitive(self):
        config1 = get_sport_config("nfl")
        config2 = get_sport_config("NFL")
        assert config1.sport_key == config2.sport_key

    def test_unknown_sport_raises(self):
        with pytest.raises(KeyError, match="Unknown sport"):
            get_sport_config("CRICKET")


class TestAvailableSports:
    def test_returns_list(self):
        sports = available_sports()
        assert isinstance(sports, list)

    def test_contains_nfl_and_nba(self):
        sports = available_sports()
        assert "NBA" in sports
        assert "NFL" in sports

    def test_sorted(self):
        sports = available_sports()
        assert sports == sorted(sports)

    def test_frozen_sports_hidden_by_default(self):
        sports = available_sports()
        assert "NHL" not in sports
        assert "MLB" not in sports

    def test_frozen_sports_still_reachable(self):
        assert "NHL" in available_sports(include_frozen=True)
        assert "MLB" in available_sports(include_frozen=True)
        assert get_sport_config("NHL").sport_key == "icehockey_nhl"


class TestSeasonForDate:
    def test_nfl_january_belongs_to_previous_season(self):
        config = get_sport_config("NFL")
        assert config.season_for_date(date(2027, 1, 10)) == 2026
        assert config.season_for_date(date(2026, 9, 10)) == 2026
        assert config.season_for_date(date(2026, 8, 1)) == 2026

    def test_nba_spring_belongs_to_previous_season(self):
        config = get_sport_config("NBA")
        assert config.season_for_date(date(2027, 4, 15)) == 2026
        assert config.season_for_date(date(2026, 10, 25)) == 2026

    def test_mlb_uses_calendar_year(self):
        config = get_sport_config("MLB")
        assert config.season_for_date(date(2026, 4, 1)) == 2026
        assert config.season_for_date(date(2026, 9, 30)) == 2026
