"""Unit tests for sport registry."""

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
