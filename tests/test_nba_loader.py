"""Unit tests for NBA loader helpers (no network calls)."""

from datetime import date

import pandas as pd
import pytest

from betting_agent.sports.nba.loader import (
    NBALoader,
    int_to_nba_season,
    _parse_nba_date,
    _pivot_game_log,
    NBA_ABBREV_TO_FULL,
    NBA_FULL_TO_ABBREV,
)


class TestIntToNbaSeason:
    def test_standard_year(self):
        assert int_to_nba_season(2024) == "2024-25"

    def test_century_boundary(self):
        assert int_to_nba_season(2099) == "2099-00"

    def test_early_year(self):
        assert int_to_nba_season(2000) == "2000-01"


class TestParseNbaDate:
    def test_standard_date(self):
        assert _parse_nba_date("OCT 28, 2024") == date(2024, 10, 28)

    def test_single_digit_day(self):
        assert _parse_nba_date("JAN 5, 2025") == date(2025, 1, 5)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            _parse_nba_date("INVALID DATE")


class TestSportKey:
    def test_sport_key(self):
        loader = NBALoader()
        assert loader.sport_key() == "basketball_nba"


class TestPivotGameLog:
    def _make_game_log(self):
        """Create a mock LeagueGameLog DataFrame with 2 rows per game."""
        return pd.DataFrame([
            {
                "GAME_ID": "0022400100",
                "TEAM_ABBREVIATION": "BOS",
                "MATCHUP": "BOS vs. MIL",
                "GAME_DATE": "OCT 28, 2024",
                "PTS": 110,
            },
            {
                "GAME_ID": "0022400100",
                "TEAM_ABBREVIATION": "MIL",
                "MATCHUP": "MIL @ BOS",
                "GAME_DATE": "OCT 28, 2024",
                "PTS": 105,
            },
            {
                "GAME_ID": "0022400101",
                "TEAM_ABBREVIATION": "LAL",
                "MATCHUP": "LAL vs. GSW",
                "GAME_DATE": "OCT 29, 2024",
                "PTS": 120,
            },
            {
                "GAME_ID": "0022400101",
                "TEAM_ABBREVIATION": "GSW",
                "MATCHUP": "GSW @ LAL",
                "GAME_DATE": "OCT 29, 2024",
                "PTS": 115,
            },
        ])

    def test_pivot_produces_one_row_per_game(self):
        df = self._make_game_log()
        rows = _pivot_game_log(df, season=2024, is_playoff=False)
        assert len(rows) == 2

    def test_pivot_home_away_correct(self):
        df = self._make_game_log()
        rows = _pivot_game_log(df, season=2024, is_playoff=False)
        game1 = rows[0]
        assert game1["home_team"] == "BOS"
        assert game1["away_team"] == "MIL"
        assert game1["home_score"] == 110
        assert game1["away_score"] == 105

    def test_pivot_metadata(self):
        df = self._make_game_log()
        rows = _pivot_game_log(df, season=2024, is_playoff=True)
        game1 = rows[0]
        assert game1["season"] == 2024
        assert game1["is_playoff"] is True
        assert game1["sport"] == "NBA"
        assert game1["status"] == "final"
        assert game1["week"] is None
        assert game1["neutral_site"] is False

    def test_pivot_empty_dataframe(self):
        df = pd.DataFrame()
        rows = _pivot_game_log(df, season=2024, is_playoff=False)
        assert rows == []

    def test_pivot_single_row_skipped(self):
        """A game with only 1 row (incomplete data) should be skipped."""
        df = pd.DataFrame([{
            "GAME_ID": "0022400100",
            "TEAM_ABBREVIATION": "BOS",
            "MATCHUP": "BOS vs. MIL",
            "GAME_DATE": "OCT 28, 2024",
            "PTS": 110,
        }])
        rows = _pivot_game_log(df, season=2024, is_playoff=False)
        assert len(rows) == 0


class TestTeamMappings:
    def test_all_30_teams_in_abbrev_to_full(self):
        assert len(NBA_ABBREV_TO_FULL) == 30

    def test_reverse_mapping_consistent(self):
        assert len(NBA_FULL_TO_ABBREV) == 30
        for abbrev, full in NBA_ABBREV_TO_FULL.items():
            assert NBA_FULL_TO_ABBREV[full] == abbrev

    def test_known_teams(self):
        assert NBA_ABBREV_TO_FULL["BOS"] == "Boston Celtics"
        assert NBA_ABBREV_TO_FULL["LAL"] == "Los Angeles Lakers"
        assert NBA_FULL_TO_ABBREV["Golden State Warriors"] == "GSW"
