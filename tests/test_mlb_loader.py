"""Unit tests for MLB loader helpers (no network calls)."""

from datetime import date
from unittest.mock import MagicMock

from betting_agent.sports.mlb.loader import (
    MLBLoader,
    _parse_schedule_games,
    _extract_team_boxscore_stats,
    MLB_ABBREV_TO_FULL,
    MLB_FULL_TO_ABBREV,
    MLB_TEAM_ABBREVS,
)


class TestSportKey:
    def test_sport_key(self):
        loader = MLBLoader()
        assert loader.sport_key() == "baseball_mlb"


def _make_mock_game(game_pk=746419, game_type="R", status="Final",
                     home_name="Detroit Tigers", away_name="Baltimore Orioles",
                     home_score=4, away_score=2, official_date="2024-09-15"):
    """Create a mock game object mimicking mlbstatsapi Schedule game."""
    game = MagicMock()
    game.game_pk = game_pk
    game.game_type = game_type
    game.official_date = official_date

    game.status = MagicMock()
    game.status.detailed_state = status

    game.teams = MagicMock()
    game.teams.home = MagicMock()
    game.teams.home.team = MagicMock()
    game.teams.home.team.name = home_name
    game.teams.home.score = home_score
    game.teams.away = MagicMock()
    game.teams.away.team = MagicMock()
    game.teams.away.team.name = away_name
    game.teams.away.score = away_score

    return game


def _make_mock_schedule(games):
    """Create a mock Schedule with a single date containing given games."""
    schedule = MagicMock()
    sched_date = MagicMock()
    sched_date.date = "2024-09-15"
    sched_date.games = games
    schedule.dates = [sched_date]
    return schedule


class TestParseScheduleGames:
    def test_parse_produces_correct_rows(self):
        games = [_make_mock_game(game_pk=1), _make_mock_game(game_pk=2)]
        schedule = _make_mock_schedule(games)
        rows = _parse_schedule_games(schedule, season=2024)
        assert len(rows) == 2

    def test_parse_home_away_correct(self):
        schedule = _make_mock_schedule([_make_mock_game()])
        rows = _parse_schedule_games(schedule, season=2024)
        game1 = rows[0]
        assert game1["home_team"] == "DET"
        assert game1["away_team"] == "BAL"
        assert game1["home_score"] == 4
        assert game1["away_score"] == 2

    def test_parse_metadata(self):
        schedule = _make_mock_schedule([_make_mock_game()])
        rows = _parse_schedule_games(schedule, season=2024)
        game1 = rows[0]
        assert game1["season"] == 2024
        assert game1["is_playoff"] is False
        assert game1["sport"] == "MLB"
        assert game1["status"] == "final"
        assert game1["week"] is None
        assert game1["neutral_site"] is False

    def test_parse_playoff_game(self):
        schedule = _make_mock_schedule([_make_mock_game(game_type="W")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert rows[0]["is_playoff"] is True

    def test_parse_skips_non_final_games(self):
        schedule = _make_mock_schedule([_make_mock_game(status="Scheduled")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert len(rows) == 0

    def test_parse_skips_spring_training(self):
        schedule = _make_mock_schedule([_make_mock_game(game_type="S")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert len(rows) == 0

    def test_parse_skips_exhibition(self):
        schedule = _make_mock_schedule([_make_mock_game(game_type="E")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert len(rows) == 0

    def test_parse_empty_schedule(self):
        schedule = MagicMock()
        schedule.dates = []
        rows = _parse_schedule_games(schedule, season=2024)
        assert rows == []

    def test_parse_game_date(self):
        schedule = _make_mock_schedule([_make_mock_game()])
        rows = _parse_schedule_games(schedule, season=2024)
        assert rows[0]["game_date"] == date(2024, 9, 15)

    def test_wild_card_is_playoff(self):
        schedule = _make_mock_schedule([_make_mock_game(game_type="F")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert rows[0]["is_playoff"] is True

    def test_division_series_is_playoff(self):
        schedule = _make_mock_schedule([_make_mock_game(game_type="D")])
        rows = _parse_schedule_games(schedule, season=2024)
        assert rows[0]["is_playoff"] is True


class TestExtractBoxscoreStats:
    def test_basic_extraction(self):
        team_stats = {
            "batting": {
                "hits": 8,
                "runs": 5,
                "homeRuns": 2,
                "rbi": 5,
                "strikeOuts": 9,
                "baseOnBalls": 3,
                "doubles": 1,
                "triples": 0,
                "stolenBases": 1,
            },
            "pitching": {
                "earnedRuns": 3,
                "strikeOuts": 7,
                "baseOnBalls": 2,
                "hits": 6,
                "inningsPitched": "9.0",
            },
        }
        stats = _extract_team_boxscore_stats(team_stats)
        assert stats["hits"] == 8
        assert stats["runs_scored"] == 5
        assert stats["home_runs"] == 2
        assert stats["rbi"] == 5
        assert stats["strikeouts"] == 9
        assert stats["walks"] == 3
        assert stats["doubles"] == 1
        assert stats["triples"] == 0
        assert stats["stolen_bases"] == 1
        assert stats["earned_runs"] == 3
        assert stats["pitching_strikeouts"] == 7
        assert stats["pitching_walks"] == 2
        assert stats["hits_allowed"] == 6
        assert stats["innings_pitched"] == 9.0

    def test_missing_stats_returns_none(self):
        stats = _extract_team_boxscore_stats({"batting": {}, "pitching": {}})
        assert stats["hits"] is None
        assert stats["earned_runs"] is None
        assert stats["innings_pitched"] is None

    def test_empty_dict(self):
        stats = _extract_team_boxscore_stats({})
        assert stats["hits"] is None
        assert stats["innings_pitched"] is None


class TestTeamMappings:
    def test_30_teams_in_abbrevs(self):
        assert len(MLB_TEAM_ABBREVS) == 30

    def test_abbrev_to_full_30_teams(self):
        assert len(MLB_ABBREV_TO_FULL) == 30

    def test_reverse_mapping_consistent(self):
        for abbrev, full in MLB_ABBREV_TO_FULL.items():
            assert MLB_FULL_TO_ABBREV[full] == abbrev

    def test_known_teams(self):
        assert MLB_ABBREV_TO_FULL["NYY"] == "New York Yankees"
        assert MLB_ABBREV_TO_FULL["LAD"] == "Los Angeles Dodgers"
        assert MLB_FULL_TO_ABBREV["Boston Red Sox"] == "BOS"
        assert MLB_FULL_TO_ABBREV["Chicago Cubs"] == "CHC"

    def test_oakland_historical_mapping(self):
        assert MLB_FULL_TO_ABBREV["Oakland Athletics"] == "ATH"
        assert MLB_FULL_TO_ABBREV["Athletics"] == "ATH"

    def test_all_abbrevs_in_mapping(self):
        for abbrev in MLB_TEAM_ABBREVS:
            assert abbrev in MLB_ABBREV_TO_FULL, f"Missing mapping for {abbrev}"
