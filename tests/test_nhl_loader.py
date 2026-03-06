"""Unit tests for NHL loader helpers (no network calls)."""

from datetime import date

from betting_agent.sports.nhl.loader import (
    NHLLoader,
    int_to_nhl_season,
    _parse_schedule_games,
    _extract_team_boxscore_stats,
    _parse_team_game_stats,
    NHL_ABBREV_TO_FULL,
    NHL_FULL_TO_ABBREV,
    NHL_TEAM_ABBREVS,
)


class TestIntToNhlSeason:
    def test_standard_year(self):
        assert int_to_nhl_season(2023) == "20232024"

    def test_next_year(self):
        assert int_to_nhl_season(2024) == "20242025"

    def test_early_year(self):
        assert int_to_nhl_season(2000) == "20002001"


class TestSportKey:
    def test_sport_key(self):
        loader = NHLLoader()
        assert loader.sport_key() == "icehockey_nhl"


class TestParseScheduleGames:
    def _make_schedule_data(self, game_state="OFF", game_type=2):
        return {
            "games": [
                {
                    "id": 2023020280,
                    "gameType": game_type,
                    "gameState": game_state,
                    "gameDate": "2024-01-15",
                    "homeTeam": {
                        "abbrev": "TOR",
                        "score": 4,
                    },
                    "awayTeam": {
                        "abbrev": "MTL",
                        "score": 2,
                    },
                },
                {
                    "id": 2023020281,
                    "gameType": game_type,
                    "gameState": game_state,
                    "gameDate": "2024-01-15",
                    "homeTeam": {
                        "abbrev": "BOS",
                        "score": 3,
                    },
                    "awayTeam": {
                        "abbrev": "NYR",
                        "score": 1,
                    },
                },
            ]
        }

    def test_parse_produces_correct_rows(self):
        data = self._make_schedule_data()
        rows = _parse_schedule_games(data, season=2023)
        assert len(rows) == 2

    def test_parse_home_away_correct(self):
        data = self._make_schedule_data()
        rows = _parse_schedule_games(data, season=2023)
        game1 = rows[0]
        assert game1["home_team"] == "TOR"
        assert game1["away_team"] == "MTL"
        assert game1["home_score"] == 4
        assert game1["away_score"] == 2

    def test_parse_metadata(self):
        data = self._make_schedule_data()
        rows = _parse_schedule_games(data, season=2023)
        game1 = rows[0]
        assert game1["season"] == 2023
        assert game1["is_playoff"] is False
        assert game1["sport"] == "NHL"
        assert game1["status"] == "final"
        assert game1["week"] is None
        assert game1["neutral_site"] is False

    def test_parse_playoff_game(self):
        data = self._make_schedule_data(game_type=3)
        rows = _parse_schedule_games(data, season=2023)
        assert rows[0]["is_playoff"] is True

    def test_parse_skips_future_games(self):
        data = self._make_schedule_data(game_state="FUT")
        rows = _parse_schedule_games(data, season=2023)
        assert len(rows) == 0

    def test_parse_skips_preseason(self):
        data = self._make_schedule_data(game_type=1)
        rows = _parse_schedule_games(data, season=2023)
        assert len(rows) == 0

    def test_parse_empty_schedule(self):
        rows = _parse_schedule_games({"games": []}, season=2023)
        assert rows == []

    def test_parse_game_date(self):
        data = self._make_schedule_data()
        rows = _parse_schedule_games(data, season=2023)
        assert rows[0]["game_date"] == date(2024, 1, 15)


class TestExtractBoxscoreStats:
    def test_basic_extraction(self):
        team_data = {
            "sog": 35,
            "powerPlayConversion": "2/4",
            "pim": 8,
            "hits": 25,
            "blocks": 15,
            "faceoffWinningPctg": 0.52,
        }
        stats = _extract_team_boxscore_stats(team_data)
        assert stats["sog"] == 35
        assert stats["ppg"] == 2
        assert stats["ppo"] == 4
        assert stats["pim"] == 8
        assert stats["hits"] == 25
        assert stats["blocks"] == 15
        assert stats["faceoff_pct"] == 0.52

    def test_missing_stats_returns_none(self):
        stats = _extract_team_boxscore_stats({})
        assert stats["sog"] is None
        # ppg/ppo come from powerPlayConversion parsing; with empty string they get default
        assert stats["hits"] is None
        assert stats["blocks"] is None

    def test_bad_pp_format(self):
        stats = _extract_team_boxscore_stats({"powerPlayConversion": "invalid"})
        assert stats["ppg"] is None
        assert stats["ppo"] is None


class TestTeamMappings:
    def test_32_current_teams_in_abbrevs(self):
        assert len(NHL_TEAM_ABBREVS) == 32

    def test_abbrev_to_full_includes_historical_and_current(self):
        # ARI (historical) and UTA (current) both present
        assert "ARI" in NHL_ABBREV_TO_FULL
        assert "UTA" in NHL_ABBREV_TO_FULL
        assert NHL_ABBREV_TO_FULL["ARI"] == "Arizona Coyotes"
        assert NHL_ABBREV_TO_FULL["UTA"] == "Utah Hockey Club"

    def test_reverse_mapping_consistent(self):
        for abbrev, full in NHL_ABBREV_TO_FULL.items():
            assert NHL_FULL_TO_ABBREV[full] == abbrev

    def test_known_teams(self):
        assert NHL_ABBREV_TO_FULL["TOR"] == "Toronto Maple Leafs"
        assert NHL_ABBREV_TO_FULL["BOS"] == "Boston Bruins"
        assert NHL_FULL_TO_ABBREV["Pittsburgh Penguins"] == "PIT"
        assert NHL_FULL_TO_ABBREV["Vegas Golden Knights"] == "VGK"

    def test_all_current_abbrevs_in_mapping(self):
        for abbrev in NHL_TEAM_ABBREVS:
            assert abbrev in NHL_ABBREV_TO_FULL, f"Missing mapping for {abbrev}"


class TestParseTeamGameStats:
    SAMPLE_STATS = [
        {"category": "sog", "awayValue": 23, "homeValue": 31},
        {"category": "faceoffWinningPctg", "awayValue": 0.42, "homeValue": 0.57},
        {"category": "powerPlay", "awayValue": "0/2", "homeValue": "2/4"},
        {"category": "pim", "awayValue": 8, "homeValue": 4},
        {"category": "hits", "awayValue": 34, "homeValue": 29},
        {"category": "blockedShots", "awayValue": 18, "homeValue": 15},
        {"category": "giveaways", "awayValue": 5, "homeValue": 7},
        {"category": "takeaways", "awayValue": 3, "homeValue": 4},
    ]

    def test_home_stats(self):
        stats = _parse_team_game_stats(self.SAMPLE_STATS, "home")
        assert stats["sog"] == 31
        assert stats["ppg"] == 2
        assert stats["ppo"] == 4
        assert stats["pim"] == 4
        assert stats["hits"] == 29
        assert stats["blocks"] == 15
        assert stats["faceoff_pct"] == 0.57

    def test_away_stats(self):
        stats = _parse_team_game_stats(self.SAMPLE_STATS, "away")
        assert stats["sog"] == 23
        assert stats["ppg"] == 0
        assert stats["ppo"] == 2
        assert stats["pim"] == 8
        assert stats["hits"] == 34
        assert stats["blocks"] == 18
        assert stats["faceoff_pct"] == 0.42

    def test_empty_stats(self):
        stats = _parse_team_game_stats([], "home")
        assert all(v is None for v in stats.values())

    def test_missing_category_ignored(self):
        partial = [{"category": "sog", "awayValue": 10, "homeValue": 20}]
        stats = _parse_team_game_stats(partial, "home")
        assert stats["sog"] == 20
        assert stats["hits"] is None

    def test_bad_powerplay_format(self):
        bad = [{"category": "powerPlay", "awayValue": "invalid", "homeValue": "bad"}]
        stats = _parse_team_game_stats(bad, "home")
        assert stats["ppg"] is None
        assert stats["ppo"] is None
