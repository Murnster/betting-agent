"""Tests for canonical team naming across the Odds API / loader divide."""

import pytest

from betting_agent.sports.teams import canonical_team, display_team, same_team


class TestCanonicalTeam:
    @pytest.mark.parametrize("sport,name,expected", [
        ("NFL", "Kansas City Chiefs", "KC"),
        ("NFL", "Los Angeles Rams", "LA"),
        ("NBA", "Boston Celtics", "BOS"),
        ("NHL", "Toronto Maple Leafs", "TOR"),
        ("MLB", "New York Yankees", "NYY"),
    ])
    def test_maps_full_names_to_abbreviations(self, sport, name, expected):
        assert canonical_team(sport, name) == expected

    def test_abbreviations_are_already_canonical(self):
        assert canonical_team("NFL", "KC") == "KC"
        assert canonical_team("NBA", "BOS") == "BOS"

    def test_case_of_sport_does_not_matter(self):
        assert canonical_team("nfl", "Kansas City Chiefs") == "KC"

    def test_whitespace_is_trimmed(self):
        assert canonical_team("NFL", "  Kansas City Chiefs ") == "KC"

    def test_unknown_names_pass_through(self):
        assert canonical_team("NFL", "Toronto Argonauts") == "Toronto Argonauts"

    def test_unknown_sport_passes_through(self):
        assert canonical_team("XFL", "Some Team") == "Some Team"

    def test_none_is_safe(self):
        assert canonical_team("NFL", None) is None


class TestDisplayTeam:
    def test_expands_abbreviation(self):
        assert display_team("NFL", "KC") == "Kansas City Chiefs"
        assert display_team("NBA", "BOS") == "Boston Celtics"

    def test_full_name_unchanged(self):
        assert display_team("NFL", "Kansas City Chiefs") == "Kansas City Chiefs"

    def test_round_trips_with_canonical(self):
        for sport, full in (
            ("NFL", "Philadelphia Eagles"),
            ("NBA", "Golden State Warriors"),
            ("NHL", "Boston Bruins"),
        ):
            assert display_team(sport, canonical_team(sport, full)) == full


class TestSameTeam:
    def test_matches_across_naming_conventions(self):
        """The bug this exists to prevent: a pick stored as an Odds API club
        name graded against a loader-seeded abbreviation."""
        assert same_team("NFL", "Kansas City Chiefs", "KC")
        assert same_team("NBA", "BOS", "Boston Celtics")

    def test_rejects_different_teams(self):
        assert not same_team("NFL", "Kansas City Chiefs", "BUF")
        assert not same_team("NBA", "Boston Celtics", "MIA")

    def test_none_never_matches(self):
        assert not same_team("NFL", None, "KC")
        assert not same_team("NFL", "KC", None)
