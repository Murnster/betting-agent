"""Tests for odds selection: a line and the price it was offered at must come
from the same bookmaker."""

import pytest

from betting_agent.api.odds import OddsAPIClient
from betting_agent.intelligence.ev import calculate_edge_fair


def _book(key, title, *, ml=None, spread=None, total=None):
    markets = []
    if ml:
        markets.append({"key": "h2h", "outcomes": [
            {"name": "Kansas City Chiefs", "price": ml[0]},
            {"name": "Buffalo Bills", "price": ml[1]},
        ]})
    if spread:
        line, home_price, away_price = spread
        markets.append({"key": "spreads", "outcomes": [
            {"name": "Kansas City Chiefs", "price": home_price, "point": line},
            {"name": "Buffalo Bills", "price": away_price, "point": -line},
        ]})
    if total:
        line, over_price, under_price = total
        markets.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": over_price, "point": line},
            {"name": "Under", "price": under_price, "point": line},
        ]})
    return {"key": key, "title": title, "markets": markets}


def _event(*books):
    return [{
        "id": "evt1",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "commence_time": "2026-09-13T17:00:00Z",
        "bookmakers": list(books),
    }]


@pytest.fixture
def client():
    return OddsAPIClient()


class TestMoneylinePairing:
    def test_best_price_carries_its_own_books_other_side(self, client):
        games = _event(
            _book("bet365", "Bet365", ml=(-150, 130)),
            _book("fanduel", "FanDuel", ml=(-135, 115)),
        )
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=[])

        # -135 is the better home price; its pair must be FanDuel's +115,
        # not Bet365's +130.
        assert odds["home_price"] == -135
        assert odds["home_pair_away_price"] == 115
        assert odds["away_price"] == 130
        assert odds["away_pair_home_price"] == -150

    def test_cross_book_pairing_misprices_the_edge(self, client):
        """The old behaviour de-vigged the best home price against the best
        away price — a two-sided market no book offers. Because best-of-both
        always has a smaller overround than any real book, that understates
        the edge on whichever side is being evaluated."""
        games = _event(
            _book("bet365", "Bet365", ml=(-150, 130)),
            _book("fanduel", "FanDuel", ml=(-135, 115)),
        )
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=[])

        honest = calculate_edge_fair(
            0.60, odds["home_price"], odds["home_pair_away_price"], pick_home=True
        )
        mixed = calculate_edge_fair(
            0.60, odds["home_price"], odds["away_price"], pick_home=True
        )
        assert mixed < honest


class TestSpreadAtomicity:
    def test_line_and_price_come_from_one_book(self, client):
        games = _event(
            _book("bet365", "Bet365", spread=(-3.5, -110, -110)),
            _book("fanduel", "FanDuel", spread=(-2.5, -105, -115)),
        )
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=[])

        # FanDuel offers the better home price (-105); the line must be its
        # -2.5, never Bet365's -3.5.
        assert odds["spread_home_price"] == -105
        assert odds["spread_home"] == -2.5
        assert odds["spread_home_pair_away_price"] == -115


class TestTotalsAtomicity:
    def test_each_side_keeps_its_own_line(self, client):
        games = _event(
            _book("bet365", "Bet365", total=(44.5, -110, -110)),
            _book("fanduel", "FanDuel", total=(46.5, -105, -118)),
        )
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=[])

        assert odds["over_price"] == -105
        assert odds["total_line"] == 46.5
        assert odds["over_pair_under_price"] == -118

        assert odds["under_price"] == -110
        assert odds["total_under_line"] == 44.5
        assert odds["under_pair_over_price"] == -110


class TestBookmakerFilter:
    def test_restricts_to_named_books(self, client):
        games = _event(
            _book("bet365", "Bet365", ml=(-150, 130)),
            _book("fanduel", "FanDuel", ml=(-135, 115)),
        )
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=["bet365"])
        assert odds["home_price"] == -150
        assert odds["books"]["moneyline_home"] == "Bet365"

    def test_falls_back_when_the_book_has_no_quote(self, client):
        games = _event(_book("fanduel", "FanDuel", ml=(-135, 115)))
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=["bet365"])
        assert odds["home_price"] == -135

    def test_matches_events_by_canonical_name(self, client):
        games = _event(_book("bet365", "Bet365", ml=(-150, 130)))
        odds = client.get_best_odds(games, "KC", bookmakers=[], sport="NFL")
        assert odds["home_price"] == -150

    def test_unknown_matchup_returns_empty_quote(self, client):
        games = _event(_book("bet365", "Bet365", ml=(-150, 130)))
        odds = client.get_best_odds(games, "Denver Broncos", bookmakers=[])
        assert odds["home_price"] is None
        assert odds["books"] == {}


class TestPartialMarkets:
    def test_one_sided_market_is_not_paired(self, client):
        """A book quoting only one side cannot supply a de-vig pair."""
        games = _event({
            "key": "bet365", "title": "Bet365",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Kansas City Chiefs", "price": -150},
            ]}],
        })
        odds = client.get_best_odds(games, "Kansas City Chiefs", bookmakers=[])
        assert odds["home_price"] is None
        assert odds["home_pair_away_price"] is None
