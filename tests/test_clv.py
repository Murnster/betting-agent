"""Tests for closing line value: the maths and, more importantly, which side
of the closing line a pick is scored against."""

from unittest.mock import MagicMock

from betting_agent.accounting.clv import _closing_price_for_side, calculate_clv


def _make(**kwargs):
    obj = MagicMock()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _pick(pick_side, bet_type="moneyline", sport="NFL",
          home_team="KC", away_team="BUF"):
    return _make(
        id=1,
        pick_side=pick_side,
        bet_type=bet_type,
        sport=sport,
        game=_make(sport=sport, home_team=home_team, away_team=away_team),
    )


def _closing(home_price=-120, away_price=100):
    return _make(home_price=home_price, away_price=away_price)


class TestCalculateClv:
    def test_better_price_than_close_is_positive(self):
        # Bet +120, closed +100: we got the better number.
        assert calculate_clv(120, 100) > 0

    def test_worse_price_than_close_is_negative(self):
        assert calculate_clv(100, 120) < 0

    def test_same_price_is_zero(self):
        assert calculate_clv(-110, -110) == 0


class TestClosingPriceForSide:
    def test_over_uses_home_price(self):
        price = _closing_price_for_side(_pick("over 45.5", "total"), _closing())
        assert price == -120

    def test_under_uses_away_price(self):
        price = _closing_price_for_side(_pick("under 45.5", "total"), _closing())
        assert price == 100

    def test_home_moneyline_across_naming_conventions(self):
        """Pick stores the Odds API club name, the game row the abbreviation.
        The old code fell through to the away price on any mismatch."""
        pick = _pick("Kansas City Chiefs")
        assert _closing_price_for_side(pick, _closing()) == -120

    def test_away_moneyline_across_naming_conventions(self):
        pick = _pick("Buffalo Bills")
        assert _closing_price_for_side(pick, _closing()) == 100

    def test_spread_pick_strips_the_line_before_matching(self):
        pick = _pick("Kansas City Chiefs -3.5", bet_type="spread")
        assert _closing_price_for_side(pick, _closing()) == -120

    def test_unmatched_team_returns_none_rather_than_guessing(self):
        pick = _pick("Toronto Argonauts")
        assert _closing_price_for_side(pick, _closing()) is None

    def test_missing_game_returns_none(self):
        pick = _pick("Kansas City Chiefs")
        pick.game = None
        assert _closing_price_for_side(pick, _closing()) is None
