"""Tests for real-closing-line grading.

The spread convention is the dangerous part: nflreadpy's `spread_line` is
positive when the HOME team is favoured, the opposite of the Odds API's point
value. Getting the sign backwards inverts every spread result while still
producing plausible-looking output.
"""

import pandas as pd
import pytest

from betting_agent.sports.nfl.market import (
    breakeven_rate,
    grade_moneyline,
    grade_spread,
    grade_total,
    has_closing_lines,
    line_coverage,
    payout,
)


class TestGradeMoneyline:
    def test_home_pick_wins(self):
        assert grade_moneyline(27, 24, "home") == "win"

    def test_home_pick_loses(self):
        assert grade_moneyline(24, 27, "home") == "loss"

    def test_away_pick_wins(self):
        assert grade_moneyline(24, 27, "away") == "win"

    def test_tie_is_a_push(self):
        assert grade_moneyline(24, 24, "home") == "push"
        assert grade_moneyline(24, 24, "away") == "push"


class TestGradeSpread:
    def test_home_favourite_covers(self):
        # spread_line +3 means home favoured by 3; winning by 7 covers.
        assert grade_spread(27, 20, 3.0, "home") == "win"

    def test_home_favourite_wins_but_does_not_cover(self):
        assert grade_spread(27, 25, 3.0, "home") == "loss"

    def test_away_underdog_covers_when_home_wins_narrowly(self):
        assert grade_spread(27, 25, 3.0, "away") == "win"

    def test_home_underdog_covers_by_losing_close(self):
        # spread_line -3 means the home team is a 3-point underdog.
        assert grade_spread(24, 26, -3.0, "home") == "win"

    def test_exact_margin_is_a_push(self):
        assert grade_spread(27, 24, 3.0, "home") == "push"
        assert grade_spread(27, 24, 3.0, "away") == "push"

    def test_sign_convention_is_not_inverted(self):
        """A home favourite that wins big must not grade as a loss."""
        assert grade_spread(35, 10, 7.0, "home") == "win"
        assert grade_spread(35, 10, 7.0, "away") == "loss"


class TestGradeTotal:
    def test_over_wins(self):
        assert grade_total(27, 24, 45.5, "over") == "win"

    def test_over_loses(self):
        assert grade_total(17, 14, 45.5, "over") == "loss"

    def test_under_wins(self):
        assert grade_total(17, 14, 45.5, "under") == "win"

    def test_exact_total_is_a_push(self):
        assert grade_total(27, 24, 51.0, "over") == "push"
        assert grade_total(27, 24, 51.0, "under") == "push"


class TestPayout:
    def test_favourite_win(self):
        assert payout(110, -110, "win") == pytest.approx(100.0)

    def test_underdog_win(self):
        assert payout(100, 150, "win") == pytest.approx(150.0)

    def test_loss_costs_the_stake(self):
        assert payout(100, -110, "loss") == -100.0

    def test_push_returns_nothing(self):
        assert payout(100, -110, "push") == 0.0


class TestBreakevenRate:
    def test_standard_juice(self):
        assert breakeven_rate(-110) == pytest.approx(0.5238, abs=1e-4)

    def test_even_money(self):
        assert breakeven_rate(100) == pytest.approx(0.5)

    def test_underdog_needs_less(self):
        assert breakeven_rate(200) == pytest.approx(1 / 3, abs=1e-4)


class TestLineDetection:
    def test_detects_closing_lines(self):
        df = pd.DataFrame({
            "spread_line": [3.0], "total_line": [45.5], "home_moneyline": [-150],
        })
        assert has_closing_lines(df)

    def test_absent_lines(self):
        assert not has_closing_lines(pd.DataFrame({"home_team": ["KC"]}))

    def test_coverage_counts_usable_rows(self):
        df = pd.DataFrame({
            "home_moneyline": [-150, None],
            "away_moneyline": [130, 120],
            "spread_line": [3.0, 2.5],
            "home_spread_odds": [-110, -110],
            "total_line": [45.5, 44.0],
            "over_odds": [-110, None],
        })
        coverage = line_coverage(df)
        assert coverage["moneyline"] == 0.5
        assert coverage["spread"] == 1.0
        assert coverage["total"] == 0.5

    def test_empty_frame_has_no_coverage(self):
        coverage = line_coverage(pd.DataFrame())
        assert coverage == {"moneyline": 0.0, "spread": 0.0, "total": 0.0}
