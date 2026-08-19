"""Tests for grading logic — moneyline, spread, and O/U (total) picks."""

from unittest.mock import MagicMock

from betting_agent.accounting.grader import _grade_moneyline, _grade_spread, _grade_total


def _make_pick(**kwargs):
    pick = MagicMock()
    for k, v in kwargs.items():
        setattr(pick, k, v)
    return pick


def _make_game(**kwargs):
    game = MagicMock()
    for k, v in kwargs.items():
        setattr(game, k, v)
    return game


# ---- Moneyline ----

def test_grade_moneyline_win():
    pick = _make_pick(pick_side="Kansas City Chiefs")
    game = _make_game(home_team="Kansas City Chiefs", away_team="Buffalo Bills", home_score=27, away_score=24)
    assert _grade_moneyline(pick, game) == "win"


def test_grade_moneyline_loss():
    pick = _make_pick(pick_side="Buffalo Bills")
    game = _make_game(home_team="Kansas City Chiefs", away_team="Buffalo Bills", home_score=27, away_score=24)
    assert _grade_moneyline(pick, game) == "loss"


def test_grade_moneyline_nba_win():
    """NBA picks now use full Odds API names — must match game table names."""
    pick = _make_pick(pick_side="Boston Celtics")
    game = _make_game(home_team="Boston Celtics", away_team="Miami Heat", home_score=110, away_score=102)
    assert _grade_moneyline(pick, game) == "win"


def test_grade_moneyline_nba_loss():
    pick = _make_pick(pick_side="Miami Heat")
    game = _make_game(home_team="Boston Celtics", away_team="Miami Heat", home_score=110, away_score=102)
    assert _grade_moneyline(pick, game) == "loss"


def test_grade_moneyline_tie():
    pick = _make_pick(pick_side="Kansas City Chiefs")
    game = _make_game(home_team="Kansas City Chiefs", away_team="Buffalo Bills", home_score=24, away_score=24)
    assert _grade_moneyline(pick, game) == "push"


# ---- Spread ----

def test_grade_spread_covers():
    pick = _make_pick(pick_side="Kansas City Chiefs +3.5")
    game = _make_game(home_team="Kansas City Chiefs", away_team="Buffalo Bills", home_score=24, away_score=27)
    # KC lost by 3, but had +3.5 → covers (24 - 27 + 3.5 = 0.5 > 0)
    assert _grade_spread(pick, game) == "win"


def test_grade_spread_fails():
    pick = _make_pick(pick_side="Kansas City Chiefs +2.5")
    game = _make_game(home_team="Kansas City Chiefs", away_team="Buffalo Bills", home_score=24, away_score=27)
    # KC lost by 3, had +2.5 → doesn't cover (24 - 27 + 2.5 = -0.5 < 0)
    assert _grade_spread(pick, game) == "loss"


def test_grade_spread_nba_covers():
    """NBA spread pick uses full team name — grader must match correctly."""
    pick = _make_pick(pick_side="Boston Celtics +4.5")
    game = _make_game(home_team="Boston Celtics", away_team="Miami Heat", home_score=100, away_score=103)
    # Lost by 3, had +4.5 → covers (100 - 103 + 4.5 = 1.5 > 0)
    assert _grade_spread(pick, game) == "win"


def test_grade_spread_nba_fails():
    pick = _make_pick(pick_side="Boston Celtics +2.5")
    game = _make_game(home_team="Boston Celtics", away_team="Miami Heat", home_score=100, away_score=103)
    # Lost by 3, had +2.5 → doesn't cover (100 - 103 + 2.5 = -0.5 < 0)
    assert _grade_spread(pick, game) == "loss"


# ---- Total (Over/Under) ----

def test_grade_total_over_win():
    pick = _make_pick(pick_side="over 45.5")
    game = _make_game(home_score=27, away_score=24)
    # Total = 51 > 45.5 → win
    assert _grade_total(pick, game) == "win"


def test_grade_total_over_loss():
    pick = _make_pick(pick_side="over 55.5")
    game = _make_game(home_score=27, away_score=24)
    # Total = 51 < 55.5 → loss
    assert _grade_total(pick, game) == "loss"


def test_grade_total_under_win():
    pick = _make_pick(pick_side="under 55.5")
    game = _make_game(home_score=27, away_score=24)
    # Total = 51 < 55.5 → win
    assert _grade_total(pick, game) == "win"


def test_grade_total_under_loss():
    pick = _make_pick(pick_side="under 45.5")
    game = _make_game(home_score=27, away_score=24)
    # Total = 51 > 45.5 → loss
    assert _grade_total(pick, game) == "loss"


def test_grade_total_push():
    pick = _make_pick(pick_side="over 51.0")
    game = _make_game(home_score=27, away_score=24)
    # Total = 51 == 51.0 → push
    assert _grade_total(pick, game) == "push"


def test_grade_total_no_scores():
    pick = _make_pick(pick_side="over 45.5")
    game = _make_game(home_score=None, away_score=None)
    assert _grade_total(pick, game) == "push"


def test_grade_total_unparseable_is_not_a_push():
    """An unreadable pick_side must stay ungraded rather than book a bet as
    break-even and quietly bury the problem."""
    pick = _make_pick(pick_side="over", bet_type="total")
    game = _make_game(home_score=27, away_score=24)
    assert _grade_total(pick, game) is None


# ---- Mixed naming conventions ----
# Picks store Odds API club names; loader-seeded games store abbreviations.
# Before canonical matching, these graded against the wrong side.

def test_grade_moneyline_pick_full_name_game_abbreviation():
    pick = _make_pick(pick_side="Boston Celtics", sport="NBA", bet_type="moneyline")
    game = _make_game(sport="NBA", home_team="BOS", away_team="MIA",
                      home_score=110, away_score=102)
    assert _grade_moneyline(pick, game) == "win"


def test_grade_moneyline_away_pick_full_name_game_abbreviation():
    pick = _make_pick(pick_side="Miami Heat", sport="NBA", bet_type="moneyline")
    game = _make_game(sport="NBA", home_team="BOS", away_team="MIA",
                      home_score=110, away_score=102)
    assert _grade_moneyline(pick, game) == "loss"


def test_grade_nfl_moneyline_across_conventions():
    pick = _make_pick(pick_side="Kansas City Chiefs", sport="NFL", bet_type="moneyline")
    game = _make_game(sport="NFL", home_team="KC", away_team="BUF",
                      home_score=27, away_score=24)
    assert _grade_moneyline(pick, game) == "win"


def test_grade_spread_away_side_measured_from_correct_team():
    """The away team's margin is its own, not the home team's negated by
    accident of a name mismatch."""
    pick = _make_pick(pick_side="Buffalo Bills +3.5", sport="NFL", bet_type="spread")
    game = _make_game(sport="NFL", home_team="KC", away_team="BUF",
                      home_score=27, away_score=24)
    # BUF lost by 3 with +3.5 → covers.
    assert _grade_spread(pick, game) == "win"


def test_grade_spread_home_side_across_conventions():
    pick = _make_pick(pick_side="Kansas City Chiefs -3.5", sport="NFL", bet_type="spread")
    game = _make_game(sport="NFL", home_team="KC", away_team="BUF",
                      home_score=27, away_score=24)
    # KC won by 3, laying 3.5 → does not cover.
    assert _grade_spread(pick, game) == "loss"


def test_unmatched_team_is_left_ungraded():
    pick = _make_pick(pick_side="Toronto Argonauts", sport="NFL", bet_type="moneyline",
                      id=1)
    game = _make_game(sport="NFL", home_team="KC", away_team="BUF",
                      home_score=27, away_score=24, id=9)
    assert _grade_moneyline(pick, game) is None


def test_unmatched_spread_team_is_left_ungraded():
    pick = _make_pick(pick_side="Toronto Argonauts +3.5", sport="NFL",
                      bet_type="spread", id=1)
    game = _make_game(sport="NFL", home_team="KC", away_team="BUF",
                      home_score=27, away_score=24, id=9)
    assert _grade_spread(pick, game) is None
