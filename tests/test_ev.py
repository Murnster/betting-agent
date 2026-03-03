"""Basic tests for EV / Kelly calculation correctness."""

from betting_agent.intelligence.ev import american_to_implied_prob, calculate_edge, remove_vig
from betting_agent.intelligence.kelly import kelly_fraction, scaled_kelly


def test_implied_prob_negative_odds():
    p = american_to_implied_prob(-110)
    assert abs(p - 0.5238) < 0.001


def test_implied_prob_positive_odds():
    p = american_to_implied_prob(150)
    assert abs(p - 0.4) < 0.001


def test_edge_positive():
    # Model says 55%, market says 52.38% → positive edge
    edge = calculate_edge(0.55, -110)
    assert edge > 0


def test_edge_negative():
    edge = calculate_edge(0.45, -110)
    assert edge < 0


def test_remove_vig():
    fair_home, fair_away = remove_vig(0.55, 0.50)
    assert abs(fair_home + fair_away - 1.0) < 1e-9


def test_kelly_fraction_positive():
    k = kelly_fraction(0.55, -110)
    assert k > 0


def test_kelly_capped():
    k = scaled_kelly(0.60, -110, edge=0.08, max_kelly=0.05)
    assert k <= 0.05
