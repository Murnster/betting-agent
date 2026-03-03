"""Tests for Kelly criterion scaling and bet sizing."""

from betting_agent.intelligence.kelly import kelly_fraction, scaled_kelly, recommended_bet


def test_kelly_scaling_direction():
    """Large edges should get LESS aggressive Kelly scaling (more conservative)."""
    prob = 0.60
    odds = -110

    k_small = scaled_kelly(prob, odds, edge=0.02, max_kelly=0.10)
    k_medium = scaled_kelly(prob, odds, edge=0.04, max_kelly=0.10)
    k_large = scaled_kelly(prob, odds, edge=0.06, max_kelly=0.10)

    # Small edge (<=3%) gets 80% Kelly → largest fraction
    # Medium edge (3-5%) gets 60% Kelly
    # Large edge (>5%) gets 40% Kelly → smallest fraction
    assert k_small > k_medium, "Small edge should get larger Kelly fraction than medium"
    assert k_medium > k_large, "Medium edge should get larger Kelly fraction than large"


def test_kelly_scaling_factors():
    """Verify exact scaling factors: 80% for small, 60% for medium, 40% for large."""
    prob = 0.55
    odds = 100
    full_k = kelly_fraction(prob, odds)

    k_small = scaled_kelly(prob, odds, edge=0.01, max_kelly=1.0)
    k_medium = scaled_kelly(prob, odds, edge=0.04, max_kelly=1.0)
    k_large = scaled_kelly(prob, odds, edge=0.06, max_kelly=1.0)

    assert abs(k_small - full_k * 0.8) < 1e-9
    assert abs(k_medium - full_k * 0.6) < 1e-9
    assert abs(k_large - full_k * 0.4) < 1e-9


def test_min_bet_floor_zero_default():
    """Default min_bet_pct should be 0.0 (no forced minimum bet)."""
    from betting_agent.config import Settings
    s = Settings()
    assert s.min_bet_pct == 0.0


def test_recommended_bet_no_forced_minimum():
    """With min_bet_pct=0.0, a tiny Kelly should produce a tiny bet."""
    kf, amount = recommended_bet(
        prob=0.52, american_odds=-110, edge=0.016,
        bankroll=100.0, min_bet_pct=0.0,
    )
    # With tiny edge, Kelly fraction is tiny, bet should be proportional
    assert amount < 5.0  # should be well under 5% of bankroll
    assert amount >= 0.0


def test_kelly_never_negative():
    """Kelly fraction and bet should never be negative."""
    kf, amount = recommended_bet(
        prob=0.45, american_odds=-110, edge=0.01,
        bankroll=100.0,
    )
    assert kf >= 0.0
    assert amount >= 0.0
