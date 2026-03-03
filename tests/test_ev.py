"""Basic tests for EV / Kelly calculation correctness."""

import numpy as np
import pandas as pd

from betting_agent.intelligence.ev import (
    american_to_implied_prob,
    calculate_edge,
    calculate_spread_edge,
    calculate_total_edge,
    remove_vig,
)
from betting_agent.intelligence.kelly import kelly_fraction, scaled_kelly
from betting_agent.models.regression import compute_residual_sigma


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


def test_total_edge_large_sigma_compresses_probability():
    """Larger sigma should push probability closer to 50%."""
    prob_small, _ = calculate_total_edge(228.5, 216.0, -110, "over", sigma=11.5)
    prob_large, _ = calculate_total_edge(228.5, 216.0, -110, "over", sigma=22.0)
    # Both should be >0.5 since predicted > line, but large sigma closer to 0.5
    assert prob_small > prob_large
    assert prob_large < 0.80  # should be much more moderate with large sigma


def test_total_edge_clipping():
    """Extreme z-scores should clip probability to [0.10, 0.90]."""
    # Massive predicted total vs low line → should clip at 0.90
    prob_over, _ = calculate_total_edge(300.0, 200.0, -110, "over", sigma=10.0)
    assert prob_over == 0.90

    # Massive under → should clip at 0.90 for under side
    prob_under, _ = calculate_total_edge(100.0, 200.0, -110, "under", sigma=10.0)
    assert prob_under == 0.90


def test_spread_edge_clipping():
    """Extreme spread predictions should clip probability to [0.10, 0.90]."""
    # Huge predicted margin vs small spread → clip at 0.90
    prob_cover, _ = calculate_spread_edge(30.0, -3.0, -110, sigma=10.0)
    assert prob_cover == 0.90

    # Huge negative margin → clip at 0.10
    prob_cover, _ = calculate_spread_edge(-30.0, -3.0, -110, sigma=10.0)
    assert prob_cover == 0.10


def test_compute_residual_sigma():
    """Synthetic data should return reasonable sigma values."""
    from unittest.mock import MagicMock

    rng = np.random.RandomState(42)
    n = 200
    X_test = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y_home = pd.Series(rng.normal(110, 10, n))
    y_away = pd.Series(rng.normal(105, 10, n))

    # Mock models that predict with some noise
    home_model = MagicMock()
    home_model.predict.return_value = y_home.values + rng.normal(0, 8, n)
    away_model = MagicMock()
    away_model.predict.return_value = y_away.values + rng.normal(0, 8, n)

    sigma = compute_residual_sigma(home_model, away_model, X_test, y_home, y_away)
    assert "total_sigma" in sigma
    assert "margin_sigma" in sigma
    # With noise std=8 for each, total residual std ≈ sqrt(8^2+8^2) ≈ 11.3
    assert 5.0 < sigma["total_sigma"] < 20.0
    assert 5.0 < sigma["margin_sigma"] < 20.0
