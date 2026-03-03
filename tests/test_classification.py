"""Tests for classification model training, calibration, and engine prediction."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from unittest.mock import MagicMock

from betting_agent.models.classification import (
    _log_calibration_diagnostics,
    train_calibrated_classifier,
)
from betting_agent.models.engine import PredictionEngine


def _make_synthetic_data(n: int = 100, seed: int = 42):
    """Generate synthetic binary classification data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    # Create a signal so XGBoost can learn something
    logits = 0.5 * X["f0"] - 0.3 * X["f1"] + 0.2 * X["f2"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.rand(n) < probs).astype(int), name="home_team_wins")
    return X, y


def test_train_returns_isotonic_calibrator():
    X, y = _make_synthetic_data(200)
    model, calibrator = train_calibrated_classifier(X, y, verbose=False)
    assert isinstance(calibrator, IsotonicRegression)


def test_calibrated_probs_in_range():
    X, y = _make_synthetic_data(200)
    model, calibrator = train_calibrated_classifier(X, y, verbose=False)
    import xgboost as xgb

    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    raw = model.predict(dmat)
    cal = calibrator.predict(raw)
    assert np.all(cal >= 0.0)
    assert np.all(cal <= 1.0)


def test_calibrator_modifies_probabilities():
    X, y = _make_synthetic_data(200)
    model, calibrator = train_calibrated_classifier(X, y, verbose=False)
    import xgboost as xgb

    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    raw = model.predict(dmat)
    cal = calibrator.predict(raw)
    # Calibrator should change at least some probabilities
    assert not np.allclose(raw, cal, atol=1e-6)


def test_diagnostics_logs_output(caplog):
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 50).astype(float)
    raw = rng.rand(50).astype(np.float64)
    cal = rng.rand(50).astype(np.float64)
    with caplog.at_level(logging.INFO):
        _log_calibration_diagnostics(y_true, raw, cal, label="unit-test")
    assert "Brier score" in caplog.text
    assert "Log loss" in caplog.text
    assert "unit-test" in caplog.text


def test_engine_handles_isotonic_calibrator():
    engine = PredictionEngine.__new__(PredictionEngine)
    engine._loaded = True
    engine._feature_names = ["f0", "f1"]
    engine._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    # Fit on simple data
    engine._calibrator.fit(np.array([0.1, 0.5, 0.9]), np.array([0.0, 0.5, 1.0]))

    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.3, 0.7])
    engine._classifier = mock_booster

    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [0.5, 0.5]})
    probs = engine.predict_win_prob(X)
    assert len(probs) == 2
    assert np.all(probs >= 0.05)
    assert np.all(probs <= 0.95)


def test_engine_handles_logistic_calibrator():
    """Backward compatibility: engine still works with Platt-scaled models."""
    engine = PredictionEngine.__new__(PredictionEngine)
    engine._loaded = True
    engine._feature_names = ["f0", "f1"]
    calibrator = LogisticRegression()
    # Fit on simple data so predict_proba works
    calibrator.fit(np.array([[0.1], [0.9], [0.2], [0.8]]), np.array([0, 1, 0, 1]))
    engine._calibrator = calibrator

    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.3, 0.7])
    engine._classifier = mock_booster

    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [0.5, 0.5]})
    probs = engine.predict_win_prob(X)
    assert len(probs) == 2
    assert np.all(probs >= 0.05)
    assert np.all(probs <= 0.95)


def test_probability_clipping():
    """Probabilities are clipped to [0.05, 0.95] after calibration."""
    engine = PredictionEngine.__new__(PredictionEngine)
    engine._loaded = True
    engine._feature_names = ["f0"]
    # Use a calibrator that returns extreme values
    calibrator = MagicMock()
    calibrator.predict.return_value = np.array([0.001, 0.999, 0.5])
    del calibrator.predict_proba  # ensure isotonic path is taken
    engine._calibrator = calibrator

    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.1, 0.9, 0.5])
    engine._classifier = mock_booster

    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    probs = engine.predict_win_prob(X)
    assert probs[0] == pytest.approx(0.05)
    assert probs[1] == pytest.approx(0.95)
    assert probs[2] == pytest.approx(0.5)
