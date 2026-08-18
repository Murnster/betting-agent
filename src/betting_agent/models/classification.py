"""
XGBoost binary classifier: predicts home team win (1) or loss (0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, log_loss

logger = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "seed": 42,
}
NUM_ROUNDS = 800
EARLY_STOPPING = 30


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    eval_split: float = 0.2,
    verbose: bool = True,
) -> xgb.Booster:
    """Train XGBoost classifier with early stopping. Returns Booster."""
    # Temporal split: first (1-eval_split) for train, last eval_split for test
    split_idx = int(len(X) * (1 - eval_split))
    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
    params = {**PARAMS, "scale_pos_weight": scale_pos_weight}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=list(X.columns))

    callbacks = []
    if EARLY_STOPPING:
        callbacks.append(xgb.callback.EarlyStopping(rounds=EARLY_STOPPING))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=[(dtrain, "train"), (dtest, "eval")],
        verbose_eval=50 if verbose else False,
        callbacks=callbacks,
    )

    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_te, y_pred)
    if verbose:
        logger.info("Classification test accuracy: %.4f", acc)
        logger.info("\n%s", classification_report(y_te, y_pred))

    return model


def _log_calibration_diagnostics(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    label: str = "test",
) -> None:
    """Log calibration quality metrics for raw vs calibrated probabilities."""
    brier_raw = brier_score_loss(y_true, raw_probs)
    brier_cal = brier_score_loss(y_true, cal_probs)
    ll_raw = log_loss(y_true, np.clip(raw_probs, 1e-15, 1 - 1e-15))
    ll_cal = log_loss(y_true, np.clip(cal_probs, 1e-15, 1 - 1e-15))

    logger.info("=== Calibration diagnostics (%s, n=%d) ===", label, len(y_true))
    logger.info("  Brier score  — raw: %.4f  calibrated: %.4f", brier_raw, brier_cal)
    logger.info("  Log loss     — raw: %.4f  calibrated: %.4f", ll_raw, ll_cal)

    for name, probs in [("raw", raw_probs), ("calibrated", cal_probs)]:
        pcts = np.percentile(probs, [0, 25, 50, 75, 100])
        logger.info(
            "  %s probs — min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f",
            name, *pcts,
        )

    # Calibration by bin
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        pred_avg = cal_probs[mask].mean()
        actual_avg = y_true[mask].mean()
        logger.info(
            "  bin [%.1f–%.1f): n=%d  pred_avg=%.3f  actual_avg=%.3f",
            lo, min(hi, 1.0), mask.sum(), pred_avg, actual_avg,
        )


def train_calibrated_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    verbose: bool = True,
) -> tuple[xgb.Booster, IsotonicRegression]:
    """
    Train XGBoost classifier with isotonic regression calibration on a held-out fold.

    Uses a 60/20/20 chronological split: train / calibration / test.
    Returns (booster, calibrator) — save calibrator alongside the model.
    """
    # 60/20/20 chronological split
    n = len(X)
    split_train = int(n * 0.60)
    split_cal = int(n * 0.80)
    X_train, X_cal, X_test = X.iloc[:split_train], X.iloc[split_train:split_cal], X.iloc[split_cal:]
    y_train, y_cal, y_test = y.iloc[:split_train], y.iloc[split_train:split_cal], y.iloc[split_cal:]

    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    params = {**PARAMS, "scale_pos_weight": scale_pos_weight}

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
    dcal = xgb.DMatrix(X_cal, label=y_cal, feature_names=list(X.columns))

    callbacks = []
    if EARLY_STOPPING:
        callbacks.append(xgb.callback.EarlyStopping(rounds=EARLY_STOPPING))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=[(dtrain, "train"), (dtest, "eval")],
        verbose_eval=50 if verbose else False,
        callbacks=callbacks,
    )

    # Fit isotonic regression calibrator on calibration fold
    cal_probs_raw = model.predict(dcal)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(cal_probs_raw, y_cal.values)

    # Warn if calibrated output range is too narrow
    cal_output = calibrator.predict(cal_probs_raw)
    cal_min, cal_max = float(cal_output.min()), float(cal_output.max())
    n_points = len(getattr(calibrator, "X_thresholds_", []))
    logger.info(
        "Calibrator: %d interpolation points, output range [%.3f, %.3f]",
        n_points, cal_min, cal_max,
    )
    if cal_min > 0.10 or cal_max < 0.90:
        logger.warning(
            "Calibration range [%.3f, %.3f] does not cover [0.10, 0.90] — "
            "model may not distinguish strong favorites/underdogs. "
            "Consider adding more training data or tuning hyperparameters.",
            cal_min, cal_max,
        )

    # Evaluate on test set using calibrated probabilities
    test_probs_raw = model.predict(dtest)
    test_probs_cal = calibrator.predict(test_probs_raw)
    y_pred = (test_probs_cal > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        logger.info("Calibrated classification test accuracy: %.4f", acc)
        logger.info("\n%s", classification_report(y_test, y_pred))

        # Log calibration diagnostics on calibration fold (in-sample)
        cal_probs_cal = calibrator.predict(cal_probs_raw)
        _log_calibration_diagnostics(y_cal.values, cal_probs_raw, cal_probs_cal, label="calibration (in-sample)")

        # Log calibration diagnostics on test fold (out-of-sample)
        _log_calibration_diagnostics(y_test.values, test_probs_raw, test_probs_cal, label="test (out-of-sample)")

    return model, calibrator


def train_final_classifier(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    """Train on full dataset (no eval split) for deployment."""
    scale_pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)
    params = {**PARAMS, "scale_pos_weight": scale_pos_weight}
    d = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    model = xgb.train(params, d, num_boost_round=NUM_ROUNDS)
    logger.info("Final classification model trained on %d rows", len(y))
    return model


def save_classifier(model: xgb.Booster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Classification model saved to %s", path)


def load_classifier(path: Path) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(str(path))
    return model
