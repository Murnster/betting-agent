"""
XGBoost binary classifier: predicts home team win (1) or loss (0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, log_loss

logger = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.01,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "seed": 42,
}
NUM_ROUNDS = 500
EARLY_STOPPING = 20


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


class IsotonicEnsemble(BaseEstimator, RegressorMixin):
    """Average of multiple isotonic calibrators (for smoother tails)."""

    def __init__(self, calibrators: list[IsotonicRegression]) -> None:
        if not calibrators:
            raise ValueError("IsotonicEnsemble requires at least one calibrator")
        self.calibrators = calibrators

    @property
    def n_calibrators(self) -> int:
        return len(self.calibrators)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        preds = [cal.predict(probs) for cal in self.calibrators]
        return np.mean(np.vstack(preds), axis=0)


def train_calibrated_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    verbose: bool = True,
    k_folds: int = 5,
) -> tuple[xgb.Booster, IsotonicEnsemble]:
    """
    Train XGBoost classifier with k-fold isotonic calibration.

    Uses an 80/20 chronological split: train+calibration / test.
    Fit k isotonic calibrators on expanding time-series folds within the 80%.
    Returns (booster, calibrator_ensemble) — save calibrator alongside the model.
    """
    n = len(X)
    split_test = int(n * 0.80)
    X_traincal, X_test = X.iloc[:split_test], X.iloc[split_test:]
    y_traincal, y_test = y.iloc[:split_test], y.iloc[split_test:]

    scale_pos_weight = len(y_traincal[y_traincal == 0]) / max(len(y_traincal[y_traincal == 1]), 1)
    params = {**PARAMS, "scale_pos_weight": scale_pos_weight}

    dtrain = xgb.DMatrix(X_traincal, label=y_traincal, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))

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

    # K-fold time-series isotonic calibration on train+cal split
    calibrators: list[IsotonicRegression] = []
    n_traincal = len(X_traincal)
    
    # Start calibration after 20% of data, but ensure meaningful split for small datasets
    if n_traincal < 100:
        min_train = max(int(n_traincal * 0.80), 1)
    else:
        min_train = max(50, int(n_traincal * 0.20))
    
    min_train = min(min_train, max(n_traincal - 1, 1))
    remaining = n_traincal - min_train

    if remaining < k_folds:
        k_folds = max(1, remaining)
    fold_size = max(1, remaining // max(k_folds, 1))

    for i in range(k_folds):
        cal_start = min_train + i * fold_size
        if cal_start >= n_traincal:
            break
        cal_end = n_traincal if i == k_folds - 1 else min(n_traincal, cal_start + fold_size)
        if cal_end <= cal_start:
            continue

        X_fold_train = X_traincal.iloc[:cal_start]
        y_fold_train = y_traincal.iloc[:cal_start]
        X_fold_cal = X_traincal.iloc[cal_start:cal_end]
        y_fold_cal = y_traincal.iloc[cal_start:cal_end]

        if len(X_fold_train) == 0 or len(X_fold_cal) == 0:
            continue

        d_fold_train = xgb.DMatrix(X_fold_train, label=y_fold_train, feature_names=list(X.columns))
        d_fold_cal = xgb.DMatrix(X_fold_cal, label=y_fold_cal, feature_names=list(X.columns))

        fold_model = xgb.train(
            params,
            d_fold_train,
            num_boost_round=NUM_ROUNDS,
            evals=[(d_fold_train, "train")],
            verbose_eval=False,
        )
        cal_probs_raw = fold_model.predict(d_fold_cal)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(cal_probs_raw, y_fold_cal.values)
        calibrators.append(calibrator)

    if not calibrators:
        raise RuntimeError("No calibrators trained; insufficient data for k-fold calibration")

    # Evaluate on test set using calibrated probabilities
    test_probs_raw = model.predict(dtest)
    calibrator_ensemble = IsotonicEnsemble(calibrators)
    test_probs_cal = calibrator_ensemble.predict(test_probs_raw)
    y_pred = (test_probs_cal > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        logger.info("Calibrated classification test accuracy: %.4f", acc)
        logger.info("\n%s", classification_report(y_test, y_pred))

        # Log calibration diagnostics on test fold (out-of-sample)
        _log_calibration_diagnostics(y_test.values, test_probs_raw, test_probs_cal, label="test (out-of-sample)")

    return model, calibrator_ensemble


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
