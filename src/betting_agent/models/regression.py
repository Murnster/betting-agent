"""
XGBoost regression models: predicts home_score and away_score separately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

HOME_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 20,
}
AWAY_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "random_state": 42,
    "early_stopping_rounds": 20,
}


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_regressors(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    eval_split: float = 0.2,
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """
    Train home/away score regressors.
    Returns (home_model, away_model).
    """
    # Temporal split: first (1-eval_split) for train, last eval_split for test
    split_idx = int(len(X) * (1 - eval_split))
    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    yh_tr, yh_te = y_home.iloc[:split_idx], y_home.iloc[split_idx:]
    ya_tr, ya_te = y_away.iloc[:split_idx], y_away.iloc[split_idx:]

    home_model = xgb.XGBRegressor(**HOME_PARAMS)
    away_model = xgb.XGBRegressor(**AWAY_PARAMS)

    home_model.fit(X_tr, yh_tr, eval_set=[(X_te, yh_te)], verbose=False)
    away_model.fit(X_tr, ya_tr, eval_set=[(X_te, ya_te)], verbose=False)

    if verbose:
        yh_pred = home_model.predict(X_te)
        ya_pred = away_model.predict(X_te)
        logger.info("Home RMSE: %.2f  MAE: %.2f", _rmse(yh_te, yh_pred), mean_absolute_error(yh_te, yh_pred))
        logger.info("Away RMSE: %.2f  MAE: %.2f", _rmse(ya_te, ya_pred), mean_absolute_error(ya_te, ya_pred))

    return home_model, away_model


def train_final_regressors(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """Train on full dataset for deployment."""
    # Remove early_stopping_rounds for full-data training (no eval set)
    home_params = {k: v for k, v in HOME_PARAMS.items() if k != "early_stopping_rounds"}
    away_params = {k: v for k, v in AWAY_PARAMS.items() if k != "early_stopping_rounds"}

    home_model = xgb.XGBRegressor(**home_params)
    away_model = xgb.XGBRegressor(**away_params)
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    logger.info("Final regression models trained on %d rows", len(y_home))
    return home_model, away_model


def save_regressors(
    home_model: xgb.XGBRegressor,
    away_model: xgb.XGBRegressor,
    save_dir: Path,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(home_model, save_dir / "home_regression.joblib")
    joblib.dump(away_model, save_dir / "away_regression.joblib")
    logger.info("Regression models saved to %s", save_dir)


def load_regressors(
    save_dir: Path,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    save_dir = Path(save_dir)
    home_model = joblib.load(save_dir / "home_regression.joblib")
    away_model = joblib.load(save_dir / "away_regression.joblib")
    return home_model, away_model
