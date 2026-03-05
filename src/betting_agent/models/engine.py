"""
Model engine: load saved models and provide a unified prediction interface.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from betting_agent.config import settings
from betting_agent.models.classification import load_classifier
from betting_agent.models.regression import load_regressors, load_sigma

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Loads trained models from saved_models/ and exposes predict().
    Supported sports: 'NFL' (more to come).
    """

    def __init__(self, sport: str = "NFL", models_dir: Path | str | None = None):
        self.sport = sport.upper()
        self.models_dir = Path(models_dir or settings.saved_models_dir) / self.sport
        self._classifier: xgb.Booster | None = None
        self._calibrator = None
        self._home_reg = None
        self._away_reg = None
        self._feature_names: list[str] | None = None
        self._total_sigma: float | None = None
        self._margin_sigma: float | None = None
        self._loaded = False

    def load(self) -> bool:
        """Load all models from disk. Returns True on success."""
        d = self.models_dir
        try:
            self._classifier = load_classifier(d / "classifier.json")
            self._home_reg, self._away_reg = load_regressors(d)

            feat_path = d / "feature_names.pkl"
            if feat_path.exists():
                with open(feat_path, "rb") as f:
                    self._feature_names = pickle.load(f)

            cal_path = d / "calibrator.joblib"
            if cal_path.exists():
                self._calibrator = joblib.load(cal_path)
                logger.info("Calibrator loaded from %s", cal_path)

            sigma_dict = load_sigma(d)
            if sigma_dict is not None:
                self._total_sigma = sigma_dict.get("total_sigma")
                self._margin_sigma = sigma_dict.get("margin_sigma")
                logger.info(
                    "Scoring sigma loaded: total=%.2f, margin=%.2f",
                    self._total_sigma, self._margin_sigma,
                )

            self._loaded = True
            logger.info("PredictionEngine loaded for %s from %s", self.sport, d)
            return True
        except FileNotFoundError as exc:
            logger.error("Model files not found: %s. Run train.py first.", exc)
            return False

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Return array of home-team win probabilities (calibrated if available)."""
        self._require_loaded()
        if self._feature_names:
            X = X.reindex(columns=self._feature_names, fill_value=0)
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        raw_probs = self._classifier.predict(dmat)
        if self._calibrator is not None:
            if hasattr(self._calibrator, "predict_proba"):
                cal_probs = self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                cal_probs = self._calibrator.predict(raw_probs)
            return np.clip(cal_probs, 0.05, 0.95)
        return raw_probs

    def predict_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (home_scores, away_scores) arrays."""
        self._require_loaded()
        if self._feature_names:
            X = X.reindex(columns=self._feature_names, fill_value=0)
        home_scores = self._home_reg.predict(X)
        away_scores = self._away_reg.predict(X)
        return home_scores, away_scores

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Full prediction pipeline.
        Returns DataFrame with:
          win_prob, home_pred_score, away_pred_score, pred_total, pred_margin
        """
        win_probs = self.predict_win_prob(X)
        home_scores, away_scores = self.predict_scores(X)

        return pd.DataFrame(
            {
                "win_prob": win_probs,
                "home_pred_score": home_scores,
                "away_pred_score": away_scores,
                "pred_total": home_scores + away_scores,
                "pred_margin": home_scores - away_scores,
            },
            index=X.index,
        )

    def get_sigma(self) -> dict[str, float]:
        """Return empirical sigma values, falling back to sport-specific defaults."""
        from betting_agent.sports.registry import get_sport_config
        fallback = get_sport_config(self.sport).scoring_sigma
        return {
            "total_sigma": self._total_sigma if self._total_sigma is not None else fallback,
            "margin_sigma": self._margin_sigma if self._margin_sigma is not None else fallback,
        }
