"""
Market-anchored NFL game model.

The August 2026 backtest showed the from-scratch game model loses to the
closing line and that its disagreements with the market are anti-predictive.
This model does not try to out-predict the market from scratch: the closing
spread and total are the prior, and it learns only the residual

    margin_resid = (home_score - away_score) - spread_line
    total_resid  = (home_score + away_score) - total_line

from the same feature matrix plus market shape (spread_abs, market_home_prob)
and QB continuity. With no learnable residual it collapses to the market,
which is the right failure mode. Win probability follows from the predicted
margin through a normal CDF with the held-out residual sigma.

`kind` selects the residual learner: "xgb" (shallow, heavily regularised
boosted trees), "ridge" (linear on imputed, standardised features) or "none"
(market only — the baseline every other kind has to beat).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET_COLS = ("home_team_wins", "home_score", "away_score")
ANCHOR_COLS = ("spread_line", "total_line")
KINDS = ("none", "ridge", "xgb")
ARTIFACT = "market_anchored.joblib"

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 25,
    "reg_lambda": 5.0,
    "random_state": 42,
    "early_stopping_rounds": 40,
}


def _learner(kind: str):
    if kind == "xgb":
        return xgb.XGBRegressor(**XGB_PARAMS)
    if kind == "ridge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(0, 4, 13))),
        ])
    if kind == "none":
        return None
    raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")


def residual_targets(X: pd.DataFrame, home_score: pd.Series, away_score: pd.Series
                     ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(margin_resid, total_resid, usable-row mask) for a feature frame carrying the lines."""
    spread = pd.to_numeric(X["spread_line"], errors="coerce")
    total = pd.to_numeric(X["total_line"], errors="coerce")
    ok = spread.notna() & total.notna() & home_score.notna() & away_score.notna()
    margin_resid = (home_score - away_score) - spread
    total_resid = (home_score + away_score) - total
    return margin_resid, total_resid, ok


class MarketAnchoredModel:
    def __init__(self, kind: str = "xgb"):
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
        self.kind = kind
        self.margin_model = _learner(kind)
        self.total_model = _learner(kind)
        self.feature_names: list[str] = []
        self.margin_sigma: float = 13.5
        self.total_sigma: float = 10.0
        self.n_rows: int = 0

    # ---- training ----
    def fit(self, X: pd.DataFrame, home_score: pd.Series, away_score: pd.Series,
            eval_split: float = 0.2) -> MarketAnchoredModel:
        for c in ANCHOR_COLS:
            if c not in X.columns:
                raise ValueError(f"market-anchored features need {c!r}; build with market_anchored=True")
        m_res, t_res, ok = residual_targets(X, home_score, away_score)
        Xf, m_res, t_res = X[ok], m_res[ok], t_res[ok]
        self.feature_names = list(Xf.columns)
        self.n_rows = len(Xf)
        split = int(len(Xf) * (1 - eval_split))
        X_tr, X_te = Xf.iloc[:split], Xf.iloc[split:]

        if self.kind == "none":
            pm, pt = np.zeros(len(X_te)), np.zeros(len(X_te))
        else:
            self._fit_one(self.margin_model, X_tr, m_res.iloc[:split], X_te, m_res.iloc[split:])
            self._fit_one(self.total_model, X_tr, t_res.iloc[:split], X_te, t_res.iloc[split:])
            pm, pt = self._raw(self.margin_model, X_te), self._raw(self.total_model, X_te)
        # Residual spread on the held-out tail — what the CDF is scaled with.
        self.margin_sigma = float(np.std(m_res.iloc[split:].to_numpy() - pm, ddof=1)) if len(X_te) > 2 else 13.5
        self.total_sigma = float(np.std(t_res.iloc[split:].to_numpy() - pt, ddof=1)) if len(X_te) > 2 else 10.0

        if self.kind != "none":
            # Refit on everything for deployment; sigma keeps the held-out estimate.
            self._fit_one(self.margin_model, Xf, m_res, X_te, m_res.iloc[split:])
            self._fit_one(self.total_model, Xf, t_res, X_te, t_res.iloc[split:])
        logger.info("MarketAnchoredModel[%s] fit on %d rows: sigma margin=%.2f total=%.2f",
                    self.kind, self.n_rows, self.margin_sigma, self.total_sigma)
        return self

    def _fit_one(self, model, X_tr, y_tr, X_te, y_te) -> None:
        if isinstance(model, xgb.XGBRegressor):
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

    def _raw(self, model, X: pd.DataFrame) -> np.ndarray:
        if model is None:
            return np.zeros(len(X))
        return np.asarray(model.predict(X), dtype=float)

    # ---- inference ----
    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_names) if self.feature_names else X

    def residuals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(margin residual, total residual) the model adds to the lines."""
        Xa = self.align(X)
        return self._raw(self.margin_model, Xa), self._raw(self.total_model, Xa)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Same columns as PredictionEngine.predict()."""
        spread = pd.to_numeric(X["spread_line"], errors="coerce").to_numpy(dtype=float)
        total = pd.to_numeric(X["total_line"], errors="coerce").to_numpy(dtype=float)
        r_m, r_t = self.residuals(X)
        pred_margin = spread + r_m
        pred_total = total + r_t
        win_prob = np.clip(norm.cdf(pred_margin / self.margin_sigma), 0.05, 0.95)
        return pd.DataFrame({
            "win_prob": win_prob,
            "home_pred_score": (pred_total + pred_margin) / 2.0,
            "away_pred_score": (pred_total - pred_margin) / 2.0,
            "pred_total": pred_total,
            "pred_margin": pred_margin,
            "margin_resid": r_m,
            "total_resid": r_t,
        }, index=X.index)

    def get_sigma(self) -> dict[str, float]:
        return {"total_sigma": self.total_sigma, "margin_sigma": self.margin_sigma}

    # ---- persistence ----
    def save(self, save_dir: Path | str) -> Path:
        path = Path(save_dir) / ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, save_dir: Path | str) -> MarketAnchoredModel:
        return joblib.load(Path(save_dir) / ARTIFACT)
