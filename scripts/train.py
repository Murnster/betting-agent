#!/usr/bin/env python
"""
Entry point: seed DB with historical data and train all models.

Usage:
    uv run python scripts/train.py [--sport NFL] [--seasons 2018 2019 2020 2021 2022 2023 2024]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from betting_agent.config import settings
from betting_agent.sports.registry import get_sport_config, available_sports
from betting_agent.models.classification import train_calibrated_classifier, save_classifier
from betting_agent.models.regression import train_final_regressors, save_regressors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train betting models")
    parser.add_argument(
        "--sport",
        type=str,
        default="NFL",
        help=f"Sport to train ({', '.join(available_sports())})",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons to include in training data (defaults per sport)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding the DB (just train from data directly)",
    )
    args = parser.parse_args()

    sport = args.sport.upper()
    config = get_sport_config(sport)
    seasons = args.seasons or config.default_seasons

    save_dir = Path(settings.saved_models_dir) / sport
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load raw schedules ----
    logger.info("Loading %s schedules for seasons: %s", sport, seasons)
    loader = config.loader_cls()
    raw = loader.load_schedules(seasons).to_pandas()
    logger.info("Raw schedules: %d rows", len(raw))

    # Optionally seed DB
    if not args.no_seed:
        try:
            n = loader.seed_games_table(seasons)
            logger.info("DB seeded: %d new games", n)
        except Exception as exc:
            logger.warning("DB seed failed (DB may not be running): %s", exc)

    # ---- 2. Feature engineering ----
    logger.info("Building feature matrix...")
    df = config.build_features(raw)
    X, y = config.split_features_targets(df)
    logger.info("Feature matrix: %d rows x %d features", len(X), len(X.columns))

    # Save feature names for alignment at predict time
    feature_names = list(X.columns)
    with open(save_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    logger.info("Feature names saved (%d features)", len(feature_names))

    # ---- 3. Train classification model + calibrator ----
    y_class = y["home_team_wins"].astype(int)
    logger.info("Training calibrated classification model on %d rows...", len(X))
    clf, calibrator = train_calibrated_classifier(X, y_class)
    save_classifier(clf, save_dir / "classifier.json")
    joblib.dump(calibrator, save_dir / "calibrator.joblib")
    logger.info("Calibrator saved to %s", save_dir / "calibrator.joblib")

    # ---- 3b. Post-training probability distribution check ----
    split_test = int(len(X) * 0.80)
    X_test_fold = X.iloc[split_test:]
    dmat_test = xgb.DMatrix(X_test_fold, feature_names=list(X.columns))
    raw_test_probs = clf.predict(dmat_test)
    cal_test_probs = calibrator.predict(raw_test_probs)
    cal_test_probs = np.clip(cal_test_probs, 0.05, 0.95)

    in_range = np.mean((cal_test_probs >= 0.25) & (cal_test_probs <= 0.75))
    logger.info("Calibrated prob distribution on test fold:")
    logger.info(
        "  min=%.3f  median=%.3f  max=%.3f  in [0.25, 0.75]: %.1f%%",
        cal_test_probs.min(), np.median(cal_test_probs), cal_test_probs.max(),
        in_range * 100,
    )
    if in_range < 0.50:
        logger.warning(
            "WARNING: <50%% of calibrated probabilities fall in [0.25, 0.75] — "
            "model may still be overconfident"
        )

    # ---- 4. Train regression models ----
    y_home = y["home_score"].astype(float)
    y_away = y["away_score"].astype(float)
    logger.info("Training regression models...")
    home_reg, away_reg = train_final_regressors(X, y_home, y_away)
    save_regressors(home_reg, away_reg, save_dir)

    logger.info("Training complete. Models saved to %s", save_dir)
    logger.info(
        "Run: uv run python scripts/picks.py --sport %s  — to generate today's picks", sport
    )


if __name__ == "__main__":
    main()
