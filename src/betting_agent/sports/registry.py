"""
Sport registry — maps sport name → loader, feature builder, and constants.

Usage:
    from betting_agent.sports.registry import get_sport_config
    config = get_sport_config("NBA")
    loader = config.loader_cls()
    df = config.build_features(raw_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

import pandas as pd

from betting_agent.sports.base import SportLoader


@dataclass
class SportConfig:
    loader_cls: Type[SportLoader]
    build_features: Callable[[pd.DataFrame], pd.DataFrame]
    split_features_targets: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    sport_key: str
    default_seasons: list[int]
    hist_avg_total: float  # for backtest O/U line generation
    total_stdev: float
    scoring_sigma: float = 14.0  # scoring variance for spread/total normal approx
    star_thresholds: tuple[float, float, float, float] = (0.04, 0.07, 0.12, 0.20)
    # (2-star, 3-star, 4-star, 5-star edge minimums)
    ml_edge_tiers: list[tuple[float, float]] | None = None
    # List of (implied_prob_max, min_edge) for ML guardrails


def _get_registry() -> dict[str, SportConfig]:
    """Build registry with lazy imports to avoid circular dependencies."""
    # NFL
    from betting_agent.sports.nfl.loader import NFLLoader
    from betting_agent.sports.nfl.features import (
        build_nfl_features,
        split_features_targets as nfl_split,
    )

    # NBA
    from betting_agent.sports.nba.loader import NBALoader
    from betting_agent.sports.nba.features import (
        build_nba_features,
        split_features_targets as nba_split,
    )

    # NHL
    from betting_agent.sports.nhl.loader import NHLLoader
    from betting_agent.sports.nhl.features import (
        build_nhl_features,
        split_features_targets as nhl_split,
    )

    # MLB
    from betting_agent.sports.mlb.loader import MLBLoader
    from betting_agent.sports.mlb.features import (
        build_mlb_features,
        split_features_targets as mlb_split,
    )

    return {
        "NFL": SportConfig(
            loader_cls=NFLLoader,
            build_features=build_nfl_features,
            split_features_targets=nfl_split,
            sport_key="americanfootball_nfl",
            default_seasons=list(range(2015, 2025)),
            hist_avg_total=45.0,
            total_stdev=7.0,
            scoring_sigma=14.0,
        ),
        "NBA": SportConfig(
            loader_cls=NBALoader,
            build_features=build_nba_features,
            split_features_targets=nba_split,
            sport_key="basketball_nba",
            default_seasons=list(range(2015, 2025)),
            hist_avg_total=220.0,
            total_stdev=15.0,
            scoring_sigma=11.5,
            star_thresholds=(0.08, 0.15, 0.25, 0.35),
            ml_edge_tiers=[
                (0.12, 0.35),
                (0.20, 0.20),
                (0.30, 0.10),
            ],
        ),
        "NHL": SportConfig(
            loader_cls=NHLLoader,
            build_features=build_nhl_features,
            split_features_targets=nhl_split,
            sport_key="icehockey_nhl",
            default_seasons=list(range(2018, 2026)),
            hist_avg_total=6.0,
            total_stdev=1.5,
            scoring_sigma=1.5,
            star_thresholds=(0.04, 0.07, 0.12, 0.20),
        ),
        "MLB": SportConfig(
            loader_cls=MLBLoader,
            build_features=build_mlb_features,
            split_features_targets=mlb_split,
            sport_key="baseball_mlb",
            default_seasons=list(range(2021, 2026)),
            hist_avg_total=8.5,
            total_stdev=2.5,
            scoring_sigma=2.5,
            star_thresholds=(0.04, 0.07, 0.12, 0.20),
        ),
    }


def get_sport_config(sport: str) -> SportConfig:
    """Get config for a sport. Raises KeyError if sport is unknown."""
    registry = _get_registry()
    key = sport.upper()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown sport '{sport}'. Available: {available}")
    return registry[key]


def available_sports() -> list[str]:
    """Return list of registered sport names."""
    return sorted(_get_registry().keys())
