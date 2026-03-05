"""
Sport registry — maps sport name → loader, feature builder, and constants.

Usage:
    from betting_agent.sports.registry import get_sport_config
    config = get_sport_config("NBA")
    loader = config.loader_cls()
    df = config.build_features(raw_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

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
