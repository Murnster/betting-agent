"""
Abstract sport interface.
Each sport implements these methods so the pipeline scaffolding stays generic.
"""

from abc import ABC, abstractmethod

import polars as pl


class SportLoader(ABC):
    """Load raw game/schedule data for a specific sport."""

    @abstractmethod
    def load_schedules(self, seasons: list[int]) -> pl.DataFrame:
        """Return one row per game with at minimum: game_date, season, week,
        home_team, away_team, home_score, away_score, is_playoff, external_id."""
        ...

    @abstractmethod
    def load_injuries(self, seasons: list[int]) -> pl.DataFrame:
        """Return injury report data keyed by team and date."""
        ...

    @abstractmethod
    def sport_key(self) -> str:
        """Return the Odds API sport key (e.g. 'americanfootball_nfl')."""
        ...


class SportFeatureBuilder(ABC):
    """Build a model-ready feature matrix for a specific sport."""

    @abstractmethod
    def build_features(self, schedules: pl.DataFrame) -> pl.DataFrame:
        """Return a feature matrix including Elo, rolling stats, H2H, rest days."""
        ...
