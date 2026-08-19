"""Unit tests for PredictionEngine feature alignment (no model files needed)."""

import logging

import pandas as pd

from betting_agent.models.engine import PredictionEngine


def _engine_with_features(names: list[str]) -> PredictionEngine:
    engine = PredictionEngine(sport="NFL")
    engine._feature_names = names
    engine._loaded = True
    return engine


class TestMissingFeatures:
    def test_reports_absent_columns(self):
        engine = _engine_with_features(["a", "b", "c"])
        X = pd.DataFrame({"a": [1], "c": [3]})
        assert engine.missing_features(X) == ["b"]

    def test_empty_when_fully_covered(self):
        engine = _engine_with_features(["a", "b"])
        X = pd.DataFrame({"b": [2], "a": [1], "extra": [9]})
        assert engine.missing_features(X) == []

    def test_no_feature_names_means_no_claim(self):
        engine = PredictionEngine(sport="NFL")
        assert engine.missing_features(pd.DataFrame({"a": [1]})) == []


class TestAlign:
    def test_reorders_and_drops_extras(self):
        engine = _engine_with_features(["a", "b"])
        aligned = engine._align(pd.DataFrame({"b": [2], "extra": [9], "a": [1]}))
        assert list(aligned.columns) == ["a", "b"]

    def test_warns_when_features_are_zero_filled(self, caplog):
        """The silent zero-fill is what made live NFL picks meaningless."""
        engine = _engine_with_features(["a", "b", "c"])
        with caplog.at_level(logging.WARNING):
            aligned = engine._align(pd.DataFrame({"a": [1]}))
        assert aligned.loc[0, "b"] == 0
        assert "2 of 3 model features missing" in caplog.text

    def test_warns_once_per_distinct_gap(self, caplog):
        engine = _engine_with_features(["a", "b"])
        X = pd.DataFrame({"a": [1]})
        with caplog.at_level(logging.WARNING):
            engine._align(X)
            engine._align(X)
        assert caplog.text.count("model features missing") == 1

    def test_silent_when_nothing_is_missing(self, caplog):
        engine = _engine_with_features(["a", "b"])
        with caplog.at_level(logging.WARNING):
            engine._align(pd.DataFrame({"a": [1], "b": [2]}))
        assert "missing" not in caplog.text


class TestTrainingSeasons:
    def test_empty_without_a_manifest(self):
        engine = PredictionEngine(sport="NFL")
        assert engine.training_seasons == []

    def test_reads_seasons_from_the_manifest(self):
        engine = PredictionEngine(sport="NFL")
        engine._training_meta = {"seasons": [2023, 2024, 2025], "n_rows": 800}
        assert engine.training_seasons == [2023, 2024, 2025]

    def test_coerces_string_seasons(self):
        engine = PredictionEngine(sport="NFL")
        engine._training_meta = {"seasons": ["2024", "2025"]}
        assert engine.training_seasons == [2024, 2025]
