"""Market-anchored game model: residual targets, market-only baseline, anchored
features, QB continuity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betting_agent.models.market_anchored import MarketAnchoredModel, residual_targets
from betting_agent.sports.nfl.features import (
    ANCHOR_COLS,
    MARKET_COLS,
    add_market_anchor_features,
    add_qb_continuity,
    build_nfl_features,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.normal(0, 5, n).round()
    total = rng.normal(45, 4, n).round() + 0.5
    margin = spread + rng.normal(0, 13, n)
    tot = total + rng.normal(0, 10, n)
    home = (tot + margin) / 2
    away = (tot - margin) / 2
    X = pd.DataFrame({
        "spread_line": spread, "total_line": total,
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
    })
    return X, pd.Series(home), pd.Series(away)


class TestMarketAnchoredModel:
    def test_none_kind_is_exactly_the_market(self):
        X, h, a = _frame()
        m = MarketAnchoredModel("none").fit(X, h, a)
        p = m.predict(X)
        assert np.allclose(p["pred_margin"], X["spread_line"])
        assert np.allclose(p["pred_total"], X["total_line"])
        assert ((p["win_prob"] > 0.5) == (X["spread_line"] > 0)).all()
        assert 10 < m.margin_sigma < 17 and 7 < m.total_sigma < 13

    @pytest.mark.parametrize("kind", ["ridge", "xgb"])
    def test_learners_stay_close_to_market_on_noise(self, kind):
        # Pure-noise residuals: a well-regularised learner must not wander far.
        X, h, a = _frame()
        m = MarketAnchoredModel(kind).fit(X, h, a)
        p = m.predict(X)
        assert np.abs(p["pred_margin"] - X["spread_line"]).mean() < 3
        assert set(p.columns) >= {"win_prob", "pred_margin", "pred_total",
                                  "home_pred_score", "away_pred_score"}
        assert p["win_prob"].between(0.05, 0.95).all()

    def test_learner_picks_up_a_real_residual_signal(self):
        X, h, a = _frame()
        # Inject a residual the market "missed": f1 moves the margin by 6 points.
        h = h + 3 * X["f1"]
        a = a - 3 * X["f1"]
        m = MarketAnchoredModel("ridge").fit(X, h, a)
        r_m, _ = m.residuals(X)
        assert np.corrcoef(r_m, X["f1"])[0, 1] > 0.8

    def test_rows_without_lines_are_dropped_from_training(self):
        X, h, a = _frame()
        X.loc[:10, "spread_line"] = np.nan
        _, _, ok = residual_targets(X, h, a)
        assert ok.sum() == len(X) - 11
        m = MarketAnchoredModel("none").fit(X, h, a)
        assert m.n_rows == len(X) - 11

    def test_missing_anchor_column_is_an_error(self):
        X, h, a = _frame()
        with pytest.raises(ValueError):
            MarketAnchoredModel("none").fit(X.drop(columns=["total_line"]), h, a)

    def test_save_and_load_round_trip(self, tmp_path):
        X, h, a = _frame()
        m = MarketAnchoredModel("ridge").fit(X, h, a)
        m.save(tmp_path)
        back = MarketAnchoredModel.load(tmp_path)
        assert np.allclose(back.predict(X)["pred_total"], m.predict(X)["pred_total"])
        assert back.get_sigma() == m.get_sigma()


class TestAnchoredFeatures:
    def _raw(self):
        return pd.DataFrame({
            "season": [2024] * 4, "week": [1, 1, 2, 2], "game_type": ["REG"] * 4,
            "gameday": ["2024-09-08", "2024-09-08", "2024-09-15", "2024-09-15"],
            "home_team": ["KC", "BUF", "KC", "BUF"], "away_team": ["BAL", "NYJ", "CIN", "MIA"],
            "home_score": [27, 31, 26, 20], "away_score": [20, 10, 25, 24],
            "spread_line": [3.0, 6.5, 2.5, 3.0], "total_line": [46.0, 47.5, 48.0, 44.5],
            "home_moneyline": [-150, -270, -135, -150], "away_moneyline": [130, 220, 115, 130],
            "home_spread_odds": [-110] * 4, "away_spread_odds": [-110] * 4,
            "over_odds": [-110] * 4, "under_odds": [-110] * 4,
            "home_qb_id": ["mahomes", "allen", "mahomes", "allen"],
            "away_qb_id": ["lamar", "rodgers", "burrow", "tua"],
            "location": ["Home"] * 4, "roof": ["outdoors"] * 4, "surface": ["grass"] * 4,
            "div_game": [0, 1, 0, 1], "temp": [80, 70, 75, 72], "wind": [5, 8, 3, 6],
        })

    def test_default_build_still_leaks_no_market_columns(self):
        feats = build_nfl_features(self._raw())
        assert not [c for c in MARKET_COLS if c in feats.columns]
        assert {"home_qb_change", "away_qb_change"} <= set(feats.columns)

    def test_anchored_build_keeps_lines_but_never_prices(self):
        feats = build_nfl_features(self._raw(), market_anchored=True)
        assert set(ANCHOR_COLS) <= set(feats.columns)
        assert {"market_home_prob", "spread_abs"} <= set(feats.columns)
        prices = set(MARKET_COLS) - set(ANCHOR_COLS)
        assert not (prices & set(feats.columns))
        # vig-removed -150/+130 → ~0.578
        assert feats["market_home_prob"].iloc[0] == pytest.approx(0.578, abs=0.01)

    def test_market_prob_falls_back_to_spread_when_prices_missing(self):
        raw = self._raw()
        raw["home_moneyline"] = np.nan
        raw["away_moneyline"] = np.nan
        out = add_market_anchor_features(raw)
        assert out["market_home_prob"].notna().all()
        assert (out["market_home_prob"] > 0.5).all()      # every home side is favoured here


class TestQbContinuity:
    def test_change_flags_follow_each_team_across_home_and_away(self):
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-09-08", "2024-09-15", "2024-09-22"]),
            "home_team": ["KC", "CIN", "KC"], "away_team": ["BAL", "KC", "ATL"],
            "home_qb_id": ["mahomes", "burrow", "backup"],
            "away_qb_id": ["lamar", "mahomes", "cousins"],
        })
        out = add_qb_continuity(df)
        # KC: mahomes (home) → mahomes (away) → backup (home): change only in game 3.
        assert out["home_qb_change"].tolist() == [0, 0, 1]
        assert out["away_qb_change"].tolist() == [0, 0, 0]

    def test_frame_without_qb_ids_gets_zeros(self):
        out = add_qb_continuity(pd.DataFrame({"home_team": ["KC"], "away_team": ["BAL"]}))
        assert out["home_qb_change"].tolist() == [0] and out["away_qb_change"].tolist() == [0]
