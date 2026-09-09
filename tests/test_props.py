"""Tests for NFL prop projection models and prop grading."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from betting_agent.accounting.grader import _calculate_pnl, _grade_prop
from betting_agent.db.models import Pick
from betting_agent.sports.nfl.props import (
    DNP,
    ReceivingPropsModel,
    build_receiving_history,
    normalize_player,
)


def _history(n_weeks=12, seasons=(2023, 2024)):
    """Synthetic league: a few pass-catchers with stable rates."""
    rng = np.random.default_rng(7)
    players = [
        ("p1", "Alpha Receiver", "WR", "KC", 7.0, 80.0),
        ("p2", "Beta Tight-End", "TE", "KC", 4.0, 45.0),
        ("p3", "Gamma Receiver", "WR", "BUF", 5.5, 65.0),
        ("p4", "Delta Back", "RB", "BUF", 3.0, 25.0),
    ]
    rows = []
    for season in seasons:
        for week in range(1, n_weeks + 1):
            for pid, name, pos, team, rec_mu, yds_mu in players:
                opp = "BUF" if team == "KC" else "KC"
                rows.append({
                    "player_id": pid, "player_display_name": name,
                    "position": pos, "season": season, "week": week,
                    "season_type": "REG", "team": team, "opponent_team": opp,
                    "receptions": int(rng.poisson(rec_mu)),
                    "targets": int(rng.poisson(rec_mu * 1.5)),
                    "receiving_yards": float(max(0, rng.normal(yds_mu, yds_mu * 0.5))),
                })
    return build_receiving_history(pd.DataFrame(rows))


class TestNormalizePlayer:
    def test_strips_punctuation_and_suffix(self):
        assert normalize_player("Ja'Marr Chase") == "jamarr chase"
        assert normalize_player("Odell Beckham Jr.") == "odell beckham"
        assert normalize_player("Amon-Ra St. Brown") == "amon ra st brown"

    def test_matches_across_conventions(self):
        assert normalize_player("Marvin Harrison Jr") == normalize_player("Marvin Harrison Jr.")

    def test_empty(self):
        assert normalize_player(None) == ""


class TestReceptionsModel:
    def test_projection_centers_near_player_rate(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        proj = model.project("alpha receiver", 2024, 13)
        assert proj is not None
        assert 5.0 < proj.mean < 9.0

    def test_prob_over_under_are_complements_on_half_lines(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        proj = model.project("alpha receiver", 2024, 13)
        assert proj.prob_over(5.5) + proj.prob_under(5.5) == pytest.approx(1.0)

    def test_prob_over_decreases_with_line(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        proj = model.project("alpha receiver", 2024, 13)
        assert proj.prob_over(3.5) > proj.prob_over(6.5) > proj.prob_over(9.5)

    def test_too_few_games_returns_none(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        assert model.project("alpha receiver", 2023, 2) is None

    def test_asof_excludes_current_and_future_weeks(self):
        history = _history()
        model = ReceivingPropsModel("player_receptions").fit(history)
        # Corrupt every game from week 10 on — a leak-free projection for
        # week 10 must not move.
        clean = model.project("alpha receiver", 2024, 10)
        poisoned = history.copy()
        poisoned.loc[
            (poisoned["player_key"] == "alpha receiver") & (poisoned["t"] >= 202410),
            "receptions",
        ] = 99
        model.history = poisoned
        assert model.project("alpha receiver", 2024, 10).mean == pytest.approx(clean.mean)

    def test_unknown_market_rejected(self):
        with pytest.raises(ValueError):
            ReceivingPropsModel("player_pass_tds")


class TestYardsModel:
    def test_projection_positive_and_reasonable(self):
        model = ReceivingPropsModel("player_reception_yds").fit(_history())
        proj = model.project("gamma receiver", 2024, 13)
        assert proj is not None
        assert 35.0 < proj.mean < 95.0
        assert 0.0 < proj.prob_over(proj.mean) < 1.0

    def test_prob_over_complement(self):
        model = ReceivingPropsModel("player_reception_yds").fit(_history())
        proj = model.project("gamma receiver", 2024, 13)
        assert proj.prob_over(60.5) + proj.prob_under(60.5) == pytest.approx(1.0)


class TestDefenseFactor:
    def test_factor_is_clipped(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        f = model._defense_factor("BUF", "WR", 202413)
        assert 0.8 <= f <= 1.2

    def test_unknown_opponent_is_neutral(self):
        model = ReceivingPropsModel("player_receptions").fit(_history())
        assert model._defense_factor("ZZZ", "WR", 202413) == 1.0


class TestGradeProp:
    def _pick(self, side="over", line=5.5, **kw):
        return Pick(pick_side=side, line=line, bet_type="prop",
                    odds=-110, recommended_bet=10.0, **kw)

    def test_over_win_loss(self):
        assert _grade_prop(self._pick("over", 5.5), 7.0) == "win"
        assert _grade_prop(self._pick("over", 5.5), 4.0) == "loss"

    def test_under_win_loss(self):
        assert _grade_prop(self._pick("under", 5.5), 4.0) == "win"
        assert _grade_prop(self._pick("under", 5.5), 7.0) == "loss"

    def test_whole_line_push(self):
        assert _grade_prop(self._pick("over", 5.0), 5.0) == "push"

    def test_unknown_stat_leaves_ungraded(self):
        assert _grade_prop(self._pick("over", 5.5), None) is None

    def test_missing_line_leaves_ungraded(self):
        assert _grade_prop(self._pick("over", None), 6.0) is None

    def test_dnp_voids(self):
        assert _grade_prop(self._pick("over", 5.5), DNP) == "void"

    def test_void_pnl_is_zero(self):
        pick = Pick(recommended_bet=25.0, odds=-110)
        assert _calculate_pnl(pick, "void") == 0.0


class TestStatLookupDnp:
    """The DNP sentinel fires only when the week's stats are published."""

    def _make_lookup(self, monkeypatch):
        import betting_agent.sports.nfl.features as feat_mod
        import betting_agent.sports.nfl.loader as loader_mod
        import betting_agent.sports.nfl.props as props_mod

        stats = pd.DataFrame({
            "player_display_name": ["Teammate Guy"],
            "season": [2025], "week": [1], "team": ["KC"],
            "receptions": [4.0],
        })
        monkeypatch.setattr(props_mod, "load_player_stats", lambda seasons: stats)

        sched = pd.DataFrame({
            "home_team": ["KC", "KC"], "away_team": ["BUF", "BUF"],
            "game_date": [pd.Timestamp("2025-09-07"), pd.Timestamp("2025-09-14")],
            "season": [2025, 2025], "week": [1, 2],
        })

        class _FakePolars:
            def to_pandas(self):
                return sched

        class _FakeLoader:
            def load_schedules(self, seasons):
                return _FakePolars()

        monkeypatch.setattr(loader_mod, "NFLLoader", _FakeLoader)
        monkeypatch.setattr(feat_mod, "normalise_raw_schedules", lambda df: df)
        return props_mod.make_stat_lookup([2025])

    def _game(self, day):
        from types import SimpleNamespace
        return SimpleNamespace(home_team="KC", away_team="BUF", game_date=day)

    def test_player_with_row_returns_stat(self, monkeypatch):
        lookup = self._make_lookup(monkeypatch)
        assert lookup("Teammate Guy", "player_receptions",
                      self._game(date(2025, 9, 7))) == 4.0

    def test_published_week_without_player_is_dnp(self, monkeypatch):
        lookup = self._make_lookup(monkeypatch)
        assert lookup("Missing Player", "player_receptions",
                      self._game(date(2025, 9, 7))) == DNP

    def test_unpublished_week_stays_ungraded(self, monkeypatch):
        lookup = self._make_lookup(monkeypatch)
        # Week 2 is on the schedule but has no stat rows yet — not a DNP.
        assert lookup("Teammate Guy", "player_receptions",
                      self._game(date(2025, 9, 14))) is None


class TestPnlUsesActualBet:
    def test_actual_bet_and_odds_override_recommendation(self):
        pick = Pick(recommended_bet=100.0, odds=-110, actual_bet=50.0, actual_odds=120)
        assert _calculate_pnl(pick, "win") == pytest.approx(60.0)
        assert _calculate_pnl(pick, "loss") == pytest.approx(-50.0)

    def test_falls_back_to_recommendation(self):
        pick = Pick(recommended_bet=100.0, odds=-110)
        assert _calculate_pnl(pick, "win") == pytest.approx(100.0 * 100 / 110)
        assert _calculate_pnl(pick, "push") == 0.0


class TestCalibratorDoesNotExtrapolate:
    """
    The isotonic calibrator is fit on lines near the projection, so a real
    book line far from it produces a raw P(over) outside the fitted range.
    sklearn snaps that to an end bin, and end bins collapse to exactly 0 or 1
    — the 2026 week-1 rehearsal put three receiving-yards picks at 99.0% that
    way (Holani/Tuten under 11.5, Robinson over 11.5).
    """

    @staticmethod
    def _collapsed_calibrator():
        from sklearn.isotonic import IsotonicRegression

        # Fitted range [0.15, 0.70]; the lowest bin is all misses (-> 0.0) and
        # the highest is all hits (-> 1.0), the interior is sensible.
        raw = [0.15, 0.16, 0.30, 0.30, 0.30, 0.50, 0.50, 0.50, 0.69, 0.70]
        hit = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, hit)

    def test_out_of_range_uses_the_raw_tail(self):
        from betting_agent.sports.nfl.props import calibrated_prob

        cal = self._collapsed_calibrator()
        assert calibrated_prob(cal, 0.09) == pytest.approx(0.09)   # below X_min_
        assert calibrated_prob(cal, 0.78) == pytest.approx(0.78)   # above X_max_

    def test_collapsed_end_bin_uses_the_raw_tail(self):
        from betting_agent.sports.nfl.props import calibrated_prob

        cal = self._collapsed_calibrator()
        assert float(cal.predict([0.155])[0]) == 0.0      # in range, but the 0.0 bin
        assert calibrated_prob(cal, 0.155) == pytest.approx(0.155)
        assert float(cal.predict([0.695])[0]) == 1.0
        assert calibrated_prob(cal, 0.695) == pytest.approx(0.695)

    def test_interior_still_calibrates(self):
        from betting_agent.sports.nfl.props import calibrated_prob

        cal = self._collapsed_calibrator()
        assert calibrated_prob(cal, 0.50) == pytest.approx(float(cal.predict([0.50])[0]))
        assert 0.0 < calibrated_prob(cal, 0.50) < 1.0

    def test_projection_no_longer_claims_99_on_a_far_line(self):
        from scipy import stats as sps

        from betting_agent.sports.nfl.props import Projection

        cal = self._collapsed_calibrator()
        # A 30-yard projection against an 11.5 line: raw P(over) ~0.8, which
        # the collapsed calibrator used to turn into 0.99.
        dist = sps.lognorm(0.6, scale=30.0)
        proj = Projection(player="x", market="player_reception_yds", mean=30.0, games=10,
                          _dist=dist, _calibrator=cal)
        p = proj.prob_over(11.5)
        assert p == pytest.approx(float(dist.sf(11.5)))
        assert p < 0.99
