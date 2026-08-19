"""
Tests for the prop pick-selection policy: book-proxy lines, per-player
deduplication, per-market edge floors, and NFL game finalization.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import scripts.props as props_script
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.sports.nfl.props import MIN_QUOTABLE_LINE, book_proxy_line


class TestBookProxyLine:
    def test_receptions_quoted_to_half_point_above_median(self):
        # median 4 → 4.5
        assert book_proxy_line([3, 4, 4, 5, 6], "player_receptions") == 4.5

    def test_yards_snapped_to_half_point(self):
        assert book_proxy_line([40.0, 50.0, 60.0], "player_reception_yds") == 50.5

    def test_uses_only_last_eight_games(self):
        old_high = [20] * 5
        recent_low = [2, 2, 2, 2, 2, 2, 2, 2]
        assert book_proxy_line(old_high + recent_low, "player_receptions") == 2.5

    def test_below_quotable_threshold_is_none(self):
        assert book_proxy_line([0, 0, 1], "player_receptions") is None
        assert book_proxy_line([1.0, 2.0, 3.0], "player_reception_yds") is None

    def test_thresholds_are_the_documented_ones(self):
        assert MIN_QUOTABLE_LINE["player_receptions"] == 1.5
        assert MIN_QUOTABLE_LINE["player_reception_yds"] == 10.0

    def test_empty_history(self):
        assert book_proxy_line([], "player_receptions") is None


def _cand(player: str, market: str, edge: float, event: str = "g1") -> BetCandidate:
    return BetCandidate(
        game_id=0, external_id=event, home_team="KC", away_team="BUF",
        game_date=date(2026, 9, 13), sport="NFL", bet_type="prop",
        pick_side="under", player=player, market=market, line=4.5,
        model_prob=0.5 + edge, implied_prob=0.5, edge=edge, odds=-110,
        kelly_fraction=0.04, recommended_bet=40.0, bankroll_at_pick=1000.0,
    )


class TestDeduplicateByPlayer:
    def test_keeps_only_the_highest_edge_market_per_player(self):
        out = props_script._deduplicate_by_player([
            _cand("Travis Kelce", "player_receptions", 0.12),
            _cand("Travis Kelce", "player_reception_yds", 0.18),
            _cand("Rashee Rice", "player_receptions", 0.11),
        ])
        assert len(out) == 2
        kelce = [c for c in out if c.player == "Travis Kelce"]
        assert len(kelce) == 1
        assert kelce[0].market == "player_reception_yds"

    def test_matches_players_across_name_spellings(self):
        out = props_script._deduplicate_by_player([
            _cand("Amon-Ra St. Brown", "player_receptions", 0.20),
            _cand("Amon Ra St Brown", "player_reception_yds", 0.09),
        ])
        assert len(out) == 1
        assert out[0].edge == pytest.approx(0.20)

    def test_distinct_players_all_survive(self):
        out = props_script._deduplicate_by_player([
            _cand("A Receiver", "player_receptions", 0.11),
            _cand("B Receiver", "player_receptions", 0.12),
        ])
        assert len(out) == 2


class TestEdgeFloors:
    def test_yards_floor_is_stricter_than_receptions(self):
        # Receiving yards ran ~5pp worse per the walk-forward diagnostic.
        assert (props_script.PROP_EDGE_FLOORS["player_reception_yds"]
                > props_script.PROP_EDGE_FLOORS["player_receptions"])

    def test_floors_exceed_measured_overconfidence(self):
        # Residual overconfidence after recalibration is ~4.6-6.0pp; a floor
        # at or below that would systematically bet negative-EV picks.
        assert min(props_script.PROP_EDGE_FLOORS.values()) >= 0.06


class TestGeneratePropCandidates:
    """End-to-end selection: floors, per-player dedup, correlation scaling."""

    def _event(self):
        def outcome(name, player, point, price):
            return {"name": name, "description": player, "point": point, "price": price}

        return {
            "id": "evt-1", "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills", "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{"key": "bet365", "markets": [
                {"key": "player_receptions", "outcomes": [
                    outcome("Over", "Star Guy", 4.5, -110),
                    outcome("Under", "Star Guy", 4.5, -110),
                    outcome("Over", "Other Guy", 3.5, -110),
                    outcome("Under", "Other Guy", 3.5, -110),
                ]},
                {"key": "player_reception_yds", "outcomes": [
                    outcome("Over", "Star Guy", 55.5, -110),
                    outcome("Under", "Star Guy", 55.5, -110),
                ]},
            ]}],
        }

    class _Model:
        """Stub returning a fixed, deliberately lopsided projection."""

        def __init__(self, p_under):
            self._p = p_under
            self.history = pd.DataFrame({
                "player_key": ["star guy", "other guy"], "team": ["KC", "KC"],
            })

        def project(self, player_key, season, week, opponent=None):
            model = self

            class _P:
                mean = 3.0
                games = 8

                def prob_over(self, line):
                    return 1.0 - model._p

                def prob_under(self, line):
                    return model._p

            return _P()

    def _generate(self, models, min_edge=None):
        return props_script.generate_prop_candidates(
            [self._event()], models, bankroll=1000.0,
            min_edge=min_edge, season=2026,
        )

    def test_one_pick_per_player_and_correlation_scaled(self):
        # 0.72 under vs 0.50 fair = 22% edge, clears both floors.
        models = {"player_receptions": self._Model(0.72),
                  "player_reception_yds": self._Model(0.72)}
        out = self._generate(models)
        assert len(out) == 2, "Star Guy must not be bet in both markets"
        assert {c.player for c in out} == {"Star Guy", "Other Guy"}
        # Two picks in one game → Kelly scaled by 1/sqrt(2).
        for c in out:
            assert c.recommended_bet < c.bankroll_at_pick * 0.07
            assert c.kelly_fraction > 0

    def test_yards_floor_rejects_what_receptions_floor_accepts(self):
        # 12% edge: above the receptions floor (10%), below yards (15%).
        models = {"player_receptions": self._Model(0.62),
                  "player_reception_yds": self._Model(0.62)}
        out = self._generate(models)
        assert out, "receptions picks should survive"
        assert all(c.market == "player_receptions" for c in out)

    def test_below_all_floors_yields_nothing(self):
        models = {"player_receptions": self._Model(0.56),
                  "player_reception_yds": self._Model(0.56)}
        assert self._generate(models) == []

    def test_explicit_min_edge_overrides_market_floors(self):
        models = {"player_receptions": self._Model(0.56),
                  "player_reception_yds": self._Model(0.56)}
        assert self._generate(models, min_edge=0.05)


class TestFinalizeNflGames:
    """props.py writes games as 'scheduled'; grading needs them 'final'."""

    def _patch(self, monkeypatch, games, sched):
        import betting_agent.sports.nfl.results as results_mod

        class _Session:
            def query(self, model):
                return self

            def filter(self, *a, **kw):
                return self

            def all(self):
                return games

        class _Ctx:
            def __enter__(self):
                return _Session()

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(results_mod, "get_session", lambda: _Ctx())

        class _FakePolars:
            def to_pandas(self):
                return sched

        class _FakeLoader:
            def load_schedules(self, seasons):
                return _FakePolars()

        import betting_agent.sports.nfl.loader as loader_mod
        import betting_agent.sports.nfl.features as feat_mod
        monkeypatch.setattr(loader_mod, "NFLLoader", _FakeLoader)
        monkeypatch.setattr(feat_mod, "normalise_raw_schedules", lambda df: df)
        return results_mod

    def _sched(self):
        return pd.DataFrame({
            "home_team": ["KC"], "away_team": ["BUF"],
            "game_date": [pd.Timestamp("2025-09-07")],
            "home_score": [27.0], "away_score": [20.0],
        })

    def _game(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            sport="NFL", status="scheduled", game_date=date(2025, 9, 7),
            home_team="KC", away_team="BUF", home_score=None, away_score=None,
        )

    def test_fills_scores_and_marks_final(self, monkeypatch):
        game = self._game()
        mod = self._patch(monkeypatch, [game], self._sched())
        assert mod.finalize_nfl_games() == 1
        assert game.status == "final"
        assert (game.home_score, game.away_score) == (27, 20)

    def test_unplayed_game_is_left_alone(self, monkeypatch):
        game = self._game()
        sched = self._sched()
        sched.loc[0, ["home_score", "away_score"]] = [None, None]
        mod = self._patch(monkeypatch, [game], sched)
        assert mod.finalize_nfl_games() == 0
        assert game.status == "scheduled"

    def test_date_mismatch_beyond_tolerance_does_not_match(self, monkeypatch):
        game = self._game()
        sched = self._sched()
        sched.loc[0, "game_date"] = pd.Timestamp("2025-10-19")
        mod = self._patch(monkeypatch, [game], sched)
        assert mod.finalize_nfl_games() == 0
        assert game.status == "scheduled"

    def test_no_pending_games_skips_the_load(self, monkeypatch):
        mod = self._patch(monkeypatch, [], self._sched())
        assert mod.finalize_nfl_games() == 0
