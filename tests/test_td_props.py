"""Anytime-TD props: market-implied touchdowns, board de-vig, the model,
candidate generation (edge window, per-game cap, shadow), grading and
closing-quote support."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.props as props_script
from betting_agent.accounting.grader import _grade_prop
from betting_agent.accounting.prop_clv import _closing_quote
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.notifications.discord import _build_td_embed
from betting_agent.sports.nfl.props import MARKET_STAT_COLUMNS, edge_cap, edge_floor, make_stat_lookup
from betting_agent.sports.nfl.td_props import (
    TD_MARKET,
    TD_STAT_COL,
    TouchdownPropsModel,
    add_anytime_td_column,
    build_td_history,
    expected_game_tds,
    expected_team_tds,
    fair_yes_probabilities,
    team_expected_tds_lookup,
    yes_outcomes,
)


# ---- data helpers ----

def _stats(rows):
    cols = ["player_id", "player_display_name", "position", "season", "week", "season_type",
            "team", "opponent_team", "carries", "targets", "rushing_tds", "receiving_tds",
            "special_teams_tds", "fumble_recovery_tds"]
    return pd.DataFrame(rows, columns=cols)


def _season(player_id, name, pos, team, opp, tds_per_week, season=2025):
    return [(player_id, name, pos, season, w, "REG", team, opp, 10, 5, td, 0, 0, 0)
            for w, td in enumerate(tds_per_week, start=1)]


def _history():
    rows = []
    rows += _season("p1", "Scorer One", "RB", "KC", "BUF", [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    rows += _season("p2", "Quiet Two", "WR", "KC", "BUF", [0] * 12)
    rows += _season("p3", "Bills Back", "RB", "BUF", "KC", [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    rows += _season("p4", "Bills Tight", "TE", "BUF", "KC", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return build_td_history(_stats(rows))


class TestMarketImpliedTouchdowns:
    def test_expected_team_tds_is_linear_in_points_with_floor(self):
        assert expected_team_tds(24.0) == pytest.approx(-0.671 + 0.1407 * 24.0)
        assert expected_team_tds(0.0) == 0.3        # floored, never negative
        assert expected_team_tds(None) == pytest.approx(2.38)

    def test_expected_game_tds_splits_total_by_spread(self):
        # nflreadpy convention: positive spread_line = home favourite.
        home, away = expected_game_tds(7.0, 47.0)
        assert home == pytest.approx(expected_team_tds(27.0))
        assert away == pytest.approx(expected_team_tds(20.0))
        assert expected_game_tds(None, None) == (2.38, 2.38)
        # Total without a spread: even split.
        h, a = expected_game_tds(None, 44.0)
        assert h == a == pytest.approx(expected_team_tds(22.0))

    def test_lookup_from_schedule(self):
        sched = pd.DataFrame({
            "season": [2025], "week": [3], "home_team": ["KC"], "away_team": ["BUF"],
            "spread_line": [3.0], "total_line": [50.0],
        })
        look = team_expected_tds_lookup(sched)
        assert look[(2025, 3, "KC")] == pytest.approx(expected_team_tds(26.5))
        assert look[(2025, 3, "BUF")] == pytest.approx(expected_team_tds(23.5))
        assert team_expected_tds_lookup(None) == {}
        assert team_expected_tds_lookup(pd.DataFrame({"season": [1]})) == {}


class TestBoardDevig:
    def test_yes_outcomes_ignores_other_names(self):
        m = {"outcomes": [{"name": "Yes", "description": "A", "price": 120},
                          {"name": "No", "description": "A", "price": -150},
                          {"name": "Yes", "description": "B"}]}
        assert set(yes_outcomes(m)) == {"A"}

    def test_fair_board_sums_to_expected_tds_and_reports_hold(self):
        prices = {"A": -110, "B": 150, "C": 300, "D": 600}
        fair, hold = fair_yes_probabilities(prices, expected_tds=1.5)
        lam = {k: -np.log1p(-p) for k, p in fair.items()}
        assert sum(lam.values()) == pytest.approx(1.5)
        # The board sells more probability than it is worth.
        assert hold > 0
        assert all(0 < v < 1 for v in fair.values())
        # Ordering preserved: shortest price stays the likeliest scorer.
        assert fair["A"] > fair["B"] > fair["C"] > fair["D"]

    def test_hold_is_zero_when_board_is_already_fair(self):
        lam = {"A": 0.6, "B": 0.4}
        p = {k: 1 - np.exp(-v) for k, v in lam.items()}
        prices = {k: int(round(100 * (1 - v) / v)) for k, v in p.items()}
        fair, hold = fair_yes_probabilities(prices, expected_tds=1.0)
        assert abs(hold) < 0.01
        assert fair["A"] == pytest.approx(p["A"], abs=0.005)

    def test_empty_board(self):
        assert fair_yes_probabilities({}, 4.0) == ({}, 0.0)


class TestHistoryAndModel:
    def test_history_derives_anytime_tds_and_keeps_skill_positions(self):
        stats = _stats(_season("p1", "A. Player", "RB", "KC", "BUF", [1, 0])
                       + [("ol", "Big Guard", "G", 2025, 1, "REG", "KC", "BUF", 0, 0, 0, 1, 0, 0)])
        stats.loc[0, "receiving_tds"] = 1
        hist = build_td_history(stats)
        assert set(hist["position"]) == {"RB"}
        assert hist[TD_STAT_COL].tolist() == [2.0, 0.0]
        assert hist["player_key"].iloc[0] == "a player"
        assert hist["t"].tolist() == [202501, 202502]

    def test_add_column_tolerates_missing_td_columns(self):
        df = pd.DataFrame({"rushing_tds": [1, 0], "receiving_tds": [0, 2]})
        assert add_anytime_td_column(df)[TD_STAT_COL].tolist() == [1, 2]

    def test_project_uses_prior_games_only_and_ranks_scorers(self):
        model = TouchdownPropsModel().fit(_history())
        hot = model.project("scorer one", 2025, 13, opponent="BUF")
        cold = model.project("quiet two", 2025, 13, opponent="BUF")
        assert hot is not None and cold is not None
        assert hot.games == 12 and hot.prob > cold.prob
        assert 0.01 <= cold.prob <= 0.95
        # Week 5 projection sees only weeks 1-4.
        early = model.project("scorer one", 2025, 5)
        assert early.games == 4
        # Too little history → None.
        assert model.project("scorer one", 2025, 3) is None
        assert model.project("nobody", 2025, 13) is None

    def test_scoring_environment_moves_the_rate(self):
        model = TouchdownPropsModel().fit(_history())
        base = model.project("scorer one", 2025, 13, team="KC")
        boosted = model.project("scorer one", 2025, 13, team="KC", team_expected_tds=5.0)
        assert boosted.rate > base.rate
        # Clipped: no more than ENV_CLIP[1] times the base.
        assert boosted.rate <= base.rate * TouchdownPropsModel.ENV_CLIP[1] + 1e-9

    def test_calibrate_fits_isotonic_when_enough_rows(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(40):
            rate = rng.uniform(0.05, 0.9)
            tds = rng.poisson(rate, size=17).tolist()
            rows += _season(f"p{i}", f"Player {i}", "RB", "KC", "BUF", tds, season=2024)
            rows += _season(f"p{i}", f"Player {i}", "RB", "KC", "BUF",
                            rng.poisson(rate, size=17).tolist(), season=2025)
        model = TouchdownPropsModel().fit(build_td_history(_stats(rows)))
        model.calibrate([2025])
        assert model.calibrator is not None
        proj = model.project("player 1", 2025, 17)
        assert proj is not None and 0.01 <= proj.prob <= 0.95

    def test_extend_history_reindexes(self):
        hist = _history()
        model = TouchdownPropsModel().fit(hist[hist["week"] <= 8])
        assert model.project("scorer one", 2025, 13).games == 8
        model.extend_history(hist[hist["week"] > 8])
        assert model.project("scorer one", 2025, 13).games == 12


# ---- candidate generation ----

class _StubModel:
    """project() returns fixed probabilities; _players supplies the team."""

    def __init__(self, probs: dict[str, float], teams: dict[str, str]):
        self.probs = probs
        self._players = {pk: ([1], np.array([0.0]), np.array([5.0]), [team], "RB")
                         for pk, team in teams.items()}
        self.calls = []

    def project(self, pk, season, week, opponent=None, team=None, team_expected_tds=None):
        self.calls.append((pk, opponent, team, team_expected_tds))
        p = self.probs.get(pk)
        return None if p is None else SimpleNamespace(prob=p, rate=-np.log1p(-p), games=10)


def _event(prices: dict[str, int], book="draftkings", eid="evt1"):
    return {
        "id": eid, "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
        "commence_time": "2026-09-13T17:00:00Z",
        "bookmakers": [{"key": book, "markets": [
            {"key": "player_receptions", "outcomes": []},
            {"key": TD_MARKET, "outcomes": [
                {"name": "Yes", "description": pl, "price": pr} for pl, pr in prices.items()]},
        ]}],
    }


class TestGenerateTdCandidates:
    PRICES = {"Travis Kelce": 100, "Isiah Pacheco": 120, "Khalil Shakir": 300, "Dawson Knox": 400,
              "Nobody Known": 800}
    TEAMS = {"travis kelce": "KC", "isiah pacheco": "KC", "khalil shakir": "BUF",
             "dawson knox": "BUF"}

    def _fair(self):
        from betting_agent.sports.nfl.td_props import BOARD_COVERAGE
        fair, _ = fair_yes_probabilities(self.PRICES, BOARD_COVERAGE * 2 * 2.38)
        return fair

    def test_best_scorer_is_always_returned_and_window_marks_picks(self):
        fair = self._fair()
        floor, cap = edge_floor(TD_MARKET), edge_cap(TD_MARKET)
        probs = {
            "travis kelce": fair["Travis Kelce"] + floor + 0.01,      # in the window → PICK
            "isiah pacheco": fair["Isiah Pacheco"] + cap + 0.05,      # above the cap → dropped
            "khalil shakir": fair["Khalil Shakir"] + floor + 0.03,    # in the window, best → PICK
            "dawson knox": fair["Dawson Knox"] + floor - 0.01,        # below the floor
        }
        model = _StubModel(probs, self.TEAMS)
        out = props_script.generate_td_candidates(
            [_event(self.PRICES)], model, 1000.0, 2026, current_teams=self.TEAMS,
            book_order=["bet365", "draftkings"],
        )
        # per_game=2: the two picks; Knox (below floor) only if he were the best.
        assert [c.player for c in out] == ["Khalil Shakir", "Travis Kelce"]
        c = out[0]
        assert (c.bet_type, c.pick_side, c.market, c.line) == ("prop", "yes", TD_MARKET, 0.5)
        assert c.odds == 300 and c.implied_prob == pytest.approx(fair["Khalil Shakir"])
        assert c.extra["td_pick"] is True and c.recommended_bet > 0
        assert c.extra["bookmaker"] == "draftkings" and c.extra["team"] == "BUF"
        assert isinstance(c.extra["board_hold"], float) and c.extra["expected_game_tds"] > 0
        assert ("khalil shakir", "KC", "BUF", pytest.approx(2.38)) in model.calls
        assert not any(call[0] == "nobody known" for call in model.calls)

    def test_below_floor_best_scorer_becomes_a_zero_stake_lean(self):
        fair = self._fair()
        floor = edge_floor(TD_MARKET)
        probs = {"travis kelce": fair["Travis Kelce"] + floor - 0.02,
                 "khalil shakir": fair["Khalil Shakir"] + floor - 0.05}
        out = props_script.generate_td_candidates(
            [_event(self.PRICES)], _StubModel(probs, self.TEAMS), 1000.0, 2026,
            current_teams=self.TEAMS)
        assert [c.player for c in out] == ["Travis Kelce"]         # best edge only
        assert out[0].extra["td_pick"] is False
        assert out[0].recommended_bet == 0.0 and out[0].kelly_fraction == 0.0
        assert out[0].edge < floor

    def test_long_shots_never_lead_the_card(self):
        fair = self._fair()
        # "Nobody Known" at +800 is priced under MIN_FAIR_PROB; give it a team
        # and a huge model edge — it must still be ignored.
        teams = dict(self.TEAMS, **{"nobody known": "KC"})
        probs = {"nobody known": 0.5, "travis kelce": fair["Travis Kelce"] + 0.01}
        out = props_script.generate_td_candidates(
            [_event(self.PRICES)], _StubModel(probs, teams), 1000.0, 2026, current_teams=teams)
        assert [c.player for c in out] == ["Travis Kelce"]

    def test_per_game_cap_keeps_best_edges(self):
        fair = self._fair()
        floor = edge_floor(TD_MARKET)
        probs = {pk: fair[name] + floor + 0.005 * i for i, (pk, name) in enumerate(
            [("travis kelce", "Travis Kelce"), ("isiah pacheco", "Isiah Pacheco"),
             ("khalil shakir", "Khalil Shakir"), ("dawson knox", "Dawson Knox")])}
        out = props_script.generate_td_candidates(
            [_event(self.PRICES)], _StubModel(probs, self.TEAMS), 1000.0, 2026,
            current_teams=self.TEAMS, per_game=1)
        assert [c.player for c in out] == ["Dawson Knox"]
        assert out[0].extra["td_pick"] is True

    def test_same_game_kelly_scaling_and_injury_drop(self, monkeypatch):
        fair = self._fair()
        floor = edge_floor(TD_MARKET)
        probs = {"travis kelce": fair["Travis Kelce"] + floor + 0.01,
                 "khalil shakir": fair["Khalil Shakir"] + floor + 0.01}
        solo = props_script.generate_td_candidates(
            [_event(self.PRICES)], _StubModel({"travis kelce": probs["travis kelce"]}, self.TEAMS),
            1000.0, 2026, current_teams=self.TEAMS)
        model = _StubModel(probs, self.TEAMS)
        both = props_script.generate_td_candidates(
            [_event(self.PRICES)], model, 1000.0, 2026, current_teams=self.TEAMS)
        kelce_both = next(c for c in both if c.player == "Travis Kelce")
        assert len(both) == 2 and kelce_both.kelly_fraction < solo[0].kelly_fraction
        injuries = pd.DataFrame(
            [("Travis Kelce", "KC", "TE", "Out", None, "1")],
            columns=["full_name", "team", "position", "report_status", "practice_status", "gsis_id"],
        )
        kept = props_script.generate_td_candidates(
            [_event(self.PRICES)], model, 1000.0, 2026, current_teams=self.TEAMS,
            injuries=injuries, qb1={})
        assert [c.player for c in kept] == ["Khalil Shakir"]

    def test_game_without_a_td_board_is_skipped(self):
        event = _event({})
        event["bookmakers"][0]["markets"] = [{"key": "player_receptions",
                                             "outcomes": [{"name": "Over", "description": "X",
                                                           "point": 3.5, "price": -110}]}]
        out = props_script.generate_td_candidates([event], _StubModel({}, {}), 1000.0, 2026)
        assert out == []

    def test_min_edge_override_replaces_the_floor(self):
        fair = self._fair()
        probs = {"travis kelce": fair["Travis Kelce"] + 0.02}
        out = props_script.generate_td_candidates(
            [_event(self.PRICES)], _StubModel(probs, self.TEAMS), 1000.0, 2026,
            current_teams=self.TEAMS, min_edge=0.01)
        assert [c.player for c in out] == ["Travis Kelce"]


# ---- grading, closing quote, Discord ----

def _pick(side, line=0.5):
    return SimpleNamespace(id=1, pick_side=side, line=line, player="Travis Kelce", market=TD_MARKET)


class TestGradingAndClosing:
    def test_yes_grades_as_over_half(self):
        assert _grade_prop(_pick("yes"), 1.0) == "win"
        assert _grade_prop(_pick("yes"), 2.0) == "win"
        assert _grade_prop(_pick("yes"), 0.0) == "loss"
        assert _grade_prop(_pick("no"), 0.0) == "win"
        assert _grade_prop(_pick("yes"), "DNP") == "void"
        assert _grade_prop(_pick("yes"), None) is None

    def test_market_maps_to_derived_stat_column(self):
        assert MARKET_STAT_COLUMNS[TD_MARKET] == TD_STAT_COL

    def test_stat_lookup_sums_td_columns(self, monkeypatch):
        stats = _stats(_season("p1", "Travis Kelce", "TE", "KC", "BUF", [0]))
        stats.loc[0, "receiving_tds"] = 1
        stats.loc[0, "rushing_tds"] = 1
        stats["receptions"] = 5
        sched = pd.DataFrame({"season": [2025], "week": [1], "home_team": ["KC"],
                              "away_team": ["BUF"], "game_date": [pd.Timestamp("2025-09-07")]})
        monkeypatch.setattr("betting_agent.sports.nfl.props.load_player_stats", lambda s: stats)
        monkeypatch.setattr("betting_agent.sports.nfl.props.load_season_schedule", lambda s: sched)
        lookup = make_stat_lookup([2025])
        game = SimpleNamespace(game_date=date(2025, 9, 7), home_team="KC", away_team="BUF")
        assert lookup("Travis Kelce", TD_MARKET, game) == 2.0
        assert lookup("Travis Kelce", "player_receptions", game) == 5.0   # untouched

    def test_closing_quote_reads_the_yes_price(self):
        event = _event({"Travis Kelce": 115, "Isiah Pacheco": 130})
        quote = _closing_quote(event, "travis kelce", TD_MARKET, "yes", 0.5, ["draftkings"])
        assert quote == (0.5, 115)
        assert _closing_quote(event, "unknown guy", TD_MARKET, "yes", 0.5, ["draftkings"]) is None


class TestTdEmbed:
    def _cand(self, td_pick: bool):
        return BetCandidate(
            game_id=0, external_id="evt1", home_team="KC", away_team="BUF",
            game_date=date(2026, 9, 13), sport="NFL", bet_type="prop", pick_side="yes",
            player="Travis Kelce", market=TD_MARKET, line=0.5, model_prob=0.41,
            implied_prob=0.33, edge=0.08, odds=150, kelly_fraction=0.02 if td_pick else 0.0,
            recommended_bet=20.0 if td_pick else 0.0, bankroll_at_pick=1000.0,
            extra={"bookmaker": "draftkings", "board_hold": 0.27, "td_pick": td_pick,
                   "flags": [{"detail": "Travis Kelce listed Questionable (knee)"}]},
        )

    def test_pick_embed(self):
        embed = _build_td_embed(self._cand(True), 1)
        assert embed["title"] == "TD SCORER #1  Travis Kelce anytime TD (PICK)"
        d = embed["description"]
        assert "+150" in d and "hold 27%" in d and "$20.00" in d and "Questionable" in d

    def test_lean_embed_is_zero_stake(self):
        embed = _build_td_embed(self._cand(False), 2)
        assert embed["title"].endswith("(LEAN)")
        assert "Stake 0" in embed["description"] and "$" not in embed["description"]
