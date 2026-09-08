"""Ladder hits — the card's "best overs" sub-section: rung pricing on the
Over-only alternate boards, the milestone / fair / edge windows, the tail
calibrator, its own paper book (strategy + bankroll) and the Discord card."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from betting_agent.accounting import ledger as ledger_mod
from betting_agent.accounting.grader import _grade_prop
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.notifications.discord import (
    _build_ladder_embed,
    _build_results_embed,
    _ladder_line,
    send_slate_to_discord,
)
from betting_agent.sports.nfl.props import (
    ALTERNATE_MARKETS,
    DEFAULT_LADDER_HOLD,
    LADDER_EDGE_CAPS,
    LADDER_EDGE_FLOORS,
    LADDER_MAX_FAIR_PROB,
    LADDER_MIN_FAIR_PROB,
    LADDER_MIN_RUNG,
    LADDER_RUNGS,
    MARKET_STAT_COLUMNS,
    MODELED_MARKETS,
    Projection,
    ReceivingPropsModel,
    base_market,
    build_rushing_history,
    is_alternate_market,
    ladder_edge_cap,
    ladder_edge_floor,
    ladder_label,
    ladder_min_rung,
)

_SPEC = importlib.util.spec_from_file_location(
    "props_script", Path(__file__).resolve().parents[1] / "scripts" / "props.py"
)
props_script = importlib.util.module_from_spec(_SPEC)
sys.modules["props_script"] = props_script
_SPEC.loader.exec_module(props_script)


# ---------------------------------------------------------------- policy ----

class TestLadderPolicy:
    def test_alternate_markets_grade_off_the_base_stat(self):
        for base, alt in ALTERNATE_MARKETS.items():
            assert MARKET_STAT_COLUMNS[alt] == MARKET_STAT_COLUMNS[base]
            assert base_market(alt) == base and is_alternate_market(alt)
            assert not is_alternate_market(base)

    def test_labels_read_as_milestones(self):
        assert ladder_label("A.J. Brown", "player_reception_yds_alternate", 59.5) == \
            "A.J. Brown 60+ receiving yds"
        assert ladder_label("X", "player_receptions", 5.5) == "X 6+ receptions"
        assert ladder_label("Y", "player_rush_yds_alternate", 39.5) == "Y 40+ rushing yds"

    def test_windows_are_the_documented_ones(self):
        assert LADDER_MIN_FAIR_PROB == 0.20 and LADDER_MAX_FAIR_PROB == 0.65
        assert ladder_min_rung("player_reception_yds_alternate") == 39.5
        assert ladder_min_rung("player_rush_yds") == 39.5
        # Experimental pool (user's call): floors at the main card's MIN_EDGE tier.
        assert ladder_edge_floor("player_rush_yds_alternate") == LADDER_EDGE_FLOORS["player_rush_yds"] == 0.03
        assert ladder_edge_floor("player_reception_yds") == 0.03
        assert ladder_edge_floor("player_receptions") == 0.03
        assert ladder_edge_floor("player_rush_yds", 0.08) == 0.08
        # Rushing turns over above 15% claimed edge; receiving yards do not.
        assert ladder_edge_cap("player_rush_yds") == 0.15 < ladder_edge_cap("player_reception_yds")
        assert set(LADDER_EDGE_CAPS) == set(LADDER_EDGE_FLOORS) == set(LADDER_MIN_RUNG) == set(LADDER_RUNGS)

    def test_all_three_ladders_are_on_by_default(self):
        from betting_agent.config import Settings

        markets = Settings(_env_file=None).ladder_markets.split(",")
        assert set(markets) == {"player_receptions", "player_reception_yds", "player_rush_yds"}

    def test_rushing_model_is_ladder_only(self):
        assert "player_rush_yds" not in MODELED_MARKETS
        ReceivingPropsModel("player_rush_yds")   # but a distribution exists for it


# ------------------------------------------------------ model / calibrator ----

def _synthetic_stats(n_players: int = 40, weeks: int = 17, seasons=(2024, 2025), seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for i in range(n_players):
                pos = "RB" if i % 3 == 0 else ("QB" if i % 7 == 0 else "WR")
                mean = {"RB": 55, "QB": 20, "WR": 3}[pos]
                if i == 0:
                    mean = 95          # the bell-cow: well above the RB position mean
                rows.append({
                    "player_id": f"p{i}", "player_display_name": f"Player {i}", "position": pos,
                    "season": season, "week": week, "season_type": "REG",
                    "team": f"T{i % 8}", "opponent_team": f"T{(i + week) % 8}",
                    "rushing_yards": max(0.0, rng.gamma(2.0, mean / 2.0)),
                    "carries": rng.poisson(mean / 4.5),
                    "receptions": rng.poisson(3), "targets": rng.poisson(5),
                    "receiving_yards": max(0.0, rng.gamma(2.0, 15.0)),
                })
    return pd.DataFrame(rows)


class TestRushingModelAndTail:
    def test_rushing_history_keeps_carriers_and_stat(self):
        hist = build_rushing_history(_synthetic_stats())
        assert {"rushing_yards", "carries", "t", "player_key"} <= set(hist.columns)
        assert set(hist["position"]) == {"RB", "QB", "WR"}

    def test_tail_calibrator_is_fit_and_used_by_prob_hit(self):
        hist = build_rushing_history(_synthetic_stats())
        model = ReceivingPropsModel("player_rush_yds").fit(hist)
        model.tune_dispersion([2024], sample=600)
        assert model.tail_calibrator is not None
        proj = model.project_ladder("player 0", 2025, 10, opponent="T3")
        assert proj is not None and proj._tail_calibrator is model.tail_calibrator
        main = model.project("player 0", 2025, 10, opponent="T3")
        assert main._tail_calibrator is None        # fit on the ladder projection only
        # A rung far above the projection is a real, small probability — the
        # main calibrator would clip it at the lowest value it ever saw.
        assert 0.01 <= proj.prob_hit(proj.mean * 2.5) < proj.prob_hit(proj.mean)
        assert 0.01 <= proj.prob_hit(99.5) <= 0.99

    def test_ladder_projection_shrinks_less_than_the_main_one(self):
        hist = build_rushing_history(_synthetic_stats())
        model = ReceivingPropsModel("player_rush_yds").fit(hist)
        # The bell-cow averages ~95 against an RB position mean near 60, so
        # lighter shrinkage leaves him nearer his own average.
        main = model.project("player 0", 2025, 10)
        ladder = model.project_ladder("player 0", 2025, 10)
        assert ladder.mean > main.mean
        assert model.LADDER_PRIOR_GAMES < model.PRIOR_GAMES

    def test_prob_hit_falls_back_to_main_calibrator(self):
        from scipy import stats as sps

        proj = Projection(player="x", market="player_rush_yds", mean=40.0, games=8,
                          _dist=sps.norm(40, 20))
        assert proj.prob_hit(39.5) == pytest.approx(proj.prob_over(39.5))


# ------------------------------------------------------------ generation ----

def _outcome(name, player, point, price):
    return {"name": name, "description": player, "point": point, "price": price}


def _event(rush_alt=True):
    markets = [
        {"key": "player_reception_yds", "outcomes": [
            _outcome("Over", "Star Guy", 62.5, -113), _outcome("Under", "Star Guy", 62.5, -113),
            _outcome("Over", "Main Under", 40.5, -110), _outcome("Under", "Main Under", 40.5, -110),
            _outcome("Over", "Backup", 24.5, -110), _outcome("Under", "Backup", 24.5, -110),
        ]},
        {"key": "player_reception_yds_alternate", "outcomes": [
            _outcome("Over", "Star Guy", 24.5, -1060),    # fair > 0.65: chalk, skipped
            _outcome("Over", "Star Guy", 59.5, -129),
            _outcome("Over", "Star Guy", 69.5, 118),
            _outcome("Over", "Star Guy", 89.5, 264),
            _outcome("Over", "Star Guy", 149.5, 2300),    # fair < 0.20: long shot, skipped
            _outcome("Over", "Main Under", 59.5, 150),
            _outcome("Over", "Backup", 14.5, -120),       # below the milestone rung
            _outcome("Over", "Backup", 39.5, 260),
        ]},
    ]
    if rush_alt:
        markets.append({"key": "player_rush_yds_alternate", "outcomes": [
            _outcome("Over", "Star Guy", 59.5, 119),
            _outcome("Over", "Back One", 39.5, -279),
            _outcome("Over", "Back One", 59.5, 119),
            _outcome("Over", "Back One", 79.5, 322),
        ]})
    return {
        "id": "evt-1", "home_team": "Seattle Seahawks", "away_team": "New England Patriots",
        "commence_time": "2026-09-10T00:20:00Z",
        "bookmakers": [{"key": "draftkings", "markets": markets}],
    }


class _Model:
    """Stub: a fixed P(hit) per rung, recorded per player."""

    def __init__(self, hits: dict[str, dict[float, float]], mean=50.0):
        self._hits = hits
        self._mean = mean
        self.stat_col = "receiving_yards"
        self.history = pd.DataFrame({
            "player_key": list(hits), "team": ["SEA"] * len(hits), "t": [202601] * len(hits),
            "receiving_yards": [50.0] * len(hits),
        })
        self.calls = []

    def project_ladder(self, player_key, season, week, opponent=None):
        self.calls.append((player_key, opponent))
        table = self._hits.get(player_key)
        if table is None:
            return None
        model = self

        class _P:
            mean = model._mean
            games = 9

            def prob_hit(self, line):
                return table.get(line, 0.01)

        return _P()


def _generate(models, **kw):
    kw.setdefault("markets", list(models))
    return props_script.generate_ladder_candidates(
        [_event()], models, bankroll=100.0, season=2026, book_order=["draftkings"], **kw,
    )


class TestGenerateLadderCandidates:
    def test_best_rung_per_player_with_hold_from_the_main_pair(self):
        # Star Guy's main pair is -113/-113 → hold 6.1%. 69.5 at +118: implied
        # .459 / 1.061 = .432 fair. Model says .58 there → edge +14.8%.
        model = _Model({"star guy": {59.5: 0.60, 69.5: 0.58, 89.5: 0.28}})
        out = _generate({"player_reception_yds": model})
        assert [c.player for c in out] == ["Star Guy"]
        c = out[0]
        assert c.line == 69.5 and c.odds == 118 and c.pick_side == "over"
        assert c.market == "player_reception_yds_alternate"
        assert c.strategy == "ladder" and c.bet_type == "prop"
        assert c.implied_prob == pytest.approx(0.4587 / 1.0603, abs=2e-3)
        assert c.edge == pytest.approx(0.58 - c.implied_prob, abs=1e-6)
        assert c.extra["book_hold"] == pytest.approx(0.061, abs=1e-3)
        assert c.extra["base_market"] == "player_reception_yds"
        assert c.bankroll_at_pick == 100.0 and c.recommended_bet > 0
        assert "ladder_pick" not in c.extra            # no lean tier: everything is a pick

    def test_best_over_per_game_is_a_staked_pick_even_below_the_floor(self):
        # Main Under 59.5 at +150: implied .40 / 1.048 hold = .382 fair; model .40 → +1.8%.
        model = _Model({"star guy": {69.5: 0.44}, "main under": {59.5: 0.40}})   # +0.8% / +1.8%
        out = _generate({"player_reception_yds": model})
        assert len(out) == 1                       # the game's best over only
        best = out[0]
        assert best.player == "Main Under" and 0 < best.edge < 0.03
        assert best.recommended_bet > 0 and best.kelly_fraction > 0 and best.strategy == "ladder"

    def test_further_players_must_clear_the_floor(self):
        model = _Model({"star guy": {69.5: 0.58}, "main under": {59.5: 0.40}})   # +14.8% / +1.8%
        assert [c.player for c in _generate({"player_reception_yds": model})] == ["Star Guy"]
        model = _Model({"star guy": {69.5: 0.58}, "main under": {59.5: 0.42}})   # +14.8% / +3.8%
        assert len(_generate({"player_reception_yds": model})) == 2

    def test_no_edge_means_no_entry(self):
        model = _Model({"star guy": {69.5: 0.40}})                              # -3%
        assert _generate({"player_reception_yds": model}) == []

    def test_main_line_over_is_a_rung_priced_from_its_pair(self):
        model = _Model({"star guy": {62.5: 0.65}})
        out = _generate({"player_reception_yds": model})
        assert len(out) == 1 and out[0].line == 62.5
        assert out[0].market == "player_reception_yds"         # main key, pairwise de-vig
        assert out[0].implied_prob == pytest.approx(0.5, abs=1e-9)

    def test_default_hold_when_no_main_pair(self):
        model = _Model({"back one": {59.5: 0.55}})
        model.stat_col = "rushing_yards"
        out = _generate({"player_rush_yds": model})
        assert len(out) == 1
        c = out[0]
        assert c.implied_prob == pytest.approx((100 / 219) / (1 + DEFAULT_LADDER_HOLD), abs=1e-4)
        assert c.market == "player_rush_yds_alternate"

    def test_fair_window_and_milestone_floor_are_enforced(self):
        # Everything the model loves: only rungs inside the fair window and at
        # or above the milestone survive.
        model = _Model({"star guy": {24.5: 0.99, 59.5: 0.70, 149.5: 0.30},
                        "backup": {14.5: 0.90, 39.5: 0.40}})
        out = _generate({"player_reception_yds": model})
        by_player = {c.player: c for c in out}
        assert by_player["Star Guy"].line == 59.5          # 24.5 chalk, 149.5 long shot
        assert by_player["Backup"].line == 39.5            # 14.5 is not a milestone

    def test_main_line_over_qualifies_below_the_milestone(self):
        # Backup's main line is 24.5 (< 39.5): the book's own number is a real over.
        model = _Model({"backup": {24.5: 0.56}})            # fair .50 → +6%
        out = _generate({"player_reception_yds": model})
        assert len(out) == 1 and out[0].line == 24.5 and out[0].market == "player_reception_yds"
        model = _Model({"backup": {14.5: 0.90}})            # alt rung below the milestone: still out
        assert _generate({"player_reception_yds": model}) == []

    def test_edge_cap_drops_the_model_is_wrong_rungs(self):
        model = _Model({"star guy": {69.5: 0.99}})           # edge ~+56%: capped, no entry at all
        assert _generate({"player_reception_yds": model}) == []
        model = _Model({"star guy": {69.5: 0.58}})
        assert len(_generate({"player_reception_yds": model})) == 1

    def test_main_card_players_are_excluded(self):
        model = _Model({"star guy": {69.5: 0.58}, "main under": {59.5: 0.60}})
        out = _generate({"player_reception_yds": model}, exclude_players={"main under"})
        assert [c.player for c in out] == ["Star Guy"]

    def test_one_rung_per_player_across_markets(self):
        rec = _Model({"star guy": {69.5: 0.56}})             # edge +12.8%
        rush = _Model({"star guy": {59.5: 0.70}})            # fair .435 → edge +27%: capped
        rush2 = _Model({"star guy": {59.5: 0.575}})          # edge +14.0% > receiving's +12.8%
        assert len(_generate({"player_reception_yds": rec, "player_rush_yds": rush})) == 1
        out = _generate({"player_reception_yds": rec, "player_rush_yds": rush2})
        assert len(out) == 1 and out[0].market == "player_rush_yds_alternate"

    def test_same_game_kelly_scaling(self):
        model = _Model({"star guy": {69.5: 0.58}, "main under": {59.5: 0.60}})
        out = _generate({"player_reception_yds": model})
        assert len(out) == 2
        solo = _generate({"player_reception_yds": _Model({"star guy": {69.5: 0.58}})})[0]
        star = next(c for c in out if c.player == "Star Guy")
        assert star.kelly_fraction == pytest.approx(solo.kelly_fraction / np.sqrt(2))

    def test_opponent_resolved_from_roster_overlay(self):
        model = _Model({"star guy": {69.5: 0.58}})
        _generate({"player_reception_yds": model}, current_teams={"star guy": "NE"})
        assert [c for c in model.calls if c[0] == "star guy"] == [("star guy", "SEA")]

    def test_min_edge_override_gates_the_second_player(self):
        model = _Model({"star guy": {69.5: 0.58}, "main under": {59.5: 0.44}})  # +14.8% / +5.8%
        assert len(_generate({"player_reception_yds": model})) == 2
        assert len(_generate({"player_reception_yds": model}, min_edge=0.08)) == 1
        assert _generate({"player_reception_yds": model}, min_edge=0.08)[0].extra["edge_floor"] == 0.08

    def test_ladder_markets_setting_is_honoured(self, monkeypatch):
        monkeypatch.setattr(props_script.settings, "ladder_markets", "player_rush_yds")
        assert props_script.ladder_markets() == ["player_rush_yds"]
        monkeypatch.setattr(props_script.settings, "ladder_markets", "player_pass_yds,bogus")
        assert props_script.ladder_markets() == []


# ------------------------------------------------- grading / accounting ----

class TestLadderAccounting:
    def test_alternate_rung_grades_as_an_over(self):
        pick = SimpleNamespace(id=1, player="Star Guy", line=69.5, pick_side="over",
                               market="player_reception_yds_alternate")
        assert _grade_prop(pick, 71.0) == "win"
        assert _grade_prop(pick, 69.0) == "loss"

    def test_save_key_and_row_carry_the_strategy(self):
        import inspect

        from betting_agent.intelligence import picks as picks_mod

        src = inspect.getsource(picks_mod.save_picks_to_db)
        assert 'getattr(c_or_p, "strategy", None) or ""' in src
        assert "strategy=c.strategy" in src

    def test_ledger_filters_by_strategy_with_its_own_bankroll(self, monkeypatch):
        rows = [
            (SimpleNamespace(id=1, sport="NFL", bet_type="prop", pick_side="under", stake=10.0,
                             price=-110, result="win", pnl=9.09, graded_at=None, clv=None,
                             strategy=None),
             SimpleNamespace(game_date=date(2026, 9, 9))),
            (SimpleNamespace(id=2, sport="NFL", bet_type="prop", pick_side="over", stake=5.0,
                             price=150, result="loss", pnl=-5.0, graded_at=None, clv=None,
                             strategy="ladder"),
             SimpleNamespace(game_date=date(2026, 9, 9))),
        ]
        captured = {}

        class _Q:
            def __init__(self, rows):
                self.rows = rows

            def join(self, *_a, **_k):
                return self

            def filter(self, *clauses):
                captured.setdefault("filters", []).extend(str(c) for c in clauses)
                return self

            def all(self):
                return self.rows

        class _S:
            def query(self, *_a):
                return _Q(rows)

        from contextlib import contextmanager

        @contextmanager
        def fake_session():
            yield _S()

        monkeypatch.setattr(ledger_mod, "get_session", fake_session)
        monkeypatch.setattr(ledger_mod.settings, "starting_bankroll", 100.0)
        monkeypatch.setattr(ledger_mod.settings, "ladder_bankroll", 50.0)
        assert ledger_mod.starting_bankroll_for("ladder") == 50.0
        assert ledger_mod.starting_bankroll_for(None) == 100.0
        summary = ledger_mod.ledger_summary(strategy="ladder")
        assert summary["starting_bankroll"] == 50.0
        assert any("picks.strategy = " in f for f in captured["filters"])
        main = ledger_mod.ledger_summary(exclude_strategy="ladder")
        assert main["starting_bankroll"] == 100.0
        assert any("picks.strategy IS NULL" in f for f in captured["filters"])

    def test_get_summary_accepts_strategy_filters(self, monkeypatch):
        from betting_agent.accounting import roi as roi_mod

        seen = []

        class _Q:
            def join(self, *_a):
                return self

            def filter(self, *clauses):
                seen.extend(str(c) for c in clauses)
                return self

            def all(self):
                return []

        class _S:
            def query(self, *_a):
                return _Q()

        from contextlib import contextmanager

        @contextmanager
        def fake_session():
            yield _S()

        monkeypatch.setattr(roi_mod, "get_session", fake_session)
        roi_mod.get_summary(strategy="ladder")
        assert any("picks.strategy = " in s for s in seen)
        seen.clear()
        roi_mod.get_summary(exclude_strategy="ladder")
        assert any("picks.strategy IS NULL" in s for s in seen)


# ---------------------------------------------------------------- discord ----

def _ladder_pick(**kw):
    base = dict(game_id=0, external_id="evt-1", home_team="Seattle Seahawks",
                away_team="New England Patriots", game_date=date(2026, 9, 8), sport="NFL",
                bet_type="prop", pick_side="over", player="A.J. Brown",
                market="player_reception_yds_alternate", line=69.5, model_prob=0.55,
                implied_prob=0.43, edge=0.12, odds=118, kelly_fraction=0.02,
                recommended_bet=2.0, bankroll_at_pick=100.0, strategy="ladder",
                extra={"bookmaker": "draftkings", "projection_mean": 66.4})
    base.update(kw)
    return BetCandidate(**base)


class TestLadderDiscord:
    def test_ladder_embed_names_the_milestone_and_its_book(self):
        embed = _build_ladder_embed(_ladder_pick(), 1)
        assert embed["title"] == "LADDER #1  A.J. Brown 70+ receiving yds"
        assert "+118" in embed["description"] and "ladder bankroll" in embed["description"]
        assert "55.0%" in embed["description"] and "43.0%" in embed["description"]
        assert "$2.00" in embed["description"] and "LEAN" not in embed["description"]

    def test_slate_card_puts_ladder_after_props_under_its_own_header(self, monkeypatch):
        sent = []
        monkeypatch.setattr("betting_agent.notifications.discord._get_webhook_url",
                            lambda *_a: "http://hook")
        monkeypatch.setattr("betting_agent.notifications.discord._send_webhook",
                            lambda url, payload: sent.append(payload) or True)
        prop = _ladder_pick(strategy=None, market="player_receptions", pick_side="under",
                            line=4.5, player="Someone Else")
        assert send_slate_to_discord("Card", [], [prop], 100.0, ladder=[_ladder_pick()],
                                     ladder_saved=2, ladder_bankroll=100.0)
        titles = [e.get("title", "") for e in sent[0]["embeds"]]
        assert titles[1].startswith("#1") and titles[2].startswith("LADDER HITS")
        assert titles[3].startswith("LADDER #1")
        header = sent[0]["embeds"][2]["description"]
        assert "$100.00" in header and "2 more ladder pick(s) saved off-card" in header

    def test_ladder_only_card_still_posts(self, monkeypatch):
        sent = []
        monkeypatch.setattr("betting_agent.notifications.discord._get_webhook_url",
                            lambda *_a: "http://hook")
        monkeypatch.setattr("betting_agent.notifications.discord._send_webhook",
                            lambda url, payload: sent.append(payload) or True)
        assert send_slate_to_discord("Card", [], [], 100.0, ladder=[_ladder_pick()])
        assert sent and any(e.get("title", "").startswith("LADDER #1") for e in sent[0]["embeds"])

    def test_results_line_reports_the_ladder_as_its_own_book(self):
        summary = {"total_bets": 3, "wins": 1, "losses": 2, "pushes": 0, "total_pnl": -1.5,
                   "roi_pct": -25.0, "win_rate_pct": 33.3, "avg_fair_pct": 41.0}
        line = _ladder_line(summary, bankroll=100.0, label="Ladder hits all-time")
        assert line.startswith("**Ladder hits all-time (own bankroll):** 1-2-0")
        assert "-$1.50" in line and "$100.00 → $98.50" in line and "vs fair 41.0%" in line
        assert _ladder_line({"message": "none"}) is None
        embed = _build_results_embed(
            {"total_bets": 1, "wins": 1, "losses": 0, "pushes": 0, "win_rate_pct": 100.0,
             "total_pnl": 5.0, "roi_pct": 50.0, "avg_edge_pct": 10.0},
            "NFL", date(2026, 9, 10), ladder_summary=summary,
            pick_details=[{"result": "win", "pnl": 2.36, "odds": 118, "bet_type": "prop",
                           "player": "A.J. Brown", "market": "player_reception_yds_alternate",
                           "line": 69.5, "pick_side": "over", "strategy": "ladder",
                           "home_team": "SEA", "away_team": "NE"}],
        )
        assert "**Ladder hits (own bankroll):**" in embed["description"]
        assert "LADDER A.J. Brown 70+ receiving yds (+118)" in embed["description"]
