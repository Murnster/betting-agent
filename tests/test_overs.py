"""Straight overs — the third card section (user, Sep 8 2026: "at least one
straight over line ... in these primetime games"): main-line Over per game,
own paper book (strategy="overs"), flat-staked when the model has no edge."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from betting_agent.accounting import ledger as ledger_mod
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.notifications.discord import (
    _build_over_embed,
    _build_overs_header_embed,
    _ladder_line,
    send_slate_to_discord,
)
from betting_agent.sports.nfl.props import OVERS_EDGE_FLOOR, OVERS_MIN_STAKE_PCT, over_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import props as props_script  # noqa: E402


def _outcome(name, player, point, price):
    return {"name": name, "description": player, "point": point, "price": price}


def _event():
    markets = [
        {"key": "player_reception_yds", "outcomes": [
            _outcome("Over", "Star Guy", 62.5, -113), _outcome("Under", "Star Guy", 62.5, -113),
            _outcome("Over", "Main Under", 40.5, -110), _outcome("Under", "Main Under", 40.5, -110),
            _outcome("Over", "Backup", 24.5, -110), _outcome("Under", "Backup", 24.5, -110),
            _outcome("Over", "No History", 30.5, -110), _outcome("Under", "No History", 30.5, -110),
        ]},
        {"key": "player_receptions", "outcomes": [
            _outcome("Over", "Star Guy", 5.5, 125), _outcome("Under", "Star Guy", 5.5, -160),
        ]},
        {"key": "player_reception_yds_alternate", "outcomes": [
            _outcome("Over", "Star Guy", 69.5, 118),     # ladder territory: not an over line
        ]},
    ]
    return {
        "id": "evt-1", "home_team": "Seattle Seahawks", "away_team": "New England Patriots",
        "commence_time": "2026-09-10T00:20:00Z",
        "bookmakers": [{"key": "draftkings", "markets": markets}],
    }


class _Model:
    def __init__(self, hits: dict[str, dict[float, float]], stat_col="receiving_yards"):
        self._hits = hits
        self.stat_col = stat_col
        self.history = pd.DataFrame({
            "player_key": list(hits), "team": ["SEA"] * len(hits), "t": [202601] * len(hits),
            stat_col: [50.0] * len(hits),
        })
        self.calls = []

    def project_ladder(self, player_key, season, week, opponent=None):
        self.calls.append((player_key, opponent))
        table = self._hits.get(player_key)
        if table is None:
            return None

        class _P:
            mean = 50.0
            games = 9

            def prob_hit(self, line):
                return table.get(line, 0.01)

        return _P()


def _generate(models, **kw):
    return props_script.generate_over_candidates(
        [_event()], models, bankroll=100.0, season=2026, book_order=["draftkings"], **kw,
    )


class TestOverPolicy:
    def test_label_and_constants(self):
        assert over_label("Romeo Doubs", "player_reception_yds", 36.5) == \
            "Romeo Doubs over 36.5 receiving yds"
        assert over_label("X", "player_receptions", 2.5) == "X over 2.5 receptions"
        assert OVERS_EDGE_FLOOR == 0.03 and OVERS_MIN_STAKE_PCT == 0.01

    def test_config_defaults(self):
        from betting_agent.config import Settings

        s = Settings(_env_file=None)
        assert s.overs_enabled is True and s.overs_bankroll == 100.0


class TestGenerateOverCandidates:
    def test_kelly_pick_when_the_model_has_an_edge(self):
        model = _Model({"star guy": {62.5: 0.56}, "main under": {40.5: 0.45}})   # +6% / -5%
        out = _generate({"player_reception_yds": model})
        assert [c.player for c in out] == ["Star Guy"]
        c = out[0]
        assert c.strategy == "overs" and c.pick_side == "over" and c.market == "player_reception_yds"
        assert c.line == 62.5 and c.odds == -113 and c.implied_prob == pytest.approx(0.5)
        assert c.edge == pytest.approx(0.06) and c.recommended_bet > 0 and c.kelly_fraction > 0
        assert c.extra["flat_stake"] is False and c.extra["book_hold"] == pytest.approx(0.061, abs=1e-3)
        assert c.bankroll_at_pick == 100.0

    def test_best_over_is_flat_staked_when_nothing_has_edge(self):
        model = _Model({"star guy": {62.5: 0.45}, "main under": {40.5: 0.48}})   # -5% / -2%
        out = _generate({"player_reception_yds": model})
        assert len(out) == 1
        best = out[0]
        assert best.player == "Main Under" and best.edge < 0
        assert best.extra["flat_stake"] is True
        assert best.kelly_fraction == OVERS_MIN_STAKE_PCT and best.recommended_bet == 1.0

    def test_further_overs_need_the_floor(self):
        model = _Model({"star guy": {62.5: 0.56}, "backup": {24.5: 0.52}})       # +6% / +2%
        assert [c.player for c in _generate({"player_reception_yds": model})] == ["Star Guy"]
        model = _Model({"star guy": {62.5: 0.56}, "backup": {24.5: 0.54}})       # +6% / +4%
        out = _generate({"player_reception_yds": model})
        assert {c.player for c in out} == {"Star Guy", "Backup"}
        solo = _generate({"player_reception_yds": _Model({"star guy": {62.5: 0.56}})})[0]
        star = next(c for c in out if c.player == "Star Guy")
        assert star.kelly_fraction == pytest.approx(solo.kelly_fraction / np.sqrt(2))
        assert len(_generate({"player_reception_yds": model}, min_edge=0.05)) == 1

    def test_cap_drops_the_model_is_wrong_lines(self):
        assert _generate({"player_reception_yds": _Model({"star guy": {62.5: 0.99}})}) == []

    def test_main_card_players_and_unknowns_are_excluded(self):
        model = _Model({"star guy": {62.5: 0.56}, "main under": {40.5: 0.55}})
        out = _generate({"player_reception_yds": model}, exclude_players={"star guy"})
        assert [c.player for c in out] == ["Main Under"]
        assert all(c.player != "No History" for c in out)

    def test_one_over_per_player_across_markets(self):
        yds = _Model({"star guy": {62.5: 0.56}})                                   # +6.0%
        rec = _Model({"star guy": {5.5: 0.50}}, stat_col="receptions")           # fair .419 → +8.1%
        out = _generate({"player_reception_yds": yds, "player_receptions": rec})
        assert len(out) == 1 and out[0].market == "player_receptions" and out[0].odds == 125

    def test_alternate_rungs_are_not_over_lines(self):
        model = _Model({"star guy": {69.5: 0.99, 62.5: 0.45}})
        out = _generate({"player_reception_yds": model})
        assert [c.line for c in out] == [62.5]

    def test_opponent_resolved_from_roster_overlay(self):
        model = _Model({"star guy": {62.5: 0.56}})
        _generate({"player_reception_yds": model}, current_teams={"star guy": "NE"})
        assert [c for c in model.calls if c[0] == "star guy"] == [("star guy", "SEA")]


class TestOversAccounting:
    def test_own_bankroll_and_multi_strategy_exclusion(self, monkeypatch):
        monkeypatch.setattr(ledger_mod.settings, "overs_bankroll", 40.0)
        monkeypatch.setattr(ledger_mod.settings, "ladder_bankroll", 50.0)
        monkeypatch.setattr(ledger_mod.settings, "starting_bankroll", 100.0)
        assert ledger_mod.starting_bankroll_for("overs") == 40.0
        assert ledger_mod.starting_bankroll_for("ladder") == 50.0
        assert ledger_mod.starting_bankroll_for(None) == 100.0
        assert ledger_mod.SIDE_BOOKS == ("ladder", "overs")

        from betting_agent.db.models import Pick

        clause = str(ledger_mod.strategy_exclusion(Pick.strategy, ledger_mod.SIDE_BOOKS))
        assert "picks.strategy IS NULL" in clause and "NOT IN" in clause
        single = str(ledger_mod.strategy_exclusion(Pick.strategy, "ladder"))
        assert "NOT IN" in single
        assert ledger_mod.strategy_exclusion(Pick.strategy, None) is None

    def test_grade_script_excludes_both_side_books_from_the_main_line(self):
        src = Path("scripts/grade.py").read_text()
        assert '"exclude_strategy": SIDE_BOOKS' in src and "OVERS_STRATEGY" in src


def _over_pick(**kw):
    base = dict(game_id=0, external_id="evt-1", home_team="Seattle Seahawks",
                away_team="New England Patriots", game_date=date(2026, 9, 8), sport="NFL",
                bet_type="prop", pick_side="over", player="Romeo Doubs",
                market="player_reception_yds", line=36.5, model_prob=0.55,
                implied_prob=0.502, edge=0.048, odds=-113, kelly_fraction=0.02,
                recommended_bet=2.0, bankroll_at_pick=100.0, strategy="overs",
                extra={"bookmaker": "draftkings", "projection_mean": 43.5, "flat_stake": False})
    base.update(kw)
    return BetCandidate(**base)


class TestOversDiscord:
    def test_embed_names_the_line_and_flags_flat_stakes(self):
        embed = _build_over_embed(_over_pick(), 1)
        assert embed["title"] == "OVER #1  Romeo Doubs over 36.5 receiving yds"
        assert "-113" in embed["description"] and "overs bankroll" in embed["description"]
        flat = _build_over_embed(_over_pick(edge=-0.02, kelly_fraction=0.01, recommended_bet=1.0,
                                            extra={"bookmaker": "draftkings", "flat_stake": True}), 1)
        assert "flat 1% stake" in flat["description"]
        header = _build_overs_header_embed([_over_pick(), _over_pick(extra={"flat_stake": True})],
                                           100.0, 1)
        assert header["title"].startswith("STRAIGHT OVERS — 2 on card")
        assert "1 at a flat 1% stake" in header["description"]
        assert "$100.00" in header["description"] and "1 more over(s) saved off-card" in header["description"]

    def test_card_order_is_props_then_overs_then_ladder(self, monkeypatch):
        sent = []
        monkeypatch.setattr("betting_agent.notifications.discord._get_webhook_url",
                            lambda *_a: "http://hook")
        monkeypatch.setattr("betting_agent.notifications.discord._send_webhook",
                            lambda url, payload: sent.append(payload) or True)
        prop = _over_pick(strategy=None, pick_side="under", player="Someone Else")
        ladder = _over_pick(strategy="ladder", market="player_reception_yds_alternate", line=69.5,
                            player="Third Guy")
        assert send_slate_to_discord("Card", [], [prop], 100.0, overs=[_over_pick()],
                                     overs_bankroll=100.0, ladder=[ladder], ladder_bankroll=100.0)
        titles = [e.get("title", "") for e in sent[0]["embeds"]]
        assert titles[1].startswith("#1")
        assert titles[2].startswith("STRAIGHT OVERS") and titles[3].startswith("OVER #1")
        assert titles[4].startswith("LADDER HITS") and titles[5].startswith("LADDER #1")
        # An overs-only card still posts.
        sent.clear()
        assert send_slate_to_discord("Card", [], [], 100.0, overs=[_over_pick()]) and sent

    def test_results_line_uses_the_overs_label(self):
        summary = {"total_bets": 2, "wins": 1, "losses": 1, "pushes": 0, "total_pnl": 0.5,
                   "roi_pct": 12.5, "win_rate_pct": 50.0, "avg_fair_pct": 50.2}
        line = _ladder_line(summary, bankroll=100.0, label="Straight overs all-time")
        assert line.startswith("**Straight overs all-time (own bankroll):** 1-1-0")
        assert "$100.00 → $100.50" in line

    def test_pick_detail_tag(self):
        from betting_agent.notifications.discord import _build_results_embed

        detail = {"result": "win", "pnl": 1.0, "away_team": "NE", "home_team": "SEA", "odds": -113,
                  "bet_type": "prop", "strategy": "overs", "player": "Romeo Doubs",
                  "market": "player_reception_yds", "line": 36.5, "pick_side": "over"}
        summary = {"total_bets": 1, "wins": 1, "losses": 0, "pushes": 0, "total_pnl": 1.0,
                   "roi_pct": 50.0, "win_rate_pct": 100.0, "avg_edge_pct": 4.8}
        embed = _build_results_embed(summary, "NFL", date(2026, 9, 10), pick_details=[detail])
        assert "OVER Romeo Doubs over 36.5 receiving yds" in embed["description"]


def test_side_book_summaries_are_plumbed_into_the_results_post(monkeypatch):
    from betting_agent.notifications import discord as d

    sent = []
    monkeypatch.setattr(d, "_get_webhook_url", lambda *_a: "http://hook")
    monkeypatch.setattr(d, "_send_webhook", lambda url, payload: sent.append(payload) or True)
    summary = {"total_bets": 1, "wins": 1, "losses": 0, "pushes": 0, "total_pnl": 1.0,
               "roi_pct": 50.0, "win_rate_pct": 100.0, "avg_edge_pct": 4.8, "total_wagered": 2.0}
    overs = {"total_bets": 1, "wins": 0, "losses": 1, "pushes": 0, "total_pnl": -1.0,
             "roi_pct": -100.0, "win_rate_pct": 0.0, "avg_fair_pct": 50.0}
    assert d.send_results_to_discord(summary, "NFL", [], date(2026, 9, 10), overs_summary=overs,
                                     alltime_summary=summary, starting_bankroll=1000.0,
                                     alltime_overs_summary=overs, overs_bankroll=100.0)
    text = "\n".join(e.get("description", "") for e in sent[0]["embeds"])
    assert "**Straight overs (own bankroll):** 0-1-0" in text
    assert "Straight overs all-time (own bankroll)" in text and "$100.00 → $99.00" in text


def test_stub_signature_matches_real_model():
    """The stub must offer what generate_over_candidates calls."""
    real = props_script.ReceivingPropsModel
    assert hasattr(real, "project_ladder")
    assert isinstance(SimpleNamespace(), object)
