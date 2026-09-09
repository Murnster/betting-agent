"""Slate windows, per-slate caps, game leans vs a sharp reference, and the
closing capture for game picks."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from betting_agent.accounting.prop_clv import capture_game_closing_lines
from betting_agent.intelligence.game_lean import (
    MARGIN_SIGMA,
    fair_from_reference,
    game_leans,
)
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.slate import (
    PRIMETIME_PROP_CAP,
    WINDOW_PROP_CAP,
    cap_per_slate,
    group_events_by_slate,
    select_card,
    slate_for,
)


def _ev(eid, ct, home="Kansas City Chiefs", away="Buffalo Bills"):
    return {"id": eid, "commence_time": ct, "home_team": home, "away_team": away}


def _candidate(external_id, edge):
    return BetCandidate(game_id=0, external_id=external_id, home_team="", away_team="",
                        game_date=date(2026, 10, 11), sport="NFL", bet_type="prop",
                        pick_side="under", model_prob=.6, implied_prob=.5, edge=edge,
                        odds=-110)


class TestSlates:
    @pytest.mark.parametrize("ct,label", [
        ("2026-09-10T00:20:00Z", "Wednesday"),        # Wed 20:20 ET opener
        ("2026-09-11T00:15:00Z", "Thursday Night"),
        ("2026-09-13T13:30:00Z", "Sunday Early"),     # 9:30 ET London game
        ("2026-09-13T17:00:00Z", "Sunday Early"),
        ("2026-09-13T20:25:00Z", "Sunday Late"),
        ("2026-09-14T00:20:00Z", "Sunday Night"),     # 20:20 ET Sunday
        ("2026-09-15T00:15:00Z", "Monday Night"),
    ])
    def test_window_labels_follow_eastern_kickoff(self, ct, label):
        assert slate_for(ct)[1] == label

    def test_grouping_orders_by_kickoff_and_sets_caps(self):
        events = [_ev("snf", "2026-09-14T00:20:00Z"), _ev("e1", "2026-09-13T17:00:00Z"),
                  _ev("e2", "2026-09-13T17:00:00Z", "Dallas Cowboys", "New York Giants"),
                  _ev("late", "2026-09-13T20:25:00Z")]
        slates = group_events_by_slate(events)
        assert [s.label for s in slates] == ["Sunday Early", "Sunday Late", "Sunday Night"]
        early, late, snf = slates
        assert early.prop_cap == WINDOW_PROP_CAP and early.lean_cap == 3
        assert snf.single_game and snf.prop_cap == PRIMETIME_PROP_CAP and snf.lean_cap == 1
        assert "Buffalo Bills @ Kansas City Chiefs" in snf.title()

    def test_select_card_takes_top_edges_in_slate_only(self):
        slate = group_events_by_slate([_ev("e1", "2026-09-14T00:20:00Z")])[0]
        def cand(eid, edge):
            return BetCandidate(game_id=0, external_id=eid, home_team="", away_team="",
                                game_date=date(2026, 9, 13), sport="NFL", bet_type="prop",
                                pick_side="under", model_prob=.6, implied_prob=.5, edge=edge,
                                odds=-110)
        cands = [cand("e1", .10), cand("e1", .30), cand("e1", .20), cand("other", .99)]
        card = select_card(cands, slate, 2)
        assert [c.edge for c in card] == [.30, .20]
        assert all(c.extra.get("card") for c in card)
        assert not cands[0].extra.get("card") and not cands[3].extra.get("card")


def _book(key, ml=(-150, 130), spread=(-3.0, -110, -110), total=(45.5, -110, -110)):
    return {"key": key, "title": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": ml[0]},
                                    {"name": "Buffalo Bills", "price": ml[1]}]},
        {"key": "spreads", "outcomes": [
            {"name": "Kansas City Chiefs", "point": spread[0], "price": spread[1]},
            {"name": "Buffalo Bills", "point": -spread[0], "price": spread[2]}]},
        {"key": "totals", "outcomes": [{"name": "Over", "point": total[0], "price": total[1]},
                                       {"name": "Under", "point": total[0], "price": total[2]}]},
    ]}


class TestGameLeans:
    def test_fair_parameters_from_reference(self):
        fair = fair_from_reference({"ml_home": -150.0, "ml_away": 130.0,
                                    "spread_home_line": -3.0, "spread_home_price": -110.0,
                                    "spread_away_price": -110.0, "total_over_line": 45.5,
                                    "over_price": -110.0, "under_price": -110.0})
        assert fair["home_prob"] == pytest.approx(0.578, abs=0.01)
        assert fair["mu_margin"] == pytest.approx(3.0)      # even money at -3 → expect +3
        assert fair["mu_total"] == pytest.approx(45.5)

    def test_off_market_book_line_is_the_lean(self):
        # Pinnacle: KC -3 even. DraftKings hangs KC -4.5 → Bills +4.5 is the value.
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("pinnacle"), _book("draftkings", spread=(-4.5, -110, -110))]}
        leans = game_leans([event], ["bet365", "draftkings"], bankroll=1000.0)
        assert len(leans) == 1
        lean = leans[0]
        assert lean.bet_type == "spread" and lean.pick_side == "Buffalo Bills +4.5"
        assert lean.line == 4.5 and lean.extra["bookmaker"] == "draftkings"
        # P(margin < 4.5 | mu 3, sigma) ≈ 0.547 vs 0.5 fair-vig → ~+4.7%
        expected = 0.5 + (4.5 - 3.0) / MARGIN_SIGMA * 0.3989 - 0.5  # linearised, loose
        assert 0.03 < lean.edge < 0.07 and abs(lean.edge - expected) < 0.02
        assert lean.recommended_bet > 0 and lean.extra["lean"] is True

    def test_book_matching_pinnacle_has_only_vig_negative_edge(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("pinnacle"), _book("draftkings")]}
        lean = game_leans([event], ["draftkings"], bankroll=1000.0)[0]
        assert lean.edge == pytest.approx(0.0, abs=1e-6)
        assert lean.recommended_bet == 0.0        # a zero-edge lean is paper with no stake

    def test_no_reference_quote_means_no_lean(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"), "bookmakers": [_book("draftkings")]}
        assert game_leans([event], ["draftkings"], bankroll=1000.0) == []

    def test_reference_is_never_the_bettable_book(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"), "bookmakers": [_book("pinnacle")]}
        assert game_leans([event], ["pinnacle"], bankroll=1000.0) == []


class TestGameClosingCapture:
    def _pick(self, bet_type, side, line, odds=-110):
        return SimpleNamespace(bet_type=bet_type, pick_side=side, line=line, odds=odds,
                               sport="NFL", closing_odds=None, closing_line=None, clv=None,
                               game=SimpleNamespace(external_id="e1"))

    def test_spread_held_gets_clv_moved_does_not(self):
        held = self._pick("spread", "Buffalo Bills +3.0", 3.0)
        moved = self._pick("spread", "Kansas City Chiefs -3.0", -3.0)
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("draftkings", spread=(-3.5, -105, -115))]}
        # DK closed KC -3.5: Bills side is +3.5 (moved for the Bills pick, line differs),
        # KC side moved from -3 to -3.5 as well.
        assert capture_game_closing_lines([event], [held, moved], ["bet365", "draftkings"]) == 2
        assert held.closing_line == 3.5 and held.closing_odds == -115 and held.clv is None
        assert moved.closing_line == -3.5 and moved.clv is None

    def test_moneyline_and_total(self):
        ml = self._pick("moneyline", "Kansas City Chiefs", None, odds=-140)
        tot = self._pick("total", "under 45.5", 45.5, odds=-110)
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("draftkings", ml=(-160, 140), total=(45.5, -105, -115))]}
        assert capture_game_closing_lines([event], [ml, tot], ["draftkings"]) == 2
        assert ml.closing_odds == -160 and ml.clv is not None and ml.clv > 0   # took -140, closed -160
        assert tot.closing_line == 45.5 and tot.closing_odds == -115 and tot.clv > 0


class TestCapPerSlate:
    """
    The main-card candidate cut used to be global (`candidates[:max_picks]`),
    which could spend its whole allowance on one kickoff window and leave a
    later one with nothing to card at all.
    """

    @staticmethod
    def _slates():
        return group_events_by_slate([
            {"id": "early1", "commence_time": "2026-10-11T17:00:00Z"},   # 1pm ET
            {"id": "early2", "commence_time": "2026-10-11T17:00:00Z"},
            {"id": "late1", "commence_time": "2026-10-11T20:25:00Z"},    # 4:25pm ET
            {"id": "night1", "commence_time": "2026-10-12T00:20:00Z"},   # 8:20pm ET
        ])

    def test_late_and_night_survive_a_cut_the_early_window_would_have_eaten(self):
        # Ten fat edges in the early window, thin ones everywhere else: a global
        # top-10 by edge keeps nothing but early games.
        candidates = [_candidate(external_id="early1", edge=0.90 - i * 0.01) for i in range(10)]
        candidates += [_candidate(external_id="late1", edge=0.20)]
        candidates += [_candidate(external_id="night1", edge=0.19)]

        kept = cap_per_slate(candidates, self._slates(), cap=10)
        surviving = {c.external_id for c in kept}
        assert "late1" in surviving, "the 4pm window was starved by the early window"
        assert "night1" in surviving, "Sunday night was starved by the early window"

        # The old global cut is what this guards against.
        assert {c.external_id for c in sorted(
            candidates, key=lambda c: c.edge, reverse=True)[:10]} == {"early1"}

    def test_each_slate_keeps_its_own_best_cap(self):
        candidates = [_candidate(external_id="early1", edge=0.5 - i * 0.01) for i in range(5)]
        candidates += [_candidate(external_id="late1", edge=0.4 - i * 0.01) for i in range(5)]

        kept = cap_per_slate(candidates, self._slates(), cap=2)
        assert len(kept) == 4
        by_event: dict[str, list[float]] = {}
        for c in kept:
            by_event.setdefault(c.external_id, []).append(c.edge)
        assert sorted(by_event) == ["early1", "late1"]
        assert by_event["early1"] == pytest.approx([0.50, 0.49])
        assert by_event["late1"] == pytest.approx([0.40, 0.39])

    def test_input_order_is_preserved(self):
        candidates = [
            _candidate(external_id="early1", edge=0.10),
            _candidate(external_id="late1", edge=0.90),
            _candidate(external_id="early1", edge=0.50),
        ]
        kept = cap_per_slate(candidates, self._slates(), cap=5)
        assert [c.edge for c in kept] == [0.10, 0.90, 0.50]

    def test_a_candidate_on_no_slate_is_kept_not_silently_dropped(self):
        candidates = [_candidate(external_id="not-an-event-today", edge=0.3)]
        assert len(cap_per_slate(candidates, self._slates(), cap=1)) == 1
