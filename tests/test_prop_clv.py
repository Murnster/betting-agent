"""Closing-line capture for props: matching, CLV vs. moved lines, the kickoff
window, and the re-grade reset leaving prop CLV alone."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from betting_agent.accounting.prop_clv import (
    capture_prop_closing_lines,
    events_kicking_off_within,
    line_moved_for_pick,
)


def _pick(player="Travis Kelce", market="player_receptions", side="under",
          line=4.5, odds=-110, external_id="evt-1"):
    return SimpleNamespace(
        id=1, player=player, market=market, pick_side=side, line=line, odds=odds,
        closing_odds=None, closing_line=None, clv=None, bet_type="prop",
        game=SimpleNamespace(external_id=external_id),
    )


def _event(outcomes, market="player_receptions", event_id="evt-1"):
    return {"id": event_id, "bookmakers": [{"key": "bet365", "markets": [
        {"key": market, "outcomes": [
            {"name": n, "description": p, "point": pt, "price": pr} for n, p, pt, pr in outcomes
        ]},
    ]}]}


class TestCaptureProbClosingLines:
    def test_same_line_sets_closing_price_and_clv(self):
        pick = _pick(odds=-110)
        event = _event([("Over", "Travis Kelce", 4.5, -105), ("Under", "Travis Kelce", 4.5, -125)])
        assert capture_prop_closing_lines([event], [pick]) == 1
        assert pick.closing_line == 4.5
        assert pick.closing_odds == -125
        # Under closed shorter than we took it → positive CLV.
        assert pick.clv is not None and pick.clv > 0

    def test_moved_line_stores_number_but_no_clv(self):
        pick = _pick(line=4.5)
        event = _event([("Over", "Travis Kelce", 5.5, -110), ("Under", "Travis Kelce", 5.5, -110)])
        assert capture_prop_closing_lines([event], [pick]) == 1
        assert pick.closing_line == 5.5
        assert pick.closing_odds == -110
        assert pick.clv is None

    def test_alt_lines_pick_the_closest_number(self):
        pick = _pick(line=4.5)
        event = _event([
            ("Over", "Travis Kelce", 3.5, -160), ("Under", "Travis Kelce", 3.5, 120),
            ("Over", "Travis Kelce", 4.5, -105), ("Under", "Travis Kelce", 4.5, -125),
            ("Over", "Travis Kelce", 6.5, 150), ("Under", "Travis Kelce", 6.5, -190),
        ])
        capture_prop_closing_lines([event], [pick])
        assert (pick.closing_line, pick.closing_odds) == (4.5, -125)

    def test_closing_quote_reads_the_same_book_as_the_pick(self):
        pick = _pick(line=4.5, odds=-110)
        event = _event([("Over", "Travis Kelce", 4.5, -105), ("Under", "Travis Kelce", 4.5, -125)])
        event["bookmakers"][0]["key"] = "fanduel"
        event["bookmakers"].append({"key": "draftkings", "markets": [
            {"key": "player_receptions", "outcomes": [
                {"name": "Over", "description": "Travis Kelce", "point": 4.5, "price": -120},
                {"name": "Under", "description": "Travis Kelce", "point": 4.5, "price": -100},
            ]}]})
        capture_prop_closing_lines([event], [pick], book_order=["bet365", "draftkings", "fanduel"])
        assert pick.closing_odds == -100          # DraftKings, not FanDuel's -125

    def test_player_matched_across_spellings(self):
        pick = _pick(player="Amon-Ra St. Brown")
        event = _event([("Over", "Amon Ra St Brown", 4.5, -110),
                        ("Under", "Amon Ra St Brown", 4.5, -110)])
        assert capture_prop_closing_lines([event], [pick]) == 1

    def test_unmatched_player_or_market_is_untouched(self):
        pick = _pick(player="Rashee Rice")
        other_market = _pick(market="player_reception_yds")
        event = _event([("Over", "Travis Kelce", 4.5, -110), ("Under", "Travis Kelce", 4.5, -110)])
        assert capture_prop_closing_lines([event], [pick, other_market]) == 0
        for p in (pick, other_market):
            assert p.closing_line is None and p.closing_odds is None and p.clv is None

    def test_pick_in_another_game_is_untouched(self):
        pick = _pick(external_id="evt-9")
        event = _event([("Over", "Travis Kelce", 4.5, -110), ("Under", "Travis Kelce", 4.5, -110)])
        assert capture_prop_closing_lines([event], [pick]) == 0


class TestKickoffWindow:
    def _ev(self, minutes_from_now, now):
        return {"id": str(minutes_from_now),
                "commence_time": (now + timedelta(minutes=minutes_from_now))
                .strftime("%Y-%m-%dT%H:%M:%SZ")}

    def test_keeps_only_games_inside_the_window(self):
        now = datetime(2026, 9, 13, 16, 0, tzinfo=timezone.utc)
        events = [self._ev(-30, now), self._ev(45, now), self._ev(90, now), self._ev(200, now)]
        kept = events_kicking_off_within(events, 90, now=now)
        assert [e["id"] for e in kept] == ["45", "90"]

    def test_malformed_times_are_dropped(self):
        assert events_kicking_off_within([{"commence_time": "nope"}, {}], 90) == []


class TestLineMoved:
    def test_over_benefits_from_a_lower_number(self):
        assert line_moved_for_pick("over", 4.5, 3.5) is True
        assert line_moved_for_pick("over", 4.5, 5.5) is False

    def test_under_benefits_from_a_higher_number(self):
        assert line_moved_for_pick("under", 55.5, 60.5) is True
        assert line_moved_for_pick("under", 55.5, 50.5) is False

    def test_held_line_is_none(self):
        assert line_moved_for_pick("over", 4.5, 4.5) is None


class TestRegradeResetLeavesPropClv:
    def test_prop_clv_survives_a_dated_regrade(self, monkeypatch):
        import betting_agent.accounting.grader as grader_mod

        prop = SimpleNamespace(bet_type="prop", result="win", pnl=1.0, graded_at="x",
                               clv=0.03, closing_odds=-120, closing_line=4.5)
        ml = SimpleNamespace(bet_type="moneyline", result="win", pnl=1.0, graded_at="x",
                             clv=0.02, closing_odds=-115)

        class _Session:
            def query(self, model):
                return self

            def filter(self, *a, **k):
                return self

            def all(self):
                return [prop, ml]

            def flush(self):
                pass

        class _Ctx:
            def __enter__(self):
                return _Session()

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(grader_mod, "get_session", lambda: _Ctx())
        monkeypatch.setattr(grader_mod, "get_ungraded_picks", lambda session, pick_date=None: [])

        assert grader_mod.grade_picks(target_date=date(2026, 9, 13)) == 0
        assert prop.result is None and prop.clv == pytest.approx(0.03)
        assert prop.closing_odds == -120
        assert ml.clv is None and ml.closing_odds is None


class TestRoiLineMoves:
    def test_summary_counts_moves_for_and_against(self, monkeypatch):
        from contextlib import contextmanager

        from betting_agent.accounting.roi import get_summary

        def pick(side, line, closing_line):
            return SimpleNamespace(
                result="win", pnl=1.0, recommended_bet=1.0, actual_bet=None, edge=0.1,
                clv=None, stake=1.0, pick_side=side, line=line, closing_line=closing_line,
            )

        picks = [pick("over", 4.5, 3.5), pick("under", 50.5, 40.5), pick("over", 4.5, 4.5),
                 pick("over", 4.5, None)]

        class _Q:
            def join(self, *a, **k):
                return self

            def filter(self, *a, **k):
                return self

            def all(self):
                return picks

        class _S:
            def query(self, *a, **k):
                return _Q()

        @contextmanager
        def _fake():
            yield _S()

        monkeypatch.setattr("betting_agent.accounting.roi.get_session", _fake)
        s = get_summary(sport="NFL")
        assert (s["line_moves_for"], s["line_moves_against"]) == (1, 1)
