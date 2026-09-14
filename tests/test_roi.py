from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from betting_agent.accounting.roi import get_summary
from betting_agent.config import settings


@dataclass
class _Pick:
    result: str
    pnl: float
    recommended_bet: float
    edge: float
    clv: float | None = None
    sport: str = "NBA"
    actual_bet: float | None = None

    @property
    def stake(self) -> float:
        if self.actual_bet is not None:
            return self.actual_bet
        return self.recommended_bet or 0.0


class _Query:
    def __init__(self, picks):
        self._picks = picks

    def join(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._picks


class _Session:
    def __init__(self, picks):
        self._picks = picks

    def query(self, *_args, **_kwargs):
        return _Query(self._picks)


def test_get_summary_uses_total_wagered_for_roi(monkeypatch):
    picks = [
        _Pick(result="win", pnl=20.0, recommended_bet=40.0, edge=0.10, clv=0.02),
        _Pick(result="loss", pnl=-10.0, recommended_bet=20.0, edge=0.05, clv=None),
    ]

    @contextmanager
    def _fake_session():
        yield _Session(picks)

    monkeypatch.setattr("betting_agent.accounting.roi.get_session", _fake_session)

    summary = get_summary(sport="NBA")

    assert summary["total_wagered"] == 60.0
    assert summary["total_pnl"] == 10.0
    assert summary["roi_pct"] == 16.67


def test_carded_only_is_true_for_nfl_and_none_elsewhere():
    from betting_agent.accounting.roi import carded_only

    assert carded_only("NFL") is True
    assert carded_only("nfl") is True
    assert carded_only("NBA") is None
    assert carded_only(None) is None


def test_get_summary_filters_to_carded_picks(monkeypatch):
    """Off-card picks are model evaluation, never part of a reported record."""
    from betting_agent.accounting import roi as roi_mod

    seen: list[str] = []

    class _FilterQuery(_Query):
        def filter(self, *clauses, **kwargs):
            seen.extend(str(c) for c in clauses)
            return self

    class _FilterSession:
        def query(self, *_args, **_kwargs):
            return _FilterQuery([_Pick(result="win", pnl=5.0, recommended_bet=10.0, edge=0.1)])

    @contextmanager
    def _fake_session():
        yield _FilterSession()

    monkeypatch.setattr(roi_mod, "get_session", _fake_session)

    roi_mod.get_summary(sport="NFL", on_card=True)
    assert any("picks.on_card IS true" in f or "picks.on_card = true" in f for f in seen), seen

    seen.clear()
    roi_mod.get_summary(sport="NBA", on_card=None)
    assert not any("on_card" in f for f in seen), seen


def test_graded_window_can_be_bounded_at_both_ends(monkeypatch):
    """--repost re-reads one already-graded day, so it bounds graded_at."""
    from datetime import datetime

    from betting_agent.accounting import roi as roi_mod

    seen: list[str] = []

    class _FilterQuery(_Query):
        def filter(self, *clauses, **kwargs):
            seen.extend(str(c) for c in clauses)
            return self

    class _FilterSession:
        def query(self, *_args, **_kwargs):
            return _FilterQuery([_Pick(result="win", pnl=5.0, recommended_bet=10.0, edge=0.1)])

    @contextmanager
    def _fake_session():
        yield _FilterSession()

    monkeypatch.setattr(roi_mod, "get_session", _fake_session)

    roi_mod.get_summary(sport="NFL", graded_since=datetime(2026, 9, 10),
                        graded_until=datetime(2026, 9, 10, 23, 59, 59))
    assert sum("picks.graded_at" in f for f in seen) == 2, seen


def test_report_headline_and_bankroll_exclude_every_side_book(monkeypatch):
    """The headline record is the MAIN book. The ladder, the straight overs
    and the anytime-TD scorers stake their own paper bankrolls, so blending
    them into one record and one equity line makes the headline a record of
    nothing."""
    from betting_agent.accounting import roi as roi_mod
    from betting_agent.accounting.ledger import LADDER_STRATEGY, OVERS_STRATEGY, SIDE_BOOKS
    from betting_agent.sports.nfl.td_props import TD_MARKET

    summary_calls, ledger_calls = [], []

    def fake_summary(**kwargs):
        summary_calls.append(kwargs)
        return {"total_bets": 16, "wins": 8, "losses": 7, "pushes": 0, "voids": 0,
                "win_rate_pct": 53.3, "total_wagered": 29.1, "total_pnl": 3.03,
                "roi_pct": 11.1, "avg_edge_pct": 4.0}

    def fake_ledger(**kwargs):
        ledger_calls.append(kwargs)
        return {"starting_bankroll": 100.0, "current_bankroll": 103.03, "total_pnl": 3.03,
                "peak_equity": 103.03, "max_drawdown": 0.0, "settled_picks": 16}

    monkeypatch.setattr(roi_mod, "get_summary", fake_summary)
    monkeypatch.setattr(roi_mod, "get_breakdown_by_bet_type", lambda **kw: [])
    monkeypatch.setattr("betting_agent.accounting.ledger.ledger_summary", fake_ledger)

    report = roi_mod.format_roi_report(sport="NFL", on_card=True)

    assert summary_calls[0]["exclude_market"] == TD_MARKET
    assert summary_calls[0]["exclude_strategy"] == SIDE_BOOKS
    # Main bankroll charges neither the strategy-keyed books nor the TD board.
    main_ledger = ledger_calls[0]
    assert main_ledger["exclude_strategy"] == SIDE_BOOKS
    assert TD_MARKET in main_ledger["exclude_market"]
    # Each side book is then reported on its own bankroll, the TD one by
    # market because its saved picks carry a NULL strategy.
    side_scopes = ledger_calls[1:]
    assert {c.get("strategy") for c in side_scopes} == {OVERS_STRATEGY, LADDER_STRATEGY, None}
    td_scope = next(c for c in side_scopes if c.get("market") == TD_MARKET)
    assert td_scope["starting_bankroll"] == settings.td_bankroll
    assert "TD scorers (own bankroll)" in report


def test_breakdown_forwards_the_same_exclusions_as_the_headline(monkeypatch):
    """The per-bet-type breakdown sits directly under the headline record in
    the results post. Unscoped, its PROP row read 15-28 / -16.7% beneath a
    headline of 8-7 / +11.1% — it was still blending in the TD, ladder and
    overs books."""
    from betting_agent.accounting import roi as roi_mod

    seen = []

    def fake_summary(**kwargs):
        seen.append(kwargs)
        return {"total_bets": 0}

    monkeypatch.setattr(roi_mod, "get_summary", fake_summary)
    roi_mod.get_breakdown_by_bet_type(sport="NFL", on_card=True,
                                      exclude_market="player_anytime_td",
                                      exclude_strategy=("ladder", "overs"))
    assert seen and all(c["exclude_market"] == "player_anytime_td" for c in seen)
    assert all(c["exclude_strategy"] == ("ladder", "overs") for c in seen)
