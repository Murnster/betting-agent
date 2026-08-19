"""Tests for the bankroll ledger."""

from contextlib import contextmanager
from datetime import date, datetime

import pytest

from betting_agent.accounting import ledger as ledger_mod


class _Pick:
    def __init__(self, pid, pnl, result="win", stake=10.0, price=-110,
                 graded_at=None, clv=None, sport="NFL", bet_type="moneyline"):
        self.id = pid
        self.pnl = pnl
        self.result = result
        self.stake = stake
        self.price = price
        self.graded_at = graded_at or datetime(2026, 1, 1)
        self.clv = clv
        self.sport = sport
        self.bet_type = bet_type
        self.pick_side = "X"


class _Game:
    def __init__(self, game_date):
        self.game_date = game_date


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def join(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _Query(self._rows)


def _patch(monkeypatch, rows):
    @contextmanager
    def fake_session():
        yield _Session(rows)
    monkeypatch.setattr(ledger_mod, "get_session", fake_session)


def test_equity_runs_in_event_order(monkeypatch):
    rows = [
        (_Pick(2, -10.0, "loss"), _Game(date(2026, 1, 12))),
        (_Pick(1, +9.09, "win"), _Game(date(2026, 1, 5))),
    ]
    _patch(monkeypatch, rows)
    curve = ledger_mod.equity_curve(starting_bankroll=100.0)
    assert [e.pick_id for e in curve] == [1, 2]
    assert curve[0].equity == pytest.approx(109.09)
    assert curve[1].equity == pytest.approx(99.09)


def test_summary_tracks_peak_and_drawdown(monkeypatch):
    rows = [
        (_Pick(1, +50.0), _Game(date(2026, 1, 1))),
        (_Pick(2, -80.0, "loss"), _Game(date(2026, 1, 2))),
        (_Pick(3, +10.0), _Game(date(2026, 1, 3))),
    ]
    _patch(monkeypatch, rows)
    monkeypatch.setattr(ledger_mod.settings, "starting_bankroll", 100.0)
    s = ledger_mod.ledger_summary()
    assert s["current_bankroll"] == pytest.approx(80.0)
    assert s["peak_equity"] == pytest.approx(150.0)
    assert s["max_drawdown"] == pytest.approx(80.0)
    assert s["settled_picks"] == 3


def test_empty_ledger(monkeypatch):
    _patch(monkeypatch, [])
    monkeypatch.setattr(ledger_mod.settings, "starting_bankroll", 100.0)
    s = ledger_mod.ledger_summary()
    assert s["current_bankroll"] == 100.0
    assert s["settled_picks"] == 0
    assert ledger_mod.current_bankroll() == 100.0
