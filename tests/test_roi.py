from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from betting_agent.accounting.roi import get_summary


@dataclass
class _Pick:
    result: str
    pnl: float
    recommended_bet: float
    edge: float
    clv: float | None = None
    sport: str = "NBA"


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
