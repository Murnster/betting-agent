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
