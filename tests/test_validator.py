from __future__ import annotations

import pytest

from datetime import date

from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.orchestrator import (
    AgentValidationRecord,
    save_agent_validations_to_db,
    validate_picks,
)
from betting_agent.intelligence.validator.schemas import (
    GameValidationResult,
    UsageTokens,
    ValidationPickResult,
)


def _candidate(game_id: int, edge: float, pick_side: str = "TeamA", external_id: str | None = None):
    return BetCandidate(
        game_id=game_id,
        external_id=external_id or str(game_id),
        home_team="TeamA",
        away_team="TeamB",
        game_date=date(2026, 1, 15),
        scheduled_game_date=date(2026, 1, 16),
        sport="NFL",
        bet_type="moneyline",
        pick_side=pick_side,
        model_prob=0.58,
        implied_prob=0.50,
        edge=edge,
        odds=-110,
        kelly_fraction=0.04,
        recommended_bet=40.0,
        bankroll_at_pick=1000.0,
    )


class _NoSearch:
    def search(self, query: str, team_hint: str | None = None):
        return []


def test_validate_picks_top_mode_limits_games(monkeypatch):
    calls = []
    payload_dates = []

    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            calls.append(payload.game_id)
            payload_dates.append(payload.game_date)
            return GameValidationResult(
                game_id=payload.game_id,
                results=[
                    ValidationPickResult(
                        bet_type="moneyline",
                        pick_side="TeamA",
                        verdict="UNCHANGED",
                        edge_adjustment=0.0,
                        adjusted_edge=0.08,
                        reasons=["no material issues"],
                    )
                ],
                tokens_used=UsageTokens(input=100, output=20),
                estimated_cost_usd=0.01,
            )

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    candidates = [
        _candidate(1, 0.08),
        _candidate(2, 0.05, external_id="2"),
        _candidate(3, 0.03, external_id="3"),
    ]
    validated, summary = validate_picks(candidates, sport="NFL", mode="top", max_games=2)

    assert len(validated) == 3
    assert summary.validated_games == 2
    assert calls == ["1", "2"]
    assert payload_dates == [date(2026, 1, 16), date(2026, 1, 16)]
    assert validated[2].agent_verdict == "SKIPPED"


def test_validate_picks_reduces_and_drops(monkeypatch):
    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            return GameValidationResult(
                game_id=payload.game_id,
                results=[
                    ValidationPickResult(
                        bet_type="moneyline",
                        pick_side="TeamA",
                        verdict="REDUCED",
                        edge_adjustment=-0.02,
                        adjusted_edge=0.04,
                        kelly_multiplier=0.5,
                        reasons=["starter uncertainty"],
                    ),
                    ValidationPickResult(
                        bet_type="total",
                        pick_side="over 44.5",
                        verdict="NO_BET",
                        edge_adjustment=-0.03,
                        adjusted_edge=0.02,
                        reasons=["weather risk"],
                    ),
                ],
                tokens_used=UsageTokens(input=100, output=20),
                estimated_cost_usd=0.02,
            )

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    ml = _candidate(1, 0.06)
    total = BetCandidate(
        game_id=1,
        external_id="1",
        home_team="TeamA",
        away_team="TeamB",
        game_date=date(2026, 1, 15),
        sport="NFL",
        bet_type="total",
        pick_side="over 44.5",
        model_prob=0.57,
        implied_prob=0.50,
        edge=0.05,
        odds=-110,
        kelly_fraction=0.02,
        recommended_bet=20.0,
        bankroll_at_pick=1000.0,
    )

    validated, summary = validate_picks([ml, total], sport="NFL", mode="all", shadow=False)

    assert len(validated) == 1
    assert validated[0].agent_verdict == "REDUCED"
    assert validated[0].original_edge == 0.06
    assert validated[0].edge == 0.04
    assert validated[0].recommended_bet == 20.0
    assert summary.validated_games == 1
    assert len(summary.records) == 2


def test_failed_call_spend_is_booked_against_the_budget(monkeypatch):
    class _Validator:
        last_call_cost_usd = 0.42   # killed by --max-budget-usd after spending

        def is_available(self):
            return True

        def validate(self, payload):
            return None

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    a, b = _candidate(1, 0.06), _candidate(1, 0.05)
    b.bet_type = "total"
    validated, summary = validate_picks([a, b], sport="NFL", mode="all")

    assert summary.skipped_games == 1
    assert summary.total_cost_usd == pytest.approx(0.42)
    assert [r.verdict for r in summary.records] == ["SKIPPED", "SKIPPED"]
    assert sum(r.cost_usd for r in summary.records) == pytest.approx(0.42)
    assert all(c.agent_verdict == "SKIPPED" and c.edge in (0.06, 0.05) for c in validated)


def test_validate_picks_fails_open_when_validator_errors(monkeypatch):
    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            return None

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    candidate = _candidate(1, 0.06)
    validated, summary = validate_picks([candidate], sport="NFL", mode="all")

    assert len(validated) == 1
    assert validated[0].agent_verdict == "SKIPPED"
    assert validated[0].edge == 0.06
    assert summary.validated_games == 0
    assert summary.skipped_games == 1


def test_validate_picks_continues_after_single_game_failure(monkeypatch):
    calls = []

    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            calls.append(payload.game_id)
            if payload.game_id == "1":
                return None
            return GameValidationResult(
                game_id=payload.game_id,
                results=[
                    ValidationPickResult(
                        bet_type="moneyline",
                        pick_side="TeamA",
                        verdict="UNCHANGED",
                        edge_adjustment=0.0,
                        adjusted_edge=0.05,
                        reasons=["no material issues"],
                    )
                ],
                tokens_used=UsageTokens(input=100, output=20),
                estimated_cost_usd=0.01,
            )

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    validated, summary = validate_picks(
        [_candidate(1, 0.06), _candidate(2, 0.05, external_id="2")],
        sport="NFL",
        mode="all",
    )

    assert calls == ["1", "2"]
    assert summary.validated_games == 1
    assert summary.skipped_games == 1
    assert validated[0].agent_verdict == "SKIPPED"
    assert validated[1].agent_verdict == "UNCHANGED"


def test_validate_picks_skips_search_when_validator_unavailable(monkeypatch):
    search_calls = []

    class _Search:
        def search(self, query: str, team_hint: str | None = None):
            search_calls.append(query)
            return []

    class _Validator:
        def is_available(self):
            return False

        def validate(self, payload):
            raise AssertionError("validate should not be called when validator is unavailable")

    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _Search(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: _Validator(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )

    candidate = _candidate(1, 0.06)
    validated, summary = validate_picks([candidate], sport="NFL", mode="all")

    assert len(validated) == 1
    assert validated[0].agent_verdict == "SKIPPED"
    assert validated[0].agent_reasons == ["validator unavailable"]
    assert search_calls == []
    assert summary.skipped_games == 1


def test_save_agent_validations_preserves_duplicate_attempts(monkeypatch):
    added = []

    class _Session:
        def add(self, row):
            added.append(row)

    from contextlib import contextmanager

    @contextmanager
    def _fake_session():
        yield _Session()

    monkeypatch.setattr("betting_agent.intelligence.validator.orchestrator.get_session", _fake_session)

    record = AgentValidationRecord(
        game_id=1,
        external_id="game-1",
        pick_date=date(2026, 1, 15),
        sport="NBA",
        bet_type="moneyline",
        pick_side="TeamA",
        verdict="REDUCED",
        original_edge=0.05,
        adjusted_edge=0.04,
        reasons=["lineup risk"],
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.01,
    )

    save_agent_validations_to_db([record, record])

    assert len(added) == 2


def _prop(player: str, side: str = "over", edge: float = 0.15):
    return BetCandidate(
        game_id=0, external_id="evt-1", home_team="Kansas City Chiefs",
        away_team="Buffalo Bills", game_date=date(2026, 9, 12),
        scheduled_game_date=date(2026, 9, 13), sport="NFL", bet_type="prop",
        pick_side=side, player=player, market="player_receptions", line=4.5,
        model_prob=0.5 + edge, implied_prob=0.5, edge=edge, odds=-110,
        kelly_fraction=0.04, recommended_bet=40.0, bankroll_at_pick=1000.0,
        extra={"projection_mean": 3.4, "projection_games": 12,
               "recent_values": [3, 4, 2, 5], "team": "KC"},
    )


def _prop_result(*items):
    return GameValidationResult(
        game_id="evt-1",
        results=[ValidationPickResult(
            bet_type="prop", pick_side=side, player=player, verdict=verdict,
            edge_adjustment=adj, adjusted_edge=0.1, kelly_multiplier=mult, reasons=[reason],
        ) for player, side, verdict, adj, mult, reason in items],
        tokens_used=UsageTokens(input=500, output=50),
        estimated_cost_usd=0.04,
    )


def _wire(monkeypatch, validator):
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.TavilySearchClient",
        lambda: _NoSearch(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.make_validator",
        lambda: validator,
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.orchestrator.budget_allows",
        lambda target_date, run_cost: True,
    )


def test_shadow_mode_records_verdicts_without_touching_stakes(monkeypatch):
    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            return _prop_result(
                ("Travis Kelce", "over", "NO_BET", -0.03, 1.0, "ruled out Friday"),
                ("Rashee Rice", "over", "REDUCED", -0.02, 0.5, "questionable"),
            )

    _wire(monkeypatch, _Validator())
    kelce, rice = _prop("Travis Kelce"), _prop("Rashee Rice")
    validated, summary = validate_picks([kelce, rice], sport="NFL", mode="all", shadow=True)

    assert summary.shadow is True
    assert validated == [kelce, rice]                    # NO_BET kept
    assert kelce.agent_verdict == "NO_BET" and rice.agent_verdict == "REDUCED"
    for c in (kelce, rice):
        assert c.edge == 0.15 and c.kelly_fraction == 0.04 and c.recommended_bet == 40.0
        assert c.extra["agent"]["shadow"] is True
    assert rice.extra["agent"]["proposed_kelly_multiplier"] == 0.5
    assert [r.verdict for r in summary.records] == ["NO_BET", "REDUCED"]
    assert summary.records[0].pick_side.startswith("Travis Kelce")
    assert summary.records[0].adjusted_edge == 0.12   # what it WOULD have been


def test_live_mode_applies_prop_verdicts(monkeypatch):
    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            return _prop_result(
                ("Travis Kelce", "over", "NO_BET", -0.03, 1.0, "ruled out"),
                ("Rashee Rice", "over", "REDUCED", -0.02, 0.5, "questionable"),
            )

    _wire(monkeypatch, _Validator())
    kelce, rice = _prop("Travis Kelce"), _prop("Rashee Rice")
    validated, _ = validate_picks([kelce, rice], sport="NFL", mode="all", shadow=False)
    assert validated == [rice]
    assert rice.edge == 0.13 and rice.recommended_bet == 20.0


def test_two_same_side_props_in_one_game_are_keyed_by_player(monkeypatch):
    """Before, results keyed on (bet_type, pick_side) and the second 'over'
    silently overwrote the first."""
    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            assert len(payload.picks) == 2
            assert {p.player for p in payload.picks} == {"Travis Kelce", "Rashee Rice"}
            assert payload.picks[0].recent_values == [3.0, 4.0, 2.0, 5.0]
            return _prop_result(
                ("Travis Kelce", "over", "UNCHANGED", 0.0, 1.0, "fine"),
                ("Rashee Rice", "over", "REDUCED", -0.01, 0.8, "limited"),
            )

    _wire(monkeypatch, _Validator())
    kelce, rice = _prop("Travis Kelce"), _prop("Rashee Rice")
    validate_picks([kelce, rice], sport="NFL", mode="all", shadow=True)
    assert kelce.agent_verdict == "UNCHANGED"
    assert rice.agent_verdict == "REDUCED"


def test_injury_flags_and_game_lines_reach_the_payload(monkeypatch):
    import pandas as pd

    captured = {}

    class _Validator:
        def is_available(self):
            return True

        def validate(self, payload):
            captured["payload"] = payload
            return _prop_result(("Travis Kelce", "over", "UNCHANGED", 0.0, 1.0, "ok"))

    _wire(monkeypatch, _Validator())
    kelce = _prop("Travis Kelce")
    kelce.extra["flags"] = [{"type": "qb_out", "team": "KC", "severity": "high",
                             "detail": "KC QB1 Patrick Mahomes is Out", "player": "Travis Kelce"}]
    injuries = pd.DataFrame({
        "full_name": ["Travis Kelce"], "team": ["KC"], "position": ["TE"],
        "report_status": ["Questionable"], "practice_status": [None], "gsis_id": ["1"],
    })
    validate_picks([kelce], sport="NFL", mode="all", shadow=True, injuries=injuries,
                   qb1={}, player_teams={"travis kelce": "KC"},
                   game_lines={"evt-1": (-3.5, 47.5)})
    payload = captured["payload"]
    assert payload.spread_line == -3.5 and payload.total_line == 47.5
    assert {f.type for f in payload.deterministic_flags} == {"player_injury", "qb_out"}
