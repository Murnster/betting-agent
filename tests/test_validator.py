from __future__ import annotations

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
        "betting_agent.intelligence.validator.orchestrator.GeminiValidator",
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
        "betting_agent.intelligence.validator.orchestrator.GeminiValidator",
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

    validated, summary = validate_picks([ml, total], sport="NFL", mode="all")

    assert len(validated) == 1
    assert validated[0].agent_verdict == "REDUCED"
    assert validated[0].original_edge == 0.06
    assert validated[0].edge == 0.04
    assert validated[0].recommended_bet == 20.0
    assert summary.validated_games == 1
    assert len(summary.records) == 2


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
        "betting_agent.intelligence.validator.orchestrator.GeminiValidator",
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
        "betting_agent.intelligence.validator.orchestrator.GeminiValidator",
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
        "betting_agent.intelligence.validator.orchestrator.GeminiValidator",
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
