from __future__ import annotations

import logging
from urllib.parse import quote
from datetime import date
from types import SimpleNamespace

import requests

from betting_agent.config import settings
from betting_agent.intelligence.validator.llm_validator import GeminiValidator
from betting_agent.intelligence.validator.schemas import CandidateValidationInput, ValidatorInput


def _payload() -> ValidatorInput:
    return ValidatorInput(
        game_id="game-123",
        external_id="external-123",
        sport="NBA",
        game_date=date(2026, 3, 12),
        home_team="Detroit Pistons",
        away_team="Philadelphia 76ers",
        picks=[
            CandidateValidationInput(
                bet_type="moneyline",
                pick_side="Philadelphia 76ers",
                model_prob=0.61,
                implied_prob=0.53,
                edge=0.08,
                odds=120,
                kelly_fraction=0.03,
                recommended_bet=3.0,
            )
        ],
    )


class _FakeResponse:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self._payload = payload or {}
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


def _http_error(status_code: int, url: str) -> requests.HTTPError:
    error = requests.HTTPError(f"{status_code} Server Error: Service Unavailable for url: {url}")
    error.response = SimpleNamespace(status_code=status_code)
    return error


def test_validate_retries_transient_http_error_and_succeeds(monkeypatch):
    attempts = []
    sleeps = []
    secret = "secret-key"
    validator = GeminiValidator(api_key=secret, model="gemini/gemini-2.5-flash")

    monkeypatch.setattr(settings, "agent_request_retries", 2, raising=False)
    monkeypatch.setattr(settings, "agent_request_retry_backoff_seconds", 0.25, raising=False)
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.requests.get",
        lambda *args, **kwargs: _FakeResponse(),
    )

    responses = [
        _FakeResponse(
            error=_http_error(
                503,
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={secret}",
            )
        ),
        _FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"game_id":"game-123","results":[{"bet_type":"moneyline",'
                                        '"pick_side":"Philadelphia 76ers","verdict":"UNCHANGED",'
                                        '"edge_adjustment":0.0,"adjusted_edge":0.08,'
                                        '"kelly_multiplier":1.0,"reasons":["no material issues"]}],'
                                        '"tokens_used":{"input":10,"output":5},"estimated_cost_usd":0.0}'
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        ),
    ]

    def _post(*args, **kwargs):
        attempts.append(kwargs["params"]["key"])
        return responses.pop(0)

    monkeypatch.setattr("betting_agent.intelligence.validator.llm_validator.requests.post", _post)

    result = validator.validate(_payload())

    assert result is not None
    assert result.results[0].verdict == "UNCHANGED"
    assert attempts == [secret, secret]
    assert sleeps == [0.25]


def test_validate_redacts_api_key_in_failure_logs(monkeypatch, caplog):
    secret = "secret-key"
    validator = GeminiValidator(api_key=secret, model="gemini/gemini-2.5-flash")

    monkeypatch.setattr(settings, "agent_request_retries", 0, raising=False)
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.requests.get",
        lambda *args, **kwargs: _FakeResponse(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.requests.post",
        lambda *args, **kwargs: _FakeResponse(
            error=_http_error(
                503,
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={secret}",
            )
        ),
    )

    caplog.set_level(logging.WARNING)

    result = validator.validate(_payload())

    assert result is None
    assert secret not in caplog.text
    assert "key=***" in caplog.text


def test_validate_redacts_urlencoded_api_key_in_failure_logs(monkeypatch, caplog):
    secret = "secret/key+="
    encoded_secret = quote(secret, safe="")
    validator = GeminiValidator(api_key=secret, model="gemini/gemini-2.5-flash")

    monkeypatch.setattr(settings, "agent_request_retries", 0, raising=False)
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.requests.get",
        lambda *args, **kwargs: _FakeResponse(),
    )
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.llm_validator.requests.post",
        lambda *args, **kwargs: _FakeResponse(
            error=_http_error(
                503,
                (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.5-flash:generateContent?key={encoded_secret}"
                ),
            )
        ),
    )

    caplog.set_level(logging.WARNING)

    result = validator.validate(_payload())

    assert result is None
    assert secret not in caplog.text
    assert encoded_secret not in caplog.text
    assert "key=***" in caplog.text
