"""
Tests for Odds API key rotation.

Each key carries its own monthly quota (500 on the free tier) and the NFL
props loop peaks around 564 credits in November, so the season depends on the
client walking to a fallback key rather than dying on an exhausted one.
"""

import pytest
import requests

from betting_agent.api import odds as odds_module
from betting_agent.api.odds import OddsAPIClient, key_failure_reason


class _Resp:
    def __init__(self, status_code=200, body=None, json_data=None, remaining=None):
        self.status_code = status_code
        self.text = body or ""
        self._json = json_data if json_data is not None else []
        self.headers = {} if remaining is None else {"x-requests-remaining": remaining}

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def _reset():
    odds_module._reset_key_rotation()
    yield
    odds_module._reset_key_rotation()


@pytest.fixture
def three_keys(monkeypatch):
    monkeypatch.setattr(odds_module.settings, "odds_api_key", "key1")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_2", "key2")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_3", "key3")


def _record_calls(monkeypatch, responses):
    """Serve `responses` in order, capturing the apiKey each call was made with."""
    used = []

    def fake_get(url, params=None, timeout=None):
        used.append((params or {}).get("apiKey"))
        return responses[len(used) - 1]

    monkeypatch.setattr(odds_module.requests, "get", fake_get)
    return used


def test_keys_collapse_blanks_and_duplicates(monkeypatch):
    monkeypatch.setattr(odds_module.settings, "odds_api_key", "key1")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_2", "   ")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_3", "key1")
    assert odds_module.settings.odds_api_keys == ["key1"]


def test_exhausted_key_falls_back_to_the_next_one(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [
        _Resp(401, body='{"message":"Usage quota has been reached"}'),
        _Resp(200, json_data=[{"id": "evt"}]),
    ])
    assert OddsAPIClient()._get("nfl/events", {}) == [{"id": "evt"}]
    assert used == ["key1", "key2"]


def test_rotation_survives_two_exhausted_keys(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [
        _Resp(401, body="Usage quota has been reached"),
        _Resp(429, body="too many requests"),
        _Resp(200, json_data=[{"id": "evt"}]),
    ])
    assert OddsAPIClient()._get("nfl/events", {}) == [{"id": "evt"}]
    assert used == ["key1", "key2", "key3"]


def test_rotation_is_remembered_so_the_next_call_skips_the_dead_key(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [
        _Resp(401, body="Usage quota has been reached"),
        _Resp(200, json_data=[{"id": "a"}]),
        _Resp(200, json_data=[{"id": "b"}]),
    ])
    client = OddsAPIClient()
    client._get("nfl/events", {})
    client._get("nfl/odds", {})
    assert used == ["key1", "key2", "key2"]


def test_zero_remaining_retires_the_key_without_a_wasted_request(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [
        _Resp(200, json_data=[{"id": "a"}], remaining="0"),
        _Resp(200, json_data=[{"id": "b"}]),
    ])
    client = OddsAPIClient()
    assert client._get("nfl/events", {}) == [{"id": "a"}]
    client._get("nfl/odds", {})
    # The first call still succeeded on key1; only the follow-up moved on.
    assert used == ["key1", "key2"]


def test_revoked_key_also_rotates(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [
        _Resp(401, body='{"message":"Invalid API key"}'),
        _Resp(200, json_data=[{"id": "evt"}]),
    ])
    assert OddsAPIClient()._get("nfl/events", {}) == [{"id": "evt"}]
    assert used == ["key1", "key2"]


def test_all_keys_exhausted_returns_none(three_keys, monkeypatch):
    _record_calls(monkeypatch, [_Resp(401, body="Usage quota has been reached")] * 3)
    assert OddsAPIClient()._get("nfl/events", {}) is None


def test_422_is_not_a_key_problem_and_does_not_rotate(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [_Resp(422, body="market not available")])
    assert OddsAPIClient()._get("nfl/odds", {}) == []
    assert used == ["key1"]


def test_server_error_does_not_burn_through_the_keys(three_keys, monkeypatch):
    used = _record_calls(monkeypatch, [_Resp(500, body="boom")])
    assert OddsAPIClient()._get("nfl/odds", {}) is None
    assert used == ["key1"]


def test_no_keys_configured_returns_none(monkeypatch):
    monkeypatch.setattr(odds_module.settings, "odds_api_key", "")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_2", "")
    monkeypatch.setattr(odds_module.settings, "odds_api_key_3", "")
    assert OddsAPIClient()._get("nfl/events", {}) is None


def test_network_error_does_not_rotate(three_keys, monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(odds_module.requests, "get", boom)
    assert OddsAPIClient()._get("nfl/events", {}) is None
    assert odds_module._key_index == 0


def test_api_key_property_reports_the_key_in_use(three_keys, monkeypatch):
    _record_calls(monkeypatch, [
        _Resp(401, body="Usage quota has been reached"),
        _Resp(200, json_data=[]),
    ])
    client = OddsAPIClient()
    assert client.api_key == "key1"
    client._get("nfl/events", {})
    assert client.api_key == "key2"


@pytest.mark.parametrize("status,body,expected", [
    (200, "", None),
    (422, "market not available", None),
    (500, "server error", None),
    (429, "", "quota"),
    (401, '{"message":"Usage quota has been reached"}', "quota"),
    (401, '{"message":"Invalid API key"}', "rejected"),
])
def test_key_failure_reason(status, body, expected):
    assert key_failure_reason(_Resp(status, body=body)) is expected
