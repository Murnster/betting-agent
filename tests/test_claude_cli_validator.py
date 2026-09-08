"""The `claude -p` validator provider: envelope parsing, failure modes, and
the command it builds."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from types import SimpleNamespace

from betting_agent.intelligence.validator.claude_cli import ClaudeCliValidator, parse_cli_envelope
from betting_agent.intelligence.validator.orchestrator import make_validator
from betting_agent.intelligence.validator.schemas import CandidateValidationInput, ValidatorInput


def _payload() -> ValidatorInput:
    return ValidatorInput(
        game_id="evt-1", external_id="evt-1", sport="NFL", game_date=date(2026, 9, 13),
        home_team="Kansas City Chiefs", away_team="Buffalo Bills",
        picks=[CandidateValidationInput(
            bet_type="prop", pick_side="under", model_prob=0.66, implied_prob=0.5,
            edge=0.16, odds=-110, kelly_fraction=0.03, recommended_bet=3.0,
            player="Travis Kelce", team="Kansas City Chiefs", market="player_receptions",
            line=4.5, projection_mean=3.4, projection_games=20,
            recent_values=[3, 4, 2, 5, 3, 3, 4, 2],
        )],
    )


def _result_body(game_id="evt-1"):
    return {
        "game_id": game_id,
        "results": [{
            "bet_type": "prop", "pick_side": "under", "player": "Travis Kelce",
            "market": "player_receptions", "verdict": "REDUCED", "edge_adjustment": -0.02,
            "adjusted_edge": 0.14, "kelly_multiplier": 0.5,
            "reasons": ["Kelce limited in practice Wed (team report)"],
        }],
    }


def _available(monkeypatch):
    monkeypatch.setattr("betting_agent.intelligence.validator.claude_cli.shutil.which",
                        lambda name: "/usr/bin/claude")


class TestParseEnvelope:
    def test_structured_output_field(self):
        env = {"type": "result", "is_error": False, "structured_output": _result_body(),
               "usage": {"input_tokens": 1200, "output_tokens": 80,
                         "cache_read_input_tokens": 300},
               "total_cost_usd": 0.0412}
        res = parse_cli_envelope(json.dumps(env), "evt-1")
        assert res is not None
        assert res.results[0].verdict == "REDUCED"
        assert res.results[0].player == "Travis Kelce"
        assert res.tokens_used.input == 1500 and res.tokens_used.output == 80
        assert res.estimated_cost_usd == 0.0412

    def test_fenced_json_in_result_text(self):
        env = {"type": "result", "result": "Here you go:\n```json\n"
               + json.dumps(_result_body()) + "\n```", "total_cost_usd": 0.01}
        res = parse_cli_envelope(json.dumps(env), "evt-1")
        assert res is not None and res.results[0].verdict == "REDUCED"

    def test_error_envelope_is_none(self):
        env = {"type": "result", "is_error": True, "result": "budget exceeded"}
        assert parse_cli_envelope(json.dumps(env), "evt-1") is None

    def test_garbage_is_none(self):
        assert parse_cli_envelope("not json at all", "evt-1") is None
        assert parse_cli_envelope(json.dumps({"result": "plain prose"}), "evt-1") is None

    def test_schema_violation_is_none(self):
        env = {"structured_output": {"game_id": "evt-1", "results": [{"bet_type": "prop"}]}}
        assert parse_cli_envelope(json.dumps(env), "evt-1") is None


class TestValidate:
    def test_runs_cli_without_claudecode_env_and_parses(self, monkeypatch):
        _available(monkeypatch)
        monkeypatch.setenv("CLAUDECODE", "1")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs["env"]
            seen["input"] = kwargs["input"]
            seen["cwd"] = kwargs["cwd"]
            return SimpleNamespace(returncode=0, stdout=json.dumps(
                {"structured_output": _result_body(), "total_cost_usd": 0.02}), stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        v = ClaudeCliValidator(model="claude/sonnet", web_search=True, max_turns=4)
        res = v.validate(_payload())
        assert res is not None and res.estimated_cost_usd == 0.02
        assert "CLAUDECODE" not in seen["env"]
        cmd = seen["cmd"]
        assert cmd[1] == "-p" and "--bare" not in cmd   # --bare skips the keychain login
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert cmd[cmd.index("--tools") + 1] == "WebSearch"
        assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch"   # else silently denied
        assert cmd[cmd.index("--max-turns") + 1] == "4"
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        assert "--max-budget-usd" in cmd and "--json-schema" in cmd
        rules = cmd[cmd.index("--system-prompt") + 1]
        assert "web search" in rules.lower() and "Player props" in rules
        assert "Travis Kelce" in seen["input"]          # payload goes on stdin
        assert seen["cwd"] != str(__import__("pathlib").Path.cwd())

    def test_no_web_search_passes_empty_tools(self, monkeypatch):
        _available(monkeypatch)
        seen = {}
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (
            seen.setdefault("cmd", cmd),
            SimpleNamespace(returncode=0, stdout=json.dumps(
                {"structured_output": _result_body()}), stderr=""))[1])
        ClaudeCliValidator(model="claude/haiku", web_search=False).validate(_payload())
        assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""
        assert "--allowedTools" not in seen["cmd"]

    def test_nonzero_exit_is_none(self, monkeypatch):
        _available(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="boom"))
        assert ClaudeCliValidator(model="claude/sonnet").validate(_payload()) is None

    def test_budget_killed_call_reports_its_spend(self, monkeypatch):
        # --max-budget-usd kills the CLI mid-search: exit 1, but the JSON
        # envelope still carries what was spent. That money must be booked.
        _available(monkeypatch)
        env = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True,
               "stop_reason": "tool_use", "total_cost_usd": 0.4197, "result": ""}
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout=json.dumps(env), stderr=""))
        v = ClaudeCliValidator(model="claude/sonnet")
        assert v.validate(_payload()) is None
        assert v.last_call_cost_usd == 0.4197

    def test_successful_call_records_spend_too(self, monkeypatch):
        _available(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps(
                {"structured_output": _result_body(), "total_cost_usd": 0.02}), stderr=""))
        v = ClaudeCliValidator(model="claude/sonnet")
        assert v.validate(_payload()) is not None
        assert v.last_call_cost_usd == 0.02

    def test_timeout_is_none(self, monkeypatch):
        _available(monkeypatch)

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

        monkeypatch.setattr(subprocess, "run", boom)
        assert ClaudeCliValidator(model="claude/sonnet").validate(_payload()) is None

    def test_missing_binary_is_unavailable(self, monkeypatch):
        monkeypatch.setattr("betting_agent.intelligence.validator.claude_cli.shutil.which",
                            lambda name: None)
        v = ClaudeCliValidator(model="claude/sonnet")
        assert v.is_available() is False
        assert v.validate(_payload()) is None


class TestProviderFactory:
    def test_claude_prefix_selects_cli_provider(self):
        v = make_validator("claude/sonnet")
        assert isinstance(v, ClaudeCliValidator) and v.model == "sonnet"

    def test_gemini_prefix_selects_gemini(self):
        from betting_agent.intelligence.validator.llm_validator import GeminiValidator
        v = make_validator("gemini/gemini-2.5-flash")
        assert isinstance(v, GeminiValidator) and v.model == "gemini-2.5-flash"
