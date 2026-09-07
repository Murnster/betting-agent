"""
Validator provider that runs the local `claude -p` CLI headless.

Same interface as GeminiValidator: `is_available()` and
`validate(payload) -> GameValidationResult | None`. Any failure logs a
warning and returns None so the picks pipeline fails open.

`--json-schema` makes the CLI enforce the result shape; the envelope field
that carries it is not documented in `claude --help`, so parsing tries
`structured_output` first and falls back to the JSON in `result`. Spend is
capped per call with `--max-budget-usd` in addition to the daily gate in
costs.budget_allows().

Context hygiene matters for cost: a default `claude -p` run from the repo
loads Claude Code's own system prompt plus CLAUDE.md and memory (~29k
tokens, several cents per call before any reasoning). The validator instead
replaces the system prompt with its rules (`--system-prompt`), loads no
settings (`--setting-sources ""`), and runs from a temp directory — ~250
input tokens plus the payload. `--bare` would do the same but also skips
keychain reads, which is where the login lives, so it is NOT used.

Measured Sep 2026 (Sonnet, one prop pick): ~$0.014 without search;
~$0.09 with WebSearch (two searches, 4 turns; the search summarisation runs
on Haiku inside the CLI). Web search needs BOTH `--tools WebSearch` and
`--allowedTools WebSearch` in print mode — with only the former the model
reports the tool as unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile

from betting_agent.config import settings
from betting_agent.intelligence.validator.llm_validator import _extract_json_text
from betting_agent.intelligence.validator.prompt import (
    GAME_VALIDATION_JSON_SCHEMA,
    system_prompt,
    user_message,
)
from betting_agent.intelligence.validator.schemas import GameValidationResult, ValidatorInput

logger = logging.getLogger(__name__)


class ClaudeCliValidator:
    def __init__(
        self,
        model: str | None = None,
        web_search: bool | None = None,
        max_turns: int | None = None,
        binary: str = "claude",
    ):
        self.model = (model or settings.agent_model).split("/", 1)[-1]
        self.web_search = (
            settings.agent_claude_web_search if web_search is None else web_search
        )
        self.max_turns = max_turns or settings.agent_claude_max_turns
        self.binary = binary
        self._path: str | None = None
        self._checked = False

    def is_available(self) -> bool:
        if not self._checked:
            self._path = shutil.which(self.binary)
            self._checked = True
            if self._path is None:
                logger.warning("claude CLI not found on PATH — validator unavailable")
        return self._path is not None

    def _command(self, has_props: bool) -> list[str]:
        cmd = [
            self._path or self.binary, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(GAME_VALIDATION_JSON_SCHEMA),
            "--system-prompt", system_prompt(has_props, self.web_search),
            "--setting-sources", "",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--max-budget-usd", f"{settings.agent_claude_max_call_usd:.2f}",
            "--no-session-persistence",
            "--tools", "WebSearch" if self.web_search else "",
        ]
        if self.web_search:
            # --tools only exposes the tool; print mode still needs it
            # pre-approved or every call is silently denied.
            cmd += ["--allowedTools", "WebSearch"]
        return cmd

    def validate(self, payload: ValidatorInput) -> GameValidationResult | None:
        if not self.is_available():
            return None
        has_props = any(p.bet_type == "prop" for p in payload.picks)
        # A nested Claude Code session refuses to start while CLAUDECODE is set.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        try:
            proc = subprocess.run(
                self._command(has_props), input=user_message(payload),
                capture_output=True, text=True, cwd=tempfile.gettempdir(),
                timeout=settings.agent_request_timeout * 3, env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("claude validator failed to run for %s: %s", payload.game_id, exc)
            return None
        if proc.returncode != 0:
            logger.warning(
                "claude validator exited %d for %s: %s",
                proc.returncode, payload.game_id, (proc.stderr or proc.stdout)[:200].strip(),
            )
            return None
        return parse_cli_envelope(proc.stdout, payload.game_id)


def parse_cli_envelope(stdout: str, game_id: str) -> GameValidationResult | None:
    """Turn `claude -p --output-format json` stdout into a GameValidationResult."""
    # The envelope itself is plain JSON; only fall back to fence-stripping
    # when it is not (the fence may legitimately sit INSIDE the result text).
    try:
        envelope = json.loads(stdout)
    except ValueError:
        try:
            envelope = json.loads(_extract_json_text(stdout))
        except ValueError as exc:
            logger.warning("claude validator returned non-JSON for %s: %s", game_id, exc)
            return None
    if not isinstance(envelope, dict):
        logger.warning("claude validator envelope for %s is not an object", game_id)
        return None
    if envelope.get("is_error"):
        logger.warning("claude validator reported an error for %s: %s",
                       game_id, str(envelope.get("result", ""))[:200])
        return None

    body = envelope.get("structured_output")
    if body is None:
        raw = envelope.get("result")
        if isinstance(raw, dict):
            body = raw
        elif isinstance(raw, str):
            try:
                body = json.loads(_extract_json_text(raw))
            except ValueError as exc:
                logger.warning("claude validator result for %s is not JSON: %s", game_id, exc)
                return None
    if not isinstance(body, dict):
        logger.warning("claude validator produced no result body for %s", game_id)
        return None

    body.setdefault("game_id", game_id)
    try:
        result = GameValidationResult.model_validate(body)
    except ValueError as exc:
        logger.warning("claude validator result for %s failed validation: %s", game_id, exc)
        return None

    usage = envelope.get("usage") or {}
    if isinstance(usage, dict):
        result.tokens_used.input = int(
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        result.tokens_used.output = int(usage.get("output_tokens", 0))
    cost = envelope.get("total_cost_usd")
    if cost is not None:
        result.estimated_cost_usd = float(cost)
    return result
