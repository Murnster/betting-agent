from __future__ import annotations

import json
import logging

import requests

from betting_agent.config import settings
from betting_agent.intelligence.validator.costs import estimate_gemini_cost
from betting_agent.intelligence.validator.schemas import GameValidationResult, ValidatorInput

logger = logging.getLogger(__name__)


class GeminiValidator:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = (model or settings.agent_model).split("/", 1)[-1]
        self._availability: bool | None = None

    def is_available(self) -> bool:
        if self._availability is not None:
            return self._availability

        if not self.api_key:
            self._availability = False
            return False

        try:
            resp = requests.get(
                self.MODEL_URL.format(model=self.model),
                params={"key": self.api_key},
                timeout=settings.agent_request_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Gemini validator unavailable for model %s: %s", self.model, exc)
            self._availability = False
            return False

        self._availability = True
        return True

    def validate(self, payload: ValidatorInput) -> GameValidationResult | None:
        if not self.is_available():
            return None

        prompt = _build_prompt(payload)
        try:
            resp = requests.post(
                self.BASE_URL.format(model=self.model),
                params={"key": self.api_key},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                        "maxOutputTokens": 2048,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                timeout=settings.agent_request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Gemini validator request failed for %s: %s", payload.game_id, exc)
            return None

        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(str(part.get("text", "")) for part in parts)
            parsed = json.loads(_extract_json_text(text))
            result = GameValidationResult.model_validate(parsed)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            snippet = ""
            try:
                snippet = text[:200].replace("\n", " ")
            except Exception:
                pass
            logger.warning(
                "Gemini validator returned invalid JSON for %s: %s | response=%r",
                payload.game_id,
                exc,
                snippet,
            )
            return None

        usage = data.get("usageMetadata", {})
        input_tokens = int(usage.get("promptTokenCount", result.tokens_used.input))
        output_tokens = int(usage.get("candidatesTokenCount", result.tokens_used.output))
        result.tokens_used.input = input_tokens
        result.tokens_used.output = output_tokens
        result.estimated_cost_usd = estimate_gemini_cost(self.model, input_tokens, output_tokens)
        return result


def _build_prompt(payload: ValidatorInput) -> str:
    return (
        "You are validating sports bets after a quantitative model has already generated them.\n"
        "Return strict JSON with this shape:\n"
        '{"game_id":"string","results":[{"bet_type":"moneyline|spread|total","pick_side":"string",'
        '"verdict":"UNCHANGED|REDUCED|NO_BET|SKIPPED","edge_adjustment":0.0,'
        '"adjusted_edge":0.0,"kelly_multiplier":1.0,"reasons":["short reason"]}],'
        '"tokens_used":{"input":0,"output":0},"estimated_cost_usd":0.0}\n'
        "Rules:\n"
        "- Only use verdicts UNCHANGED, REDUCED, NO_BET, or SKIPPED.\n"
        "- Keep reasons short and factual.\n"
        f"- Never adjust edge by more than {settings.agent_max_edge_adjustment:.2f} in absolute value.\n"
        "- Use REDUCED when information adds risk but does not fully kill the pick.\n"
        "- Use NO_BET only when the available findings materially undermine the pick.\n"
        "Payload:\n"
        f"{payload.model_dump_json(indent=2, exclude_none=True)}"
    )


def _extract_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if "```" in cleaned:
        start = cleaned.find("```")
        body_start = cleaned.find("\n", start)
        if body_start == -1:
            body_start = start + 3
        else:
            body_start += 1
        end = cleaned.find("```", body_start)
        cleaned = cleaned[body_start:end if end != -1 else None].strip()

    if not cleaned.startswith("{"):
        brace = cleaned.find("{")
        if brace != -1:
            cleaned = cleaned[brace:]

    if not cleaned.startswith("{"):
        return cleaned

    depth = 0
    end_pos = -1
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    if end_pos != -1:
        cleaned = cleaned[: end_pos + 1]
    return cleaned
