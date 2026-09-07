from __future__ import annotations

import json
import logging
import re
import time

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

        resp = self._request_with_retries(
            method="get",
            url=self.MODEL_URL.format(model=self.model),
            params={"key": self.api_key},
            request_label=f"availability check for model {self.model}",
        )
        if resp is None:
            self._availability = False
            return False

        self._availability = True
        return True

    def validate(self, payload: ValidatorInput) -> GameValidationResult | None:
        if not self.is_available():
            return None

        prompt = _build_prompt(payload)
        resp = self._request_with_retries(
            method="post",
            url=self.BASE_URL.format(model=self.model),
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
            request_label=f"request for {payload.game_id}",
        )
        if resp is None:
            return None
        data = resp.json()

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

    def _request_with_retries(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, str],
        request_label: str,
        json: dict | None = None,
    ):
        attempts = max(1, settings.agent_request_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                if method == "get":
                    response = requests.get(
                        url,
                        params=params,
                        timeout=settings.agent_request_timeout,
                    )
                elif method == "post":
                    response = requests.post(
                        url,
                        params=params,
                        json=json,
                        timeout=settings.agent_request_timeout,
                    )
                else:
                    raise ValueError(f"Unsupported request method: {method}")

                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                sanitized_error = self._sanitize_error(exc)
                if attempt < attempts and _is_retryable_request_exception(exc):
                    delay = settings.agent_request_retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini validator %s transient failure on attempt %d/%d, retrying in %.1fs: %s",
                        request_label,
                        attempt,
                        attempts,
                        delay,
                        sanitized_error,
                    )
                    time.sleep(delay)
                    continue

                if method == "get":
                    logger.warning(
                        "Gemini validator unavailable for model %s: %s",
                        self.model,
                        sanitized_error,
                    )
                else:
                    logger.warning(
                        "Gemini validator %s failed: %s",
                        request_label,
                        sanitized_error,
                    )
                return None

    def _sanitize_error(self, exc: requests.RequestException) -> str:
        text = str(exc)
        if self.api_key:
            text = text.replace(self.api_key, "***")
        return _redact_query_secrets(text)


def _build_prompt(payload: ValidatorInput) -> str:
    # Shared with the claude CLI provider; kept under the old name for callers.
    from betting_agent.intelligence.validator.prompt import build_prompt

    return build_prompt(payload)


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


def _is_retryable_request_exception(exc: requests.RequestException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True

    if isinstance(exc, requests.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        return status_code in {408, 429, 500, 502, 503, 504}

    return False


def _redact_query_secrets(text: str) -> str:
    return re.sub(r"([?&](?:key|api_key)=)[^&\s]+", r"\1***", text)
