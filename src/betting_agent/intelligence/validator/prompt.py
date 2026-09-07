"""
Shared prompt + output schema for the validator providers.

Both the Gemini API provider and the local `claude -p` provider receive the
same `ValidatorInput` and must return a `GameValidationResult`. Game-market
picks and player props are described together so one prompt serves both.
"""

from __future__ import annotations

from betting_agent.config import settings
from betting_agent.intelligence.validator.schemas import GameValidationResult, ValidatorInput

#: JSON schema handed to `claude -p --json-schema` so the CLI enforces the
#: output shape. Pydantic generates it from the result model.
GAME_VALIDATION_JSON_SCHEMA: dict = GameValidationResult.model_json_schema()


def system_prompt(has_props: bool, web_search: bool = False) -> str:
    """The rules. Sent as the system prompt to `claude -p`, prepended for Gemini."""
    lines = [
        "You are validating sports bets after a quantitative model has already generated them.",
        "Return strict JSON with this shape:",
        '{"game_id":"string","results":[{"bet_type":"moneyline|spread|total|prop",'
        '"pick_side":"string","player":"string or null","market":"string or null",'
        '"verdict":"UNCHANGED|REDUCED|NO_BET|SKIPPED","edge_adjustment":0.0,'
        '"adjusted_edge":0.0,"kelly_multiplier":1.0,"reasons":["short reason"]}],'
        '"tokens_used":{"input":0,"output":0},"estimated_cost_usd":0.0}',
        "Rules:",
        "- Return one result per pick in the payload, echoing its bet_type, pick_side, and "
        "(for props) player and market exactly.",
        "- Only use verdicts UNCHANGED, REDUCED, NO_BET, or SKIPPED.",
        "- Keep reasons short and factual; cite the source (feed, headline) when one exists.",
        f"- Never adjust edge by more than {settings.agent_max_edge_adjustment:.2f} "
        "in absolute value.",
        "- Use REDUCED when information adds risk but does not fully kill the pick.",
        "- Use NO_BET only when the available findings materially undermine the pick.",
        "- Do not re-derive the model. Your job is information the model cannot see: "
        "injury designations, a quarterback change, a trade or suspension, a new "
        "offensive coordinator, weather, or a line the market has moved sharply.",
    ]
    if has_props:
        lines += [
            "Player props: each prop pick is over/under a player's stat line "
            "(market player_receptions = receptions, player_reception_yds = receiving yards). "
            "projection_mean is the model's expected value, projection_games how many games it "
            "rests on, recent_values the player's last games of that stat (oldest first). "
            "deterministic_flags already encode the official injury report and QB1 status; "
            "treat a player_injury flag with drop=true as decisive.",
        ]
    if web_search:
        lines += [
            "You may use web search. Search ONLY for this week's news on the listed players "
            "and teams (injury status, trade, QB change, suspension, coaching change). Do not "
            "search for odds, picks, or predictions. Cite what you found in reasons.",
        ]
    return "\n".join(lines)


def user_message(payload: ValidatorInput) -> str:
    return "Payload:\n" + payload.model_dump_json(indent=2, exclude_none=True)


def build_prompt(payload: ValidatorInput, web_search: bool = False) -> str:
    """Single-string form (rules + payload) for providers without a system slot."""
    has_props = any(p.bet_type == "prop" for p in payload.picks)
    return system_prompt(has_props, web_search) + "\n" + user_message(payload)
