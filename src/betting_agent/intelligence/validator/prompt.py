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
        "You are reviewing sports bets after a quantitative model has already generated them.",
        "Return strict JSON with this shape:",
        '{"game_id":"string","results":[{"bet_type":"moneyline|spread|total|prop",'
        '"pick_side":"string","player":"string or null","market":"string or null",'
        '"verdict":"UNCHANGED|REDUCED|NO_BET","edge_adjustment":0.0,'
        '"adjusted_edge":0.0,"kelly_multiplier":1.0,'
        '"why":"one or two sentences making the case for this pick"}],'
        '"tokens_used":{"input":0,"output":0},"estimated_cost_usd":0.0}',
        "Rules:",
        "- Return one result per pick in the payload, echoing its bet_type, pick_side, and "
        "(for props) player and market exactly.",
        "- `why` is the case for the pick as a bettor reads it: what the player or team has "
        "been doing lately against this number, the role and usage to expect in this game, "
        "and the matchup. Make it specific to this market and this side — an Under explains "
        "why he stays under the line, an Over why he clears it, a touchdown pick why he "
        "reaches the end zone. Two picks on one player get two different `why` texts. "
        "Never write 'no news found' or a sample-size caveat as the whole text.",
        "- Do not recompute the model's probabilities. Explain the pick, then adjust only for "
        "what the model cannot see: an injury designation, a quarterback change, a trade or "
        "suspension, a new offensive coordinator, weather, or a line the market has moved "
        "sharply. When such a finding changes your verdict, state it inside `why` as a clause "
        "and cite the source (feed, headline).",
        "- Only use verdicts UNCHANGED, REDUCED, or NO_BET. Never SKIPPED: when nothing argues "
        "against the pick, return UNCHANGED.",
        f"- Never adjust edge by more than {settings.agent_max_edge_adjustment:.2f} "
        "in absolute value.",
        "- Use REDUCED when information adds risk but does not fully kill the pick; "
        "use NO_BET only when the findings materially undermine it. A REDUCED or NO_BET "
        "verdict must name its concrete cause in `why`.",
    ]
    if has_props:
        lines += [
            "Player props: each prop pick is over/under a player's stat line "
            "(market player_receptions = receptions, player_reception_yds = receiving yards, "
            "player_rush_yds = rushing yards; a market ending in _alternate is a ladder rung — "
            "side over means the player reaches that milestone; "
            "player_anytime_td with side yes = the player scores a touchdown, line 0.5). "
            "projection_mean is the model's expected value, projection_games how many games it "
            "rests on, recent_games the player's last games of that stat with the week and "
            "opponent (oldest first; recent_values is the same series as bare numbers), "
            "position and opponent are his position and the club he faces. "
            "section says which paper book the pick belongs to: main is the tracked card, "
            "td_scorer the anytime-TD board, straight_over the book's main-line Over, ladder an "
            "alternate milestone rung — the last three are experiment pools. "
            "deterministic_flags already encode the official injury report and QB1 status; "
            "treat a player_injury flag with drop=true as decisive.",
            "Every listed player IS in this game: the sportsbook hangs these lines for this "
            "matchup and `team` is the player's current club from the official roster. "
            "Players change teams every offseason, so a source placing a player on another "
            "club is stale — it is not evidence he is absent. Never return NO_BET (or any "
            "reduction) on the grounds that a player is not part of this game.",
        ]
    if web_search:
        lines += [
            "You may use web search. Search ONLY for this week's news on the listed players "
            "and teams (injury status, role, depth chart and usage, trade, QB change, "
            "suspension, coaching change); put the player's `team` and the season in the "
            "query so stale prior-club results do not mislead you. Do not search for odds, "
            "picks, or predictions. Cite what you found in `why`.",
        ]
    return "\n".join(lines)


def user_message(payload: ValidatorInput) -> str:
    return "Payload:\n" + payload.model_dump_json(indent=2, exclude_none=True)


def build_prompt(payload: ValidatorInput, web_search: bool = False) -> str:
    """Single-string form (rules + payload) for providers without a system slot."""
    has_props = any(p.bet_type == "prop" for p in payload.picks)
    return system_prompt(has_props, web_search) + "\n" + user_message(payload)
