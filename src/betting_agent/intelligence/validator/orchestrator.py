from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from betting_agent.config import settings
from betting_agent.db.models import AgentValidation
from betting_agent.db.queries import get_game_by_external_id
from betting_agent.db.session import get_session
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.costs import budget_allows
from betting_agent.intelligence.validator.deterministic_checks import build_deterministic_flags
from betting_agent.intelligence.validator.llm_validator import GeminiValidator
from betting_agent.intelligence.validator.schemas import (
    CandidateValidationInput,
    GameValidationResult,
    ValidatorInput,
)
from betting_agent.intelligence.validator.search import TavilySearchClient, build_queries

logger = logging.getLogger(__name__)


@dataclass
class AgentValidationRecord:
    game_id: int | None
    external_id: str | None
    pick_date: date
    sport: str
    bet_type: str
    pick_side: str
    verdict: str
    original_edge: float
    adjusted_edge: float
    reasons: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class AgentValidationSummary:
    validated_games: int = 0
    total_cost_usd: float = 0.0
    skipped_games: int = 0
    records: list[AgentValidationRecord] = field(default_factory=list)


def validate_picks(
    candidates: list[BetCandidate],
    sport: str,
    history_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    mode: str | None = None,
    max_games: int | None = None,
) -> tuple[list[BetCandidate], AgentValidationSummary]:
    if not candidates:
        return candidates, AgentValidationSummary()

    resolved_mode = (mode or settings.agent_mode).lower()
    if resolved_mode == "off":
        return candidates, AgentValidationSummary(skipped_games=len(_group_candidates(candidates)))

    groups = _group_candidates(candidates)
    ranked_groups = sorted(groups.values(), key=lambda items: max(p.edge for p in items), reverse=True)
    if resolved_mode == "top":
        ranked_groups = ranked_groups[: max_games or settings.agent_max_games_per_run]

    metadata_map = _metadata_map(metadata_df)
    search_client = TavilySearchClient()
    validator = GeminiValidator()
    summary = AgentValidationSummary()
    validated_keys = {_group_key(group[0]) for group in ranked_groups}
    survivors: list[BetCandidate] = []
    validator_available = validator.is_available()

    for original in candidates:
        original.original_edge = original.edge
        original.agent_reasons = []

    if not validator_available:
        for group in ranked_groups:
            summary.skipped_games += 1
            _mark_group(group, "SKIPPED", ["validator unavailable"])
        for candidate in candidates:
            if _group_key(candidate) not in validated_keys and candidate.agent_verdict is None:
                candidate.agent_verdict = "SKIPPED"
                candidate.agent_reasons = ["not selected for validation"]
            survivors.append(candidate)
        return survivors, summary

    for group in ranked_groups:
        game = group[0]
        current_cost = summary.total_cost_usd
        if not budget_allows(game.game_date, current_cost):
            summary.skipped_games += 1
            _mark_group(group, "SKIPPED", ["daily validator budget reached"])
            continue

        flags = build_deterministic_flags(group, sport, history_df=history_df, metadata_map=metadata_map)
        findings = []
        for query, team_hint in build_queries(sport, game.home_team, game.away_team)[
            : settings.agent_search_queries_per_game
        ]:
            findings.extend(search_client.search(query, team_hint=team_hint))

        payload = ValidatorInput(
            game_id=str(game.game_id or game.external_id or f"{game.away_team}@{game.home_team}"),
            external_id=game.external_id,
            sport=sport,
            game_date=game.event_date,
            home_team=game.home_team,
            away_team=game.away_team,
            picks=[
                CandidateValidationInput(
                    bet_type=pick.bet_type,
                    pick_side=pick.pick_side,
                    model_prob=pick.model_prob,
                    implied_prob=pick.implied_prob,
                    edge=pick.edge,
                    odds=pick.odds,
                    kelly_fraction=pick.kelly_fraction,
                    recommended_bet=pick.recommended_bet,
                )
                for pick in group
            ],
            deterministic_flags=flags,
            search_findings=findings,
        )
        result = validator.validate(payload)
        if result is None:
            summary.skipped_games += 1
            _mark_group(group, "SKIPPED", ["validator request failed"])
            continue

        summary.validated_games += 1
        summary.total_cost_usd += result.estimated_cost_usd
        summary.records.extend(_apply_result(group, result))

    for candidate in candidates:
        if _group_key(candidate) not in validated_keys and candidate.agent_verdict is None:
            candidate.agent_verdict = "SKIPPED"
            candidate.agent_reasons = ["not selected for validation"]

        if candidate.agent_verdict != "NO_BET":
            survivors.append(candidate)

    return survivors, summary


def save_agent_validations_to_db(records: list[AgentValidationRecord]) -> None:
    if not records:
        return

    with get_session() as session:
        for record in records:
            game_id = record.game_id
            if game_id in (None, 0) and record.external_id:
                game = get_game_by_external_id(session, record.external_id)
                game_id = game.id if game else None

            session.add(
                AgentValidation(
                    game_id=game_id,
                    external_id=record.external_id,
                    pick_date=record.pick_date,
                    sport=record.sport,
                    bet_type=record.bet_type,
                    pick_side=record.pick_side,
                    verdict=record.verdict,
                    original_edge=record.original_edge,
                    adjusted_edge=record.adjusted_edge,
                    reasons_json=record.reasons,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost_usd=record.cost_usd,
                )
            )


def _group_candidates(candidates: list[BetCandidate]) -> dict[str, list[BetCandidate]]:
    groups: dict[str, list[BetCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(_group_key(candidate), []).append(candidate)
    return groups


def _group_key(candidate: BetCandidate) -> str:
    return str(candidate.external_id or candidate.game_id or f"{candidate.away_team}@{candidate.home_team}")


def _metadata_map(metadata_df: pd.DataFrame | None) -> dict[str, dict]:
    if metadata_df is None or metadata_df.empty:
        return {}

    result: dict[str, dict] = {}
    for row in metadata_df.to_dict(orient="records"):
        key = str(row.get("external_id") or row.get("game_id") or "")
        if key:
            result[key] = row
    return result


def _mark_group(group: list[BetCandidate], verdict: str, reasons: list[str]) -> None:
    for pick in group:
        pick.agent_verdict = verdict
        pick.agent_reasons = reasons[:]


def _apply_result(
    group: list[BetCandidate], result: GameValidationResult
) -> list[AgentValidationRecord]:
    per_pick_cost = result.estimated_cost_usd / max(len(result.results), 1)
    records: list[AgentValidationRecord] = []
    by_key = {(item.bet_type, item.pick_side): item for item in group}

    for item in result.results:
        pick = by_key.get((item.bet_type, item.pick_side))
        if pick is None:
            continue

        bounded_adjustment = max(
            -settings.agent_max_edge_adjustment,
            min(settings.agent_max_edge_adjustment, item.edge_adjustment),
        )
        pick.original_edge = pick.original_edge if pick.original_edge is not None else pick.edge
        pick.edge = round(max(0.0, pick.original_edge + bounded_adjustment), 6)
        pick.agent_verdict = item.verdict
        pick.agent_reasons = item.reasons[:3]
        pick.agent_cost_usd = per_pick_cost
        pick.extra["agent"] = {"reasons": item.reasons[:3], "verdict": item.verdict}

        if item.verdict == "REDUCED":
            multiplier = min(max(item.kelly_multiplier, 0.0), 1.0)
            pick.kelly_fraction *= multiplier
            pick.recommended_bet *= multiplier

        pick.agent_adjusted_kelly = pick.kelly_fraction
        pick.agent_adjusted_bet = pick.recommended_bet

        records.append(
            AgentValidationRecord(
                game_id=pick.game_id or None,
                external_id=pick.external_id,
                pick_date=pick.game_date,
                sport=pick.sport,
                bet_type=pick.bet_type,
                pick_side=pick.pick_side,
                verdict=item.verdict,
                original_edge=pick.original_edge,
                adjusted_edge=pick.edge,
                reasons=item.reasons[:3],
                input_tokens=result.tokens_used.input,
                output_tokens=result.tokens_used.output,
                cost_usd=per_pick_cost,
            )
        )

    for pick in group:
        if pick.agent_verdict is None:
            pick.agent_verdict = "SKIPPED"
            pick.agent_reasons = ["validator returned no result for this pick"]

    return records
