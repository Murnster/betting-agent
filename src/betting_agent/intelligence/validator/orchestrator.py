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
    DeterministicFlag,
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
    shadow: bool = False
    records: list[AgentValidationRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "validated_games": self.validated_games,
            "skipped_games": self.skipped_games,
            "total_cost_usd": self.total_cost_usd,
            "shadow": self.shadow,
        }


def make_validator(model: str | None = None):
    """
    Provider from the `agent_model` prefix: `claude/<model>` runs the local
    `claude -p` CLI, anything else (`gemini/<model>`) calls the Gemini API.
    """
    name = model or settings.agent_model
    if name.lower().startswith("claude/"):
        from betting_agent.intelligence.validator.claude_cli import ClaudeCliValidator

        return ClaudeCliValidator(model=name)
    return GeminiValidator(model=name)


def validate_picks(
    candidates: list[BetCandidate],
    sport: str,
    history_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    mode: str | None = None,
    max_games: int | None = None,
    shadow: bool | None = None,
    injuries: pd.DataFrame | None = None,
    qb1: dict | None = None,
    player_teams: dict[str, str] | None = None,
    game_lines: dict[str, tuple[float | None, float | None]] | None = None,
) -> tuple[list[BetCandidate], AgentValidationSummary]:
    """
    Run the LLM validator over the picks, grouped by game.

    Shadow mode (`settings.agent_shadow`, default on): the provider still
    runs and every verdict is recorded — on the candidate, in the summary
    records for `agent_validations`, on the pick card and in Discord — but
    edge, Kelly fraction and recommended bet are left untouched and NO_BET
    picks are kept. Flip `AGENT_SHADOW=false` once the paper trade shows
    the verdicts add value.

    `game_lines` maps a group key (external_id) to (spread_line, total_line)
    from the published schedule; `injuries`/`qb1`/`player_teams` feed the
    deterministic prop flags.
    """
    shadow = settings.agent_shadow if shadow is None else shadow
    if not candidates:
        return candidates, AgentValidationSummary(shadow=shadow)

    resolved_mode = (mode or settings.agent_mode).lower()
    if resolved_mode == "off":
        return candidates, AgentValidationSummary(
            skipped_games=len(_group_candidates(candidates)), shadow=shadow
        )

    groups = _group_candidates(candidates)
    ranked_groups = sorted(
        groups.values(), key=lambda items: max(p.edge for p in items), reverse=True
    )
    if resolved_mode == "top":
        ranked_groups = ranked_groups[: max_games or settings.agent_max_games_per_run]

    metadata_map = _metadata_map(metadata_df)
    search_client = TavilySearchClient()
    validator = make_validator()
    summary = AgentValidationSummary(shadow=shadow)
    validated_keys = {_group_key(group[0]) for group in ranked_groups}
    survivors: list[BetCandidate] = []
    validator_available = validator.is_available()

    for original in candidates:
        original.original_edge = original.edge
        # Deterministic flags applied upstream (injury policy) may already
        # have left reasons on the card; keep them.
        original.agent_reasons = list(original.agent_reasons or [])

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

        flags = build_deterministic_flags(
            group, sport, history_df=history_df, metadata_map=metadata_map,
            injuries=injuries, qb1=qb1, player_teams=player_teams,
        )
        flags.extend(_flags_from_candidates(group, flags))
        findings = []
        for query, team_hint in build_queries(sport, game.home_team, game.away_team)[
            : settings.agent_search_queries_per_game
        ]:
            findings.extend(search_client.search(query, team_hint=team_hint))

        spread_line, total_line = (game_lines or {}).get(_group_key(game), (None, None))
        payload = ValidatorInput(
            game_id=str(game.game_id or game.external_id or f"{game.away_team}@{game.home_team}"),
            external_id=game.external_id,
            sport=sport,
            game_date=game.event_date,
            home_team=game.home_team,
            away_team=game.away_team,
            picks=[_candidate_input(pick) for pick in group],
            deterministic_flags=flags,
            search_findings=findings,
            spread_line=spread_line,
            total_line=total_line,
        )
        result = validator.validate(payload)
        if result is None:
            summary.skipped_games += 1
            _mark_group(group, "SKIPPED", ["validator request failed"])
            # A call can fail after spending (the CLI's --max-budget-usd
            # kills it mid-search); book that spend so the daily gate sees it.
            wasted = float(getattr(validator, "last_call_cost_usd", 0.0) or 0.0)
            if wasted > 0:
                summary.total_cost_usd += wasted
                summary.records.extend(_failed_call_records(group, wasted))
            continue

        summary.validated_games += 1
        summary.total_cost_usd += result.estimated_cost_usd
        summary.records.extend(_apply_result(group, result, shadow=shadow))

    for candidate in candidates:
        if _group_key(candidate) not in validated_keys and candidate.agent_verdict is None:
            candidate.agent_verdict = "SKIPPED"
            candidate.agent_reasons = ["not selected for validation"]

        if shadow or candidate.agent_verdict != "NO_BET":
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


def _candidate_input(pick: BetCandidate) -> CandidateValidationInput:
    extra = pick.extra or {}
    recent = extra.get("recent_values")
    return CandidateValidationInput(
        bet_type=pick.bet_type,
        pick_side=pick.pick_side,
        model_prob=pick.model_prob,
        implied_prob=pick.implied_prob,
        edge=pick.edge,
        odds=pick.odds,
        kelly_fraction=pick.kelly_fraction,
        recommended_bet=pick.recommended_bet,
        player=pick.player,
        market=pick.market,
        line=pick.line,
        projection_mean=extra.get("projection_mean"),
        projection_games=extra.get("projection_games"),
        recent_values=[float(v) for v in recent] if recent else None,
    )


def _flags_from_candidates(
    group: list[BetCandidate], already: list[DeterministicFlag]
) -> list[DeterministicFlag]:
    """Flags the injury policy attached to candidates upstream, if not rebuilt here."""
    seen = {(f.type, f.player, f.detail) for f in already}
    out: list[DeterministicFlag] = []
    for pick in group:
        for raw in (pick.extra or {}).get("flags", []):
            try:
                flag = DeterministicFlag.model_validate(raw)
            except ValueError:
                continue
            key = (flag.type, flag.player, flag.detail)
            if key not in seen:
                seen.add(key)
                out.append(flag)
    return out


def _group_candidates(candidates: list[BetCandidate]) -> dict[str, list[BetCandidate]]:
    groups: dict[str, list[BetCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(_group_key(candidate), []).append(candidate)
    return groups


def _group_key(candidate: BetCandidate) -> str:
    return str(
        candidate.external_id or candidate.game_id or f"{candidate.away_team}@{candidate.home_team}"
    )


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


def _result_key(bet_type: str, pick_side: str, player: str | None) -> tuple[str, str, str]:
    """
    Two "over" props in one game must not collide, so props key on the
    player as well. `market` is not needed: one pick per player per slate.
    """
    from betting_agent.sports.nfl.props import normalize_player

    return (bet_type, (pick_side or "").strip().lower(), normalize_player(player) if player else "")


def _apply_result(
    group: list[BetCandidate], result: GameValidationResult, shadow: bool = False
) -> list[AgentValidationRecord]:
    per_pick_cost = result.estimated_cost_usd / max(len(result.results), 1)
    records: list[AgentValidationRecord] = []
    by_key = {_result_key(p.bet_type, p.pick_side, p.player): p for p in group}

    for item in result.results:
        pick = by_key.get(_result_key(item.bet_type, item.pick_side, item.player))
        if pick is None:
            continue

        bounded_adjustment = max(
            -settings.agent_max_edge_adjustment,
            min(settings.agent_max_edge_adjustment, item.edge_adjustment),
        )
        pick.original_edge = pick.original_edge if pick.original_edge is not None else pick.edge
        proposed_edge = round(max(0.0, pick.original_edge + bounded_adjustment), 6)
        multiplier = 1.0
        if item.verdict == "REDUCED":
            multiplier = min(max(item.kelly_multiplier, 0.0), 1.0)

        pick.agent_verdict = item.verdict
        pick.agent_reasons = item.reasons[:3]
        pick.agent_cost_usd = per_pick_cost
        pick.extra["agent"] = {
            "reasons": item.reasons[:3],
            "verdict": item.verdict,
            "shadow": shadow,
            "proposed_edge": proposed_edge,
            "proposed_kelly_multiplier": multiplier,
        }

        if not shadow:
            pick.edge = proposed_edge
            pick.kelly_fraction *= multiplier
            pick.recommended_bet *= multiplier
        pick.agent_adjusted_kelly = pick.kelly_fraction * (1.0 if not shadow else multiplier)
        pick.agent_adjusted_bet = pick.recommended_bet * (1.0 if not shadow else multiplier)

        records.append(
            AgentValidationRecord(
                game_id=pick.game_id or None,
                external_id=pick.external_id,
                pick_date=pick.game_date,
                sport=pick.sport,
                bet_type=pick.bet_type,
                pick_side=_record_side(pick),
                verdict=item.verdict,
                original_edge=pick.original_edge,
                adjusted_edge=proposed_edge,
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


def _failed_call_records(group: list[BetCandidate], cost_usd: float
                         ) -> list[AgentValidationRecord]:
    """SKIPPED records carrying the spend of a failed call, split per pick."""
    per_pick = cost_usd / max(len(group), 1)
    out = []
    for pick in group:
        pick.agent_cost_usd = per_pick
        out.append(AgentValidationRecord(
            game_id=pick.game_id or None, external_id=pick.external_id,
            pick_date=pick.game_date, sport=pick.sport, bet_type=pick.bet_type,
            pick_side=_record_side(pick), verdict="SKIPPED",
            original_edge=pick.edge, adjusted_edge=pick.edge,
            reasons=["validator request failed"], input_tokens=0, output_tokens=0,
            cost_usd=per_pick,
        ))
    return out


def _record_side(pick: BetCandidate) -> str:
    """agent_validations.pick_side: for props, name the player so rows are distinct."""
    if pick.bet_type == "prop" and pick.player:
        line = f" {pick.line:g}" if pick.line is not None else ""
        return f"{pick.player} {pick.pick_side}{line}"[:100]
    return pick.pick_side
