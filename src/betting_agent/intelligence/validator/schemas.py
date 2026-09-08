from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class DeterministicFlag(BaseModel):
    type: str
    team: str | None = None
    severity: str = "low"
    detail: str
    player: str | None = None   # props: the player the flag is about
    drop: bool = False          # policy says the pick should not be made


class SearchFinding(BaseModel):
    category: str
    team: str | None = None
    headline: str
    detail: str
    source_url: str
    published_at: datetime | None = None
    freshness_hours: float | None = None
    relevance_score: float | None = None


class CandidateValidationInput(BaseModel):
    bet_type: str
    pick_side: str
    model_prob: float
    implied_prob: float
    edge: float
    odds: int
    kelly_fraction: float
    recommended_bet: float
    # Props only
    player: str | None = None
    team: str | None = None     # the player's CURRENT club (official roster + the book's board)
    market: str | None = None
    line: float | None = None
    projection_mean: float | None = None
    projection_games: int | None = None
    recent_values: list[float] | None = None   # last 8 games of the modeled stat


class ValidatorInput(BaseModel):
    game_id: str
    external_id: str | None = None
    sport: str
    game_date: date
    home_team: str
    away_team: str
    picks: list[CandidateValidationInput]
    deterministic_flags: list[DeterministicFlag] = Field(default_factory=list)
    search_findings: list[SearchFinding] = Field(default_factory=list)
    spread_line: float | None = None   # home-team spread from the published schedule
    total_line: float | None = None


class ValidationPickResult(BaseModel):
    bet_type: str
    pick_side: str
    player: str | None = None
    market: str | None = None
    verdict: str
    edge_adjustment: float = 0.0
    adjusted_edge: float
    kelly_multiplier: float = 1.0
    reasons: list[str] = Field(default_factory=list)


class UsageTokens(BaseModel):
    input: int = 0
    output: int = 0


class GameValidationResult(BaseModel):
    game_id: str
    results: list[ValidationPickResult]
    tokens_used: UsageTokens = Field(default_factory=UsageTokens)
    estimated_cost_usd: float = 0.0
