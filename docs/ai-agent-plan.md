# Plan: Lean AI Validator Layer

## Deliverable
Implement a cheap, effective post-picks validator that uses search plus a single Gemini pass to reduce or kill risky bets without disturbing the core quant model.

## Status
This document now reflects the implementation in the repo.

Implemented areas:
- `scripts/picks.py` post-picks validation flow
- `src/betting_agent/intelligence/validator/` package
- validator config in `src/betting_agent/config.py`
- validator persistence via `agent_validations`
- CLI and Discord display of validator verdicts
- tests for validator behavior and display

Not yet verified in this environment:
- applying the Alembic migration to a live PostgreSQL database

---

## 1. Goal and Approach

The validator is a **post-model**, **post-pick-generation** safety layer.

Current live flow:
```text
Load models → Fetch odds → Build features → Predict → Generate picks → [Validator] → Output / Save / Discord
```

Design intent:
- keep the quant model as the primary signal
- use real-time search to catch late-breaking news
- use one cheap reasoning call per validated game
- fail open when APIs or budget checks fail

This intentionally replaces the earlier 3-agent / LangGraph concept with a leaner game-level validator.

---

## 2. Validation Scope

Validation runs at the **game level**, not once per individual pick in isolation.

Process:
1. Generate `BetCandidate[]` as usual
2. Group candidates by `external_id` or `game_id`
3. Rank games by the highest pick edge in each game
4. Validate only the selected games
5. Apply verdicts back to each pick in that game

Supported modes:
- `off`: no validation
- `top`: validate only the top `N` games
- `all`: validate all games that produced picks

Current CLI flags in `scripts/picks.py`:
- `--agent-mode off|top|all`
- `--agent-max-games N`

---

## 3. Validator Architecture

Implementation location:
```text
src/betting_agent/intelligence/validator/
  __init__.py
  schemas.py
  deterministic_checks.py
  search.py
  llm_validator.py
  costs.py
  orchestrator.py
```

Execution stages:

### Stage 1: Deterministic Checks
Run cheap local checks first.

Implemented:
- NBA back-to-back warning
- NHL back-to-back warning
- NFL weather warning when weather data already exists in metadata

These checks produce structured flags used by the validator prompt.

### Stage 2: Search
Use Tavily search for a small number of targeted queries per game.

Current default:
- `agent_search_queries_per_game = 2`

Current query pattern:
- shared query:
  - `"{away_team} {home_team} injury report lineup starter today"`
- sport-specific query:
  - NHL: `"{away_team} {home_team} starting goalie today"`
  - NBA: `"{away_team} {home_team} starting lineup today"`
  - NFL: `"{away_team} {home_team} starting quarterback injury report today"`
  - MLB: `"{away_team} {home_team} probable pitcher today"`

Search output is normalized into structured findings with:
- headline
- detail
- source URL
- published timestamp when available
- freshness estimate
- relevance score when available

### Stage 3: Single LLM Validation
Use one Gemini call per validated game.

Current implementation:
- provider: direct Gemini API
- model default: `gemini/gemini-2.5-flash`
- output: strict JSON

The prompt includes:
- game summary
- all picks for the game
- deterministic flags
- search findings

Allowed verdicts:
- `UNCHANGED`
- `REDUCED`
- `NO_BET`
- `SKIPPED`

---

## 4. Verdict Semantics

### `UNCHANGED`
Keep the pick as-is.

### `REDUCED`
Keep the pick, but reduce exposure.

Current implementation:
- `kelly_fraction *= kelly_multiplier`
- `recommended_bet *= kelly_multiplier`

### `NO_BET`
Remove the pick from:
- CLI output
- DB pick save
- Discord output

### `SKIPPED`
Leave the pick unchanged because validation was not attempted or failed.

Examples of skip reasons:
- not selected for validation
- daily validator budget reached
- validator request failed

---

## 5. Edge Adjustment Rules

The validator may adjust edge, but it is bounded.

Current config:
- `agent_max_edge_adjustment = 0.03`

Implementation behavior:
- edge adjustment is clamped to `[-0.03, +0.03]`
- adjusted edge is rounded to 6 decimals
- edge never drops below `0.0`

The validator is advisory. It should not overpower the quant model.

---

## 6. Data Structures

### Validator Input
The validator runs on a game payload, not a single-pick payload.

Core fields:
- game id / external id
- sport
- game date
- home team
- away team
- list of picks
- deterministic flags
- search findings

Implemented in:
- `src/betting_agent/intelligence/validator/schemas.py`

### BetCandidate Extensions
The validator annotates `BetCandidate` with:
- `original_edge`
- `agent_verdict`
- `agent_reasons`
- `agent_adjusted_bet`
- `agent_adjusted_kelly`
- `agent_cost_usd`

Implemented in:
- `src/betting_agent/intelligence/picks.py`

---

## 7. Cost Strategy

The validator is optimized for “cheap enough to run daily.”

Current implementation uses:
- Tavily for search
- Gemini 2.5 Flash for reasoning

Relevant current pricing assumptions used by the design:
- Gemini 2.5 Flash:
  - `$0.30 / 1M` input tokens
  - `$2.50 / 1M` output tokens
- Gemini 2.5 Pro:
  - `$1.25 / 1M` input tokens
  - `$10.00 / 1M` output tokens
- Tavily:
  - `1,000` free credits/month
  - paid usage after free credits

Current config:
```python
gemini_api_key: str = ""
tavily_api_key: str = ""
agent_enabled: bool = False
agent_mode: str = "top"
agent_max_games_per_run: int = 5
agent_search_queries_per_game: int = 2
agent_model: str = "gemini/gemini-2.5-flash"
agent_premium_model: str = "gemini/gemini-2.5-pro"
agent_enable_premium_escalation: bool = False
agent_premium_max_games_per_day: int = 1
agent_max_edge_adjustment: float = 0.03
agent_daily_budget_usd: float = 0.50
agent_monthly_budget_target_usd: float = 15.0
agent_request_timeout: int = 20
```

Current enforcement:
- daily spend is computed from `agent_validations` rows for the target date
- if the estimated run would exceed daily budget, the game is marked `SKIPPED`
- the picks pipeline never crashes due to validator budget or request failure

Cost estimation implementation:
- `src/betting_agent/intelligence/validator/costs.py`

---

## 8. Persistence

The validator stores per-pick validation outcomes in a dedicated table.

Current table:
- `agent_validations`

Fields:
- `game_id`
- `external_id`
- `pick_date`
- `sport`
- `bet_type`
- `pick_side`
- `verdict`
- `original_edge`
- `adjusted_edge`
- `reasons_json`
- `input_tokens`
- `output_tokens`
- `cost_usd`
- `validated_at`

ORM model:
- `src/betting_agent/db/models.py`

Migration:
- `src/betting_agent/db/migrations/versions/e4d5b8b7d9c1_add_agent_validations_table.py`

Important note:
- the migration was created but could not be applied in this environment because PostgreSQL was not reachable from the configured `DATABASE_URL`

---

## 9. UI / Output Behavior

### CLI
`format_picks_cli()` now shows:
- original edge vs adjusted edge
- verdict
- short validator reasons
- validator summary line with validated games, skipped games, and total cost

Example shape:
```text
  Validator: 3 games  |  Skipped: 2  |  Cost: $0.0142

  #1 ...
  Edge: +6.0% -> +4.0%
  Verdict: REDUCED
  Why: lineup uncertainty, starter news stale
```

### Discord
Pick embeds now include:
- adjusted edge when changed
- verdict
- short reasons

Summary embed now includes:
- validated games
- skipped games
- validator cost

---

## 10. Files Changed

### New
- `src/betting_agent/intelligence/validator/__init__.py`
- `src/betting_agent/intelligence/validator/schemas.py`
- `src/betting_agent/intelligence/validator/deterministic_checks.py`
- `src/betting_agent/intelligence/validator/search.py`
- `src/betting_agent/intelligence/validator/llm_validator.py`
- `src/betting_agent/intelligence/validator/costs.py`
- `src/betting_agent/intelligence/validator/orchestrator.py`
- `src/betting_agent/db/migrations/versions/e4d5b8b7d9c1_add_agent_validations_table.py`

### Modified
- `scripts/picks.py`
- `src/betting_agent/config.py`
- `src/betting_agent/db/models.py`
- `src/betting_agent/db/queries.py`
- `src/betting_agent/intelligence/picks.py`
- `src/betting_agent/notifications/discord.py`
- `tests/test_picks.py`
- `tests/test_discord.py`
- `tests/test_validator.py`

---

## 11. Verification

Completed:
- targeted validator-related tests
- full repository test suite

Full suite result:
- `226 passed`
- `1 warning`

Command used:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v
```

Lint checks completed for the validator-related code:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check ...
```

Not completed:
- `uv run alembic upgrade head`

Result:
- failed due to database connectivity / `psycopg2.OperationalError`

---

## 12. Follow-Up Work

Reasonable next steps:
- bring up the configured PostgreSQL instance and apply the migration
- run a live validator smoke test with real `GEMINI_API_KEY` and `TAVILY_API_KEY`
- decide whether to keep the legacy Ollama analysis path long-term
- add richer deterministic rules only after adding reliable live structured data for them

## 13. Summary

The implemented design is a **lean game-level validator**, not a multi-agent orchestration layer.

It is designed to:
- stay cheap
- stay resilient
- improve pick quality by filtering late-breaking risk
- preserve the current workflow when disabled

This document should be treated as the source of truth for the current validator implementation.
