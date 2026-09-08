# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated sports betting ETL pipeline and prediction engine. Generates "+EV Picks" using XGBoost game models, distributional NFL player-prop models, Kelly Criterion sizing, and optional Ollama sentiment analysis. Four sports are implemented (NFL, NBA, NHL, MLB) but **the owner only cares about NFL**: NHL and MLB are frozen (`active=False` in the registry — reachable by explicit `--sport`, hidden from `available_sports()` and routine loops) and the NBA pick path is known-broken and deprioritized (see TODO.md). The Aug 2026 real-line backtest showed the NFL game-level model cannot beat closing prices, so the live NFL strategy is **player props** (`scripts/props.py`, paper-trading through the 2026 season). Runs entirely on local hardware ($0-$15/month).

Open follow-ups and deferred decisions live in `TODO.md` at the repo root — check it before starting work, and add to it rather than leaving loose ends in commit messages. `AGENTS.md` holds style conventions (Ruff, 100-char lines, test naming); don't duplicate them here.

## Development Commands

```bash
# Setup
cp .env.example .env                    # fill in ODDS_API_KEY, DATABASE_URL, etc.
uv run alembic upgrade head             # requires PostgreSQL running

# Training
uv run python scripts/train.py --sport NFL --seasons 2018 2019 2020 2021 2022 2023 2024
uv run python scripts/train.py --sport NBA --seasons 2022 2023 2024

# Reset DB from scratch (wipes all games, odds, sentiment, picks)
uv run alembic downgrade base && uv run alembic upgrade head
# Then re-seed by training without --no-seed, or run extract.py

# Picks (game markets)
uv run python scripts/picks.py --sport NFL --bankroll 1000
uv run python scripts/picks.py --sport NBA --bankroll 1000 --save

# NFL player props (paper trading; one API call per event — cap with --max-events)
uv run python scripts/props.py --save --max-events 5
uv run python scripts/props.py --pick-games          # choose games interactively (saves credits)
uv run python scripts/props.py --suggest 3           # free pre-screen, auto-picks 3 hottest games
uv run python scripts/props.py --today --save        # cron mode: today's slate only, off-days exit free
uv run python scripts/props.py --closing             # pre-kickoff: closing price/line on held picks → CLV
uv run python scripts/props.py --today --save --no-leans   # skip the 3-credit game-lean call
uv run python scripts/props.py --today --save --agent-mode all   # LLM validator on every game (shadow)
uv run python scripts/replay.py --random             # dress-rehearse a past week (synthetic lines)
uv run python scripts/replay.py --random-sunday --suggest 4   # weekend rehearsal: heat board + top-4 games
uv run python scripts/replay.py --date 2025-12-28 --suggest 4 # same, for a specific calendar day
uv run python scripts/props_diagnostic.py            # walk-forward pick-level backtest (sets the floors)
uv run python scripts/props_calibration.py --train-seasons 2020 2021 2022 2023 --eval-seasons 2024 2025

# Ledger / recording actual bets placed at the book
uv run python scripts/bets.py list
uv run python scripts/bets.py set <pick_id> --stake 25 --odds -115
uv run python scripts/bets.py ledger

# Daily extraction (morning → closing → postgame → grade)
uv run python scripts/extract.py morning --sport NBA
uv run python scripts/extract.py closing --sport NBA
uv run python scripts/extract.py postgame --sport NBA
uv run python scripts/grade.py

# Backtesting
uv run python scripts/backtest.py --sport NFL --start-season 2019 --end-season 2023

# Tests
uv run pytest tests/ -v                 # all tests (~480 tests, ~8s)
uv run pytest tests/test_ev.py -v       # single file
uv run pytest tests/test_ev.py::test_name -v  # single test

# Reports / diagnostics
uv run python scripts/report.py --sport NFL          # ROI + CLV + bankroll report
uv run python scripts/model_vs_market.py             # game-model vs closing-line diagnostic
uv run python scripts/game_model_gate.py --start-season 2012 --end-season 2025   # market-anchored game model vs the close (fails: see TODO)
uv run python scripts/backup_db.py                   # pg_dump to backups/postgres/, prunes >30d

# Lint (scripts/ is linted too)
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/         # auto-fix

# Install packages
uv add <package>
```

## Architecture

Three layers: **Extraction** (data loaders + Odds API + weather) → **Intelligence** (features + models + EV + Kelly + sentiment + optional LLM validator in `intelligence/validator.py`) → **Accounting** (grading + CLV + ROI + bankroll ledger in `accounting/ledger.py`).

### NFL Props (Phase 3)

`sports/nfl/props.py` projects receptions (negative binomial) and receiving yards (shifted lognormal, level-dependent residuals) from nflreadpy weekly stats: shrunk exponentially-weighted player means × opponent-position defense factor, with an isotonic P(over) calibration layer. Always call `tune_dispersion()` after `fit()`. Prop odds come from the per-event endpoint (`OddsAPIClient.fetch_event_odds()` — props are NOT on the sport-level `/odds` endpoint). Prop picks grade from player stats via `grade_prop_picks()`; a player missing from a published week's stats voids the pick (result `"void"`, pnl 0), while an unpublished week leaves it ungraded. **Grading finalizes NFL games itself** via `sports/nfl/results.py finalize_nfl_games()` (free, from nflreadpy) — props.py writes its Game rows as `"scheduled"` and grading requires `"final"`, so without this nothing settles.

Selection policy (set by `scripts/props_diagnostic.py`, a walk-forward pick-level backtest — re-run it after any projection change): per-market edge floors in `PROP_EDGE_FLOORS` (10% receptions, 15% receiving yards; `--min-edge` overrides), one pick per player per slate (both markets on one player are near-duplicate bets), and same-game Kelly scaling. The isotonic calibrator trains on `book_proxy_line()` placements as well as pseudo-lines, because real lines sit above our shrunk projection. Residual overconfidence is ~5pp, which the floors are sized to cover.

### In-season NFL props loop (what actually runs Sep–Feb)

Everything except the per-event odds fetch is free. The Odds API free tier is 500 credits/month; `fetch_events()` costs nothing, `fetch_event_odds()` costs one credit **per market** (two per game for the two modeled markets), so a full 16-game slate is ~32 credits and `--suggest N` / `--max-events N` bound the spend. Up to 10 bookmakers in one request still count as one region, so requesting the whole fallback list is free.

**The Odds API has no bet365 player-prop feed.** bet365 posted no receiving props in any region on the 2026 opener while DraftKings, FanDuel, BetOnline, BetRivers, Bovada (US) and Pinnacle/Unibet/Sportsbet (EU/AU) all did. `prop_bookmaker_order()` therefore requests the preferred book (`PREFERRED_BOOKMAKERS`, default bet365) **plus** `PROP_FALLBACK_BOOKMAKERS` (default `draftkings,fanduel`), and `books_in_preference()` prices each game against the **first** book in that order that posted — one book per game, never best-of-N across books (that would inflate edges past what the floors were set on). The pick card shows the book; the user checks bet365's number by hand before staking and records the real price with `bets.py set --odds`. `--closing` reads the same book order so CLV compares like with like.

1. Game day (Thu/Sun/Mon): `props.py --today --save [--suggest N]`. Output is **one card per slate** (`intelligence/slate.py`: Thursday/Sunday Early/Sunday Late/Sunday Night/Monday by Eastern kickoff). A single-game slate carries 1 game lean + up to 2 props; a Sunday window 3 leans + 3 props, chosen top-down by edge. Everything that clears the floors is still saved (`Pick.on_card` marks the card). The **game lean** (`intelligence/game_lean.py`) is not a model pick: Pinnacle's vig-removed price is the fair value, converted to the bettable book's line through the gate-measured margin/total sigmas, and the lean is the best-edge side at that book. It is labelled LEAN on the card and Discord, saved as a paper pick (stake 0 when edge ≤ 0), and `--closing` captures its close so a season of CLV decides whether early numbers beat closing ones. One sport-level odds call (`fetch_odds(..., bookmakers=)`) = 3 credits per run regardless of game count; `--no-leans` skips it. `--today` matches kickoff to the **machine's local calendar day** (this box is America/Halifax, UTC−3, so US primetime still lands on its own day) and drops games that have already kicked off, so a re-run never refreshes saved picks with in-play prices. Off-days exit before any model fitting or paid call. After the models fit, the run loads free season context: the published roster (`current_teams()` — offseason movers land on their new side; 72 re-teamed in Sep 2026), the schedule (`nfl_week_for()` gives the exact week and spread/total for the validator payload), the official injury report and depth charts (`sports/nfl/injuries.py`). Out/Doubtful players are dropped; Questionable and a QB1 who is Out are attached as flags on the pick card. Then the LLM validator runs if `AGENT_ENABLED` (shadow mode by default — see below).
2. Pre-kickoff: `props.py --closing [--window-minutes 90]`. Also captures held game leans (3 credits for all of them, same book order + Pinnacle). Free events call → held ungraded prop picks in games kicking off inside the window → per-event odds fetch **only for those games** → `closing_line`/`closing_odds` stored on each pick; `clv` is set only when the line held (a price at a different number is not comparable) and `report.py` counts line moves for/against otherwise. Zero credits when nothing is due. This is the only source of CLV for props — `update_clv_for_picks()` reads the game-market `odds` table, and `grade.py --date` deliberately leaves prop CLV alone on reset.
3. Next morning: `grade.py`. It finalizes NFL Game rows from nflreadpy schedules, then grades props from `load_player_stats()`. Week N's `stats_player_week_<season>.parquet` doesn't exist on nflverse until week N is played — `load_player_stats()` tolerates a missing season (warns, continues), and a pending pick simply stays ungraded until the parquet appears.
4. Weekly: `report.py --sport NFL` / `bets.py ledger`. Real stakes go through `bets.py set`.

**Validator (shadow).** `AGENT_MODEL` picks the provider by prefix: `claude/<model>` runs the local `claude -p` CLI (`intelligence/validator/claude_cli.py`), `gemini/<model>` the Gemini API. With `AGENT_SHADOW=true` (default) verdicts are recorded to `agent_validations`, shown on pick cards/Discord as `SHADOW`, but never change edge, sizing, or the slate — flip it only once `agent_validations.verdict` vs `picks.result` shows the verdicts add value. The CLI provider replaces the system prompt with the rules, passes the payload on stdin, loads no settings, and runs from a temp dir (~250 input tokens + payload; a default run from the repo would load ~29k tokens of CLAUDE.md/memory). Do NOT use `--bare` — it skips keychain reads and reports "Not logged in". WebSearch needs both `--tools WebSearch` and `--allowedTools WebSearch` in print mode. Measured Sep 2026 on Sonnet: ~$0.014/game without search, ~$0.09 with for a one-pick game, **$0.42 for a six-pick game with search** (the model searches per player). `--max-budget-usd` caps each call at `AGENT_CLAUDE_MAX_CALL_USD` (0.25) and `AGENT_DAILY_BUDGET_USD` (1.00) gates the day; a call the CLI kills on its cap exits non-zero **after spending**, so the provider reads `total_cost_usd` from the error envelope (`last_call_cost_usd`) and the orchestrator books it as SKIPPED records — otherwise the daily gate never sees the money. Prop results are keyed by `(bet_type, pick_side, player)` — two "over" props in one game used to collide.

nflreadpy gates `load_injuries`/`load_rosters_weekly` on its own `get_current_season()`, which flips to the new season on the Thursday after Labor Day — requesting the new season before then raises `ValueError: Season must be between …`. `load_schedules` and `load_rosters` accept the new season earlier. Week-1 projections therefore rest entirely on the prior two seasons of stats.

`scripts/daily_workflow.sh` / `setup_cron.sh` are the older game-market (NBA/NHL) automation and are not what the props loop uses; no crontab is installed as of Sep 2026 (the user runs the sequence above by hand; the proposed schedule is in TODO.md).

### Multi-Sport Registry

All scripts accept `--sport NFL|NBA` and dispatch through `sports/registry.py`. `get_sport_config("NFL")` returns a `SportConfig` dataclass with: `loader_cls`, `build_features` (callable), `split_features_targets` (callable), `sport_key` (Odds API key), `default_seasons`, `hist_avg_total`, `total_stdev`. To add a new sport: create `sports/<sport>/loader.py` + `features.py`, register a `SportConfig` in the registry, and all scripts pick it up.

### Data Flow Through the Pipeline

1. **Loaders** return Polars DataFrames (`nflreadpy` / `nba_api`)
2. Scripts call `.to_pandas()` at the call site — all feature engineering operates on **pandas**
3. `build_features(raw_df)` runs: column normalization → Elo (8 variants) → rolling averages → streaks → H2H → rest days → sport-specific features → OHE → leakage column drops
4. `split_features_targets(df)` filters to completed games, returns `(X, y)` where y has `home_team_wins`, `home_score`, `away_score`
5. `PredictionEngine.predict(X)` aligns features via `X.reindex(columns=feature_names, fill_value=0)` and returns `win_prob`, `home_pred_score`, `away_pred_score`, `pred_total`, `pred_margin`

### Model Artifacts

Saved to `saved_models/<SPORT>/`: `classifier.json`, `calibrator.joblib`, `home_regression.joblib`, `away_regression.joblib`, `feature_names.pkl`, `scoring_sigma.pkl`, `feature_importance.json`, `calibration_report.json`, `calibration_bins.csv`. The calibrator (`IsotonicEnsemble`) is trained via k-fold expanding-window isotonic calibration during `train_calibrated_classifier()` using an 80/20 train+calibration/test split. Each fold fits an `IsotonicRegression` on an expanding window; the ensemble averages their predictions. Probabilities are clipped to [0.05, 0.95] at predict time. `scoring_sigma.pkl` contains `{"total_sigma": float, "margin_sigma": float}` — the empirical standard deviation of regression residuals, computed during training on a held-out 20% temporal split. Used instead of hardcoded sigma for totals/spread normal CDF probability calculations. Totals/spread probabilities are clipped to [0.10, 0.90].

## Critical Gotchas

**Column normalization is mandatory for NFL data.** `nflreadpy` uses `gameday` (not `game_date`), `game_type` (not `is_playoff`), `location` (not `neutral_site`); its weather columns are `temp`/`wind`. `normalise_raw_schedules()` in `sports/nfl/features.py` handles all renames — it's called automatically by `build_nfl_features()`, but if you pass raw nflreadpy data anywhere else, you must normalize first.

**OHE feature alignment.** One-hot encoding generates different columns per training run depending on which teams appear. `feature_names.pkl` captures the exact column list. At predict time, `X.reindex(columns=feature_names, fill_value=0)` ensures alignment — missing columns get zero-filled.

**The `_is_upcoming` tag pattern.** In `picks.py`, upcoming games (from Odds API) are combined with historical games (for Elo warm-up) into one DataFrame. Since `build_features()` sorts by `game_date` internally, upcoming rows are tagged with `_is_upcoming=True` before entering the pipeline, then retrieved by this tag after feature building.

**NBA team name bridging.** `nba_api` uses 3-letter abbreviations (`"BOS"`), the Odds API uses full names (`"Boston Celtics"`). The `NBA_ABBREV_TO_FULL` / `NBA_FULL_TO_ABBREV` dicts in `sports/nba/loader.py` handle mapping. In `picks.py`, a `home_team_odds` metadata column carries the Odds API name for odds matching while `home_team` carries the abbreviation.

**The game model cannot beat the close, even anchored on it.** `models/market_anchored.py` keeps the closing spread/total as the prior and learns only the residual (`build_nfl_features(..., market_anchored=True)` keeps `ANCHOR_COLS`, adds `market_home_prob`/`spread_abs`/QB continuity, never prices). `scripts/game_model_gate.py` walk-forwards it: ridge equals the market (Brier 0.2100 both), XGB is worse, spread ROI negative, totals within noise. It is not wired into picks.py; a game line on a card is a market lean, not a pick.

**NFL backtests use real closing lines** (`sports/nfl/market.py` from nflreadpy schedules); other sports still use synthetic odds from `_generate_market_odds()` (model-centered + Gaussian noise + 4.5% vig, never included as features). Use `--flat-stake` results to judge selection skill — Kelly ROI on small samples is sizing variance.

**The props time index `t = season*100 + week` is not contiguous across seasons.** `ReceivingPropsModel` and `props.py` order history by `t`, which is fine for "strictly before" comparisons but wrong for arithmetic windows: `t - 3` in week 1 of a new season points at a `t` no game has (e.g. 202599), so any "seen in the last K weeks" filter written as subtraction silently excludes every player at the start of a season. Use `active_player_keys()` (`sports/nfl/props.py`): it windows by rank of distinct `t` values and counts **regular-season slates only** — the postseason weeks carry two teams' worth of players and would otherwise make week 1 nearly blind again.

**nflreadpy depth charts ignore the season filter** (`load_depth_charts([2026])` returns the whole ~499k-row current-format dataset, `dt` is an ISO string) and teams publish on different days: take `dt.max()` **per team**, then `pos_abb == "QB"` (not `pos_grp`, which is a formation label like "3WR 1TE") and `pos_rank == 1`. The name column is `player_name`, not `full_name`.

**Sentiment is optional.** `is_ollama_available()` is checked before any Ollama call. If unavailable, picks generate from ML alone. Sentiment shifts the model probability (by at most `sentiment_weight`, default 0.02) before the edge is computed, so edge and Kelly sizing stay consistent.

**Season is sport-aware.** Use `SportConfig.season_for_date(date)` — an NFL/NBA game in January belongs to the previous year's season, and mistagging it triggers Elo's per-season mean reversion mid-season.

**DB is optional for training.** `train.py --no-seed` skips DB operations. DB failures during seeding are caught and logged as warnings.

**SQLAlchemy 2.x has no TIMESTAMPTZ.** The codebase uses `TIMESTAMP` instead (aliased for readability in imports).

## Configuration

All settings via pydantic-settings in `config.py`, reading from `.env`. Key tunables: `min_edge_pct` (0.015), `max_kelly_pct` (0.05), `max_bet_pct` (0.07), `sentiment_weight` (0.02), `starting_bankroll` (1000), `ollama_model` ("llama3.1:8b"), `nba_api_rate_limit` (0.6s). Validator: `agent_enabled`, `agent_model` (`claude/sonnet` in this box's `.env`), `agent_shadow` (True), `agent_claude_web_search`, `agent_claude_max_turns` (4), `agent_claude_max_call_usd` (0.25), `agent_daily_budget_usd`.

## Key Design Decisions

- **Edge-based filtering:** Only surfaces picks where model probability exceeds vig-removed implied probability by `MIN_EDGE_PCT`
- **Scaled Kelly sizing:** 40% Kelly at edge <= 3%, 60% at <= 5%, 80% above, capped at `MAX_KELLY_PCT`
- **Walk-forward backtesting:** Train on seasons [start..N-1], test on season N — no look-ahead bias
- **All compute local:** No paid LLM APIs; Ollama for sentiment, everything else is statistical
