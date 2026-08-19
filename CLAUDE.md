# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated sports betting ETL pipeline and prediction engine. Generates "+EV Picks" using XGBoost game models, distributional NFL player-prop models, Kelly Criterion sizing, and optional Ollama sentiment analysis. Four sports are implemented (NFL, NBA, NHL, MLB) but **NHL and MLB are frozen** (`active=False` in the registry — reachable by explicit `--sport`, hidden from `available_sports()` and routine loops) until NFL+NBA are proven. The Aug 2026 real-line backtest showed the NFL game-level model cannot beat closing prices, so the live NFL strategy is **player props** (`scripts/props.py`, paper-trading from Sep 2026). Runs entirely on local hardware ($0-$15/month).

Open follow-ups and deferred decisions live in `TODO.md` at the repo root — check it before starting work, and add to it rather than leaving loose ends in commit messages.

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
uv run pytest tests/ -v                 # all tests (391 tests)
uv run pytest tests/test_ev.py -v       # single file
uv run pytest tests/test_ev.py::test_name -v  # single test

# Lint
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/     # auto-fix

# Install packages
uv add <package>
```

## Architecture

Three layers: **Extraction** (data loaders + Odds API + weather) → **Intelligence** (features + models + EV + Kelly + sentiment + optional LLM validator in `intelligence/validator.py`) → **Accounting** (grading + CLV + ROI + bankroll ledger in `accounting/ledger.py`).

### NFL Props (Phase 3)

`sports/nfl/props.py` projects receptions (negative binomial) and receiving yards (shifted lognormal, level-dependent residuals) from nflreadpy weekly stats: shrunk exponentially-weighted player means × opponent-position defense factor, with an isotonic P(over) calibration layer. Always call `tune_dispersion()` after `fit()`. Prop odds come from the per-event endpoint (`OddsAPIClient.fetch_event_odds()` — props are NOT on the sport-level `/odds` endpoint). Prop picks grade from player stats via `grade_prop_picks()`; a player missing from a published week's stats voids the pick (result `"void"`, pnl 0), while an unpublished week leaves it ungraded. Prop edge floor is 5% (`--min-edge`), fatter than game markets, to absorb residual calibration drift.

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

**NFL backtests use real closing lines** (`sports/nfl/market.py` from nflreadpy schedules); other sports still use synthetic odds from `_generate_market_odds()` (model-centered + Gaussian noise + 4.5% vig, never included as features). Use `--flat-stake` results to judge selection skill — Kelly ROI on small samples is sizing variance.

**Sentiment is optional.** `is_ollama_available()` is checked before any Ollama call. If unavailable, picks generate from ML alone. Sentiment shifts the model probability (by at most `sentiment_weight`, default 0.02) before the edge is computed, so edge and Kelly sizing stay consistent.

**Season is sport-aware.** Use `SportConfig.season_for_date(date)` — an NFL/NBA game in January belongs to the previous year's season, and mistagging it triggers Elo's per-season mean reversion mid-season.

**DB is optional for training.** `train.py --no-seed` skips DB operations. DB failures during seeding are caught and logged as warnings.

**SQLAlchemy 2.x has no TIMESTAMPTZ.** The codebase uses `TIMESTAMP` instead (aliased for readability in imports).

## Configuration

All settings via pydantic-settings in `config.py`, reading from `.env`. Key tunables: `min_edge_pct` (0.015), `max_kelly_pct` (0.05), `max_bet_pct` (0.07), `sentiment_weight` (0.02), `starting_bankroll` (1000), `ollama_model` ("llama3.1:8b"), `nba_api_rate_limit` (0.6s).

## Key Design Decisions

- **Edge-based filtering:** Only surfaces picks where model probability exceeds vig-removed implied probability by `MIN_EDGE_PCT`
- **Scaled Kelly sizing:** 40% Kelly at edge <= 3%, 60% at <= 5%, 80% above, capped at `MAX_KELLY_PCT`
- **Walk-forward backtesting:** Train on seasons [start..N-1], test on season N — no look-ahead bias
- **All compute local:** No paid LLM APIs; Ollama for sentiment, everything else is statistical
