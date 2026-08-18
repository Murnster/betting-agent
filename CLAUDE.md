# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated sports betting ETL pipeline and prediction engine. Generates daily "+EV Picks of the Day" using XGBoost models, Kelly Criterion sizing, and optional Ollama sentiment analysis. Supports NFL and NBA, designed to extend to NHL+. Runs entirely on local hardware ($0-$15/month).

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

# Picks
uv run python scripts/picks.py --sport NFL --bankroll 1000
uv run python scripts/picks.py --sport NBA --bankroll 1000 --save

# Daily extraction (morning → closing → postgame → grade)
uv run python scripts/extract.py morning --sport NBA
uv run python scripts/extract.py closing --sport NBA
uv run python scripts/extract.py postgame --sport NBA
uv run python scripts/grade.py

# Backtesting
uv run python scripts/backtest.py --sport NFL --start-season 2019 --end-season 2023

# Tests
uv run pytest tests/ -v                 # all tests (54 tests)
uv run pytest tests/test_ev.py -v       # single file
uv run pytest tests/test_ev.py::test_name -v  # single test

# Lint
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/     # auto-fix

# Install packages
uv add <package>
```

## Architecture

Three layers: **Extraction** (data loaders + Odds API + weather) → **Intelligence** (features + models + EV + Kelly + sentiment) → **Accounting** (grading + CLV + ROI).

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

**Column normalization is mandatory for NFL data.** `nflreadpy` uses `gameday` (not `game_date`), `game_type` (not `is_playoff`), `location` (not `neutral_site`), and has no weather columns. `normalise_raw_schedules()` in `sports/nfl/features.py` handles all renames — it's called automatically by `build_nfl_features()`, but if you pass raw nflreadpy data anywhere else, you must normalize first.

**OHE feature alignment.** One-hot encoding generates different columns per training run depending on which teams appear. `feature_names.pkl` captures the exact column list. At predict time, `X.reindex(columns=feature_names, fill_value=0)` ensures alignment — missing columns get zero-filled.

**The `_is_upcoming` tag pattern.** In `picks.py`, upcoming games (from Odds API) are combined with historical games (for Elo warm-up) into one DataFrame. Since `build_features()` sorts by `game_date` internally, upcoming rows are tagged with `_is_upcoming=True` before entering the pipeline, then retrieved by this tag after feature building.

**NBA team name bridging.** `nba_api` uses 3-letter abbreviations (`"BOS"`), the Odds API uses full names (`"Boston Celtics"`). The `NBA_ABBREV_TO_FULL` / `NBA_FULL_TO_ABBREV` dicts in `sports/nba/loader.py` handle mapping. In `picks.py`, a `home_team_odds` metadata column carries the Odds API name for odds matching while `home_team` carries the abbreviation.

**Backtest uses synthetic odds.** Historical closing odds aren't available, so `_generate_market_odds()` simulates a book using league-average home win rate + Gaussian noise + 4.5% vig overround. These odds are never included as features (no leakage).

**Sentiment is optional.** `is_ollama_available()` is checked before any Ollama call. If unavailable, picks generate from ML alone. Sentiment adjusts edge by at most `sentiment_weight` (default 0.02).

**DB is optional for training.** `train.py --no-seed` skips DB operations. DB failures during seeding are caught and logged as warnings.

**SQLAlchemy 2.x has no TIMESTAMPTZ.** The codebase uses `TIMESTAMP` instead (aliased for readability in imports).

## Configuration

All settings via pydantic-settings in `config.py`, reading from `.env`. Key tunables: `min_edge_pct` (0.015), `max_kelly_pct` (0.05), `max_bet_pct` (0.07), `sentiment_weight` (0.02), `starting_bankroll` (1000), `ollama_model` ("llama3.1:8b"), `nba_api_rate_limit` (0.6s).

## Key Design Decisions

- **Edge-based filtering:** Only surfaces picks where model probability exceeds vig-removed implied probability by `MIN_EDGE_PCT`
- **Scaled Kelly sizing:** 40% Kelly at edge <= 3%, 60% at <= 5%, 80% above, capped at `MAX_KELLY_PCT`
- **Walk-forward backtesting:** Train on seasons [start..N-1], test on season N — no look-ahead bias
- **All compute local:** No paid LLM APIs; Ollama for sentiment, everything else is statistical
