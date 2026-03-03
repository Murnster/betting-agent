# Betting Agent — Architecture & Build Plan

> **Status: All phases complete.** The system is fully operational end-to-end.

## Role & Objective

A local, automated sports betting ETL pipeline and prediction engine for the NFL (extensible to NBA/NHL). Generates daily **+EV Picks of the Day (POTD)** and tracks **Closing Line Value (CLV)** and **ROI** over time.

## Tech Stack & Constraints

- **Language:** Python 3.12
- **Data Manipulation:** Pandas (for XGBoost/sklearn compatibility)
- **Data Sources:** `nflreadpy`, The Odds API, OpenWeatherMap
- **AI/Sentiment:** Ollama (local execution, default: `phi4`)
- **Database:** PostgreSQL (localhost:5432)
- **Budget:** $0–$15/month — all compute runs locally

## System Architecture

Three layers process data end-to-end each day:

### 1. Extraction Layer
- **`scripts/extract.py morning`** — Fetch daily schedules + opening odds from The Odds API, upsert game rows, fetch stadium weather via OpenWeatherMap
- **`scripts/extract.py closing`** — Capture closing odds (`is_closing=True`) right before game time for CLV tracking
- **`scripts/extract.py postgame`** — Fetch final scores from The Odds API, update game status to `"final"`
- Historical data loaded via `nflreadpy` (`load_schedules`, `load_injuries`, `load_player_stats`)

### 2. Intelligence Layer
- **Feature engineering** — Elo ratings (8 variants with 3-season reset), rolling averages (L3/L5), win streaks, H2H record, rest days, home/away splits (~125 features)
- **Dual-model prediction** — XGBoost classifier (win probability) + two regressors (home/away scores), calibrated with IsotonicRegression
- **Edge calculation** — Vig-removed implied probability compared against model probability
- **Sentiment modifier** — Ollama analyzes injury reports per team, produces a -1.0 to +1.0 score that adjusts edge by `sentiment_weight` (default 2%)
- **Kelly Criterion sizing** — Fractional Kelly with hard caps (`MAX_KELLY_PCT=0.05`, `MAX_BET_PCT=0.07`)
- **Filtering** — Only surfaces picks with `edge >= MIN_EDGE_PCT` (default 1.5%)

### 3. Accounting Layer
- **Grading** — Moneyline (team won?), spread (team + line > 0?), total (over/under line comparison)
- **CLV** — Compares pick-time odds vs closing odds in implied probability space
- **ROI** — Cumulative P&L tracking across all graded picks

## DB Schema

| Table | Purpose |
|-------|---------|
| `games` | Schedules, scores, weather, metadata per game |
| `odds` | Opening and closing odds per game per bookmaker (h2h, spreads, totals, props) |
| `sentiment` | Ollama AI sentiment scores per team/game |
| `picks` | Daily POTD output, graded results, CLV, ROI |

## Build Phases — All Complete

### Phase 1 — Project Scaffolding ✅
- `uv` project, `pyproject.toml`, directory structure, `.env.example`

### Phase 2 — Database Layer ✅
- SQLAlchemy 2.x ORM models (`games`, `odds`, `sentiment`, `picks`)
- Alembic migrations, session context manager, common queries

### Phase 3 — NFL Data Extraction ✅
- `nflreadpy` wrapper (`loader.py`), schedule normalisation
- Multi-sport abstract interface (`sports/base.py`), NBA/NHL stubs

### Phase 4 — Feature Engineering ✅
- Elo system (8 variants, 3-season reset decay)
- Rolling stats, streaks, H2H, rest days, home/away splits
- Full NFL feature assembly pipeline (~125 features)

### Phase 5 — ML Models ✅
- XGBoost classifier + IsotonicRegression calibrator
- Home/away score regressors
- `PredictionEngine` (load/predict with feature alignment)

### Phase 6 — Odds API + EV Calculation ✅
- `OddsAPIClient` (fetch events, odds, scores; parse & store)
- Edge calculation (fair odds, vig removal, spread edge, total edge)
- Kelly Criterion bet sizing with hard caps

### Phase 7 — Pick Generation ✅
- `BetCandidate` dataclass, `generate_picks()` pipeline
- Moneyline, spread, and total (over/under) picks
- CLI formatter, DB persistence
- Sentiment-adjusted edges (optional, Ollama)

### Phase 8 — Accounting (Grading + CLV + ROI) ✅
- Pick grading against final scores (moneyline/spread/total)
- O/U picks stored as `"over 45.5"` format for correct grading
- CLV calculation from closing odds
- ROI summary reporting

### Phase 9 — Backtesting ✅
- Walk-forward backtest (train on prior seasons, test next)
- No data leakage (odds excluded from features, no future data)
- Outputs `backtest_results.csv` + `equity_curve.png`

### Phase 10 — Daily Extraction Orchestrator ✅
- `scripts/extract.py` with `morning`, `closing`, `postgame` subcommands
- Opening odds + weather fetch, closing odds capture, final score ingestion
- NFL stadium city mapping for weather lookups

### Phase 11 — Ollama Sentiment Integration ✅
- `is_ollama_available()` health check with graceful degradation
- Injury context from `nflreadpy.load_injuries()`
- Structured JSON prompt → sentiment score (-1.0 to +1.0)
- Edge adjustment: `edge += score * sentiment_weight`
- Averaged sentiment for totals bets

## Daily Workflow

```bash
# 1. Morning: fetch schedule + opening odds + weather
uv run python scripts/extract.py morning

# 2. Pre-game (~5 min before): capture closing odds
uv run python scripts/extract.py closing

# 3. Post-game: fetch final scores
uv run python scripts/extract.py postgame

# 4. Next morning: grade + CLV + ROI
uv run python scripts/grade.py

# 5. Afternoon: generate picks (with sentiment if Ollama running)
uv run python scripts/picks.py --bankroll 1000 --save
```

## Extending to Other Sports

The codebase is multi-sport from day one. `sports/base.py` defines an abstract interface. Add a `loader.py` and `features.py` under `sports/nba/` or `sports/nhl/` following the NFL pattern, then pass `--sport NBA` to any script.
