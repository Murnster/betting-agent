# betting-agent

A local, automated sports betting ETL pipeline and prediction engine. Generates daily **+EV Picks of the Day (POTD)**, backtests historical games to validate strategy, and tracks **Closing Line Value (CLV)** and **ROI** over time.

Currently supports **NFL** and **NBA**. Runs entirely on local hardware — no paid LLM APIs, $0–$15/month budget.

---

## How It Works

Three layers process data end-to-end each day:

```
Extraction Layer       Intelligence Layer        Accounting Layer
─────────────────      ──────────────────────    ──────────────────────
nflreadpy / nba_api ─► Feature engineering   ─► Grade yesterday's picks
The Odds API        ─► XGBoost prediction    ─► Log CLV
OpenWeatherMap      ─► Vig-removed edge calc ─► Update ROI report
                       Kelly Criterion sizing
                       Ollama sentiment (LLM)
```

### Prediction Pipeline

1. **Feature engineering** — Elo ratings (8 variants), rolling averages (L3/L5/L10), win streaks, head-to-head record, rest days, home/away splits, plus sport-specific features (NFL: weather/surface; NBA: back-to-back, conference/division, advanced stats — TS%, eFG%, ORTG/DRTG, pace, AST/TOV rates, OREB)
2. **Dual-model prediction** — XGBoost classifier (win probability) + two regressors (home/away scores), calibrated with IsotonicRegression
3. **Edge calculation** — Compares model probability against vig-removed implied probability from The Odds API
4. **Kelly Criterion sizing** — Fractional Kelly with hard caps to size each bet as a % of bankroll
5. **Filtering** — Only surfaces picks with `edge >= MIN_EDGE_PCT` (default 1.5%)

---

## Tech Stack

| Component       | Technology                                                                  |
| --------------- | --------------------------------------------------------------------------- |
| Language        | Python 3.12                                                                 |
| Package manager | `uv`                                                                        |
| NFL data        | `nflreadpy` (`load_schedules`, `load_injuries`, `load_player_stats`)        |
| NBA data        | `nba_api` (`LeagueGameLog`, `LeagueDashTeamStats`, `LeagueDashPlayerStats`) |
| Odds data       | [The Odds API](https://the-odds-api.com)                                    |
| Weather         | OpenWeatherMap                                                              |
| ML models       | XGBoost + scikit-learn                                                      |
| Calibration     | IsotonicRegression                                                          |
| Sentiment/AI    | Ollama (local LLM, default: `llama3.1:8b`)                                  |
| Database        | PostgreSQL                                                                  |
| ORM/Migrations  | SQLAlchemy 2.x + Alembic                                                    |

---

## Prerequisites

**Required:**

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Required for live picks** (not needed for training or backtesting):

- [The Odds API key](https://the-odds-api.com) (free tier: 500 requests/month) — provides live odds for edge calculation

**Optional:**

- PostgreSQL — enables the daily extract/grade/CLV/ROI workflow. Training and picks work without it (`--no-seed`)
- [Ollama](https://ollama.com) — adds LLM sentiment analysis to picks (`ollama pull llama3.1:8b`). If unavailable, picks generate from the ML model alone
- OpenWeatherMap API key — NFL weather features (free tier available)

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url> betting-agent
cd betting-agent
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
ODDS_API_KEY=your_key_here              # required for live picks
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/betting_agent  # optional
WEATHER_API_KEY=your_key_here           # optional
OLLAMA_URL=http://localhost:11434/api/generate  # optional
OLLAMA_MODEL=llama3.1:8b                        # optional
```

Strategy defaults (override in `.env` if desired):

```env
MIN_EDGE_PCT=0.015     # minimum edge % to surface a pick
MAX_KELLY_PCT=0.05     # Kelly fraction cap
MAX_BET_PCT=0.07       # max % of bankroll per bet
STARTING_BANKROLL=100.0
SENTIMENT_WEIGHT=0.02  # max edge adjustment from Ollama sentiment (-1 to +1 scaled)

# Discord notifications (optional — one webhook per sport/channel)
# DISCORD_WEBHOOK_NFL_PICKS=https://discord.com/api/webhooks/...
# DISCORD_WEBHOOK_NBA_PICKS=https://discord.com/api/webhooks/...
# DISCORD_WEBHOOK_NFL_RESULTS=https://discord.com/api/webhooks/...
# DISCORD_WEBHOOK_NBA_RESULTS=https://discord.com/api/webhooks/...
# DISCORD_ENABLED=true
```

### 3. Train the models

#### NFL

Downloads schedule data via nflreadpy, builds a ~62-feature matrix, trains XGBoost models, and saves to `saved_models/NFL/`.

```bash
uv run python scripts/train.py --sport NFL --seasons 2018 2019 2020 2021 2022 2023 2024
```

#### NBA

Downloads schedule data via nba_api, builds a ~91–97 feature matrix (including advanced stats like TS%, eFG%, ORTG/DRTG, pace, AST/TOV rates, back-to-back, conference, and division features), and saves to `saved_models/NBA/`.

```bash
uv run python scripts/train.py --sport NBA --seasons 2022 2023 2024
```

Training takes ~1–2 minutes per sport. Add `--no-seed` to skip DB seeding if PostgreSQL is not running.

Saved artifacts (per sport):

```
saved_models/<SPORT>/
├── classifier.json          # XGBoost win/loss model
├── calibrator.joblib        # IsotonicRegression probability calibrator
├── home_regression.joblib   # home score regressor
├── away_regression.joblib   # away score regressor
├── feature_names.pkl        # feature alignment at predict time
├── scoring_sigma.pkl        # empirical regression residual std devs
└── feature_importance.json  # gain-based feature importance per model
```

### 5. Generate today's picks

```bash
# NFL picks
uv run python scripts/picks.py --sport NFL --bankroll 100

# NBA picks
uv run python scripts/picks.py --sport NBA --bankroll 100
```

If Ollama is running, each matchup gets an LLM-powered analysis (injury context, team stats, key matchup factors). If Ollama is unavailable, picks generate from the ML model alone — no errors.

---

## Database Setup (Optional)

PostgreSQL is required for the daily extract/grade/CLV/ROI workflow and for saving picks with `--save`. If you only want to train models, generate picks, or run backtests, you can skip this.

```bash
# Create DB (run once)
sudo -u postgres psql -c "CREATE DATABASE betting_agent;"

# Apply schema (creates games, odds, sentiment, picks tables)
uv run alembic upgrade head
```

### Resetting the Database

If you suspect bad data in the DB (e.g., from a corrupted seed or stale scores), you can wipe all tables and start fresh:

```bash
# Drop all tables and re-apply the schema
uv run alembic downgrade base && uv run alembic upgrade head
```

This deletes **all** games, odds, sentiment, and picks data. To re-populate:

- **Re-seed game schedules:** run `train.py` without `--no-seed` (seeding happens automatically)
- **Re-fetch odds/weather:** run `extract.py morning --sport NBA` (and/or NFL)
- **Saved picks and grading history cannot be recovered** — only reset if you're sure

---

## Daily Workflow

Requires PostgreSQL. All scripts accept `--sport NFL` or `--sport NBA` (default: NFL).

### 1. Morning — fetch schedule + opening odds + weather

```bash
uv run python scripts/extract.py morning --sport NBA
```

Pulls today's games from The Odds API, stores opening odds in the DB, and fetches weather for each stadium city via OpenWeatherMap.

### 2. Pre-game — capture closing odds

```bash
uv run python scripts/extract.py closing --sport NBA
```

Re-fetches odds with `is_closing=True` so CLV can be calculated after grading.

### 3. Post-game — fetch final scores

```bash
uv run python scripts/extract.py postgame --sport NBA
```

Pulls scores for games completed in the last 3 days and updates the DB with final results.

### 4. Next morning — grade + CLV + ROI

```bash
uv run python scripts/grade.py
```

Grades each pick against final scores, calculates CLV (did the closing line move in your favor?), and prints an ROI summary.

### 5. Generate picks

```bash
# Print to terminal
uv run python scripts/picks.py --sport NBA --bankroll 100

# Save picks to DB
uv run python scripts/picks.py --sport NBA --bankroll 100 --save
```

### 6. Standalone matchup analysis

```bash
# Analyze all today's NBA games with Ollama
uv run python scripts/analyze.py --sport NBA

# Analyze a specific NFL matchup
uv run python scripts/analyze.py --sport NFL --home "Kansas City Chiefs" --away "Buffalo Bills"
```

## Automation

To run the daily workflow automatically (e.g., on a server), use the provided helper scripts:

1. **`scripts/daily_workflow.sh`**: A wrapper that runs the appropriate extraction and prediction steps based on the time of day (`morning`, `pregame`, `postgame`).
2. **`scripts/setup_cron.sh`**: Generates the `crontab` entries to schedule the workflow.

Run the setup script to see the recommended configuration:

```bash
./scripts/setup_cron.sh
```

---

## Backtesting

Validate the model with a walk-forward backtest before trusting it with real money. Each season is trained on all prior seasons and tested on the next — no look-ahead bias.

```bash
# NFL backtest
uv run python scripts/backtest.py --sport NFL --start-season 2019 --end-season 2023

# NBA backtest
uv run python scripts/backtest.py --sport NBA --start-season 2018 --end-season 2023
```

Options:

```bash
--bankroll 2000          # starting bankroll (default: 1000)
--flat-stake 25          # flat bet amount instead of Kelly sizing
--reset-bankroll         # reset bankroll at the start of each season
--min-train-seasons 3    # minimum training seasons before first test
```

Output:

- `backtest_results.csv` — per-season stats (bets, wins, P&L, ROI)
- `equity_curve.png` — cumulative bankroll chart

---

## Project Structure

```
betting-agent/
├── scripts/
│   ├── train.py          # Train XGBoost models (--sport NFL|NBA)
│   ├── picks.py          # Generate today's POTD (with optional Ollama sentiment)
│   ├── grade.py          # Grade yesterday's picks + ROI report
│   ├── backtest.py       # Walk-forward historical backtest
│   ├── extract.py        # Daily extraction orchestrator (morning/closing/postgame)
│   └── analyze.py        # Standalone Ollama matchup analysis
├── src/betting_agent/
│   ├── config.py         # Pydantic-settings (reads .env)
│   ├── api/
│   │   ├── odds.py       # The Odds API client (h2h, spreads, totals, props)
│   │   └── weather.py    # OpenWeatherMap client
│   ├── db/
│   │   ├── models.py     # SQLAlchemy ORM (games, odds, sentiment, picks)
│   │   ├── session.py    # DB session context manager
│   │   ├── queries.py    # Common DB queries
│   │   └── migrations/   # Alembic migration scripts
│   ├── pipeline/
│   │   ├── elo.py        # Elo rating system (8 variants, 3-season reset)
│   │   └── features.py   # Shared features: rolling avgs, streaks, H2H, rest days
│   ├── sports/
│   │   ├── base.py       # Abstract sport interface
│   │   ├── registry.py   # Sport registry (SportConfig dispatch)
│   │   ├── nfl/
│   │   │   ├── loader.py   # nflreadpy wrapper
│   │   │   └── features.py # NFL feature assembly + normalisation
│   │   └── nba/
│   │       ├── loader.py   # nba_api wrapper (LeagueGameLog pivot, team mappings)
│   │       └── features.py # NBA features (advanced stats, back-to-back, conference, division)
│   ├── models/
│   │   ├── classification.py  # XGBoost classifier + IsotonicRegression calibrator
│   │   ├── regression.py      # Score regressors (home/away)
│   │   └── engine.py          # PredictionEngine: load/predict
│   ├── intelligence/
│   │   ├── ev.py         # Edge calculation (vig-removed implied probability)
│   │   ├── kelly.py      # Scaled Kelly Criterion bet sizing
│   │   ├── picks.py      # BetCandidate, generate_picks, CLI formatter
│   │   └── sentiment.py  # Ollama-based matchup analysis (NFL + NBA context)
│   └── accounting/
│       ├── grader.py     # Grade picks against results
│       ├── clv.py        # Closing Line Value tracking
│       └── roi.py        # ROI reporting
├── tests/                # 114 unit tests (uv run pytest tests/ -v)
├── saved_models/         # Trained model artifacts (per sport)
├── .env.example          # Environment variable template
├── pyproject.toml        # Project metadata + dependencies
└── alembic.ini           # Alembic migration config
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Database Schema

| Table       | Purpose                                                  |
| ----------- | -------------------------------------------------------- |
| `games`     | Schedules, scores, metadata per game (NFL + NBA)         |
| `odds`      | Opening and closing odds per game (h2h, spreads, totals) |
| `sentiment` | Ollama AI sentiment scores per team/game                 |
| `picks`     | Daily POTD output, graded results, CLV, ROI              |

---

## Extending to Other Sports

The codebase is multi-sport by design. To add a new sport (e.g. NHL):

1. Create `src/betting_agent/sports/nhl/loader.py` — subclass `SportLoader`, implement `load_schedules()`, `load_injuries()`, `sport_key()`
2. Create `src/betting_agent/sports/nhl/features.py` — implement `build_nhl_features()` and `split_features_targets()`
3. Register in `src/betting_agent/sports/registry.py` — add a `SportConfig` entry
4. Add NBA-style team context to `intelligence/sentiment.py` if Ollama analysis is desired
5. All scripts automatically pick up the new sport via `--sport NHL`
