# Gemini CLI Context for `betting-agent`

This `GEMINI.md` provides essential context and instructions for the Gemini CLI agent when working with the `betting-agent` project.

## Project Overview

**betting-agent** is a local, automated sports betting ETL pipeline and prediction engine. It currently supports **NFL** and **NBA**, with an architecture designed to be extensible to other sports (e.g., NHL). It runs entirely on local hardware without paid APIs, leveraging free tiers and local LLMs.

### Core Objectives
1.  **Generate +EV Picks**: Identify value bets (Moneyline, Spread, Totals) using dual-model predictions (XGBoost classification + Regression).
2.  **Track Performance**: Calculate Closing Line Value (CLV) and ROI to validate strategies.
3.  **Local Execution**: Operate within a strict budget ($0–$15/mo) using local compute for ML and LLM tasks.

## Architecture

The system operates in three distinct layers:

1.  **Extraction Layer** (`src/betting_agent/sports/*/loader.py`, `scripts/extract.py`)
    *   Fetches schedules, odds (The Odds API), and weather (OpenWeatherMap).
    *   Loads historical data via sport-specific libraries (`nflreadpy`, `nba_api`).
    *   Stores raw data in PostgreSQL.

2.  **Intelligence Layer** (`src/betting_agent/intelligence/`, `src/betting_agent/models/`)
    *   **Feature Engineering**: Elo ratings, rolling averages, streaks, rest days.
    *   **Prediction**: XGBoost classifier (win prob) + Isotonic Calibration + Score Regressors.
    *   **Decision Engine**: Kelly Criterion sizing, EV calculation (vig-removed).
    *   **Sentiment**: Optional local LLM (Ollama) analysis of injury reports/matchups.

3.  **Accounting Layer** (`src/betting_agent/accounting/`)
    *   **Grading**: Validates picks against final scores.
    *   **CLV**: Tracks line movement to measure market efficiency.
    *   **ROI**: Reports profitability.

## Tech Stack

*   **Language**: Python 3.12
*   **Package Manager**: `uv` (Astral)
*   **Database**: PostgreSQL (with SQLAlchemy 2.x + Alembic)
*   **ML**: XGBoost, Scikit-learn, Pandas, Polars
*   **LLM**: Ollama (local, e.g., `llama3.1:8b`, `phi4`)
*   **APIs**: The Odds API, OpenWeatherMap, `nflreadpy`, `nba_api`

## Development Workflow

### Prerequisites
*   Python 3.11+
*   `uv` installed
*   PostgreSQL running locally
*   Ollama running (optional, for sentiment analysis)

### Setup & Installation
1.  **Install Dependencies**: `uv sync`
2.  **Environment**: `cp .env.example .env` and populate keys (`ODDS_API_KEY`, `DATABASE_URL`, etc.).
3.  **Database**:
    *   Create DB: `sudo -u postgres psql -c "CREATE DATABASE betting_agent;"`
    *   Migrate: `uv run alembic upgrade head`

### Daily Routine (Agent Tasks)
The agent may be asked to perform parts of the daily cycle:

1.  **Morning Extraction**: `uv run python scripts/extract.py morning --sport [NFL|NBA]`
2.  **Closing Odds**: `uv run python scripts/extract.py closing --sport [NFL|NBA]`
3.  **Post-Game**: `uv run python scripts/extract.py postgame --sport [NFL|NBA]`
4.  **Grading**: `uv run python scripts/grade.py`
5.  **Generate Picks**: `uv run python scripts/picks.py --sport [NFL|NBA] --bankroll 1000 [--save]`

### Training Models
To retrain models (e.g., mid-season update):
```bash
uv run python scripts/train.py --sport NFL --seasons 2018 2019 2020 2021 2022 2023 2024
uv run python scripts/train.py --sport NBA --seasons 2022 2023 2024
```

### Backtesting
To validate strategy changes:
```bash
uv run python scripts/backtest.py --sport NFL --start-season 2019 --end-season 2023
```

## Key Configuration (`src/betting_agent/config.py`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MIN_EDGE_PCT` | `0.015` | Minimum edge (1.5%) to trigger a pick. |
| `MAX_KELLY_PCT` | `0.05` | Max bankroll fraction per bet (Kelly criterion). |
| `SENTIMENT_WEIGHT` | `0.02` | Max adjustment to edge from LLM sentiment. |
| `ODDS_API_KEY` | - | Required for live odds. |
| `OLLAMA_URL` | `http://localhost:11434` | URL for local LLM inference. |

## File Structure Highlights

*   `scripts/`: Executable entry points for all major tasks.
*   `src/betting_agent/sports/`: Sport-specific logic (loaders, features). To add a sport, implement `loader.py` and `features.py` here and register in `registry.py`.
*   `src/betting_agent/models/`: Core ML logic (XGBoost, calibration, regression).
*   `src/betting_agent/db/`: Database models and migrations.

## Common Commands Reference

| Task | Command |
| :--- | :--- |
| **Run Tests** | `uv run pytest tests/ -v` |
| **Linting** | `uv run ruff check .` |
| **Format** | `uv run ruff format .` |
| **DB Migration** | `uv run alembic upgrade head` |
| **New Migration** | `uv run alembic revision --autogenerate -m "message"` |
