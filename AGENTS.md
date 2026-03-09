# Repository Guidelines

## Project Structure & Module Organization
Code lives under `src/betting_agent/` using a standard `src/` layout. Core areas are `sports/` for sport-specific loaders and feature builders, `pipeline/` for shared feature logic, `models/` for training and prediction, `intelligence/` for EV/Kelly/picks logic, `accounting/` for grading and ROI, and `db/` for SQLAlchemy models and Alembic migrations. CLI entry points are in `scripts/` (`train.py`, `picks.py`, `extract.py`, `backtest.py`). Tests mirror the codebase in `tests/`. Generated artifacts belong in `saved_models/`; local configuration starts from `.env.example`.

## Build, Test, and Development Commands
Use `uv` for environment and command execution.

- `uv sync` installs runtime and dev dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest tests/ -v` runs the unit test suite.
- `uv run pytest --cov=src/betting_agent tests/` checks coverage when touching model, pipeline, or grading logic.
- `uv run ruff check src tests scripts` runs linting.
- `uv run python scripts/train.py --sport NFL --seasons 2022 2023 2024` trains models and writes artifacts to `saved_models/NFL/`.
- `uv run alembic upgrade head` applies database migrations.

## Coding Style & Naming Conventions
Target Python 3.11+ and keep lines within Ruff’s configured `100` characters. Follow PEP 8: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase sport codes like `NFL`, `NBA`, `NHL`, and `MLB` in CLI flags or saved model paths. Prefer small, explicit functions over shared implicit state. Keep sport-specific code inside `src/betting_agent/sports/<sport>/`.

## Testing Guidelines
Pytest is the test framework. Add or update tests in `tests/test_<area>.py`; mirror the module or behavior you changed, for example `tests/test_nhl_features.py` for `sports/nhl/features.py`. Cover edge calculations, loaders, grading, and registry changes with deterministic assertions. Run the full suite before opening a PR; add `--cov` for higher-risk changes.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `MLB added`, `Fixed results`, and `discord all-time all sports`. Keep commit titles concise, specific, and in sentence case. PRs should include a brief summary, affected sports or workflows, any required env or migration changes, and sample output or screenshots when CLI/reporting behavior changes.

## Configuration & Data Notes
Do not commit secrets from `.env`. Use `.env.example` as the template for API keys and `DATABASE_URL`. Treat `saved_models/`, generated CSVs, and local logs as build artifacts unless a change explicitly updates checked-in fixtures or migration files.
