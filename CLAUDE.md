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
uv run python scripts/td_props_diagnostic.py             # walk-forward anytime-TD picks (sets the TD edge window)
uv run python scripts/props.py --today --save --no-td     # skip the anytime-TD board (1 credit/game)
uv run python scripts/ladder_diagnostic.py               # walk-forward ladder hits (sets the ladder window)
uv run python scripts/props.py --today --save --no-ladder # skip the alternate boards (1 credit/game/market)
uv run python scripts/props.py --today --save --no-overs  # skip the straight-overs section (free)

# Unattended loop (cron entry point; docs/NFL_SETUP.md)
scripts/nfl_loop.sh card | closing | grade | backup      # what cron calls
scripts/nfl_loop.sh crontab                              # print the schedule to install

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

**Anytime TD (`sports/nfl/td_props.py`, shadow from Sep 2026).** `player_anytime_td` is a **Yes-only board** (no "No" side, Pinnacle does not post it), so it cannot be de-vigged pairwise: `fair_yes_probabilities()` treats each Yes price as a Poisson rate, scales every rate by one factor until the board sums to the touchdowns the market expects for the game (`expected_game_tds()` from the schedule's spread/total: team TDs ≈ −0.671 + 0.1407 × implied points, 2020-25 fit; × `BOARD_COVERAGE` 0.98 for linemen/defenders), and reports the hold it removed. `TouchdownPropsModel` = shrunk EW TD rate × scoring environment (market-implied team TDs vs trailing) × opponent-vs-position factor → Poisson P(≥1) → isotonic layer (`calibrate()` on the two prior seasons with the schedule). Picks are `pick_side="yes"`, `line=0.5` so grading, `--closing` and labels work unchanged; grading sums rushing+receiving+return+fumble-recovery TDs (`add_anytime_td_column`). The prior is **usage-scaled** (position TDs per touch × the player's touches per game): shrinking toward the bare position rate gave a half-touch-a-game backup the same 10% floor as a starter and the first live run picked two +3000/+4500 bench players. `MIN_FAIR_PROB` (0.10) refuses long shots outright — the favourite-longshot bias lives there, and so does depth-chart knowledge the model lacks. **Edge window, not a floor:** `scripts/td_props_diagnostic.py` (2024-25 walk-forward vs a usage-aware proxy board) shows picks at 8–15% claimed edge hitting 30.4% vs 34.4% claimed with +10% flat ROI at the measured 25% board hold (same in both seasons), but realised probability *falling* above 15% (claimed 38%, realised 24%) — when the model disagrees with the market that hard, the model is wrong. Hence `PROP_EDGE_FLOORS` 0.08 and `PROP_EDGE_CAPS` 0.15 (`edge_cap()`), at most `TD_PROPS_PER_GAME` (2) per game. TD candidates are generated separately (`generate_td_candidates`): never deduped against or scaled with the receiving picks. **The TD scorer sits on the card like the game lean (user's call, Sep 7 2026):** the best-edge player per game always appears, up to the slate's lean cap, labelled PICK (paper stake) when its edge is inside the window and LEAN (stake 0) otherwise (`extra.td_pick`); further in-window players up to `TD_PROPS_PER_GAME` are saved off-card. Card TD scorers go through the LLM validator with the props. Results and the all-time recap report TD scorers on their own line (`get_summary(market=TD_MARKET)`, hit rate vs `avg_fair_pct` — never vs 50%) and exclude them from the receiving-prop headline (`exclude_market`). `--no-td` skips the board (1 credit per game).

**Ladder hits (`generate_ladder_candidates` in `scripts/props.py`, user request Sep 8 2026: "the best overs picks as its own running section … ladder hits like 60+ receiving yards, 6 receptions, 40+ rushing").** The card's "best overs" sub-section, posted in the same Discord block after the props under its own `LADDER HITS` header and run as **its own paper book**: picks carry `Pick.strategy = "ladder"` (nullable column, migration `f2a3b4c5d6e7`; NULL = every other strategy), are sized against `LADDER_BANKROLL` (own ledger via `ledger_summary(strategy=)` / `get_summary(strategy=)`; the main prop summaries pass `exclude_strategy`), and get their own results line and all-time bankroll line. Pricing: the books' `*_alternate` markets are **Over-only ladders** (DraftKings hangs 10–18 rungs per player, 39.5–199.5 yards), so a rung cannot be de-vigged against its own Under — the hold is measured from the player's main-line pair on the same book (`DEFAULT_LADDER_HOLD` 5% when there is none) and divided out; the main-line Over is a rung too. Model side: `ReceivingPropsModel.project_ladder()` uses `LADDER_PRIOR_GAMES` (1 vs the main 4 — the main shrinkage pulls a WR1 ~17 yards under his own average, which is why the main card is unders and why the main projection could never see an over on a milestone board) and `Projection.prob_hit()` maps through a **tail calibrator** fit on `LADDER_RUNGS` (the main isotonic layer never sees the tails and would clip). A rushing-yards model (`build_rushing_history`, RB/FB/QB/WR) exists for the ladder only — `DISTRIBUTION_MARKETS` vs `MODELED_MARKETS`; it has no main-card diagnostic. Policy from `scripts/ladder_diagnostic.py` (2024-25 walk-forward vs a trailing-mean proxy book, soft, so absolute ROI is optimistic): alternate rungs must be milestones (`LADDER_MIN_RUNG` 39.5 yards / 4.5 receptions — below that the "edge" is 10+ yards for backups; the book's main-line Over qualifies at any number), fair probability inside 20–65%, one rung per player, players on the main card excluded so the card never argues with itself. **The pool is an experiment, not a validated edge (user's decision, Sep 8 2026: "I want the ladders and the potential overs to be picks in their own separate pool. I don't care if overall long term they lose money, I just want to experiment with the models and see if they somehow pick good overs or ladders like the main picks for the unders").** So: every entry is a staked PICK from the ladder bankroll (there is no LEAN tier), the floors sit at the main card's min-edge tier (`LADDER_EDGE_FLOORS` 3% for all three markets) rather than where the diagnostic found the edge real, the game's best over is always kept whenever the model has any positive edge on it (further players must clear the floor), and **all three ladders are on** (`LADDER_MARKETS` includes receptions, 3 credits per game). Caps stay as a sanity guard (receiving yards 25%, rushing 15% — realised probability turns over above them, so those are model errors, not picks). What the diagnostic said, for the eventual review: rushing yards are calibrated from ~5% up (44.2% hit vs 43.5% claimed); receiving yards over-claim ~6pp at a 10% floor and more below it; receptions over-claim 10–20pp everywhere. Expect the pool to run under its claimed probabilities and to lose at the book's hold — the question is by how much, and whether any market or edge band holds up. Card slots = the slate's prop cap; card ladder entries go through the validator. On the 2026 opener the book priced its stars far above even the ladder projection (JSN 82.5 line vs 65 projected, the defence factor at its 0.8 floor); review with the TD window. `--no-ladder` skips the boards; `LADDER_MARKETS` drops one at a credit per game.

**Straight overs (`generate_over_candidates` in `scripts/props.py`, user request Sep 8 2026: "there should be at least one straight over line to add in these primetime games").** The third card section, between the props and the ladder under its own `STRAIGHT OVERS` header, its own paper book (`Pick.strategy = "overs"`, `OVERS_BANKROLL`, own results/all-time lines; `ledger.SIDE_BOOKS` lists every side book and the main summaries exclude them all). It prices the book's **main-line Over** on the receiving markets (no extra credits — those markets are fetched for the unders card), fair from the Over/Under pair, P(over) from the ladder projection (`project_ladder` + `prob_hit`, the same lighter shrinkage the ladder uses). **The game's best over by edge is always a pick:** Kelly-staked when the model has an edge, otherwise flat at `OVERS_MIN_STAKE_PCT` (1%) of the overs bankroll with `extra.flat_stake` set, so the selection is tracked whatever the model thinks; further overs need `OVERS_EDGE_FLOOR` (3%), the ladder caps drop the model-is-wrong lines, one over per player. Ordering on the card: main card → overs (excluding main-card players) → ladder (excluding both), so no player appears twice. On the 2026 opener the ladder projection sat under the book on 20 of 22 main lines (Doubs +4.8% and Kupp +0.8% were the only positive overs) — expect many flat-stake entries; the review should split flat-stake from Kelly picks (`recommended_bet` vs `edge` sign). `--no-overs` skips the section.

### In-season NFL props loop (what actually runs Sep–Feb)

Everything except the per-event odds fetch is free. The Odds API free tier is 500 credits/month; `fetch_events()` costs nothing, `fetch_event_odds()` costs one credit **per market** (two per game for the two receiving markets, three with the anytime-TD board, **six with the three ladder boards**), so a full 16-game slate is ~32–96 credits and `--suggest N` / `--max-events N` bound the spend. A full week (TNF + 14 Sunday + MNF) at six markets is ~96 credits plus closing captures — over the free tier's month if every game is fetched; `--no-ladder`, `LADDER_MARKETS`, `--no-td` and `--suggest N` are the dials. Up to 10 bookmakers in one request still count as one region, so requesting the whole fallback list is free.

**The Odds API has no bet365 player-prop feed.** bet365 posted no receiving props in any region on the 2026 opener while DraftKings, FanDuel, BetOnline, BetRivers, Bovada (US) and Pinnacle/Unibet/Sportsbet (EU/AU) all did. `prop_bookmaker_order()` therefore requests the preferred book (`PREFERRED_BOOKMAKERS`, default bet365) **plus** `PROP_FALLBACK_BOOKMAKERS` (default `draftkings,fanduel`), and `books_in_preference()` prices each game against the **first** book in that order that posted — one book per game, never best-of-N across books (that would inflate edges past what the floors were set on). The pick card shows the book; the user checks bet365's number by hand before staking and records the real price with `bets.py set --odds`. `--closing` reads the same book order so CLV compares like with like.

1. Game day (Thu/Sun/Mon): `props.py --today --save [--suggest N]`. Output is **one card per slate** (`intelligence/slate.py`: Thursday/Sunday Early/Sunday Late/Sunday Night/Monday by Eastern kickoff). A single-game slate carries 1 game lean + up to 2 props; a Sunday window 3 leans + 3 props, chosen top-down by edge. Everything that clears the floors is still saved (`Pick.on_card` marks the card). The **game lean** (`intelligence/game_lean.py`) is not a model pick: Pinnacle's vig-removed price is the fair value, converted to the bettable book's line through the gate-measured margin/total sigmas, and the lean is the best-edge side at that book. It is labelled LEAN on the card and Discord, saved as a paper pick (stake 0 when edge ≤ 0), and `--closing` captures its close so a season of CLV decides whether early numbers beat closing ones. One sport-level odds call (`fetch_odds(..., bookmakers=)`) = 3 credits per run regardless of game count; `--no-leans` skips it. `--today` matches kickoff to the **machine's local calendar day** (this box is America/Halifax, UTC−3, so US primetime still lands on its own day) and drops games that have already kicked off, so a re-run never refreshes saved picks with in-play prices. Off-days exit before any model fitting or paid call. After the models fit, the run loads free season context: the published roster (`current_teams()` — offseason movers land on their new side; 72 re-teamed in Sep 2026), the schedule (`nfl_week_for()` gives the exact week and spread/total for the validator payload), the official injury report and depth charts (`sports/nfl/injuries.py`). Out/Doubtful players are dropped; Questionable and a QB1 who is Out are attached as flags on the pick card. Then the LLM validator runs if `AGENT_ENABLED` (shadow mode by default — see below).
2. Pre-kickoff: `props.py --closing [--window-minutes 90]`. Also captures held game leans (3 credits for all of them, same book order + Pinnacle). Free events call → held ungraded prop picks in games kicking off inside the window → per-event odds fetch **only for those games** → `closing_line`/`closing_odds` stored on each pick; `clv` is set only when the line held (a price at a different number is not comparable) and `report.py` counts line moves for/against otherwise. Zero credits when nothing is due. This is the only source of CLV for props — `update_clv_for_picks()` reads the game-market `odds` table, and `grade.py --date` deliberately leaves prop CLV alone on reset.
3. Next morning: `grade.py`. It finalizes NFL Game rows from nflreadpy schedules, then grades props from `load_player_stats()`. Week N's `stats_player_week_<season>.parquet` doesn't exist on nflverse until week N is played — `load_player_stats()` tolerates a missing season (warns, continues), and a pending pick simply stays ungraded until the parquet appears.
4. Weekly: `report.py --sport NFL` / `bets.py ledger`. Real stakes go through `bets.py set`.

**Validator (shadow).** `AGENT_MODEL` picks the provider by prefix: `claude/<model>` runs the local `claude -p` CLI (`intelligence/validator/claude_cli.py`), `gemini/<model>` the Gemini API. With `AGENT_SHADOW=true` (default) verdicts are recorded to `agent_validations`, shown on pick cards/Discord as `SHADOW`, but never change edge, sizing, or the slate — flip it only once `agent_validations.verdict` vs `picks.result` shows the verdicts add value. The CLI provider replaces the system prompt with the rules, passes the payload on stdin, loads no settings, and runs from a temp dir (~250 input tokens + payload; a default run from the repo would load ~29k tokens of CLAUDE.md/memory). Do NOT use `--bare` — it skips keychain reads and reports "Not logged in". WebSearch needs both `--tools WebSearch` and `--allowedTools WebSearch` in print mode. Measured Sep 2026 on Sonnet: ~$0.014/game without search, ~$0.09 with for a one-pick game, **$0.42 for a six-pick game with search** (the model searches per player). `--max-budget-usd` caps each call at `AGENT_CLAUDE_MAX_CALL_USD` (0.25) and `AGENT_DAILY_BUDGET_USD` (1.00) gates the day; a call the CLI kills on its cap exits non-zero **after spending**, so the provider reads `total_cost_usd` from the error envelope (`last_call_cost_usd`) and the orchestrator books it as SKIPPED records — otherwise the daily gate never sees the money. Prop results are keyed by `(bet_type, pick_side, player)` — two "over" props in one game used to collide. **The payload carries each player's current club (`team`, from the roster overlay) and the prompt states that every listed player is in the game** — on the 2026 opener the model searched bare names, found two offseason movers' old clubs and returned SKIPPED for "not in this matchup". SKIPPED is the harness's word (no call / budget / failed call); a model that returns it is recorded as UNCHANGED. **A skipped validation must never come into play (user, Sep 8 2026):** a failed or cap-killed call is retried (`AGENT_RETRIES`, default 1) and this box's `.env` runs `claude/opus` with caps sized not to kill anything ("it doesn't matter how long it takes": `AGENT_CLAUDE_MAX_CALL_USD` 10.00, `AGENT_DAILY_BUDGET_USD` 100.00, timeout 900s, 16 turns) — read the validator cost line after the first Sunday and tighten from what it actually spent.

nflreadpy gates `load_injuries`/`load_rosters_weekly` on its own `get_current_season()`, which flips to the new season on the Thursday after Labor Day — requesting the new season before then raises `ValueError: Season must be between …`. `load_schedules` and `load_rosters` accept the new season earlier. Week-1 projections therefore rest entirely on the prior two seasons of stats.

`scripts/nfl_loop.sh {card|closing|grade|backup}` is the cron entry point for this loop (sets PATH for `uv`/`claude`, unsets `CLAUDECODE`, logs to `logs/nfl_*.log`; `nfl_loop.sh crontab` prints the schedule: card 12:45 Sun/Sat + 19:00 Thu/Mon, closing hourly 12–23 on game days, grade 09:00, backup 03:15, local time in a North American zone). The step-by-step install for a fresh machine — Postgres, `pg_restore` of this box's dump so the ledger carries over, `.env`, Claude login, the verification sequence, credit budget — is `docs/NFL_SETUP.md` (written Sep 8 2026 for the user's second laptop, which is to run the loop non-stop; run it on ONE machine only or the ledger forks). `scripts/daily_workflow.sh` / `setup_cron.sh` are the older game-market (NBA/NHL) automation and are not used for NFL. `.env` here runs `AGENT_MODE=all` — `top` caps the games validated per run and leaves the rest SKIPPED, which the user does not want.

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

All settings via pydantic-settings in `config.py`, reading from `.env`. Key tunables: `min_edge_pct` (0.015), `max_kelly_pct` (0.05), `max_bet_pct` (0.07), `sentiment_weight` (0.02), `starting_bankroll` (1000), `ollama_model` ("llama3.1:8b"), `nba_api_rate_limit` (0.6s). Validator: `agent_enabled`, `agent_model` (`claude/opus` in this box's `.env`), `agent_shadow` (True), `agent_claude_web_search`, `agent_claude_max_turns` (4 default; 16 here), `agent_claude_max_call_usd` (0.25 default; 10.00 here), `agent_daily_budget_usd` (100.00 here), `agent_claude_timeout` (900 here), `agent_retries` (1).

## Key Design Decisions

- **Edge-based filtering:** Only surfaces picks where model probability exceeds vig-removed implied probability by `MIN_EDGE_PCT`
- **Scaled Kelly sizing:** 40% Kelly at edge <= 3%, 60% at <= 5%, 80% above, capped at `MAX_KELLY_PCT`
- **Walk-forward backtesting:** Train on seasons [start..N-1], test on season N — no look-ahead bias
- **All compute local:** No paid LLM APIs; Ollama for sentiment, everything else is statistical
