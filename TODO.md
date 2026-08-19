# TODO

Open follow-ups. Add items here rather than leaving them in commit messages
or session scrollback.

## Review roadmap (Aug 2026 audit)

Phased plan from the full project review. Full rationale, evidence, and the
market-strategy context live in the review artifact:
https://claude.ai/code/artifact/c04ec512-2b1f-43cd-a6b6-a86cccbac268

Work the phases in order — each gates the next. Don't start props (Phase 3)
until the real-line backtest (Phase 1) shows what the game-level model is
worth and the ledger (Phase 2) can be trusted to score it.

### Phase 0 — Correctness fixes (then retrain NFL) — DONE 2026-08-18

All landed in commits 48699b5, f02ca6e, 80f55c7, 9950d2b. NFL retrained on
2016-2025 (54 features, 2751 rows, 65.5% test accuracy, Brier 0.2144).
Test count went 253 → 333.

- [x] **F1 — Market-line leakage into NFL features.** Market columns dropped
  in `build_nfl_features()`; constant `home_is_favorite` removed.
- [x] **F8 (found during the work, worse than F1) — the NFL pick path had no
  team-name bridge.** Odds API sends "Kansas City Chiefs", history keys on
  "KC", so upcoming rows matched no history: base Elo 1000 for both teams,
  empty rolling averages, no head-to-head. Combined with F1 and missing
  roof/surface/div_game, 21 of 63 features were zero-filled at pick time.
  Fixed with `NFL_FULL_TO_ABBREV` and `attach_schedule_context()`.
  `PredictionEngine._align()` now warns when it zero-fills a trained feature.
- [x] **F3 — Canonical team names.** Canon is the sport-native abbreviation
  (not the Odds API name as the review proposed — every loader already writes
  abbreviations, so converting on entry is the cheaper direction).
  `sports/teams.py` provides `canonical_team`/`same_team`; grading and CLV
  match through it and leave a pick ungraded rather than guessing.
  `scripts/normalize_teams.py` backfills existing rows.
- [x] **F4 — Atomic (line, price) pairs per book.** Per-book quotes; each side
  reports its best price with that book's line and opposing price.
  `PREFERRED_BOOKMAKERS=bet365` prices against the book actually used.
  Correction to the review: cross-book pairing *understated* moneyline edges
  rather than inflating them — best-of-both always has a smaller overround
  than any real book.
- [x] **F6 — Real weather.** `temp`/`wind` mapped in
  `normalise_raw_schedules()`; dead `weather_Unknown` OHE removed.
- [x] **F5 — Elo warm-up skew.** `training_meta.json` records the training
  seasons; picks.py warms up over exactly those. Costs ~6s per run.

### Phase 1 — Real-line NFL backtest (the go/no-go gate) — DONE 2026-08-18

Built in commit b7b71f6 (`src/betting_agent/sports/nfl/market.py` +
`_simulate_season_real_lines` in `scripts/backtest.py`). Synthetic path kept
for the other sports and behind `--synthetic-odds`.

CLV vs close was dropped from scope: nflreadpy carries only the closing
line, so there is no opening price to measure against. Betting *at* the
close is the stricter test anyway.

**Result — the gate says no for game markets.** 2016-2025 walk-forward,
3,475 flat-staked bets at real closing prices:

| market | bets | win% | needed | ROI |
|---|---|---|---|---|
| moneyline | 1032 | 47.3% | 49.7% | −5.08% |
| spread | 1165 | 51.1% | 51.8% | −1.37% |
| total | 1278 | 47.7% | 51.8% | −7.95% |
| **overall** | **3475** | **48.7%** | | **−4.89%** |

Every test season is negative. Reproduce with:

```
uv run python scripts/backtest.py --sport NFL --start-season 2016 \
  --end-season 2025 --bankroll 5000 --min-train-seasons 4 --flat-stake 10
```

Read `--flat-stake` results, not Kelly ones, when judging selection skill:
the Kelly run showed spread at +4.55% ROI on a sub-breakeven win rate,
which was purely sizing variance.

**Decisions this forces:**

- [ ] Do not bet NFL game markets at closing prices with real money.
- [x] **Investigated the 6.9% median claimed edge** with
  `scripts/model_vs_market.py` (walk-forward 2016-2025, 1,693 games):

  |  | Brier | LogLoss | Accuracy |
  |---|---|---|---|
  | model | 0.2334 | 0.6597 | 60.5% |
  | market (close) | 0.2102 | 0.6075 | 66.7% |

  Correlation 0.718; mean absolute gap 0.109. When the two pick opposite
  sides, the model is right 52.0% at a 2-5% gap, 47.7% at 5-10%, and
  **35.2% at 10%+ (349 games)**. The model gets *more wrong* as it disagrees
  more — the exact inverse of edge. Its confident deviations are its errors.

  This closes the question: no threshold tuning, market selection, or
  guardrail change rescues the game-level model, because the disagreements
  it is built to bet on are anti-predictive. Improving it needs better
  information (injuries, personnel, line movement), not better filtering.
- [ ] Re-run the same gate for NBA/NHL/MLB once real closing lines exist
  for them; assume they fail until shown otherwise.
- [ ] Spread is the least-bad market (−1.37%) and totals the worst
  (−7.95%). If anything gets a second look, it is spreads.

### Phase 2 — Trustworthy ledger — DONE 2026-08-18

Migration `b8c1d2e3f4a5` (applied). The ledger is a view over picks, not a
second store: equity = starting bankroll + cumulative graded P&L, with
stakes/prices taken from `Pick.stake`/`Pick.price` (actual over recommended).

- [x] Bankroll ledger: `accounting/ledger.py` — `equity_curve()`,
  `current_bankroll()`, `ledger_summary()` (peak, max drawdown). Shown in
  `scripts/report.py` and `scripts/bets.py ledger`.
- [x] `actual_bet` / `actual_odds` on picks; `scripts/bets.py list|set`
  records what was actually placed at the book (stake 0 = logged, not bet).
  Grading and ROI pay out from actual when recorded; `bets.py set` on an
  already-graded pick rebooks its P&L.
- [x] Report: CLV hit-rate + bankroll section added to the ROI report
  (grade.py already pushes it to Discord).
- [x] Dedup fixed: re-runs now refresh ungraded picks in place (odds, edge,
  sizing, line); settled picks are never rewritten. Props key on
  `(game_id, bet_type, player, market)`.

### Phase 3 — Props MVP (paper-traded) — BUILT 2026-08-18, calibration PASSED

Models in `sports/nfl/props.py`: negative binomial for receptions, shifted
lognormal (level-dependent residual location/scale) for receiving yards,
both centered on shrunk EW player means × opponent-position defense factor,
with a final isotonic layer mapping raw P(over) to empirical over-rates.
`tune_dispersion()` fits spread + calibrator on walk-forward training
residuals — always call it after `fit()`.

- [x] Calibration gate (`scripts/props_calibration.py`, train 2020-23, eval
  2024-25, 10,444 player-weeks/market): both markets beat the naive-constant
  Brier (receptions 0.1908 vs 0.2414; yards 0.2210 vs 0.2301) and sit within
  ~3pp of the diagonal on the 0.2-0.8 bins where real lines live. Residual
  +2-3pp under-prediction of overs on 2024-25 → props edge floor set to 5%
  (`--min-edge` default in props.py). Interval coverage for receptions reads
  over-nominal because discrete intervals carry point masses — judge counts
  by the reliability table, not coverage.
- [x] Per-event prop odds: `OddsAPIClient.fetch_event_odds()` (props are NOT
  on the sport-level /odds endpoint), filtered to `PREFERRED_BOOKMAKERS`;
  `scripts/props.py [--save] [--max-events N]` generates paper picks
  (verified live against bet365 preseason lines).
- [x] Props log as `bet_type="prop"` with structured `player`/`market`/`line`
  columns; graded from `nfl.load_player_stats()` via
  `grade_prop_picks()` + `make_stat_lookup()` (wired into grade.py; a player
  with no stat row stays ungraded rather than guessing).
- [ ] **Paper-trade 4–6 weeks once the 2026 season starts (September).**
  Run `props.py --save` 1-2×/week, `grade.py` after each slate, judge with
  `report.py` / `bets.py ledger`. No real stakes until paper ROI and the
  reliability of live-line P(over) hold up.
- [ ] Watch API credit spend: one call per event per snapshot; `--max-events`
  caps it. Evaluate SportsGameOdds/OddsPapi/PropLine if credits pinch.

### Phase 4 — Scale what the ledger endorses

- [ ] Real stakes only where paper CLV + ROI are positive over a meaningful
  sample; expand prop markets or upgrade the odds tier (seasonal, Sep–Feb)
  only if the MVP earns it.

### Housekeeping (any time)

- [ ] Refresh CLAUDE.md + project memory: 333 tests (not 54), four sports
  implemented, validator subsystem, nflreadpy has weather columns.
- [ ] `picks.py` derives `season` from the calendar year, so NFL games in
  January/February are tagged as the wrong season — that feeds Elo's
  per-season mean reversion. Derive the season properly per sport.
- [ ] The NBA/NHL/MLB pick paths still have no equivalent of
  `attach_schedule_context()`, so venue/scheduling features available for
  those sports may still arrive empty. Check each against the
  `_align()` warning before trusting their picks.
- [ ] Stop writing `saved_models/*_backtest_tmp/` into the real models dir
  (use a temp path) and delete the existing ones.
- [ ] Sentiment adjusts `edge` but not `model_prob`, so Kelly sizes off a
  probability inconsistent with the edge that selected the bet — reconcile.
- [ ] Backtest sizes Kelly from vig-included `calculate_edge()` while live
  picks use vig-removed fair edges; align the two.
- [ ] Freeze MLB/NHL (leave code, drop from docs/routine runs) until NFL+NBA
  are proven.
- [ ] `db/models.py` Game defines `home_hits`/`away_hits` twice (NHL block,
  then MLB block silently overrides it) — rename one pair with a migration.
- [ ] Prop grading treats a DNP as "no stat row → leave ungraded" forever;
  decide a void policy (e.g. void after stats for that week are published
  and the player is absent).

## Tune or validate the `max_edge_pct` guardrail

`max_edge_pct` currently hard-rejects any pick with edge > 15%, across all
bet types (`_passes_guardrails` in `src/betting_agent/intelligence/picks.py`).

The cap may be tighter than intended. Three pre-existing tests in
`TestCorrelationAdjustment` were built on fixtures with moneyline edges of
17–18% and a totals edge of ~36%, i.e. the codebase previously treated edges
in that range as unremarkable. Those fixtures were moved into a lower band so
the tests keep exercising dedup/sorting/`max_picks` rather than edge magnitude
(commit 83fd4a3).

What to do:
- Watch for `"Guardrail: rejecting pick"` warnings during normal runs. Frequent
  firing means the cap is filtering real picks, not just pathological ones.
- Decide whether 15% is right, or whether the cap should be per-bet-type
  (totals and spreads can legitimately show wider edges than moneylines).
- Consider whether a large edge should reject the pick outright or just flag
  it for the validator to review.

## Decide the fate of the `guardrails-wip` branch

Local-only branch `guardrails-wip` (`1bacf2b`) holds classifier changes that
were deliberately left out of `main`:

- Hyperparameters: `max_depth` 3→4, `eta` 0.01→0.02, `NUM_ROUNDS` 500→800,
  `EARLY_STOPPING` 20→30
- A 60/20/20 calibration split and a calibrator output-range warning — both
  superseded by the k-fold expanding-window `IsotonicEnsemble` and
  `_log_calibration_diagnostics` now on `main`

Only the hyperparameters are still live, and evaluating them requires a
retrain against the current calibration code. Either run that experiment and
land the result, or delete the branch. It is unpushed, so it is not backed up.
