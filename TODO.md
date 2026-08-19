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

### Phase 1 — Real-line NFL backtest (the go/no-go gate)

- [ ] Replace synthetic odds in `scripts/backtest.py` for NFL with the real
  closing lines already in nflreadpy schedules: bet model vs `spread_line` /
  `total_line` / `home_moneyline` at real prices (`home_spread_odds`,
  `over_odds`, …), grade against real scores.
- [ ] Report ROI and simulated CLV vs close, per season, walk-forward.
- [ ] Keep the synthetic path (with its warning) for NBA until real
  historical odds exist for it.

### Phase 2 — Trustworthy ledger

- [ ] Bankroll ledger: running equity from graded P&L (a table or a view over
  picks); `bankroll_at_pick` currently just echoes the CLI flag.
- [ ] `actual_bet` / `actual_odds` fields on picks (default = recommended) so
  the ledger reflects bets actually placed.
- [ ] Weekly report (CLI + Discord embed): record, ROI, cumulative P&L, and
  CLV hit-rate headlined.
- [ ] Fix pick dedup: `save_picks_to_db()` keys on `(game_id, bet_type)` so
  re-runs after a line move can never refresh a pick.

### Phase 3 — Props MVP (paper-traded)

- [ ] Start with receptions + receiving yards. Distributional player models
  (player volume share × team volume × opponent positional defense; negative
  binomial for counts, quantile/lognormal for yardage → P(over line)) over
  `nfl.load_player_stats()`. Stubs exist in
  `src/betting_agent/sports/nfl/props.py`.
- [ ] Prop odds via Odds API per-event endpoint, filtered to bet365,
  1–2 snapshots/week (~70 credits per full Sunday slate; free tier is
  500/month). Verify NFL prop availability + per-call cost with a live key
  before designing around it; evaluate SportsGameOdds/OddsPapi/PropLine if
  credits pinch.
- [ ] Log picks through the existing table as `bet_type="prop"`; validate
  projection calibration against historical actuals, then paper-trade live
  lines 4–6 weeks. No real stakes in this phase.

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
