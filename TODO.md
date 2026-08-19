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
### Phase 3.5 — Pick-selection fixes from the walk-forward pick diagnostic (2026-08-19)

`scripts/props_diagnostic.py` (train 2020-23, eval 2024-25, 10,395 picks at a
5% floor against trailing-median book-proxy lines) drove four changes. Re-run
it after any projection/calibration change — the floors are only right while
the numbers under them hold.

- [x] **Grading blocker.** `props.py --save` writes its Game row as
  `status="scheduled"` (an Odds API event carries no score) and
  `grade_prop_picks` skips any pick whose game is not final, so the whole
  paper trade would have graded ZERO picks. `sports/nfl/results.py`
  `finalize_nfl_games()` now fills scores from nflreadpy (free, whole season)
  and grade.py calls it before grading props. `extract.py postgame` is no
  longer required in the weekly loop.
- [x] **Correlated exposure.** Props never ran through
  `_apply_same_game_correlation_adjustment`, and exposure was extreme: 19.3
  picks/game on average, 72.8% of picks a player bet in BOTH markets, 88% of
  those the same side, and doubled players lost both legs 27.7% of the time
  vs 17.1% under independence. props.py now keeps one pick per player (best
  market) and applies the same-game Kelly scaling. De-concentrating also
  *raised* the hit rate (59.7% → 62.2%).
- [x] **Edge floors raised and split per market.** Realized hit rate rises
  monotonically with the floor; receiving yards runs ~5pp worse than
  receptions at every floor. `PROP_EDGE_FLOORS` = 10% receptions / 15% yards
  (was a flat 5%), in `sports/nfl/props.py` so props.py and replay.py share
  them; `--min-edge` still overrides.
- [x] **Calibrator anchored on realistic lines.** The isotonic layer was fit
  only on pseudo-lines around our own shrunk mean, but real lines sit above
  it, so we were betting in a region the calibrator never saw. It now also
  trains on `book_proxy_line()` placements. Overconfidence narrowed from
  -6.0/-7.3pp (over/under) to -4.6/-6.0pp, and receiving-yards ROI went
  +7.5% → +11.2%. The Phase 3 gate still passes (receptions Brier 0.1920 vs
  0.2414 naive; yards 0.2205 vs 0.2301).

Net on the same eval data: **59.7% hit / +13.9% ROI → 64.2% hit / +22.6% ROI**,
with the confidence gap down from -8.9pp to -5.4pp. Absolute ROI is against
soft median lines and is NOT a forecast — the comparison is the point.

- [ ] Residual overconfidence is still ~5pp and grows with claimed
  confidence (the 25%+ bucket claims 77% and delivers 65%). The floors cover
  it, but a per-projection uncertainty signal (games played, role stability)
  would beat a global isotonic map. Revisit if paper trading confirms it.
  - Tested and REJECTED (2026-08-19): capping claimed p at 0.72-0.80 —
    Kelly ROI on the 7,504-pick eval was flat-to-slightly-worse, because the
    80%+ bucket, though overconfident (claims 84.5%), still realizes 68.9%,
    the best of any bucket. Probability shrinkage (λ=0.7-0.8 + refilter) is
    algebraically identical to raising the floors: +3.6pp ROI per bet but
    28% fewer picks and lower total profit. A two-week replay showing 50%
    at 80%+ was n=10 noise. Leave selection as is; do NOT re-add a cap
    without new evidence from live lines.

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

### Housekeeping — swept 2026-08-19

All done in one pass (the sweep also caught a live ImportError:
`make_stat_lookup` imported `NFLDataLoader`, but the class is `NFLLoader` —
prop grading would have crashed on first real use):

- [x] CLAUDE.md refreshed: 391 tests, four sports (NHL/MLB frozen), props +
  ledger commands, validator/ledger in the architecture map, real-line NFL
  backtest, nflreadpy `temp`/`wind` weather columns.
- [x] Season is sport-aware: `SportConfig.season_for_date(date)`
  (`season_start_month=8` for NFL/NBA/NHL); picks.py uses it, so Jan/Feb
  games no longer trigger Elo mean reversion mid-season.
- [x] Stale `saved_models/*_backtest_tmp/` dirs deleted (the code already
  writes fold models to a scratch dir outside saved_models).
- [x] Sentiment now shifts `model_prob` (clipped [0.01, 0.99]) before the
  edge is computed, so the edge that selects the bet and the probability
  Kelly sizes with agree.
- [x] Synthetic backtest moneyline now uses vig-removed
  `calculate_edge_fair()`, matching live picks. (Real-line path already did.)
- [x] MLB/NHL frozen: `SportConfig.active=False`; `available_sports()`
  hides them by default (`include_frozen=True` to list); explicit `--sport`
  still works; grade.py's Discord loop skips them.
- [x] `home_hits` collision fixed: NHL body checks renamed to
  `home_nhl_hits`/`away_nhl_hits` (migration `c9d2e3f4a5b6`, applied);
  `home_hits` stays MLB batting.
- [x] Prop DNP void policy: when a week's stats are published for the
  game's teams but the player has no row, the pick voids (result `"void"`,
  pnl 0, stake excluded from total_wagered) — matching book settlement.
  Unpublished week still leaves it ungraded. A name that never matches
  nflreadpy's spelling voids too — check the void log line if a star's
  pick voids unexpectedly.

Still open:

- [ ] **NBA pick path is broken and deprioritized** (user is NFL-only for
  fall 2026). Confirmed 2026-08-19: `picks.py`'s `keep_cols` trim drops the
  box-score columns from history, so `compute_nba_advanced_rolling()`
  no-ops and 52 of 91 trained NBA features arrive zero-filled at pick time
  (NFL-F8 all over again). Fix before ever trusting NBA picks: pass the
  loader's box columns through the history trim, retrain NBA to write
  `training_meta.json`, and verify `_align()` reports no zero-fills.

## `max_edge_pct` guardrail — decision recorded 2026-08-19

Keep the 15% cap as-is. Game markets are not being bet with real money
(Phase 1 verdict), so the cap only protects paper runs; and a game-market
edge over 15% is, per the model-vs-market diagnostic, almost certainly model
error rather than value. Props don't route through `_passes_guardrails` —
their protection is the 5% edge floor plus paper trading. Revisit only if a
game market ever comes back into scope.

## `guardrails-wip` branch — deleted 2026-08-19

The calibration changes were already superseded on `main`, and the remaining
hyperparameter experiment (`max_depth` 3→4, `eta` 0.01→0.02) is moot: the
game-level model loses to the close regardless of tuning (Phase 1), so the
retrain it required isn't worth the compute. The commit was `1bacf2b` if it
ever needs archaeology.
