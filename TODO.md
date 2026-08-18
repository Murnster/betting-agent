# TODO

Open follow-ups. Add items here rather than leaving them in commit messages
or session scrollback.

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
