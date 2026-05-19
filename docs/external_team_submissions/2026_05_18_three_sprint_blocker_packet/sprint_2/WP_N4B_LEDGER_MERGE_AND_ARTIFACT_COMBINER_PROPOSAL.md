# Proposal: WP-N4B Cross-Restart Ledger Merge and Artifact Combiner

Status: proposed
Sprint: 2
Blocker IDs: WP-N4B
Claim allowed: a proposed engineering mechanism to combine N segmented-run
restarts into one whole-run ledger and artifact.
Claim forbidden: a completed 12 us run; an accepted whole-shot artifact.

## 1. Scope

The package-native segmented whole-shot runner
`src/dpf/first_principles/segmented_whole_shot.py`. This proposal covers
combining the outputs of N separate operating-system-level process restarts —
each a distinct `run_segmented_whole_shot()` invocation — into one whole-run
cumulative ledger and one whole-run artifact. Pure engineering orchestration; no
physics change.

## 2. Local Source Authority

Not applicable — engineering orchestration. The authority is the audit's WP-N4B
questions and the Sprint 1 segmented runner code (commit `55e3f94`). No
`KnowledgeReference/` physics source is invoked.

## 3. Equations And Symbol Map

Not applicable. The ledger merge is integer/float counter addition; no physics
equations.

## 4. Validity Regime

Applies to multi-restart runs where each restart is a separate
`run_segmented_whole_shot()` invocation resuming from the previous restart's
final checkpoint. The merge is valid only when the restarts tile the horizon
contiguously (no step gap, no overlap).

## 5. Method

Sprint 1 (commit `55e3f94`) delivered single-resume rehydration: resuming within
one `run_segmented_whole_shot()` call reconstructs `_CumulativeLedgers` from a
per-checkpoint sidecar. **Cross-restart merge across N separate process
invocations is absent**, and so is a whole-run artifact combiner.

Cross-restart ledger merge — `merge_cumulative_ledgers(manifest_paths)`:
1. Read each restart's `run_manifest.json["cumulative_ledgers"]`.
2. Sort manifests by `resume_started_at_step`.
3. Verify the first restart starts at step 0; fail closed if not (suffix run,
   not whole run).
4. Verify that every manifest's cumulative counters are non-decreasing relative
   to the preceding manifest in step order (monotonicity invariant); fail
   closed with an attributable `LedgerMergeError` if any counter decreases.
   This proves every manifest was rehydrated from the A-6 per-checkpoint
   sidecar (which carries the full prefix into every resume), not just per-
   restart-segment totals.
5. Verify the restarts tile contiguously (no step gap, no step overlap); fail
   closed otherwise.
6. Take the whole-run additive counters **directly from the terminal restart's
   `cumulative_ledgers`** (the manifest with the highest
   `total_steps_completed`).  Do NOT sum the per-restart ledger blocks across
   all N manifests.

   Rationale: summing would double-count.  Because the A-6 sidecar rehydrates
   the full prefix into every resumed run, restart k's `cumulative_ledgers`
   already contains the totals for steps 0..k_end, not only for its own
   post-resume steps.  Summing restart 0's ledger (steps 0..2) with restart
   1's ledger (steps 0..4) would count steps 0..2 twice.  The terminal
   manifest's ledger is the single correct whole-run total; we take it once.

7. Take final-state scalars (`final_circuit_current_A`, `final_*`) from the
   same terminal manifest.
8. Emit a merged ledger with
   `covers_executed_horizon = sum(executed_steps) >= planned_total_steps`,
   where `executed_steps` is the sum of each restart's own contribution
   (`total_steps_completed - resume_started_at_step`) to avoid re-counting
   the sidecar prefix.

Whole-run artifact combiner — `combine_whole_run_artifacts(run_dirs)`:
1. Input: an ordered list of `run_dir` paths, each with `run_manifest.json`,
   `plan.json`, `segments/`.
2. Contiguity check:
   `run_dirs[k].resume_started_at_step == run_dirs[k-1].total_steps_completed`
   for every `k`; fail closed otherwise.
3. Segment inventory: collect every `segment_NNNN.manifest.json` in global step
   order, re-indexing segment numbers monotonically across restarts.
4. Combined ledger: via `merge_cumulative_ledgers` above.
5. Checkpoint inventory: list every `.npz` with its step offset; the latest
   checkpoint is the resume point for any further restart.
6. Output: one `whole_run_combined_manifest.json` with
   `horizon_complete = total_combined_steps >= planned_total_steps` and all
   fail-closed labels.

Field time-series note: an NPZ checkpoint cannot be stream-appended, so
per-segment `.npz` checkpoints stay individually addressable by step index
rather than being concatenated.

## 6. Implementation Plan

New functions in `src/dpf/first_principles/segmented_whole_shot.py` (or a new
`segmented_whole_shot_combine.py`): `merge_cumulative_ledgers()` and
`combine_whole_run_artifacts()`. Reuse `_CumulativeLedgers.from_state_dict()` /
`to_state_dict()` from Sprint 1. Do not modify the checkpoint `.npz` schema or
its fail-closed loader. Proposed for a dedicated WP-N4B implementation session.

## 7. Test Plan

- Positive: a 4-step horizon executed as two separate `run_segmented_whole_shot`
  invocations (2 + 2); assert the merged ledger equals an uninterrupted 4-step
  run's for every counter — the same equivalence form as the Sprint 1 A-6 test.
- Negative: a non-contiguous restart pair (a step gap, and an overlap) => the
  merge/combiner fails closed with an attributable error.
- Negative: a missing per-restart `run_manifest.json` => fail closed.

## 8. Runtime Artifacts

`whole_run_combined_manifest.json` carrying the merged cumulative ledger, the
re-indexed segment inventory, the checkpoint inventory, `horizon_complete`, and
all fail-closed manifest labels. No acceptance promotion.

## 9. Acceptance And Rejection Criteria

Accept engineering progress when: `merge_cumulative_ledgers` and
`combine_whole_run_artifacts` produce `covers_executed_horizon=True` and
`horizon_complete=True` from N>=2 separate invocations tiling a test horizon,
and the contiguity negative tests fail closed. This is engineering
infrastructure only — it is never evidence of a completed or accepted 12 us run.
Reject if a combined artifact claims `horizon_complete` without a contiguity
check, or claims acceptance.

## 10. Open Questions

- Does the 12 us run need retained field snapshots beyond each segment's
  terminal `.npz`? If yes, the combiner must index — not concatenate — them.
  Owner: WP-N4B implementation.

## 11. AI And External Tool Disclosure

Engineering analysis by a Claude Opus 4.7 (Claude Code) agent, lead-reviewed
against the Sprint 1 segmented-runner code. No external sources used. No code
implemented this sprint.
