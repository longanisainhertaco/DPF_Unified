# Proposal: S1-A6 Segmented Resume Cumulative-Ledger Continuity

Status: implemented_candidate
Sprint: 1
Blocker IDs: A-6
Claim allowed: a segmented whole-shot run resumed from a mid-horizon checkpoint
reconstructs candidate cumulative ledgers covering the full executed horizon.
Claim forbidden: a completed or accepted 12 us source-sign whole-shot run;
accepted restart-equivalence authority.

## 1. Scope

The package-native segmented whole-shot runner
`src/dpf/first_principles/segmented_whole_shot.py`,
`run_segmented_whole_shot()`. Phase interval: any multi-segment horizon executed
with checkpointing. Observable: the `cumulative_ledgers` block of a resumed
run's `run_manifest`, specifically `covers_executed_horizon`,
`limiter_steps_observed`, `cumulative_j_dot_e_step_count`, and
`cumulative_active_port_step_count`. General to the segmented runner; not
PF-1000-specific.

## 2. Local Source Authority

| Source path | Lines | Supports | Limits |
| --- | --- | --- | --- |
| docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md | 199-228 | A-6 resume-ledger requirement and the required regression test | control-gate / runtime-engineering only |

No `KnowledgeReference/` physics source is invoked: the cumulative ledger is a
bookkeeping accumulator, not a physics model.

## 3. Equations And Symbol Map

The cumulative ledger is an additive accumulator. For a horizon executed as
segments `0..N`, each counter `X` satisfies

  `X_cumulative = sum over k in 0..N of X_segment_k`.

The defect: on resume from the checkpoint after segment `m`, segments `0..m` are
skipped and never added, so the resumed run computed
`sum over k in m+1..N` instead of the full sum. No physical units or sign
conventions are involved; the counters are step counts (dimensionless), work
(J), and final-state scalars (A, C, K, m^-3, count).

## 4. Validity Regime

Resume is valid only when the fail-closed checkpoint loader accepts the
checkpoint (same grid shape and closure configuration as the resumed deck — see
`test_resume_from_mismatched_deck_fails_closed`). The cumulative-ledger sidecar
must sit beside the resumed-from `.npz`; if it is absent (a legacy or
hand-supplied checkpoint), resume degrades to a zero ledger baseline — the
pre-fix behavior — rather than crashing.

## 5. Method

A per-checkpoint cumulative-ledger sidecar:

1. When the runner writes a checkpoint after segment `k`, `_CumulativeLedgers`
   has already accumulated segments `0..k`. The runner writes
   `_CumulativeLedgers.to_state_dict()` to a sidecar JSON beside the checkpoint
   (`<checkpoint-stem>.cumulative_ledger.json`).
2. On resume from `segment_kkkk.npz`, the runner derives the sidecar path,
   reads it, and rehydrates `_CumulativeLedgers` via the new `from_state_dict`
   classmethod before the segment loop — instead of `_CumulativeLedgers()`.
3. Fresh (non-resumed) runs are unchanged: still `_CumulativeLedgers()`.

The sidecar was chosen over embedding counters in the checkpoint `.npz` because
it keeps the change contained to `segmented_whole_shot.py` and leaves the
checkpoint schema and its fail-closed loader untouched (smallest blast radius;
`../UNKNOWN_AND_INFERENCE_LOG.md` I-4). Loading the prior run manifest was
rejected because the manifest holds only the final full-horizon ledger, not the
as-of-segment-`k` state a mid-horizon resume needs.

## 6. Implementation Plan

Implemented in commit `55e3f94`. `src/dpf/first_principles/segmented_whole_shot.py`:
`_CumulativeLedgers._STATE_FIELDS`, `.to_state_dict()`, `.from_state_dict()`;
helpers `_cumulative_ledger_sidecar_path()` and
`_load_cumulative_ledger_sidecar()`; sidecar write at checkpoint time; resume
rehydration. No other file changed.

## 7. Test Plan

`tests/test_first_principles_segmented_whole_shot.py` —
`test_resume_cumulative_ledgers_cover_full_executed_horizon` (new): a 4-step
horizon, `segment_steps=2`, resumed from the segment-0 checkpoint at step 2,
asserts `total_steps_completed == 4`, `limiter_steps_observed == 4`,
`covers_executed_horizon is True`, J·E and active-port step counts span 4 steps,
and — strongest form — every cumulative counter equals a fresh uninterrupted
4-step run's. The existing `test_resume_from_checkpoint_completes_the_horizon`,
`test_resume_from_mismatched_deck_fails_closed`, and restart-equivalence tests
still pass: 21 passed total. Negative control: reverting the rehydration line
fails the new test with `assert 2 == 4`, reproducing the audit probe.

## 8. Runtime Artifacts

New: `<checkpoint-stem>.cumulative_ledger.json` sidecars beside each non-final
segment checkpoint, carrying `ledger_sidecar_status`,
`candidate_cumulative_segmented_ledger_resume_sidecar`, `checkpoint_path`,
`segment_index`, `total_steps_completed_after_segment`, and
`cumulative_ledger_state`. The sidecar is candidate engineering telemetry; it
carries no acceptance field and supports no first-principles claim. The
resumed run's `run_manifest` retains all fail-closed labels.

## 9. Acceptance And Rejection Criteria

Accept as engineering progress when: a resumed run's `cumulative_ledgers`
covers the full executed horizon and equals an uninterrupted run's; fresh runs
are unregressed; restart-equivalence still holds. Reject if: resume completes
while `covers_executed_horizon` is false; or the change touches the checkpoint
schema / fail-closed loader; or it claims a completed 12 us run. None hold.

This does not constitute a completed or accepted 12 us whole-shot run; the
compute-wall blocker (WP-N4B) is open.

## 10. Open Questions

- A legacy checkpoint with no sidecar resumes from a zero ledger baseline. This
  is the documented graceful-degradation path, not a silent error; a follow-up
  could emit a one-line warning. Owner: runtime engineering. Not blocking.

## 11. AI And External Tool Disclosure

Implemented by a Claude Opus 4.7 (Claude Code) delegated agent under the Codex
audit specification, then verified by the lead against the live diff, the
21-test segmented suite, the negative control, and the 275-test broad suite. No
external papers, repositories, or web sources were used. Pending human review
before push. The checkpoint-metadata embedding option was considered and not
used (blast radius); the sidecar option was implemented.
