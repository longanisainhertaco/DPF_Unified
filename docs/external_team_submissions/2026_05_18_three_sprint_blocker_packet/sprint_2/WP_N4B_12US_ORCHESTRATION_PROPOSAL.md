# Proposal: WP-N4B 12 us Source-Sign Run Orchestration

Status: proposed
Sprint: 2
Blocker IDs: WP-N4B, DPF-PHYS-023
Claim allowed: a job plan, a measured per-step compute floor, and a production
ladder for a 12 us segmented run.
Claim forbidden: a completed 12 us source-sign run; an accepted whole-shot
result; a production wall-clock estimate (still blocked on grid size).

## 1. Scope

A 12 us source-sign whole-shot run of the package-native first-principles 3-D
runtime via `src/dpf/first_principles/segmented_whole_shot.py`. PF-1000/Akel
scoped. Engineering orchestration; no physics change.

## 2. Local Source Authority

Not applicable — engineering. Authority: the audit WP-N4B questions, the Sprint 1
segmented runner (`55e3f94`), and the timing measured for this proposal
(section 5).

## 3. Equations And Symbol Map

Not applicable.

## 4. Validity Regime

Applies to fixed-step runs at `dt = 1.0e-13 s` — the verified default of the
first-principles-3d and experimental-whole-shot paths (`runner.py:153`,
`cli/main.py:425`, `cli/main.py:2743`, and the `experimental_shot.py` presets).
The experimental dt-policy logic only lowers `dt`, never raises it, so 1e-13 s
is the step-count floor.

## 5. Job Plan, Measured Compute Floor, and Production Ladder

12 us at `dt = 1e-13 s` is **120,000,000 steps**. At a segment size of 10,000
steps that is **12,000 segments**; at 100,000 steps, **1,200 segments**.

Measured per-step compute floor (this session, this machine, Apple Silicon /
MLX): `dpf first-principles-3d --deck-preset pf1000_akel_16kv` at the compact
grid `[5,5,5]`, `dt=1e-13`:

- 2 steps: 2.27 s wall (fixed process + import + MLX init + artifact write)
- 300 steps: 3.83 s wall
- marginal: 298 steps in 1.56 s => **5.23 ms/step** at grid `[5,5,5]`

12 us floor at the `[5,5,5]` compact grid: `1.2e8 x 5.23e-3 s ~ 6.28e5 s ~
174 hours ~ 7.3 days`.

This is a **lower bound only**. A production PF-1000 run uses the reviewed WP-N3
geometry grid, which is far larger than 125 cells; per-step cost grows roughly
with cell count. **The production-grid 12 us wall-clock is BLOCKED** until the
WP-N3 grid size is fixed. What is now established: even at the smallest possible
grid, 12 us is a multi-day run — the audit's `B-WPN4-12US-COMPUTE-WALL` blocker
verdict is confirmed quantitatively.

Production ladder (step counts exact; wall-clock = `[5,5,5]` floor, a lower
bound):

| Rung | Horizon | Steps | Segments @10k | Floor wall-clock @[5,5,5] |
| --- | --- | --- | --- | --- |
| 1 | 10 ns | 100,000 | 10 | ~8.7 min |
| 2 | 100 ns | 1,000,000 | 100 | ~1.5 h |
| 3 | 1 us | 10,000,000 | 1,000 | ~14.5 h |
| 4 | 12 us | 120,000,000 | 12,000 | ~174 h (7.3 days) |

## 6. Implementation Plan

The Sprint 1 segmented runner already provides the static planner, segmented
execution, checkpointing, per-segment manifests, cumulative ledgers, and
single-resume rehydration. WP-N4B still needs:
- the cross-restart ledger merge and artifact combiner —
  `WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`;
- a wall-time-sliced production driver that runs rung-by-rung, resuming each
  restart from the prior restart's final checkpoint;
- a production-grid per-step measurement once the WP-N3 grid exists.

## 7. Test Plan

Climb the ladder; each rung gates the next:
- Rung 1 (10 ns): `build_staged_restart_equivalence_evidence` must prove
  bit-identical restart across the 10 segments.
- Rung 2 (100 ns): resume from rung 1's final checkpoint; verify ledger
  continuity (the Sprint 1 A-6 mechanism).
- Rung 3 (1 us): first multi-process-restart; requires the cross-restart merge
  to produce `covers_executed_horizon=True`.
- Rung 4 (12 us): requires merge + combiner exercised at rung 3, and a
  production-grid wall-clock estimate.

## 8. Runtime Artifacts

Per-segment `segment_NNNN.manifest.json`, per-checkpoint `.npz` and
cumulative-ledger sidecars, per-restart `run_manifest.json`, and (via the
combiner proposal) a `whole_run_combined_manifest.json`. Every partial run
carries the `B-WPN4-12US-COMPUTE-WALL` and, if tripped, `B-WPN4-WALL-TIME-CAP`
blocker verdicts.

## 9. Acceptance And Rejection Criteria

`horizon_complete` is set only by `completed >= plan.total_steps`
(`segmented_whole_shot.py:557`), guarded by: the wall-time-cap break, step counts
taken from real `telemetry.n_steps_completed` (never speculative), and the
restart-equivalence gate refusing to run if `not horizon_complete`.

Accept engineering progress when ladder rungs 1-2 pass with proven restart
equivalence and ledger continuity. A completed 12 us run is **not claimable**
until: rungs 3-4 pass, the cross-restart merge + combiner exist, and a
production-grid wall-clock is measured and operationally feasible. Reject any
artifact labelled `horizon_complete` for a partial run.

## 10. Open Questions

- Production grid size (WP-N3 dependency) — blocks the real wall-clock. A 7.3-day
  floor at `[5,5,5]` implies a production run could be weeks to months; whether
  that is operationally acceptable, or whether a faster path is needed, is an
  open decision for the project owner. Owner: WP-N4B + WP-N3.
- The audit notes the dt-policy can lower `dt`; if a source-sign run needs a
  smaller `dt` than 1e-13 s the step count and wall-clock rise proportionally.

## 11. AI And External Tool Disclosure

Engineering analysis by a Claude Opus 4.7 (Claude Code) agent; the per-step
timing was measured by the lead on this machine this session. No external
sources used. No production code implemented this sprint.
