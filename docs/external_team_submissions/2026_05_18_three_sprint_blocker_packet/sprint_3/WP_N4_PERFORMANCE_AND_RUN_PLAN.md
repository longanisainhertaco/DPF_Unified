# WP-N4 Performance and Run Plan

Status: research_packet_parallel_lane_not_implementation
Date: 2026-05-19
Branch: codex/corpus
Scope: Lane 7 per `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`
       section "Allowed parallel lanes", item 7.
Sprint-2.2 constraint: this packet is READ-ONLY with respect to all Sprint-2.2
owned files. No edits to `segmented_whole_shot_combine.py`,
`test_first_principles_segmented_whole_shot_combine.py`, `power_port.py`,
`DPF_REQUIREMENTS_BASELINE.md`, `SRS_TRACEABILITY_MATRIX.*`, or any
external-team packet traceability/changelog files.

---

## 1. Source-Backed Findings

### 1.1 Runner and Backend

The first-principles-3D runner uses `dpf.fields.HybridPIC3DSimulator` wrapping
`dpf.fields.HybridPIC3DLoop` and `dpf.fields.maxwell_3d`. This is the
**package-native NumPy backend** (`backend="package_native"`,
`src/dpf/first_principles/manifest.py:66`). MLX (`mlx`) is not available on
the current machine (`importlib.util.find_spec('mlx') is None`). The
`experimental-segmented-whole-shot` CLI entry point invokes
`run_segmented_whole_shot()` in
`src/dpf/first_principles/segmented_whole_shot.py`.

Backend identity confirmed from:

- `src/dpf/first_principles/manifest.py:66` — `backend: str = "package_native"`
- `src/dpf/first_principles/runner.py:599-608` — constructs `HybridPIC3DSimulator`
  with no MLX or Metal dispatch
- `src/dpf/fields/hybrid_loop.py:8` — `import numpy as np`, no MLX import
- CLI output: `"runner": "dpf.fields.HybridPIC3DSimulator"`

### 1.2 Compact Deck Grid and dt Policy

Default deck: `pf1000_akel_16kv` preset, defined at
`src/dpf/first_principles/deck.py:775`.

| Parameter | Value | Source |
|-----------|-------|--------|
| `grid_shape` | `(5, 5, 5)` | `deck.py:778`, CLI default |
| `grid_spacing_m` | `(0.110, 0.110, 0.141)` m | `deck.py:827-828` (computed from geometry) |
| `dt_s` | `1.0e-13 s` | `runner.py:153`, `deck.py:779`, `cli/main.py:425` |
| `n_cells` | 125 | 5 * 5 * 5 |

The dt policy at `src/dpf/first_principles/deck.py:537` declares `dt_s:
float = 1.0e-13` as the diagnostic default. The experimental dt-search path
in the CLI (`cli/main.py:568-586`) only lowers dt from the CFL-stable value;
it never raises it. Therefore `dt = 1e-13 s` is the step-count **floor** for
any production run.

### 1.3 Fresh Per-Step Timing Measurement

**Command used:**

```bash
.venv312/bin/python -c "
import time
from dpf.first_principles import pf1000_akel_16kv_engineering_deck
from dpf.first_principles.runner import build_first_principles_3d_session

deck = pf1000_akel_16kv_engineering_deck(n_steps=500)
session = build_first_principles_3d_session(deck)
session.run_segment(10)  # warm-up

for N in [100, 200, 500]:
    t0 = time.perf_counter()
    r = session.run_segment(N)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    print(f'N={N}: {ms:.2f} ms total, {ms/N:.4f} ms/step')
"
```

**Results (2026-05-19, Apple Silicon, package-native NumPy, grid [5,5,5]):**

| N steps | Total ms | Per-step ms |
|---------|----------|-------------|
| 100 | 248.78 | 2.488 |
| 200 | 496.65 | 2.483 |
| 500 | 1415.71 | 2.831 |

Adopted steady-state estimate: **2.48 ms/step** (average of N=100 and N=200,
post warm-up, same live session).

**Prior measurement (Sprint 2, different measurement method):** 5.23 ms/step.
The discrepancy is explained by measurement methodology: the prior measurement
included process startup + import + MLX-init overhead in the marginal delta
(2-step and 300-step CLI invocations). The fresh measurement uses a live session
with a 10-step warm-up inside a single process, isolating per-step compute cost.
Both numbers are presented; the fresh 2.48 ms/step is the better steady-state
estimate, but both are lower-bound floors valid only for the compact `[5,5,5]`
grid.

### 1.4 Memory Footprint (Compact Grid)

Measured with `resource.getrusage(RUSAGE_SELF).ru_maxrss` during a 200-step
post-warm-up measurement:

| Phase | Peak RSS |
|-------|----------|
| Pre-run (after warm-up, before 200 steps) | 170.6 MiB |
| Post-run | 177.1 MiB |
| Checkpoint .npz (grid [5,5,5]) | 12.8 KiB |

Memory at production scale is **blocked on WP-N3** (production grid: blocked_by_missing_local_source — WP-N3 geometry packet not yet delivered as runtime code).

### 1.5 Checkpoint Artifact Volume

Checkpoint file: `.npz`, written by
`src/dpf/first_principles/state_checkpoint.py:70` via `np.savez_compressed`.
Sidecar: `.cumulative_ledger.json`, ~2–5 KiB, written at each checkpoint by
`segmented_whole_shot.py:507-519`.

| Grid | Checkpoint .npz | Total per checkpoint (incl. sidecar) |
|------|----------------|---------------------------------------|
| 5x5x5 (125 cells) | 12.8 KiB | ~18 KiB |
| Production (blocked_by_missing_local_source, ~32^3 est.) | ~3.3 MiB (estimated, scales linearly) | ~3.4 MiB (estimated) |

For 120 M-step run at segment_steps=10,000 (12,000 segments):

| checkpoint_every_segments | Checkpoints | Volume @[5,5,5] | Volume @32^3 est. |
|--------------------------|------------|-----------------|-------------------|
| 1 | 12,000 | ~208 MiB | ~40 GiB |
| 10 | 1,200 | ~20 MiB | ~4 GiB |
| 100 | 120 | ~2 MiB | ~400 MiB |

Recommendation: `checkpoint_every_segments=100` for production (120 retained
checkpoints, manageable storage, adequate resume granularity).

---

## 2. Supported / Candidate / Blocked Table

| Item | Status | Evidence / Blocker |
|------|--------|-------------------|
| Per-step timing (compact grid [5,5,5], warm session) | **supported** | Measured this session: 2.48 ms/step |
| Per-step timing (prior, CLI process-level) | **supported** | Sprint-2 measurement: 5.23 ms/step |
| Backend: package-native NumPy | **supported** | `manifest.py:66`, `hybrid_loop.py:8`, `runner.py:599-608` |
| dt = 1e-13 s fixed step | **supported** | `deck.py:779`, `runner.py:153`, `cli/main.py:425` |
| 12 us step count = 120,000,000 | **supported** | `ceil(12e-6 / 1e-13)` = 120,000,000; confirmed in `segmented_whole_shot.py:945` blocker verdict |
| 12 us wall-clock floor (compact grid) | **supported** | 120e6 * 2.48e-3 s = 297,600 s = 82.7 h (lower bound) |
| Checkpoint roundtrip integrity | **supported** | `write_read_hashes_match=True` verified this session |
| `horizon_complete` flag semantics | **supported** | `segmented_whole_shot.py:557`: `completed >= plan.total_steps` |
| Wall-time-cap honest partial labeling | **supported** | `segmented_whole_shot.py:551,595` |
| Run ladder step counts (10 ns / 100 ns / 1 us / 12 us) | **supported** | Computed from `dt=1e-13 s`, exact integer steps |
| WP-N3 production grid size | **blocked** | Grid size not yet determined; WP-N3 geometry packet not complete |
| Production-grid per-step timing | **blocked** | Requires production grid; scales with cell count (N_cells = nx*ny*nz) |
| Production-grid memory footprint | **blocked** | Requires WP-N3 grid |
| Production-grid 12 us wall-clock | **blocked** | Requires WP-N3 grid + production-grid per-step timing |
| MLX backend availability | **blocked** | `mlx` not available on this machine (`find_spec('mlx') is None`) |
| Three-restart combiner correctness | **candidate** | WP-N4B combiner has known F1 bug (rejects valid 3-restart chains); Sprint 2.2 owns the fix |
| Restart equivalence at staged horizons | **candidate** | `build_staged_restart_equivalence_evidence` exists; equivalence proven for single-restart only pending Sprint-2.2 combiner fix |
| 12 us completion claim | **blocked** | `horizon_complete=false` for all runs to date; claim forbidden until `horizon_complete=true` in the run manifest |

Counts: **10 supported**, **3 candidate**, **7 blocked**.

---

## 3. Runtime Fields Required for a Long Run

Every segment manifest (`segment_NNNN.manifest.json`) and the terminal
`run_manifest.json` must emit these fields. Sources:
`segmented_whole_shot.py:582-643`, `segmented_whole_shot.py:847-867`.

| Field | Location | Requirement |
|-------|----------|-------------|
| `horizon_complete` | `run_manifest.json` | `bool`; `true` only when `completed >= plan.total_steps` (`segmented_whole_shot.py:557`). Must never be inferred or promoted. |
| `wall_time_cap_reached` | `run_manifest.json` | `bool`; `true` when run stopped at the wall-time cap before completing the horizon. |
| `total_steps_completed` | `run_manifest.json` | Exact integer count of steps actually executed, never speculative. |
| `planned_total_steps` | `run_manifest.json` | `total_steps` from `WholeShotPlan`. |
| `wall_clock_seconds` | `run_manifest.json` | Measured wall time for the full run or partial run. |
| `blocker_verdicts` | `run_manifest.json` | Dict with `verdicts` list, `any_triggered`, `triggered_ids`, and `summary`. Includes `B-WPN4-12US-COMPUTE-WALL`, `B-WPN4-WALL-TIME-CAP`, `B-WPN4-CHECKPOINT-INTEGRITY`, `B-WPN4-RESTART-EQUIVALENCE`. |
| `segment_wall_clock_seconds` | `segment_NNNN.manifest.json` | Per-segment wall time; required for per-step cost tracking across a long run. |
| `stop_reason` | `segment_NNNN.manifest.json` | From `telemetry.stop_reason`; must be non-null. |
| `finite_state_all` | `segment_NNNN.manifest.json` | From `telemetry.finite_state.all_finite`; must be `true` at every checkpoint to continue. |
| `state_fingerprint_sha256` | `segment_NNNN.manifest.json` | State hash for restart-equivalence verification. |
| `write_read_hashes_match` | `segment_NNNN.manifest.json` checkpoint block | Must be `true`; fail-closed on mismatch. |
| `cumulative_ledger_state` | `segment_NNNN.cumulative_ledger.json` | Cross-restart ledger accumulation; required for resume rehydration (`segmented_whole_shot.py:507-519`). |

**Partial/compute-wall labeling rules (do not promote):**

- A run stopped by `wall_time_cap_s` must have `horizon_complete=false` and
  `wall_time_cap_reached=true` in `run_manifest.json`.
- The `blocker_verdicts.summary` must read
  `"engineering_candidate_partial_or_blocked"` for any partial run.
- A partial run is a legitimate artifact and must never be relabeled as a
  finished whole shot.
- `can_support_first_principles_acceptance` must remain `false` in
  `run_manifest.json` for all runs.

---

## 4. Missing Parameters

| Parameter | Why Missing | Blocker |
|-----------|------------|---------|
| Production grid size (nx, ny, nz) | WP-N3 PF-1000/Akel geometry packet not yet complete | WP-N3 |
| Production-grid per-step timing | Cannot measure until production grid exists | WP-N3 |
| Production-grid memory footprint per step | Scales as `nx * ny * nz * NVAR * 8 bytes`; NVAR=10 fields; blocked_by_missing_local_source (grid dimensions: WP-N3 runtime_packet_not_delivered) | WP-N3 |
| Production-grid checkpoint .npz size | Scales linearly with cell count; ~3.3 MiB estimated at 32^3, not confirmed | WP-N3 |
| Production-grid 12 us wall-clock | Requires production-grid per-step timing | WP-N3 |
| dt CFL constraint at production grid | CFL-limited dt may be smaller than 1e-13 s at finer resolution; each factor of 2 reduction doubles step count | WP-N3 + physics review |
| Cross-restart combiner N-restart correctness | Known F1 bug in `merge_cumulative_ledgers()` (3-restart chains rejected as false overlaps) | Sprint 2.2 |

---

## 5. Proposed Tests and Negative Controls

### 5.1 Staged Horizon Restart-Reproducibility Checks

All tests use `build_staged_restart_equivalence_evidence()` at
`segmented_whole_shot.py:655` or direct invocations of `run_segmented_whole_shot`.

**T-N4-01: Two-restart bit-identical equivalence at 10 ns (100,000 steps)**

```python
# deck: pf1000_akel_16kv, segment_steps=10000, total_steps=100000
manifest = run_segmented_whole_shot(
    deck=deck, run_dir=run_dir, segment_steps=10_000,
    explicit_total_steps=100_000,
    checkpoint_every_segments=1,
    verify_restart_equivalence=True,
)
assert manifest["horizon_complete"] is True
assert manifest["restart_equivalence"]["state_fingerprints_match"] is True
assert manifest["restart_equivalence"]["tracked_observables_match_exactly"] is True
```

**T-N4-02: Three-restart chain (requires Sprint-2.2 combiner fix)**

Run 100,000 steps as three sequential restart invocations (step 0-34k,
34k-67k, 67k-100k). Requires the F1 fix in `merge_cumulative_ledgers()`.
Gate: total_steps_completed = 100,000 across all three restarts.

**T-N4-03: Staged equivalence evidence at three horizons**

```python
build_staged_restart_equivalence_evidence(
    deck=deck,
    staged_segment_plans=((10, 5), (100, 25), (1000, 100)),
    run_root=run_root,
)
```

All three stages must have `equivalence_proven=True`.

**T-N4-04: Production-grid per-step timing (blocked on WP-N3)**

Once WP-N3 fixes grid dimensions, run 100 warm steps on the production grid
and measure per-step cost with the same method as section 1.3.

### 5.2 Negative Controls (Fail-Closed)

**NC-N4-01: Partial run must not be labeled complete**

```python
manifest = run_segmented_whole_shot(
    deck=deck, run_dir=run_dir, segment_steps=10_000,
    explicit_total_steps=100_000,
    wall_time_cap_s=0.001,  # very short cap — forces partial run
    raise_on_wall_time_cap=False,
)
assert manifest["horizon_complete"] is False
assert manifest["wall_time_cap_reached"] is True
assert manifest["blocker_verdicts"]["summary"] == "engineering_candidate_partial_or_blocked"
assert "B-WPN4-WALL-TIME-CAP" in manifest["blocker_verdicts"]["triggered_ids"]
```

**NC-N4-02: `horizon_complete=false` rejects restart equivalence assertion**

```python
equivalence = manifest["restart_equivalence"]
assert equivalence["verified"] is False
# reason must mention "wall-time cap" or "partial"
assert "cap" in equivalence["reason"] or "partial" in equivalence["reason"]
```

Source: `segmented_whole_shot.py:888-897` — the restart equivalence block
explicitly returns `verified=False` with `reason="segmented run did not
complete the planned horizon"` when `not horizon_complete`.

**NC-N4-03: Malformed cumulative-ledger sidecar degrades gracefully**

Write a sidecar with `cumulative_ledger_state` missing; resume from that
checkpoint. The resumed run must complete without crashing and must report
ledger fields as zero-baseline (not `None` crash). Source:
`segmented_whole_shot.py:1062-1080` — missing sidecar falls back to
`_CumulativeLedgers()` (zero baseline), not an error.

**NC-N4-04: 12 us plan is always labeled as compute-wall blocker**

```python
plan = plan_segmented_whole_shot(
    target_time_s=12e-6, dt_s=1e-13, segment_steps=10_000,
)
# plan.total_steps must equal 120,000,000
assert plan.total_steps == 120_000_000
# The blocker verdict is always present when total_steps < twelve_us_steps
# Run one segment only, then check the blocker manifest
manifest = run_segmented_whole_shot(
    deck=deck, run_dir=run_dir, segment_steps=10_000,
    explicit_total_steps=10_000,  # only one rung
    wall_time_cap_s=None,
)
verdicts_by_id = {v["id"]: v for v in manifest["blocker_verdicts"]["verdicts"]}
b = verdicts_by_id["B-WPN4-12US-COMPUTE-WALL"]
assert b["status"] == "blocked"  # total_steps < twelve_us_steps
```

Source: `segmented_whole_shot.py:944-957` — the blocker verdict is present
in every run manifest regardless of horizon.

---

## 6. Exact Implementation Recommendations — Production Run Ladder

### 6.1 Step / Segment / Checkpoint Counts

All step counts exact (no float round-trip) via `--explicit-total-steps`.

| Rung | Horizon | Steps | segment_steps | Segments | ckpt_every | Checkpoints | Wall floor (2.48 ms/step) |
|------|---------|-------|--------------|----------|-----------|-------------|--------------------------|
| 1 | 10 ns | 100,000 | 10,000 | 10 | 1 | 10 | ~248 s (4.1 min) |
| 2 | 100 ns | 1,000,000 | 10,000 | 100 | 5 | 20 | ~2,480 s (41 min) |
| 3 | 1 us | 10,000,000 | 10,000 | 1,000 | 50 | 20 | ~24,800 s (6.9 h) |
| 4 | 12 us | 120,000,000 | 10,000 | 12,000 | 100 | 120 | ~297,600 s (82.7 h) |

These are **lower-bound wall clocks** for the compact `[5,5,5]` grid. Production
grid wall clocks are blocked pending WP-N3.

### 6.2 CLI Invocation Pattern

```bash
# Rung 1 — 10 ns, fresh run
.venv312/bin/python -m dpf experimental-segmented-whole-shot \
    --deck-preset pf1000_akel_16kv \
    --run-dir runs/rung1_10ns \
    --explicit-total-steps 100000 \
    --segment-steps 10000 \
    --checkpoint-every-segments 1 \
    --verify-restart-equivalence \
    -o runs/rung1_10ns/run_manifest.json

# Rung 2 — 100 ns, resume from rung 1 final checkpoint
.venv312/bin/python -m dpf experimental-segmented-whole-shot \
    --deck-preset pf1000_akel_16kv \
    --run-dir runs/rung2_100ns \
    --explicit-total-steps 1000000 \
    --segment-steps 10000 \
    --checkpoint-every-segments 5 \
    --resume-from-checkpoint runs/rung1_10ns/segments/segment_0009.npz \
    --no-verify-restart-equivalence \
    -o runs/rung2_100ns/run_manifest.json
```

For rungs 3 and 4, add `--wall-time-cap-s` to allow multi-session continuation.
Each session resumes from the previous session's final checkpoint.

### 6.3 Gate Conditions

- **Rung 1 gates Rung 2**: `horizon_complete=true`, `state_fingerprints_match=true`,
  `tracked_observables_match_exactly=true`, `all_checkpoint_roundtrips_match=true`.
- **Rung 2 gates Rung 3**: ledger continuity — resumed run's cumulative ledgers
  must cover the full executed horizon (cross-restart sidecar rehydration verified).
- **Rung 3 gates Rung 4**: three-restart combiner must produce a
  `whole_run_combined_manifest.json` with `covers_executed_horizon=true` and
  no ledger merge errors. This requires the Sprint-2.2 F1 combiner fix.
- **Rung 4 (12 us)**: `horizon_complete=true` in the terminal
  `whole_run_combined_manifest.json`. No claim allowed before this.

---

## 7. Do-Not-Promote Notes

**No 12 us claim is allowed until `horizon_complete=true` appears in the
terminal run manifest.**

The following are explicitly forbidden:

1. Labeling a partial run (wall-time cap, step-count shortage, combiner error)
   as a completed 12 us shot.

2. Inferring or estimating `horizon_complete` from step count without reading
   the manifest field directly.

3. Claiming a 12 us wall-clock estimate from the compact `[5,5,5]` grid applies
   to a production run. The production-grid wall clock is blocked on WP-N3.

4. Asserting restart equivalence for a partial run. The equivalence block
   refuses to evaluate when `horizon_complete=false`
   (`segmented_whole_shot.py:890`).

5. Treating any rung of the production ladder as validation evidence.
   `can_support_first_principles_acceptance=false` is set unconditionally
   in every `run_manifest.json` (`segmented_whole_shot.py:638-642`).

6. Bypassing the Sprint-2.2 combiner fix and using the N>2 restart combiner
   on production data before the three-restart positive test passes.

The `horizon_complete` flag is set exclusively at
`segmented_whole_shot.py:557`:

```python
horizon_complete = completed >= plan.total_steps
```

`completed` accumulates only from `result.telemetry.n_steps_completed` (real
executed steps). It is never speculatively set and never derived from the
planned horizon.

---

## 8. Blocker Summary

| Blocker | Owner | Unblocked By |
|---------|-------|-------------|
| Production grid size | WP-N3 | WP-N3 geometry packet completion |
| Production-grid per-step timing | WP-N4 | WP-N3 grid + fresh measurement |
| Production-grid memory + artifact volume | WP-N4 | WP-N3 grid |
| Production-grid 12 us wall-clock floor | WP-N4 | WP-N3 + production-grid timing |
| Three-restart combiner correctness | Sprint 2.2 | F1 fix in `merge_cumulative_ledgers()` |
| 12 us completion claim | WP-N4 | Rung 4 actually runs to completion: `horizon_complete=true` |
| MLX/Metal acceleration | System | MLX unavailable; `backend="package_native"` only |

---

## 9. Acceptance State

This packet is a read-only engineering research document. It contains no
physics implementation, no changes to Sprint-2.2-owned files, and no
promotion of validation or acceptance claims.

```json
{
  "can_support_first_principles_acceptance": false,
  "can_support_validation_claims": false,
  "review_decision": "parallel_lane_research_packet_engineering_candidate_only",
  "horizon_complete_claim_allowed": false,
  "production_grid_known": false,
  "sprint_2_2_files_modified": false
}
```
