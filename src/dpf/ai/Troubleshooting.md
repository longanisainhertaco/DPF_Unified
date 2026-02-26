# AI Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 14 files (~4,200 LOC)

---

## CRITICAL

### CRIT-1: Incorrect WALRUS Import in preconditioner.py ✅ FIXED
- **File:Line**: `preconditioner.py:20`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `from walrus.models import WalrusModel` imports a non-existent class. The correct class is `IsotropicModel`.
- **Fix applied**: Changed import to `from walrus.models import IsotropicModel` and updated all references from `WalrusModel` to `IsotropicModel`.

### CRIT-2: Code Duplication — kinetic/hybrid.py is near-identical copy of experimental/pic/hybrid.py
- **File:Line**: `kinetic/hybrid.py` (978 LOC) vs `experimental/pic/hybrid.py` (988 LOC)
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (verified via diff)
- **Description**: Files differ only in cosmetic formatting and RNG pattern divergence:
  - `experimental/pic/hybrid.py:903`: `rng = np.random.default_rng()` (modern, reproducible)
  - `kinetic/hybrid.py:897`: `np.random.random()` (legacy global RNG, not reproducible)
- **Proposed fix**: Delete `kinetic/hybrid.py`, have `kinetic/__init__.py` re-export from `experimental.pic.hybrid`. Standardize on `np.random.default_rng()`.
- **Impact**: ~978 LOC of pure duplication. Bug fixes applied to one copy may not reach the other.
- **Status**: Deferred — outside AI module scope. Requires coordination with kinetic module owner.

### CRIT-3: Duplicate WellExporter Implementations — Both Used in Production
- **File:Line**: `ai/well_exporter.py` (273 LOC) vs `io/well_exporter.py` (178 LOC)
- **Found by**: py-ai-diag
- **Cross-review verdict**: MODIFIED (more severe than originally assessed)
- **Description**: Two completely separate `WellExporter` classes with different APIs:
  - `ai/well_exporter.py`: `add_snapshot(state, time, scalars)` — used by CLI `export-well`
  - `io/well_exporter.py`: `append_state(state)` — used by `engine.py:214` (production engine)
- **Status**: Deferred — consolidation requires coordinated refactor across engine.py and CLI.

---

## HIGH

### HIGH-1: Dead Code in surrogate.py — Two Unused Methods (~170 LOC) ✅ FIXED
- **File:Line**: `surrogate.py:625-674` (`_states_to_walrus_tensor`) and `surrogate.py:753-795` (`_tensor_to_state`)
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (grep finds no callers)
- **Fix applied**: Removed both dead methods. Active replacements are `_build_walrus_batch()` and `_well_output_to_state()`. Updated tests in `test_phase_j2_walrus_integration.py` accordingly.

### HIGH-2: Bug M9 — instability_detector.py Uses Loop Index as Time ✅ FIXED
- **File:Line**: `instability_detector.py:201`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (known bug M9 from memory)
- **Fix applied**: Added `dt: float = 1.0` parameter to `monitor_trajectory()`. Time now computed as `time=float(i * dt)` instead of `time=float(i)`. Default `dt=1.0` preserves backward compatibility.

### HIGH-3: torch.load Without weights_only in preconditioner.py ✅ FIXED
- **File:Line**: `preconditioner.py:66`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix applied**: Added `weights_only=True` to `torch.load()` call. SimpleUNet3D only needs state_dict, so no pickle deserialization is needed.

### HIGH-4: Hardcoded Relative Path in preconditioner.py ✅ FIXED
- **File:Line**: `preconditioner.py:61`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix applied**: Changed `Path("src/dpf/ai/poisson_net.pt")` to `Path(__file__).parent / "poisson_net.pt"`. Removed stale `custom_model_path` variable and comments.

### HIGH-5: _lists_to_arrays Blindly Converts Lists to NumPy Arrays ✅ FIXED
- **File:Line**: `realtime_server.py:99-117`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix applied**: Added dtype check after `np.array()` conversion: rejects arrays with `dtype.kind not in ("f", "i", "u", "b")` by raising ValueError, which is caught and treated as non-numeric.

### HIGH-6: Shallow Copy in ai/well_exporter.py ✅ FIXED
- **File:Line**: `ai/well_exporter.py:85`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix applied**: Changed `state.copy()` (shallow dict copy) to `{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}` (deep copy of NumPy arrays).

### HIGH-7: Dead Loop in io/well_exporter.py ✅ FIXED
- **File:Line**: `io/well_exporter.py:125-132`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix applied**: Dead loop with `dpf_scalar_to_well` call and `pass` was already removed (cleaned up in prior session). Unused imports also removed.

---

## MEDIUM

### MOD-1: Bug C3 — well_loader.py compute_stats() Can Return Empty Dict
- **File:Line**: `well_loader.py:162-248`
- **Found by**: py-ai-diag (analysis), known bug C3 from memory
- **Cross-review verdict**: CONFIRMED with nuance
- **Description**: `compute_stats()` returns `{}` when: no HDF5 paths exist, files have no density dataset, or `len(self) == 0`. Strategy 2 (online computation) handles the empty case by returning whatever partial stats exist (line 200-202 short-circuits).
- **Assessment**: Not a logic bug per se — it's graceful degradation for missing data. But callers (WALRUS RevIN) don't check for empty stats, causing silently degraded inference.
- **Proposed fix**: Log a warning when returning empty stats and document the requirement for callers to check.

### MOD-2: confidence.py Scale-Dependent Confidence
- **File:Line**: `confidence.py`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `_compute_confidence` uses absolute std, making confidence scale-dependent.
- **Impact**: Low — affects AI module only, not physics.

### MOD-3: dataset_validator.py Loads Full Dataset Into Memory
- **File:Line**: `dataset_validator.py:176-177`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Impact**: Will OOM on large datasets (>10GB). Use chunked iteration.

### MOD-4: Dead _validate_step in hybrid_engine.py ✅ FIXED
- **File:Line**: `hybrid_engine.py:229-263`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (grep finds only the definition, no callers)
- **Fix applied**: Removed dead `_validate_step` method (~35 LOC). Updated `test_phase_h_hybrid.py` to remove stale tests referencing removed `max_div_b_error` parameter and `_compute_div_b_error` method.

---

## REJECTED FINDINGS

### REJECTED: HIGH-4 (torch.load in surrogate.py) — Partially Rejected
- **Reason**: The Python reviewer listed `surrogate.py` as having `torch.load()` without `weights_only=True`. However, `surrogate.py:183` explicitly uses `weights_only=False`, which is intentional — WALRUS checkpoints contain embedded config objects that require pickle deserialization. The reviewer should have distinguished between the intentional case (surrogate.py) and the oversight case (preconditioner.py).

### REJECTED: "io/well_exporter.py never used by any production code path"
- **Reason**: `engine.py:214` imports `from dpf.io.well_exporter import WellExporter`. This is a production code path. The finding was upgraded to CRIT-3 (MODIFIED) to reflect the true severity.
