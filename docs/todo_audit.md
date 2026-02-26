# DPF-Unified TODO / Placeholder Audit

Generated: 2026-02-25
Scope: `src/dpf/**/*.py`, `tests/**/*.py`

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL (broken physics / silently wrong) | 3 |
| MEDIUM (missing feature, incomplete) | 9 |
| LOW (cleanup / minor) | 5 |
| BENIGN (correct except-pass patterns) | 5 |
| EXPERIMENTAL (by design, not bugs) | 4 |

---

## CRITICAL

### C1 — Back-EMF hardcoded to zero
**File:** `src/dpf/engine.py:643-644`
**Context:**
```python
# TODO: Compute Back-EMF from field motion (-v x B)
back_emf = 0.0
```
**Function:** `SimulationEngine._apply_source_terms()` (circuit step)
**Impact:** The circuit solver never receives any motional EMF from MHD field motion. The `-v×B` term couples fluid velocity to circuit voltage and is physically essential for current quench dynamics. Currently the circuit sees only the plasma inductance change (dL/dt), not the field-advection EMF. Missing for ALL backends.
**Fix:** Compute volume-averaged `-v×B` from MHD state and pass to `circuit.step()`.

---

### C2 — Radial zipper BC only implemented for Python backend
**File:** `src/dpf/engine.py:1089-1092`
**Context:**
```python
elif self.backend == "athena":
    pass
else:
    pass
```
**Function:** `SimulationEngine._apply_radial_zipper_bc()`
**Impact:** The zipper boundary condition (zeros B_theta outside r_shock during radial phase, enforcing thin-sheath topology) is silently skipped for Athena++ and Metal backends. Cross-backend physics diverge at radial phase entry.
**Fix:** Implement zipper BC for Athena++ (via EnrollUserBoundaryFunction) and Metal (Metal tensor slice zeroing).

---

### C3 — WALRUS normalization stats not computed (returns empty dict)
**File:** `src/dpf/ai/well_loader.py:170-185`
**Context:**
```python
pass  # Doing proper pass requires refactoring slightly.
...
return {} # Placeholder
```
**Function:** `WellDataLoader._compute_normalization_stats()`
**Impact:** WALRUS RevIN normalization requires per-field statistics (mean/std or RMS). This method returns `{}` unconditionally. Downstream callers that use these stats will apply zero normalization or fall back to identity, silently degrading inference quality.
**Fix:** Implement HDF5 attribute read (many Well datasets expose `mean`/`std` attrs) or compute from sampled trajectories.

---

## MEDIUM

### M1 — J_kin shape mismatch silently ignored
**File:** `src/dpf/fluid/mhd_solver.py:1541-1542`
**Context:**
```python
else:
    pass # TODO: Add shape check/warning
```
**Function:** `MHDSolver._compute_mhd_rhs()` (kinetic current subtraction)
**Impact:** If `J_kin` has the wrong shape, the kinetic current is silently dropped and the MHD-PIC coupling fails without any diagnostic. Should at least emit a `logger.warning`.
**Fix:** Add `logger.warning("J_kin shape mismatch: %s vs %s; skipping kinetic subtraction", J_kin.shape, J.shape)`.

---

### M2 — Default checkpoint auto-discovery not implemented
**File:** `src/dpf/ai/surrogate.py:104-106`
**Context:**
```python
def _find_default_checkpoint(self) -> Path | None:
    """Attempt to locate a default checkpoint."""
    return None  # Placeholder logic for now, or scan directories
```
**Impact:** `DPFSurrogate()` without an explicit `checkpoint_path` always falls back to placeholder prediction. Users who install the pretrained checkpoint at `models/walrus-pretrained/walrus.pt` still get placeholder behavior unless they pass the path explicitly.
**Fix:** Scan `[project_root]/models/walrus-pretrained/walrus.pt`, `~/.dpf/walrus.pt`, and `WALRUS_CHECKPOINT` env var.

---

### M3 — Well exporter has dead `pass` block before re-implementation
**File:** `src/dpf/io/well_exporter.py:131-134`
**Context:**
```python
# Optimize: direct assignment if dpf_scalar_to_well returns full array
# But dpf_scalar_to_well returns a full (1,1,...) array.
# Let's perform a simpler loop here to avoid alloc turnover.
pass
```
**Function:** `WellExporter._write_scalar_fields()` (inner loop)
**Impact:** Dead code block. The `pass` is followed by a working `np.stack` re-implementation. Confusing and suggests abandoned optimization attempt. No functional bug, but code quality issue.
**Fix:** Remove the dead comment block and `pass`.

---

### M4 — Kinetic manager uses placeholder dt and "ions" comment
**File:** `src/dpf/kinetic/manager.py:34,42-44`
**Context:**
```python
dt=1e-9,  # placeholder, will be overridden in step
...
# Let's create a placeholder "ions" species
```
**Function:** `KineticManager.__init__()`
**Impact:** The `HybridPIC` driver is initialized with `dt=1e-9` which is overridden later, but the ion species initialization comment indicates incomplete design. The "beam" species is created but no thermal background species is initialized, which may cause unphysical behavior for non-beam runs.
**Fix:** Remove placeholder dt entirely (pass dt in `step()` only) and clarify the species initialization design.

---

### M5 — Orszag-Tang periodic BCs not supported
**File:** `src/dpf/verification/orszag_tang.py:290-293`
**Context:**
```python
bc_note = (
    "Zero-gradient (Neumann) BCs used; periodic BCs not yet supported. "
    "Boundary effects may degrade accuracy at late times."
)
```
**Function:** `run_orszag_tang()`
**Impact:** The Orszag-Tang vortex is a canonical *periodic* MHD test. Using Neumann BCs reduces accuracy at late times and prevents comparison with published periodic-BC benchmarks.
**Fix:** Add periodic BC option to the MHD solver (wrap ghost cells from opposite boundary).

---

### M6 — Multi-species WENO5 advection deferred to future phase
**File:** `src/dpf/experimental/species.py:264-265`
**Context:**
```python
Higher-order reconstruction (WENO5) will be added in a future phase.
```
**Function:** `SpeciesMixture.advect_species()` docstring
**Impact:** Only first-order upwind advection implemented for multi-species transport. Module is in `experimental/` so this is expected, but the gap is documented.
**Severity:** LOW within the experimental context.

---

### M7 — Beam-target neutron yield requires particle tracking (deferred)
**File:** `src/dpf/diagnostics/neutron_yield.py:14-15`
**Context:**
```python
This module computes the thermonuclear component; beam-target is a
correction that requires particle tracking (future Phase).
```
**Impact:** For most DPF devices, the beam-target mechanism dominates (10-100× over thermonuclear). The missing beam-target component means neutron yield estimates are likely 1-2 orders of magnitude low for realistic DPF conditions. Note: beam-target diagnostic was added in Phase Y (`diagnostics/beam_target.py`), so this comment in `neutron_yield.py` is now partially outdated — the beam-target module exists but is separate, not integrated into `neutron_yield.py`.
**Fix:** Update docstring to reference `diagnostics/beam_target.py` for beam-target yields.

---

### M8 — RHO_FLOOR source ambiguous in Metal cylindrical solver
**File:** `src/dpf/metal/metal_solver.py:528`
**Context:**
```python
rho_safe = torch.clamp(rho, min=1e-12) # RHO_FLOOR hardcoded or import?
```
**Function:** `MetalMHDSolver._cylindrical_source_terms()`
**Impact:** The floor value `1e-12` is hardcoded inline rather than imported from a central `RHO_FLOOR` constant. Minor inconsistency risk if the floor is changed elsewhere.
**Fix:** Import `RHO_FLOOR` from `metal_solver.py` module-level constant (already defined at line ~35 as `P_FLOOR = 1e-20`) or create a shared constants module.

---

### M9 — Instability detector uses loop index as placeholder time
**File:** `src/dpf/ai/instability_detector.py:200-201`
**Context:**
```python
# Use index as step number, placeholder time
event = self.check(history, actual_next, step=i, time=float(i))
```
**Function:** `InstabilityDetector.monitor_trajectory()`
**Impact:** Events are tagged with `time=float(i)` (integer step index) rather than actual simulation time in seconds. Post-processing code that sorts or filters events by time will get meaningless timestamps.
**Fix:** Accept an optional `times` list parameter and pass `time=times[i] if times else float(i)`.

---

## LOW

### L1 — Incomplete test: Whistler wave parity test missing
**File:** `tests/test_phase_g_parity.py:65`
**Context:**
```python
# TODO: Implement a specific test case (e.g. Whistler wave)
```
**Impact:** Cross-backend Hall MHD parity test has a placeholder comment instead of an actual test case. Reduces coverage of Hall MHD correctness verification.
**Fix:** Implement Whistler wave dispersion test (ω = k²c²/ωci) comparing Python and Metal backends.

---

### L2 — `_find_default_checkpoint` (surrogate.py) cannot scan for checkpoints
*(Listed above as M2, repeated here for completeness at LOW since fallback is graceful.)*

---

## BENIGN (Correct Pattern — No Action Needed)

These `pass` statements are idiomatic Python in exception handlers or TYPE_CHECKING guards:

| File | Line | Context | Why Benign |
|------|------|---------|-----------|
| `src/dpf/benchmarks/metal_benchmark.py` | 38, 45 | `except ImportError: pass` | Optional dependency |
| `src/dpf/server/app.py` | 263, 289 | `except WebSocketDisconnect: pass` | Expected disconnect |
| `src/dpf/metal/metal_transport.py` | 23 | `if TYPE_CHECKING: pass` | Type-checking guard |
| `src/dpf/metal/device.py` | 193 | `except ValueError: pass` | Graceful sysctl failure |

---

## EXPERIMENTAL (By Design — Unintegrated Modules)

The `src/dpf/experimental/` package contains complete-but-unintegrated modules. These are not bugs; they are explicitly documented as building blocks for future development.

| Module | LOC | Status |
|--------|-----|--------|
| `experimental/amr/` | ~756 | Block-structured AMR, gradient tagging — needs solver refactoring |
| `experimental/pic/` | ~979 | Boris pusher + CIC deposition — never instantiated in engine |
| `experimental/species.py` | ~410 | SpeciesMixture class — not wired into engine.py |
| `experimental/gpu_backend.py` | ~119 | CuPy detection only, no actual kernels (irrelevant on M3 Pro) |

---

## Priority Action List

1. **[C1]** Implement Back-EMF coupling in `engine.py` — affects circuit accuracy for all backends
2. **[C3]** Implement `WellDataLoader._compute_normalization_stats()` — affects WALRUS inference quality
3. **[C2]** Implement zipper BC for Athena++ and Metal backends — cross-backend physics parity
4. **[M2]** Implement `_find_default_checkpoint()` scan — improves surrogate.py UX
5. **[M1]** Add shape-mismatch warning in `mhd_solver.py` — debug quality
6. **[M5]** Add periodic BCs for Orszag-Tang verification — accuracy testing
7. **[M7]** Update `neutron_yield.py` docstring to reference `beam_target.py`
8. **[L1]** Implement Whistler wave parity test in `test_phase_g_parity.py`
9. **[M3]** Remove dead `pass` block in `well_exporter.py`
