# Sprint 4 Gap Analysis: MLX Solver vs DoD Acceptance Criteria

**Date**: 2026-03-24
**Author**: dpf-engine-architect (Cortana)
**Status**: Pre-Sprint 4 readiness assessment
**Scope**: Gap analysis for METAL_V2_DOD.md M1-M8, S1-S9 against current MLX solver state

---

## 1. Current State Summary

### 1.1 Files Implemented (Sprints 0-3)

| File | Status | LOC (approx) |
|------|--------|-------------|
| `mlx_device.py` | Complete | 120 |
| `mlx_grid.py` | Complete | 100 |
| `mlx_state.py` | Complete | 150 |
| `mlx_kernels.py` | Complete | 400 |
| `mlx_primitives.py` | Complete | 180 |
| `mlx_reconstruction.py` | Complete | 250 |
| `mlx_riemann.py` | Complete | 200 |
| `mlx_sources.py` | Complete | 150 |
| `mlx_ct.py` | Complete | 120 |
| `mlx_transport.py` | Complete | 150 |
| `mlx_timestepper.py` | Complete | 628 |
| `mlx_solver.py` | Complete | 541 |
| `mlx_circuit.py` | **MISSING** | 0 |
| `mlx_benchmark.py` | **MISSING** | 0 |

### 1.2 Test Files Implemented

| File | Status |
|------|--------|
| `test_mlx_device.py` | Complete |
| `test_mlx_grid.py` | Complete |
| `test_mlx_state.py` | Complete |
| `test_mlx_kernels.py` | Complete |
| `test_mlx_primitives.py` | Complete |
| `test_mlx_reconstruction.py` | Complete |
| `test_mlx_riemann.py` | Complete |
| `test_mlx_sources.py` | Complete |
| `test_mlx_ct.py` | Complete |
| `test_mlx_transport.py` | Complete |
| `test_mlx_timestepper.py` | Complete |
| `test_mlx_solver.py` | Complete (15 tests) |
| `test_mlx_cross_backend.py` | Complete (14 tests) |
| `test_mlx_engine_integration.py` | Complete |
| `test_mlx_pf1000.py` | **MISSING** |
| `test_mlx_multidevice.py` | **MISSING** |
| `test_mlx_acceptance.py` | **MISSING** |

---

## 2. Must-Have Criteria Gap Table (M1-M8)

| ID | Criterion | Status | Gap | Fix Required | LOC | Risk |
|----|-----------|--------|-----|-------------|-----|------|
| M1 | No negative pressure (entropy formulation) | PARTIAL | Dual-energy entropy tracer IS implemented (`_resync_energy` in timestepper, `recover_pressure_dual_energy` in primitives). Pressure floors enforce `P_FLOOR`. **Not tested at beta=7e-7 electrode conditions.** | Write `test_v2_entropy_positivity` with electrode-condition state (rho=1e-3, p=160, B_theta=24T). Verify `p > 0` after 100 steps. | 30 | LOW -- mechanism exists, just needs validation |
| M2 | PF-1000 I_peak within 10% of 1.2 MA | NOT TESTED | No PF-1000 full discharge test exists for MLX backend. `coupling_interface()` returns only current/voltage passthrough (no R_plasma, no Lp). Circuit coupling is done by engine.py via `self.fluid.geom` which is **not attached** to MLX solver. | 1. Attach `.geom` in engine.py MLX block (5 LOC). 2. Write `test_mlx_pf1000.py` with full discharge. 3. Fix `coupling_interface()` to compute real Lp/R_plasma or rely on engine-level coupler. | 350 | HIGH -- engine integration gap blocks full discharge |
| M3 | Mass conservation < 5% | NOT TESTED | No mass conservation test for MLX. The conservative formulation should conserve mass, but outflow BCs and cylindrical geometry may leak. | Write mass tracking test in `test_mlx_pf1000.py`. | 30 | LOW -- conservative scheme should satisfy this |
| M4 | Energy conservation < 10% | PARTIAL | Cross-backend test checks < 5% drift over 20 steps on normalised uniform state. **Not tested on PF-1000 discharge with circuit input and radiation.** Ohmic heating in entropy tracer source is implemented in `_do_resistive_diffusion` (updates `ISR` alongside `IEN`). | Write energy budget test: `|dE_total/dt - P_circuit + P_rad| / P_circuit < 0.10` over full discharge. | 50 | MEDIUM -- entropy source term wiring needs verification at DPF conditions |
| M5 | No NaN propagation | PARTIAL | Tested for 3 steps (uniform), 50 steps (Sod), 20 steps (Brio-Wu). **Not tested over full PF-1000 discharge (~10,000+ steps).** HLL fallback with NaN guard exists in `_hll_flux`. | Full discharge NaN check in `test_mlx_pf1000.py`. | 10 | MEDIUM -- long integration may expose edge cases |
| M6 | Completes 5 phases (t > 2 * t_peak) | NOT TESTED | No full discharge test. Engine integration incomplete (missing `.geom`). | Fix engine integration + write full discharge test. | 30 | HIGH -- blocked by M2 gap |
| M7 | Float32 on Metal GPU | SATISFIED | MLX solver uses float32 throughout (`mx.float32` in all arrays). `is_available()` checks for GPU device. HLL fallback uses float64 NumPy bridge for safety -- acceptable. | None (verify in acceptance test). | 0 | LOW |
| M8 | div(B) = 0 to relative 1e-6 | PARTIAL | `mlx_ct.py` exists. CT implemented. **No test measures `max(|div(B)|) * dx / max(|B|)` quantitatively.** | Write div(B) measurement test. Need to verify CT is actually called during `ssp_rk3_step` -- currently, geometric sources are computed but CT update is NOT visible in `mhd_rhs`. | 80 | MEDIUM -- CT may not be wired into the RHS |

### M-Criteria Summary

- **SATISFIED**: M7 (1/8)
- **PARTIAL**: M1, M4, M5, M8 (4/8)
- **NOT TESTED**: M2, M3, M6 (3/8)

---

## 3. Should-Have Criteria Gap Table (S1-S9)

| ID | Criterion | Status | Gap | Fix Required | LOC | Risk |
|----|-----------|--------|-----|-------------|-----|------|
| S1 | I(t) waveform NRMSE < 0.25 | NOT TESTED | Requires full PF-1000 discharge + waveform comparison. Blocked by M2/M6. | Include NRMSE computation in `test_mlx_pf1000.py`. | 40 | HIGH -- depends on M2 fix |
| S2 | Current dip 30-70% at pinch | NOT TESTED | Same blocker as S1. | Include dip measurement in PF-1000 test. | 20 | HIGH |
| S3 | Pinch voltage > 20 kV | NOT TESTED | Requires circuit coupling feedback. | Test voltage spike in PF-1000 test. | 10 | HIGH |
| S4 | Multi-device (3+ devices) | NOT TESTED | No UNU-ICTP or NX2 validation tests for MLX. Presets exist in `presets.py`. | Write `test_mlx_multidevice.py` using presets. | 150 | MEDIUM |
| S5 | Cross-backend Sod L1(rho) < 15% | SATISFIED | `test_mlx_cross_backend.py::TestSodShockParity::test_sod_density_l1_parity` tests this at < 15%. | None. | 0 | LOW |
| S6 | Brio-Wu compound wave structure | SATISFIED | `test_mlx_solver.py::test_brio_wu_no_nan` verifies no NaN + B_theta structure. | None (could strengthen with wave-counting). | 0 | LOW |
| S7 | Sod L1(rho) < 1e-2 at N=256 | PARTIAL | Current test uses N=16x16 with < 15% threshold. **Need N=256 test with < 1e-2 threshold.** | Add high-resolution Sod test. | 40 | LOW |
| S8 | Diffusion convergence >= 1.9 | NOT TESTED | `test_mlx_transport.py` tests diffusion changes B-field, but no multi-resolution convergence study. | Write 4-resolution (32, 64, 128, 256) convergence test. | 80 | MEDIUM |
| S9 | Faster than Athena++ at 128x512 | NOT TESTED | No benchmark script exists. `mlx_benchmark.py` is missing. | Write benchmark with grid scaling. | 200 | MEDIUM -- MLX vs C++ depends on kernel efficiency |

### S-Criteria Summary

- **SATISFIED**: S5, S6 (2/9)
- **PARTIAL**: S7 (1/9)
- **NOT TESTED**: S1, S2, S3, S4, S8, S9 (6/9)

---

## 4. Critical Bugs and Missing Infrastructure

### 4.1 Missing `.geom` Attribute on MLX Solver (BLOCKER)

**Location**: `engine.py` lines 199-218

The Metal (`backend="metal"`) block attaches `.geom = CylindricalGeometry(...)` at line 190-192.
The MLX (`backend="mlx"`) block does NOT. This means:

- `self.fluid.geom.cell_volumes()` -- called 15+ times in engine.py -- will raise `AttributeError`
- `self.fluid.geom.curl()` -- used for J-field computation -- will fail
- `self.fluid.geom.r` -- used for radial grid coordinates -- will fail
- Snowplow mass deposition, radiation energy tracking, neutron yield -- all broken

**Fix**: Add 4 lines after line 216 in engine.py:

```python
if self.geometry_type == "cylindrical":
    from dpf.geometry.cylindrical import CylindricalGeometry
    self.fluid.geom = CylindricalGeometry(nr=nx, nz=nz, dr=dx, dz=dz)
```

Also fix `_cell_volume` to use cylindrical volumes instead of `dx * dx * dz`.

**LOC**: 5
**Risk**: LOW (direct copy from Metal block)

### 4.2 `coupling_interface()` Returns Stub Data

**Location**: `mlx_solver.py` lines 511-519

`coupling_interface()` returns a `CouplingState` with only `current` and `voltage` populated.
`CouplingState` has fields: `Lp`, `emf`, `current`, `voltage`, `dL_dt`, `R_plasma`, `Z_bar`.
All physics fields (`Lp`, `emf`, `dL_dt`, `R_plasma`) are left at 0.0.

**Impact**: The engine-level `CircuitCoupler` handles Lp/R_plasma extraction from the state dict
independently (via `coupler.py`), so this stub may be acceptable IF the engine coupler works
with MLX-produced state dicts. However, any code path that reads `solver.coupling_interface().Lp`
directly will get zero.

**Decision**: LOW priority for Sprint 4. The engine coupler computes Lp/R_plasma from the state
dict, bypassing `coupling_interface()`. Verify this path works with MLX state dicts.

**LOC**: 0 (defer)
**Risk**: LOW

### 4.3 `mx.compile()` Not Applied (WU-4.4)

**Location**: `mlx_reconstruction.py` mentions `mx.compile()` in docstring but no actual
`@mx.compile` decorator or `mx.compile()` wrapping anywhere in the codebase.

**Impact**: Performance only. The solver is correct without compilation. `mx.compile()` fuses
elementwise chains (pressure recovery, floor clamping) into single Metal kernels.

**Fix**: Wrap `_resync_energy`, `_apply_floors`, `_clamp_velocity`, and `_geometric_sources`
with `mx.compile()`. Test for correctness regression.

**LOC**: 30
**Risk**: LOW (performance optimization, no physics change)

### 4.4 Missing `mlx_circuit.py` (WU-3.2)

**Planned**: Sprint 3 deliverable, 100 LOC circuit coupling adapter.

**Current state**: Does not exist. The engine's `CircuitCoupler` handles circuit integration
at the engine level, extracting Lp/R_plasma from the state dict. The MLX solver does not need
its own circuit module IF the engine coupler works with MLX state dicts.

**Decision**: DEFER. Verify engine coupler compatibility in integration tests.

**LOC**: 0 (defer)
**Risk**: LOW

### 4.5 Missing `mlx_benchmark.py` (WU-4.3)

**Planned**: Sprint 4 deliverable, 200 LOC benchmark script.

**Fix**: Create `src/dpf/benchmarks/mlx_benchmark.py` with grid scaling tests at
64x256, 128x512, 256x1024. Measure wall-clock per step. Compare vs Athena++.

**LOC**: 200
**Risk**: LOW

### 4.6 CT Not Wired Into `mhd_rhs`

**Location**: `mlx_timestepper.py::mhd_rhs()` (lines 59-162)

The `mhd_rhs` function computes radial flux divergence, axial flux divergence, and geometric
sources. It does NOT call any CT function. The B-field is updated through the Riemann solver
fluxes (which preserves div(B) for 1D sweeps) but there is no explicit CT correction step.

The `mlx_ct.py` module exists but is never imported or called from `mhd_rhs` or `ssp_rk3_step`.

**Impact**: div(B) may not be maintained to machine precision over long integrations.
For axisymmetric (r,z) geometry, the Riemann solver approach may be sufficient because
B_theta is cell-centered and does not participate in CT. But the spec requires `max(|div(B)|) * dx / max(|B|) < 1e-6`.

**Fix**: Wire `mlx_ct.py` into `ssp_rk3_step` as a post-stage correction, OR verify that
the current approach satisfies M8 empirically.

**LOC**: 20-40
**Risk**: MEDIUM -- may require restructuring the B-field update

---

## 5. Pre-Sprint-4 Action Items (Must Fix BEFORE Sprint 4 Starts)

These are blockers that prevent Sprint 4 work units from executing.

| # | Action | File | LOC | Priority | Blocks |
|---|--------|------|-----|----------|--------|
| P1 | Attach `.geom` to MLX solver in engine.py | `engine.py` | 5 | CRITICAL | M2, M3, M4, M5, M6, S1-S3 |
| P2 | Fix `_cell_volume` for cylindrical MLX | `engine.py` | 3 | CRITICAL | M3, M4 |
| P3 | Verify engine coupler works with MLX state dicts | `tests/test_mlx_engine_integration.py` | 30 | HIGH | M2, S1 |
| P4 | Wire CT into timestepper OR measure div(B) | `mlx_timestepper.py` or test | 40 | HIGH | M8 |

**Total pre-sprint effort**: ~78 LOC, approximately 2-3 hours.

---

## 6. Sprint 4 Work Unit Breakdown

### Week 1 (Days 1-3): Engine Integration + PF-1000 Discharge

| Day | Work Unit | Deliverable | DoD Criteria |
|-----|-----------|-------------|-------------|
| 1 | WU-4.0 (pre-sprint) | Fix P1-P4 from section 5. Run existing tests. | Unblocks all |
| 1 | WU-4.1a | `test_mlx_pf1000.py` scaffold: PF-1000 preset, 64x256 grid, 12 us | M2, M5, M6 |
| 2 | WU-4.1b | PF-1000 full discharge runs to completion (may need debug) | M5, M6 |
| 2 | WU-4.1c | Add M1-M8 assertions to PF-1000 test | M1-M8 |
| 3 | WU-4.1d | Debug any M-criteria failures. Priority: M1 (pressure), M5 (NaN) | M1, M5 |

### Week 2 (Days 4-6): Multi-Device + Convergence

| Day | Work Unit | Deliverable | DoD Criteria |
|-----|-----------|-------------|-------------|
| 4 | WU-4.2a | `test_mlx_multidevice.py`: UNU-ICTP (tutorial preset) | S4 |
| 4 | WU-4.2b | NX2 preset validation | S4 |
| 5 | WU-4.2c | Sod at N=256, L1 < 1e-2 | S7 |
| 5 | WU-4.2d | Diffusion convergence study (4 resolutions) | S8 |
| 6 | WU-4.1e | S1-S3 assertions (waveform NRMSE, current dip, voltage spike) | S1, S2, S3 |

### Week 3 (Days 7-9): Performance + Acceptance

| Day | Work Unit | Deliverable | DoD Criteria |
|-----|-----------|-------------|-------------|
| 7 | WU-4.3 | `src/dpf/benchmarks/mlx_benchmark.py`: grid scaling | S9 |
| 7 | WU-4.4 | `mx.compile()` wrapping on hot paths | S9 |
| 8 | WU-4.5a | `test_mlx_acceptance.py`: all M1-M8 + S1-S9 in one suite | ALL |
| 9 | WU-4.5b | Fix any remaining failures. Final pass. | ALL |

---

## 7. Test Plan Matrix

| Test File | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|-----------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|
| `test_mlx_solver.py` (existing) | . | . | . | . | x | . | x | . | . | . | . | . | . | x | . | . | . |
| `test_mlx_cross_backend.py` (existing) | . | . | . | x | x | . | . | . | . | . | . | . | x | . | . | . | . |
| `test_mlx_pf1000.py` (NEW) | x | x | x | x | x | x | x | x | x | x | x | . | . | . | . | . | . |
| `test_mlx_multidevice.py` (NEW) | x | . | . | . | x | x | . | . | . | . | . | x | . | . | . | . | . |
| `test_mlx_acceptance.py` (NEW) | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `test_mlx_convergence.py` (NEW) | . | . | . | . | . | . | . | . | . | . | . | . | . | . | x | x | . |
| `mlx_benchmark.py` (NEW, bench) | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | x |

Legend: `x` = primary verification, `.` = not tested by this file

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PF-1000 discharge crashes mid-run (NaN at pinch) | HIGH | HIGH | Velocity clamping + dual-energy already implemented. Debug with reduced grid (32x128) first. |
| I_peak > 10% off experimental | MEDIUM | HIGH | Engine coupler Lp extraction may need tuning. Compare against Python cylindrical backend first. |
| CT insufficient -- div(B) grows over 10k steps | MEDIUM | MEDIUM | Wire `mlx_ct.py` post-stage correction. Measure empirically before adding complexity. |
| `mx.compile()` causes correctness regression | LOW | MEDIUM | Test with and without compile on Sod/Brio-Wu. Only apply to pure functions. |
| Performance does not beat Athena++ | MEDIUM | LOW | S9 is should-have, not must-have. Document MLX value as zero-copy Python integration. |
| Multi-device presets expose solver tuning | MEDIUM | MEDIUM | UNU-ICTP is small (15 kV, 30 uF) -- gentler conditions. Test this first. |

---

## 9. Estimated Total Sprint 4 Effort

| Category | LOC | Hours |
|----------|-----|-------|
| Pre-sprint fixes (P1-P4) | 78 | 3 |
| `test_mlx_pf1000.py` | 300 | 8 |
| `test_mlx_multidevice.py` | 150 | 4 |
| `test_mlx_convergence.py` | 120 | 3 |
| `test_mlx_acceptance.py` | 200 | 5 |
| `mlx_benchmark.py` | 200 | 4 |
| `mx.compile()` wrapping | 30 | 2 |
| Debug / iteration | 100 | 12 |
| **Total** | **~1,178** | **~41** |

Sprint 4 is approximately 2 weeks of focused work, assuming pre-sprint blockers are resolved
on day 1. The highest-risk item is the PF-1000 full discharge (WU-4.1), which may surface
NaN stability issues at the pinch phase requiring debug cycles.
