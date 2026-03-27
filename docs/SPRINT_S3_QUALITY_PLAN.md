# Sprint S-3: Quality Hardening Plan

**Date**: 2026-03-27
**Objective**: Fix verified defects, establish process gates, build V&V foundation
**Principle**: No new physics features until process reaches 3.0 sigma

**Source**: Panel audit by Dr. Vasquez (physics), Marcus Chen (code), Sandra Nakamura (Six Sigma)
**All findings verified against source code before inclusion.**

---

## Phase 1: Process Gates (Sessions 1-2)

### 1.1 Automated Pre-Commit Verification Hook

**Problem**: 50% rework rate (6/12 commits were fixes). Root cause: no automated quality gate.

**Deliverable**: Git pre-commit hook at `.git/hooks/pre-commit` that BLOCKS commits failing any check.

**Checks** (verified necessary from actual bugs):
1. **Attribute validation**: `grep 'self\._[a-z]' changed_files` → verify attribute exists in class
2. **Import validation**: `python3 -c "from module import func"` for every new import in diff
3. **Analytical stencil test**: If diff adds `def.*_rhs` or `def.*laplacian`, require matching `test_analytical_*`
4. **Test fixture audit**: If diff modifies `test_*.py`, log which backend/grid/preset the test uses
5. **Module pytest**: `pytest changed_test_files -x -q` must pass

**Implementation**:
- File: `scripts/pre_commit_check.sh` (exists, needs expansion)
- Wire: `ln -s ../../scripts/pre_commit_check.sh .git/hooks/pre-commit`
- Test: Intentionally introduce each bug type, verify hook blocks it

**Acceptance criteria**: Hook blocks a commit with self._nr, blocks a commit with missing import, passes clean commits in <5 seconds.

---

### 1.2 Physics Constants Single Source of Truth

**Problem**: c_boris hardcoded in 65 locations across 5 files. P_FLOOR in 5 files. RHO_FLOOR in 4 files.

**Deliverable**: `src/dpf/metal/constants.py` with ALL physics constants. Every Python file imports from it.

**Constants to centralize** (verified duplicated):
| Constant | Current locations | Value |
|----------|------------------|-------|
| C_BORIS / C_BORIS_SQ | mlx_riemann.py (4x), mlx_kernels.py (2x), mlx_primitives.py (1x), mlx_timestepper.py (1x), _riemann_reconstruction.py (1x) | 5e5 / 2.5e11 |
| P_FLOOR | mlx_primitives.py, mlx_kernels.py (2x), mlx_sources.py, mlx_transport.py | 1e-12 |
| RHO_FLOOR | mlx_primitives.py, mlx_kernels.py (2x), mlx_sources.py | 1e-12 |
| GAMMA_DEFAULT | mlx_kernels.py (2x), mlx_primitives.py | 5/3 |
| MU_0 | mlx_solver.py, mlx_transport.py, mlx_sources.py | 4pi*1e-7 |

**Metal shader exception**: `mlx_kernels.py` Metal shader constants are MSL, cannot import Python. Add a test that extracts the Metal value and compares to Python value.

**Implementation**:
1. Create `src/dpf/metal/constants.py`
2. Find-and-replace all Python-side definitions to import from constants
3. Add `test_constants_consistency.py` that verifies Metal shader values match Python
4. Run full test suite to verify no regressions

**Acceptance criteria**: `grep -rn "2.5e11\|_C_BORIS_SQ = " src/dpf/metal/*.py` returns only constants.py and Metal shader.

---

### 1.3 Formal V&V Plan Document

**Problem**: No formal V&V plan. No traceability matrix. Fidelity rating is subjective.

**Deliverable**: `docs/VV_PLAN.md` with:

1. **Requirements table**: Each physics capability with quantitative acceptance criterion
2. **Traceability matrix**: Each requirement → specific test(s) that verify it
3. **Analytical benchmarks**: For each solver component, the analytical solution it must match
4. **Convergence study template**: Grid refinement with Richardson extrapolation
5. **Experimental validation targets**: For each device preset, the acceptance tolerance

**Template for each requirement**:
```
REQ-XXX: [Physics capability]
Acceptance: [Quantitative threshold]
Test: [test_file.py::test_name]
Analytical reference: [Citation or formula]
Status: [VERIFIED / PENDING / FAILING]
```

**Implementation**: Manual document creation. No code changes.

**Acceptance criteria**: Every physics module (HLLS, HLL, HLLD, Boris, Lee-More, anomalous, RKL2, conduction, flux limiter) has at least one entry in the V&V matrix.

---

## Phase 2: Physics Defect Fixes (Sessions 3-4)

### 2.1 Fix Cylindrical Energy Source Term

**Problem** (VERIFIED): `apply_geometric_sources` computes `dE = v·(rho*S_acceleration)*dt` but the correct cylindrical energy source is `S_E = [(E + p_total)*vr - Br*(v·B)] / r`.

**Missing terms**: `E*vr/r - Br²*vr/r + Bt²*vr/r - Br*Bz*vz/r - 2*Br*Bt*vt/r`

**Verification** (from Stone & Norman 1992, eq. 3.4): The energy flux in r-direction is `F_E_r = (E+p_total)*vr - Br*(v·B)`. The geometric source is `F_E_r / r`.

**Implementation**:
- File: `src/dpf/metal/mlx_sources.py:apply_geometric_sources` (lines 170-192)
- Replace `dE = vr * dmr + vz * dmz + vt * dmt` with the full `S_E * dt`
- Also update the Metal MSL kernel in `mlx_kernels.py` to include the energy source
- Also update the NumPy reference `cylindrical_source_numpy`

**Test**: Create `test_cylindrical_energy_conservation.py`:
- Uniform rotating plasma: verify dE/dt matches analytical `(p_total*vr/r)*V*dt`
- Radial implosion: verify total energy conservation within 0.1% over 100 steps

**Acceptance criteria**: Energy drift < 0.1% over 100 steps for a cylindrical implosion test.

---

### 2.2 Make HLL-GPU the Default Riemann Solver

**Problem** (VERIFIED): HLLS uses entropy-derived `E_tot` for energy flux (non-conservative). HLL-GPU uses `U[IEN]` directly (conservative). For energy conservation, HLL-GPU should be default.

**Evidence**: mlx_riemann.py line 163 (HLLS) vs line 326 (HLL) — the comments explicitly state the difference.

**Implementation**:
- File: `src/dpf/metal/mlx_solver.py` constructor
- Change default: `self._riemann: str = "hll"` (currently "hlld")
- Update calibration scripts to use "hll" explicitly
- Document: HLLS remains available for debugging float32 cancellation issues

**Test**: Run Sod shock tube with both HLL and HLLS, compare energy conservation over 1000 steps.

**Acceptance criteria**: HLL energy conservation < 1e-6 relative error. HLLS > HLL (expected).

---

### 2.3 Add Boris Correction to HLLD Metal Kernel

**Problem** (VERIFIED): The HLLD Metal kernel's `fast_magnetosonic()` function (mlx_kernels.py:365-378) uses raw Alfven speed without Boris capping. HLLS and HLL have Boris. Switching between solvers changes the physics.

**Implementation**:
- File: `src/dpf/metal/mlx_kernels.py`, function `fast_magnetosonic` in `_HLLD_HEADER`
- Add Boris capping: `va_sq = va_sq * C_BORIS_SQ / (va_sq + C_BORIS_SQ)` after line 371
- Invalidate kernel cache (change kernel name)

**Test**: Run HLLD on a vacuum-adjacent state, verify wavespeeds are bounded at c_boris.

**Acceptance criteria**: `max(cf) <= c_boris * 1.1` for all cells including vacuum.

---

### 2.4 Fix Anomalous Resistivity Saturation

**Problem** (VERIFIED): `ratio_sq = np.minimum((v_d/v_ti)**2, 1.0)` caps the anomalous resistivity at v_d = v_ti. Physically, the instability grows stronger with increasing v_d/v_ti. The saturation should be at a higher value or removed.

**Reference**: Huba 1985, GORGON uses uncapped `(v_d/v_ti - 1)` scaling.

**Implementation**:
- File: `src/dpf/metal/mlx_transport.py:anomalous_resistivity`, line 423
- Change: `ratio_sq = np.minimum((v_d / np.maximum(v_ti, 1.0))**2, 100.0)` (cap at 100x, not 1x)
- The global Bohm cap (`np.clip(eta_anom, 0.0, 1e-2)`) at line 430 still provides an upper bound

**Test**: Verify that at v_d = 10*v_ti, anomalous eta is 100x larger than at v_d = v_ti (currently it's capped at 1x).

**Acceptance criteria**: `eta_anom(v_d=10*v_ti) > 10 * eta_anom(v_d=v_ti)`.

---

## Phase 3: Code Quality (Session 5)

### 3.1 Fix Frontend Backend Slider

**Problem** (VERIFIED): `state.py:327-329` hardcodes `hlls+plm+ssp_rk2` regardless of `backend_level` selection.

**Implementation**:
- File: `frontendv2/frontendv2/state.py`, `run_simulation` method
- Map backend_level to actual solver config:
  - Level 1: Lee model only (no MHD)
  - Level 2: Python backend, plm+hll+ssp_rk2
  - Level 3: MLX, plm+hlls+ssp_rk2
  - Level 4: MLX, weno5z+hlls+ssp_rk3
  - Level 5: MLX, weno5z+hlld+ssp_rk3, 64x128 grid, anomalous+hall enabled

**Test**: Run backend level 3 and 5 on same preset, verify different I_peak (WENO5 should give sharper sheath).

---

### 3.2 Wire Real Data to Frontend

**Problem** (VERIFIED): Neutron yield hardcoded `1.3e11`. Energy partition synthetic from I^2 proportions.

**Implementation**:
- Wire `YieldTracker` output to `state.neutron_yield`
- Wire actual energy balance from solver's energy tracking to `energy_magnetic`, `energy_kinetic`, `energy_thermal`
- Label any remaining estimated values as "(estimated)" in the UI

---

### 3.3 Fix Dead Code in Timestepper

**Problem** (VERIFIED): `_stage_post_impl` has `drho = mx.zeros_like(rho)` (line 76) making the energy correction on line 88-91 always zero. Dead code from removed density floor.

**Implementation**: Remove the dead `drho` computation and the zero energy correction. Simplify `E_floored = mx.maximum(U[IEN], P_FLOOR)`.

---

### 3.4 Fix Bare Exception in Electrode BC

**Problem** (VERIFIED): `mlx_solver.py:431` catches `except Exception` silently. Logic bugs in the MLX path will never surface.

**Implementation**: Change to `except (RuntimeError, IndexError, ValueError) as exc: logger.warning(f"MLX electrode BC failed: {exc}, using NumPy fallback")`.

---

## Phase 4: Convergence Studies (Session 6)

### 4.1 Cylindrical MHD Convergence Study

**Problem**: No convergence study exists for cylindrical geometry (the code's primary use case).

**Deliverable**: `scripts/convergence_study_cylindrical.py`
- Problem: resistive z-pinch equilibrium (analytical solution exists)
- Grids: 16x32, 32x64, 64x128, 128x256
- Measure: L1 error vs analytical, compute convergence order
- Expected: 2nd order for PLM, ~1.8 for WENO5 (limited by MHD nonlinearity)

---

## Execution Order

| Session | Tasks | Exit Criteria |
|---------|-------|---------------|
| **1** | 1.1 (pre-commit hook), 1.2 (constants.py) | Hook blocks bad commits, grep shows single source |
| **2** | 1.3 (V&V plan), 3.3 (dead code), 3.4 (bare exception) | V&V doc exists with matrix, dead code removed |
| **3** | 2.1 (energy source), 2.2 (HLL default) | Energy conservation <0.1%, HLL is default |
| **4** | 2.3 (Boris HLLD), 2.4 (anomalous cap) | Boris consistent across all solvers, anomalous scales properly |
| **5** | 3.1 (backend slider), 3.2 (real data wiring) | Slider changes physics, no hardcoded values in UI |
| **6** | 4.1 (convergence study) | Measured convergence order documented |

**Total estimated effort**: 6 focused sessions (~3-4 hours each)
**Target sigma improvement**: 1.5 → 3.0 (50% rework → <7% rework)
**Target fidelity**: 7.0 → 8.5
