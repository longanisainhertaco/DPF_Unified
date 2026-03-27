# Sprint S-2 DMAIC Plan: Performance + HLLD-S + Current Dip Fix

**Date**: 2026-03-26
**Phase**: S-2 (post S-1 physics hardening)
**Fidelity target**: 8.3 -> 8.7/10
**Sprint duration**: 3-5 sessions

---

## DEFINE

### Objective

Eliminate the CPU transport bottleneck (85% of step time), enable float32 HLLD
on GPU without precision loss, and fix the current dip signature -- the single
most important DPF diagnostic that currently reads 0% vs the >5% expected.

### Deliverables

| # | Item | Exit Criterion |
|---|------|----------------|
| D1 | RKL2 wiring for resistive diffusion | `_do_resistive_diffusion` uses RKL2 on MLX; Thomas fallback preserved via `diffusion_method` config; full-step time < 8 ms at 32x64 (was 38.5 ms) |
| D2 | RKL2 wiring for thermal conduction | `_do_thermal_conduction` uses RKL2 on MLX; same config switch |
| D3 | HLLD-S hybrid Riemann solver | `cons_to_prim` in Metal kernel reads entropy slot for pressure; `riemann_solver="hlld_s"` option; Brio-Wu L1 < 1% vs float64 HLLD |
| D4 | Anomalous resistivity in MLX solver | Drift-velocity threshold from `turbulence.anomalous` wired into MLX transport; activates at pinch |
| D5 | Current dip fix (>5% dip depth) | `test_current_dip_present` passes; dip depth > 5% on PF-1000 20kV at 32x64 |

### Out of Scope

- AMR refluxing (Phase S+)
- Implicit Hall MHD (Phase S+)
- 3D DPF physics (Phase T+)
- Batched Metal tridiagonal solver (research incomplete)
- fc/fm recalibration (deferred to post-sprint validation)

### Fidelity Rating Criteria

| Rating | Meaning |
|--------|---------|
| 8.3 (current) | Boris vacuum + Lee-More + flux-limited conduction; transport CPU-bottlenecked; no anomalous resistivity; no current dip |
| 8.5 | + RKL2 GPU transport + HLLD-S float32 (performance + precision) |
| 8.7 | + anomalous resistivity + current dip fix (physics completeness) |
| 9.0 | + recalibrated fc/fm with new physics (future sprint) |

---

## MEASURE

### Baseline Performance (32x64 grid, full physics step)

| Component | Time (ms) | Fraction |
|-----------|-----------|----------|
| Resistive diffusion (Thomas CPU) | 23.0 | 60% |
| Thermal conduction (Thomas CPU) | 9.5 | 25% |
| Core hyperbolic (GPU) | 5.9 | 15% |
| Ghost BC | < 0.1 | < 1% |
| **Total** | **38.5** | 100% |

### Baseline Accuracy

| Metric | Value | Source |
|--------|-------|--------|
| I_peak error vs Gribkov | 6.3% | fc=0.797, fm=0.084 |
| t_peak error vs Gribkov | 10.1% | 0-3% timing ambiguity |
| Current dip depth | 0% | test_current_dip_present FAILS |
| Brio-Wu L1(rho) HLLD float64 | ~5% | reference |
| Brio-Wu L1(rho) HLL float32 | ~15% | reference |

### Test Coverage Gaps

| Area | Status |
|------|--------|
| RKL2 diffusion convergence (analytical) | NOT TESTED |
| RKL2 vs Thomas parity | NOT TESTED |
| HLLD-S vs HLLD float64 parity | NOT TESTED |
| Anomalous resistivity activation at pinch | NOT TESTED in MLX path |
| Post-pinch column expansion model | Implemented in snowplow.py; not validated against MHD dip depth |

### Existing Module Inventory

| Module | Status | LOC |
|--------|--------|-----|
| `mlx_sts.py` | COMPLETE (generic RKL2 integrator) | 106 |
| `mlx_sts_operators.py` | DOES NOT EXIST (needed) | ~100 est |
| `mlx_transport.py` | Thomas CPU path (to be preserved) | existing |
| `turbulence/anomalous.py` | COMPLETE (3 threshold models, Numba) | ~200 |
| `mlx_kernels.py` | HLLD Metal kernel (needs ISR slot read) | existing |
| `validation/pinch_physics.py` | expansion_timescale() exists | existing |

---

## ANALYZE

### FMEA (Failure Modes and Effects Analysis)

| # | Failure Mode | Severity (1-10) | Occurrence (1-10) | Detection (1-10) | RPN | Mitigation |
|---|-------------|-----------------|-------------------|-------------------|-----|------------|
| F1 | RKL2 cylindrical stencil incorrect (1/r terms) | 8 | 4 | 3 | 96 | Unit test: Gaussian diffusion on cylinder vs analytical; cross-check vs Thomas output |
| F2 | RKL2 stages hit cap (s=20) at sheath, sub-cycling needed | 5 | 6 | 2 | 60 | Implement sub-cycling from research doc Section 7.2; cap at 5 sub-cycles |
| F3 | HLLD-S entropy tracer inaccurate at shocks | 7 | 3 | 4 | 84 | entropy_resync() already handles this; test on Brio-Wu specifically |
| F4 | HLLD-S cons_to_prim change breaks Metal kernel buffer layout | 9 | 3 | 2 | 54 | ISR slot (index 5) already in state; just read it in kernel |
| F5 | Anomalous eta too large -> RKL2 stages explode | 6 | 5 | 3 | 90 | Cap eta_anom at 0.01 Ohm-m (same as Lee-More cap); monitor s count |
| F6 | Current dip still absent after anomalous resistivity | 7 | 5 | 2 | 70 | Root cause is post-pinch Lp model, not just resistivity; may need MHD-coupled expansion tuning |
| F7 | Float32 accumulation in RKL2 s=16 stages | 4 | 4 | 5 | 80 | Ohmic heating dB^2 is same precision risk as Thomas; DPF is resistivity-limited not precision-limited |
| F8 | Thomas fallback path regresses silently | 6 | 3 | 7 | 126 | Regression test: run Brio-Wu with `diffusion_method="thomas"` explicitly |

**Top 3 by RPN**: F8 (126), F1 (96), F5 (90). These get dedicated test coverage.

### Dependency Graph

```
D1 (RKL2 resistive) ──────────────────────┐
    requires: mlx_sts_operators.py         │
    requires: mlx_solver.py branching      │
                                           ├──► D5 (current dip fix)
D2 (RKL2 conduction) ─────────────────────┤     requires: D1 + D4 + expansion model
    requires: mlx_sts_operators.py         │     tuning on GPU (fast iteration)
    requires: mlx_solver.py branching      │
                                           │
D3 (HLLD-S) ──────────────────────────────┤
    independent of D1/D2                   │
    requires: mlx_kernels.py ISR read      │
                                           │
D4 (anomalous resistivity) ───────────────┘
    requires: D1 (RKL2 must handle variable eta)
    requires: turbulence/anomalous.py MLX port
```

**Critical path**: D1 -> D4 -> D5. D2 and D3 are parallel with D1.

### Pareto Analysis (20% effort -> 80% value)

| Item | Effort (LOC) | Value |
|------|-------------|-------|
| **D1 (RKL2 resistive)** | ~120 | **HIGH** — eliminates 60% of step time |
| **D3 (HLLD-S)** | ~50 | **HIGH** — enables float32 HLLD on GPU, 2-3x over CPU float64 |
| D2 (RKL2 conduction) | ~30 (reuses D1 infrastructure) | MEDIUM — eliminates 25% of step time |
| D4 (anomalous resistivity) | ~80 | MEDIUM — physics completeness for pinch |
| D5 (current dip fix) | ~50 (tuning + test) | HIGH for validation, but depends on D1+D4 |

**80/20 conclusion**: D1 + D3 deliver the largest performance and accuracy gains.
D2 is nearly free once D1 is done. D4 + D5 are the physics payoff.

---

## IMPROVE

### Implementation Plan

#### Item D1: RKL2 Resistive Diffusion Wiring (~120 LOC)

**New file**: `src/dpf/metal/mlx_sts_operators.py`

```
make_resistive_rhs(eta, dr, dz, r_cell, coordinates) -> Callable
  - Vectorized 2D cylindrical Laplacian: (1/r) d/dr(r * alpha * dB/dr) + d^2B/dz^2
  - alpha = eta / mu_0
  - Face-centered r-fluxes with r_face weighting
  - Neumann BC (zero-flux) at domain boundaries
  - Returns closure rhs_fn(B) -> dB/dt

make_conduction_rhs(chi_r, chi_z, dr, dz, r_cell, coordinates) -> Callable
  - Same pattern for anisotropic thermal conduction
  - chi weighted by B-field direction (Braginskii)
```

**Modified file**: `src/dpf/metal/mlx_solver.py`

```
MLXMHDSolver.__init__:
  + self._diffusion_method = diffusion_method  # "rkl2" (default) or "thomas"

_do_resistive_diffusion:
  if self._diffusion_method == "rkl2":
    rhs_fn = make_resistive_rhs(eta, dr, dz, r_cell, self.coordinates)
    dt_explicit = min(dr, dz)**2 / (2.0 * max(alpha_max, 1e-30))
    s = compute_sts_stages(dt, dt_explicit)
    Br_new = rkl2_step_mlx(Br, rhs_fn, dt, s_stages=s)
    Bz_new = rkl2_step_mlx(Bz, rhs_fn, dt, s_stages=s)
    Bt_new = rkl2_step_mlx(Bt, rhs_fn, dt, s_stages=s)
    # Ohmic heating from dB^2
  else:
    # existing Thomas path (unchanged)
```

**Tests** (~60 LOC in `tests/test_mlx_sts_operators.py`):
1. Gaussian blob diffusion on cylinder: convergence rate = 2nd order (RKL2)
2. RKL2 vs Thomas parity: L2 difference < O(dt) on identical problem
3. Stage count: verify s=4-8 for typical DPF conditions, s=16 for extreme eta
4. Sub-cycling: verify correctness when s hits cap (s=20, n_sub > 1)
5. Regression: Sod + Brio-Wu with `diffusion_method="rkl2"` produce same results as `"thomas"`
6. Regression: explicit `diffusion_method="thomas"` still works (F8 mitigation)

**Performance target**: 32x64 step time < 8 ms (from 38.5 ms). Transport fraction < 10%.

---

#### Item D2: RKL2 Thermal Conduction (~30 LOC)

Reuses `mlx_sts_operators.py` from D1. Only requires:
- `make_conduction_rhs()` closure (already in D1 file)
- Branching in `_do_thermal_conduction` (same pattern as D1)
- Anisotropy: chi_parallel >> chi_perp, weighted by local B-field direction

**Tests**: Same pattern as D1 — convergence, parity, regression.

---

#### Item D3: HLLD-S Hybrid Riemann Solver (~50 LOC)

**Modified file**: `src/dpf/metal/mlx_kernels.py`

In `_HLLD_HEADER` Metal kernel, change `cons_to_prim`:
```metal
// BEFORE (cancellation-prone):
float p = max((gamma - 1.0f) * (E - ke - mag), P_FLOOR);

// AFTER (entropy-derived):
float Srho = max(U[ISR * stride + idx], 1e-30f);
float p_entropy = max(Srho * pow(rho, gamma - 1.0f), P_FLOOR);
float p = p_entropy;  // Use for star-state structure
// E still used for physical flux (preserves energy conservation)
```

The ISR slot is already in the state array (index 5). The kernel just needs to
read it. No buffer layout change.

**Modified file**: `src/dpf/metal/mlx_riemann.py`

Add `riemann_solver="hlld_s"` option to `compute_fluxes()`. Internally this
calls the same `hlld_flux_mlx` kernel but with the entropy-pressure flag.

**Tests** (~40 LOC):
1. HLLD-S vs HLLD float64 on Brio-Wu: L1(rho) < 1%
2. HLLD-S energy conservation on Orszag-Tang: dE/E < 1e-6
3. HLLD-S no NaN fallback on DPF full discharge (32x64, 100 steps)
4. Benchmark: HLLD-S(float32 GPU) vs HLLD(float64 CPU) timing

---

#### Item D4: Anomalous Resistivity in MLX (~80 LOC)

**Problem**: `turbulence/anomalous.py` uses Numba `@njit`. Cannot call directly
from MLX arrays. Need an MLX-native wrapper.

**New function** in `mlx_sts_operators.py` or `mlx_transport.py`:

```python
def anomalous_resistivity_mlx(
    J_mag: mx.array,      # |J| = |curl B| / mu_0
    ne: mx.array,          # electron density
    Te: mx.array,          # electron temperature [K]
    ion_mass: float,
    alpha: float = 0.01,   # turbulence parameter
    threshold: str = "ion_acoustic",
) -> mx.array:
    """MLX-native anomalous resistivity from drift-velocity threshold."""
    v_drift = J_mag / (ne * e_charge + 1e-30)
    if threshold == "ion_acoustic":
        cs = mx.sqrt(k_B * Te / ion_mass)
        triggered = v_drift > cs
    # eta_anom = alpha * m_e * omega_pe / (ne * e^2)
    omega_pe = mx.sqrt(ne * e_charge**2 / (m_e * epsilon_0))
    eta_anom = alpha * m_e * omega_pe / (ne * e_charge**2 + 1e-30)
    eta_anom = mx.where(triggered, eta_anom, mx.zeros_like(eta_anom))
    return mx.minimum(eta_anom, 0.01)  # cap at 0.01 Ohm-m
```

Wire into `_do_resistive_diffusion`: compute J from curl(B), compute eta_total
= eta_spitzer_or_leemore + eta_anom, pass to RKL2.

**Tests**:
1. Unit: anomalous eta activates above cs threshold, zero below
2. Integration: DPF discharge with anomalous resistivity enabled, verify eta_anom > 0 in pinch region
3. Stability: no NaN with eta_anom active at s=16 RKL2

---

#### Item D5: Current Dip Fix (~50 LOC tuning + diagnostics)

**Root cause analysis**: The current dip requires dL_p/dt > 0 during pinch
(rising plasma inductance from column compression). Currently 0% dip because:

1. **Post-pinch Lp model is too weak**: The MHD-computed Lp doesn't increase
   fast enough during pinch to produce a measurable back-EMF.
2. **Missing anomalous resistivity**: Without anomalous eta at pinch, the
   column doesn't heat enough to expand quickly, so Lp stagnates.
3. **Column expansion timescale**: The m=0 disruption that drives rapid
   expansion (and dip recovery) isn't modeled in the MHD coupling.

**Fix strategy** (three-pronged):
1. D4 provides anomalous resistivity -> stronger Ohmic heating at pinch -> faster expansion
2. Verify Lp computation uses density-peak sheath detection (not B_theta extent)
3. If still insufficient: add explicit column expansion model from
   `snowplow.py:post_pinch_rhs` into the MHD-circuit coupling

**Tests**:
1. `test_current_dip_present`: dip_depth > 5% (currently 0%)
2. Lp monotonicity during pinch: dLp/dt > 0 for at least 3 consecutive timesteps
3. Comparison against Lee model dip depth (should be within factor of 2)

---

### Implementation Order

```
Session 1: D1 (RKL2 resistive) + D2 (RKL2 conduction)
  - Create mlx_sts_operators.py
  - Wire into mlx_solver.py
  - Write tests
  - Benchmark: verify < 8 ms step time
  Time estimate: 2-3 hours

Session 2: D3 (HLLD-S)
  - Modify Metal kernel cons_to_prim
  - Add riemann_solver="hlld_s" option
  - Write parity + conservation tests
  - Benchmark: float32 GPU vs float64 CPU
  Time estimate: 1-2 hours

Session 3: D4 (anomalous resistivity) + D5 (current dip)
  - Port anomalous eta to MLX
  - Wire into transport
  - Tune post-pinch coupling
  - Fix test_current_dip_present
  Time estimate: 2-3 hours
```

---

## CONTROL

### Regression Gates (must pass before sprint completion)

| Gate | Test | Criterion |
|------|------|-----------|
| G1 | `pytest tests/test_mlx_sts_operators.py -v` | All pass |
| G2 | `pytest tests/ -k "brio_wu" -v` | All pass (no regression from HLLD-S) |
| G3 | `pytest tests/ -k "sod" -v` | All pass |
| G4 | `pytest tests/test_mlx_solver.py -v` | All pass |
| G5 | `pytest tests/ -k "current_dip" -v` | At least 1 MHD-path test passes with >5% dip |
| G6 | `pytest tests/ -k "thomas" --diffusion-method=thomas` | Thomas fallback still works |
| G7 | `pytest tests/ -x -q -m "not slow"` | >= 4000 tests pass (CI gate) |

### Performance Benchmarks

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Full step time (32x64) | 38.5 ms | < 8 ms | `mlx_benchmark.py --profile` |
| Transport fraction | 85% | < 15% | Profile breakdown |
| HLLD-S vs HLL throughput | N/A | HLLD-S >= 0.7x HLL speed | `mlx_benchmark.py --riemann` |
| RKL2 stage count (typical) | N/A | 4-8 (pre-pinch), 8-16 (sheath) | Logged during sim |

### Calibration Validation

After sprint completion, run the fc/fm smoke test:
```
python3 -c "
from dpf.metal.mlx_solver import MLXMHDSolver
# 32x64, fc=0.797, fm=0.084, 100 steps
# Verify I_peak within 20% of 1.87 MA reference
"
```

If I_peak drifts > 20%, schedule recalibration sprint. Do NOT adjust fc/fm
during this sprint -- isolate performance/precision changes from physics
parameter changes.

### Fidelity Advancement Criteria

| From | To | Requires |
|------|----|---------|
| 8.3 | 8.5 | D1 + D2 + D3 complete; step time < 10 ms; HLLD-S parity confirmed |
| 8.5 | 8.7 | D4 + D5 complete; current dip > 5%; anomalous eta active at pinch |
| 8.7 | 9.0 | Future sprint: recalibrate fc/fm with full S-2 physics; AMR refluxing; implicit Hall |

### Knowledge Capture (post-sprint)

After each item completes, update:
- `CLAUDE.md` Lessons Learned: RKL2 stage count behavior, HLLD-S float32 parity results
- `memory/observations.md`: any unexpected behavior (e.g., RKL2 accuracy better than Thomas)
- `docs/RESEARCH_INDEX.md`: link to this DMAIC and any new research docs

### Monitoring During Sprint

- Log RKL2 stage count per timestep (diagnostic, not production)
- Log HLLD-S NaN fallback rate (should be 0%)
- Log anomalous eta activation fraction (cells where eta_anom > 0)
- Compare energy conservation with/without HLLD-S

---

## Appendix A: Risk Mitigation Decision Tree

```
RKL2 stages > 20?
  ├── YES: sub-cycle (n_sub = ceil(dt / dt_super_max), cap at 5)
  │   └── Still too slow? → fall back to Thomas for that timestep
  └── NO: proceed normally

HLLD-S produces NaN?
  ├── YES at > 0.1% interfaces: proceed to Phase 2 (double-float D_L/D_R)
  └── YES at < 0.1%: existing Lax-Friedrichs fallback handles it

Current dip still 0% after D4?
  ├── YES: root cause is Lp model, not resistivity
  │   ├── Check sheath detection method (density peak vs B_theta extent)
  │   ├── Add explicit m=0 expansion from snowplow.py
  │   └── If still 0%: escalate to dpf-mhd-physicist for conservation audit
  └── NO (dip > 5%): success
```

## Appendix B: File Change Summary

| File | Action | LOC |
|------|--------|-----|
| `src/dpf/metal/mlx_sts_operators.py` | CREATE | ~100 |
| `src/dpf/metal/mlx_solver.py` | MODIFY (diffusion_method branching) | ~40 |
| `src/dpf/metal/mlx_kernels.py` | MODIFY (ISR read in cons_to_prim) | ~10 |
| `src/dpf/metal/mlx_riemann.py` | MODIFY (hlld_s option) | ~15 |
| `src/dpf/metal/mlx_transport.py` | MODIFY (anomalous eta integration) | ~30 |
| `tests/test_mlx_sts_operators.py` | CREATE | ~80 |
| `tests/test_mlx_hlld_s.py` | CREATE | ~60 |
| **Total new code** | | **~335** |

## Appendix C: References

1. Meyer, Balsara & Aslam, JCP 231:2963 (2012) -- RKL2 method
2. Miyoshi & Kusano, JCP 208:315 (2005) -- HLLD solver
3. Popovas et al., A&A 694 (2025) -- HLLS entropy-stable solver (arXiv:2211.02438)
4. Stone et al., ApJS 249:4 (2020) -- Athena++ RKL2 STS implementation
5. Sagdeev, Rev. Plasma Phys. 4:23 (1966) -- anomalous resistivity
6. Haines, PPCF 53:093001 (2011) -- DPF review (current dip physics)
