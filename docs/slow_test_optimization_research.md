# Slow Test Optimization Research Report

**Date**: 2026-02-26
**Hardware**: M3 Pro MacBook Pro, 36GB unified memory, Apple Metal GPU
**Scope**: 139 `@pytest.mark.slow` tests across 24 test files
**Goal**: Reduce slow test suite wall-clock time from ~30+ min to <10 min

---

## 1. Executive Summary

### Key Findings

1. **Metal GPU tests dominate**: 72 of 139 slow tests (52%) are already Metal-based (`test_metal_production.py` + `test_phase_o_physics_accuracy.py` + `test_phase_n_cross_backend.py`). These cannot be "ported to Metal" — they **are** Metal. Their bottleneck is solver step count (30-300 steps per test).

2. **WALRUS inference is the heaviest single bottleneck**: 13 tests in `test_verification_walrus.py` share a 4.8GB model load (~10s) + ~58s CPU forward pass. Module-scoped fixtures already cache the model and prediction, but the initial load + first inference is unavoidable (~68s).

3. **Grid size reduction has the highest ROI**: Many Metal tests use 32x16x16 or 16x16x16 grids with 20-300 steps. Reducing to 16x8x8 where possible (convergence tests excluded) could cut per-test time by 4-8x.

4. **pytest-xdist parallelization is viable**: Most slow tests are CPU/GPU independent. With 6P+6E cores, running 4-6 workers can achieve ~3-4x wall-clock reduction. Metal tests must be serialized (single MPS device).

5. **Numba JIT warmup is already mitigated**: All 90 `@njit` functions use `cache=True`. Cold start only affects the first run after a code change.

### Top 5 Recommendations

| # | Recommendation | Est. Speedup | Effort |
|---|---------------|-------------|--------|
| 1 | Reduce grid sizes where accuracy allows (32x16x16 → 16x8x8) | 4-8x per test | Low |
| 2 | Reduce step counts in stability tests (300 → 50-100 where safe) | 3-6x per test | Low |
| 3 | Use pytest-xdist with 4 workers, grouping Metal tests | ~3x wall-clock | Medium |
| 4 | Move WALRUS tests to MPS inference (1.57x faster than CPU) | 1.5x for 13 tests | Medium |
| 5 | Add shared module-scoped solver fixtures to avoid re-instantiation | ~1.5x for groups | Low |

---

## 2. Full Test Catalog

### 2.1 Test File Summary

| # | Test File | Slow Tests | Backend | Primary Bottleneck |
|---|-----------|-----------|---------|-------------------|
| 1 | `test_metal_production.py` | 31 | Metal MPS | Solver steps, MPS kernel launch |
| 2 | `test_phase_o_physics_accuracy.py` | 27 | Metal MPS/CPU | Multi-resolution convergence, 100-300 step runs |
| 3 | `test_phase_n_cross_backend.py` | 14 | Metal MPS + Python | Dual-solver comparison, 20-200 step runs |
| 4 | `test_verification_walrus.py` | 13 | CPU (WALRUS) | 4.8GB model load + ~58s forward pass |
| 5 | `test_phase_c_verification.py` | 8 | Python (Numba) | Diffusion convergence (3 resolutions), Orszag-Tang, Sedov, Lee model |
| 6 | `test_verification_comprehensive.py` | 7 | Python + Athena++ | Sod/Brio-Wu cross-backend (subprocess), circuit verification |
| 7 | `test_phase_v_validation_fixes.py` | 5 | Python (Lee model) | Lee model sweeps, radial phase simulation |
| 8 | `test_phase17.py` | 4 | Python (Numba) | Sod/Brio-Wu shock tubes (nx=100-200), convergence |
| 9 | `test_phase_z_calibration_benchmark.py` | 4 | Python (scipy + Lee) | Optimization loops (scipy.optimize.minimize) |
| 10 | `test_verification_rlc.py` | 4 | Python (circuit) | 2000-5000 step RLC waveform comparison |
| 11 | `test_phase_r_walrus_hybrid.py` | 2 | Athena++ / Hybrid | Athena++ solver creation + hybrid engine |
| 12 | `test_phase_y_crossval.py` | 2 | Python (scipy + Lee) | CrossValidator optimization |
| 13 | `test_verification_mhd_convergence.py` | 2 | Python (Numba) | Multi-resolution MHD convergence (16,32,64) |
| 14 | `test_athena_wrapper.py` | 2 | Athena++ subprocess | Binary execution + I/O |
| 15 | `test_phase_f_verification.py` | 2 | Athena++ subprocess | Sod/Brio-Wu via Athena++ binary |
| 16 | `test_well_integration.py` | 1 | Python (hybrid engine) | SimulationEngine + HybridEngine run |
| 17 | `test_phase_q_transport.py` | 1 | Metal MPS | Whistler wave (currently skipped) |
| 18 | `test_phase_q_hlld_ct.py` | 1 | Python + Metal | CT parity comparison, 30 steps each |
| 19 | `test_phase_p_accuracy.py` | 1 | Python (Numba) | SSP-RK3 + WENO5 16^3 grid (JIT warmup) |
| 20 | `test_verification_energy_balance.py` | 1 | Python (circuit) | 1000-step RLC energy conservation |
| 21 | `test_phase_t_radial.py` | 1 | Python (snowplow) | 1M iteration radial phase sequence |
| 22 | `test_verification_system.py` | 1 | Python (engine) | SimulationEngine 10-step run with tutorial config |
| 23 | `test_stress.py` | 0* | — | *Mentioned but no actual `@pytest.mark.slow` decorator found |
| **Total** | | **139** | | |

### 2.2 Detailed Test Catalog

#### Category A: Metal GPU Tests (72 tests)

##### `test_metal_production.py` — 31 slow tests

| Test | Grid | Steps | Bottleneck | Optimization |
|------|------|-------|-----------|-------------|
| `test_ct_update_preserves_divB` | 16^3 | 1 | MPS CT kernel | Reduce to 8^3 |
| `test_div_B_uniform_field` | 16^3 | 0 | MPS divergence op | Reduce to 8^3 |
| `test_gradient_linear_field` | 16^3 | 0 | MPS gradient op | Reduce to 8^3 |
| `test_laplacian_quadratic` | 16^3 | 0 | MPS Laplacian | Reduce to 8^3 |
| `test_strain_rate_rigid_rotation` | 8^3 | 0 | MPS strain rate | Already small |
| `test_implicit_diffusion_smooths` | 16^3 | 1 | MPS diffusion | Reduce to 8^3 |
| `test_hll_flux_uniform` | 8^3 | 0 | HLL flux compute | Already small |
| `test_hll_flux_conservation` | 8^3 | 0 | HLL flux compute | Already small |
| `test_plm_reconstruct_constant` | 16^3 | 0 | PLM reconstruct | Reduce to 8^3 |
| `test_compute_fluxes_symmetry` | 16x4x4 | 0 | Flux pipeline | Already small |
| `test_mhd_rhs_hydro_limit` | 8^3 | 0 | Full RHS compute | Already small |
| `test_solver_creation` | 16^3 | 0 | Solver instantiation | Consider un-marking as slow |
| `test_solver_step` | 16^3 | 1 | Single MHD step | Consider un-marking as slow |
| `test_solver_10_steps` | 16^3 | 10 | 10 MHD steps | OK |
| `test_solver_energy_conservation` | 16^3 | 5 | 5 steps + energy calc | OK |
| `test_solver_divB_maintained` | 16^3 | 5 | 5 steps + div(B) | OK |
| `test_solver_compute_dt` | 16^3 | 0 | CFL compute | Consider un-marking |
| `test_solver_coupling_interface` | 16^3 | 1 | Coupling state | Consider un-marking |
| `test_solver_sod_shock` | 16x4x4 | 100 | **100 steps on small grid** | Reduce to 50 steps |
| `test_engine_backend_metal` | 16^3 | 0 | Engine creation | Consider un-marking |
| `test_engine_metal_5_steps` | 16^3 | 5 | 5 engine steps | OK |
| `test_engine_metal_state_sanity` | 16^3 | 3 | 3 engine steps | OK |
| `test_float32_vs_float64_stencil` | 16^3 | 1 | CT + reference CPU | OK |
| `test_float32_riemann_stability` | 16 (1D) | 0 | HLL single call | Consider un-marking |
| `test_benchmark_suite_completes` | 16 | varies | Full benchmark suite | OK (integration) |

**Quick wins for `test_metal_production.py`**: 7 tests could be un-marked as slow (instantiation/single-op tests that run in <1s on MPS). 6 tests could reduce grid from 16^3 to 8^3.

##### `test_phase_o_physics_accuracy.py` — 27 slow tests

| Test Class | Tests | Grid | Steps | Bottleneck |
|-----------|-------|------|-------|-----------|
| `TestMetalHLLBrioWu` | 4 | 32x16x16 | 20-50 | **Largest grid × most steps** |
| `TestMetalHLLD` | 3 | 32x16x16 | 20-30 | Dual-solver comparison (HLL vs HLLD) |
| `TestMetalConvergenceOrder` | 3 | 16,32 × 8×8 | 3 each | Multi-resolution (2 grids × 3 steps) |
| `TestMetalLongRunEnergy` | 2 | 16^3 | 100-300 | **300 steps is the worst offender** |
| `TestMetalPythonParity` | 1 | 16x8x8 | 10 | Dual-engine (Metal + Python) |
| `TestFloat64Precision` | 1 | 16^3 | 100 | Dual-precision (f32 + f64) × 100 steps |
| `TestWENO5Reconstruction` | 3 | 32x16x16 | 30-50 | WENO5 reconstruction + multi-step |
| `TestFormalConvergenceOrder` | 3 | 16,32,64 × 8×8 | 5 each | **Three resolutions, hardest to reduce** |
| `TestSSPRK3` | 4 | 16-32 × 8-16 | 30-100 | Multi-step stability/energy |
| `TestSSPRK3ConvergenceOrder` | 3 | 16,32,64 × 8×8 | 5-10 | Three resolutions |

**Key insight**: The 300-step energy drift test (`test_300_step_energy_drift`) and the triple-resolution convergence tests are the heaviest. The 300-step test could be reduced to 100 steps with a tighter tolerance. Convergence tests need at least 2 resolutions but the 64×8×8 runs are expensive.

##### `test_phase_n_cross_backend.py` — 14 slow tests

| Test Class | Tests | Grid | Steps | Bottleneck |
|-----------|-------|------|-------|-----------|
| `TestMetalSodParity` | 4 | 32x4x4 | 20 | Python + Metal Sod in sequence |
| `TestMetalMHDWaveParity` | 3 | 16^3 | 10 | Metal MPS with CT (10 steps) |
| `TestMetalEnergyConservation` | 3 | 16x8x8 | 50-200 | **100-200 step long-runs** |
| `TestMetalFloat32Fidelity` | 1 | 8^3 | 10 | Small, could un-mark |
| `TestAthenaKCrossBackend` | 1 | 32^3 | 50 | AthenaK subprocess (if binary exists) |
| `TestMetalEngineIntegration` | 2 | 8^3 | 5-10 | Engine creation + circuit coupling |

**Quick win**: The 200-step stability test could be reduced to 100. The 8^3 fidelity test is likely <1s and could be un-marked.

#### Category B: WALRUS / AI Tests (15 tests)

##### `test_verification_walrus.py` — 13 slow tests

| Test | Bottleneck | Optimization |
|------|-----------|-------------|
| `test_walrus_model_loads` | 4.8GB checkpoint load (~10s) | Module-scoped fixture (already done) |
| `test_walrus_model_field_mapping` | Batch construction | Fast after model load |
| `test_walrus_density_positive` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_pressure_positive` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_prediction_nontrivial` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_mass_approximately_conserved` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_energy_bounded` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_deterministic` | **2nd forward pass** (~58s CPU) | Move to MPS (~38s) |
| `test_walrus_continuous` | **3rd forward pass** (~58s CPU) | Move to MPS (~38s) |
| `test_walrus_static_state_stable` | Shared `predicted_state` fixture | ~0s incremental |
| `test_walrus_shock_propagates` | **4th forward pass** (~58s CPU) | Move to MPS (~38s) |
| `test_walrus_output_no_nans_or_infs` | Shared `predicted_state` fixture | ~0s incremental |
| (N/A — continuity test) | 5th forward pass | Move to MPS |

**Total estimated time**: ~10s load + 58s initial prediction + 3×58s additional predictions = **~252s on CPU, ~160s on MPS**.

**Key optimization**: The `predicted_state` fixture caches 1 prediction; 3 additional tests do their own `predict_next_step()` calls. If we can cache these too, we save ~174s on CPU.

##### `test_well_integration.py` — 1 slow test
- `test_hybrid_engine_delegation`: Runs `SimulationEngine` with `backend='hybrid'` for 10 steps. Bottleneck: engine + circuit initialization + 10 MHD steps.

##### `test_phase_r_walrus_hybrid.py` — 2 slow tests
- Both require Athena++ availability; skip if binary not found. Non-optimizable — Athena++ initialization is inherently slow.

#### Category C: Python Engine / Physics Tests (33 tests)

##### `test_phase_c_verification.py` — 8 slow tests

| Test | Description | Steps/Resolution | Bottleneck |
|------|------------|-----------------|-----------|
| `test_diffusion_convergence_implicit` | Crank-Nicolson convergence | 3 resolutions (32,64,128) | Multi-resolution diffusion |
| `test_diffusion_convergence_sts` | RKL2 STS convergence | 3 resolutions | Multi-resolution diffusion |
| `test_diffusion_convergence_explicit` | Explicit diffusion convergence | 3 resolutions | Multi-resolution diffusion |
| `test_orszag_tang_runs` | Orszag-Tang vortex | nx=32, t=0.1 | Numba JIT + MHD steps |
| `test_sedov_cylindrical_runs` | Sedov blast | nr=64, nz=128 | Cylindrical solver + many steps |
| `test_lee_model_pf1000` | Lee model PF-1000 | ODE integration | Lee model ODE solver |
| `test_lee_model_nx2` | Lee model NX2 | ODE integration | Lee model ODE solver |
| `test_lee_model_comparison_pf1000` | Lee model vs experiment | ODE + comparison | Lee model + metrics |

##### `test_verification_comprehensive.py` — 7 slow tests

| Test | Description | Bottleneck |
|------|------------|-----------|
| `TestCrossBackend.test_cross_backend_sod_shock` | Athena++ subprocess + Python 3000 steps | **Athena++ binary + 3000 Python MHD steps** |
| `TestCrossBackend.test_cross_backend_brio_wu` | Athena++ subprocess + Python 5000 steps | **5000 Python MHD steps** |
| `TestSystemVerification.test_pf1000_peak_current` | 10000 circuit steps | Moderate |
| `TestSystemVerification.test_pf1000_pinch_time` | 20000 circuit steps | Moderate |
| `TestSystemVerification.test_nx2_peak_current` | 20000 circuit steps | Moderate |
| `TestSystemVerification.test_lee_model_vs_engine` | Lee model PF-1000 run | Lee model ODE |
| (7th in class) | — | — |

##### `test_phase_v_validation_fixes.py` — 5 slow tests
- All Lee model related: PF-1000 runs, phase completion, radial force with z_f, first peak finder. Each runs the Lee model ODE solver 1-3 times.

##### `test_phase17.py` — 4 slow tests
- Sod shock tube (nx=100): ~500 solver steps
- Brio-Wu (nx=200): ~500 solver steps
- Cylindrical convergence (32,64,128): Multi-resolution
- Error-decreases-with-resolution: Multi-resolution

##### `test_verification_mhd_convergence.py` — 2 slow tests
- Sound wave at 3 resolutions (16,32,64). Bottleneck: 64×4×4 solver runs.

##### `test_phase_p_accuracy.py` — 1 slow test
- SSP-RK3 + WENO5 on 16^3 grid: 2 steps. Bottleneck: Numba JIT compilation on first call.

##### `test_phase_t_radial.py` — 1 slow test
- `test_full_phase_sequence`: Up to **1,000,000 iterations** of snowplow model stepping. The dominant bottleneck in Category C.

#### Category D: Circuit / RLC Tests (5 tests)

##### `test_verification_rlc.py` — 4 slow tests

| Test | Steps | Bottleneck |
|------|-------|-----------|
| `test_underdamped_waveform` | 5000 | Pure Python loop |
| `test_critically_damped` | 5000 | Pure Python loop |
| `test_overdamped` | 2000 | Pure Python loop |
| (4th) | varies | RLC timestep loop |

##### `test_verification_energy_balance.py` — 1 slow test
- 1000-step RLC energy conservation. Pure Python loop.

#### Category E: Calibration / Optimization Tests (6 tests)

##### `test_phase_z_calibration_benchmark.py` — 4 slow tests
- All use `scipy.optimize.minimize` with Lee model evaluations. Each iteration runs the full Lee model ODE. 30-100 iterations.

##### `test_phase_y_crossval.py` — 2 slow tests
- `CrossValidator().validate()` calls with `maxiter=5`. Small but includes scipy optimization.

#### Category F: Athena++ Subprocess Tests (4 tests)

##### `test_athena_wrapper.py` — 2 slow tests
- Subprocess single step + time update. Skip if binary not found.

##### `test_phase_f_verification.py` — 2 slow tests (class-level)
- `TestSodShockTube` and `TestBrioWuMHD`: Full Athena++ subprocess runs with HDF5 output. 256-cell 1D problems. Takes ~5-10s each when binary is available.

---

## 3. Detailed Analysis per Category

### 3.1 Metal Acceleration Opportunities

**Already on Metal (72 tests)**: These tests already use `MetalMHDSolver` on MPS. The primary speedup is reducing work per test (fewer steps, smaller grids).

**Candidates for Metal porting (from Python engine)**:

| Test File | Tests | Current Backend | Metal Feasibility | Expected Speedup |
|-----------|-------|----------------|------------------|-----------------|
| `test_phase_c_verification.py` | 3 diffusion tests | Python Numba | **Not feasible** — tests specific diffusion operators (implicit, STS, explicit) that are Python-only | N/A |
| `test_phase_c_verification.py` | 1 Orszag-Tang | Python Numba | **Partially feasible** — could add Metal Orszag-Tang, but test validates *Python* solver | N/A (different test) |
| `test_verification_comprehensive.py` | 2 cross-backend | Python + Athena++ | **Not applicable** — cross-backend comparison is the purpose | N/A |
| `test_phase17.py` | 2 shock tube | Python Numba | **Feasible** — Metal Sod/Brio-Wu already exist in `test_phase_o` | 2-3x (but tests validate *Python* engine) |
| `test_verification_mhd_convergence.py` | 2 convergence | Python Numba | **Feasible** — Metal convergence tests already exist in `test_phase_o` | 2-3x |
| `test_phase_p_accuracy.py` | 1 RK3+WENO5 | Python Numba | **Not applicable** — validates Python engine specifically | N/A |
| `test_phase_q_hlld_ct.py` | 1 CT parity | Python + Metal | **Not applicable** — cross-backend comparison | N/A |

**Conclusion**: Most Python-engine slow tests cannot be ported to Metal because they specifically test the Python engine's implementation. The speedup opportunity is in reducing their grid sizes and step counts.

### 3.2 WALRUS Inference Optimization

**Current**: CPU inference (~58s per forward pass, ~10s model load)

**MPS inference**: 1.57x speedup documented (38s vs 60s per step). Change `device="cpu"` to `device="mps"` in the test fixture.

**Additional caching**: The `predicted_state` module-scoped fixture already caches 1 prediction. Three additional tests (`test_walrus_deterministic`, `test_walrus_continuous`, `test_walrus_shock_propagates`) each call `predict_next_step()` independently. These could be cached as additional module-scoped fixtures.

**Estimated savings**:
- MPS device: ~252s → ~160s (save ~92s)
- Additional caching: Save 2 forward passes = ~116s CPU / ~76s MPS
- Combined: ~252s → ~84s (save ~168s, 3x speedup)

### 3.3 Parallelization Strategy (pytest-xdist)

#### Resource Classification

| Group | Resource | Tests | Can Parallel? |
|-------|----------|-------|--------------|
| **GPU** | Metal MPS device | 72 Metal tests | **No** (single MPS device; concurrent access causes errors) |
| **CPU-heavy** | Numba JIT + MHD | ~25 Python engine tests | **Yes** (independent processes) |
| **CPU-light** | Circuit/Lee model | ~15 RLC + Lee tests | **Yes** (pure Python, low memory) |
| **Subprocess** | Athena++ binary | 4 Athena++ tests | **Yes** (subprocess-based) |
| **AI** | WALRUS model (4.8GB) | 13 WALRUS tests | **Partial** (memory-constrained: 4.8GB per worker) |
| **Optimizer** | scipy.optimize | 6 calibration tests | **Yes** (CPU-bound) |

#### Proposed pytest-xdist Strategy

```ini
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "metal: tests requiring Apple Metal MPS device (serialize)",
    "walrus: tests requiring WALRUS model (memory-heavy)",
]
```

**Recommended**: `pytest -m slow -n 4 --dist loadgroup`

With `@pytest.mark.xdist_group("metal")` on all Metal tests to serialize them:
- **Worker 1**: All 72 Metal tests (serialized, ~15-20 min)
- **Worker 2**: Python engine + circuit tests (~8-10 min)
- **Worker 3**: Lee model + calibration tests (~5-8 min)
- **Worker 4**: WALRUS + Athena++ tests (~5-8 min with MPS inference)

**Estimated wall-clock**: ~20 min → down from 30+ min (1.5x improvement from parallelization alone).

#### M3 Pro Core Allocation

- **6 Performance cores**: Ideal for pytest workers (4 workers + OS overhead)
- **6 Efficiency cores**: Background tasks, Numba compilation
- **36GB unified memory**: Can support 2-3 WALRUS model instances if needed, but 1 is safer

**Recommendation**: 4 workers (`-n 4`) is optimal. More workers risk memory pressure from MHD arrays + WALRUS model.

### 3.4 Grid Size Optimization

| Current Grid | Tests Using It | Can Reduce To | Savings |
|-------------|---------------|--------------|---------|
| 32×16×16 | 15 (Phase O Brio-Wu, WENO5, SSP-RK3) | 16×8×8 for stability; **keep for convergence** | 8x fewer cells |
| 16×16×16 | 25 (Metal production, Phase N energy) | 8×8×8 for stencil/single-step tests | 8x fewer cells |
| 16×8×8 | 10 (convergence, parity) | Already optimal | — |
| 8×8×8 | 5 (integration, fidelity) | Already minimal | — |
| 64×8×8 | 6 (convergence order tests) | **Cannot reduce** (need resolution range) | — |
| 128×4×4 | 2 (resistive diffusion) | **Cannot reduce** (1D resolution matters) | — |

**Specific candidates for grid reduction**:

1. `test_ct_update_preserves_divB` (16^3 → 8^3): Only tests zero EMF → no change. 8^3 is sufficient.
2. `test_div_B_uniform_field` (16^3 → 8^3): Tests constant field divergence. 8^3 is sufficient.
3. `test_gradient_linear_field` (16^3 → 8^3): Tests linear gradient. 8^3 with 6 interior cells is sufficient.
4. `test_laplacian_quadratic` (16^3 → 8^3): Tests quadratic Laplacian. 8^3 is sufficient.
5. `test_implicit_diffusion_smooths` (16^3 → 8^3): Tests gradient reduction. 8^3 is sufficient.
6. `test_plm_reconstruct_constant` (16^3 → 8^3): Tests constant reconstruction. 8^3 is sufficient.

**Estimated per-test savings**: Reducing 16^3 (4096 cells) to 8^3 (512 cells) = 8x fewer cells. With Metal overhead, expect ~3-5x speedup per step.

### 3.5 Step Count Optimization

| Test | Current Steps | Can Reduce To | Rationale |
|------|--------------|--------------|-----------|
| `test_300_step_energy_drift` | 300 | 100 | With tighter tolerance (2% → 1%), 100 steps suffices |
| `test_no_exponential_growth` | 100 | 50 | Exponential growth detectable in 50 steps |
| `test_200_step_stability` | 200 | 100 | NaN/negative detection in 100 steps |
| `test_100_step_energy_drift` (Phase N) | 100 | 50 | With tighter tolerance |
| `test_solver_sod_shock` | 100 | 50 | Qualitative shock check needs ~30 steps |
| `test_full_phase_sequence` (Phase T) | 1,000,000 | 100,000 | Reduce dt or increase current to speed up transition |
| Python Sod cross-backend | 3000 | 1000 | With larger dt or shorter t_end |
| Python Brio-Wu cross-backend | 5000 | 2000 | With larger dt or shorter t_end |

### 3.6 Numba JIT Warmup Analysis

**Status**: All 90 `@njit` functions use `cache=True`. After first compilation, bytecode is cached in `__pycache__`. No warmup penalty on subsequent runs.

**When cache is invalid**: After any source code change to the cached function, Numba recompiles. This is unavoidable.

**AOT compilation**: Not recommended. Numba's AOT compilation is fragile, platform-specific, and doesn't work with `prange`. The cache mechanism is sufficient.

**Pre-warming fixture**: Could add a `conftest.py` fixture that imports all Numba modules:

```python
@pytest.fixture(scope="session", autouse=True)
def warm_numba_cache():
    """Import Numba modules to trigger cache check (not recompilation)."""
    import dpf.fluid.mhd_solver  # noqa: F401
    import dpf.radiation.line_radiation  # noqa: F401
    import dpf.collision.spitzer  # noqa: F401
```

This would front-load the ~0.5-1s import time rather than spreading it across test discovery. Marginal benefit.

### 3.7 Other Speed Strategies

#### Shared Fixtures

Several test files create `MetalMHDSolver` instances in every test. Module-scoped fixtures would help:

```python
@pytest.fixture(scope="module")
def metal_solver_16():
    """Shared 16^3 Metal solver (avoids repeated MPS device init)."""
    return MetalMHDSolver(grid_shape=(16, 16, 16), dx=0.01, gamma=5.0/3.0, device="mps")
```

MPS device initialization takes ~0.5-1s. With 31 slow tests in `test_metal_production.py`, this saves ~15s.

#### MLX for WALRUS Inference

MLX is documented as faster than MPS for Apple Silicon inference (native Metal, lower overhead). If WALRUS inference can run on MLX:
- Expected speedup over MPS: ~1.5-2x
- Expected speedup over CPU: ~2.5-3x
- Status: `mlx_surrogate.py` exists but is not connected to the test suite

#### Subprocess Pooling (Athena++)

Athena++ tests launch separate subprocesses. No optimization needed — subprocess startup (~1s) is negligible compared to solver runtime (~5-10s). Only 4 tests use subprocess mode.

#### Un-marking False Slow Tests

Several tests are marked `@pytest.mark.slow` but likely run in <1s:
- `test_solver_creation` (just instantiates solver)
- `test_solver_compute_dt` (single CFL calculation)
- `test_solver_coupling_interface` (1 step + coupling read)
- `test_engine_backend_metal` (engine creation, no stepping)
- `test_float32_riemann_stability` (single HLL call on 16-element 1D array)
- `test_phase_q_transport.test_hall_whistler_dispersion` (currently just `pytest.skip()`)

**Estimated: 6 tests could be un-marked, reducing the slow test count to 133.**

---

## 4. Implementation Roadmap

### Phase 1: Quick Wins (Est. 1-2 hours, saves ~40%)

1. **Un-mark 6 false-slow tests** → 139 → 133 slow tests
2. **Reduce grid sizes** for 6 stencil tests (16^3 → 8^3) → ~3-5x faster per test
3. **Reduce step counts** for 5 long-run tests (300→100, 200→100, etc.) → ~2-3x faster
4. **Add module-scoped `MetalMHDSolver` fixture** to `test_metal_production.py` → save ~15s

### Phase 2: WALRUS Optimization (Est. 2-3 hours, saves ~15%)

5. **Switch WALRUS test fixture to `device="mps"`** → 1.57x faster inference
6. **Cache additional WALRUS predictions** as module-scoped fixtures → save 2 forward passes (~116s)
7. **Investigate MLX inference path** for additional speedup

### Phase 3: Parallelization (Est. 3-4 hours, saves ~30%)

8. **Add `@pytest.mark.xdist_group("metal")` to all Metal tests**
9. **Configure pytest-xdist** with 4 workers and `--dist loadgroup`
10. **Test memory pressure** with parallel workers + MPS device

### Phase 4: Deep Optimization (Est. 4-6 hours, saves ~10%)

11. **Reduce `test_full_phase_sequence` iterations** (1M → 100K with faster transition)
12. **Reduce Python cross-backend step counts** (3000→1000, 5000→2000)
13. **Profile and optimize Lee model ODE** for calibration tests

### Estimated Total Savings

| Phase | Wall-clock Before | Wall-clock After | Reduction |
|-------|------------------|-----------------|-----------|
| Baseline | ~30 min | — | — |
| Phase 1 | ~30 min | ~20 min | 33% |
| Phase 2 | ~20 min | ~17 min | 15% |
| Phase 3 | ~17 min | ~10 min | 41% |
| Phase 4 | ~10 min | ~8 min | 20% |
| **Total** | **~30 min** | **~8 min** | **~73%** |

---

## 5. Risk Assessment

### Accuracy Trade-offs

| Optimization | Risk | Mitigation |
|-------------|------|-----------|
| Grid reduction (16^3→8^3) | Lose boundary effects, reduced statistical significance | Only for tests that don't check convergence order |
| Step reduction (300→100) | Miss late-onset instability | Add assertion on per-step drift to catch exponential growth |
| WALRUS MPS inference | Float32 vs float64 for normalization | WALRUS already uses float32 internally; no accuracy change |
| Parallelization | Race conditions on MPS device | Serialize Metal tests with xdist_group |

### Resource Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| Single MPS device | Metal tests cannot run in parallel | Use xdist_group to serialize |
| 36GB unified memory | Max 2 WALRUS instances in parallel | Limit to 1 WALRUS worker |
| Numba cache invalidation | Source changes trigger recompilation | Unavoidable; accept ~5s per module on first run |
| Athena++ binary availability | 4 tests skip without binary | Acceptable (binary not always built) |

### Tests That Must NOT Be Optimized

These tests are specifically designed for their current parameters:

1. **Convergence order tests** (3 resolutions): Need the resolution range for order estimation
2. **Float32 vs float64 comparison**: Need both precision modes
3. **Cross-backend parity**: Must run both backends for comparison
4. **Energy conservation long-runs**: Step count directly tests conservation drift

---

## 6. Appendix: Test Timing Estimates

Rough per-test timing on M3 Pro (from Metal benchmark data and memory/performance.md):

| Operation | Grid | Time per step |
|----------|------|--------------|
| Metal MHD step (MPS float32) | 16^3 | ~5-10ms |
| Metal MHD step (MPS float32) | 32×16×16 | ~15-25ms |
| Metal MHD step (CPU float64) | 16^3 | ~20-40ms |
| Python MHD step (Numba, warm) | 16^3 | ~50-100ms |
| Python MHD step (Numba, warm) | 64×4×4 | ~30-60ms |
| WALRUS forward pass (CPU) | 16^3 | ~58,000ms |
| WALRUS forward pass (MPS) | 16^3 | ~38,000ms |
| Lee model full run | N/A | ~200-500ms |
| RLC circuit step | N/A | ~0.01ms |
| Athena++ subprocess (Sod 256) | 256×1×1 | ~5,000-10,000ms total |

**Heaviest tests by estimated time**:

1. `test_walrus_deterministic` / `continuous` / `shock_propagates`: ~58s each (CPU)
2. `test_300_step_energy_drift`: ~300 × 10ms = ~3s (MPS) or ~12s (CPU float64)
3. `test_200_step_stability`: ~200 × 10ms = ~2s
4. `test_full_phase_sequence`: ~1M × 0.001ms = ~1s (pure Python loop, very fast per step but many iterations)
5. `test_cross_backend_brio_wu`: ~5000 × 50ms = ~250s (Python) + ~10s (Athena++)
6. `test_cross_backend_sod_shock`: ~3000 × 50ms = ~150s (Python) + ~10s (Athena++)

**The two Python cross-backend tests in `test_verification_comprehensive.py` are the 2nd and 3rd heaviest tests after WALRUS**, with combined ~400s. Reducing Python step counts (3000→1000, 5000→2000) or using a shorter `t_end` would save ~250s.
