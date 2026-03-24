# Phase B Research: Clean-Room MLX MHD Solver

**Date**: 2026-03-24
**Author**: dpf-engine-architect (Cortana)
**Status**: Research complete -- ready for implementation
**Scope**: MLX framework assessment, prototype repo analysis, sprint plan, risk register

---

## 1. MLX Framework Assessment (as of March 2026)

### 1.1 Version and Stability

MLX is at **v0.31.1** (released 2026-03-12). The framework has been under continuous
development with ~15 releases since v0.29 (October 2024). Key observations:

- **Core array API is stable.** NumPy-compatible operations (`mx.zeros`, `mx.sum`,
  `mx.sqrt`, slicing, broadcasting) have not changed since v0.1. Safe to depend on.
- **`mx.fast.metal_kernel()` is stable.** Introduced via PR #1325, documented since v0.30.0,
  with no breaking changes through v0.31.1. The API accepts `name`, `input_names`,
  `output_names`, `source` (MSL body), `ensure_row_contiguous`, `atomic_outputs`, and
  `init_value`. Kernel invocation takes `grid`, `threadgroup`, `template`, `output_shapes`,
  `output_dtypes`, and `verbose` parameters.
- **`mx.compile()` fuses elementwise operations** into single Metal kernels. A `gelu`
  example achieves 5x speedup on M1 Max. Limitations: compiled functions must be pure
  (no side effects), control-flow-dependent shapes fail with `shapeless=True`, and
  recompilation occurs when input shapes/types change.
- **JACCL backend** (v0.30.1+) enables RDMA over Thunderbolt for multi-device training.
  Not relevant for single-machine DPF but shows Apple is investing in the framework.
- **CUDA backend** added in v0.30+. MLX now runs on NVIDIA GPUs. We only care about
  Metal, but this shows the project is not Apple-Silicon-only anymore.

**Recommendation**: Pin to `mlx>=0.30.0,<1.0` in requirements. The custom kernel API is
mature enough for production use.

### 1.2 Custom Metal Kernel Capabilities

The `mx.fast.metal_kernel()` API provides:

| Feature | Detail |
|---------|--------|
| Input arrays | Automatically passed as `device const float*` with shape/stride metadata |
| Output arrays | Pre-allocated by caller, passed as `device float*` |
| Thread indexing | `[[thread_position_in_grid]]` auto-injected when detected in source |
| Template params | Type substitution (`T` -> `float`) via `[("T", mx.float32)]` |
| Grid/threadgroup | Metal `dispatchThreads` semantics; user specifies both |
| JIT compilation | Kernels compiled once, cached for reuse |
| Row contiguous | `ensure_row_contiguous=True` (default) copies inputs if needed |
| Atomic outputs | `atomic_outputs=True` for concurrent writes |
| Init values | `init_value` pre-fills outputs before kernel launch |
| Debugging | `verbose=True` prints generated MSL with full function signature |

**Key limitation**: No shared threadgroup memory (`threadgroup float[]`) support in the
Python API. The `metal_kernel` only generates a flat kernel signature. For stencil
operations that need shared memory tiling, the workaround is to read from global memory
with the understanding that Apple Silicon's unified memory has low latency (no PCIe hop).
For our grid sizes (128x512 = 65K cells), this is acceptable.

**Limitation for stencils**: No `F.pad` equivalent in MLX. We must write a custom Metal
kernel for ghost cell padding, which is exactly what the prototype repo provides.

### 1.3 mx.compile() for MHD Workloads

`mx.compile()` can fuse:
- Elementwise arithmetic chains (pressure recovery, floor clamping, source terms)
- Broadcast operations
- Simple reductions along axes

`mx.compile()` **cannot** fuse:
- Stencil operations with neighbor access (reconstruction, flux divergence)
- Custom Metal kernels (they are already compiled)
- Operations with data-dependent control flow
- Functions with side effects

**Strategy for Phase B**: Use `mx.compile()` to wrap the SSP-RK3 outer loop and
elementwise chains (conserved-to-primitive, pressure recovery, entropy sync). Use custom
Metal kernels for the 3 compute-intensive stencil operations (ghost pad, HLLD flux,
geometric sources). Use standard MLX ops for WENO5-Z reconstruction (vectorizable as
array slicing + arithmetic).

### 1.4 MLX vs PyTorch MPS: Performance Reality

Based on multiple independent benchmarks (TristanBilot/mlx-benchmark, LucasSte/MLX-vs-Pytorch,
arXiv:2510.18921):

| Workload | MLX GPU | PyTorch MPS | Winner |
|----------|---------|-------------|--------|
| Elementwise ops | ~1.2x faster | baseline | MLX |
| Matmul (large) | ~0.9x | ~1.1x | MPS (slightly) |
| Sort | 2-3x faster | baseline | MLX |
| Memory transfer | Zero-copy | Copy to GPU-visible | MLX |
| Training loops | ~1.5-2x slower | baseline | MPS |
| Inference | ~1.3x faster | baseline | MLX |

**Key finding**: MLX's advantage is in zero-copy unified memory access and elementwise
fusion. PyTorch MPS has better matmul (uses MPSGraph's tuned GEMM). For MHD stencil
operations, which are bandwidth-bound elementwise + neighbor access, MLX is the better fit.

A 2025 benchmark on M3 Pro found PyTorch 2.6 MPS slightly faster than MLX for some
workloads. This underscores the need to benchmark our specific stencil patterns early in
implementation.

### 1.5 No Existing MLX MHD/PDE Solvers

Web search found zero published MLX implementations for:
- MHD solvers
- CFD/finite-volume PDE solvers
- Stencil-based scientific computing

The closest related work:
- **ZMLX** (github.com/Hmbown/ZMLX): Triton-style kernel toolkit for MLX. Provides
  primitives for custom fusions but no PDE solver.
- **arXiv:2510.18921**: Benchmarks transformer inference on Apple Silicon, not
  scientific computing.

**Implication**: We are the first MLX MHD solver. No reference implementations exist to
compare against. The prototype repo (section 2) is the only prior art.

---

## 2. Prototype Repo Analysis: `longanisainhertaco/mlx_mhd`

### 2.1 Repository Structure

```
mlx_mhd/
  __init__.py      (105 LOC) -- public API exports
  kernels.py       (964 LOC) -- Metal kernels + NumPy reference implementations
  solver.py       (1082 LOC) -- full MHD solver: WENO5-Z, HLLD, SSP-RK3, CT, circuit
  driver.py         (62 LOC) -- CLI driver for PF-1000 simulation
mhd_kernels.py     (871 LOC) -- standalone Metal kernel definitions + wrappers
mhd_reference.py   (433 LOC) -- NumPy reference for kernel validation
test_mhd_kernels.py(425 LOC) -- kernel unit tests
test_solver.py     (256 LOC) -- solver integration tests
pf1000_driver.py    (17 LOC) -- PF-1000 convenience script
```

**Total**: ~4,320 LOC (excluding tests: ~3,215 LOC)

### 2.2 What the Prototype Implements

The prototype is a **complete NumPy-first cylindrical MHD solver** with 3 custom Metal
kernels. It implements exactly the architecture described in METAL_V2_SPEC.md:

| Component | Status | Quality |
|-----------|--------|---------|
| 10-component state vector (rho, mom, E, Srho, B, Ee) | Complete | Matches spec exactly |
| WENO5-Z reconstruction | Complete (NumPy) | FV coefficients (Jiang-Shu), correct for cell averages |
| HLLD Riemann solver | Complete (NumPy + Metal) | Full 4-intermediate-state, entropy tracer passthrough |
| SSP-RK3 time integration | Complete | Correct Shu-Osher coefficients |
| Dual-energy pressure recovery | Complete | Entropy-based switching, cubic Hermite blend |
| Entropy resynchronization | Complete | Shock detection via div(v) + pressure jump |
| Constrained transport | Complete (lightweight) | Edge EMF, gradient-based |
| Implicit resistive diffusion | Complete | Thomas solver, operator-split |
| Cylindrical geometric sources | Complete (NumPy + Metal) | L'Hopital at axis |
| Ghost cell padding | Complete (Metal) | Reflecting/electrode BCs |
| Circuit coupling (RLC) | Complete | Density-weighted Lp, back-EMF, monotonicity |
| Spitzer resistivity | Complete | Temperature-dependent, capped |
| Bremsstrahlung radiation | Complete | Standard formula |
| PF-1000 initialization | Complete | Dense sheath, circuit params |
| Validation framework | Complete | 8 validation targets with ranges |

### 2.3 Metal Kernels Implemented

Three custom Metal kernels via `mx.fast.metal_kernel()`:

1. **Ghost Cell Padding** (~90 LOC MSL in `mhd_kernels.py`, ~65 LOC in `kernels.py`)
   - Grid: `(10, nr+2*ng, nz)` -- one thread per output cell per variable
   - Threadgroup: `(1, 8, 8)` = 64 threads
   - Inner BC: reflecting with sign flips for vr, Br, Btheta
   - Outer BC: `Btheta = mu0*I/(2*pi*r)`, zero-gradient others
   - Handles both conserved and primitive variable sign conventions

2. **HLLD Flux** (~300 LOC MSL in `mhd_kernels.py`)
   - Grid: `(10, nr, nz)` -- one thread per interface per variable
   - Full HLLD with Prim struct, fast magnetosonic speeds, 4 intermediate states
   - Entropy tracer (Srho) upwinded through contact wave
   - Electron energy (Ee) upwinded through contact wave
   - Direction parameter for radial vs axial sweeps
   - Lax-Friedrichs fallback for degenerate cases

3. **Cylindrical Geometric Sources** (~100 LOC MSL in `mhd_kernels.py`)
   - Grid: `(nr, nz)` -- one thread per cell
   - L'Hopital rule at axis (`r_min + 0.5*dr`)
   - Hoop stress, magnetic pressure, tension terms
   - Operates on primitive variables

### 2.4 What Needs Adaptation for DPF-Unified Integration

| Aspect | Prototype State | Needed for DPF-Unified |
|--------|----------------|----------------------|
| Framework | Pure NumPy solver + standalone Metal kernels | MLX-native solver using `mx.array` throughout |
| WENO5-Z | NumPy loop over interfaces | MLX vectorized (slicing + arithmetic) or Metal kernel |
| Flux divergence | NumPy with grid geometry arrays | MLX with r-weighted face areas |
| Config | Frozen dataclasses | Integration with Pydantic SimConfig |
| Backend interface | Standalone `MHDSolver` class | Must implement `PlasmaSolverBase` |
| State dict convention | `(10, nr, nz)` array | Must translate to/from DPF state dict |
| Engine wiring | Own `run_pf1000_simulation` | Must plug into `engine.py` Strang splitting |
| Testing | Own test suite | Must pass existing cross-backend parity tests |
| CT implementation | Lightweight gradient-based | Need proper face-centered EMF (match Phase A) |
| Axial BCs | Ghost cells in both r and z | Matches spec requirements |

### 2.5 Code Reuse Assessment

| Component | Reuse Strategy | Estimated Adaptation |
|-----------|---------------|---------------------|
| Metal kernel MSL sources | **Direct reuse** | ~5% modification (buffer layout tweaks) |
| `SolverConfig` dataclass | **Adapt** to map from SimConfig | ~50 LOC wrapper |
| `CylindricalGrid` | **Replace** with existing GeometryConfig | Use DPF's grid system |
| `recover_pressure()` | **Port** NumPy -> MLX (mechanical) | ~30 LOC |
| `weno5z_left()` | **Port** NumPy -> MLX (vectorize) | ~50 LOC |
| `reconstruct_interfaces()` | **Redesign** -- loop-based, needs vectorization | ~100 LOC |
| `hlld_flux_numpy()` | **Reference only** -- Metal kernel used in production | Keep for testing |
| `entropy_resynchronize()` | **Port** NumPy -> MLX (mechanical) | ~40 LOC |
| SSP-RK3 stepper | **Port** + wrap in `mx.compile()` | ~60 LOC |
| Circuit coupling | **Reuse** DPF's existing `CircuitCoupler` | Adapter only |
| Thomas solver | **Port** or keep as NumPy (runs on CPU anyway) | ~50 LOC |
| PF-1000 initialization | **Reuse** existing DPF presets | Adapter only |

**Bottom line**: The prototype provides ~80% of the physics logic and all 3 Metal kernels.
The main work is (1) converting from NumPy to MLX arrays, (2) integrating with DPF's
config/engine/state-dict system, and (3) vectorizing the WENO5-Z reconstruction loop.

---

## 3. Existing DPF Metal Codebase (v1)

### 3.1 File Inventory

| File | LOC | Phase B Action |
|------|-----|---------------|
| `metal_solver.py` | 2,381 | Replace with MLX solver |
| `metal_transport.py` | 919 | Port Thomas solver; rest is PyTorch-specific |
| `metal_stencil.py` | 866 | Replace with Metal kernels from prototype |
| `metal_riemann.py` | 539 | Replace with MLX HLLD |
| `mlx_surrogate.py` | 492 | Keep as-is (WALRUS, unrelated) |
| `_riemann_solvers.py` | 414 | Reference for HLLD validation |
| `_riemann_reconstruction.py` | 391 | Reference for WENO5-Z validation |
| `device.py` | 336 | Simplify for MLX device detection |
| `_dual_energy.py` | 318 | Port switching logic to MLX |
| `_riemann_primitives.py` | 274 | Replace with MLX cons/prim conversion |
| `_riemann_nan_safety.py` | 84 | Port NaN guards to MLX |
| `_utils.py` | 41 | Keep general utilities |
| `_riemann_constants.py` | 26 | Keep constants |
| `__init__.py` | 15 | Update exports |

**Total v1 Metal code**: ~7,096 LOC (excluding `mlx_surrogate.py`)
**Estimated v2 replacement**: ~4,200 LOC (cleaner architecture, less framework boilerplate)

### 3.2 Existing Metal Shader Files

```
src/dpf/metal/kernels/
  common.metal
  flux_divergence.metal
  hll_flux.metal
  mhd_sweep_x.metal
  plm_reconstruct_x.metal
  source_terms.metal
  time_integrator.metal
```

These are PyTorch-era `.metal` files, not `mx.fast.metal_kernel()` sources. They cannot
be reused directly. The prototype repo's MSL strings are the correct format.

---

## 4. Sprint Plan (8-10 Weeks, 5 Sprints)

### Sprint 0: Foundation (Week 3-4)

**Goal**: MLX device layer + state dict bridge + all 3 Metal kernels integrated and tested.

| Work Unit | File | LOC | Description |
|-----------|------|-----|-------------|
| WU-0.1 | `src/dpf/metal/mlx_device.py` | ~120 | MLX device detection, stream management, dtype helpers |
| WU-0.2 | `src/dpf/metal/mlx_state.py` | ~150 | State dict <-> `(10, nr, nz)` mx.array conversion |
| WU-0.3 | `src/dpf/metal/mlx_kernels.py` | ~400 | Port 3 Metal kernels from prototype with Python wrappers |
| WU-0.4 | `src/dpf/metal/mlx_grid.py` | ~100 | Cylindrical grid geometry (face areas, volumes) as mx.arrays |
| WU-0.5 | `tests/test_mlx_kernels.py` | ~300 | Unit tests: each kernel vs NumPy reference |

**Dependencies**: None (standalone foundation).
**Exit gate**: All 3 kernels produce output matching NumPy reference to < 1e-5 relative error.
**Risk**: MLX import failures on CI (no Apple Silicon). Mitigation: `pytest.importorskip("mlx")`.

### Sprint 1: Reconstruction + Riemann (Week 5-6)

**Goal**: WENO5-Z + HLLD + flux divergence producing correct 1D shock solutions.

| Work Unit | File | LOC | Description |
|-----------|------|-----|-------------|
| WU-1.1 | `src/dpf/metal/mlx_reconstruction.py` | ~250 | WENO5-Z in MLX (vectorized, no Python loops) |
| WU-1.2 | `src/dpf/metal/mlx_riemann.py` | ~200 | HLLD wrapper: reconstruction -> Metal kernel -> flux divergence |
| WU-1.3 | `src/dpf/metal/mlx_primitives.py` | ~180 | Cons/prim conversion, dual-energy pressure recovery in MLX |
| WU-1.4 | `tests/test_mlx_reconstruction.py` | ~200 | WENO5-Z vs v1 `_riemann_reconstruction.py` |
| WU-1.5 | `tests/test_mlx_riemann.py` | ~200 | Sod + Brio-Wu 1D shock tubes |

**Dependencies**: Sprint 0 (kernels, grid, state conversion).
**Exit gate**: Sod L1(rho) < 0.02 at N=256. Brio-Wu no NaN, compound wave structure visible.
**Risk**: WENO5-Z vectorization in MLX may need creative slicing. The prototype uses Python
loops over interfaces. MLX version must use `mx.array` gather/scatter or rolling windows.

**Key design decision**: WENO5-Z can be implemented as pure MLX ops (slicing + arithmetic)
without a custom Metal kernel. The 5-point stencil is just 5 overlapping array slices, and
the nonlinear weight computation is elementwise. This avoids writing a 4th Metal kernel and
lets `mx.compile()` fuse the entire reconstruction.

### Sprint 2: Time Integration + Source Terms (Week 6-7)

**Goal**: SSP-RK3 stepping with geometric sources, entropy sync, and CT.

| Work Unit | File | LOC | Description |
|-----------|------|-----|-------------|
| WU-2.1 | `src/dpf/metal/mlx_timestepper.py` | ~200 | SSP-RK3 with CFL, entropy resync, floor enforcement |
| WU-2.2 | `src/dpf/metal/mlx_sources.py` | ~150 | Cylindrical source wrapper + ohmic/bremsstrahlung |
| WU-2.3 | `src/dpf/metal/mlx_ct.py` | ~120 | Constrained transport for face-centered Br, Bz |
| WU-2.4 | `src/dpf/metal/mlx_transport.py` | ~150 | Implicit resistive diffusion (Thomas solver, CPU float64) |
| WU-2.5 | `tests/test_mlx_timestepper.py` | ~250 | Uniform state preservation, energy conservation, diffusion convergence |

**Dependencies**: Sprint 1 (reconstruction, Riemann, primitives).
**Exit gate**: Uniform state preserved to 1e-6 over 100 steps. Diffusion convergence >= 1.9.
Energy drift < 1e-5 over 100 steps at beta << 1.

### Sprint 3: Solver Assembly + Engine Integration (Week 8-9)

**Goal**: Complete MLX solver implementing `PlasmaSolverBase`, wired into `engine.py`.

| Work Unit | File | LOC | Description |
|-----------|------|-----|-------------|
| WU-3.1 | `src/dpf/metal/mlx_solver.py` | ~500 | Main solver class, implements PlasmaSolverBase |
| WU-3.2 | `src/dpf/metal/mlx_circuit.py` | ~100 | Circuit coupling adapter (reuses DPF CircuitCoupler) |
| WU-3.3 | `src/dpf/engine.py` (modify) | ~50 | Add `backend="mlx"` to backend selection cascade |
| WU-3.4 | `src/dpf/metal/mlx_device.py` (update) | ~30 | Add MLX to backend availability detection |
| WU-3.5 | `tests/test_mlx_solver.py` | ~300 | Integration tests: Sod, Brio-Wu, linear waves |
| WU-3.6 | `tests/test_mlx_cross_backend.py` | ~200 | Cross-backend parity: MLX vs Python engine |

**Dependencies**: Sprint 2 (timestepper, sources, CT, transport).
**Exit gate**: Cross-backend Sod L1(rho) < 5% vs Python WENO5+HLLD. All existing engine
tests pass with `backend="mlx"`. Backend fallback cascade works.

### Sprint 4: DPF Validation + Performance (Week 10-12)

**Goal**: PF-1000 full discharge, multi-device validation, performance benchmarks.

| Work Unit | File | LOC | Description |
|-----------|------|-----|-------------|
| WU-4.1 | `tests/test_mlx_pf1000.py` | ~300 | PF-1000 full discharge: M1-M8 DoD criteria |
| WU-4.2 | `tests/test_mlx_multidevice.py` | ~150 | UNU-ICTP, NX2, POSEIDON validation |
| WU-4.3 | `src/dpf/benchmarks/mlx_benchmark.py` | ~200 | Grid scaling benchmarks: 64x256, 128x512, 256x1024 |
| WU-4.4 | Performance tuning | ~100 | `mx.compile()` wrapping, threadgroup optimization |
| WU-4.5 | `tests/test_mlx_acceptance.py` | ~200 | Final acceptance: all DoD M1-M8 + S1-S9 |

**Dependencies**: Sprint 3 (solver, engine integration).
**Exit gate**: PF-1000 I_peak < 10% of 1.2 MA. No negative pressure. No NaN. Full 5
phases complete. Performance > Athena++ at 128x512. All DoD M1-M8 pass.

---

## 5. File-by-File Implementation Order

### New Files (Phase B)

| # | File | LOC Est. | Sprint | Dependencies |
|---|------|----------|--------|-------------|
| 1 | `src/dpf/metal/mlx_device.py` | 120 | 0 | None |
| 2 | `src/dpf/metal/mlx_grid.py` | 100 | 0 | mlx_device |
| 3 | `src/dpf/metal/mlx_kernels.py` | 400 | 0 | mlx_device |
| 4 | `src/dpf/metal/mlx_state.py` | 150 | 0 | mlx_grid |
| 5 | `src/dpf/metal/mlx_primitives.py` | 180 | 1 | mlx_state, mlx_kernels |
| 6 | `src/dpf/metal/mlx_reconstruction.py` | 250 | 1 | mlx_primitives |
| 7 | `src/dpf/metal/mlx_riemann.py` | 200 | 1 | mlx_reconstruction, mlx_kernels |
| 8 | `src/dpf/metal/mlx_sources.py` | 150 | 2 | mlx_primitives, mlx_kernels |
| 9 | `src/dpf/metal/mlx_ct.py` | 120 | 2 | mlx_grid |
| 10 | `src/dpf/metal/mlx_transport.py` | 150 | 2 | mlx_grid |
| 11 | `src/dpf/metal/mlx_timestepper.py` | 200 | 2 | 6-10 (all above) |
| 12 | `src/dpf/metal/mlx_circuit.py` | 100 | 3 | mlx_state |
| 13 | `src/dpf/metal/mlx_solver.py` | 500 | 3 | 11, 12 |
| 14 | `src/dpf/benchmarks/mlx_benchmark.py` | 200 | 4 | mlx_solver |

**Total new code**: ~2,820 LOC (production) + ~1,900 LOC (tests) = ~4,720 LOC

### Modified Files

| File | Change | LOC Delta | Sprint |
|------|--------|-----------|--------|
| `src/dpf/engine.py` | Add `backend="mlx"` selection | +50 | 3 |
| `src/dpf/metal/device.py` | Add MLX availability check | +30 | 0 |
| `src/dpf/metal/__init__.py` | Update exports | +20 | 3 |

---

## 6. Dependency Graph

```
mlx_device (0)
  |
  +-- mlx_grid (0)
  |     |
  |     +-- mlx_state (0)
  |     |     |
  |     |     +-- mlx_primitives (1)
  |     |     |     |
  |     |     |     +-- mlx_reconstruction (1)
  |     |     |     |     |
  |     |     |     |     +-- mlx_riemann (1)
  |     |     |     |
  |     |     |     +-- mlx_sources (2)
  |     |     |
  |     |     +-- mlx_circuit (3)
  |     |
  |     +-- mlx_ct (2)
  |     |
  |     +-- mlx_transport (2)
  |
  +-- mlx_kernels (0)
        |
        +-- mlx_riemann (1)
        +-- mlx_sources (2)

mlx_timestepper (2) -- depends on: reconstruction, riemann, sources, ct, transport, primitives
mlx_solver (3) -- depends on: timestepper, circuit
engine.py (3) -- depends on: mlx_solver
```

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | WENO5-Z vectorization in MLX fails to achieve 5th-order convergence | Medium | High | Validate convergence order on smooth sinusoidal profile at 4 resolutions in Sprint 1. If <4th order, fall back to prototype's loop-based approach with `mx.eval()` per interface batch. |
| R2 | Metal kernel debugging is opaque (no printf, no breakpoints) | Medium | Medium | Use `verbose=True` to inspect generated MSL. Unit-test every kernel against NumPy reference. Use Xcode Metal GPU debugger for crash diagnosis. |
| R3 | `mx.compile()` does not fuse SSP-RK3 stages effectively | Medium | Low | Benchmark with and without compile. If no speedup, drop compile for the outer loop -- the Metal kernels dominate compute anyway. |
| R4 | Grid too small for GPU advantage (< 128x512) | HIGH | Medium | Phase B is explicitly for production grids. Small grids use Python or Athena++ backend. Add automatic backend selection threshold in engine.py. |
| R5 | Float32 entropy switching produces artifacts at electrode | Medium | High | Tune eta1/eta2 on Brio-Wu and electrode test in Sprint 1-2. Phase A PyTorch patch validates the physics first. Prototype already implements the correct switching. |
| R6 | MLX API breaking change in v0.32+ | Low | Low | Pin version. Core array API stable since inception. Custom kernel API stable since v0.30. |
| R7 | Prototype kernels have bugs not caught by their tests | Medium | Medium | Every kernel gets independent validation against DPF v1 PyTorch HLLD and NumPy reference in Sprint 0. Do not trust prototype tests alone. |
| R8 | Circuit coupling oscillates due to MLX float32 | Low | High | Circuit solver runs in CPU float64 (explicit in spec). Only Lp/R_plasma scalars extracted from MLX arrays via `.item()`. |
| R9 | CT implementation insufficient for div(B) at production resolution | Medium | Medium | Prototype uses gradient-based CT. If div(B) > 1e-6 at 128x512, upgrade to proper face-centered EMF CT matching Phase A implementation. |
| R10 | Performance does not beat Athena++ at 128x512 | Medium | Medium | This is a "should-have" (S9), not a "must-have". MLX value is in zero-copy integration with the Python ecosystem, not raw throughput vs compiled C++. |

---

## 8. What Can Be Ported Mechanically vs. What Needs Redesign

### Mechanical Ports (NumPy -> MLX, ~1:1 translation)

These functions translate directly because MLX's API mirrors NumPy:

- `recover_pressure()` -- replace `np.maximum` with `mx.maximum`, etc.
- `primitive_to_conserved()` / `conserved_to_primitive()` -- array indexing identical
- `enforce_physical_floors()` -- elementwise clamps
- `entropy_resynchronize()` -- masking + elementwise ops
- `estimate_timestep()` -- reductions + arithmetic
- `divergence_velocity()` -- `np.gradient` -> `mx.array` finite differences
- `divergence_b()` -- same pattern
- `compute_current_density()` -- `np.gradient` -> finite differences
- `compute_source_terms()` -- elementwise with function calls
- `spitzer_resistivity()` -- elementwise

**Estimated effort**: ~2 days. Pure search-and-replace with `np.` -> `mx.` plus `.astype(np.float64)` -> `.astype(mx.float32)` (GPU stays float32).

### Needs Redesign

1. **WENO5-Z reconstruction** (`reconstruct_interfaces`): The prototype loops over `nr+1`
   interfaces in Python. In MLX, this must be vectorized using array slicing to extract the
   5-point stencil windows as `(10, 5, nr+1, nz)` tensors, then compute all weights and
   candidate polynomials in parallel. This is the single hardest piece of the port.

2. **Flux divergence with r-weighting**: The prototype uses `grid.radial_face_areas` and
   `grid.cell_volumes` as NumPy arrays. These need to be pre-computed as `mx.array` and
   cached on the grid object. The divergence itself is just array slicing + multiplication.

3. **`np.gradient` replacement**: MLX has no `np.gradient` equivalent. Replace with explicit
   finite differences: `(f[2:] - f[:-2]) / (2*dx)` for interior, one-sided at boundaries.
   Affects: `divergence_velocity`, `divergence_b`, `compute_current_density`, CT update.

4. **Thomas solver**: Must stay on CPU in float64 (tridiagonal solve is inherently sequential).
   Extract B-field components from MLX to NumPy, solve, put back. The `.item()` extraction
   per cell is too slow for full grids -- use `np.array(mx_array)` for zero-copy bulk transfer.

5. **State dict bridge**: DPF state dict uses separate keys (`rho`, `velocity`, `B`, etc.)
   while the solver uses `(10, nr, nz)` packed array. Need fast pack/unpack functions.

6. **Engine integration**: `PlasmaSolverBase` interface requires `step(state_dict, dt) ->
   state_dict`. The MLX solver works with packed `mx.array`. The adapter must handle the
   conversion without unnecessary copies (leverage zero-copy MLX<->NumPy).

---

## 9. Testing Strategy

### Sprint 0: Kernel Validation
- Each Metal kernel tested against NumPy reference at 3 grid sizes (16x32, 64x128, 128x512)
- Relative error < 1e-5 (float32 arithmetic)
- Edge cases: axis cells (r=0), cathode cells, zero current, large current
- Performance: kernel launch time measured, must complete in < 1ms for 128x512

### Sprint 1: Numerical Accuracy
- WENO5-Z convergence on smooth sin profile: 32, 64, 128, 256 cells. Rate >= 4.0
- Sod shock tube: L1(rho) < 0.02 at N=256, float32
- Brio-Wu: 7 waves visible, no NaN, no negative pressure
- Cross-reference every reconstruction output against `_riemann_reconstruction.py`

### Sprint 2: Conservation + Stability
- Uniform state preservation: 100 steps, max change < 1e-6
- Energy conservation: beta=0.001, 100 steps, |dE/E| < 1e-5
- Diffusion convergence: rate >= 1.9
- Entropy tracer: K constant on uniform state, increases at shocks
- Float32 pressure positivity: electrode conditions (ME/p=10^6), 50 steps

### Sprint 3: Integration
- All existing `test_metal_production.py` tests pass (adapted for MLX backend)
- Cross-backend parity: MLX vs Python engine, Sod L1(rho) < 5%
- Backend fallback: MLX -> Python when MLX unavailable
- State dict round-trip: pack -> MLX step -> unpack preserves keys/shapes

### Sprint 4: Validation
- PF-1000 DoD M1-M8 (section 3.1 of METAL_V2_DOD.md)
- PF-1000 DoD S1-S9 (section 3.2)
- Multi-device: UNU-ICTP, NX2 complete without crash
- Performance: wall-clock measured at 64x256, 128x512, 256x1024

---

## 10. Hardware Requirements and Performance Targets

### Development Hardware
- Apple Silicon Mac with Metal GPU (M1/M2/M3/M4)
- Minimum 16GB unified memory
- Recommended: M3 Pro 36GB (current dev machine)

### Performance Targets

| Grid | Python (NumPy) | Python (Numba) | MLX (Metal) | Athena++ | Target |
|------|---------------|----------------|-------------|----------|--------|
| 64x256 | ~2 s/step | ~0.1 s/step | **< 0.05 s** | ~0.01 s | 2x Numba |
| 128x512 | ~8 s/step | ~0.4 s/step | **< 0.08 s** | ~0.04 s | Match Athena++ |
| 256x1024 | ~32 s/step | ~1.6 s/step | **< 0.2 s** | ~0.15 s | Within 1.5x |

**Rationale**: MLX on M3 Pro has ~4 TFLOPS float32 GPU throughput. The 128x512 grid has
65K cells x 10 variables x ~100 FLOPs/cell/stage x 3 stages = ~195 MFLOP per timestep.
At 4 TFLOPS, compute time is ~50 microseconds. Memory bandwidth (150 GB/s) limits to
~2.6 ms for reading/writing the state array 3 times. Adding kernel launch overhead (~0.5 ms
per kernel x 6 kernels per stage x 3 stages = ~9 ms), total should be ~15-20 ms per step.

The 0.08 s target at 128x512 is conservative (4x the theoretical minimum) to account for
Python overhead, MLX scheduling, and non-fused operations.

### Memory Requirements

| Grid | State Array | Padded + Intermediates | Total |
|------|------------|----------------------|-------|
| 64x256 | 0.6 MB | ~10 MB | ~15 MB |
| 128x512 | 2.5 MB | ~40 MB | ~60 MB |
| 256x1024 | 10 MB | ~160 MB | ~250 MB |
| 512x2048 | 40 MB | ~640 MB | ~1 GB |

All fit comfortably in 36GB unified memory. The 512x2048 grid is feasible for production
runs if needed.

---

## 11. Open Questions for Implementation

1. **WENO5-Z vectorization strategy**: Should we use `mx.as_strided` (if available) to
   create stencil windows, or explicit index arithmetic with `mx.take`? Need to prototype
   both and benchmark.

2. **CT implementation fidelity**: The prototype's gradient-based CT may not maintain div(B)
   to machine precision at 128x512. Should we implement proper face-centered EMF CT from
   the start (matching Athena++/FLASH) or start with gradient-based and upgrade if needed?
   **Recommendation**: Start with gradient-based (Sprint 2), measure div(B), upgrade in
   Sprint 4 if needed.

3. **`mx.compile()` scope**: Should we compile the entire SSP-RK3 step function, or just
   the elementwise chains? Compiling the outer function may fail due to Metal kernel calls
   inside. **Recommendation**: Compile individual elementwise functions (pressure recovery,
   entropy sync, floor enforcement) but not the RK3 outer loop.

4. **Thread group optimization**: The prototype uses (1,8,8)=64 threads. M3 Pro has SIMD
   width 32 with 14 GPU cores. Should we use (16,8)=128 for 2D kernels?
   **Recommendation**: Use prototype defaults initially, profile with Xcode Instruments
   in Sprint 4, optimize then.

5. **Backward compatibility**: Should `backend="metal"` still use PyTorch MPS (Phase A)?
   **Recommendation**: Yes. Add `backend="mlx"` as the new backend. Keep `metal` for PyTorch.
   Users can switch explicitly. Engine auto-resolution: `athenak > athena > mlx > metal > python`.

---

## 12. Comparison: Prototype vs. DPF v1 Metal vs. Phase B Target

| Feature | DPF v1 (PyTorch) | Prototype (NumPy+Metal) | Phase B (MLX) |
|---------|-----------------|------------------------|---------------|
| Framework | PyTorch MPS | NumPy + mlx kernels | MLX native |
| State vector | 9 components | 10 (+ Srho) | 10 (+ Srho) |
| Reconstruction | WENO5/PLM (PyTorch) | WENO5-Z (NumPy loops) | WENO5-Z (MLX vectorized) |
| Riemann solver | HLLD (PyTorch) | HLLD (Metal kernel) | HLLD (Metal kernel) |
| Time integrator | SSP-RK2/RK3 | SSP-RK3 | SSP-RK3 |
| Dual energy | Phase A patch | Full implementation | Full implementation |
| CT | PyTorch EMF | Gradient-based | Gradient-based (upgradeable) |
| Resistive MHD | Implicit Thomas | Implicit Thomas | Implicit Thomas (CPU f64) |
| Circuit coupling | CircuitCoupler (buggy back-EMF) | Own RLC (correct) | Via DPF CircuitCoupler (fixed) |
| Memory transfer | Copy (MPS) | Zero-copy (MLX) | Zero-copy (MLX) |
| Precision | float32 (corrupted) | float32 (entropy-safe) | float32 (entropy-safe) |
| Coordinates | 2D Cartesian | 2D Cylindrical | 2D Cylindrical |
| Geometry sources | Basic | L'Hopital at axis | L'Hopital at axis (Metal kernel) |
| Lines of code | ~7,100 | ~3,200 | ~2,800 (est.) |

---

## 13. Summary and Recommendation

Phase B is ready for implementation. The prototype repo provides a complete, tested
NumPy-first solver with 3 validated Metal kernels that implement the exact architecture
specified in METAL_V2_SPEC.md. The work is primarily an integration and optimization
effort, not a from-scratch design.

**Critical path**: Sprint 0 (kernels) -> Sprint 1 (reconstruction/Riemann) -> Sprint 2
(time integration) -> Sprint 3 (solver/engine). Each sprint has a clear exit gate.

**Highest risk item**: WENO5-Z vectorization in MLX (R1). This is the only component that
requires genuine algorithmic redesign rather than mechanical porting. If it fails, the
fallback is a batched Python loop with MLX arrays, which will be slower but correct.

**Expected outcome**: A production MLX MHD solver at ~2,800 LOC that matches Phase A
physics accuracy, runs on Metal GPU for grids >= 128x512, and integrates seamlessly with
DPF-Unified's engine and configuration system.

---

## References

- [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX Compilation Documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX GitHub Releases](https://github.com/ml-explore/mlx/releases)
- [MLX Benchmark (TristanBilot)](https://github.com/TristanBilot/mlx-benchmark)
- [MLX vs PyTorch Benchmarks (LucasSte)](https://github.com/LucasSte/MLX-vs-Pytorch)
- [Benchmarking On-Device ML on Apple Silicon (arXiv:2510.18921)](https://arxiv.org/html/2510.18921v1)
- [WWDC25: Get started with MLX for Apple Silicon](https://developer.apple.com/videos/play/wwdc2025/315/)
- [ZMLX Triton-style toolkit for MLX](https://github.com/Hmbown/ZMLX)
- [MLX metal_kernel API Reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)
- [matmul MPS vs MLX comparison (2025)](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)
- Prototype repo: `github.com/longanisainhertaco/mlx_mhd` (cloned and analyzed locally)
