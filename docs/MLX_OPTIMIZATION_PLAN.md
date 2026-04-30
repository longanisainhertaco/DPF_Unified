# MLX MHD Solver Optimization Plan

## 1. Executive Summary

The MLX solver (8,235 LOC across 19 modules) is the first-ever MLX PDE solver, achieving
2.68x speedup over PyTorch MPS on M3 Pro. Current PF-1000 64x128 full discharge takes ~8 min.
Target: <2 min (4x speedup). This plan identifies 23 optimization opportunities across 3 phases.

The critical finding: **the Riemann solvers (HLL/HLLD/HLLS) run on CPU via NumPy in float64**,
constituting the single largest bottleneck. The `_hll_flux()` and `_hlls_flux()` functions in
`mlx_riemann.py` (lines 223-344, 86-215) convert mx.array to np.ndarray, compute in float64,
and convert back. This CPU round-trip happens **6 times per SSP-RK3 step** (2 dims x 3 stages).
Eliminating this one bottleneck would yield an estimated 2-3x speedup alone.

Secondary bottlenecks: operator-split transport modules (resistive diffusion, thermal conduction,
viscosity) use Python-loop Thomas solvers on CPU. The bremsstrahlung source term forces a
GPU-to-CPU-to-GPU round-trip for float64 computation.

mx.compile() is already applied to the right places (cons_to_prim, pressure recovery, stage
post-processing, WENO5-Z kernel). The HLLD Metal kernel exists and works on GPU but is only
used for `riemann="hlld"` in float32 mode.

## 2. Current Performance Profile

### Time-per-step Breakdown (estimated from code structure, 64x128 grid)

| Operation | GPU/CPU | Calls/Step | Est. % of Time | Notes |
|-----------|---------|------------|-----------------|-------|
| HLL/HLLS Riemann (NumPy f64) | CPU | 6 | ~45-55% | `np.asarray()` round-trip x6 |
| WENO5-Z reconstruction | GPU | 6 | ~10-15% | mx.compile() already applied |
| Stage post-processing | GPU | 3 | ~5-8% | Fused and compiled |
| Geometric sources | GPU | 3 | ~3-5% | Metal kernel |
| Ghost cell padding | Mixed | 1 | ~5-8% | Metal kernel + NumPy fixups |
| Resistive diffusion (Thomas) | CPU | 2 | ~8-12% | Python loops, Strang split |
| State pack/unpack | CPU | 2 | ~3-5% | np.asarray conversions |
| CFL computation | Mixed | 1 | ~2-3% | GPU compute, CPU transfer |
| Dedner/Powell div(B) | GPU | 1 | ~1-2% | Gradient computations |

### Existing mx.compile() Coverage

| Function | File:Line | Compiled | Speedup |
|----------|-----------|----------|---------|
| `_cons_to_prim_impl` | mlx_primitives.py:75 | Yes | 1.21-1.30x |
| `_recover_pressure_impl` | mlx_primitives.py:94 | Yes | 1.21-1.30x |
| `_weno5z_left_biased` | mlx_reconstruction.py:183 | Yes | 1.21-1.30x |
| `_stage_post_impl` | mlx_timestepper.py:56 | Yes | 1.21-1.30x |

### Existing Metal Kernels (3)

| Kernel | File | Lines | Function |
|--------|------|-------|----------|
| Ghost cell padding | mlx_kernels.py:63-168 | 105 MSL | Electrode BC + energy fix |
| HLLD Riemann | mlx_kernels.py:354-607 | 253 MSL | Full 4-wave HLLD solver |
| Geometric source | mlx_kernels.py (later) | ~80 MSL | Cylindrical source terms |

## 3. Optimization Inventory (ranked by speedup/effort)

### Tier 1: Critical Path (expected 2.5-3.5x total)

| # | Optimization | File | Lines to Change | Expected Speedup | Effort |
|---|-------------|------|-----------------|------------------|--------|
| 1 | **Port HLL flux to MLX GPU** | mlx_riemann.py:223-344 | ~120 rewrite | 1.8-2.5x | Medium |
| 2 | **Port HLLS flux to MLX GPU** | mlx_riemann.py:86-215 | ~130 rewrite | 1.5-2.0x | Medium |
| 3 | **Eliminate NumPy fixups in ghost padding** | mlx_solver.py:325-367 | ~40 | 1.1-1.2x | Low |
| 4 | **Fuse PLM+HLL into single Metal kernel** | new module (TBD) | ~150 new MSL | 1.2-1.4x | High |

### Tier 2: Medium Impact (expected 1.3-1.8x additional)

| # | Optimization | File | Lines | Expected Speedup | Effort |
|---|-------------|------|-------|------------------|--------|
| 5 | Port Thomas solver to MLX parallel scan | mlx_transport.py:34-85 | ~80 | 1.1-1.3x | High |
| 6 | Bremsstrahlung in float32 with compensated sum | mlx_sources.py:446-455 | ~15 | 1.05x | Low |
| 7 | Batch mx.eval() calls (reduce sync points) | mlx_solver.py:660-724 | ~20 | 1.1-1.2x | Low |
| 8 | Pre-allocate dU_dt in mhd_rhs | mlx_riemann.py:527 | ~10 | 1.05x | Low |
| 9 | Fuse geometric source + flux divergence | mlx_timestepper.py + mlx_riemann.py | ~80 | 1.1x | Medium |
| 10 | Compile `_clamp_reconstructed` | mlx_riemann.py:60-78 | ~5 | 1.02x | Trivial |

### Tier 3: Deep Optimization (expected 1.1-1.3x additional)

| # | Optimization | File | Lines | Expected Speedup | Effort |
|---|-------------|------|-------|------------------|--------|
| 11 | Metal kernel for flux divergence | mlx_riemann.py | ~100 MSL | 1.1x | High |
| 12 | SIMD group reductions in HLLD kernel | mlx_kernels.py | ~30 MSL | 1.05x | Medium |
| 13 | Threadgroup shared memory for stencil ops | mlx_kernels.py | ~50 MSL | 1.1x | High |
| 14 | mx.vmap for column-wise Thomas solver | mlx_transport.py | ~40 | 1.15x | Medium |
| 15 | Overlap RHS computation with eval | mlx_timestepper.py | ~20 | 1.05x | Low |

## 4. Detailed Analysis of Top Optimizations

### OPT-1: Port HLL Flux to Pure MLX (Highest Impact)

**Current**: `_hll_flux()` at mlx_riemann.py:223 converts QL/QR to NumPy float64, computes
wavespeeds, physical fluxes, HLL combination, NaN fallback, then converts back to mx.array float32.

**Problem**: For 64x128 grid, each call processes ~8,000 interface points through:
- 2x `np.asarray()` GPU-to-CPU transfers
- ~40 NumPy array operations in float64
- 1x `mx.array()` CPU-to-GPU transfer
- Called 6x per SSP-RK3 step (2 dims x 3 stages)

**Fix**: Rewrite as pure mx.array operations. The Boris correction and wavespeed computation
are all elementwise -- perfectly suited for MLX. Key concern: float32 cancellation in pressure
recovery `p = (gamma-1)(E - KE - ME)`. Mitigation: use dual-energy pressure from entropy tracer
(already available in ISR slot) instead of E-KE-ME subtraction, exactly as HLLS does.

**Implementation sketch** (mlx_riemann.py):
```python
def _hll_flux_mlx(QL: mx.array, QR: mx.array, gamma: float, dim: int) -> mx.array:
    # All operations stay on GPU as mx.array
    rho_L = mx.maximum(QL[IDN], RHO_FLOOR)
    inv_rL = mx.reciprocal(rho_L)
    vn_L = QL[im_n] * inv_rL
    # ... wavespeeds, fluxes, HLL combination ...
    # NaN fallback: mx.where(mx.isnan(F_hll), F_LF, F_hll)
    return F_out
```

**Risk**: Float32 precision in wavespeed computation. Mitigated by Boris correction (already
caps wave speeds at 5e5 m/s) and dual-energy pressure recovery.

**LOC**: ~120 lines of MLX code replacing ~120 lines of NumPy code.

### OPT-2: Port HLLS Flux to Pure MLX

Same pattern as OPT-1 but for the entropy-based solver. HLLS already uses entropy-derived
pressure (no E-KE-ME cancellation), making it inherently float32-safe. This is the ideal
candidate for an all-GPU Riemann solver.

### OPT-3: Eliminate NumPy Fixups in Ghost Padding

**Current**: `_pad_electrode_ghost()` at mlx_solver.py:307 calls the Metal ghost kernel, then
immediately converts the result back to NumPy (`np.asarray(U_padded)` at line 325) for Python-loop
electrode B_theta injection and energy consistency fixups (lines 327-367). This negates the
Metal kernel's GPU advantage.

**Fix**: Extend the Metal ghost kernel to handle the SI-to-HL conversion and interior-cell
B_theta blending. Alternatively, write the fixup loop as vectorized MLX operations:
```python
# Vectorized outer ghost + interior electrode injection
for ig in range(ng):
    out_idx = ng + self.nr + ig
    # ... can be vectorized as slice operations on mx.array
```

### OPT-4: Fused PLM+HLL Metal Kernel

**Current**: A planning stub `mlx_fused_flux.py` previously documented this optimization
but only delegated to the separate PLM reconstruction + HLL flux path
[Deleted 2026-04-24, 48 LOC dead code — the fused kernel was never implemented].
The current pipeline still materializes intermediate UL/UR arrays
(10 x nr x nz each, ~2.6 MB at 128x256) and then immediately consumes them.
A new fused kernel would need to be implemented from scratch in `mlx_kernels.py`
or a fresh module.

**Fix**: Single Metal kernel that reads Q, applies PLM reconstruction per-cell, and immediately
computes HLL flux without materializing UL/UR. Saves ~5 MB memory bandwidth per RK stage.

### OPT-7: Batch mx.eval() Calls

**Current**: `mlx_solver.py` step() method calls `mx.eval(U)` after every operator-split
substep (lines 660, 672, 684, 696, 701, 707, 712, 724, 767). Each eval forces a GPU sync.

**Fix**: Group eval calls. The Strang-split resistive diffusion (first half at line 659,
second half at line 700) could defer eval until after the hyperbolic step. The sequence
ghost_pad -> RK step -> strip_ghost could be a single eval at the end.

## 5. Porting Opportunities

### From Athena++ C++ Patterns

| Pattern | Athena++ Location | MLX Equivalent | Benefit |
|---------|-------------------|----------------|---------|
| Reconstruction + Riemann in single loop | hydro/rsolvers/ | Fused Metal kernel (OPT-4) | Eliminate UL/UR temp |
| CT EMF averaging with upwind bias | field/ct.cpp | Not yet in mlx_ct.py | Better div(B) control |
| Orbital advection (shearing box) | orbital_advection/ | N/A for DPF | -- |
| Characteristic decomposition | reconstruct/ | Too expensive for Metal | Skip |

### From PyTorch Metal Patterns

| Pattern | Metal solver location | MLX Equivalent | Benefit |
|---------|----------------------|----------------|---------|
| torch.where for flux selection | metal_riemann.py | mx.where (already used) | Parity |
| Contiguous tensor layout | metal_solver.py | mx.array is always contiguous | Built-in |
| MPS batch operations | metal_stencil.py | mx.compile for fusion | Better fusion |

### From Numba Patterns

| Pattern | Location | MLX Equivalent | Benefit |
|---------|----------|----------------|---------|
| @njit column loops for Thomas | cylindrical_mhd.py | mx.vmap or parallel scan | GPU parallelism |
| @njit WENO stencil | mhd_solver.py | Already ported as mx ops | Done |
| @njit resistivity subcycle | cylindrical_mhd.py | RKL2 STS (mlx_sts.py) | Already done |

### From CUDA MHD Patterns

| CUDA Pattern | Description | MLX Equivalent | Feasibility |
|--------------|-------------|----------------|-------------|
| Shared memory stencil tiling | Load 5-point stencil to threadgroup_memory | `threadgroup float tile[]` in MSL | Medium (OPT-13) |
| Warp-level reduction for CFL | `__shfl_down_sync` for parallel max | SIMD group ops in Metal | Medium (OPT-12) |
| Texture memory for read-only | Bind state as texture for cached reads | Metal `texture2d` | Low priority |
| Stream overlap | Kernel + transfer overlap | MLX async eval | Already implicit |

## 6. Custom Metal Kernel Candidates

### Profitability Analysis

| Candidate | Current Impl | Compute/Launch Ratio (64x128) | Worthwhile? |
|-----------|-------------|-------------------------------|-------------|
| HLL flux | NumPy CPU | Very high (eliminates CPU trip) | **Yes** |
| PLM+HLL fused | Separate MLX ops | Medium (~2.6 MB saved) | Yes |
| Flux divergence | MLX slicing | Low (simple subtraction) | No |
| WENO5-Z | mx.compile | Low (already compiled) | No |
| CFL reduction | mx.max calls | Medium (global reduction) | Maybe |
| Thomas solver (parallel) | Python loops | Very high | **Yes** (if feasible) |

### SIMD Group Opportunities

The HLLD Metal kernel at mlx_kernels.py:422 processes one interface point per thread with
no inter-thread communication. Opportunities:

1. **SIMD group max for NaN detection**: Instead of per-thread NaN check (lines 593-596),
   use `simd_any()` to check the entire SIMD group at once.
2. **SIMD group broadcast for Bn**: `Bn = 0.5*(Bn_L + Bn_R)` is computed per-thread but
   shared across the interface -- could broadcast within a SIMD group for adjacent cells.

### Threadgroup Shared Memory

For WENO5-Z reconstruction, the 5-point stencil means each output reads 5 input values.
With `threadgroup float tile[BLOCK+4]`, a block of threads can cooperatively load the
stencil neighborhood once, reducing global memory reads by ~4x for that pass. At 64x128,
the data fits in L1 cache anyway, so benefit is marginal. At 256x1024, this becomes more
significant.

## 7. Memory Layout Analysis

### Current Layout

State array: `(NVAR=10, nr, nz)` -- variables-first (SoA). This is optimal for MLX because:
- Accessing `U[IDN]` (all density values) is a contiguous slice
- WENO5-Z reconstruction along axis 1 or 2 accesses contiguous memory
- Metal kernels index as `U[var * stride + spatial_idx]` -- stride is nr*nz

### SoA vs AoS

| Layout | Access Pattern | Metal Cache Behavior | Verdict |
|--------|---------------|---------------------|---------|
| SoA (current) | Per-variable vectorization | Coalesced for single-var ops | **Keep** |
| AoS (NVAR last) | Per-cell access | Coalesced for Riemann solver | Marginal gain |

The Riemann solver accesses all 10 variables per cell, which would favor AoS. But the
reconstruction accesses single variables across many cells, which favors SoA. Since
reconstruction runs 6x per step and Riemann runs 6x, they're balanced. SoA wins because
MLX slice operations (`U[IDN]`) are zero-copy on SoA but require a transpose on AoS.

### In-Place Operations

MLX arrays are immutable by design (functional semantics for `mx.compile`). The current
pattern of `mx.stack([...], axis=0)` to rebuild state is correct but creates temporaries.
The `_stage_post_impl` function (mlx_timestepper.py:56-131) already fuses the rebuild into
a single compiled function, which is the right approach.

### Unified Memory Implications

On M3 Pro, MLX's `mx.array(np_arr)` is zero-copy when `np_arr` is float32 and C-contiguous
(mlx_state.py:433-453). The `np.asarray(mx_arr)` direction is also zero-copy.
**The real cost is not the copy but the synchronization**: `np.asarray()` forces MLX to
complete all pending GPU work before the CPU can read the result.

The Riemann solvers' `np.asarray(QL)` calls at mlx_riemann.py:115,241 are the most expensive
sync points because they interrupt the GPU pipeline mid-RHS-computation.

## 8. DMAIC Analysis

### Define

**Goal**: PF-1000 64x128 full discharge in <2 min (4x speedup from ~8 min).
**Constraint**: No physics accuracy regression (Sod L1 error, conservation, cross-backend parity).
**Scope**: MLX solver modules only (no engine/circuit changes).

### Measure

Current bottleneck ranking (from code analysis, not runtime profiling):

1. Riemann solver CPU round-trips: ~45-55% of step time
2. Operator-split transport (Thomas loops): ~8-12%
3. Ghost cell NumPy fixups: ~5-8%
4. mx.eval() sync overhead: ~3-5%
5. State pack/unpack: ~3-5%
6. Everything else (already on GPU): ~20-30%

### Analyze (Pareto 80/20)

**Top 3 optimizations covering ~80% of potential speedup:**

1. Port HLL/HLLS to pure MLX (OPT-1, OPT-2): eliminates 45-55% bottleneck
2. Fuse PLM+HLL Metal kernel (OPT-4): eliminates intermediate allocation
3. Batch eval() calls (OPT-7): reduces sync overhead

These 3 changes touch ~300 LOC and are expected to deliver 2.5-3x speedup.

### Improve

| Phase | Optimizations | Expected Speedup | LOC | Duration |
|-------|---------------|------------------|-----|----------|
| Quick Wins | OPT-1,2,3,6,7,8,10 | 2.0-2.5x | ~350 | 1-2 days |
| Medium | OPT-4,5,9,14 | 1.3-1.5x (cumulative 3-3.5x) | ~400 | 3-5 days |
| Deep | OPT-11,12,13,15 | 1.1-1.2x (cumulative 3.5-4x) | ~300 | 5-7 days |

### Control

1. **Benchmark gate**: `mlx_benchmark.py` must show >= target speedup before merge
2. **Physics gate**: All 471 MLX tests must pass; Sod L1 < 1e-3; conservation < 1e-4
3. **Regression CI**: Add `@pytest.mark.benchmark` test that asserts zone-cycles/sec >= threshold
4. **Performance tracking**: JSON benchmark output committed to `benchmarks/` directory

## 9. FMEA Table

| # | Optimization | Speedup | LOC | Sev (1-10) | Occur (1-10) | Detect (1-10) | RPN | Recommendation |
|---|-------------|---------|-----|------------|--------------|---------------|-----|----------------|
| 1 | HLL to MLX GPU | 1.8-2.5x | 120 | 7 | 4 | 2 | 56 | **Do first** -- high impact, medium risk. Float32 mitigated by Boris + dual-energy. |
| 2 | HLLS to MLX GPU | 1.5-2.0x | 130 | 5 | 3 | 2 | 30 | HLLS is inherently float32-safe (entropy pressure). Low risk. |
| 3 | Ghost pad NumPy removal | 1.1-1.2x | 40 | 4 | 3 | 3 | 36 | Low risk -- electrode BC is well-tested. |
| 4 | Fused PLM+HLL kernel | 1.2-1.4x | 150 | 8 | 5 | 4 | 160 | **Highest RPN** -- complex Metal kernel, hard to debug. Defer to Phase 2. |
| 5 | Thomas parallel scan | 1.1-1.3x | 80 | 6 | 6 | 5 | 180 | **Highest RPN** -- tridiagonal solver on GPU is non-trivial. Consider cyclic reduction. |
| 6 | Brem float32 | 1.05x | 15 | 3 | 2 | 2 | 12 | Trivial -- use log-space computation. |
| 7 | Batch eval() | 1.1-1.2x | 20 | 5 | 3 | 2 | 30 | Low risk -- just move eval() calls. |
| 8 | Pre-alloc dU_dt | 1.05x | 10 | 2 | 2 | 2 | 8 | Trivial. |
| 9 | Fuse geom src + div | 1.1x | 80 | 5 | 4 | 4 | 80 | Medium -- touches RHS structure. |
| 10 | Compile clamp_recon | 1.02x | 5 | 1 | 1 | 1 | 1 | Trivial. |
| 11 | Metal flux divergence | 1.1x | 100 | 6 | 5 | 5 | 150 | Complex kernel for marginal gain. Defer. |
| 12 | SIMD group in HLLD | 1.05x | 30 | 4 | 3 | 4 | 48 | Medium -- MSL SIMD API is well-documented. |
| 13 | Threadgroup shared mem | 1.1x | 50 | 6 | 5 | 5 | 150 | Only worthwhile at 256x1024+. |
| 14 | mx.vmap Thomas | 1.15x | 40 | 5 | 4 | 3 | 60 | Good ROI if mx.vmap supports the pattern. |
| 15 | Overlap RHS + eval | 1.05x | 20 | 3 | 3 | 3 | 27 | Low risk -- MLX handles async natively. |

**Risk summary**: OPT-4 (fused kernel) and OPT-5 (parallel Thomas) have highest RPN due
to implementation complexity and debugging difficulty. Schedule these after validating the
simpler optimizations deliver the expected gains.

## 10. Implementation Phases

### Phase 1: Quick Wins (1-2 days, expected 2.0-2.5x)

1. **OPT-1**: Rewrite `_hll_flux()` as pure MLX operations in mlx_riemann.py
   - Keep NumPy version as `_hll_flux_cpu64()` fallback
   - Test: Sod shock tube parity between MLX and NumPy versions
2. **OPT-2**: Rewrite `_hlls_flux()` as pure MLX operations
3. **OPT-7**: Consolidate mx.eval() calls in solver step()
   - Remove intermediate evals between ghost_pad and RK step
   - Single eval after hyperbolic step + div(B) cleaning
4. **OPT-10**: Wrap `_clamp_reconstructed` with `_compile_if_available`
5. **OPT-6**: Bremsstrahlung in log-space: `log(Q_rad) = log(coeff) + 2*log(ne) + 0.5*log(Te)`
6. **OPT-8**: Pre-allocate `dU_dt = mx.zeros_like(U)` outside the dim loop in mhd_rhs

### Phase 2: Medium Effort (3-5 days, expected 1.3-1.5x additional)

7. **OPT-3**: Vectorize ghost padding fixups as MLX slice operations
8. **OPT-4**: Fused PLM+HLL Metal kernel
   - Start with 2D (r,z) version, extend to 3D later
   - Benchmark at 64x128 and 256x1024 to confirm bandwidth win
9. **OPT-9**: Fuse geometric source into mhd_rhs return (avoid separate stack)
10. **OPT-14**: Explore mx.vmap for Thomas solver column-parallel execution
    - Fallback: keep Python loops but batch `np.asarray` calls

### Phase 3: Deep Optimization (5-7 days, expected 1.1-1.2x additional)

11. **OPT-5**: Parallel tridiagonal solver on GPU (cyclic reduction algorithm)
12. **OPT-12**: SIMD group ops in HLLD kernel for NaN check and Bn broadcast
13. **OPT-13**: Threadgroup shared memory for WENO5-Z stencil at large grids
14. **OPT-15**: Experiment with async eval patterns (stream-like overlap)
15. **OPT-11**: Metal kernel for r-weighted flux divergence (cylindrical)

### Validation at Each Phase

After each phase:
- `pytest tests/test_mlx_*.py -v` (471 tests)
- `python3 -m dpf.benchmarks.mlx_benchmark --steps 50 --output benchmarks/opt_phaseN.json`
- Compare Sod L1 error, conservation metrics, and zone-cycles/sec vs baseline

## 11. Estimated Total Speedup

| Scenario | Speedup | Confidence | Basis |
|----------|---------|------------|-------|
| Phase 1 only | 2.0-2.5x | 80% | Riemann CPU elimination is well-understood |
| Phase 1 + 2 | 2.8-3.5x | 65% | Fused kernel depends on Metal compiler |
| Phase 1 + 2 + 3 | 3.5-4.2x | 50% | Deep opts have diminishing returns |
| Theoretical max | ~5x | 20% | Assumes zero CPU round-trips |

**Confidence-weighted estimate**: 3.0x speedup (Phase 1+2), bringing PF-1000 64x128 from
~8 min to ~2.7 min. Phase 3 could push to ~2 min but with uncertainty.

The 4x target requires all three phases. Phase 1 alone gets 80% of the way to "good enough"
(~3.2-4 min) with high confidence and low risk.

## 12. Functions Not Yet Using mx.compile()

| Function | File | Line | Compilable? | Reason if No |
|----------|------|------|-------------|--------------|
| `_clamp_reconstructed` | mlx_riemann.py | 60 | Yes | Pure elementwise |
| `plm_reconstruct` | mlx_reconstruction.py | 115 | Partial | Python control flow (n<2 check) |
| `_mc_limit` | mlx_reconstruction.py | 86 | Yes | Pure elementwise |
| `_minmod` | mlx_reconstruction.py | 69 | Yes | Pure elementwise |
| `fast_magnetosonic` | mlx_primitives.py | 233 | Yes | Pure elementwise |
| `entropy_resync` | mlx_primitives.py | 358 | Partial | Gradient ops with concat |
| `_geometric_sources` | mlx_timestepper.py | 184 | Yes | Pure elementwise + broadcast |
| `compute_dt_cfl` | mlx_timestepper.py | 232 | No | CPU scalar return (`float()`) |
| `_gradient_1d` | mlx_divb.py | 37 | Partial | Dynamic slicing |
| `dedner_source` | mlx_divb.py | 120 | Partial | Grid object access |

Priority for compilation: `fast_magnetosonic` (called in CFL and stage post), `_mc_limit`
and `_minmod` (called per PLM reconstruction), `_clamp_reconstructed` (called per flux
computation).

## 13. NumPy Bridge Calls That Could Stay on GPU

| Call Site | File:Line | Direction | Avoidable? |
|-----------|-----------|-----------|------------|
| `np.asarray(QL)` in `_hll_flux` | mlx_riemann.py:241 | GPU->CPU | **Yes** (OPT-1) |
| `np.asarray(QL)` in `_hlls_flux` | mlx_riemann.py:115 | GPU->CPU | **Yes** (OPT-2) |
| `np.asarray(U_padded)` in ghost fixup | mlx_solver.py:325 | GPU->CPU | **Yes** (OPT-3) |
| `np.asarray()` in bremsstrahlung | mlx_sources.py:447-448 | GPU->CPU | **Yes** (OPT-6) |
| `np.asarray()` in Thomas solver | mlx_transport.py:256-260 | GPU->CPU | Partially (OPT-5) |
| `np.asarray()` in viscosity CFL | mlx_viscosity.py:323-333 | GPU->CPU | Keep (1x/step) |
| `np.asarray()` in state unpack | mlx_state.py:380 | GPU->CPU | Keep (output) |
| `np.asarray()` in mask_ghost_rhs | mlx_timestepper.py:529 | CPU->GPU | **Yes** (pre-alloc mask) |

The first four are high-priority elimination targets. The Thomas solver transfers are harder
to avoid without a GPU tridiagonal algorithm.

## 14. Algorithmic Optimization Notes

### WENO5-Z Smoothness Indicators

Current implementation (mlx_reconstruction.py:232-243) computes beta0, beta1, beta2 with
6 squarings and 6 additions each (18 total). The Jiang-Shu indicators share common terms:
- `qm2 - 2*qm1 + q0` appears in beta0
- `qm1 - 2*q0 + qp1` appears in beta1
- `q0 - 2*qp1 + qp2` appears in beta2

These second differences could be pre-computed once and reused, saving ~6 multiply-add ops
per cell. At 64x128 with 10 variables, that is ~50K FLOPs saved per reconstruction call.
Marginal but free in terms of code complexity.

### SSP-RK3 Stage Overlap

The three RK stages are strictly sequential (stage 2 depends on stage 1). However, within
each stage, the r-direction and z-direction flux computations are independent until the
divergence summation. With MLX's lazy evaluation, these are already implicitly overlapped
when both are enqueued before `mx.eval()`. No code change needed.

### Operator-Split Fusion

Currently, resistive diffusion, thermal conduction, and viscosity are three separate
operator-split steps, each with its own cons_to_prim decomposition. Fusing them into a
single operator-split step that shares the primitive variable computation would save ~2
cons_to_prim calls per timestep. At 64x128 with 10 variables, that is ~160K FLOPs -- small
but the real saving is avoiding 2 extra `mx.eval()` sync points.
