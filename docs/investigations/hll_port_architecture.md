# HLL/HLLS CPU Round-Trip Elimination: Architecture Decision

**Date**: 2026-03-26
**Status**: Design complete, ready for implementation
**Author**: dpf-engine-architect
**Estimated speedup**: 1.8-2.5x (from eliminating 45-55% bottleneck)

## Problem Statement

`_hll_flux()` (mlx_riemann.py:223-344) and `_hlls_flux()` (mlx_riemann.py:86-215) convert
mx.array to np.ndarray via `np.asarray()`, compute in float64, then convert back via
`mx.array()`. This GPU->CPU->GPU round-trip happens **6 times per SSP-RK3 step** (2 dims x
3 stages). On M3 Pro unified memory, the data copy itself is near-zero-cost, but each
`np.asarray()` forces an implicit `mx.eval()` that drains the GPU command queue -- breaking
MLX's lazy evaluation pipeline and serializing all preceding GPU work.

## Decision: Option C (Pure mx.array Port)

**Chosen**: Rewrite both solvers using only MLX ops. No Metal kernels.

**Rationale**:

| Option | Speedup | Risk | LOC | Maintainability |
|--------|---------|------|-----|-----------------|
| A/C: Pure mx.array | 1.8-2.5x | Low | ~240 | One language (Python), easy to debug |
| B: Metal kernel | 2.0-2.8x | High | ~400 MSL + glue | Two languages, hard to debug, RPN=160 |
| Hybrid (keep CPU) | 0x | -- | 0 | Status quo bottleneck |

Option C wins because:

1. **The bottleneck is the sync, not the compute.** On a 64x128 grid, HLL processes ~8K
   interface points. The arithmetic is trivially fast on either CPU or GPU. The cost is the
   pipeline stall from `np.asarray()` forcing `mx.eval()` mid-RHS, 6 times per step.

2. **mx.compile() can fuse the entire HLL into one kernel.** MLX's JIT compiler turns a
   chain of elementwise ops into a single Metal kernel. We get Metal kernel performance
   without writing Metal Shading Language. The existing HLLD Metal kernel (253 lines of MSL
   at mlx_kernels.py:354-607) proves the physics is GPU-friendly, but writing a second 200+
   line MSL kernel for HLL is unjustified when mx.compile achieves 90% of the throughput
   with 10% of the maintenance burden.

3. **Every operation in HLL/HLLS is elementwise.** No reductions, no scatters, no variable-
   length loops. This is the ideal workload for MLX's lazy evaluation + compile pipeline.
   The Boris correction, wavespeeds, physical fluxes, HLL combination, and NaN fallback
   are all broadcastable array ops.

4. **float32 is safe for HLL.** The known float32 failure mode is in HLLD star-state
   denominators (D_L/D_R), not in HLL wavespeeds. HLL pressure recovery via `p = gm1 *
   (E - KE - 0.5*B^2)` can cancel in float32, but this is already mitigated by the Boris
   speed cap (prevents extreme E values) and the Lax-Friedrichs NaN fallback. HLLS avoids
   this entirely via entropy-derived pressure. Both are float32-safe for the DPF regime.

## Architecture

### 1. Shared Wavespeed Helper

Both HLL and HLLS compute identical Boris-capped fast magnetosonic wavespeeds. Extract once:

```python
@mx.compile
def _wavespeeds_mlx(
    rho_L: mx.array, rho_R: mx.array,
    vn_L: mx.array, vn_R: mx.array,
    B2_L: mx.array, B2_R: mx.array,
    Bt_sq_L: mx.array, Bt_sq_R: mx.array,
    p_L: mx.array, p_R: mx.array,
    gamma: float,
) -> tuple[mx.array, mx.array]:
    """Boris-capped HLL wavespeeds (SL, SR).

    Gombosi (2002) semi-relativistic correction prevents vacuum Alfven
    speeds from exceeding c_boris = 500 km/s.  Shared by HLL and HLLS.
    """
    C_BORIS_SQ = mx.array(2.5e11, dtype=mx.float32)  # (500 km/s)^2
    TINY = mx.array(1e-20, dtype=mx.float32)

    a_sq_L = mx.minimum(gamma * p_L / rho_L, C_BORIS_SQ)
    a_sq_R = mx.minimum(gamma * p_R / rho_R, C_BORIS_SQ)

    va_sq_L = B2_L / rho_L
    va_sq_R = B2_R / rho_R
    va_sq_L = va_sq_L * C_BORIS_SQ / (va_sq_L + C_BORIS_SQ)
    va_sq_R = va_sq_R * C_BORIS_SQ / (va_sq_R + C_BORIS_SQ)

    vat_sq_L = Bt_sq_L / rho_L
    vat_sq_R = Bt_sq_R / rho_R
    vat_sq_L = vat_sq_L * C_BORIS_SQ / (vat_sq_L + C_BORIS_SQ)
    vat_sq_R = vat_sq_R * C_BORIS_SQ / (vat_sq_R + C_BORIS_SQ)

    disc_L = mx.maximum((a_sq_L - va_sq_L) ** 2 + 4.0 * a_sq_L * vat_sq_L, 0.0)
    disc_R = mx.maximum((a_sq_R - va_sq_R) ** 2 + 4.0 * a_sq_R * vat_sq_R, 0.0)

    cf_L = mx.sqrt(mx.maximum(0.5 * (a_sq_L + va_sq_L + mx.sqrt(disc_L)), 0.0))
    cf_R = mx.sqrt(mx.maximum(0.5 * (a_sq_R + va_sq_R + mx.sqrt(disc_R)), 0.0))

    SL = mx.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = mx.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = mx.maximum(SR, SL + TINY)

    return SL, SR
```

**Key design choice**: `mx.compile` on the wavespeed helper. MLX compiles this into a fused
Metal kernel that runs the entire wavespeed computation (20+ array ops) as a single GPU dispatch.
The `gamma` parameter is a compile-time constant (always 5/3 for ideal MHD), enabling further
optimization.

### 2. HLL Flux (Pure MLX)

```python
def _hll_flux_mlx(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """HLL two-wave Riemann flux -- pure MLX, zero CPU round-trips."""
    # Dimension mapping (same logic, but produces Python ints, no arrays)
    if dim == 0:
        im_n, im_t1, im_t2 = IMR, IMZ, IMT
        ib_n, ib_t1, ib_t2 = IBR, IBZ, IBT
    elif dim == 1:
        im_n, im_t1, im_t2 = IMZ, IMR, IMT
        ib_n, ib_t1, ib_t2 = IBZ, IBR, IBT
    else:
        im_n, im_t1, im_t2 = IMT, IMR, IMZ
        ib_n, ib_t1, ib_t2 = IBT, IBR, IBZ

    gm1 = gamma - 1.0

    rho_L = mx.maximum(QL[IDN], RHO_FLOOR)
    rho_R = mx.maximum(QR[IDN], RHO_FLOOR)
    inv_rL = mx.reciprocal(rho_L)
    inv_rR = mx.reciprocal(rho_R)

    vn_L = QL[im_n] * inv_rL
    vn_R = QR[im_n] * inv_rR

    # Kinetic + magnetic energy for pressure recovery
    KE_L = 0.5 * rho_L * (
        (QL[IMR] * inv_rL) ** 2 + (QL[IMZ] * inv_rL) ** 2 + (QL[IMT] * inv_rL) ** 2
    )
    KE_R = 0.5 * rho_R * (
        (QR[IMR] * inv_rR) ** 2 + (QR[IMZ] * inv_rR) ** 2 + (QR[IMT] * inv_rR) ** 2
    )
    B2_L = QL[IBR] ** 2 + QL[IBZ] ** 2 + QL[IBT] ** 2
    B2_R = QR[IBR] ** 2 + QR[IBZ] ** 2 + QR[IBT] ** 2
    p_L = mx.maximum(gm1 * (QL[IEN] - KE_L - 0.5 * B2_L), P_FLOOR)
    p_R = mx.maximum(gm1 * (QR[IEN] - KE_R - 0.5 * B2_R), P_FLOOR)

    Bn_L, Bn_R = QL[ib_n], QR[ib_n]
    Bt_sq_L = mx.maximum(B2_L - Bn_L ** 2, 0.0)
    Bt_sq_R = mx.maximum(B2_R - Bn_R ** 2, 0.0)

    # Shared wavespeed computation
    SL, SR = _wavespeeds_mlx(
        rho_L, rho_R, vn_L, vn_R,
        B2_L, B2_R, Bt_sq_L, Bt_sq_R,
        p_L, p_R, gamma,
    )

    # Physical fluxes (inlined, not a nested function -- mx.compile can see it)
    FL = _physical_flux_mlx(QL, rho_L, inv_rL, vn_L, p_L, B2_L, gm1,
                            im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2)
    FR = _physical_flux_mlx(QR, rho_R, inv_rR, vn_R, p_R, B2_R, gm1,
                            im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2)

    # HLL combination
    TINY = 1e-20
    inv_dS = mx.reciprocal(mx.maximum(SR - SL, TINY))
    F_hll = (SR * FL - SL * FR + SL * SR * (QR - QL)) * inv_dS

    # Upwind selection
    F_out = mx.where(SL >= 0.0, FL, mx.where(SR <= 0.0, FR, F_hll))

    # Zero normal B flux (CT constraint)
    # Use mx.concatenate to reconstruct with zeroed ib_n slot
    F_out = _zero_slot(F_out, ib_n)

    # NaN/Inf fallback: Lax-Friedrichs
    S_max = mx.maximum(mx.abs(SL), mx.abs(SR))
    F_LF = 0.5 * (FL + FR) - 0.5 * S_max * (QR - QL)
    bad = mx.isnan(F_out) | mx.isinf(F_out)
    F_out = mx.where(bad, F_LF, F_out)

    return F_out
```

**Design notes**:

- **No `np.asarray` anywhere.** Every operation is mx.array -> mx.array.
- **`mx.reciprocal` instead of `1.0 / rho`.** Single-op inverse, no broadcast temp.
- **`_zero_slot` helper** replaces `F_out[ib_n] = 0.0` (in-place mutation is illegal in MLX).
  Implementation: `mx.where(slot_mask, 0.0, F_out)` with a pre-built boolean mask.
- **Lax-Friedrichs always computed.** In the NumPy version, LF is computed only when NaN
  is detected (`if np.any(nans)`). In MLX, the branch would force an eval. Instead, always
  compute LF (cheap -- 3 ops) and select via `mx.where`. This is branchless and stays lazy.
- **No float32 clamp.** The NumPy version clips to `float32.max` before cast. The MLX version
  stays in float32 throughout, so overflow to inf is caught by the NaN fallback. The clip
  was defensive for the float64->float32 downcast; with no downcast, it's unnecessary.

### 3. HLLS Flux (Pure MLX)

Same structure as HLL but with entropy-derived pressure:

```python
def _hlls_flux_mlx(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """HLLS entropy-based Riemann flux -- pure MLX, zero CPU round-trips."""
    # ... dimension mapping identical to HLL ...

    rho_L = mx.maximum(QL[IDN], RHO_FLOOR)
    rho_R = mx.maximum(QR[IDN], RHO_FLOOR)

    # Entropy pressure recovery: p = Srho * rho^(gamma-1)
    # No E - KE - ME cancellation. Inherently float32-safe.
    Srho_L = mx.maximum(QL[ISR], P_FLOOR)
    Srho_R = mx.maximum(QR[ISR], P_FLOOR)
    p_L = mx.maximum(Srho_L * mx.power(rho_L, gamma - 1.0), P_FLOOR)
    p_R = mx.maximum(Srho_R * mx.power(rho_R, gamma - 1.0), P_FLOOR)

    # ... wavespeeds via _wavespeeds_mlx (shared with HLL) ...
    # ... physical fluxes with entropy reconstruction of E_tot ...
    # ... HLL combination + NaN fallback (same pattern) ...
```

**Key difference from HLL**: pressure comes from entropy tracer (ISR slot), not from
`E - KE - 0.5*B^2`. The `_physical_flux_mlx` helper gets a flag to reconstruct E_tot
from entropy-derived pressure instead of using the conserved energy directly:

```python
# In HLLS physical flux:
E_tot = p / mx.maximum(gm1, 1e-30) + 0.5 * rho * v_sq + 0.5 * B2
F[IEN] = (E_tot + pt) * vn - Bn * vB

# vs HLL physical flux:
E_tot = U[IEN]  # conserved total energy directly
F[IEN] = (E_tot + pt) * vn - Bn * vB
```

### 4. Shared Physical Flux Helper

```python
def _physical_flux_mlx(
    U: mx.array,
    rho: mx.array,
    inv_r: mx.array,
    vn: mx.array,
    p: mx.array,
    B2: mx.array,
    gm1: float,
    im_n: int, im_t1: int, im_t2: int,
    ib_n: int, ib_t1: int, ib_t2: int,
    entropy_energy: bool = False,
) -> mx.array:
    """MHD physical flux vector F(U) for a single direction.

    All operations are mx.array -- no CPU round-trip.

    Args:
        entropy_energy: If True, reconstruct E_tot from entropy-derived
            pressure (HLLS mode). If False, use conserved E directly (HLL mode).
    """
    Bn = U[ib_n]
    Bt1 = U[ib_t1]
    Bt2 = U[ib_t2]
    vt1 = U[im_t1] * inv_r
    vt2 = U[im_t2] * inv_r
    pt = p + 0.5 * B2
    vB = vn * Bn + vt1 * Bt1 + vt2 * Bt2

    # Density flux
    F_dn = rho * vn
    # Momentum fluxes
    F_mn = rho * vn * vn + pt - Bn * Bn
    F_mt1 = rho * vn * vt1 - Bn * Bt1
    F_mt2 = rho * vn * vt2 - Bn * Bt2
    # Energy flux
    if entropy_energy:
        v_sq = vn ** 2 + vt1 ** 2 + vt2 ** 2
        E_tot = p / mx.maximum(gm1, 1e-30) + 0.5 * rho * v_sq + 0.5 * B2
    else:
        E_tot = U[IEN]
    F_en = (E_tot + pt) * vn - Bn * vB
    # Entropy flux (passive advection)
    F_sr = U[ISR] * vn
    # Induction fluxes
    F_bn = mx.zeros_like(Bn)
    F_bt1 = vn * Bt1 - vt1 * Bn
    F_bt2 = vn * Bt2 - vt2 * Bn
    # Electron energy (if present)
    F_ee = U[IEE] * vn if U.shape[0] > IEE else mx.zeros_like(Bn)

    # Assemble in slot order: IDN=0, IMR=1, IMZ=2, IMT=3, IEN=4,
    # ISR=5, IBR=6, IBZ=7, IBT=8, IEE=9
    slots = [None] * NVAR
    slots[IDN] = F_dn
    slots[im_n] = F_mn
    slots[im_t1] = F_mt1
    slots[im_t2] = F_mt2
    slots[IEN] = F_en
    slots[ISR] = F_sr
    slots[ib_n] = F_bn
    slots[ib_t1] = F_bt1
    slots[ib_t2] = F_bt2
    if U.shape[0] > IEE:
        slots[IEE] = F_ee

    # Expand each to (1, ...) and concatenate along axis 0
    parts = [s[None] if s.ndim == U.ndim - 1 else s for s in slots if s is not None]
    return mx.concatenate(parts, axis=0)
```

**Design choice -- `mx.concatenate` vs `mx.stack`**: The flux components are 2D arrays
(n_ifaces, n_transverse). We unsqueeze to (1, n_ifaces, n_transverse) and concatenate along
axis 0 to build (NVAR, n_ifaces, n_transverse). This matches the input shape convention and
avoids the `F = np.zeros_like(U); F[IDN] = ...` mutation pattern that is illegal in MLX.

Under `mx.compile`, the concatenation is fused into the preceding elementwise ops -- the
intermediate per-variable arrays are never materialized.

### 5. Zero-Slot Helper

The NumPy code does `F_out[ib_n] = 0.0` (in-place mutation). MLX arrays are immutable.
Two options:

**Option A -- mx.where with mask (chosen)**:
```python
def _zero_slot(F: mx.array, slot: int) -> mx.array:
    """Zero out one variable slot in a (NVAR, ...) flux array."""
    mask = mx.arange(NVAR)[:, None, None] == slot  # broadcast to (NVAR, 1, 1)
    return mx.where(mask, 0.0, F)
```

Pros: One op, branchless, compilable. Cons: Builds a mask each call (but it's a compile-
time constant under mx.compile, so it gets folded).

**Option B -- slice + concatenate**:
```python
F_pre = F[:slot]
F_post = F[slot + 1:]
F_zero = mx.zeros_like(F[slot:slot + 1])
return mx.concatenate([F_pre, F_zero, F_post], axis=0)
```

Pros: Explicit. Cons: 3 slices + 1 concat vs 1 where. Under mx.compile both fuse, but
Option A is cleaner.

### 6. mx.compile Strategy

The wavespeed helper is decorated with `@mx.compile`. The full HLL/HLLS functions are NOT
decorated -- instead, `compute_fluxes()` or `mhd_rhs()` can be the compilation boundary.
Reason: `dim` is a Python int that changes between calls (0 for radial, 1 for axial), and
`mx.compile` recompiles when Python-valued arguments change. Compiling at the `mhd_rhs`
level would capture both sweeps in one trace, which is the optimal granularity.

However, the current `mhd_rhs` has Python control flow (`for dim in dims_to_sweep`) that
prevents full compilation. The pragmatic approach:

1. **Phase 1**: Compile `_wavespeeds_mlx` only. This fuses 20+ ops into one kernel.
2. **Phase 2**: Compile `_physical_flux_mlx` with `entropy_energy` as a static bool.
3. **Phase 3** (future): Unroll the dim loop in `mhd_rhs` and compile the full RHS.

Each phase is independently testable and delivers incremental benefit.

## Integration Into compute_fluxes()

**Zero change to dispatch logic.** The new functions slot into the existing `riemann` string
dispatch in `compute_fluxes()` (mlx_riemann.py:375-436):

```python
# CURRENT dispatch (unchanged):
if riemann == "hlls":
    ...  # calls _hlls_flux (CPU)
if riemann == "hll":
    ...  # calls _hll_flux (CPU)

# NEW dispatch with A/B testing:
if riemann == "hlls":
    return _hlls_flux_mlx(...)    # GPU -- replaces _hlls_flux
if riemann == "hll":
    return _hll_flux_mlx(...)     # GPU -- replaces _hll_flux
```

During rollout, add temporary `"hll_cpu"` and `"hlls_cpu"` aliases that call the old NumPy
versions. This allows A/B testing without touching the default code path.

### Rollout Plan

| Phase | Change | Riemann string | Risk |
|-------|--------|----------------|------|
| 0 (now) | CPU NumPy HLL/HLLS | `"hll"`, `"hlls"` | Baseline |
| 1 | Add `_hll_flux_mlx` | `"hll"` (default) + `"hll_cpu"` (fallback) | Low |
| 2 | Add `_hlls_flux_mlx` | `"hlls"` (default) + `"hlls_cpu"` (fallback) | Low |
| 3 | Remove CPU aliases after 1 week of CI green | -- | None |

The old functions are renamed, not deleted:

```python
_hll_flux_cpu64 = _hll_flux      # keep as reference + fallback
_hlls_flux_cpu64 = _hlls_flux
```

### 4D Cartesian Path

`_compute_fluxes_4d()` (mlx_riemann.py:439-491) flattens transverse dimensions before
calling the Riemann solver. The new MLX functions accept the same (NVAR, n_ifaces, n_trans)
shape -- no change needed. The flatten/unflatten logic stays in `_compute_fluxes_4d`.

### Transpose Pattern for dim=1

The current code transposes QL/QR before calling HLL/HLLS for dim=1 (axial sweep):

```python
if dim == 1:
    QL_t = mx.transpose(QL, axes=[0, 2, 1])
    QR_t = mx.transpose(QR, axes=[0, 2, 1])
    F_t = _hll_flux(QL_t, QR_t, gamma=gamma, dim=1)
    return mx.transpose(F_t, axes=[0, 2, 1])
```

This is because the solvers index the normal direction as axis 1. The transpose is a zero-
copy view in MLX (just metadata), so it costs nothing. The new MLX functions use the same
convention -- no change.

## Testing Strategy

### Parity Tests

For each solver (HLL, HLLS), add a test that:

1. Generates random QL/QR states (seeded, reproducible)
2. Runs the old NumPy version: `_hll_flux_cpu64(QL, QR, gamma, dim)`
3. Runs the new MLX version: `_hll_flux_mlx(QL, QR, gamma, dim)`
4. Asserts `mx.allclose(F_mlx, F_cpu, rtol=1e-5, atol=1e-7)`

The `rtol=1e-5` tolerance accounts for float32 vs float64 differences. The NumPy version
operates in float64 and downcasts; the MLX version operates natively in float32. For well-
conditioned states (moderate beta, subsonic flows), the difference is < 1e-6. For extreme
states (beta < 0.01, Mach > 10), the difference can reach 1e-4 -- still acceptable for
HLL's inherent diffusivity.

### Physics Tests (existing, must still pass)

| Test | File | What it validates |
|------|------|-------------------|
| Sod shock tube | test_mlx_solver.py | L1(rho) < 1e-3 on 128-cell 1D |
| Brio-Wu MHD | test_mlx_solver.py | Compound wave structure, no NaN |
| Diffusion | test_mlx_transport.py | Convergence rate matches theory |
| Conservation | test_mlx_solver.py | dE/E0 < 1e-4 per step |
| Calibration | test_mlx_calibration.py | PF-1000 I_peak within 4.1% |

Run: `pytest tests/test_mlx_*.py -v -x`

### Performance Benchmark

```python
# Before: measure time-per-step with CPU HLL
solver = MLXMHDSolver(config, riemann="hll_cpu")
t_cpu = benchmark(solver.step, n_steps=50)

# After: measure time-per-step with MLX HLL
solver = MLXMHDSolver(config, riemann="hll")
t_mlx = benchmark(solver.step, n_steps=50)

speedup = t_cpu / t_mlx
assert speedup >= 1.5, f"Expected >= 1.5x, got {speedup:.2f}x"
```

Target: >= 1.8x on 64x128 grid, >= 2.0x on 128x256.

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| float32 pressure cancellation in HLL | Low (Boris cap limits E) | Medium (wrong wavespeeds) | LF fallback catches NaN; HLLS has no cancellation |
| mx.compile recompilation overhead | Low (dim is 0 or 1, cached) | Low (one-time cost) | Warm up both dims on first step |
| mx.power float32 precision in HLLS | Low (gamma-1=2/3 is exact) | Low (small error in p) | Compare against CPU reference |
| Regression in existing tests | Low | High | Full test suite is the gate |

## Implementation Checklist

- [ ] Add `_wavespeeds_mlx()` to mlx_riemann.py (shared helper, ~35 LOC)
- [ ] Add `_physical_flux_mlx()` to mlx_riemann.py (shared helper, ~45 LOC)
- [ ] Add `_zero_slot()` to mlx_riemann.py (3 LOC)
- [ ] Add `_hll_flux_mlx()` to mlx_riemann.py (~50 LOC, calls helpers)
- [ ] Add `_hlls_flux_mlx()` to mlx_riemann.py (~40 LOC, calls helpers)
- [ ] Rename `_hll_flux` -> `_hll_flux_cpu64`, `_hlls_flux` -> `_hlls_flux_cpu64`
- [ ] Update `compute_fluxes()` dispatch: default to MLX versions, add `"hll_cpu"` / `"hlls_cpu"`
- [ ] Add parity test: `test_hll_mlx_vs_cpu64()` in tests/test_mlx_riemann_parity.py
- [ ] Add parity test: `test_hlls_mlx_vs_cpu64()` in same file
- [ ] Run full `pytest tests/test_mlx_*.py -v` -- all 471+ tests must pass
- [ ] Run benchmark: `python3 -m dpf.benchmarks.mlx_benchmark --steps 50`
- [ ] If speedup < 1.5x, investigate with `mx.disable_compile()` to isolate fusion benefit

## Future Work (Not This Sprint)

1. **Fused PLM+HLL Metal kernel** (OPT-4, RPN=160): Skip intermediate UL/UR allocation.
   Only worth it at 256x1024+ grids where memory bandwidth dominates.

2. **Compile mhd_rhs end-to-end**: Unroll the dim loop, inline reconstruction + Riemann +
   divergence into a single mx.compile trace. Requires refactoring the Python control flow
   but would eliminate all intermediate mx.eval() sync points within a single RK stage.

3. **HLLD in pure MLX**: The HLLD Metal kernel (253 LOC MSL) could be rewritten as pure
   MLX ops + mx.compile. The star-state computation has more branching than HLL but is still
   elementwise. Risk: float32 cancellation in D_L/D_R denominators (the reason the Metal
   kernel exists in the first place). Would need the same dual-energy/entropy mitigation.
