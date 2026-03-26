# MLX Optimization Phase 1 — Prototype Code

**Date**: 2026-03-26
**Scope**: OPT-1 (HLL GPU), OPT-7 (eval consolidation), OPT-6 (bremsstrahlung log-space), mx.compile candidates
**Prereq**: OPT-2 (HLLS GPU) already done as `_hlls_flux_gpu` in mlx_riemann.py:86-205

---

## 1. `_hll_flux_gpu()` — Pure MLX HLL Riemann Solver (~120 LOC)

Drop-in replacement for `_hll_flux()` (mlx_riemann.py:350-471). Key difference from
the HLLS GPU version: HLL uses E-KE-ME pressure recovery in the **energy flux** (the
conserved total energy E, not entropy-reconstructed E_tot), but switches to entropy
pressure for **wavespeed computation** where float32 cancellation actually bites.

This is the correct hybrid: wavespeeds from entropy (no cancellation), fluxes from
conserved energy (conservative). The HLLS variant reconstructs E_tot from entropy for
both — slightly more diffusive but safer.

```python
def _hll_flux_gpu(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """HLL two-wave Riemann flux -- pure MLX, zero CPU round-trips.

    Wavespeeds use entropy-derived pressure (ISR slot) to avoid float32
    catastrophic cancellation in E-KE-ME. Energy flux uses conserved E
    directly (conservative formulation).

    Args:
        QL: Left state at interfaces, shape (NVAR, n_ifaces, n_transverse).
        QR: Right state at interfaces, shape (NVAR, n_ifaces, n_transverse).
        gamma: Adiabatic index (scalar float).
        dim: Normal direction (0=radial, 1=axial, 2=y-Cartesian).

    Returns:
        Numerical flux, shape (NVAR, n_ifaces, n_transverse), mx.array float32.
    """
    TINY = 1e-20
    _C_BORIS_SQ = 2.5e11  # (500 km/s)^2

    # --- Dimension mapping (Python control flow, not data-dependent) ---
    if dim == 0:
        im_n, im_t1, im_t2 = IMR, IMZ, IMT
        ib_n, ib_t1, ib_t2 = IBR, IBZ, IBT
    elif dim == 1:
        im_n, im_t1, im_t2 = IMZ, IMR, IMT
        ib_n, ib_t1, ib_t2 = IBZ, IBR, IBT
    else:
        im_n, im_t1, im_t2 = IMT, IMR, IMZ
        ib_n, ib_t1, ib_t2 = IBT, IBR, IBZ

    # --- Primitive extraction (all mx.array, stays on GPU) ---
    rho_L = mx.maximum(QL[IDN], RHO_FLOOR)
    rho_R = mx.maximum(QR[IDN], RHO_FLOOR)
    inv_rL = 1.0 / rho_L
    inv_rR = 1.0 / rho_R

    vn_L = QL[im_n] * inv_rL
    vn_R = QR[im_n] * inv_rR
    vt1_L = QL[im_t1] * inv_rL
    vt2_L = QL[im_t2] * inv_rL
    vt1_R = QR[im_t1] * inv_rR
    vt2_R = QR[im_t2] * inv_rR

    Bn_L, Bn_R = QL[ib_n], QR[ib_n]
    Bt1_L, Bt1_R = QL[ib_t1], QR[ib_t1]
    Bt2_L, Bt2_R = QL[ib_t2], QR[ib_t2]

    # --- Pressure from entropy tracer FOR WAVESPEEDS (no E-KE-ME) ---
    gm1 = gamma - 1.0
    Srho_L = mx.maximum(QL[ISR], P_FLOOR)
    Srho_R = mx.maximum(QR[ISR], P_FLOOR)
    p_L = mx.maximum(Srho_L * mx.power(rho_L, gm1), P_FLOOR)
    p_R = mx.maximum(Srho_R * mx.power(rho_R, gm1), P_FLOOR)

    # --- Magnetic field magnitudes ---
    B2_L = QL[IBR] ** 2 + QL[IBZ] ** 2 + QL[IBT] ** 2
    B2_R = QR[IBR] ** 2 + QR[IBZ] ** 2 + QR[IBT] ** 2
    Bt_sq_L = mx.maximum(B2_L - Bn_L ** 2, 0.0)
    Bt_sq_R = mx.maximum(B2_R - Bn_R ** 2, 0.0)

    # --- Boris-corrected wavespeeds (caps at c_boris = 500 km/s) ---
    a_sq_L = mx.minimum(gamma * p_L * inv_rL, _C_BORIS_SQ)
    a_sq_R = mx.minimum(gamma * p_R * inv_rR, _C_BORIS_SQ)
    va_sq_L = B2_L * inv_rL
    va_sq_R = B2_R * inv_rR
    va_sq_L = va_sq_L * _C_BORIS_SQ / (va_sq_L + _C_BORIS_SQ)
    va_sq_R = va_sq_R * _C_BORIS_SQ / (va_sq_R + _C_BORIS_SQ)
    vat_sq_L = Bt_sq_L * inv_rL
    vat_sq_R = Bt_sq_R * inv_rR
    vat_sq_L = vat_sq_L * _C_BORIS_SQ / (vat_sq_L + _C_BORIS_SQ)
    vat_sq_R = vat_sq_R * _C_BORIS_SQ / (vat_sq_R + _C_BORIS_SQ)

    # Fast magnetosonic speed (stable discriminant form)
    disc_L = mx.maximum((a_sq_L - va_sq_L) ** 2 + 4.0 * a_sq_L * vat_sq_L, 0.0)
    disc_R = mx.maximum((a_sq_R - va_sq_R) ** 2 + 4.0 * a_sq_R * vat_sq_R, 0.0)
    cf_L = mx.sqrt(mx.maximum(0.5 * (a_sq_L + va_sq_L + mx.sqrt(disc_L)), 0.0))
    cf_R = mx.sqrt(mx.maximum(0.5 * (a_sq_R + va_sq_R + mx.sqrt(disc_R)), 0.0))

    SL = mx.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = mx.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = mx.maximum(SR, SL + TINY)

    # --- Physical flux F(U) using conserved energy E directly ---
    # This is the key difference from HLLS: energy flux is E-based (conservative),
    # not entropy-reconstructed. More accurate for energy conservation.
    def _pflux(U, rho, inv_r, vn, p):
        Bt1 = U[ib_t1]
        Bt2 = U[ib_t2]
        vt1 = U[im_t1] * inv_r
        vt2 = U[im_t2] * inv_r
        Bn = U[ib_n]
        B2 = Bn ** 2 + Bt1 ** 2 + Bt2 ** 2
        pt = p + 0.5 * B2
        vB = vn * Bn + vt1 * Bt1 + vt2 * Bt2
        E = U[IEN]  # conserved total energy (NOT reconstructed from entropy)

        F_dn = rho * vn
        F_sr = U[ISR] * vn
        F_en = (E + pt) * vn - Bn * vB  # conservative energy flux
        F_ee = U[IEE] * vn if U.shape[0] > IEE else mx.zeros_like(F_dn)

        slots = [None] * NVAR
        slots[IDN] = F_dn
        # Normal momentum: rho*vn^2 + p_tot - Bn^2
        slots[im_n] = rho * vn * vn + pt - Bn * Bn
        # Transverse momenta: rho*vn*vt - Bn*Bt
        slots[im_t1] = rho * vn * vt1 - Bn * Bt1
        slots[im_t2] = rho * vn * vt2 - Bn * Bt2
        slots[IEN] = F_en
        slots[ISR] = F_sr
        # Induction: vn*Bt - vt*Bn (Bn flux = 0)
        slots[ib_n] = mx.zeros_like(F_dn)
        slots[ib_t1] = vn * Bt1 - vt1 * Bn
        slots[ib_t2] = vn * Bt2 - vt2 * Bn
        slots[IEE] = F_ee

        # Fill any remaining None slots (safety for index aliasing)
        for k in range(NVAR):
            if slots[k] is None:
                slots[k] = mx.zeros_like(F_dn)
        return mx.stack(slots, axis=0)

    FL = _pflux(QL, rho_L, inv_rL, vn_L, p_L)
    FR = _pflux(QR, rho_R, inv_rR, vn_R, p_R)

    # --- HLL combination ---
    inv_dS = 1.0 / mx.maximum(SR - SL, TINY)
    F_hll = (SR * FL - SL * FR + SL * SR * (QR - QL)) * inv_dS

    # Region selection: supersonic-L / subsonic / supersonic-R
    F_out = mx.where(SL >= 0.0, FL, mx.where(SR <= 0.0, FR, F_hll))

    # Zero normal B flux (divergence-free constraint)
    F_zero = mx.zeros_like(F_out[0:1])
    parts = [F_out[i : i + 1] if i != ib_n else F_zero for i in range(NVAR)]
    F_out = mx.concatenate(parts, axis=0)

    # --- Branchless NaN fallback: Lax-Friedrichs ---
    # Always compute LF (costs ~20% extra flops) to avoid GPU->CPU sync
    # from `mx.any(nans)` that would stall the pipeline.
    S_max = mx.maximum(mx.abs(SL), mx.abs(SR))
    F_LF = 0.5 * (FL + FR) - 0.5 * S_max * (QR - QL)
    is_bad = mx.isnan(F_out) | mx.isinf(F_out)
    F_out = mx.where(is_bad, F_LF, F_out)

    return F_out
```

### Wiring into `compute_fluxes` (mlx_riemann.py:552-558)

```diff
-    if riemann in ("hll", "hll_cpu"):
+    if riemann == "hll_cpu":
         if dim == 1:
             QL_t = mx.transpose(QL, axes=[0, 2, 1])
             QR_t = mx.transpose(QR, axes=[0, 2, 1])
             F_t = _hll_flux(QL_t, QR_t, gamma=gamma, dim=1)
             return mx.transpose(F_t, axes=[0, 2, 1])
         return _hll_flux(QL, QR, gamma=gamma, dim=dim)
+
+    if riemann == "hll":
+        if dim == 1:
+            QL_t = mx.transpose(QL, axes=[0, 2, 1])
+            QR_t = mx.transpose(QR, axes=[0, 2, 1])
+            F_t = _hll_flux_gpu(QL_t, QR_t, gamma=gamma, dim=1)
+            return mx.transpose(F_t, axes=[0, 2, 1])
+        return _hll_flux_gpu(QL, QR, gamma=gamma, dim=dim)
```

Same change in `_compute_fluxes_4d` (line 611): route `"hll"` to `_hll_flux_gpu`,
add `"hll_cpu"` branch for the NumPy fallback.

---

## 2. mx.eval() Consolidation Diff

Current `step()` in mlx_solver.py has 9 `mx.eval()` calls (lines 660, 672, 684, 696,
701, 707, 712, 724, 767). Each forces a GPU sync. Analysis:

| Line | Current eval | Can remove? | Rationale |
|------|-------------|-------------|-----------|
| 660 | After resistive half-step | **Yes** | Ghost pad reads from U but through MLX ops, not np.asarray. Lazy eval chains. |
| 672 | After ghost padding | **Yes** | RK step reads U through MLX ops. Let lazy eval handle it. |
| 684 | After hyperbolic step | **Keep** | Hyperbolic step is the largest graph. Evaluate here to bound peak memory. |
| 696 | After div(B) cleaning | **Merge with 684** | Dedner/Powell is small — chain with hyperbolic eval. |
| 701 | After resistive 2nd half | **Yes** | Conduction reads U through MLX ops. |
| 707 | After conduction | **Yes** | Viscosity reads U through MLX ops. |
| 712 | After viscosity | **Yes** | Hall reads U through MLX ops. |
| 724 | After Hall MHD | **Merge** | Single eval after all operator-split steps. |
| 767 | After species advection | **Keep** | Species state is used by external code (engine coupling). |

**Proposed: 9 evals -> 3 evals.** Expected 1.1-1.2x from reduced sync overhead.

```diff
 # mlx_solver.py step() method

         # -- 3.1. Strang split: first half-step resistive diffusion --
         if _eta_arg is not None:
             U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)
-            mx.eval(U)

         # -- 3.2. Electrode BC (ghost-cell padding) --
         ...
         if _ghost_active:
             U, grid_for_rk = self._pad_electrode_ghost(U, current)
-            mx.eval(U)

         # -- 4. Hyperbolic step --
         ...
         U = step_fn(U, grid_for_rk, dt, ...)
-        mx.eval(U)

         # -- 4.1. Strip ghost cells --
         if _ghost_active:
             U = self._strip_ghost(U, self._GHOST_NG)

         # -- 4.5. div(B) control --
         if self._use_ct and ...:
             U = self._apply_ct_correction(U, dt)
         if self._enable_dedner or self._enable_powell:
             U = self._apply_divb_cleaning(U, dt)
-            mx.eval(U)
+        # Single eval after the entire hyperbolic+divB pipeline
+        mx.eval(U)

         # -- 5. Strang split: second half-step resistive diffusion --
         if _eta_arg is not None:
             U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)
-            mx.eval(U)

         # -- 6. Braginskii conduction --
         if self.enable_braginskii_conduction:
             ...
             U = self._do_thermal_conduction(U, dt, kappa)
-            mx.eval(U)

         # -- 6.5. Braginskii viscosity --
         if self.enable_braginskii_viscosity:
             U = self._do_braginskii_viscosity(U, dt)
-            mx.eval(U)

         # -- 6.6. Hall MHD --
         if self.enable_hall and ...:
             ...
             U = apply_hall_mhd(U, dt, ...)
-            mx.eval(U)
+        # Single eval after all operator-split physics
+        mx.eval(U)

         # -- 6.7. Species advection --
         if self._species_mgr is not None and ...:
             ...
             mx.eval(self._Y)  # Keep: external state coupling
```

**Caveat**: Eval 672 (ghost padding) can only be removed after OPT-3 lands, because
`_pad_electrode_ghost` calls `np.asarray()` internally. Remove 660 + 696-724 now
(safe, 9->4 evals), then 672 after OPT-3 (4->3 evals).

---

## 3. Bremsstrahlung Log-Space Prototype (~25 LOC)

Replaces mlx_sources.py:445-455 (the NumPy float64 detour). The problem: `_BREM_COEFF
= 1.42e-40` is subnormal in float32 (min normal ~1.175e-38) and flushes to zero.

Solution: compute entirely in log-space where `log(1.42e-40) = -91.76` is well within
float32 range (float32 handles +-126 in log-space).

```python
def _bremsstrahlung_logspace(
    rho: mx.array,
    p: mx.array,
    gamma: float,
    Z_eff: float | mx.array = 1.0,
    gaunt_factor: float = 1.2,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Compute bremsstrahlung power Q_rad in pure MLX via log-space arithmetic.

    Q_rad = BREM_COEFF * g_ff * Z * ne^2 * sqrt(Te)

    In log-space:
        log(Q_rad) = log(BREM_COEFF) + log(g_ff) + log(Z)
                     + 2*log(ne) + 0.5*log(Te)

    All intermediate values stay in [-100, +100] range, well within float32.

    Args:
        rho: Mass density (nr, nz), float32, already floored.
        p: Pressure (nr, nz), float32, already floored.
        gamma: Adiabatic index.
        Z_eff: Effective ion charge (scalar or spatial array).
        gaunt_factor: Free-free Gaunt factor.
        ion_mass: Ion mass [kg].

    Returns:
        Q_rad: Volumetric radiation power [W/m^3], shape (nr, nz), float32.
    """
    _KBOLTZ = 1.380649e-23

    # Pre-computed log constants (evaluated once at import, not per call)
    _LOG_BREM = mx.array(-91.7621, dtype=mx.float32)  # log(1.42e-40)
    _LOG_GFF = mx.array(float(np.log(gaunt_factor)), dtype=mx.float32)
    _LOG_Z = mx.array(float(np.log(max(float(Z_eff) if isinstance(Z_eff, (int, float)) else 1.0, 1e-30))), dtype=mx.float32)
    _LOG_MI = float(np.log(ion_mass))  # log(3.34e-27) ~ -61.07

    # ne = rho / ion_mass  ->  log(ne) = log(rho) - log(ion_mass)
    log_rho = mx.log(mx.maximum(rho, 1e-30))
    log_ne = log_rho - _LOG_MI

    # Te = p * ion_mass / (2 * rho * kB)
    # log(Te) = log(p) + log(ion_mass) - log(2) - log(rho) - log(kB)
    _LOG_2KB = float(np.log(2.0 * _KBOLTZ))  # log(2*kB) ~ -52.17
    log_p = mx.log(mx.maximum(p, 1e-30))
    log_Te = log_p + _LOG_MI - _LOG_2KB - log_rho
    # Floor Te at 1 K in log-space = 0.0
    log_Te = mx.maximum(log_Te, 0.0)

    # log(Q_rad) = log(BREM) + log(g_ff) + log(Z) + 2*log(ne) + 0.5*log(Te)
    log_Q = _LOG_BREM + _LOG_GFF + _LOG_Z + 2.0 * log_ne + 0.5 * log_Te

    # Handle spatial Z_eff array (if not scalar, add log(Z_eff) per-cell)
    if isinstance(Z_eff, mx.array):
        log_Q = log_Q - _LOG_Z + mx.log(mx.maximum(Z_eff, 1e-30))

    # exp back to linear. Clamp log_Q to prevent overflow (exp(88) ~ 3.4e38).
    log_Q = mx.minimum(log_Q, 80.0)
    Q_rad = mx.exp(log_Q)

    return Q_rad
```

### Integration into `apply_bremsstrahlung` (mlx_sources.py:438-455)

Replace the NumPy detour block:

```diff
-    # Compute Q_rad in float64 via NumPy: 1.42e-40 is subnormal in float32 and
-    # would flush to zero if left in the MLX float32 graph.
-    rho_np = np.asarray(rho).astype(np.float64)
-    p_np = np.asarray(p).astype(np.float64)
-    ne_np = rho_np / ion_mass
-    # T = p*m_i/(2*rho*kB) for fully ionized Z=1 plasma (n_e + n_i = 2*n_i)
-    Te_np = np.maximum(p_np * ion_mass / (2.0 * rho_np * _KBOLTZ), 1.0)
-    Q_rad_np = (_BREM_COEFF * gaunt_factor * Z_eff * ne_np * ne_np * np.sqrt(Te_np)).astype(
-        np.float32
-    )
-    Q_rad = mx.array(Q_rad_np)
+    Q_rad = _bremsstrahlung_logspace(
+        rho, p, gamma,
+        Z_eff=Z_eff if isinstance(Z_eff, mx.array) else float(Z_eff),
+        gaunt_factor=gaunt_factor,
+        ion_mass=ion_mass,
+    )
```

### Validation

Log-space and direct float64 must agree to within ~1e-4 relative (log-space
introduces ~1e-5 relative error from float32 intermediate log values). Test with
random (rho, p) spanning 4 orders of magnitude, compare against NumPy float64 reference.

---

## 4. mx.compile() Candidate Analysis

### Candidate 1: `_hll_flux_gpu` (this prototype)

- **Calls/step**: 6 (2 dims x 3 RK stages)
- **Op count**: ~50 elementwise MLX ops per call
- **Expected speedup**: 1.2-1.3x on the Riemann portion (~0.06 ms saved per call)
- **Compilable?**: YES — pure elementwise, no Python control flow on data, no `np.asarray`.
  The `if dim == 0/1/2` branching is on a Python int (constant per call), which `mx.compile`
  handles via tracing.
- **Gotcha**: The inner `_pflux` closure references `ib_n`, `ib_t1` etc. from the outer scope.
  These are Python ints, so mx.compile traces them as constants. BUT if `_hll_flux_gpu` is
  compiled at module level, `dim` must be fixed. Solution: compile three variants.

```python
# Compile three dim-specialized versions at first use
_HLL_GPU_COMPILED = {}

def _get_hll_gpu_compiled(dim: int):
    if dim not in _HLL_GPU_COMPILED:
        def _hll_dim(QL, QR, gamma):
            return _hll_flux_gpu(QL, QR, gamma, dim=dim)
        _HLL_GPU_COMPILED[dim] = _compile_if_available(_hll_dim)
    return _HLL_GPU_COMPILED[dim]
```

### Candidate 2: `_clamp_reconstructed` (mlx_riemann.py:60-78)

- **Calls/step**: 6 (called in `compute_fluxes` per dim per stage)
- **Op count**: ~12 ops (maximum, concatenate, slice)
- **Expected speedup**: 1.02x (tiny function, overhead-dominated)
- **Compilable?**: YES — pure elementwise with concatenation.
- **Gotcha**: None. Trivial to wrap.

```python
_clamp_reconstructed = _compile_if_available(_clamp_reconstructed_impl)
```

### Candidate 3: `fast_magnetosonic` (mlx_primitives.py:233)

- **Calls/step**: 1 (CFL computation, but also implicitly in stage post-processing)
- **Op count**: ~20 ops (sqrt, maximum, division, Boris correction)
- **Expected speedup**: 1.05x on CFL computation
- **Compilable?**: YES — pure elementwise.
- **Gotcha**: `dim` parameter selects Bn component — same solution as HLL (trace per dim).

```python
_FAST_MS_COMPILED = {}

def fast_magnetosonic_compiled(rho, p, Br, Bz, Bt, gamma, dim):
    if dim not in _FAST_MS_COMPILED:
        def _fms(rho, p, Br, Bz, Bt, gamma):
            return fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim)
        _FAST_MS_COMPILED[dim] = _compile_if_available(_fms)
    return _FAST_MS_COMPILED[dim](rho, p, Br, Bz, Bt, gamma)
```

### Candidate 4: `_minmod` + `_mc_limit` (mlx_reconstruction.py:69, 86)

- **Calls/step**: 6 x NVAR = 60 (PLM reconstruction, 10 vars x 2 dims x 3 stages)
- **Op count**: 5-8 ops each (sign, minimum, abs, where, multiply)
- **Expected speedup**: 1.02-1.05x cumulative (many small calls)
- **Compilable?**: YES — pure elementwise, no control flow.
- **Gotcha**: These are already called inside `plm_reconstruct` which itself is not compiled.
  Compiling the limiters alone gives marginal benefit. Better to compile the entire
  `plm_reconstruct` but it has Python control flow (`if limiter == "mc"`) that
  mx.compile can't handle. Solution: compile the limiter-specific PLM path.

```python
def _plm_mc_reconstruct(Q, axis):
    """PLM with MC limiter -- compilable (no Python branching on data)."""
    Qm1 = _take(Q, slice(None, -2), axis)
    Q0 = _take(Q, slice(1, -1), axis)
    Qp1 = _take(Q, slice(2, None), axis)
    dL = Q0 - Qm1
    dR = Qp1 - Q0
    slope = _mc_limit(dL, dR)
    QL = Q0 + 0.5 * slope
    QR = Q0 - 0.5 * slope
    return QL, QR

_plm_mc_compiled = _compile_if_available(_plm_mc_reconstruct)
```

### Candidate 5: `_bremsstrahlung_logspace` (this prototype)

- **Calls/step**: 1 (operator-split source term)
- **Op count**: ~15 ops (log, exp, multiply, add, maximum)
- **Expected speedup**: 1.3x on bremsstrahlung (mainly from eliminating CPU round-trip,
  but mx.compile adds fusion benefit on top)
- **Compilable?**: PARTIAL — the `isinstance(Z_eff, mx.array)` check is Python control
  flow. Solution: split into scalar-Z and array-Z variants, compile each.

```python
def _brem_logspace_scalar_z(rho, p, log_brem, log_gff, log_z, log_mi, log_2kb):
    log_rho = mx.log(mx.maximum(rho, 1e-30))
    log_ne = log_rho - log_mi
    log_p = mx.log(mx.maximum(p, 1e-30))
    log_Te = mx.maximum(log_p + log_mi - log_2kb - log_rho, 0.0)
    log_Q = log_brem + log_gff + log_z + 2.0 * log_ne + 0.5 * log_Te
    return mx.exp(mx.minimum(log_Q, 80.0))

_brem_compiled = _compile_if_available(_brem_logspace_scalar_z)
```

### Summary Table

| # | Function | Calls/Step | Compilable | Expected Speedup | Priority |
|---|----------|-----------|------------|------------------|----------|
| 1 | `_hll_flux_gpu` | 6 | Yes (per-dim) | 1.2-1.3x on Riemann | **High** |
| 2 | `_clamp_reconstructed` | 6 | Yes | 1.02x | Low |
| 3 | `fast_magnetosonic` | 1 | Yes (per-dim) | 1.05x on CFL | Medium |
| 4 | `_plm_mc_reconstruct` | 6 | Yes (fused) | 1.05x on reconstruction | Medium |
| 5 | `_brem_logspace` | 1 | Yes (scalar-Z path) | 1.3x on brem (already fast) | Low |

**Recommendation**: Prioritize #1 (HLL GPU compile) — highest call count x largest op
graph = most fusion opportunity. #4 is the best bang-for-buck among the others because
it fuses the limiter + reconstruction into one compiled kernel.

---

## Implementation Order

1. Add `_hll_flux_gpu()` to mlx_riemann.py, wire into `compute_fluxes`
2. Add `_bremsstrahlung_logspace()` to mlx_sources.py
3. Remove evals 660, 701, 707, 712, 724 from mlx_solver.py
4. Wrap `_hll_flux_gpu` with `_compile_if_available` (per-dim variants)
5. Wrap `_clamp_reconstructed` with `_compile_if_available`
6. Validate: `pytest tests/test_mlx_*.py -v` (471 tests), Sod L1 parity < 1e-3
