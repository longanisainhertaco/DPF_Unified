# RKL2 Super-Timestepping: Wiring into MLX Solver

Research document for replacing the CPU Thomas tridiagonal solver with GPU-resident
RKL2 super-timestepping for resistive diffusion and thermal conduction in the MLX
MHD solver.

**Date**: 2026-03-26
**Status**: Research only -- no source modifications

---

## 1. What Does mlx_sts.py Currently Implement?

`src/dpf/metal/mlx_sts.py` is a **complete, functional** RKL2 integrator for MLX.
It is not a stub. It provides two public functions:

### `rkl2_step_mlx(U, rhs_fn, dt, s_stages=8) -> mx.array`
A generic s-stage RKL2 integrator that advances any parabolic operator on MLX arrays.
The caller provides a callable `rhs_fn(U) -> dU/dt` representing the diffusion operator
L(U). The integrator applies the full Meyer et al. (2012) recursion:

```
Y_0 = U
Y_1 = Y_0 + mu_tilde_1 * dt * L(Y_0)
Y_j = mu_j*Y_{j-1} + nu_j*Y_{j-2} + (1 - mu_j - nu_j)*Y_0
      + mu_tilde_j * dt * L(Y_{j-1}) + gamma_tilde_j * dt * L(Y_0)
U_new = Y_s
```

All stages run on MLX arrays (Metal GPU). Coefficients are computed once via
`dpf.fluid.super_time_step.rkl2_coefficients(s)`, which uses the exact Chebyshev
polynomial recursion from Meyer, Balsara & Aslam (2012). Coefficients are cast to
Python floats before use (zero overhead).

### `compute_sts_stages(dt_mhd, dt_parabolic, max_stages=20) -> int`
Computes the minimum number of RKL2 stages needed so that the super-step covers
the MHD timestep: `s >= ceil(sqrt(dt_mhd / (0.25 * dt_parabolic)))`, clamped to
[2, max_stages].

### What's Missing
The module has no knowledge of specific physics operators (resistive diffusion,
thermal conduction). It is a generic integrator that requires the caller to provide
a `rhs_fn`. The wiring task is to write appropriate `rhs_fn` closures for each
physics operator and call `rkl2_step_mlx` from the solver.

---

## 2. The RKL2 Algorithm

### Background
RKL2 (Runge-Kutta-Legendre, 2nd order) is a stabilized explicit Runge-Kutta method
designed for parabolic PDEs. It extends the explicit stability region along the
negative real axis by using s stages whose coefficients are derived from Chebyshev
polynomial zeros. Unlike implicit methods (Thomas, ADI, multigrid), it requires no
matrix solves -- only repeated evaluations of the explicit RHS operator L(U).

### Key Reference
Meyer, Balsara & Aslam, "A stabilized Runge-Kutta-Legendre method for explicit
super-time-stepping of parabolic and mixed equations," JCP 231:2963-2988 (2012).

### Stability Region
The RKL2 stability region extends to `|z| <= z_max` where `z = lambda * dt` and
lambda is the most negative eigenvalue of the diffusion operator. For s stages:

    z_max ~ 0.55 * s^2

Since standard explicit Euler is stable for `|z| <= 2`, the acceleration factor is:

    acceleration = z_max / 2 ~ 0.275 * s^2

With a safety factor of 0.9, the practical limit is:

    dt_super <= 0.25 * s^2 * dt_explicit

where `dt_explicit = dx^2 / (2 * D_max)` is the standard parabolic CFL.

### Second-Order Accuracy
RKL2 achieves 2nd-order temporal accuracy through the choice of b_j coefficients:
`b_j = T''_j(w0) / (T'_j(w0))^2` with w0 = 1 + 2/(s^2+s). This ensures the
stability polynomial G(z) satisfies G(0)=1, G'(0)=1, G''(0)=1.

### Athena++ Implementation
Athena++ implements RKL2 in `src/task_list/sts_task_list.cpp`. Key details:
- Used for resistive diffusion, viscosity, and thermal conduction
- Stages computed dynamically from the ratio dt_hyperbolic / dt_parabolic
- Coefficients cached per stage count
- Applied as an operator-split step after the hyperbolic integrator
- Same approach as our `mlx_sts.py` -- generic recursion with pluggable RHS

---

## 3. How RKL2 Replaces the Thomas Solver

### Current Implementation (mlx_transport.py)

The current resistive diffusion (`apply_resistive_diffusion`) and thermal conduction
(`apply_thermal_conduction`) both:

1. **Convert MLX arrays to NumPy float64** -- full GPU-to-CPU transfer
2. **Loop over grid columns/rows** -- Python `for ir in range(nr)` / `for iz in range(nz)`
3. **Build tridiagonal systems** -- `_build_diffusion_system()` or `_build_cylindrical_diffusion_system()`
4. **Solve via Thomas algorithm** -- sequential forward/back substitution per pencil
5. **Convert back to MLX** -- CPU-to-GPU transfer

This is a fully implicit, unconditionally stable solve. But it is entirely CPU-bound
with O(nr*nz) Python loop iterations and two full GPU<->CPU round-trips.

### RKL2 Replacement Strategy

Replace the implicit Thomas solve with explicit RKL2 stages that run entirely on MLX.

#### For Resistive Diffusion (`apply_resistive_diffusion`)

The diffusion equation is:
```
dB/dt = (eta/mu_0) * Laplacian(B)  (Cartesian)
dB/dt = (eta/mu_0) * [(1/r) d/dr(r dB/dr) + d^2B/dz^2]  (cylindrical)
```

The `rhs_fn` for RKL2 computes the explicit Laplacian of B using MLX array operations:

```python
def resistive_rhs(B_component: mx.array) -> mx.array:
    """Explicit cylindrical diffusion: (eta/mu0) * [(1/r) d/dr(r dB/dr) + d^2B/dz^2]"""
    alpha = eta / MU_0  # diffusivity, shape (nr, nz) or scalar

    # z-direction: standard second-order central difference
    d2B_dz2 = (B[..., 2:] - 2*B[..., 1:-1] + B[..., :-2]) / dz^2
    # pad with Neumann BC (zero-flux)

    # r-direction: cylindrical (1/r) d/dr(r * alpha * dB/dr)
    # Interface values at r_{i+1/2}
    r_face_p = r_cell + 0.5*dr
    r_face_m = maximum(r_cell - 0.5*dr, 0)
    alpha_face_p = 0.5*(alpha[:-1,:] + alpha[1:,:])
    flux_p = r_face_p * alpha_face_p * (B[1:,:] - B[:-1,:]) / dr
    flux_m = r_face_m * alpha_face_m * (B[1:,:] - B[:-1,:]) / dr
    L_r = (flux_p - flux_m) / (r_cell * dr)

    return L_r + alpha * d2B_dz2
```

Then call:
```python
from dpf.metal.mlx_sts import rkl2_step_mlx, compute_sts_stages

dt_explicit = min(dr, dz)**2 / (2.0 * alpha_max)
s = compute_sts_stages(dt, dt_explicit)
B_new = rkl2_step_mlx(B, resistive_rhs, dt, s_stages=s)
```

#### For Thermal Conduction (`apply_thermal_conduction`)

The anisotropic conduction equation is:
```
dT/dt = div(chi * grad(T))
```
where chi is the effective thermal diffusivity (direction-weighted by B-field).
The same pattern applies: write an explicit `rhs_fn` that computes the Laplacian
with anisotropic chi on MLX, then pass to `rkl2_step_mlx`.

### Mapping: Implicit Thomas -> Explicit RKL2

| Aspect | Thomas (current) | RKL2 (proposed) |
|--------|-----------------|-----------------|
| Matrix assembly | `_build_diffusion_system()` per pencil | Not needed |
| Linear solve | Thomas forward/back sweep per pencil | Not needed |
| RHS evaluations | 0 (implicit) | s evaluations of L(U) |
| Python loops | `for ir in range(nr): for iz in range(nz):` | None (vectorized MLX) |
| GPU utilization | 0% (all CPU) | 100% (all MLX) |
| Memory transfer | 2 full GPU<->CPU round-trips | 0 |
| Stability | Unconditional | Conditional: dt <= 0.25*s^2*dt_explicit |
| Temporal order | 1st (backward Euler) | 2nd (RKL2) |
| Spatial order | 2nd (central difference) | 2nd (same stencil) |

---

## 4. Stability Limit and Stage Count for DPF Conditions

### Parabolic CFL for Resistive Diffusion

The explicit CFL limit for resistive diffusion is:
```
dt_explicit = dx^2 * mu_0 / (2 * eta_max)
```

For typical DPF conditions:

| Condition | eta [Ohm*m] | dr [m] | dt_explicit [s] | dt_mhd [s] | s needed |
|-----------|-------------|--------|-----------------|-------------|----------|
| Pre-pinch (Spitzer, Te~10eV) | 1e-4 | 1e-3 | 6.3e-9 | 1e-8 | 3 |
| Pre-pinch (Spitzer, Te~10eV) | 1e-4 | 5e-4 | 1.6e-9 | 5e-9 | 4 |
| Pinch (Te~1keV, Spitzer) | 1e-7 | 5e-4 | 1.6e-6 | 1e-9 | 2 (trivial) |
| Sheath (Lee-More, warm dense) | 1e-3 | 1e-3 | 6.3e-10 | 1e-8 | 8 |
| Sheath (Lee-More, warm dense) | 1e-2 | 1e-3 | 6.3e-11 | 1e-8 | 13 |
| Extreme (eta_cap=1e-2, fine grid) | 1e-2 | 5e-4 | 1.6e-11 | 1e-8 | 16 |

### Parabolic CFL for Thermal Conduction

The explicit CFL for conduction is:
```
dt_explicit = dx^2 * (1.5 * n_e * k_B) / (2 * kappa_max)
```

Braginskii parallel kappa for deuterium at Te ~ 1 keV, n_e ~ 1e25 m^-3:
```
kappa_par ~ 3.16 * n_e * k_B^2 * Te * tau_e / m_e ~ 1e6 W/(m*K)
```

| Condition | kappa [W/(m*K)] | chi [m^2/s] | dr [m] | dt_explicit [s] | s needed (for dt_mhd=1e-9) |
|-----------|----------------|-------------|--------|-----------------|---------------------------|
| Moderate conduction | 1e3 | ~5e2 | 1e-3 | 1e-9 | 2 |
| Strong conduction (pinch) | 1e6 | ~5e5 | 1e-3 | 1e-12 | 20 (max cap) |
| Flux-limited conduction | 1e4 | ~5e3 | 1e-3 | 1e-10 | 7 |
| Flux-limited + fine grid | 1e4 | ~5e3 | 5e-4 | 2.5e-11 | 13 |

### Summary
- Resistive diffusion: s = 2-16 stages depending on sheath conditions. Typically 4-8.
- Thermal conduction: s = 2-20 stages. Flux-limited kappa keeps it manageable (7-13).
  Without flux limiting, kappa at pinch requires s > 20, hitting the cap.
- The existing `max_stages=20` cap in `compute_sts_stages` is appropriate. At s=20,
  the acceleration is 100x over explicit Euler, covering most DPF regimes.
- When s hits the cap, the super-step is smaller than dt_mhd. Options:
  (a) sub-cycle the RKL2 step: N_sub = ceil(dt_mhd / dt_super_max)
  (b) accept the cap and let the outer Strang split handle it
  (c) fall back to implicit Thomas for extreme cases

---

## 5. Required Changes in mlx_solver.py

### 5.1. New Module: `mlx_sts_operators.py` (recommended)

Create a new file providing the physics-specific RHS closures:

```python
# src/dpf/metal/mlx_sts_operators.py

def make_resistive_rhs(eta, dr, dz, r_cell, coordinates="cylindrical"):
    """Return an rhs_fn(B) -> dB/dt for resistive diffusion on MLX."""
    alpha = eta / MU_0  # mx.array or scalar
    ...
    def rhs_fn(B):
        # Vectorized 2D Laplacian on MLX arrays
        ...
        return L_B
    return rhs_fn

def make_conduction_rhs(chi_r, chi_z, dr, dz, r_cell, coordinates="cylindrical"):
    """Return an rhs_fn(T) -> dT/dt for anisotropic thermal conduction on MLX."""
    ...
    def rhs_fn(T):
        # Vectorized 2D anisotropic diffusion on MLX arrays
        ...
        return L_T
    return rhs_fn
```

### 5.2. Modified `_do_resistive_diffusion` in mlx_solver.py

```python
def _do_resistive_diffusion(self, U, dt, eta):
    from dpf.metal.mlx_sts import rkl2_step_mlx, compute_sts_stages
    from dpf.metal.mlx_sts_operators import make_resistive_rhs

    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, self.gamma)
    alpha_max = float(mx.max(eta).item()) / MU_0
    dt_explicit = min(self._grid.dr, self._grid.dz)**2 / (2.0 * max(alpha_max, 1e-30))
    s = compute_sts_stages(dt, dt_explicit)

    rhs_fn = make_resistive_rhs(eta, self._grid.dr, self._grid.dz,
                                 self._grid.r_cell, self.coordinates)

    # Diffuse each B component independently
    Br_new = rkl2_step_mlx(Br, rhs_fn, dt, s_stages=s)
    Bz_new = rkl2_step_mlx(Bz, rhs_fn, dt, s_stages=s)
    Bt_new = rkl2_step_mlx(Bt, rhs_fn, dt, s_stages=s)

    # Ohmic heating from energy conservation
    dB_sq = (Br_new-Br)**2 + (Bz_new-Bz)**2 + (Bt_new-Bt)**2
    Q_ohmic = 0.5 * dB_sq * MU_0
    p_new = mx.maximum(p + (self.gamma - 1.0) * Q_ohmic, P_FLOOR)

    # Rebuild conserved state...
```

### 5.3. Modified `_do_thermal_conduction` in mlx_solver.py

Same pattern: compute chi from kappa/n_e/k_B, build `rhs_fn`, call `rkl2_step_mlx`.
The anisotropy weighting (b_r^2, b_z^2) is computed once and baked into the closure.

### 5.4. Configuration

Add a solver constructor parameter:
```python
diffusion_method: str = "rkl2"  # "rkl2" (GPU) or "thomas" (CPU implicit)
```
Default to RKL2 but allow Thomas fallback for regression testing and validation.

### 5.5. No Changes Needed To

- `mlx_sts.py` -- already complete and generic
- `mlx_transport.py` -- keep as-is for Thomas fallback path
- `super_time_step.py` -- coefficient computation already shared
- Strang splitting in `step()` -- same half-step/full-step structure

---

## 6. Expected GPU Speedup

### Current Performance Profile (Thomas)

For a 128x128 grid with resistive diffusion:
- GPU->CPU transfer: ~0.1 ms (256 KB)
- Python loops: 128 * 128 = 16,384 Thomas solves (z-sweep) + 16,384 (r-sweep)
- Per-Thomas: ~1 us (128-element tridiagonal)
- Total Thomas time: ~33 ms per B component, ~100 ms for Br+Bz+Bt
- CPU->GPU transfer: ~0.1 ms
- **Total**: ~100 ms per diffusion call, ~200 ms per timestep (Strang: 2 calls)

For thermal conduction: similar cost (128*128 Thomas solves for Te + Ti, r and z sweeps).

### RKL2 Performance Estimate

Each RKL2 stage evaluates L(U) = vectorized 2D stencil on MLX:
- One stencil evaluation on 128x128: ~0.01 ms on M3 Pro Metal
- s stages * 3 components: s * 3 * 0.01 ms
- For s=8: 8 * 3 * 0.01 = 0.24 ms
- For s=16: 16 * 3 * 0.01 = 0.48 ms

### Speedup Estimate

| Grid | Thomas (CPU) | RKL2 s=8 (GPU) | RKL2 s=16 (GPU) | Speedup |
|------|-------------|----------------|-----------------|---------|
| 64x64 | ~25 ms | ~0.06 ms | ~0.12 ms | 200-400x |
| 128x128 | ~100 ms | ~0.24 ms | ~0.48 ms | 200-400x |
| 256x256 | ~400 ms | ~0.96 ms | ~1.9 ms | 200-400x |
| 512x512 | ~1.6 s | ~3.8 ms | ~7.7 ms | 200-400x |

The speedup is dominated by eliminating the Python loops and CPU<->GPU transfers,
not by the number of RKL2 stages. Even at s=20, the GPU time is negligible compared
to the CPU Thomas solver.

### Fraction of Total Step Time

Currently, transport (resistive + conduction) is the dominant cost at high resolution:
- 128x128 MHD step (hyperbolic): ~5-10 ms on MLX
- 128x128 transport (Thomas): ~200-400 ms
- Transport is 95%+ of wall time

After RKL2:
- 128x128 MHD step: ~5-10 ms
- 128x128 transport (RKL2): ~0.5-1 ms
- Transport drops to <10% of wall time

This removes transport as the bottleneck entirely.

---

## 7. Risks and Mitigations

### 7.1. Accuracy vs Implicit Thomas

| Aspect | Thomas (backward Euler) | RKL2 |
|--------|------------------------|------|
| Temporal order | 1st | 2nd |
| Spatial order | 2nd (central diff) | 2nd (central diff) |
| Damping | Over-damps (implicit) | Minimal damping within stability region |

RKL2 is actually **more accurate** than backward Euler (2nd vs 1st order in time).
The Thomas solver's main advantage is unconditional stability, but this comes with
artificial numerical dissipation from the implicit scheme.

**Risk**: RKL2 may preserve fine-scale features that Thomas dampened. If these
features are noise, the solution may be less smooth.

**Mitigation**: The Ohmic heating calculation (energy conservation from delta-B)
is identical. The spatial discretization is the same 2nd-order stencil. Differences
should be confined to temporal accuracy improvements.

### 7.2. Stability When s Hits the Cap

When eta is very large (cold sheath, eta ~ 1e-2 Ohm*m) and the grid is fine,
`compute_sts_stages` may return `max_stages=20` but the actual dt_parabolic
requires more stages. In this case, the super-step is smaller than dt_mhd.

**Mitigation**: Sub-cycle the RKL2 call:
```python
dt_super_max = rkl2_stability_limit(s, dt_explicit)
n_sub = max(1, int(math.ceil(dt / dt_super_max)))
dt_sub = dt / n_sub
for _ in range(n_sub):
    B = rkl2_step_mlx(B, rhs_fn, dt_sub, s_stages=s)
```
Even with n_sub=5, the GPU cost is negligible vs one Thomas pass.

### 7.3. Operator-Split Interaction

The MLX solver uses Strang splitting for resistive diffusion:
```
half-step resistive (dt/2) -> hyperbolic (dt) -> half-step resistive (dt/2)
```

Thermal conduction is applied as a full step after the hyperbolic update (Lie split).

Switching from implicit to explicit within the same split structure does not change
the overall splitting error (which is O(dt^2) for Strang and O(dt) for Lie). The
RKL2 method is applied to the same sub-step timestep (dt/2 or dt), so the split
interaction is identical.

**Risk**: The implicit Thomas solver handles the full dt/2 in one shot. RKL2 may
need multiple internal sub-steps if s hits the cap. This changes the effective
operator ordering within the sub-step from a single implicit solve to multiple
explicit stages.

**Mitigation**: The RKL2 stages are mathematically equivalent to a single 2nd-order
accurate step over the same interval. The sub-cycling (if needed) is also mathematically
correct. No operator-ordering change at the Strang level.

### 7.4. Cylindrical Geometry Correctness

The Thomas solver uses `_build_cylindrical_diffusion_system` which correctly handles
the (1/r) d/dr(r * alpha * dB/dr) operator with face-centered coefficients. The RKL2
`rhs_fn` must implement the same cylindrical Laplacian with the same boundary conditions.

**Risk**: Incorrect cylindrical stencil in the MLX RHS closure.

**Mitigation**: The stencil is straightforward -- second-order central differences
with r-weighted face fluxes. Same formula as `_build_cylindrical_diffusion_system`
but vectorized across the full 2D array. Unit test: compare RKL2 output against
Thomas output on a known diffusion problem. They should agree to O(dt) since Thomas
is 1st-order and RKL2 is 2nd-order; the difference should *decrease* with dt.

### 7.5. Float32 Precision

The Thomas solver operates in float64 (explicit conversion in `apply_resistive_diffusion`).
RKL2 on MLX runs in float32 (MLX default on Metal).

**Risk**: Accumulation of round-off across s stages in float32. Each stage multiplies
by coefficients close to 1.0 and adds small perturbations. For s=16, this is 16
multiply-accumulate operations on the full field.

**Mitigation**:
- The RKL2 coefficients (mu, nu, mu_tilde, gamma_tilde) are computed in float64
  and cast to float scalars. The MLX operations use float32 arrays.
- The critical quantity is the Ohmic heating dB_sq = (B_new - B_old)^2. This is
  a single subtraction of two close values, which is the same cancellation risk
  as in the Thomas solver.
- For DPF, the resistive diffusion is a correction term (not the dominant physics).
  Float32 errors of O(1e-7) relative are acceptable.
- If validation shows precision issues, the `rhs_fn` can upcast to float64 for
  the stencil computation and downcast the result. Cost: ~2x per stage, still
  much faster than Thomas.

### 7.6. Convergence Order Verification

**Plan**: Run a smooth diffusion test (e.g., Gaussian blob on a cylinder) at
multiple resolutions (32x32, 64x64, 128x128, 256x256) and measure the L2 error
against the analytical solution. Verify 2nd-order convergence in both space and time.
Compare against Thomas solver on the same problem.

---

## 8. Implementation Plan (Recommended Sequence)

1. **Create `mlx_sts_operators.py`** (~100 LOC)
   - `make_resistive_rhs()`: cylindrical + Cartesian diffusion operators on MLX
   - `make_conduction_rhs()`: anisotropic thermal conduction on MLX
   - Vectorized 2D stencils, Neumann BCs, r-weighted cylindrical terms

2. **Add `diffusion_method` config** to `MLXMHDSolver.__init__`
   - `"rkl2"` (default) or `"thomas"` (legacy)
   - Stored as `self._diffusion_method`

3. **Modify `_do_resistive_diffusion`** to branch on method
   - RKL2 path: compute stages, build closure, call `rkl2_step_mlx`
   - Thomas path: existing code (unchanged)

4. **Modify `_do_thermal_conduction`** to branch on method
   - Same pattern as resistive diffusion

5. **Write tests** (~50 LOC)
   - Gaussian diffusion convergence test (RKL2 vs analytical)
   - Cross-method parity test (RKL2 vs Thomas, agree to O(dt))
   - Stage count computation test for DPF conditions
   - Full Sod/Brio-Wu regression with RKL2 enabled

6. **Benchmark** (~20 LOC)
   - Wall-time comparison: Thomas vs RKL2 at 64x64, 128x128, 256x256

Total estimated effort: ~200 LOC new code, ~50 LOC test code, ~20 LOC benchmark.
No changes to existing modules except `mlx_solver.py` (branching logic).

---

## 9. References

1. Meyer C.D., Balsara D.S., Aslam T.D., "A stabilized Runge-Kutta-Legendre method
   for explicit super-time-stepping of parabolic and mixed equations," JCP 231:2963 (2012)
2. Vaidya B. et al., "A scalable explicit approach for RHD/RMHD with super-time-stepping,"
   MNRAS 472:3147 (2017) -- PLUTO implementation
3. Alexiades V., Amiez G., Gremaud P.-A., "Super-time-stepping acceleration of explicit
   schemes for parabolic problems," CNME 12:31 (1996)
4. Stone J.M. et al., "Athena++: An Adaptive Mesh Refinement Framework for Astrophysical
   Magnetohydrodynamics," ApJS 249:4 (2020) -- RKL2 in Athena++ STS task list
