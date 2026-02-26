# Verification Module Troubleshooting

**Cross-review by**: xreview-circuit (synthesizing py-circuit-val + phys-circuit-val)
**Files reviewed**: `shock_tubes.py` (700 LOC), `sedov_cylindrical.py` (392 LOC), `orszag_tang.py` (315 LOC), `diffusion_convergence.py` (478 LOC), `cylindrical_convergence.py` (491 LOC)
**Date**: 2026-02-25

---

## CRITICAL Findings

*None.*

---

## HIGH Findings

### ✅ FIXED HIGH-1: Sedov-Taylor alpha=1.0 is approximate; should be ~1.15 for cylindrical gamma=5/3

- **File**: `sedov_cylindrical.py:146`
- **Found by**: phys-circuit-val (BUG-VV-2)
- **Cross-review verdict**: **CONFIRMED** -- alpha affects the reference solution used for error measurement
- **Status**: ✅ FIXED — `alpha = 1.152` (Kamm & Timmes 2007, Table I) with updated comment.

**Description**: The Sedov-Taylor similarity constant `alpha` is hardcoded to `1.0` at line 146. For the 2D cylindrical (nu=2) case with gamma=5/3, the correct value from Kamm & Timmes (LA-UR-07-2849, 2007) is approximately 1.15. The code comments at lines 145-146 acknowledge this: "Kamm & Timmes give alpha ~ 1.0 - 1.15", but then choose the lower bound.

**Evidence**:
```python
# sedov_cylindrical.py:146
alpha = 1.0
```

**Impact**: The reference shock position is underestimated by ~15%, which means:
1. The `relative_error` in `SedovCylindricalResult` is inaccurate -- it compares against a wrong baseline
2. Tests relying on this error metric may pass with a large tolerance (masking solver bugs) or fail spuriously

**Fix applied**:
```python
# For gamma=5/3, nu=2 (cylindrical): alpha ~ 1.152 (Kamm & Timmes 2007, Table I)
alpha = 1.152
```

### ✅ FIXED HIGH-2: Sedov pressure cap silently alters deposited energy

- **File**: `sedov_cylindrical.py:270`
- **Found by**: py-circuit-val (V-SED-2)
- **Cross-review verdict**: **CONFIRMED** -- physical energy is changed without adjusting E0
- **Status**: ✅ FIXED — `E0_actual` recomputed after cap; analytical call now uses `E0_actual`; `E0_requested` and `E0_actual` both stored in metadata.

**Description**: The deposited pressure is capped: `p_deposit = min(p_deposit, 20.0 * p_bg)`. This cap prevents numerical instability from extreme pressure ratios but silently reduces the actual energy deposited below the intended `E0 = 0.1 J`. However, the `sedov_shock_radius_cylindrical()` function at line 356 still uses the original `E0` to compute the analytical shock position.

**Evidence**:
```python
# sedov_cylindrical.py:270
p_deposit = min(p_deposit, 20.0 * p_bg)
```

**Impact**: When the cap is active (which depends on nr, nz, and deposit volume), the actual deposited energy is less than `E0`, making the analytical reference overestimate the shock radius. This compounds with HIGH-1 (alpha=1.0 underestimate), and the two errors may partially cancel -- which is the dangerous "right answer for wrong reasons" pattern.

**Fix applied**:
```python
p_deposit = min(p_deposit, 20.0 * p_bg)
# Recompute actual E0 after capping
E0_actual = p_deposit / (gamma - 1.0) * total_volume
```
`sedov_shock_radius_cylindrical()` now called with `E0_actual`. Both `E0_requested` and `E0_actual` stored in `metadata`.

---

## MEDIUM Findings

### MED-1: Orszag-Tang uses Neumann BCs for an intrinsically periodic problem

- **File**: `orszag_tang.py:290-293`
- **Found by**: phys-circuit-val (BUG-VV-1), py-circuit-val (V-OT-1)
- **Cross-review verdict**: **CONFIRMED** -- correctly documented as a limitation

**Description**: The Orszag-Tang vortex is defined on a periodic domain, but the MHDSolver uses zero-gradient (Neumann) boundary conditions. The initial conditions `vx = -sin(y)`, `vy = sin(x)`, `Bx = -sin(y)`, `By = sin(2x)` are all periodic on [0, 2pi]^2, so waves that exit one boundary should re-enter from the opposite side. With Neumann BCs, boundary reflections corrupt the solution.

**Evidence**:
```python
# orszag_tang.py:290-293
bc_note = (
    "Zero-gradient (Neumann) BCs used; periodic BCs not yet supported. "
    "Boundary effects may degrade accuracy at late times."
)
```

**Impact**: Energy "conservation" metric (line 274: `E_final / E0`) is polluted by boundary flux leakage. The benchmark is useful for qualitative solver validation at short times (t < 0.5) but quantitative comparison against published Orszag-Tang results (which use periodic BCs) is unreliable. This limits the diagnostic value of the benchmark as a code verification tool.

**Mitigation**: The limitation is correctly documented in the code. For quantitative Orszag-Tang verification, use Athena++ (which supports periodic BCs natively).

### MED-2: Explicit diffusion runner resets velocity every step

- **File**: `diffusion_convergence.py:377`
- **Found by**: py-circuit-val (V-DIFF-1)
- **Cross-review verdict**: **CONFIRMED** -- workaround that couples advection into each step

**Description**: The explicit diffusion test runs the full MHD solver but resets `state["velocity"][:] = 0.0` after each step to isolate the resistive diffusion operator. However, within each RK stage the MHD fluxes still produce non-zero velocities that couple into the B-field evolution before being zeroed out.

**Evidence**:
```python
# diffusion_convergence.py:377
state["velocity"][:] = 0.0
```

**Impact**: The velocity reset does not perfectly isolate diffusion from advection. Each timestep includes a small advective B-field contribution (driven by MHD fluxes from the non-uniform B pressure gradient) before the velocity is zeroed. This contributes an O(dt) error per step that accumulates. The mitigation is adequate -- very high background pressure (p0 = 1e6 at line 317) makes the B-field pressure gradient relatively negligible -- but the convergence rate measured for the "explicit" method is not purely testing the diffusion operator.

**Proposed fix**: The cleanest approach would be to extract the resistive diffusion operator into a standalone function (like `_run_implicit` and `_run_sts` already do for their respective methods) rather than running the full MHD solver with velocity clamping. However, the explicit test's stated purpose is to verify the diffusion operator *as embedded in the MHD solver*, which makes the current approach defensible.

### MED-3: Sod shock tube defaults to gamma=1.4 (monatomic gas convention is 5/3)

- **File**: `shock_tubes.py:443`
- **Found by**: py-circuit-val (V-SOD-1)
- **Cross-review verdict**: **DOWNGRADED from HIGH to MEDIUM** -- gamma=1.4 is the canonical Sod (1978) value

**Description**: The `run_sod_test()` function defaults to `gamma = 1.4`, which is the standard value used in the original Sod (1978) paper and in most compressible gas dynamics textbooks (Toro, LeVeque). However, DPF simulations use deuterium plasma with `gamma = 5/3` (monatomic gas / plasma). If a user runs the Sod test without specifying gamma, the default doesn't match the physics of the DPF application.

**Cross-review note**: phys-circuit-val verified that gamma=1.4 is the standard Sod convention. This is not a bug -- it's the canonical benchmark value. However, for a DPF code, having a companion test with gamma=5/3 would be more relevant. The parameter is exposed and easily overridden.

**Impact**: Low. The gamma parameter is explicit in the function signature. The exact Riemann solver at lines 86-170 correctly handles any gamma.

### ✅ FIXED MED-4: `_estimate_convergence_order` duplicated between two modules

- **File**: `diffusion_convergence.py:120-152` and `cylindrical_convergence.py:442-491`
- **Found by**: py-circuit-val (V-DUP-1)
- **Cross-review verdict**: **CONFIRMED** -- code duplication, but functionally identical
- **Status**: ✅ FIXED — extracted to `dpf.verification.utils.estimate_convergence_order()`; both modules now import it as `_estimate_convergence_order`.

**Description**: The `_estimate_convergence_order()` function appears in both `diffusion_convergence.py` (lines 120-152) and `cylindrical_convergence.py` (lines 442-491). The implementations are identical: log-log least-squares fit, same filtering logic, same return value. This is a maintenance burden -- a fix in one copy may be forgotten in the other.

**Fix applied**: Created `src/dpf/verification/utils.py` with `estimate_convergence_order()`. Both modules now:
```python
from dpf.verification.utils import estimate_convergence_order as _estimate_convergence_order
```
Local definitions removed.

---

## LOW Findings

### LOW-1: Sod analytical solution uses scalar loop instead of vectorized sampling

- **File**: `shock_tubes.py:245-275`
- **Found by**: py-circuit-val (V-SOD-2)
- **Cross-review verdict**: **CONFIRMED** -- cosmetic/performance

**Description**: The `sod_shock_tube_analytical()` function samples the exact solution using a Python `for` loop over cells (lines 245-275). This could be vectorized with `np.where` chains for ~10x speedup. However, this function is called once per test with typical nx=200, so performance is negligible.

### LOW-2: Shock tube _NY_PAD=4 deliberately disables WENO5

- **File**: `shock_tubes.py:52-53`
- **Found by**: py-circuit-val (V-SOD-3)
- **Cross-review verdict**: **CONFIRMED as intentional design** -- not a bug

**Description**: Setting `_NY_PAD = 4` (below the WENO5 minimum of 5 cells per dimension) forces the Python engine to use its non-WENO5 fallback path with Lax-Friedrichs diffusion. The extensive comment at lines 42-53 explains this is deliberate: the Python engine's hybrid WENO5/np.gradient scheme has architectural limitations at shock discontinuities.

**Impact**: None. The Metal engine uses 5-cell padding separately.

### LOW-3: Sedov CFL=0.15 is very conservative

- **File**: `sedov_cylindrical.py:300`
- **Found by**: py-circuit-val (V-SED-3)
- **Cross-review verdict**: **CONFIRMED but justified**

**Description**: The CFL number `cfl=0.15` is lower than the typical 0.3-0.4 used in other verification tests. This is conservative but appropriate for cylindrical geometry where geometric source terms (1/r factors) can cause instability near the axis.

### LOW-4: Orszag-Tang Dedner ch_dedner computed from approximate Alfven speed

- **File**: `orszag_tang.py:210-211`
- **Found by**: phys-circuit-val (observation)
- **Cross-review verdict**: **CONFIRMED** -- minor overestimate is harmless

**Description**: The Dedner cleaning speed `ch_dedner` is computed using `|B|^2 = 1^2 + 1^2` (line 210), taking the maximum component amplitudes. This slightly overestimates the true max Alfven speed but errs on the side of stronger divergence cleaning, which is safe.

### LOW-5: Cylindrical convergence test uses fixed step count, not fixed time

- **File**: `cylindrical_convergence.py:385-391`
- **Found by**: py-circuit-val (V-CYL-1)
- **Cross-review verdict**: **CONFIRMED but well-justified**

**Description**: In the default mode (t_end=None), each resolution takes exactly `n_steps=3` CFL-limited steps. Since CFL dt ~ dr, finer grids advance less physical time. The docstring at lines 33-37 explicitly explains this: "the per-step truncation error decreases as O(dr^2), so the total error after a fixed step count also decreases as O(dr^2)." This is a standard convergence testing methodology (used by Athena++, FLASH, etc.).

---

## REJECTED Findings

### REJ-1: py-circuit-val V-DIFF-2 — high p0 makes explicit test unreliable

The Python reviewer suggested that `p0 = 1e6` (line 317 of `diffusion_convergence.py`) makes the test unreliable because it doesn't represent real MHD conditions. Cross-review **REJECTS** this: the purpose of a verification test is to isolate one operator and measure convergence against an analytical solution. Using artificially high pressure to suppress advection is the standard approach (Ryu & Jones 1995). The test measures what it claims to measure.

### REJ-2: phys-circuit-val — Brio-Wu Bx tolerance too generous

The physics reviewer noted that `Bx_err < 0.1` (line 682 of `shock_tubes.py`) is a generous tolerance for B_x conservation. Cross-review **REJECTS** as a bug: this is a qualitative sanity check, not a precision metric. The B_x component in 1D MHD is the normal component (not evolved by the 1D induction equation), but the 3D solver with Dedner cleaning and transverse fluxes can perturb it. A 0.1 tolerance on a 0.75 field (13% relative) is reasonable for a multi-dimensional solver used in 1D mode.

### REJ-3: py-circuit-val V-CYL-2 — convergence test only measures B_theta

The Python reviewer suggested measuring convergence on all variables, not just B_theta. Cross-review **REJECTS**: the code measures pressure, velocity, B_theta, and density errors (lines 407-412, `check_equilibrium_preservation` at lines 184-244). The convergence *order* is estimated from B_theta because it's the cleanest diagnostic (lines 424-430, with justification in comments). All four metrics are reported.

---

## Positive Observations

1. **Sod exact Riemann solver**: Correct implementation of the Toro (Ch. 4) Newton-Raphson iteration for p_star. Pressure functions, derivatives, rarefaction/shock branches, and sampling all verified against the textbook. Research-grade.

2. **Brio-Wu qualitative checks**: Appropriate approach for a problem without analytical solution. Five independent checks (NaN, positivity, B_x, wave structure, By sign change) provide robust regression detection.

3. **Diffusion analytical solution**: The `gaussian_B_analytical()` function correctly implements the Green's function solution `B0 * (sigma0/sigma(t)) * exp(-x^2/(2*sigma(t)^2))` with `sigma(t) = sqrt(sigma0^2 + 2Dt)`. Verified against Ryu & Jones (1995).

4. **Cylindrical convergence methodology**: The `setup_zpinch_equilibrium()` function implements the Bennett equilibrium with correct Ampere's law and force balance. The choice `a = 2*R_domain` to keep the grid inside the smooth parabolic region (avoiding the B_theta kink at r=a) is a sophisticated convergence testing technique.

5. **Three independent diffusion solvers**: Testing explicit (MHD operator-split), implicit (Crank-Nicolson), and STS (RKL2) against the same analytical solution provides strong cross-verification. Each method has its own runner function with appropriate time-stepping.
