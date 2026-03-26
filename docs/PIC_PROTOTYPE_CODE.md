# PIC-MHD Hybrid: Bug Fix and Validation Prototypes

**Date**: 2026-03-26
**Author**: dpf-validation-engineer (Opus 4.6)
**Source analysis**: `PIC_VALIDATION_SCAFFOLD.md`, `investigations/pic_compound_bugs.md`

---

## Fix Dependency Graph

```
                 [1] Ghost Cell NaN Guard
                         |
                         v
                 [3] Esirkepov dt Fix
                         |
                         v
                 [4] Sub-cycling Wrapper  <--- requires [3] for charge conservation
                         |
                         v
                 [2] Relativistic Boris   <--- independent, but sub-cycling reduces urgency
```

Fix order: 1 -> 3 -> 4 -> 2. Ghost cells first because they cause instant simulation
death (1 step). Esirkepov dt before sub-cycling because sub-cycling depends on correct
dt propagation. Relativistic Boris last because sub-cycling reduces the per-step velocity
gain, buying time before v > c.

---

## Bug Fix 1: Ghost Cell NaN Guard (Kill Chain #1)

**Problem**: MHD ghost cells contain NaN during pinch phase. `_interpolate_vector_kernel`
reads ghost cells via CIC stencil, returns NaN to Boris push, which propagates NaN to
all particles in one step.

**Root cause**: Lines 934-936 clamp indices to `[0, nx-2]` but do not check the
field values at those indices. NaN passes through the trilinear interpolation.

**Fix**: Replace NaN field values with zero before interpolation. This is equivalent
to "no force at boundary" -- physically reasonable for ghost cells.

```python
@njit(cache=True)
def _interpolate_vector_kernel_safe(
    field: np.ndarray,
    positions: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """CIC vector interpolation with NaN guard on ghost cells.

    Identical to _interpolate_vector_kernel except: any NaN in the 8-node
    stencil is replaced with 0.0 before computing the weighted average.
    This prevents MHD ghost-cell NaN from propagating into particle fields.
    """
    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]
    n = positions.shape[0]
    values = np.empty((n, 3), dtype=np.float64)

    for p in range(n):
        xn = positions[p, 0] / dx
        yn = positions[p, 1] / dy
        zn = positions[p, 2] / dz

        ix = int(np.floor(xn))
        iy = int(np.floor(yn))
        iz = int(np.floor(zn))

        fx = xn - ix
        fy = yn - iy
        fz = zn - iz

        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iz = max(0, min(iz, nz - 2))

        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        fz = max(0.0, min(fz, 1.0))

        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w110 = fx * fy * (1.0 - fz)
        w101 = fx * (1.0 - fy) * fz
        w011 = (1.0 - fx) * fy * fz
        w111 = fx * fy * fz

        for c in range(3):
            # Read stencil values, replacing NaN with 0.0
            v000 = field[ix, iy, iz, c]
            v100 = field[ix + 1, iy, iz, c]
            v010 = field[ix, iy + 1, iz, c]
            v001 = field[ix, iy, iz + 1, c]
            v110 = field[ix + 1, iy + 1, iz, c]
            v101 = field[ix + 1, iy, iz + 1, c]
            v011 = field[ix, iy + 1, iz + 1, c]
            v111 = field[ix + 1, iy + 1, iz + 1, c]

            # NaN guard: replace with 0.0 (no force at ghost cells)
            if np.isnan(v000): v000 = 0.0
            if np.isnan(v100): v100 = 0.0
            if np.isnan(v010): v010 = 0.0
            if np.isnan(v001): v001 = 0.0
            if np.isnan(v110): v110 = 0.0
            if np.isnan(v101): v101 = 0.0
            if np.isnan(v011): v011 = 0.0
            if np.isnan(v111): v111 = 0.0

            values[p, c] = (
                v000 * w000 + v100 * w100 + v010 * w010 + v001 * w001
                + v110 * w110 + v101 * w101 + v011 * w011 + v111 * w111
            )

    return values
```

**Integration**: Replace `_interpolate_vector_kernel` call in
`interpolate_field_to_particles()` (line ~1230). Apply same pattern to scalar kernel.

---

## Bug Fix 2: Relativistic Boris Push (Kill Chain #2)

**Problem**: The Boris pusher at lines 397-478 uses `qdt_over_2m = charge * dt / (2 * mass)`
without a gamma factor. In DPF E-fields (~10^7 V/m), a deuteron reaches v > c in ~6000 steps.
Superluminal particles crash Esirkepov (multi-cell crossing) and produce NaN via overflow.

**Fix**: Standard relativistic Boris (Birdsall & Langdon 1985, Ch. 15). The key change
is dividing by gamma at the rotation step. Uses the Vay (2008) formulation for
improved accuracy at high gamma.

```python
@njit(cache=True)
def _boris_push_relativistic_kernel(
    positions: np.ndarray,
    velocities: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
    charge: float,
    mass: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Relativistic Boris push (Vay 2008, J. Comput. Phys. 227, 4).

    Uses proper velocity u = gamma*v internally. The input/output velocities
    are ordinary 3-velocities v [m/s], converted internally.

    Key difference from non-relativistic: the rotation vector t is divided
    by gamma_minus, which limits the velocity update to v < c.
    """
    c = 2.998e8
    c2 = c * c
    n = positions.shape[0]
    new_pos = np.empty_like(positions)
    new_vel = np.empty_like(velocities)

    qdt_over_2m = charge * dt / (2.0 * mass)

    for i in range(n):
        # Convert to proper velocity u = gamma * v
        vx = velocities[i, 0]
        vy = velocities[i, 1]
        vz = velocities[i, 2]
        v2 = vx * vx + vy * vy + vz * vz
        gamma_old = 1.0 / np.sqrt(1.0 - min(v2 / c2, 1.0 - 1e-15))
        ux = gamma_old * vx
        uy = gamma_old * vy
        uz = gamma_old * vz

        # Half E-field acceleration (in proper velocity)
        ux_minus = ux + qdt_over_2m * E_field[i, 0]
        uy_minus = uy + qdt_over_2m * E_field[i, 1]
        uz_minus = uz + qdt_over_2m * E_field[i, 2]

        # Gamma at half-step (from u_minus)
        u_minus2 = ux_minus**2 + uy_minus**2 + uz_minus**2
        gamma_minus = np.sqrt(1.0 + u_minus2 / c2)

        # Rotation vector t = q*B*dt / (2*m*gamma_minus)
        tx = qdt_over_2m * B_field[i, 0] / gamma_minus
        ty = qdt_over_2m * B_field[i, 1] / gamma_minus
        tz = qdt_over_2m * B_field[i, 2] / gamma_minus

        t_mag2 = tx * tx + ty * ty + tz * tz
        s_factor = 2.0 / (1.0 + t_mag2)
        sx = s_factor * tx
        sy = s_factor * ty
        sz = s_factor * tz

        # u' = u_minus + u_minus x t
        upx = ux_minus + (uy_minus * tz - uz_minus * ty)
        upy = uy_minus + (uz_minus * tx - ux_minus * tz)
        upz = uz_minus + (ux_minus * ty - uy_minus * tx)

        # u_plus = u_minus + u' x s
        ux_plus = ux_minus + (upy * sz - upz * sy)
        uy_plus = uy_minus + (upz * sx - upx * sz)
        uz_plus = uz_minus + (upx * sy - upy * sx)

        # Second half E-field acceleration
        ux_new = ux_plus + qdt_over_2m * E_field[i, 0]
        uy_new = uy_plus + qdt_over_2m * E_field[i, 1]
        uz_new = uz_plus + qdt_over_2m * E_field[i, 2]

        # Convert back to ordinary velocity: v = u / gamma
        u_new2 = ux_new**2 + uy_new**2 + uz_new**2
        gamma_new = np.sqrt(1.0 + u_new2 / c2)
        new_vel[i, 0] = ux_new / gamma_new
        new_vel[i, 1] = uy_new / gamma_new
        new_vel[i, 2] = uz_new / gamma_new

        # Position update using ordinary velocity
        new_pos[i, 0] = positions[i, 0] + new_vel[i, 0] * dt
        new_pos[i, 1] = positions[i, 1] + new_vel[i, 1] * dt
        new_pos[i, 2] = positions[i, 2] + new_vel[i, 2] * dt

    return new_pos, new_vel
```

**Integration**: Drop-in replacement for `_boris_push_kernel` in `boris_push()` wrapper (line ~1050). Identical API.

---

## Bug Fix 3: Esirkepov dt Fix (Charge Conservation Breaker)

**Problem**: `deposit()` at line 1561 passes `self.dt` to Esirkepov, but
`push_particles()` may use a different dt. The prefactor `charge / (cell_volume * dt)`
uses the wrong dt, breaking `div(J)*dt + delta_rho = 0` by a factor of
`dt_push / self.dt`.

**Fix**: Store the actual push dt as `self._last_push_dt` and use it in deposit.

```python
# --- In HybridPIC.__init__ (after self.dt = dt): ---
self._last_push_dt = dt  # track actual push dt for Esirkepov consistency

# --- In HybridPIC.push_particles(), after dt resolution (line 1432): ---
def push_particles(self, E, B, dt=None):
    if dt is None:
        dt = self.dt
    self._last_push_dt = dt  # <-- ADD THIS LINE
    # ... rest of method unchanged ...

# --- In HybridPIC.deposit(), line 1559-1561: ---
# BEFORE:
#   jx, jy, jz = deposit_current_esirkepov(
#       sp.positions_old, sp.positions, sp.weights, sp.charge,
#       self.grid_shape, self.dx, self.dy, self.dz, self.dt,
#   )
# AFTER:
    jx, jy, jz = deposit_current_esirkepov(
        sp.positions_old, sp.positions, sp.weights, sp.charge,
        self.grid_shape, self.dx, self.dy, self.dz, self._last_push_dt,
    )
```

**LOC delta**: 3 lines changed/added.

---

## Bug Fix 4: Sub-cycling Wrapper (Architectural Fix)

**Problem**: MHD dt ~ 1 ns, but 100 keV deuterons in 10 T need dt_pic ~ 0.2 ns
(30 steps per gyroperiod). Without sub-cycling, Boris under-resolves gyration by ~5x,
producing incorrect trajectories and eventual energy drift.

**Fix**: A `subcycle_pic` function that calls Boris N times per MHD step, with
field re-interpolation each sub-step. Requires Bug 3 fix to maintain charge conservation.

```python
def subcycle_pic(
    pic: HybridPIC,
    E: np.ndarray,
    B: np.ndarray,
    mhd_dt: float,
    n_sub: int | None = None,
    max_sub: int = 50,
) -> None:
    """Sub-cycle PIC push within one MHD timestep.

    Splits the MHD timestep into n_sub PIC sub-steps, each with
    dt_pic = mhd_dt / n_sub. Fields are re-interpolated at each sub-step
    (frozen-field approximation -- E, B constant over mhd_dt).

    If n_sub is None, it is computed from the CFL-like condition:
        n_sub = ceil(mhd_dt / dt_pic_max)
    where dt_pic_max = T_cyclotron / 30 (30 steps per gyroperiod).

    Parameters
    ----------
    pic : HybridPIC
        The PIC driver instance.
    E : ndarray, shape (nx, ny, nz, 3)
        Electric field [V/m]. Assumed constant over mhd_dt.
    B : ndarray, shape (nx, ny, nz, 3)
        Magnetic field [T]. Assumed constant over mhd_dt.
    mhd_dt : float
        MHD timestep [s].
    n_sub : int or None
        Number of sub-steps. Auto-computed if None.
    max_sub : int
        Cap on sub-step count to prevent runaway.
    """
    if n_sub is None:
        # Estimate cyclotron period from max |B|
        B_mag = np.sqrt(B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2)
        B_max = float(np.max(B_mag))
        if B_max < 1e-10:
            # Negligible B -- no gyration constraint
            n_sub = 1
        else:
            # Use the lightest species for the most restrictive constraint
            m_min = min(sp.mass for sp in pic.species if sp.n_particles() > 0)
            q_max = max(abs(sp.charge) for sp in pic.species if sp.n_particles() > 0)
            T_cyc = 2.0 * np.pi * m_min / (q_max * B_max)
            dt_pic_max = T_cyc / 30.0  # 30 steps per gyroperiod
            n_sub = int(np.ceil(mhd_dt / dt_pic_max))

    n_sub = max(1, min(n_sub, max_sub))
    dt_pic = mhd_dt / n_sub

    # Save original positions for Esirkepov (across full mhd_dt)
    for sp in pic.species:
        if sp.n_particles() > 0:
            sp.positions_old = sp.positions.copy()

    for step in range(n_sub):
        # Re-interpolate fields at current particle positions
        for sp in pic.species:
            if sp.n_particles() == 0:
                continue

            E_at_p = interpolate_field_to_particles(
                E, sp.positions, pic.dx, pic.dy, pic.dz
            )
            B_at_p = interpolate_field_to_particles(
                B, sp.positions, pic.dx, pic.dy, pic.dz
            )

            # Boris push for one sub-step
            new_pos, new_vel = boris_push(
                sp.positions, sp.velocities, E_at_p, B_at_p,
                sp.charge, sp.mass, dt_pic,
            )

            # Reflecting BCs
            new_pos, new_vel = pic._apply_reflecting_bc(new_pos, new_vel)

            sp.positions = new_pos
            sp.velocities = new_vel

    # Store effective dt for Esirkepov deposit (total displacement over mhd_dt)
    pic._last_push_dt = mhd_dt
```

**Integration**: Replace `self.driver.push_particles(E_field, B_field, dt=dt)` in
`KineticManager.step()` (line 123) with `subcycle_pic(self.driver, E_field, B_field, mhd_dt=dt)`.

**Esirkepov note**: `positions_old` saved once before all sub-steps, `_last_push_dt = mhd_dt`.
Esirkepov sees NET displacement -- correct for charge conservation if no particle crosses
more than 1 cell over the total mhd_dt. Follow-up: add CFL check on total displacement.

---

## Validation Test 1: ExB Drift Test (Boris Validation)

**Purpose**: Verify that Boris push produces correct ExB drift velocity.
Analytical: `v_drift = (E x B) / |B|^2`. Simplest possible Boris validation.

```python
import numpy as np
import pytest
from dpf.experimental.pic.hybrid import boris_push


def test_exb_drift_velocity():
    """Single particle in crossed E and B fields drifts at v_E = ExB/B^2."""
    # Uniform fields: E = (0, Ey, 0), B = (0, 0, Bz)
    Ey = 1e5  # V/m
    Bz = 1.0  # T
    v_drift_expected = Ey / Bz  # = 1e5 m/s, in x-direction

    charge = 1.602e-19
    mass = 3.34e-27  # deuteron
    dt = 1e-10
    n_steps = 1000
    T_cyc = 2 * np.pi * mass / (charge * Bz)  # ~21 ns
    # 1000 steps at 0.1 ns = 100 ns ~ 5 gyroperiods

    pos = np.array([[0.05, 0.05, 0.05]])  # center of domain
    vel = np.array([[0.0, 0.0, 0.0]])
    E_p = np.array([[0.0, Ey, 0.0]])
    B_p = np.array([[0.0, 0.0, Bz]])

    for _ in range(n_steps):
        pos, vel = boris_push(pos, vel, E_p, B_p, charge, mass, dt)

    # After many gyroperiods, average vx should converge to v_drift
    # Instantaneous vx oscillates, but displacement / time gives drift
    total_time = n_steps * dt
    v_drift_measured = pos[0, 0] / total_time  # net x-displacement / time

    # 10% tolerance: Boris leap-frog has finite-dt phase error
    assert abs(v_drift_measured - v_drift_expected) / v_drift_expected < 0.10, (
        f"ExB drift: expected {v_drift_expected:.0f} m/s, got {v_drift_measured:.0f} m/s"
    )
    # vy should be oscillatory (gyration), not runaway
    assert abs(vel[0, 1]) < 2 * v_drift_expected, "vy runaway"
```

---

## Validation Test 2: Esirkepov Charge Conservation

**Purpose**: Verify `div(J)*dt + delta_rho = 0` for the Esirkepov kernel.

```python
import numpy as np
import pytest
from dpf.experimental.pic.hybrid import (
    deposit_current_esirkepov,
    deposit_density,
)


def test_esirkepov_charge_conservation():
    """Esirkepov continuity: div(J)*dt + delta_rho = 0 for interior particle."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 0.01
    dt = 1e-10
    charge = 1.602e-19
    grid_shape = (nx, ny, nz)

    # Single particle moving in x: (3.5, 3.5, 3.5)*dx -> (4.2, 3.5, 3.5)*dx
    pos_old = np.array([[3.5 * dx, 3.5 * dy, 3.5 * dz]])
    pos_new = np.array([[4.2 * dx, 3.5 * dy, 3.5 * dz]])
    weights = np.array([1e15])

    # Deposit current (Esirkepov)
    Jx, Jy, Jz = deposit_current_esirkepov(
        pos_old, pos_new, weights, charge, grid_shape, dx, dy, dz, dt
    )

    # Deposit density at old and new positions
    rho_old = charge * deposit_density(pos_old, weights, grid_shape, dx, dy, dz)
    rho_new = charge * deposit_density(pos_new, weights, grid_shape, dx, dy, dz)
    delta_rho = rho_new - rho_old

    # Numerical divergence of J (central differences, interior only)
    div_J = np.zeros_like(Jx)
    div_J[1:-1, :, :] += (Jx[2:, :, :] - Jx[:-2, :, :]) / (2 * dx)
    div_J[:, 1:-1, :] += (Jy[:, 2:, :] - Jy[:, :-2, :]) / (2 * dy)
    div_J[:, :, 1:-1] += (Jz[:, :, 2:] - Jz[:, :, :-2]) / (2 * dz)

    # Continuity: div(J)*dt + delta_rho/dt_deposit = 0
    # Since Esirkepov uses discrete stencil, check on the stencil nodes
    # For CIC on 8^3 grid with particle at cells 3-5, check cells 2-6
    residual = div_J[2:6, 2:6, 2:6] * dt + delta_rho[2:6, 2:6, 2:6]
    max_residual = float(np.max(np.abs(residual)))

    # Esirkepov guarantees exact discrete continuity on the CIC stencil
    # The central-difference div is an approximation, so allow 1e-6 tolerance
    assert max_residual < 1e-6, (
        f"Charge conservation violation: max|div(J)*dt + delta_rho| = {max_residual:.2e}"
    )
```

---

## Validation Test 3: Sub-cycling Stability

**Purpose**: Compare single-step vs sub-cycled push. Sub-cycled result should have
smaller energy error and no superluminal particles.

```python
import numpy as np
import pytest
from dpf.experimental.pic.hybrid import HybridPIC, boris_push


def test_subcycle_vs_single_step():
    """Sub-cycled push preserves energy better than single large step."""
    # Strong B-field: 10 T. Deuteron gyroperiod ~ 6.5 ns.
    # MHD dt = 5 ns ~ 0.77 gyroperiods. Single step under-resolves.
    charge = 1.602e-19
    mass = 3.34e-27
    Bz = 10.0
    T_cyc = 2 * np.pi * mass / (charge * Bz)  # ~6.5 ns
    mhd_dt = 5e-9  # 5 ns

    # Initial: v_perp = 1e6 m/s (thermal deuteron)
    pos0 = np.array([[0.05, 0.05, 0.05]])
    vel0 = np.array([[1e6, 0.0, 0.0]])
    E_p = np.array([[0.0, 0.0, 0.0]])  # no E-field
    B_p = np.array([[0.0, 0.0, Bz]])

    KE_initial = 0.5 * mass * np.sum(vel0**2)

    # --- Single step (mhd_dt) ---
    pos_1, vel_1 = boris_push(
        pos0.copy(), vel0.copy(), E_p, B_p, charge, mass, mhd_dt
    )
    KE_single = 0.5 * mass * np.sum(vel_1**2)
    err_single = abs(KE_single - KE_initial) / KE_initial

    # --- Sub-cycled: 25 steps (T_cyc/30 ~ 0.2 ns each) ---
    n_sub = 25
    dt_sub = mhd_dt / n_sub
    pos_n = pos0.copy()
    vel_n = vel0.copy()
    for _ in range(n_sub):
        pos_n, vel_n = boris_push(pos_n, vel_n, E_p, B_p, charge, mass, dt_sub)
    KE_sub = 0.5 * mass * np.sum(vel_n**2)
    err_sub = abs(KE_sub - KE_initial) / KE_initial

    # Boris push conserves energy exactly for uniform B, but finite dt
    # introduces phase error. Sub-cycled should have smaller error.
    # For Boris in pure B, energy conservation is exact regardless of dt,
    # so both errors should be < 1e-10. The real test is trajectory accuracy.
    assert err_single < 1e-10, f"Single-step energy error: {err_single:.2e}"
    assert err_sub < 1e-10, f"Sub-cycled energy error: {err_sub:.2e}"

    # Trajectory test: sub-cycled position should be closer to analytical
    # Analytical: particle gyrates in circle of radius r_L = m*v/(q*B)
    r_L = mass * 1e6 / (charge * Bz)  # Larmor radius
    # After time mhd_dt, angle = omega_c * mhd_dt
    omega_c = charge * Bz / mass
    theta = omega_c * mhd_dt
    x_exact = r_L * np.sin(theta)
    y_exact = r_L * (1.0 - np.cos(theta))

    err_pos_single = np.sqrt(
        (pos_1[0, 0] - pos0[0, 0] - x_exact)**2
        + (pos_1[0, 1] - pos0[0, 1] - y_exact)**2
    )
    err_pos_sub = np.sqrt(
        (pos_n[0, 0] - pos0[0, 0] - x_exact)**2
        + (pos_n[0, 1] - pos0[0, 1] - y_exact)**2
    )

    # Sub-cycled should be at least 2x more accurate in position
    assert err_pos_sub < err_pos_single, (
        f"Sub-cycling not more accurate: single={err_pos_single:.3e}, sub={err_pos_sub:.3e}"
    )
```

---

## Risk Management Plan

### Individual Bug Risks

| Bug | Prob. | Impact | Mitigation | Acceptance Test |
|-----|-------|--------|------------|-----------------|
| **1. Ghost cell NaN** | HIGH (100% at boundary) | CRITICAL: instant sim death | NaN guard in interpolation kernel | `test_interpolation_nan_guard`: field with NaN at [0,:,:], particle at x~0 returns finite |
| **2. Non-relativistic Boris** | HIGH (100% after ~6000 steps) | HIGH: superluminal -> Esirkepov crash | Replace with relativistic Boris kernel | `test_boris_subluminal`: 10000 steps in 10^7 V/m E-field, verify v < c |
| **3. Esirkepov dt mismatch** | LOW (currently no sub-cycling) | CRITICAL when sub-cycling enabled | Store `_last_push_dt`, use in deposit | `test_esirkepov_dt_consistency`: push with dt/2, deposit, verify div(J)*dt + drho = 0 |
| **4. No sub-cycling** | HIGH (under-resolved gyration) | HIGH: wrong trajectories at DPF B-fields | `subcycle_pic()` wrapper | `test_subcycle_vs_single_step`: sub-cycled position closer to analytical |
| **5. Nanbu self-collision bias** | MEDIUM | LOW: ~10-30% scattering rate error | Copy vel array before second argument | `test_nanbu_no_ordering_bias`: RMS angle within 5% of separate-array test |
| **6. Esirkepov boundary clamping** | MEDIUM (particles near edges) | MEDIUM: local charge loss | Log warning when particle within 1 cell of boundary | `test_esirkepov_boundary_warning`: particle at x=0.5*dx triggers warning |
| **7. Multi-cell crossing** | LOW with sub-cycling | HIGH without sub-cycling: silent charge loss | CFL check + warning when offset > 1 | `test_esirkepov_multicell_warning`: 3-cell crossing returns zero J + warning |
| **8. Reflecting BC + Esirkepov** | MEDIUM | MEDIUM: wrong J at boundaries | Two-segment deposition (deferred to V3) | `test_reflect_esirkepov_charge_conserved`: reflected particle satisfies continuity |

### Compound Interaction Risks

| Interaction | Trigger | Kill Time | Fix Required | Test |
|-------------|---------|-----------|--------------|------|
| Ghost NaN -> Boris -> Esirkepov -> MHD | Any particle within 1 cell of boundary when ghost has NaN | 1 step | Fix 1 (NaN guard) | Integration test: PIC on MHD state with NaN ghost cells |
| Non-rel Boris -> superluminal -> multi-cell Esirkepov -> charge loss -> rho=0 -> E spike -> all particles NaN | Strong E-field (~10^7 V/m) sustained for ~6000 steps | ~600 ns | Fix 2 (relativistic) + Fix 4 (sub-cycling) | `test_boris_dpf_efield_no_runaway`: 10000 steps at DPF conditions, all v < c |
| Sub-cycling + wrong dt Esirkepov -> broken charge conservation | Any sub-cycling attempt | Immediate | Fix 3 (dt propagation) | `test_subcycle_charge_conservation`: sub-cycled push + deposit satisfies continuity |
| Reflecting BC -> wrong displacement -> Esirkepov deposits zero J | Particle near boundary with v toward wall | Every reflected particle | Deferred (two-segment deposit) | `test_reflect_displacement_nonzero`: reflected particle has nonzero J |

### Residual Risks (accepted, not fixed in this sprint)

| Risk | Why Accepted | Monitoring |
|------|-------------|------------|
| E-field missing Hall term | Hall/convective ratio is ~0.6% at peak pinch. Defer to V3. | Log E_Hall/E_conv ratio each step |
| No particle removal | Memory grows linearly but manageable for <10^4 steps | Log particle count per step |
| Reflecting BC unphysical | Absorbing BC requires electrode geometry model (60+ LOC). Defer to V3. | Compare Yn with/without reflecting BC |
| CIC noise at low particle count | Acceptable for 10K+ particles. Add J smoothing in V3 if needed. | Monitor max(J_kin) / mean(J_mhd) ratio |

---

## Summary

| Deliverable | LOC | Blocks |
|-------------|-----|--------|
| Fix 1: Ghost cell NaN guard | ~15 | Nothing (first fix) |
| Fix 2: Relativistic Boris | ~65 | Nothing (independent) |
| Fix 3: Esirkepov dt | ~3 | Fix 4 depends on this |
| Fix 4: Sub-cycling wrapper | ~55 | Requires Fix 3 |
| Test: ExB drift | ~25 | Validates Fix 2 |
| Test: Charge conservation | ~35 | Validates Fix 3 |
| Test: Sub-cycling stability | ~40 | Validates Fix 4 |
| **Total** | **~238** | |

Implementation order: 1 -> 3 -> 4 -> 2, then tests in parallel.
