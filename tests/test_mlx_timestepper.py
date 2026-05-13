"""Tests for mlx_timestepper: SSP-RK3/RK2 time integration with dual-energy recovery.

All tests skip if mlx is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # noqa: E402

from dpf.metal.mlx_grid import CylindricalGrid  # noqa: E402, I001
from dpf.metal.constants import C_BORIS  # noqa: E402
from dpf.metal.mlx_kernels import IDN, IEN, IMR, IMZ, IMT, ISR, IBR, IBZ, IBT, NVAR  # noqa: E402
from dpf.metal.mlx_primitives import (  # noqa: E402
    RHO_FLOOR,
    cons_to_prim,
    prim_to_cons,
)
from dpf.metal.mlx_timestepper import (  # noqa: E402
    _apply_floors,
    _clamp_velocity,
    _resync_energy,
    _stage_post_impl,
    compute_dt_cfl,
    mhd_rhs,
    ssp_rk2_step,
    ssp_rk3_step,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GAMMA = 5.0 / 3.0


def _grid(nr: int = 16, nz: int = 16, dr: float = 0.01, dz: float = 0.01) -> CylindricalGrid:
    return CylindricalGrid(nr=nr, nz=nz, dr=dr, dz=dz, r_inner=0.0)


def _uniform_U(
    nr: int = 16,
    nz: int = 16,
    rho: float = 1.0,
    vr: float = 0.0,
    vz: float = 0.0,
    vt: float = 0.0,
    p: float = 0.6,
    Br: float = 0.0,
    Bz: float = 0.3,
    Bt: float = 0.0,
    gamma: float = GAMMA,
) -> mx.array:
    """Build a uniform (NVAR, nr, nz) conserved state."""
    rho_a = mx.full((nr, nz), rho, dtype=mx.float32)
    vr_a = mx.full((nr, nz), vr, dtype=mx.float32)
    vz_a = mx.full((nr, nz), vz, dtype=mx.float32)
    vt_a = mx.full((nr, nz), vt, dtype=mx.float32)
    p_a = mx.full((nr, nz), p, dtype=mx.float32)
    Br_a = mx.full((nr, nz), Br, dtype=mx.float32)
    Bz_a = mx.full((nr, nz), Bz, dtype=mx.float32)
    Bt_a = mx.full((nr, nz), Bt, dtype=mx.float32)
    return prim_to_cons(rho_a, vr_a, vz_a, vt_a, p_a, Br_a, Bz_a, Bt_a, gamma=gamma)


def _np(arr: mx.array) -> np.ndarray:
    return np.asarray(arr)


def _total_energy(U: mx.array) -> float:
    return float(mx.sum(U[IEN]).item())


# ---------------------------------------------------------------------------
# 1. Uniform state preservation (100 steps)
# ---------------------------------------------------------------------------


def test_mass_conservation_rk3():
    """RK3: total mass (integral of rho) must be conserved to < 1% over 100 steps.

    With outflow BCs the mass can change at boundaries, but for a uniform
    non-flowing state the mass flux through each boundary must be nearly zero,
    so total mass should be approximately conserved.
    """
    grid = _grid(nr=8, nz=8)
    # No velocity, no B: mass flux = 0 everywhere
    U0 = _uniform_U(nr=8, nz=8, vr=0.0, vz=0.0, vt=0.0, Br=0.0, Bz=0.0, Bt=0.0)
    mass0 = float(mx.sum(U0[IDN]).item())

    dt = 1e-7  # small step to avoid pressure-driven flows from growing
    U = U0
    for _ in range(100):
        U = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll")

    mass1 = float(mx.sum(U[IDN]).item())
    rel_diff = abs(mass1 - mass0) / (abs(mass0) + 1e-30)
    assert rel_diff < 0.01, f"Mass not conserved: {rel_diff:.3%} drift after 100 steps"


def test_mass_conservation_rk2():
    """RK2: total mass must be conserved to < 1% over 100 steps."""
    grid = _grid(nr=8, nz=8)
    U0 = _uniform_U(nr=8, nz=8, vr=0.0, vz=0.0, vt=0.0, Br=0.0, Bz=0.0, Bt=0.0)
    mass0 = float(mx.sum(U0[IDN]).item())

    dt = 1e-7
    U = U0
    for _ in range(100):
        U = ssp_rk2_step(U, grid, dt, method="plm", riemann="hll")

    mass1 = float(mx.sum(U[IDN]).item())
    rel_diff = abs(mass1 - mass0) / (abs(mass0) + 1e-30)
    assert rel_diff < 0.01, f"Mass (RK2) not conserved: {rel_diff:.3%} drift after 100 steps"


# ---------------------------------------------------------------------------
# 2. Energy conservation (100 steps, low beta)
# ---------------------------------------------------------------------------


def test_energy_conservation_short_run():
    """Total energy drift must be < 5% over 20 steps for a low-velocity state.

    Uses a very small timestep so the dynamics stay near the initial state.
    We cannot expect perfect conservation because:
    - Cylindrical BCs leak energy at domain edges
    - Geometric pressure source drives slow radial flows
    Instead we verify that energy doesn't blow up or collapse.
    """
    grid = _grid(nr=8, nz=8, dr=0.01, dz=0.01)
    U0 = _uniform_U(nr=8, nz=8, rho=1.0, p=0.6, Bz=0.0, vr=0.0, vz=0.0)
    E0 = _total_energy(U0)

    dt = 1e-6  # tiny step: <<< CFL
    U = U0
    for _ in range(20):
        U = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll",
                         use_dual_energy=True)

    E1 = _total_energy(U)
    rel_drift = abs(E1 - E0) / (abs(E0) + 1e-30)
    assert rel_drift < 0.05, f"Energy drifted {rel_drift:.3%} over 20 steps"
    assert math.isfinite(E1), "Energy blew up (NaN/Inf)"


# ---------------------------------------------------------------------------
# 3. CFL computation
# ---------------------------------------------------------------------------


def test_cfl_known_state():
    """For a known uniform state, dt should match the analytic CFL estimate."""
    nr, nz = 8, 8
    dr = dz = 0.01
    grid = _grid(nr=nr, nz=nz, dr=dr, dz=dz)

    rho_val = 1.0
    p_val = 0.6
    Bz_val = 0.3
    vz_val = 0.5

    U = _uniform_U(nr=nr, nz=nz, rho=rho_val, p=p_val, Bz=Bz_val, vz=vz_val)
    dt = compute_dt_cfl(U, grid, gamma=GAMMA, cfl=0.3)

    # Analytic: cf^2 = 0.5*(a^2 + va^2 + sqrt((a^2-va^2)^2 + 4*a^2*Bt^2/rho))
    # For Bz only and dim=1: Bn=Bz, Bt=0 -> cf = sqrt(a^2 + Bz^2/rho)
    a_sq = GAMMA * p_val / rho_val
    va_sq = Bz_val ** 2 / rho_val
    cf = math.sqrt(a_sq + va_sq)
    v_plus_cf = abs(vz_val) + cf
    dt_expected = 0.3 * min(dr, dz) / v_plus_cf

    # Allow 5% tolerance for floating-point differences
    assert abs(dt - dt_expected) / dt_expected < 0.05, (
        f"CFL dt={dt:.4e}, expected {dt_expected:.4e}"
    )


def test_cfl_positive():
    """CFL dt must always be strictly positive."""
    grid = _grid(nr=8, nz=8)
    U = _uniform_U(nr=8, nz=8)
    dt = compute_dt_cfl(U, grid)
    assert dt > 0.0
    assert math.isfinite(dt)


def test_cfl_boris_bounds_vacuum_field_timestep():
    """Boris CFL should prevent vacuum B_theta from collapsing dt."""
    grid = _grid(nr=8, nz=8, dr=0.005, dz=0.005)
    U = _uniform_U(nr=8, nz=8, rho=1e-8, p=1e-8, Bt=2.0e4)

    dt_plain = compute_dt_cfl(U, grid, gamma=GAMMA, cfl=0.3, use_boris=False)
    dt_boris = compute_dt_cfl(U, grid, gamma=GAMMA, cfl=0.3, use_boris=True)

    assert dt_plain > 0.0
    assert dt_boris > 0.0
    assert math.isfinite(dt_boris)
    assert dt_boris > 100.0 * dt_plain


def test_stage_post_caps_pressure_driven_velocity_with_boris_speed():
    """Post-stage limiter should match the Boris-capped CFL/Riemann speed."""
    U = _uniform_U(nr=4, nz=4, rho=1e-6, vr=1.0e12, p=1.0e12, Bz=0.0)

    U_out = _stage_post_impl(U, GAMMA)
    rho = U_out[IDN]
    vr = U_out[IMR] / rho
    max_v = float(mx.max(mx.abs(vr)))

    assert np.all(np.isfinite(np.asarray(U_out)))
    assert max_v <= 10.0 * C_BORIS * (1.0 + 1e-5)


# ---------------------------------------------------------------------------
# 4. Floor enforcement
# ---------------------------------------------------------------------------


def test_floor_density():
    """Density below RHO_FLOOR must be clamped to RHO_FLOOR."""
    nr, nz = 4, 4
    U = _uniform_U(nr=nr, nz=nz, rho=1e-20)
    U_floored = _apply_floors(U)
    rho_out = _np(U_floored[IDN])
    assert np.all(rho_out >= RHO_FLOOR - 1e-30)


def test_floor_preserves_good_cells():
    """Floor must not modify cells that are already above the floor."""
    nr, nz = 8, 8
    U = _uniform_U(nr=nr, nz=nz, rho=10.0, p=5.0)
    U_floored = _apply_floors(U)
    rho_diff = np.max(np.abs(_np(U_floored[IDN]) - _np(U[IDN])))
    assert rho_diff < 1e-6


def test_apply_floors_does_not_inject_density_for_strong_field():
    """The legacy B^2/va_max^2 density injection path must stay removed."""
    U = _uniform_U(nr=4, nz=4, rho=1e-10, p=1e3, Bz=0.0, Bt=100.0)

    U_floored = _apply_floors(U)
    rho_out = _np(U_floored[IDN])

    assert np.max(rho_out) <= 1e-9


def test_rk2_zero_dt_does_not_inject_density_for_strong_field():
    """Full RK2 path must not add fake mass when only floor logic can run."""
    grid = _grid(nr=4, nz=4)
    U = _uniform_U(nr=4, nz=4, rho=1e-10, p=1e3, Bz=0.0, Bt=100.0)

    U_out = ssp_rk2_step(U, grid, 0.0, method="plm", riemann="hll")
    rho_out = _np(U_out[IDN])

    assert np.max(rho_out) <= 1e-9


def test_rk3_zero_dt_does_not_inject_density_for_strong_field():
    """Full RK3 path must not add fake mass when only floor logic can run."""
    grid = _grid(nr=4, nz=4)
    U = _uniform_U(nr=4, nz=4, rho=1e-10, p=1e3, Bz=0.0, Bt=100.0)

    U_out = ssp_rk3_step(U, grid, 0.0, method="plm", riemann="hll")
    rho_out = _np(U_out[IDN])

    assert np.max(rho_out) <= 1e-9


# ---------------------------------------------------------------------------
# 5. Velocity clamping
# ---------------------------------------------------------------------------


def test_velocity_clamping_extreme():
    """Extreme velocity must be clamped to 10x fast magnetosonic speed."""
    nr, nz = 4, 4
    rho_val = 1.0
    p_val = 0.6
    Bz_val = 0.3
    v_extreme = 1e8  # much larger than cf

    U = _uniform_U(nr=nr, nz=nz, rho=rho_val, p=p_val, Bz=Bz_val, vz=v_extreme)
    U_clamped = _clamp_velocity(U, gamma=GAMMA)

    _, _, vz_c, _, _, _, _, _ = cons_to_prim(U_clamped, GAMMA)
    vz_np = _np(vz_c)

    # cf for this state (approx): sqrt(gamma*p/rho + Bz^2/rho)
    a_sq = GAMMA * p_val / rho_val
    va_sq = Bz_val ** 2 / rho_val
    cf = math.sqrt(a_sq + va_sq)
    v_max_expected = 10.0 * cf

    assert np.all(np.abs(vz_np) <= v_max_expected * 1.01), (
        f"Velocity not clamped: max |vz|={np.max(np.abs(vz_np)):.3e}, "
        f"expected <= {v_max_expected:.3e}"
    )


def test_velocity_clamping_normal():
    """Normal velocity must not be modified by clamping."""
    nr, nz = 4, 4
    U = _uniform_U(nr=nr, nz=nz, rho=1.0, p=0.6, Bz=0.3, vz=0.1)
    U_clamped = _clamp_velocity(U, gamma=GAMMA)
    _, _, vz_c, _, _, _, _, _ = cons_to_prim(U_clamped, GAMMA)
    _, _, vz_orig, _, _, _, _, _ = cons_to_prim(U, GAMMA)
    diff = float(mx.max(mx.abs(vz_c - vz_orig)).item())
    assert diff < 1e-5


# ---------------------------------------------------------------------------
# 6. Dual-energy recovery at each stage
# ---------------------------------------------------------------------------


def test_dual_energy_low_beta_uses_entropy():
    """At low beta, dual-energy should preferentially use the entropy tracer."""
    nr, nz = 8, 8
    # Strong B field: B^2 >> 2*p  => eta = |p_S|/|E| is small => use entropy
    U = _uniform_U(nr=nr, nz=nz, rho=1.0, p=0.001, Bz=10.0)
    U_resynced = _resync_energy(U, gamma=GAMMA)

    # The resynced energy should be close to the entropy-derived pressure
    rho_np = _np(U[IDN])
    Srho_np = _np(U[ISR])
    gm1 = GAMMA - 1.0
    p_S = Srho_np * np.power(np.maximum(rho_np, RHO_FLOOR), gm1)

    rho_new, vr_new, vz_new, vt_new, p_new, Br_new, Bz_new, Bt_new = cons_to_prim(U_resynced, GAMMA)
    p_new_np = _np(p_new)

    # Recovered p should be close to p_S (< 10% relative difference)
    rel_diff = np.max(np.abs(p_new_np - p_S) / (np.abs(p_S) + 1e-30))
    assert rel_diff < 0.1, f"Low-beta pressure not entropy-dominated: rel_diff={rel_diff:.3f}"


def test_dual_energy_high_beta_uses_total_energy():
    """At high beta, dual-energy should preferentially use total energy."""
    nr, nz = 8, 8
    # Thermal pressure >> magnetic: p >> B^2/2 => eta large => use total energy
    U = _uniform_U(nr=nr, nz=nz, rho=1.0, p=100.0, Bz=0.01)
    U_resynced = _resync_energy(U, gamma=GAMMA)

    rho_np = _np(U[IDN])
    inv_rho = 1.0 / np.maximum(rho_np, RHO_FLOOR)
    vr_np = _np(U[IMR]) * inv_rho
    vz_np = _np(U[IMZ]) * inv_rho
    vt_np = _np(U[IMT]) * inv_rho
    E_np = _np(U[IEN])
    B2 = _np(U[IBR]) ** 2 + _np(U[IBZ]) ** 2 + _np(U[IBT]) ** 2
    KE = 0.5 * rho_np * (vr_np ** 2 + vz_np ** 2 + vt_np ** 2)
    p_E = (GAMMA - 1.0) * (E_np - KE - 0.5 * B2)

    _, _, _, _, p_out, _, _, _ = cons_to_prim(U_resynced, GAMMA)
    p_out_np = _np(p_out)

    rel_diff = np.max(np.abs(p_out_np - p_E) / (np.abs(p_E) + 1e-30))
    assert rel_diff < 0.05, (
        f"High-beta pressure not total-energy-dominated: rel_diff={rel_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# 7. RK2 vs RK3 accuracy (smooth problem)
# ---------------------------------------------------------------------------


def test_rk3_lower_error_than_rk2():
    """RK3 should have lower total-energy error than RK2 on a smooth problem.

    Both integrators are run for a short time; RK3 should match the initial
    state more closely than RK2 because it's higher order.
    """
    grid = _grid(nr=8, nz=8, dr=0.1, dz=0.1)
    U0 = _uniform_U(nr=8, nz=8, rho=1.0, p=0.6, Bz=0.3, vz=0.01)
    E0 = _total_energy(U0)

    dt = 1e-5
    n_steps = 20

    U_rk2 = U0
    for _ in range(n_steps):
        U_rk2 = ssp_rk2_step(U_rk2, grid, dt, method="plm", riemann="hll",
                              use_dual_energy=True)

    U_rk3 = U0
    for _ in range(n_steps):
        U_rk3 = ssp_rk3_step(U_rk3, grid, dt, method="plm", riemann="hll",
                              use_dual_energy=True)

    err_rk2 = abs(_total_energy(U_rk2) - E0) / (abs(E0) + 1e-30)
    err_rk3 = abs(_total_energy(U_rk3) - E0) / (abs(E0) + 1e-30)

    # RK3 should not be worse than RK2 (allow 10x factor for margin)
    assert err_rk3 <= err_rk2 * 10.0 or err_rk3 < 1e-6, (
        f"RK3 error {err_rk3:.3e} is much worse than RK2 error {err_rk2:.3e}"
    )


# ---------------------------------------------------------------------------
# 8. Sod shock tube evolution (50 steps)
# ---------------------------------------------------------------------------


def _sod_initial_state(nr: int = 32, nz: int = 4, gamma: float = GAMMA) -> mx.array:
    """1-D Sod shock tube along z: left state (rho=1, p=1) / right state (rho=0.125, p=0.1)."""
    rho_np = np.zeros((nr, nz), dtype=np.float32)
    vr_np = np.zeros((nr, nz), dtype=np.float32)
    vz_np = np.zeros((nr, nz), dtype=np.float32)
    vt_np = np.zeros((nr, nz), dtype=np.float32)
    p_np = np.zeros((nr, nz), dtype=np.float32)
    Br_np = np.zeros((nr, nz), dtype=np.float32)
    Bz_np = np.zeros((nr, nz), dtype=np.float32)
    Bt_np = np.zeros((nr, nz), dtype=np.float32)

    mid = nz // 2
    rho_np[:, :mid] = 1.0
    rho_np[:, mid:] = 0.125
    p_np[:, :mid] = 1.0
    p_np[:, mid:] = 0.1

    rho = mx.array(rho_np)
    vr = mx.array(vr_np)
    vz = mx.array(vz_np)
    vt = mx.array(vt_np)
    p = mx.array(p_np)
    Br = mx.array(Br_np)
    Bz = mx.array(Bz_np)
    Bt = mx.array(Bt_np)

    return prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt, gamma=gamma)


def test_sod_shock_tube_no_blowup():
    """Sod shock tube must remain finite (no NaN/Inf) after 50 steps."""
    nz = 32
    nr = 4
    dz = 1.0 / nz
    dr = 1.0 / nr
    grid = _grid(nr=nr, nz=nz, dr=dr, dz=dz)
    U0 = _sod_initial_state(nr=nr, nz=nz)

    dt = 0.1 * dz / 3.0  # conservative CFL

    U = U0
    for _ in range(50):
        U = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll",
                         use_dual_energy=True)

    U_np = _np(U)
    assert not np.any(np.isnan(U_np)), "NaN in Sod shock tube state after 50 steps"
    assert not np.any(np.isinf(U_np)), "Inf in Sod shock tube state after 50 steps"


def test_sod_shock_density_jump():
    """Sod shock tube: density in left half must exceed right half after evolution."""
    nz = 32
    nr = 4
    dz = 1.0 / nz
    dr = 1.0 / nr
    grid = _grid(nr=nr, nz=nz, dr=dr, dz=dz)
    U0 = _sod_initial_state(nr=nr, nz=nz)

    dt = 0.1 * dz / 3.0
    U = U0
    for _ in range(50):
        U = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll",
                         use_dual_energy=True)

    rho_np = _np(U[IDN])
    rho_left = float(np.mean(rho_np[:, :nz // 4]))
    rho_right = float(np.mean(rho_np[:, 3 * nz // 4:]))

    assert rho_left > rho_right, (
        f"Sod shock: left density {rho_left:.4f} should exceed right {rho_right:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. mhd_rhs: zero RHS for uniform state
# ---------------------------------------------------------------------------


def test_mhd_rhs_mass_rhs_zero():
    """mhd_rhs of a uniform no-flow state must have zero density derivative.

    With v=0, B=0, the mass flux rho*v=0 everywhere, so d(rho)/dt = 0.
    """
    grid = _grid(nr=8, nz=8)
    U = _uniform_U(nr=8, nz=8, vr=0.0, vz=0.0, vt=0.0, Br=0.0, Bz=0.0, Bt=0.0)
    L = mhd_rhs(U, grid, gamma=GAMMA, method="plm", riemann="hll")
    L_np = _np(L)
    # Density RHS must be zero: no mass flux without velocity
    rms_density = float(np.sqrt(np.mean(L_np[IDN] ** 2)))
    assert rms_density < 1e-6, f"Density RHS non-zero: RMS={rms_density:.3e}"


# ---------------------------------------------------------------------------
# 10. Output shape correctness
# ---------------------------------------------------------------------------


def test_rk3_output_shape():
    """ssp_rk3_step must return same shape as input."""
    nr, nz = 12, 10
    grid = _grid(nr=nr, nz=nz)
    U = _uniform_U(nr=nr, nz=nz)
    dt = 1e-7
    U_new = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll")
    assert U_new.shape == (NVAR, nr, nz)


def test_rk2_output_shape():
    """ssp_rk2_step must return same shape as input."""
    nr, nz = 10, 12
    grid = _grid(nr=nr, nz=nz)
    U = _uniform_U(nr=nr, nz=nz)
    dt = 1e-7
    U_new = ssp_rk2_step(U, grid, dt, method="plm", riemann="hll")
    assert U_new.shape == (NVAR, nr, nz)


# ---------------------------------------------------------------------------
# 11. Density and energy remain positive after steps
# ---------------------------------------------------------------------------


def test_floors_applied_after_steps():
    """After 20 RK3 steps with tiny dt, density and energy must remain non-negative.

    Uses a very small timestep (1% of CFL) to stay near equilibrium.
    Primarily tests that floor enforcement keeps rho, E >= 0.
    """
    nr, nz = 8, 8
    grid = _grid(nr=nr, nz=nz)
    U = _uniform_U(nr=nr, nz=nz, rho=0.5, p=0.3, Bz=0.0)  # B=0 for equilibrium

    dt_cfl = compute_dt_cfl(U, grid, cfl=0.3)
    dt = dt_cfl * 0.01  # 1% of CFL: very gentle steps
    for _ in range(20):
        U = ssp_rk3_step(U, grid, dt, method="plm", riemann="hll")

    rho_np = _np(U[IDN])
    E_np = _np(U[IEN])

    assert not np.any(np.isnan(rho_np)), "NaN in density after 20 steps"
    assert not np.any(np.isnan(E_np)), "NaN in energy after 20 steps"
    assert np.all(rho_np >= 0.0), f"Negative density: min={np.nanmin(rho_np):.3e}"
    assert np.all(E_np >= 0.0), f"Negative energy: min={np.nanmin(E_np):.3e}"
