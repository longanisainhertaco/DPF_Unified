"""SSP-RK3 time integrator for the MLX MHD solver.

Implements the 3-stage Strong Stability Preserving Runge-Kutta method
(Shu & Osher 1988, Gottlieb et al. 2001):

  U^(1) = U^n + dt * L(U^n)
  U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
  U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

Key feature: dual-energy pressure recovery after EVERY stage to prevent
chain-rule cancellation from corrupting intermediate fluxes.

References:
    Shu C.-W. & Osher S., JCP 77:439 (1988) -- SSP-RK schemes.
    Gottlieb S. et al., SIAM Rev. 43:89 (2001) -- SSP review.
    Bryan et al., ApJS 211:19 (2014) -- dual-energy formalism.
    Popovas et al., arXiv:2211.02438 (2025) -- DISPATCH HLLS entropy switch.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from dpf.metal.mlx_grid import CylindricalGrid
from dpf.metal.mlx_kernels import (
    IBR,
    IBT,
    IBZ,
    IDN,
    IEN,
    IMR,
    IMT,
    IMZ,
    ISR,
    NVAR,
)
from dpf.metal.mlx_primitives import (
    P_FLOOR,
    RHO_FLOOR,
    cons_to_prim,
    fast_magnetosonic,
    recover_pressure_dual_energy,
)
from dpf.metal.mlx_riemann import mhd_rhs as _riemann_mhd_rhs

# Velocity clamping: cap at V_CLAMP_FACTOR * fast magnetosonic speed
_V_CLAMP_FACTOR: float = 10.0


# ---------------------------------------------------------------------------
# Spatial operator L(U)
# ---------------------------------------------------------------------------


def mhd_rhs(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float = 5.0 / 3.0,
    dr: float | None = None,
    dz: float | None = None,
    method: str = "weno5z",
    riemann: str = "hlld",
) -> mx.array:
    """Compute the MHD right-hand side dU/dt = L(U).

    Delegates to mlx_riemann.mhd_rhs which applies _clamp_reconstructed
    guards after WENO5-Z reconstruction (prevents negative-energy states from
    reaching the Riemann solver) and correctly handles dim=1 axis transposition
    for the HLLD Metal kernel.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid instance with geometry arrays.
        gamma: Adiabatic index (default 5/3).
        dr: Radial cell spacing [m]. Defaults to grid.dr.
        dz: Axial cell spacing [m]. Defaults to grid.dz.
        method: Reconstruction method: "weno5z" or "plm".
        riemann: Riemann solver: "hlld" or "hll".

    Returns:
        dU/dt array, shape (NVAR, nr, nz), float32.
    """
    return _riemann_mhd_rhs(
        U,
        grid,
        gamma=gamma,
        dr=grid.dr if dr is None else dr,
        dz=grid.dz if dz is None else dz,
        method=method,
        riemann=riemann,
    )


def _geometric_sources(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float,
) -> mx.array:
    """Compute cylindrical geometric source terms.

    S_mr = (rho*vt^2 - Bt^2) / r     [centrifugal + hoop stress]
    S_mt = -2*(rho*vr*vt - Br*Bt) / r [Coriolis + tension]

    These are velocity-space sources; energy source is v dot S.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        grid: CylindricalGrid with inv_r array.
        gamma: Adiabatic index.

    Returns:
        Source array dU/dt, shape (NVAR, nr, nz).
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bt = U[IBT]

    inv_r = grid.inv_r[:, None]  # (nr, 1) broadcast over z

    # Momentum sources (in momentum units, i.e. rho * a)
    S_mr = (rho * vt * vt - Bt * Bt) * inv_r
    S_mt = -2.0 * (rho * vr * vt - Br * Bt) * inv_r
    S_E = vr * S_mr + vt * S_mt

    rows = [mx.zeros_like(rho)] * NVAR
    rows[IMR] = S_mr
    rows[IMT] = S_mt
    rows[IEN] = S_E

    return mx.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# CFL timestep
# ---------------------------------------------------------------------------


def compute_dt_cfl(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float = 5.0 / 3.0,
    cfl: float = 0.3,
) -> float:
    """Compute CFL-limited timestep.

    dt = cfl * min(dr, dz) / max(|v| + cf)
    where cf is the fast magnetosonic speed (capped at c=3e8).
    NaN cells filtered before computing max.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        grid: CylindricalGrid with dr, dz.
        gamma: Adiabatic index (default 5/3).
        cfl: Courant number (default 0.3).

    Returns:
        dt [s], float.
    """
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)

    cf_r = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=0)
    cf_z = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=1)

    speed_r = mx.abs(vr) + cf_r
    speed_z = mx.abs(vz) + cf_z

    speed_r_np = np.asarray(speed_r)
    speed_z_np = np.asarray(speed_z)

    max_r = float(np.nanmax(speed_r_np))
    max_z = float(np.nanmax(speed_z_np))

    if not math.isfinite(max_r) or max_r == 0.0:
        max_r = 1.0
    if not math.isfinite(max_z) or max_z == 0.0:
        max_z = 1.0

    dx_min = min(grid.dr, grid.dz)
    dt = cfl * dx_min / max(max_r, max_z)
    return float(dt)


# ---------------------------------------------------------------------------
# Floor and velocity clamping
# ---------------------------------------------------------------------------


def _apply_floors(U: mx.array) -> mx.array:
    """Enforce density and pressure floors on conserved state.

    Clamps rho >= RHO_FLOOR. Pressure floor enforced implicitly via
    energy floor: E >= P_FLOOR/(gamma-1) + KE + ME.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).

    Returns:
        U with floors applied, same shape.
    """
    rows = list(mx.split(U, NVAR, axis=0))
    rows[IDN] = mx.maximum(rows[IDN], RHO_FLOOR)
    # Ensure non-negative density also for entropy tracer
    rows[ISR] = mx.maximum(rows[ISR], 0.0)
    return mx.stack([r[0] for r in rows], axis=0)


def _clamp_velocity(U: mx.array, gamma: float) -> mx.array:
    """Clamp velocity to _V_CLAMP_FACTOR * local fast magnetosonic speed.

    Prevents extreme velocities at low-density vacuum cells from blowing
    up intermediate RK stages.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        gamma: Adiabatic index.

    Returns:
        U with velocity clamped, same shape.
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    p = mx.maximum(gm1 * (U[IEN] - KE - ME), P_FLOOR)

    cf = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=0)
    v_max = _V_CLAMP_FACTOR * cf  # (nr, nz)

    v_mag = mx.sqrt(mx.maximum(vr * vr + vz * vz + vt * vt, 0.0))
    scale = mx.where(v_mag > v_max, v_max / mx.maximum(v_mag, 1e-30), mx.ones_like(v_mag))

    vr_c = vr * scale
    vz_c = vz * scale
    vt_c = vt * scale

    KE_c = 0.5 * rho * (vr_c * vr_c + vz_c * vz_c + vt_c * vt_c)
    E_c = p / gm1 + KE_c + ME

    rows = list(mx.split(U, NVAR, axis=0))
    rows[IMR] = (rho * vr_c)[None]
    rows[IMZ] = (rho * vz_c)[None]
    rows[IMT] = (rho * vt_c)[None]
    rows[IEN] = E_c[None]
    return mx.stack([r[0] for r in rows], axis=0)


# ---------------------------------------------------------------------------
# Dual-energy pressure resync at intermediate stages
# ---------------------------------------------------------------------------


def _resync_energy(U: mx.array, gamma: float) -> mx.array:
    """Recover pressure from dual-energy and rewrite U[IEN] for consistency.

    Prevents chain-rule cancellation in float32 from leaking into the next
    RK stage. After recovery, E is reconstructed from p_recovered + KE + ME
    so the next call to cons_to_prim gets a non-corrupted energy.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        gamma: Adiabatic index.

    Returns:
        U with IEN rewritten to match recovered pressure.
    """
    p, _ = recover_pressure_dual_energy(U, gamma)
    rho, vr, vz, vt, _, Br, Bz, Bt = cons_to_prim(U, gamma)

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    E_new = p / gm1 + KE + ME

    rows = list(mx.split(U, NVAR, axis=0))
    rows[IEN] = E_new[None]
    return mx.stack([r[0] for r in rows], axis=0)


# ---------------------------------------------------------------------------
# SSP-RK3 integrator
# ---------------------------------------------------------------------------


def ssp_rk3_step(
    U: mx.array,
    grid: CylindricalGrid,
    dt: float,
    gamma: float = 5.0 / 3.0,
    method: str = "weno5z",
    riemann: str = "hlld",
    use_dual_energy: bool = True,
) -> mx.array:
    """Advance U by one SSP-RK3 timestep.

    At each intermediate stage:
    1. Compute L(U) via mhd_rhs
    2. SSP combination
    3. Enforce density/pressure floors
    4. If use_dual_energy: recover pressure from conservative E + entropy tracer
    5. Clamp velocity to _V_CLAMP_FACTOR * fast magnetosonic speed

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid instance.
        dt: Timestep [s].
        gamma: Adiabatic index (default 5/3).
        method: Reconstruction method: "weno5z" or "plm".
        riemann: Riemann solver: "hlld" or "hll".
        use_dual_energy: Apply dual-energy pressure recovery at each stage.

    Returns:
        U_new, shape (NVAR, nr, nz), float32.
    """
    dr, dz = grid.dr, grid.dz

    def _stage_post(Uk: mx.array) -> mx.array:
        Uk = _apply_floors(Uk)
        if use_dual_energy:
            Uk = _resync_energy(Uk, gamma)
        Uk = _clamp_velocity(Uk, gamma)
        return Uk

    # Stage 1: U1 = Un + dt * L(Un)
    L1 = mhd_rhs(U, grid, gamma, dr, dz, method, riemann)
    U1 = U + dt * L1
    U1 = _stage_post(U1)

    # Stage 2: U2 = 3/4 * Un + 1/4 * (U1 + dt * L(U1))
    L2 = mhd_rhs(U1, grid, gamma, dr, dz, method, riemann)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = _stage_post(U2)

    # Stage 3: Un+1 = 1/3 * Un + 2/3 * (U2 + dt * L(U2))
    L3 = mhd_rhs(U2, grid, gamma, dr, dz, method, riemann)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    U_new = _stage_post(U_new)

    mx.eval(U_new)
    return U_new


# ---------------------------------------------------------------------------
# SSP-RK2 integrator
# ---------------------------------------------------------------------------


def ssp_rk2_step(
    U: mx.array,
    grid: CylindricalGrid,
    dt: float,
    gamma: float = 5.0 / 3.0,
    method: str = "plm",
    riemann: str = "hlld",
    use_dual_energy: bool = True,
) -> mx.array:
    """Advance U by one SSP-RK2 timestep (simpler, for testing).

    Scheme (Shu & Osher 1988, 2-stage):
        U^(1) = U^n + dt * L(U^n)
        U^(n+1) = 1/2 * U^n + 1/2 * (U^(1) + dt * L(U^(1)))

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid instance.
        dt: Timestep [s].
        gamma: Adiabatic index (default 5/3).
        method: Reconstruction method: "plm".
        riemann: Riemann solver: "hlld" or "hll".
        use_dual_energy: Apply dual-energy pressure recovery at each stage.

    Returns:
        U_new, shape (NVAR, nr, nz), float32.
    """
    dr, dz = grid.dr, grid.dz

    def _stage_post(Uk: mx.array) -> mx.array:
        Uk = _apply_floors(Uk)
        if use_dual_energy:
            Uk = _resync_energy(Uk, gamma)
        Uk = _clamp_velocity(Uk, gamma)
        return Uk

    # Stage 1
    L1 = mhd_rhs(U, grid, gamma, dr, dz, method, riemann)
    U1 = U + dt * L1
    U1 = _stage_post(U1)

    # Stage 2
    L2 = mhd_rhs(U1, grid, gamma, dr, dz, method, riemann)
    U_new = 0.5 * U + 0.5 * (U1 + dt * L2)
    U_new = _stage_post(U_new)

    mx.eval(U_new)
    return U_new
