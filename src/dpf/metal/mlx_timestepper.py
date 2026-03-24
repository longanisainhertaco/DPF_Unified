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
    hlld_flux_mlx,
)
from dpf.metal.mlx_primitives import (
    P_FLOOR,
    RHO_FLOOR,
    cons_to_prim,
    fast_magnetosonic,
    recover_pressure_dual_energy,
)
from dpf.metal.mlx_reconstruction import reconstruct

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

    Implements conservative cylindrical finite-volume differencing:

        dU/dt = -(1/(r*dr)) * d(r*F_r)/dr - dF_z/dz + S_geom

    where S_geom is the cylindrical geometric source (centrifugal + hoop stress).

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
    if dr is None:
        dr = grid.dr
    if dz is None:
        dz = grid.dz

    nr, nz = grid.nr, grid.nz
    dU = mx.zeros_like(U)

    # --- Radial flux divergence ---
    if nr >= 2:
        QL_r, QR_r = reconstruct(U, dim=0, method=method)
        F_r = _riemann_flux(QL_r, QR_r, gamma=gamma, dim=0, riemann=riemann)
        # r-weighted divergence: (1/(r_cell*dr)) * (r_face[i+1]*F[i+1/2] - r_face[i]*F[i-1/2])
        # F_r shape: (NVAR, nr-1, nz) -- fluxes at interfaces i+1/2 for i=0..nr-2
        r_face = grid.r_face  # shape (nr+1,)
        r_cell = grid.r_cell      # shape (nr,)

        # Interior cells i=1..nr-2 get contributions from both faces
        # Cell i gets: rF_r[i-1] - rF_l[i-1] ... but F_r[k] is flux at i=k+1/2
        # So cell i: div = (r_{i+1/2}*F_{i+1/2} - r_{i-1/2}*F_{i-1/2}) / (r_i * dr)
        # F_r[k] = flux between cells k and k+1
        # Cell 0: has right flux F_r[0] only (no left flux from interior)
        # Cell i (1..nr-2): right=F_r[i], left=F_r[i-1]
        # Cell nr-1: has left flux F_r[nr-2] only

        inv_r_dr = (1.0 / (mx.maximum(r_cell, 1e-30) * dr))  # shape (nr,)
        inv_r_dr_bc = inv_r_dr[None, :, None]  # (1, nr, 1)

        # Build (r*F) at each interface; F_r[k] lives at face k+1/2
        # r*F right face for cell i: r_face[i+1] * F_r[i]
        # r*F left  face for cell i: r_face[i]   * F_r[i-1]
        r_face_full = r_face[None, :, None]  # (1, nr+1, 1)

        # Flux divergence via scatter:
        # dU[:, i, :] -= (r_face[i+1]*F_r[i] - r_face[i]*F_r[i-1]) / (r_i * dr)
        # Use zero-padded approach: pad F_r with zeros at both ends
        zero_pad = mx.zeros((NVAR, 1, nz), dtype=U.dtype)
        F_r_padded = mx.concatenate([zero_pad, F_r, zero_pad], axis=1)  # (NVAR, nr+1, nz)

        # r * F at each face k (k=0..nr): r_face[k] * F_r_padded[k]
        rF_all = r_face_full * F_r_padded  # (NVAR, nr+1, nz)

        # Divergence: rF_all[:, 1:nr+1, :] - rF_all[:, 0:nr, :]
        div_r = (rF_all[:, 1:, :] - rF_all[:, :-1, :]) * inv_r_dr_bc
        dU = dU - div_r

    # --- Axial flux divergence ---
    if nz >= 2:
        QL_z, QR_z = reconstruct(U, dim=1, method=method)
        F_z = _riemann_flux(QL_z, QR_z, gamma=gamma, dim=1, riemann=riemann)
        # Standard Cartesian divergence along z
        zero_pad_z = mx.zeros((NVAR, nr, 1), dtype=U.dtype)
        F_z_padded = mx.concatenate([zero_pad_z, F_z, zero_pad_z], axis=2)  # (NVAR, nr, nz+1)
        div_z = (F_z_padded[:, :, 1:] - F_z_padded[:, :, :-1]) / dz
        dU = dU - div_z

    # --- Cylindrical geometric sources ---
    dU = dU + _geometric_sources(U, grid, gamma)

    mx.eval(dU)
    return dU


def _riemann_flux(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
    riemann: str,
) -> mx.array:
    """Dispatch to HLLD or HLL Riemann solver.

    Args:
        QL: Left state at interfaces, shape (NVAR, n_ifaces, nz).
        QR: Right state at interfaces, shape (NVAR, n_ifaces, nz).
        gamma: Adiabatic index.
        dim: Normal direction (0=radial, 1=axial).
        riemann: "hlld" or "hll".

    Returns:
        Numerical flux, shape (NVAR, n_ifaces, nz).
    """
    if riemann == "hlld":
        return hlld_flux_mlx(QL, QR, gamma=gamma, dim=dim)
    # HLL fallback
    return _hll_flux(QL, QR, gamma=gamma, dim=dim)


def _hll_flux(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """HLL two-wave Riemann flux via NumPy bridge (float64 for safety).

    Args:
        QL: Left state, shape (NVAR, n_ifaces, nz).
        QR: Right state, shape (NVAR, n_ifaces, nz).
        gamma: Adiabatic index.
        dim: Normal direction.

    Returns:
        Numerical flux, shape (NVAR, n_ifaces, nz).
    """
    # Use float64 to avoid overflow at high-velocity cells
    QL_np = np.asarray(QL).astype(np.float64)
    QR_np = np.asarray(QR).astype(np.float64)
    TINY = 1e-20

    rho_L = np.maximum(QL_np[IDN], RHO_FLOOR)
    rho_R = np.maximum(QR_np[IDN], RHO_FLOOR)

    # Select normal / tangential indices
    if dim == 0:
        im_n, im_t1, im_t2 = IMR, IMZ, IMT
        ib_n, ib_t1, ib_t2 = IBR, IBZ, IBT
    else:
        im_n, im_t1, im_t2 = IMZ, IMR, IMT
        ib_n, ib_t1, ib_t2 = IBZ, IBR, IBT

    inv_rL = 1.0 / rho_L
    inv_rR = 1.0 / rho_R

    vn_L = QL_np[im_n] * inv_rL
    vn_R = QR_np[im_n] * inv_rR
    Bn_L = QL_np[ib_n]
    Bn_R = QR_np[ib_n]
    Bt1_L = QL_np[ib_t1]
    Bt1_R = QR_np[ib_t1]
    Bt2_L = QL_np[ib_t2]
    Bt2_R = QR_np[ib_t2]

    gm1 = gamma - 1.0
    KE_L = 0.5 * rho_L * (
        (QL_np[IMR] * inv_rL) ** 2 +
        (QL_np[IMZ] * inv_rL) ** 2 +
        (QL_np[IMT] * inv_rL) ** 2
    )
    KE_R = 0.5 * rho_R * (
        (QR_np[IMR] * inv_rR) ** 2 +
        (QR_np[IMZ] * inv_rR) ** 2 +
        (QR_np[IMT] * inv_rR) ** 2
    )
    B2_L = QL_np[IBR] ** 2 + QL_np[IBZ] ** 2 + QL_np[IBT] ** 2
    B2_R = QR_np[IBR] ** 2 + QR_np[IBZ] ** 2 + QR_np[IBT] ** 2
    ME_L = 0.5 * B2_L
    ME_R = 0.5 * B2_R
    p_L = np.maximum(gm1 * (QL_np[IEN] - KE_L - ME_L), P_FLOOR)
    p_R = np.maximum(gm1 * (QR_np[IEN] - KE_R - ME_R), P_FLOOR)

    # Fast magnetosonic speeds
    Bt_sq_L = np.maximum(B2_L - Bn_L ** 2, 0.0)
    Bt_sq_R = np.maximum(B2_R - Bn_R ** 2, 0.0)
    a_sq_L = np.minimum(gamma * p_L / rho_L, (3e8) ** 2)
    a_sq_R = np.minimum(gamma * p_R / rho_R, (3e8) ** 2)
    va_sq_L = np.minimum(B2_L / rho_L, (3e8) ** 2)
    va_sq_R = np.minimum(B2_R / rho_R, (3e8) ** 2)
    vat_sq_L = np.minimum(Bt_sq_L / rho_L, (3e8) ** 2)
    vat_sq_R = np.minimum(Bt_sq_R / rho_R, (3e8) ** 2)

    diff_L = a_sq_L - va_sq_L
    disc_L = np.maximum(diff_L ** 2 + 4.0 * a_sq_L * vat_sq_L, 0.0)
    cf_L = np.sqrt(np.maximum(0.5 * (a_sq_L + va_sq_L + np.sqrt(disc_L)), 0.0))

    diff_R = a_sq_R - va_sq_R
    disc_R = np.maximum(diff_R ** 2 + 4.0 * a_sq_R * vat_sq_R, 0.0)
    cf_R = np.sqrt(np.maximum(0.5 * (a_sq_R + va_sq_R + np.sqrt(disc_R)), 0.0))

    SL = np.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = np.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = np.maximum(SR, SL + TINY)

    def _phys_flux(rho, inv_r, prim_vn, prim_vt1, prim_vt2, p, Bn, Bt1, Bt2, E, U_arr):
        B2 = Bn ** 2 + Bt1 ** 2 + Bt2 ** 2
        pt = p + 0.5 * B2
        F = np.zeros_like(U_arr)
        vB = prim_vn * Bn + prim_vt1 * Bt1 + prim_vt2 * Bt2
        F[IDN] = rho * prim_vn
        F[im_n] = rho * prim_vn * prim_vn + pt - Bn * Bn
        F[im_t1] = rho * prim_vn * prim_vt1 - Bn * Bt1
        F[im_t2] = rho * prim_vn * prim_vt2 - Bn * Bt2
        F[IEN] = (E + pt) * prim_vn - Bn * vB
        F[ISR] = U_arr[ISR] * prim_vn
        F[ib_n] = 0.0
        F[ib_t1] = prim_vn * Bt1 - prim_vt1 * Bn
        F[ib_t2] = prim_vn * Bt2 - prim_vt2 * Bn
        return F

    vt1_L = QL_np[im_t1] * inv_rL
    vt2_L = QL_np[im_t2] * inv_rL
    vt1_R = QR_np[im_t1] * inv_rR
    vt2_R = QR_np[im_t2] * inv_rR

    FL = _phys_flux(rho_L, inv_rL, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L, QL_np[IEN], QL_np)
    FR = _phys_flux(rho_R, inv_rR, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R, QR_np[IEN], QR_np)

    inv_dS = 1.0 / np.maximum(SR - SL, TINY)
    F_hll = (SR[np.newaxis] * FL - SL[np.newaxis] * FR + SL[np.newaxis] * SR[np.newaxis] * (QR_np - QL_np)) * inv_dS[np.newaxis]

    F_out = np.where(SL[np.newaxis] >= 0.0, FL, np.where(SR[np.newaxis] <= 0.0, FR, F_hll))
    F_out[ib_n] = 0.0

    has_nan = np.isnan(F_out) | np.isinf(F_out)
    if np.any(has_nan):
        S_max = np.maximum(np.abs(SL), np.abs(SR))
        F_LF = 0.5 * (FL + FR) - 0.5 * S_max[np.newaxis] * (QR_np - QL_np)
        F_out = np.where(has_nan, F_LF, F_out)

    return mx.array(F_out.astype(np.float32))


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
