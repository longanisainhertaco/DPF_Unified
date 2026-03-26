"""Braginskii parallel viscosity for the MLX cylindrical MHD solver.

Implements isotropic parallel viscosity (eta_0) operator-split step.
Explicit update with sub-cycling when viscous CFL < MHD timestep.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1, 205 (1965).
    NRL Plasma Formulary (2019), p. 31-34, ion collision time.
    Miyoshi & Kusano (2005), JCP 208:315 -- variable layout.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from dpf.metal.mlx_kernels import (
    IBR,
    IBT,
    IBZ,
    IDN,
    IEN,
    IMR,
    IMT,
    IMZ,
)
from dpf.metal.mlx_primitives import P_FLOOR, RHO_FLOOR

# Physical constants (SI)
_KB: float = 1.380649e-23
_EV_TO_K: float = 11604.5
_K_TO_EV: float = 1.0 / _EV_TO_K
_M_D: float = 3.3435e-27

# NRL Formulary ion collision time coefficient [s * m^3 * K^{-1.5}]
# tau_i = _TAU_COEFF * Ti_eV^1.5 * sqrt(A) / (ni_cm3 * lnL)
_TAU_COEFF: float = 2.09e7

# Sub-cycle cap (matches resistive diffusion convention)
_MAX_SUBCYCLES: int = 20


# ---------------------------------------------------------------------------
# Viscosity coefficients
# ---------------------------------------------------------------------------


def braginskii_viscosity_coefficients(
    rho: mx.array,
    Ti: mx.array,
    B_mag: mx.array,
    ion_mass: float = _M_D,
) -> tuple[mx.array, mx.array]:
    """Compute Braginskii parallel and perpendicular viscosity coefficients.

    NRL Formulary: tau_i = 2.09e7 * Ti_eV^1.5 * sqrt(A) / (ni_cm3 * lnL)
    eta_0 = 0.96 * ni * kB * Ti * tau_i  (parallel)
    eta_1 = eta_0 / (1 + (omega_ci * tau_i)^2)  (perpendicular, strongly suppressed)

    Args:
        rho: Mass density [kg/m^3], shape (...).
        Ti: Ion temperature [K], shape (...).
        B_mag: Magnetic field magnitude [T], shape (...).
        ion_mass: Ion mass [kg]. Default: deuterium.

    Returns:
        (eta_0, eta_1): Parallel and perpendicular viscosity [Pa.s].
    """
    Ti_safe = mx.maximum(Ti, 1.0)
    ni_safe = mx.maximum(rho / ion_mass, 1.0e10)

    Ti_eV = Ti_safe * _K_TO_EV

    # Number density in cm^-3
    ni_cm3 = ni_safe * 1.0e-6

    # Atomic mass number
    A = ion_mass / 1.6605e-27

    lnL = 10.0  # simplified Coulomb logarithm

    tau_i = _TAU_COEFF * (Ti_eV ** 1.5) * math.sqrt(A) / (ni_cm3 * lnL)

    eta_0 = 0.96 * ni_safe * _KB * Ti_safe * tau_i

    # Ion cyclotron frequency: omega_ci = e * B / m_i
    e_charge = 1.602176634e-19
    omega_ci = e_charge * mx.maximum(B_mag, 1e-30) / ion_mass
    x_i = omega_ci * tau_i  # magnetization parameter

    # Perpendicular viscosity: eta_1 = eta_0 / (1 + x_i^2)
    # For strongly magnetized plasma (x_i >> 1): eta_1 << eta_0
    eta_1 = eta_0 / (1.0 + x_i * x_i)

    return eta_0, eta_1


# ---------------------------------------------------------------------------
# Gradient helpers and strain rate tensor
# ---------------------------------------------------------------------------


def _grad_r(f: mx.array, dx: float) -> mx.array:
    """Central-difference derivative along axis 0 with one-sided boundaries."""
    n = f.shape[0]
    if n == 1:
        return mx.zeros_like(f)
    lo = (f[1:2, :] - f[0:1, :]) / dx
    hi = (f[-1:, :] - f[-2:-1, :]) / dx
    if n == 2:
        return mx.concatenate([lo, hi], axis=0)
    return mx.concatenate([lo, (f[2:, :] - f[:-2, :]) * (0.5 / dx), hi], axis=0)


def _grad_z(f: mx.array, dx: float) -> mx.array:
    """Central-difference derivative along axis 1 with one-sided boundaries."""
    n = f.shape[1]
    if n == 1:
        return mx.zeros_like(f)
    lo = (f[:, 1:2] - f[:, 0:1]) / dx
    hi = (f[:, -1:] - f[:, -2:-1]) / dx
    if n == 2:
        return mx.concatenate([lo, hi], axis=1)
    return mx.concatenate([lo, (f[:, 2:] - f[:, :-2]) * (0.5 / dx), hi], axis=1)


def compute_strain_rate_cylindrical(
    vr: mx.array,
    vz: mx.array,
    vt: mx.array,
    dr: float,
    dz: float,
    inv_r: mx.array,
) -> dict[str, mx.array]:
    """Compute axisymmetric strain rate tensor components via central differences.

    Args:
        vr, vz, vt: Velocity components [m/s], shape (nr, nz).
        dr, dz: Cell spacings [m].
        inv_r: 1/r at cell centres (L'Hopital value at axis), shape (nr, 1) or (nr, nz).

    Returns:
        Dict with S_rr, S_zz, S_tt, S_rz, S_rt, S_tz, S_trace.
    """
    dvr_dr = _grad_r(vr, dr)
    dvz_dz = _grad_z(vz, dz)
    S_rr = dvr_dr
    S_zz = dvz_dz
    S_tt = vr * inv_r
    S_rz = 0.5 * (_grad_z(vr, dz) + _grad_r(vz, dr))
    S_rt = 0.5 * (_grad_r(vt, dr) - vt * inv_r)
    S_tz = 0.5 * _grad_z(vt, dz)
    S_trace = S_rr + S_tt + S_zz
    return {
        "S_rr": S_rr, "S_zz": S_zz, "S_tt": S_tt,
        "S_rz": S_rz, "S_rt": S_rt, "S_tz": S_tz,
        "S_trace": S_trace,
    }


# ---------------------------------------------------------------------------
# Core viscous update (single sub-step)
# ---------------------------------------------------------------------------


def _viscous_substep(
    U: mx.array,
    dt_sub: float,
    dr: float,
    dz: float,
    inv_r: mx.array,
    gamma: float,
    ion_mass: float,
) -> mx.array:
    """Apply one explicit Braginskii viscosity sub-step.

    Returns updated conserved state U_new with same shape as U.
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    v2 = vr * vr + vz * vz + vt * vt
    B2 = U[IBR] * U[IBR] + U[IBZ] * U[IBZ] + U[IBT] * U[IBT]
    p = mx.maximum((gamma - 1.0) * (U[IEN] - 0.5 * rho * v2 - 0.5 * B2), P_FLOOR)

    # Ion temperature (factor of 2 for n_e + n_i at Z=1)
    Ti = p * ion_mass / (2.0 * rho * _KB)

    B_mag = mx.sqrt(mx.maximum(B2, 0.0))
    eta_0, eta_1 = braginskii_viscosity_coefficients(rho, Ti, B_mag, ion_mass)

    S = compute_strain_rate_cylindrical(vr, vz, vt, dr, dz, inv_r)
    S_trace_3 = S["S_trace"] / 3.0

    # Anisotropic Braginskii stress tensor.
    # For weakly magnetized plasma (x_i << 1): eta_0 ~ eta_1, isotropic.
    # For strongly magnetized (x_i >> 1): eta_1 ~ 0, only parallel stress.
    #
    # Parallel rate of strain: S_par = b_i*b_j*S_ij - S_trace/3
    # Stress: sigma_ij = -eta_0 * (3*b_i*b_j - delta_ij) * S_par
    #         + eta_1 * (S_ij - delta_ij*S_trace/3 - (3*b_i*b_j-delta_ij)*S_par)
    Br_ = U[IBR]
    Bz_ = U[IBZ]
    Bt_ = U[IBT]
    B_inv = 1.0 / mx.maximum(B_mag, 1e-30)
    br = Br_ * B_inv
    bz = Bz_ * B_inv
    bt = Bt_ * B_inv

    # Parallel strain: b.S.b = br^2*S_rr + bz^2*S_zz + bt^2*S_tt + 2*br*bz*S_rz
    S_par = (br * br * S["S_rr"] + bz * bz * S["S_zz"] + bt * bt * S["S_tt"]
             + 2.0 * br * bz * S["S_rz"]) - S_trace_3

    # Parallel contribution (eta_0): sigma^(0) = -eta_0 * (3*bi*bj - delta_ij) * S_par
    sig_rr = -eta_0 * (3.0 * br * br - 1.0) * S_par
    sig_zz = -eta_0 * (3.0 * bz * bz - 1.0) * S_par
    sig_tt = -eta_0 * (3.0 * bt * bt - 1.0) * S_par
    sig_rz = -eta_0 * 3.0 * br * bz * S_par
    sig_rt = -eta_0 * 3.0 * br * bt * S_par
    sig_tz = -eta_0 * 3.0 * bz * bt * S_par

    # Perpendicular contribution (eta_1): add the isotropic remainder
    # sigma^(1) = eta_1 * (S_ij - delta_ij*S/3 - W^(0)_ij)
    # For strongly magnetized plasma, eta_1 ~ 0, so this is negligible.
    # Include for correctness at intermediate magnetization.
    sig_rr = sig_rr + eta_1 * (S["S_rr"] - S_trace_3 + (3.0 * br * br - 1.0) * S_par)
    sig_zz = sig_zz + eta_1 * (S["S_zz"] - S_trace_3 + (3.0 * bz * bz - 1.0) * S_par)
    sig_tt = sig_tt + eta_1 * (S["S_tt"] - S_trace_3 + (3.0 * bt * bt - 1.0) * S_par)
    sig_rz = sig_rz + eta_1 * (S["S_rz"] + 3.0 * br * bz * S_par)
    sig_rt = sig_rt + eta_1 * (S["S_rt"] + 3.0 * br * bt * S_par)
    sig_tz = sig_tz + eta_1 * (S["S_tz"] + 3.0 * bz * bt * S_par)

    # Divergence of stress tensor in cylindrical coordinates
    # (div sigma)_r = d(sig_rr)/dr + d(sig_rz)/dz + (sig_rr - sig_tt)/r
    div_r = _grad_r(sig_rr, dr) + _grad_z(sig_rz, dz) + (sig_rr - sig_tt) * inv_r
    # (div sigma)_z = d(sig_rz)/dr + d(sig_zz)/dz + sig_rz/r
    div_z = _grad_r(sig_rz, dr) + _grad_z(sig_zz, dz) + sig_rz * inv_r
    # (div sigma)_t = d(sig_rt)/dr + d(sig_tz)/dz + 2*sig_rt/r
    div_t = _grad_r(sig_rt, dr) + _grad_z(sig_tz, dz) + 2.0 * sig_rt * inv_r

    # Viscous acceleration
    a_r = div_r * inv_rho
    a_z = div_z * inv_rho
    a_t = div_t * inv_rho

    # Viscous heating rate: Q = sigma_ij * S_ij (positive definite for traceless)
    Q_visc = (
        sig_rr * S["S_rr"]
        + sig_zz * S["S_zz"]
        + sig_tt * S["S_tt"]
        + 2.0 * sig_rz * S["S_rz"]
        + 2.0 * sig_rt * S["S_rt"]
        + 2.0 * sig_tz * S["S_tz"]
    )
    Q_visc = mx.maximum(Q_visc, 0.0)

    # Update velocity
    vr_new = vr + dt_sub * a_r
    vz_new = vz + dt_sub * a_z
    vt_new = vt + dt_sub * a_t

    # Rebuild conserved state (only mutated components)
    v2_new = vr_new * vr_new + vz_new * vz_new + vt_new * vt_new
    E_new = p / (gamma - 1.0) + 0.5 * rho * v2_new + 0.5 * B2 + dt_sub * Q_visc

    # Replace IDN(0), IMR(1), IMZ(2), IMT(3), IEN(4); keep indices 5-9 unchanged
    head = mx.concatenate([
        rho[None],
        (rho * vr_new)[None],
        (rho * vz_new)[None],
        (rho * vt_new)[None],
        E_new[None],
    ], axis=0)
    return mx.concatenate([head, U[5:]], axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_braginskii_viscosity(
    U: mx.array,
    dt: float,
    grid: object,
    gamma: float = 5.0 / 3.0,
    ion_mass: float = _M_D,
) -> mx.array:
    """Apply Braginskii parallel viscosity operator-split step.

    Explicit update with sub-cycling when viscous CFL is more restrictive
    than the supplied dt. Caps sub-cycles at 20.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        dt: MHD timestep [s].
        grid: Grid object with attributes dr, dz, inv_r (shape (nr,1) or
            (nr,nz)).
        gamma: Adiabatic index.
        ion_mass: Ion mass [kg]. Default: deuterium.

    Returns:
        U_new: Updated conserved state, same shape as U.
    """
    dr: float = float(grid.dr)
    dz: float = float(grid.dz)
    inv_r: mx.array = grid.inv_r

    # Viscous CFL estimate using cell-averaged eta_0
    rho_np = np.asarray(U[IDN], dtype=np.float64)
    rho_min = float(np.maximum(rho_np.min(), RHO_FLOOR))

    # Quick eta_0 estimate from mean state
    mr_np = np.asarray(U[IMR], dtype=np.float64)
    mz_np = np.asarray(U[IMZ], dtype=np.float64)
    mt_np = np.asarray(U[IMT], dtype=np.float64)
    E_np = np.asarray(U[IEN], dtype=np.float64)
    Br_np = np.asarray(U[IBR], dtype=np.float64)
    Bz_np = np.asarray(U[IBZ], dtype=np.float64)
    Bt_np = np.asarray(U[IBT], dtype=np.float64)

    rho_np_safe = np.maximum(rho_np, RHO_FLOOR)
    v2_np = ((mr_np / rho_np_safe) ** 2
             + (mz_np / rho_np_safe) ** 2
             + (mt_np / rho_np_safe) ** 2)
    B2_np = Br_np ** 2 + Bz_np ** 2 + Bt_np ** 2
    p_np = np.maximum((gamma - 1.0) * (E_np - 0.5 * rho_np_safe * v2_np - 0.5 * B2_np),
                      P_FLOOR)
    Ti_np = p_np * ion_mass / (2.0 * rho_np_safe * _KB)
    ni_np = np.maximum(rho_np_safe / ion_mass, 1.0e10)
    Ti_eV_np = np.maximum(Ti_np * _K_TO_EV, 1.0 / _EV_TO_K)
    A = ion_mass / 1.6605e-27
    tau_i_np = _TAU_COEFF * (Ti_eV_np ** 1.5) * math.sqrt(A) / (ni_np * 1.0e-6 * 10.0)
    eta0_np = 0.96 * ni_np * _KB * np.maximum(Ti_np, 1.0) * tau_i_np
    # Use max eta_0 for CFL (eta_1 is always <= eta_0)
    eta0_max = float(eta0_np.max())

    dx_min = min(dr, dz)
    D = 2  # axisymmetric
    if eta0_max > 0.0:
        dt_visc = dx_min * dx_min * rho_min / (2.0 * D * eta0_max)
    else:
        dt_visc = dt

    n_sub = min(int(math.ceil(dt / dt_visc)) if dt_visc < dt else 1, _MAX_SUBCYCLES)
    dt_sub = dt / n_sub

    U_out = U
    for _ in range(n_sub):
        U_out = _viscous_substep(U_out, dt_sub, dr, dz, inv_r, gamma, ion_mass)

    return U_out
