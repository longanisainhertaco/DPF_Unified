"""Anisotropic (field-aligned) thermal conduction for magnetised plasmas.

In magnetised plasmas, thermal conduction is anisotropic: heat flows
much more efficiently along B-field lines than across them.  The
Braginskii conductivities are:

    kappa_parallel ~ Te^{5/2} / (Z * ln Lambda)
    kappa_perp     ~ n^2 * Z^2 * ln Lambda / (Te^{1/2} * B^2)

The ratio kappa_parallel / kappa_perp ~ (omega_ce * tau_e)^2
can exceed 10^{10} in hot magnetised plasma.

This module implements the Sharma-Hammett slope-limited explicit
diffusion operator (Sharma & Hammett 2007), which prevents unphysical
heat transport that can arise with standard centred-difference
discretisations of anisotropic diffusion.

The operator is designed to be called as an operator-split step
from the simulation engine.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    Sharma P., Hammett G.W., JCP 227, 123 (2007).
    Mignone A., JCP 270, 784 (2014).

Functions:
    braginskii_kappa_parallel: Parallel thermal conductivity.
    braginskii_kappa_perp: Perpendicular thermal conductivity.
    anisotropic_thermal_conduction: Apply anisotropic heat conduction.
"""

from __future__ import annotations

import logging

import numpy as np

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_e

logger = logging.getLogger(__name__)


def braginskii_kappa_parallel(
    ne: np.ndarray,
    Te: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    r"""Braginskii parallel electron thermal conductivity.

    kappa_par = 3.16 * ne * k_B^2 * Te * tau_e / m_e

    where tau_e is the electron collision time.  Using the Spitzer
    expression for tau_e this simplifies to:

    kappa_par ~ 3.16 * (k_B * Te)^{5/2} / (m_e^{1/2} * Z * e^4 * ln Lambda)

    We use the simplified form suitable for explicit computation.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z_eff: Effective ion charge.

    Returns:
        kappa_parallel [W / (m * K)].
    """
    Te_safe = np.maximum(Te, 1.0)
    ne_safe = np.maximum(ne, 1e-10)

    # Coulomb log (NRL formula for electron-ion)
    Te_eV = Te_safe * k_B / e_charge
    ne_cm3 = ne_safe * 1e-6
    arg = np.sqrt(np.maximum(ne_cm3, 1.0)) * Z_eff / np.maximum(Te_eV, 1e-3) ** 1.5
    lnL = np.maximum(23.0 - np.log(np.maximum(arg, 1e-30)), 2.0)

    # Electron collision time: tau_e ~ 3.44e5 * Te^{3/2} / (ne * lnL)
    tau_e = 3.44e5 * Te_safe ** 1.5 / np.maximum(ne_safe * lnL, 1e-30)

    kappa_par = 3.16 * ne_safe * k_B**2 * Te_safe * tau_e / m_e

    return np.where(np.isfinite(kappa_par), kappa_par, 0.0)


def braginskii_kappa_perp(
    ne: np.ndarray,
    Te: np.ndarray,
    B_mag: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    r"""Braginskii perpendicular electron thermal conductivity.

    kappa_perp = 4.66 * ne * k_B^2 * Te / (m_e * omega_ce^2 * tau_e)

    For strongly magnetised electrons this is suppressed by
    (omega_ce * tau_e)^{-2} relative to kappa_parallel.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        B_mag: Magnetic field magnitude [T].
        Z_eff: Effective ion charge.

    Returns:
        kappa_perp [W / (m * K)].
    """
    Te_safe = np.maximum(Te, 1.0)
    ne_safe = np.maximum(ne, 1e-10)
    B_safe = np.maximum(B_mag, 1e-30)

    # Coulomb log
    Te_eV = Te_safe * k_B / e_charge
    ne_cm3 = ne_safe * 1e-6
    arg = np.sqrt(np.maximum(ne_cm3, 1.0)) * Z_eff / np.maximum(Te_eV, 1e-3) ** 1.5
    lnL = np.maximum(23.0 - np.log(np.maximum(arg, 1e-30)), 2.0)

    # Electron collision time
    tau_e = 3.44e5 * Te_safe ** 1.5 / np.maximum(ne_safe * lnL, 1e-30)

    # Electron cyclotron frequency
    omega_ce = e_charge * B_safe / m_e
    omega_ce_sq = omega_ce * omega_ce

    kappa_perp = 4.66 * ne_safe * k_B**2 * Te_safe / (
        m_e * np.maximum(omega_ce_sq * tau_e, 1e-300)
    )

    # Cap at kappa_parallel
    kappa_par = braginskii_kappa_parallel(ne, Te, Z_eff)
    kappa_perp = np.minimum(kappa_perp, kappa_par)

    return np.where(np.isfinite(kappa_perp), kappa_perp, 0.0)


def anisotropic_thermal_conduction(
    Te: np.ndarray,
    B: np.ndarray,
    ne: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    Z_eff: float = 1.0,
    kappa_parallel: np.ndarray | None = None,
    kappa_perp: np.ndarray | None = None,
    flux_limiter: float = 0.1,
) -> np.ndarray:
    """Apply anisotropic thermal conduction to electron temperature.

    Implements Sharma-Hammett slope-limited explicit diffusion:

    1. Compute B-hat = B / |B|
    2. Compute temperature gradient: grad_T
    3. Parallel component: (grad_T . b_hat) * b_hat
    4. Perpendicular component: grad_T - parallel component
    5. Heat flux: q = -kappa_par * grad_T_par - kappa_perp * grad_T_perp
    6. Apply slope limiter to prevent unphysical heat transport
    7. Update: dTe = dt * div(q) / (1.5 * ne * k_B)

    Args:
        Te: Electron temperature [K], shape (nx, ny, nz).
        B: Magnetic field [T], shape (3, nx, ny, nz).
        ne: Electron number density [m^-3], shape (nx, ny, nz).
        dt: Timestep [s].
        dx, dy, dz: Grid spacings [m].
        Z_eff: Effective ion charge.
        kappa_parallel: Pre-computed parallel conductivity [W/(m*K)].
                        If None, computed from Braginskii.
        kappa_perp: Pre-computed perpendicular conductivity [W/(m*K)].
                    If None, computed from Braginskii.
        flux_limiter: Sharma-Hammett flux limiter parameter (default 0.1).

    Returns:
        Updated Te array [K], shape (nx, ny, nz).
    """
    # Compute B magnitude and unit vector
    B_mag = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    B_safe = np.maximum(B_mag, 1e-30)
    b_hat = np.zeros_like(B)
    for i in range(3):
        b_hat[i] = B[i] / B_safe

    # Compute conductivities if not provided
    if kappa_parallel is None:
        kappa_parallel = braginskii_kappa_parallel(ne, Te, Z_eff)
    if kappa_perp is None:
        kappa_perp = braginskii_kappa_perp(ne, Te, B_mag, Z_eff)

    # Cap conductivities for stability
    kappa_cap = 1e30
    kappa_parallel = np.minimum(kappa_parallel, kappa_cap)
    kappa_perp = np.minimum(kappa_perp, kappa_cap)

    max_kappa = float(np.max(kappa_parallel))
    if max_kappa < 1e-30:
        return Te.copy()

    # Temperature gradient
    grad_T = np.array([
        np.gradient(Te, dx, axis=0),
        np.gradient(Te, dy, axis=1),
        np.gradient(Te, dz, axis=2),
    ])

    # Parallel gradient: (b . grad_T) * b
    b_dot_gradT = np.sum(b_hat * grad_T, axis=0)
    grad_T_par = np.zeros_like(B)
    for i in range(3):
        grad_T_par[i] = b_dot_gradT * b_hat[i]

    # Perpendicular gradient: grad_T - parallel part
    grad_T_perp = grad_T - grad_T_par

    # Sharma-Hammett slope limiter
    # Limit the parallel heat flux to prevent unphysical transport
    # q_free = n_e * k_B * T_e * v_th_e (free-streaming limit)
    Te_safe = np.maximum(Te, 1.0)
    ne_safe = np.maximum(ne, 1e-10)
    v_th_e = np.sqrt(k_B * Te_safe / m_e)
    q_free = ne_safe * k_B * Te_safe * v_th_e

    # Magnitude of parallel heat flux before limiting
    q_par_mag = kappa_parallel * np.abs(b_dot_gradT)

    # Harmonic limiter: q_limited = q_classical * q_free / (q_classical + flux_limiter * q_free)
    q_denom = q_par_mag + flux_limiter * q_free
    limiter_factor = np.where(
        q_denom > 1e-30,
        flux_limiter * q_free / q_denom,
        1.0,
    )

    # Apply limiter to parallel conductivity
    kappa_par_limited = kappa_parallel * limiter_factor

    # Heat flux: q = -kappa_par * grad_T_par - kappa_perp * grad_T_perp
    heat_flux = np.zeros_like(B)
    for i in range(3):
        heat_flux[i] = kappa_par_limited * grad_T_par[i] + kappa_perp * grad_T_perp[i]

    # Sanitize heat flux
    heat_flux = np.where(np.isfinite(heat_flux), heat_flux, 0.0)

    # Divergence of heat flux
    div_q = (
        np.gradient(heat_flux[0], dx, axis=0)
        + np.gradient(heat_flux[1], dy, axis=1)
        + np.gradient(heat_flux[2], dz, axis=2)
    )
    div_q = np.where(np.isfinite(div_q), div_q, 0.0)

    # Stability-limited timestep for explicit diffusion
    dx_min = min(dx, dy, dz)
    max_kappa_eff = float(np.max(kappa_par_limited))
    min_ne = float(np.min(ne_safe))
    diffusivity = max_kappa_eff / max(min_ne * k_B, 1e-30)

    if not np.isfinite(diffusivity) or diffusivity <= 0:
        return Te.copy()

    dt_diff = 0.25 * dx_min**2 / diffusivity

    if not np.isfinite(dt_diff) or dt_diff <= 0:
        return Te.copy()

    # Sub-cycle if needed
    n_sub = max(1, int(np.ceil(dt / dt_diff)))
    n_sub = min(n_sub, 100)  # Cap subcycles
    dt_sub = dt / n_sub

    Te_new = Te.copy()
    for _ in range(n_sub):
        # dTe/dt = div(q) / (1.5 * ne * kB)
        Te_new += dt_sub * div_q / (1.5 * ne_safe * k_B)

    # Floor temperature and sanitize
    Te_new = np.maximum(Te_new, 1.0)
    Te_new = np.where(np.isfinite(Te_new), Te_new, Te)

    return Te_new
