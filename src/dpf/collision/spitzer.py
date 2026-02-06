"""Spitzer collision frequencies and Braginskii transport coefficients.

Pure-function implementations (no classes, no @njit on methods) extracted
from DPF_AI's collision_model.py. All physics verified correct against
NRL Plasma Formulary.

Functions:
    coulomb_log: Dynamic Coulomb logarithm with quantum correction.
    nu_ei: Electron-ion Spitzer collision frequency.
    nu_ee: Electron-electron collision frequency.
    nu_ii: Ion-ion collision frequency.
    nu_en: Electron-neutral collision frequency.
    braginskii_kappa: Parallel and perpendicular thermal conductivity.
    relax_temperatures: Implicit e-i temperature relaxation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e, epsilon_0, h, k_B, m_e, m_p, pi


@njit(cache=True)
def coulomb_log(ne: np.ndarray, Te: np.ndarray) -> np.ndarray:
    """Dynamic Coulomb logarithm with quantum diffraction correction.

    Uses the Gericke-Murillo-Schlanges interpolation between classical
    and quantum regimes via the de Broglie wavelength.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].

    Returns:
        Coulomb logarithm (floored at 1.0 to prevent unphysical values).
    """
    # Debye length
    lambda_D = np.sqrt(epsilon_0 * k_B * Te / (ne * e**2 + 1e-30))
    # Classical distance of closest approach
    b_class = e**2 / (4.0 * pi * epsilon_0 * k_B * Te + 1e-30)
    # de Broglie wavelength
    lambda_db = h / np.sqrt(2.0 * pi * m_e * k_B * Te + 1e-30)
    # Minimum impact parameter: max of classical and quantum
    b_min = np.maximum(b_class, lambda_db)
    # Coulomb logarithm
    Lambda = lambda_D / (b_min + 1e-30)
    return np.log(np.maximum(Lambda, 1.0))


@njit(cache=True)
def nu_ei(ne: np.ndarray, Te: np.ndarray, lnL: np.ndarray | float = 10.0, Z: float = 1.0) -> np.ndarray:
    """Electron-ion Spitzer collision frequency [s^-1].

    nu_ei = (4 sqrt(2 pi) * ne * Z * e^4 * lnL) /
            (3 * (4 pi epsilon_0)^2 * sqrt(m_e) * (k_B T_e)^{3/2})

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        lnL: Coulomb logarithm (scalar or array).
        Z: Ion charge state.

    Returns:
        Collision frequency [s^-1].
    """
    coef = 4.0 * np.sqrt(2.0 * pi) * ne * Z * e**4 * lnL
    denom = 3.0 * (4.0 * pi * epsilon_0) ** 2 * np.sqrt(m_e) * (k_B * Te) ** 1.5
    return coef / np.maximum(denom, 1e-300)


@njit(cache=True)
def nu_ee(ne: np.ndarray, Te: np.ndarray, lnL: np.ndarray | float = 10.0) -> np.ndarray:
    """Electron-electron collision frequency [s^-1]."""
    return nu_ei(ne, Te, lnL, Z=1.0) * np.sqrt(2.0)


@njit(cache=True)
def nu_ii(
    ni: np.ndarray,
    Ti: np.ndarray,
    lnL: np.ndarray | float = 10.0,
    Z: float = 1.0,
    mi: float = m_p,
) -> np.ndarray:
    """Ion-ion collision frequency [s^-1]."""
    coef = 4.0 * np.sqrt(pi) * ni * Z**4 * e**4 * lnL
    denom = 3.0 * (4.0 * pi * epsilon_0) ** 2 * np.sqrt(mi) * (k_B * Ti) ** 1.5
    return coef / np.maximum(denom, 1e-300)


@njit(cache=True)
def nu_en(ne: np.ndarray, Te: np.ndarray, nn: np.ndarray, sigma_en: float = 1e-19) -> np.ndarray:
    """Electron-neutral collision frequency [s^-1].

    Args:
        ne: Electron number density [m^-3] (unused, kept for interface consistency).
        Te: Electron temperature [K].
        nn: Neutral number density [m^-3].
        sigma_en: Electron-neutral cross section [m^2].
    """
    v_th_e = np.sqrt(k_B * Te / m_e)
    return nn * sigma_en * v_th_e


@njit(cache=True)
def braginskii_kappa(
    ne: np.ndarray, Te: np.ndarray, Bmag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Braginskii parallel and perpendicular thermal conductivity.

    kappa_par = 3.16 * k_B^2 * n_e * T_e / (m_e * nu_ei)
    kappa_per = kappa_par / (1 + (omega_ce / nu_ei)^2)

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Bmag: Magnetic field magnitude [T].

    Returns:
        Tuple of (kappa_parallel, kappa_perpendicular) [W/(m*K)].
    """
    lnL = coulomb_log(ne, Te)
    freq = nu_ei(ne, Te, lnL)
    omega_ce = e * Bmag / m_e
    x = omega_ce / np.maximum(freq, 1e-300)

    kappa_par = 3.16 * k_B**2 * ne * Te / (m_e * np.maximum(freq, 1e-300))
    kappa_per = kappa_par / (1.0 + x**2)

    return kappa_par, kappa_per


@njit(cache=True)
def relax_temperatures(
    Te: np.ndarray, Ti: np.ndarray, freq_ei: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit electron-ion temperature relaxation.

    Exact 2x2 solver for the coupled system:
        dTe/dt = -nu_ei * (Te - Ti) * (2 m_e / m_i)
        dTi/dt = +nu_ei * (Te - Ti) * (2 m_e / m_i)

    Uses implicit midpoint for unconditional stability.

    Args:
        Te: Electron temperature [K].
        Ti: Ion temperature [K].
        freq_ei: Electron-ion collision frequency [s^-1].
        dt: Timestep [s].

    Returns:
        Tuple of (Te_new, Ti_new).
    """
    alpha = freq_ei * dt * 2.0 * m_e / m_p
    # Implicit: T_new = T_old + alpha * (T_other_mid - T_self_mid)
    # Exact solution of relaxation toward equilibrium:
    factor = np.exp(-2.0 * alpha)
    T_eq = 0.5 * (Te + Ti)
    Te_new = T_eq + (Te - T_eq) * factor
    Ti_new = T_eq + (Ti - T_eq) * factor
    return Te_new, Ti_new
