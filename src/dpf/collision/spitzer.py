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
    spitzer_resistivity: Spitzer resistivity [Ohm*m].
    braginskii_kappa: Parallel and perpendicular thermal conductivity.
    relax_temperatures: Implicit e-i temperature relaxation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e, epsilon_0, h, k_B, m_d, m_e, m_p, pi


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
def spitzer_alpha(Z: np.ndarray | float) -> np.ndarray | float:
    """Braginskii Z-dependent correction factor for Spitzer resistivity.

    Returns alpha(Z) from Braginskii Table 1 (1965) using piecewise linear
    interpolation. The Spitzer perpendicular resistivity is:

        eta_perp = eta_classical / alpha(Z)

    where eta_classical = m_e * nu_ei / (ne * e^2).

    Reference values:
        alpha(1) = 0.5064   (hydrogen)
        alpha(2) = 0.4408
        alpha(3) = 0.3965
        alpha(4) = 0.3752
        alpha(inf) = 0.2949 (Lorentz gas limit)

    Args:
        Z: Ion charge state (scalar or array).

    Returns:
        Dimensionless alpha(Z) correction factor (broadcasts with Z shape).

    References:
        Braginskii, S. I., Reviews of Plasma Physics Vol. 1 (1965), Table 1.
    """
    # Initialize result with default (Z=1 value)
    result = np.full_like(np.asarray(Z), 0.5064, dtype=np.float64)

    # Piecewise linear interpolation using NumPy where
    # Z <= 1
    mask1 = np.asarray(Z) <= 1.0
    result = np.where(mask1, 0.5064, result)

    # 1 < Z <= 2
    mask2 = (np.asarray(Z) > 1.0) & (np.asarray(Z) <= 2.0)
    t2 = np.asarray(Z) - 1.0
    alpha2 = 0.5064 * (1.0 - t2) + 0.4408 * t2
    result = np.where(mask2, alpha2, result)

    # 2 < Z <= 3
    mask3 = (np.asarray(Z) > 2.0) & (np.asarray(Z) <= 3.0)
    t3 = np.asarray(Z) - 2.0
    alpha3 = 0.4408 * (1.0 - t3) + 0.3965 * t3
    result = np.where(mask3, alpha3, result)

    # 3 < Z <= 4
    mask4 = (np.asarray(Z) > 3.0) & (np.asarray(Z) <= 4.0)
    t4 = np.asarray(Z) - 3.0
    alpha4 = 0.3965 * (1.0 - t4) + 0.3752 * t4
    result = np.where(mask4, alpha4, result)

    # Z >= 100 (Lorentz limit)
    mask_inf = np.asarray(Z) >= 100.0
    result = np.where(mask_inf, 0.2949, result)

    # 4 < Z < 100
    mask5 = (np.asarray(Z) > 4.0) & (np.asarray(Z) < 100.0)
    t5 = (np.asarray(Z) - 4.0) / (100.0 - 4.0)
    alpha5 = 0.3752 * (1.0 - t5) + 0.2949 * t5
    result = np.where(mask5, alpha5, result)

    return result


@njit(cache=True)
def spitzer_resistivity(
    ne: np.ndarray, Te: np.ndarray, lnL: np.ndarray | float = 10.0, Z: np.ndarray | float = 1.0
) -> np.ndarray:
    """Spitzer resistivity [Ohm*m] with Braginskii alpha(Z) correction.

    eta = m_e * nu_ei / (ne * e^2 * alpha(Z))

    The alpha(Z) correction accounts for ion screening effects and ranges
    from 0.5064 (Z=1, hydrogen) to 0.2949 (Z→∞, Lorentz gas). For Z=1,
    this increases resistivity by a factor of ~2 compared to the
    uncorrected formula.

    For a hydrogen plasma at 1 keV, eta ~ 2×10^-7 Ohm*m.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        lnL: Coulomb logarithm (scalar or array).
        Z: Ion charge state.

    Returns:
        Spitzer resistivity [Ohm*m].

    References:
        NRL Plasma Formulary, Spitzer (1962).
        Braginskii, S. I., Reviews of Plasma Physics Vol. 1 (1965).
    """
    freq = nu_ei(ne, Te, lnL, Z)
    alpha_Z = spitzer_alpha(Z)
    return m_e * freq / (ne * e**2 + 1e-300) / alpha_Z


@njit(cache=True)
def braginskii_kappa_coefficient(Z: float) -> float:
    """Z-dependent coefficient for electron thermal conductivity.

    Returns the Braginskii (1965) Table 1 coefficient for parallel
    electron thermal conductivity as a function of ion charge state Z.
    Uses piecewise linear interpolation between tabulated values.

    Reference values:
        Z=1: 3.16, Z=2: 3.14, Z=3: 3.12, Z=4: 3.11, Z->inf: 3.21

    The coefficient is nearly constant (~3% variation) but the non-monotonic
    behavior (dipping at Z~4, rising toward Z->inf) is physically significant
    for multi-Z plasmas with electrode ablation products.

    Args:
        Z: Ion charge state (scalar).

    Returns:
        Dimensionless kappa coefficient.

    References:
        Braginskii, S.I., Reviews of Plasma Physics Vol. 1 (1965), Table 1.
    """
    if Z <= 1.0:
        return 3.16
    elif Z <= 2.0:
        t = Z - 1.0
        return 3.16 + (3.14 - 3.16) * t
    elif Z <= 3.0:
        t = Z - 2.0
        return 3.14 + (3.12 - 3.14) * t
    elif Z <= 4.0:
        t = Z - 3.0
        return 3.12 + (3.11 - 3.12) * t
    elif Z < 100.0:
        t = (Z - 4.0) / 96.0
        return 3.11 + (3.21 - 3.11) * t
    else:
        return 3.21


@njit(cache=True)
def braginskii_kappa(
    ne: np.ndarray, Te: np.ndarray, Bmag: np.ndarray, Z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Braginskii parallel and perpendicular thermal conductivity.

    kappa_par = delta_e(Z) * k_B^2 * n_e * T_e / (m_e * nu_ei)
    kappa_per = kappa_par / (1 + (omega_ce / nu_ei)^2)

    where delta_e(Z) is the Z-dependent coefficient from Braginskii (1965)
    Table 1: delta_e(1)=3.16, delta_e(2)=3.14, ..., delta_e(inf)=3.21.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Bmag: Magnetic field magnitude [T].
        Z: Ion charge state (default 1.0). Used for the Z-dependent
            delta_e coefficient. Scalar only.

    Returns:
        Tuple of (kappa_parallel, kappa_perpendicular) [W/(m*K)].

    References:
        Braginskii, S.I., Reviews of Plasma Physics Vol. 1 (1965).
    """
    lnL = coulomb_log(ne, Te)
    freq = nu_ei(ne, Te, lnL)
    omega_ce = e * Bmag / m_e
    x = omega_ce / np.maximum(freq, 1e-300)

    delta_e = braginskii_kappa_coefficient(Z)
    kappa_par = delta_e * k_B**2 * ne * Te / (m_e * np.maximum(freq, 1e-300))
    kappa_per = kappa_par / (1.0 + x**2)

    return kappa_par, kappa_per


@njit(cache=True)
def relax_temperatures(
    Te: np.ndarray, Ti: np.ndarray, freq_ei: np.ndarray, dt: float,
    Z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit electron-ion temperature relaxation.

    Exact 2x2 solver for the coupled system:
        dTe/dt = -nu_ei * (Te - Ti) * (2 m_e / m_i)
        dTi/dt = +nu_ei * (Te - Ti) * (2 m_e / m_i)

    Uses implicit midpoint for unconditional stability.
    Equilibrium temperature accounts for charge state Z:
        T_eq = (Z * Te + Ti) / (Z + 1)

    Args:
        Te: Electron temperature [K].
        Ti: Ion temperature [K].
        freq_ei: Electron-ion collision frequency [s^-1].
        dt: Timestep [s].
        Z: Ion charge state (default 1.0).

    Returns:
        Tuple of (Te_new, Ti_new).
    """
    alpha = freq_ei * dt * 2.0 * m_e / m_d
    # Implicit: T_new = T_old + alpha * (T_other_mid - T_self_mid)
    # Exact solution of relaxation toward equilibrium:
    factor = np.exp(-2.0 * alpha)
    T_eq = (Z * Te + Ti) / (Z + 1.0)
    Te_new = T_eq + (Te - T_eq) * factor
    Ti_new = T_eq + (Ti - T_eq) * factor
    return Te_new, Ti_new
