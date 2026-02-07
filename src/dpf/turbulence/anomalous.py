"""Anomalous resistivity from current-driven micro-instabilities.

Implements the Buneman instability threshold model for anomalous
resistivity in dense plasma focus devices. When the electron drift
velocity exceeds the ion thermal speed, collective plasma turbulence
enhances the effective resistivity far above the Spitzer value.

Physical picture:
- Axial current in the DPF sheath drives electrons at drift velocity
  v_d = J / (n_e * e)
- When v_d > v_ti = sqrt(k_B * Ti / m_i), the Buneman instability
  develops
- This generates electrostatic turbulence that scatters electrons,
  producing anomalous resistivity eta_anom >> eta_Spitzer
- The anomalous resistivity is parameterized as:
  eta_anom = alpha * m_e * omega_pe / (n_e * e^2)
  where alpha ~ 0.01-0.1 is a turbulence parameter

References:
    Buneman, Phys. Rev. 115:503 (1959) -- instability threshold
    Sagdeev, Rev. Plasma Phys. 4:23 (1966) -- anomalous resistivity
    Haines, Plasma Phys. Control. Fusion 53:093001 (2011) -- DPF review
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e, epsilon_0, k_B, m_e, m_p


@njit(cache=True)
def electron_drift_velocity(
    J: np.ndarray,
    ne: np.ndarray,
) -> np.ndarray:
    """Compute electron drift velocity magnitude from current density.

    v_d = |J| / (n_e * e)

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].

    Returns:
        Electron drift velocity [m/s].
    """
    return np.abs(J) / np.maximum(ne * e, 1e-300)


@njit(cache=True)
def ion_thermal_speed(
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Compute ion thermal speed.

    v_ti = sqrt(k_B * Ti / m_i)

    Args:
        Ti: Ion temperature [K].
        mi: Ion mass [kg] (default: proton mass).

    Returns:
        Ion thermal speed [m/s].
    """
    return np.sqrt(k_B * np.maximum(Ti, 0.0) / mi)


@njit(cache=True)
def plasma_frequency(ne: np.ndarray) -> np.ndarray:
    """Compute electron plasma frequency.

    omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))

    Args:
        ne: Electron number density [m^-3].

    Returns:
        Electron plasma frequency [rad/s].
    """
    return np.sqrt(np.maximum(ne, 0.0) * e**2 / (epsilon_0 * m_e))


@njit(cache=True)
def buneman_threshold(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Check Buneman instability threshold at each point.

    The instability is triggered when v_drift > v_ti:
    - v_drift = |J| / (n_e * e)
    - v_ti = sqrt(k_B * Ti / m_i)

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].
        mi: Ion mass [kg].

    Returns:
        Boolean array: True where instability is active.
    """
    v_d = electron_drift_velocity(J, ne)
    v_ti = ion_thermal_speed(Ti, mi)
    return v_d > v_ti


@njit(cache=True)
def anomalous_resistivity(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    alpha: float = 0.05,
    mi: float = m_p,
) -> np.ndarray:
    """Compute anomalous resistivity from Buneman instability.

    When the electron drift velocity exceeds the ion thermal speed,
    the anomalous resistivity is:

        eta_anom = alpha * m_e * omega_pe / (n_e * e^2)

    where omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e)).

    Below threshold, eta_anom = 0.

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].
        alpha: Turbulence parameter (0.01-0.1, default 0.05).
        mi: Ion mass [kg].

    Returns:
        Anomalous resistivity [Ohm*m].
    """
    result = np.zeros_like(J)

    v_d = electron_drift_velocity(J, ne)
    v_ti = ion_thermal_speed(Ti, mi)
    omega_pe = plasma_frequency(ne)

    for idx in np.ndindex(J.shape):
        if v_d[idx] > v_ti[idx]:
            # Buneman anomalous resistivity
            ne_local = ne[idx]
            if ne_local > 0.0:
                result[idx] = alpha * m_e * omega_pe[idx] / np.maximum(ne_local * e**2, 1e-300)

    return result


@njit(cache=True)
def anomalous_resistivity_scalar(
    J_mag: float,
    ne_val: float,
    Ti_val: float,
    alpha: float = 0.05,
    mi: float = m_p,
) -> float:
    """Scalar version of anomalous resistivity for volume-averaged values.

    Args:
        J_mag: Current density magnitude [A/m^2].
        ne_val: Electron number density [m^-3].
        Ti_val: Ion temperature [K].
        alpha: Turbulence parameter (0.01-0.1, default 0.05).
        mi: Ion mass [kg].

    Returns:
        Anomalous resistivity [Ohm*m].
    """
    if ne_val <= 0.0:
        return 0.0

    v_d = abs(J_mag) / max(ne_val * e, 1e-300)
    v_ti = (k_B * max(Ti_val, 0.0) / mi) ** 0.5

    if v_d <= v_ti:
        return 0.0

    omega_pe = (ne_val * e**2 / (epsilon_0 * m_e)) ** 0.5
    return alpha * m_e * omega_pe / max(ne_val * e**2, 1e-300)


@njit(cache=True)
def total_resistivity(
    eta_spitzer: np.ndarray,
    eta_anomalous: np.ndarray,
) -> np.ndarray:
    """Combine Spitzer and anomalous resistivity.

    eta_total = eta_spitzer + eta_anomalous

    The anomalous contribution is additive (parallel scattering mechanisms).

    Args:
        eta_spitzer: Spitzer (collisional) resistivity [Ohm*m].
        eta_anomalous: Anomalous (turbulent) resistivity [Ohm*m].

    Returns:
        Total resistivity [Ohm*m].
    """
    return eta_spitzer + eta_anomalous


@njit(cache=True)
def total_resistivity_scalar(
    eta_spitzer: float,
    eta_anomalous: float,
) -> float:
    """Scalar version of total resistivity.

    Args:
        eta_spitzer: Spitzer resistivity [Ohm*m].
        eta_anomalous: Anomalous resistivity [Ohm*m].

    Returns:
        Total resistivity [Ohm*m].
    """
    return eta_spitzer + eta_anomalous
