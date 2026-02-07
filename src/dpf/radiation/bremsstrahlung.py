"""Bremsstrahlung (free-free) radiation for two-temperature plasma.

Provides volumetric power density for electron-ion bremsstrahlung,
the dominant radiation loss mechanism in DPF plasma at ~1-10 keV.

Physics:
    P_ff = 1.69e-32 * g_ff * Z^2 * ne^2 * sqrt(Te)  [W/m^3]

    where:
        g_ff  = Gaunt factor (dimensionless, ~1.0-1.5 for DPF conditions)
        Z     = Ion charge state
        ne    = Electron number density [m^-3]
        Te    = Electron temperature [K]

Reference: NRL Plasma Formulary (2019), p. 58
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Bremsstrahlung coefficient in SI (W m^3 K^{-1/2})
# P_ff = BREM_COEFF * g_ff * Z^2 * ne^2 * sqrt(Te)
BREM_COEFF = 1.69e-32


@njit(cache=True)
def bremsstrahlung_power(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
) -> np.ndarray:
    """Compute bremsstrahlung volumetric power density [W/m^3].

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state (default 1 for hydrogen).
        gaunt_factor: Gaunt factor g_ff (default 1.2, typical for DPF).

    Returns:
        Volumetric power density P_ff [W/m^3], same shape as ne.
    """
    # Ensure Te is non-negative for sqrt
    Te_safe = np.maximum(Te, 0.0)
    ne_safe = np.maximum(ne, 0.0)

    P_ff = BREM_COEFF * gaunt_factor * Z * Z * ne_safe * ne_safe * np.sqrt(Te_safe)
    return P_ff


@njit(cache=True)
def bremsstrahlung_cooling_rate(
    ne: np.ndarray,
    Te: np.ndarray,
    rho: np.ndarray,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
) -> np.ndarray:
    """Compute bremsstrahlung specific cooling rate [K/s].

    This returns the rate of electron temperature decrease due to
    bremsstrahlung losses:  dTe/dt = -P_ff / (1.5 * ne * k_B)

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        rho: Mass density [kg/m^3] (unused but kept for interface consistency).
        Z: Ion charge state.
        gaunt_factor: Gaunt factor.

    Returns:
        Temperature cooling rate [K/s] (positive = cooling).
    """
    k_B = 1.380649e-23  # Boltzmann constant
    P_ff = bremsstrahlung_power(ne, Te, Z, gaunt_factor)
    # Cooling rate: dTe/dt = -P_ff / (1.5 * ne * k_B)
    # Return positive value (represents cooling magnitude)
    denom = 1.5 * np.maximum(ne, 1e-10) * k_B
    return P_ff / denom


def apply_bremsstrahlung_losses(
    Te: np.ndarray,
    ne: np.ndarray,
    dt: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    Te_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bremsstrahlung cooling to electron temperature (implicit).

    Uses backward Euler to avoid negative temperatures:
        Te_new = Te_old - dt * P_ff(ne, Te_new) / (1.5 * ne * k_B)

    For P_ff ~ sqrt(Te), the implicit solve is:
        Let alpha = dt * BREM_COEFF * g_ff * Z^2 * ne / (1.5 * k_B)
        Then: Te_new + alpha * sqrt(Te_new) = Te_old
        Solve via Newton iteration (2-3 iterations suffice).

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].
        dt: Timestep [s].
        Z: Ion charge state.
        gaunt_factor: Gaunt factor.
        Te_floor: Minimum temperature [K].

    Returns:
        Tuple of (Te_new, P_radiated) where P_radiated is the volumetric
        power density actually removed [W/m^3].
    """
    k_B = 1.380649e-23

    # Coefficient for implicit solve
    # alpha = dt * BREM_COEFF * g_ff * Z^2 * ne / (1.5 * k_B)
    ne_safe = np.maximum(ne, 0.0)
    alpha = dt * BREM_COEFF * gaunt_factor * Z * Z * ne_safe / (1.5 * k_B)

    # Newton iteration for: f(T) = T + alpha * sqrt(T) - Te_old = 0
    # f'(T) = 1 + alpha / (2 * sqrt(T))
    Te_new = Te.copy()
    for _ in range(4):  # 4 Newton iterations (converges in 2-3)
        sqrt_T = np.sqrt(np.maximum(Te_new, Te_floor))
        f = Te_new + alpha * sqrt_T - Te
        fp = 1.0 + alpha / (2.0 * sqrt_T)
        Te_new = Te_new - f / fp
        Te_new = np.maximum(Te_new, Te_floor)

    # Compute actual radiated power from temperature change
    # P_rad = 1.5 * ne * k_B * (Te_old - Te_new) / dt
    P_radiated = 1.5 * ne_safe * k_B * np.maximum(Te - Te_new, 0.0) / max(dt, 1e-30)

    return Te_new, P_radiated
