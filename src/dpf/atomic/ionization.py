"""Coronal / Saha equilibrium ionization for hydrogen and deuterium.

For a hydrogen-like species (H, D, T), the average ionization state Z_bar
is either 0 (neutral) or 1 (fully ionized). The Saha equation gives the
ionization fraction as a function of electron temperature and density.

The Saha equation for single ionization:
    Z / (1 - Z) = (1/ne) * (2*pi*me*kB*T/h^2)^{3/2} * (g1/g0) * exp(-E_ion/(kB*T))

where:
    g1/g0 = 2 / 1 = 2  (statistical weights: ion ground state / neutral ground state)
    E_ion = 13.6 eV     (hydrogen ionization energy)

Solving for Z_bar:
    S = (1/ne) * (2*pi*me*kB*T/h^2)^{3/2} * 2 * exp(-E_ion/(kB*T))
    Z_bar = S / (1 + S)

At low Te: Z_bar -> 0 (neutral gas)
At high Te: Z_bar -> 1 (fully ionized plasma)
The transition is sharp around Te ~ 1-2 eV depending on density.

Reference:
    NRL Plasma Formulary (2019)
    Griem, "Principles of Plasma Spectroscopy" (1997)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import eV, h, k_B, m_e, pi

# Hydrogen ionization energy
E_ION_H = 13.6 * eV  # 13.6 eV in Joules


@njit(cache=True)
def _saha_parameter(Te: float, ne: float) -> float:
    """Compute the Saha parameter S for hydrogen.

    S = (1/ne) * (2*pi*me*kB*T/h^2)^{3/2} * 2 * exp(-E_ion/(kB*T))

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].

    Returns:
        Saha parameter S (dimensionless).
    """
    if Te < 1.0 or ne < 1.0:
        return 0.0

    kT = k_B * Te
    # Thermal de Broglie: (2*pi*me*kB*T/h^2)^{3/2}
    thermal_factor = (2.0 * pi * m_e * kT / (h * h)) ** 1.5

    # Exponential: exp(-E_ion / kT)
    # Guard against overflow for very low temperatures
    exponent = -E_ION_H / kT
    if exponent < -500.0:
        return 0.0

    exp_factor = np.exp(exponent)

    # Statistical weight ratio g_ion/g_neutral = 2/1 for hydrogen
    g_ratio = 2.0

    # S = (thermal_factor * g_ratio * exp_factor) / ne
    S = thermal_factor * g_ratio * exp_factor / ne

    return S


@njit(cache=True)
def saha_ionization_fraction(Te: float, ne: float) -> float:
    """Compute the average ionization state Z_bar for hydrogen.

    Uses the Saha equation:
        Z_bar = S / (1 + S)

    where S is the Saha parameter.

    For cold gas (Te << 1 eV): Z_bar ~ 0 (neutral)
    For hot plasma (Te >> 1 eV): Z_bar ~ 1 (fully ionized)

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].

    Returns:
        Average ionization state Z_bar in [0, 1].
    """
    S = _saha_parameter(Te, ne)
    Z_bar = S / (1.0 + S)
    # Clamp to [0, 1] for safety
    return min(max(Z_bar, 0.0), 1.0)


@njit(cache=True)
def saha_ionization_fraction_array(
    Te: np.ndarray, ne: np.ndarray
) -> np.ndarray:
    """Vectorized Saha ionization fraction for arrays.

    Computes Z_bar element-wise for temperature and density arrays.

    Args:
        Te: Electron temperature array [K].
        ne: Electron number density array [m^-3].

    Returns:
        Z_bar array with same shape as input.
    """
    Z_bar = np.empty_like(Te)
    for i in range(Te.size):
        Z_bar.flat[i] = saha_ionization_fraction(Te.flat[i], ne.flat[i])
    return Z_bar
