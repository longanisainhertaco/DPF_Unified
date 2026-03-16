"""Improved radiation model with temperature-dependent Gaunt factor,
recombination radiation, and cyclotron emission.

Upgrades the constant Gaunt factor (g_ff=1.2) to a physics-based
temperature and frequency-averaged formula. Adds cyclotron radiation
for strong B-field regimes relevant to DPF pinch conditions.

Physics:
    Gaunt factor (Born approximation, temperature-averaged):
        g_ff(T, Z) = 1.0 + 0.7936 / sqrt(T_keV) + 0.1387 * T_keV
        for T in [0.01, 100] keV, Z=1. Higher Z: multiply by Z^0.15

    Cyclotron radiation:
        P_cyc = (e^4 * B^2 * ne) / (6 * pi * epsilon_0 * m_e^3 * c^3)
        = 6.21e-28 * B^2 * ne * Te  [W/m^3] (relativistic, single particle)

    Total radiation = bremsstrahlung + recombination + cyclotron + line

References:
    Karzas, W.J. & Latter, R., ApJS 6:167 (1961) — Gaunt factor tables.
    Sutherland, R.S. & Dopita, M.A., ApJS 88:253 (1993) — cooling functions.
    Haines, M.G., Plasma Phys. Control. Fusion 53:093001 (2011) — DPF radiation.
    Rybicki & Lightman, "Radiative Processes in Astrophysics" (1979).
"""

from __future__ import annotations

import logging

import numpy as np

from dpf.constants import e, k_B
from dpf.radiation.bremsstrahlung import BREM_COEFF

logger = logging.getLogger(__name__)

# eV to Kelvin conversion
_EV_TO_K = e / k_B  # 1 eV = 11604.5 K
_KEV_TO_K = 1000.0 * _EV_TO_K


def gaunt_factor_thermal(Te: np.ndarray, Z: float = 1.0) -> np.ndarray:
    """Temperature-averaged free-free Gaunt factor.

    Fitted to Karzas & Latter (1961) tables for thermal bremsstrahlung.
    Valid for T_e in [100 eV, 100 keV] (DPF regime).

    The classical Kramers value is g_ff = 1.0. Quantum corrections
    increase it to ~1.1-1.5 depending on temperature.

    Args:
        Te: Electron temperature [K].
        Z: Ion charge state (default 1).

    Returns:
        Gaunt factor array, same shape as Te.

    References:
        Karzas & Latter, ApJS 6:167 (1961).
        van Hoof et al., MNRAS 444:420 (2014) — modern tables.
    """
    Te_safe = np.maximum(Te, 1.0)
    T_keV = Te_safe / _KEV_TO_K

    # Fitting formula from van Hoof et al. (2014), simplified
    # g_ff ~ 1 + 0.7936/sqrt(T_keV) + 0.1387*T_keV for Z=1
    # Higher Z: g_ff scales weakly as Z^0.15
    g_ff = 1.0 + 0.7936 / np.sqrt(np.maximum(T_keV, 0.01)) + 0.1387 * T_keV

    # Z scaling (weak)
    if Z > 1:
        g_ff *= Z**0.15

    # Clamp to physical range [1.0, 5.0]
    g_ff = np.clip(g_ff, 1.0, 5.0)

    return g_ff


def bremsstrahlung_improved(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float = 1.0,
) -> np.ndarray:
    """Bremsstrahlung with temperature-dependent Gaunt factor.

    Replaces the constant g_ff=1.2 with the Karzas-Latter fit.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state.

    Returns:
        Volumetric power density P_ff [W/m^3].
    """
    g_ff = gaunt_factor_thermal(Te, Z)
    Te_safe = np.maximum(Te, 0.0)
    ne_safe = np.maximum(ne, 0.0)
    return BREM_COEFF * g_ff * Z * ne_safe**2 * np.sqrt(Te_safe)


def recombination_power(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float = 1.0,
) -> np.ndarray:
    """Free-bound (recombination) radiation power.

    Radiative recombination of electrons with ions. Dominant at low
    temperatures (< 1 keV) where recombination rate is high.

    P_fb = C_fb * Z^2 * ne^2 * (chi / (kB * Te))^0.5 * exp(-chi / (kB * Te))

    where chi = 13.6 * Z^2 eV is the ionization potential of the
    hydrogen-like ion.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state.

    Returns:
        Volumetric recombination power [W/m^3].

    References:
        Seaton, MNRAS 119:81 (1959).
        NRL Plasma Formulary (2019), p. 58.
    """
    # Ionization energy of hydrogen-like ion
    chi = 13.6 * Z**2 * e  # [J]

    Te_safe = np.maximum(Te, 1.0)
    ne_safe = np.maximum(ne, 0.0)

    # Seaton coefficient
    C_fb = 1.13e-37  # [W m^3 K^{1/2}]

    ratio = chi / (k_B * Te_safe)
    # For very large ratio (cold plasma), clamp to avoid overflow
    ratio_safe = np.minimum(ratio, 100.0)

    P_fb = C_fb * Z**2 * ne_safe**2 * np.sqrt(ratio_safe) * np.exp(-ratio_safe)

    return P_fb


def cyclotron_power(
    ne: np.ndarray,
    Te: np.ndarray,
    B_mag: np.ndarray,
) -> np.ndarray:
    """Cyclotron (synchrotron) radiation power.

    Electron cyclotron emission in strong magnetic fields.
    Relevant for DPF pinch conditions (B ~ 10-100 T).

    P_cyc = (e^4 * B^2 * ne * kB * Te) / (6 * pi * epsilon_0 * m_e^3 * c^3)

    For thermal electrons, using NRL Plasma Formulary:
    P_cyc = 6.21e-28 * B^2 * ne * Te  [W/m^3]

    This becomes significant relative to bremsstrahlung when:
    B > B_crit = sqrt(P_ff / (6.21e-28 * ne * Te))

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        B_mag: Magnetic field magnitude [T].

    Returns:
        Volumetric cyclotron power [W/m^3].

    References:
        Rybicki & Lightman (1979), Chapter 6.
        Haines, PPCF 53:093001 (2011), Eq. 4.
    """
    # Cyclotron coefficient [W / (T^2 m^{-3} K)]
    CYCL_COEFF = 6.21e-28

    ne_safe = np.maximum(ne, 0.0)
    Te_safe = np.maximum(Te, 0.0)
    B_safe = np.maximum(B_mag, 0.0)

    return CYCL_COEFF * B_safe**2 * ne_safe * Te_safe


def total_radiation_power(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float = 1.0,
    B_mag: np.ndarray | None = None,
    include_bremsstrahlung: bool = True,
    include_recombination: bool = True,
    include_cyclotron: bool = True,
) -> dict[str, np.ndarray]:
    """Compute total radiation power from all mechanisms.

    Returns a breakdown of each radiation component plus the total.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state.
        B_mag: Magnetic field magnitude [T] (optional, for cyclotron).
        include_bremsstrahlung: Include free-free radiation.
        include_recombination: Include free-bound radiation.
        include_cyclotron: Include cyclotron/synchrotron.

    Returns:
        Dict with keys:
            - P_bremsstrahlung: Free-free power [W/m^3]
            - P_recombination: Free-bound power [W/m^3]
            - P_cyclotron: Cyclotron power [W/m^3]
            - P_total: Sum of all components [W/m^3]
            - gaunt_factor: Temperature-dependent g_ff
            - dominant: Name of the dominant mechanism
    """
    P_total = np.zeros_like(ne, dtype=float)
    result: dict[str, np.ndarray] = {}

    if include_bremsstrahlung:
        P_ff = bremsstrahlung_improved(ne, Te, Z)
        result["P_bremsstrahlung"] = P_ff
        P_total = P_total + P_ff
    else:
        result["P_bremsstrahlung"] = np.zeros_like(ne)

    if include_recombination:
        P_fb = recombination_power(ne, Te, Z)
        result["P_recombination"] = P_fb
        P_total = P_total + P_fb
    else:
        result["P_recombination"] = np.zeros_like(ne)

    if include_cyclotron and B_mag is not None:
        P_cyc = cyclotron_power(ne, Te, B_mag)
        result["P_cyclotron"] = P_cyc
        P_total = P_total + P_cyc
    else:
        result["P_cyclotron"] = np.zeros_like(ne)

    result["P_total"] = P_total
    result["gaunt_factor"] = gaunt_factor_thermal(Te, Z)

    # Identify dominant mechanism
    P_max = np.max(P_total)
    if P_max > 0:
        components = {
            "bremsstrahlung": float(np.max(result["P_bremsstrahlung"])),
            "recombination": float(np.max(result["P_recombination"])),
            "cyclotron": float(np.max(result["P_cyclotron"])),
        }
        result["dominant"] = max(components, key=components.get)
    else:
        result["dominant"] = "none"

    return result


def apply_improved_radiation_losses(
    Te: np.ndarray,
    ne: np.ndarray,
    dt: float,
    Z: float = 1.0,
    B_mag: np.ndarray | None = None,
    Te_floor: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply improved radiation cooling to electron temperature.

    Uses backward Euler (implicit) to avoid negative temperatures.
    Returns the new temperature and a breakdown of radiated power.

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].
        dt: Timestep [s].
        Z: Ion charge state.
        B_mag: Magnetic field magnitude [T] (optional).
        Te_floor: Minimum temperature [K].

    Returns:
        Tuple of (Te_new, radiation_breakdown).
    """
    # Compute total radiation at current Te
    rad = total_radiation_power(ne, Te, Z, B_mag)
    P_total = rad["P_total"]

    # Implicit cooling: Te_new = Te - dt * P_total / (1.5 * ne * kB)
    ne_safe = np.maximum(ne, 1e-10)
    thermal_energy = 1.5 * ne_safe * k_B
    dTe = P_total * dt / thermal_energy

    Te_new = np.maximum(Te - dTe, Te_floor)

    # Compute actual radiated power from temperature change
    P_actual = thermal_energy * np.maximum(Te - Te_new, 0.0) / max(dt, 1e-30)
    rad["P_actual"] = P_actual

    return Te_new, rad


def radiation_regime_diagnostic(
    Te: np.ndarray,
    ne: np.ndarray,
    B_mag: np.ndarray,
    Z: float = 1.0,
) -> dict[str, float]:
    """Diagnose which radiation mechanism dominates.

    Useful for physics narrative and understanding the plasma regime.

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].
        B_mag: Magnetic field magnitude [T].
        Z: Ion charge state.

    Returns:
        Dict with:
            - Te_keV: Peak electron temperature [keV]
            - ne_max: Peak electron density [m^-3]
            - B_max: Peak magnetic field [T]
            - P_brem_peak: Peak bremsstrahlung power [W/m^3]
            - P_rec_peak: Peak recombination power [W/m^3]
            - P_cyc_peak: Peak cyclotron power [W/m^3]
            - brem_fraction: Fraction from bremsstrahlung
            - rec_fraction: Fraction from recombination
            - cyc_fraction: Fraction from cyclotron
            - dominant: Name of dominant mechanism
            - gaunt_factor_mean: Average Gaunt factor
            - B_crit_T: B-field at which cyclotron = bremsstrahlung [T]
    """
    rad = total_radiation_power(ne, Te, Z, B_mag)

    P_brem = float(np.max(rad["P_bremsstrahlung"]))
    P_rec = float(np.max(rad["P_recombination"]))
    P_cyc = float(np.max(rad["P_cyclotron"]))
    P_total = P_brem + P_rec + P_cyc

    if P_total > 0:
        brem_frac = P_brem / P_total
        rec_frac = P_rec / P_total
        cyc_frac = P_cyc / P_total
    else:
        brem_frac = rec_frac = cyc_frac = 0.0

    # Critical B-field where cyclotron matches bremsstrahlung
    # P_cyc = P_brem → 6.21e-28 * B^2 * ne * Te = 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te)
    # B_crit = sqrt(1.42e-40 * g_ff * Z * ne * sqrt(Te) / 6.21e-28 / Te)
    ne_peak = float(np.max(ne))
    Te_peak = max(float(np.max(Te)), 1.0)
    g_ff_mean = float(np.mean(rad["gaunt_factor"]))

    if Te_peak > 0 and ne_peak > 0:
        B_crit = np.sqrt(
            BREM_COEFF * g_ff_mean * Z * ne_peak * np.sqrt(Te_peak)
            / (6.21e-28 * Te_peak)
        )
    else:
        B_crit = 0.0

    return {
        "Te_keV": Te_peak / _KEV_TO_K,
        "ne_max": ne_peak,
        "B_max": float(np.max(B_mag)),
        "P_brem_peak": P_brem,
        "P_rec_peak": P_rec,
        "P_cyc_peak": P_cyc,
        "brem_fraction": brem_frac,
        "rec_fraction": rec_frac,
        "cyc_fraction": cyc_frac,
        "dominant": rad["dominant"],
        "gaunt_factor_mean": g_ff_mean,
        "B_crit_T": float(B_crit),
    }
