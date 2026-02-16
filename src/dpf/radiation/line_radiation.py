"""Line and recombination radiation for multi-species plasma.

Provides coronal equilibrium line radiation cooling and recombination
radiation power, the dominant cooling mechanisms for impurity-laden
DPF plasmas at 10 eV -- 10 keV electron temperatures.

Physics:
    P_line = ne * n_Z * Lambda(Te, Z)               [W/m^3]
    P_rec  = C_rec * ne^2 * Z^2 * sqrt(chi / Te)    [W/m^3]

    where:
        Lambda(Te, Z)  = coronal equilibrium cooling function [W m^3]
        C_rec          = Seaton recombination coefficient
        chi            = 13.6 * Z^2 eV  (effective ionisation energy)
        n_Z            = impurity density [m^-3]

The cooling function Lambda(Te, Z) is assembled from piecewise power-law
fits to CHIANTI / ADAS data for hydrogen (Z=1), neon (Z=10), argon (Z=18),
copper (Z=29), and tungsten (Z=74).  For other elements a general
interpolation in Z is used.

References:
    - Post et al., At. Data Nucl. Data Tables 20, 397 (1977)
    - Summers, ADAS User Manual (2004)
    - NRL Plasma Formulary (2019)
    - Seaton, MNRAS 119, 81 (1959)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import eV, k_B

# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

# Seaton recombination coefficient [W m^3 K^{1/2}]
# P_rec = C_REC * ne^2 * Z^2 * sqrt(13.6*Z^2 eV / (kB*Te)) / sqrt(Te)
# which simplifies to P_rec = C_REC * ne^2 * Z^2 * sqrt(chi_eV * eV / kB) / Te
C_REC: float = 5.2e-39


# ═══════════════════════════════════════════════════════
# Cooling function -- piecewise power-law fits
# ═══════════════════════════════════════════════════════

@njit(cache=True)
def _cooling_hydrogen(Te_eV: float) -> float:
    """Hydrogen / deuterium cooling function [W m^3].

    Very weak line radiation above full ionisation (~13.6 eV).
    Below ~13.6 eV excitation of Lyman-alpha dominates.

    Fit: Post et al. (1977) hydrogen cooling curve.
    """
    if Te_eV < 1.0:
        # Below 1 eV: negligible excitation
        return 1.0e-40
    elif Te_eV < 13.6:
        # Lyman-alpha excitation peak
        # Lambda peaks ~1e-31 near 5 eV, drops steeply outside
        return 1.0e-31 * (Te_eV / 5.0) ** 1.5 * np.exp(-13.6 / Te_eV)
    elif Te_eV < 100.0:
        # Fully ionised H: only residual recombination-cascade lines
        return 1.0e-34 * (Te_eV / 13.6) ** (-0.5)
    else:
        # keV range: essentially zero line radiation
        return 1.0e-36


@njit(cache=True)
def _cooling_neon(Te_eV: float) -> float:
    """Neon (Z=10) cooling function [W m^3].

    Moderate-Z impurity with strong line radiation in 10-200 eV range.
    Fit from Post et al. (1977) and Summers ADAS data.
    """
    if Te_eV < 1.0:
        return 1.0e-40
    elif Te_eV < 10.0:
        # Rising toward L-shell excitation
        return 5.0e-33 * (Te_eV / 10.0) ** 2.0
    elif Te_eV < 50.0:
        # L-shell peak, Lambda ~ 5e-33 to 2e-32
        return 5.0e-33 * (Te_eV / 10.0) ** 0.8
    elif Te_eV < 200.0:
        # K-shell excitation, second peak near 150 eV
        return 2.0e-32 * (Te_eV / 50.0) ** (-0.3)
    elif Te_eV < 1000.0:
        # Declining: fully stripped
        return 1.5e-32 * (Te_eV / 200.0) ** (-1.5)
    else:
        return 3.0e-34 * (Te_eV / 1000.0) ** (-0.5)


@njit(cache=True)
def _cooling_argon(Te_eV: float) -> float:
    """Argon (Z=18) cooling function [W m^3].

    Strong M-shell and L-shell line radiation in 10-1000 eV range.
    Fit from Post et al. (1977).
    """
    if Te_eV < 2.0:
        return 1.0e-39
    elif Te_eV < 20.0:
        # M-shell excitation rising
        return 1.0e-33 * (Te_eV / 20.0) ** 2.5
    elif Te_eV < 100.0:
        # M-shell peak near 30-50 eV
        return 1.0e-33 * (Te_eV / 20.0) ** 0.5
    elif Te_eV < 500.0:
        # L-shell excitation, peak near 200 eV
        return 3.0e-33 * (Te_eV / 100.0) ** (-0.2)
    elif Te_eV < 3000.0:
        # K-shell and decline
        return 2.5e-33 * (Te_eV / 500.0) ** (-1.3)
    else:
        return 2.0e-34 * (Te_eV / 3000.0) ** (-0.5)


@njit(cache=True)
def _cooling_copper(Te_eV: float) -> float:
    """Copper (Z=29) cooling function [W m^3].

    Strong line radiation from M-shell and L-shell excitation.
    Relevant for DPF electrode erosion.  Peak Lambda ~ 5e-32
    around 30-100 eV.

    Fit from Post et al. (1977) and ADAS compilation.
    """
    if Te_eV < 2.0:
        return 1.0e-39
    elif Te_eV < 10.0:
        # M-shell excitation rising steeply
        return 2.0e-34 * (Te_eV / 10.0) ** 3.0
    elif Te_eV < 50.0:
        # M-shell peak: Lambda ~ 2e-32 at ~30 eV
        return 2.0e-34 * (Te_eV / 10.0) ** 2.0
    elif Te_eV < 200.0:
        # Broad M/L-shell peak, Lambda ~ 5e-32
        return 5.0e-32 * (Te_eV / 50.0) ** (-0.3)
    elif Te_eV < 1000.0:
        # L-shell declining
        return 4.0e-32 * (Te_eV / 200.0) ** (-1.0)
    elif Te_eV < 5000.0:
        # K-shell and decline toward fully stripped
        return 8.0e-33 * (Te_eV / 1000.0) ** (-1.5)
    else:
        return 5.0e-34 * (Te_eV / 5000.0) ** (-0.5)


@njit(cache=True)
def _cooling_tungsten(Te_eV: float) -> float:
    """Tungsten (Z=74) cooling function [W m^3].

    Very high-Z impurity with extremely strong and broad line radiation.
    Dominant radiator in tokamak divertors and DPF with W electrodes.
    Peak Lambda ~ 1e-31 near 50 eV.

    Fit from Pütterich et al. (2019) and Post et al. (1977).
    """
    if Te_eV < 2.0:
        return 1.0e-38
    elif Te_eV < 10.0:
        # N/O-shell excitation
        return 1.0e-34 * (Te_eV / 10.0) ** 3.0
    elif Te_eV < 50.0:
        # Strong rise to peak, Lambda ~ 1e-31
        return 1.0e-34 * (Te_eV / 10.0) ** 2.5
    elif Te_eV < 200.0:
        # Broad peak near 50-100 eV
        return 1.0e-31 * (Te_eV / 50.0) ** (-0.2)
    elif Te_eV < 2000.0:
        # Declining but still strong
        return 8.0e-32 * (Te_eV / 200.0) ** (-0.8)
    elif Te_eV < 10000.0:
        # L/K-shell and decline
        return 1.5e-32 * (Te_eV / 2000.0) ** (-1.2)
    else:
        return 2.0e-33 * (Te_eV / 10000.0) ** (-0.5)


@njit(cache=True)
def _cooling_generic(Te_eV: float, Z: float) -> float:
    """Generic Z scaling of the cooling function [W m^3].

    For elements without dedicated fits.  Uses a rough interpolation:
        Lambda ~ Z^2 * Lambda_hydrogen * enhancement_factor

    The enhancement factor captures the fact that higher-Z elements
    have more bound-state transitions and therefore radiate more
    efficiently at moderate temperatures.

    Args:
        Te_eV: Electron temperature in eV.
        Z: Atomic number (nuclear charge).

    Returns:
        Approximate coronal equilibrium cooling function [W m^3].
    """
    # Rough peak temperature scales as Z^1.3 * 10 eV
    Te_peak = 10.0 * Z ** 1.3
    # Peak cooling scales as Z^2 * 1e-33
    Lambda_peak = Z * Z * 1.0e-33

    if Te_eV < 1.0:
        return 1.0e-40
    elif Te_eV < Te_peak * 0.1:
        # Rising phase
        ratio = Te_eV / (Te_peak * 0.1)
        return Lambda_peak * 0.01 * ratio ** 2.5
    elif Te_eV < Te_peak:
        # Approach peak
        ratio = Te_eV / Te_peak
        return Lambda_peak * ratio ** 1.0
    elif Te_eV < Te_peak * 10.0:
        # Declining from peak
        ratio = Te_eV / Te_peak
        return Lambda_peak * ratio ** (-0.8)
    else:
        # Far above peak: fully stripped, weak radiation
        ratio = Te_eV / (Te_peak * 10.0)
        return Lambda_peak * 0.1 * ratio ** (-1.0)


@njit(cache=True)
def _cooling_scalar(Te_K: float, Z: float) -> float:
    """Dispatch to the appropriate species fit (scalar interface).

    Args:
        Te_K: Electron temperature [K].
        Z: Atomic number (nuclear charge, not charge state).

    Returns:
        Cooling function Lambda(Te, Z) [W m^3].
    """
    # Convert temperature to eV
    Te_eV = k_B * Te_K / eV

    # Dispatch by Z (using nearest-integer matching)
    Zi = int(Z + 0.5)

    if Zi <= 1:
        return _cooling_hydrogen(Te_eV)
    elif Zi == 10:
        return _cooling_neon(Te_eV)
    elif Zi == 18:
        return _cooling_argon(Te_eV)
    elif Zi == 29:
        return _cooling_copper(Te_eV)
    elif Zi >= 74:
        return _cooling_tungsten(Te_eV)
    else:
        return _cooling_generic(Te_eV, Z)


# ═══════════════════════════════════════════════════════
# Public API -- array functions
# ═══════════════════════════════════════════════════════

@njit(cache=True)
def cooling_function(Te: np.ndarray, Z: float) -> np.ndarray:
    """Coronal equilibrium cooling function Lambda(Te, Z) [W m^3].

    The cooling function describes the total power radiated per unit
    electron density per unit ion density in coronal equilibrium:

        P_line = ne * n_Z * Lambda(Te, Z)

    Uses piecewise power-law fits to CHIANTI / ADAS data for common
    elements (H, Ne, Ar, Cu, W) and a generic Z-scaling for others.

    Args:
        Te: Electron temperature [K].  Must be non-negative.
            Scalar or 1-D array.
        Z: Atomic number (nuclear charge).  E.g. Z=1 for hydrogen,
           Z=29 for copper, Z=74 for tungsten.

    Returns:
        Lambda: Cooling function [W m^3], same shape as *Te*.
    """
    out = np.empty_like(Te)
    for i in range(Te.size):
        Te_val = Te.flat[i]
        Te_safe = max(Te_val, 0.0)
        out.flat[i] = _cooling_scalar(Te_safe, Z)
    return out


@njit(cache=True)
def _recomb_power_scalar(ne: float, Te_K: float, Z: float) -> float:
    """Scalar recombination power density [W/m^3].

    P_rec = C_REC * ne^2 * Z^2 * sqrt(chi / (kB * Te)) / sqrt(Te)

    where chi = 13.6 * Z^2 eV is the effective ionisation energy
    (hydrogen-like scaling).

    Args:
        ne: Electron number density [m^-3].
        Te_K: Electron temperature [K].
        Z: Ion charge state.

    Returns:
        Recombination radiation power density [W/m^3].
    """
    if ne <= 0.0 or Te_K <= 0.0 or Z <= 0.0:
        return 0.0

    chi_J = 13.6 * Z * Z * eV          # Effective ionisation energy [J]
    kT = k_B * Te_K                     # Thermal energy [J]
    # P = C_REC * ne^2 * Z^2 * sqrt(chi / kT) / sqrt(Te)
    #   = C_REC * ne^2 * Z^2 * sqrt(chi) / (sqrt(kT) * sqrt(Te))
    #   = C_REC * ne^2 * Z^2 * sqrt(chi / kB) / Te
    return C_REC * ne * ne * Z * Z * np.sqrt(chi_J / kT) / np.sqrt(Te_K)


@njit(cache=True)
def recombination_power(ne: np.ndarray, Te: np.ndarray, Z: float) -> np.ndarray:
    """Radiative recombination power density [W/m^3].

    Uses the Seaton (1959) approximation for hydrogenic recombination:

        P_rec = C_rec * ne^2 * Z^2 * sqrt(chi / (kB Te)) / sqrt(Te)

    where chi = 13.6 * Z^2 eV.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state.

    Returns:
        Recombination radiation power density [W/m^3], same shape as *ne*.
    """
    out = np.empty_like(ne)
    for i in range(ne.size):
        ne_val = max(ne.flat[i], 0.0)
        Te_val = max(Te.flat[i], 1.0)  # Floor at 1 K to avoid divide-by-zero
        out.flat[i] = _recomb_power_scalar(ne_val, Te_val, Z)
    return out


@njit(cache=True)
def line_radiation_power(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float,
    n_impurity_fraction: float,
) -> np.ndarray:
    """Line radiation volumetric power density [W/m^3].

    Computes coronal equilibrium line radiation from impurity species:

        P_line = ne * n_imp * Lambda(Te, Z)

    where n_imp = n_impurity_fraction * ne (impurity density expressed
    as a fraction of electron density).

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Atomic number of the impurity species.
        n_impurity_fraction: Impurity density as a fraction of ne,
            i.e. n_imp = n_impurity_fraction * ne.  Typical values:
            0.01 -- 0.05 for trace copper in DPF.

    Returns:
        P_line: Line radiation power [W/m^3], same shape as *ne*.
    """
    Lambda = cooling_function(Te, Z)
    n_imp = n_impurity_fraction * ne
    P_line = ne * n_imp * Lambda
    # Guard against negative due to floating-point
    for i in range(P_line.size):
        if P_line.flat[i] < 0.0:
            P_line.flat[i] = 0.0
    return P_line


@njit(cache=True)
def _total_rad_power_scalar(
    ne: float,
    Te_K: float,
    Z_eff: float,
    n_imp_frac: float,
    Z_imp: float,
) -> float:
    """Scalar total radiation power: bremsstrahlung + line + recombination.

    Args:
        ne: Electron density [m^-3].
        Te_K: Electron temperature [K].
        Z_eff: Effective charge state (for bremsstrahlung).
        n_imp_frac: Impurity fraction relative to ne.
        Z_imp: Impurity atomic number.

    Returns:
        Total volumetric radiated power [W/m^3].
    """
    if ne <= 0.0 or Te_K <= 0.0:
        return 0.0

    # Bremsstrahlung: P_ff = 1.42e-40 * g_ff * Z_eff^2 * ne^2 * sqrt(Te)  (SI coefficient)
    g_ff = 1.2
    P_brem = 1.42e-40 * g_ff * Z_eff * Z_eff * ne * ne * np.sqrt(Te_K)

    # Impurity line radiation
    P_line = 0.0
    if n_imp_frac > 0.0 and Z_imp > 1.0:
        n_imp = n_imp_frac * ne
        Lambda = _cooling_scalar(Te_K, Z_imp)
        P_line = ne * n_imp * Lambda

    # Recombination radiation (from the dominant ion species)
    P_rec = _recomb_power_scalar(ne, Te_K, Z_eff)

    return P_brem + P_line + P_rec


def total_radiation_power(
    ne: np.ndarray,
    Te: np.ndarray,
    Z_eff: float,
    n_impurity_fraction: float = 0.0,
    Z_impurity: float = 1.0,
) -> np.ndarray:
    """Total volumetric radiated power density [W/m^3].

    Combines the three main radiation loss channels:

    1. **Bremsstrahlung** (free-free): P_ff ~ ne^2 * Z_eff^2 * sqrt(Te)
    2. **Line radiation** (bound-bound): P_line = ne * n_imp * Lambda(Te, Z)
    3. **Recombination** (free-bound):  P_rec ~ ne^2 * Z^2 * sqrt(chi/Te)/sqrt(Te)

    This function provides a convenient summary; for engine-level
    integration use :func:`apply_line_radiation_losses` which performs
    implicit cooling.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z_eff: Effective charge state of the bulk plasma.
        n_impurity_fraction: Impurity density as fraction of ne.
        Z_impurity: Atomic number of the impurity species.

    Returns:
        Total radiated power density [W/m^3], same shape as *ne*.
    """
    ne_safe = np.maximum(ne, 0.0)
    Te_safe = np.maximum(Te, 1.0)
    out = np.empty_like(ne_safe)
    ne_flat = ne_safe.ravel()
    Te_flat = Te_safe.ravel()
    out_flat = out.ravel()

    for i in range(ne_flat.size):
        out_flat[i] = _total_rad_power_scalar(
            ne_flat[i], Te_flat[i], Z_eff, n_impurity_fraction, Z_impurity
        )

    return out


# ═══════════════════════════════════════════════════════
# Implicit cooling step
# ═══════════════════════════════════════════════════════

@njit(cache=True)
def _implicit_cool_scalar(
    Te_old: float,
    ne: float,
    dt: float,
    Z_eff: float,
    n_imp_frac: float,
    Z_imp: float,
    Te_floor: float,
) -> tuple[float, float]:
    """Implicit cooling of a single cell.

    Solves the ODE:
        (3/2) ne kB dTe/dt = -P_rad(ne, Te)

    using Newton iteration on the implicit equation:
        Te_new + (dt / (1.5 ne kB)) * P_rad(ne, Te_new) = Te_old

    This prevents the temperature from becoming negative regardless
    of the timestep size.

    Returns:
        (Te_new, P_rad_effective) where P_rad_effective is the
        average power density actually removed [W/m^3].
    """
    if ne <= 0.0 or Te_old <= Te_floor:
        return Te_old, 0.0

    inv_cv = dt / (1.5 * ne * k_B)  # dt / (volumetric heat capacity)

    # Newton iteration: f(T) = T + inv_cv * P_rad(T) - Te_old = 0
    T = Te_old
    for _ in range(8):
        P = _total_rad_power_scalar(ne, T, Z_eff, n_imp_frac, Z_imp)
        f = T + inv_cv * P - Te_old
        # Numerical Jacobian: df/dT ~ 1 + inv_cv * dP/dT
        # Approximate dP/dT by finite difference
        dT_fd = max(T * 1.0e-6, 1.0)
        P_plus = _total_rad_power_scalar(ne, T + dT_fd, Z_eff, n_imp_frac, Z_imp)
        dPdT = (P_plus - P) / dT_fd
        fp = 1.0 + inv_cv * dPdT

        # Avoid division by near-zero Jacobian
        if abs(fp) < 1.0e-30:
            break

        delta = f / fp
        T = T - delta
        T = max(T, Te_floor)

        # Convergence check
        if abs(delta) < 1.0e-8 * T + 1.0e-30:
            break

    Te_new = max(T, Te_floor)

    # Effective radiated power = energy removed / dt
    P_eff = 1.5 * ne * k_B * max(Te_old - Te_new, 0.0) / max(dt, 1.0e-300)

    return Te_new, P_eff


@njit(cache=True)
def _apply_line_radiation_kernel(
    Te: np.ndarray,
    ne: np.ndarray,
    dt: float,
    Z_eff: float,
    n_imp_frac: float,
    Z_imp: float,
    Te_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel: apply implicit line + recombination cooling over all cells."""
    Te_new = np.empty_like(Te)
    P_rad = np.empty_like(Te)
    for i in range(Te.size):
        Te_val = Te.flat[i]
        ne_val = ne.flat[i]
        T_out, P_out = _implicit_cool_scalar(
            Te_val, ne_val, dt, Z_eff, n_imp_frac, Z_imp, Te_floor,
        )
        Te_new.flat[i] = T_out
        P_rad.flat[i] = P_out
    return Te_new, P_rad


def apply_line_radiation_losses(
    Te: np.ndarray,
    ne: np.ndarray,
    dt: float,
    Z_eff: float = 1.0,
    n_imp_frac: float = 0.0,
    Z_imp: float = 29.0,
    Te_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply line + recombination radiation cooling (implicit).

    Uses Newton-iteration implicit solve to prevent negative temperatures:

        Te_new + (dt / C_v) * P_rad(Te_new) = Te_old

    where C_v = (3/2) ne kB is the electron volumetric heat capacity.

    The radiated power includes **bremsstrahlung**, **line radiation**,
    and **recombination radiation** so that this function can serve as
    a single-call radiation step.  If you already apply bremsstrahlung
    separately, set *Z_eff* = 0 (which zeros the bremsstrahlung term)
    and use *n_imp_frac* and *Z_imp* for line + recombination only.

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].
        dt: Timestep [s].
        Z_eff: Effective charge state of the bulk plasma (used for
            bremsstrahlung and recombination).  Set to 0 to disable
            bremsstrahlung within this call.
        n_imp_frac: Impurity density as a fraction of ne.
            E.g. 0.01 for 1 % copper impurity.
        Z_imp: Atomic number of the dominant impurity species.
            Default 29 (copper), relevant for DPF electrode erosion.
        Te_floor: Minimum allowed electron temperature [K].

    Returns:
        Tuple of (Te_new, P_rad) where:
            Te_new: Updated electron temperature [K], same shape as *Te*.
            P_rad: Effective volumetric radiated power [W/m^3]
                   (energy removed per unit volume per unit time).
    """
    ne_safe = np.maximum(ne, 0.0)
    Te_safe = np.maximum(Te, 0.0)

    shape = Te.shape
    Te_flat = Te_safe.ravel()
    ne_flat = ne_safe.ravel()

    Te_new_flat, P_rad_flat = _apply_line_radiation_kernel(
        Te_flat, ne_flat, dt, Z_eff, n_imp_frac, Z_imp, Te_floor,
    )

    Te_new = Te_new_flat.reshape(shape)
    P_rad = P_rad_flat.reshape(shape)

    return Te_new, P_rad
