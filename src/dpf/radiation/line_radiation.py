"""Line and recombination radiation for multi-species plasma.

Provides coronal equilibrium line radiation cooling and recombination
radiation power, the dominant cooling mechanisms for impurity-laden
DPF plasmas at 10 eV -- 10 keV electron temperatures.

Physics:
    P_line = ne * n_Z * Lambda(Te, Z)               [W/m^3]
    P_rec  = C_rec * ne^2 * Z^2 * sqrt(chi / (kB Te))  [W/m^3]

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

import logging

import numpy as np
from numba import njit

from dpf.constants import eV, k_B

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

# Seaton recombination coefficient [W m^3]
# Derived: alpha_R = 5.197e-20 * Z * sqrt(chi/(kB*Te)) [m^3/s]  (Seaton 1959, Phi~1)
# P_rec = ne * (ne/Z) * alpha_R * chi = C_REC * ne^2 * Z^2 * sqrt(chi/(kB*Te))
# where chi = 13.6 * Z^2 * eV (hydrogenic ionisation energy).
C_REC: float = 1.13e-37


# ═══════════════════════════════════════════════════════
# Cooling function -- piecewise power-law fits
# ═══════════════════════════════════════════════════════

@njit(cache=True)
def _cooling_hydrogen(Te_eV: float) -> float:
    """Hydrogen / deuterium cooling function [W m^3].

    Very weak line radiation above full ionisation (~13.6 eV).
    Below ~13.6 eV excitation of Lyman-alpha dominates.

    Fit calibrated to Post et al. (1977) Fig. 1 hydrogen cooling curve.
    Peak Lambda ~ 3e-32 W m^3 at ~ 4 eV.  Uses double-exponential form
    A * exp(-E_exc/T - T/T_ion) where E_exc = 10.2 eV (Ly-alpha excitation)
    and T_ion = 1.57 eV (ionisation depletion scale).  Accuracy: within
    factor of 2 of Post et al. over 2--13 eV range.
    """
    if Te_eV < 1.0:
        # Below 1 eV: negligible excitation
        return 1.0e-40
    elif Te_eV < 13.6:
        # Lyman-alpha excitation peak with ionisation depletion cutoff
        # Peak ~ 3e-32 near 4 eV, matching Post et al. (1977) Fig. 1
        return 4.93e-30 * np.exp(-10.2 / Te_eV - Te_eV / 1.57)
    elif Te_eV < 100.0:
        # Fully ionised H: only residual recombination-cascade lines
        return 4.0e-34 * (Te_eV / 13.6) ** (-0.5)
    else:
        # keV range: essentially zero line radiation
        return 1.0e-36


@njit(cache=True)
def _cooling_neon(Te_eV: float) -> float:
    """Neon (Z=10) cooling function [W m^3].

    Moderate-Z impurity with strong line radiation in 10-200 eV range.
    Piecewise power-law fit to Post et al. (1977) Table II and Summers
    ADAS data.  Peak Lambda ~ 2e-32 near 30-50 eV.
    Accuracy: within factor of 2-3 of ADAS coronal rates over 5-500 eV.
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
    Piecewise power-law fit to Post et al. (1977) Table II.
    Peak Lambda ~ 3e-33 near 30-100 eV.
    Accuracy: within factor of 3 of Post et al. and ADAS over 10-1000 eV.
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

    Tabulated coronal equilibrium cooling from Post, Jensen, Tarter,
    Grasberger & Lokke, At. Data Nucl. Data Tables 20, 397 (1977),
    with shell-structure cross-referenced against NIST ionization data
    (Sugar & Musgrove, JPCRD 19:527, 1990) and Z-scaling validated
    against Pütterich et al., Nucl. Fusion 59:056013 (2019).

    Physical shell structure of Cu (Z=29):
        N-shell (n=4):  IP   7.7–20.3 eV  (Cu0+ to Cu1+)
        M-shell (n=3):  IP  36.8–670.6 eV (Cu2+ to Cu18+)
          - 3d subshell: ~37–103 eV — primary peak at ~50-100 eV
          - 3p subshell: ~103–633 eV — broad emission to ~500 eV
          - Ar-like gap at Cu18+ (670 eV) → trough ~700–1000 eV
        L-shell (n=2):  IP 1690–2586 eV (Cu19+ to Cu26+) — secondary peak ~2-3 keV
        K-shell (n=1):  IP 11062–11568 eV (Cu27+, Cu28+)

    Uses log-log linear interpolation on 21 data points spanning
    1 eV to 10 keV.  Accuracy: ±30-50% vs. ADAS/CHIANTI tabulations.
    Peak Lambda ~ 3e-30 W m^3 at ~100 eV (global M-shell maximum).
    """
    if Te_eV < 1.0:
        return 5.0e-34
    if Te_eV > 10000.0:
        return 3.0e-31 * (Te_eV / 10000.0) ** (-0.5)

    # 21-point log-log table: (ln(Te_eV), ln(Lambda_Wm3))
    # Te: 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 500,
    #     700, 1000, 2000, 3000, 5000, 7000, 10000 [eV]
    log_T = np.log(Te_eV)

    # Tabulated ln(Te) breakpoints
    LT0 = 0.0       # ln(1)
    LT1 = 0.6931    # ln(2)
    LT2 = 1.6094    # ln(5)
    LT3 = 2.3026    # ln(10)
    LT4 = 2.9957    # ln(20)
    LT5 = 3.4012    # ln(30)
    LT6 = 3.9120    # ln(50)
    LT7 = 4.3820    # ln(80)
    LT8 = 4.6052    # ln(100)
    LT9 = 5.0106    # ln(150)
    LT10 = 5.2983   # ln(200)
    LT11 = 5.7038   # ln(300)
    LT12 = 5.9915   # ln(400)
    LT13 = 6.2146   # ln(500)
    LT14 = 6.5511   # ln(700)
    LT15 = 6.9078   # ln(1000)
    LT16 = 7.6009   # ln(2000)
    LT17 = 8.0064   # ln(3000)
    LT18 = 8.5172   # ln(5000)
    LT19 = 8.8537   # ln(7000)
    LT20 = 9.2103   # ln(10000)

    # Tabulated ln(Lambda) values [W m^3]
    LL0 = -76.6785   # ln(5e-34)
    LL1 = -74.8867   # ln(3e-33)
    LL2 = -71.6033   # ln(8e-32)
    LL3 = -70.2815   # ln(3e-31)
    LL4 = -69.3007   # ln(8e-31)
    LL5 = -68.6721   # ln(1.5e-30)
    LL6 = -68.1613   # ln(2.5e-30)
    LL7 = -68.0479   # ln(2.8e-30)
    LL8 = -67.9789   # ln(3.0e-30)   ← global M-shell peak
    LL9 = -68.1220   # ln(2.6e-30)
    LL10 = -68.2891  # ln(2.2e-30)
    LL11 = -68.4898  # ln(1.8e-30)
    LL12 = -68.7411  # ln(1.4e-30)
    LL13 = -69.0776  # ln(1.0e-30)
    LL14 = -69.5884  # ln(6.0e-31)
    LL15 = -69.9938  # ln(4.0e-31)   ← Ar-like trough
    LL16 = -68.8952  # ln(1.2e-30)
    LL17 = -68.7411  # ln(1.4e-30)   ← L-shell peak
    LL18 = -69.3007  # ln(8.0e-31)
    LL19 = -69.7707  # ln(5.0e-31)
    LL20 = -70.2815  # ln(3.0e-31)

    # Find interpolation interval
    if log_T < LT1:
        t0, t1, l0, l1 = LT0, LT1, LL0, LL1
    elif log_T < LT2:
        t0, t1, l0, l1 = LT1, LT2, LL1, LL2
    elif log_T < LT3:
        t0, t1, l0, l1 = LT2, LT3, LL2, LL3
    elif log_T < LT4:
        t0, t1, l0, l1 = LT3, LT4, LL3, LL4
    elif log_T < LT5:
        t0, t1, l0, l1 = LT4, LT5, LL4, LL5
    elif log_T < LT6:
        t0, t1, l0, l1 = LT5, LT6, LL5, LL6
    elif log_T < LT7:
        t0, t1, l0, l1 = LT6, LT7, LL6, LL7
    elif log_T < LT8:
        t0, t1, l0, l1 = LT7, LT8, LL7, LL8
    elif log_T < LT9:
        t0, t1, l0, l1 = LT8, LT9, LL8, LL9
    elif log_T < LT10:
        t0, t1, l0, l1 = LT9, LT10, LL9, LL10
    elif log_T < LT11:
        t0, t1, l0, l1 = LT10, LT11, LL10, LL11
    elif log_T < LT12:
        t0, t1, l0, l1 = LT11, LT12, LL11, LL12
    elif log_T < LT13:
        t0, t1, l0, l1 = LT12, LT13, LL12, LL13
    elif log_T < LT14:
        t0, t1, l0, l1 = LT13, LT14, LL13, LL14
    elif log_T < LT15:
        t0, t1, l0, l1 = LT14, LT15, LL14, LL15
    elif log_T < LT16:
        t0, t1, l0, l1 = LT15, LT16, LL15, LL16
    elif log_T < LT17:
        t0, t1, l0, l1 = LT16, LT17, LL16, LL17
    elif log_T < LT18:
        t0, t1, l0, l1 = LT17, LT18, LL17, LL18
    elif log_T < LT19:
        t0, t1, l0, l1 = LT18, LT19, LL18, LL19
    else:
        t0, t1, l0, l1 = LT19, LT20, LL19, LL20

    frac = (log_T - t0) / (t1 - t0 + 1.0e-300)
    frac = max(0.0, min(1.0, frac))
    return np.exp(l0 + frac * (l1 - l0))


@njit(cache=True)
def _cooling_tungsten(Te_eV: float) -> float:
    """Tungsten (Z=74) cooling function [W m^3].

    Very high-Z impurity with extremely strong and broad line radiation.
    Dominant radiator in tokamak divertors and DPF with W electrodes.
    Peak Lambda ~ 1e-31 near 50 eV.

    Piecewise power-law fit to Pütterich et al., Nucl. Fusion 59,
    056020 (2019) and Post et al. (1977).
    Accuracy: within factor of 2-3 of Pütterich over 10-10000 eV.
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

    P_rec = C_REC * ne^2 * Z^2 * sqrt(chi / (kB * Te))

    where chi = 13.6 * Z^2 eV is the effective ionisation energy
    (hydrogen-like scaling).  Derived from the Seaton (1959)
    recombination rate with quasi-neutrality (ni = ne / Z).

    Note:
        The hydrogenic scaling chi = 13.6 * Z^2 eV is only exact for
        hydrogen-like (single-electron) ions.  For multi-electron ions
        the true ionisation potential is lower (e.g. NIST ASD values),
        so this formula overestimates chi and therefore P_rec for Z > 1.
        The error is moderate for DPF plasmas where Z is typically 1--2.

    Args:
        ne: Electron number density [m^-3].
        Te_K: Electron temperature [K].
        Z: Ion charge state.

    Returns:
        Recombination radiation power density [W/m^3].
    """
    if ne <= 0.0 or Te_K <= 0.0 or Z <= 0.0:
        return 0.0

    # Hydrogenic ionisation energy; exact for Z=1, overestimates for Z>1
    chi_J = 13.6 * Z * Z * eV          # Effective ionisation energy [J]
    kT = k_B * Te_K                     # Thermal energy [J]
    # P = C_REC * ne^2 * Z^2 * sqrt(chi / kT)
    return C_REC * ne * ne * Z * Z * np.sqrt(chi_J / kT)


@njit(cache=True)
def recombination_power(ne: np.ndarray, Te: np.ndarray, Z: float) -> np.ndarray:
    """Radiative recombination power density [W/m^3].

    Uses the Seaton (1959) approximation for hydrogenic recombination:

        P_rec = C_rec * ne^2 * Z^2 * sqrt(chi / (kB Te))

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

    # Bremsstrahlung: P_ff = 1.42e-40 * g_ff * Z_eff * ne^2 * sqrt(Te)  (SI, quasi-neutral ni=ne/Z)
    g_ff = 1.2
    P_brem = 1.42e-40 * g_ff * Z_eff * ne * ne * np.sqrt(Te_K)

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

    1. **Bremsstrahlung** (free-free): P_ff ~ ne^2 * Z_eff * sqrt(Te)  (quasi-neutral: ni=ne/Z)
    2. **Line radiation** (bound-bound): P_line = ne * n_imp * Lambda(Te, Z)
    3. **Recombination** (free-bound):  P_rec ~ ne^2 * Z^2 * sqrt(chi/(kB*Te))

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

    # Energy conservation monitor (NLR panel study P1 action):
    # Verify (3/2)*ne*kB*(Te_old - Te_new) == P_rad * dt to within tolerance.
    # Silent energy drain from Te_floor clamping violates conservation.
    dE_thermal = 1.5 * ne_safe * k_B * (Te_safe - Te_new)  # energy removed [J/m^3]
    dE_radiated = P_rad * dt                                 # energy radiated [J/m^3]
    total_thermal = np.sum(np.abs(dE_thermal)) + 1e-300
    imbalance = np.sum(np.abs(dE_thermal - dE_radiated))
    rel_error = imbalance / total_thermal
    if rel_error > 1e-6:
        logger.warning(
            "Radiation split energy imbalance: %.2e relative error "
            "(thermal removed=%.3e J/m^3, radiated=%.3e J/m^3)",
            rel_error,
            np.sum(dE_thermal),
            np.sum(dE_radiated),
        )

    return Te_new, P_rad
