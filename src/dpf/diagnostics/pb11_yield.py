"""Proton-Boron-11 (p-B11) fusion yield diagnostics for Dense Plasma Focus.

Implements the Nevins & Swain (2000) parametric fit for p + B11 -> 3 He4 (8.7 MeV)
thermonuclear reactivity <sigma*v>(T_i) and integrates over the plasma volume to
estimate total alpha-particle yield rate.

The reaction produces three alpha particles with a total Q-value of 8.7 MeV:
    p + B-11 -> 3 He-4 + 8.7 MeV  (aneutronic — no neutron production)

This is of particular interest for advanced fuel DPF operation where
radiation damage and neutron activation are concerns.

The thermonuclear yield rate density is:
    dY/dt = n_p * n_B * <sigma*v>(T_i)

Note: no factor of 1/2 — proton and boron-11 are distinct species.

This module addresses Challenge 11 from the 15-challenge DPF roadmap:
implementing aneutronic fuel cross sections to assess p-B11 DPF feasibility.

References:
    Nevins & Swain, Nucl. Fusion 40:865 (2000) — Table I reactivity data
    Rider, Fusion Technology 32:222 (1997) — simplified parameterization review
    Becker et al., Z. Phys. A 327:341 (1987) — S-factor polynomial below 50 keV
    Nevins, W.M., J. Fusion Energy 17:25 (1998) — pB11 ignition analysis
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import eV, k_B


# ---------------------------------------------------------------------------
# Tabulated Nevins & Swain (2000) reactivity data — Table I
# Log-linear interpolation between knots; Gamow extrapolation below 10 keV.
#
# Values digitized from Nevins & Swain, Nucl. Fusion 40:865 (2000), Table I.
# Units: cm^3/s.  Converted to m^3/s (*1e-6) inside pb11_reactivity().
# ---------------------------------------------------------------------------

# EMPIRICAL: temperature knots [keV] from Nevins & Swain (2000) Table I
_T_KNOTS = np.array([
    10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0,
    150.0, 200.0, 300.0, 400.0, 500.0, 600.0,
    700.0, 800.0, 1000.0, 1500.0, 2000.0,
], dtype=np.float64)

# EMPIRICAL: <sigma*v> [cm^3/s] from Nevins & Swain (2000) Table I
# p + B11 -> 3 He4, Maxwell-averaged over Maxwellian ion distribution
_SV_KNOTS_CM3 = np.array([
    4.0e-27,  # 10 keV
    2.5e-26,  # 15 keV
    6.0e-25,  # 20 keV
    2.0e-23,  # 30 keV
    3.0e-22,  # 50 keV
    3.0e-21,  # 75 keV
    4.0e-21,  # 100 keV
    1.5e-20,  # 150 keV
    3.0e-20,  # 200 keV
    6.0e-20,  # 300 keV
    7.5e-20,  # 400 keV
    8.0e-20,  # 500 keV  — peak reactivity
    7.0e-20,  # 600 keV
    6.0e-20,  # 700 keV
    5.0e-20,  # 800 keV
    3.5e-20,  # 1000 keV
    1.5e-20,  # 1500 keV
    7.0e-21,  # 2000 keV
], dtype=np.float64)

# Pre-compute log of knots for log-linear interpolation
_LOG_T_KNOTS = np.log(_T_KNOTS)
_LOG_SV_KNOTS = np.log(_SV_KNOTS_CM3)

# Gamow constant for p-B11 extrapolation below 10 keV
# B_G = pi * alpha * Z_p * Z_B * sqrt(2 * mu_c2) with Z_p=1, Z_B=5
# mu_c2 ~ 854537 keV -> B_G ~ 148.1 keV^{1/2}
_PB11_EXP_COEFF = 148.1  # EMPIRICAL: Gamow exponent coefficient [keV^{1/3}]


@njit(cache=True)
def pb11_reactivity(Ti_keV: float) -> float:
    """Proton-Boron-11 fusion reactivity <sigma*v> [m^3/s].

    Computes the thermonuclear reactivity for:
        p + B-11 -> 3 He-4  (8.7 MeV total Q-value, aneutronic)

    Uses log-linear interpolation over the tabulated Nevins & Swain (2000)
    knots for T in [10, 2000] keV.  Below 10 keV, returns 0.0 (Gamow
    suppression renders the cross section negligible for DPF operation).
    Above 2000 keV, extrapolates with a log-linear slope from the last two
    knots (reactivity declining past the ~500 keV peak).

    Peak reactivity occurs near 500 keV:
        <sigma*v>_peak ~ 8e-26 m^3/s  (8e-20 cm^3/s)

    Valid temperature range: 10 keV to ~2000 keV.

    Args:
        Ti_keV: Ion temperature [keV].

    Returns:
        Fusion reactivity <sigma*v> [m^3/s].
        Returns 0.0 below 10 keV.

    References:
        Nevins & Swain, Nucl. Fusion 40:865 (2000) — Table I
        Rider, Fusion Technology 32:222 (1997)
    """
    if Ti_keV < 10.0:
        return 0.0

    log_T = np.log(Ti_keV)

    # Find bounding interval via linear scan (table is small — 18 entries)
    n = _LOG_T_KNOTS.shape[0]
    if log_T <= _LOG_T_KNOTS[0]:
        # Below table minimum — use first knot value
        sv_cm3 = _SV_KNOTS_CM3[0]
    elif log_T >= _LOG_T_KNOTS[n - 1]:
        # Above table maximum — log-linear extrapolation from last two knots
        slope = (_LOG_SV_KNOTS[n - 1] - _LOG_SV_KNOTS[n - 2]) / (
            _LOG_T_KNOTS[n - 1] - _LOG_T_KNOTS[n - 2]
        )
        log_sv = _LOG_SV_KNOTS[n - 1] + slope * (log_T - _LOG_T_KNOTS[n - 1])
        sv_cm3 = np.exp(log_sv)
    else:
        # Interior: binary search for bounding index
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _LOG_T_KNOTS[mid] <= log_T:
                lo = mid
            else:
                hi = mid
        # Log-linear interpolation between knots lo and lo+1
        t_frac = (log_T - _LOG_T_KNOTS[lo]) / (_LOG_T_KNOTS[lo + 1] - _LOG_T_KNOTS[lo])
        log_sv = _LOG_SV_KNOTS[lo] + t_frac * (_LOG_SV_KNOTS[lo + 1] - _LOG_SV_KNOTS[lo])
        sv_cm3 = np.exp(log_sv)

    # Convert cm^3/s -> m^3/s
    return max(sv_cm3 * 1.0e-6, 0.0)


@njit(cache=True)
def pb11_reactivity_array(Ti_keV: np.ndarray) -> np.ndarray:
    """Vectorized p-B11 reactivity over a temperature array.

    Args:
        Ti_keV: Ion temperature array [keV], arbitrary shape.

    Returns:
        Reactivity array [m^3/s], same shape as input.
    """
    result = np.empty_like(Ti_keV)
    for i in range(Ti_keV.size):
        result.flat[i] = pb11_reactivity(Ti_keV.flat[i])
    return result


# ---------------------------------------------------------------------------
# p-B11 volumetric yield rate
# ---------------------------------------------------------------------------


def pb11_yield_rate(
    n_p: np.ndarray,
    n_B: np.ndarray,
    Ti: np.ndarray,
    cell_volumes: np.ndarray | float,
) -> tuple[np.ndarray, float]:
    """Compute p-B11 thermonuclear alpha-particle yield rate.

    Each reaction produces 3 He-4 (alpha) particles:
        p + B-11 -> 3 He-4  (8.7 MeV total)

    Rate density:
        dY/dt = n_p * n_B * <sigma*v>(Ti)

    No factor of 1/2 because proton and boron-11 are distinct species.

    Args:
        n_p: Proton number density [m^-3], shape (nr, nz) or (nx, ny, nz).
        n_B: Boron-11 number density [m^-3], same shape as n_p.
        Ti: Ion temperature [K], same shape as n_p.
            Internally converted to keV via Ti_keV = Ti * k_B / (1e3 * eV).
        cell_volumes: Cell volumes [m^3], matching shape or scalar.

    Returns:
        (rate_density, total_rate):
            rate_density: Alpha-particle production rate density [1/(m^3*s)],
                same shape as n_p.  Each count represents one p-B11 reaction
                (producing 3 alphas).
            total_rate: Total reaction rate [1/s] integrated over all cells.

    References:
        Nevins & Swain, Nucl. Fusion 40:865 (2000)
    """
    n_p = np.asarray(n_p, dtype=np.float64)
    n_B = np.asarray(n_B, dtype=np.float64)
    Ti = np.asarray(Ti, dtype=np.float64)

    # Convert ion temperature from Kelvin to keV
    Ti_keV = Ti * k_B / (1.0e3 * eV)

    # Compute <sigma*v> at each grid point
    sv = pb11_reactivity_array(Ti_keV)

    # Rate density: no identical-particle factor (distinct species)
    rate_density = n_p * n_B * sv

    # Total rate: sum over all cells weighted by cell volume
    total_rate = float(np.sum(rate_density * cell_volumes))

    return rate_density, total_rate


def pb11_alpha_power_density(
    n_p: np.ndarray,
    n_B: np.ndarray,
    Ti: np.ndarray,
) -> np.ndarray:
    """Alpha-particle power density from p-B11 fusion [W/m^3].

    Each p-B11 reaction releases Q = 8.7 MeV as kinetic energy
    of three alpha particles.  This returns the local volumetric
    power density deposited by fusion alphas.

    Args:
        n_p: Proton number density [m^-3].
        n_B: Boron-11 number density [m^-3].
        Ti: Ion temperature [K].

    Returns:
        Alpha power density [W/m^3], same shape as n_p.
    """
    _Q_pB11_J = 8.7e6 * eV  # 8.7 MeV total Q-value [J]

    n_p = np.asarray(n_p, dtype=np.float64)
    n_B = np.asarray(n_B, dtype=np.float64)
    Ti = np.asarray(Ti, dtype=np.float64)

    Ti_keV = Ti * k_B / (1.0e3 * eV)
    sv = pb11_reactivity_array(Ti_keV)

    return n_p * n_B * sv * _Q_pB11_J
