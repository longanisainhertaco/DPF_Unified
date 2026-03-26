"""Runaway electron diagnostics for DPF.

Computes the Dreicer field threshold and estimates runaway electron
generation rate during m=0 instability disruption. Runaway electrons
produce hard X-rays (>100 keV) observed experimentally.

The Dreicer field is the threshold electric field above which electrons
accelerate faster than they scatter:
    E_D = n_e * e^3 * ln(Lambda) / (4 * pi * epsilon_0^2 * k_B * T_e)

When E > E_D * E_crit_fraction, a fraction of electrons runs away.

References:
    Dreicer, Phys. Rev. 115:238 (1959).
    Connor & Hastie, Nucl. Fusion 15:415 (1975).
"""

from __future__ import annotations

import numpy as np

# Physical constants
_E_CHARGE = 1.602176634e-19
_EPSILON_0 = 8.854187817e-12
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31
_C = 299792458.0
_PI = 3.141592653589793


def dreicer_field(
    ne: np.ndarray | float,
    Te: np.ndarray | float,
    lnL: float = 10.0,
) -> np.ndarray | float:
    """Dreicer critical electric field [V/m].

    E_D = n_e * e^3 * ln(Lambda) / (4 * pi * epsilon_0^2 * k_B * T_e)

    For PF-1000 at pinch: n_e ~ 1e25, T_e ~ 300 eV = 3.5e6 K:
        E_D ~ 1e25 * (1.6e-19)^3 * 10 / (4*pi*(8.85e-12)^2 * 1.38e-23 * 3.5e6)
            ~ 8.5e5 V/m

    The actual inductive E-field during m=0 disruption can reach ~10^7 V/m,
    exceeding E_D by ~10x.

    Args:
        ne: Electron density [m^-3].
        Te: Electron temperature [K].
        lnL: Coulomb logarithm.

    Returns:
        Dreicer field [V/m].
    """
    ne_arr = np.maximum(np.asarray(ne, dtype=np.float64), 1.0)
    Te_arr = np.maximum(np.asarray(Te, dtype=np.float64), 1.0)
    return (
        ne_arr * _E_CHARGE**3 * lnL
        / (4.0 * _PI * _EPSILON_0**2 * _K_B * Te_arr)
    )


def runaway_fraction(
    E_field: np.ndarray | float,
    E_dreicer: np.ndarray | float,
) -> np.ndarray | float:
    """Estimate of runaway electron fraction (Connor-Hastie 1975).

    For E > E_D: f_runaway ~ exp(-E_D / (2*E) - sqrt(E_D / E))

    This is a crude estimate — the full kinetic treatment requires
    solving the Fokker-Planck equation. Useful as a diagnostic flag.

    Args:
        E_field: Applied electric field [V/m].
        E_dreicer: Dreicer critical field [V/m].

    Returns:
        Runaway fraction (0 to ~0.1). Zero below threshold.
    """
    E = np.maximum(np.asarray(E_field, dtype=np.float64), 1e-10)
    Ed = np.maximum(np.asarray(E_dreicer, dtype=np.float64), 1e-10)
    ratio = Ed / E
    # Below threshold: no runaways
    mask = 0.1 * Ed < E  # onset at ~10% of Dreicer
    f_run = np.where(
        mask,
        np.exp(-ratio / 2.0 - np.sqrt(ratio)),
        0.0,
    )
    return np.clip(f_run, 0.0, 0.5)


def hard_xray_power(
    ne: np.ndarray | float,
    f_runaway: np.ndarray | float,
    Te: np.ndarray | float,
    E_field: np.ndarray | float,
) -> np.ndarray | float:
    """Estimate hard X-ray power from runaway Bremsstrahlung [W/m^3].

    P_hxr ~ n_runaway * E_field * e * c * Z_eff * sigma_brem

    Crude order-of-magnitude estimate for diagnostic purposes.

    Args:
        ne: Electron density [m^-3].
        f_runaway: Runaway fraction.
        Te: Background electron temperature [K].
        E_field: Applied electric field [V/m].

    Returns:
        Hard X-ray emission power [W/m^3].
    """
    n_run = np.asarray(ne, dtype=np.float64) * np.asarray(f_runaway, dtype=np.float64)
    # Runaway energy ~ e * E * mean_free_path ~ e * E * c * tau_collision
    # Simplified: P ~ n_run * e * E * v_run, v_run ~ c for relativistic
    E_arr = np.asarray(E_field, dtype=np.float64)
    return n_run * _E_CHARGE * np.abs(E_arr) * _C * 1e-30  # sigma_brem ~ 1e-30 m^2
