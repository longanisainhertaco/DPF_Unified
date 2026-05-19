"""Neutron yield diagnostics for DD and DT fusion in Dense Plasma Focus.

Implements the Bosch-Hale parametric fit for DD fusion reactivity <sigma*v>(T_i)
and integrates over the pinch volume and time to estimate total neutron yield.

The thermonuclear neutron yield rate is:
    dY/dt = (1/4) * n_D^2 * <sigma*v>(T_i) * V_pinch

where the factor 1/4 accounts for identical particle reactions (DD).

For beam-target contribution (dominant in many DPF devices):
    dY/dt_beam ~ n_D * n_beam * <sigma*v>(E_beam) * V_interaction

This module computes the thermonuclear component.  For beam-target yields
(dominant in many DPF devices), see ``dpf.diagnostics.beam_target``.

Reference:
    Bosch & Hale, Nuclear Fusion 32:611 (1992)
    — Parametric fit valid for 0.2 keV < T < 100 keV
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import eV, k_B

# S3.6 / WP-N6 §4 uncited-coefficient isolation.
# The Bosch-Hale reactivity fit <sigma v>(Ti) is source-backed verbatim
# (KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:59-93,106-109).
# The volumetric reaction-rate prefactor 1/4 in dY/dt = (1/4) n_D^2 <sigma v> V
# has NO verbatim KnowledgeReference formula for the full reaction-rate
# equation — KR supplies <sigma v> only. The 1/4 (identical-particle 1/2 x
# neutron-branch 1/2) is standard physics but inferred, so it is labelled
# inferred_candidate and isolated from neutron authority: the thermonuclear
# channel in dpf.first_principles.neutron_authority stays missing_or_blocked
# until prefactor_citation is a reviewed KR source. This constant is the
# explicit flag; the 1/4 is NOT silently treated as a source-backed default.
THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS = {
    "coefficient": "thermonuclear_volumetric_prefactor_one_quarter",
    "value": 0.25,
    "status": "inferred_candidate",
    "kr_source": "none_for_full_reaction_rate_equation",
    "reactivity_fit_kr_source": (
        "KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:59-93,106-109"
    ),
    "isolation": (
        "thermonuclear neutron authority stays blocked until the 1/4 prefactor "
        "has a reviewed KR citation (WP-N6 §4); not a source-backed default"
    ),
    "can_support_first_principles_acceptance": False,
}


@njit(cache=True)
def dd_reactivity(Ti_keV: float) -> float:
    """Bosch-Hale DD fusion reactivity <sigma*v> [m^3/s].

    Uses the parametric fit from Bosch & Hale (1992) for the DD reaction:
        D + D -> He-3 + n  (2.45 MeV neutron)
        D + D -> T + p     (3.02 MeV proton)

    Both branches have roughly equal probability at DPF temperatures.
    This returns the total reactivity for both branches.

    The fit is:
        <sigma*v> = C1 * theta * sqrt(xi / (mu_c^2 * T^3)) * exp(-3*xi)

    where:
        theta = T / (1 - T*(C2 + T*(C4 + T*C6)) / (1 + T*(C3 + T*(C5 + T*C7))))
        xi = (B_G^2 / (4*theta))^(1/3)

    Args:
        Ti_keV: Ion temperature in keV.

    Returns:
        Fusion reactivity <sigma*v> in m^3/s.
        Returns 0 for T < 0.2 keV (below fit validity).
    """
    if Ti_keV < 0.2:
        return 0.0
    if Ti_keV > 100.0:
        Ti_keV = 100.0  # Cap at fit validity range

    # Bosch-Hale coefficients for DD (both branches combined)
    # Branch 1: D(d,n)He3
    B_G = 31.3970  # Gamow constant [keV^{1/2}]
    mu_c2 = 937814.0  # Reduced mass * c^2 [keV]

    # Fit coefficients (Table IV of Bosch & Hale 1992, D(d,n)He3)
    C1 = 5.43360e-12  # [keV*cm^3/s] -> will convert to m^3/s
    C2 = 5.85778e-3
    C3 = 7.68222e-3
    C4 = 0.0
    C5 = -2.96400e-6
    C6 = 0.0
    C7 = 0.0

    T = Ti_keV
    denom = 1.0 + T * (C3 + T * (C5 + T * C7))
    if abs(denom) < 1e-30:
        return 0.0

    numer = T * (C2 + T * (C4 + T * C6))
    theta = T / (1.0 - numer / denom)

    if theta <= 0:
        return 0.0

    xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)

    # Reactivity for branch 1 (D(d,n)He3)
    sv_1 = C1 * theta * np.sqrt(xi / (mu_c2 * T**3)) * np.exp(-3.0 * xi)

    # Branch 2: D(d,p)T — separate Bosch-Hale coefficients (Table IV)
    # B_G and mu_c2 are the same (same entrance channel D+D)
    C1_2 = 5.65718e-12
    C2_2 = 3.41267e-3
    C3_2 = 1.99167e-3
    C4_2 = 0.0
    C5_2 = 1.05060e-5
    C6_2 = 0.0
    C7_2 = 0.0

    denom_2 = 1.0 + T * (C3_2 + T * (C5_2 + T * C7_2))
    if abs(denom_2) < 1e-30:
        return max(sv_1 * 1e-6, 0.0)
    numer_2 = T * (C2_2 + T * (C4_2 + T * C6_2))
    theta_2 = T / (1.0 - numer_2 / denom_2)

    if theta_2 <= 0:
        return max(sv_1 * 1e-6, 0.0)

    xi_2 = (B_G**2 / (4.0 * theta_2)) ** (1.0 / 3.0)
    sv_2 = C1_2 * theta_2 * np.sqrt(xi_2 / (mu_c2 * T**3)) * np.exp(-3.0 * xi_2)

    # Total reactivity (both branches)
    # Convert from cm^3/s to m^3/s: multiply by 1e-6
    sv_total = (sv_1 + sv_2) * 1e-6

    return max(sv_total, 0.0)


@njit(cache=True)
def dd_reactivity_array(Ti_keV: np.ndarray) -> np.ndarray:
    """Vectorized DD reactivity for temperature arrays.

    Args:
        Ti_keV: Ion temperature array [keV].

    Returns:
        Reactivity array [m^3/s].
    """
    result = np.empty_like(Ti_keV)
    for i in range(Ti_keV.size):
        result.flat[i] = dd_reactivity(Ti_keV.flat[i])
    return result


def neutron_yield_rate(
    n_D: np.ndarray,
    Ti: np.ndarray,
    cell_volumes: np.ndarray | float,
) -> tuple[np.ndarray, float]:
    """Compute thermonuclear DD neutron yield rate.

    dY/dt = (1/4) * n_D^2 * <sigma*v>(Ti) * dV

    The prefactor 1/4 (identical-particle 1/2 x neutron-branch 1/2) is an
    inferred coefficient with no verbatim KnowledgeReference formula — see
    ``THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS``. This function is a runtime
    diagnostic; it cannot support neutron authority. The thermonuclear channel
    in ``dpf.first_principles.neutron_authority`` stays ``missing_or_blocked``
    while the prefactor is uncited (WP-N6 §4).

    Args:
        n_D: Deuterium number density [m^-3], shape (nr, nz) or (nx, ny, nz).
        Ti: Ion temperature [K], shape matching n_D.
        cell_volumes: Cell volumes [m^3], matching shape or scalar.

    Returns:
        (rate_density, total_rate):
            rate_density: Neutron production rate density [1/(m^3*s)], same shape as n_D.
            total_rate: Total neutron production rate [1/s] (integrated over volume).
    """
    # Convert Ti from Kelvin to keV
    Ti_keV = Ti * k_B / (1000.0 * eV)

    # Compute reactivity <sigma*v>(Ti)
    sv = dd_reactivity_array(Ti_keV)

    # Thermonuclear neutron rate density: (1/4) * n_D^2 * <sigma*v>_total
    # Factor breakdown:
    #   1/2 for identical-particle DD reactions
    #   1/2 because only the D(d,n)He3 branch produces neutrons
    #   (dd_reactivity sums both branches: D(d,n)He3 + D(d,p)T)
    # Combined: (1/2) * (1/2) = 1/4
    rate_density = 0.25 * n_D**2 * sv

    # Total rate: integrate over volume
    total_rate = float(np.sum(rate_density * cell_volumes))

    return rate_density, total_rate


def integrate_neutron_yield(
    n_D: np.ndarray,
    Ti: np.ndarray,
    cell_volumes: np.ndarray | float,
    dt: float,
) -> float:
    """Compute neutron yield for one timestep.

    Y = dY/dt * dt

    Args:
        n_D: Deuterium number density [m^-3].
        Ti: Ion temperature [K].
        cell_volumes: Cell volumes [m^3].
        dt: Timestep [s].

    Returns:
        Neutron yield for this timestep (dimensionless count).
    """
    _, total_rate = neutron_yield_rate(n_D, Ti, cell_volumes)
    return total_rate * dt
