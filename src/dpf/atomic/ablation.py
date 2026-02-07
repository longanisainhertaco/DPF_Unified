"""Electrode ablation model for Dense Plasma Focus devices.

Simple surface ablation driven by Ohmic power deposition at electrode
boundaries. Material is injected into the first boundary cell as a mass
source term in the continuity equation.

Physics:
    dm/dt = efficiency * P_surface

where:
    P_surface = eta * J^2 * V_cell   [W]  (Ohmic heating in boundary cell)
    efficiency                        [kg/J] (material-dependent ablation yield)
    dm/dt                             [kg/s] (mass ablation rate)

The volumetric mass source injected at the boundary is:

    S_rho = dm/dt / V_cell = efficiency * eta * J^2   [kg/(m^3 s)]

Typical ablation efficiencies for common electrode materials:
    Copper:   ~5e-5 kg/J  (high thermal conductivity, moderate Z)
    Tungsten: ~2e-5 kg/J  (high melting point, refractory)

These values are empirical and encompass the combined effects of
evaporation, sublimation, and micro-droplet ejection under intense
pulsed heating (timescales ~100 ns to ~1 us).

References:
    Bruzzone & Aranchuk, J. Phys. D: Appl. Phys. 36 (2003) 2218
    Vikhrev & Korolev, Plasma Physics Reports 33 (2007) 356
    Lee & Serban, IEEE Trans. Plasma Sci. 24 (1996) 1101
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

AMU = 1.66054e-27  # Atomic mass unit [kg]

# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------

# Copper electrode parameters
COPPER_ABLATION_EFFICIENCY = 5.0e-5    # [kg/J]
COPPER_MASS = 63.546 * AMU            # Atomic mass [kg]

# Tungsten electrode parameters
TUNGSTEN_ABLATION_EFFICIENCY = 2.0e-5  # [kg/J]
TUNGSTEN_MASS = 183.84 * AMU          # Atomic mass [kg]


# ---------------------------------------------------------------------------
# Core ablation functions
# ---------------------------------------------------------------------------


@njit(cache=True)
def ablation_rate(P_surface: float, efficiency: float) -> float:
    """Compute mass ablation rate from deposited surface power.

    The ablation rate is linearly proportional to the power deposited
    on the electrode surface:

        dm/dt = efficiency * P_surface

    This linear model is valid for moderate power densities typical
    of DPF operation (~10^8 to 10^11 W/m^2). At higher fluences,
    plasma shielding reduces the effective efficiency.

    Args:
        P_surface: Power deposited on the electrode surface [W].
                   Must be non-negative.
        efficiency: Ablation efficiency [kg/J]. Typical values:
                    ~5e-5 for copper, ~2e-5 for tungsten.

    Returns:
        Mass ablation rate [kg/s]. Non-negative.
    """
    if P_surface <= 0.0 or efficiency <= 0.0:
        return 0.0
    return efficiency * P_surface


@njit(cache=True)
def ablation_source(
    rho_boundary: float,
    Te_boundary: float,
    ne_boundary: float,
    J_boundary: float,
    eta_boundary: float,
    dx: float,
    ablation_efficiency: float,
    material_mass: float,
) -> float:
    """Compute volumetric mass source from electrode ablation.

    Evaluates Ohmic power deposition in the boundary cell and converts
    it to a volumetric mass injection rate:

        P_ohmic = eta * J^2          [W/m^3]  (volumetric Ohmic heating)
        S_rho   = efficiency * P_ohmic  [kg/(m^3 s)]

    The ablated material enters as neutral atoms at the local electron
    temperature. In a 1D slab geometry the cell volume is A * dx where
    A is the cross-sectional area; since both P_ohmic and S_rho are
    volumetric densities, the area cancels and the result is
    independent of the transverse geometry.

    Args:
        rho_boundary: Mass density at the boundary cell [kg/m^3].
                      Used for diagnostic context; does not affect the
                      source magnitude in this simple model.
        Te_boundary: Electron temperature at the boundary cell [K].
                     Included for interface consistency and future
                     temperature-dependent efficiency models.
        ne_boundary: Electron number density at the boundary [m^-3].
                     Included for interface consistency.
        J_boundary: Current density magnitude at the boundary [A/m^2].
        eta_boundary: Resistivity at the boundary [Ohm m].
        dx: Grid spacing [m]. Included for interface consistency and
            future models that need cell volume explicitly.
        ablation_efficiency: Material ablation efficiency [kg/J].
        material_mass: Atomic mass of the electrode material [kg].
                       Included for downstream species tracking and
                       momentum injection calculations.

    Returns:
        Volumetric mass source rate S_rho [kg/(m^3 s)].
        Non-negative by construction.
    """
    # Guard against unphysical inputs
    if J_boundary <= 0.0 or eta_boundary <= 0.0 or ablation_efficiency <= 0.0:
        return 0.0

    # Ohmic volumetric power density [W/m^3]
    P_ohmic = eta_boundary * J_boundary * J_boundary

    # Volumetric mass source [kg/(m^3 s)]
    S_rho = ablation_efficiency * P_ohmic

    return S_rho


@njit(cache=True)
def ablation_source_array(
    J: np.ndarray,
    eta: np.ndarray,
    ablation_efficiency: float,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    """Compute ablation mass source on a 1D grid with boundary mask.

    Applies the ablation source calculation only at cells flagged as
    electrode boundary cells (boundary_mask == 1). Interior cells
    receive zero source.

    Args:
        J: Current density magnitude array [A/m^2].
        eta: Resistivity array [Ohm m].
        ablation_efficiency: Material ablation efficiency [kg/J].
        boundary_mask: Integer array, 1 at electrode boundary cells,
                       0 elsewhere. Same shape as J.

    Returns:
        Volumetric mass source array S_rho [kg/(m^3 s)].
        Same shape as J.
    """
    S_rho = np.zeros_like(J)
    for i in range(J.size):
        if boundary_mask.flat[i] == 1:
            J_val = J.flat[i]
            eta_val = eta.flat[i]
            if J_val > 0.0 and eta_val > 0.0:
                S_rho.flat[i] = ablation_efficiency * eta_val * J_val * J_val
    return S_rho


@njit(cache=True)
def ablation_particle_flux(
    S_rho: float,
    material_mass: float,
) -> float:
    """Convert volumetric mass source to particle injection rate density.

    Useful for coupling ablation to ionization and charge-state models:

        S_n = S_rho / m_atom   [particles/(m^3 s)]

    Args:
        S_rho: Volumetric mass source rate [kg/(m^3 s)].
        material_mass: Atomic mass of the ablated species [kg].

    Returns:
        Volumetric particle injection rate [m^-3 s^-1].
    """
    if S_rho <= 0.0 or material_mass <= 0.0:
        return 0.0
    return S_rho / material_mass


@njit(cache=True)
def ablation_momentum_source(
    S_rho: float,
    v_boundary: float,
) -> float:
    """Compute momentum source from ablated material injection.

    Ablated neutrals are assumed to enter with velocity v_boundary
    (typically the local thermal speed or a fraction thereof):

        S_mom = S_rho * v_boundary   [kg/(m^2 s^2) = Pa/m]

    Args:
        S_rho: Volumetric mass source rate [kg/(m^3 s)].
        v_boundary: Injection velocity of ablated material [m/s].
                    Typically ~sqrt(k_B * T_surface / m_atom).

    Returns:
        Volumetric momentum source [N/m^3].
    """
    return S_rho * v_boundary
