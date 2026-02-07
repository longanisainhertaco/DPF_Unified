"""Plasma sheath physics: Bohm criterion, Child-Langmuir law, Debye sheath.

Provides the boundary conditions at electrode surfaces in DPF:
- Bohm velocity (minimum ion speed at sheath edge)
- Child-Langmuir current density (space-charge-limited current)
- Debye length (natural sheath scale)
- Sheath thickness (from Bohm-CL balance)
- 1D Poisson solver for sheath potential profile

All functions are pure (no class state) and use SI units.

Reference:
    Lieberman & Lichtenberg, "Principles of Plasma Discharges" (2005)
    NRL Plasma Formulary (2019)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e, epsilon_0, k_B, m_e, m_p


@njit(cache=True)
def bohm_velocity(Te: float, mi: float = m_p) -> float:
    """Compute the Bohm velocity (minimum ion speed at sheath edge).

    v_B = sqrt(k_B * Te / m_i)

    This is the minimum velocity ions must have when entering the sheath
    for a stable sheath to form (Bohm criterion).

    Args:
        Te: Electron temperature [K].
        mi: Ion mass [kg] (default: proton mass).

    Returns:
        Bohm velocity [m/s].
    """
    Te_safe = max(Te, 0.0)
    return np.sqrt(k_B * Te_safe / mi)


@njit(cache=True)
def child_langmuir_current(V: float, d: float, mi: float = m_p) -> float:
    """Compute the Child-Langmuir space-charge-limited current density.

    J_CL = (4/9) * epsilon_0 * sqrt(2*e/m_i) * V^{3/2} / d^2

    This is the maximum current density that can flow through a planar
    gap of width d under applied voltage V, limited by space charge.

    Args:
        V: Sheath voltage drop [V] (positive).
        d: Sheath thickness [m] (positive).
        mi: Ion mass [kg].

    Returns:
        Current density J_CL [A/m^2].
    """
    V_safe = max(abs(V), 0.0)
    d_safe = max(d, 1e-30)
    return (4.0 / 9.0) * epsilon_0 * np.sqrt(2.0 * e / mi) * V_safe**1.5 / (d_safe * d_safe)


@njit(cache=True)
def debye_length(ne: float, Te: float) -> float:
    """Compute the Debye length.

    lambda_D = sqrt(epsilon_0 * k_B * Te / (ne * e^2))

    The natural length scale of charge separation in a plasma.

    Args:
        ne: Electron density [m^-3].
        Te: Electron temperature [K].

    Returns:
        Debye length [m].
    """
    ne_safe = max(ne, 1e-10)
    Te_safe = max(Te, 0.0)
    return np.sqrt(epsilon_0 * k_B * Te_safe / (ne_safe * e * e))


@njit(cache=True)
def sheath_thickness(ne: float, Te: float, V_sheath: float, mi: float = m_p) -> float:
    """Estimate sheath thickness from Bohm-Child-Langmuir balance.

    At steady state, the ion current entering the sheath (Bohm flux)
    equals the Child-Langmuir current through the sheath:

        n_e * v_B * e = J_CL(V, d)

    Solving for d:
        d = sqrt((4/9) * epsilon_0 * sqrt(2e/m_i) * V^{3/2} / (n_e * v_B * e))

    Args:
        ne: Electron density at sheath edge [m^-3].
        Te: Electron temperature [K].
        V_sheath: Sheath voltage drop [V].
        mi: Ion mass [kg].

    Returns:
        Sheath thickness [m].
    """
    v_B = bohm_velocity(Te, mi)
    ne_safe = max(ne, 1e-10)
    V_safe = max(abs(V_sheath), 0.0)

    # J_Bohm = ne * v_B * e
    J_Bohm = ne_safe * v_B * e

    if J_Bohm <= 0:
        return debye_length(ne_safe, Te)

    # d = sqrt(J_CL_coeff * V^{3/2} / J_Bohm)
    # where J_CL_coeff = (4/9) * epsilon_0 * sqrt(2e/mi)
    J_CL_coeff = (4.0 / 9.0) * epsilon_0 * np.sqrt(2.0 * e / mi)
    d_sq = J_CL_coeff * V_safe**1.5 / J_Bohm
    d = np.sqrt(max(d_sq, 0.0))

    # Minimum: a few Debye lengths
    lam_D = debye_length(ne_safe, Te)
    return max(d, 3.0 * lam_D)


@njit(cache=True)
def floating_potential(Te: float) -> float:
    """Compute the floating potential of an isolated surface.

    V_f = -(k_B * Te / (2*e)) * ln(m_i / (2*pi*m_e))

    At floating potential, the electron and ion currents to the surface
    balance. For hydrogen plasma:
        V_f ~ -2.84 * k_B * Te / e  (in Volts, Te in K)
        V_f ~ -2.84 * Te_eV         (in Volts, Te in eV)

    Args:
        Te: Electron temperature [K].

    Returns:
        Floating potential [V] (negative for electron-repelling).
    """
    Te_safe = max(Te, 0.0)
    # V_f = -(kT/2e) * ln(mi/(2*pi*me))
    # For hydrogen: ln(mp/(2*pi*me)) ~ 5.68
    return -(k_B * Te_safe / (2.0 * e)) * np.log(m_p / (2.0 * np.pi * m_e))


def poisson_1d(
    ne_profile: np.ndarray,
    ni_profile: np.ndarray,
    dx: float,
    V_left: float = 0.0,
    V_right: float = 0.0,
    Z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve 1D Poisson equation for electrostatic potential.

    d^2 phi / dx^2 = -rho / epsilon_0 = -e * (Z*ni - ne) / epsilon_0

    Uses tridiagonal solve (Thomas algorithm via scipy).

    Args:
        ne_profile: Electron density profile [m^-3], shape (N,).
        ni_profile: Ion density profile [m^-3], shape (N,).
        dx: Grid spacing [m].
        V_left: Potential at left boundary [V].
        V_right: Potential at right boundary [V].
        Z: Ion charge state.

    Returns:
        Tuple of (phi, E_field) where:
            phi: Electrostatic potential [V], shape (N,).
            E_field: Electric field [V/m], shape (N,).
    """
    from scipy.linalg import solve_banded

    N = len(ne_profile)
    if N < 3:
        return np.zeros(N), np.zeros(N)

    # RHS: charge density
    rho = e * (Z * ni_profile - ne_profile)
    b = -rho / epsilon_0 * dx * dx

    # Tridiagonal matrix: [1, -2, 1] (stored in banded form)
    # ab shape: (3, N) for solve_banded with l=1, u=1
    ab = np.zeros((3, N))
    ab[0, 1:] = 1.0    # upper diagonal
    ab[1, :] = -2.0     # main diagonal
    ab[2, :-1] = 1.0    # lower diagonal

    # Boundary conditions (Dirichlet)
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    b[0] = V_left

    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    b[-1] = V_right

    # Solve
    phi = solve_banded((1, 1), ab, b)

    # Electric field: E = -d(phi)/dx
    E_field = np.zeros(N)
    E_field[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * dx)
    E_field[0] = -(phi[1] - phi[0]) / dx
    E_field[-1] = -(phi[-1] - phi[-2]) / dx

    return phi, E_field


def apply_sheath_bc(
    state: dict[str, np.ndarray],
    ne_boundary: float,
    Te_boundary: float,
    V_sheath: float,
    mi: float = m_p,
    Z: float = 1.0,
    boundary: str = "z_high",
) -> dict[str, np.ndarray]:
    """Apply Bohm sheath boundary condition to fluid state.

    Sets the ion velocity at the specified boundary to the Bohm velocity
    and limits the density to maintain the Bohm flux.

    Args:
        state: Simulation state dict with 'velocity', 'rho', 'Te'.
        ne_boundary: Electron density at sheath edge [m^-3].
        Te_boundary: Electron temperature at boundary [K].
        V_sheath: Sheath voltage [V].
        mi: Ion mass [kg].
        Z: Ion charge state.
        boundary: Which boundary to apply ("z_high", "z_low", "x_high", "x_low").

    Returns:
        Updated state dict.
    """
    v_B = float(bohm_velocity(Te_boundary, mi))

    vel = state["velocity"]  # (3, nx, ny, nz)

    # Map boundary string to axis and slice
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_name = boundary[0]
    side = boundary.split("_")[1]
    axis = axis_map[axis_name]

    # Build slice for the boundary layer
    sl = [slice(None)] * 4  # (component, nx, ny, nz)
    if side == "high":
        sl[axis + 1] = -1
    else:
        sl[axis + 1] = 0

    # Set velocity component along the boundary axis to Bohm velocity
    # Direction: outward (toward wall)
    sign = 1.0 if side == "high" else -1.0
    vel_sl = list(sl)
    vel_sl[0] = axis
    vel[tuple(vel_sl)] = sign * v_B

    state["velocity"] = vel

    return state
