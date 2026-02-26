"""Dynamic Z-pinch analytical slug model for radial implosion.

Provides the 1D cylindrical slug model for a current-driven Z-pinch
implosion, tracking both the magnetic piston and the shock front.
Used as a verification benchmark for MHD radial compression dynamics.

Physics Model
-------------
A cylindrical shell (piston) of swept-up mass is driven inward by
magnetic pressure from axial current I(t).  A strong shock propagates
ahead of the piston into undisturbed gas.

State variables: r_p (piston radius), v_p (piston velocity).
Shock radius derived from geometric mass conservation::

    r_s^2 = (gamma+1)/2 * r_p^2 - (gamma-1)/2 * r_0^2

Momentum equation (per unit axial length)::

    d/dt[m * v_p] = -mu_0 * I^2 / (4*pi*r_p) + 2*pi*r_p*p_back

where m = rho_0 * pi * (r_0^2 - r_p^2) is swept mass.

Strong-shock Rankine-Hugoniot: X = (gamma+1)/(gamma-1).

References
----------
Mesa Dame M. et al., "A Comprehensive Analytical Model of the
Dynamic Z-Pinch", arXiv:2505.18067 (2025).

Potter D.E., Phys. Fluids 14, 1911 (1971).

Rosenbluth M.N., LANL Report LA-1850 (1954).

Lee S. & Serban A., J. Plasma Phys. 55, 83 (1996).

Haines M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from dpf.constants import mu_0, pi

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZPinchResult:
    """Solution of the dynamic Z-pinch slug model.

    Attributes
    ----------
    t : ndarray
        Time array [s].
    r_piston : ndarray
        Piston radius vs time [m].
    v_piston : ndarray
        Piston velocity vs time [m/s] (negative = inward).
    r_shock : ndarray
        Shock radius vs time [m].
    v_shock : ndarray
        Shock velocity vs time [m/s] (negative = inward).
    current : ndarray
        Current vs time [A].
    r0 : float
        Initial radius [m].
    rho0 : float
        Initial gas density [kg/m^3].
    gamma : float
        Adiabatic index.
    compression_ratio : float
        Strong-shock compression ratio X = (gamma+1)/(gamma-1).
    """

    t: np.ndarray
    r_piston: np.ndarray
    v_piston: np.ndarray
    r_shock: np.ndarray
    v_shock: np.ndarray
    current: np.ndarray
    r0: float
    rho0: float
    gamma: float
    compression_ratio: float


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------


def strong_shock_compression_ratio(gamma: float) -> float:
    """Strong-shock compression ratio X = (gamma+1)/(gamma-1).

    Args:
        gamma: Adiabatic index (must be > 1).

    Returns:
        Compression ratio X.

    Raises:
        ValueError: If gamma <= 1.
    """
    if gamma <= 1.0:
        raise ValueError(f"gamma must be > 1, got {gamma}")
    return (gamma + 1.0) / (gamma - 1.0)


def shock_radius_from_piston(
    r_p: float | np.ndarray, r0: float, gamma: float
) -> float | np.ndarray:
    """Compute shock radius from piston radius via geometric mass conservation.

    For a strong cylindrical shock with compression ratio X = (gamma+1)/(gamma-1),
    mass conservation in the annulus gives::

        r_s^2 = (gamma+1)/2 * r_p^2 - (gamma-1)/2 * r_0^2

    Args:
        r_p: Piston radius [m].
        r0: Initial radius [m].
        gamma: Adiabatic index.

    Returns:
        Shock radius r_s [m].
    """
    r_s_sq = (gamma + 1.0) / 2.0 * np.asarray(r_p) ** 2 - (gamma - 1.0) / 2.0 * r0**2
    return np.sqrt(np.maximum(r_s_sq, 0.0))


def implosion_time_constant_current(
    r0: float, rho0: float, I0: float
) -> float:
    """Estimate implosion time for constant-current thin-shell Z-pinch.

    The exact thin-shell solution gives r_p(t) = r_0 * sqrt(1 - (t/t_0)^4)
    where::

        t_0^4 = 2 * pi^2 * rho_0 * r_0^4 / (mu_0 * I_0^2)

    The implosion time is t_imp = t_0.

    Args:
        r0: Initial radius [m].
        rho0: Gas density [kg/m^3].
        I0: Constant current [A].

    Returns:
        Implosion time [s].
    """
    t0_4 = 2.0 * pi**2 * rho0 * r0**4 / (mu_0 * I0**2)
    return t0_4**0.25


def stagnation_pressure(
    rho0: float, v_p: float, gamma: float = 5.0 / 3.0
) -> float:
    """Estimate stagnation pressure from ram pressure at pinch.

    For a strong shock::

        p_stag = rho_0 * v_p^2 * (gamma+1) / (2*(gamma-1))

    Args:
        rho0: Initial gas density [kg/m^3].
        v_p: Piston velocity at stagnation [m/s] (sign ignored).
        gamma: Adiabatic index.

    Returns:
        Stagnation pressure [Pa].
    """
    return rho0 * v_p**2 * (gamma + 1.0) / (2.0 * (gamma - 1.0))


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------


def slug_model_rhs(
    t: float,
    y: np.ndarray,
    I_func: Callable[[float], float],
    r0: float,
    rho0: float,
    gamma: float,
    p_fill: float,
) -> np.ndarray:
    """Right-hand side of the slug model ODE system.

    State vector y = [r_p, v_p]::

        dr_p/dt = v_p
        dv_p/dt = (F_mag + F_back - v_p * dm/dt) / m

    where:
        m = rho_0 * pi * (r_0^2 - r_p^2)      [swept mass per unit length]
        dm/dt = -2*pi*rho_0*r_p*v_p             [mass accretion rate]
        F_mag = -mu_0*I^2 / (4*pi*r_p)          [magnetic drive, inward]
        F_back = 2*pi*r_p*p_fill*(r_0/r_p)^(2*gamma)  [adiabatic back-pressure]

    Args:
        t: Time [s].
        y: State vector [r_p, v_p].
        I_func: Current as function of time I(t) [A].
        r0: Initial radius [m].
        rho0: Initial gas density [kg/m^3].
        gamma: Adiabatic index.
        p_fill: Fill gas pressure [Pa] (0 to disable back-pressure).

    Returns:
        dy/dt = [v_p, dv_p/dt].
    """
    r_p = max(float(y[0]), 1e-10)
    v_p = float(y[1])

    I = I_func(t)  # noqa: E741 — physics convention for current

    # Swept mass per unit length [kg/m]
    m = rho0 * pi * (r0**2 - r_p**2)
    m = max(m, 1e-30)

    # Magnetic driving force per unit length [N/m] (inward = negative)
    F_mag = -mu_0 * I**2 / (4.0 * pi * r_p)

    # Mass accretion rate
    dm_dt = -2.0 * pi * rho0 * r_p * v_p

    # Adiabatic back-pressure
    F_back = 0.0
    if p_fill > 0.0 and r_p < r0:
        p_back = p_fill * (r0 / r_p) ** (2.0 * gamma)
        F_back = 2.0 * pi * r_p * p_back

    # Momentum equation: m * dv/dt = F_mag + F_back - v_p * dm_dt
    dv_dt = (F_mag + F_back - v_p * dm_dt) / m

    return np.array([v_p, dv_dt])


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_zpinch_implosion(
    I_func: Callable[[float], float],
    r0: float,
    rho0: float,
    t_end: float,
    gamma: float = 5.0 / 3.0,
    p_fill: float = 0.0,
    v0: float = 0.0,
    n_points: int = 1000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    r_min_fraction: float = 0.05,
) -> ZPinchResult:
    """Solve the dynamic Z-pinch slug model for radial implosion.

    Integrates the ODE system from t=0 until the piston reaches
    r_min = r_min_fraction * r0 or t_end is reached.

    Args:
        I_func: Current as function of time I(t) [A].
        r0: Initial radius [m].
        rho0: Initial gas density [kg/m^3].
        t_end: Maximum integration time [s].
        gamma: Adiabatic index.
        p_fill: Fill gas pressure [Pa] for adiabatic back-pressure.
        v0: Initial piston velocity [m/s] (negative = inward).
        n_points: Number of output time points.
        rtol: Relative tolerance for ODE solver.
        atol: Absolute tolerance for ODE solver.
        r_min_fraction: Stop when r_p < r_min_fraction * r0.

    Returns:
        ZPinchResult with time histories.

    Raises:
        RuntimeError: If ODE integration fails.
    """
    r_min = r_min_fraction * r0

    def piston_at_rmin(t: float, y: np.ndarray, *args: object) -> float:
        return y[0] - r_min

    piston_at_rmin.terminal = True  # type: ignore[attr-defined]
    piston_at_rmin.direction = -1  # type: ignore[attr-defined]

    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        slug_model_rhs,
        [0, t_end],
        [r0, v0],
        args=(I_func, r0, rho0, gamma, p_fill),
        method="RK45",
        t_eval=t_eval,
        events=[piston_at_rmin],
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t = sol.t
    r_piston = sol.y[0]
    v_piston = sol.y[1]

    # Shock radius from geometric relation
    r_shock = np.asarray(
        shock_radius_from_piston(r_piston, r0, gamma), dtype=np.float64
    )

    # Shock velocity: d(r_s)/dt = (gamma+1)*r_p*v_p / (2*r_s)
    v_shock = np.zeros_like(r_shock)
    valid = r_shock > 1e-10
    v_shock[valid] = (
        (gamma + 1.0) * r_piston[valid] * v_piston[valid] / (2.0 * r_shock[valid])
    )

    current = np.array([I_func(ti) for ti in t])
    X = strong_shock_compression_ratio(gamma)

    return ZPinchResult(
        t=t,
        r_piston=r_piston,
        v_piston=v_piston,
        r_shock=r_shock,
        v_shock=v_shock,
        current=current,
        r0=r0,
        rho0=rho0,
        gamma=gamma,
        compression_ratio=X,
    )


# ---------------------------------------------------------------------------
# Profile reconstruction
# ---------------------------------------------------------------------------


def zpinch_profiles(
    r: np.ndarray,
    r_p: float,
    r_s: float,
    v_p: float,
    r0: float,
    rho0: float,
    gamma: float,
    I_total: float,
) -> dict[str, np.ndarray]:
    """Compute radial profiles at a given instant during Z-pinch implosion.

    Three regions:
        1. Undisturbed (0 < r < r_s): rho = rho_0, v = 0, B = 0
        2. Post-shock  (r_s < r < r_p): compressed, shocked gas
        3. Vacuum      (r > r_p): near-vacuum, magnetic field from current

    Args:
        r: Radial positions [m].
        r_p: Piston radius [m].
        r_s: Shock radius [m].
        v_p: Piston velocity [m/s].
        r0: Initial radius [m].
        rho0: Initial density [kg/m^3].
        gamma: Adiabatic index.
        I_total: Total current [A].

    Returns:
        Dict with keys: rho, vr, pressure, B_theta, region.
        region: 0=undisturbed, 1=post-shock, 2=vacuum.
    """
    r = np.asarray(r, dtype=np.float64)
    X = strong_shock_compression_ratio(gamma)

    # Shock velocity
    v_s = 0.0
    if r_s > 1e-10:
        v_s = (gamma + 1.0) * r_p * v_p / (2.0 * r_s)

    # Region masks
    postshock = (r >= r_s) & (r <= r_p)
    vacuum = r > r_p

    # Initialize with undisturbed values
    rho = np.full_like(r, rho0)
    vr = np.zeros_like(r)
    pressure = np.zeros_like(r)
    B_theta = np.zeros_like(r)
    region = np.zeros_like(r, dtype=int)

    # Region 2: Post-shock compressed gas
    if np.any(postshock):
        region[postshock] = 1

        # Density from mass conservation in the annulus
        if r_p > r_s > 0:
            rho_post = rho0 * (r0**2 - r_s**2) / (r_p**2 - r_s**2)
        else:
            rho_post = X * rho0
        rho[postshock] = rho_post

        # Velocity: linear interpolation from v_s at r_s to v_p at r_p
        if r_p > r_s:
            frac = (r[postshock] - r_s) / (r_p - r_s)
            vr[postshock] = v_s + frac * (v_p - v_s)

        # Pressure from strong-shock R-H jump
        if abs(v_s) > 0:
            p_post = rho0 * v_s**2 * (1.0 - 1.0 / X)
        else:
            p_post = 0.0
        pressure[postshock] = max(p_post, 0.0)

    # Region 3: Vacuum/driver outside piston
    region[vacuum] = 2
    rho[vacuum] = rho0 * 1e-6
    safe_r = np.where(r[vacuum] > 0, r[vacuum], 1e-30)
    B_theta[vacuum] = mu_0 * I_total / (2.0 * pi * safe_r)

    return {
        "rho": rho,
        "vr": vr,
        "pressure": pressure,
        "B_theta": B_theta,
        "region": region,
        "r_shock": r_s,
        "r_piston": r_p,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_geometric_relation(result: ZPinchResult) -> np.ndarray:
    """Check the geometric mass-conservation relation at each timestep.

    Verifies::

        r_p^2 - r_s^2 = (gamma-1)/2 * (r_0^2 - r_p^2)

    Args:
        result: Solution from solve_zpinch_implosion.

    Returns:
        Relative error at each timestep.
    """
    r_p = result.r_piston
    r_s = result.r_shock
    gamma = result.gamma

    lhs = r_p**2 - r_s**2
    rhs = (gamma - 1.0) / 2.0 * (result.r0**2 - r_p**2)

    scale = np.maximum(np.abs(lhs), 1e-30)
    return np.abs(lhs - rhs) / scale


def kinetic_energy(result: ZPinchResult) -> np.ndarray:
    """Kinetic energy of the slug per unit axial length [J/m].

    KE = 0.5 * m * v_p^2 where m = rho_0 * pi * (r_0^2 - r_p^2).

    Args:
        result: Solution from solve_zpinch_implosion.

    Returns:
        Kinetic energy per unit length [J/m] at each timestep.
    """
    m = result.rho0 * pi * (result.r0**2 - result.r_piston**2)
    return 0.5 * m * result.v_piston**2


def magnetic_energy(result: ZPinchResult) -> np.ndarray:
    """Magnetic energy in the vacuum annulus per unit axial length [J/m].

    W_B = (mu_0 * I^2) / (4*pi) * ln(r_0 / r_p)

    Args:
        result: Solution from solve_zpinch_implosion.

    Returns:
        Magnetic energy per unit length [J/m] at each timestep.
    """
    ratio = np.maximum(result.r0 / result.r_piston, 1.0)
    return mu_0 * result.current**2 / (4.0 * pi) * np.log(ratio)


# ---------------------------------------------------------------------------
# DPF state dict creation
# ---------------------------------------------------------------------------


def create_zpinch_state(
    nr: int,
    nz: int,
    r_max: float,
    r_p: float,
    r_s: float,
    v_p: float,
    r0: float,
    rho0: float,
    gamma: float,
    I_total: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Create a 2D cylindrical MHD state dict from Z-pinch slug model profiles.

    Generates arrays compatible with CylindricalMHDSolver (nr, 1, nz).
    Profiles are axially uniform (infinite-cylinder approximation).

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.
        r_max: Outer radial boundary [m].
        r_p: Piston radius [m].
        r_s: Shock radius [m].
        v_p: Piston velocity [m/s].
        r0: Initial radius [m].
        rho0: Initial density [kg/m^3].
        gamma: Adiabatic index.
        I_total: Total current [A].

    Returns:
        (state, info): DPF state dict and metadata dict.
    """
    dr = r_max / nr
    r_centers = np.linspace(0.5 * dr, r_max - 0.5 * dr, nr)

    profiles = zpinch_profiles(
        r_centers, r_p, r_s, v_p, r0, rho0, gamma, I_total
    )

    rho_3d = np.broadcast_to(
        profiles["rho"][:, None, None], (nr, 1, nz)
    ).copy()
    p_3d = np.broadcast_to(
        profiles["pressure"][:, None, None], (nr, 1, nz)
    ).copy()

    velocity = np.zeros((3, nr, 1, nz))
    velocity[0, :, 0, :] = profiles["vr"][:, None]

    B = np.zeros((3, nr, 1, nz))
    B[1, :, 0, :] = profiles["B_theta"][:, None]

    Te_3d = np.full((nr, 1, nz), 1.0)
    Ti_3d = np.full((nr, 1, nz), 1.0)
    psi_3d = np.zeros((nr, 1, nz))

    state = {
        "rho": rho_3d,
        "velocity": velocity,
        "pressure": p_3d,
        "B": B,
        "Te": Te_3d,
        "Ti": Ti_3d,
        "psi": psi_3d,
    }

    info = {
        "r_piston": r_p,
        "r_shock": r_s,
        "v_piston": v_p,
        "I_total": I_total,
        "r_centers": r_centers,
        "dr": dr,
    }

    return state, info
