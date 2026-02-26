"""Magnetized Noh Z-pinch exact self-similar solution.

Provides the analytical solution for a radially imploding cylindrical plasma
with an embedded azimuthal magnetic field, stagnating through an outward-
propagating strong shock.  This is the gold-standard Z-pinch verification
benchmark for MHD codes.

Self-similar variable:  xi = r / (V_s * t)
  - Upstream  (xi > 1): cold, converging flow with cylindrical compression
  - Downstream (xi < 1): stagnant post-shock gas with uniform density and B

The compression ratio X across the shock satisfies::

    (gamma + 1) - (gamma - 1)*X
        = beta_A * (X - 1)**2 * [(2 - gamma)*X + gamma]

where beta_A = B_0**2 / (mu_0 * rho_0 * V_0**2) is the upstream
magnetization parameter.

References:
    Velikovich A.L., Giuliani J.L., Zalesak S.T., Gardiner T.A.,
    "Exact self-similar solutions for the magnetized Noh Z pinch problem",
    Phys. Plasmas 19, 012707 (2012).

    Noh W.F., "Errors for calculations of strong shocks using an artificial
    viscosity and an artificial heat flux", J. Comput. Phys. 72, 78-120 (1987).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from dpf.constants import mu_0


def compression_ratio(
    gamma: float,
    beta_A: float,
    tol: float = 1e-12,
) -> float:
    """Solve for the shock compression ratio X.

    Parameters
    ----------
    gamma : float
        Adiabatic index (ratio of specific heats).
    beta_A : float
        Upstream magnetization parameter B_0**2 / (mu_0 * rho_0 * V_0**2).
        Zero for unmagnetized Noh.
    tol : float
        Root-finding tolerance.

    Returns
    -------
    float
        Compression ratio X = rho_downstream / rho_upstream_at_shock.
        Always in the range (1, (gamma+1)/(gamma-1)].
    """
    if beta_A < 0.0:
        raise ValueError(f"beta_A must be non-negative, got {beta_A}")

    X_hydro = (gamma + 1.0) / (gamma - 1.0)

    if beta_A < 1e-15:
        return X_hydro

    def _residual(X: float) -> float:
        return (
            (gamma + 1.0)
            - (gamma - 1.0) * X
            - beta_A * (X - 1.0) ** 2 * ((2.0 - gamma) * X + gamma)
        )

    # X is bounded: X -> (gamma+1)/(gamma-1) as beta_A -> 0
    #               X -> 1 as beta_A -> inf
    return brentq(_residual, 1.0 + 1e-14, X_hydro, xtol=tol)


def shock_velocity(V_0: float, X: float) -> float:
    """Shock propagation velocity V_s = V_0 / (X - 1).

    Parameters
    ----------
    V_0 : float
        Initial inflow speed [m/s] (positive).
    X : float
        Compression ratio.

    Returns
    -------
    float
        Shock velocity [m/s] (positive, outward).
    """
    return V_0 / (X - 1.0)


def noh_upstream(
    r: np.ndarray,
    t: float,
    rho_0: float,
    V_0: float,
    B_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Upstream (pre-shock) profiles at positions *r* and time *t*.

    Valid for r > V_s * t (ahead of the shock).

    Parameters
    ----------
    r : ndarray
        Radial positions [m].
    t : float
        Time [s] (must be > 0).
    rho_0 : float
        Initial density [kg/m^3].
    V_0 : float
        Initial inflow speed [m/s] (positive).
    B_0 : float
        Initial azimuthal field magnitude [T].

    Returns
    -------
    rho, vr, B_theta, p : ndarray
        Density [kg/m^3], radial velocity [m/s] (negative = inward),
        azimuthal magnetic field [T], pressure [Pa].
    """
    factor = 1.0 + V_0 * t / r
    rho = rho_0 * factor
    vr = np.full_like(r, -V_0)
    B_theta = B_0 * factor
    p = np.zeros_like(r)
    return rho, vr, B_theta, p


def noh_downstream(
    rho_0: float,
    V_0: float,
    B_0: float,
    gamma: float,
    X: float,
) -> tuple[float, float, float, float]:
    """Uniform post-shock (downstream) state.

    Parameters
    ----------
    rho_0 : float
        Initial density [kg/m^3].
    V_0 : float
        Initial inflow speed [m/s].
    B_0 : float
        Initial azimuthal field magnitude [T].
    gamma : float
        Adiabatic index.
    X : float
        Compression ratio.

    Returns
    -------
    rho, vr, B_theta, p : float
        Post-shock density, velocity (0), B_theta, pressure.
    """
    rho = X**2 * rho_0
    vr = 0.0
    B_theta = X**2 * B_0
    p = (
        rho_0 * V_0**2 * X**2 / (X - 1.0)
        - B_0**2 * X**2 * (X**2 - 1.0) / (2.0 * mu_0)
    )
    return rho, vr, B_theta, p


def noh_exact_solution(
    r: np.ndarray,
    t: float,
    rho_0: float = 1.0,
    V_0: float = 1.0,
    B_0: float = 0.0,
    gamma: float = 5.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Full magnetized Noh exact solution at arbitrary (r, t).

    Combines upstream and downstream profiles with the shock location
    at r_s = V_s * t.

    Parameters
    ----------
    r : ndarray
        Radial positions [m] (must be > 0).
    t : float
        Time [s] (must be > 0).
    rho_0 : float
        Initial density [kg/m^3].
    V_0 : float
        Initial inflow speed [m/s].
    B_0 : float
        Initial azimuthal magnetic field magnitude [T].
    gamma : float
        Adiabatic index.

    Returns
    -------
    dict
        State dict with keys: rho, vr, B_theta, pressure.
        Each is an ndarray with the same shape as *r*.
    """
    if t <= 0:
        raise ValueError(f"Time must be positive, got {t}")

    r = np.asarray(r, dtype=np.float64)

    beta_A = B_0**2 / (mu_0 * rho_0 * V_0**2) if B_0 != 0.0 else 0.0
    X = compression_ratio(gamma, beta_A)
    V_s = shock_velocity(V_0, X)
    r_shock = V_s * t

    # Upstream profiles
    rho_up, vr_up, Bt_up, p_up = noh_upstream(r, t, rho_0, V_0, B_0)

    # Downstream (uniform) values
    rho_d, vr_d, Bt_d, p_d = noh_downstream(rho_0, V_0, B_0, gamma, X)

    # Assemble piecewise solution
    downstream = r < r_shock

    rho = np.where(downstream, rho_d, rho_up)
    vr = np.where(downstream, vr_d, vr_up)
    B_theta = np.where(downstream, Bt_d, Bt_up)
    pressure = np.where(downstream, p_d, p_up)

    return {
        "rho": rho,
        "vr": vr,
        "B_theta": B_theta,
        "pressure": pressure,
        "r_shock": r_shock,
        "V_s": V_s,
        "X": X,
        "beta_A": beta_A,
    }


def verify_rankine_hugoniot(
    rho_0: float,
    V_0: float,
    B_0: float,
    gamma: float,
    X: float | None = None,
) -> dict[str, float]:
    """Verify Rankine-Hugoniot jump conditions at the shock.

    Checks mass, momentum, energy, and induction jumps.  All residuals
    should be zero to machine precision for the exact solution.

    Parameters
    ----------
    rho_0, V_0, B_0, gamma : float
        Problem parameters.
    X : float or None
        Compression ratio (computed if None).

    Returns
    -------
    dict
        Residuals: mass_residual, momentum_residual, energy_residual,
        induction_residual, max_relative_residual.
    """
    beta_A = B_0**2 / (mu_0 * rho_0 * V_0**2) if B_0 != 0.0 else 0.0
    if X is None:
        X = compression_ratio(gamma, beta_A)

    V_s = shock_velocity(V_0, X)

    # Upstream state at shock
    rho_1 = rho_0 * X
    u_1 = V_0 + V_s  # shock-frame velocity (into shock)
    B_1 = B_0 * X

    # Downstream state
    rho_2 = X**2 * rho_0
    u_2 = u_1 / X
    B_2 = X**2 * B_0
    p_2 = (
        rho_0 * V_0**2 * X**2 / (X - 1.0)
        - B_0**2 * X**2 * (X**2 - 1.0) / (2.0 * mu_0)
    )

    # Mass flux: rho * u
    mass_res = rho_1 * u_1 - rho_2 * u_2

    # Momentum flux: rho * u^2 + p + B^2/(2*mu_0)
    mom_1 = rho_1 * u_1**2 + 0.0 + B_1**2 / (2.0 * mu_0)
    mom_2 = rho_2 * u_2**2 + p_2 + B_2**2 / (2.0 * mu_0)
    mom_res = mom_1 - mom_2

    # Energy flux: u * (rho*u^2/2 + gamma*p/(gamma-1) + B^2/mu_0)
    en_1 = u_1 * (rho_1 * u_1**2 / 2.0 + B_1**2 / mu_0)
    en_2 = u_2 * (
        rho_2 * u_2**2 / 2.0
        + gamma * p_2 / (gamma - 1.0)
        + B_2**2 / mu_0
    )
    en_res = en_1 - en_2

    # Induction: B * u = const (frozen-in for perpendicular shock)
    ind_res = B_1 * u_1 - B_2 * u_2

    # Scale for relative errors
    scales = [
        abs(rho_1 * u_1),
        abs(mom_1),
        abs(en_1) if abs(en_1) > 0 else 1.0,
        abs(B_1 * u_1) if abs(B_1 * u_1) > 0 else 1.0,
    ]
    residuals = [abs(mass_res), abs(mom_res), abs(en_res), abs(ind_res)]
    rel_residuals = [r / max(s, 1e-30) for r, s in zip(residuals, scales, strict=True)]

    return {
        "mass_residual": float(mass_res),
        "momentum_residual": float(mom_res),
        "energy_residual": float(en_res),
        "induction_residual": float(ind_res),
        "max_relative_residual": float(max(rel_residuals)),
    }


def create_noh_state(
    nr: int,
    nz: int,
    r_max: float,
    t: float,
    rho_0: float = 1.0,
    V_0: float = 1.0,
    B_0: float = 0.0,
    gamma: float = 5.0 / 3.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Create a 2D cylindrical MHD state dict with magnetized Noh profiles.

    Generates arrays compatible with CylindricalMHDSolver (nr, 1, nz).
    The Noh solution is axially uniform (z-independent).

    Parameters
    ----------
    nr : int
        Number of radial cells.
    nz : int
        Number of axial cells.
    r_max : float
        Outer radial boundary [m].
    t : float
        Solution time [s].
    rho_0, V_0, B_0, gamma : float
        Problem parameters.

    Returns
    -------
    (state, info) : tuple
        state: DPF state dict with rho, velocity, pressure, B, Te, Ti, psi.
        info: Dict with shock metadata (r_shock, V_s, X, beta_A).
    """
    dr = r_max / nr
    dz = r_max / nz
    r_centers = np.linspace(0.5 * dr, r_max - 0.5 * dr, nr)

    exact = noh_exact_solution(r_centers, t, rho_0, V_0, B_0, gamma)

    # Reshape for cylindrical solver: (nr, 1, nz)
    rho_3d = np.broadcast_to(exact["rho"][:, None, None], (nr, 1, nz)).copy()
    p_3d = np.broadcast_to(exact["pressure"][:, None, None], (nr, 1, nz)).copy()

    velocity = np.zeros((3, nr, 1, nz))
    velocity[0, :, 0, :] = exact["vr"][:, None]  # v_r

    B = np.zeros((3, nr, 1, nz))
    B[1, :, 0, :] = exact["B_theta"][:, None]  # B_theta

    # Temperature from ideal gas: p = n k_B T, assume Te = Ti = p/(2*n*k_B)
    # For code verification, just set a reference temperature
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
        "r_shock": exact["r_shock"],
        "V_s": exact["V_s"],
        "X": exact["X"],
        "beta_A": exact["beta_A"],
        "r_centers": r_centers,
        "dr": dr,
        "dz": dz,
    }

    return state, info
