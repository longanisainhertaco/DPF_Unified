"""Circuit-MHD coupling and two-temperature source terms for the MLX solver.

Extracted from MLXMHDSolver (God Class decomposition). Contains:
  - Plasma inductance computation (Lee formula, density-weighted r_eff)
  - Two-temperature electron-ion equilibration source terms
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dpf.core.bases import CouplingState
from dpf.metal.constants import MU_0


def _bdf2_dLp_dt(history: list[tuple[float, float]], Lp: float, t: float) -> float:
    """Compute dLp/dt using BDF2 (3-point) when history is available.

    Falls back to backward difference with 2 points, returns 0.0 with < 2.

    BDF2: dLp/dt = (3*Lp_n - 4*Lp_{n-1} + Lp_{n-2}) / (2*dt)
    Non-uniform dt variant used when timestep varies.

    References
    ----------
    Hairer & Wanner, "Solving ODEs II", Springer 1996, Sec. III.1.
    Matches CircuitCoupler._compute_dLp_dt in circuit/coupler.py.
    """
    if len(history) >= 2:
        t1, Lp1 = history[-1]
        t0, Lp0 = history[-2]
        dt1 = t - t1
        dt0 = t1 - t0
        if dt1 > 0 and dt0 > 0:
            # Non-uniform BDF2: weighted finite difference
            r = dt1 / dt0
            denom = dt1 * (1.0 + r)
            return ((1.0 + 2.0 * r) * Lp - (1.0 + r) ** 2 * Lp1 + r**2 * Lp0) / denom
    if len(history) >= 1:
        t1, Lp1 = history[-1]
        dt1 = t - t1
        if dt1 > 0:
            return (Lp - Lp1) / dt1
    return 0.0


def update_coupling(
    U: Any,
    current: float,
    voltage: float,
    dt: float,
    grid: Any,
    cathode_radius: float,
    r_inner: float,
    prev_Lp: float,
    Lp_max: float,
    coordinates: str,
    Lp_history: list[tuple[float, float]] | None = None,
    sim_time: float = 0.0,
) -> tuple[CouplingState, float, float]:
    """Compute plasma inductance and return circuit coupling state.

    Uses the Lee formula: Lp = (mu0/2pi) * z_sheath * ln(b/r_eff)
    where r_eff is the density-weighted effective radius.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    current, voltage : float
        Circuit current [A] and voltage [V].
    dt : float
        Timestep [s].
    grid : object
        Grid with .dr, .dz attributes.
    cathode_radius : float
        Outer electrode radius [m].
    r_inner : float
        Inner radial boundary [m].
    prev_Lp : float
        Previous timestep inductance [H].
    Lp_max : float
        Peak inductance seen so far [H].
    coordinates : str
        "cylindrical" or "cartesian".
    Lp_history : list of (time, Lp) tuples, optional
        History for BDF2 dLp/dt. Mutated in-place (appended to).
        Kept to max 3 entries.
    sim_time : float
        Current simulation time [s] (for BDF2 time stamps).

    Returns
    -------
    tuple[CouplingState, float, float]
        (coupling_state, new_prev_Lp, new_Lp_max)

    References
    ----------
    Lee & Saw, Phys. Plasmas 21, 072501 (2014).
    """
    from dpf.metal.mlx_kernels import IDN

    if coordinates == "cartesian":
        return CouplingState(current=current, voltage=voltage), prev_Lp, Lp_max

    rho_np = np.asarray(U[IDN])  # (nr, nz)
    nr, nz = rho_np.shape
    dr = grid.dr
    dz = grid.dz
    r_arr = r_inner + (np.arange(nr) + 0.5) * dr

    # Sheath position from column density peak
    col_density = np.sum(rho_np * r_arr[:, np.newaxis], axis=0) * dr
    iz_sheath = int(np.argmax(col_density))
    z_sheath = (iz_sheath + 0.5) * dz

    # Density-weighted effective radius
    rho_region = rho_np[:, : iz_sheath + 1]
    r_col = r_arr[:, np.newaxis]
    dV = 2.0 * math.pi * r_col * dr * dz
    mass = rho_region * dV
    total_mass = float(np.sum(mass))
    if total_mass > 0:
        r_eff = float(np.sum(r_col * mass) / total_mass)
    else:
        r_eff = 0.5 * cathode_radius
    r_eff = max(r_eff, 1e-6)
    r_eff = min(r_eff, cathode_radius * 0.999)

    # Lee formula
    if r_eff > 0 and z_sheath > 0:
        Lp = (MU_0 / (2.0 * math.pi)) * z_sheath * math.log(
            cathode_radius / r_eff
        )
    else:
        Lp = 0.0

    # Phase-aware monotonicity
    if Lp > Lp_max:
        Lp_max = Lp
    elif Lp < Lp_max * 0.98:
        pass
    else:
        Lp = Lp_max

    # dL/dt via BDF2 (3-point) or backward difference (2-point)
    if Lp_history is not None:
        dL_dt = _bdf2_dLp_dt(Lp_history, Lp, sim_time)
        Lp_history.append((sim_time, Lp))
        # Keep only last 3 entries
        while len(Lp_history) > 3:
            Lp_history.pop(0)
    else:
        dL_dt = (Lp - prev_Lp) / dt if prev_Lp > 0 and dt > 0 else None
    prev_Lp = Lp

    coupling = CouplingState(
        Lp=Lp, current=current, voltage=voltage, dL_dt=dL_dt,
    )
    return coupling, prev_Lp, Lp_max


def do_two_temperature_sources(
    result: dict[str, np.ndarray],
    dt: float,
    eta_field: float | np.ndarray | None,
    ion_mass: float,
    dx: float,
    Z_eff: float,
    gaunt_factor: float,
    gamma: float,
) -> None:
    """Apply electron-ion equilibration, Ohmic heating, bremsstrahlung.

    Modifies result dict in-place.
    """
    from dpf.fluid.two_temperature import step_electron_energy

    rho = result["rho"]
    n_i = rho / ion_mass
    n_i_safe = np.maximum(n_i, 1e-30)
    eta_np = (
        np.asarray(eta_field) if eta_field is not None
        else np.zeros_like(rho)
    )
    J_sq = np.zeros_like(rho)
    e_e_new, Te_new, Ti_new = step_electron_energy(
        rho_e_e=result["e_electron"],
        rho=rho,
        velocity=result["velocity"],
        eta=eta_np,
        J_sq=J_sq,
        Te=result["Te"],
        Ti=result["Ti"],
        n_e=n_i_safe,
        n_i=n_i_safe,
        dx=dx,
        dt=dt,
        Z=Z_eff,
        gaunt_factor=gaunt_factor,
        gamma=gamma,
    )
    result["Te"] = np.maximum(Te_new, 1.0)
    result["Ti"] = np.maximum(Ti_new, 1.0)
    result["e_electron"] = e_e_new
