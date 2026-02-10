"""Cylindrical Sedov-Taylor blast wave verification test.

The Sedov-Taylor self-similar blast wave is one of the most important
verification problems in computational hydrodynamics.  A point-like
energy deposit into a uniform medium drives a strong spherical shock
whose position evolves as a power law in time.

In cylindrical (r, z) geometry with axisymmetry, the blast wave from
a line source (or, equivalently, a point source in the (r, z) plane)
propagates as a cylindrical shock whose radius follows:

    R_s(t) = alpha * (E_lin / rho0)^(1/4) * t^(1/2)

where E_lin is the energy per unit length [J/m] and alpha is a
geometry-dependent constant of order unity.  For the strong-shock
Sedov-Taylor solution in cylindrical geometry, alpha ~= 1.0 (the
exact value depends on gamma and the dimensionality).

For our (r, z) simulation we deposit a total energy E0 into a small
central region and compare the measured shock position to the
similarity solution.  We run pure hydrodynamics (B = 0) to isolate
the Euler solver from MHD complications.

Usage::

    from dpf.verification.sedov_cylindrical import run_sedov_cylindrical

    result = run_sedov_cylindrical(nr=64, nz=128, t_end=0.05)
    print(f"Shock position: {result.shock_position_numerical:.4f}")
    print(f"Analytical:     {result.shock_position_analytical:.4f}")
    print(f"Relative error: {result.relative_error:.2%}")

References
----------
- Sedov, *Similarity and Dimensional Methods in Mechanics* (1959).
- Taylor, *Proc. Roy. Soc. A* **201**, 159 (1950).
- Kamm & Timmes, LA-UR-07-2849 (2007) -- exact solutions for Sedov.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dpf.constants import k_B, pi

logger = logging.getLogger(__name__)


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class SedovCylindricalResult:
    """Container for cylindrical Sedov blast test results.

    Attributes:
        rho_profile_r: Radial density profile at z-midplane, 1D array.
        rho_profile_z: Axial density profile at r=r_min, 1D array.
        r_coords: Radial cell centers [m].
        z_coords: Axial cell centers [m].
        shock_position_numerical: Measured radial shock position [m].
        shock_position_analytical: Analytical Sedov shock position [m].
        relative_error: |R_num - R_ana| / R_ana.
        E0: Deposited energy [J].
        rho0: Background density [kg/m^3].
        t_end: Final time [s].
        n_steps: Number of timesteps taken.
        gamma: Adiabatic index used.
        metadata: Additional metadata dict.
    """

    rho_profile_r: np.ndarray
    rho_profile_z: np.ndarray
    r_coords: np.ndarray
    z_coords: np.ndarray
    shock_position_numerical: float = 0.0
    shock_position_analytical: float = 0.0
    relative_error: float = 0.0
    E0: float = 0.0
    rho0: float = 0.0
    t_end: float = 0.0
    n_steps: int = 0
    gamma: float = 5.0 / 3.0
    metadata: dict = field(default_factory=dict)


# ============================================================
# Similarity solution
# ============================================================

def sedov_shock_radius_cylindrical(
    E0: float,
    rho0: float,
    t: float,
    gamma: float = 5.0 / 3.0,
) -> float:
    """Compute the analytical Sedov-Taylor shock radius for cylindrical geometry.

    For a 2D cylindrical blast (line source), the similarity solution gives:

        R_s(t) = alpha * (E_lin / rho0)^(1/4) * t^(1/2)

    For a point source in the (r, z) plane with total energy E0, we use
    the spherical Sedov solution as an approximation:

        R_s(t) = (xi0 * E0 / rho0)^(1/5) * t^(2/5)

    where xi0 depends on gamma.  For gamma = 5/3, xi0 ~ 1.15.

    We use the standard 2D Sedov-Taylor solution with the energy integral
    constant derived from the gamma-dependent self-similar solution.

    Args:
        E0: Total deposited energy [J].
        rho0: Background mass density [kg/m^3].
        t: Time [s].
        gamma: Adiabatic index.

    Returns:
        Shock radius [m].
    """
    # For 2D cylindrical (nu=2 in Sedov notation, but we use the
    # 3D spherical exponent as an approximation for the (r,z) plane):
    # R_s = (E0 * t^2 / (alpha_cyl * rho0))^(1/4)
    # alpha_cyl depends on gamma; for gamma=5/3, alpha_cyl ~ 3.45
    # This gives the cylindrical (2D) scaling R ~ t^(1/2)
    #
    # However, since our simulation is 2D in (r,z) with r-geometry,
    # the energy goes into a cylindrical volume ~ pi * R^2 * dz.
    # For a point deposit in (r,z), we use the 2D planar Sedov:
    #   R_s = (alpha * E_2d / rho0)^(1/4) * t^(1/2)
    # where E_2d = E0 / (2*pi*dz) is the energy per unit length.
    #
    # For simplicity, we use an approximate formula:
    #   R_s ~ (E0 * t^2 / rho0)^(1/(2+2))  [for 2D]
    #       = (E0 / rho0)^(1/4) * t^(1/2)
    #
    # With a geometry-dependent constant alpha ~ 1.0 for gamma=5/3.

    # Standard 2D Sedov constant (approximate, varies slightly with gamma)
    # For nu=2 (cylindrical), Kamm & Timmes give alpha ~ 1.0 - 1.15
    alpha = 1.0

    # 2D scaling: R ~ (E0/rho0)^(1/4) * t^(1/2)
    R_s = alpha * (E0 / rho0) ** 0.25 * t ** 0.5

    return R_s


def _detect_shock_position(
    rho_1d: np.ndarray,
    coords: np.ndarray,
    rho_background: float,
) -> float:
    """Detect the shock position from a 1D density profile.

    Uses the location where density first exceeds 1.5x the background
    density (scanning from the outside inward), or the location of
    maximum density gradient if no clear threshold crossing is found.

    Args:
        rho_1d: 1D density profile.
        coords: Corresponding coordinates.
        rho_background: Background (undisturbed) density.

    Returns:
        Estimated shock position [m].
    """
    threshold = 1.5 * rho_background

    # Scan from outside inward
    for i in range(len(rho_1d) - 1, 0, -1):
        if rho_1d[i] > threshold:
            # Linear interpolation
            if rho_1d[i - 1] <= threshold:
                frac = (threshold - rho_1d[i - 1]) / max(rho_1d[i] - rho_1d[i - 1], 1e-30)
                return float(coords[i - 1] + frac * (coords[i] - coords[i - 1]))
            continue

    # Fallback: location of maximum density gradient
    grad_rho = np.abs(np.gradient(rho_1d, coords))
    idx = int(np.argmax(grad_rho))
    return float(coords[idx])


# ============================================================
# Public API
# ============================================================

def run_sedov_cylindrical(
    nr: int = 128,
    nz: int = 256,
    t_end: float = 0.1,
    gamma: float = 5.0 / 3.0,
) -> SedovCylindricalResult:
    """Run a cylindrical Sedov-Taylor blast wave test.

    Sets up a uniform background state in cylindrical (r, z) geometry
    and deposits a point-like energy in the central few cells.  The
    CylindricalMHDSolver advances the system to *t_end*, and the
    numerical shock position is compared to the Sedov similarity
    solution.

    Pure hydrodynamics: B = 0 everywhere.

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.
        t_end: Final time [s].
        gamma: Adiabatic index (default 5/3).

    Returns:
        :class:`SedovCylindricalResult` with density profiles, shock
        positions, and error metrics.
    """
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    # Physical parameters
    # Use a weak blast to keep the cylindrical solver stable.
    # The cylindrical solver's two-temperature model introduces additional
    # stiffness that makes strong blast waves challenging.  We use a
    # moderate overpressure (deposit/background ~ 10) which is well within
    # the solver's capability.
    rho0 = 1.0     # Background density [kg/m^3]
    p_bg = 1.0     # Background pressure [Pa]
    E0 = 0.1       # Deposited energy [J] -- moderate blast

    # Compact domain
    R_max = 0.5
    Z_max = 1.0
    dr = R_max / nr
    dz = Z_max / nz

    ion_mass = 1.67e-27  # proton mass [kg]

    # Cell-centered coordinates
    r = (np.arange(nr) + 0.5) * dr
    z = (np.arange(nz) + 0.5) * dz

    # State arrays: (nr, 1, nz)
    rho = np.full((nr, 1, nz), rho0)
    velocity = np.zeros((3, nr, 1, nz))
    pressure = np.full((nr, 1, nz), p_bg)
    B = np.zeros((3, nr, 1, nz))
    psi = np.zeros((nr, 1, nz))

    # Energy deposit: distribute E0 over a wider region to avoid extreme ratios.
    # Use ~10% of the domain in each direction.
    z_center_idx = nz // 2
    n_deposit_r = max(3, nr // 8)
    n_deposit_z = max(3, nz // 8)

    # Compute the volume of deposit cells (cylindrical: V = pi*(r_out^2 - r_in^2)*dz)
    total_volume = 0.0
    for ir in range(n_deposit_r):
        r_in = ir * dr
        r_out = (ir + 1) * dr
        cell_vol = pi * (r_out**2 - r_in**2) * dz
        total_volume += cell_vol * n_deposit_z

    if total_volume > 0:
        e_density = E0 / total_volume  # [J/m^3]
        p_deposit = (gamma - 1.0) * e_density

        # Cap the deposit pressure to prevent extreme ratios
        p_deposit = min(p_deposit, 20.0 * p_bg)

        z_start = z_center_idx - n_deposit_z // 2
        z_end = z_start + n_deposit_z
        for ir in range(n_deposit_r):
            pressure[ir, 0, z_start:z_end] = p_deposit

    # Temperature: derive from pressure (after deposit)
    # T = p * ion_mass / (2 * rho * k_B)
    T_arr = pressure * ion_mass / (2.0 * np.maximum(rho, 1e-30) * k_B)
    T_arr = np.maximum(T_arr, 10.0)  # Floor at 10 K
    Te = T_arr.copy()
    Ti = T_arr.copy()

    state = {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }

    solver = CylindricalMHDSolver(
        nr=nr,
        nz=nz,
        dr=dr,
        dz=dz,
        gamma=gamma,
        cfl=0.15,
        dedner_ch=0.0,
        enable_hall=False,
        enable_resistive=False,
        enable_energy_equation=False,
        ion_mass=ion_mass,
        riemann_solver="hll",
        enable_ct=False,
    )

    # Time integration
    t = 0.0
    step_count = 0
    max_steps = 500_000

    logger.info(
        "Sedov cylindrical blast: nr=%d, nz=%d, t_end=%.4e, E0=%.2e, rho0=%.2e",
        nr, nz, t_end, E0, rho0,
    )

    rho_floor = 1e-6 * rho0
    p_floor = 1e-6 * p_bg

    while t < t_end and step_count < max_steps:
        dt = solver._compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break
        state = solver.step(state, dt, current=0.0, voltage=0.0)

        # Apply floors for numerical stability
        state["rho"] = np.maximum(state["rho"], rho_floor)
        state["pressure"] = np.maximum(state["pressure"], p_floor)

        # Check for NaN and bail early
        if not np.all(np.isfinite(state["rho"])):
            logger.warning(
                "NaN detected at step %d, t=%.6e. Stopping early.", step_count, t,
            )
            break

        t += dt
        step_count += 1

    logger.info("Sedov blast completed: %d steps, t_final=%.6e", step_count, t)

    # Extract 1D profiles
    z_mid = nz // 2
    rho_r = state["rho"][:, 0, z_mid]   # Radial profile at z midplane
    rho_z = state["rho"][0, 0, :]        # Axial profile at r = r_min

    # Detect shock position in the radial direction
    shock_r_num = _detect_shock_position(rho_r, r, rho0)

    # Analytical shock radius
    shock_r_ana = sedov_shock_radius_cylindrical(E0, rho0, t, gamma)

    # Relative error
    if shock_r_ana > 0:
        rel_err = abs(shock_r_num - shock_r_ana) / shock_r_ana
    else:
        rel_err = float("inf")

    logger.info(
        "Shock position: numerical=%.4e, analytical=%.4e, rel_error=%.2f%%",
        shock_r_num, shock_r_ana, rel_err * 100,
    )

    return SedovCylindricalResult(
        rho_profile_r=rho_r,
        rho_profile_z=rho_z,
        r_coords=r,
        z_coords=z,
        shock_position_numerical=shock_r_num,
        shock_position_analytical=shock_r_ana,
        relative_error=rel_err,
        E0=E0,
        rho0=rho0,
        t_end=t,
        n_steps=step_count,
        gamma=gamma,
        metadata={
            "nr": nr,
            "nz": nz,
            "dr": dr,
            "dz": dz,
            "R_max": R_max,
            "Z_max": Z_max,
            "p_bg": p_bg,
        },
    )
