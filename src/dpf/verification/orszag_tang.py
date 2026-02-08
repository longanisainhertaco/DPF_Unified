"""Orszag-Tang vortex benchmark for the MHD solver.

The Orszag-Tang vortex (Orszag & Tang, 1979) is the canonical 2D MHD
benchmark problem.  It features the development of supersonic MHD
turbulence from smooth initial conditions and is widely used to verify
MHD codes against each other.

Setup
-----
- Domain: [0, 2*pi] x [0, 2*pi], periodic BCs
- gamma = 5/3
- rho  = 25 / (36 * pi)
- p    = 5 / (12 * pi)
- vx   = -sin(y)
- vy   =  sin(x)
- Bx   = -sin(y)
- By   =  sin(2*x)
- vz = Bz = 0

The solution develops thin current sheets, shocks, and vortical
structures.  At t = pi the density field shows a characteristic
pattern with thin shock fronts and a central high-density region.

Since the MHD solver uses zero-gradient (Neumann) boundary conditions
by default (not periodic), this benchmark applies the initial condition
and runs with Neumann BCs.  This limits accuracy at the domain
boundaries after waves reach them, but the interior solution remains
valid for comparison.  A note is included in the result metadata.

Usage::

    from dpf.verification.orszag_tang import run_orszag_tang

    result = run_orszag_tang(nx=128, t_end=0.5)
    print(f"rho range: [{result.rho_min:.4f}, {result.rho_max:.4f}]")
    print(f"Energy conservation: {result.energy_conservation:.4f}")

References
----------
- Orszag & Tang, *J. Fluid Mech.* **90**, 129 (1979).
- Picone & Dahlburg, *Phys. Fluids B* **3**, 29 (1991).
- Londrillo & Del Zanna, *ApJ* **530**, 508 (2000).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dpf.constants import k_B, mu_0, pi

logger = logging.getLogger(__name__)

# z-direction padding: use 5 cells for WENO5 compatibility
_NZ_PAD = 5


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class OrszagTangResult:
    """Container for Orszag-Tang vortex benchmark results.

    Attributes:
        rho_final: Density field at final time, shape (nx, nx).
        rho_min: Minimum density (should remain positive).
        rho_max: Maximum density.
        energy_initial: Total energy at t=0.
        energy_final: Total energy at t_end.
        energy_conservation: Ratio energy_final / energy_initial (ideal = 1.0).
        max_div_B: Maximum |div(B)| at final time.
        nx: Grid size.
        t_end: Final time reached.
        n_steps: Number of timesteps taken.
        gamma: Adiabatic index used.
        bc_note: Note about boundary conditions.
        metadata: Additional metadata dict.
    """

    rho_final: np.ndarray
    rho_min: float = 0.0
    rho_max: float = 0.0
    energy_initial: float = 0.0
    energy_final: float = 0.0
    energy_conservation: float = 0.0
    max_div_B: float = 0.0
    nx: int = 0
    t_end: float = 0.0
    n_steps: int = 0
    gamma: float = 5.0 / 3.0
    bc_note: str = ""
    metadata: dict = field(default_factory=dict)


# ============================================================
# Total energy computation
# ============================================================

def _total_energy(
    rho: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    B: np.ndarray,
    gamma: float,
) -> float:
    """Compute total MHD energy (thermal + kinetic + magnetic).

    E_total = integral[ p/(gamma-1) + 0.5*rho*|v|^2 + |B|^2/(2*mu_0) ] dV

    Since the grid is uniform with unit cell volume (absorbed), we just sum.

    Args:
        rho: Density, shape (nx, ny, nz).
        velocity: Velocity, shape (3, nx, ny, nz).
        pressure: Pressure, shape (nx, ny, nz).
        B: Magnetic field, shape (3, nx, ny, nz).
        gamma: Adiabatic index.

    Returns:
        Total energy (sum over all cells).
    """
    e_thermal = pressure / (gamma - 1.0)
    e_kinetic = 0.5 * rho * np.sum(velocity**2, axis=0)
    e_magnetic = np.sum(B**2, axis=0) / (2.0 * mu_0)
    return float(np.sum(e_thermal + e_kinetic + e_magnetic))


# ============================================================
# Public API
# ============================================================

def run_orszag_tang(
    nx: int = 256,
    t_end: float = 0.5,
    gamma: float = 5.0 / 3.0,
) -> OrszagTangResult:
    """Run the Orszag-Tang vortex benchmark.

    Sets up the standard Orszag-Tang initial conditions on a 2D grid
    (mapped to 3D as (nx, nx, nz_pad)) and advances to *t_end* using
    the MHD solver.

    The solver uses zero-gradient (Neumann) boundary conditions.
    For best results, use *t_end* < 1.0 or moderate resolution so
    that boundary effects remain small.

    Args:
        nx: Number of cells in x and y (square grid).
        t_end: Final time. Standard choices: 0.5, pi.
        gamma: Adiabatic index (default 5/3).

    Returns:
        :class:`OrszagTangResult` with density field, diagnostics, and
        metadata.
    """
    from dpf.fluid.mhd_solver import MHDSolver

    nz = _NZ_PAD
    dx = 2.0 * pi / nx
    grid_shape = (nx, nx, nz)

    # Cell-centered coordinates
    x = np.linspace(0.5 * dx, 2.0 * pi - 0.5 * dx, nx)
    y = np.linspace(0.5 * dx, 2.0 * pi - 0.5 * dx, nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial conditions (Orszag-Tang standard)
    rho0 = 25.0 / (36.0 * pi)
    p0 = 5.0 / (12.0 * pi)

    rho = np.full(grid_shape, rho0)
    pressure = np.full(grid_shape, p0)
    velocity = np.zeros((3,) + grid_shape)
    B = np.zeros((3,) + grid_shape)

    # Fill x-y plane, uniform in z
    for k in range(nz):
        velocity[0, :, :, k] = -np.sin(Y)
        velocity[1, :, :, k] = np.sin(X)
        B[0, :, :, k] = -np.sin(Y)
        B[1, :, :, k] = np.sin(2.0 * X)

    # Temperature (arbitrary; not used by ideal MHD fluxes)
    ion_mass = 1.67e-27
    rho_safe = np.maximum(rho, 1e-30)
    T = pressure * ion_mass / (2.0 * rho_safe * k_B)

    psi = np.zeros(grid_shape)

    state = {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": T.copy(),
        "Ti": T.copy(),
        "psi": psi,
    }

    # Compute initial energy
    E0 = _total_energy(rho, velocity, pressure, B, gamma)

    # Use a moderate CFL and Dedner divergence cleaning for stability.
    # Dedner ch = max signal speed ~ sqrt(gamma * p0 / rho0 + |B|^2 / (mu_0 * rho0))
    cs = np.sqrt(gamma * p0 / rho0)
    va = np.sqrt((1.0**2 + 1.0**2) / (mu_0 * rho0))  # Approximate Alfven speed
    ch_dedner = 2.0 * max(cs, va)

    solver = MHDSolver(
        grid_shape=grid_shape,
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        dedner_ch=ch_dedner,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        enable_energy_equation=True,
        ion_mass=ion_mass,
        riemann_solver="hlld",
    )

    # Time integration
    t = 0.0
    step_count = 0
    max_steps = 500_000

    logger.info(
        "Orszag-Tang vortex: nx=%d, t_end=%.4f, gamma=%.3f, dx=%.4e",
        nx, t_end, gamma, dx,
    )

    rho_floor = 1e-6 * rho0
    p_floor = 1e-6 * p0

    while t < t_end and step_count < max_steps:
        dt = solver._compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break
        state = solver.step(state, dt, current=0.0, voltage=0.0)

        # Apply floors to prevent negative density/pressure
        state["rho"] = np.maximum(state["rho"], rho_floor)
        state["pressure"] = np.maximum(state["pressure"], p_floor)

        # Check for NaN and bail early if detected
        if not np.all(np.isfinite(state["rho"])):
            logger.warning(
                "NaN detected at step %d, t=%.6e. Stopping early.", step_count, t,
            )
            break

        t += dt
        step_count += 1

    logger.info(
        "Orszag-Tang completed: %d steps, t_final=%.6e", step_count, t,
    )

    # Extract 2D density (midplane in z)
    km = nz // 2
    rho_final = state["rho"][:, :, km]

    # Final energy
    E_final = _total_energy(
        state["rho"], state["velocity"], state["pressure"], state["B"], gamma,
    )
    energy_conservation = E_final / E0 if E0 > 0 else 0.0

    # div(B) diagnostic
    Bx = state["B"][0]
    By = state["B"][1]
    Bz = state["B"][2]
    div_B = (
        np.gradient(Bx, dx, axis=0)
        + np.gradient(By, dx, axis=1)
        + np.gradient(Bz, dx, axis=2)
    )
    max_div_B = float(np.max(np.abs(div_B)))

    rho_min = float(np.min(state["rho"]))
    rho_max = float(np.max(state["rho"]))

    bc_note = (
        "Zero-gradient (Neumann) BCs used; periodic BCs not yet supported. "
        "Boundary effects may degrade accuracy at late times."
    )

    return OrszagTangResult(
        rho_final=rho_final,
        rho_min=rho_min,
        rho_max=rho_max,
        energy_initial=E0,
        energy_final=E_final,
        energy_conservation=energy_conservation,
        max_div_B=max_div_B,
        nx=nx,
        t_end=t,
        n_steps=step_count,
        gamma=gamma,
        bc_note=bc_note,
        metadata={
            "rho0": rho0,
            "p0": p0,
            "dx": dx,
            "domain": "[0, 2*pi]^2",
        },
    )
