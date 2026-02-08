"""Diffusion convergence test for verifying resistive magnetic diffusion solvers.

Tests the numerical solution of the 1D magnetic diffusion equation against
the analytical Gaussian diffusion solution.  Three methods are supported:

1. **Explicit**: Full MHD solver with zero velocity / pressure gradient,
   letting only the resistive diffusion term act.
2. **Implicit**: Crank-Nicolson ADI solver from ``dpf.fluid.implicit_diffusion``.
3. **STS** (super time-stepping): RKL2 solver from ``dpf.fluid.super_time_step``.

Test problem
------------
Initial condition for *B_y*:

    B_y(x, t=0) = B0 * exp(-x^2 / (2 * sigma0^2))

Analytical solution (Green's function for the 1D heat equation):

    B_y(x, t) = B0 * (sigma0 / sigma(t)) * exp(-x^2 / (2 * sigma(t)^2))

where:

    sigma(t) = sqrt(sigma0^2 + 2 * D * t)
    D        = eta / mu_0

The domain is [-L, L] with zero-gradient (Neumann) boundary conditions.
The integration runs for one diffusion time t_end = sigma0^2 / D, at which
point sigma(t_end) = sqrt(3) * sigma0.

Usage::

    from dpf.verification.diffusion_convergence import run_diffusion_convergence

    result = run_diffusion_convergence(method="implicit", resolutions=[32, 64, 128, 256])
    print(f"Convergence order: {result.convergence_order:.2f}")
    print(f"Errors: {result.errors}")

References
----------
- Ryu & Jones, ApJ 442, 228 (1995) -- resistive MHD test problems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dpf.constants import k_B, mu_0

logger = logging.getLogger(__name__)

# Transverse padding for the 3D grid (MHD solver requires >= 5 cells for WENO5)
_NY_PAD = 5
_NZ_PAD = 5


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class DiffusionConvergenceResult:
    """Container for diffusion convergence test results.

    Attributes:
        method: Diffusion method tested ("explicit", "implicit", or "sts").
        resolutions: List of grid sizes tested.
        errors: L2 errors at each resolution.
        convergence_order: Estimated convergence order from log-log fit.
        eta: Resistivity used [Ohm*m].
        sigma0: Initial Gaussian width [m].
        t_end: Integration time [s].
    """

    method: str
    resolutions: list[int]
    errors: list[float] = field(default_factory=list)
    convergence_order: float = 0.0
    eta: float = 0.0
    sigma0: float = 0.0
    t_end: float = 0.0


# ============================================================
# Analytical solution
# ============================================================

def gaussian_B_analytical(
    x: np.ndarray,
    t: float,
    B0: float,
    sigma0: float,
    D: float,
) -> np.ndarray:
    """Analytical solution for 1D Gaussian magnetic diffusion.

    B_y(x, t) = B0 * (sigma0 / sigma(t)) * exp(-x^2 / (2 * sigma(t)^2))

    Args:
        x: Spatial coordinates [m], 1D array.
        t: Time [s].
        B0: Peak field amplitude [T].
        sigma0: Initial Gaussian width [m].
        D: Magnetic diffusivity eta/mu_0 [m^2/s].

    Returns:
        B_y field at time t, same shape as x.
    """
    sigma_sq = sigma0**2 + 2.0 * D * t
    sigma = np.sqrt(sigma_sq)
    return B0 * (sigma0 / sigma) * np.exp(-x**2 / (2.0 * sigma_sq))


# ============================================================
# Convergence order estimation
# ============================================================

def _estimate_convergence_order(
    resolutions: list[int],
    errors: list[float],
) -> float:
    """Estimate convergence order from (resolution, error) pairs via log-log fit.

    Args:
        resolutions: Grid sizes.
        errors: Corresponding L2 errors.

    Returns:
        Estimated convergence order (positive = error decreases with N).
    """
    if len(resolutions) < 2:
        return 0.0

    log_N: list[float] = []
    log_e: list[float] = []
    for N, err in zip(resolutions, errors, strict=False):
        if np.isfinite(err) and err > 0 and N > 0:
            log_N.append(np.log(float(N)))
            log_e.append(np.log(err))

    if len(log_N) < 2:
        return 0.0

    log_N_arr = np.array(log_N)
    log_e_arr = np.array(log_e)
    A = np.vstack([log_N_arr, np.ones(len(log_N_arr))]).T
    result = np.linalg.lstsq(A, log_e_arr, rcond=None)
    slope = result[0][0]

    return float(-slope)


# ============================================================
# Runner for implicit method
# ============================================================

def _run_implicit(
    nx: int,
    L: float,
    B0: float,
    sigma0: float,
    eta: float,
    t_end: float,
) -> float:
    """Run the implicit Crank-Nicolson diffusion solver and return L2 error.

    Uses the 1D Crank-Nicolson solver (``diffuse_field_1d``) directly
    on x-direction pencils to avoid ADI splitting artifacts from the
    transverse (y, z) dimensions.

    Args:
        nx: Number of cells in x.
        L: Half-domain size [m].
        B0: Peak field amplitude [T].
        sigma0: Initial Gaussian width [m].
        eta: Resistivity [Ohm*m].
        t_end: Integration time [s].

    Returns:
        L2 error between numerical and analytical solution.
    """
    from dpf.fluid.implicit_diffusion import diffuse_field_1d

    dx = 2.0 * L / nx
    x = np.linspace(-L + 0.5 * dx, L - 0.5 * dx, nx)
    D = eta / mu_0

    # Initial condition: B_y Gaussian in x
    By = B0 * np.exp(-x**2 / (2.0 * sigma0**2))

    # Diffusion coefficient array: D = eta / mu_0
    coeff = np.full(nx, D)

    # Time integration: enough steps to resolve temporal discretization.
    # Crank-Nicolson is 2nd order in time, so we need dt ~ O(dx^2).
    n_steps = max(50, nx * 2)
    dt = t_end / n_steps

    for _ in range(n_steps):
        By = diffuse_field_1d(By, coeff, dt, dx)

    # Analytical solution
    By_ana = gaussian_B_analytical(x, t_end, B0, sigma0, D)

    # Relative L2 error
    err = np.sqrt(np.mean((By - By_ana)**2))
    norm = np.sqrt(np.mean(By_ana**2))
    if norm > 1e-30:
        err /= norm

    return float(err)


# ============================================================
# Runner for STS (RKL2) method
# ============================================================

def _run_sts(
    nx: int,
    L: float,
    B0: float,
    sigma0: float,
    eta: float,
    t_end: float,
) -> float:
    """Run the RKL2 super time-stepping diffusion solver and return L2 error.

    Uses the 1D RKL2 step function directly on x-direction data to
    avoid transverse artifacts from the 3D ADI approach.

    Args:
        nx: Number of cells in x.
        L: Half-domain size [m].
        B0: Peak field amplitude [T].
        sigma0: Initial Gaussian width [m].
        eta: Resistivity [Ohm*m].
        t_end: Integration time [s].

    Returns:
        L2 error between numerical and analytical solution.
    """
    from dpf.fluid.super_time_step import rkl2_diffusion_step

    dx = 2.0 * L / nx
    x = np.linspace(-L + 0.5 * dx, L - 0.5 * dx, nx)
    D = eta / mu_0

    # Initial condition: B_y Gaussian in x
    By = B0 * np.exp(-x**2 / (2.0 * sigma0**2))

    # RKL2 with s stages extends the CFL by factor ~0.275*s^2.
    # Explicit CFL: dt_exp < dx^2 / (2*D).
    s_stages = 8
    cfl_factor = 0.275 * s_stages**2
    dt_exp = dx**2 / (2.0 * D)
    dt_sts = 0.8 * cfl_factor * dt_exp
    n_steps = max(10, int(np.ceil(t_end / dt_sts)))
    dt = t_end / n_steps

    for _ in range(n_steps):
        By = rkl2_diffusion_step(By, D, dt, dx, s_stages=s_stages)

    # Analytical solution
    By_ana = gaussian_B_analytical(x, t_end, B0, sigma0, D)

    # Relative L2 error
    err = np.sqrt(np.mean((By - By_ana)**2))
    norm = np.sqrt(np.mean(By_ana**2))
    if norm > 1e-30:
        err /= norm

    return float(err)


# ============================================================
# Runner for explicit MHD solver method
# ============================================================

def _run_explicit(
    nx: int,
    L: float,
    B0: float,
    sigma0: float,
    eta: float,
    t_end: float,
) -> float:
    """Run the full MHD solver in pure-diffusion mode and return L2 error.

    Sets velocity to zero and uses a uniform, high pressure so the MHD
    advection fluxes produce negligible changes; only the resistive
    induction term drives the evolution of B.

    Args:
        nx: Number of cells in x.
        L: Half-domain size [m].
        B0: Peak field amplitude [T].
        sigma0: Initial Gaussian width [m].
        eta: Resistivity [Ohm*m].
        t_end: Integration time [s].

    Returns:
        L2 error between numerical and analytical solution.
    """
    from dpf.fluid.mhd_solver import MHDSolver

    dx = 2.0 * L / nx
    x = np.linspace(-L + 0.5 * dx, L - 0.5 * dx, nx)
    D = eta / mu_0

    ny, nz = _NY_PAD, _NZ_PAD
    grid_shape = (nx, ny, nz)

    # Uniform density and high pressure (suppress advection dynamics)
    rho0 = 1.0
    p0 = 1e6  # Very high pressure to keep sound speed >> diffusion speed
    ion_mass = 1.67e-27  # proton mass

    rho = np.full(grid_shape, rho0)
    velocity = np.zeros((3,) + grid_shape)
    pressure = np.full(grid_shape, p0)

    # B_y Gaussian in x-direction
    By_init = B0 * np.exp(-x**2 / (2.0 * sigma0**2))
    B = np.zeros((3,) + grid_shape)
    B[1] = np.broadcast_to(By_init[:, np.newaxis, np.newaxis], grid_shape).copy()

    # Temperature
    T0 = p0 * ion_mass / (2.0 * rho0 * k_B)
    Te = np.full(grid_shape, T0)
    Ti = np.full(grid_shape, T0)
    psi = np.zeros(grid_shape)

    state = {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }

    eta_field = np.full(grid_shape, eta)

    solver = MHDSolver(
        grid_shape=grid_shape,
        dx=dx,
        gamma=5.0 / 3.0,
        cfl=0.3,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=True,
        enable_energy_equation=False,
        ion_mass=ion_mass,
        riemann_solver="hll",
    )

    # Time integration
    # Diffusion CFL: dt < 0.25 * dx^2 / D
    t = 0.0
    max_steps = 2_000_000
    step_count = 0
    dt_diff = 0.4 * dx**2 / D

    while t < t_end and step_count < max_steps:
        dt = solver._compute_dt(state)
        dt = min(dt, dt_diff)
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break
        state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=eta_field)
        # Reset velocity to zero each step to suppress advection
        state["velocity"][:] = 0.0
        t += dt
        step_count += 1

    jm = ny // 2
    km = nz // 2
    By_num = state["B"][1, :, jm, km]

    By_ana = gaussian_B_analytical(x, t_end, B0, sigma0, D)

    err = np.sqrt(np.mean((By_num - By_ana)**2))
    norm = np.sqrt(np.mean(By_ana**2))
    if norm > 1e-30:
        err /= norm

    return float(err)


# ============================================================
# Public API
# ============================================================

def run_diffusion_convergence(
    method: str = "implicit",
    resolutions: list[int] | None = None,
) -> DiffusionConvergenceResult:
    """Run diffusion convergence study at multiple resolutions.

    Sets up a 1D Gaussian B-field diffusion problem and measures the L2
    error against the analytical solution at each resolution.  The
    convergence order is estimated from the log-log slope of (N, error).

    Args:
        method: Diffusion method to test.  One of ``"explicit"``,
            ``"implicit"``, or ``"sts"``.
        resolutions: List of grid sizes to test.
            Defaults to ``[32, 64, 128, 256]``.

    Returns:
        :class:`DiffusionConvergenceResult` with resolutions, errors,
        and estimated convergence order.

    Raises:
        ValueError: If *method* is not one of the supported values.
    """
    if resolutions is None:
        resolutions = [32, 64, 128, 256]

    if method not in ("explicit", "implicit", "sts"):
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'explicit', 'implicit', 'sts'."
        )

    # Physical parameters
    # Choose D (magnetic diffusivity) to be moderate so that:
    #   1. t_end is long enough for time integration to be accurate
    #   2. Spatial resolution dominates the error (not time discretization)
    #   3. The Gaussian stays well within the domain boundaries
    D_target = 0.1   # Target diffusivity [m^2/s]
    B0 = 1.0         # Peak field [T]
    sigma0 = 0.05    # Initial Gaussian width [m]
    eta = D_target * mu_0  # Resistivity [Ohm*m]
    D = D_target
    # Run for 0.5 diffusion times: sigma grows from sigma0 to sqrt(2)*sigma0
    t_end = 0.5 * sigma0**2 / D  # 0.0125 s
    # Large domain to eliminate boundary effects:
    # At t_end, sigma(t) = sqrt(1.5)*sigma0 ~ 0.061 m
    # L = 10*sigma0 = 0.5 m => boundary at 0.5/0.061 ~ 8 sigmas
    L = 10.0 * sigma0  # Domain half-width: 10 sigma0 = 0.5 m

    logger.info(
        "Diffusion convergence test: method=%s, eta=%.2e, D=%.2e, "
        "sigma0=%.2e, t_end=%.2e, L=%.2e, resolutions=%s",
        method, eta, D, sigma0, t_end, L, resolutions,
    )

    runner = {
        "explicit": _run_explicit,
        "implicit": _run_implicit,
        "sts": _run_sts,
    }[method]

    errors: list[float] = []

    for nx in resolutions:
        err = runner(nx, L, B0, sigma0, eta, t_end)
        errors.append(err)
        logger.info("  nx=%4d  L2_error=%.6e", nx, err)

    order = _estimate_convergence_order(resolutions, errors)
    logger.info("Estimated convergence order: %.2f", order)

    return DiffusionConvergenceResult(
        method=method,
        resolutions=resolutions,
        errors=errors,
        convergence_order=order,
        eta=eta,
        sigma0=sigma0,
        t_end=t_end,
    )
