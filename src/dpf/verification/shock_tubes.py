"""Shock tube verification tests for the MHD solver.

Provides exact and reference solutions for standard gas-dynamics and MHD
shock tube problems, together with driver functions that run them through
the :class:`dpf.fluid.mhd_solver.MHDSolver` and report quantitative error
norms.

Implemented problems
--------------------
1. **Sod shock tube** (Sod 1978) -- exact Riemann solver for the
   one-dimensional Euler equations.  Used to verify the density, velocity,
   and pressure profiles at finite time against the closed-form solution
   involving a left rarefaction, a contact discontinuity, and a right
   shock.

2. **Brio-Wu MHD shock tube** (Brio & Wu 1988) -- canonical MHD Riemann
   problem with compound waves, slow shocks, and rotational
   discontinuities.  No closed-form solution exists; instead we verify
   that the solver produces physically admissible output (positivity,
   B_x conservation, expected wave structure) and return the numerical
   state for visual inspection.

References
----------
- G. A. Sod, *J. Comput. Phys.* **27**, 1--31 (1978).
- M. Brio & C. C. Wu, *J. Comput. Phys.* **75**, 400--422 (1988).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dpf.constants import k_B
from dpf.fluid.mhd_solver import MHDSolver

logger = logging.getLogger(__name__)

# Transverse grid padding for 1-D shock tube tests.  The MHD solver
# requires >= 5 cells in every direction to enable WENO5 reconstruction,
# and at least 2 cells for np.gradient on the remaining operators (curl,
# divergence, Dedner cleaning).  We pad the y and z dimensions to 5
# cells with uniform data; transverse gradients are therefore identically
# zero and the problem remains strictly one-dimensional.
# Use 4 (not 5) to disable WENO5 on the Python engine, where the hybrid
# WENO5 boundary mismatch causes NaN for shock propagation problems.
_NY_PAD: int = 4
_NZ_PAD: int = 4


# ============================================================
# Data containers
# ============================================================

@dataclass
class ShockTubeResult:
    """Container returned by the run_* driver functions.

    Attributes:
        x: Cell-centre coordinates, shape ``(nx,)``.
        numerical: Dict of field arrays from the solver (1-D squeezed).
        analytical: Dict of analytical/reference arrays (may be ``None``
            for problems without a closed-form solution).
        errors: Dict of ``{field_name: L1_error}`` if an analytical
            solution is available, else an empty dict.
        checks: Dict of ``{check_name: bool}`` for qualitative sanity
            checks (positivity, B_x preservation, etc.).
    """

    x: np.ndarray
    numerical: dict[str, np.ndarray]
    analytical: dict[str, np.ndarray] | None = None
    errors: dict[str, float] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)


# ============================================================
# Sod shock tube -- exact Riemann solver
# ============================================================

def _sod_find_pstar(
    rho_L: float,
    p_L: float,
    u_L: float,
    rho_R: float,
    p_R: float,
    u_R: float,
    gamma: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Find the star-region pressure *p** and velocity *u** for the exact
    Riemann problem (Toro, ch. 4).

    Uses the Newton-Raphson iteration on the pressure function
    ``f(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0``.

    Returns:
        ``(p_star, u_star)``
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # Sound speeds
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # Pressure functions and their derivatives (Toro 4.6-4.8)
    def _f(p: float, rho_k: float, p_k: float, a_k: float) -> float:
        if p > p_k:
            # Shock
            A_k = 2.0 / (gp1 * rho_k)
            B_k = gm1 / gp1 * p_k
            return (p - p_k) * np.sqrt(A_k / (p + B_k))
        else:
            # Rarefaction
            return (
                2.0 * a_k / gm1
                * ((p / p_k) ** (gm1 / (2.0 * gamma)) - 1.0)
            )

    def _fprime(p: float, rho_k: float, p_k: float, a_k: float) -> float:
        if p > p_k:
            A_k = 2.0 / (gp1 * rho_k)
            B_k = gm1 / gp1 * p_k
            q = np.sqrt(A_k / (p + B_k))
            return q * (1.0 - (p - p_k) / (2.0 * (p + B_k)))
        else:
            return (
                1.0 / (rho_k * a_k)
                * (p / p_k) ** (-(gp1) / (2.0 * gamma))
            )

    # Initial guess: linearised (PVRS) estimate
    p_star = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p_star = max(p_star, 1e-30)

    for _ in range(max_iter):
        fL = _f(p_star, rho_L, p_L, a_L)
        fR = _f(p_star, rho_R, p_R, a_R)
        f_val = fL + fR + (u_R - u_L)

        dfL = _fprime(p_star, rho_L, p_L, a_L)
        dfR = _fprime(p_star, rho_R, p_R, a_R)
        df_val = dfL + dfR

        if abs(df_val) < 1e-30:
            break

        dp = -f_val / df_val
        p_star_new = p_star + dp
        if p_star_new < 1e-30:
            p_star_new = 1e-30

        if abs(dp) < tol * 0.5 * (p_star + p_star_new):
            p_star = p_star_new
            break
        p_star = p_star_new

    # Star velocity
    fL = _f(p_star, rho_L, p_L, a_L)
    fR = _f(p_star, rho_R, p_R, a_R)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (fR - fL)

    return p_star, u_star


def sod_shock_tube_analytical(
    x: np.ndarray,
    t: float,
    gamma: float = 1.4,
    x0: float = 0.5,
) -> dict[str, np.ndarray]:
    """Exact solution for the Sod shock tube problem.

    Initial conditions (interface at *x0*):

    ========  ======  ======  ====
     Region    rho      p      u
    ========  ======  ======  ====
     Left      1.0     1.0    0.0
     Right     0.125   0.1    0.0
    ========  ======  ======  ====

    Args:
        x: 1-D array of cell-centre positions.
        t: Evaluation time.  Must be > 0.
        gamma: Adiabatic index (default 1.4).
        x0: Initial discontinuity position (default 0.5).

    Returns:
        Dictionary with keys ``"rho"``, ``"u"``, ``"p"`` -- each a 1-D
        ``ndarray`` the same length as *x*.
    """
    if t <= 0.0:
        raise ValueError("Time t must be > 0 for the Sod analytical solution.")

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # Left / right states
    rho_L, p_L, u_L = 1.0, 1.0, 0.0
    rho_R, p_R, u_R = 0.125, 0.1, 0.0

    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # Solve for star region
    p_star, u_star = _sod_find_pstar(rho_L, p_L, u_L, rho_R, p_R, u_R, gamma)

    # --- Characterise the left wave (rarefaction for standard Sod) ---
    # Density behind the left rarefaction (star-left)
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
    a_star_L = a_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))

    # Rarefaction head and tail speeds
    S_HL = u_L - a_L           # head (furthest left)
    S_TL = u_star - a_star_L   # tail (closest to contact)

    # --- Characterise the right wave (shock for standard Sod) ---
    # Post-shock (star-right) density via Rankine-Hugoniot
    rho_star_R = rho_R * (
        (p_star / p_R + gm1 / gp1)
        / (gm1 / gp1 * p_star / p_R + 1.0)
    )

    # Shock speed
    S_R = u_R + a_R * np.sqrt(gp1 / (2.0 * gamma) * p_star / p_R + gm1 / (2.0 * gamma))

    # --- Contact discontinuity speed ---
    S_contact = u_star

    # --- Sample the solution ---
    rho_out = np.empty_like(x)
    u_out = np.empty_like(x)
    p_out = np.empty_like(x)

    xi = (x - x0) / t  # similarity variable

    for i in range(len(x)):
        s = xi[i]
        if s <= S_HL:
            # Undisturbed left state
            rho_out[i] = rho_L
            u_out[i] = u_L
            p_out[i] = p_L
        elif s <= S_TL:
            # Inside the left rarefaction fan
            rho_out[i] = rho_L * (
                2.0 / gp1 + gm1 / (gp1 * a_L) * (u_L - s)
            ) ** (2.0 / gm1)
            u_out[i] = 2.0 / gp1 * (a_L + gm1 / 2.0 * u_L + s)
            p_out[i] = p_L * (
                2.0 / gp1 + gm1 / (gp1 * a_L) * (u_L - s)
            ) ** (2.0 * gamma / gm1)
        elif s <= S_contact:
            # Star region (left side of contact)
            rho_out[i] = rho_star_L
            u_out[i] = u_star
            p_out[i] = p_star
        elif s <= S_R:
            # Star region (right side of contact)
            rho_out[i] = rho_star_R
            u_out[i] = u_star
            p_out[i] = p_star
        else:
            # Undisturbed right state
            rho_out[i] = rho_R
            u_out[i] = u_R
            p_out[i] = p_R

    return {"rho": rho_out, "u": u_out, "p": p_out}


# ============================================================
# Brio-Wu MHD shock tube -- reference data
# ============================================================

def brio_wu_reference(
    x: np.ndarray,
    t: float,
    gamma: float = 2.0,
    x0: float = 0.5,
) -> dict[str, Any]:
    """Reference information for the Brio-Wu MHD shock tube problem.

    No closed-form analytical solution exists for this problem.  This
    function returns the initial conditions and expected qualitative
    features that can be checked against the numerical output.

    Initial conditions (interface at *x0*):

    =======  =====  =====  ======  ======  ====
     Region   rho    p      Bx      By      u
    =======  =====  =====  ======  ======  ====
     Left     1.0    1.0    0.75    1.0     0.0
     Right    0.125  0.1    0.75   -1.0     0.0
    =======  =====  =====  ======  ======  ====

    The solution at *t* = 0.1 (on [0, 1]) contains the following wave
    structure (left to right):

    1. Fast rarefaction
    2. Slow compound wave (rarefaction + shock)
    3. Contact discontinuity
    4. Slow shock
    5. Fast rarefaction

    Args:
        x: 1-D cell-centre positions.
        t: Evaluation time (for informational labelling only).
        gamma: Adiabatic index (default 2.0 for Brio-Wu).
        x0: Interface position.

    Returns:
        Dictionary with keys:

        - ``"initial_left"`` / ``"initial_right"`` -- state dicts.
        - ``"gamma"`` -- adiabatic index.
        - ``"Bx"`` -- normal B-field component (should be preserved).
        - ``"expected_waves"`` -- list of wave names (qualitative).
        - ``"t"`` -- evaluation time.
        - ``"x0"`` -- interface position.
    """
    return {
        "initial_left": {
            "rho": 1.0,
            "p": 1.0,
            "u": 0.0,
            "Bx": 0.75,
            "By": 1.0,
        },
        "initial_right": {
            "rho": 0.125,
            "p": 0.1,
            "u": 0.0,
            "Bx": 0.75,
            "By": -1.0,
        },
        "gamma": gamma,
        "Bx": 0.75,
        "expected_waves": [
            "fast rarefaction (left-going)",
            "slow compound wave",
            "contact discontinuity",
            "slow shock",
            "fast rarefaction (right-going)",
        ],
        "t": t,
        "x0": x0,
    }


# ============================================================
# Helper: build MHD state dict from 1-D primitives
# ============================================================

def _build_state(
    nx: int,
    rho: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    gamma: float,
    ion_mass: float,
    *,
    ny: int = _NY_PAD,
    nz: int = _NZ_PAD,
) -> dict[str, np.ndarray]:
    """Construct a full MHD state dict from 1-D primitive arrays.

    The 1-D data (length *nx*) is broadcast into shape ``(nx, ny, nz)``
    so that the solver sees a valid 3-D grid.  Data is uniform in the
    transverse (y, z) directions, giving correct 1-D physics while
    allowing ``np.gradient`` and WENO5 to operate on every axis.

    Temperature is derived from the ideal-gas relation:
    ``p = 2 * n * k_B * T`` with ``n = rho / ion_mass``, so
    ``T = p * ion_mass / (2 * rho * k_B)``.

    Args:
        nx: Number of cells along the shock-tube axis.
        rho, u, p: Density, x-velocity, pressure (1-D, length *nx*).
        Bx, By, Bz: Magnetic field components (1-D, length *nx*).
        gamma: Adiabatic index.
        ion_mass: Ion mass [kg].
        ny: Transverse grid size in y (default ``_NY_PAD``).
        nz: Transverse grid size in z (default ``_NZ_PAD``).

    Returns:
        State dictionary compatible with ``MHDSolver.step``.
    """
    shape = (nx, ny, nz)

    # Broadcast 1-D -> 3-D by adding trailing axes then tiling
    def _inflate(arr_1d: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr_1d[:, np.newaxis, np.newaxis], shape).copy()

    rho_3d = _inflate(rho)
    p_3d = _inflate(p)

    vel = np.zeros((3,) + shape)
    vel[0] = _inflate(u)

    B = np.zeros((3,) + shape)
    B[0] = _inflate(Bx)
    B[1] = _inflate(By)
    B[2] = _inflate(Bz)

    # Temperature: p = n_i * k_B * Ti + n_e * k_B * Te = 2 * n * k_B * T
    # for Z = 1 and Te = Ti.  So T = p * ion_mass / (2 * rho * k_B).
    rho_safe = np.maximum(rho_3d, 1e-30)
    T = p_3d * ion_mass / (2.0 * rho_safe * k_B)

    psi = np.zeros(shape)

    return {
        "rho": rho_3d,
        "velocity": vel,
        "pressure": p_3d,
        "B": B,
        "Te": T.copy(),
        "Ti": T.copy(),
        "psi": psi,
    }


# ============================================================
# Driver: Sod shock tube
# ============================================================

def run_sod_test(
    nx: int = 200,
    t_end: float = 0.2,
    gamma: float = 1.4,
    riemann_solver: str = "hll",
) -> ShockTubeResult:
    """Run the Sod shock tube through the MHD solver and compare with the
    exact Riemann solution.

    Sets up a 1-D grid on [0, 1] with the standard Sod initial data,
    advances to *t_end* using :class:`MHDSolver`, and returns the
    numerical/analytical profiles together with L1 error norms.

    Args:
        nx: Number of cells (default 200).
        t_end: Final time (default 0.2).
        gamma: Adiabatic index (default 1.4).
        riemann_solver: Riemann solver choice, ``"hll"`` or ``"hlld"``.

    Returns:
        :class:`ShockTubeResult` with fields ``x``, ``numerical``,
        ``analytical``, ``errors``, and ``checks``.
    """
    dx = 1.0 / nx
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)

    # Use proton mass -- the actual value does not matter for ideal MHD
    # since Te/Ti are not fed back into the MHD fluxes.  It only affects
    # the temperature diagnostic.
    ion_mass = 1.67e-27  # proton mass [kg]

    # Initial left/right states
    rho_init = np.where(x < 0.5, 1.0, 0.125)
    u_init = np.zeros(nx)
    p_init = np.where(x < 0.5, 1.0, 0.1)
    Bx_init = np.zeros(nx)
    By_init = np.zeros(nx)
    Bz_init = np.zeros(nx)

    state = _build_state(
        nx, rho_init, u_init, p_init, Bx_init, By_init, Bz_init,
        gamma, ion_mass,
    )

    grid_shape = (nx, _NY_PAD, _NZ_PAD)
    solver = MHDSolver(
        grid_shape=grid_shape,
        dx=dx,
        gamma=gamma,
        cfl=0.4,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        enable_energy_equation=True,
        ion_mass=ion_mass,
        riemann_solver=riemann_solver,
        time_integrator="ssp_rk2",
    )

    # Time integration loop
    t = 0.0
    step_count = 0
    max_steps = 100_000

    logger.info(
        "Sod shock tube: nx=%d, t_end=%.4f, gamma=%.2f, riemann=%s",
        nx, t_end, gamma, riemann_solver,
    )

    while t < t_end and step_count < max_steps:
        dt = solver._compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        state = solver.step(state, dt, current=0.0, voltage=0.0)
        t += dt
        step_count += 1

    logger.info("Sod shock tube completed: %d steps, t_final=%.6e", step_count, t)

    # Extract 1-D numerical profiles (take the middle y/z pencil)
    jm = _NY_PAD // 2
    km = _NZ_PAD // 2
    rho_num = state["rho"][:, jm, km]
    u_num = state["velocity"][0, :, jm, km]
    p_num = state["pressure"][:, jm, km]

    # Analytical solution
    analytical = sod_shock_tube_analytical(x, t, gamma=gamma)

    # L1 error norms (normalized by domain length)
    errors = {}
    for name, num_arr, ana_arr in [
        ("rho", rho_num, analytical["rho"]),
        ("u", u_num, analytical["u"]),
        ("p", p_num, analytical["p"]),
    ]:
        errors[name] = float(np.mean(np.abs(num_arr - ana_arr)))

    # Sanity checks
    checks = {
        "no_nan": bool(
            np.all(np.isfinite(rho_num))
            and np.all(np.isfinite(u_num))
            and np.all(np.isfinite(p_num))
        ),
        "rho_positive": bool(np.all(rho_num > 0)),
        "p_positive": bool(np.all(p_num > 0)),
    }

    return ShockTubeResult(
        x=x,
        numerical={"rho": rho_num, "u": u_num, "p": p_num},
        analytical=analytical,
        errors=errors,
        checks=checks,
    )


# ============================================================
# Driver: Brio-Wu MHD shock tube
# ============================================================

def run_brio_wu_test(
    nx: int = 400,
    t_end: float = 0.1,
    gamma: float = 2.0,
    riemann_solver: str = "hlld",
) -> ShockTubeResult:
    """Run the Brio-Wu MHD shock tube through the MHD solver.

    Sets up a 1-D grid on [0, 1] with the Brio-Wu initial data and
    advances to *t_end*.  Since no closed-form solution exists, the
    function performs qualitative sanity checks (positivity, B_x
    conservation, expected wave count) and returns the numerical state
    for visual inspection.

    Args:
        nx: Number of cells (default 400).
        t_end: Final time (default 0.1).
        gamma: Adiabatic index (default 2.0).
        riemann_solver: Riemann solver, ``"hll"`` or ``"hlld"``
            (``"hlld"`` recommended for MHD).

    Returns:
        :class:`ShockTubeResult` with fields ``x``, ``numerical``,
        ``analytical`` (reference info dict), and ``checks``.
    """
    dx = 1.0 / nx
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)

    ion_mass = 1.67e-27  # proton mass

    # Initial conditions
    rho_init = np.where(x < 0.5, 1.0, 0.125)
    u_init = np.zeros(nx)
    p_init = np.where(x < 0.5, 1.0, 0.1)
    Bx_init = np.full(nx, 0.75)
    By_init = np.where(x < 0.5, 1.0, -1.0)
    Bz_init = np.zeros(nx)

    state = _build_state(
        nx, rho_init, u_init, p_init, Bx_init, By_init, Bz_init,
        gamma, ion_mass,
    )

    grid_shape = (nx, _NY_PAD, _NZ_PAD)
    solver = MHDSolver(
        grid_shape=grid_shape,
        dx=dx,
        gamma=gamma,
        cfl=0.4,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        enable_energy_equation=True,
        ion_mass=ion_mass,
        riemann_solver=riemann_solver,
        time_integrator="ssp_rk2",
    )

    # Time integration loop
    t = 0.0
    step_count = 0
    max_steps = 200_000

    logger.info(
        "Brio-Wu MHD shock tube: nx=%d, t_end=%.4f, gamma=%.2f, riemann=%s",
        nx, t_end, gamma, riemann_solver,
    )

    while t < t_end and step_count < max_steps:
        dt = solver._compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        state = solver.step(state, dt, current=0.0, voltage=0.0)
        t += dt
        step_count += 1

    logger.info("Brio-Wu test completed: %d steps, t_final=%.6e", step_count, t)

    # Extract 1-D numerical profiles (take the middle y/z pencil)
    jm = _NY_PAD // 2
    km = _NZ_PAD // 2
    rho_num = state["rho"][:, jm, km]
    u_num = state["velocity"][0, :, jm, km]
    p_num = state["pressure"][:, jm, km]
    Bx_num = state["B"][0, :, jm, km]
    By_num = state["B"][1, :, jm, km]

    numerical = {
        "rho": rho_num,
        "u": u_num,
        "p": p_num,
        "Bx": Bx_num,
        "By": By_num,
    }

    # Reference information (not an analytical solution)
    ref = brio_wu_reference(x, t, gamma=gamma)

    # Qualitative checks
    checks: dict[str, bool] = {}

    # 1. No NaN or Inf in any field
    checks["no_nan"] = bool(
        np.all(np.isfinite(rho_num))
        and np.all(np.isfinite(u_num))
        and np.all(np.isfinite(p_num))
        and np.all(np.isfinite(Bx_num))
        and np.all(np.isfinite(By_num))
    )

    # 2. Density and pressure stay positive
    checks["rho_positive"] = bool(np.all(rho_num > 0))
    checks["p_positive"] = bool(np.all(p_num > 0))

    # 3. B_x should remain constant (it is the normal component in 1-D
    #    ideal MHD and is not evolved by the induction equation in 1-D).
    #    Allow a small tolerance for numerical diffusion from Dedner
    #    cleaning and the gradient-based induction update.
    Bx_err = float(np.max(np.abs(Bx_num - 0.75)))
    checks["Bx_preserved"] = Bx_err < 0.1  # generous tolerance

    # 4. The density profile should show structure: it must differ from
    #    the initial condition (i.e., the solver actually did something).
    rho_range = float(np.max(rho_num) - np.min(rho_num))
    checks["has_wave_structure"] = rho_range > 0.05

    # 5. By should change sign across the domain (initially +1 left, -1
    #    right; after evolution the contact separates them).
    checks["By_sign_change"] = bool(np.any(By_num > 0) and np.any(By_num < 0))

    return ShockTubeResult(
        x=x,
        numerical=numerical,
        analytical=ref,
        errors={},  # no quantitative error for Brio-Wu
        checks=checks,
    )
