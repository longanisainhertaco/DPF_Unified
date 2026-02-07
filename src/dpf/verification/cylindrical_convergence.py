"""Radial equilibrium convergence test for the cylindrical MHD solver.

Verifies that the solver correctly maintains a known static equilibrium
-- the Bennett z-pinch -- and that truncation errors converge at the
expected rate as grid resolution increases.

Z-pinch (Bennett) equilibrium
------------------------------
A cylindrical plasma column carries uniform axial current density J_z = J0
inside radius *a*.  Ampere's law gives the azimuthal magnetic field:

    B_theta(r) = mu_0 * J0 * r / 2          for r <= a
    B_theta(r) = mu_0 * J0 * a^2 / (2 * r)  for r >  a

Radial force balance (dp/dr + J_z * B_theta = 0) yields a parabolic
pressure profile inside the column:

    p(r) = p_axis - mu_0 * J0^2 * r^2 / 4   for r <= a
    p(r) = p_edge                              for r >  a

with p_edge = p_axis - mu_0 * J0^2 * a^2 / 4.

The density is uniform (rho = rho0), velocity is zero, and B_r = B_z = 0.
For an ideal MHD solver this configuration is a stationary solution; any
departure from it is numerical error that should diminish with resolution.

For the convergence test the column radius *a* is chosen larger than the
domain radius so the entire grid sits inside the smooth parabolic region,
avoiding the kink in dB_theta/dr at r = a that would otherwise limit
convergence to first order.

Each resolution is advanced for the same number of CFL-limited timesteps.
Because dt ~ dr at each resolution, finer grids advance a shorter physical
time, but the per-step truncation error decreases as O(dr^2) (SSP-RK2 +
second-order spatial operators).  The accumulated error over a fixed step
count therefore also decreases as O(dr^2), demonstrating second-order
convergence.

Usage::

    from dpf.verification.cylindrical_convergence import run_convergence_test

    results = run_convergence_test()
    print(f"Convergence order: {results['convergence_order']:.2f}")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dpf.constants import k_B, mu_0

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Equilibrium setup
# ──────────────────────────────────────────────────────────────────────

def setup_zpinch_equilibrium(
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    J0: float,
    a: float,
    p_axis: float,
    rho0: float,
    ion_mass: float,
) -> dict[str, np.ndarray]:
    """Create the initial state for a Bennett z-pinch equilibrium.

    Constructs a radially-resolved equilibrium on a cell-centred
    cylindrical grid with ``ny = 1`` (axisymmetric).  All arrays are
    returned in the 3-D shape ``(nr, 1, nz)`` expected by
    :class:`~dpf.fluid.cylindrical_mhd.CylindricalMHDSolver`.

    Parameters
    ----------
    nr : int
        Number of radial cells.
    nz : int
        Number of axial cells.
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    J0 : float
        Uniform axial current density inside the pinch column [A/m^2].
    a : float
        Pinch column radius [m].  For a smooth convergence test, set
        ``a >= nr * dr`` so the entire domain is inside the column.
        When ``a < nr * dr``, the exterior vacuum profile is also
        initialised correctly.
    p_axis : float
        On-axis thermal pressure [Pa].  Must be large enough that
        pressure stays positive everywhere:
        ``p_axis > mu_0 * J0^2 * min(a, R_max)^2 / 4``.
    rho0 : float
        Uniform mass density [kg/m^3].
    ion_mass : float
        Ion mass [kg] (used for temperature computation).

    Returns
    -------
    dict[str, np.ndarray]
        State dictionary with keys ``rho``, ``velocity``, ``pressure``,
        ``B``, ``Te``, ``Ti``, ``psi`` in the ``(nr, 1, nz)`` layout.

    Raises
    ------
    ValueError
        If the on-axis pressure is too low to keep p > 0 everywhere.
    """
    R_max = nr * dr

    # Maximum radius that the parabolic profile extends to
    r_eff = min(a, R_max)
    p_min = p_axis - mu_0 * J0**2 * r_eff**2 / 4.0
    if p_min <= 0:
        raise ValueError(
            f"On-axis pressure p_axis={p_axis:.3e} Pa is too low: "
            f"minimum pressure would be {p_min:.3e} Pa at r={r_eff:.3e} m.  "
            f"Need p_axis > {mu_0 * J0**2 * r_eff**2 / 4.0:.3e} Pa."
        )

    # Cell-centred radial coordinates: r[i] = (i + 0.5) * dr
    r = (np.arange(nr) + 0.5) * dr

    # ── Magnetic field: only B_theta is non-zero ──
    B_theta = np.where(
        r <= a,
        mu_0 * J0 * r / 2.0,
        mu_0 * J0 * a**2 / (2.0 * np.maximum(r, 1e-30)),
    )

    # ── Pressure profile ──
    p_edge = p_axis - mu_0 * J0**2 * a**2 / 4.0
    pressure = np.where(
        r <= a,
        p_axis - mu_0 * J0**2 * r**2 / 4.0,
        np.maximum(p_edge, 1e-10),
    )
    pressure = np.maximum(pressure, 1e-10)

    # ── Build 3-D state arrays (nr, 1, nz) ──
    rho = np.full((nr, 1, nz), rho0)
    velocity = np.zeros((3, nr, 1, nz))

    pressure_3d = np.empty((nr, 1, nz))
    pressure_3d[:, 0, :] = pressure[:, np.newaxis]

    B = np.zeros((3, nr, 1, nz))
    B[1, :, 0, :] = B_theta[:, np.newaxis]  # B_theta component

    psi = np.zeros((nr, 1, nz))

    # ── Temperatures from p = 2 * n_i * k_B * T  (quasi-neutral: ne = ni) ──
    n_i = rho0 / ion_mass
    T = pressure_3d / np.maximum(2.0 * n_i * k_B, 1e-30)
    T = np.maximum(T, 1.0)  # Temperature floor [K]

    Te = T.copy()
    Ti = T.copy()

    return {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure_3d,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }


# ──────────────────────────────────────────────────────────────────────
# Error diagnostics
# ──────────────────────────────────────────────────────────────────────

def check_equilibrium_preservation(
    state_initial: dict[str, np.ndarray],
    state_final: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compare initial and final states to quantify departure from equilibrium.

    All error norms are normalised so that they are independent of the
    number of grid cells (the L2 norms are root-mean-square, i.e.
    divided by sqrt(N)).

    Parameters
    ----------
    state_initial : dict[str, np.ndarray]
        State dictionary at t = 0.
    state_final : dict[str, np.ndarray]
        State dictionary at t = t_end.

    Returns
    -------
    dict[str, float]
        Diagnostic error metrics:

        - **pressure_error_L2** -- relative RMS of the pressure change,
          ``rms(p_final - p_initial) / rms(p_initial)``.
        - **velocity_error_L2** -- RMS of the radial velocity [m/s]
          (should be close to zero for a maintained equilibrium).
        - **Btheta_error_L2** -- relative RMS of the B_theta change.
        - **max_rho_change** -- maximum relative density change over the
          entire grid.
    """
    def _rms(x: np.ndarray) -> float:
        """Root-mean-square of a flat array."""
        return float(np.sqrt(np.mean(x**2)))

    p_i = state_initial["pressure"].ravel()
    p_f = state_final["pressure"].ravel()
    p_rms = _rms(p_i)
    pressure_error_L2 = _rms(p_f - p_i) / p_rms if p_rms > 0 else 0.0

    # Radial velocity: component 0
    vr_f = state_final["velocity"][0].ravel()
    velocity_error_L2 = _rms(vr_f)

    # B_theta: component 1
    Bt_i = state_initial["B"][1].ravel()
    Bt_f = state_final["B"][1].ravel()
    Bt_rms = _rms(Bt_i)
    Btheta_error_L2 = _rms(Bt_f - Bt_i) / Bt_rms if Bt_rms > 0 else 0.0

    # Density
    rho_i = state_initial["rho"].ravel()
    rho_f = state_final["rho"].ravel()
    rel_drho = np.abs(rho_f - rho_i) / np.maximum(rho_i, 1e-30)
    max_rho_change = float(np.max(rel_drho))

    return {
        "pressure_error_L2": float(pressure_error_L2),
        "velocity_error_L2": float(velocity_error_L2),
        "Btheta_error_L2": float(Btheta_error_L2),
        "max_rho_change": float(max_rho_change),
    }


# ──────────────────────────────────────────────────────────────────────
# Convergence study driver
# ──────────────────────────────────────────────────────────────────────

def run_convergence_test(
    resolutions: list[int] | None = None,
    t_end: float | None = None,
    J0: float = 1e6,
    a_frac: float = 2.0,
    n_steps: int = 3,
) -> dict[str, Any]:
    """Run the z-pinch equilibrium at multiple resolutions and measure convergence.

    For each resolution the solver is initialised with the analytic
    Bennett equilibrium and advanced for *n_steps* CFL-limited timesteps.
    Because the CFL timestep scales linearly with the grid spacing, each
    resolution reaches a different physical time -- but the *per-step*
    truncation error of the SSP-RK2 time integrator combined with the
    second-order spatial discretisation decreases as O(dr^2), so the
    total error after a fixed number of steps also decreases as O(dr^2).

    If *t_end* is given instead, all resolutions are advanced to the same
    physical time (the finest grid will take proportionally more steps).
    This can be slower and may require a short integration time to avoid
    nonlinear error growth.

    By default ``a_frac = 2.0`` so the column radius is twice the domain
    radius.  This keeps the entire computational domain inside the smooth
    parabolic region of the equilibrium, avoiding the kink at r = a that
    would otherwise pollute the convergence rate.

    Parameters
    ----------
    resolutions : list[int] | None
        List of grid sizes; ``nr = nz = res`` for each entry.
        Defaults to ``[32, 64, 128]``.
    t_end : float | None
        Physical end time [s].  If *None*, the solver is advanced for
        exactly *n_steps* CFL-limited steps at each resolution.  If
        given, the solver advances until t = t_end (taking as many CFL
        steps as needed, up to a hard cap of ``8 * n_steps * max(resolutions)``
        to prevent runaway).
    J0 : float
        Axial current density [A/m^2].  Default ``1e6`` gives a
        high-beta equilibrium that is well-resolved even at 32 cells.
    a_frac : float
        Column radius as a fraction of the domain radius:
        ``a = a_frac * R_domain``.  Default ``2.0`` ensures no kink
        lies within the grid.
    n_steps : int
        Number of CFL-limited solver timesteps per resolution (when
        *t_end* is None).  Default 3.

    Returns
    -------
    dict[str, Any]
        - **resolutions** -- list of grid sizes tested.
        - **pressure_errors** -- pressure RMS errors at each resolution.
        - **velocity_errors** -- radial-velocity RMS errors at each resolution.
        - **Btheta_errors** -- B_theta RMS errors at each resolution.
        - **convergence_order** -- estimated convergence order (slope
          of log(error) vs log(1/N)).  Positive means error decreases
          with resolution.
        - **details** -- per-resolution ``check_equilibrium_preservation``
          results.
    """
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    if resolutions is None:
        resolutions = [32, 64, 128]

    # Fixed physical domain
    R_domain = 0.05   # 5 cm outer radius
    L_domain = 0.05   # 5 cm axial length

    # Physics parameters
    gamma = 5.0 / 3.0
    ion_mass = 3.34e-27   # deuterium [kg]
    rho0 = 1e-3           # kg/m^3  (moderate fill density)

    # Column radius: well outside domain for smooth interior
    a = a_frac * R_domain

    # Magnetic pressure scale at the domain edge (r = R_domain, inside column)
    p_mag_edge = mu_0 * J0**2 * R_domain**2 / 4.0

    # High-beta regime: p_axis >> p_mag so that the equilibrium perturbation
    # is small relative to the background pressure.  This keeps errors
    # manageable and the solver stable.
    p_axis = 100.0 * p_mag_edge + 1.0  # +1.0 guards against J0=0

    # Sound speed estimate for logging
    cs_est = np.sqrt(gamma * p_axis / rho0)

    logger.info(
        "Z-pinch convergence test: J0=%.2e A/m^2, a=%.3e m, p_axis=%.3e Pa, "
        "rho0=%.2e kg/m^3, cs=%.2e m/s, n_steps=%d, t_end=%s",
        J0, a, p_axis, rho0, cs_est, n_steps,
        f"{t_end:.3e}" if t_end is not None else "None (fixed-step mode)",
    )

    pressure_errors: list[float] = []
    velocity_errors: list[float] = []
    Btheta_errors: list[float] = []
    details: list[dict[str, float]] = []

    for res in resolutions:
        nr = nz = res
        dr = R_domain / nr
        dz = L_domain / nz

        # Build equilibrium
        state0 = setup_zpinch_equilibrium(
            nr=nr, nz=nz, dr=dr, dz=dz,
            J0=J0, a=a, p_axis=p_axis, rho0=rho0, ion_mass=ion_mass,
        )

        # Deep copy for comparison
        state_initial = {k: v.copy() for k, v in state0.items()}

        # Create solver (pure ideal MHD: no Hall, no resistivity)
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=gamma,
            cfl=0.4,
            dedner_ch=0.0,
            enable_hall=False,
            enable_resistive=False,
            enable_energy_equation=True,
            ion_mass=ion_mass,
            riemann_solver="hll",
        )

        # Advance the solver
        state = state0
        t = 0.0
        steps_taken = 0

        if t_end is None:
            # Fixed-step mode: take exactly n_steps CFL steps
            for _ in range(n_steps):
                dt = solver._compute_dt(state)
                state = solver.step(state, dt, current=0.0, voltage=0.0)
                t += dt
                steps_taken += 1
        else:
            # Fixed-time mode: advance to t_end
            max_iter = 8 * n_steps * max(resolutions)
            for _ in range(max_iter):
                dt = solver._compute_dt(state)
                remaining = t_end - t
                if remaining <= 0:
                    break
                dt = min(dt, remaining)
                if dt <= 0:
                    break
                state = solver.step(state, dt, current=0.0, voltage=0.0)
                t += dt
                steps_taken += 1

        # Measure error
        diag = check_equilibrium_preservation(state_initial, state)
        pressure_errors.append(diag["pressure_error_L2"])
        velocity_errors.append(diag["velocity_error_L2"])
        Btheta_errors.append(diag["Btheta_error_L2"])
        details.append(diag)

        logger.info(
            "  res=%3d  steps=%3d  t_final=%.3e s  "
            "p_err=%.3e  vr_rms=%.3e  Bt_err=%.3e  drho_max=%.3e",
            res, steps_taken, t,
            diag["pressure_error_L2"],
            diag["velocity_error_L2"],
            diag["Btheta_error_L2"],
            diag["max_rho_change"],
        )

    # ── Estimate convergence order from B_theta error ──
    # B_theta is the cleanest indicator for this equilibrium because it
    # depends only on the discrete curl operator's accuracy and is not
    # affected by the geometric source terms that complicate pressure
    # convergence at low resolution.  Pressure convergence is also
    # returned for reference.
    convergence_order = _estimate_convergence_order(resolutions, Btheta_errors)

    return {
        "resolutions": resolutions,
        "pressure_errors": pressure_errors,
        "velocity_errors": velocity_errors,
        "Btheta_errors": Btheta_errors,
        "convergence_order": convergence_order,
        "details": details,
    }


def _estimate_convergence_order(
    resolutions: list[int],
    errors: list[float],
) -> float:
    """Estimate the order of convergence from (resolution, error) pairs.

    Uses a least-squares fit in log-log space:
    ``log(error) ~ -order * log(N) + const``.

    If any error is zero or negative, or if there are fewer than two
    usable data points, returns ``0.0``.

    Parameters
    ----------
    resolutions : list[int]
        Grid sizes (N values).
    errors : list[float]
        Corresponding error norms.

    Returns
    -------
    float
        Estimated convergence order (positive means error decreases with
        increasing resolution).
    """
    if len(resolutions) < 2:
        return 0.0

    # Filter out zero / negative / NaN errors
    log_N: list[float] = []
    log_e: list[float] = []
    for N, err in zip(resolutions, errors, strict=False):
        if np.isfinite(err) and err > 0 and N > 0:
            log_N.append(np.log(float(N)))
            log_e.append(np.log(err))

    if len(log_N) < 2:
        return 0.0

    # Least-squares: log_e = slope * log_N + intercept
    log_N_arr = np.array(log_N)
    log_e_arr = np.array(log_e)
    A = np.vstack([log_N_arr, np.ones(len(log_N_arr))]).T
    result = np.linalg.lstsq(A, log_e_arr, rcond=None)
    slope = result[0][0]

    # Convergence order is the negative of the slope:
    #   error ~ N^{-order}  =>  log(e) = -order * log(N) + const
    return float(-slope)
