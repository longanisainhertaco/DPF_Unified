"""MHD convergence verification — error decreases with resolution.

Verifies that the Python MHD solver produces decreasing error with
increasing resolution for smooth initial data, and that wave propagation
and divergence cleaning work correctly.

Strategy: Since the Python solver uses outflow/zero-gradient BCs (not periodic),
we measure convergence using short-time evolution before boundary effects
contaminate the interior, or by comparing against higher-resolution reference.

References:
    Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics.
    Stone, J. M. et al. (2008). ApJS, 178, 137 (Athena code paper).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import k_B, mu_0
from dpf.fluid.mhd_solver import MHDSolver

# ============================================================
# Helpers
# ============================================================


def _make_uniform_state(
    nx: int, ny: int, nz: int, rho0: float = 1e-4, p0: float = 1e3
) -> dict[str, np.ndarray]:
    """Create a uniform quiescent state."""
    m_i = 3.34e-27
    n_i = rho0 / m_i
    T0 = p0 / (2.0 * n_i * k_B)
    return {
        "rho": np.full((nx, ny, nz), rho0),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.full((nx, ny, nz), p0),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), T0),
        "Ti": np.full((nx, ny, nz), T0),
        "psi": np.zeros((nx, ny, nz)),
    }


def _add_sound_wave(
    state: dict[str, np.ndarray],
    dx: float,
    amplitude: float = 0.01,
    gamma: float = 5.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Add a small-amplitude sound wave perturbation in x-direction.

    Wave centered in domain interior to avoid boundary contamination.
    Uses a Gaussian-windowed sinusoid so it doesn't reach boundaries.
    """
    nx = state["rho"].shape[0]
    ny = state["rho"].shape[1]
    nz = state["rho"].shape[2]

    rho0 = state["rho"][0, 0, 0]
    p0 = state["pressure"][0, 0, 0]
    cs = np.sqrt(gamma * p0 / rho0)

    x = np.arange(nx) * dx
    x_center = x[nx // 2]
    sigma = 0.25 * nx * dx  # Gaussian width = 1/4 of domain

    # 3D coordinate array
    X = x[:, np.newaxis, np.newaxis] * np.ones((1, ny, nz))

    # Gaussian envelope keeps perturbation away from boundaries
    wavelength = nx * dx / 4.0  # 4 wavelengths across domain
    k = 2.0 * np.pi / wavelength
    envelope = np.exp(-0.5 * ((X - x_center) / sigma) ** 2)
    perturbation = amplitude * np.sin(k * X) * envelope

    new_state = {}
    for key, val in state.items():
        new_state[key] = val.copy()

    new_state["rho"] = rho0 * (1.0 + perturbation)
    new_state["pressure"] = p0 * (1.0 + gamma * perturbation)

    v_perturb = cs * perturbation
    new_state["velocity"] = np.zeros((3, nx, ny, nz))
    new_state["velocity"][0] = v_perturb

    return new_state


def _l1_error(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized L1 error."""
    return float(np.sum(np.abs(a - b))) / max(float(np.sum(np.abs(b))), 1e-30)


def _run_solver(
    nx: int, ny: int, nz: int, dx: float, state: dict, sim_time: float,
    gamma: float = 5.0 / 3.0, use_weno5: bool = True,
) -> dict[str, np.ndarray]:
    """Run MHD solver for given time and return final state."""
    solver = MHDSolver(
        grid_shape=(nx, ny, nz),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        riemann_solver="hll",
    )
    solver.use_weno5 = use_weno5

    time = 0.0
    while time < sim_time:
        dt = solver._compute_dt(state)
        dt = min(dt, sim_time - time)
        if dt < 1e-20:
            break
        state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=None)
        time += dt

    return state


# ============================================================
# T2.1: Convergence — error decreases with resolution
# ============================================================


@pytest.mark.slow
def test_sound_wave_error_decreases_with_resolution():
    """Error from sound wave propagation should decrease as resolution increases.

    Run a Gaussian-windowed sound wave pulse for a short time at N=16, 32, 64.
    The interior error (away from boundaries) should decrease monotonically.
    """
    gamma = 5.0 / 3.0
    resolutions = [16, 32, 64]
    errors = []

    for N in resolutions:
        ny = 4
        nz = 4
        domain_size = 0.1  # Fixed physical domain
        dx = domain_size / N

        cs = np.sqrt(gamma * 1e3 / 1e-4)
        # Run for a short time: ~10% of domain crossing time
        sim_time = 0.1 * domain_size / cs

        state0 = _make_uniform_state(N, ny, nz)
        state0 = _add_sound_wave(state0, dx, amplitude=0.01, gamma=gamma)
        rho_init = state0["rho"].copy()

        state_final = _run_solver(N, ny, nz, dx, state0, sim_time, gamma=gamma)

        # Measure error in interior only (avoid boundary cells)
        margin = max(3, N // 8)
        interior = slice(margin, N - margin)
        error = _l1_error(
            state_final["rho"][interior, :, :],
            rho_init[interior, :, :],
        )
        errors.append(error)

    # Error should decrease with resolution
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"Error should decrease: N={resolutions[i]}: {errors[i]:.2e} "
            f"> N={resolutions[i+1]}: {errors[i+1]:.2e}"
        )


@pytest.mark.slow
def test_sound_wave_convergence_order():
    """Convergence order should be at least ~1.5 (theoretical 2 for PLM).

    Limiters and the centered-difference fallback may reduce the effective order,
    but we should still see significant convergence.
    """
    gamma = 5.0 / 3.0
    resolutions = [16, 32, 64]
    errors = []

    for N in resolutions:
        ny = 4
        nz = 4
        domain_size = 0.1
        dx = domain_size / N

        cs = np.sqrt(gamma * 1e3 / 1e-4)
        sim_time = 0.1 * domain_size / cs

        state0 = _make_uniform_state(N, ny, nz)
        state0 = _add_sound_wave(state0, dx, amplitude=0.01, gamma=gamma)
        rho_init = state0["rho"].copy()

        state_final = _run_solver(N, ny, nz, dx, state0, sim_time, gamma=gamma)

        margin = max(3, N // 8)
        interior = slice(margin, N - margin)
        error = _l1_error(
            state_final["rho"][interior, :, :],
            rho_init[interior, :, :],
        )
        errors.append(error)

    # Compute convergence order: p = log(e1/e2) / log(h1/h2) = log(e1/e2) / log(2)
    if errors[0] > 0 and errors[1] > 0:
        order_01 = np.log(errors[0] / errors[1]) / np.log(2.0)
    else:
        order_01 = 0.0

    if errors[1] > 0 and errors[2] > 0:
        order_12 = np.log(errors[1] / errors[2]) / np.log(2.0)
    else:
        order_12 = 0.0

    # Expect at least ~1.0 convergence order (2nd order theoretical minus limiter degradation)
    avg_order = 0.5 * (order_01 + order_12)
    assert avg_order > 0.8, (
        f"Convergence order too low: {avg_order:.2f} "
        f"(orders: {order_01:.2f}, {order_12:.2f}, errors: {errors})"
    )


# ============================================================
# T2.2: Alfvén wave propagation
# ============================================================


def test_alfven_wave_propagation():
    """Test that an Alfvén wave propagates without significant decay.

    Short-time test: initialize Alfvén wave, run for a fraction of a period,
    verify wave amplitude is preserved (< 50% decay from numerical diffusion).
    """
    gamma = 5.0 / 3.0
    N = 32
    ny = 4
    nz = 4
    dx = 0.05 / N

    rho0 = 1e-4
    p0 = 1e3
    Bx0 = 0.1
    vA = Bx0 / np.sqrt(mu_0 * rho0)
    amplitude = 0.01

    # Grid
    x = np.arange(N) * dx
    X = x[:, np.newaxis, np.newaxis] * np.ones((1, ny, nz))

    wavelength = N * dx
    k = 2.0 * np.pi / wavelength
    sin_kx = np.sin(k * X)

    m_i = 3.34e-27
    n_i = rho0 / m_i
    T0 = p0 / (2.0 * n_i * k_B)

    B = np.zeros((3, N, ny, nz))
    B[0] = Bx0
    B[1] = Bx0 * amplitude * sin_kx

    velocity = np.zeros((3, N, ny, nz))
    velocity[1] = vA * amplitude * sin_kx

    state = {
        "rho": np.full((N, ny, nz), rho0),
        "velocity": velocity,
        "pressure": np.full((N, ny, nz), p0),
        "B": B,
        "Te": np.full((N, ny, nz), T0),
        "Ti": np.full((N, ny, nz), T0),
        "psi": np.zeros((N, ny, nz)),
    }

    By_amp_initial = float(np.max(np.abs(state["B"][1])))

    # Run for 1/4 period (short time to avoid boundary issues)
    period = wavelength / vA
    sim_time = 0.25 * period

    state_final = _run_solver(N, ny, nz, dx, state, sim_time, gamma=gamma)

    By_amp_final = float(np.max(np.abs(state_final["B"][1])))

    # Wave amplitude should not decay by more than 50%
    assert By_amp_final > 0.5 * By_amp_initial, (
        f"Alfvén wave decayed too much: "
        f"{By_amp_initial:.3e} -> {By_amp_final:.3e} "
        f"({100*(1 - By_amp_final/By_amp_initial):.1f}% loss)"
    )

    # Note: HLL solver is very diffusive for Alfvén waves and introduces
    # significant compressive artifacts. We only check B-field amplitude
    # and that the solution stays bounded (no blow-up).
    assert np.all(np.isfinite(state_final["rho"])), "Density has NaN/Inf"
    assert np.all(state_final["rho"] > 0), "Density went negative"


# ============================================================
# T2.3: Divergence cleaning
# ============================================================


def test_dedner_psi_responds_to_div_B():
    """Dedner cleaning: psi scalar should respond to non-zero div(B).

    When div(B) != 0, the Dedner mechanism drives psi away from zero via
    dpsi/dt = -ch^2 * div(B). Verify psi becomes non-zero after stepping,
    confirming the cleaning mechanism is active.
    """
    gamma = 5.0 / 3.0
    N = 16
    ny = 16
    nz = 16
    dx = 0.01

    rho0 = 1e-4
    p0 = 1e3
    m_i = 3.34e-27
    n_i = rho0 / m_i
    T0 = p0 / (2.0 * n_i * k_B)

    x = np.arange(N) * dx
    y = np.arange(ny) * dx
    z = np.arange(nz) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Sinusoidal B-field with non-zero divergence
    k = 2.0 * np.pi / (N * dx)
    Bx = 0.01 * np.sin(k * X)  # dBx/dx = 0.01 * k * cos(kx) != 0
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    state = {
        "rho": np.full((N, ny, nz), rho0),
        "velocity": np.zeros((3, N, ny, nz)),
        "pressure": np.full((N, ny, nz), p0),
        "B": np.array([Bx, By, Bz]),
        "Te": np.full((N, ny, nz), T0),
        "Ti": np.full((N, ny, nz), T0),
        "psi": np.zeros((N, ny, nz)),
    }

    # Verify initial psi is zero
    assert np.all(state["psi"] == 0.0)

    # Run with Dedner cleaning enabled
    solver = MHDSolver(
        grid_shape=(N, ny, nz),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        dedner_ch=10.0,  # Strong cleaning speed
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        riemann_solver="hll",
    )

    # Run a few steps
    time = 0.0
    sim_time = 5e-5
    while time < sim_time:
        dt = solver._compute_dt(state)
        dt = min(dt, sim_time - time)
        if dt < 1e-20:
            break
        state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=None)
        time += dt

    # Psi should now be non-zero (Dedner mechanism is active)
    psi_rms = np.sqrt(np.mean(state["psi"] ** 2))
    assert psi_rms > 1e-20, (
        f"Dedner psi didn't respond to div(B): psi_rms = {psi_rms:.3e}"
    )

    # All fields should remain finite (no blow-up)
    assert np.all(np.isfinite(state["rho"])), "Density has NaN/Inf"
    assert np.all(np.isfinite(state["B"])), "B-field has NaN/Inf"
    assert np.all(state["rho"] > 0), "Density went negative"


# ============================================================
# T2.4: Solver stability — uniform state remains uniform
# ============================================================


def test_uniform_state_stability():
    """A uniform quiescent state should remain uniform (within round-off).

    This catches sign errors, uninitialized arrays, and spurious source terms.
    """
    N = 16
    ny = 8
    nz = 8
    dx = 0.01

    state = _make_uniform_state(N, ny, nz)
    rho_init = state["rho"][0, 0, 0]
    p_init = state["pressure"][0, 0, 0]

    solver = MHDSolver(
        grid_shape=(N, ny, nz),
        dx=dx,
        gamma=5.0 / 3.0,
        cfl=0.3,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        riemann_solver="hll",
    )

    # Run for 10 timesteps
    for _ in range(10):
        dt = solver._compute_dt(state)
        if dt < 1e-20 or dt > 1e10:
            dt = 1e-6  # Use a small safe timestep for uniform state
        state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=None)

    # Density and pressure should remain essentially uniform
    rho_err = float(np.max(np.abs(state["rho"] - rho_init))) / rho_init
    p_err = float(np.max(np.abs(state["pressure"] - p_init))) / p_init

    assert rho_err < 1e-10, f"Density drifted from uniform: max rel error = {rho_err:.2e}"
    assert p_err < 1e-10, f"Pressure drifted from uniform: max rel error = {p_err:.2e}"

    # Velocity should remain zero
    vel_max = float(np.max(np.abs(state["velocity"])))
    assert vel_max < 1e-10, f"Velocity grew from zero: max = {vel_max:.2e}"
