"""MHD linear wave convergence test for the Metal MHD solver.

Propagates small-amplitude fast magnetosonic and Alfven waves through
a magnetized plasma and measures the L1 error against the exact linear
solution. This tests the MHD solver's accuracy on smooth magnetized flow,
complementing the hydro-only sound wave convergence test.

Fast magnetosonic wave (propagating along B, degenerate k||B):
  Perturbations in rho, p, vx only (no transverse B), wave speed c_f = max(cs, vA)

Alfven wave (propagating along B):
  Perturbations in vy, Bz only (incompressible), wave speed vA = Bx/sqrt(rho)

Reference: Stone J.M. et al. (2008), ApJS 178, 137-177, Section 6.2.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001


def _make_fast_wave_ic(
    nx: int,
    amplitude: float = 1e-4,
    gamma: float = 5.0 / 3.0,
    rho0: float = 1.0,
    p0: float = 1.0,
    Bx0: float = 1.0,
) -> tuple[dict, float, float, float]:
    """Create fast magnetosonic wave ICs on a quasi-1D grid.

    For parallel propagation (k || B), the fast mode is a pure sound wave
    with no transverse B perturbation. The wave speed is c_f = max(cs, vA)
    but the eigenstructure only involves rho, p, vx.

    Returns (state, dx, c_fast, L).
    """
    ny = nz = 4
    L = 1.0
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    k = 2.0 * np.pi / L

    cs = np.sqrt(gamma * p0 / rho0)
    vA = Bx0 / np.sqrt(rho0)
    c_fast = max(cs, vA)  # degenerate fast = sound for k || B

    # Pure acoustic eigenmode in magnetized background (rightward)
    rho = rho0 * (1.0 + amplitude * np.sin(k * xc))
    vx = amplitude * cs * np.sin(k * xc)  # velocity perturbation at cs
    p = p0 * (1.0 + gamma * amplitude * np.sin(k * xc))

    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(p[:, None, None], (nx, ny, nz)).copy()

    vel = np.zeros((3, nx, ny, nz), dtype=np.float64)
    vel[0] = np.broadcast_to(vx[:, None, None], (nx, ny, nz)).copy()

    B = np.zeros((3, nx, ny, nz), dtype=np.float64)
    B[0, :, :, :] = Bx0  # uniform Bx (no transverse B for k||B fast mode)

    state = {
        "rho": rho_3d,
        "velocity": vel,
        "pressure": p_3d,
        "B": B,
        "Te": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, c_fast, L


def _make_alfven_wave_ic(
    nx: int,
    amplitude: float = 1e-4,
    rho0: float = 1.0,
    Bx0: float = 1.0,
) -> tuple[dict, float, float, float]:
    """Create Alfven wave ICs on a quasi-1D grid.

    The Alfven wave has perturbations only in vy and Bz, with no density
    or pressure change (incompressible mode).

    Returns (state, dx, vA, L).
    """
    ny = nz = 4
    L = 1.0
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    k = 2.0 * np.pi / L

    vA = Bx0 / np.sqrt(rho0)

    # Alfven eigenmode: vy and Bz perturbations (rightward propagating)
    vy = amplitude * vA * np.sin(k * xc)
    Bz = -amplitude * Bx0 * np.sin(k * xc)  # dBz = -sqrt(rho) * dvy

    vel = np.zeros((3, nx, ny, nz), dtype=np.float64)
    vel[1] = np.broadcast_to(vy[:, None, None], (nx, ny, nz)).copy()

    B = np.zeros((3, nx, ny, nz), dtype=np.float64)
    B[0, :, :, :] = Bx0
    B[2] = np.broadcast_to(Bz[:, None, None], (nx, ny, nz)).copy()

    p0 = 1.0
    state = {
        "rho": np.full((nx, ny, nz), rho0, dtype=np.float64),
        "velocity": vel,
        "pressure": np.full((nx, ny, nz), p0, dtype=np.float64),
        "B": B,
        "Te": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, vA, L


class TestFastWaveStability:
    """Fast magnetosonic wave should be stable."""

    def test_fast_wave_no_nan(self):
        """HLL+PLM runs 100 steps on fast wave without NaN."""
        nx = 32
        state, dx, _, _ = _make_fast_wave_ic(nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))
        assert not np.any(np.isnan(state["B"]))
        assert np.all(state["rho"] > 0)
        assert np.all(state["pressure"] > 0)


class TestAlfvenWaveStability:
    """Alfven wave should be stable and preserve density/pressure."""

    def test_alfven_wave_no_nan(self):
        """HLL+PLM runs 100 steps on Alfven wave without NaN."""
        nx = 32
        state, dx, _, _ = _make_alfven_wave_ic(nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["B"]))

    def test_alfven_density_unchanged(self):
        """Alfven wave is incompressible — density should stay constant."""
        nx = 64
        state, dx, _, _ = _make_alfven_wave_ic(nx)
        rho0 = state["rho"][0, 0, 0]

        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(50):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        rho_1d = state["rho"][:, 2, 2]
        max_dev = np.max(np.abs(rho_1d - rho0)) / rho0
        # Alfven wave should not perturb density (to numerical precision)
        assert max_dev < 0.01, (
            f"Alfven wave density deviation: {max_dev:.2%}, expected < 1%"
        )


class TestFastWaveConvergence:
    """Convergence for fast magnetosonic wave."""

    @pytest.mark.slow
    def test_fast_wave_convergence(self):
        """PLM+HLL error should decrease with resolution on fast wave.

        For k||B, the fast mode propagates as a sound wave at cs.
        Uses float64 precision and T/8 propagation.
        """
        gamma = 5.0 / 3.0
        rho0 = 1.0
        p0 = 1.0
        Bx0 = 1.0
        amplitude = 1e-3
        cs = np.sqrt(gamma * p0 / rho0)
        L = 1.0
        t_end = 0.125 * L / cs

        resolutions = [32, 64, 128]
        errors = {}

        for nx in resolutions:
            state, dx, _, _ = _make_fast_wave_ic(
                nx, gamma=gamma, Bx0=Bx0, amplitude=amplitude,
            )
            xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)

            solver = MetalMHDSolver(
                grid_shape=(nx, 4, 4),
                dx=dx,
                gamma=gamma,
                cfl=0.3,
                device="cpu",
                riemann_solver="hll",
                reconstruction="plm",
                time_integrator="ssp_rk2",
                precision="float64",
                use_ct=False,
            )

            t_total = 0.0
            for _ in range(50000):
                dt = solver.compute_dt(state)
                if t_total + dt > t_end:
                    dt = t_end - t_total
                state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
                t_total += dt
                if t_total >= t_end * 0.9999:
                    break

            # Compare against initial state shifted at cs
            k = 2.0 * np.pi / L
            rho_exact = rho0 * (
                1.0 + amplitude * np.sin(k * (xc - cs * t_total))
            )

            rho_num = state["rho"][:, 2, 2]
            margin = nx // 8
            interior = slice(margin, nx - margin)
            errors[nx] = np.mean(np.abs(rho_num[interior] - rho_exact[interior]))

        # Error should decrease with resolution
        assert errors[128] < errors[32], (
            f"No convergence: L1(32)={errors[32]:.4e}, L1(128)={errors[128]:.4e}"
        )


class TestBxConservation:
    """Normal B-field should be conserved in all MHD tests."""

    def test_bx_constant_fast_wave(self):
        """Bx should remain constant during fast wave propagation."""
        nx = 64
        state, dx, _, _ = _make_fast_wave_ic(nx, Bx0=1.0)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(50):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        Bx = state["B"][0, :, 2, 2]
        max_dev = np.max(np.abs(Bx - 1.0))
        assert max_dev < 0.01, f"Bx deviation: {max_dev:.4f}"

    def test_bx_constant_alfven_wave(self):
        """Bx should remain constant during Alfven wave propagation."""
        nx = 64
        state, dx, _, _ = _make_alfven_wave_ic(nx, Bx0=1.0)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(50):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        Bx = state["B"][0, :, 2, 2]
        max_dev = np.max(np.abs(Bx - 1.0))
        assert max_dev < 0.01, f"Bx deviation: {max_dev:.4f}"
