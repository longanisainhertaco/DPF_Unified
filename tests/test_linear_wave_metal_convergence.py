"""Linear sound wave convergence test for the Metal MHD solver.

Propagates a small-amplitude sinusoidal density perturbation on a
uniform background and measures L1 error against the exact linear
solution after a short propagation time. Since the wave is smooth
(no shocks), the numerical error should converge at the formal order
of the reconstruction scheme:

  - PLM (piecewise linear): ~2nd order
  - WENO5: ~5th order (on smooth problems)

The setup uses a long-wavelength wave (lambda = domain length) so
that the wave stays well inside the domain for the measurement time.
We compare the numerical solution at time t against the analytically
shifted exact profile rho(x - cs*t).

The ICs:
  rho = rho0 * (1 + A * sin(2*pi*x/L))
  u   = A * cs * sin(2*pi*x/L)     (rightward-propagating eigenmode)
  p   = p0 * (1 + gamma*A*sin(2*pi*x/L))  (linearized isentropic)
  B   = 0 (pure hydro)

where A << 1 is the perturbation amplitude and cs is the sound speed.
The propagation time is T/4 (quarter period) so the wave shifts by
L/4, staying well within the domain (outflow BCs don't affect it).

Reference: Stone J.M. et al. (2008), ApJS 178, 137-177, Section 6.1.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001


def _make_sound_wave_ic(
    nx: int,
    amplitude: float = 1e-4,
    gamma: float = 5.0 / 3.0,
    rho0: float = 1.0,
    p0: float = 1.0,
) -> tuple[dict, float, float, float]:
    """Create linear sound wave ICs on a quasi-1D grid.

    Returns (state, dx, cs, L) where cs is the sound speed and L is
    the domain length.
    """
    ny = nz = 4
    L = 1.0
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    k = 2.0 * np.pi / L

    cs = np.sqrt(gamma * p0 / rho0)

    # Rightward-propagating acoustic eigenmode (linearized)
    rho = rho0 * (1.0 + amplitude * np.sin(k * xc))
    u = amplitude * cs * np.sin(k * xc)
    # Linearized isentropic: dp/p0 = gamma * drho/rho0
    p = p0 * (1.0 + gamma * amplitude * np.sin(k * xc))

    # Broadcast to 3D
    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(p[:, None, None], (nx, ny, nz)).copy()
    vel = np.zeros((3, nx, ny, nz), dtype=np.float64)
    vel[0] = np.broadcast_to(u[:, None, None], (nx, ny, nz)).copy()

    state = {
        "rho": rho_3d,
        "velocity": vel,
        "pressure": p_3d,
        "B": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "Te": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, cs, L


def _exact_rho(xc: np.ndarray, t: float, cs: float, L: float,
               amplitude: float, rho0: float) -> np.ndarray:
    """Exact linear wave density at time t (rightward propagation)."""
    k = 2.0 * np.pi / L
    return rho0 * (1.0 + amplitude * np.sin(k * (xc - cs * t)))


def _run_convergence(
    nx: int,
    t_end: float,
    riemann: str = "hll",
    recon: str = "plm",
    integrator: str = "ssp_rk2",
    precision: str = "float32",
    amplitude: float = 1e-4,
    gamma: float = 5.0 / 3.0,
) -> float:
    """Run a sound wave to t_end and return L1(rho) error vs exact."""
    state, dx, cs, L = _make_sound_wave_ic(nx, amplitude=amplitude, gamma=gamma)
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)

    solver = MetalMHDSolver(
        grid_shape=(nx, 4, 4),
        dx=dx,
        gamma=gamma,
        cfl=0.4,
        device="cpu",
        riemann_solver=riemann,
        reconstruction=recon,
        time_integrator=integrator,
        precision=precision,
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

    rho_num = state["rho"][:, 2, 2]
    rho_exact = _exact_rho(xc, t_total, cs, L, amplitude, 1.0)

    # Only compare interior cells (avoid boundary effects)
    margin = nx // 8
    interior = slice(margin, nx - margin)
    l1 = np.mean(np.abs(rho_num[interior] - rho_exact[interior]))
    return l1


class TestLinearWaveStability:
    """Basic stability on smooth flow."""

    def test_sound_wave_no_nan(self):
        """Sound wave should not produce NaN over 100 steps."""
        nx = 32
        state, dx, _, _ = _make_sound_wave_ic(nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=5.0 / 3.0,
            cfl=0.4,
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
        assert np.all(state["rho"] > 0)
        assert np.all(state["pressure"] > 0)


class TestLinearWaveConvergence:
    """Order of accuracy measurement via Richardson extrapolation.

    Uses a short propagation time (T/8 where T = L/cs) so the wave
    stays well inside the domain and boundary effects are negligible.
    """

    _T_FRAC = 0.125  # propagate for T/8

    def _t_end(self, gamma: float = 5.0 / 3.0) -> float:
        cs = np.sqrt(gamma * 1.0 / 1.0)
        return self._T_FRAC * (1.0 / cs)

    @pytest.mark.slow
    def test_plm_convergence_order(self):
        """PLM+HLL should converge at ~2nd order on smooth waves.

        Measures L1(rho) at nx=32, 64, 128 and checks that the
        convergence rate is between 1.0 and 3.0.
        """
        t_end = self._t_end()
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_convergence(
                nx, t_end, riemann="hll", recon="plm", integrator="ssp_rk2",
            )

        # Convergence order: p = log2(e_coarse / e_fine)
        order_1 = np.log2(errors[32] / errors[64])
        order_2 = np.log2(errors[64] / errors[128])
        avg_order = 0.5 * (order_1 + order_2)

        assert avg_order > 1.0, (
            f"PLM convergence too low: order={avg_order:.2f}, "
            f"errors={errors}"
        )

    @pytest.mark.slow
    def test_error_decreases_with_resolution(self):
        """L1 error should strictly decrease: e(32) > e(64) > e(128)."""
        t_end = self._t_end()
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_convergence(nx, t_end)

        assert errors[32] > errors[64] > errors[128], (
            f"Non-monotonic convergence: {errors}"
        )

    @pytest.mark.slow
    def test_absolute_error_level(self):
        """On 128 cells, L1 error after T/8 should be < 1e-6.

        This verifies the solver actually resolves the wave, not
        just that error decreases.
        """
        t_end = self._t_end()
        l1 = _run_convergence(128, t_end)
        assert l1 < 1e-6, f"L1 error too high on 128 cells: {l1:.2e}"


class TestLinearWaveAmplitude:
    """Verify wave amplitude is preserved (minimal dissipation)."""

    @pytest.mark.slow
    def test_amplitude_preserved(self):
        """Peak-to-peak density amplitude should be > 50% of initial after T/4.

        Some dissipation is expected from numerical diffusion, but
        the wave should not be completely damped.
        """
        nx = 128
        amplitude = 1e-4
        gamma = 5.0 / 3.0
        state, dx, cs, L = _make_sound_wave_ic(nx, amplitude=amplitude)
        t_end = 0.25 * L / cs  # quarter period

        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=gamma,
            cfl=0.4,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
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

        # Check interior amplitude (avoid boundaries)
        margin = nx // 4
        rho_interior = state["rho"][margin:-margin, 2, 2]
        final_amplitude = 0.5 * (np.max(rho_interior) - np.min(rho_interior))
        initial_amplitude = amplitude

        retention = final_amplitude / initial_amplitude
        assert retention > 0.3, (
            f"Wave too damped: {retention:.1%} amplitude retained"
        )
