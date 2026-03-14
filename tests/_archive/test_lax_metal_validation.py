"""Lax shock tube validation for the Metal MHD solver.

Runs the Metal engine on the Lax problem (Lax 1954) and compares density
and pressure profiles against the exact Riemann solution. The Lax problem
is harder than Sod — it involves a stronger shock and non-zero initial
velocity — making it a good stress test for numerical dissipation.

The Lax ICs:
  Left:  rho=0.445, u=0.698, p=3.528
  Right: rho=0.5, u=0.0, p=0.571
  gamma=1.4, domain [0,1], interface at x=0.5

Reference: Lax P.D. (1954), Comm. Pure Appl. Math. 7, 159-193.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001
from dpf.validation.riemann_exact import (  # noqa: E402
    LAX_LEFT,
    LAX_RIGHT,
    ExactRiemannSolver,
)


def _make_lax_ic(
    nx: int = 200,
    gamma: float = 1.4,
    x0: float = 0.5,
) -> tuple[dict, float]:
    """Create Lax shock tube ICs on a quasi-1D grid (nx, 4, 4)."""
    ny = nz = 4
    dx = 1.0 / nx
    xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)

    rho = np.empty((nx, ny, nz), dtype=np.float64)
    pressure = np.empty((nx, ny, nz), dtype=np.float64)
    velocity = np.zeros((3, nx, ny, nz), dtype=np.float64)

    for i in range(nx):
        if xc[i] < x0:
            rho[i, :, :] = LAX_LEFT.rho
            pressure[i, :, :] = LAX_LEFT.p
            velocity[0, i, :, :] = LAX_LEFT.u
        else:
            rho[i, :, :] = LAX_RIGHT.rho
            pressure[i, :, :] = LAX_RIGHT.p
            velocity[0, i, :, :] = LAX_RIGHT.u

    state = {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "Te": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx


class TestLaxMetalStability:
    """Basic stability: no NaN, positive quantities."""

    def test_lax_hll_plm_no_nan(self):
        """HLL+PLM runs 100 steps on Lax without NaN."""
        nx = 100
        state, dx = _make_lax_ic(nx=nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
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

        assert not np.any(np.isnan(state["rho"])), "NaN in density"
        assert not np.any(np.isnan(state["pressure"])), "NaN in pressure"
        assert np.all(state["rho"] > 0), "Negative density"
        assert np.all(state["pressure"] > 0), "Negative pressure"


class TestLaxMetalAccuracy:
    """Quantitative accuracy against the exact Riemann solution."""

    def _run_lax_to_time(
        self, nx: int, t_target: float, riemann: str = "hll", recon: str = "plm",
    ) -> tuple[dict, float, float]:
        """Run Metal on Lax until t_target. Returns (state, dx, t_actual)."""
        state, dx = _make_lax_ic(nx=nx, gamma=1.4)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
            cfl=0.4,
            device="cpu",
            riemann_solver=riemann,
            reconstruction=recon,
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        t_total = 0.0
        for _ in range(10000):
            dt = solver.compute_dt(state)
            if t_total + dt > t_target:
                dt = t_target - t_total
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            t_total += dt
            if t_total >= t_target * 0.999:
                break

        return state, dx, t_total

    @pytest.mark.slow
    def test_lax_density_l1_hll(self):
        """HLL+PLM density L1 error < 20% at t=0.15 on 200 cells."""
        nx = 200
        t_target = 0.15
        state, dx, t_actual = self._run_lax_to_time(nx, t_target)

        rho_1d = state["rho"][:, 2, 2]

        exact = ExactRiemannSolver(LAX_LEFT, LAX_RIGHT, gamma=1.4)
        xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        rho_exact, _, _ = exact.sample(xc, t=t_actual)

        l1_abs = np.mean(np.abs(rho_1d - rho_exact))
        l1_rel = l1_abs / np.mean(rho_exact)
        assert l1_rel < 0.20, f"L1(rho) = {l1_rel:.2%}, expected < 20%"

    @pytest.mark.slow
    def test_lax_pressure_l1_hll(self):
        """HLL+PLM pressure L1 error < 20% at t=0.15 on 200 cells."""
        nx = 200
        t_target = 0.15
        state, dx, t_actual = self._run_lax_to_time(nx, t_target)

        p_1d = state["pressure"][:, 2, 2]

        exact = ExactRiemannSolver(LAX_LEFT, LAX_RIGHT, gamma=1.4)
        xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        _, _, p_exact = exact.sample(xc, t=t_actual)

        l1_abs = np.mean(np.abs(p_1d - p_exact))
        l1_rel = l1_abs / np.mean(p_exact)
        assert l1_rel < 0.20, f"L1(p) = {l1_rel:.2%}, expected < 20%"

    def test_lax_mass_conservation(self):
        """Mass should be conserved over 50 steps."""
        nx = 100
        state, dx = _make_lax_ic(nx=nx, gamma=1.4)
        mass_0 = np.sum(state["rho"]) * dx**3

        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
            cfl=0.4,
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

        mass_f = np.sum(state["rho"]) * dx**3
        rel_error = abs(mass_f - mass_0) / mass_0
        # Lax has non-zero initial velocity, so mass exits the domain faster
        assert rel_error < 0.05, f"Mass conservation error: {rel_error:.2%}"

    def test_lax_energy_conservation(self):
        """Total energy should be conserved over 50 steps."""
        nx = 100
        gamma = 1.4
        state, dx = _make_lax_ic(nx=nx, gamma=gamma)

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

        def total_energy(s):
            rho = s["rho"]
            v = s["velocity"]
            p = s["pressure"]
            ke = 0.5 * rho * np.sum(v**2, axis=0)
            ie = p / (gamma - 1.0)
            return np.sum((ke + ie) * dx**3)

        E0 = total_energy(state)
        for _ in range(50):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        E_final = total_energy(state)

        rel_error = abs(E_final - E0) / E0
        # Non-zero initial velocity causes energy to exit through boundaries
        assert rel_error < 0.10, f"Energy conservation error: {rel_error:.2%}"


class TestLaxMetalConvergence:
    """Resolution convergence test for the Lax problem."""

    @pytest.mark.slow
    def test_lax_convergence(self):
        """Error should decrease with finer grid (100 vs 200 cells)."""
        t_target = 0.12
        errors = {}

        for nx in [100, 200]:
            state, dx = _make_lax_ic(nx=nx, gamma=1.4)
            solver = MetalMHDSolver(
                grid_shape=(nx, 4, 4),
                dx=dx,
                gamma=1.4,
                cfl=0.4,
                device="cpu",
                riemann_solver="hll",
                reconstruction="plm",
                time_integrator="ssp_rk2",
                precision="float32",
                use_ct=False,
            )

            t_total = 0.0
            for _ in range(5000):
                dt = solver.compute_dt(state)
                if t_total + dt > t_target:
                    dt = t_target - t_total
                state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
                t_total += dt
                if t_total >= t_target * 0.999:
                    break

            rho_1d = state["rho"][:, 2, 2]
            xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
            exact = ExactRiemannSolver(LAX_LEFT, LAX_RIGHT, gamma=1.4)
            rho_exact, _, _ = exact.sample(xc, t=t_total)

            l1 = np.mean(np.abs(rho_1d - rho_exact))
            errors[nx] = l1

        assert errors[200] < errors[100], (
            f"No convergence: L1(100)={errors[100]:.4f}, L1(200)={errors[200]:.4f}"
        )
