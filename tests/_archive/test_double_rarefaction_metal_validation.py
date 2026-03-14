"""Double rarefaction (123 problem) validation for the Metal MHD solver.

The 123 problem is a symmetric Riemann problem with two rarefaction fans
expanding outward, creating a near-vacuum region at the center. This is
a severe test for positivity preservation — many solvers produce negative
density or pressure in the low-density central region.

ICs (symmetric, counter-propagating):
  Left:  rho=1.0, u=-2.0, p=0.4
  Right: rho=1.0, u=+2.0, p=0.4
  gamma=1.4, domain [0,1], interface at x=0.5

Key features to test:
1. Positivity — rho and p remain positive in the near-vacuum region
2. Symmetry — solution should be symmetric about x=0.5
3. L1 accuracy against the exact Riemann solution

Reference: Toro E.F. (2009), Riemann Solvers, 3rd ed., Test 2.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001
from dpf.validation.riemann_exact import (  # noqa: E402
    DOUBLE_RAREFACTION_LEFT,
    DOUBLE_RAREFACTION_RIGHT,
    ExactRiemannSolver,
)


def _make_double_rarefaction_ic(
    nx: int = 200,
    gamma: float = 1.4,
    x0: float = 0.5,
) -> tuple[dict, float]:
    """Create 123 problem ICs on a quasi-1D grid (nx, 4, 4)."""
    ny = nz = 4
    dx = 1.0 / nx
    xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)

    rho = np.empty((nx, ny, nz), dtype=np.float64)
    pressure = np.empty((nx, ny, nz), dtype=np.float64)
    velocity = np.zeros((3, nx, ny, nz), dtype=np.float64)

    for i in range(nx):
        if xc[i] < x0:
            rho[i, :, :] = DOUBLE_RAREFACTION_LEFT.rho
            pressure[i, :, :] = DOUBLE_RAREFACTION_LEFT.p
            velocity[0, i, :, :] = DOUBLE_RAREFACTION_LEFT.u
        else:
            rho[i, :, :] = DOUBLE_RAREFACTION_RIGHT.rho
            pressure[i, :, :] = DOUBLE_RAREFACTION_RIGHT.p
            velocity[0, i, :, :] = DOUBLE_RAREFACTION_RIGHT.u

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


class TestDoubleRarefactionStability:
    """Positivity preservation in the near-vacuum region."""

    def test_hll_plm_positivity(self):
        """HLL+PLM preserves positivity over 100 steps."""
        nx = 100
        state, dx = _make_double_rarefaction_ic(nx=nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
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

        assert not np.any(np.isnan(state["rho"])), "NaN in density"
        assert not np.any(np.isnan(state["pressure"])), "NaN in pressure"
        assert np.all(state["rho"] > 0), "Negative density in near-vacuum"
        assert np.all(state["pressure"] > 0), "Negative pressure in near-vacuum"


class TestDoubleRarefactionAccuracy:
    """Quantitative accuracy and symmetry tests."""

    def _run_to_time(
        self, nx: int, t_target: float,
    ) -> tuple[dict, float, float]:
        """Run Metal on 123 problem until t_target."""
        state, dx = _make_double_rarefaction_ic(nx=nx, gamma=1.4)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
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
    def test_density_l1_error(self):
        """HLL+PLM density L1 error < 25% at t=0.15 on 200 cells.

        Tolerance is higher than Sod because HLL smears the central
        near-vacuum region significantly.
        """
        nx = 200
        t_target = 0.15
        state, dx, t_actual = self._run_to_time(nx, t_target)

        rho_1d = state["rho"][:, 2, 2]

        exact = ExactRiemannSolver(
            DOUBLE_RAREFACTION_LEFT, DOUBLE_RAREFACTION_RIGHT, gamma=1.4,
        )
        xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        rho_exact, _, _ = exact.sample(xc, t=t_actual)

        l1_abs = np.mean(np.abs(rho_1d - rho_exact))
        l1_rel = l1_abs / np.mean(rho_exact)
        assert l1_rel < 0.25, f"L1(rho) = {l1_rel:.2%}, expected < 25%"

    @pytest.mark.slow
    def test_symmetry(self):
        """Solution should be approximately symmetric about x=0.5."""
        nx = 200
        t_target = 0.15
        state, dx, t_actual = self._run_to_time(nx, t_target)

        rho_1d = state["rho"][:, 2, 2]
        rho_flipped = rho_1d[::-1]

        # Relative asymmetry should be small
        asymmetry = np.mean(np.abs(rho_1d - rho_flipped)) / np.mean(rho_1d)
        assert asymmetry < 0.05, f"Asymmetry = {asymmetry:.2%}, expected < 5%"

    def test_mass_conservation(self):
        """Mass should be conserved over 20 steps.

        Uses fewer steps than Sod because the supersonic |u|=2.0 initial
        velocity causes rapid boundary mass loss through outflow BCs.
        """
        nx = 100
        state, dx = _make_double_rarefaction_ic(nx=nx)
        mass_0 = np.sum(state["rho"]) * dx**3

        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=1.4,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(20):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        mass_f = np.sum(state["rho"]) * dx**3
        rel_error = abs(mass_f - mass_0) / mass_0
        assert rel_error < 0.10, f"Mass conservation error: {rel_error:.2%}"
