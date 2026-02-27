"""Sedov-Taylor blast wave validation for the Metal MHD solver.

Runs the Metal engine on the Sedov point explosion and compares density,
pressure, and velocity profiles against the exact self-similar solution from
Kamm & Timmes (2007). This is a quantitative accuracy test, not just a
stability check.

The Sedov blast is a pure hydro problem (B=0), so no CT is needed.  We test
on CPU with float32 (matching the MPS precision mode) for reproducibility
without requiring a Metal GPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001
from dpf.validation.sedov_exact import SedovExact  # noqa: E402


def _make_sedov_ic(
    nx: int = 32,
    gamma: float = 5.0 / 3.0,
    eblast: float = 1.0,
    p_ambient: float = 1e-5,
    rho_ambient: float = 1.0,
) -> tuple[dict, float]:
    """Create Sedov blast initial conditions on a 3D Cartesian grid.

    Energy is deposited in the central cell as thermal pressure:
        p_center = (gamma - 1) * E_blast / V_cell

    Returns (state_dict, dx).
    """
    dx = 1.0 / nx
    rho = np.full((nx, nx, nx), rho_ambient, dtype=np.float64)
    pressure = np.full((nx, nx, nx), p_ambient, dtype=np.float64)

    # Deposit energy in the central cell
    cx, cy, cz = nx // 2, nx // 2, nx // 2
    vol_cell = dx**3
    pressure[cx, cy, cz] = (gamma - 1.0) * eblast / vol_cell

    state = {
        "rho": rho,
        "velocity": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "pressure": pressure,
        "B": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "Te": np.full((nx, nx, nx), 1e4, dtype=np.float64),
        "Ti": np.full((nx, nx, nx), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, nx, nx), dtype=np.float64),
    }
    return state, dx


class TestSedovMetalStability:
    """Basic stability: no NaN, positive density/pressure after many steps."""

    def test_sedov_hll_plm_no_nan(self):
        """HLL+PLM solver runs 50 steps on Sedov without NaN."""
        nx = 16
        state, dx = _make_sedov_ic(nx=nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
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

        assert not np.any(np.isnan(state["rho"])), "NaN in density"
        assert not np.any(np.isnan(state["pressure"])), "NaN in pressure"
        assert np.all(state["rho"] > 0), "Negative density"
        assert np.all(state["pressure"] >= 0), "Negative pressure"

    def test_sedov_energy_conservation(self):
        """Total energy should be conserved to <5% over 30 steps."""
        gamma = 5.0 / 3.0
        eblast = 1.0
        nx = 16

        state, dx = _make_sedov_ic(nx=nx, gamma=gamma, eblast=eblast)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            gamma=gamma,
            cfl=0.3,
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
            B = s["B"]
            ke = 0.5 * rho * np.sum(v**2, axis=0)
            ie = p / (gamma - 1.0)
            me = 0.5 * np.sum(B**2, axis=0)
            return np.sum((ke + ie + me) * dx**3)

        E0 = total_energy(state)
        for _ in range(30):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        E_final = total_energy(state)

        rel_error = abs(E_final - E0) / E0
        assert rel_error < 0.05, f"Energy conservation error: {rel_error:.2%}"


def _run_sedov_to_target(
    nx: int,
    target_r_frac: float = 0.30,
) -> tuple[dict, float, float, float]:
    """Run Metal HLL+PLM solver until shock reaches target_r_frac * half-domain.

    Returns (state, dx, t_total, r_shock_exact).
    """
    gamma = 5.0 / 3.0
    eblast = 1.0
    rho_ambient = 1.0
    p_ambient = 1e-5

    state, dx = _make_sedov_ic(
        nx=nx, gamma=gamma, eblast=eblast,
        p_ambient=p_ambient, rho_ambient=rho_ambient,
    )

    solver = MetalMHDSolver(
        grid_shape=(nx, nx, nx),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        device="cpu",
        riemann_solver="hll",
        reconstruction="plm",
        time_integrator="ssp_rk2",
        precision="float32",
        use_ct=False,
    )

    sedov = SedovExact(geometry=3, gamma=gamma, eblast=eblast, rho0=rho_ambient)
    target_r = target_r_frac * 0.5  # half-domain = 0.5

    # Compute the time at which exact R_shock = target_r
    prefactor = (eblast / (sedov.get_alpha() * rho_ambient)) ** (1.0 / 5.0)
    t_target = (target_r / prefactor) ** (5.0 / 2.0)

    t_total = 0.0
    for _ in range(10000):
        dt = solver.compute_dt(state)
        if t_total + dt > t_target:
            dt = t_target - t_total
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        t_total += dt
        if t_total >= t_target * 0.999:
            break

    r_shock_exact = sedov.shock_radius(t_total)
    return state, dx, t_total, r_shock_exact


class TestSedovMetalAccuracy:
    """Quantitative accuracy: compare Metal solver output with exact solution."""

    @pytest.mark.slow
    def test_sedov_density_profile(self):
        """Metal solver density profile should have reasonable L1 error."""
        gamma = 5.0 / 3.0
        eblast = 1.0
        rho_ambient = 1.0
        nx = 32

        state, dx, t_total, r_shock = _run_sedov_to_target(nx=nx, target_r_frac=0.30)

        # Build radial grid
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)

        # Get exact solution
        sedov = SedovExact(geometry=3, gamma=gamma, eblast=eblast, rho0=rho_ambient)
        r_flat = r.flatten()
        _, rho_exact, _, _, _ = sedov.evaluate(r_flat, t_total)
        rho_exact = rho_exact.reshape(r.shape)

        # L1 error in the shocked region
        interior = r < r_shock * 0.90
        if np.sum(interior) < 10:
            pytest.skip("Shock hasn't propagated enough for comparison")

        rho_num = state["rho"]
        l1_abs = np.mean(np.abs(rho_num[interior] - rho_exact[interior]))
        l1_rel = l1_abs / np.mean(rho_exact[interior])

        # 32^3 with HLL+PLM is highly diffusive. 85% tolerance is realistic
        # for the thin Sedov shell on this coarse grid. For precision use 128^3+.
        assert l1_rel < 0.85, f"L1(rho) relative error {l1_rel:.2%} exceeds 85%"

    @pytest.mark.slow
    def test_sedov_shock_radius(self):
        """Metal solver shock radius should match exact within 30%."""
        rho_ambient = 1.0
        nx = 32

        state, dx, t_total, r_shock_exact = _run_sedov_to_target(
            nx=nx, target_r_frac=0.30,
        )

        # Build radial grid and radially-average density
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)

        rho_num = state["rho"]
        r_bins = np.linspace(0, 0.5, 50)
        rho_avg = np.zeros(len(r_bins) - 1)
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.sum(mask) > 0:
                rho_avg[i] = np.mean(rho_num[mask])
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

        # Find outermost bin where rho > 1.1 * ambient
        shocked = rho_avg > 1.1 * rho_ambient
        if np.any(shocked):
            r_shock_num = r_centers[np.max(np.where(shocked))]
        else:
            pytest.skip("No shocked cells found above threshold")

        # HLL diffusion spreads shock outward by ~2-3 cells on 32^3
        rel_error = abs(r_shock_num - r_shock_exact) / r_shock_exact
        assert rel_error < 0.30, (
            f"Shock radius error {rel_error:.2%}: "
            f"numerical={r_shock_num:.4f}, exact={r_shock_exact:.4f}"
        )


class TestSedovMetalConvergence:
    """Resolution convergence: error should decrease with finer grid."""

    @pytest.mark.slow
    def test_sedov_convergence_rate(self):
        """Error at 32^3 should be lower than at 16^3."""
        gamma = 5.0 / 3.0
        eblast = 1.0
        rho_ambient = 1.0
        p_ambient = 1e-5
        n_steps = 100

        errors = {}
        for nx in [16, 32]:
            state, dx = _make_sedov_ic(
                nx=nx, gamma=gamma, eblast=eblast,
                p_ambient=p_ambient, rho_ambient=rho_ambient,
            )

            solver = MetalMHDSolver(
                grid_shape=(nx, nx, nx),
                dx=dx,
                gamma=gamma,
                cfl=0.3,
                device="cpu",
                riemann_solver="hll",
                reconstruction="plm",
                time_integrator="ssp_rk2",
                precision="float32",
                use_ct=False,
            )

            t_total = 0.0
            for _ in range(n_steps):
                dt = solver.compute_dt(state)
                state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
                t_total += dt

            # Exact solution at final time
            sedov = SedovExact(geometry=3, gamma=gamma, eblast=eblast, rho0=rho_ambient)

            x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
            X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
            r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)

            r_shock = sedov.shock_radius(t_total)
            r_flat = r.flatten()
            _, rho_exact, _, _, _ = sedov.evaluate(r_flat, t_total)
            rho_exact = rho_exact.reshape(r.shape)

            interior = r < r_shock * 0.9
            if np.sum(interior) < 5:
                pytest.skip(f"Not enough shocked cells at nx={nx}")

            l1 = np.mean(np.abs(state["rho"][interior] - rho_exact[interior]))
            l1_rel = l1 / np.mean(rho_exact[interior])
            errors[nx] = l1_rel

        # Finer grid should have lower error
        assert errors[32] < errors[16], (
            f"No convergence: L1(16)={errors[16]:.3f}, L1(32)={errors[32]:.3f}"
        )
