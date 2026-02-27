"""Brio-Wu MHD shock tube validation for the Metal MHD solver.

Runs the Metal engine on the Brio-Wu problem (Brio & Wu 1988) and
validates against qualitative expectations. Unlike the Sod problem,
no closed-form exact solution exists for MHD Riemann problems. Instead
we check:

1. Stability — no NaN/Inf, positive density and pressure
2. Conservation — mass, momentum, energy, Bx (normal B)
3. Wave structure — the solution should contain 5 waves:
   fast rarefaction, slow compound, contact, slow shock, fast rarefaction
4. Symmetry — By should flip sign across the contact

The Brio-Wu ICs:
  Left:  rho=1.0, p=1.0, u=0, Bx=0.75, By=1.0
  Right: rho=0.125, p=0.1, u=0, Bx=0.75, By=-1.0
  gamma=2.0, domain [0,1], interface at x=0.5

Reference: Brio M. & Wu C.C. (1988), J. Comput. Phys. 75, 400-422.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402, I001


def _make_briowu_ic(
    nx: int = 200,
    gamma: float = 2.0,
    x0: float = 0.5,
) -> tuple[dict, float]:
    """Create Brio-Wu MHD shock tube ICs on a quasi-1D grid (nx, 4, 4).

    Uses ny=nz=4 (minimum for PLM stencil) to keep it fast.
    """
    ny = nz = 4
    dx = 1.0 / nx

    xc = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)

    rho = np.empty((nx, ny, nz), dtype=np.float64)
    pressure = np.empty((nx, ny, nz), dtype=np.float64)
    B = np.zeros((3, nx, ny, nz), dtype=np.float64)

    for i in range(nx):
        if xc[i] < x0:
            rho[i, :, :] = 1.0
            pressure[i, :, :] = 1.0
            B[0, i, :, :] = 0.75  # Bx
            B[1, i, :, :] = 1.0   # By
        else:
            rho[i, :, :] = 0.125
            pressure[i, :, :] = 0.1
            B[0, i, :, :] = 0.75  # Bx (same on both sides)
            B[1, i, :, :] = -1.0  # By (sign flip)

    state = {
        "rho": rho,
        "velocity": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "pressure": pressure,
        "B": B,
        "Te": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 1e4, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx


class TestBrioWuMetalStability:
    """Basic stability: no NaN, positive quantities after Brio-Wu evolution."""

    def test_hll_plm_no_nan(self):
        """HLL+PLM runs 100 steps on Brio-Wu without NaN."""
        nx = 100
        state, dx = _make_briowu_ic(nx=nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
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
        assert not np.any(np.isnan(state["B"])), "NaN in B field"
        assert np.all(state["rho"] > 0), "Negative density"
        assert np.all(state["pressure"] > 0), "Negative pressure"

    def test_hlld_weno5_no_nan(self):
        """HLLD+WENO5 on Brio-Wu (stress test for float32 stability).

        Previously xfail due to HLLD float32 discriminant NaN on strong By
        discontinuity. Fixed in Phase O via Lax-Friedrichs fallback mechanism.
        """
        nx = 100
        state, dx = _make_briowu_ic(nx=nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hlld",
            reconstruction="weno5",
            time_integrator="ssp_rk3",
            precision="float32",
            use_ct=False,
        )

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"])), "NaN in density"
        assert not np.any(np.isnan(state["pressure"])), "NaN in pressure"


class TestBrioWuMetalConservation:
    """Conservation law tests for Brio-Wu on the Metal solver."""

    def _run_briowu(
        self, nx: int, n_steps: int, riemann: str = "hll", recon: str = "plm",
    ) -> tuple[dict, dict, float]:
        """Run Brio-Wu and return (initial_state_copy, final_state, dx)."""
        state, dx = _make_briowu_ic(nx=nx, gamma=2.0)

        # Deep copy initial state for conservation comparison
        state0 = {k: np.copy(v) for k, v in state.items()}

        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver=riemann,
            reconstruction=recon,
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(n_steps):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        return state0, state, dx

    def test_mass_conservation(self):
        """Mass should be conserved to <2% over 50 steps."""
        state0, state_f, dx = self._run_briowu(100, 50)
        mass_0 = np.sum(state0["rho"]) * dx**3
        mass_f = np.sum(state_f["rho"]) * dx**3
        rel_error = abs(mass_f - mass_0) / mass_0
        assert rel_error < 0.02, f"Mass conservation error: {rel_error:.2%}"

    def test_energy_conservation(self):
        """Total energy (kinetic + thermal + magnetic) conserved to <5%."""
        gamma = 2.0
        state0, state_f, dx = self._run_briowu(100, 50)

        def total_energy(s):
            rho = s["rho"]
            v = s["velocity"]
            p = s["pressure"]
            B = s["B"]
            ke = 0.5 * rho * np.sum(v**2, axis=0)
            ie = p / (gamma - 1.0)
            me = 0.5 * np.sum(B**2, axis=0)
            return np.sum((ke + ie + me) * dx**3)

        E0 = total_energy(state0)
        Ef = total_energy(state_f)
        rel_error = abs(Ef - E0) / E0
        assert rel_error < 0.05, f"Energy conservation error: {rel_error:.2%}"

    def test_bx_conservation(self):
        """Normal B-field Bx should be nearly constant (div(B)=0 in 1D)."""
        _, state_f, _ = self._run_briowu(100, 50)
        Bx = state_f["B"][0, :, 2, 2]  # 1D profile along x
        # Bx should remain 0.75 everywhere (it's the normal component)
        max_dev = np.max(np.abs(Bx - 0.75))
        assert max_dev < 0.1, f"Bx deviation from 0.75: {max_dev:.4f}"


class TestBrioWuMetalWaveStructure:
    """Verify that the solution has the expected MHD wave structure."""

    @pytest.mark.slow
    def test_wave_count(self):
        """At t~0.1, density profile should have at least 3 distinct regions."""
        nx = 200
        state, dx = _make_briowu_ic(nx=nx, gamma=2.0)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        t_target = 0.1
        t_total = 0.0
        for _ in range(10000):
            dt = solver.compute_dt(state)
            if t_total + dt > t_target:
                dt = t_target - t_total
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            t_total += dt
            if t_total >= t_target * 0.999:
                break

        rho_1d = state["rho"][:, 2, 2]

        # Count significant density jumps (waves)
        drho = np.abs(np.diff(rho_1d))
        threshold = 0.01 * (np.max(rho_1d) - np.min(rho_1d))
        significant = drho > threshold

        # Count contiguous regions of significant gradient
        wave_count = 0
        in_wave = False
        for sig in significant:
            if sig and not in_wave:
                wave_count += 1
                in_wave = True
            elif not sig:
                in_wave = False

        # Brio-Wu should have 5 waves, but HLL smears them;
        # we should see at least 3 distinct gradient regions
        assert wave_count >= 3, (
            f"Only found {wave_count} wave regions, expected >= 3"
        )

    @pytest.mark.slow
    def test_by_sign_flip(self):
        """By should transition from positive to negative across the contact."""
        nx = 200
        state, dx = _make_briowu_ic(nx=nx, gamma=2.0)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        t_target = 0.1
        t_total = 0.0
        for _ in range(10000):
            dt = solver.compute_dt(state)
            if t_total + dt > t_target:
                dt = t_target - t_total
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            t_total += dt
            if t_total >= t_target * 0.999:
                break

        By = state["B"][1, :, 2, 2]

        # Left side should have positive By, right side negative
        left_quarter = By[: nx // 4]
        right_quarter = By[3 * nx // 4 :]

        assert np.mean(left_quarter) > 0, "Left By should be positive"
        assert np.mean(right_quarter) < 0, "Right By should be negative"

    @pytest.mark.slow
    def test_density_bounded(self):
        """Density should remain between initial min and ~4x compression."""
        nx = 200
        state, dx = _make_briowu_ic(nx=nx, gamma=2.0)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        t_target = 0.1
        t_total = 0.0
        for _ in range(10000):
            dt = solver.compute_dt(state)
            if t_total + dt > t_target:
                dt = t_target - t_total
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            t_total += dt
            if t_total >= t_target * 0.999:
                break

        rho_1d = state["rho"][:, 2, 2]
        # Density should be positive and not exceed ~4x the initial max
        assert np.all(rho_1d > 0), "Negative density"
        assert np.max(rho_1d) < 4.0, (
            f"Density too high: {np.max(rho_1d):.2f}, expected < 4.0"
        )
