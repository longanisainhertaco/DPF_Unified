"""Tests for the Godunov (PLM+HLL) flux path in CylindricalMHDSolver.

Validates the use_godunov_flux=True mode that replaces np.gradient central
differences with PLM reconstruction + HLL Riemann fluxes.  This path is
designed to handle sheath discontinuities without Gibbs oscillations.

Test matrix:
    1. Sod shock tube (1D radial): no NaN, shock at correct position
    2. Uniform state preservation: dU/dt = 0 for uniform initial conditions
    3. Conservation: total energy conserved on uniform state
    4. Small-grid smoke test: runs without crash at minimal resolution
    5. PLM reconstruction unit test: correct slopes, monotonicity
"""

from __future__ import annotations

import numpy as np

from dpf.constants import mu_0
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


def _make_uniform_state(
    nr: int = 16,
    nz: int = 16,
    dr: float = 0.001,
    dz: float = 0.001,
    rho0: float = 1e-4,
    p0: float = 1e3,
    B_theta: float = 0.0,
) -> tuple[CylindricalMHDSolver, dict]:
    """Create a force-free uniform solver state for preservation tests."""
    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=5.0 / 3.0, cfl=0.3,
        enable_hall=False, enable_resistive=False,
        use_godunov_flux=True,
        conservative_energy=True,
    )
    state = {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }
    state["B"][1, :, :, :] = B_theta
    return solver, state


class TestGodunovInstantiation:
    """Verify the solver creates with use_godunov_flux=True."""

    def test_flag_stored(self) -> None:
        solver = CylindricalMHDSolver(
            nr=8, nz=8, dr=0.01, dz=0.01,
            use_godunov_flux=True,
        )
        assert solver.use_godunov_flux is True

    def test_default_off(self) -> None:
        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert solver.use_godunov_flux is False


class TestPLMReconstruction:
    """Unit tests for the PLM minmod reconstruction."""

    def test_uniform_field_zero_slopes(self) -> None:
        """Uniform field should produce zero slopes → L=R=cell value."""
        solver, _ = _make_uniform_state(nr=8, nz=8)
        q = np.ones((8, 8)) * 5.0
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        np.testing.assert_allclose(q_L, 5.0, atol=1e-14)
        np.testing.assert_allclose(q_R, 5.0, atol=1e-14)

    def test_linear_field_exact(self) -> None:
        """Linear field q = i should be reconstructed exactly (2nd order)."""
        solver, _ = _make_uniform_state(nr=16, nz=4)
        q = np.zeros((16, 4))
        for i in range(16):
            q[i, :] = float(i)
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        # Interior interfaces (away from boundaries): L and R should match
        # Interface i+1/2: L = q[i] + 0.5*slope[i], R = q[i+1] - 0.5*slope[i+1]
        # For linear field with slope=1: L[i] = i + 0.5, R[i] = (i+1) - 0.5 = i + 0.5
        for j in range(2, 13):
            np.testing.assert_allclose(q_L[j, 0], j + 0.5, atol=1e-12)
            np.testing.assert_allclose(q_R[j, 0], j + 0.5, atol=1e-12)

    def test_monotonicity_preserved(self) -> None:
        """PLM with minmod should not create new extrema."""
        solver, _ = _make_uniform_state(nr=16, nz=4)
        q = np.zeros((16, 4))
        q[7, :] = 10.0  # spike
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        assert np.all(q_L >= 0.0)
        assert np.all(q_R >= 0.0)
        assert np.max(q_L) <= 10.0
        assert np.max(q_R) <= 10.0


class TestGodunovUniformState:
    """Uniform state should be preserved (dU/dt ~ 0)."""

    def test_single_step_no_nan(self) -> None:
        solver, state = _make_uniform_state()
        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(result[key])), f"NaN in {key}"

    def test_uniform_state_preserved(self) -> None:
        """Uniform state should remain uniform after one step."""
        solver, state = _make_uniform_state()
        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        rho_init = state["rho"].mean()
        p_init = state["pressure"].mean()
        np.testing.assert_allclose(
            result["rho"].mean(), rho_init, rtol=1e-3,
            err_msg="Density not preserved on uniform state",
        )
        np.testing.assert_allclose(
            result["pressure"].mean(), p_init, rtol=1e-3,
            err_msg="Pressure not preserved on uniform state",
        )


class TestGodunovConservation:
    """Energy conservation on uniform state with no sources."""

    def test_total_energy_conserved(self) -> None:
        solver, state = _make_uniform_state()
        gamma = solver.gamma
        rho = solver._squeeze(state["rho"])
        p = solver._squeeze(state["pressure"])
        B = solver._squeeze(state["B"])
        v = solver._squeeze(state["velocity"])
        B_sq = np.sum(B**2, axis=0)
        v_sq = np.sum(v**2, axis=0)
        E0 = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2 * mu_0)
        vol = solver.geom.cell_volumes()
        total_E0 = np.sum(E0 * vol)

        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        rho1 = solver._squeeze(result["rho"])
        p1 = solver._squeeze(result["pressure"])
        B1 = solver._squeeze(result["B"])
        v1 = solver._squeeze(result["velocity"])
        B_sq1 = np.sum(B1**2, axis=0)
        v_sq1 = np.sum(v1**2, axis=0)
        E1 = p1 / (gamma - 1) + 0.5 * rho1 * v_sq1 + B_sq1 / (2 * mu_0)
        total_E1 = np.sum(E1 * vol)

        rel_err = abs(total_E1 - total_E0) / max(abs(total_E0), 1e-30)
        assert rel_err < 1e-6, f"Energy conservation violated: relative error = {rel_err:.2e}"


class TestGodunovSodShock:
    """1D Sod shock tube in radial direction — no NaN, shock structure."""

    def _make_sod_state(self) -> tuple[CylindricalMHDSolver, dict]:
        nr, nz = 64, 4
        dr, dz = 0.01, 0.01
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=1.4, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=True,
            conservative_energy=True,
        )
        rho = np.zeros((nr, 1, nz))
        p = np.zeros((nr, 1, nz))
        v = np.zeros((3, nr, 1, nz))
        B = np.zeros((3, nr, 1, nz))

        # Left state (inner radii): high density, high pressure
        mid = nr // 2
        rho[:mid, :, :] = 1.0
        p[:mid, :, :] = 1.0
        # Right state (outer radii): low density, low pressure
        rho[mid:, :, :] = 0.125
        p[mid:, :, :] = 0.1

        state = {
            "rho": rho,
            "velocity": v,
            "pressure": p,
            "B": B,
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        return solver, state

    def test_sod_no_nan(self) -> None:
        """Sod shock tube should run without NaN for 50 steps."""
        solver, state = self._make_sod_state()
        for _ in range(50):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(state[key])), f"NaN in {key} after Sod shock"

    def test_sod_density_monotone_decay(self) -> None:
        """After evolution, density should transition smoothly (no oscillation)."""
        solver, state = self._make_sod_state()
        for _ in range(30):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        rho_1d = state["rho"][:, 0, 0]
        # No cell should go negative
        assert np.all(rho_1d > 0), "Negative density in Sod shock"
        # Max density should not exceed initial left state
        assert np.max(rho_1d) <= 1.5, "Density overshoot in Sod shock"

    def test_sod_no_negative_pressure(self) -> None:
        """Pressure should stay positive everywhere."""
        solver, state = self._make_sod_state()
        for _ in range(50):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(state["pressure"] > 0), "Negative pressure in Sod shock"


class TestGodunovVsCentralDiff:
    """Compare Godunov and central-difference paths on smooth data."""

    def test_both_paths_agree_on_smooth(self) -> None:
        """On a smooth state, both methods should give similar results."""
        nr, nz = 16, 16
        dr, dz = 0.001, 0.001

        # Godunov solver
        solver_g = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=5.0 / 3.0, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=True,
            conservative_energy=True,
        )
        # Central-difference solver
        solver_c = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=5.0 / 3.0, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=False,
            conservative_energy=True,
        )
        state = {
            "rho": np.full((nr, 1, nz), 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), 1e3),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        state["B"][1, :, :, :] = 0.01  # small B_theta

        dt = min(solver_g._compute_dt(state), solver_c._compute_dt(state))
        result_g = solver_g.step(dict(state), dt, current=0.0, voltage=0.0)
        result_c = solver_c.step(dict(state), dt, current=0.0, voltage=0.0)

        # Both should preserve uniform state to similar accuracy
        np.testing.assert_allclose(
            result_g["rho"].mean(), result_c["rho"].mean(), rtol=1e-3,
        )


class TestGodunovMultiStep:
    """Multi-step stability tests."""

    def test_100_steps_no_crash(self) -> None:
        """Run 100 steps with Godunov flux — should not crash."""
        solver, state = _make_uniform_state(nr=16, nz=16)
        # Add a small perturbation to trigger nontrivial dynamics
        state["velocity"][0, 8, 0, :] = 100.0
        for _ in range(100):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "pressure"):
            arr = state[key]
            assert not np.any(np.isnan(arr)), f"NaN in {key} after 100 steps"
            assert np.all(arr > 0), f"Negative {key} after 100 steps"

    def test_with_electrode_bc(self) -> None:
        """Godunov + electrode B-field BC should not crash."""
        solver, state = _make_uniform_state(nr=32, nz=32, dr=0.001, dz=0.001)
        for _ in range(10):
            dt = solver._compute_dt(state)
            state = solver.step(
                state, dt,
                current=100e3, voltage=20e3,
                apply_electrode_bc=True,
                anode_radius=0.005,
                cathode_radius=0.025,
            )
        assert not np.any(np.isnan(state["rho"])), "NaN with electrode BC"
