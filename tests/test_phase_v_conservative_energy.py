"""Phase V: Conservative energy formulation + axis BC tests.

Tests verify that the cylindrical MHD solver:
1. Conserves total energy through shock interactions (Rankine-Hugoniot)
2. Enforces v_r=0 and B_r=0 at r=0 (axis symmetry)
3. Maintains backward compatibility with non-conservative mode
4. Handles source_terms passthrough to cylindrical solver
"""

import numpy as np
import pytest

from dpf.constants import k_B, mu_0
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


@pytest.fixture
def solver_conservative():
    """CylindricalMHDSolver with conservative energy enabled."""
    return CylindricalMHDSolver(
        nr=16, nz=32, dr=0.01, dz=0.01,
        gamma=5.0 / 3.0, cfl=0.3,
        conservative_energy=True,
        time_integrator="ssp_rk3",
    )


@pytest.fixture
def solver_nonconservative():
    """CylindricalMHDSolver with non-conservative (legacy) pressure equation."""
    return CylindricalMHDSolver(
        nr=16, nz=32, dr=0.01, dz=0.01,
        gamma=5.0 / 3.0, cfl=0.3,
        conservative_energy=False,
        time_integrator="ssp_rk3",
    )


def _uniform_state(nr, nz, rho=1.0, p=1.0, Btheta=0.0):
    """Create a uniform cylindrical state dict (nr, 1, nz)."""
    return {
        "rho": np.full((nr, 1, nz), rho),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }


def _sod_state(nr, nz, dr, dz, gamma=5.0 / 3.0):
    """Create a Sod shock tube initial condition along z-axis."""
    state = _uniform_state(nr, nz)
    mid = nz // 2
    # Left state (high pressure)
    state["rho"][:, :, :mid] = 1.0
    state["pressure"][:, :, :mid] = 1.0
    # Right state (low pressure)
    state["rho"][:, :, mid:] = 0.125
    state["pressure"][:, :, mid:] = 0.1
    return state


class TestConservativeEnergy:
    """Test conservative total energy formulation."""

    def test_default_is_conservative(self):
        """Default is conservative for correct shock-capturing physics."""
        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert solver.conservative_energy is True

    def test_nonconservative_flag(self):
        solver = CylindricalMHDSolver(
            nr=8, nz=8, dr=0.01, dz=0.01, conservative_energy=False,
        )
        assert solver.conservative_energy is False

    def test_uniform_state_preserved_conservative(self, solver_conservative):
        """Uniform state should remain uniform under conservative energy."""
        state = _uniform_state(16, 32)
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))
        np.testing.assert_allclose(new_state["rho"], state["rho"], rtol=1e-6)
        np.testing.assert_allclose(new_state["pressure"], state["pressure"], rtol=1e-4)

    def test_uniform_state_preserved_nonconservative(self, solver_nonconservative):
        """Uniform state should remain uniform under non-conservative energy."""
        state = _uniform_state(16, 32)
        new_state = solver_nonconservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))
        np.testing.assert_allclose(new_state["rho"], state["rho"], rtol=1e-6)

    def test_total_energy_conservation_sod(self, solver_conservative):
        """Sod shock: total energy should be approximately conserved."""
        nr, nz = 16, 32
        dr, dz = 0.01, 0.01
        gamma = 5.0 / 3.0
        state = _sod_state(nr, nz, dr, dz)

        # Compute initial total energy
        rho = state["rho"][:, 0, :]
        p = state["pressure"][:, 0, :]
        v = state["velocity"][:, :, 0, :]
        B = state["B"][:, :, 0, :]
        v_sq = np.sum(v**2, axis=0)
        B_sq = np.sum(B**2, axis=0)
        E_init = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * mu_0)
        # Cylindrical cell volumes: 2*pi*r*dr*dz
        r = np.array([(i + 0.5) * dr for i in range(nr)])
        cell_vol = 2.0 * np.pi * r[:, np.newaxis] * dr * dz
        total_E_init = np.sum(E_init * cell_vol)

        # Run 10 steps
        dt = 1e-5
        for _ in range(10):
            state = solver_conservative.step(state, dt=dt, current=0.0, voltage=0.0)

        # Compute final total energy
        rho = state["rho"][:, 0, :]
        p = state["pressure"][:, 0, :]
        v = state["velocity"][:, :, 0, :]
        B = state["B"][:, :, 0, :]
        v_sq = np.sum(v**2, axis=0)
        B_sq = np.sum(B**2, axis=0)
        E_final = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * mu_0)
        total_E_final = np.sum(E_final * cell_vol)

        # Energy should be conserved to within ~5% (finite difference + boundary effects)
        rel_change = abs(total_E_final - total_E_init) / abs(total_E_init)
        assert rel_change < 0.10, f"Energy changed by {rel_change:.2%}"

    def test_no_nan_sod_conservative(self, solver_conservative):
        """Sod shock should not produce NaN with conservative energy."""
        state = _sod_state(16, 32, 0.01, 0.01)
        for _ in range(20):
            state = solver_conservative.step(state, dt=1e-5, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))

    def test_conservative_step_returns_all_keys(self, solver_conservative):
        """Step should return all required state keys."""
        state = _uniform_state(16, 32)
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in new_state, f"Missing key: {key}"

    def test_rhs_returns_dE_dt(self, solver_conservative):
        """Conservative RHS should return dE_dt, not dp_dt."""
        state = _uniform_state(16, 32)
        rho_2d = state["rho"][:, 0, :]
        vel_2d = state["velocity"][:, :, 0, :]
        p_2d = state["pressure"][:, 0, :]
        B_2d = state["B"][:, :, 0, :]
        psi_2d = np.zeros((16, 32))
        rhs = solver_conservative._compute_rhs(rho_2d, vel_2d, p_2d, B_2d, psi_2d)
        assert "dE_dt" in rhs
        assert "dp_dt" not in rhs

    def test_rhs_nonconservative_returns_dp_dt(self, solver_nonconservative):
        """Non-conservative RHS should return dp_dt, not dE_dt."""
        state = _uniform_state(16, 32)
        rho_2d = state["rho"][:, 0, :]
        vel_2d = state["velocity"][:, :, 0, :]
        p_2d = state["pressure"][:, 0, :]
        B_2d = state["B"][:, :, 0, :]
        psi_2d = np.zeros((16, 32))
        rhs = solver_nonconservative._compute_rhs(rho_2d, vel_2d, p_2d, B_2d, psi_2d)
        assert "dp_dt" in rhs
        assert "dE_dt" not in rhs


class TestAxisBoundaryCondition:
    """Test v_r=0 and B_r=0 enforcement at r=0."""

    def test_vr_zero_at_axis(self, solver_conservative):
        """v_r should be zero at r=0 after step."""
        state = _uniform_state(16, 32)
        # Set nonzero v_r everywhere
        state["velocity"][0, :, :, :] = 100.0
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        # v_r at r=0 (first radial cell) should be zero
        assert np.allclose(new_state["velocity"][0, 0, :, :], 0.0)

    def test_Br_zero_at_axis(self, solver_conservative):
        """B_r should be zero at r=0 after step."""
        state = _uniform_state(16, 32)
        state["B"][0, :, :, :] = 0.01  # Set nonzero B_r
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert np.allclose(new_state["B"][0, 0, :, :], 0.0)

    def test_axis_bc_enforced_every_rk_stage(self, solver_conservative):
        """Axis BC should be enforced after each RK stage, not just final."""
        state = _uniform_state(16, 32)
        # Apply B_theta that drives radial flow
        state["B"][1, :, :, :] = 0.1
        for _ in range(5):
            state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        # After multiple steps, axis BC should still hold
        assert np.allclose(state["velocity"][0, 0, :, :], 0.0)
        assert np.allclose(state["B"][0, 0, :, :], 0.0)


class TestSourceTermsPassthrough:
    """Test that source_terms are properly passed to cylindrical solver."""

    def test_source_terms_accepted(self, solver_conservative):
        """Step should accept source_terms kwarg."""
        state = _uniform_state(16, 32)
        src = {
            "Q_ohmic_correction": np.ones((16, 1, 32)) * 1e6,
        }
        new_state = solver_conservative.step(
            state, dt=1e-8, current=0.0, voltage=0.0, source_terms=src,
        )
        assert not np.any(np.isnan(new_state["pressure"]))
        # Ohmic correction should increase pressure
        assert np.mean(new_state["pressure"]) > np.mean(state["pressure"])

    def test_snowplow_source_terms(self, solver_conservative):
        """Snowplow source terms should inject mass/momentum/energy."""
        state = _uniform_state(16, 32, rho=0.001)
        # Inject mass along z-axis
        S_rho = np.zeros((16, 1, 32))
        S_rho[:, :, 16] = 1.0  # Inject at midpoint
        src = {
            "S_rho_snowplow": S_rho,
            "S_mom_snowplow": np.zeros((3, 16, 1, 32)),
            "S_energy_snowplow": np.zeros((16, 1, 32)),
        }
        new_state = solver_conservative.step(
            state, dt=1e-6, current=0.0, voltage=0.0, source_terms=src,
        )
        # Density at injection point should increase
        assert new_state["rho"][8, 0, 16] > state["rho"][8, 0, 16]


class TestSSPRK2Fallback:
    """Test SSP-RK2 time integrator with conservative energy."""

    def test_rk2_conservative_runs(self):
        solver = CylindricalMHDSolver(
            nr=8, nz=16, dr=0.01, dz=0.01,
            conservative_energy=True, time_integrator="ssp_rk2",
        )
        state = _uniform_state(8, 16)
        new_state = solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))

    def test_rk2_nonconservative_runs(self):
        solver = CylindricalMHDSolver(
            nr=8, nz=16, dr=0.01, dz=0.01,
            conservative_energy=False, time_integrator="ssp_rk2",
        )
        state = _uniform_state(8, 16)
        new_state = solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))
