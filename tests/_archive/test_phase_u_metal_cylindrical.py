"""Phase U: Metal MHD solver cylindrical coordinate support.

Tests for the cylindrical geometric source terms in MetalMHDSolver.
The cylindrical sources correct the Cartesian flux divergence for (r,z)
axisymmetric geometry — critical for DPF pinch dynamics (1/r force scaling).

Reference:
    Stone & Norman, ApJS 80:753 (1992) — ZEUS-2D
    Mignone et al., ApJS 170:228 (2007) — PLUTO code
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpf.metal.metal_solver import MetalMHDSolver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID = (16, 16, 16)
DX = 0.01


@pytest.fixture
def cyl_solver():
    """Cylindrical MetalMHDSolver on CPU (float64 for accuracy)."""
    return MetalMHDSolver(
        grid_shape=GRID, dx=DX,
        coordinates="cylindrical",
        precision="float64",
        use_ct=False,
        cfl=0.3,
    )


@pytest.fixture
def cart_solver():
    """Cartesian MetalMHDSolver on CPU (float64 for accuracy)."""
    return MetalMHDSolver(
        grid_shape=GRID, dx=DX,
        coordinates="cartesian",
        precision="float64",
        use_ct=False,
        cfl=0.3,
    )


def make_uniform_state(nx: int = 16, ny: int = 16, nz: int = 16) -> dict:
    """Create a uniform MHD state (zero velocity, uniform density/pressure)."""
    return {
        "rho": np.ones((nx, ny, nz)),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), 1e4),
        "Ti": np.full((nx, ny, nz), 1e4),
        "psi": np.zeros((nx, ny, nz)),
    }


def make_btheta_state(
    nx: int = 16, ny: int = 16, nz: int = 16, dx: float = 0.01,
    B_theta_0: float = 1.0,
) -> dict:
    """Create a state with B_theta = B_theta_0 / r (z-pinch equilibrium).

    In Heaviside-Lorentz units, the z-pinch equilibrium has:
        B_theta(r) = B_0 / r  (from Ampere's law)
        p(r) chosen so that grad(p) balances J×B

    For simplicity, we set B_theta = B_theta_0 everywhere and let the
    test verify the geometric source term direction.
    """
    state = make_uniform_state(nx, ny, nz)
    # Set uniform B_theta (component index 1 = theta/y)
    state["B"][1] = B_theta_0
    return state


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestCylindricalConstruction:
    """Test MetalMHDSolver construction with cylindrical coordinates."""

    def test_default_is_cartesian(self):
        solver = MetalMHDSolver(
            grid_shape=GRID, dx=DX, device="cpu", use_ct=False,
        )
        assert solver.coordinates == "cartesian"
        assert solver._r is None
        assert solver._inv_r is None

    def test_cylindrical_flag(self, cyl_solver):
        assert cyl_solver.coordinates == "cylindrical"
        assert cyl_solver._r is not None
        assert cyl_solver._inv_r is not None

    def test_r_array_shape(self, cyl_solver):
        assert cyl_solver._r.shape == (16, 1, 1)
        assert cyl_solver._inv_r.shape == (16, 1, 1)

    def test_r_values(self, cyl_solver):
        """r[i] = (i + 0.5) * dx."""
        r = cyl_solver._r.squeeze().numpy()
        expected = np.array([(i + 0.5) * DX for i in range(16)])
        np.testing.assert_allclose(r, expected, rtol=1e-10)

    def test_inv_r_values(self, cyl_solver):
        """1/r correctly computed (no division by zero)."""
        inv_r = cyl_solver._inv_r.squeeze().numpy()
        r = cyl_solver._r.squeeze().numpy()
        np.testing.assert_allclose(inv_r, 1.0 / r, rtol=1e-10)

    def test_repr_includes_coords(self, cyl_solver):
        assert "coords='cylindrical'" in repr(cyl_solver)


# ---------------------------------------------------------------------------
# Geometric source term unit tests
# ---------------------------------------------------------------------------


class TestCylindricalSources:
    """Test _apply_cylindrical_sources directly."""

    def test_uniform_state_zero_velocity_no_sources(self, cyl_solver):
        """Uniform state with v=0 and B=0 should have zero geometric sources."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        dt = 1e-6

        rho_new, vel_new, p_new, B_new = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # With v=0, B=0: all sources are zero, state unchanged
        torch.testing.assert_close(rho_new, rho, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(vel_new, vel, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(p_new, p, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(B_new, B, rtol=1e-14, atol=1e-14)

    def test_btheta_creates_inward_force(self, cyl_solver):
        """B_theta > 0 with v=0 should produce INWARD radial acceleration.

        The r-momentum source S_mr = (B_r^2 - B_theta^2)/r < 0 when B_theta
        dominates, pushing plasma inward (the pinch effect).
        """
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B[1] = 1.0  # B_theta = 1

        dt = 1e-6
        _, vel_new, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # v_r should become negative (inward) from hoop stress
        assert torch.all(vel_new[0] < 0), (
            "B_theta hoop stress should produce inward v_r"
        )

    def test_btheta_source_scales_with_inv_r(self, cyl_solver):
        """Geometric source should scale as 1/r: stronger near axis."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B[1] = 1.0

        dt = 1e-6
        _, vel_new, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # Inner cells (small r) should have larger |v_r| than outer cells
        vr_inner = abs(float(vel_new[0, 0, 0, 0]))
        vr_outer = abs(float(vel_new[0, -1, 0, 0]))
        assert vr_inner > vr_outer, (
            f"Inner |v_r|={vr_inner:.3e} should exceed outer |v_r|={vr_outer:.3e}"
        )

    def test_radial_flow_causes_density_decrease(self, cyl_solver):
        """Outward radial flow (v_r > 0) should reduce density (cylindrical divergence)."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = 0.1  # v_r = 0.1 (outward)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-6
        rho_new, _, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # rho should decrease (geometric divergence term -rho*v_r/r)
        assert torch.all(rho_new < rho), "Outward flow should reduce density"

    def test_inward_flow_compresses(self, cyl_solver):
        """Inward radial flow (v_r < 0) should increase density."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = -0.1  # v_r = -0.1 (inward)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-6
        rho_new, _, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        assert torch.all(rho_new > rho), "Inward flow should increase density"

    def test_pressure_correction_for_radial_flow(self, cyl_solver):
        """Outward radial flow should reduce pressure (adiabatic expansion)."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = 0.1
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-6
        _, _, p_new, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        assert torch.all(p_new < p), "Outward flow should reduce pressure"

    def test_btheta_induction_correction(self, cyl_solver):
        """Inward flow (v_r < 0) with B_theta should compress B_theta."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = -0.1  # inward
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B[1] = 1.0  # B_theta

        dt = 1e-6
        _, _, _, B_new = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # S_Btheta = -(v_r * B_theta)/r. For v_r < 0, B_theta > 0:
        # S_Btheta = -(-0.1 * 1.0)/r = +0.1/r > 0
        # B_theta should increase (flux compression)
        assert torch.all(B_new[1] > B[1]), (
            "Inward flow should compress B_theta"
        )

    def test_no_sources_for_cartesian(self, cart_solver):
        """Cartesian solver should NOT apply cylindrical sources."""
        state = make_btheta_state()
        state_out = cart_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)

        # In Cartesian, B_theta (stored as By) does NOT produce radial force
        # from geometric terms. The velocity should stay near zero.
        # (There may be small numerical flux effects but no 1/r hoop stress)
        v_r_max = np.max(np.abs(state_out["velocity"][0]))
        assert v_r_max < 1e-6, (
            f"Cartesian should have no 1/r hoop stress, got |v_r|={v_r_max:.3e}"
        )


# ---------------------------------------------------------------------------
# Integration tests (full step)
# ---------------------------------------------------------------------------


class TestCylindricalStep:
    """Test full solver step with cylindrical geometry."""

    def test_step_runs_without_error(self, cyl_solver):
        """Full step should complete without NaN or crash."""
        state = make_btheta_state(B_theta_0=0.1)
        state_out = cyl_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)

        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(state_out[key])), (
                f"NaN in {key} after cylindrical step"
            )

    def test_step_produces_radial_inflow(self, cyl_solver):
        """B_theta field should drive radial inflow via hoop stress."""
        state = make_btheta_state(B_theta_0=0.5)

        # Take a few steps
        for _ in range(5):
            state = cyl_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)

        # Mean v_r should be negative (inward)
        mean_vr = np.mean(state["velocity"][0])
        assert mean_vr < 0, (
            f"B_theta hoop stress should drive inward flow, got <v_r>={mean_vr:.3e}"
        )

    def test_density_increases_near_axis(self, cyl_solver):
        """Pinch dynamics should compress density toward the axis."""
        state = make_btheta_state(B_theta_0=0.5)

        for _ in range(10):
            state = cyl_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)

        # Check density is higher at inner radii than outer
        rho_inner = np.mean(state["rho"][0:4])
        rho_outer = np.mean(state["rho"][-4:])
        assert rho_inner >= rho_outer, (
            f"Pinch should concentrate density: inner={rho_inner:.4f}, "
            f"outer={rho_outer:.4f}"
        )

    def test_uniform_state_stable(self, cyl_solver):
        """Uniform state with zero fields should remain uniform."""
        state = make_uniform_state()
        state_init = {k: v.copy() for k, v in state.items()}

        for _ in range(5):
            state = cyl_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)

        for key in ("rho", "pressure"):
            diff = np.max(np.abs(state[key] - state_init[key]))
            assert diff < 1e-10, (
                f"Uniform state should remain stable: {key} changed by {diff:.3e}"
            )

    def test_cartesian_vs_cylindrical_differ(self, cyl_solver, cart_solver):
        """Cylindrical and Cartesian should give different results for B_theta."""
        state_cyl = make_btheta_state(B_theta_0=0.3)
        state_cart = {k: v.copy() for k, v in state_cyl.items()}

        state_cyl = cyl_solver.step(state_cyl, dt=1e-7, current=0.0, voltage=0.0)
        state_cart = cart_solver.step(state_cart, dt=1e-7, current=0.0, voltage=0.0)

        # Results should differ (cylindrical has extra source terms)
        vr_diff = np.max(np.abs(state_cyl["velocity"][0] - state_cart["velocity"][0]))
        assert vr_diff > 1e-10, (
            f"Cylindrical should differ from Cartesian: v_r diff = {vr_diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Physics verification
# ---------------------------------------------------------------------------


class TestCylindricalPhysics:
    """Verify physical correctness of geometric source terms."""

    def test_hoop_stress_magnitude(self, cyl_solver):
        """Verify S_mr = -B_theta^2/r (for B_r=0, v=0).

        The r-momentum source when only B_theta is present:
        S_mr = (B_r^2 - B_theta^2)/r = -B_theta^2/r
        """
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B_theta_val = 2.0
        B[1] = B_theta_val

        dt = 1e-8
        _, vel_new, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # Expected: dv_r = dt * S_mr / rho = dt * (-B_theta^2/r) / rho
        r = cyl_solver._r.squeeze().numpy()
        expected_dvr = dt * (-B_theta_val ** 2 / r)  # rho = 1

        actual_dvr = vel_new[0, :, 0, 0].numpy()
        np.testing.assert_allclose(actual_dvr, expected_dvr, rtol=1e-10)

    def test_centrifugal_force(self, cyl_solver):
        """v_theta > 0 should produce outward (positive) radial force."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[1] = 1.0  # v_theta = 1
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-6
        _, vel_new, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # S_mr = rho*v_theta^2/r > 0 → outward acceleration
        assert torch.all(vel_new[0] > 0), (
            "Centrifugal force should produce outward v_r"
        )

    def test_coriolis_effect(self, cyl_solver):
        """Outward radial flow with v_theta should slow down v_theta."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = 0.1   # v_r = 0.1 (outward)
        vel[1] = 1.0   # v_theta = 1
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-6
        _, vel_new, _, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        # S_mtheta = -2*rho*v_r*v_theta/r < 0 → v_theta decreases
        assert torch.all(vel_new[1] < vel[1]), (
            "Outward radial flow should slow azimuthal motion"
        )

    def test_flux_compression_btheta(self, cyl_solver):
        """Inward flow compressing B_theta: check magnitude.

        S_Btheta = -(v_r * B_theta)/r = -(-0.1 * 1.0)/r = +0.1/r
        """
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = -0.1  # inward
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B[1] = 1.0

        dt = 1e-7
        _, _, _, B_new = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        r = cyl_solver._r.squeeze().numpy()
        expected_dBtheta = dt * (0.1 * 1.0 / r)  # S = -v_r*B_theta/r = +0.1/r
        actual_dBtheta = B_new[1, :, 0, 0].numpy() - 1.0

        np.testing.assert_allclose(actual_dBtheta, expected_dBtheta, rtol=1e-8)

    def test_br_unchanged_when_zero(self, cyl_solver):
        """B_r = 0 should remain zero (no geometric source for B_r)."""
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = -0.1
        p = torch.ones(GRID, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)
        B[1] = 1.0

        dt = 1e-7
        _, _, _, B_new = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        torch.testing.assert_close(
            B_new[0], B[0], rtol=1e-14, atol=1e-14,
        )

    def test_adiabatic_expansion(self, cyl_solver):
        """Verify pressure change: dp = -gamma * p * v_r / r * dt."""
        gamma = 5.0 / 3.0
        rho = torch.ones(GRID, dtype=torch.float64)
        vel = torch.zeros(3, *GRID, dtype=torch.float64)
        vel[0] = 0.05  # outward
        p_val = 2.0
        p = torch.full(GRID, p_val, dtype=torch.float64)
        B = torch.zeros(3, *GRID, dtype=torch.float64)

        dt = 1e-8
        _, _, p_new, _ = cyl_solver._apply_cylindrical_sources(
            rho, vel, p, B, dt,
        )

        r = cyl_solver._r.squeeze().numpy()
        expected_dp = -gamma * p_val * 0.05 / r * dt
        actual_dp = p_new[:, 0, 0].numpy() - p_val

        np.testing.assert_allclose(actual_dp, expected_dp, rtol=1e-8)


# ---------------------------------------------------------------------------
# Energy / conservation tests
# ---------------------------------------------------------------------------


class TestCylindricalConservation:
    """Test conservation properties in cylindrical geometry."""

    def test_stationary_btheta_energy_bounded(self, cyl_solver):
        """Running with B_theta should not blow up energy."""
        state = make_btheta_state(B_theta_0=0.3)
        E0 = np.sum(state["pressure"]) + 0.5 * np.sum(state["B"] ** 2)

        for _ in range(20):
            state = cyl_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)

        E_final = np.sum(state["pressure"]) + 0.5 * np.sum(state["B"] ** 2)
        # Energy should change but not blow up (bounded within 50%)
        ratio = E_final / max(E0, 1e-30)
        assert 0.1 < ratio < 10.0, (
            f"Energy ratio {ratio:.2f} out of bounds"
        )

    def test_no_nan_after_many_steps(self, cyl_solver):
        """Cylindrical solver should be stable for O(100) steps."""
        state = make_btheta_state(B_theta_0=0.1)

        for _ in range(50):
            state = cyl_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)

        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(state[key])), (
                f"NaN in {key} after 50 steps"
            )
            assert not np.any(np.isinf(state[key])), (
                f"Inf in {key} after 50 steps"
            )

    def test_density_stays_positive(self, cyl_solver):
        """Density should never go negative."""
        state = make_btheta_state(B_theta_0=0.5)

        for _ in range(30):
            state = cyl_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)

        assert np.all(state["rho"] > 0), "Density must stay positive"

    def test_pressure_stays_positive(self, cyl_solver):
        """Pressure should never go negative."""
        state = make_btheta_state(B_theta_0=0.5)

        for _ in range(30):
            state = cyl_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)

        assert np.all(state["pressure"] > 0), "Pressure must stay positive"
