"""Tests for Phase 3: cylindrical (r,z) axisymmetric geometry.

Test categories:
1. CylindricalGeometry differential operators (div, grad, curl, laplacian)
2. Geometric source terms (hoop stress)
3. Cell volumes and face areas
4. CylindricalMHDSolver basic functionality
5. GeometryConfig validation
6. Engine cylindrical mode end-to-end
7. Bennett pinch equilibrium (J_z * B_theta = dp/dr)
"""

from __future__ import annotations

import numpy as np

from dpf.constants import mu_0, pi

# ═══════════════════════════════════════════════════════
# CylindricalGeometry Operator Tests
# ═══════════════════════════════════════════════════════

class TestCylindricalDivergence:
    """Tests for divergence in cylindrical coordinates."""

    def test_uniform_field_zero_div(self):
        """Uniform radial + axial field has div=0 only if F_r=0 and dFz/dz=0.

        div(F) = dF_r/dr + F_r/r + dF_z/dz
        For F = (0, 0, const): div = 0.
        """
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        F = np.zeros((3, 16, 16))
        F[2] = 1.0  # Uniform F_z
        div = geom.divergence(F)
        np.testing.assert_allclose(div, 0.0, atol=1e-10)

    def test_radial_1_over_r_zero_div(self):
        """For F_r = C/r (vacuum field from line charge), div ~ 0.

        d(r * C/r)/dr = d(C)/dr = 0, so (1/r) * d(r * F_r)/dr = 0.
        """
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        F = np.zeros((3, 32, 8))
        C = 1.0
        for i in range(32):
            r = geom.r[i]
            if r > 1e-10:
                F[0, i, :] = C / r
        div = geom.divergence(F)
        # Interior cells (avoid boundary artifacts)
        interior = div[4:-4, 2:-2]
        assert np.max(np.abs(interior)) < 0.5, f"max div = {np.max(np.abs(interior))}"

    def test_divergence_of_r_hat(self):
        """div(r_hat) = 1/r in cylindrical coords.

        F = (1, 0, 0): div = d(1)/dr + 1/r = 0 + 1/r = 1/r.
        """
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        F = np.zeros((3, 32, 8))
        F[0] = 1.0  # F_r = 1 everywhere

        div = geom.divergence(F)
        # Expected: 1/r (approximately, using d(r*1)/dr / r = (1 + r*0)/r... wait)
        # Actually div(F_r=1) = (1/r)*d(r*1)/dr = (1/r)*1 = 1/r
        # But the discrete version uses d(r*1)/dr ≈ 1, so div ≈ 1/r
        # Check that it's roughly 1/r for interior cells
        for i in range(4, 28):
            expected = 1.0 / geom.r[i]
            # Loose tolerance due to finite differences
            assert abs(div[i, 4] - expected) / expected < 0.2, (
                f"At r={geom.r[i]:.3f}: div={div[i,4]:.3f}, expected={expected:.3f}"
            )


class TestCylindricalGradient:
    """Tests for gradient in cylindrical coordinates."""

    def test_constant_zero_gradient(self):
        """Gradient of a constant is zero."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        p = np.full((16, 16), 100.0)
        grad = geom.gradient(p)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_linear_r_gradient(self):
        """grad(p = r) = (1, 0, 0) in cylindrical."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        p = geom.r[:, np.newaxis] * np.ones((1, 8))
        grad = geom.gradient(p)
        # dp/dr = 1 (approximately for interior cells)
        interior_grad_r = grad[0, 4:-4, :]
        np.testing.assert_allclose(interior_grad_r, 1.0, atol=0.1)
        # dp/dz = 0
        np.testing.assert_allclose(grad[2], 0.0, atol=1e-10)


class TestCylindricalCurl:
    """Tests for curl in cylindrical coordinates."""

    def test_curl_of_constant_zero(self):
        """Curl of a constant vector field is zero."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        B = np.ones((3, 16, 16))
        curl = geom.curl(B)
        # Should be approximately zero (B_theta constant -> d(r*const)/dr = const ≠ 0)
        # Actually (curl B)_z = (1/r)*d(r*B_theta)/dr for B_theta=1:
        #   = (1/r) * d(r)/dr = 1/r  (not zero!)
        # So curl of uniform B_theta = 1/r in z-direction
        # Only r and z components from theta-independent fields should be zero:
        np.testing.assert_allclose(curl[0], 0.0, atol=1e-10)  # -dB_theta/dz = 0
        np.testing.assert_allclose(curl[1], 0.0, atol=1e-10)  # dB_r/dz - dB_z/dr = 0

    def test_b_theta_from_line_current(self):
        """For B_theta = mu_0*I/(2*pi*r), curl should give J_z = I/(pi*r^2)... but actually
        the field from an infinite wire: curl(B) = 0 for r > 0.

        (curl B)_z = (1/r)*d(r * mu_0*current/(2*pi*r))/dr = 0
        """
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        current = 1000.0  # 1 kA
        B = np.zeros((3, 32, 8))
        for i in range(32):
            r = geom.r[i]
            if r > 1e-10:
                B[1, i, :] = mu_0 * current / (2.0 * pi * r)

        curl = geom.curl(B)
        # (curl B)_z should be ~0 for r > 0 (outside wire)
        interior_curl_z = curl[2, 4:-4, 2:-2]
        assert np.max(np.abs(interior_curl_z)) < 50.0, (
            f"max |curl_z| = {np.max(np.abs(interior_curl_z))}"
        )


class TestCylindricalLaplacian:
    """Tests for Laplacian in cylindrical coordinates."""

    def test_constant_zero_laplacian(self):
        """Laplacian of a constant is zero."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        T = np.full((16, 16), 100.0)
        lap = geom.laplacian(T)
        np.testing.assert_allclose(lap, 0.0, atol=1e-6)


# ═══════════════════════════════════════════════════════
# Cell Volume and Face Area Tests
# ═══════════════════════════════════════════════════════

class TestCylindricalVolumes:
    """Tests for cell volumes and face areas."""

    def test_cell_volumes_positive(self):
        """All cell volumes should be positive."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        V = geom.cell_volumes()
        assert V.shape == (16, 16)
        assert np.all(V > 0)

    def test_cell_volumes_increase_with_r(self):
        """Cell volumes should increase with radius (V ~ r * dr * dz)."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        V = geom.cell_volumes()
        # For fixed j, V[i,j] should increase with i
        for j in range(8):
            for i in range(15):
                assert V[i + 1, j] > V[i, j], f"V not increasing at i={i}, j={j}"

    def test_total_volume_matches_cylinder(self):
        """Total volume should match pi * R^2 * L for the full domain."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        nr, nz = 32, 16
        dr, dz = 0.01, 0.02
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)
        V_total = np.sum(geom.cell_volumes())

        R = nr * dr  # Outer radius
        L = nz * dz  # Length
        V_expected = pi * R**2 * L
        np.testing.assert_allclose(V_total, V_expected, rtol=1e-10)

    def test_face_areas_radial_shape(self):
        """Radial face areas should have shape (nr+1, nz)."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        A = geom.face_areas_radial()
        assert A.shape == (17, 8)

    def test_face_area_at_axis_zero(self):
        """Radial face area at r=0 (axis) should be zero."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        A = geom.face_areas_radial()
        np.testing.assert_allclose(A[0, :], 0.0, atol=1e-20)


# ═══════════════════════════════════════════════════════
# CylindricalMHDSolver Tests
# ═══════════════════════════════════════════════════════

class TestCylindricalMHDSolver:
    """Tests for the cylindrical MHD solver."""

    def _make_state(self, nr=16, nz=16):
        """Create a basic cylindrical state dict with ny=1."""
        rho0 = 1e-4
        T0 = 1e4
        p0 = (rho0 / 1.672e-27) * 1.38e-23 * T0
        return {
            "rho": np.full((nr, 1, nz), rho0),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), p0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), T0),
            "Ti": np.full((nr, 1, nz), T0),
            "psi": np.zeros((nr, 1, nz)),
        }

    def test_step_returns_correct_keys(self):
        """Solver returns all expected state keys."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in result

    def test_step_preserves_shape(self):
        """Output shapes match input (nr, 1, nz)."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 16
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=0.01, dz=0.01)
        state = self._make_state(nr, nz)
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        assert result["rho"].shape == (nr, 1, nz)
        assert result["velocity"].shape == (3, nr, 1, nz)
        assert result["B"].shape == (3, nr, 1, nz)

    def test_uniform_state_preserved(self):
        """A uniform state with zero velocity should be (nearly) preserved."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(
            nr=16, nz=16, dr=0.01, dz=0.01, enable_hall=False,
        )
        state = self._make_state()
        rho0 = state["rho"][0, 0, 0]
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        # Density should be well-preserved for uniform quiescent state
        np.testing.assert_allclose(result["rho"], rho0, rtol=1e-3)

    def test_density_stays_positive(self):
        """Density floor prevents negative values."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01, enable_hall=False)
        state = self._make_state()
        state["rho"][:] = 1e-30  # Very low density
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(result["rho"] > 0)

    def test_coupling_interface(self):
        """Solver returns valid CouplingState."""
        from dpf.core.bases import CouplingState
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        cs = solver.coupling_interface()
        assert isinstance(cs, CouplingState)

    def test_no_nan_after_step(self):
        """No NaN values after a step."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        result = solver.step(state, dt=1e-10, current=1000.0, voltage=5000.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti"]:
            assert np.all(np.isfinite(result[key])), f"NaN in {key}"

    def test_compute_dt_positive(self):
        """Computed timestep is positive."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        dt = solver._compute_dt(state)
        assert dt > 0


# ═══════════════════════════════════════════════════════
# GeometryConfig Tests
# ═══════════════════════════════════════════════════════

class TestGeometryConfig:
    """Tests for the geometry configuration."""

    def test_default_cartesian(self):
        """Default geometry is Cartesian."""
        from dpf.config import GeometryConfig

        cfg = GeometryConfig()
        assert cfg.type == "cartesian"
        assert cfg.dz is None

    def test_cylindrical_config(self):
        """Cylindrical config with explicit dz."""
        from dpf.config import GeometryConfig

        cfg = GeometryConfig(type="cylindrical", dz=0.005)
        assert cfg.type == "cylindrical"
        assert cfg.dz == 0.005

    def test_invalid_type_raises(self):
        """Invalid geometry type raises."""
        import pytest

        from dpf.config import GeometryConfig

        with pytest.raises(ValueError, match="geometry type"):
            GeometryConfig(type="spherical")

    def test_cylindrical_grid_validation(self):
        """Cylindrical geometry requires ny=1."""
        import pytest

        from dpf.config import SimulationConfig

        with pytest.raises(ValueError, match="grid_shape\\[1\\]=1"):
            SimulationConfig(
                grid_shape=[16, 4, 16],  # ny != 1 -> error
                dx=0.01,
                sim_time=1e-6,
                circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                         "anode_radius": 0.005, "cathode_radius": 0.01},
                geometry={"type": "cylindrical"},
            )

    def test_cylindrical_grid_valid(self):
        """Cylindrical geometry with ny=1 is valid."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[16, 1, 16],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            geometry={"type": "cylindrical"},
        )
        assert config.geometry.type == "cylindrical"

    def test_simulation_config_default_geometry(self):
        """SimulationConfig defaults to Cartesian geometry."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.geometry.type == "cartesian"


# ═══════════════════════════════════════════════════════
# Engine Cylindrical Mode Tests
# ═══════════════════════════════════════════════════════

class TestEngineCylindrical:
    """Tests for the simulation engine in cylindrical mode."""

    def _make_cyl_config(self):
        """Create a cylindrical simulation config."""
        from dpf.config import SimulationConfig

        return SimulationConfig(
            grid_shape=[16, 1, 16],
            dx=0.01,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": False},
        )

    def test_engine_init_cylindrical(self):
        """Engine initializes with cylindrical geometry."""
        from dpf.engine import SimulationEngine

        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        assert engine.geometry_type == "cylindrical"

    def test_engine_runs_cylindrical(self):
        """Engine completes a short run in cylindrical mode without errors."""
        from dpf.engine import SimulationEngine

        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5
        assert summary["energy_conservation"] > 0.9

    def test_engine_no_nan_cylindrical(self):
        """No NaN in state after cylindrical engine run."""
        from dpf.engine import SimulationEngine

        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        engine.run(max_steps=3)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti"]:
            assert np.all(np.isfinite(engine.state[key])), f"NaN in {key}"

    def test_engine_cartesian_unchanged(self):
        """Existing Cartesian mode still works identically."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            radiation={"bremsstrahlung_enabled": False},
        )
        engine = SimulationEngine(config)
        assert engine.geometry_type == "cartesian"
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5


# ═══════════════════════════════════════════════════════
# Bennett Pinch Equilibrium Test
# ═══════════════════════════════════════════════════════

class TestBennettPinch:
    """Test Bennett pinch equilibrium: J_z produces B_theta that confines plasma.

    In equilibrium: dp/dr = -J_z * B_theta = -(B_theta / (mu_0 * r)) * d(r * B_theta)/dr
    This is the fundamental MHD equilibrium of a z-pinch.
    """

    def test_bennett_pressure_balance(self):
        """For a z-pinch with uniform J_z, verify that the equilibrium
        pressure gradient balances the Lorentz force.

        Given: J_z = const, B_theta = mu_0 * J_z * r / 2
        Then: J × B force in r-direction = J_z * B_theta = mu_0 * J_z^2 * r / 2
        And: dp/dr = -J_z * B_theta (inward pinch)
        """
        from dpf.geometry.cylindrical import CylindricalGeometry

        nr, nz = 64, 4
        dr, dz = 0.001, 0.01
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)

        # Set up a uniform current density
        J_z = 1e8  # A/m^2 (typical for DPF pinch)

        # B_theta = mu_0 * J_z * r / 2 (from Ampere's law for uniform J_z)
        B = np.zeros((3, nr, nz))
        for i in range(nr):
            B[1, i, :] = mu_0 * J_z * geom.r[i] / 2.0

        # Compute J from curl(B) / mu_0
        curl_B = geom.curl(B)
        J_computed = curl_B / mu_0

        # J_z should be approximately J_z everywhere
        J_z_computed = J_computed[2, 4:-4, :]  # Interior cells
        np.testing.assert_allclose(
            J_z_computed, J_z, rtol=0.15,
            err_msg="Computed J_z doesn't match input J_z for uniform current",
        )

    def test_lorentz_force_direction(self):
        """The J × B force should point inward (negative r) for a z-pinch."""
        from dpf.geometry.cylindrical import CylindricalGeometry

        nr, nz = 32, 4
        dr, dz = 0.001, 0.01
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)

        J_z = 1e8
        B = np.zeros((3, nr, nz))
        for i in range(nr):
            B[1, i, :] = mu_0 * J_z * geom.r[i] / 2.0

        # J × B: only J_z × B_theta contributes
        # (J × B)_r = J_theta * B_z - J_z * B_theta = -J_z * B_theta (since J_theta=0)
        curl_B = geom.curl(B)
        J = curl_B / mu_0

        # JxB_r = J[1]*B[2] - J[2]*B[1]
        JxB_r = J[1] * B[2] - J[2] * B[1]

        # For r > 0, B_theta > 0 and J_z > 0, so JxB_r should be negative (inward)
        interior = JxB_r[4:-4, :]
        assert np.all(interior <= 0.0), "Lorentz force should point inward for z-pinch"
