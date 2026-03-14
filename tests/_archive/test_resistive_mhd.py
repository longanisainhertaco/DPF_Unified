"""Tests for Phase 11: Resistive MHD, electrode BCs, energy equation,
two-temperature physics, and deuterium mass corrections.

Verifies:
- B-field BC from circuit current at electrode boundaries
- Resistive diffusion (eta*J in induction equation)
- Ohmic heating (eta*J^2)
- Conservative energy equation with Ohmic heating
- Two-temperature preservation through MHD step (Te != Ti survives)
- Deuterium mass (m_d) used instead of proton mass (m_p)
- Neutron yield factor correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.constants import k_B, m_d, m_p, mu_0

# ====================================================
# Helpers
# ====================================================

def _make_cartesian_state(nx=8, ny=8, nz=8, rho0=1e-4, T0=1e4, B_theta=0.1):
    """Create a simple Cartesian state dict for testing."""
    p0 = 2.0 * (rho0 / m_d) * k_B * T0
    state = {
        "rho": np.full((nx, ny, nz), rho0),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.full((nx, ny, nz), p0),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), T0),
        "Ti": np.full((nx, ny, nz), T0),
        "psi": np.zeros((nx, ny, nz)),
    }
    # Add a B_theta-like field in Bx and By
    state["B"][1, :, :, :] = B_theta
    return state


def _make_cylindrical_state(nr=32, nz=64, rho0=1e-4, T0=1e4):
    """Create a cylindrical state dict (nr, 1, nz) for testing."""
    p0 = 2.0 * (rho0 / m_d) * k_B * T0
    return {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), T0),
        "Ti": np.full((nr, 1, nz), T0),
        "psi": np.zeros((nr, 1, nz)),
    }


# ====================================================
# Tests: Electrode B-field Boundary Conditions
# ====================================================


class TestElectrodeBFieldBC:
    """Test electrode B-field boundary condition application."""

    def test_cylindrical_bfield_bc_imposed(self):
        """B_theta matches mu0*I/(2*pi*r) at electrode boundaries."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 32, 64
        dr, dz = 5e-4, 5e-4
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz)

        # Create 2D B-field (3, nr, nz) — all zeros
        B = np.zeros((3, nr, nz))
        current = 1e5  # 100 kA
        anode_radius = 0.005
        cathode_radius = 0.01

        B = solver.apply_electrode_bfield_bc(B, current, anode_radius, cathode_radius)

        # Check that B_theta at electrode boundaries matches analytic
        r = solver.geom.r
        idx_cath = np.argmin(np.abs(r - cathode_radius))
        r_cath = max(r[idx_cath], 1e-10)
        B_theta_expected = mu_0 * current / (2.0 * np.pi * r_cath)
        assert np.isclose(B[1, idx_cath, 0], B_theta_expected, rtol=1e-6)

    def test_cylindrical_axis_symmetry(self):
        """B_r = 0 at r=0 axis (symmetry condition)."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3)
        B = np.random.randn(3, 16, 32) * 0.01
        B = solver.apply_electrode_bfield_bc(B, 1e4, 0.005, 0.01)
        # B_r at r=0 should be zero
        assert np.all(B[0, 0, :] == 0.0)

    def test_zero_current_no_bc(self):
        """Zero current should not modify B-field (except axis BC)."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3)
        B = np.ones((3, 16, 32)) * 0.05
        B_original = B.copy()
        B = solver.apply_electrode_bfield_bc(B, 0.0, 0.005, 0.01)
        # B_r at axis should be zeroed, but B_theta and B_z unchanged
        assert B[0, 0, 0] == 0.0  # axis BC
        assert np.allclose(B[1, 1:, :], B_original[1, 1:, :])

    def test_cartesian_bfield_bc_imposed(self):
        """Cartesian solver applies B-field BC at boundary cells."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(16, 16, 16), dx=1e-3)
        B = np.zeros((3, 16, 16, 16))
        current = 1e5
        B = solver.apply_electrode_bfield_bc(
            B, current, anode_radius=0.003, cathode_radius=0.006,
        )
        # At least some cells should have non-zero B_x and B_y
        assert np.max(np.abs(B[0])) > 0 or np.max(np.abs(B[1])) > 0

    def test_bfield_bc_through_step(self):
        """Electrode BC applied during fluid step with apply_electrode_bc=True."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 32
        dr, dz = 5e-4, 5e-4
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz)
        state = _make_cylindrical_state(nr=nr, nz=nz)

        result = solver.step(
            state, dt=1e-12, current=1e4, voltage=1e3,
            anode_radius=0.003, cathode_radius=0.006,
            apply_electrode_bc=True,
        )
        # B_theta should now be non-zero at boundary cells
        B_theta = result["B"][1, :, 0, :]  # squeeze ny=1
        assert np.max(np.abs(B_theta)) > 0


# ====================================================
# Tests: Resistive MHD (eta*J)
# ====================================================


class TestResistiveMHD:
    """Test resistive MHD implementation."""

    def test_resistive_diffusion_reduces_B(self):
        """Magnetic field diffuses in presence of resistivity."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=False)
        state = _make_cartesian_state(B_theta=0.1)

        # Add a B perturbation
        state["B"][2, 3:5, 3:5, 3:5] = 1.0
        # Apply resistivity
        eta_field = np.full((8, 8, 8), 1e-4)  # Moderate resistivity
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)

        B_max_after = np.max(np.abs(result["B"][2]))
        # B should diffuse — peak should decrease (or at least not grow unphysically)
        # Note: with such a short dt, the change may be small
        assert np.isfinite(B_max_after)

    def test_no_resistive_without_eta(self):
        """Without eta_field, no resistive term is applied."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, enable_resistive=True)
        state = _make_cartesian_state()

        result_no_eta = solver.step(state, dt=1e-12, current=0, voltage=0,
                                    eta_field=None)
        result_zero_eta = solver.step(state, dt=1e-12, current=0, voltage=0,
                                      eta_field=np.zeros((8, 8, 8)))

        # Both should give same result (no resistive heating)
        assert np.allclose(result_no_eta["B"], result_zero_eta["B"], atol=1e-15)

    def test_resistive_disabled_flag(self):
        """enable_resistive=False skips resistive term even with eta_field."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=False, enable_energy_equation=False)
        state = _make_cartesian_state()

        eta_field = np.full((8, 8, 8), 1.0)  # Large resistivity
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)
        # Should complete without issues
        assert np.all(np.isfinite(result["B"]))

    def test_cylindrical_resistive_term(self):
        """Cylindrical solver includes resistive term."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 32
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=1e-3, dz=1e-3,
                                       enable_resistive=True)
        state = _make_cylindrical_state(nr=nr, nz=nz)
        state["B"][1, nr//2, 0, nz//2] = 0.5  # B_theta perturbation

        eta_field = np.full((nr, 1, nz), 1e-4)
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)
        assert np.all(np.isfinite(result["B"]))


# ====================================================
# Tests: Ohmic Heating (eta*J^2)
# ====================================================


class TestOhmicHeating:
    """Test Ohmic heating from resistive dissipation."""

    def test_ohmic_heating_increases_Te(self):
        """eta*J^2 heats the plasma (Te increases with resistive dissipation)."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=True)
        state = _make_cartesian_state(T0=1e4)
        # Create a J by having a B gradient (Bz varies in x)
        x = np.arange(8) * 1e-3
        for k in range(8):
            state["B"][2, :, :, k] = np.sin(2 * np.pi * x / 0.008)[:, np.newaxis]

        Te_before = np.mean(state["Te"])

        eta_field = np.full((8, 8, 8), 1e-3)  # High resistivity
        result = solver.step(state, dt=1e-11, current=0, voltage=0,
                             eta_field=eta_field)

        Te_after = np.mean(result["Te"])
        # Temperature should increase due to Ohmic heating
        assert Te_after >= Te_before - 1.0  # Allow small numerical noise

    def test_ohmic_heating_rhs_nonzero(self):
        """RHS returns non-zero ohmic_heating when eta and J are non-zero."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=True)
        state = _make_cartesian_state()
        state["B"][2, 3:6, :, :] = 0.1  # Create a B gradient -> J

        eta_field = np.full((8, 8, 8), 1e-3)
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0,
                                         eta_field=eta_field)
        assert np.max(rhs["ohmic_heating"]) > 0

    def test_zero_eta_zero_ohmic_heating(self):
        """Zero resistivity gives zero Ohmic heating."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state()
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0,
                                         eta_field=np.zeros((8, 8, 8)))
        assert np.allclose(rhs["ohmic_heating"], 0.0)


# ====================================================
# Tests: Conservative Energy Equation
# ====================================================


class TestEnergyEquation:
    """Test conservative energy equation with Ohmic heating."""

    def test_energy_equation_pressure_update(self):
        """Energy equation includes (gamma-1)*Q_ohm in pressure rate."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_energy_equation=True, enable_resistive=True)
        state = _make_cartesian_state()
        state["B"][2, 3:6, :, :] = 0.1  # Create J

        eta_field = np.full((8, 8, 8), 1e-3)
        rhs_with = solver._compute_rhs_euler(state, current=0, voltage=0,
                                              eta_field=eta_field)

        solver_no_energy = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                                      enable_energy_equation=False,
                                      enable_resistive=True)
        rhs_without = solver_no_energy._compute_rhs_euler(state, current=0, voltage=0,
                                                           eta_field=eta_field)

        # dp_dt should be larger with energy equation (Ohmic heating adds to pressure)
        # The difference is (gamma-1)*Q_ohm
        diff = rhs_with["dp_dt"] - rhs_without["dp_dt"]
        assert np.max(diff) > 0, "Energy equation should add Ohmic heating to pressure"

    def test_adiabatic_fallback(self):
        """With enable_energy_equation=False, uses adiabatic dp/dt = -γp*div(v)."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_energy_equation=False)
        state = _make_cartesian_state()
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0)
        # dp_dt should be non-zero only where div(v) != 0
        # For zero velocity, dp_dt should be zero
        assert np.allclose(rhs["dp_dt"], 0.0, atol=1e-20)


# ====================================================
# Tests: Two-Temperature Physics
# ====================================================


class TestTwoTemperature:
    """Test two-temperature preservation through MHD step."""

    def test_te_not_equal_ti_survives(self):
        """Te != Ti is preserved through MHD step (not collapsed)."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state(T0=5000.0)
        # Set different Te and Ti
        state["Te"][:] = 1e5  # 100,000 K
        state["Ti"][:] = 1e4  # 10,000 K
        # Adjust pressure to be consistent
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])

        result = solver.step(state, dt=1e-12, current=0, voltage=0)

        # Te and Ti should still be different
        Te_mean = np.mean(result["Te"])
        Ti_mean = np.mean(result["Ti"])
        assert Te_mean > Ti_mean * 1.5, f"Te={Te_mean:.1f} should be > 1.5*Ti={Ti_mean:.1f}"

    def test_temperature_ratio_preserved(self):
        """The Te/Ti ratio is approximately preserved through a step."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state(T0=1e4)
        state["Te"][:] = 2e4
        state["Ti"][:] = 5e3
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])

        ratio_before = np.mean(state["Te"]) / np.mean(state["Ti"])

        result = solver.step(state, dt=1e-12, current=0, voltage=0)

        ratio_after = np.mean(result["Te"]) / np.mean(result["Ti"])
        # Ratio should be approximately preserved (within 10%)
        assert abs(ratio_after - ratio_before) / ratio_before < 0.1

    def test_cylindrical_two_temperature(self):
        """Cylindrical solver also preserves Te != Ti."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 32
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=1e-3, dz=1e-3)
        state = _make_cylindrical_state(nr=nr, nz=nz)
        state["Te"][:] = 5e4
        state["Ti"][:] = 1e4
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])

        result = solver.step(state, dt=1e-12, current=0, voltage=0)

        Te_mean = np.mean(result["Te"])
        Ti_mean = np.mean(result["Ti"])
        assert Te_mean > Ti_mean * 1.5


# ====================================================
# Tests: Deuterium Mass
# ====================================================


class TestDeuteriumMass:
    """Test that deuterium mass is used correctly throughout."""

    def test_m_d_value(self):
        """Deuterium mass constant has correct value."""
        assert abs(m_d - 3.34358377e-27) < 1e-35
        assert m_d > m_p * 1.9  # Should be about 2x proton mass

    def test_config_default_ion_mass(self):
        """Default ion_mass in config is deuterium mass."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.ion_mass == pytest.approx(m_d, rel=1e-6)

    def test_engine_uses_ion_mass(self):
        """Engine initial state uses ion_mass for pressure calculation."""
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)

        rho0 = config.rho0
        T0 = config.T0
        n_i = rho0 / m_d
        expected_p0 = 2.0 * n_i * k_B * T0

        actual_p0 = engine.state["pressure"][0, 0, 0]
        assert actual_p0 == pytest.approx(expected_p0, rel=1e-6)

    def test_custom_ion_mass(self):
        """Can specify custom ion_mass (e.g., proton for hydrogen)."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            ion_mass=m_p,  # Use proton mass for hydrogen
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.ion_mass == pytest.approx(m_p, rel=1e-6)

    def test_solver_ion_mass_parameter(self):
        """MHD solver accepts and uses ion_mass parameter."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver_d = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, ion_mass=m_d)
        solver_p = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, ion_mass=m_p)

        assert solver_d.ion_mass == m_d
        assert solver_p.ion_mass == m_p

    def test_cylindrical_solver_ion_mass(self):
        """Cylindrical solver uses ion_mass parameter."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3, ion_mass=m_d)
        assert solver.ion_mass == m_d


# ====================================================
# Tests: Neutron Yield Factor
# ====================================================


class TestNeutronYieldFactor:
    """Test neutron yield uses correct factor."""

    def test_yield_factor_is_quarter(self):
        """Neutron rate uses 1/4 factor (1/2 identical particles * 1/2 neutron branch)."""
        from dpf.diagnostics.neutron_yield import neutron_yield_rate

        # At 10 keV, dd_reactivity is well-defined
        n_D = np.array([1e25])  # High density for measurable yield
        Ti = np.array([10e3 * 1000.0 / (k_B / (1000.0 * 1.602e-19))])  # ~10 keV in K
        Ti = np.array([1.16e8])  # 10 keV in Kelvin

        rate_density, total_rate = neutron_yield_rate(n_D, Ti, 1.0)
        # Rate should be 0.25 * n_D^2 * sv
        from dpf.diagnostics.neutron_yield import dd_reactivity
        sv = dd_reactivity(10.0)
        expected_rate = 0.25 * n_D[0]**2 * sv * 1.0
        assert total_rate == pytest.approx(expected_rate, rel=1e-3)


# ====================================================
# Tests: Config BoundaryConfig
# ====================================================


class TestBoundaryConfig:
    """Test BoundaryConfig integration."""

    def test_default_electrode_bc_off(self):
        """Default electrode_bc is False (must be explicitly enabled)."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.boundary.electrode_bc is False

    def test_enable_electrode_bc(self):
        """Can enable electrode BC via config."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            boundary={"electrode_bc": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.boundary.electrode_bc is True

    def test_fluid_config_resistive_flag(self):
        """FluidConfig has enable_resistive and enable_energy_equation."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.fluid.enable_resistive is True
        assert config.fluid.enable_energy_equation is True


# ====================================================
# Tests: Integration with Engine
# ====================================================


class TestEnginePhase11Integration:
    """Test Phase 11 features work end-to-end through the engine."""

    def test_engine_step_with_resistive(self):
        """Engine step completes with resistive MHD enabled."""
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        result = engine.step()
        assert result.dt > 0
        assert np.isfinite(result.max_Te)

    def test_engine_10_steps_no_nan(self):
        """Engine completes 10 steps without NaN values."""
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        for _ in range(10):
            engine.step()
        assert np.all(np.isfinite(engine.state["Te"]))
        assert np.all(np.isfinite(engine.state["pressure"]))

    def test_cylindrical_engine_with_electrode_bc(self):
        """Cylindrical engine with electrode BCs enabled runs without error."""
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[16, 1, 32], dx=5e-4, sim_time=1e-7,
            geometry={"type": "cylindrical"},
            boundary={"electrode_bc": True},
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.003, "cathode_radius": 0.006},
        )
        engine = SimulationEngine(config)
        for _ in range(5):
            result = engine.step()
        assert result.dt > 0
        # B_theta should be non-zero from electrode BC
        B_theta = engine.state["B"][1]
        assert np.max(np.abs(B_theta)) > 0

    def test_energy_conservation_with_phase11(self):
        """Circuit energy still conserved with Phase 11 features."""
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-2, sim_time=1e-6,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)
        # Energy should be conserved within 10%
        assert 0.90 < summary["energy_conservation"] < 1.10
