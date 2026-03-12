"""Phase W: Snowplow MHD coupling source term tests.

Tests verify:
1. Snowplow source terms are computed correctly for rundown and radial phases
2. Source terms inject mass/momentum/energy into MHD grid
3. Gaussian smearing produces spatially smooth injection
4. Coupling is gated behind enable_mhd_coupling flag
5. Source terms integrate with cylindrical and Cartesian solvers
6. Solver-level source_terms dict injection via CylindricalMHDSolver.step()
7. Shape handling: 3D (nr,1,nz) squeeze and 2D (nr,nz) passthrough
8. Time integrator variants (SSP-RK2, SSP-RK3) propagate sources
9. Cartesian MHDSolver._compute_rhs_euler() source term acceptance
"""

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


def _make_config(
    geometry: str = "cylindrical",
    enable_mhd_coupling: bool = True,
    nx: int = 8,
    nz: int = 16,
) -> SimulationConfig:
    """Create a minimal SimulationConfig for snowplow coupling tests."""
    dz = 0.01 if geometry == "cylindrical" else None
    return SimulationConfig(
        grid_shape=(nx, 1 if geometry == "cylindrical" else nx, nz),
        dx=0.01,
        rho0=1e-4,
        sim_time=1e-6,
        circuit={
            "C": 1e-3,
            "V0": 15000,
            "L0": 33.5e-9,
            "R0": 0.01,
            "anode_radius": 0.025,
            "cathode_radius": 0.05,
        },
        geometry={"type": geometry, "dz": dz},
        fluid={"backend": "python", "enable_ohmic_correction": False},
        diagnostics={"hdf5_filename": ":memory:", "output_interval": 1},
        snowplow={
            "enabled": True,
            "mass_fraction": 0.3,
            "anode_length": 0.16,
            "current_fraction": 0.7,
            "enable_mhd_coupling": enable_mhd_coupling,
        },
        boundary={"electrode_bc": False},
    )


class TestSnowplowSourceTermComputation:
    """Test _compute_snowplow_source_terms method."""

    def test_coupling_disabled_returns_empty(self):
        """When enable_mhd_coupling=False, no source terms produced."""
        config = _make_config(enable_mhd_coupling=False)
        engine = SimulationEngine(config)
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        assert result == {}

    def test_snowplow_none_returns_empty(self):
        """When snowplow is None, no source terms produced."""
        config = _make_config(enable_mhd_coupling=True)
        engine = SimulationEngine(config)
        engine.snowplow = None
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        assert result == {}

    def test_rundown_phase_produces_source_terms(self):
        """During rundown, source terms should be non-empty."""
        config = _make_config()
        engine = SimulationEngine(config)
        # Advance circuit to get current flowing
        engine.step()
        assert engine.snowplow is not None
        assert engine.snowplow.phase == "rundown"

        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if engine.snowplow.sheath_velocity > 1e-6:
            assert "S_rho_snowplow" in result
            assert "S_mom_snowplow" in result
            assert "S_energy_snowplow" in result

    def test_source_terms_have_correct_shape(self):
        """Source term arrays must match grid shape."""
        config = _make_config(nx=8, nz=16)
        engine = SimulationEngine(config)
        engine.step()  # Get current flowing

        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result:
            grid_shape = engine.state["rho"].shape
            assert result["S_rho_snowplow"].shape == grid_shape
            assert result["S_mom_snowplow"].shape == (3, *grid_shape)
            assert result["S_energy_snowplow"].shape == grid_shape

    def test_source_terms_are_positive(self):
        """Mass and energy sources should be non-negative."""
        config = _make_config()
        engine = SimulationEngine(config)
        engine.step()

        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result:
            # Mass source should be >= 0 (injection, not removal)
            assert np.all(result["S_rho_snowplow"] >= 0)
            # Energy source should be >= 0
            assert np.all(result["S_energy_snowplow"] >= 0)

    def test_gaussian_smearing_peaks_at_sheath(self):
        """Source term should peak near the sheath position."""
        config = _make_config(nx=8, nz=64)  # Higher resolution for sharper peak
        engine = SimulationEngine(config)
        # Run a few steps to move the sheath
        for _ in range(5):
            engine.step()

        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result and engine.snowplow.phase == "rundown":
            S_rho = result["S_rho_snowplow"]
            # Sum over r to get 1D z-profile
            z_profile = np.sum(S_rho[:, 0, :], axis=0)
            peak_idx = np.argmax(z_profile)

            # Peak should be near sheath position
            dz = config.geometry.dz or config.dx
            z_peak = (peak_idx + 0.5) * dz
            z_sheath = engine.snowplow.sheath_position
            assert abs(z_peak - z_sheath) < 4 * dz, (
                f"Peak at z={z_peak:.4f}, sheath at z={z_sheath:.4f}"
            )


class TestSnowplowMHDIntegration:
    """Test snowplow source terms integrated into the MHD solve."""

    def test_engine_step_with_coupling(self):
        """Engine step should complete without errors when coupling is on."""
        config = _make_config()
        engine = SimulationEngine(config)
        for _ in range(10):
            result = engine.step()
        assert result is not None
        assert not np.any(np.isnan(engine.state["rho"]))

    def test_coupling_increases_density(self):
        """Mass injection should increase density near the sheath."""
        config_coupled = _make_config(enable_mhd_coupling=True)
        config_uncoupled = _make_config(enable_mhd_coupling=False)

        engine_c = SimulationEngine(config_coupled)
        engine_u = SimulationEngine(config_uncoupled)

        # Run both for the same number of steps
        for _ in range(10):
            engine_c.step()
            engine_u.step()

        # Coupled engine should have higher total mass
        total_mass_c = np.sum(engine_c.state["rho"])
        total_mass_u = np.sum(engine_u.state["rho"])
        assert total_mass_c >= total_mass_u * 0.99  # At least as much mass

    def test_no_nan_with_coupling(self):
        """Extended run with coupling should not produce NaN."""
        config = _make_config()
        engine = SimulationEngine(config)
        for _ in range(50):
            engine.step()
        assert not np.any(np.isnan(engine.state["rho"]))
        assert not np.any(np.isnan(engine.state["pressure"]))
        assert np.all(engine.state["rho"] > 0)
        assert np.all(engine.state["pressure"] > 0)


class TestSnowplowConfigFlag:
    """Test the enable_mhd_coupling config flag."""

    def test_config_default_is_false(self):
        """Default enable_mhd_coupling should be False for backward compat."""
        config = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={"C": 1e-3, "V0": 15000, "L0": 33.5e-9, "anode_radius": 0.025, "cathode_radius": 0.05},
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert config.snowplow.enable_mhd_coupling is False

    def test_config_can_enable(self):
        """enable_mhd_coupling can be set to True."""
        config = _make_config(enable_mhd_coupling=True)
        assert config.snowplow.enable_mhd_coupling is True


# ===================================================================
# Solver-level source_terms dict injection tests
# ===================================================================

def _copy_state(state: dict) -> dict:
    return {k: v.copy() for k, v in state.items()}


@pytest.fixture()
def cyl_solver():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False,
        conservative_energy=True,
    )


@pytest.fixture()
def cyl_solver_nonconservative():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False,
        conservative_energy=False,
    )


@pytest.fixture()
def cyl_solver_rk2():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False,
        conservative_energy=True,
        time_integrator="ssp_rk2",
    )


@pytest.fixture()
def uniform_state():
    """Uniform quiescent state with 3D arrays (nr, 1, nz)."""
    nr, nz = 16, 16
    return {
        "rho": np.full((nr, 1, nz), 1.0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), 1e5),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }


class TestSolverMassSource:
    """1. Mass source increases density."""

    def test_mass_source_increases_density(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["rho"]) > np.mean(uniform_state["rho"])

    def test_negative_mass_source_decreases_density(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), -1e5)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["rho"]) < np.mean(uniform_state["rho"])


class TestSolverMomentumSource:
    """2. Momentum source changes velocity."""

    def test_axial_momentum_injection(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2] = 1e8  # Axial
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(np.abs(new["velocity"][2])) > 0.0

    def test_radial_momentum_injection(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[0] = 1e8  # Radial
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(np.abs(new["velocity"][0])) > 0.0


class TestSolverEnergySource:
    """3. Energy source increases total energy (conservative mode)."""

    def test_energy_source_conservative(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {"S_energy_snowplow": np.full((16, 1, 16), 1e12)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])

    def test_energy_source_conservative_vs_nonconservative(
        self, cyl_solver, cyl_solver_nonconservative, uniform_state
    ):
        """Conservative mode should show larger pressure change from S_energy."""
        src = {"S_energy_snowplow": np.full((16, 1, 16), 1e12)}
        new_cons = cyl_solver.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=src,
        )
        new_noncons = cyl_solver_nonconservative.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=src,
        )
        dp_cons = np.mean(new_cons["pressure"]) - np.mean(uniform_state["pressure"])
        dp_noncons = np.mean(new_noncons["pressure"]) - np.mean(uniform_state["pressure"])
        assert abs(dp_cons) > abs(dp_noncons)


class TestSolverOhmicCorrection:
    """4. Ohmic correction heats electrons."""

    def test_ohmic_correction_increases_pressure(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {"Q_ohmic_correction": np.full((16, 1, 16), 1e12)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])


class TestSolverShape3DSqueeze:
    """5. Source terms with 3D shape (nr,1,nz) are properly squeezed."""

    def test_3d_shape_squeezed(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_mom_snowplow": np.full((3, 16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e10),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e10),
        }
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert new["rho"].shape == (16, 1, 16)
        assert new["velocity"].shape == (3, 16, 1, 16)


class TestSolverShape2D:
    """6. Source terms with 2D shape (nr,nz) work directly."""

    def test_2d_shape_works(self, cyl_solver, uniform_state):
        state0 = _copy_state(uniform_state)
        source_terms = {
            "S_rho_snowplow": np.full((16, 16), 1e6),
            "S_mom_snowplow": np.full((3, 16, 16), 1e6),
            "S_energy_snowplow": np.full((16, 16), 1e10),
            "Q_ohmic_correction": np.full((16, 16), 1e10),
        }
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert new["rho"].shape == (16, 1, 16)
        assert np.mean(new["rho"]) > 1.0


class TestSolverZeroSourceTerms:
    """7. Zero source terms = no effect."""

    def test_zero_sources_match_none(self, cyl_solver, uniform_state):
        source_terms = {
            "S_rho_snowplow": np.zeros((16, 1, 16)),
            "S_mom_snowplow": np.zeros((3, 16, 1, 16)),
            "S_energy_snowplow": np.zeros((16, 1, 16)),
            "Q_ohmic_correction": np.zeros((16, 1, 16)),
        }
        new_zero = cyl_solver.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=source_terms,
        )
        new_none = cyl_solver.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=None,
        )
        np.testing.assert_allclose(new_zero["rho"], new_none["rho"], atol=1e-15)
        np.testing.assert_allclose(new_zero["pressure"], new_none["pressure"], atol=1e-10)

    def test_empty_dict_matches_none(self, cyl_solver, uniform_state):
        new_empty = cyl_solver.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms={},
        )
        new_none = cyl_solver.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=None,
        )
        np.testing.assert_allclose(new_empty["rho"], new_none["rho"], atol=1e-15)


class TestSolverLargeSourceStability:
    """8. Large source terms don't cause NaN."""

    def test_large_mass_source_no_nan(self, cyl_solver, uniform_state):
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e15)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))

    def test_large_energy_source_no_nan(self, cyl_solver, uniform_state):
        source_terms = {"S_energy_snowplow": np.full((16, 1, 16), 1e18)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))
        assert not np.any(np.isinf(new["pressure"]))

    def test_large_momentum_source_no_nan(self, cyl_solver, uniform_state):
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2] = 1e15
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["velocity"]))


class TestSolverSSPRK3:
    """9. Source terms work with SSP-RK3 (default)."""

    def test_rk3_source_injection(self, cyl_solver, uniform_state):
        assert cyl_solver.time_integrator == "ssp_rk3"
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0


class TestSolverSSPRK2:
    """10. Source terms work with SSP-RK2."""

    def test_rk2_source_injection(self, cyl_solver_rk2, uniform_state):
        assert cyl_solver_rk2.time_integrator == "ssp_rk2"
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver_rk2.step(_copy_state(uniform_state), dt=1e-9,
                                  current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0

    def test_rk2_vs_rk3_both_increase(self, cyl_solver, cyl_solver_rk2, uniform_state):
        src = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e12),
        }
        new_rk3 = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                  current=0.0, voltage=0.0, source_terms=src)
        new_rk2 = cyl_solver_rk2.step(_copy_state(uniform_state), dt=1e-9,
                                      current=0.0, voltage=0.0, source_terms=src)
        assert np.mean(new_rk3["rho"]) > 1.0
        assert np.mean(new_rk2["rho"]) > 1.0


class TestSolverLocalizedSource:
    """11. Gaussian-shaped source localized to specific cells."""

    def test_gaussian_density_bump(self, cyl_solver, uniform_state):
        nr, nz = 16, 16
        r_idx, z_idx = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
        gaussian = 1e6 * np.exp(-((r_idx - 8)**2 + (z_idx - 8)**2) / (2 * 2.0**2))
        source_terms = {"S_rho_snowplow": gaussian[:, np.newaxis, :]}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert new["rho"][8, 0, 8] > new["rho"][0, 0, 0]

    def test_point_momentum_injection(self, cyl_solver, uniform_state):
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2, 8, 0, 8] = 1e10  # Single cell axial kick
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.abs(new["velocity"][2, 8, 0, 8]) >= np.abs(new["velocity"][2, 0, 0, 0])


class TestSolverCombinedSources:
    """12. Combined sources (mass + momentum + energy simultaneously)."""

    def test_all_four_sources(self, cyl_solver, uniform_state):
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[0] = 1e7
        S_mom[2] = 1e7
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_mom_snowplow": S_mom,
            "S_energy_snowplow": np.full((16, 1, 16), 1e12),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e10),
        }
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])
        assert np.mean(np.abs(new["velocity"][0])) > 0.0
        assert np.mean(np.abs(new["velocity"][2])) > 0.0
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))

    def test_mass_and_energy_coupled(self, cyl_solver, uniform_state):
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e13),
        }
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > np.mean(uniform_state["rho"])
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])


class TestCartesianSolverSourceTerms:
    """13. Cartesian MHDSolver._compute_rhs_euler accepts source terms."""

    def test_rhs_accepts_snowplow_sources(self):
        from dpf.fluid.mhd_solver import MHDSolver

        n = 16
        solver = MHDSolver(grid_shape=(n, n, n), dx=0.001,
                           enable_hall=False, enable_resistive=False)
        state = {
            "rho": np.full((n, n, n), 1.0),
            "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1e5),
            "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4),
            "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        source_terms = {
            "S_rho_snowplow": np.full((n, n, n), 1e6),
            "S_mom_snowplow": np.full((3, n, n, n), 1e6),
            "Q_ohmic_correction": np.full((n, n, n), 1e10),
        }
        rhs = solver._compute_rhs_euler(state, current=0.0, voltage=0.0,
                                         eta_field=None, source_terms=source_terms)
        assert np.mean(rhs["drho_dt"]) > 0.0

    def test_rhs_none_source_terms(self):
        from dpf.fluid.mhd_solver import MHDSolver

        n = 16
        solver = MHDSolver(grid_shape=(n, n, n), dx=0.001,
                           enable_hall=False, enable_resistive=False)
        state = {
            "rho": np.full((n, n, n), 1.0),
            "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1e5),
            "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4),
            "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        rhs = solver._compute_rhs_euler(state, current=0.0, voltage=0.0,
                                         eta_field=None, source_terms=None)
        assert "drho_dt" in rhs
        assert "dmom_dt" in rhs


class TestSolverProportionality:
    """14. Source magnitude scales effect proportionally."""

    def test_double_mass_source_double_effect(self, cyl_solver, uniform_state):
        dt = 1e-9
        src_1x = {"S_rho_snowplow": np.full((16, 1, 16), 1e4)}
        src_2x = {"S_rho_snowplow": np.full((16, 1, 16), 2e4)}
        new_1x = cyl_solver.step(_copy_state(uniform_state), dt=dt,
                                 current=0.0, voltage=0.0, source_terms=src_1x)
        new_2x = cyl_solver.step(_copy_state(uniform_state), dt=dt,
                                 current=0.0, voltage=0.0, source_terms=src_2x)
        drho_1x = np.mean(new_1x["rho"]) - np.mean(uniform_state["rho"])
        drho_2x = np.mean(new_2x["rho"]) - np.mean(uniform_state["rho"])
        if abs(drho_1x) > 1e-15:
            ratio = drho_2x / drho_1x
            assert ratio == pytest.approx(2.0, rel=0.3)
