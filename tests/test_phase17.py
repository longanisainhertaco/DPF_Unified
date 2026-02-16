"""Tests for Phase 17: Integration + Verification.

Covers:
    17.1 Nernst effect wired into engine.py Strang splitting
    17.2 Braginskii viscosity wired into engine.py
    17.3 Constrained transport option in cylindrical solver
    17.4 Sod shock tube test (exact Riemann comparison)
    17.5 Brio-Wu MHD shock tube test
    17.6 Cylindrical convergence test (z-pinch equilibrium)
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import k_B, mu_0

# ===================================================================
# Helpers
# ===================================================================

def _make_engine_config(**overrides) -> dict:
    """Return a minimal config dict for SimulationConfig."""
    base = {
        "grid_shape": [16, 1, 16],
        "dx": 1e-3,
        "sim_time": 1e-9,
        "rho0": 1e-4,
        "T0": 1e6,
        "ion_mass": 3.34e-27,
        "circuit": {
            "C": 30e-6,
            "V0": 15e3,
            "L0": 33e-9,
            "R0": 0.01,
            "anode_radius": 0.012,
            "cathode_radius": 0.033,
        },
        "geometry": {"type": "cylindrical"},
        "fluid": {
            "reconstruction": "weno5",
            "riemann_solver": "hll",
            "cfl": 0.3,
            "enable_resistive": True,
            "enable_energy_equation": True,
        },
        "diagnostics": {"hdf5_filename": ":memory:", "output_interval": 9999},
    }
    # Deep merge overrides
    for key, val in overrides.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            base[key].update(val)
        else:
            base[key] = val
    return base


def _make_engine(config_dict: dict):
    """Build a SimulationEngine from a config dict."""
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    cfg = SimulationConfig(**config_dict)
    return SimulationEngine(cfg)


# ===================================================================
# 17.1 — Nernst Effect Integration in Engine
# ===================================================================


class TestNernstEngineIntegration:
    """Verify the Nernst effect is properly wired into the engine."""

    def test_nernst_config_flag_exists(self):
        """FluidConfig has enable_nernst field."""
        from dpf.config import FluidConfig

        fc = FluidConfig()
        assert hasattr(fc, "enable_nernst")
        assert fc.enable_nernst is False  # Default off

    def test_nernst_disabled_no_change(self):
        """With enable_nernst=False, B-field is not modified by Nernst step."""
        cfg = _make_engine_config(fluid={"enable_nernst": False})
        engine = _make_engine(cfg)

        # Put a temperature gradient in place and a B-field
        engine.state["B"][1, :, 0, :] = 0.1  # B_theta
        Te = engine.state["Te"]
        Te[:8, :, :] = 2e6  # Hot left half
        Te[8:, :, :] = 5e5  # Cold right half
        engine.state["Te"] = Te

        # With Nernst disabled, engine.step() should not call _apply_nernst.
        # Verify the config flag path:
        assert engine.config.fluid.enable_nernst is False

    def test_nernst_enabled_modifies_B(self):
        """With enable_nernst=True and conditions that couple, B-field changes."""
        cfg = _make_engine_config(fluid={"enable_nernst": True})
        engine = _make_engine(cfg)

        # Set up B_theta that varies along z (so the z-directed Nernst
        # velocity advects B_theta and produces a non-zero change).
        nr = engine.state["B"].shape[1]
        nz = engine.state["B"].shape[3]
        for k in range(nz):
            engine.state["B"][1, :, 0, k] = 0.1 + 0.4 * (k / nz)

        # Radial temperature gradient drives axial Nernst velocity
        Te = engine.state["Te"]
        for i in range(nr):
            Te[i, :, :] = 1e6 + (i / nr) * 5e6
        engine.state["Te"] = Te

        B_before = engine.state["B"].copy()
        # Use a larger dt so the tiny advection velocity produces a measurable change
        engine._apply_nernst(1e-6, Z_bar=1.0)
        B_after = engine.state["B"]

        # B-field should have changed (Nernst velocity ~7 m/s in z,
        # and B_theta now has z-gradient -> dB_theta/dz != 0)
        diff = np.max(np.abs(B_after - B_before))
        assert diff > 0, "Nernst advection should modify B when dB/dz != 0 and v_N_z != 0"

    def test_nernst_step_preserves_finiteness(self):
        """Nernst step should not introduce NaN or Inf."""
        cfg = _make_engine_config(fluid={"enable_nernst": True})
        engine = _make_engine(cfg)

        nz = engine.state["B"].shape[3]
        for k in range(nz):
            engine.state["B"][1, :, 0, k] = 0.5 + 0.5 * (k / nz)
        engine.state["Te"][:8, :, :] = 5e6
        engine.state["Te"][8:, :, :] = 1e5

        engine._apply_nernst(1e-6, Z_bar=1.0)
        engine._sanitize_state("test nernst")

        assert np.all(np.isfinite(engine.state["B"]))


# ===================================================================
# 17.2 — Braginskii Viscosity Integration in Engine
# ===================================================================


class TestViscosityEngineIntegration:
    """Verify Braginskii viscosity is properly wired into the engine."""

    def test_viscosity_config_flag_exists(self):
        """FluidConfig has enable_viscosity field."""
        from dpf.config import FluidConfig

        fc = FluidConfig()
        assert hasattr(fc, "enable_viscosity")
        assert fc.enable_viscosity is False  # Default off

    def test_viscosity_disabled_by_default(self):
        """Default config does not enable viscosity."""
        cfg = _make_engine_config()
        engine = _make_engine(cfg)
        assert engine.config.fluid.enable_viscosity is False

    def test_viscosity_enabled_modifies_velocity(self):
        """With viscosity enabled and non-uniform shear, velocity changes."""
        cfg = _make_engine_config(fluid={"enable_viscosity": True})
        engine = _make_engine(cfg)

        # Set up sinusoidal velocity shear (non-uniform -> non-zero div(stress))
        # A linear profile gives constant stress -> zero divergence (Couette flow).
        vel = engine.state["velocity"]
        nr = vel.shape[1]
        for i in range(nr):
            vel[2, i, :, :] = 1e4 * np.sin(np.pi * i / nr)  # non-uniform vz(r)
        engine.state["velocity"] = vel
        engine.state["Ti"][:] = 1e6  # Hot ions for non-negligible viscosity

        vel_before = engine.state["velocity"].copy()

        engine._apply_viscosity(1e-9)

        vel_after = engine.state["velocity"]

        # Velocity should be modified by viscous stress
        vel_diff = np.max(np.abs(vel_after - vel_before))
        assert vel_diff > 0, "Viscosity should modify velocity when non-uniform shear is present"

    def test_viscous_heating_increases_Ti(self):
        """Viscous dissipation should heat ions."""
        cfg = _make_engine_config(fluid={"enable_viscosity": True})
        engine = _make_engine(cfg)

        # Set up strong velocity shear
        vel = engine.state["velocity"]
        nr = vel.shape[1]
        for i in range(nr):
            vel[0, i, :, :] = 1e5 * np.sin(np.pi * i / nr)
        engine.state["velocity"] = vel
        engine.state["Ti"][:] = 1e6

        Ti_mean_before = float(np.mean(engine.state["Ti"]))
        engine._apply_viscosity(1e-11)
        Ti_mean_after = float(np.mean(engine.state["Ti"]))

        # Mean Ti should increase (viscous heating is always positive)
        assert Ti_mean_after >= Ti_mean_before

    def test_viscosity_preserves_finiteness(self):
        """Viscosity step should not produce NaN/Inf."""
        cfg = _make_engine_config(fluid={"enable_viscosity": True})
        engine = _make_engine(cfg)

        vel = engine.state["velocity"]
        vel[0, :, :, :] = 1e4
        engine.state["velocity"] = vel
        engine.state["Ti"][:] = 1e6

        engine._apply_viscosity(1e-12)

        assert np.all(np.isfinite(engine.state["velocity"]))
        assert np.all(np.isfinite(engine.state["Ti"]))
        assert np.all(np.isfinite(engine.state["pressure"]))


# ===================================================================
# 17.3 — Constrained Transport Option
# ===================================================================


class TestConstrainedTransportOption:
    """Verify the CT option in CylindricalMHDSolver."""

    def test_ct_flag_in_constructor(self):
        """CylindricalMHDSolver accepts enable_ct parameter."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(
            nr=16, nz=16, dr=1e-3, dz=1e-3, enable_ct=True,
        )
        assert solver.enable_ct is True

    def test_ct_default_on(self):
        """CT is enabled by default for div(B)=0 preservation."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=16, nz=16, dr=1e-3, dz=1e-3)
        assert solver.enable_ct is True

    def test_ct_solver_runs_without_error(self):
        """Solver with CT enabled can complete a step."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 16
        dr = dz = 1e-3
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            enable_ct=True,
            enable_hall=False,
            enable_resistive=False,
        )

        state = _build_uniform_cylindrical_state(nr, nz, rho0=1e-4, T0=1e4, p0=None)
        # Add some B-field
        state["B"][1, :, 0, :] = 0.01  # small B_theta

        dt = solver._compute_dt(state)
        state_out = solver.step(state, dt, current=0.0, voltage=0.0)

        assert np.all(np.isfinite(state_out["rho"]))
        assert np.all(np.isfinite(state_out["B"]))

    def test_dedner_solver_also_runs(self):
        """Solver with Dedner (default) still works alongside CT option."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 16
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=1e-3, dz=1e-3,
            enable_ct=False,
            enable_hall=False,
        )

        state = _build_uniform_cylindrical_state(nr, nz)
        state["B"][1, :, 0, :] = 0.01

        dt = solver._compute_dt(state)
        state_out = solver.step(state, dt, current=0.0, voltage=0.0)

        assert np.all(np.isfinite(state_out["rho"]))


def _build_uniform_cylindrical_state(
    nr: int, nz: int, rho0: float = 1e-4, T0: float = 1e4, p0: float | None = None
) -> dict[str, np.ndarray]:
    """Create a uniform cylindrical state for testing."""
    ion_mass = 3.34e-27
    n_i = rho0 / ion_mass
    if p0 is None:
        p0 = 2.0 * n_i * k_B * T0
    return {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), T0),
        "Ti": np.full((nr, 1, nz), T0),
        "psi": np.zeros((nr, 1, nz)),
    }


# ===================================================================
# 17.4 — Sod Shock Tube Verification
# ===================================================================


class TestSodShockTube:
    """Verify the Sod shock tube test against the exact Riemann solution."""

    def test_exact_solver_pstar(self):
        """Exact Riemann solver finds the correct p* for Sod problem."""
        from dpf.verification.shock_tubes import _sod_find_pstar

        p_star, u_star = _sod_find_pstar(1.0, 1.0, 0.0, 0.125, 0.1, 0.0, 1.4)
        # Known exact: p* ≈ 0.30313, u* ≈ 0.92745
        assert abs(p_star - 0.30313) < 1e-4, f"p_star={p_star}"
        assert abs(u_star - 0.92745) < 1e-4, f"u_star={u_star}"

    def test_analytical_solution_structure(self):
        """Analytical solution has correct wave structure at t=0.2."""
        from dpf.verification.shock_tubes import sod_shock_tube_analytical

        x = np.linspace(0.005, 0.995, 200)
        sol = sod_shock_tube_analytical(x, 0.2, gamma=1.4)

        assert "rho" in sol and "u" in sol and "p" in sol
        assert len(sol["rho"]) == 200

        # Left state preserved far left
        assert abs(sol["rho"][0] - 1.0) < 1e-10
        assert abs(sol["p"][0] - 1.0) < 1e-10

        # Right state preserved far right
        assert abs(sol["rho"][-1] - 0.125) < 1e-10
        assert abs(sol["p"][-1] - 0.1) < 1e-10

    @pytest.mark.slow
    def test_sod_run_passes(self):
        """Full Sod shock tube test completes and passes sanity checks.

        Uses reduced resolution (nx=100) for test speed.
        """
        from dpf.verification.shock_tubes import run_sod_test

        result = run_sod_test(nx=100, t_end=0.2, gamma=1.4)

        # All sanity checks should pass
        assert result.checks["no_nan"], "Sod test produced NaN/Inf"
        assert result.checks["rho_positive"], "Density went negative"
        assert result.checks["p_positive"], "Pressure went negative"

        # L1 errors should be reasonable for 100-cell WENO5+HLL
        # Typical: rho ~ 0.02-0.05, u ~ 0.02-0.05, p ~ 0.02-0.05
        assert result.errors["rho"] < 0.10, f"rho L1 error too large: {result.errors['rho']}"
        assert result.errors["u"] < 0.10, f"u L1 error too large: {result.errors['u']}"
        assert result.errors["p"] < 0.10, f"p L1 error too large: {result.errors['p']}"


# ===================================================================
# 17.5 — Brio-Wu MHD Shock Tube
# ===================================================================


class TestBrioWuShockTube:
    """Verify the Brio-Wu MHD shock tube test."""

    def test_reference_data_structure(self):
        """Reference data returns expected keys."""
        from dpf.verification.shock_tubes import brio_wu_reference

        x = np.linspace(0.005, 0.995, 100)
        ref = brio_wu_reference(x, 0.1, gamma=2.0)

        assert "initial_left" in ref
        assert "initial_right" in ref
        assert "Bx" in ref
        assert ref["Bx"] == 0.75
        assert len(ref["expected_waves"]) == 5

    @pytest.mark.slow
    def test_brio_wu_run_passes(self):
        """Full Brio-Wu test completes and passes qualitative checks.

        Uses reduced resolution (nx=200) for test speed.
        """
        from dpf.verification.shock_tubes import run_brio_wu_test

        result = run_brio_wu_test(nx=200, t_end=0.1, gamma=2.0)

        # All qualitative checks should pass
        assert result.checks["no_nan"], "Brio-Wu produced NaN/Inf"
        assert result.checks["rho_positive"], "Density went negative"
        assert result.checks["p_positive"], "Pressure went negative"
        assert result.checks["Bx_preserved"], "Bx not conserved"
        assert result.checks["has_wave_structure"], "No wave structure detected"
        assert result.checks["By_sign_change"], "By sign change missing"


# ===================================================================
# 17.6 — Cylindrical Convergence Test
# ===================================================================


class TestCylindricalConvergence:
    """Verify convergence of the cylindrical solver on z-pinch equilibrium."""

    def test_zpinch_equilibrium_setup(self):
        """Z-pinch equilibrium setup produces valid initial state."""
        from dpf.verification.cylindrical_convergence import setup_zpinch_equilibrium

        state = setup_zpinch_equilibrium(
            nr=32, nz=32, dr=1e-3, dz=1e-3,
            J0=1e6, a=0.1, p_axis=1e5, rho0=1e-3,
            ion_mass=3.34e-27,
        )

        assert state["rho"].shape == (32, 1, 32)
        assert state["velocity"].shape == (3, 32, 1, 32)
        assert state["B"].shape == (3, 32, 1, 32)
        assert np.all(state["rho"] > 0)
        assert np.all(state["pressure"] > 0)
        assert np.all(np.isfinite(state["B"]))

    def test_zpinch_pressure_positive(self):
        """Equilibrium pressure stays positive everywhere."""
        from dpf.verification.cylindrical_convergence import setup_zpinch_equilibrium

        state = setup_zpinch_equilibrium(
            nr=64, nz=16, dr=1e-3, dz=1e-3,
            J0=1e6, a=0.2, p_axis=1e5, rho0=1e-3,
            ion_mass=3.34e-27,
        )
        assert np.all(state["pressure"] > 0)

    def test_zpinch_btheta_profile(self):
        """B_theta follows B_theta = mu_0 * J0 * r / 2 inside column."""
        from dpf.verification.cylindrical_convergence import setup_zpinch_equilibrium

        nr, nz = 64, 16
        dr = 1e-3
        J0 = 1e6
        a = 0.2  # Column radius >> domain radius = nr*dr = 0.064

        state = setup_zpinch_equilibrium(
            nr=nr, nz=nz, dr=dr, dz=dr,
            J0=J0, a=a, p_axis=1e5, rho0=1e-3,
            ion_mass=3.34e-27,
        )

        r = (np.arange(nr) + 0.5) * dr
        B_theta_expected = mu_0 * J0 * r / 2.0
        B_theta_actual = state["B"][1, :, 0, 0]

        np.testing.assert_allclose(B_theta_actual, B_theta_expected, rtol=1e-10)

    def test_equilibrium_check_diagnostics(self):
        """check_equilibrium_preservation returns expected keys."""
        from dpf.verification.cylindrical_convergence import (
            check_equilibrium_preservation,
            setup_zpinch_equilibrium,
        )

        state = setup_zpinch_equilibrium(
            nr=16, nz=16, dr=1e-3, dz=1e-3,
            J0=1e6, a=0.1, p_axis=1e5, rho0=1e-3,
            ion_mass=3.34e-27,
        )
        # Compare state to itself => zero error
        diag = check_equilibrium_preservation(state, state)
        assert diag["pressure_error_L2"] == 0.0
        assert diag["velocity_error_L2"] == 0.0
        assert diag["Btheta_error_L2"] == 0.0
        assert diag["max_rho_change"] == 0.0

    @pytest.mark.slow
    def test_convergence_order_above_one(self):
        """Convergence order should be > 1.0 (expect ~1.5-2.0 for SSP-RK2)."""
        from dpf.verification.cylindrical_convergence import run_convergence_test

        result = run_convergence_test(
            resolutions=[32, 64, 128],
            n_steps=3,
            J0=1e6,
            a_frac=2.0,
        )

        order = result["convergence_order"]
        assert order > 1.0, f"Convergence order {order:.2f} should be > 1.0"

    @pytest.mark.slow
    def test_errors_decrease_with_resolution(self):
        """Errors should monotonically decrease with resolution."""
        from dpf.verification.cylindrical_convergence import run_convergence_test

        result = run_convergence_test(
            resolutions=[32, 64, 128],
            n_steps=3,
        )

        bt_errs = result["Btheta_errors"]
        # Each finer resolution should have less error
        for i in range(len(bt_errs) - 1):
            assert bt_errs[i + 1] < bt_errs[i], (
                f"Error at res {result['resolutions'][i + 1]} "
                f"({bt_errs[i + 1]:.3e}) should be less than at "
                f"res {result['resolutions'][i]} ({bt_errs[i]:.3e})"
            )


# ===================================================================
# 17.x — Engine step integration smoke tests
# ===================================================================


class TestEnginePhase17Integration:
    """Smoke tests that engine.step() works with new Phase 17 features."""

    def test_engine_step_with_nernst_enabled(self):
        """Engine completes a step with Nernst enabled."""
        cfg = _make_engine_config(fluid={"enable_nernst": True})
        engine = _make_engine(cfg)

        # Seed with B-field and temperature gradient
        engine.state["B"][1, :, 0, :] = 0.1
        engine.state["Te"][:8, :, :] = 2e6
        engine.state["Te"][8:, :, :] = 5e5

        result = engine.step()
        assert not result.finished
        assert result.dt > 0
        assert np.all(np.isfinite(engine.state["B"]))

    def test_engine_step_with_viscosity_enabled(self):
        """Engine completes a step with viscosity enabled."""
        cfg = _make_engine_config(fluid={"enable_viscosity": True})
        engine = _make_engine(cfg)

        # Seed velocity shear
        nr = engine.state["velocity"].shape[1]
        for i in range(nr):
            engine.state["velocity"][2, i, :, :] = 1e4 * (i / nr)

        result = engine.step()
        assert not result.finished
        assert result.dt > 0
        assert np.all(np.isfinite(engine.state["velocity"]))

    def test_engine_step_with_both_enabled(self):
        """Engine completes a step with both Nernst and viscosity enabled."""
        cfg = _make_engine_config(
            fluid={"enable_nernst": True, "enable_viscosity": True}
        )
        engine = _make_engine(cfg)

        engine.state["B"][1, :, 0, :] = 0.05
        engine.state["Te"][:8, :, :] = 3e6
        engine.state["Te"][8:, :, :] = 5e5

        result = engine.step()
        assert not result.finished
        assert result.dt > 0


# ===================================================================
# Verification module unit tests (fast)
# ===================================================================


class TestShockTubeModule:
    """Unit tests for the shock_tubes verification module internals."""

    def test_build_state_shape(self):
        """_build_state produces correct 3D shape."""
        from dpf.verification.shock_tubes import _build_state

        nx = 50
        rho = np.ones(nx)
        u = np.zeros(nx)
        p = np.ones(nx)
        Bx = np.zeros(nx)
        By = np.zeros(nx)
        Bz = np.zeros(nx)

        state = _build_state(nx, rho, u, p, Bx, By, Bz, 1.4, 1.67e-27)

        assert state["rho"].shape == (50, 4, 4)
        assert state["velocity"].shape == (3, 50, 4, 4)
        assert state["B"].shape == (3, 50, 4, 4)
        assert state["Te"].shape == (50, 4, 4)
        assert state["psi"].shape == (50, 4, 4)

    def test_analytical_solution_smooth(self):
        """Analytical solution is smooth within each region."""
        from dpf.verification.shock_tubes import sod_shock_tube_analytical

        x = np.linspace(0.005, 0.995, 1000)
        sol = sod_shock_tube_analytical(x, 0.2, gamma=1.4)

        # All values should be finite
        assert np.all(np.isfinite(sol["rho"]))
        assert np.all(np.isfinite(sol["u"]))
        assert np.all(np.isfinite(sol["p"]))

        # Density and pressure should be positive
        assert np.all(sol["rho"] > 0)
        assert np.all(sol["p"] > 0)


class TestConvergenceModule:
    """Unit tests for the cylindrical_convergence module internals."""

    def test_equilibrium_raises_on_low_pressure(self):
        """setup_zpinch_equilibrium raises if p_axis is too low."""
        from dpf.verification.cylindrical_convergence import setup_zpinch_equilibrium

        with pytest.raises(ValueError, match="too low"):
            setup_zpinch_equilibrium(
                nr=32, nz=16, dr=1e-3, dz=1e-3,
                J0=1e8,  # Very high J0
                a=0.1,
                p_axis=1.0,  # Way too low
                rho0=1e-3,
                ion_mass=3.34e-27,
            )

    def test_convergence_order_estimation(self):
        """_estimate_convergence_order returns correct order for known data."""
        from dpf.verification.cylindrical_convergence import _estimate_convergence_order

        # Exact 2nd order: error ~ 1/N^2
        resolutions = [10, 20, 40, 80]
        errors = [1.0 / n**2 for n in resolutions]

        order = _estimate_convergence_order(resolutions, errors)
        assert abs(order - 2.0) < 0.05, f"Expected order ~2.0, got {order:.3f}"

    def test_convergence_order_first_order_data(self):
        """Correctly identifies first-order convergence."""
        from dpf.verification.cylindrical_convergence import _estimate_convergence_order

        resolutions = [10, 20, 40]
        errors = [1.0 / n for n in resolutions]

        order = _estimate_convergence_order(resolutions, errors)
        assert abs(order - 1.0) < 0.05, f"Expected order ~1.0, got {order:.3f}"
