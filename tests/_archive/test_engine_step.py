"""Tests for Phase 7: Engine step() refactor, field output, config, NaN guard, derived quantities.

Test categories:
1. StepResult dataclass and step() method
2. run() uses step() internally (backward compatibility)
3. Field snapshot HDF5 output
4. Config initial conditions (rho0, T0, anomalous_alpha)
5. NaN/Inf guard
6. Derived quantity computations (J, beta, Mach)
7. get_field_snapshot() method
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from dpf.constants import mu_0

# ====================================================
# Helper: create a test engine
# ====================================================

def _make_config(**overrides):
    """Build a minimal SimulationConfig with optional overrides."""
    from dpf.config import SimulationConfig

    defaults = {
        "grid_shape": (4, 4, 4),
        "dx": 1e-3,
        "sim_time": 1e-7,
        "dt_init": 1e-10,
        "circuit": {
            "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
            "anode_radius": 0.005, "cathode_radius": 0.01,
        },
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _make_engine(**config_overrides):
    from dpf.engine import SimulationEngine
    config = _make_config(**config_overrides)
    return SimulationEngine(config)


# ====================================================
# StepResult and step() Tests
# ====================================================

class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_fields(self):
        """StepResult has all expected fields."""
        from dpf.core.bases import StepResult

        r = StepResult()
        assert hasattr(r, "time")
        assert hasattr(r, "step")
        assert hasattr(r, "dt")
        assert hasattr(r, "current")
        assert hasattr(r, "voltage")
        assert hasattr(r, "energy_conservation")
        assert hasattr(r, "max_Te")
        assert hasattr(r, "max_rho")
        assert hasattr(r, "Z_bar")
        assert hasattr(r, "R_plasma")
        assert hasattr(r, "eta_anomalous")
        assert hasattr(r, "total_radiated_energy")
        assert hasattr(r, "finished")

    def test_step_result_defaults(self):
        """StepResult defaults are sensible."""
        from dpf.core.bases import StepResult

        r = StepResult()
        assert r.time == 0.0
        assert r.step == 0
        assert r.finished is False


class TestEngineStep:
    """Tests for engine.step() method."""

    def test_step_returns_step_result(self):
        """Single step returns a StepResult."""
        from dpf.core.bases import StepResult

        engine = _make_engine()
        result = engine.step()
        assert isinstance(result, StepResult)
        assert not result.finished

    def test_step_advances_time(self):
        """step() increments time and step_count."""
        engine = _make_engine()
        assert engine.time == 0.0
        assert engine.step_count == 0

        result = engine.step()
        assert result.time > 0.0
        assert result.step == 1
        assert engine.time == result.time
        assert engine.step_count == 1

    def test_step_dt_positive(self):
        """step() uses a positive timestep."""
        engine = _make_engine()
        result = engine.step()
        assert result.dt > 0.0

    def test_step_scalars_populated(self):
        """StepResult has nonzero physics scalars after a step."""
        engine = _make_engine()
        result = engine.step()
        assert result.current != 0.0 or result.voltage != 0.0
        assert result.max_Te > 0.0
        assert result.max_rho > 0.0
        assert result.energy_conservation > 0.0

    def test_multiple_steps(self):
        """Multiple steps advance monotonically."""
        engine = _make_engine()
        times = []
        for _ in range(5):
            result = engine.step()
            times.append(result.time)
        # Time should be strictly increasing
        for i in range(len(times) - 1):
            assert times[i + 1] > times[i]

    def test_step_finished_at_sim_time(self):
        """step() sets finished=True when sim_time is reached."""
        engine = _make_engine(sim_time=1e-10, dt_init=1e-10)
        # Run enough steps to reach sim_time
        for _ in range(100):
            result = engine.step()
            if result.finished:
                break
        assert result.finished


# ====================================================
# Backward Compatibility: run() still works
# ====================================================

class TestRunUsesStep:
    """Tests that run() produces same results as before."""

    def test_run_returns_summary(self):
        """run() still returns a summary dict."""
        engine = _make_engine()
        summary = engine.run(max_steps=5)
        assert isinstance(summary, dict)
        assert summary["steps"] == 5
        assert "energy_conservation" in summary
        assert "wall_time_s" in summary

    def test_run_energy_conservation(self):
        """run() maintains energy conservation."""
        engine = _make_engine()
        summary = engine.run(max_steps=10)
        assert 0.95 <= summary["energy_conservation"] <= 1.05


# ====================================================
# Field Snapshot HDF5 Output
# ====================================================

class TestFieldSnapshots:
    """Tests for field snapshot recording in HDF5."""

    def test_field_snapshot_written(self):
        """With field_output_interval > 0, HDF5 has field groups."""
        import h5py

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine = _make_engine(
                diagnostics={"output_interval": 1, "field_output_interval": 1, "hdf5_filename": fname},
            )
            engine.run(max_steps=3)

            with h5py.File(fname, "r") as f:
                assert "fields" in f
                assert f["fields"].attrs["num_snapshots"] > 0
                # Check first snapshot has expected arrays
                snap = f["fields/snapshot_0000"]
                assert "rho" in snap
                assert "Te" in snap
                assert "B" in snap
                assert "time" in snap.attrs
        finally:
            os.unlink(fname)

    def test_no_field_snapshots_by_default(self):
        """With field_output_interval=0, no field groups written."""
        import h5py

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine = _make_engine(
                diagnostics={"output_interval": 1, "hdf5_filename": fname},
            )
            engine.run(max_steps=3)

            with h5py.File(fname, "r") as f:
                assert "fields" not in f
                assert "scalars" in f
        finally:
            os.unlink(fname)


# ====================================================
# Config Initial Conditions
# ====================================================

class TestConfigInitialConditions:
    """Tests for rho0, T0, anomalous_alpha config parameters."""

    def test_custom_rho0(self):
        """Engine uses rho0 from config."""
        engine = _make_engine(rho0=1e-2)
        assert np.allclose(engine.state["rho"], 1e-2)

    def test_custom_T0(self):
        """Engine uses T0 from config."""
        engine = _make_engine(T0=5000.0)
        assert np.allclose(engine.state["Te"], 5000.0)
        assert np.allclose(engine.state["Ti"], 5000.0)

    def test_default_rho0(self):
        """Default rho0 is 1e-4."""
        engine = _make_engine()
        assert np.allclose(engine.state["rho"], 1e-4)

    def test_anomalous_alpha_in_config(self):
        """anomalous_alpha parameter exists in config."""
        config = _make_config()
        assert config.anomalous_alpha == pytest.approx(0.05)
        config2 = _make_config(anomalous_alpha=0.1)
        assert config2.anomalous_alpha == pytest.approx(0.1)


# ====================================================
# NaN / Inf Guard
# ====================================================

class TestNaNGuard:
    """Tests for NaN/Inf detection and repair."""

    def test_nan_in_rho_repaired(self):
        """NaN in density is replaced with floor value."""
        engine = _make_engine()
        engine.state["rho"][0, 0, 0] = np.nan
        count = engine._sanitize_state("test")
        assert count > 0
        assert np.all(np.isfinite(engine.state["rho"]))

    def test_inf_in_Te_repaired(self):
        """Inf in electron temperature is replaced with floor."""
        engine = _make_engine()
        engine.state["Te"][1, 1, 1] = np.inf
        count = engine._sanitize_state("test")
        assert count > 0
        assert np.all(np.isfinite(engine.state["Te"]))

    def test_clean_state_no_repairs(self):
        """Clean state returns zero repairs."""
        engine = _make_engine()
        count = engine._sanitize_state("test")
        assert count == 0

    def test_velocity_nan_repaired(self):
        """NaN in velocity is replaced with 0.0."""
        engine = _make_engine()
        engine.state["velocity"][0, 0, 0, 0] = np.nan
        engine._sanitize_state("test")
        assert engine.state["velocity"][0, 0, 0, 0] == 0.0


# ====================================================
# get_field_snapshot()
# ====================================================

class TestFieldSnapshot:
    """Tests for get_field_snapshot() method."""

    def test_returns_copy(self):
        """get_field_snapshot() returns independent copies."""
        engine = _make_engine()
        snap = engine.get_field_snapshot()
        # Modify snapshot — should not affect engine state
        snap["rho"][:] = 999.0
        assert not np.allclose(engine.state["rho"], 999.0)

    def test_has_all_fields(self):
        """Snapshot contains all state arrays."""
        engine = _make_engine()
        snap = engine.get_field_snapshot()
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi"):
            assert key in snap


# ====================================================
# Derived Quantity Tests
# ====================================================

class TestDerivedQuantities:
    """Tests for derived diagnostic computations."""

    def test_plasma_beta_uniform(self):
        """Uniform p and B gives correct beta."""
        from dpf.diagnostics.derived import plasma_beta

        p = np.full((4, 4, 4), 1e5)  # 1 bar
        B = np.zeros((3, 4, 4, 4))
        B[2, :, :, :] = 0.1  # 0.1 T in z direction

        beta = plasma_beta(p, B)
        # beta = 2 * mu_0 * p / B^2 = 2 * 4pi*1e-7 * 1e5 / 0.01
        expected = 2.0 * mu_0 * 1e5 / 0.01
        np.testing.assert_allclose(beta[2, 2, 2], expected, rtol=1e-10)

    def test_plasma_beta_zero_B(self):
        """Zero B gives large beta (capped by floor)."""
        from dpf.diagnostics.derived import plasma_beta

        p = np.full((4, 4, 4), 1e5)
        B = np.zeros((3, 4, 4, 4))
        beta = plasma_beta(p, B)
        assert np.all(beta > 0)
        assert np.all(np.isfinite(beta))

    def test_mach_number_at_rest(self):
        """Zero velocity gives zero Mach number."""
        from dpf.diagnostics.derived import mach_number

        v = np.zeros((3, 4, 4, 4))
        p = np.full((4, 4, 4), 1e5)
        rho = np.full((4, 4, 4), 1.0)
        M = mach_number(v, p, rho)
        np.testing.assert_allclose(M, 0.0)

    def test_mach_number_supersonic(self):
        """Flow faster than sound speed gives M > 1."""
        from dpf.diagnostics.derived import mach_number

        gamma = 5.0 / 3.0
        rho = np.full((4, 4, 4), 1.0)
        p = np.full((4, 4, 4), 1e5)
        c_s = np.sqrt(gamma * 1e5 / 1.0)  # Sound speed
        v = np.zeros((3, 4, 4, 4))
        v[0, :, :, :] = 2.0 * c_s  # Mach 2 in x
        M = mach_number(v, p, rho, gamma)
        np.testing.assert_allclose(M[2, 2, 2], 2.0, rtol=1e-10)

    def test_alfven_speed_formula(self):
        """Alfven speed matches analytic formula."""
        from dpf.diagnostics.derived import alfven_speed

        rho = np.full((4, 4, 4), 1e-4)
        B = np.zeros((3, 4, 4, 4))
        B[2, :, :, :] = 0.01  # 10 mT
        v_A = alfven_speed(B, rho)
        expected = 0.01 / np.sqrt(mu_0 * 1e-4)
        np.testing.assert_allclose(v_A[2, 2, 2], expected, rtol=1e-10)

    def test_current_density_uniform_B(self):
        """Uniform B gives zero current density (curl(const) = 0)."""
        from dpf.diagnostics.derived import current_density_magnitude

        B = np.zeros((3, 8, 8, 8))
        B[2, :, :, :] = 0.1  # Uniform Bz
        J = current_density_magnitude(B, dx=1e-3)
        # Interior cells should have ~0 current (boundaries have gradient artifacts)
        assert J[4, 4, 4] < 1e-5  # Very small in interior
