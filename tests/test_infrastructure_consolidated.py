"""Consolidated infrastructure tests.

Coverage:
- test_checkpoint.py: Checkpoint save/load HDF5, engine checkpoint/restart, HLL njit
- test_engine_step.py: StepResult, engine.step(), field snapshots, NaN guard, derived quantities
- test_integration.py: End-to-end pipeline: config -> engine -> simulation -> diagnostics
- test_stress.py: Stress tests (NaN, energy, neutron yield, interferometry, cylindrical, presets)
- test_phase_r_engine_demotion.py: Python engine demotion to teaching-only tier
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
import warnings

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.constants import mu_0
from dpf.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_config_engine(**overrides):
    """Build a minimal SimulationConfig for engine step tests (grid 4x4x4)."""
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


def _make_engine_step(**config_overrides):
    engine = SimulationEngine(_make_config_engine(**config_overrides))
    return engine


def _make_config_stress(**overrides):
    """Build a SimulationConfig for stress tests (grid 8x8x8)."""
    defaults = {
        "grid_shape": [8, 8, 8],
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(sample_config_dict):
    """Write a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config_dict, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def small_python_config() -> dict:
    """Config dict for a small Python-backend simulation."""
    return {
        "grid_shape": [8, 8, 8],
        "dx": 1e-3,
        "sim_time": 1e-8,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
            "anode_radius": 0.005, "cathode_radius": 0.01,
        },
        "fluid": {"backend": "python"},
        "diagnostics": {"hdf5_filename": ":memory:"},
    }


@pytest.fixture
def large_python_config() -> dict:
    """Config for Python backend exceeding the demotion threshold (grid > 16^3)."""
    return {
        "grid_shape": [32, 32, 32],
        "dx": 5e-4,
        "sim_time": 5e-7,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 5e-6, "V0": 5e3, "L0": 5e-8, "R0": 0.01,
            "anode_radius": 0.005, "cathode_radius": 0.01,
        },
        "fluid": {"backend": "python"},
        "diagnostics": {"hdf5_filename": ":memory:"},
    }


# ---------------------------------------------------------------------------
# Section: Checkpoint Save/Load
# ---------------------------------------------------------------------------

class TestCheckpointSaveLoad:
    """Tests for low-level checkpoint save and load."""

    def test_save_creates_file(self):
        """Saving a checkpoint creates an HDF5 file."""
        from dpf.diagnostics.checkpoint import save_checkpoint

        state = {
            "rho": np.ones((4, 4, 4)),
            "velocity": np.zeros((3, 4, 4, 4)),
            "B": np.zeros((3, 4, 4, 4)),
            "pressure": np.ones((4, 4, 4)) * 1e5,
            "Te": np.full((4, 4, 4), 1e4),
            "Ti": np.full((4, 4, 4), 1e4),
        }
        circuit = {"current": 100.0, "voltage": 5000.0, "energy_cap": 0.5}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            save_checkpoint(fname, state, circuit, time=1e-7, step_count=50)
            assert os.path.exists(fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_load_recovers_state(self):
        """Loading a checkpoint recovers the saved state."""
        from dpf.diagnostics.checkpoint import load_checkpoint, save_checkpoint

        state = {
            "rho": np.random.rand(4, 4, 4),
            "velocity": np.random.rand(3, 4, 4, 4),
            "B": np.random.rand(3, 4, 4, 4),
        }
        circuit = {"current": 123.4, "voltage": 5678.9}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            save_checkpoint(fname, state, circuit, time=2.5e-7, step_count=100)
            data = load_checkpoint(fname)

            assert data["time"] == pytest.approx(2.5e-7)
            assert data["step_count"] == 100
            np.testing.assert_array_equal(data["state"]["rho"], state["rho"])
            np.testing.assert_array_equal(data["state"]["velocity"], state["velocity"])
            np.testing.assert_array_equal(data["state"]["B"], state["B"])
            assert data["circuit"]["current"] == pytest.approx(123.4)
            assert data["circuit"]["voltage"] == pytest.approx(5678.9)
        finally:
            os.unlink(fname)

    def test_config_json_preserved(self):
        """Config JSON string is preserved in checkpoint."""
        from dpf.diagnostics.checkpoint import load_checkpoint, save_checkpoint

        state = {"rho": np.ones((2, 2, 2))}
        circuit = {"current": 0.0}
        config_json = '{"grid_shape": [2, 2, 2], "dx": 0.001}'

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            save_checkpoint(fname, state, circuit, 0.0, 0, config_json)
            data = load_checkpoint(fname)
            assert data["config_json"] is not None
            assert "grid_shape" in data["config_json"]
        finally:
            os.unlink(fname)

    def test_round_trip_all_arrays(self):
        """Full round-trip preserves all array types."""
        from dpf.diagnostics.checkpoint import load_checkpoint, save_checkpoint

        state = {
            "rho": np.full((4, 4, 4), 1e-4),
            "velocity": np.random.randn(3, 4, 4, 4) * 1e3,
            "pressure": np.full((4, 4, 4), 1e5),
            "B": np.random.randn(3, 4, 4, 4) * 0.1,
            "Te": np.full((4, 4, 4), 5e6),
            "Ti": np.full((4, 4, 4), 3e6),
            "psi": np.zeros((4, 4, 4)),
        }
        circuit = {
            "current": 1e6, "voltage": 1e4,
            "energy_cap": 100.0, "energy_ind": 50.0, "energy_res": 10.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            save_checkpoint(fname, state, circuit, time=5e-7, step_count=250)
            data = load_checkpoint(fname)

            for key in state:
                np.testing.assert_array_almost_equal(
                    data["state"][key], state[key],
                    err_msg=f"Mismatch in state['{key}']",
                )
            for key in circuit:
                assert data["circuit"][key] == pytest.approx(circuit[key])
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# Section: Engine Checkpoint Integration
# ---------------------------------------------------------------------------

class TestEngineCheckpoint:
    """Tests for engine checkpoint/restart integration."""

    def _make_engine(self):
        config = SimulationConfig(
            grid_shape=(4, 4, 4),
            dx=1e-3,
            sim_time=1e-7,
            dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )
        return SimulationEngine(config)

    def test_engine_save_checkpoint(self):
        """Engine can save a checkpoint."""
        engine = self._make_engine()
        engine.run(max_steps=5)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine.save_checkpoint(fname)
            assert os.path.exists(fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_engine_restart_continues(self):
        """Engine restart continues from saved state."""
        engine1 = self._make_engine()
        engine1.run(max_steps=5)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine1.save_checkpoint(fname)
            time_at_checkpoint = engine1.time
            steps_at_checkpoint = engine1.step_count

            engine2 = self._make_engine()
            engine2.load_from_checkpoint(fname)

            assert engine2.time == pytest.approx(time_at_checkpoint)
            assert engine2.step_count == steps_at_checkpoint

            engine2.run(max_steps=steps_at_checkpoint + 3)
            assert engine2.step_count > steps_at_checkpoint
            assert engine2.time > time_at_checkpoint
        finally:
            os.unlink(fname)

    def test_auto_checkpoint(self):
        """Auto-checkpoint saves file at specified interval."""
        engine = self._make_engine()
        engine.checkpoint_interval = 5

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        engine.checkpoint_filename = fname
        try:
            engine.run(max_steps=10)
            assert os.path.exists(fname)
            assert os.path.getsize(fname) > 0
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_restart_state_matches(self):
        """Restarted state arrays match checkpoint state."""
        from dpf.diagnostics.checkpoint import load_checkpoint

        engine = self._make_engine()
        engine.run(max_steps=5)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine.save_checkpoint(fname)
            data = load_checkpoint(fname)

            np.testing.assert_array_almost_equal(
                data["state"]["rho"], engine.state["rho"],
            )
            np.testing.assert_array_almost_equal(
                data["state"]["velocity"], engine.state["velocity"],
            )
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# Section: HLL Numba Acceleration
# ---------------------------------------------------------------------------

class TestHLLNjit:
    """Verify the @njit HLL core gives same results as before."""

    def test_hll_core_matches_wrapper(self):
        """The njit core and dict wrapper return identical results."""
        from dpf.fluid.mhd_solver import _hll_flux_1d, _hll_flux_1d_core

        n = 10
        rho = np.ones(n) + 0.1 * np.random.rand(n)
        u = np.random.randn(n) * 100
        p = np.ones(n) * 1e5
        Bn = np.random.randn(n) * 0.01
        gamma = 5.0 / 3.0

        F_rho, F_mom, F_ene = _hll_flux_1d_core(
            rho, rho, u, u, p, p, Bn, Bn, gamma,
        )
        fluxes = _hll_flux_1d(rho, rho, u, u, p, p, Bn, Bn, gamma)

        np.testing.assert_array_almost_equal(F_rho, fluxes["mass_flux"])
        np.testing.assert_array_almost_equal(F_mom, fluxes["momentum_flux"])
        np.testing.assert_array_almost_equal(F_ene, fluxes["energy_flux"])


# ---------------------------------------------------------------------------
# Section: StepResult and engine.step()
# ---------------------------------------------------------------------------

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

        engine = _make_engine_step()
        result = engine.step()
        assert isinstance(result, StepResult)
        assert not result.finished

    def test_step_advances_time(self):
        """step() increments time and step_count."""
        engine = _make_engine_step()
        assert engine.time == 0.0
        assert engine.step_count == 0

        result = engine.step()
        assert result.time > 0.0
        assert result.step == 1
        assert engine.time == result.time
        assert engine.step_count == 1

    def test_step_dt_positive(self):
        """step() uses a positive timestep."""
        engine = _make_engine_step()
        result = engine.step()
        assert result.dt > 0.0

    def test_step_scalars_populated(self):
        """StepResult has nonzero physics scalars after a step."""
        engine = _make_engine_step()
        result = engine.step()
        assert result.current != 0.0 or result.voltage != 0.0
        assert result.max_Te > 0.0
        assert result.max_rho > 0.0
        assert result.energy_conservation > 0.0

    def test_multiple_steps(self):
        """Multiple steps advance monotonically."""
        engine = _make_engine_step()
        times = []
        for _ in range(5):
            result = engine.step()
            times.append(result.time)
        for i in range(len(times) - 1):
            assert times[i + 1] > times[i]

    def test_step_finished_at_sim_time(self):
        """step() sets finished=True when sim_time is reached."""
        engine = _make_engine_step(sim_time=1e-10, dt_init=1e-10)
        for _ in range(100):
            result = engine.step()
            if result.finished:
                break
        assert result.finished


# ---------------------------------------------------------------------------
# Section: Backward Compatibility — run()
# ---------------------------------------------------------------------------

class TestRunUsesStep:
    """Tests that run() produces same results as before."""

    def test_run_returns_summary(self):
        """run() still returns a summary dict."""
        engine = _make_engine_step()
        summary = engine.run(max_steps=5)
        assert isinstance(summary, dict)
        assert summary["steps"] == 5
        assert "energy_conservation" in summary
        assert "wall_time_s" in summary

    def test_run_energy_conservation(self):
        """run() maintains energy conservation."""
        engine = _make_engine_step()
        summary = engine.run(max_steps=10)
        assert 0.95 <= summary["energy_conservation"] <= 1.05


# ---------------------------------------------------------------------------
# Section: Field Snapshot HDF5 Output
# ---------------------------------------------------------------------------

class TestFieldSnapshots:
    """Tests for field snapshot recording in HDF5."""

    def test_field_snapshot_written(self):
        """With field_output_interval > 0, HDF5 has field groups."""
        import h5py

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name

        try:
            engine = _make_engine_step(
                diagnostics={"output_interval": 1, "field_output_interval": 1, "hdf5_filename": fname},
            )
            engine.run(max_steps=3)

            with h5py.File(fname, "r") as f:
                assert "fields" in f
                assert f["fields"].attrs["num_snapshots"] > 0
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
            engine = _make_engine_step(
                diagnostics={"output_interval": 1, "hdf5_filename": fname},
            )
            engine.run(max_steps=3)

            with h5py.File(fname, "r") as f:
                assert "fields" not in f
                assert "scalars" in f
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# Section: Config Initial Conditions
# ---------------------------------------------------------------------------

class TestConfigInitialConditions:
    """Tests for rho0, T0, anomalous_alpha config parameters."""

    def test_custom_rho0(self):
        """Engine uses rho0 from config."""
        engine = _make_engine_step(rho0=1e-2)
        assert np.allclose(engine.state["rho"], 1e-2)

    def test_custom_T0(self):
        """Engine uses T0 from config."""
        engine = _make_engine_step(T0=5000.0)
        assert np.allclose(engine.state["Te"], 5000.0)
        assert np.allclose(engine.state["Ti"], 5000.0)

    def test_default_rho0(self):
        """Default rho0 is 1e-4."""
        engine = _make_engine_step()
        assert np.allclose(engine.state["rho"], 1e-4)

    def test_anomalous_alpha_in_config(self):
        """anomalous_alpha parameter exists in config."""
        config = _make_config_engine()
        assert config.anomalous_alpha == pytest.approx(0.05)
        config2 = _make_config_engine(anomalous_alpha=0.1)
        assert config2.anomalous_alpha == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Section: NaN / Inf Guard
# ---------------------------------------------------------------------------

class TestNaNGuard:
    """Tests for NaN/Inf detection and repair."""

    def test_nan_in_rho_repaired(self):
        """NaN in density is replaced with floor value."""
        engine = _make_engine_step()
        engine.state["rho"][0, 0, 0] = np.nan
        count = engine._sanitize_state("test")
        assert count > 0
        assert np.all(np.isfinite(engine.state["rho"]))

    def test_inf_in_Te_repaired(self):
        """Inf in electron temperature is replaced with floor."""
        engine = _make_engine_step()
        engine.state["Te"][1, 1, 1] = np.inf
        count = engine._sanitize_state("test")
        assert count > 0
        assert np.all(np.isfinite(engine.state["Te"]))

    def test_clean_state_no_repairs(self):
        """Clean state returns zero repairs."""
        engine = _make_engine_step()
        count = engine._sanitize_state("test")
        assert count == 0

    def test_velocity_nan_repaired(self):
        """NaN in velocity is replaced with 0.0."""
        engine = _make_engine_step()
        engine.state["velocity"][0, 0, 0, 0] = np.nan
        engine._sanitize_state("test")
        assert engine.state["velocity"][0, 0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# Section: get_field_snapshot()
# ---------------------------------------------------------------------------

class TestFieldSnapshot:
    """Tests for get_field_snapshot() method."""

    def test_returns_copy(self):
        """get_field_snapshot() returns independent copies."""
        engine = _make_engine_step()
        snap = engine.get_field_snapshot()
        snap["rho"][:] = 999.0
        assert not np.allclose(engine.state["rho"], 999.0)

    def test_has_all_fields(self):
        """Snapshot contains all state arrays."""
        engine = _make_engine_step()
        snap = engine.get_field_snapshot()
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi"):
            assert key in snap


# ---------------------------------------------------------------------------
# Section: Derived Quantity Tests
# ---------------------------------------------------------------------------

class TestDerivedQuantities:
    """Tests for derived diagnostic computations."""

    def test_plasma_beta_uniform(self):
        """Uniform p and B gives correct beta."""
        from dpf.diagnostics.derived import plasma_beta

        p = np.full((4, 4, 4), 1e5)
        B = np.zeros((3, 4, 4, 4))
        B[2, :, :, :] = 0.1

        beta = plasma_beta(p, B)
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
        c_s = np.sqrt(gamma * 1e5 / 1.0)
        v = np.zeros((3, 4, 4, 4))
        v[0, :, :, :] = 2.0 * c_s
        M = mach_number(v, p, rho, gamma)
        np.testing.assert_allclose(M[2, 2, 2], 2.0, rtol=1e-10)

    def test_alfven_speed_formula(self):
        """Alfven speed matches analytic formula."""
        from dpf.diagnostics.derived import alfven_speed

        rho = np.full((4, 4, 4), 1e-4)
        B = np.zeros((3, 4, 4, 4))
        B[2, :, :, :] = 0.01
        v_A = alfven_speed(B, rho)
        expected = 0.01 / np.sqrt(mu_0 * 1e-4)
        np.testing.assert_allclose(v_A[2, 2, 2], expected, rtol=1e-10)

    def test_current_density_uniform_B(self):
        """Uniform B gives zero current density (curl(const) = 0)."""
        from dpf.diagnostics.derived import current_density_magnitude

        B = np.zeros((3, 8, 8, 8))
        B[2, :, :, :] = 0.1
        J = current_density_magnitude(B, dx=1e-3)
        assert J[4, 4, 4] < 1e-5


# ---------------------------------------------------------------------------
# Section: End-to-End Integration
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full simulation pipeline tests."""

    def test_engine_runs_10_steps(self, sample_config_dict):
        """Engine completes 10 steps without crashing."""
        config = SimulationConfig(**sample_config_dict)
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=10)

        assert summary["steps"] == 10
        assert summary["sim_time"] > 0
        assert summary["wall_time_s"] > 0

    def test_energy_conservation(self, sample_config_dict):
        """Circuit energy is conserved over a short run."""
        config = SimulationConfig(**sample_config_dict)
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=100)

        assert 0.90 < summary["energy_conservation"] < 1.10

    def test_config_from_file(self, tmp_config):
        """Config loads from JSON file and creates a valid engine."""
        config = SimulationConfig.from_file(tmp_config)
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5

    def test_diagnostics_output(self, sample_config_dict, tmp_path):
        """Diagnostics file is created with data."""
        hdf5_file = str(tmp_path / "test_diag.h5")
        sample_config_dict["diagnostics"] = {
            "hdf5_filename": hdf5_file,
            "output_interval": 1,
        }
        config = SimulationConfig(**sample_config_dict)
        engine = SimulationEngine(config)
        engine.run(max_steps=10)

        try:
            import h5py

            assert os.path.exists(hdf5_file)
            with h5py.File(hdf5_file, "r") as f:
                assert "scalars" in f
                assert "time" in f["scalars"]
                assert len(f["scalars"]["time"]) == 10
        except ImportError:
            pass

    def test_current_flows(self, sample_config_dict):
        """After running, current should be non-zero (circuit is discharging)."""
        config = SimulationConfig(**sample_config_dict)
        engine = SimulationEngine(config)
        engine.run(max_steps=50)
        assert abs(engine.circuit.current) > 0


# ---------------------------------------------------------------------------
# Section: Neutron Yield Integration
# ---------------------------------------------------------------------------

class TestNeutronYieldIntegration:
    """Tests that neutron yield is computed inside the engine."""

    def test_step_result_has_neutron_fields(self):
        """StepResult includes neutron_rate and total_neutron_yield."""
        from dpf.core.bases import StepResult

        r = StepResult()
        assert hasattr(r, "neutron_rate")
        assert hasattr(r, "total_neutron_yield")
        assert r.neutron_rate == 0.0
        assert r.total_neutron_yield == 0.0

    def test_engine_tracks_neutron_yield(self):
        """Engine accumulates total_neutron_yield over steps."""
        engine = SimulationEngine(_make_config_stress())
        engine.run(max_steps=5)
        assert engine.total_neutron_yield >= 0.0

    def test_neutron_yield_in_summary(self):
        """run() summary includes total_neutron_yield."""
        engine = SimulationEngine(_make_config_stress())
        summary = engine.run(max_steps=5)
        assert "total_neutron_yield" in summary

    def test_step_result_neutron_rate_nonnegative(self):
        """neutron_rate is always non-negative."""
        engine = SimulationEngine(_make_config_stress())
        for _ in range(5):
            result = engine.step()
            assert result.neutron_rate >= 0.0
            assert result.total_neutron_yield >= 0.0


# ---------------------------------------------------------------------------
# Section: Interferometry Integration
# ---------------------------------------------------------------------------

class TestInterferometryIntegration:
    """Tests that interferometry is computed for cylindrical geometry."""

    def test_cylindrical_has_fringe_shifts(self):
        """Cylindrical engine computes fringe shifts."""
        config = _make_config_stress(
            grid_shape=[8, 1, 8],
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        engine.step()
        assert engine._last_fringe_shifts is not None
        assert len(engine._last_fringe_shifts) == 8

    def test_cartesian_no_fringe_shifts(self):
        """Cartesian engine does NOT compute fringe shifts."""
        engine = SimulationEngine(_make_config_stress())
        engine.step()
        assert engine._last_fringe_shifts is None

    def test_fringe_shifts_finite(self):
        """Fringe shifts are finite numbers."""
        config = _make_config_stress(
            grid_shape=[8, 1, 8],
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        for _ in range(3):
            engine.step()
        assert np.all(np.isfinite(engine._last_fringe_shifts))


# ---------------------------------------------------------------------------
# Section: Cartesian Stress Tests
# ---------------------------------------------------------------------------

class TestCartesianStress:
    """Cartesian stress tests with all physics enabled."""

    def test_8_cubed_50_steps(self):
        """8^3 grid, up to 50 steps — no NaN, energy conserved."""
        config = _make_config_stress(
            grid_shape=[8, 8, 8],
            sim_time=1e-5,
            radiation={"bremsstrahlung_enabled": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)

        assert summary["steps"] >= 10
        assert 0.95 <= summary["energy_conservation"] <= 1.05
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf found in {key}"

    def test_all_physics_combined(self):
        """All optional physics enabled simultaneously."""
        config = _make_config_stress(
            grid_shape=[8, 8, 8],
            sim_time=1e-5,
            radiation={"bremsstrahlung_enabled": True, "fld_enabled": True},
            sheath={"enabled": True, "boundary": "z_high"},
            collision={"dynamic_coulomb_log": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=20)

        assert summary["steps"] >= 10
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key} with all physics"


# ---------------------------------------------------------------------------
# Section: Cylindrical Stress Tests
# ---------------------------------------------------------------------------

class TestCylindricalStress:
    """Cylindrical stress tests with radiation and sheath."""

    def test_8x16_50_steps(self):
        """8x1x16 cylindrical, up to 50 steps — no NaN, energy conserved."""
        config = _make_config_stress(
            grid_shape=[8, 1, 16],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)

        assert summary["steps"] >= 10
        assert 0.95 <= summary["energy_conservation"] <= 1.05
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key}"

    def test_cylindrical_with_sheath(self):
        """Cylindrical with sheath BCs enabled."""
        config = _make_config_stress(
            grid_shape=[8, 1, 8],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": True},
            sheath={"enabled": True, "boundary": "z_high"},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=20)

        assert summary["steps"] >= 10
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key}"

    def test_cylindrical_neutron_yield_tracked(self):
        """Cylindrical engine tracks neutron yield."""
        config = _make_config_stress(
            grid_shape=[8, 1, 8],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        engine.run(max_steps=10)
        assert engine.total_neutron_yield >= 0.0


# ---------------------------------------------------------------------------
# Section: Preset Smoke Tests
# ---------------------------------------------------------------------------

class TestPresetSmoke:
    """Smoke tests: each preset creates an engine and runs a few steps."""

    def test_tutorial_preset_runs(self):
        """Tutorial preset runs 5 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("tutorial"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5

    def test_pf1000_preset_runs(self):
        """PF-1000 preset runs 3 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("pf1000"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=3)
        assert summary["steps"] == 3

    def test_nx2_preset_runs(self):
        """NX2 preset runs 3 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("nx2"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=3)
        assert summary["steps"] == 3

    def test_all_presets_no_nan(self):
        """All presets produce finite state after 3 steps."""
        from dpf.presets import get_preset, get_preset_names

        for name in get_preset_names():
            config = SimulationConfig(**get_preset(name))
            engine = SimulationEngine(config)
            engine.run(max_steps=3)
            for key, arr in engine.state.items():
                if isinstance(arr, np.ndarray):
                    assert np.all(np.isfinite(arr)), (
                        f"NaN/Inf in {key} for preset '{name}'"
                    )


# ---------------------------------------------------------------------------
# Section: Python Engine Demotion (Phase R)
# ---------------------------------------------------------------------------

class TestEngineTier:
    """Tests for the engine_tier property."""

    def test_python_backend_is_teaching_tier(self, small_python_config: dict):
        """Python backend should report 'teaching' tier."""
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        assert engine.engine_tier == "teaching"

    def test_python_backend_name(self, small_python_config: dict):
        """Python backend should resolve correctly."""
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        assert engine.backend == "python"


class TestPythonDeprecationWarnings:
    """Tests for deprecation warnings on Python backend."""

    def test_large_grid_deprecation_warning(self, large_python_config: dict):
        """Python backend on large grid should emit DeprecationWarning."""
        config = SimulationConfig(**large_python_config)
        engine = SimulationEngine(config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.step()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            msg = str(dep_warnings[0].message)
            assert "non-conservative" in msg or "dp/dt" in msg

    def test_small_grid_no_deprecation(self, small_python_config: dict):
        """Python backend on small grid with short sim_time should NOT warn."""
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.step()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


class TestMHDSolverDocstring:
    """Tests for the MHD solver docstring warning."""

    def test_mhd_solver_docstring_warns_teaching(self):
        """MHDSolver module docstring should mention teaching/fallback."""
        from dpf.fluid import mhd_solver
        doc = mhd_solver.__doc__
        assert "teaching" in doc.lower() or "fallback" in doc.lower()
        assert "non-conservative" in doc.lower() or "dp/dt" in doc.lower()

    def test_mhd_solver_docstring_recommends_metal(self):
        """MHDSolver module docstring should recommend Metal/Athena++."""
        from dpf.fluid import mhd_solver
        doc = mhd_solver.__doc__
        assert "metal" in doc.lower()
        assert "athena" in doc.lower()


class TestBackendResolution:
    """Tests for backend auto-resolution and tier classification."""

    def test_resolve_python_explicit(self):
        """Explicitly requesting Python returns 'python'."""
        result = SimulationEngine._resolve_backend("python")
        assert result == "python"

    def test_resolve_unknown_raises(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            SimulationEngine._resolve_backend("nonexistent_backend")


class TestMetalPhysicsCoverage:
    """Tests verifying Metal solver supports key physics modules."""

    def test_metal_solver_has_transport_flags(self):
        """MetalMHDSolver should accept transport physics flags."""
        pytest.importorskip("torch")
        from dpf.metal.metal_solver import MetalMHDSolver
        solver = MetalMHDSolver(
            grid_shape=(8, 8, 8),
            dx=1e-3,
            device="cpu",
            use_ct=False,
            enable_hall=True,
            enable_braginskii_conduction=True,
            enable_braginskii_viscosity=True,
            enable_nernst=True,
        )
        assert solver.enable_hall is True
        assert solver.enable_braginskii_conduction is True
        assert solver.enable_braginskii_viscosity is True
        assert solver.enable_nernst is True

    def test_metal_solver_conservative_formulation(self):
        """Metal solver uses conservative variables (8-component: rho, mom, E, B)."""
        from dpf.metal.metal_riemann import IEN, NVAR
        assert NVAR == 8
        assert IEN == 4


class TestBackendPhysicsWarnings:
    """Tests for backend-specific physics warnings."""

    def test_python_engine_no_physics_skip_warning(
        self, small_python_config: dict, caplog: pytest.LogCaptureFixture,
    ):
        """Python engine should NOT warn about skipped physics (it has all)."""
        with caplog.at_level(logging.WARNING):
            config = SimulationConfig(**small_python_config)
            SimulationEngine(config)
        skip_msgs = [r for r in caplog.records if "skips physics" in r.message]
        assert len(skip_msgs) == 0


class TestRPlasmaCapIncrease:
    """Test that R_plasma cap was increased from 10 to 1000 Ohm."""

    def test_r_plasma_cap_is_1000(self):
        """The R_plasma cap in engine.py should allow up to 1000 Ohm."""
        source = inspect.getsource(SimulationEngine.step)
        assert "1000.0" in source or "1000" in source
