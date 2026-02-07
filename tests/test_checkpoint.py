"""Tests for Phase 6: Checkpoint/restart and infrastructure.

Test categories:
1. Save checkpoint creates valid HDF5 file
2. Load checkpoint recovers saved state
3. Round-trip: save then load gives identical state
4. Engine checkpoint integration
5. Engine restart continues correctly
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# ====================================================
# Checkpoint Save/Load Tests
# ====================================================

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
            "current": 1e6,
            "voltage": 1e4,
            "energy_cap": 100.0,
            "energy_ind": 50.0,
            "energy_res": 10.0,
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


# ====================================================
# Engine Checkpoint Integration Tests
# ====================================================

class TestEngineCheckpoint:
    """Tests for engine checkpoint/restart integration."""

    def _make_engine(self):
        """Helper to create a test engine."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

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

            # Create new engine and restore
            engine2 = self._make_engine()
            engine2.load_from_checkpoint(fname)

            assert engine2.time == pytest.approx(time_at_checkpoint)
            assert engine2.step_count == steps_at_checkpoint

            # Continue running — request more steps than checkpoint
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
            # Should have auto-checkpointed at step 5 and 10
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

            # Load raw data
            data = load_checkpoint(fname)

            # Verify key arrays match engine state
            np.testing.assert_array_almost_equal(
                data["state"]["rho"], engine.state["rho"],
            )
            np.testing.assert_array_almost_equal(
                data["state"]["velocity"], engine.state["velocity"],
            )
        finally:
            os.unlink(fname)


# ====================================================
# HLL Numba Acceleration Test
# ====================================================

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
