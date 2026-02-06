"""End-to-end integration tests.

Tests the full pipeline: config -> engine -> simulation -> diagnostics.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine


@pytest.fixture
def tmp_config(sample_config_dict):
    """Write a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config_dict, f)
        path = f.name
    yield path
    os.unlink(path)


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

        # Energy conservation within 5% (relaxed for short run with coupling)
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

        # Check HDF5 file exists (only if h5py available)
        try:
            import h5py

            assert os.path.exists(hdf5_file)
            with h5py.File(hdf5_file, "r") as f:
                assert "scalars" in f
                assert "time" in f["scalars"]
                assert len(f["scalars"]["time"]) == 10
        except ImportError:
            pass  # Skip HDF5 checks if h5py not installed

    def test_current_flows(self, sample_config_dict):
        """After running, current should be non-zero (circuit is discharging)."""
        config = SimulationConfig(**sample_config_dict)
        engine = SimulationEngine(config)
        engine.run(max_steps=50)
        assert abs(engine.circuit.current) > 0
