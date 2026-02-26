"""Verification test for Phase J.2: WALRUS Live Integration.

Tests:
1. WellExporter functionality (writing .npz files).
2. HybridEngine delegation in SimulationEngine.run().
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

OUTPUT_DIR = Path("tests/output_well_test")


@pytest.fixture
def clean_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    yield
    # Cleanup after test
    # if OUTPUT_DIR.exists():
    #     shutil.rmtree(OUTPUT_DIR)


def test_well_exporter_integration(clean_output):
    """Test that WellExporter writes files during simulation."""
    config_dict = {
        "grid_shape": [32, 32, 32],
        "dx": 0.01,
        "sim_time": 1e-7,
        "circuit": {
            "C": 1e-6, "V0": 1000.0, "L0": 1e-9, "R0": 0.01,
            "anode_radius": 0.01, "cathode_radius": 0.02
        },
        "diagnostics": {
            "hdf5_filename": str(OUTPUT_DIR / "test_diag.h5"),
            "output_interval": 10,
            "well_output_interval": 2,  # Export every 2 steps
            "well_filename_prefix": "well_test"
        },
        "fluid": {
            "backend": "python"
        }
    }
    config = SimulationConfig(**config_dict)
    engine = SimulationEngine(config)

    # Run for 5 steps
    engine.run(max_steps=5)

    # Check if exporter flushed
    # Since buffer_size defaults to 100, we need to call close() or wait validation
    # SimulationEngine.close() calls exporter.close()
    if hasattr(engine, "close"):
        engine.close()

    # Check for output files (h5 or npz)
    npz_files = list(OUTPUT_DIR.glob("well_test_*.npz"))
    h5_files = list(OUTPUT_DIR.glob("well_test_*.h5"))
    files = npz_files + h5_files
    assert len(files) > 0, "No Well export files found"

    # Load and check content
    if h5_files:
        import h5py
        print(f"Loading HDF5 file: {h5_files[0]}")
        with h5py.File(h5_files[0], "r") as f:
             data = {}
             for k in f:
                 data[k] = f[k][:]
    else:
        print(f"Loading NPZ file: {npz_files[0]}")
        data = np.load(npz_files[0])

    print(f"Well file keys: {list(data.keys())}")

    # Check required fields
    assert "density" in data
    assert "velocity" in data
    assert "pressure" in data

    # Check shapes
    # density: (n_traj=1, n_steps, nx, ny, nz)
    rho = data["density"]
    assert rho.ndim == 5
    assert rho.shape[0] == 1
    # We ran 5 steps, interval 2. Should save steps 0, 2, 4? Or 2, 4?
    # Logic: step_count % interval == 0.
    # step 0, 1, 2, 3, 4.
    # 0%2==0 (Yes), 2%2==0 (Yes), 4%2==0 (Yes). Total 3 steps.
    assert rho.shape[1] >= 2


@pytest.mark.slow
def test_hybrid_engine_delegation(clean_output):
    """Test that backend='hybrid' delegates to HybridEngine."""
    config_dict = {
        "grid_shape": [8, 8, 8],
        "dx": 0.01,
        "sim_time": 1e-6,
        "circuit": {
            "C": 1e-6, "V0": 1000.0, "L0": 1e-9, "R0": 0.01,
            "anode_radius": 0.01, "cathode_radius": 0.02
        },
        "fluid": {
            "backend": "hybrid",
            "handoff_fraction": 0.5, # 50% physics, 50% surrogate
            "validation_interval": 10
        },
        "ai": {
            "device": "cpu"
            # No checkpoint -> fallback mock
        },
        "diagnostics": {
            "hdf5_filename": str(OUTPUT_DIR / "hybrid_diag.h5"),
            "well_output_interval": 0 # Disable well export to avoid clash/complexity
        }
    }
    config = SimulationConfig(**config_dict)
    engine = SimulationEngine(config)

    # Run short simulation
    # Need enough steps to satisfy history_length=4
    # max_steps=10, handoff=0.5 -> 5 physics steps (>4)
    summary = engine.run(max_steps=10)

    # Check summary contains HybridEngine keys
    assert "total_steps" in summary
    assert "physics_steps" in summary
    assert "surrogate_steps" in summary

    # Since max_steps=10, handoff=0.5 -> 5 physics, 5 surrogate
    assert summary["physics_steps"] == 5
    assert summary["surrogate_steps"] == 5
