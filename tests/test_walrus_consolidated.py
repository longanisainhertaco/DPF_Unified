"""Consolidated WALRUS/AI test suite for DPF-Unified.

Coverage (17 source files merged):
- test_phase_h_field_mapping.py      — DPF <-> Well field name/shape transforms
- test_phase_h_dataset_validator.py  — Well HDF5 schema + NaN/Inf/energy checks
- test_phase_h_well_exporter.py      — WellExporter (DPF state -> Well HDF5)
- test_phase_h_batch_runner.py       — BatchRunner, ParameterRange, Latin Hypercube
- test_phase_h_hybrid.py             — HybridEngine validation (Phase H)
- test_phase_i_confidence.py         — EnsemblePredictor / PredictionWithConfidence
- test_phase_i_hybrid_engine.py      — HybridEngine step-by-step (Phase I)
- test_phase_i_instability.py        — InstabilityDetector
- test_phase_i_surrogate.py          — DPFSurrogate (torch-mocked unit tests)
- test_phase_i_inverse_design.py     — InverseDesigner
- test_phase_j_well_loader.py        — WellDataset / collate_well_to_dpf
- test_phase_j2_walrus_integration.py — Phase J.2 WALRUS live integration
- test_phase_r_walrus_hybrid.py      — Phase R WALRUS-hybrid engine
- test_phase_z_walrus_benchmarks.py  — Bennett / Noh benchmark validation
- test_verification_walrus.py        — Real-checkpoint verification (all slow)
- test_well_integration.py           — WellExporter integration (live engine)
- test_well_loader_stats.py          — WellDataset stats / normalization
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None

from dpf.ai.batch_runner import BatchResult, BatchRunner, ParameterRange  # noqa: E402
from dpf.ai.confidence import EnsemblePredictor, PredictionWithConfidence  # noqa: E402
from dpf.ai.dataset_validator import DatasetValidator, ValidationResult  # noqa: E402
from dpf.ai.field_mapping import (  # noqa: E402
    CIRCUIT_SCALARS,
    DPF_TO_WELL_NAMES,
    FIELD_UNITS,
    REQUIRED_STATE_KEYS,
    SCALAR_FIELDS,
    VECTOR_FIELDS,
    WELL_TO_DPF_NAMES,
    dpf_scalar_to_well,
    dpf_vector_to_well,
    infer_geometry,
    spatial_shape,
    validate_state_dict,
    well_scalar_to_dpf,
    well_vector_to_dpf,
)
from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402
from dpf.ai.instability_detector import InstabilityDetector, InstabilityEvent  # noqa: E402
from dpf.ai.inverse_design import InverseDesigner, InverseResult  # noqa: E402
from dpf.ai.well_exporter import WellExporter  # noqa: E402
from dpf.config import SimulationConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Section: Phase H — field_mapping
# ---------------------------------------------------------------------------


def test_scalar_fields_constant():
    """Verify SCALAR_FIELDS contains expected DPF -> Well mappings."""
    assert "rho" in SCALAR_FIELDS
    assert "Te" in SCALAR_FIELDS
    assert "Ti" in SCALAR_FIELDS
    assert "pressure" in SCALAR_FIELDS
    assert "psi" in SCALAR_FIELDS
    assert SCALAR_FIELDS["rho"] == "density"
    assert SCALAR_FIELDS["Te"] == "electron_temp"
    assert len(SCALAR_FIELDS) == 5


def test_vector_fields_constant():
    """Verify VECTOR_FIELDS contains expected DPF -> Well mappings."""
    assert "B" in VECTOR_FIELDS
    assert "velocity" in VECTOR_FIELDS
    assert VECTOR_FIELDS["B"] == "magnetic_field"
    assert VECTOR_FIELDS["velocity"] == "velocity"
    assert len(VECTOR_FIELDS) == 2


def test_dpf_to_well_names_merges_scalar_and_vector():
    """Verify DPF_TO_WELL_NAMES combines scalar and vector mappings."""
    assert set(SCALAR_FIELDS.keys()).issubset(DPF_TO_WELL_NAMES.keys())
    assert set(VECTOR_FIELDS.keys()).issubset(DPF_TO_WELL_NAMES.keys())
    assert len(DPF_TO_WELL_NAMES) == len(SCALAR_FIELDS) + len(VECTOR_FIELDS)


def test_well_to_dpf_names_is_reverse():
    """Verify WELL_TO_DPF_NAMES is the reverse mapping of DPF_TO_WELL_NAMES."""
    for dpf_key, well_key in DPF_TO_WELL_NAMES.items():
        assert WELL_TO_DPF_NAMES[well_key] == dpf_key
    assert len(WELL_TO_DPF_NAMES) == len(DPF_TO_WELL_NAMES)


def test_field_units_constant():
    """Verify FIELD_UNITS contains units for all DPF fields."""
    assert FIELD_UNITS["rho"] == "kg/m^3"
    assert FIELD_UNITS["Te"] == "K"
    assert FIELD_UNITS["Ti"] == "K"
    assert FIELD_UNITS["pressure"] == "Pa"
    assert FIELD_UNITS["psi"] == "T*m/s"
    assert FIELD_UNITS["B"] == "T"
    assert FIELD_UNITS["velocity"] == "m/s"
    assert len(FIELD_UNITS) == 7


def test_circuit_scalars_constant():
    """Verify CIRCUIT_SCALARS contains non-spatial circuit quantities."""
    assert "current" in CIRCUIT_SCALARS
    assert "voltage" in CIRCUIT_SCALARS
    assert "R_plasma" in CIRCUIT_SCALARS
    assert "Z_bar" in CIRCUIT_SCALARS
    assert len(CIRCUIT_SCALARS) == 8


def test_required_state_keys_constant():
    """Verify REQUIRED_STATE_KEYS contains all mandatory state dict keys."""
    expected = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
    assert set(REQUIRED_STATE_KEYS) == expected
    assert len(REQUIRED_STATE_KEYS) == 7


def test_dpf_scalar_to_well_shape():
    """Verify dpf_scalar_to_well converts (nx,ny,nz) -> (n_traj,n_steps,nx,ny,nz)."""
    field = np.random.rand(4, 4, 4)
    result = dpf_scalar_to_well(field, traj_idx=0, step_idx=0, n_traj=2, n_steps=3)
    assert result.shape == (2, 3, 4, 4, 4)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], field, rtol=1e-6)
    assert np.all(result[1, 0] == 0)
    assert np.all(result[0, 1] == 0)


def test_dpf_vector_to_well_shape_and_moveaxis():
    """Verify dpf_vector_to_well converts (3,nx,ny,nz) -> (n_traj,n_steps,nx,ny,nz,3)."""
    field = np.random.rand(3, 4, 4, 4)
    result = dpf_vector_to_well(field, traj_idx=0, step_idx=0, n_traj=1, n_steps=1)
    assert result.shape == (1, 1, 4, 4, 4, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0, :, :, :, 0], field[0], rtol=1e-6)
    np.testing.assert_allclose(result[0, 0, :, :, :, 1], field[1], rtol=1e-6)
    np.testing.assert_allclose(result[0, 0, :, :, :, 2], field[2], rtol=1e-6)


def test_well_scalar_to_dpf_reverse():
    """Verify well_scalar_to_dpf extracts (nx,ny,nz) from (n_traj,n_steps,nx,ny,nz)."""
    well_field = np.random.rand(2, 3, 4, 4, 4).astype(np.float32)
    result = well_scalar_to_dpf(well_field, traj_idx=1, step_idx=2)
    assert result.shape == (4, 4, 4)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, well_field[1, 2], rtol=1e-6)


def test_well_vector_to_dpf_reverse():
    """Verify well_vector_to_dpf extracts (3,nx,ny,nz) from (n_traj,n_steps,nx,ny,nz,3)."""
    well_field = np.random.rand(1, 2, 4, 4, 4, 3).astype(np.float32)
    result = well_vector_to_dpf(well_field, traj_idx=0, step_idx=1)
    assert result.shape == (3, 4, 4, 4)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[0], well_field[0, 1, :, :, :, 0], rtol=1e-6)
    np.testing.assert_allclose(result[1], well_field[0, 1, :, :, :, 1], rtol=1e-6)
    np.testing.assert_allclose(result[2], well_field[0, 1, :, :, :, 2], rtol=1e-6)


def test_round_trip_scalar():
    """Verify DPF -> Well -> DPF preserves scalar field values."""
    original = np.random.rand(8, 8, 8) * 1e3
    well_fmt = dpf_scalar_to_well(original)
    recovered = well_scalar_to_dpf(well_fmt)
    np.testing.assert_allclose(recovered, original, rtol=1e-6)


def test_round_trip_vector():
    """Verify DPF -> Well -> DPF preserves vector field values."""
    original = np.random.rand(3, 8, 8, 8) * 1e3
    well_fmt = dpf_vector_to_well(original)
    recovered = well_vector_to_dpf(well_fmt)
    np.testing.assert_allclose(recovered, original, rtol=1e-6)


def test_infer_geometry_cartesian():
    """Verify infer_geometry returns 'cartesian' when ny > 1."""
    state = {"rho": np.zeros((4, 4, 4))}
    assert infer_geometry(state) == "cartesian"


def test_infer_geometry_cylindrical():
    """Verify infer_geometry returns 'cylindrical' when ny == 1."""
    state = {"rho": np.zeros((8, 1, 8))}
    assert infer_geometry(state) == "cylindrical"


def test_spatial_shape_from_state():
    """Verify spatial_shape extracts (nx, ny, nz) from state dict."""
    state = {
        "rho": np.zeros((4, 5, 6)),
        "velocity": np.zeros((3, 4, 5, 6)),
    }
    shape = spatial_shape(state)
    assert shape == (4, 5, 6)


def test_spatial_shape_empty_state_raises():
    """Verify spatial_shape raises ValueError on empty state dict."""
    with pytest.raises(ValueError, match="No scalar field found"):
        spatial_shape({})


def test_validate_state_dict_empty_returns_all_missing():
    """Verify validate_state_dict returns errors for all missing fields."""
    errors = validate_state_dict({})
    assert len(errors) == len(REQUIRED_STATE_KEYS)
    assert all("Missing required field" in err for err in errors)


def test_validate_state_dict_valid_state_returns_empty():
    """Verify validate_state_dict returns empty list for valid state."""
    state = {
        "rho": np.zeros((4, 4, 4)),
        "Te": np.zeros((4, 4, 4)),
        "Ti": np.zeros((4, 4, 4)),
        "pressure": np.zeros((4, 4, 4)),
        "psi": np.zeros((4, 4, 4)),
        "velocity": np.zeros((3, 4, 4, 4)),
        "B": np.zeros((3, 4, 4, 4)),
    }
    errors = validate_state_dict(state)
    assert errors == []


def test_validate_state_dict_wrong_vector_shape():
    """Verify validate_state_dict detects incorrect vector field shape."""
    state = {
        "rho": np.zeros((4, 4, 4)),
        "Te": np.zeros((4, 4, 4)),
        "Ti": np.zeros((4, 4, 4)),
        "pressure": np.zeros((4, 4, 4)),
        "psi": np.zeros((4, 4, 4)),
        "velocity": np.zeros((4, 4, 4)),  # Wrong: should be (3, 4, 4, 4)
        "B": np.zeros((3, 4, 4, 4)),
    }
    errors = validate_state_dict(state)
    assert any("velocity" in err and "shape" in err for err in errors)


def test_validate_state_dict_inconsistent_scalar_shapes():
    """Verify validate_state_dict detects inconsistent scalar field shapes."""
    state = {
        "rho": np.zeros((4, 4, 4)),
        "Te": np.zeros((5, 5, 5)),  # Inconsistent with rho
        "Ti": np.zeros((4, 4, 4)),
        "pressure": np.zeros((4, 4, 4)),
        "psi": np.zeros((4, 4, 4)),
        "velocity": np.zeros((3, 4, 4, 4)),
        "B": np.zeros((3, 4, 4, 4)),
    }
    errors = validate_state_dict(state)
    assert any("Inconsistent scalar field shapes" in err for err in errors)


# ---------------------------------------------------------------------------
# Section: Phase H — dataset_validator
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_well_file(tmp_path: Path) -> Path:
    """Create a valid Well HDF5 file with 4x4x4 grid, 1 trajectory, 5 timesteps."""
    path = tmp_path / "valid_well.h5"
    nx, ny, nz = 4, 4, 4
    n_traj = 1
    n_steps = 5

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test_dpf_dataset"
        f.attrs["n_trajectories"] = n_traj

        t0_group = f.create_group("t0_fields")
        density = np.random.uniform(1.0, 2.0, (n_traj, n_steps, nz, ny, nx))
        t0_group.create_dataset("density", data=density)
        pressure = np.random.uniform(0.5, 1.5, (n_traj, n_steps, nz, ny, nx))
        t0_group.create_dataset("pressure", data=pressure)

        t1_group = f.create_group("t1_fields")
        velocity = np.random.uniform(-0.1, 0.1, (n_traj, n_steps, nz, ny, nx, 3))
        t1_group.create_dataset("velocity", data=velocity)
        B_field = np.random.uniform(-0.5, 0.5, (n_traj, n_steps, nz, ny, nx, 3))
        t1_group.create_dataset("B", data=B_field)

        dim_group = f.create_group("dimensions")
        dim_group.create_dataset("x", data=np.linspace(0.0, 1.0, nx))
        dim_group.create_dataset("y", data=np.linspace(0.0, 1.0, ny))
        dim_group.create_dataset("z", data=np.linspace(0.0, 1.0, nz))

        bc_group = f.create_group("boundary_conditions")
        bc_group.attrs["x_bc"] = "periodic"
        bc_group.attrs["y_bc"] = "periodic"
        bc_group.attrs["z_bc"] = "outflow"

    return path


def test_validation_result_defaults():
    """ValidationResult defaults to valid=True with empty errors/warnings."""
    result = ValidationResult()
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.n_trajectories == 0
    assert result.n_timesteps == 0
    assert result.field_stats == {}
    assert result.energy_drift == 0.0


def test_validate_file_valid(valid_well_file: Path):
    """validate_file on valid file returns valid=True with correct counts."""
    validator = DatasetValidator()
    result = validator.validate_file(valid_well_file)

    assert result.valid is True
    assert result.n_trajectories == 1
    assert result.n_timesteps == 5


def test_validate_file_nonexistent():
    """validate_file on nonexistent file returns valid=False with error."""
    validator = DatasetValidator()
    result = validator.validate_file("/nonexistent/path/to/file.h5")

    assert result.valid is False
    assert len(result.errors) == 1
    assert "not found" in result.errors[0].lower()


def test_validate_file_empty_errors_for_valid(valid_well_file: Path):
    """validate_file error list is empty for valid file."""
    validator = DatasetValidator()
    result = validator.validate_file(valid_well_file)

    assert result.errors == []


def test_check_well_schema_valid(valid_well_file: Path):
    """check_well_schema passes for valid Well file."""
    validator = DatasetValidator()
    errors = validator.check_well_schema(valid_well_file)

    assert errors == []


def test_check_well_schema_missing_fields(tmp_path: Path):
    """check_well_schema returns error when missing t0_fields AND t1_fields."""
    path = tmp_path / "no_fields.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    errors = validator.check_well_schema(path)

    assert len(errors) == 1
    assert "t0_fields or t1_fields" in errors[0]


def test_check_well_schema_missing_dimensions(tmp_path: Path):
    """check_well_schema returns error when missing dimensions."""
    path = tmp_path / "no_dimensions.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.zeros((1, 5, 4, 4, 4)))
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    errors = validator.check_well_schema(path)

    assert len(errors) == 1
    assert "dimensions" in errors[0]


def test_check_well_schema_missing_boundary_conditions(tmp_path: Path):
    """check_well_schema returns error when missing boundary_conditions."""
    path = tmp_path / "no_bc.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.zeros((1, 5, 4, 4, 4)))
        f.create_group("dimensions")

    validator = DatasetValidator()
    errors = validator.check_well_schema(path)

    assert len(errors) == 1
    assert "boundary_conditions" in errors[0]


def test_check_well_schema_missing_root_attrs(tmp_path: Path):
    """check_well_schema returns error when missing root attrs."""
    path = tmp_path / "no_attrs.h5"

    with h5py.File(path, "w") as f:
        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.zeros((1, 5, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    errors = validator.check_well_schema(path)

    assert len(errors) == 1
    assert "n_trajectories or dataset_name" in errors[0]


def test_check_nan_inf_clean_data(valid_well_file: Path):
    """check_nan_inf returns empty list for clean data."""
    validator = DatasetValidator()
    bad_fields = validator.check_nan_inf(valid_well_file)

    assert bad_fields == []


def test_check_nan_inf_detects_nan_scalar(tmp_path: Path):
    """check_nan_inf detects NaN in scalar field."""
    path = tmp_path / "nan_scalar.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.attrs["n_trajectories"] = 1

        t0_group = f.create_group("t0_fields")
        density = np.ones((1, 5, 4, 4, 4))
        density[0, 2, 1, 1, 1] = np.nan
        t0_group.create_dataset("density", data=density)

        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    bad_fields = validator.check_nan_inf(path)

    assert len(bad_fields) == 1
    assert "t0_fields/density" in bad_fields


def test_check_nan_inf_detects_inf_vector(tmp_path: Path):
    """check_nan_inf detects Inf in vector field."""
    path = tmp_path / "inf_vector.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.attrs["n_trajectories"] = 1

        t1_group = f.create_group("t1_fields")
        velocity = np.zeros((1, 5, 4, 4, 4, 3))
        velocity[0, 3, 2, 2, 2, 1] = np.inf
        t1_group.create_dataset("velocity", data=velocity)

        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    bad_fields = validator.check_nan_inf(path)

    assert len(bad_fields) == 1
    assert "t1_fields/velocity" in bad_fields


def test_check_energy_conservation_no_field(valid_well_file: Path):
    """check_energy_conservation returns 0.0 when no energy_conservation field."""
    validator = DatasetValidator()
    drift = validator.check_energy_conservation(valid_well_file)

    assert drift == 0.0


def test_check_energy_conservation_returns_drift(tmp_path: Path):
    """check_energy_conservation returns correct drift value."""
    path = tmp_path / "with_energy.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.attrs["n_trajectories"] = 1

        scalars_group = f.create_group("scalars")
        energy_cons = np.array([1.0, 0.99, 1.01, 0.97, 1.02])
        scalars_group.create_dataset("energy_conservation", data=energy_cons)

        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.ones((1, 5, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    drift = validator.check_energy_conservation(path)

    assert drift == pytest.approx(0.03, abs=1e-6)


def test_check_energy_conservation_triggers_warning(tmp_path: Path):
    """check_energy_conservation with large drift triggers warning in validate_file."""
    path = tmp_path / "large_drift.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.attrs["n_trajectories"] = 1

        scalars_group = f.create_group("scalars")
        energy_cons = np.array([1.0, 0.95, 1.05, 0.90, 1.10])
        scalars_group.create_dataset("energy_conservation", data=energy_cons)

        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.ones((1, 5, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator(energy_drift_threshold=0.05)
    result = validator.validate_file(path)

    assert len(result.warnings) == 1
    assert "Energy drift" in result.warnings[0]
    assert "exceeds threshold" in result.warnings[0]


def test_compute_field_statistics(valid_well_file: Path):
    """compute_field_statistics returns mean/std/min/max/n_nan/n_inf for each field."""
    validator = DatasetValidator()
    stats = validator.compute_field_statistics(valid_well_file)

    assert "density" in stats
    assert "pressure" in stats
    assert "velocity" in stats
    assert "B" in stats

    density_stats = stats["density"]
    assert "mean" in density_stats
    assert "std" in density_stats
    assert "min" in density_stats
    assert "max" in density_stats
    assert "n_nan" in density_stats
    assert "n_inf" in density_stats

    assert 1.0 <= density_stats["mean"] <= 2.0
    assert density_stats["n_nan"] == 0
    assert density_stats["n_inf"] == 0


def test_validate_directory_finds_all_files(tmp_path: Path):
    """validate_directory finds and validates all .h5 files."""
    for i in range(3):
        path = tmp_path / f"dataset_{i}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["dataset_name"] = f"test_{i}"
            f.attrs["n_trajectories"] = 1
            t0_group = f.create_group("t0_fields")
            t0_group.create_dataset("density", data=np.ones((1, 5, 4, 4, 4)))
            f.create_group("dimensions")
            f.create_group("boundary_conditions")

    validator = DatasetValidator()
    results = validator.validate_directory(tmp_path)

    assert len(results) == 3
    assert all(result.valid for result in results.values())


def test_validate_directory_nonexistent(tmp_path: Path):
    """validate_directory raises FileNotFoundError for nonexistent dir."""
    validator = DatasetValidator()
    with pytest.raises(FileNotFoundError, match="Validation directory not found"):
        validator.validate_directory(tmp_path / "nonexistent")


def test_summary_report_formatting(tmp_path: Path):
    """summary_report formats readable output with VALID/INVALID status."""
    valid_path = tmp_path / "valid.h5"
    with h5py.File(valid_path, "w") as f:
        f.attrs["dataset_name"] = "valid_test"
        f.attrs["n_trajectories"] = 2
        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.ones((2, 10, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    invalid_path = tmp_path / "invalid.h5"
    with h5py.File(invalid_path, "w") as f:
        f.attrs["dataset_name"] = "invalid_test"

    validator = DatasetValidator()
    results = {
        "valid.h5": validator.validate_file(valid_path),
        "invalid.h5": validator.validate_file(invalid_path),
    }

    report = validator.summary_report(results)

    assert "VALID" in report
    assert "INVALID" in report
    assert "valid.h5" in report
    assert "invalid.h5" in report
    assert "1/2 files valid" in report


def test_summary_report_includes_errors_and_warnings(tmp_path: Path):
    """summary_report includes error and warning lines."""
    path = tmp_path / "errors.h5"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "error_test"

    path2 = tmp_path / "warnings.h5"
    with h5py.File(path2, "w") as f:
        f.attrs["dataset_name"] = "warning_test"
        f.attrs["n_trajectories"] = 1

        scalars_group = f.create_group("scalars")
        energy_cons = np.array([1.0, 0.90, 1.10])
        scalars_group.create_dataset("energy_conservation", data=energy_cons)

        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.ones((1, 3, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator(energy_drift_threshold=0.05)
    results = {
        "errors.h5": validator.validate_file(path),
        "warnings.h5": validator.validate_file(path2),
    }

    report = validator.summary_report(results)

    assert "Errors:" in report
    assert "Warnings:" in report
    assert "Energy drift" in report


# ---------------------------------------------------------------------------
# Section: Phase H — well_exporter
# ---------------------------------------------------------------------------


def make_dpf_state(nx: int = 4, ny: int = 4, nz: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "rho": rng.random((nx, ny, nz)),
        "pressure": rng.random((nx, ny, nz)),
        "Te": rng.random((nx, ny, nz)) * 1e4,
        "Ti": rng.random((nx, ny, nz)) * 1e4,
        "psi": rng.random((nx, ny, nz)) * 1e-3,
        "velocity": rng.random((3, nx, ny, nz)),
        "B": rng.random((3, nx, ny, nz)),
    }


def make_circuit_scalars() -> dict[str, float]:
    return {
        "current": 1.5e5,
        "voltage": 2.5e3,
        "energy_conservation": 0.999,
        "R_plasma": 0.05,
        "Z_bar": 3.5,
    }


def test_well_exporter_init_stores_parameters(tmp_path: Path) -> None:
    """WellExporter __init__ stores all parameters correctly."""
    output_path = tmp_path / "test.h5"
    grid_shape = (8, 8, 8)
    dx = 0.001
    dz = 0.002
    geometry = "cylindrical"
    sim_params = {"foo": "bar", "voltage": 1.5e4}

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=grid_shape,
        dx=dx,
        dz=dz,
        geometry=geometry,
        sim_params=sim_params,
    )

    assert exporter.output_path == output_path
    assert exporter.grid_shape == grid_shape
    assert exporter.dx == dx
    assert exporter.dz == dz
    assert exporter.geometry == geometry
    assert exporter.sim_params == sim_params
    assert exporter._snapshots == []
    assert exporter._times == []
    assert exporter._circuit_history == []


def test_well_exporter_init_defaults(tmp_path: Path) -> None:
    """WellExporter __init__ applies default values for optional parameters."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    assert exporter.dz == 0.001
    assert exporter.geometry == "cartesian"
    assert exporter.sim_params == {}


def test_well_exporter_init_accepts_str_path(tmp_path: Path) -> None:
    """WellExporter __init__ accepts string path and converts to Path."""
    output_path = str(tmp_path / "test.h5")
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    assert isinstance(exporter.output_path, Path)
    assert exporter.output_path == Path(output_path)


def test_add_snapshot_stores_state_and_time(tmp_path: Path) -> None:
    """add_snapshot stores state dict and time in internal buffers."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    state = make_dpf_state()
    time = 1.5e-6
    exporter.add_snapshot(state, time)

    assert len(exporter._snapshots) == 1
    assert len(exporter._times) == 1
    assert exporter._times[0] == time
    assert np.allclose(exporter._snapshots[0]["rho"], state["rho"])


def test_add_snapshot_accumulates_multiple_snapshots(tmp_path: Path) -> None:
    """add_snapshot accumulates multiple snapshots over time."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    for i in range(5):
        exporter.add_snapshot(make_dpf_state(), i * 1e-7)

    assert len(exporter._snapshots) == 5
    assert len(exporter._times) == 5
    assert exporter._times == [0.0, 1e-7, 2e-7, 3e-7, 4e-7]


def test_add_snapshot_stores_circuit_scalars(tmp_path: Path) -> None:
    """add_snapshot stores circuit scalars when provided."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    circuit_scalars = make_circuit_scalars()
    exporter.add_snapshot(make_dpf_state(), 1e-6, circuit_scalars)

    assert len(exporter._circuit_history) == 1
    assert exporter._circuit_history[0]["current"] == circuit_scalars["current"]
    assert exporter._circuit_history[0]["voltage"] == circuit_scalars["voltage"]


def test_add_snapshot_handles_missing_circuit_scalars(tmp_path: Path) -> None:
    """add_snapshot stores empty dict when circuit_scalars is None."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)

    assert len(exporter._circuit_history) == 1
    assert exporter._circuit_history[0] == {}


def test_add_snapshot_copies_state_dict(tmp_path: Path) -> None:
    """add_snapshot creates a shallow copy of the state dict."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)

    original_keys = set(state.keys())
    del state["rho"]

    assert set(exporter._snapshots[0].keys()) == original_keys


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_raises_when_no_snapshots(tmp_path: Path) -> None:
    """finalize raises ValueError when no snapshots have been accumulated."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    with pytest.raises(ValueError, match="No snapshots accumulated"):
        exporter.finalize()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_writes_hdf5_file_to_disk(tmp_path: Path) -> None:
    """finalize writes HDF5 file to disk at specified path."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    result_path = exporter.finalize()

    assert result_path == output_path
    assert output_path.exists()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_root_attributes(tmp_path: Path) -> None:
    """finalize creates root-level HDF5 attributes."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert f.attrs["dataset_name"] == "dpf_simulation"
        assert f.attrs["grid_type"] == "cartesian"
        assert f.attrs["n_spatial_dims"] == 3
        assert f.attrs["n_trajectories"] == 1


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_dimensions_group_cartesian(tmp_path: Path) -> None:
    """finalize creates /dimensions group with x/y/z coordinates for Cartesian."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    dx, dz = 0.001, 0.002

    exporter = WellExporter(
        output_path=output_path, grid_shape=(nx, ny, nz), dx=dx, dz=dz, geometry="cartesian"
    )
    exporter.add_snapshot(make_dpf_state(nx, ny, nz), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        dims = f["dimensions"]
        assert "x" in dims
        assert "y" in dims
        assert "z" in dims
        assert dims["x"].shape == (nx,)
        assert dims["y"].shape == (ny,)
        assert dims["z"].shape == (nz,)
        assert dims["x"].dtype == np.float32
        assert pytest.approx(dims["x"][-1], rel=1e-5) == nx * dx
        assert pytest.approx(dims["z"][-1], rel=1e-5) == nz * dz


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_dimensions_group_cylindrical(tmp_path: Path) -> None:
    """finalize creates /dimensions group with r/theta/z for cylindrical."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4

    exporter = WellExporter(
        output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001, geometry="cylindrical"
    )
    exporter.add_snapshot(make_dpf_state(nx, ny, nz), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        dims = f["dimensions"]
        assert "r" in dims
        assert "theta" in dims
        assert "z" in dims
        assert "x" not in dims
        assert "y" not in dims
        assert dims["r"].shape == (nx,)
        assert dims["theta"].shape == (ny,)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_time_array(tmp_path: Path) -> None:
    """finalize creates /dimensions/time array with all timesteps."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    times = [0.0, 1e-7, 2e-7, 3e-7]
    for t in times:
        exporter.add_snapshot(make_dpf_state(), t)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        time_arr = f["dimensions/time"][:]
        assert len(time_arr) == len(times)
        assert time_arr.dtype == np.float32
        assert np.allclose(time_arr, times)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_t0_fields_group(tmp_path: Path) -> None:
    """finalize creates /t0_fields group with scalar fields."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    exporter = WellExporter(output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001)
    exporter.add_snapshot(make_dpf_state(nx, ny, nz), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        t0_grp = f["t0_fields"]
        for _dpf_name, well_name in SCALAR_FIELDS.items():
            assert well_name in t0_grp
            dataset = t0_grp[well_name]
            assert dataset.shape == (1, 1, nx, ny, nz)
            assert dataset.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_t1_fields_group(tmp_path: Path) -> None:
    """finalize creates /t1_fields group with vector fields."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    exporter = WellExporter(output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001)
    exporter.add_snapshot(make_dpf_state(nx, ny, nz), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        t1_grp = f["t1_fields"]
        for _dpf_name, well_name in VECTOR_FIELDS.items():
            assert well_name in t1_grp
            dataset = t1_grp[well_name]
            assert dataset.shape == (1, 1, nx, ny, nz, 3)
            assert dataset.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_scalar_field_shapes(tmp_path: Path) -> None:
    """Scalar fields have correct shape: (n_traj, n_steps, nx, ny, nz)."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    n_steps = 3
    n_traj = 1

    exporter = WellExporter(output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001)
    for i in range(n_steps):
        exporter.add_snapshot(make_dpf_state(nx, ny, nz), i * 1e-7)
    exporter.finalize(n_trajectories=n_traj)

    with h5py.File(output_path, "r") as f:
        density = f["t0_fields/density"]
        assert density.shape == (n_traj, n_steps, nx, ny, nz)
        assert density.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_vector_field_shapes(tmp_path: Path) -> None:
    """Vector fields have correct shape: (n_traj, n_steps, nx, ny, nz, 3)."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    n_steps = 3
    n_traj = 1

    exporter = WellExporter(output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001)
    for i in range(n_steps):
        exporter.add_snapshot(make_dpf_state(nx, ny, nz), i * 1e-7)
    exporter.finalize(n_trajectories=n_traj)

    with h5py.File(output_path, "r") as f:
        magnetic_field = f["t1_fields/magnetic_field"]
        assert magnetic_field.shape == (n_traj, n_steps, nx, ny, nz, 3)
        assert magnetic_field.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_vector_field_component_order(tmp_path: Path) -> None:
    """Vector fields use moveaxis to convert from (3, nx, ny, nz) to (..., 3)."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4

    exporter = WellExporter(output_path=output_path, grid_shape=(nx, ny, nz), dx=0.001)
    state = make_dpf_state(nx, ny, nz)
    state["B"] = np.zeros((3, nx, ny, nz))
    state["B"][0, 0, 0, 0] = 1.0
    state["B"][1, 0, 0, 0] = 2.0
    state["B"][2, 0, 0, 0] = 3.0

    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        B = f["t1_fields/magnetic_field"][0, 0, 0, 0, 0, :]
        assert pytest.approx(B[0]) == 1.0
        assert pytest.approx(B[1]) == 2.0
        assert pytest.approx(B[2]) == 3.0


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_scalars_group(tmp_path: Path) -> None:
    """finalize creates /scalars group with circuit quantities."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6, make_circuit_scalars())
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert "scalars" in f
        scalars_grp = f["scalars"]
        assert "current" in scalars_grp
        assert "voltage" in scalars_grp
        assert scalars_grp["current"].shape == (1, 1)
        assert scalars_grp["current"].dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_boundary_conditions_group(tmp_path: Path) -> None:
    """finalize creates /boundary_conditions group with appropriate attrs."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, geometry="cartesian"
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        bc_grp = f["boundary_conditions"]
        assert bc_grp.attrs["geometry_type"] == "cartesian"
        assert bc_grp.attrs["all"] == "periodic"


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cylindrical_boundary_conditions(tmp_path: Path) -> None:
    """Cylindrical geometry creates appropriate boundary condition attrs."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, geometry="cylindrical"
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        bc_grp = f["boundary_conditions"]
        assert bc_grp.attrs["geometry_type"] == "cylindrical"
        assert bc_grp.attrs["r_inner"] == "axis"
        assert bc_grp.attrs["r_outer"] == "outflow"
        assert bc_grp.attrs["z_low"] == "wall"
        assert bc_grp.attrs["z_high"] == "wall"


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cartesian_boundary_conditions(tmp_path: Path) -> None:
    """Cartesian geometry creates appropriate boundary condition attrs."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, geometry="cartesian"
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        bc_grp = f["boundary_conditions"]
        assert bc_grp.attrs["geometry_type"] == "cartesian"
        assert bc_grp.attrs["all"] == "periodic"


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_custom_sim_params_stored(tmp_path: Path) -> None:
    """Custom sim_params are stored as root-level attributes."""
    output_path = tmp_path / "test.h5"
    sim_params = {
        "voltage": 1.5e4,
        "capacitance": 4.0e-6,
        "inductance": 1.0e-8,
        "fill_pressure": 5.0,
    }
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, sim_params=sim_params
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert f.attrs["voltage"] == sim_params["voltage"]
        assert f.attrs["capacitance"] == sim_params["capacitance"]
        assert f.attrs["inductance"] == sim_params["inductance"]
        assert f.attrs["fill_pressure"] == sim_params["fill_pressure"]


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_returns_output_path(tmp_path: Path) -> None:
    """finalize returns the output path as a Path object."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    result = exporter.finalize()

    assert isinstance(result, Path)
    assert result == output_path


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_with_n_trajectories_parameter(tmp_path: Path) -> None:
    """finalize with n_trajectories parameter sets correct array shapes."""
    output_path = tmp_path / "test.h5"
    n_traj = 3
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize(n_trajectories=n_traj)

    with h5py.File(output_path, "r") as f:
        assert f.attrs["n_trajectories"] == n_traj
        assert f["t0_fields/density"].shape[0] == n_traj
        assert f["t1_fields/magnetic_field"].shape[0] == n_traj


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_missing_fields_handled_with_zeros(tmp_path: Path) -> None:
    """Missing fields in some snapshots are filled with zeros."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    state1 = make_dpf_state()
    exporter.add_snapshot(state1, 1e-7)

    state2 = make_dpf_state()
    del state2["Te"]
    exporter.add_snapshot(state2, 2e-7)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        electron_temp = f["t0_fields/electron_temp"][0, :, 0, 0, 0]
        assert electron_temp[0] > 0
        assert electron_temp[1] == 0.0


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_circuit_scalars_only_created_when_nonzero(tmp_path: Path) -> None:
    """Circuit scalars only created when at least one non-zero value exists."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    state = make_dpf_state()
    circuit_scalars = {
        "current": 1.5e5,
        "voltage": 0.0,
        "neutron_rate": 0.0,
    }
    exporter.add_snapshot(state, 1e-7, circuit_scalars)
    exporter.add_snapshot(state, 2e-7, circuit_scalars)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        scalars_grp = f["scalars"]
        assert "current" in scalars_grp
        assert "voltage" not in scalars_grp
        assert "neutron_rate" not in scalars_grp


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_field_units_stored_as_attrs(tmp_path: Path) -> None:
    """Field units are stored as dataset attributes."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        density = f["t0_fields/density"]
        assert density.attrs["units"] == FIELD_UNITS["rho"]

        electron_temp = f["t0_fields/electron_temp"]
        assert electron_temp.attrs["units"] == FIELD_UNITS["Te"]

        magnetic_field = f["t1_fields/magnetic_field"]
        assert magnetic_field.attrs["units"] == FIELD_UNITS["B"]

        velocity = f["t1_fields/velocity"]
        assert velocity.attrs["units"] == FIELD_UNITS["velocity"]


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_multiple_snapshots_produce_time_series(tmp_path: Path) -> None:
    """Multiple snapshots produce correct time series in HDF5."""
    output_path = tmp_path / "test.h5"
    n_steps = 5
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    for i in range(n_steps):
        state = make_dpf_state()
        state["rho"][:] = float(i + 1)
        exporter.add_snapshot(state, i * 1e-7, {"current": float((i + 1) * 1e5)})

    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        density = f["t0_fields/density"][0, :, 0, 0, 0]
        assert len(density) == n_steps
        assert pytest.approx(density[0]) == 1.0
        assert pytest.approx(density[4]) == 5.0

        current = f["scalars/current"][0, :]
        assert len(current) == n_steps
        assert pytest.approx(current[0]) == 1e5
        assert pytest.approx(current[4]) == 5e5


def test_add_from_dpf_hdf5_raises_importerror_when_h5py_unavailable(tmp_path: Path) -> None:
    """add_from_dpf_hdf5 raises ImportError when h5py is unavailable."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)

    with (
        patch("dpf.ai.well_exporter.HAS_H5PY", False),
        pytest.raises(ImportError, match="h5py is required"),
    ):
        exporter.add_from_dpf_hdf5(tmp_path / "input.h5")


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_add_from_dpf_hdf5_raises_filenotfounderror(tmp_path: Path) -> None:
    """add_from_dpf_hdf5 raises FileNotFoundError for missing file."""
    exporter = WellExporter(output_path=tmp_path / "test.h5", grid_shape=(4, 4, 4), dx=0.001)
    with pytest.raises(FileNotFoundError, match="DPF HDF5 file not found"):
        exporter.add_from_dpf_hdf5(tmp_path / "nonexistent.h5")


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cylindrical_writes_r_theta(tmp_path: Path) -> None:
    """Cylindrical geometry writes r/theta instead of x/y in dimensions."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, geometry="cylindrical"
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        dims = f["dimensions"]
        assert "r" in dims
        assert "theta" in dims
        assert "z" in dims
        assert "x" not in dims
        assert "y" not in dims


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cylindrical_n_spatial_dims(tmp_path: Path) -> None:
    """Cylindrical geometry sets n_spatial_dims to 2."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path, grid_shape=(4, 4, 4), dx=0.001, geometry="cylindrical"
    )
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert f.attrs["n_spatial_dims"] == 2


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_no_scalars_group_when_no_circuit_data(tmp_path: Path) -> None:
    """finalize does not create /scalars group when no circuit data exists."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)
    exporter.add_snapshot(make_dpf_state(), 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert "scalars" not in f


# ---------------------------------------------------------------------------
# Section: Phase H — batch_runner
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Create base simulation config for testing."""
    return SimulationConfig(
        grid_shape=[8, 8, 8],
        dx=1e-2,
        sim_time=1e-6,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )


@pytest.fixture
def batch_parameter_ranges():
    """Create sample parameter ranges for batch runner testing."""
    return [
        ParameterRange(name="circuit.V0", low=500.0, high=2000.0, log_scale=False),
        ParameterRange(name="circuit.C", low=1e-7, high=1e-5, log_scale=True),
        ParameterRange(name="dx", low=0.005, high=0.02, log_scale=False),
    ]


class TestParameterRange:
    """Test ParameterRange dataclass."""

    def test_parameter_range_creation(self):
        """Test ParameterRange dataclass fields."""
        param = ParameterRange(name="test_param", low=1.0, high=10.0, log_scale=False)
        assert param.name == "test_param"
        assert param.low == 1.0
        assert param.high == 10.0
        assert param.log_scale is False

    def test_parameter_range_default_log_scale(self):
        """Test ParameterRange default log_scale is False."""
        param = ParameterRange(name="test", low=1.0, high=10.0)
        assert param.log_scale is False

    def test_parameter_range_with_log_scale(self):
        """Test ParameterRange with log_scale enabled."""
        param = ParameterRange(name="test", low=1e-3, high=1e3, log_scale=True)
        assert param.log_scale is True


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_batch_result_defaults(self):
        """Test BatchResult default values."""
        result = BatchResult()
        assert result.n_total == 0
        assert result.n_success == 0
        assert result.n_failed == 0
        assert result.output_dir == ""
        assert result.failed_configs == []

    def test_batch_result_creation(self):
        """Test BatchResult with explicit values."""
        result = BatchResult(
            n_total=100,
            n_success=95,
            n_failed=5,
            output_dir="/tmp/test",
            failed_configs=[(2, "error1"), (5, "error2")],
        )
        assert result.n_total == 100
        assert result.n_success == 95
        assert result.n_failed == 5
        assert result.output_dir == "/tmp/test"
        assert len(result.failed_configs) == 2

    def test_batch_result_n_failed_computation(self):
        """Test n_failed is computed correctly from counts."""
        result = BatchResult(n_total=100, n_success=92, n_failed=8)
        assert result.n_failed == 8
        assert result.n_total == result.n_success + result.n_failed

    def test_batch_result_failed_configs_list(self):
        """Test failed_configs stores index and error message tuples."""
        failed = [(0, "timeout"), (3, "convergence failure"), (7, "NaN detected")]
        result = BatchResult(failed_configs=failed)
        assert len(result.failed_configs) == 3
        assert result.failed_configs[0] == (0, "timeout")
        assert result.failed_configs[1] == (3, "convergence failure")
        assert result.failed_configs[2] == (7, "NaN detected")


class TestBatchRunnerInit:
    """Test BatchRunner initialization."""

    def test_batch_runner_init_stores_parameters(self, base_config, batch_parameter_ranges):
        """Test BatchRunner __init__ stores all parameters."""
        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=50,
            output_dir="test_output",
            workers=2,
            field_interval=5,
        )

        assert runner.base_config is base_config
        assert runner.parameter_ranges is batch_parameter_ranges
        assert runner.n_samples == 50
        assert runner.output_dir == Path("test_output")
        assert runner.workers == 2
        assert runner.field_interval == 5

    def test_batch_runner_init_default_values(self, base_config, batch_parameter_ranges):
        """Test BatchRunner default values."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=batch_parameter_ranges)

        assert runner.n_samples == 100
        assert runner.output_dir == Path("training_data")
        assert runner.workers == 4
        assert runner.field_interval == 10


class TestLatinHypercube:
    """Test Latin Hypercube sampling."""

    def test_latin_hypercube_returns_correct_shape(self):
        """Test _latin_hypercube returns array with correct shape."""
        samples = BatchRunner._latin_hypercube(n_samples=50, n_dims=3)
        assert samples.shape == (50, 3)

    def test_latin_hypercube_values_in_range(self):
        """Test _latin_hypercube values are in [0, 1]."""
        samples = BatchRunner._latin_hypercube(n_samples=100, n_dims=5)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_latin_hypercube_reproducible_with_seed(self):
        """Test _latin_hypercube with same seed produces identical results."""
        samples1 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=123)
        samples2 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=123)
        np.testing.assert_array_equal(samples1, samples2)

    def test_latin_hypercube_different_seeds_differ(self):
        """Test _latin_hypercube with different seeds produces different results."""
        samples1 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=1)
        samples2 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=2)
        assert not np.allclose(samples1, samples2)

    def test_latin_hypercube_fallback_without_scipy(self, monkeypatch):
        """Test _latin_hypercube fallback when scipy not available."""
        import builtins

        original_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if "scipy" in name:
                raise ImportError("No module named 'scipy'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        samples = BatchRunner._latin_hypercube(n_samples=30, n_dims=2, seed=42)
        assert samples.shape == (30, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


class TestGenerateSamples:
    """Test sample generation."""

    def test_generate_samples_returns_correct_count(self, base_config, batch_parameter_ranges):
        """Test generate_samples returns correct number of samples."""
        runner = BatchRunner(
            base_config=base_config, parameter_ranges=batch_parameter_ranges, n_samples=25
        )
        samples = runner.generate_samples()
        assert len(samples) == 25

    def test_generate_samples_linear_scaling_maps_to_range(self, base_config):
        """Test generate_samples linear scaling maps to correct range."""
        param_ranges = [ParameterRange(name="test_param", low=10.0, high=20.0, log_scale=False)]
        runner = BatchRunner(base_config=base_config, parameter_ranges=param_ranges, n_samples=100)
        samples = runner.generate_samples()

        values = [s["test_param"] for s in samples]
        assert all(10.0 <= v <= 20.0 for v in values)
        assert min(values) < 11.0
        assert max(values) > 19.0

    def test_generate_samples_log_scale_mapping(self, base_config):
        """Test generate_samples with log_scale enabled."""
        param_ranges = [ParameterRange(name="test_log", low=1e-6, high=1e-2, log_scale=True)]
        runner = BatchRunner(base_config=base_config, parameter_ranges=param_ranges, n_samples=100)
        samples = runner.generate_samples()

        values = [s["test_log"] for s in samples]
        assert all(1e-6 <= v <= 1e-2 for v in values)

        log_values = np.log10(values)
        assert min(log_values) < -5.5
        assert max(log_values) > -2.5

    def test_generate_samples_each_sample_has_correct_keys(self, base_config, batch_parameter_ranges):
        """Test each sample dictionary has correct parameter keys."""
        runner = BatchRunner(
            base_config=base_config, parameter_ranges=batch_parameter_ranges, n_samples=10
        )
        samples = runner.generate_samples()

        expected_keys = {"circuit.V0", "circuit.C", "dx"}
        for sample in samples:
            assert set(sample.keys()) == expected_keys


class TestBuildConfig:
    """Test configuration building."""

    def test_build_config_applies_top_level_parameter(self, base_config):
        """Test build_config applies top-level parameter overrides."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"dx": 0.015}

        config = runner.build_config(params)
        assert config.dx == pytest.approx(0.015)
        assert config.grid_shape == [8, 8, 8]

    def test_build_config_applies_nested_dot_notation(self, base_config):
        """Test build_config applies nested dot-notation overrides."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"circuit.V0": 1500.0, "circuit.C": 5e-6}

        config = runner.build_config(params)
        assert pytest.approx(1500.0) == config.circuit.V0
        assert pytest.approx(5e-6) == config.circuit.C
        assert pytest.approx(1e-7) == config.circuit.L0

    def test_build_config_returns_valid_simulation_config(self, base_config):
        """Test build_config returns valid SimulationConfig."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"circuit.V0": 1200.0, "dx": 0.01}

        config = runner.build_config(params)
        assert isinstance(config, SimulationConfig)
        assert config.grid_shape == [8, 8, 8]
        assert pytest.approx(0.01) == config.dx
        assert pytest.approx(1200.0) == config.circuit.V0


class TestRunSingle:
    """Test single simulation runs."""

    @staticmethod
    def _make_mock_engine():
        mock_engine = MagicMock()
        step_result = MagicMock()
        step_result.finished = True
        mock_engine.step.return_value = step_result
        mock_engine.get_field_snapshot.return_value = {
            "rho": np.zeros((8, 8, 8)),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.zeros((8, 8, 8)),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.zeros((8, 8, 8)),
            "Ti": np.zeros((8, 8, 8)),
        }
        mock_engine.circuit = MagicMock()
        mock_engine.circuit.current = 0.0
        mock_engine.circuit.voltage = 1000.0
        mock_engine.time = 1e-7
        mock_engine.diagnostics = MagicMock()
        return mock_engine

    def test_run_single_returns_success_tuple(self, base_config, batch_parameter_ranges, monkeypatch, tmp_path):
        """Test run_single returns (idx, None) on success."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=5,
            output_dir=tmp_path / "test_single",
        )
        params = {"circuit.V0": 1000.0}

        idx, error = runner.run_single(0, params)
        assert idx == 0
        assert error is None
        mock_engine.step.assert_called()
        mock_engine.get_field_snapshot.assert_called()
        mock_exporter.finalize.assert_called_once()

    def test_run_single_returns_error_on_failure(self, base_config, batch_parameter_ranges, monkeypatch):
        """Test run_single returns (idx, error_msg) on failure."""
        mock_engine = self._make_mock_engine()
        mock_engine.step.side_effect = RuntimeError("Simulation diverged")
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        runner = BatchRunner(
            base_config=base_config, parameter_ranges=batch_parameter_ranges, n_samples=5
        )
        params = {"circuit.V0": 1000.0}

        idx, error = runner.run_single(0, params)
        assert idx == 0
        assert error is not None
        assert "RuntimeError" in error
        assert "Simulation diverged" in error


class TestRun:
    """Test batch run execution."""

    @staticmethod
    def _make_mock_engine():
        mock_engine = MagicMock()
        step_result = MagicMock()
        step_result.finished = True
        mock_engine.step.return_value = step_result
        mock_engine.get_field_snapshot.return_value = {
            "rho": np.zeros((8, 8, 8)),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.zeros((8, 8, 8)),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.zeros((8, 8, 8)),
            "Ti": np.zeros((8, 8, 8)),
        }
        mock_engine.circuit = MagicMock()
        mock_engine.circuit.current = 0.0
        mock_engine.circuit.voltage = 1000.0
        mock_engine.time = 1e-7
        mock_engine.diagnostics = MagicMock()
        return mock_engine

    def test_run_with_workers_1_sequential(self, base_config, batch_parameter_ranges, monkeypatch, tmp_path):
        """Test run with workers=1 runs sequentially."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=3,
            workers=1,
            output_dir=tmp_path / "sequential",
        )
        result = runner.run()

        assert mock_engine_class.call_count == 3
        assert result.n_total == 3

    def test_run_returns_batch_result_with_correct_counts(self, base_config, batch_parameter_ranges, monkeypatch, tmp_path):
        """Test run returns BatchResult with correct counts."""
        call_count = [0]

        def make_engine(*args, **kwargs):
            call_count[0] += 1
            engine = self._make_mock_engine()
            if call_count[0] == 2:
                engine.step.side_effect = ValueError("Simulation 2 failed")
            return engine

        monkeypatch.setattr("dpf.engine.SimulationEngine", make_engine)

        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=5,
            workers=1,
            output_dir=tmp_path / "batch_test",
        )
        result = runner.run()

        assert result.n_total == 5
        assert result.n_success == 4
        assert result.n_failed == 1
        assert len(result.failed_configs) == 1

    def test_run_progress_callback_is_called(self, base_config, batch_parameter_ranges, monkeypatch, tmp_path):
        """Test run calls progress_callback after each simulation."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=3,
            workers=1,
            output_dir=tmp_path / "progress_test",
        )
        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        runner.run(progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    def test_run_creates_output_directory(self, base_config, batch_parameter_ranges, monkeypatch, tmp_path):
        """Test run creates output directory."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        output_dir = tmp_path / "test_output_dir"
        assert not output_dir.exists()

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=batch_parameter_ranges,
            n_samples=1,
            workers=1,
            output_dir=output_dir,
        )
        runner.run()

        assert output_dir.exists()
        assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# Section: Phase H — hybrid_validation
# ---------------------------------------------------------------------------


class TestHybridValidation:

    def setup_method(self):
        self.config = SimulationConfig(
            grid_shape=(10, 10, 10),
            dx=0.1,
            sim_time=1e-6,
            circuit={
                "type": "rlc",
                "L": 1e-9, "C": 1e-6, "R": 0.0, "V0": 1.0,
                "L0": 10e-9, "anode_radius": 0.05, "cathode_radius": 0.1,
            },
        )
        self.surrogate = MagicMock()
        self.surrogate.history_length = 2

        self.engine = HybridEngine(
            self.config,
            self.surrogate,
            handoff_fraction=0.1,
            validation_interval=1,
            max_l2_divergence=1e-2,
        )

    def test_surrogate_phase_runs(self):
        """Surrogate phase runs and returns predictions."""
        grid = (10, 10, 10)
        B = np.zeros((3, *grid))
        B[0] = 1.0
        B[1] = 2.0
        B[2] = 3.0

        state = {
            "B": B,
            "rho": np.ones(grid),
            "velocity": np.zeros((3, *grid)),
            "pressure": np.ones(grid),
        }
        self.surrogate.predict_next_step = MagicMock(return_value=state)

        history = [state, state]
        res = self.engine._run_surrogate_phase(history, n_steps=2)

        assert len(res) == 2

    def test_surrogate_nan_fallback(self):
        """Surrogate phase detects NaN and falls back."""
        grid = (10, 10, 10)
        nan_state = {
            "B": np.full((3, *grid), np.nan),
            "rho": np.full(grid, np.nan),
        }
        good_state = {
            "B": np.ones((3, *grid)),
            "rho": np.ones(grid),
        }
        self.surrogate.predict_next_step = MagicMock(return_value=nan_state)

        history = [good_state, good_state]
        res = self.engine._run_surrogate_phase(history, n_steps=5)

        assert len(res) == 1

    def test_dead_validate_step_removed(self):
        """_validate_step dead code has been removed from HybridEngine."""
        assert not hasattr(self.engine, "_validate_step")


# ---------------------------------------------------------------------------
# Section: Phase I — confidence
# ---------------------------------------------------------------------------


def _make_state_confidence(shape=(8, 8, 8), noise=0.0):
    """Create a synthetic DPF state dict with optional noise (confidence section)."""
    return {
        "rho": np.ones(shape) + np.random.normal(0, noise, shape),
        "velocity": np.zeros((3, *shape)) + np.random.normal(0, noise, (3, *shape)),
        "pressure": np.ones(shape) * 100.0 + np.random.normal(0, noise, shape),
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4 + np.random.normal(0, noise, shape),
        "Ti": np.ones(shape) * 1e4 + np.random.normal(0, noise, shape),
        "psi": np.zeros(shape),
    }


class MockModel:
    """Mock DPFSurrogate model for testing without PyTorch."""

    def __init__(self, noise=0.0):
        self.history_length = 4
        self.is_loaded = True
        self.noise = noise

    def predict_next_step(self, history):
        last = history[-1]
        return {k: v.copy() + np.random.normal(0, self.noise, v.shape) for k, v in last.items()}


@pytest.fixture
def ensemble():
    """Create EnsemblePredictor with mocked internals, bypassing __init__."""
    predictor = object.__new__(EnsemblePredictor)
    predictor.checkpoint_paths = []
    predictor.device = "cpu"
    predictor.history_length = 4
    predictor.confidence_threshold = 0.8
    predictor._models = [MockModel(noise=0.01), MockModel(noise=0.01), MockModel(noise=0.01)]
    return predictor


def test_prediction_with_confidence_defaults():
    """PredictionWithConfidence initializes with correct default values."""
    pred = PredictionWithConfidence()
    assert pred.mean_state == {}
    assert pred.std_state == {}
    assert pred.confidence == pytest.approx(1.0)
    assert pred.ood_score == pytest.approx(0.0)
    assert pred.n_models == 1


def test_prediction_with_confidence_custom_values():
    """PredictionWithConfidence stores custom values correctly."""
    mean_state = {"rho": np.ones((8, 8, 8))}
    std_state = {"rho": np.zeros((8, 8, 8))}
    pred = PredictionWithConfidence(
        mean_state=mean_state,
        std_state=std_state,
        confidence=0.95,
        ood_score=0.1,
        n_models=5,
    )
    assert pred.mean_state == mean_state
    assert pred.std_state == std_state
    assert pred.confidence == pytest.approx(0.95)
    assert pred.ood_score == pytest.approx(0.1)
    assert pred.n_models == 5


def test_ensemble_predictor_raises_importerror_without_torch():
    """EnsemblePredictor raises ImportError when HAS_TORCH is False."""
    with (
        patch("dpf.ai.confidence.HAS_TORCH", False),
        pytest.raises(ImportError, match="PyTorch required"),
    ):
        EnsemblePredictor(checkpoint_paths=["dummy.pt"])


def test_ensemble_predictor_n_models_property(ensemble):
    """n_models property returns correct count of ensemble members."""
    assert ensemble.n_models == 3


def test_predict_returns_prediction_with_confidence(ensemble):
    """predict returns PredictionWithConfidence instance."""
    history = [_make_state_confidence() for _ in range(4)]
    result = ensemble.predict(history)
    assert isinstance(result, PredictionWithConfidence)


def test_predict_mean_state_has_correct_keys(ensemble):
    """predict mean_state contains all expected DPF state keys."""
    history = [_make_state_confidence() for _ in range(4)]
    result = ensemble.predict(history)
    expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
    assert set(result.mean_state.keys()) == expected_keys


def test_predict_std_state_has_correct_keys(ensemble):
    """predict std_state contains all expected DPF state keys."""
    history = [_make_state_confidence() for _ in range(4)]
    result = ensemble.predict(history)
    expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
    assert set(result.std_state.keys()) == expected_keys


def test_predict_mean_state_values_are_mean_of_predictions(ensemble):
    """predict mean_state values are the mean of individual model predictions."""
    np.random.seed(42)

    predictor = object.__new__(EnsemblePredictor)
    predictor.checkpoint_paths = []
    predictor.device = "cpu"
    predictor.history_length = 4
    predictor.confidence_threshold = 0.8
    predictor._models = [MockModel(noise=0.0), MockModel(noise=0.0), MockModel(noise=0.0)]

    history = [_make_state_confidence(noise=0.0) for _ in range(4)]
    result = predictor.predict(history)

    last_state = history[-1]
    for key in last_state:
        np.testing.assert_allclose(result.mean_state[key], last_state[key], rtol=1e-6)


def test_predict_confidence_in_valid_range(ensemble):
    """predict confidence is in [0, 1] range."""
    history = [_make_state_confidence() for _ in range(4)]
    result = ensemble.predict(history)
    assert 0.0 <= result.confidence <= 1.0


def test_predict_n_models_matches_ensemble_size(ensemble):
    """predict result contains correct n_models count."""
    history = [_make_state_confidence() for _ in range(4)]
    result = ensemble.predict(history)
    assert result.n_models == 3


def test_is_confident_returns_true_when_above_threshold(ensemble):
    """is_confident returns True when confidence >= threshold."""
    pred = PredictionWithConfidence(confidence=0.85)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is True


def test_is_confident_returns_false_when_below_threshold(ensemble):
    """is_confident returns False when confidence < threshold."""
    pred = PredictionWithConfidence(confidence=0.75)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is False


def test_is_confident_returns_true_at_exact_threshold(ensemble):
    """is_confident returns True when confidence equals threshold."""
    pred = PredictionWithConfidence(confidence=0.8)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is True


def test_ood_detection_returns_zero_without_training_stats(ensemble):
    """ood_detection returns 0.0 when no training_stats provided."""
    state = _make_state_confidence()
    ood_score = ensemble.ood_detection(state, training_stats=None)
    assert ood_score == pytest.approx(0.0)


def test_ood_detection_returns_float_score_with_training_stats(ensemble):
    """ood_detection returns float score when training_stats provided."""
    state = _make_state_confidence()
    training_stats = {
        "rho": {"mean": 1.0, "std": 0.1},
        "Te": {"mean": 1e4, "std": 1e3},
        "Ti": {"mean": 1e4, "std": 1e3},
        "pressure": {"mean": 100.0, "std": 10.0},
        "psi": {"mean": 0.0, "std": 1.0},
    }
    ood_score = ensemble.ood_detection(state, training_stats=training_stats)
    assert isinstance(ood_score, float)
    assert ood_score >= 0.0


def test_ood_detection_score_increases_for_out_of_distribution_states(ensemble):
    """ood_detection score increases for states far from training distribution."""
    in_dist_state = _make_state_confidence(noise=0.0)
    training_stats = {
        "rho": {"mean": 1.0, "std": 0.1},
        "Te": {"mean": 1e4, "std": 1e3},
        "Ti": {"mean": 1e4, "std": 1e3},
        "pressure": {"mean": 100.0, "std": 10.0},
        "psi": {"mean": 0.0, "std": 1.0},
    }
    in_dist_score = ensemble.ood_detection(in_dist_state, training_stats=training_stats)

    ood_state = {
        "rho": np.ones((8, 8, 8)) * 10.0,
        "velocity": np.zeros((3, 8, 8, 8)),
        "pressure": np.ones((8, 8, 8)) * 1000.0,
        "B": np.zeros((3, 8, 8, 8)),
        "Te": np.ones((8, 8, 8)) * 1e5,
        "Ti": np.ones((8, 8, 8)) * 1e5,
        "psi": np.zeros((8, 8, 8)),
    }
    ood_score = ensemble.ood_detection(ood_state, training_stats=training_stats)

    assert ood_score > in_dist_score
    assert ood_score > 1.0


def test_compute_confidence_returns_one_for_zero_std():
    """_compute_confidence returns 1.0 for zero std (perfect agreement)."""
    predictor = object.__new__(EnsemblePredictor)
    std_state = {
        "rho": np.zeros((8, 8, 8)),
        "Te": np.zeros((8, 8, 8)),
        "Ti": np.zeros((8, 8, 8)),
        "pressure": np.zeros((8, 8, 8)),
    }
    confidence = predictor._compute_confidence(std_state)
    assert confidence == pytest.approx(1.0)


def test_compute_confidence_decreases_with_higher_std():
    """_compute_confidence returns lower values for higher std (disagreement)."""
    predictor = object.__new__(EnsemblePredictor)

    low_std_state = {
        "rho": np.ones((8, 8, 8)) * 0.1,
        "Te": np.ones((8, 8, 8)) * 0.1,
    }
    low_confidence = predictor._compute_confidence(low_std_state)

    high_std_state = {
        "rho": np.ones((8, 8, 8)) * 10.0,
        "Te": np.ones((8, 8, 8)) * 10.0,
    }
    high_confidence = predictor._compute_confidence(high_std_state)

    assert high_confidence < low_confidence
    assert 0.0 <= high_confidence <= 1.0
    assert 0.0 <= low_confidence <= 1.0


def test_compute_confidence_returns_zero_for_empty_std_state():
    """_compute_confidence returns 0.0 for empty std_state dict."""
    predictor = object.__new__(EnsemblePredictor)
    std_state = {}
    confidence = predictor._compute_confidence(std_state)
    assert confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section: Phase I — hybrid_engine
# ---------------------------------------------------------------------------


class MockSurrogateHybrid:
    """Mock WALRUS surrogate for HybridEngine testing."""

    def __init__(self, history_length: int = 4):
        self.history_length = history_length
        self.is_loaded = True

    def predict_next_step(self, history: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        last = history[-1]
        return {k: v.copy() + np.random.normal(0, 1e-10, v.shape) for k, v in last.items()}


class MockEngineHybrid:
    """Mock SimulationEngine for HybridEngine testing."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._step_count = 0

    def step(self) -> MagicMock:
        self._step_count += 1
        return MagicMock(finished=False)

    def get_field_snapshot(self) -> dict[str, np.ndarray]:
        shape = tuple(self.config.grid_shape)
        return _make_state_hybrid(shape)


def _make_state_hybrid(shape: tuple[int, int, int] = (8, 8, 8)) -> dict[str, np.ndarray]:
    """Create a valid DPF state dict (hybrid_engine section)."""
    return {
        "rho": np.ones(shape),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.ones(shape) * 100.0,
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4,
        "Ti": np.ones(shape) * 1e4,
        "psi": np.zeros(shape),
    }


def _make_config_hybrid(grid_shape: tuple[int, int, int] = (8, 8, 8)) -> SimulationConfig:
    """Create a valid SimulationConfig (hybrid_engine section)."""
    return SimulationConfig(
        grid_shape=list(grid_shape),
        dx=1e-2,
        sim_time=1e-6,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )


def test_init_raises_for_handoff_fraction_too_low():
    """HybridEngine.__init__ raises ValueError for handoff_fraction < 0."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()

    with pytest.raises(ValueError, match="handoff_fraction must be in"):
        HybridEngine(config, surrogate, handoff_fraction=-0.1)


def test_init_raises_for_handoff_fraction_too_high():
    """HybridEngine.__init__ raises ValueError for handoff_fraction > 1."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()

    with pytest.raises(ValueError, match="handoff_fraction must be in"):
        HybridEngine(config, surrogate, handoff_fraction=1.1)


def test_init_stores_parameters_correctly():
    """HybridEngine.__init__ stores all parameters correctly."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()

    engine = HybridEngine(
        config,
        surrogate,
        handoff_fraction=0.3,
        validation_interval=25,
        max_l2_divergence=0.05,
    )

    assert engine.config is config
    assert engine.surrogate is surrogate
    assert engine.handoff_fraction == 0.3
    assert engine.validation_interval == 25
    assert engine.max_l2_divergence == 0.05
    assert engine._trajectory == []


def test_init_with_valid_parameters_succeeds():
    """HybridEngine.__init__ succeeds with handoff_fraction in [0, 1]."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()

    engine_zero = HybridEngine(config, surrogate, handoff_fraction=0.0)
    assert engine_zero.handoff_fraction == 0.0

    engine_one = HybridEngine(config, surrogate, handoff_fraction=1.0)
    assert engine_one.handoff_fraction == 1.0

    engine_mid = HybridEngine(config, surrogate, handoff_fraction=0.5)
    assert engine_mid.handoff_fraction == 0.5


def test_run_returns_summary_dict_with_expected_keys():
    """HybridEngine.run returns summary dict with all expected keys."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.2)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=10)

    expected_keys = {
        "total_steps",
        "physics_steps",
        "surrogate_steps",
        "wall_time_s",
        "fallback_to_physics",
    }
    assert set(summary.keys()) == expected_keys
    assert isinstance(summary["wall_time_s"], float)
    assert summary["wall_time_s"] > 0


def test_run_summary_has_correct_total_steps():
    """HybridEngine.run summary total_steps matches trajectory length."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.2)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=10)

    assert summary["total_steps"] == len(engine.trajectory)
    assert summary["total_steps"] == 10


def test_run_physics_steps_matches_handoff_fraction():
    """HybridEngine.run physics_steps matches handoff_fraction * max_steps."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.3)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=20)

    assert summary["physics_steps"] == 6


def test_run_surrogate_steps_is_remainder():
    """HybridEngine.run surrogate_steps is the remainder after physics."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.25)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=20)

    assert summary["physics_steps"] == 5
    assert summary["surrogate_steps"] == 15


def test_trajectory_property_returns_accumulated_states():
    """HybridEngine.trajectory returns all accumulated states."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.5)

    assert engine.trajectory == []

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        engine.run(max_steps=10)

    trajectory = engine.trajectory
    assert len(trajectory) == 10
    assert all(isinstance(state, dict) for state in trajectory)
    assert all("rho" in state for state in trajectory)


def test_run_physics_phase_returns_list_of_states():
    """HybridEngine._run_physics_phase returns list of field snapshots."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate)

    mock_engine = MockEngineHybrid(config)
    history = engine._run_physics_phase(mock_engine, n_steps=5)

    assert isinstance(history, list)
    assert len(history) == 5
    assert all(isinstance(state, dict) for state in history)
    assert all("rho" in state and "B" in state for state in history)


def test_run_physics_phase_state_count_matches_n_steps():
    """HybridEngine._run_physics_phase state count matches n_steps."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate)

    mock_engine = MockEngineHybrid(config)
    for n in [1, 5, 10, 20]:
        history = engine._run_physics_phase(mock_engine, n_steps=n)
        assert len(history) == n


def test_run_surrogate_phase_returns_predicted_states():
    """HybridEngine._run_surrogate_phase returns list of predicted states."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid(history_length=4)
    engine = HybridEngine(config, surrogate, validation_interval=100)

    initial_history = [_make_state_hybrid() for _ in range(10)]

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        surrogate_history = engine._run_surrogate_phase(initial_history, n_steps=5)

    assert isinstance(surrogate_history, list)
    assert len(surrogate_history) == 5
    assert all(isinstance(state, dict) for state in surrogate_history)
    assert all("rho" in state for state in surrogate_history)


def test_validate_step_removed_as_dead_code():
    """HybridEngine._validate_step was removed (dead code, MOD-4 fix)."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate)

    assert not hasattr(engine, "_validate_step")


def test_surrogate_fallback_when_divergence_exceeds_threshold():
    """HybridEngine falls back to physics when divergence exceeds threshold."""
    config = _make_config_hybrid()

    class DivergentSurrogate(MockSurrogateHybrid):
        def predict_next_step(self, history):
            last = history[-1]
            return {k: v.copy() * 100.0 for k, v in last.items()}

    surrogate = DivergentSurrogate()
    engine = HybridEngine(
        config, surrogate, handoff_fraction=0.2, validation_interval=2, max_l2_divergence=0.1
    )

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=10)

    assert summary["physics_steps"] == 2
    assert summary["surrogate_steps"] <= 8


def test_run_uses_default_when_max_steps_none():
    """HybridEngine.run uses default max_steps=1000 when max_steps=None."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=0.2)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=None)

    assert summary["physics_steps"] == 200
    assert summary["total_steps"] == 1000


def test_run_with_zero_surrogate_steps():
    """HybridEngine.run with handoff_fraction=1.0 runs only physics."""
    config = _make_config_hybrid()
    surrogate = MockSurrogateHybrid()
    engine = HybridEngine(config, surrogate, handoff_fraction=1.0)

    with patch("dpf.engine.SimulationEngine", MockEngineHybrid):
        summary = engine.run(max_steps=10)

    assert summary["physics_steps"] == 10
    assert summary["surrogate_steps"] == 0
    assert summary["total_steps"] == 10
    assert summary["fallback_to_physics"] is False


# ---------------------------------------------------------------------------
# Section: Phase I — instability
# ---------------------------------------------------------------------------


class MockSurrogateInstability:
    """Mock surrogate for testing InstabilityDetector."""

    def __init__(self, noise_level: float = 0.0):
        self.history_length = 4
        self.noise_level = noise_level

    def predict_next_step(self, history: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        last = history[-1]
        return {k: v.copy() + np.random.normal(0, self.noise_level, v.shape) for k, v in last.items()}


def _make_state_instability(shape: tuple[int, ...] = (8, 8, 8), value: float = 1.0) -> dict[str, np.ndarray]:
    """Create standard physics state dict for instability testing."""
    return {
        "rho": np.ones(shape) * value,
        "velocity": np.zeros((3, *shape)),
        "pressure": np.ones(shape) * 100.0,
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4,
        "Ti": np.ones(shape) * 1e4,
        "psi": np.zeros(shape),
    }


class TestInstabilityEvent:
    """Tests for InstabilityEvent dataclass."""

    def test_event_stores_all_fields(self):
        """InstabilityEvent stores step, time, divergence, field divergences, and severity."""
        field_divergences = {"rho": 0.1, "pressure": 0.2}
        event = InstabilityEvent(
            step=42,
            time=1.5e-6,
            l2_divergence=0.15,
            field_divergences=field_divergences,
            severity="medium",
        )

        assert event.step == 42
        assert event.time == pytest.approx(1.5e-6)
        assert event.l2_divergence == pytest.approx(0.15)
        assert event.field_divergences == field_divergences
        assert event.severity == "medium"


class TestInstabilityDetectorInit:
    """Tests for InstabilityDetector initialization."""

    def test_init_stores_threshold_values(self):
        """InstabilityDetector __init__ stores surrogate and threshold values."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(
            surrogate, threshold_low=0.1, threshold_medium=0.2, threshold_high=0.4
        )

        assert detector.surrogate is surrogate
        assert detector.threshold_low == pytest.approx(0.1)
        assert detector.threshold_medium == pytest.approx(0.2)
        assert detector.threshold_high == pytest.approx(0.4)

    def test_init_default_thresholds(self):
        """InstabilityDetector uses default thresholds when not specified."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        assert detector.threshold_low == pytest.approx(0.05)
        assert detector.threshold_medium == pytest.approx(0.15)
        assert detector.threshold_high == pytest.approx(0.3)


class TestClassifySeverity:
    """Tests for severity classification."""

    def test_classify_severity_high(self):
        """classify_severity returns 'high' when divergence >= threshold_high."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate, threshold_high=0.3)

        assert detector.classify_severity(0.3) == "high"
        assert detector.classify_severity(0.5) == "high"
        assert detector.classify_severity(1.0) == "high"

    def test_classify_severity_medium(self):
        """classify_severity returns 'medium' when threshold_medium <= divergence < threshold_high."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate, threshold_medium=0.15, threshold_high=0.3)

        assert detector.classify_severity(0.15) == "medium"
        assert detector.classify_severity(0.2) == "medium"
        assert detector.classify_severity(0.29) == "medium"

    def test_classify_severity_low(self):
        """classify_severity returns 'low' when threshold_low <= divergence < threshold_medium."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(
            surrogate, threshold_low=0.05, threshold_medium=0.15, threshold_high=0.3
        )

        assert detector.classify_severity(0.05) == "low"
        assert detector.classify_severity(0.1) == "low"
        assert detector.classify_severity(0.14) == "low"


class TestComputeDivergence:
    """Tests for divergence computation."""

    def test_compute_divergence_identical_states(self):
        """compute_divergence returns 0.0 for identical states."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        state = _make_state_instability()
        overall, field_divs = detector.compute_divergence(state, state)

        assert overall == pytest.approx(0.0, abs=1e-10)
        for field_div in field_divs.values():
            assert field_div == pytest.approx(0.0, abs=1e-10)

    def test_compute_divergence_different_states(self):
        """compute_divergence returns positive value for different states."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        state1 = _make_state_instability(value=1.0)
        state2 = _make_state_instability(value=1.5)

        overall, field_divs = detector.compute_divergence(state1, state2)

        assert overall > 0.0
        assert field_divs["rho"] > 0.0
        assert field_divs["pressure"] == pytest.approx(0.0, abs=1e-10)
        assert field_divs["Te"] == pytest.approx(0.0, abs=1e-10)
        assert field_divs["Ti"] == pytest.approx(0.0, abs=1e-10)

    def test_compute_divergence_missing_fields(self):
        """compute_divergence handles missing fields via empty intersection."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        predicted = {"rho": np.ones((8, 8, 8))}
        actual = {"pressure": np.ones((8, 8, 8)) * 100.0}

        overall, field_divs = detector.compute_divergence(predicted, actual)

        assert overall == pytest.approx(0.0)
        assert len(field_divs) == 0

    def test_compute_divergence_shape_mismatch(self, caplog):
        """compute_divergence logs warning and skips field for shape mismatch."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        predicted = {"rho": np.ones((8, 8, 8)), "pressure": np.ones((8, 8, 8)) * 100.0}
        actual = {"rho": np.ones((4, 4, 4)), "pressure": np.ones((8, 8, 8)) * 100.0}

        overall, field_divs = detector.compute_divergence(predicted, actual)

        assert "rho" not in field_divs
        assert "pressure" in field_divs
        assert "Shape mismatch for field rho" in caplog.text


class TestCheck:
    """Tests for single-step instability checking."""

    def test_check_returns_none_below_threshold(self):
        """check returns None when divergence < threshold_low with zero-noise mock."""
        surrogate = MockSurrogateInstability(noise_level=0.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state_instability()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is None

    def test_check_returns_event_above_threshold(self):
        """check returns InstabilityEvent when divergence > threshold_low with high-noise mock."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state_instability()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is not None
        assert isinstance(event, InstabilityEvent)

    def test_check_event_correct_severity(self):
        """check event has correct severity classification."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=50.0)
        detector = InstabilityDetector(
            surrogate, threshold_low=0.05, threshold_medium=0.15, threshold_high=0.3
        )

        state = _make_state_instability()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is not None
        if event.l2_divergence >= 0.3:
            assert event.severity == "high"
        elif event.l2_divergence >= 0.15:
            assert event.severity == "medium"
        else:
            assert event.severity == "low"

    def test_check_event_correct_step_and_time(self):
        """check event has correct step and time values."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state_instability()
        history = [state] * 4

        event = detector.check(history, state, step=42, time=2.5e-6)

        assert event is not None
        assert event.step == 42
        assert event.time == pytest.approx(2.5e-6)


class TestMonitorTrajectory:
    """Tests for trajectory monitoring."""

    def test_monitor_trajectory_empty_for_short_trajectory(self):
        """monitor_trajectory returns empty list for trajectory too short."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        trajectory = [_make_state_instability()] * 3

        events = detector.monitor_trajectory(trajectory)

        assert events == []

    def test_monitor_trajectory_detects_increasing_divergence(self):
        """monitor_trajectory detects instabilities in trajectory with increasing divergence."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=0.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.01)

        trajectory = []
        for i in range(10):
            factor = 1.0 + i * 0.5
            state = {
                "rho": np.ones((8, 8, 8)) * factor,
                "velocity": np.ones((3, 8, 8, 8)) * factor * 0.1,
                "pressure": np.ones((8, 8, 8)) * 100.0 * factor,
                "B": np.ones((3, 8, 8, 8)) * factor * 0.01,
                "Te": np.ones((8, 8, 8)) * 1e4 * factor,
                "Ti": np.ones((8, 8, 8)) * 1e4 * factor,
                "psi": np.ones((8, 8, 8)) * factor * 0.001,
            }
            trajectory.append(state)

        events = detector.monitor_trajectory(trajectory)

        assert len(events) > 0
        for i in range(len(events) - 1):
            assert events[i].step < events[i + 1].step

    def test_monitor_trajectory_no_events_for_stable_trajectory(self):
        """monitor_trajectory returns empty list for stable trajectory."""
        surrogate = MockSurrogateInstability(noise_level=0.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        trajectory = [_make_state_instability(value=1.0)] * 10

        events = detector.monitor_trajectory(trajectory)

        assert events == []

    def test_monitor_trajectory_uses_explicit_times(self):
        """monitor_trajectory uses explicit times array for event timestamps (M9 fix)."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.01)

        trajectory = [_make_state_instability()] * 10
        times = [i * 1e-7 + i**2 * 1e-9 for i in range(10)]

        events = detector.monitor_trajectory(trajectory, times=times)

        assert len(events) > 0
        for event in events:
            assert event.time == pytest.approx(times[event.step])

    def test_monitor_trajectory_uses_state_time_key(self):
        """monitor_trajectory reads 'time' key from state dicts when present (M9 fix)."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.01)

        trajectory = []
        for i in range(10):
            state = _make_state_instability()
            state["time"] = np.float64(i * 2.5e-7)
            trajectory.append(state)

        events = detector.monitor_trajectory(trajectory)

        assert len(events) > 0
        for event in events:
            expected_time = event.step * 2.5e-7
            assert event.time == pytest.approx(expected_time)

    def test_monitor_trajectory_explicit_times_overrides_state_time(self):
        """Explicit times parameter takes priority over per-state 'time' key."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.01)

        trajectory = []
        for _i in range(10):
            state = _make_state_instability()
            state["time"] = np.float64(999.0)
            trajectory.append(state)

        real_times = [i * 1e-8 for i in range(10)]
        events = detector.monitor_trajectory(trajectory, times=real_times)

        assert len(events) > 0
        for event in events:
            assert event.time == pytest.approx(real_times[event.step])
            assert event.time != pytest.approx(999.0)

    def test_monitor_trajectory_times_length_mismatch_raises(self):
        """monitor_trajectory raises ValueError if times length != trajectory length."""
        surrogate = MockSurrogateInstability()
        detector = InstabilityDetector(surrogate)

        trajectory = [_make_state_instability()] * 10
        wrong_times = [0.0, 1.0, 2.0]

        with pytest.raises(ValueError, match="times length"):
            detector.monitor_trajectory(trajectory, times=wrong_times)

    def test_monitor_trajectory_dt_fallback(self):
        """monitor_trajectory falls back to i*dt when no times or state time key."""
        np.random.seed(42)
        surrogate = MockSurrogateInstability(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.01)

        trajectory = [_make_state_instability()] * 10
        dt = 5e-8

        events = detector.monitor_trajectory(trajectory, dt=dt)

        assert len(events) > 0
        for event in events:
            expected_time = event.step * dt
            assert event.time == pytest.approx(expected_time)


# ---------------------------------------------------------------------------
# Section: Phase I — surrogate
# ---------------------------------------------------------------------------


def _make_state_surrogate(shape=(8, 8, 8)):
    """Create a fake DPF state dict for surrogate testing."""
    return {
        "rho": np.ones(shape),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.ones(shape) * 100.0,
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4,
        "Ti": np.ones(shape) * 1e4,
        "psi": np.zeros(shape),
    }


@pytest.fixture
def mock_torch(monkeypatch, tmp_path):
    """Mock torch module and create fake checkpoint file."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"fake")

    mock_torch_mod = MagicMock()
    mock_torch_mod.load.return_value = {"state_dict": {}, "metadata": "test"}
    mock_torch_mod.from_numpy.return_value = MagicMock()

    mock_tensor = MagicMock()
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.float.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor
    mock_torch_mod.from_numpy.return_value = mock_tensor

    monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
    monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
    monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
    monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

    def mock_scalar_to_well(field, *args, **kwargs):
        return field.astype(np.float32)

    def mock_vector_to_well(field, *args, **kwargs):
        return field.astype(np.float32)

    def mock_scalar_from_well(field, *args, **kwargs):
        return field.astype(np.float64)

    def mock_vector_from_well(field, *args, **kwargs):
        return field.astype(np.float64)

    monkeypatch.setattr("dpf.ai.surrogate.dpf_scalar_to_well", mock_scalar_to_well)
    monkeypatch.setattr("dpf.ai.surrogate.dpf_vector_to_well", mock_vector_to_well)
    monkeypatch.setattr("dpf.ai.surrogate.well_scalar_to_dpf", mock_scalar_from_well)
    monkeypatch.setattr("dpf.ai.surrogate.well_vector_to_dpf", mock_vector_from_well)

    return ckpt


class TestDPFSurrogateInitialization:
    """Test DPFSurrogate initialization and loading."""

    def test_init_raises_import_error_when_torch_not_available(self, monkeypatch, tmp_path):
        """__init__ raises ImportError when HAS_TORCH is False."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", False)

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        with pytest.raises(ImportError, match="PyTorch is required"):
            DPFSurrogate(ckpt)

    def test_init_falls_back_when_checkpoint_missing(self, mock_torch):
        """__init__ warns and falls back to placeholder when checkpoint missing."""
        import warnings  # noqa: E402, I001

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        nonexistent = Path("/nonexistent/path/model.pt")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            surrogate = DPFSurrogate(nonexistent)
            assert surrogate is not None

    def test_init_with_valid_checkpoint_sets_attributes(self, mock_torch):
        """__init__ with valid checkpoint sets attributes correctly."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, device="cpu", history_length=5)

        assert surrogate.checkpoint_path == mock_torch
        assert surrogate.device == "cpu"
        assert surrogate.history_length == 5

    def test_init_loads_model(self, mock_torch, monkeypatch):
        """__init__ calls _load_model and sets _model."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        mock_torch_mod = sys.modules["torch"]

        surrogate = DPFSurrogate(mock_torch)

        mock_torch_mod.load.assert_called_once()
        assert surrogate._model is not None


class TestDPFSurrogateLoading:
    """Test model loading behavior."""

    def test_is_loaded_returns_true_when_model_loaded(self, mock_torch):
        """is_loaded property returns True when model loaded."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        assert surrogate.is_loaded is True

    def test_is_loaded_returns_false_when_model_is_none(self, mock_torch):
        """is_loaded returns False when _model is None."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)
        surrogate._model = None

        assert surrogate.is_loaded is False

    def test_load_model_sets_placeholder_dict(self, mock_torch):
        """_load_model sets _model to a placeholder dict when walrus is not installed."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, device="mps")

        assert isinstance(surrogate._model, dict)
        assert surrogate._model.get("placeholder") is True or "data" in surrogate._model

    def test_load_model_handles_torch_load_failure(self, mock_torch, monkeypatch):
        """_load_model handles torch.load failure gracefully with placeholder fallback."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        mock_torch_mod = sys.modules["torch"]
        mock_torch_mod.load.side_effect = RuntimeError("Corrupted checkpoint")

        surrogate = DPFSurrogate(mock_torch)

        assert surrogate.is_loaded is True
        assert surrogate._is_walrus_model is False


class TestDPFSurrogatePrediction:
    """Test single-step prediction."""

    def test_predict_next_step_raises_runtime_error_when_not_loaded(self, mock_torch):
        """predict_next_step raises RuntimeError when not loaded."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)
        surrogate._model = None

        history = [_make_state_surrogate() for _ in range(4)]

        with pytest.raises(RuntimeError, match="Model not loaded"):
            surrogate.predict_next_step(history)

    def test_predict_next_step_raises_value_error_when_history_too_short(self, mock_torch):
        """predict_next_step raises ValueError when history too short."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        history = [_make_state_surrogate() for _ in range(2)]

        with pytest.raises(ValueError, match="Insufficient history"):
            surrogate.predict_next_step(history)

    def test_predict_next_step_returns_state_dict_with_same_keys(self, mock_torch):
        """predict_next_step returns state dict with same keys as input."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        history = [_make_state_surrogate() for _ in range(4)]
        predicted = surrogate.predict_next_step(history)

        assert set(predicted.keys()) == set(history[0].keys())

    def test_predict_next_step_returns_correct_shapes_for_scalar_fields(self, mock_torch):
        """predict_next_step returns correct shapes for scalar fields."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        history = [_make_state_surrogate(shape=(8, 8, 8)) for _ in range(4)]
        predicted = surrogate.predict_next_step(history)

        for fkey in ["rho", "pressure", "Te", "Ti", "psi"]:
            assert predicted[fkey].shape == (8, 8, 8)

    def test_predict_next_step_returns_correct_shapes_for_vector_fields(self, mock_torch):
        """predict_next_step returns correct shapes for vector fields."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        history = [_make_state_surrogate(shape=(8, 8, 8)) for _ in range(4)]
        predicted = surrogate.predict_next_step(history)

        for fkey in ["velocity", "B"]:
            assert predicted[fkey].shape == (3, 8, 8, 8)

    def test_predict_next_step_uses_last_history_length_states(self, mock_torch):
        """predict_next_step uses only last history_length states."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=3)

        history = [_make_state_surrogate() for _ in range(5)]
        history[0]["rho"][:] = 999.0

        predicted = surrogate.predict_next_step(history)

        assert predicted is not None


class TestDPFSurrogateRollout:
    """Test autoregressive rollout."""

    def test_rollout_raises_value_error_when_initial_states_too_short(self, mock_torch):
        """rollout raises ValueError when initial_states too short."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        initial_states = [_make_state_surrogate() for _ in range(2)]

        with pytest.raises(ValueError, match="Need at least"):
            surrogate.rollout(initial_states, n_steps=10)

    def test_rollout_returns_list_of_n_steps_states(self, mock_torch):
        """rollout returns list of n_steps states."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        initial_states = [_make_state_surrogate() for _ in range(4)]
        predictions = surrogate.rollout(initial_states, n_steps=5)

        assert len(predictions) == 5

    def test_rollout_each_state_has_correct_structure(self, mock_torch):
        """rollout each state has correct structure."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=4)

        initial_states = [_make_state_surrogate() for _ in range(4)]
        predictions = surrogate.rollout(initial_states, n_steps=3)

        for state in predictions:
            assert set(state.keys()) == {
                "rho", "velocity", "pressure", "B", "Te", "Ti", "psi",
            }

    def test_rollout_autoregressive_states_accumulate(self, mock_torch):
        """rollout autoregressive (states accumulate in trajectory)."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=2)

        initial_states = [_make_state_surrogate() for _ in range(2)]
        predictions = surrogate.rollout(initial_states, n_steps=3)

        assert len(predictions) == 3
        for pred in predictions:
            assert pred is not None


class TestDPFSurrogateParameterSweep:
    """Test parameter sweep functionality."""

    def test_parameter_sweep_returns_list_of_result_dicts(self, mock_torch):
        """parameter_sweep returns list of result dicts."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=2)

        configs = [{"rho0": 1e-6, "pressure0": 100.0}, {"rho0": 2e-6, "pressure0": 200.0}]

        results = surrogate.parameter_sweep(configs, n_steps=5)

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_parameter_sweep_handles_error_in_single_config(self, mock_torch, monkeypatch):
        """parameter_sweep handles error in single config."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=2)

        original_predict = surrogate.predict_next_step

        def failing_predict(history):
            if len(history) > 2:
                raise ValueError("Prediction failed")
            return original_predict(history)

        monkeypatch.setattr(surrogate, "predict_next_step", failing_predict)

        configs = [{"rho0": 1e-6}]
        results = surrogate.parameter_sweep(configs, n_steps=5)

        assert len(results) == 1
        assert "error" in results[0]

    def test_parameter_sweep_includes_summary_metrics(self, mock_torch):
        """parameter_sweep includes expected summary metrics."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=2)

        configs = [{"rho0": 1e-6, "Te0": 5.0}]
        results = surrogate.parameter_sweep(configs, n_steps=3)

        assert len(results) == 1
        result = results[0]

        assert "config" in result
        assert "metrics" in result

        metrics = result["metrics"]
        assert "max_rho" in metrics
        assert "max_Te" in metrics
        assert "max_Ti" in metrics
        assert "max_pressure" in metrics
        assert "mean_B" in metrics
        assert "max_B" in metrics
        assert "final_rho" in metrics
        assert "final_pressure" in metrics
        assert "n_steps" in metrics

        assert result["config"]["rho0"] == 1e-6
        assert result["config"]["Te0"] == 5.0


class TestDPFSurrogateHelpers:
    """Test helper methods."""

    def test_create_initial_state_creates_correct_field_shapes(self, mock_torch):
        """_create_initial_state creates correct field shapes."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        config = {"rho0": 1e-6}
        shape = (10, 10, 10)

        state = surrogate._create_initial_state(config, shape)

        assert state["rho"].shape == (10, 10, 10)
        assert state["velocity"].shape == (3, 10, 10, 10)
        assert state["pressure"].shape == (10, 10, 10)
        assert state["B"].shape == (3, 10, 10, 10)
        assert state["Te"].shape == (10, 10, 10)
        assert state["Ti"].shape == (10, 10, 10)
        assert state["psi"].shape == (10, 10, 10)

    def test_create_initial_state_uses_config_parameters(self, mock_torch):
        """_create_initial_state uses config parameters (rho0, Te0, etc.)."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        config = {"rho0": 2e-6, "pressure0": 250.0, "Te0": 5.0, "Ti0": 3.0}
        shape = (4, 4, 4)

        state = surrogate._create_initial_state(config, shape)

        assert np.allclose(state["rho"], 2e-6)
        assert np.allclose(state["pressure"], 250.0)
        assert np.allclose(state["Te"], 5.0)
        assert np.allclose(state["Ti"], 3.0)

    def test_create_initial_state_uses_defaults_when_params_missing(self, mock_torch):
        """_create_initial_state uses defaults when params missing."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        config = {}
        shape = (4, 4, 4)

        state = surrogate._create_initial_state(config, shape)

        assert np.allclose(state["rho"], 1e-6)
        assert np.allclose(state["pressure"], 100.0)
        assert np.allclose(state["Te"], 1.0)
        assert np.allclose(state["Ti"], 1.0)

    def test_extract_summary_returns_expected_keys(self, mock_torch):
        """_extract_summary returns expected keys."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        trajectory = [_make_state_surrogate() for _ in range(5)]
        config = {"V0": 10e3}

        summary = surrogate._extract_summary(trajectory, config)

        assert set(summary.keys()) == {"config", "metrics"}

        expected_metric_keys = {
            "max_rho", "max_Te", "max_Ti", "max_pressure",
            "mean_B", "max_B", "final_rho", "final_pressure", "n_steps",
        }
        assert set(summary["metrics"].keys()) == expected_metric_keys
        assert summary["config"] == {"V0": 10e3}

    def test_extract_summary_max_values_correct(self, mock_torch):
        """_extract_summary max values are correct."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        trajectory = []
        for i in range(3):
            state = _make_state_surrogate()
            state["rho"][:] = float(i + 1)
            state["Te"][:] = float(i + 2) * 1e4
            trajectory.append(state)

        config = {}
        summary = surrogate._extract_summary(trajectory, config)

        assert summary["metrics"]["max_rho"] == pytest.approx(3.0)
        assert summary["metrics"]["max_Te"] == pytest.approx(4.0 * 1e4)

    def test_extract_summary_final_values_correct(self, mock_torch):
        """_extract_summary final values are correct."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        trajectory = [_make_state_surrogate() for _ in range(3)]
        trajectory[-1]["rho"][:] = 5.0
        trajectory[-1]["pressure"][:] = 300.0

        config = {}
        summary = surrogate._extract_summary(trajectory, config)

        assert summary["metrics"]["final_rho"] == pytest.approx(5.0)
        assert summary["metrics"]["final_pressure"] == pytest.approx(300.0)

    def test_extract_summary_includes_config_parameters(self, mock_torch):
        """_extract_summary includes original config parameters."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        trajectory = [_make_state_surrogate() for _ in range(2)]
        config = {"V0": 15e3, "pressure0": 200.0, "custom_param": "test"}

        summary = surrogate._extract_summary(trajectory, config)

        assert summary["config"]["V0"] == 15e3
        assert summary["config"]["pressure0"] == 200.0
        assert summary["config"]["custom_param"] == "test"


class TestDPFSurrogateAttributes:
    """Test attribute storage and access."""

    def test_history_length_attribute_stored_correctly(self, mock_torch):
        """history_length attribute stored correctly."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, history_length=8)

        assert surrogate.history_length == 8

    def test_device_attribute_stored_correctly(self, mock_torch):
        """device attribute stored correctly."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch, device="cuda")

        assert surrogate.device == "cuda"

    def test_checkpoint_path_stored_as_path(self, mock_torch):
        """checkpoint_path stored as Path object."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_torch)

        assert isinstance(surrogate.checkpoint_path, Path)
        assert surrogate.checkpoint_path == mock_torch


# ---------------------------------------------------------------------------
# Section: Phase I — inverse_design
# ---------------------------------------------------------------------------


class MockSurrogateInverse:
    """Mock surrogate that returns predictable results for inverse design."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        results = []
        for config in configs:
            V0 = config.get("V0", 1e4)
            C = config.get("C", 1e-6)
            results.append({
                "config": config,
                "metrics": {
                    "max_rho": (V0 / 1e4) * (C / 1e-6),
                    "max_Te": V0 * 0.1,
                    "max_Ti": V0 * 0.05,
                    "max_B": (V0 / 1e4) * (C / 1e-6) * 0.01,
                    "n_steps": n_steps,
                },
            })
        return results


class FailingSurrogate:
    """Mock surrogate that always fails."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        return [{"error": "Mock failure"}]


class ExceptionSurrogate:
    """Mock surrogate that raises exceptions."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        raise RuntimeError("Mock exception")


@pytest.fixture
def inverse_mock_surrogate():
    """Create a mock surrogate model for inverse design tests."""
    return MockSurrogateInverse()


@pytest.fixture
def inverse_parameter_ranges():
    """Standard parameter ranges for inverse design testing."""
    return {
        "V0": (5e3, 2e4),
        "C": (5e-7, 2e-6),
    }


def test_inverse_result_defaults():
    """Test InverseResult default initialization."""
    result = InverseResult()
    assert result.best_params == {}
    assert result.best_score == float("inf")
    assert result.all_trials == []
    assert result.n_trials == 0


def test_inverse_result_custom_values():
    """Test InverseResult stores custom values."""
    params = {"V0": 1e4, "C": 1e-6}
    score = 0.123
    trials = [({"V0": 8e3}, 0.5), ({"V0": 1.2e4}, 0.2)]
    n_trials = 10

    result = InverseResult(
        best_params=params,
        best_score=score,
        all_trials=trials,
        n_trials=n_trials,
    )

    assert result.best_params == params
    assert result.best_score == score
    assert result.all_trials == trials
    assert result.n_trials == n_trials


def test_inverse_designer_init(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test InverseDesigner stores surrogate and parameter_ranges."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    assert designer.surrogate is inverse_mock_surrogate
    assert designer.parameter_ranges == inverse_parameter_ranges


def test_objective_exact_match(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test _objective returns 0.0 when prediction exactly matches targets."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {
        "max_rho": 1.0,
        "max_Te": 1000.0,
    }

    score = designer._objective(params, targets, None)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_objective_mismatch(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test _objective returns positive value for mismatched targets."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {
        "max_rho": 2.0,
        "max_Te": 2000.0,
    }

    score = designer._objective(params, targets, None)
    expected = 0.5 + 0.5
    assert score == pytest.approx(expected)


def test_objective_surrogate_failure():
    """Test _objective returns large penalty (1e10) when surrogate fails."""
    failing_surrogate = FailingSurrogate()
    parameter_ranges = {"V0": (5e3, 2e4), "C": (5e-7, 2e-6)}
    designer = InverseDesigner(failing_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}

    score = designer._objective(params, targets, None)
    assert score == 1e10


def test_objective_surrogate_exception():
    """Test _objective returns large penalty (1e10) when surrogate raises exception."""
    exception_surrogate = ExceptionSurrogate()
    parameter_ranges = {"V0": (5e3, 2e4), "C": (5e-7, 2e-6)}
    designer = InverseDesigner(exception_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}

    score = designer._objective(params, targets, None)
    assert score == 1e10


def test_objective_constraint_violation(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test _objective applies constraint penalty for violated constraints."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}
    constraints = {"max_Te": 500.0}

    score = designer._objective(params, targets, constraints)

    expected = 0.0 + 10.0
    assert score == pytest.approx(expected)


def test_objective_constraint_satisfied(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test _objective does not add penalty when constraints are satisfied."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}
    constraints = {"max_Te": 2000.0}

    score = designer._objective(params, targets, constraints)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_objective_missing_metric(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test _objective handles missing metric in prediction."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"nonexistent_metric": 100.0}

    score = designer._objective(params, targets, None)
    assert score == pytest.approx(1e6)


def test_find_config_invalid_method(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test find_config raises ValueError for invalid method."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ValueError, match="Invalid method"):
        designer.find_config(targets, method="invalid_method")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_find_config_evolutionary(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test find_config with method='evolutionary' returns InverseResult."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert result.best_score < float("inf")
    assert result.n_trials > 0
    assert len(result.all_trials) > 0


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
def test_find_config_bayesian(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test find_config with method='bayesian' returns InverseResult."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="bayesian",
        n_trials=20,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert result.best_score < float("inf")
    assert result.n_trials == 20
    assert len(result.all_trials) == 20


def test_bayesian_search_no_optuna(inverse_mock_surrogate, inverse_parameter_ranges, monkeypatch):
    """Test _bayesian_search raises ImportError when optuna unavailable."""
    import dpf.ai.inverse_design

    monkeypatch.setattr(dpf.ai.inverse_design, "HAS_OPTUNA", False)

    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ImportError, match="Optuna required"):
        designer._bayesian_search(targets, None, 10, 42)


def test_evolutionary_search_no_scipy(inverse_mock_surrogate, inverse_parameter_ranges, monkeypatch):
    """Test _evolutionary_search raises ImportError when scipy unavailable."""
    import dpf.ai.inverse_design

    monkeypatch.setattr(dpf.ai.inverse_design, "HAS_SCIPY", False)

    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ImportError, match="SciPy required"):
        designer._evolutionary_search(targets, None, 10, 42)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_parameters_within_bounds(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test parameters stay within specified ranges during optimization."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    for param_name, value in result.best_params.items():
        low, high = inverse_parameter_ranges[param_name]
        assert low <= value <= high

    for params, _ in result.all_trials:
        for param_name, value in params.items():
            low, high = inverse_parameter_ranges[param_name]
            assert low <= value <= high


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_parameter_ranges(inverse_mock_surrogate):
    """Test multiple parameter ranges handled correctly."""
    parameter_ranges = {
        "V0": (5e3, 2e4),
        "C": (5e-7, 2e-6),
        "L0": (1e-8, 1e-7),
    }

    designer = InverseDesigner(inverse_mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert len(result.best_params) == 3
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert "L0" in result.best_params


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_all_trials_populated(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test InverseResult.all_trials populated after optimization."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert len(result.all_trials) > 0
    for params, score in result.all_trials:
        assert isinstance(params, dict)
        assert isinstance(score, (int, float))
        assert score >= 0.0


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_n_trials_matches_actual_count(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test InverseResult.n_trials matches actual trial count."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert result.n_trials == len(result.all_trials)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_targets(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test optimization with multiple target metrics."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {
        "max_rho": 1.5,
        "max_Te": 1200.0,
    }

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert result.best_score < float("inf")
    assert result.best_score < 10.0


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_constraints(inverse_mock_surrogate, inverse_parameter_ranges):
    """Test optimization with multiple constraints."""
    designer = InverseDesigner(inverse_mock_surrogate, inverse_parameter_ranges)
    targets = {"max_rho": 1.0}
    constraints = {
        "max_Te": 1500.0,
        "max_Ti": 600.0,
    }

    result = designer.find_config(
        targets=targets,
        constraints=constraints,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert result.best_score < float("inf")


# --- Section: Phase J — well_loader ---

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


class TestWellLoader(unittest.TestCase):
    def setUp(self):
        import h5py as _h5py  # noqa: E402, I001
        self.temp_dir = tempfile.mkdtemp()
        self.h5_path = Path(self.temp_dir) / "mock_well.h5"

        with _h5py.File(self.h5_path, "w") as f:
            n_traj = 2
            n_steps = 20
            spatial = (16, 16, 16)

            f.create_dataset(
                "density",
                data=np.random.rand(n_traj, n_steps, *spatial).astype(np.float32),
            )
            f.create_dataset(
                "velocity",
                data=np.random.rand(n_traj, n_steps, *spatial, 3).astype(np.float32),
            )
            f.create_dataset(
                "magnetic_field",
                data=np.random.rand(n_traj, n_steps, *spatial, 3).astype(np.float32),
            )
            f.attrs["dt"] = 0.01

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_well_loader_shapes(self):
        import torch as _torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        ds = WellDataset(
            hdf5_paths=[str(self.h5_path)],
            fields=["density", "velocity", "magnetic_field"],
            sequence_length=5,
            stride=1,
            normalize=False,
        )

        self.assertTrue(len(ds) > 0)

        sample = ds[0]
        self.assertIn("rho", sample)
        self.assertIn("velocity", sample)
        self.assertIn("B", sample)

        rho = sample["rho"]
        self.assertEqual(rho.shape, (5, 1, 16, 16, 16))

        B = sample["B"]
        self.assertEqual(B.shape, (5, 3, 16, 16, 16))

    def test_collate_logic(self):
        import torch as _torch  # noqa: E402, I001
        from dpf.ai.train_surrogate import collate_well_to_dpf  # noqa: E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        ds = WellDataset(
            hdf5_paths=[str(self.h5_path)],
            fields=["density", "velocity", "magnetic_field"],
            sequence_length=5,
            normalize=False,
        )

        batch = [ds[0], ds[1]]
        tensor, mask = collate_well_to_dpf(batch)

        self.assertEqual(tensor.shape, (2, 5, 11, 16, 16, 16))

        expected_mask = _torch.tensor(
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=_torch.float32
        )
        self.assertTrue(
            _torch.allclose(mask, expected_mask), f"Got mask {mask}"
        )
        self.assertTrue(
            _torch.allclose(tensor[0, :, 0], batch[0]["rho"].squeeze(1))
        )


# --- Section: Phase J2 — walrus_integration ---


def _make_state_j2(shape=(8, 8, 8)):
    """Create a standard DPF state dict for J2 testing."""
    return {
        "rho": np.ones(shape, dtype=np.float64),
        "velocity": np.zeros((3, *shape), dtype=np.float64),
        "pressure": np.ones(shape, dtype=np.float64) * 100.0,
        "B": np.zeros((3, *shape), dtype=np.float64),
        "Te": np.ones(shape, dtype=np.float64) * 1e4,
        "Ti": np.ones(shape, dtype=np.float64) * 1e4,
        "psi": np.zeros(shape, dtype=np.float64),
    }


def _fake_load_walrus(self, checkpoint_data, mock_instantiate, mock_formatter_class, mock_revin):
    """Mimics _load_walrus_model using pre-built mocks."""
    from dpf.ai.surrogate import _N_CHANNELS  # noqa: E402, I001

    config = checkpoint_data.get("config")
    if config is None:
        self._model = {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "data": checkpoint_data,
        }
        return

    model = mock_instantiate(config.model, n_states=_N_CHANNELS)

    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict"))
    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.eval()
    model.to(self.device)
    self._model = model
    self._revin = mock_revin()
    self._formatter = mock_formatter_class()


class TestWALRUSModelLoading:
    """Test surrogate model loading with real WALRUS model (mocked)."""

    @pytest.fixture
    def mock_walrus_env(self, monkeypatch, tmp_path):
        """Set up mocked walrus/hydra/torch environment."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        mock_torch_mod = MagicMock()
        mock_torch_mod.no_grad.return_value.__enter__ = MagicMock()
        mock_torch_mod.no_grad.return_value.__exit__ = MagicMock()

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_config = MagicMock()
        mock_revin_instance = MagicMock()
        mock_revin_constructor = MagicMock(return_value=mock_revin_instance)
        mock_config.trainer.revin = mock_revin_constructor

        checkpoint_data = {
            "model_state_dict": {"layer.weight": "fake_tensor"},
            "config": mock_config,
        }
        mock_torch_mod.load.return_value = checkpoint_data

        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        mock_instantiate = MagicMock(return_value=mock_model)

        mock_walrus = MagicMock()
        mock_formatter_class = MagicMock()
        mock_formatter_instance = MagicMock()
        mock_formatter_class.return_value = mock_formatter_instance

        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setitem(sys.modules, "walrus", mock_walrus)
        monkeypatch.setitem(sys.modules, "walrus.models", MagicMock())
        monkeypatch.setitem(sys.modules, "walrus.data", MagicMock())
        monkeypatch.setitem(
            sys.modules, "walrus.data.well_to_multi_transformer", MagicMock()
        )

        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        monkeypatch.setattr(
            "dpf.ai.surrogate.DPFSurrogate._load_walrus_model",
            lambda self, sd, cyp, cd: _fake_load_walrus(
                self, cd, mock_instantiate, mock_formatter_class, mock_revin_constructor
            ),
        )

        return {
            "ckpt": ckpt,
            "mock_torch": mock_torch_mod,
            "mock_model": mock_model,
            "mock_instantiate": mock_instantiate,
            "mock_config": mock_config,
            "mock_formatter_class": mock_formatter_class,
            "checkpoint_data": checkpoint_data,
        }

    def test_walrus_model_loaded_not_dict(self, mock_walrus_env):
        """When walrus is available, _model is NOT a dict (it's the real model)."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert not isinstance(surrogate._model, dict)
        assert surrogate._is_walrus_model is True

    def test_walrus_model_is_loaded_true(self, mock_walrus_env):
        """is_loaded returns True when WALRUS model loaded."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate.is_loaded is True

    def test_walrus_model_instantiate_called_with_n_states_11(self, mock_walrus_env):
        """instantiate is called with n_states=11 (5 scalars + 6 vector components)."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_instantiate"].assert_called_once()
        call_kwargs = mock_walrus_env["mock_instantiate"].call_args
        assert call_kwargs[1]["n_states"] == 11

    def test_walrus_model_load_state_dict_called(self, mock_walrus_env):
        """model.load_state_dict is called with checkpoint weights."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_model"].load_state_dict.assert_called_once()

    def test_walrus_model_set_to_eval_mode(self, mock_walrus_env):
        """model.eval() is called after loading."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_model"].eval.assert_called_once()

    def test_walrus_model_moved_to_device(self, mock_walrus_env):
        """model.to(device) is called."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        DPFSurrogate(mock_walrus_env["ckpt"], device="cpu")

        mock_walrus_env["mock_model"].to.assert_called_with("cpu")

    def test_walrus_revin_created(self, mock_walrus_env):
        """RevIN is instantiated from config."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate._revin is not None

    def test_walrus_formatter_created(self, mock_walrus_env):
        """ChannelsFirstWithTimeFormatter is created."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate._formatter is not None

    def test_walrus_missing_config_falls_back_to_placeholder(self, monkeypatch, tmp_path):
        """If checkpoint has no 'config' key, falls back to placeholder dict."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        mock_torch_mod = MagicMock()
        mock_torch_mod.load.return_value = {"state_dict": {}}
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        monkeypatch.setattr(
            "dpf.ai.surrogate.DPFSurrogate._load_walrus_model",
            lambda self, sd, cyp, cd: _fake_load_walrus(
                self, cd, MagicMock(), MagicMock(), MagicMock()
            ),
        )

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        surrogate = DPFSurrogate(ckpt)

        assert isinstance(surrogate._model, dict)
        assert surrogate._is_walrus_model is False


class TestWALRUSTensorConversion:
    """Test _build_walrus_batch and _well_output_to_state."""

    @pytest.fixture
    def surrogate(self, monkeypatch, tmp_path):
        """Create a DPFSurrogate in placeholder mode for tensor tests."""
        torch = pytest.importorskip("torch")

        import dpf.ai.surrogate as surrogate_mod  # noqa: E402, I001

        monkeypatch.setattr(surrogate_mod, "torch", torch)

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        obj = object.__new__(DPFSurrogate)
        obj.checkpoint_path = ckpt
        obj.device = "cpu"
        obj.history_length = 4
        obj._model = {"placeholder": True}
        obj._revin = None
        obj._formatter = None
        obj._walrus_config = None
        obj._field_to_index_map = None
        obj._dpf_field_indices = torch.zeros(11, dtype=torch.long)
        return obj

    def test_dead_methods_removed(self, surrogate):
        """_states_to_walrus_tensor and _tensor_to_state have been removed."""
        assert not hasattr(surrogate, "_states_to_walrus_tensor")
        assert not hasattr(surrogate, "_tensor_to_state")

    def test_build_walrus_batch_exists(self, surrogate):
        """_build_walrus_batch is the active batch builder."""
        assert hasattr(surrogate, "_build_walrus_batch")

    def test_well_output_to_state_exists(self, surrogate):
        """_well_output_to_state is the active output converter."""
        assert hasattr(surrogate, "_well_output_to_state")

    def test_well_output_to_state_correct_shapes(self, surrogate):
        """_well_output_to_state produces correct shapes for scalars and vectors."""
        ref_state = _make_state_j2(shape=(4, 4, 4))
        pred_array = np.zeros((4, 4, 4, 11), dtype=np.float32)

        result = surrogate._well_output_to_state(pred_array, ref_state)

        for key in ["rho", "Te", "Ti", "pressure", "psi"]:
            assert result[key].shape == (4, 4, 4), f"{key} shape mismatch"
        for key in ["B", "velocity"]:
            assert result[key].shape == (3, 4, 4, 4), f"{key} shape mismatch"

    def test_well_output_to_state_channel_order(self, surrogate):
        """Channel order: rho=0, Te=1, Ti=2, pressure=3, psi=4, Bx=5..Bz=7, vx=8..vz=10."""
        ref_state = _make_state_j2(shape=(2, 2, 2))
        pred_array = np.zeros((2, 2, 2, 11), dtype=np.float32)
        for ch in range(11):
            pred_array[..., ch] = float(ch + 1)

        result = surrogate._well_output_to_state(pred_array, ref_state)

        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["Te"], 2.0, atol=1e-6)
        np.testing.assert_allclose(result["Ti"], 3.0, atol=1e-6)
        np.testing.assert_allclose(result["pressure"], 4.0, atol=1e-6)
        np.testing.assert_allclose(result["psi"], 5.0, atol=1e-6)
        np.testing.assert_allclose(result["B"][0], 6.0, atol=1e-6)
        np.testing.assert_allclose(result["B"][1], 7.0, atol=1e-6)
        np.testing.assert_allclose(result["B"][2], 8.0, atol=1e-6)
        np.testing.assert_allclose(result["velocity"][0], 9.0, atol=1e-6)
        np.testing.assert_allclose(result["velocity"][1], 10.0, atol=1e-6)
        np.testing.assert_allclose(result["velocity"][2], 11.0, atol=1e-6)


class TestWellExporterCompliance:
    """Test Well format spec compliance fixes."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create a WellExporter instance."""
        h5py = pytest.importorskip("h5py")  # noqa: F841
        from dpf.ai.well_exporter import WellExporter  # noqa: E402, I001

        output = tmp_path / "test.h5"
        return WellExporter(
            output_path=output,
            grid_shape=(8, 8, 8),
            dx=0.01,
            geometry="cartesian",
        )

    def test_grid_type_is_cartesian(self, exporter, tmp_path):
        """Well files use grid_type='cartesian', not 'uniform'."""
        import h5py  # noqa: E402, I001

        state = _make_state_j2(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            assert f.attrs["grid_type"] == "cartesian"

    def test_t0_fields_have_varying_attributes(self, exporter):
        """t0_fields datasets have dim_varying, sample_varying, time_varying."""
        import h5py  # noqa: E402, I001

        state = _make_state_j2(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            if "t0_fields" in f:
                for ds_name in f["t0_fields"]:
                    ds = f["t0_fields"][ds_name]
                    assert "dim_varying" in ds.attrs, f"Missing dim_varying on {ds_name}"
                    assert "sample_varying" in ds.attrs
                    assert "time_varying" in ds.attrs
                    assert ds.attrs["dim_varying"] is True or ds.attrs["dim_varying"]
                    assert ds.attrs["sample_varying"] is True or ds.attrs["sample_varying"]
                    assert ds.attrs["time_varying"] is True or ds.attrs["time_varying"]

    def test_t1_fields_have_varying_attributes(self, exporter):
        """t1_fields datasets have dim_varying, sample_varying, time_varying."""
        import h5py  # noqa: E402, I001

        state = _make_state_j2(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            if "t1_fields" in f:
                for ds_name in f["t1_fields"]:
                    ds = f["t1_fields"][ds_name]
                    assert "dim_varying" in ds.attrs
                    assert "sample_varying" in ds.attrs
                    assert "time_varying" in ds.attrs


class TestBatchRunnerAPI:
    """Test batch_runner.py WellExporter API usage."""

    def test_batch_runner_well_exporter_constructor_args(self, monkeypatch):
        """BatchRunner passes grid_shape, dx, geometry, sim_params to WellExporter."""
        from dpf.ai.batch_runner import BatchRunner  # noqa: E402, I001

        captured_args: dict = {}

        class FakeWellExporter:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

            def add_snapshot(self, **kwargs):
                pass

            def finalize(self):
                pass

        mock_config = MagicMock()
        mock_config.grid_shape = [8, 8, 8]
        mock_config.dx = 0.01
        mock_config.geometry.type = "cartesian"
        mock_config.model_dump.return_value = {}

        runner = object.__new__(BatchRunner)
        runner.base_config = mock_config
        runner.parameter_ranges = []
        runner.n_samples = 1
        runner.output_dir = Path("/tmp/test_batch")
        runner.workers = 1
        runner.field_interval = 10

        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", FakeWellExporter)
        monkeypatch.setattr(
            "dpf.ai.batch_runner.SimulationConfig", lambda **kw: mock_config
        )

        def mock_run_single(idx, params):
            from dpf.ai.batch_runner import WellExporter  # noqa: E402, I001

            output_path = runner.output_dir / f"trajectory_{idx:04d}.h5"
            exporter = WellExporter(
                output_path=output_path,
                grid_shape=tuple(mock_config.grid_shape),
                dx=mock_config.dx,
                geometry=mock_config.geometry.type,
                sim_params=params,
            )
            exporter.finalize()
            return (idx, None)

        monkeypatch.setattr(runner, "run_single", mock_run_single)
        mock_run_single(0, {"V0": 10e3})

        assert "grid_shape" in captured_args
        assert captured_args["grid_shape"] == (8, 8, 8)
        assert "dx" in captured_args
        assert captured_args["dx"] == 0.01
        assert "geometry" in captured_args
        assert "sim_params" in captured_args

    def test_batch_runner_add_snapshot_uses_state_time(self, monkeypatch):
        """BatchRunner calls add_snapshot with state= and time= kwargs."""
        captured_calls = []

        class FakeWellExporter:
            def __init__(self, **kwargs):
                pass

            def add_snapshot(self, **kwargs):
                captured_calls.append(kwargs)

            def finalize(self):
                pass

        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", FakeWellExporter)

        exporter = FakeWellExporter()
        snapshot = {"rho": np.ones((8, 8, 8)), "time": 1.5}
        t = snapshot.get("time", 0.0)
        exporter.add_snapshot(state=snapshot, time=t)

        assert len(captured_calls) == 1
        assert "state" in captured_calls[0]
        assert "time" in captured_calls[0]
        assert captured_calls[0]["time"] == 1.5


class TestServerEndpoints:
    """Test realtime_server.py endpoint fixes."""

    def test_sweep_uses_surrogate_parameter_sweep(self):
        """POST /api/ai/sweep calls surrogate.parameter_sweep."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001
        from dpf.server.app import app  # noqa: E402, I001

        mock_surrogate_j2 = MagicMock()
        mock_surrogate_j2.parameter_sweep.return_value = [{"max_rho": 1.0}]

        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate_j2

        try:
            client = TestClient(app)
            configs = [{"V0": 10e3}]
            response = client.post("/api/ai/sweep?n_steps=5", json=configs)

            assert response.status_code == 200
            mock_surrogate_j2.parameter_sweep.assert_called_once()
        finally:
            realtime_server._surrogate = original

    def test_inverse_endpoint_uses_find_config_not_optimize(self):
        """ai_inverse endpoint calls designer.find_config, not designer.optimize."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server.ai_inverse)
        assert "find_config" in source
        assert "designer.optimize(" not in source
        assert "inverse_result.best_params" in source
        assert "inverse_result.best_score" in source
        assert "inverse_result.n_trials" in source

    def test_inverse_endpoint_returns_correct_result_structure(self, monkeypatch):
        """ai_inverse returns best_config, loss, n_trials from InverseResult."""
        pytest.importorskip("fastapi")

        from dpf.ai import realtime_server  # noqa: E402, I001

        @dataclass
        class FakeInverseResult:
            best_params: dict = field(default_factory=lambda: {"V0": 10e3})
            best_score: float = 0.01
            all_trials: list = field(default_factory=list)
            n_trials: int = 10

        mock_designer_j2 = MagicMock()
        mock_designer_j2.find_config.return_value = FakeInverseResult()

        mock_surrogate_j2 = MagicMock()
        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate_j2

        try:
            monkeypatch.setattr(
                "dpf.ai.inverse_design.InverseDesigner",
                MagicMock(return_value=mock_designer_j2),
            )

            import asyncio  # noqa: E402, I001

            result = asyncio.get_event_loop().run_until_complete(
                realtime_server.ai_inverse(
                    targets={"max_Te": 5e3},
                    method="bayesian",
                    n_trials=10,
                )
            )

            assert "best_config" in result
            assert result["best_config"] == {"V0": 10e3}
            assert result["loss"] == 0.01
            assert result["n_trials"] == 10
            mock_designer_j2.find_config.assert_called_once()
        finally:
            realtime_server._surrogate = original

    def test_confidence_returns_prediction_with_confidence_fields(self):
        """POST /api/ai/confidence returns PredictionWithConfidence dataclass fields."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001
        from dpf.ai.confidence import PredictionWithConfidence  # noqa: E402, I001
        from dpf.server.app import app  # noqa: E402, I001

        mock_ensemble_j2 = MagicMock()

        sample_state = {
            "rho": np.ones((2, 2, 2)),
            "Te": np.ones((2, 2, 2)) * 1e4,
        }
        mock_ensemble_j2.predict.return_value = PredictionWithConfidence(
            mean_state=sample_state,
            std_state={
                "rho": np.ones((2, 2, 2)) * 0.1,
                "Te": np.ones((2, 2, 2)) * 100.0,
            },
            confidence=0.92,
            ood_score=0.08,
            n_models=3,
        )

        original_surrogate = realtime_server._surrogate
        original_ensemble = realtime_server._ensemble
        realtime_server._surrogate = MagicMock()
        realtime_server._ensemble = mock_ensemble_j2

        try:
            client = TestClient(app)
            history = [{"rho": [[1.0, 2.0], [3.0, 4.0]]}]
            response = client.post("/api/ai/confidence", json=history)

            assert response.status_code == 200
            data = response.json()

            assert "predicted_state" in data
            assert "confidence" in data
            assert "ood_score" in data
            assert "confidence_score" in data
            assert "n_models" in data
            assert "inference_time_ms" in data

            assert data["ood_score"] == pytest.approx(0.08, abs=1e-6)
            assert data["confidence_score"] == pytest.approx(0.92, abs=1e-6)
            assert data["n_models"] == 3
        finally:
            realtime_server._surrogate = original_surrogate
            realtime_server._ensemble = original_ensemble

    def test_confidence_endpoint_no_parameter_sweep_import_error(self):
        """Sweep endpoint does not import from dpf.ai.parameter_sweep."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server)
        assert "from dpf.ai.parameter_sweep" not in source
        assert "import parameter_sweep" not in source


class TestPlaceholderFallback:
    """Test that surrogate works in placeholder mode when HAS_WALRUS=False."""

    @pytest.fixture
    def placeholder_surrogate(self, monkeypatch, tmp_path):
        """Create a surrogate in placeholder mode."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        mock_torch_mod = MagicMock()
        mock_torch_mod.load.return_value = {"state_dict": {}}

        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_scalar_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )
        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_vector_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        return DPFSurrogate(ckpt, history_length=2)

    def test_placeholder_is_not_walrus_model(self, placeholder_surrogate):
        """_is_walrus_model is False in placeholder mode."""
        assert placeholder_surrogate._is_walrus_model is False

    def test_placeholder_predict_returns_copy_of_last_state(self, placeholder_surrogate):
        """predict_next_step returns copy of last state in placeholder mode."""
        history = [_make_state_j2() for _ in range(2)]
        history[-1]["rho"][:] = 42.0

        result = placeholder_surrogate.predict_next_step(history)

        np.testing.assert_allclose(result["rho"], 42.0)
        assert result["rho"] is not history[-1]["rho"]


class TestModuleConstants:
    """Test module-level constants in surrogate.py."""

    def test_n_channels_equals_11(self):
        """_N_CHANNELS is 11 (5 scalars + 2 vectors x 3 components)."""
        from dpf.ai.surrogate import _N_CHANNELS  # noqa: E402, I001

        assert _N_CHANNELS == 11

    def test_scalar_keys_tuple(self):
        """_SCALAR_KEYS contains the 5 DPF scalar field names."""
        from dpf.ai.surrogate import _SCALAR_KEYS  # noqa: E402, I001

        assert _SCALAR_KEYS == ("rho", "Te", "Ti", "pressure", "psi")

    def test_vector_keys_tuple(self):
        """_VECTOR_KEYS contains the 2 DPF vector field names."""
        from dpf.ai.surrogate import _VECTOR_KEYS  # noqa: E402, I001

        assert _VECTOR_KEYS == ("B", "velocity")


class TestConfidenceFix:
    """Test that confidence.py uses constructor, not .load()."""

    def test_ensemble_predictor_uses_constructor(self):
        """EnsemblePredictor uses DPFSurrogate() constructor, not .load()."""
        import inspect  # noqa: E402, I001

        from dpf.ai.confidence import EnsemblePredictor  # noqa: E402, I001

        source = inspect.getsource(EnsemblePredictor)
        assert "DPFSurrogate.load(" not in source
        assert "DPFSurrogate(" in source


class TestValidateAgainstPhysics:
    """Test DPFSurrogate.validate_against_physics cross-validation method."""

    @pytest.fixture
    def placeholder_surrogate(self, monkeypatch, tmp_path):
        """Create placeholder surrogate for validation testing."""
        mock_torch = MagicMock()
        mock_torch.load.return_value = {"state_dict": {}}
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_tensor

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch)
        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_scalar_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )
        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_vector_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )

        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        return DPFSurrogate(ckpt, history_length=2)

    def test_validate_returns_report_dict(self, placeholder_surrogate):
        """validate_against_physics returns a dict with required keys."""
        trajectory = [_make_state_j2() for _ in range(5)]
        report = placeholder_surrogate.validate_against_physics(trajectory)

        assert "n_steps" in report
        assert "per_field_l2" in report
        assert "mean_l2" in report
        assert "max_l2" in report
        assert "diverging_steps" in report

    def test_validate_n_steps_correct(self, placeholder_surrogate):
        """n_steps = len(trajectory) - history_length."""
        trajectory = [_make_state_j2() for _ in range(7)]
        report = placeholder_surrogate.validate_against_physics(trajectory)
        assert report["n_steps"] == 5

    def test_validate_identical_trajectory_zero_error(self, placeholder_surrogate):
        """Placeholder surrogate copies last state, so identical trajectory -> 0 L2."""
        state = _make_state_j2()
        trajectory = [state.copy() for _ in range(5)]
        for t in trajectory:
            for k in t:
                t[k] = state[k].copy()

        report = placeholder_surrogate.validate_against_physics(trajectory)
        assert report["mean_l2"] == pytest.approx(0.0, abs=1e-10)

    def test_validate_diverging_steps_detected(self, placeholder_surrogate):
        """States diverging from placeholder should produce nonzero L2."""
        trajectory = [_make_state_j2() for _ in range(5)]
        trajectory[3]["rho"][:] = 1e6
        trajectory[4]["rho"][:] = 1e6

        report = placeholder_surrogate.validate_against_physics(trajectory)
        assert report["max_l2"] > 0

    def test_validate_too_short_trajectory_raises(self, placeholder_surrogate):
        """Trajectory shorter than history_length + 1 raises ValueError."""
        trajectory = [_make_state_j2() for _ in range(2)]
        with pytest.raises(ValueError, match="too short"):
            placeholder_surrogate.validate_against_physics(trajectory)

    def test_validate_specific_fields(self, placeholder_surrogate):
        """Can validate only specific fields."""
        trajectory = [_make_state_j2() for _ in range(5)]
        report = placeholder_surrogate.validate_against_physics(
            trajectory, fields=["rho", "Te"]
        )
        assert set(report["per_field_l2"].keys()) == {"rho", "Te"}

    def test_validate_method_exists(self):
        """DPFSurrogate has validate_against_physics method."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        assert hasattr(DPFSurrogate, "validate_against_physics")
        assert callable(DPFSurrogate.validate_against_physics)


class TestDeadCodeRemoval:
    """Verify dead _states_to_tensor was removed from surrogate.py."""

    def test_dead_methods_removed(self):
        """Dead _states_to_walrus_tensor and _tensor_to_state have been removed."""
        from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

        assert not hasattr(DPFSurrogate, "_states_to_walrus_tensor")
        assert not hasattr(DPFSurrogate, "_tensor_to_state")
        assert hasattr(DPFSurrogate, "_build_walrus_batch")
        assert hasattr(DPFSurrogate, "_well_output_to_state")

    def test_validate_endpoint_exists(self):
        """realtime_server has /validate endpoint."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server)
        assert "ai_validate" in source
        assert "/validate" in source


class TestWebSocketEnhancements:
    """Test enhanced WebSocket streaming rollout support."""

    def test_websocket_supports_rollout_type(self):
        """WebSocket handler recognizes 'rollout' message type."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server.ai_stream)
        assert '"rollout"' in source or "'rollout'" in source

    def test_websocket_supports_stop_type(self):
        """WebSocket handler recognizes 'stop' message type."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server.ai_stream)
        assert '"stop"' in source or "'stop'" in source

    def test_websocket_rollout_sends_step_events(self):
        """WebSocket rollout sends 'rollout_step' events per step."""
        import inspect  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001

        source = inspect.getsource(realtime_server.ai_stream)
        assert "rollout_step" in source
        assert "rollout_start" in source
        assert "rollout_complete" in source


class TestValidateEndpointHTTP:
    """HTTP-level tests for POST /api/ai/validate."""

    @pytest.fixture
    def mock_surrogate_with_validate(self, monkeypatch):
        """Install a mock surrogate with validate_against_physics."""
        from dpf.ai import realtime_server  # noqa: E402, I001

        mock_surrogate_val = MagicMock()
        mock_surrogate_val.history_length = 2
        mock_surrogate_val.validate_against_physics.return_value = {
            "n_steps": 3,
            "mean_l2": 0.05,
            "max_l2": 0.12,
            "diverging_steps": [],
            "per_field_l2": {
                "rho": [0.04, 0.05, 0.06],
                "Te": [0.03, 0.04, 0.05],
            },
        }

        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate_val
        yield mock_surrogate_val
        realtime_server._surrogate = original

    def test_validate_returns_200(self, mock_surrogate_with_validate):
        """POST /api/ai/validate with valid trajectory returns 200."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        traj = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        response = client.post("/api/ai/validate", json={"trajectory": traj})

        assert response.status_code == 200

    def test_validate_response_has_required_keys(self, mock_surrogate_with_validate):
        """Response contains n_steps, mean_l2, max_l2, diverging_steps, per_field_l2."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        traj = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        response = client.post("/api/ai/validate", json={"trajectory": traj})

        data = response.json()
        assert "n_steps" in data
        assert "mean_l2" in data
        assert "max_l2" in data
        assert "diverging_steps" in data
        assert "per_field_l2" in data
        assert "inference_time_ms" in data

    def test_validate_response_values_correct(self, mock_surrogate_with_validate):
        """Response values match the mocked validate_against_physics return."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        traj = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        response = client.post("/api/ai/validate", json={"trajectory": traj})

        data = response.json()
        assert data["n_steps"] == 3
        assert data["mean_l2"] == pytest.approx(0.05, abs=1e-6)
        assert data["max_l2"] == pytest.approx(0.12, abs=1e-6)
        assert data["diverging_steps"] == []

    def test_validate_too_short_trajectory_returns_422(self):
        """Trajectory with 1 state (< 2 required) returns 422."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001
        from dpf.server.app import app  # noqa: E402, I001

        mock_surrogate_short = MagicMock()
        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate_short

        try:
            client = TestClient(app)
            traj = [{"rho": [[1.0, 2.0]]}]
            response = client.post("/api/ai/validate", json={"trajectory": traj})
            assert response.status_code == 422
        finally:
            realtime_server._surrogate = original

    def test_validate_no_surrogate_returns_503(self):
        """If no surrogate loaded, /validate returns 503."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.ai import realtime_server  # noqa: E402, I001
        from dpf.server.app import app  # noqa: E402, I001

        original = realtime_server._surrogate
        realtime_server._surrogate = None

        try:
            client = TestClient(app)
            traj = [{"rho": [[1.0]]} for _ in range(4)]
            response = client.post("/api/ai/validate", json={"trajectory": traj})
            assert response.status_code == 503
        finally:
            realtime_server._surrogate = original

    def test_validate_calls_validate_against_physics(self, mock_surrogate_with_validate):
        """Surrogate.validate_against_physics is called once with trajectory arrays."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        traj = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        client.post("/api/ai/validate", json={"trajectory": traj})

        mock_surrogate_with_validate.validate_against_physics.assert_called_once()


class TestRolloutEndpointHTTP:
    """HTTP-level tests for POST /api/ai/rollout."""

    @pytest.fixture
    def mock_surrogate_with_rollout(self, monkeypatch):
        """Install a mock surrogate with rollout method."""
        from dpf.ai import realtime_server  # noqa: E402, I001

        predicted_state = {"rho": np.ones((2, 2, 2))}

        mock_surrogate_roll = MagicMock()
        mock_surrogate_roll.history_length = 2
        mock_surrogate_roll.rollout.return_value = [predicted_state] * 5

        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate_roll
        yield mock_surrogate_roll
        realtime_server._surrogate = original

    def test_rollout_returns_200(self, mock_surrogate_with_rollout):
        """POST /api/ai/rollout with valid history returns 200."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        history = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        response = client.post("/api/ai/rollout?n_steps=5", json=history)

        assert response.status_code == 200

    def test_rollout_response_has_trajectory_key(self, mock_surrogate_with_rollout):
        """Rollout response has 'trajectory', 'n_steps', 'total_inference_time_ms'."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        history = [{"rho": [[1.0, 2.0], [3.0, 4.0]]} for _ in range(4)]
        response = client.post("/api/ai/rollout?n_steps=5", json=history)

        data = response.json()
        assert "trajectory" in data
        assert "n_steps" in data
        assert "total_inference_time_ms" in data

    def test_rollout_zero_steps_returns_422(self, mock_surrogate_with_rollout):
        """n_steps=0 returns 422 (must be positive)."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        history = [{"rho": [[1.0, 2.0]]} for _ in range(4)]
        response = client.post("/api/ai/rollout?n_steps=0", json=history)

        assert response.status_code == 422

    def test_rollout_too_many_steps_returns_422(self, mock_surrogate_with_rollout):
        """n_steps > 1000 returns 422."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient  # noqa: E402, I001

        from dpf.server.app import app  # noqa: E402, I001

        client = TestClient(app)
        history = [{"rho": [[1.0, 2.0]]} for _ in range(4)]
        response = client.post("/api/ai/rollout?n_steps=1001", json=history)

        assert response.status_code == 422


# --- Section: Phase R — walrus_hybrid ---


class MockSurrogatePR:
    """Mock WALRUS surrogate for testing HybridEngine without ML dependencies."""

    def __init__(self, history_length: int = 4, behavior: str = "stable") -> None:
        self.history_length = history_length
        self.behavior = behavior
        self._call_count = 0

    def predict_next_step(
        self, history: list[dict]
    ) -> dict:
        self._call_count += 1
        last_state = history[-1]

        if self.behavior == "stable":
            predicted = {}
            for key, val in last_state.items():
                if isinstance(val, np.ndarray):
                    noise = np.random.normal(0, 0.01, val.shape)
                    predicted[key] = val + noise * np.abs(val + 1e-10)
                else:
                    predicted[key] = val
            return predicted

        elif self.behavior == "nan":
            if self._call_count >= 3:
                predicted = {}
                for key, val in last_state.items():
                    if isinstance(val, np.ndarray):
                        predicted[key] = np.full_like(val, np.nan)
                    else:
                        predicted[key] = val
                return predicted
            else:
                return {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in last_state.items()}

        elif self.behavior == "variance_growth":
            growth_factor = 2.5 ** self._call_count
            predicted = {}
            for key, val in last_state.items():
                if isinstance(val, np.ndarray):
                    noise = np.random.normal(0, 0.1 * growth_factor, val.shape)
                    predicted[key] = val + noise
                else:
                    predicted[key] = val
            return predicted

        else:
            raise ValueError(f"Unknown behavior: {self.behavior}")


class MockEnginePR:
    """Mock SimulationEngine for testing HybridEngine (Phase R)."""

    def __init__(self, config) -> None:
        self.config = config
        self._step_count = 0
        self._shape = (8, 8, 8)

    def step(self) -> None:
        self._step_count += 1

    def get_field_snapshot(self) -> dict:
        seed = self._step_count
        rng = np.random.RandomState(seed)

        return {
            "rho": rng.uniform(1e-6, 1e-5, self._shape),
            "velocity": rng.uniform(-1e3, 1e3, (3, *self._shape)),
            "pressure": rng.uniform(100, 200, self._shape),
            "B": rng.uniform(-0.1, 0.1, (3, *self._shape)),
            "Te": rng.uniform(1.0, 2.0, self._shape),
            "Ti": rng.uniform(1.0, 2.0, self._shape),
            "psi": rng.uniform(-1e-3, 1e-3, self._shape),
        }


@pytest.fixture
def minimal_config():
    """Create minimal simulation config for hybrid tests."""
    from dpf.config import SimulationConfig  # noqa: E402, I001

    return SimulationConfig(
        grid_shape=(8, 8, 8),
        dx=0.01,
        sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.2, "validation_interval": 10},
    )


@pytest.fixture
def walrus_mock_surrogate():
    """Create mock WALRUS surrogate."""
    return MockSurrogatePR(history_length=4, behavior="stable")


def test_hybrid_engine_init_valid_handoff(minimal_config, walrus_mock_surrogate) -> None:
    """Test HybridEngine initialization with valid handoff_fraction."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
        validation_interval=50,
        max_l2_divergence=0.1,
    )

    assert engine.handoff_fraction == pytest.approx(0.2)
    assert engine.validation_interval == 50
    assert engine.max_l2_divergence == pytest.approx(0.1)
    assert len(engine._trajectory) == 0


def test_hybrid_engine_init_invalid_handoff_negative(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test HybridEngine rejects handoff_fraction < 0."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    with pytest.raises(ValueError, match=r"handoff_fraction must be in \[0, 1\]"):
        HybridEngine(
            config=minimal_config,
            surrogate=walrus_mock_surrogate,
            handoff_fraction=-0.1,
        )


def test_hybrid_engine_init_invalid_handoff_too_large(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test HybridEngine rejects handoff_fraction > 1."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    with pytest.raises(ValueError, match=r"handoff_fraction must be in \[0, 1\]"):
        HybridEngine(
            config=minimal_config,
            surrogate=walrus_mock_surrogate,
            handoff_fraction=1.5,
        )


def test_hybrid_engine_init_boundary_values(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test HybridEngine accepts boundary values 0.0 and 1.0."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine_zero = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.0,
    )
    assert engine_zero.handoff_fraction == pytest.approx(0.0)

    engine_one = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=1.0,
    )
    assert engine_one.handoff_fraction == pytest.approx(1.0)


def test_run_physics_phase_calls_step_correct_count(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test _run_physics_phase calls engine.step() correct number of times."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine_pr = MockEnginePR(minimal_config)
    n_steps = 20

    history = engine._run_physics_phase(mock_engine_pr, n_steps)

    assert mock_engine_pr._step_count == n_steps
    assert len(history) == n_steps

    for state in history:
        assert "rho" in state
        assert "velocity" in state
        assert "pressure" in state
        assert "B" in state
        assert "Te" in state
        assert "Ti" in state
        assert "psi" in state


def test_run_physics_phase_snapshots_are_independent(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test _run_physics_phase returns independent state copies."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine_pr = MockEnginePR(minimal_config)
    history = engine._run_physics_phase(mock_engine_pr, 5)

    rho_0 = history[0]["rho"]
    rho_1 = history[1]["rho"]
    assert not np.allclose(rho_0, rho_1)


def test_run_surrogate_phase_detects_nan(minimal_config) -> None:
    """Test _run_surrogate_phase detects NaN and falls back."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    surrogate_nan_pr = MockSurrogatePR(history_length=4, behavior="nan")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_nan_pr,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=10)

    assert len(surrogate_history) <= 4
    assert len(surrogate_history) > 0


def test_run_surrogate_phase_detects_inf(minimal_config) -> None:
    """Test _run_surrogate_phase detects Inf values and falls back."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    class InfSurrogate:
        def __init__(self) -> None:
            self.history_length = 4
            self._call_count = 0

        def predict_next_step(self, history):
            self._call_count += 1
            last_state = history[-1]

            if self._call_count >= 2:
                return {
                    "rho": np.full_like(last_state["rho"], np.inf),
                    "velocity": np.full_like(last_state["velocity"], 0.0),
                    "pressure": np.full_like(last_state["pressure"], 100.0),
                    "B": np.full_like(last_state["B"], 0.0),
                    "Te": np.full_like(last_state["Te"], 1.0),
                    "Ti": np.full_like(last_state["Ti"], 1.0),
                    "psi": np.full_like(last_state["psi"], 0.0),
                }
            else:
                return {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in last_state.items()}

    surrogate_inf = InfSurrogate()

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_inf,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=10)

    assert len(surrogate_history) <= 4


def test_run_surrogate_phase_detects_variance_growth(minimal_config) -> None:
    """Test _run_surrogate_phase detects exponential variance growth."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    surrogate_growth_pr = MockSurrogatePR(history_length=4, behavior="variance_growth")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_growth_pr,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    np.random.seed(42)

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=20)

    assert len(surrogate_history) <= 20


def test_run_surrogate_phase_stable_variance_continues(minimal_config) -> None:
    """Test _run_surrogate_phase continues when variance is stable."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    surrogate_stable_pr = MockSurrogatePR(history_length=4, behavior="stable")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_stable_pr,
        handoff_fraction=0.2,
        validation_interval=5,
    )

    np.random.seed(42)

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    n_steps = 15
    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps)

    assert len(surrogate_history) == n_steps


def test_validate_step_removed_as_dead_code_phase_r(
    minimal_config, walrus_mock_surrogate
) -> None:
    """HybridEngine._validate_step was removed (dead code, MOD-4 fix) — Phase R."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )
    assert not hasattr(engine, "_validate_step")


def test_run_complete_flow_with_mock_surrogate(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.run() complete flow with mock surrogate."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
        validation_interval=10,
    )

    max_steps = 50
    summary = engine.run(max_steps=max_steps)

    assert "total_steps" in summary
    assert "physics_steps" in summary
    assert "surrogate_steps" in summary
    assert "wall_time_s" in summary
    assert "fallback_to_physics" in summary

    expected_physics_steps = int(max_steps * 0.2)
    expected_surrogate_steps = max_steps - expected_physics_steps

    assert summary["physics_steps"] == expected_physics_steps
    assert summary["total_steps"] == max_steps
    assert summary["surrogate_steps"] == expected_surrogate_steps
    assert summary["fallback_to_physics"] is False
    assert summary["wall_time_s"] > 0


def test_run_fallback_to_physics_on_nan(minimal_config, monkeypatch) -> None:
    """Test HybridEngine.run() sets fallback flag when surrogate produces NaN."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    surrogate_nan_pr2 = MockSurrogatePR(history_length=4, behavior="nan")

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_nan_pr2,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    summary = engine.run(max_steps=50)

    assert summary["fallback_to_physics"] is True
    assert summary["total_steps"] < 50


def test_run_uses_default_max_steps_when_none(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.run() uses default max_steps=1000 when None passed."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.3,
    )

    summary = engine.run(max_steps=None)

    expected_physics_steps = int(1000 * 0.3)
    expected_surrogate_steps = 1000 - expected_physics_steps

    assert summary["physics_steps"] == expected_physics_steps
    assert summary["total_steps"] == 1000
    assert summary["surrogate_steps"] == expected_surrogate_steps


def test_run_handoff_fraction_zero_all_surrogate(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.run() with handoff_fraction=0 falls back (no physics history)."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.0,
    )

    summary = engine.run(max_steps=20)

    assert summary["physics_steps"] == 0
    assert summary["fallback_to_physics"] is True
    assert summary["surrogate_steps"] == 0


def test_run_handoff_fraction_one_all_physics(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.run() with handoff_fraction=1.0 (all physics)."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=1.0,
    )

    summary = engine.run(max_steps=20)

    assert summary["physics_steps"] == 20
    assert summary["surrogate_steps"] == 0
    assert summary["total_steps"] == 20


def test_trajectory_accumulates_states(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.trajectory accumulates physics + surrogate states."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.4,
    )

    assert len(engine.trajectory) == 0

    engine.run(max_steps=50)

    assert len(engine.trajectory) == 50

    for state in engine.trajectory:
        assert "rho" in state
        assert "velocity" in state
        assert "pressure" in state
        assert "B" in state
        assert "Te" in state
        assert "Ti" in state
        assert "psi" in state


def test_trajectory_empty_before_run(minimal_config, walrus_mock_surrogate) -> None:
    """Test HybridEngine.trajectory is empty before run()."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )

    assert len(engine.trajectory) == 0


def test_trajectory_persists_across_multiple_runs(
    minimal_config, walrus_mock_surrogate, monkeypatch
) -> None:
    """Test HybridEngine.trajectory accumulates across multiple run() calls."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    mock_engine_instance_pr = MockEnginePR(minimal_config)

    def mock_simulation_engine_constructor(config):
        return mock_engine_instance_pr

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.5,
    )

    engine.run(max_steps=20)
    first_run_length = len(engine.trajectory)
    assert first_run_length == 20

    engine.run(max_steps=10)
    second_run_length = len(engine.trajectory)
    assert second_run_length == 30


@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("dpf.athena_wrapper", reason="Athena++ not available"),
    reason="Athena++ not available",
)
def test_engine_hybrid_backend_creates_athena_solver(minimal_config) -> None:
    """Test engine.py hybrid backend creates AthenaPPSolver."""
    from dpf.athena_wrapper import AthenaPPSolver  # noqa: E402, I001
    from dpf.engine import SimulationEngine  # noqa: E402, I001

    from dpf.config import SimulationConfig  # noqa: E402, I001

    config = SimulationConfig(
        grid_shape=(8, 8, 8), dx=0.01, sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.2},
    )

    engine = SimulationEngine(config)

    assert engine.backend == "hybrid"
    assert isinstance(engine.fluid, AthenaPPSolver)
    assert engine._hybrid_engine is None


@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("dpf.athena_wrapper", reason="Athena++ not available"),
    reason="Athena++ not available",
)
def test_engine_hybrid_backend_respects_handoff_fraction(minimal_config) -> None:
    """Test engine.py hybrid backend respects handoff_fraction from config."""
    from dpf.engine import SimulationEngine  # noqa: E402, I001

    from dpf.config import SimulationConfig  # noqa: E402, I001

    config = SimulationConfig(
        grid_shape=(8, 8, 8), dx=0.01, sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.35},
    )

    engine = SimulationEngine(config)

    assert engine.config.fluid.handoff_fraction == pytest.approx(0.35)


def test_fluid_config_handoff_fraction_validation() -> None:
    """Test FluidConfig validates handoff_fraction range."""
    from dpf.config import FluidConfig  # noqa: E402, I001

    valid_config_low = FluidConfig(backend="hybrid", handoff_fraction=0.0)
    assert valid_config_low.handoff_fraction == pytest.approx(0.0)

    valid_config_high = FluidConfig(backend="hybrid", handoff_fraction=1.0)
    assert valid_config_high.handoff_fraction == pytest.approx(1.0)

    valid_config_mid = FluidConfig(backend="hybrid", handoff_fraction=0.5)
    assert valid_config_mid.handoff_fraction == pytest.approx(0.5)

    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", handoff_fraction=-0.1)

    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", handoff_fraction=1.5)


def test_fluid_config_validation_interval_validation() -> None:
    """Test FluidConfig validates validation_interval >= 1."""
    from dpf.config import FluidConfig  # noqa: E402, I001

    valid_config = FluidConfig(backend="hybrid", validation_interval=50)
    assert valid_config.validation_interval == 50

    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", validation_interval=0)


def test_fluid_config_defaults() -> None:
    """Test FluidConfig uses correct defaults for hybrid backend."""
    from dpf.config import FluidConfig  # noqa: E402, I001

    config = FluidConfig(backend="hybrid")

    assert config.handoff_fraction == pytest.approx(0.1)
    assert config.validation_interval == 50


def test_run_surrogate_phase_with_zero_steps(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test _run_surrogate_phase handles n_steps=0 gracefully."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=0)

    assert len(surrogate_history) == 0


def test_run_physics_phase_with_zero_steps(
    minimal_config, walrus_mock_surrogate
) -> None:
    """Test _run_physics_phase handles n_steps=0 gracefully."""
    from dpf.ai.hybrid_engine import HybridEngine  # noqa: E402, I001

    engine = HybridEngine(
        config=minimal_config,
        surrogate=walrus_mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine_pr2 = MockEnginePR(minimal_config)
    history = engine._run_physics_phase(mock_engine_pr2, n_steps=0)

    assert len(history) == 0
    assert mock_engine_pr2._step_count == 0


# --- Section: Phase Z — walrus_benchmarks ---


class _PlaceholderSurrogate:
    """Lightweight stand-in for DPFSurrogate that returns last input state."""

    def __init__(self, history_length: int = 4) -> None:
        self.history_length = history_length

    def predict_next_step(
        self, history: list[dict]
    ) -> dict:
        return {k: v.copy() for k, v in history[-1].items()}

    @property
    def is_loaded(self) -> bool:
        return True

    def validate_against_physics(
        self,
        trajectory: list[dict],
        fields: list | None = None,
    ) -> dict:
        """Replicate the core logic from DPFSurrogate.validate_against_physics."""
        hl = self.history_length
        if len(trajectory) < hl + 1:
            raise ValueError(
                f"Trajectory too short: need {hl + 1}, got {len(trajectory)}"
            )

        if fields is None:
            fields = ["rho", "Te", "Ti", "pressure", "psi", "B", "velocity"]

        per_field_l2: dict = {f: [] for f in fields}
        diverging_steps: list = []
        all_l2: list = []

        for i in range(hl, len(trajectory)):
            history = trajectory[i - hl:i]
            actual = trajectory[i]
            predicted = self.predict_next_step(history)

            step_l2_values: list = []
            for fname in fields:
                if fname not in actual or fname not in predicted:
                    continue
                pred_arr = predicted[fname]
                actual_arr = actual[fname]
                if pred_arr.shape != actual_arr.shape:
                    continue
                diff_norm = float(np.linalg.norm(pred_arr - actual_arr))
                actual_norm = max(float(np.linalg.norm(actual_arr)), 1e-10)
                l2 = diff_norm / actual_norm
                per_field_l2[fname].append(l2)
                step_l2_values.append(l2)

            if step_l2_values:
                step_mean = float(np.mean(step_l2_values))
                all_l2.append(step_mean)
                if step_mean > 0.3:
                    diverging_steps.append(i)

        return {
            "n_steps": len(all_l2),
            "per_field_l2": per_field_l2,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else 0.0,
            "max_l2": float(np.max(all_l2)) if all_l2 else 0.0,
            "diverging_steps": diverging_steps,
        }


class TestBennettTrajectory:
    """Tests for create_bennett_trajectory."""

    def test_returns_correct_length(self) -> None:
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=8, nr=16, nz=4)
        assert len(traj) == 8

    def test_all_states_have_required_keys(self) -> None:
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=3, nr=16, nz=4)
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        for state in traj:
            assert required <= set(state.keys())

    def test_all_states_identical(self) -> None:
        """Bennett equilibrium is stationary -- all timesteps must be equal."""
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=6, nr=16, nz=4)
        for i in range(1, len(traj)):
            for key in traj[0]:
                np.testing.assert_array_equal(
                    traj[0][key], traj[i][key],
                    err_msg=f"State {i} differs from state 0 for field '{key}'",
                )

    def test_states_are_deep_copies(self) -> None:
        """Mutating one state must not affect others."""
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=3, nr=16, nz=4)
        original_rho = traj[1]["rho"].copy()
        traj[0]["rho"][:] = 999.0
        np.testing.assert_array_equal(traj[1]["rho"], original_rho)

    def test_density_positive(self) -> None:
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        assert np.all(traj[0]["rho"] > 0)

    def test_pressure_positive(self) -> None:
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        assert np.all(traj[0]["pressure"] > 0)

    def test_velocity_zero(self) -> None:
        """Bennett equilibrium has no bulk flow."""
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        np.testing.assert_array_equal(traj[0]["velocity"], 0.0)

    def test_shapes_cylindrical(self) -> None:
        from dpf.ai.benchmark_validation import create_bennett_trajectory  # noqa: E402, I001
        nr, nz = 32, 8
        traj = create_bennett_trajectory(n_steps=2, nr=nr, nz=nz)
        state = traj[0]
        assert state["rho"].shape == (nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)


class TestNohTrajectory:
    """Tests for create_noh_trajectory."""

    def test_returns_correct_length(self) -> None:
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(n_steps=8, nr=16, nz=4)
        assert len(traj) == 8

    def test_all_states_have_required_keys(self) -> None:
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(n_steps=3, nr=16, nz=4)
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        for state in traj:
            assert required <= set(state.keys())

    def test_states_evolve(self) -> None:
        """Noh solution is time-dependent -- states must differ."""
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(
            n_steps=5, nr=32, nz=4, B_0=0.0, t_start=0.5, t_end=2.0,
        )
        rho_first = traj[0]["rho"]
        rho_last = traj[-1]["rho"]
        assert not np.allclose(rho_first, rho_last)

    def test_density_positive(self) -> None:
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(n_steps=3, nr=16, nz=4)
        for state in traj:
            assert np.all(state["rho"] > 0)

    def test_inflow_velocity_negative(self) -> None:
        """Upstream radial velocity should be negative (inward flow)."""
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(
            n_steps=2, nr=32, nz=4, r_max=1.0, B_0=0.0,
            t_start=0.5, t_end=1.0,
        )
        vr = traj[0]["velocity"][0, :, 0, 0]
        assert np.any(vr < 0)

    def test_shapes_cylindrical(self) -> None:
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        nr, nz = 32, 8
        traj = create_noh_trajectory(n_steps=2, nr=nr, nz=nz)
        state = traj[0]
        assert state["rho"].shape == (nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)

    def test_unmagnetized_noh(self) -> None:
        """B_0=0 should give B=0 everywhere."""
        from dpf.ai.benchmark_validation import create_noh_trajectory  # noqa: E402, I001
        traj = create_noh_trajectory(n_steps=2, nr=16, nz=4, B_0=0.0)
        np.testing.assert_array_equal(traj[0]["B"], 0.0)


class TestValidateSurrogateAgainstBennett:
    """Tests for validate_surrogate_against_bennett with placeholder model."""

    def test_report_structure(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_bennett  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert "n_steps" in report
        assert "per_field_l2" in report
        assert "mean_l2" in report
        assert "max_l2" in report
        assert "diverging_steps" in report

    def test_placeholder_bennett_l2_zero(self) -> None:
        """Placeholder returns last state = next state for stationary Bennett."""
        from dpf.ai.benchmark_validation import validate_surrogate_against_bennett  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert report["mean_l2"] == pytest.approx(0.0, abs=1e-15)
        assert report["max_l2"] == pytest.approx(0.0, abs=1e-15)

    def test_no_diverging_steps_for_bennett(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_bennett  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert report["diverging_steps"] == []

    def test_n_steps_correct(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_bennett  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=10, nr=16, nz=4,
        )
        assert report["n_steps"] == 6

    def test_specific_fields(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_bennett  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=2)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=6, nr=16, nz=4,
            fields=["rho", "pressure"],
        )
        assert "rho" in report["per_field_l2"]
        assert "pressure" in report["per_field_l2"]


class TestValidateSurrogateAgainstNoh:
    """Tests for validate_surrogate_against_noh with placeholder model."""

    def test_report_structure(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_noh  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert "n_steps" in report
        assert "per_field_l2" in report
        assert "mean_l2" in report
        assert "max_l2" in report
        assert "diverging_steps" in report

    def test_placeholder_noh_l2_nonzero(self) -> None:
        """Placeholder returns last state, but Noh evolves -- L2 should be > 0."""
        from dpf.ai.benchmark_validation import validate_surrogate_against_noh  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=32, nz=4,
            B_0=0.0, t_start=0.5, t_end=2.0,
        )
        assert report["mean_l2"] > 0.0

    def test_n_steps_correct(self) -> None:
        from dpf.ai.benchmark_validation import validate_surrogate_against_noh  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=3)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert report["n_steps"] == 5

    def test_ensures_minimum_trajectory_length(self) -> None:
        """Even if n_steps < history_length+2, wrapper should pad up."""
        from dpf.ai.benchmark_validation import validate_surrogate_against_noh  # noqa: E402, I001
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=3, nr=16, nz=4,
        )
        assert report["n_steps"] >= 1


# --- Section: Verification — walrus ---

_CHECKPOINT_DIR_VERIF = Path(__file__).resolve().parent.parent / "models" / "walrus-pretrained"
_CHECKPOINT_PT_VERIF = _CHECKPOINT_DIR_VERIF / "walrus.pt"

_NX_VERIF = 16


def _skip_if_no_checkpoint_verif() -> None:
    """Skip current test if the WALRUS checkpoint is not on disk."""
    if not _CHECKPOINT_PT_VERIF.exists():
        pytest.skip("WALRUS checkpoint not available at models/walrus-pretrained/walrus.pt")


def make_state(
    nx: int = _NX_VERIF,
    rho: float = 1e-3,
    pressure: float = 1e5,
    Te: float = 1e6,  # noqa: N803
    Ti: float = 1e6,  # noqa: N803
) -> dict:
    return {
        "rho": np.full((nx, nx, nx), rho, dtype=np.float64),
        "velocity": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "pressure": np.full((nx, nx, nx), pressure, dtype=np.float64),
        "B": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "Te": np.full((nx, nx, nx), Te, dtype=np.float64),
        "Ti": np.full((nx, nx, nx), Ti, dtype=np.float64),
        "psi": np.zeros((nx, nx, nx), dtype=np.float64),
    }


def make_history(
    n: int = 4, nx: int = _NX_VERIF, **kwargs: float
) -> list:
    return [make_state(nx=nx, **kwargs) for _ in range(n)]


@pytest.fixture(scope="module")
def surrogate():
    """Module-scoped DPFSurrogate backed by the real WALRUS checkpoint."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("walrus")
    _skip_if_no_checkpoint_verif()

    from dpf.ai.surrogate import DPFSurrogate  # noqa: E402, I001

    return DPFSurrogate(checkpoint_path=_CHECKPOINT_DIR_VERIF, device="cpu")


@pytest.fixture(scope="module")
def predicted_state(surrogate):
    """Module-scoped prediction from a uniform quiescent input."""
    history = make_history(n=surrogate.history_length)
    return surrogate.predict_next_step(history)


class TestT41ModelLoading:
    """Verify that the checkpoint loads into a real WALRUS IsotropicModel."""

    @pytest.mark.slow
    def test_walrus_model_loads(self, surrogate):
        """T4.1.1 -- _is_walrus_model is True after loading checkpoint."""
        assert surrogate._is_walrus_model is True

    @pytest.mark.slow
    def test_walrus_model_field_mapping(self, surrogate):
        """T4.1.2 -- DPF -> WALRUS batch -> DPF roundtrip preserves shapes."""
        history = make_history(n=surrogate.history_length)
        ref = history[0]

        batch = surrogate._build_walrus_batch(history)

        inp = batch["input_fields"]
        assert inp.shape[0] == 1
        assert inp.shape[1] == surrogate.history_length
        assert inp.shape[2] == _NX_VERIF
        assert inp.shape[3] == _NX_VERIF
        assert inp.shape[4] == _NX_VERIF
        assert inp.shape[5] == 11

        inp_np = inp.cpu().numpy()
        rho_channel = inp_np[0, 0, :, :, :, 0]
        np.testing.assert_allclose(
            rho_channel,
            np.float32(ref["rho"].flat[0]),
            rtol=1e-5,
            err_msg="rho channel value mismatch after DPF->batch",
        )


class TestT42PhysicalConstraints:
    """Verify that WALRUS predictions obey basic physical constraints."""

    @pytest.mark.slow
    def test_walrus_density_positive(self, predicted_state):
        """T4.2.1 -- Predicted density should be positive everywhere."""
        rho = predicted_state["rho"]
        assert np.all(rho > -1e-10)
        if np.any(rho < 0):
            neg_frac = np.mean(rho < 0)
            assert neg_frac < 0.01

    @pytest.mark.slow
    def test_walrus_pressure_positive(self, predicted_state):
        """T4.2.2 -- Predicted pressure should be positive everywhere."""
        p = predicted_state["pressure"]
        assert np.all(p > -1e-10)
        if np.any(p < 0):
            neg_frac = np.mean(p < 0)
            assert neg_frac < 0.01

    @pytest.mark.slow
    def test_walrus_prediction_nontrivial(self, surrogate, predicted_state):
        """T4.2.3 -- Output should differ from the last input state."""
        last_input = make_state()

        any_changed = False
        for key in ("rho", "pressure", "Te", "Ti"):
            diff = np.abs(predicted_state[key] - last_input[key])
            rel_change = np.max(diff) / max(np.max(np.abs(last_input[key])), 1e-30)
            if rel_change > 1e-6:
                any_changed = True
                break

        assert any_changed


class TestT43Conservation:
    """Verify approximate conservation of integral quantities."""

    @pytest.mark.slow
    def test_walrus_mass_approximately_conserved(self, surrogate, predicted_state):
        """T4.3.1 -- Total mass should change less than 10% per step."""
        input_state = make_state()
        mass_in = np.sum(input_state["rho"])
        mass_out = np.sum(predicted_state["rho"])

        if mass_in > 0:
            rel_change = abs(mass_out - mass_in) / mass_in
            assert rel_change < 0.10

    @pytest.mark.slow
    def test_walrus_energy_bounded(self, surrogate, predicted_state):
        """T4.3.2 -- Total energy should not grow unboundedly."""
        def total_energy(state: dict) -> float:
            gamma = 5.0 / 3.0
            E_th = np.sum(state["pressure"]) / (gamma - 1.0)
            v2 = np.sum(state["velocity"] ** 2, axis=0)
            E_kin = 0.5 * np.sum(state["rho"] * v2)
            mu0 = 4.0 * np.pi * 1e-7
            B2 = np.sum(state["B"] ** 2, axis=0)
            E_mag = np.sum(B2) / (2.0 * mu0)
            return float(E_th + E_kin + E_mag)

        input_state = make_state()
        E_in = total_energy(input_state)
        E_out = total_energy(predicted_state)

        assert E_out < 10.0 * max(E_in, 1e-30)


class TestT44Consistency:
    """Verify determinism and Lipschitz continuity."""

    @pytest.mark.slow
    def test_walrus_deterministic(self, surrogate):
        """T4.4.1 -- Same inputs yield identical outputs."""
        history = make_history(n=surrogate.history_length)

        pred_a = surrogate.predict_next_step(history)
        pred_b = surrogate.predict_next_step(history)

        for key in pred_a:
            np.testing.assert_array_equal(
                pred_a[key],
                pred_b[key],
                err_msg=f"Field '{key}' differs between two identical runs",
            )

    @pytest.mark.slow
    def test_walrus_continuous(self, surrogate):
        """T4.4.2 -- Small input perturbation produces small output change."""
        history_base = make_history(n=surrogate.history_length)

        history_pert = [
            {k: v.copy() for k, v in s.items()} for s in history_base
        ]
        epsilon = 0.01 * history_pert[-1]["rho"].mean()
        history_pert[-1]["rho"] += epsilon

        pred_base = surrogate.predict_next_step(history_base)
        pred_pert = surrogate.predict_next_step(history_pert)

        max_rel_change = 0.0
        for key in ("rho", "pressure", "Te", "Ti", "psi"):
            diff = np.max(np.abs(pred_pert[key] - pred_base[key]))
            scale = max(np.max(np.abs(pred_base[key])), 1e-30)
            max_rel_change = max(max_rel_change, diff / scale)

        K_bound = 100.0
        assert max_rel_change < K_bound


class TestT45PhysicsSanity:
    """Sanity checks grounded in physical intuition."""

    @pytest.mark.slow
    def test_walrus_static_state_stable(self, surrogate, predicted_state):
        """T4.5.1 -- Uniform quiescent state should remain approximately uniform."""
        for key in ("rho", "pressure", "Te", "Ti"):
            field = predicted_state[key]
            field_mean = np.mean(field)
            field_std = np.std(field)

            if abs(field_mean) > 1e-30:
                cv = field_std / abs(field_mean)
                assert cv < 0.50

    @pytest.mark.slow
    def test_walrus_shock_propagates(self, surrogate):
        """T4.5.2 -- State with a sharp gradient should show the gradient moving."""
        nx = _NX_VERIF
        mid = nx // 2

        def sod_state() -> dict:
            s = make_state(nx=nx)
            s["rho"][:mid, :, :] = 1.0
            s["rho"][mid:, :, :] = 0.125
            s["pressure"][:mid, :, :] = 1e5
            s["pressure"][mid:, :, :] = 1e4
            return s

        history = [sod_state() for _ in range(surrogate.history_length)]
        pred = surrogate.predict_next_step(history)

        input_rho = history[-1]["rho"]
        pred_rho = pred["rho"]

        diff = np.abs(pred_rho - input_rho)
        max_diff = np.max(diff)

        assert max_diff > 1e-10

    @pytest.mark.slow
    def test_walrus_output_no_nans_or_infs(self, predicted_state):
        """T4.5.3 -- Output fields should contain no NaN or Inf values."""
        for _key, arr in predicted_state.items():
            assert np.all(np.isfinite(arr))


# --- Section: Well Integration ---

_OUTPUT_DIR_WELL = Path("tests/output_well_test")


@pytest.fixture
def clean_output():
    if _OUTPUT_DIR_WELL.exists():
        shutil.rmtree(_OUTPUT_DIR_WELL)
    _OUTPUT_DIR_WELL.mkdir(parents=True)
    yield


def test_well_exporter_integration(clean_output):
    """Test that WellExporter writes files during simulation."""
    from dpf.config import SimulationConfig  # noqa: E402, I001
    from dpf.engine import SimulationEngine  # noqa: E402, I001

    config_dict = {
        "grid_shape": [32, 32, 32],
        "dx": 0.01,
        "sim_time": 1e-7,
        "circuit": {
            "C": 1e-6, "V0": 1000.0, "L0": 1e-9, "R0": 0.01,
            "anode_radius": 0.01, "cathode_radius": 0.02
        },
        "diagnostics": {
            "hdf5_filename": str(_OUTPUT_DIR_WELL / "test_diag.h5"),
            "output_interval": 10,
            "well_output_interval": 2,
            "well_filename_prefix": "well_test"
        },
        "fluid": {
            "backend": "python"
        }
    }
    config = SimulationConfig(**config_dict)
    engine = SimulationEngine(config)

    engine.run(max_steps=5)

    if hasattr(engine, "close"):
        engine.close()

    npz_files = list(_OUTPUT_DIR_WELL.glob("well_test_*.npz"))
    h5_files = list(_OUTPUT_DIR_WELL.glob("well_test_*.h5"))
    files = npz_files + h5_files
    assert len(files) > 0, "No Well export files found"

    if h5_files:
        import h5py  # noqa: E402, I001
        with h5py.File(h5_files[0], "r") as f:
            assert "t0_fields" in f
            t0 = f["t0_fields"]
            t1 = f.get("t1_fields", {})

            assert "density" in t0
            assert "pressure" in t0 or "electron_temperature" in t0

            rho = t0["density"][:]
            assert rho.ndim == 5
            assert rho.shape[0] == 1
            assert rho.shape[1] >= 2

            if "velocity" in t1:
                vel = t1["velocity"][:]
                assert vel.ndim == 6
    else:
        import numpy as _np  # noqa: E402, I001
        data = _np.load(npz_files[0])
        assert "density" in data


@pytest.mark.slow
@pytest.mark.xfail(
    reason="WALRUS IsotropicModel requires grid >= 16x16x16 but test uses 8x8x8; "
    "kernel size assertion fails in flexi_utils.choose_kernel_size_deterministic",
    strict=False,
)
def test_hybrid_engine_delegation(clean_output):
    """Test that backend='hybrid' delegates to HybridEngine."""
    from dpf.config import SimulationConfig  # noqa: E402, I001
    from dpf.engine import SimulationEngine  # noqa: E402, I001

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
            "handoff_fraction": 0.5,
            "validation_interval": 10
        },
        "ai": {
            "device": "cpu"
        },
        "diagnostics": {
            "hdf5_filename": str(_OUTPUT_DIR_WELL / "hybrid_diag.h5"),
            "well_output_interval": 0
        }
    }
    config = SimulationConfig(**config_dict)
    engine = SimulationEngine(config)

    summary = engine.run(max_steps=10)

    assert "total_steps" in summary
    assert "physics_steps" in summary
    assert "surrogate_steps" in summary

    assert summary["physics_steps"] == 5
    assert summary["surrogate_steps"] == 5


# --- Section: Well Loader Stats ---


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _create_mock_h5(
    path: str,
    n_traj: int = 1,
    n_steps: int = 20,
    spatial: tuple = (8, 8, 8),
    density_fill: float | None = None,
    add_attrs: bool = False,
    inject_nan: bool = False,
) -> str:
    """Create a mock Well HDF5 file with scalar/vector fields."""
    import h5py  # noqa: E402, I001

    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(42)
        shape_scalar = (n_traj, n_steps, *spatial)
        shape_vector = (n_traj, n_steps, *spatial, 3)

        if density_fill is not None:
            density_data = np.full(shape_scalar, density_fill, dtype=np.float32)
        else:
            density_data = rng.standard_normal(shape_scalar).astype(np.float32) + 5.0

        if inject_nan:
            nan_mask = rng.random(shape_scalar) < 0.1
            density_data[nan_mask] = np.nan

        ds = f.create_dataset("density", data=density_data)
        if add_attrs:
            ds.attrs["mean"] = 5.0
            ds.attrs["std"] = 1.0
            ds.attrs["rms"] = 5.1

        vel_data = rng.standard_normal(shape_vector).astype(np.float32)
        vs = f.create_dataset("velocity", data=vel_data)
        if add_attrs:
            vs.attrs["mean"] = 0.0
            vs.attrs["std"] = 1.0

    return path


class TestComputeStatsBasic:
    """Test compute_stats() produces valid non-empty stats."""

    def test_stats_not_empty(self, tmp_dir):
        """C3 core fix: compute_stats() must never return empty dict."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert len(stats) > 0, "compute_stats() returned empty dict (C3 bug)"
        assert "density" in stats
        assert "velocity" in stats

    def test_stats_has_mean_std_rms(self, tmp_dir):
        """Each field entry must have mean, std, and rms keys."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        for field_name in ["density", "velocity"]:
            assert "mean" in stats[field_name]
            assert "std" in stats[field_name]
            assert "rms" in stats[field_name]

    def test_stats_values_correct_for_constant(self, tmp_dir):
        """Constant field should have std near 0 and rms equal to |mean|."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "const.h5"), density_fill=3.0
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        assert abs(s["mean"] - 3.0) < 0.01
        assert s["std"] < 0.01
        assert abs(s["rms"] - 3.0) < 0.01

    def test_rms_formula(self, tmp_dir):
        """RMS should satisfy: rms^2 = mean^2 + std^2 (approximately)."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        rms_expected = np.sqrt(s["mean"] ** 2 + s["std"] ** 2)
        assert abs(s["rms"] - rms_expected) < 0.01


class TestComputeStatsEdgeCases:
    """Edge cases that previously caused C3 (empty dict return)."""

    def test_empty_dataset_returns_defaults(self, tmp_dir):
        """No valid files -> stats should have safe defaults, not {}."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        ds = WellDataset.__new__(WellDataset)
        ds.paths = []
        ds.fields = ["density", "velocity"]
        ds.sequence_length = 5
        ds.stride = 1
        ds.normalize = True
        ds.stats = {}
        ds.samples = []

        stats = ds.compute_stats()
        assert len(stats) == 2
        for fname in ["density", "velocity"]:
            assert stats[fname]["mean"] == 0.0
            assert stats[fname]["std"] == 1.0
            assert stats[fname]["rms"] == 1.0

    def test_nan_values_filtered(self, tmp_dir):
        """Fields with NaN values should still produce valid stats."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "nan.h5"), inject_nan=True
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["std"])
        assert np.isfinite(s["rms"])

    def test_single_sample_dataset(self, tmp_dir):
        """Dataset with exactly 1 sample should still compute stats."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "single.h5"),
            n_traj=1,
            n_steps=6,
            spatial=(4, 4, 4),
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        assert len(ds) >= 1
        stats = ds.compute_stats(max_samples=1)
        assert "density" in stats
        assert stats["density"]["rms"] > 0

    def test_std_floor_prevents_zero(self, tmp_dir):
        """std should never be exactly 0 (floor at 1e-10)."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "const.h5"), density_fill=7.0
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert stats["density"]["std"] >= 1e-10


class TestComputeStatsHDF5Attrs:
    """Test Strategy 1: reading stats from HDF5 attributes."""

    def test_reads_attrs_when_present(self, tmp_dir):
        """Stats should be loaded from HDF5 attrs without sampling data."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "attrs.h5"), add_attrs=True
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert abs(stats["density"]["mean"] - 5.0) < 0.01
        assert abs(stats["density"]["std"] - 1.0) < 0.01
        assert abs(stats["density"]["rms"] - 5.1) < 0.01

    def test_attrs_rms_derived_when_missing(self, tmp_dir):
        """When HDF5 attrs have mean/std but no rms, derive rms from them."""
        import h5py  # noqa: E402, I001
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = str(Path(tmp_dir) / "partial_attrs.h5")
        with h5py.File(h5_path, "w") as f:
            data = np.ones((1, 20, 4, 4, 4), dtype=np.float32) * 3.0
            ds_f = f.create_dataset("density", data=data)
            ds_f.attrs["mean"] = 3.0
            ds_f.attrs["std"] = 4.0
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert abs(stats["density"]["rms"] - 5.0) < 0.01


class TestAutoComputeStats:
    """Test that __init__ auto-calls compute_stats() when needed."""

    def test_auto_compute_on_init_with_normalize(self, tmp_dir):
        """When normalize=True and no stats provided, stats populated in __init__."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "auto.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
        )
        assert len(ds.stats) > 0
        assert "density" in ds.stats

    def test_no_auto_compute_when_stats_provided(self, tmp_dir):
        """External stats should be used as-is without re-computation."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "ext.h5"))
        external_stats = {"density": {"mean": 99.0, "std": 1.0, "rms": 99.0}}
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
            normalization_stats=external_stats,
        )
        assert ds.stats["density"]["mean"] == 99.0

    def test_no_auto_compute_when_normalize_false(self, tmp_dir):
        """When normalize=False, stats should remain empty."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(str(Path(tmp_dir) / "noauto.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        assert len(ds.stats) == 0


class TestFindFieldDataset:
    """Test the _find_field_dataset static helper."""

    def test_finds_root_level(self, tmp_dir):
        import h5py  # noqa: E402, I001
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = str(Path(tmp_dir) / "root.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("density", data=np.zeros((1, 10, 4, 4, 4)))
            dset = WellDataset._find_field_dataset(f, "density")
            assert dset is not None

    def test_finds_t0_fields(self, tmp_dir):
        import h5py  # noqa: E402, I001
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = str(Path(tmp_dir) / "t0.h5")
        with h5py.File(h5_path, "w") as f:
            g = f.create_group("t0_fields")
            g.create_dataset("pressure", data=np.zeros((1, 10, 4, 4, 4)))
            dset = WellDataset._find_field_dataset(f, "pressure")
            assert dset is not None

    def test_finds_t1_fields(self, tmp_dir):
        import h5py  # noqa: E402, I001
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = str(Path(tmp_dir) / "t1.h5")
        with h5py.File(h5_path, "w") as f:
            g = f.create_group("t1_fields")
            g.create_dataset("velocity", data=np.zeros((1, 10, 4, 4, 4, 3)))
            dset = WellDataset._find_field_dataset(f, "velocity")
            assert dset is not None

    def test_returns_none_for_missing(self, tmp_dir):
        import h5py  # noqa: E402, I001
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = str(Path(tmp_dir) / "empty.h5")
        with h5py.File(h5_path, "w") as f:
            dset = WellDataset._find_field_dataset(f, "nonexistent")
            assert dset is None


class TestStatsUsedInNormalization:
    """Verify that computed stats actually affect __getitem__ output."""

    def test_normalization_changes_values(self, tmp_dir):
        """Data should be different with and without normalization."""
        import torch  # noqa: F401, E402, I001
        from dpf.ai.well_loader import WellDataset  # noqa: E402, I001

        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "norm.h5"), density_fill=5.0
        )
        ds_raw = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        ds_norm = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
        )
        raw_val = ds_raw[0]["rho"].mean().item()
        norm_val = ds_norm[0]["rho"].mean().item()
        assert abs(raw_val - 5.0) < 0.01
        assert abs(norm_val) < abs(raw_val)
