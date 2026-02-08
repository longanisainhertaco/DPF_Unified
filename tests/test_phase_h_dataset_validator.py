"""Tests for Well format HDF5 dataset validation.

Phase H: WALRUS pipeline — dataset validation, preprocessing, model training.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dpf.ai.dataset_validator import DatasetValidator, ValidationResult


@pytest.fixture
def valid_well_file(tmp_path: Path) -> Path:
    """Create a valid Well HDF5 file with 4x4x4 grid, 1 trajectory, 5 timesteps.

    Returns:
        Path to created HDF5 file.
    """
    path = tmp_path / "valid_well.h5"

    # Dimensions for 4x4x4 grid
    nx, ny, nz = 4, 4, 4
    n_traj = 1
    n_steps = 5

    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["dataset_name"] = "test_dpf_dataset"
        f.attrs["n_trajectories"] = n_traj

        # t0_fields (scalar fields): (n_traj, n_steps, nz, ny, nx)
        t0_group = f.create_group("t0_fields")
        density = np.random.uniform(1.0, 2.0, (n_traj, n_steps, nz, ny, nx))
        t0_group.create_dataset("density", data=density)
        pressure = np.random.uniform(0.5, 1.5, (n_traj, n_steps, nz, ny, nx))
        t0_group.create_dataset("pressure", data=pressure)

        # t1_fields (vector fields): (n_traj, n_steps, nz, ny, nx, 3)
        t1_group = f.create_group("t1_fields")
        velocity = np.random.uniform(-0.1, 0.1, (n_traj, n_steps, nz, ny, nx, 3))
        t1_group.create_dataset("velocity", data=velocity)
        B_field = np.random.uniform(-0.5, 0.5, (n_traj, n_steps, nz, ny, nx, 3))
        t1_group.create_dataset("B", data=B_field)

        # dimensions group
        dim_group = f.create_group("dimensions")
        dim_group.create_dataset("x", data=np.linspace(0.0, 1.0, nx))
        dim_group.create_dataset("y", data=np.linspace(0.0, 1.0, ny))
        dim_group.create_dataset("z", data=np.linspace(0.0, 1.0, nz))

        # boundary_conditions group
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
        # No dataset_name or n_trajectories attribute
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
        density[0, 2, 1, 1, 1] = np.nan  # Inject NaN
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
        velocity[0, 3, 2, 2, 2, 1] = np.inf  # Inject Inf
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

        # Create scalars group with energy_conservation field
        scalars_group = f.create_group("scalars")
        # Energy conservation should be ~1.0; create drift of 0.03
        energy_cons = np.array([1.0, 0.99, 1.01, 0.97, 1.02])
        scalars_group.create_dataset("energy_conservation", data=energy_cons)

        # Add required groups
        t0_group = f.create_group("t0_fields")
        t0_group.create_dataset("density", data=np.ones((1, 5, 4, 4, 4)))
        f.create_group("dimensions")
        f.create_group("boundary_conditions")

    validator = DatasetValidator()
    drift = validator.check_energy_conservation(path)

    # Max drift is |1.0 - 0.97| = 0.03
    assert drift == pytest.approx(0.03, abs=1e-6)


def test_check_energy_conservation_triggers_warning(tmp_path: Path):
    """check_energy_conservation with large drift triggers warning in validate_file."""
    path = tmp_path / "large_drift.h5"

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "test"
        f.attrs["n_trajectories"] = 1

        # Create large energy drift (10%)
        scalars_group = f.create_group("scalars")
        energy_cons = np.array([1.0, 0.95, 1.05, 0.90, 1.10])
        scalars_group.create_dataset("energy_conservation", data=energy_cons)

        # Add required groups
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

    # Should have stats for density, pressure, velocity, B
    assert "density" in stats
    assert "pressure" in stats
    assert "velocity" in stats
    assert "B" in stats

    # Check density stats structure
    density_stats = stats["density"]
    assert "mean" in density_stats
    assert "std" in density_stats
    assert "min" in density_stats
    assert "max" in density_stats
    assert "n_nan" in density_stats
    assert "n_inf" in density_stats

    # Verify reasonable values (density was uniform(1.0, 2.0))
    assert 1.0 <= density_stats["mean"] <= 2.0
    assert density_stats["n_nan"] == 0
    assert density_stats["n_inf"] == 0


def test_validate_directory_finds_all_files(tmp_path: Path):
    """validate_directory finds and validates all .h5 files."""
    # Create multiple valid HDF5 files
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
    """validate_directory returns empty dict for nonexistent dir."""
    validator = DatasetValidator()
    results = validator.validate_directory(tmp_path / "nonexistent")

    assert results == {}


def test_summary_report_formatting(tmp_path: Path):
    """summary_report formats readable output with VALID/INVALID status."""
    # Create one valid and one invalid file
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
        # Missing required groups
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
    # Create file with schema error
    path = tmp_path / "errors.h5"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "error_test"
        # Missing dimensions and boundary_conditions

    # Create file with energy drift warning
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
