"""Tests for Phase H: field_mapping module (DPF <-> Well format conversion).

Tests field name mappings, unit constants, shape transformations, and validation
utilities for interfacing with the WALRUS physics surrogate system.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.ai.field_mapping import (
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
    # Other indices should be zero
    assert np.all(result[1, 0] == 0)
    assert np.all(result[0, 1] == 0)


def test_dpf_vector_to_well_shape_and_moveaxis():
    """Verify dpf_vector_to_well converts (3,nx,ny,nz) -> (n_traj,n_steps,nx,ny,nz,3)."""
    field = np.random.rand(3, 4, 4, 4)
    result = dpf_vector_to_well(field, traj_idx=0, step_idx=0, n_traj=1, n_steps=1)
    assert result.shape == (1, 1, 4, 4, 4, 3)
    assert result.dtype == np.float32
    # Verify moveaxis: first component should match
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
    # Verify reverse moveaxis
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
