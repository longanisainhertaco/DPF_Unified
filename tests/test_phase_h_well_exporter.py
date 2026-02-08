"""Phase H tests for WellExporter class.

Tests the DPF → Well HDF5 export functionality for WALRUS neural operator training.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from dpf.ai.field_mapping import FIELD_UNITS, SCALAR_FIELDS, VECTOR_FIELDS
from dpf.ai.well_exporter import WellExporter

# ── Test Fixtures ────────────────────────────────────────────────────────


def make_dpf_state(nx: int = 4, ny: int = 4, nz: int = 4) -> dict[str, np.ndarray]:
    """Generate a fake DPF state dict for testing.

    Args:
        nx: Grid size in x (or r) direction.
        ny: Grid size in y (or theta) direction.
        nz: Grid size in z direction.

    Returns:
        State dict with all required fields populated with random data.
    """
    rng = np.random.default_rng(42)
    state = {
        "rho": rng.random((nx, ny, nz)),
        "pressure": rng.random((nx, ny, nz)),
        "Te": rng.random((nx, ny, nz)) * 1e4,
        "Ti": rng.random((nx, ny, nz)) * 1e4,
        "psi": rng.random((nx, ny, nz)) * 1e-3,
        "velocity": rng.random((3, nx, ny, nz)),
        "B": rng.random((3, nx, ny, nz)),
    }
    return state


def make_circuit_scalars() -> dict[str, float]:
    """Generate fake circuit scalar data for testing.

    Returns:
        Dict with typical circuit quantities.
    """
    return {
        "current": 1.5e5,
        "voltage": 2.5e3,
        "energy_conservation": 0.999,
        "R_plasma": 0.05,
        "Z_bar": 3.5,
    }


# ── Constructor Tests ────────────────────────────────────────────────────


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
    grid_shape = (4, 4, 4)
    dx = 0.001

    exporter = WellExporter(output_path=output_path, grid_shape=grid_shape, dx=dx)

    assert exporter.dz == dx  # dz defaults to dx
    assert exporter.geometry == "cartesian"
    assert exporter.sim_params == {}


def test_well_exporter_init_accepts_str_path(tmp_path: Path) -> None:
    """WellExporter __init__ accepts string path and converts to Path."""
    output_path = str(tmp_path / "test.h5")
    exporter = WellExporter(output_path=output_path, grid_shape=(4, 4, 4), dx=0.001)

    assert isinstance(exporter.output_path, Path)
    assert exporter.output_path == Path(output_path)


# ── add_snapshot Tests ───────────────────────────────────────────────────


def test_add_snapshot_stores_state_and_time(tmp_path: Path) -> None:
    """add_snapshot stores state dict and time in internal buffers."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    time = 1.5e-6
    exporter.add_snapshot(state, time)

    assert len(exporter._snapshots) == 1
    assert len(exporter._times) == 1
    assert exporter._times[0] == time
    assert np.allclose(exporter._snapshots[0]["rho"], state["rho"])


def test_add_snapshot_accumulates_multiple_snapshots(tmp_path: Path) -> None:
    """add_snapshot accumulates multiple snapshots over time."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    for i in range(5):
        state = make_dpf_state()
        time = i * 1e-7
        exporter.add_snapshot(state, time)

    assert len(exporter._snapshots) == 5
    assert len(exporter._times) == 5
    assert exporter._times == [0.0, 1e-7, 2e-7, 3e-7, 4e-7]


def test_add_snapshot_stores_circuit_scalars(tmp_path: Path) -> None:
    """add_snapshot stores circuit scalars when provided."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    circuit_scalars = make_circuit_scalars()
    exporter.add_snapshot(state, 1e-6, circuit_scalars)

    assert len(exporter._circuit_history) == 1
    assert exporter._circuit_history[0]["current"] == circuit_scalars["current"]
    assert exporter._circuit_history[0]["voltage"] == circuit_scalars["voltage"]


def test_add_snapshot_handles_missing_circuit_scalars(tmp_path: Path) -> None:
    """add_snapshot stores empty dict when circuit_scalars is None."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)

    assert len(exporter._circuit_history) == 1
    assert exporter._circuit_history[0] == {}


def test_add_snapshot_copies_state_dict(tmp_path: Path) -> None:
    """add_snapshot creates a shallow copy of the state dict."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)

    # Remove a key from original state dict
    original_keys = set(state.keys())
    del state["rho"]

    # Stored snapshot should still have all keys (shallow copy)
    assert set(exporter._snapshots[0].keys()) == original_keys


# ── finalize Tests ───────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_raises_when_no_snapshots(tmp_path: Path) -> None:
    """finalize raises ValueError when no snapshots have been accumulated."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    with pytest.raises(ValueError, match="No snapshots accumulated"):
        exporter.finalize()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_writes_hdf5_file_to_disk(tmp_path: Path) -> None:
    """finalize writes HDF5 file to disk at specified path."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)

    result_path = exporter.finalize()

    assert result_path == output_path
    assert output_path.exists()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_root_attributes(tmp_path: Path) -> None:
    """finalize creates root-level HDF5 attributes."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert f.attrs["dataset_name"] == "dpf_simulation"
        assert f.attrs["grid_type"] == "uniform"
        assert f.attrs["n_spatial_dims"] == 3
        assert f.attrs["n_trajectories"] == 1


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_dimensions_group_cartesian(tmp_path: Path) -> None:
    """finalize creates /dimensions group with x/y/z coordinates for Cartesian."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    dx, dz = 0.001, 0.002

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=dx,
        dz=dz,
        geometry="cartesian",
    )

    state = make_dpf_state(nx, ny, nz)
    exporter.add_snapshot(state, 1e-6)
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
    dx = 0.001

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=dx,
        geometry="cylindrical",
    )

    state = make_dpf_state(nx, ny, nz)
    exporter.add_snapshot(state, 1e-6)
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
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    times = [0.0, 1e-7, 2e-7, 3e-7]
    for t in times:
        state = make_dpf_state()
        exporter.add_snapshot(state, t)

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
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=0.001,
    )

    state = make_dpf_state(nx, ny, nz)
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        t0_grp = f["t0_fields"]
        for _dpf_name, well_name in SCALAR_FIELDS.items():
            assert well_name in t0_grp
            dataset = t0_grp[well_name]
            # Shape: (n_traj, n_steps, nx, ny, nz)
            assert dataset.shape == (1, 1, nx, ny, nz)
            assert dataset.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_t1_fields_group(tmp_path: Path) -> None:
    """finalize creates /t1_fields group with vector fields."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=0.001,
    )

    state = make_dpf_state(nx, ny, nz)
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        t1_grp = f["t1_fields"]
        for _dpf_name, well_name in VECTOR_FIELDS.items():
            assert well_name in t1_grp
            dataset = t1_grp[well_name]
            # Shape: (n_traj, n_steps, nx, ny, nz, 3)
            assert dataset.shape == (1, 1, nx, ny, nz, 3)
            assert dataset.dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_scalar_field_shapes(tmp_path: Path) -> None:
    """Scalar fields have correct shape: (n_traj, n_steps, nx, ny, nz)."""
    output_path = tmp_path / "test.h5"
    nx, ny, nz = 4, 4, 4
    n_steps = 3
    n_traj = 1

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=0.001,
    )

    for i in range(n_steps):
        state = make_dpf_state(nx, ny, nz)
        exporter.add_snapshot(state, i * 1e-7)

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

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=0.001,
    )

    for i in range(n_steps):
        state = make_dpf_state(nx, ny, nz)
        exporter.add_snapshot(state, i * 1e-7)

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

    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(nx, ny, nz),
        dx=0.001,
    )

    state = make_dpf_state(nx, ny, nz)
    # Set a distinctive pattern in the vector field
    state["B"] = np.zeros((3, nx, ny, nz))
    state["B"][0, 0, 0, 0] = 1.0  # Bx component
    state["B"][1, 0, 0, 0] = 2.0  # By component
    state["B"][2, 0, 0, 0] = 3.0  # Bz component

    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        B = f["t1_fields/magnetic_field"][0, 0, 0, 0, 0, :]  # Extract single point
        # Component order should be preserved: [Bx, By, Bz]
        assert pytest.approx(B[0]) == 1.0
        assert pytest.approx(B[1]) == 2.0
        assert pytest.approx(B[2]) == 3.0


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_scalars_group(tmp_path: Path) -> None:
    """finalize creates /scalars group with circuit quantities."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    circuit_scalars = make_circuit_scalars()
    exporter.add_snapshot(state, 1e-6, circuit_scalars)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert "scalars" in f
        scalars_grp = f["scalars"]
        assert "current" in scalars_grp
        assert "voltage" in scalars_grp
        assert scalars_grp["current"].shape == (1, 1)  # (n_traj, n_steps)
        assert scalars_grp["current"].dtype == np.float32


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_creates_boundary_conditions_group(tmp_path: Path) -> None:
    """finalize creates /boundary_conditions group with appropriate attrs."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        geometry="cartesian",
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
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
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        geometry="cylindrical",
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
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
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        geometry="cartesian",
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
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
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        sim_params=sim_params,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
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
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    result = exporter.finalize()

    assert isinstance(result, Path)
    assert result == output_path


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_with_n_trajectories_parameter(tmp_path: Path) -> None:
    """finalize with n_trajectories parameter sets correct array shapes."""
    output_path = tmp_path / "test.h5"
    n_traj = 3
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize(n_trajectories=n_traj)

    with h5py.File(output_path, "r") as f:
        assert f.attrs["n_trajectories"] == n_traj
        # All fields should have n_traj in first dimension
        assert f["t0_fields/density"].shape[0] == n_traj
        assert f["t1_fields/magnetic_field"].shape[0] == n_traj


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_missing_fields_handled_with_zeros(tmp_path: Path) -> None:
    """Missing fields in some snapshots are filled with zeros."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    # First snapshot has all fields
    state1 = make_dpf_state()
    exporter.add_snapshot(state1, 1e-7)

    # Second snapshot is missing Te
    state2 = make_dpf_state()
    del state2["Te"]
    exporter.add_snapshot(state2, 2e-7)

    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        electron_temp = f["t0_fields/electron_temp"][0, :, 0, 0, 0]
        # First timestep should have data
        assert electron_temp[0] > 0
        # Second timestep should be zero
        assert electron_temp[1] == 0.0


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_circuit_scalars_only_created_when_nonzero(tmp_path: Path) -> None:
    """Circuit scalars only created when at least one non-zero value exists."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    # Add snapshots with some zero-valued circuit scalars
    state = make_dpf_state()
    circuit_scalars = {
        "current": 1.5e5,  # Non-zero
        "voltage": 0.0,  # All zeros
        "neutron_rate": 0.0,  # All zeros
    }
    exporter.add_snapshot(state, 1e-7, circuit_scalars)
    exporter.add_snapshot(state, 2e-7, circuit_scalars)

    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        scalars_grp = f["scalars"]
        # current should be present (non-zero)
        assert "current" in scalars_grp
        # voltage should NOT be present (all zeros)
        assert "voltage" not in scalars_grp
        # neutron_rate should NOT be present (all zeros)
        assert "neutron_rate" not in scalars_grp


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_field_units_stored_as_attrs(tmp_path: Path) -> None:
    """Field units are stored as dataset attributes."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        # Check scalar field units
        density = f["t0_fields/density"]
        assert density.attrs["units"] == FIELD_UNITS["rho"]

        electron_temp = f["t0_fields/electron_temp"]
        assert electron_temp.attrs["units"] == FIELD_UNITS["Te"]

        # Check vector field units
        magnetic_field = f["t1_fields/magnetic_field"]
        assert magnetic_field.attrs["units"] == FIELD_UNITS["B"]

        velocity = f["t1_fields/velocity"]
        assert velocity.attrs["units"] == FIELD_UNITS["velocity"]


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_multiple_snapshots_produce_time_series(tmp_path: Path) -> None:
    """Multiple snapshots produce correct time series in HDF5."""
    output_path = tmp_path / "test.h5"
    n_steps = 5
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    # Add snapshots with increasing values
    for i in range(n_steps):
        state = make_dpf_state()
        state["rho"][:] = float(i + 1)  # 1.0, 2.0, 3.0, 4.0, 5.0
        circuit_scalars = {"current": float((i + 1) * 1e5)}
        exporter.add_snapshot(state, i * 1e-7, circuit_scalars)

    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        # Check time series in scalar fields
        density = f["t0_fields/density"][0, :, 0, 0, 0]
        assert len(density) == n_steps
        assert pytest.approx(density[0]) == 1.0
        assert pytest.approx(density[4]) == 5.0

        # Check time series in circuit scalars
        current = f["scalars/current"][0, :]
        assert len(current) == n_steps
        assert pytest.approx(current[0]) == 1e5
        assert pytest.approx(current[4]) == 5e5


# ── add_from_dpf_hdf5 Tests ──────────────────────────────────────────────


def test_add_from_dpf_hdf5_raises_importerror_when_h5py_unavailable(tmp_path: Path) -> None:
    """add_from_dpf_hdf5 raises ImportError when h5py is unavailable."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    with (
        patch("dpf.ai.well_exporter.HAS_H5PY", False),
        pytest.raises(ImportError, match="h5py is required"),
    ):
        exporter.add_from_dpf_hdf5(tmp_path / "input.h5")


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_add_from_dpf_hdf5_raises_filenotfounderror(tmp_path: Path) -> None:
    """add_from_dpf_hdf5 raises FileNotFoundError for missing file."""
    exporter = WellExporter(
        output_path=tmp_path / "test.h5",
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    missing_path = tmp_path / "nonexistent.h5"
    with pytest.raises(FileNotFoundError, match="DPF HDF5 file not found"):
        exporter.add_from_dpf_hdf5(missing_path)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cylindrical_writes_r_theta(tmp_path: Path) -> None:
    """Cylindrical geometry writes r/theta instead of x/y in dimensions."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        geometry="cylindrical",
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        dims = f["dimensions"]
        # Should have r, theta, z
        assert "r" in dims
        assert "theta" in dims
        assert "z" in dims
        # Should NOT have x, y
        assert "x" not in dims
        assert "y" not in dims


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_cylindrical_n_spatial_dims(tmp_path: Path) -> None:
    """Cylindrical geometry sets n_spatial_dims to 2."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
        geometry="cylindrical",
    )

    state = make_dpf_state()
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        assert f.attrs["n_spatial_dims"] == 2


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
def test_finalize_no_scalars_group_when_no_circuit_data(tmp_path: Path) -> None:
    """finalize does not create /scalars group when no circuit data exists."""
    output_path = tmp_path / "test.h5"
    exporter = WellExporter(
        output_path=output_path,
        grid_shape=(4, 4, 4),
        dx=0.001,
    )

    state = make_dpf_state()
    # Add snapshot without circuit_scalars
    exporter.add_snapshot(state, 1e-6)
    exporter.finalize()

    with h5py.File(output_path, "r") as f:
        # scalars group should not be created
        assert "scalars" not in f
