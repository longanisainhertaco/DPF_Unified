"""Export DPF simulation data to The Well HDF5 format for WALRUS training.

The Well format expects:
    t0_fields: scalar fields, shape (n_traj, n_steps, *spatial)
    t1_fields: vector fields, shape (n_traj, n_steps, *spatial, D)
    scalars: time-varying circuit quantities, shape (n_traj, n_steps)

References:
    The Well data format: https://github.com/PolymathicAI/the_well
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dpf.ai.field_mapping import (
    CIRCUIT_SCALARS,
    FIELD_UNITS,
    SCALAR_FIELDS,
    VECTOR_FIELDS,
)

logger = logging.getLogger(__name__)

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available; Well export disabled")


class WellExporter:
    """Export DPF simulation data to The Well HDF5 format.

    Accumulates snapshots from DPF simulations and exports them to
    The Well format used by WALRUS neural operator training pipeline.

    Args:
        output_path: Output HDF5 file path.
        grid_shape: Spatial grid shape (nx, ny, nz).
        dx: Grid spacing in x (or r) direction [m].
        dz: Grid spacing in z direction [m] (defaults to dx if None).
        geometry: Coordinate system ("cartesian" or "cylindrical").
        sim_params: Optional simulation parameters to store as root attributes.
    """

    def __init__(
        self,
        output_path: str | Path,
        grid_shape: tuple[int, int, int],
        dx: float,
        dz: float | None = None,
        geometry: str = "cartesian",
        sim_params: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.grid_shape = grid_shape
        self.dx = dx
        self.dz = dz if dz is not None else dx
        self.geometry = geometry
        self.sim_params = sim_params or {}
        self._snapshots: list[dict] = []
        self._times: list[float] = []
        self._circuit_history: list[dict] = []

    def add_snapshot(
        self,
        state: dict[str, np.ndarray],
        time: float,
        circuit_scalars: dict[str, float] | None = None,
    ) -> None:
        """Add a simulation snapshot to the export buffer.

        Args:
            state: DPF state dict with fields like rho, B, velocity, etc.
            time: Simulation time [s].
            circuit_scalars: Optional circuit quantities (current, voltage, etc.).
        """
        self._snapshots.append(state.copy())
        self._times.append(time)
        self._circuit_history.append(circuit_scalars.copy() if circuit_scalars else {})

    def add_from_dpf_hdf5(self, hdf5_path: str | Path) -> None:
        """Load snapshots from a DPF HDF5Writer output file.

        Args:
            hdf5_path: Path to DPF diagnostics HDF5 file.

        Raises:
            ImportError: If h5py is not available.
            FileNotFoundError: If the file does not exist.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required to load DPF HDF5 files")

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"DPF HDF5 file not found: {hdf5_path}")

        logger.info("Loading snapshots from %s", hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            # Load field snapshots
            if "fields" in f:
                fields_grp = f["fields"]
                num_snapshots = fields_grp.attrs.get("num_snapshots", 0)

                for idx in range(num_snapshots):
                    snap_grp = fields_grp[f"snapshot_{idx:04d}"]
                    time = float(snap_grp.attrs["time"])

                    # Reconstruct state dict
                    state: dict[str, np.ndarray] = {}
                    for key in snap_grp:
                        state[key] = snap_grp[key][:]

                    # Load corresponding circuit scalars if available
                    circuit_scalars: dict[str, float] = {}
                    if "scalars" in f:
                        scalars_grp = f["scalars"]
                        for scalar_name in CIRCUIT_SCALARS:
                            if scalar_name in scalars_grp:
                                arr = scalars_grp[scalar_name][:]
                                if idx < len(arr):
                                    circuit_scalars[scalar_name] = float(arr[idx])

                    self.add_snapshot(state, time, circuit_scalars)

        logger.info("Loaded %d snapshots from DPF HDF5", len(self._snapshots))

    def finalize(self, n_trajectories: int = 1) -> Path:
        """Write all accumulated data to Well HDF5 format.

        Args:
            n_trajectories: Number of trajectories (typically 1 for single sim).

        Returns:
            Path to the written HDF5 file.

        Raises:
            ImportError: If h5py is not available.
            ValueError: If no snapshots have been accumulated.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for Well export")

        if not self._snapshots:
            raise ValueError("No snapshots accumulated; cannot export")

        n_steps = len(self._snapshots)
        nx, ny, nz = self.grid_shape

        logger.info(
            "Writing %d snapshots to Well format: %s",
            n_steps,
            self.output_path,
        )

        with h5py.File(self.output_path, "w") as f:
            # Root attributes
            f.attrs["dataset_name"] = "dpf_simulation"
            f.attrs["grid_type"] = "uniform"
            f.attrs["n_spatial_dims"] = 2 if self.geometry == "cylindrical" else 3
            f.attrs["n_trajectories"] = n_trajectories

            # Add simulation parameters as root attributes
            for key, value in self.sim_params.items():
                f.attrs[key] = value

            # Dimensions group
            dims_grp = f.create_group("dimensions")
            if self.geometry == "cylindrical":
                dims_grp.create_dataset("r", data=np.linspace(0, nx * self.dx, nx, dtype=np.float32))
                dims_grp.create_dataset("theta", data=np.linspace(0, ny * self.dx, ny, dtype=np.float32))
            else:
                dims_grp.create_dataset("x", data=np.linspace(0, nx * self.dx, nx, dtype=np.float32))
                dims_grp.create_dataset("y", data=np.linspace(0, ny * self.dx, ny, dtype=np.float32))
            dims_grp.create_dataset("z", data=np.linspace(0, nz * self.dz, nz, dtype=np.float32))
            dims_grp.create_dataset("time", data=np.array(self._times, dtype=np.float32))

            # Boundary conditions group
            bc_grp = f.create_group("boundary_conditions")
            if self.geometry == "cylindrical":
                bc_grp.attrs["geometry_type"] = "cylindrical"
                bc_grp.attrs["r_inner"] = "axis"
                bc_grp.attrs["r_outer"] = "outflow"
                bc_grp.attrs["z_low"] = "wall"
                bc_grp.attrs["z_high"] = "wall"
            else:
                bc_grp.attrs["geometry_type"] = "cartesian"
                bc_grp.attrs["all"] = "periodic"

            # t0_fields: scalar fields
            t0_grp = f.create_group("t0_fields")
            for dpf_name, well_name in SCALAR_FIELDS.items():
                # Check if field exists in at least one snapshot
                if not any(dpf_name in snap for snap in self._snapshots):
                    continue

                # Build full array (n_traj, n_steps, nx, ny, nz)
                field_array = np.zeros((n_trajectories, n_steps, nx, ny, nz), dtype=np.float32)
                for step_idx, snap in enumerate(self._snapshots):
                    if dpf_name in snap:
                        field_array[0, step_idx] = snap[dpf_name].astype(np.float32)

                dataset = t0_grp.create_dataset(well_name, data=field_array)
                if dpf_name in FIELD_UNITS:
                    dataset.attrs["units"] = FIELD_UNITS[dpf_name]

            # t1_fields: vector fields
            t1_grp = f.create_group("t1_fields")
            for dpf_name, well_name in VECTOR_FIELDS.items():
                # Check if field exists in at least one snapshot
                if not any(dpf_name in snap for snap in self._snapshots):
                    continue

                # Build full array (n_traj, n_steps, nx, ny, nz, 3)
                field_array = np.zeros((n_trajectories, n_steps, nx, ny, nz, 3), dtype=np.float32)
                for step_idx, snap in enumerate(self._snapshots):
                    if dpf_name in snap:
                        # Convert from (3, nx, ny, nz) to (nx, ny, nz, 3)
                        vec = snap[dpf_name]
                        field_array[0, step_idx] = np.moveaxis(vec, 0, -1).astype(np.float32)

                dataset = t1_grp.create_dataset(well_name, data=field_array)
                if dpf_name in FIELD_UNITS:
                    dataset.attrs["units"] = FIELD_UNITS[dpf_name]

            # Scalars: circuit quantities
            if any(self._circuit_history):
                scalars_grp = f.create_group("scalars")
                for scalar_name in CIRCUIT_SCALARS:
                    # Extract time series for this scalar
                    values = []
                    for circuit_dict in self._circuit_history:
                        values.append(circuit_dict.get(scalar_name, 0.0))

                    if any(v != 0.0 for v in values):
                        # Shape: (n_traj, n_steps)
                        scalar_array = np.zeros((n_trajectories, n_steps), dtype=np.float32)
                        scalar_array[0, :] = np.array(values, dtype=np.float32)
                        scalars_grp.create_dataset(scalar_name, data=scalar_array)

        logger.info("Wrote Well HDF5 to %s", self.output_path)
        return self.output_path
