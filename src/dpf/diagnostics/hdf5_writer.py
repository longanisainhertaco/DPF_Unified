"""HDF5 time-series diagnostics writer.

Records scalar and field quantities at each output step into
an HDF5 file for post-processing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dpf.core.bases import DiagnosticsBase

logger = logging.getLogger(__name__)

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available; HDF5 diagnostics disabled")

# Field names to include in snapshots
_SNAPSHOT_FIELDS = ("rho", "B", "Te", "Ti", "velocity", "pressure")
_SCALAR_UNITS = {
    "time": "s",
    "current": "A",
    "voltage": "V",
    "energy_cap": "J",
    "energy_ind": "J",
    "energy_res": "J",
    "energy_total": "J",
    "max_div_B": "T/cell",
    "mean_rho": "kg/m^3",
    "max_Te": "K",
    "max_Ti": "K",
}
_SCALAR_METADATA = {
    "max_div_B": {
        "diagnostic_status": "rough_array_metric_not_physical_divergence",
        "validation_status": "not_validation_evidence",
        "diagnostic_method": (
            "component-wise np.gradient over array indices for B shaped "
            "(3, nx, ny, nz); geometry and grid spacing are not applied"
        ),
    },
}
_FIELD_UNITS = {
    "rho": "kg/m^3",
    "B": "T",
    "Te": "K",
    "Ti": "K",
    "velocity": "m/s",
    "pressure": "Pa",
}


def _rough_array_max_div_b(B: np.ndarray) -> float:
    """Return a compatibility metric, not a physical divergence diagnostic."""
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim != 4 or B_arr.shape[0] != 3:
        raise ValueError("B must have shape (3, nx, ny, nz)")

    div_B = np.zeros(B_arr.shape[1:], dtype=float)
    if B_arr.shape[1] > 1:
        div_B = div_B + np.gradient(B_arr[0], axis=0)
    if B_arr.shape[2] > 1:
        div_B = div_B + np.gradient(B_arr[1], axis=1)
    if B_arr.shape[3] > 1:
        div_B = div_B + np.gradient(B_arr[2], axis=2)
    if div_B.size == 0:
        return 0.0
    return float(np.max(np.abs(div_B)))


class HDF5Writer(DiagnosticsBase):
    """Write simulation diagnostics to an HDF5 file.

    Creates datasets for:
    - Scalar time series: time, current, voltage, energy_cap, energy_ind,
      max_div_B, mean_density, max_temperature
    - Field snapshots (optional): rho, B, velocity, pressure, Te, Ti
      at specified intervals

    Args:
        filename: Output HDF5 file path.
        field_output_interval: Write full field data every N output calls (0 = never).
    """

    def __init__(
        self,
        filename: str = "diagnostics.h5",
        field_output_interval: int = 0,
    ) -> None:
        self.filename = filename
        self.field_output_interval = field_output_interval
        self._call_count = 0
        self._file: Any | None = None
        self._scalars: dict[str, list] = {
            "time": [],
            "current": [],
            "voltage": [],
            "energy_cap": [],
            "energy_ind": [],
            "energy_res": [],
            "energy_total": [],
            "max_div_B": [],
            "mean_rho": [],
            "max_Te": [],
            "max_Ti": [],
        }
        self._field_snapshots: list[dict[str, Any]] = []

    @property
    def field_snapshots(self) -> list[dict[str, Any]]:
        """Public access to accumulated field snapshots."""
        return self._field_snapshots

    def record(self, state: dict[str, Any], time: float) -> None:
        """Record diagnostics from current simulation state.

        Args:
            state: Dictionary with keys including 'rho', 'B', 'Te', 'Ti',
                   'circuit' (a dict with current, voltage, energies).
            time: Current simulation time [s].
        """
        self._call_count += 1

        # Extract circuit info
        circuit = state.get("circuit", {})

        self._scalars["time"].append(time)
        self._scalars["current"].append(circuit.get("current", 0.0))
        self._scalars["voltage"].append(circuit.get("voltage", 0.0))
        self._scalars["energy_cap"].append(circuit.get("energy_cap", 0.0))
        self._scalars["energy_ind"].append(circuit.get("energy_ind", 0.0))
        self._scalars["energy_res"].append(circuit.get("energy_res", 0.0))
        self._scalars["energy_total"].append(circuit.get("energy_total", 0.0))

        # Field diagnostics
        rho = state.get("rho")
        B = state.get("B")
        Te = state.get("Te")
        Ti = state.get("Ti")

        if rho is not None:
            self._scalars["mean_rho"].append(float(np.mean(rho)))
        else:
            self._scalars["mean_rho"].append(0.0)

        if B is not None:
            self._scalars["max_div_B"].append(_rough_array_max_div_b(B))
        else:
            self._scalars["max_div_B"].append(0.0)

        if Te is not None:
            self._scalars["max_Te"].append(float(np.max(Te)))
        else:
            self._scalars["max_Te"].append(0.0)

        if Ti is not None:
            self._scalars["max_Ti"].append(float(np.max(Ti)))
        else:
            self._scalars["max_Ti"].append(0.0)

        # === Field snapshots ===
        if (
            self.field_output_interval > 0
            and self._call_count % self.field_output_interval == 0
        ):
            snapshot: dict[str, Any] = {"time": time}
            for field_name in _SNAPSHOT_FIELDS:
                arr = state.get(field_name)
                if arr is not None and isinstance(arr, np.ndarray):
                    snapshot[field_name] = arr.copy()
            self._field_snapshots.append(snapshot)
            logger.debug(
                "Captured field snapshot %d at t=%.4e",
                len(self._field_snapshots), time,
            )

    def finalize(self) -> None:
        """Write all accumulated data to the HDF5 file."""
        if not HAS_H5PY:
            logger.warning("Cannot write HDF5: h5py not installed")
            return

        logger.info("Writing diagnostics to %s", self.filename)
        with h5py.File(self.filename, "w") as f:
            # Write scalar time series
            grp = f.create_group("scalars")
            for key, values in self._scalars.items():
                dataset = grp.create_dataset(key, data=np.array(values))
                dataset.attrs["units"] = _SCALAR_UNITS.get(key, "unknown")
                for attr_key, attr_value in _SCALAR_METADATA.get(key, {}).items():
                    dataset.attrs[attr_key] = attr_value

            # Write field snapshots
            if self._field_snapshots:
                fields_grp = f.create_group("fields")
                for idx, snap in enumerate(self._field_snapshots):
                    snap_grp = fields_grp.create_group(f"snapshot_{idx:04d}")
                    snap_grp.attrs["time"] = snap["time"]
                    for key, val in snap.items():
                        if key != "time" and isinstance(val, np.ndarray):
                            dataset = snap_grp.create_dataset(key, data=val)
                            dataset.attrs["units"] = _FIELD_UNITS.get(key, "unknown")
                fields_grp.attrs["num_snapshots"] = len(self._field_snapshots)
                logger.info(
                    "Wrote %d field snapshots", len(self._field_snapshots),
                )

            f.attrs["schema_version"] = "dpf-hdf5-diagnostics-v1"
            f.attrs["time_base_units"] = "s"
            f.attrs["num_records"] = self._call_count

        logger.info("Wrote %d diagnostic records to %s", self._call_count, self.filename)
