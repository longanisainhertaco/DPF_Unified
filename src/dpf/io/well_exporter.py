"""Well format exporter for DPF simulation data.

Saves simulation trajectories in the "Well" format (HDF5 or NumPy-based)
compatible with WALRUS training.

References:
    https://github.com/PolymathicAI/the_well
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# Try importing h5py, fall back to numpy.savez if unavailable
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from dpf.ai.field_mapping import (
    SCALAR_FIELDS,
    VECTOR_FIELDS,
    dpf_scalar_to_well,
)

logger = logging.getLogger(__name__)


class WellExporter:
    """Buffer and save simulation states in Well format.

    Accumulates a list of state dictionaries and flushes them to disk
    either periodically or at the end of a simulation.
    """

    def __init__(
        self,
        output_dir: str | Path,
        filename_prefix: str = "simulation",
        buffer_size: int = 100,
        enable: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.buffer_size = buffer_size
        self.enable = enable

        self._buffer: list[dict[str, np.ndarray]] = []
        self._batch_count = 0

        if self.enable:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not HAS_H5PY:
                logger.warning(
                    "h5py not installed; WellExporter will use .npz format, "
                    "which may be less efficient for large datasets."
                )

    def append_state(self, state: dict[str, np.ndarray]) -> None:
        """Add a state to the buffer.

        Args:
            state: DPF state dict (keys: rho, velocity, B, etc.)
        """
        if not self.enable:
            return

        # Deep copy to ensure we don't store references to mutating arrays
        state_copy = {k: v.copy() for k, v in state.items()}
        self._buffer.append(state_copy)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered states to disk."""
        if not self.enable or not self._buffer:
            return

        filename = f"{self.filename_prefix}_{self._batch_count:04d}"
        if HAS_H5PY:
            path = self.output_dir / f"{filename}.h5"
            self._save_h5(path)
        else:
            path = self.output_dir / f"{filename}.npz"
            self._save_npz(path)

        logger.info(f"Saved {len(self._buffer)} steps to {path}")
        self._buffer.clear()
        self._batch_count += 1

    def close(self) -> None:
        """Flush remaining data."""
        self.flush()

    def _prepare_well_arrays(self) -> dict[str, np.ndarray]:
        """Convert buffered DPF states to Well-format arrays.

        Returns:
            Dict with keys corresponding to Well fields (e.g. 'density', 'velocity').
            Shapes will be (n_traj=1, n_steps, *spatial, [dims]).
        """
        n_steps = len(self._buffer)
        n_traj = 1
        traj_idx = 0

        well_data = {}

        # 1. Inspect first state to get spatial shapes
        ref = self._buffer[0]

        # 2. Convert Scalars
        for dpf_name, well_name in SCALAR_FIELDS.items():
            if dpf_name not in ref:
                continue

            # Allocate: (n_traj, n_steps, nx, ny, nz)
            spatial = ref[dpf_name].shape
            arr = np.zeros((n_traj, n_steps, *spatial), dtype=np.float32)

            for t, state in enumerate(self._buffer):
                arr = dpf_scalar_to_well(
                    state[dpf_name], traj_idx, t, n_traj, n_steps
                )
                # Optimize: direct assignment if dpf_scalar_to_well returns full array
                # But dpf_scalar_to_well returns a full (1,1,...) array.
                # Let's perform a simpler loop here to avoid alloc turnover.
                pass

            # Re-implementation for efficiency avoiding repeated allocs
            # dpf.ai.field_mapping helpers are 'one-shot'.
            # We'll just loop and assign.
            arr = np.stack(
                [s[dpf_name].astype(np.float32) for s in self._buffer], axis=0
            )
            # arr shape: (n_steps, nx, ny, nz)
            # Add n_traj dim
            arr = arr[np.newaxis, ...]
            well_data[well_name] = arr

        # 3. Convert Vectors
        for dpf_name, well_name in VECTOR_FIELDS.items():
            if dpf_name not in ref:
                continue

            # dpf vector: (3, nx, ny, nz) -> well: (nx, ny, nz, 3)
            # Stack over time
            # list of (3, nx, ny, nz)
            vec_list = [s[dpf_name] for s in self._buffer]
            # stack -> (n_steps, 3, nx, ny, nz)
            stacked = np.stack(vec_list, axis=0).astype(np.float32)
            # move axis 1 (components) to last -> (n_steps, nx, ny, nz, 3)
            transposed = np.moveaxis(stacked, 1, -1)
            # add traj dim
            well_data[well_name] = transposed[np.newaxis, ...]

        return well_data

    def _save_h5(self, path: Path) -> None:
        """Save buffer to HDF5."""
        data = self._prepare_well_arrays()
        with h5py.File(path, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v, compression="gzip")

            # Add metadata
            f.attrs["n_steps"] = len(self._buffer)
            f.attrs["source"] = "DPF-Unified WellExporter"

    def _save_npz(self, path: Path) -> None:
        """Save buffer to .npz."""
        data = self._prepare_well_arrays()
        np.savez_compressed(path, **data)
