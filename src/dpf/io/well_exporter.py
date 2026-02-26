"""Well format exporter for DPF simulation data.

Thin adapter around dpf.ai.well_exporter.WellExporter that provides
a buffered append_state / flush / close API for use by engine.py.

References:
    https://github.com/PolymathicAI/the_well
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dpf.ai.well_exporter import WellExporter as _FullWellExporter

logger = logging.getLogger(__name__)


class WellExporter:
    """Buffer and save simulation states in Well format.

    Wraps the full-featured :class:`dpf.ai.well_exporter.WellExporter`
    with a simpler append / flush API suitable for engine.py.
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
        self._times: list[float] = []
        self._batch_count = 0

        if self.enable:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def append_state(
        self, state: dict[str, np.ndarray], time: float = 0.0,
    ) -> None:
        """Add a state to the buffer.

        Args:
            state: DPF state dict (keys: rho, velocity, B, etc.)
            time: Simulation time [s].
        """
        if not self.enable:
            return

        state_copy = {k: v.copy() for k, v in state.items() if isinstance(v, np.ndarray)}
        self._buffer.append(state_copy)
        self._times.append(time)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered states to Well HDF5 via the full exporter."""
        if not self.enable or not self._buffer:
            return

        # Infer grid shape from first snapshot
        ref = self._buffer[0]
        rho = ref.get("rho")
        if rho is None:
            logger.warning("No 'rho' field in state — cannot infer grid shape for Well export")
            self._buffer.clear()
            self._times.clear()
            return

        grid_shape = rho.shape
        # Pad to 3D if needed
        while len(grid_shape) < 3:
            grid_shape = (*grid_shape, 1)

        filename = f"{self.filename_prefix}_{self._batch_count:04d}.h5"
        path = self.output_dir / filename

        exporter = _FullWellExporter(
            output_path=path,
            grid_shape=grid_shape,
            dx=1.0,  # placeholder — engine doesn't pass dx to exporter
        )

        for state_copy, t in zip(self._buffer, self._times, strict=True):
            exporter.add_snapshot(state_copy, time=t)

        try:
            exporter.finalize()
            logger.info("Saved %d steps to %s (Well format)", len(self._buffer), path)
        except ImportError:
            logger.warning("h5py not available — skipping Well export")

        self._buffer.clear()
        self._times.clear()
        self._batch_count += 1

    def close(self) -> None:
        """Flush remaining data."""
        self.flush()
