"""Checkpoint/restart support for DPF simulations.

Saves and loads full simulation state (fluid arrays, circuit state,
time, step count) to HDF5 files for restart capability.

Usage:
    # Save checkpoint
    save_checkpoint("checkpoint.h5", state, circuit_state, time, step_count, config_dict)

    # Load checkpoint
    data = load_checkpoint("checkpoint.h5")
    state = data["state"]
    circuit = data["circuit"]
    time = data["time"]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available; checkpoint/restart disabled")


def save_checkpoint(
    filename: str,
    state: dict[str, np.ndarray],
    circuit_state: dict[str, float],
    time: float,
    step_count: int,
    config_json: str | None = None,
) -> None:
    """Save full simulation state to an HDF5 checkpoint file.

    Args:
        filename: Output HDF5 file path.
        state: Plasma state dictionary with arrays (rho, velocity, B, etc.).
        circuit_state: Circuit scalars (current, voltage, energies).
        time: Current simulation time [s].
        step_count: Current timestep number.
        config_json: JSON string of the simulation config (for reference).
    """
    if not HAS_H5PY:
        logger.warning("Cannot save checkpoint: h5py not installed")
        return

    logger.info("Saving checkpoint to %s at t=%.4e s, step=%d", filename, time, step_count)

    with h5py.File(filename, "w") as f:
        # Metadata
        f.attrs["time"] = time
        f.attrs["step_count"] = step_count
        f.attrs["checkpoint_version"] = 1

        if config_json is not None:
            f.attrs["config_json"] = config_json

        # Plasma state arrays
        grp_state = f.create_group("state")
        for key, arr in state.items():
            if isinstance(arr, np.ndarray):
                grp_state.create_dataset(key, data=arr)

        # Circuit state scalars
        grp_circuit = f.create_group("circuit")
        for key, val in circuit_state.items():
            grp_circuit.attrs[key] = val

    logger.info("Checkpoint saved: %s", filename)


def load_checkpoint(filename: str) -> dict[str, Any]:
    """Load simulation state from an HDF5 checkpoint file.

    Args:
        filename: Input HDF5 file path.

    Returns:
        Dictionary with keys:
            - "state": dict of numpy arrays (plasma state)
            - "circuit": dict of floats (circuit state)
            - "time": float (simulation time)
            - "step_count": int (timestep number)
            - "config_json": str or None (config for reference)
    """
    if not HAS_H5PY:
        raise RuntimeError("Cannot load checkpoint: h5py not installed")

    logger.info("Loading checkpoint from %s", filename)

    with h5py.File(filename, "r") as f:
        time = float(f.attrs["time"])
        step_count = int(f.attrs["step_count"])

        config_json = None
        if "config_json" in f.attrs:
            config_json = str(f.attrs["config_json"])

        # Load plasma state arrays
        state = {}
        for key in f["state"]:
            state[key] = np.array(f["state"][key])

        # Load circuit state scalars
        circuit = {}
        for key in f["circuit"].attrs:
            circuit[key] = float(f["circuit"].attrs[key])

    logger.info(
        "Checkpoint loaded: t=%.4e s, step=%d, state keys=%s",
        time, step_count, list(state.keys()),
    )

    return {
        "state": state,
        "circuit": circuit,
        "time": time,
        "step_count": step_count,
        "config_json": config_json,
    }
