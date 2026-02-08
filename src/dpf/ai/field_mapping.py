"""DPF state dict <-> Well format field mappings.

Defines constants for field names, shapes, units, and tensor order,
plus utility functions for converting between DPF and Well layouts.

The Well format (used by WALRUS) expects:
    t0_fields: scalar fields, shape (n_traj, n_steps, *spatial)
    t1_fields: vector fields, shape (n_traj, n_steps, *spatial, D)

DPF state dict uses:
    Scalars: (nx, ny, nz)
    Vectors: (3, nx, ny, nz) — component-first

References:
    The Well data format: https://github.com/PolymathicAI/the_well
"""

from __future__ import annotations

import numpy as np

# ── Field name mappings ──────────────────────────────────────────

SCALAR_FIELDS: dict[str, str] = {
    "rho": "density",
    "Te": "electron_temp",
    "Ti": "ion_temp",
    "pressure": "pressure",
    "psi": "dedner_psi",
}

VECTOR_FIELDS: dict[str, str] = {
    "B": "magnetic_field",
    "velocity": "velocity",
}

DPF_TO_WELL_NAMES: dict[str, str] = {**SCALAR_FIELDS, **VECTOR_FIELDS}
WELL_TO_DPF_NAMES: dict[str, str] = {v: k for k, v in DPF_TO_WELL_NAMES.items()}

# ── Units ────────────────────────────────────────────────────────

FIELD_UNITS: dict[str, str] = {
    "rho": "kg/m^3",
    "Te": "K",
    "Ti": "K",
    "pressure": "Pa",
    "psi": "T*m/s",
    "B": "T",
    "velocity": "m/s",
}

# ── Circuit scalar fields (time-varying, non-spatial) ────────────

CIRCUIT_SCALARS: tuple[str, ...] = (
    "current",
    "voltage",
    "energy_conservation",
    "R_plasma",
    "Z_bar",
    "total_radiated_energy",
    "neutron_rate",
    "total_neutron_yield",
)

# ── Required state dict keys ─────────────────────────────────────

REQUIRED_STATE_KEYS: tuple[str, ...] = (
    "rho", "velocity", "pressure", "B", "Te", "Ti", "psi",
)


# ── Conversion functions ─────────────────────────────────────────


def dpf_scalar_to_well(
    field: np.ndarray,
    traj_idx: int = 0,
    step_idx: int = 0,
    n_traj: int = 1,
    n_steps: int = 1,
) -> np.ndarray:
    """Reshape DPF scalar field to Well layout.

    Args:
        field: DPF scalar array of shape (nx, ny, nz).
        traj_idx: Trajectory index in output array.
        step_idx: Timestep index in output array.
        n_traj: Total number of trajectories.
        n_steps: Total number of timesteps.

    Returns:
        Well-format array of shape (n_traj, n_steps, nx, ny, nz), float32.
    """
    spatial = field.shape
    out = np.zeros((n_traj, n_steps, *spatial), dtype=np.float32)
    out[traj_idx, step_idx] = field.astype(np.float32)
    return out


def dpf_vector_to_well(
    field: np.ndarray,
    traj_idx: int = 0,
    step_idx: int = 0,
    n_traj: int = 1,
    n_steps: int = 1,
) -> np.ndarray:
    """Reshape DPF vector field to Well layout.

    Transposes from component-first (3, nx, ny, nz) to
    component-last (n_traj, n_steps, nx, ny, nz, 3).

    Args:
        field: DPF vector array of shape (3, nx, ny, nz).
        traj_idx: Trajectory index in output array.
        step_idx: Timestep index in output array.
        n_traj: Total number of trajectories.
        n_steps: Total number of timesteps.

    Returns:
        Well-format array of shape (n_traj, n_steps, nx, ny, nz, 3), float32.
    """
    # (3, nx, ny, nz) -> (nx, ny, nz, 3)
    spatial_last = np.moveaxis(field, 0, -1).astype(np.float32)
    spatial = spatial_last.shape  # (nx, ny, nz, 3)
    out = np.zeros((n_traj, n_steps, *spatial), dtype=np.float32)
    out[traj_idx, step_idx] = spatial_last
    return out


def well_scalar_to_dpf(
    field: np.ndarray,
    traj_idx: int = 0,
    step_idx: int = 0,
) -> np.ndarray:
    """Extract DPF scalar from Well layout.

    Args:
        field: Well-format array of shape (n_traj, n_steps, nx, ny, nz).
        traj_idx: Trajectory index to extract.
        step_idx: Timestep index to extract.

    Returns:
        DPF scalar array of shape (nx, ny, nz), float64.
    """
    return field[traj_idx, step_idx].astype(np.float64)


def well_vector_to_dpf(
    field: np.ndarray,
    traj_idx: int = 0,
    step_idx: int = 0,
) -> np.ndarray:
    """Extract DPF vector from Well layout.

    Transposes from component-last (..., nx, ny, nz, 3) to
    component-first (3, nx, ny, nz).

    Args:
        field: Well-format array of shape (n_traj, n_steps, nx, ny, nz, 3).
        traj_idx: Trajectory index to extract.
        step_idx: Timestep index to extract.

    Returns:
        DPF vector array of shape (3, nx, ny, nz), float64.
    """
    spatial_last = field[traj_idx, step_idx]  # (nx, ny, nz, 3)
    return np.moveaxis(spatial_last, -1, 0).astype(np.float64)


def infer_geometry(state: dict[str, np.ndarray]) -> str:
    """Infer coordinate geometry from a DPF state dict.

    Args:
        state: DPF state dict with at least one scalar field.

    Returns:
        ``"cylindrical"`` if ny == 1, else ``"cartesian"``.
    """
    shape = spatial_shape(state)
    return "cylindrical" if shape[1] == 1 else "cartesian"


def spatial_shape(state: dict[str, np.ndarray]) -> tuple[int, int, int]:
    """Extract spatial grid shape from a DPF state dict.

    Args:
        state: DPF state dict with at least one scalar field.

    Returns:
        Tuple (nx, ny, nz).

    Raises:
        ValueError: If no suitable field is found.
    """
    for key in ("rho", "Te", "Ti", "pressure", "psi"):
        if key in state and isinstance(state[key], np.ndarray):
            return tuple(state[key].shape)  # type: ignore[return-value]
    raise ValueError("No scalar field found in state dict")


def validate_state_dict(state: dict[str, np.ndarray]) -> list[str]:
    """Validate a DPF state dict for Well export.

    Args:
        state: DPF state dict to validate.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []

    for key in REQUIRED_STATE_KEYS:
        if key not in state:
            errors.append(f"Missing required field: {key}")
            continue
        if not isinstance(state[key], np.ndarray):
            errors.append(f"Field '{key}' is not a numpy array")
            continue

    # Check shape consistency
    scalar_shapes = set()
    for key in SCALAR_FIELDS:
        if key in state and isinstance(state[key], np.ndarray):
            scalar_shapes.add(state[key].shape)
    if len(scalar_shapes) > 1:
        errors.append(f"Inconsistent scalar field shapes: {scalar_shapes}")

    # Check vector shapes
    for key in VECTOR_FIELDS:
        if key in state and isinstance(state[key], np.ndarray):
            arr = state[key]
            if arr.ndim != 4 or arr.shape[0] != 3:
                errors.append(
                    f"Vector field '{key}' has shape {arr.shape}, "
                    f"expected (3, nx, ny, nz)"
                )

    return errors
