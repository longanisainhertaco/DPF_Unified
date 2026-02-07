"""Binary field encoding for WebSocket transfer.

Converts numpy field arrays to compact float32 binary with optional
spatial downsampling.  The wire format is:

    1. JSON text frame: FieldHeader with per-field shape, dtype, byte offset
    2. Binary frame: concatenated float32 arrays in declared order

A 32^3 grid with 5 scalar fields is only ~640 KB at float32,
well within comfortable WebSocket bandwidth at 10 Hz on localhost.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def downsample_field(arr: np.ndarray, factor: int) -> np.ndarray:
    """Spatially downsample a field array by *factor* in each spatial dimension.

    Args:
        arr: Field array, either (nx, ny, nz) or (3, nx, ny, nz) for vector fields.
        factor: Downsample factor (1 = no change).

    Returns:
        Downsampled array in float32.
    """
    if factor <= 1:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        return arr[::factor, ::factor, ::factor].astype(np.float32)
    elif arr.ndim == 4:
        # Vector field: (components, nx, ny, nz)
        return arr[:, ::factor, ::factor, ::factor].astype(np.float32)
    else:
        # Unknown layout — return as-is
        return arr.astype(np.float32)


def encode_fields(
    snapshot: dict[str, np.ndarray],
    field_names: list[str],
    downsample: int = 1,
) -> tuple[dict[str, dict[str, Any]], bytes]:
    """Encode requested fields into a binary blob + metadata header.

    Args:
        snapshot: Full field snapshot from engine.get_field_snapshot().
        field_names: Which fields to include (e.g. ["rho", "Te"]).
        downsample: Spatial downsampling factor (1 = full resolution).

    Returns:
        (header_dict, binary_blob) where header_dict maps field name ->
        {shape, dtype, offset, nbytes} and binary_blob is the concatenated
        float32 data.
    """
    header: dict[str, dict[str, Any]] = {}
    chunks: list[bytes] = []
    offset = 0

    for name in field_names:
        arr = snapshot.get(name)
        if arr is None:
            continue
        arr32 = downsample_field(arr, downsample)
        raw = arr32.tobytes()
        header[name] = {
            "shape": list(arr32.shape),
            "dtype": "float32",
            "offset": offset,
            "nbytes": len(raw),
        }
        chunks.append(raw)
        offset += len(raw)

    blob = b"".join(chunks)
    return header, blob
