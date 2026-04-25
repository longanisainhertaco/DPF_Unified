"""Telemetry-aware numerical floors for MHD solver.

Replaces silent np.maximum(rho, FLOOR) with tracked apply_floor() that logs
every activation. Enables diagnosis of whether floors are masking physics
errors vs protecting legitimate vacuum regions.

Usage:
    from dpf.metal.floor_telemetry import apply_floor, report

    rho = apply_floor(rho, 1e-12, "rho", step)  # instead of np.maximum
    p = apply_floor(p, 1e-12, "pressure", step)

    report()  # prints summary at end of simulation
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

_counts: dict[str, int] = defaultdict(int)
_max_corrections: dict[str, float] = defaultdict(float)
_total_corrections: dict[str, float] = defaultdict(float)


def apply_floor(
    arr: np.ndarray,
    floor_val: float,
    name: str,
    step: int = -1,
) -> np.ndarray:
    """Apply numerical floor with telemetry tracking.

    Args:
        arr: Array to floor.
        floor_val: Minimum allowed value.
        name: Human-readable name for logging.
        step: Current timestep (for logging).

    Returns:
        Floored array (same shape).
    """
    below = arr < floor_val
    n_below = int(np.sum(below))
    if n_below > 0:
        _counts[name] += n_below
        correction = float(np.max(floor_val - arr[below]))
        _max_corrections[name] = max(_max_corrections[name], correction)
        _total_corrections[name] += float(np.sum(floor_val - arr[below]))
    return np.maximum(arr, floor_val)  # no-floor-check


def report() -> dict[str, dict]:
    """Print and return floor activation summary."""
    if not _counts:
        logger.info("Floor telemetry: no activations.")
        return {}

    results = {}
    for name in sorted(_counts):
        info = {
            "activations": _counts[name],
            "max_correction": _max_corrections[name],
            "total_correction": _total_corrections[name],
        }
        results[name] = info
        logger.warning(
            "Floor '%s': %d activations, max_correction=%.3e, total=%.3e",
            name, info["activations"], info["max_correction"], info["total_correction"],
        )
    return results


def reset() -> None:
    """Clear all telemetry counters."""
    _counts.clear()
    _max_corrections.clear()
    _total_corrections.clear()
