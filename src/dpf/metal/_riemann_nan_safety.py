"""NaN-check safety level management and positivity repair statistics.

Controls CPU-GPU synchronisation frequency for NaN detection and
tracks positivity-fallback repair counts.
"""

from __future__ import annotations

# ============================================================
# Positivity fallback repair diagnostics
# ============================================================

_repair_stats: dict[str, int] = {"total_checked": 0, "total_repaired": 0, "calls": 0}


def get_repair_stats() -> dict[str, int]:
    """Return cumulative repair statistics from positivity fallback.

    Returns
    -------
    dict[str, int]
        Keys: ``total_checked`` (interfaces evaluated), ``total_repaired``
        (interfaces replaced with donor cell), ``calls`` (number of
        ``_positivity_fallback`` invocations).
    """
    return dict(_repair_stats)


def reset_repair_stats() -> None:
    """Reset positivity fallback repair counters to zero."""
    _repair_stats["total_checked"] = 0
    _repair_stats["total_repaired"] = 0
    _repair_stats["calls"] = 0


# ============================================================
# NaN-check safety level (controls CPU-GPU sync frequency)
# ============================================================

_nan_safety: dict[str, object] = {
    "level": "normal",   # "strict" | "normal" | "fast"
    "stride": 10,        # check every N steps in "normal" mode
    "step_count": 0,     # incremented by MetalMHDSolver after each step
}

_NAN_CHECK_STRIDES: dict[str, int] = {
    "strict": 1,    # every step
    "normal": 10,   # every 10th step
    "fast": 0,      # never (0 = disabled)
}


def set_nan_safety_level(level: str) -> None:
    """Set the NaN-checking frequency for the Metal Riemann solver.

    Controls how often CPU-GPU synchronisation barriers are inserted for
    NaN detection.  Lower frequency = higher throughput, less safety.

    Parameters
    ----------
    level : str
        ``"strict"``  — check every step (original behaviour).
        ``"normal"``  — check every 10th step (default; recommended).
        ``"fast"``    — never check (maximum throughput, no NaN recovery).
    """
    if level not in _NAN_CHECK_STRIDES:
        raise ValueError(f"safety_level must be 'strict', 'normal', or 'fast'; got {level!r}")
    _nan_safety["level"] = level
    _nan_safety["stride"] = _NAN_CHECK_STRIDES[level]


def advance_nan_step_count() -> None:
    """Increment the global step counter used for periodic NaN checks."""
    _nan_safety["step_count"] = int(_nan_safety["step_count"]) + 1  # type: ignore[arg-type]


def _should_check_nan() -> bool:
    """Return True if a NaN check (CPU-GPU sync) should run this step."""
    stride = int(_nan_safety["stride"])  # type: ignore[arg-type]
    if stride == 0:
        return False
    if stride == 1:
        return True
    return int(_nan_safety["step_count"]) % stride == 0  # type: ignore[arg-type]
