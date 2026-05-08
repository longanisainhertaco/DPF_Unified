"""MLX device detection, dtype helpers, and stream management."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any

_VALID_PRECISIONS = ("float32", "float16", "bfloat16")
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _safe_mlx_available() -> bool:
    """Return whether ``mlx.core`` can be imported without risking this process."""
    if os.environ.get("DPF_DISABLE_MLX", "").strip().lower() in _TRUE_VALUES:
        return False
    if os.environ.get("DPF_MLX_ASSUME_AVAILABLE", "").strip().lower() in _TRUE_VALUES:
        return True

    try:
        timeout = float(os.environ.get("DPF_MLX_PROBE_TIMEOUT", "10"))
    except ValueError:
        timeout = 10.0

    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import mlx.core"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


# Module-level availability flag — set at import time, never changes.
# The probe intentionally runs in a child process because some broken MLX
# installations abort the interpreter instead of raising ImportError.
HAS_MLX: bool = _safe_mlx_available()


def require_mlx():
    """Import and return mlx.core, raising ImportError with install instructions if unavailable.

    Returns
    -------
    module
        The ``mlx.core`` module.

    Raises
    ------
    ImportError
        When MLX is not installed, with pip install instructions.
    """
    if not HAS_MLX:
        raise ImportError(
            "MLX is required for the Metal v2 solver but is not available.\n"
            "Install with:  pip install mlx\n"
            "Requires macOS 13.3+ on Apple Silicon."
        )

    try:
        return importlib.import_module("mlx.core")
    except ImportError as exc:
        raise ImportError(
            "MLX is required for the Metal v2 solver but is not installed.\n"
            "Install with:  pip install mlx\n"
            "Requires macOS 13.3+ on Apple Silicon."
        ) from exc


def mlx_dtype(precision: str = "float32") -> Any:
    """Map a precision string to the corresponding ``mx.Dtype``.

    Parameters
    ----------
    precision:
        One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.
        On GPU, ``"float32"`` is the standard compute type.
        ``"float16"`` / ``"bfloat16"`` are suitable for inference workloads.

    Returns
    -------
    mx.Dtype
        The matching MLX dtype object.

    Raises
    ------
    ImportError
        If MLX is not installed.
    ValueError
        If *precision* is not a recognised string.
    """
    mx = require_mlx()

    mapping: dict[str, Any] = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }

    if precision not in mapping:
        raise ValueError(
            f"Unknown precision {precision!r}. "
            f"Expected one of: {', '.join(_VALID_PRECISIONS)}"
        )
    return mapping[precision]


def mlx_default_stream() -> Any | None:
    """Return the default Metal GPU stream.

    On non-Apple hardware (or when MLX is unavailable) returns ``None``.

    Returns
    -------
    mx.Stream or None
        The default GPU stream, or ``None`` if Metal is not available.
    """
    if not HAS_MLX:
        return None

    mx = require_mlx()
    device = mx.default_device()
    if device.type == mx.gpu:
        return mx.default_stream(device)
    return None


def mlx_device_info() -> dict[str, Any]:
    """Return a snapshot of the MLX / Metal environment.

    Returns
    -------
    dict
        Keys:

        - ``has_mlx`` (bool): Whether MLX is importable.
        - ``mlx_version`` (str): MLX version string, or ``"unavailable"``.
        - ``metal_available`` (bool): Whether the default device is a GPU.
        - ``device_name`` (str): String representation of the default device,
          or ``"cpu"`` when Metal is absent.
    """
    if not HAS_MLX:
        return {
            "has_mlx": False,
            "mlx_version": "unavailable",
            "metal_available": False,
            "device_name": "cpu",
        }

    mx = require_mlx()

    try:
        version: str = mx.__version__
    except AttributeError:
        version = "unknown"

    device = mx.default_device()
    metal_available = device.type == mx.gpu

    return {
        "has_mlx": True,
        "mlx_version": version,
        "metal_available": metal_available,
        "device_name": str(device),
    }
