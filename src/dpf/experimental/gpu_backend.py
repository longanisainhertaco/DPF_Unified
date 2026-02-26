"""GPU acceleration backend: CuPy when available, NumPy fallback.

.. deprecated::
    **This module is dead code on Apple Silicon (the DPF target platform).**
    CuPy requires CUDA, which is not available on M-series Macs.  The
    production GPU path uses PyTorch MPS via
    :class:`dpf.metal.metal_solver.MetalMHDSolver`.  This module is
    retained only for Linux/CUDA environments and for historical reference.

Provides a unified array interface ``xp`` that is either CuPy (GPU) or
NumPy (CPU), allowing the same code to run on both without modification.

Usage::

    from dpf.fluid.gpu_backend import xp, is_gpu_available, to_numpy, to_device

    # Create arrays on the active device
    a = xp.zeros((100, 100))
    b = xp.ones((100, 100))
    c = a + b  # GPU if available, CPU otherwise

    # Transfer back to CPU for I/O
    c_cpu = to_numpy(c)
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.warn(
    "dpf.experimental.gpu_backend is a CUDA/CuPy backend and is not functional "
    "on Apple Silicon.  Use dpf.metal.metal_solver.MetalMHDSolver for GPU "
    "acceleration on M-series Macs.",
    stacklevel=2,
    category=DeprecationWarning,
)

try:
    import cupy as cp  # type: ignore[import-untyped]

    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False


def is_gpu_available() -> bool:
    """Return ``True`` if CuPy is installed and a CUDA GPU is accessible."""
    return _GPU_AVAILABLE


# Unified array module: ``cp`` on GPU, ``np`` on CPU.
xp = cp if _GPU_AVAILABLE else np


def to_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert an array to a NumPy (CPU) array.

    If *arr* is already a NumPy array this is a no-op (returns
    ``np.asarray(arr)``).  If *arr* is a CuPy array it is transferred
    from device memory to host memory.

    Args:
        arr: Array to convert.

    Returns:
        NumPy ndarray on the host.
    """
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr: np.ndarray) -> np.ndarray:
    """Move an array to the active compute device.

    If a GPU is available the array is transferred to device memory via
    ``cupy.asarray``.  Otherwise it is returned as a NumPy array on the
    CPU.

    Args:
        arr: Array to transfer.

    Returns:
        Array on the active device (CuPy ndarray or NumPy ndarray).
    """
    if _GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)


def synchronize() -> None:
    """Synchronize the default CUDA stream.

    This is a no-op when running on the CPU.  On GPU it blocks until all
    previously enqueued CUDA operations have completed, which is useful
    for accurate timing measurements or before host-side I/O.
    """
    if _GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()


def get_device_info() -> dict[str, str | int]:
    """Return a dictionary describing the active compute device.

    Keys:
        type: ``'gpu'`` or ``'cpu'``.
        name: Human-readable device name.
        memory_total: Total device memory in bytes (0 for CPU).
        memory_free: Free device memory in bytes (0 for CPU).
    """
    if _GPU_AVAILABLE:
        dev = cp.cuda.Device()
        mem_free, mem_total = dev.mem_info
        # Try to get a human-readable device name; fall back gracefully
        try:
            name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"]
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:  # noqa: BLE001
            name = "CUDA GPU"
        return {
            "type": "gpu",
            "name": name,
            "memory_total": mem_total,
            "memory_free": mem_free,
        }
    return {
        "type": "cpu",
        "name": "NumPy",
        "memory_total": 0,
        "memory_free": 0,
    }
