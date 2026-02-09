"""Apple Silicon device detection and management.

This module provides hardware capability detection for M-series chips,
including MPS (Metal Performance Shaders), MLX (Apple's ML framework),
and Accelerate BLAS integration.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Singleton instance
_device_manager: DeviceManager | None = None


class DeviceManager:
    """Detects and manages Apple Silicon hardware capabilities.

    Provides methods to detect MPS, MLX, Accelerate BLAS, and query
    hardware specifications (GPU cores, unified memory, chip name).
    """

    def __init__(self) -> None:
        """Initialize device manager."""
        self._cache: dict[str, Any] = {}
        logger.debug("DeviceManager initialized")

    def detect_mps(self) -> bool:
        """Check if Metal Performance Shaders (MPS) backend is available.

        Returns
        -------
        bool
            True if PyTorch MPS backend is built and available on this system.

        Notes
        -----
        MPS provides GPU acceleration for PyTorch on Apple Silicon.
        Requires torch to be installed with MPS support.
        """
        if "mps" in self._cache:
            return self._cache["mps"]

        try:
            import torch
            available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            self._cache["mps"] = available
            logger.debug(f"MPS detection: {available}")
            return available
        except (ImportError, AttributeError) as e:
            logger.debug(f"MPS detection failed: {e}")
            self._cache["mps"] = False
            return False

    def detect_mlx(self) -> bool:
        """Check if MLX (Apple ML framework) is available.

        Returns
        -------
        bool
            True if MLX can be imported successfully.

        Notes
        -----
        MLX is Apple's native ML framework optimized for unified memory architecture.
        Often faster than PyTorch MPS for inference on Apple Silicon.
        """
        if "mlx" in self._cache:
            return self._cache["mlx"]

        try:
            import mlx.core  # noqa: F401
            self._cache["mlx"] = True
            logger.debug("MLX detection: True")
            return True
        except ImportError as e:
            logger.debug(f"MLX detection failed: {e}")
            self._cache["mlx"] = False
            return False

    def detect_accelerate(self) -> bool:
        """Check if NumPy is using Apple Accelerate BLAS.

        Returns
        -------
        bool
            True if NumPy BLAS backend is Apple Accelerate.

        Notes
        -----
        Accelerate provides optimized linear algebra for Apple Silicon.
        Parsing `np.show_config()` output to detect backend.
        """
        if "accelerate" in self._cache:
            return self._cache["accelerate"]

        try:
            import contextlib
            import io

            import numpy as np
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            config_str = buf.getvalue()
            # Check for Accelerate in BLAS/LAPACK info
            has_accelerate = (
                "accelerate" in config_str.lower() or "veclib" in config_str.lower()
            )
            self._cache["accelerate"] = has_accelerate
            logger.debug(f"Accelerate BLAS detection: {has_accelerate}")
            return has_accelerate
        except Exception as e:
            logger.debug(f"Accelerate detection failed: {e}")
            self._cache["accelerate"] = False
            return False

    def _sysctl_query(self, key: str) -> str | None:
        """Query macOS sysctl for hardware info.

        Parameters
        ----------
        key : str
            Sysctl key to query (e.g., 'hw.memsize').

        Returns
        -------
        str | None
            Value from sysctl, or None if query fails.
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"sysctl query {key} failed: {e}")
            return None

    def get_gpu_info(self) -> dict[str, Any]:
        """Get comprehensive GPU and system information.

        Returns
        -------
        dict
            Dictionary containing:
            - gpu_cores: int, number of GPU cores (0 if unknown)
            - memory_gb: float, total unified memory in GB
            - chip_name: str, CPU brand string (e.g., "Apple M3 Pro")
            - mps_available: bool, MPS backend available
            - mlx_available: bool, MLX framework available
            - accelerate_blas: bool, NumPy using Accelerate

        Notes
        -----
        GPU core count is queried from sysctl hw.perflevel0.physicalcpu.
        This is a proxy for performance cores; actual GPU core count
        requires more complex detection.
        """
        info: dict[str, Any] = {
            "gpu_cores": 0,
            "memory_gb": 0.0,
            "chip_name": "Unknown",
            "mps_available": self.detect_mps(),
            "mlx_available": self.detect_mlx(),
            "accelerate_blas": self.detect_accelerate(),
        }

        # Get chip name
        chip_name = self._sysctl_query("machdep.cpu.brand_string")
        if chip_name:
            info["chip_name"] = chip_name

        # Get total memory
        memsize_str = self._sysctl_query("hw.memsize")
        if memsize_str:
            try:
                memsize_bytes = int(memsize_str)
                info["memory_gb"] = memsize_bytes / (1024**3)
            except ValueError:
                pass

        # Get performance core count (proxy for GPU capability)
        perf_cores_str = self._sysctl_query("hw.perflevel0.physicalcpu")
        if perf_cores_str:
            with contextlib.suppress(ValueError):
                info["gpu_cores"] = int(perf_cores_str)

        # Fallback: try to get total physical CPU count
        if info["gpu_cores"] == 0:
            physicalcpu_str = self._sysctl_query("hw.physicalcpu")
            if physicalcpu_str:
                with contextlib.suppress(ValueError):
                    info["gpu_cores"] = int(physicalcpu_str)

        logger.debug(f"GPU info: {info}")
        return info

    def select_best_device(self) -> str:
        """Select the best available compute device.

        Returns
        -------
        str
            Device string: "mlx" if available, else "mps" if available, else "cpu".

        Notes
        -----
        Priority order:
        1. MLX (fastest for Apple Silicon inference)
        2. MPS (PyTorch GPU acceleration)
        3. CPU (fallback)
        """
        if self.detect_mlx():
            logger.info("Selected device: mlx")
            return "mlx"
        if self.detect_mps():
            logger.info("Selected device: mps")
            return "mps"
        logger.info("Selected device: cpu")
        return "cpu"

    def memory_pressure(self) -> float:
        """Get current unified memory pressure.

        Returns
        -------
        float
            Memory usage fraction (0.0 = no usage, 1.0 = full).

        Notes
        -----
        Uses psutil if available, falls back to os.sysconf.
        Returns 0.0 if detection fails.
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            pressure = mem.percent / 100.0
            logger.debug(f"Memory pressure (psutil): {pressure:.2%}")
            return pressure
        except ImportError:
            # Fallback to sysctl-based detection
            try:
                memsize_str = self._sysctl_query("hw.memsize")
                if not memsize_str:
                    return 0.0

                total_bytes = int(memsize_str)

                # Try to get available memory via vm_stat
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )

                if result.returncode == 0:
                    # Parse vm_stat output for free pages
                    for line in result.stdout.splitlines():
                        if "Pages free" in line:
                            free_pages_str = line.split(":")[1].strip().rstrip(".")
                            free_pages = int(free_pages_str)
                            page_size = os.sysconf("SC_PAGESIZE")
                            free_bytes = free_pages * page_size
                            used_bytes = total_bytes - free_bytes
                            pressure = used_bytes / total_bytes
                            logger.debug(f"Memory pressure (vm_stat): {pressure:.2%}")
                            return max(0.0, min(1.0, pressure))

                return 0.0
            except Exception as e:
                logger.debug(f"Memory pressure detection failed: {e}")
                return 0.0

    def summary(self) -> str:
        """Generate human-readable summary of device capabilities.

        Returns
        -------
        str
            Multi-line summary string with all detected capabilities.
        """
        info = self.get_gpu_info()
        pressure = self.memory_pressure()
        best_device = self.select_best_device()

        lines = [
            "Apple Silicon Device Summary",
            "=" * 50,
            f"Chip: {info['chip_name']}",
            f"Unified Memory: {info['memory_gb']:.1f} GB ({pressure:.1%} in use)",
            f"Performance Cores: {info['gpu_cores']}",
            "",
            "Compute Backends:",
            f"  MPS (PyTorch GPU): {'✓ Available' if info['mps_available'] else '✗ Not available'}",
            f"  MLX (Apple ML): {'✓ Available' if info['mlx_available'] else '✗ Not available'}",
            f"  Accelerate BLAS: {'✓ Active' if info['accelerate_blas'] else '✗ Not active'}",
            "",
            f"Recommended Device: {best_device}",
        ]

        return "\n".join(lines)


def get_device_manager() -> DeviceManager:
    """Get singleton DeviceManager instance.

    Returns
    -------
    DeviceManager
        Cached singleton instance.

    Notes
    -----
    Ensures only one DeviceManager exists per process to avoid
    redundant hardware queries.
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
