"""AthenaK (Kokkos) MHD solver wrapper for DPF Unified.

Provides a subprocess-based interface to AthenaK — the Kokkos rewrite of
Athena++.  AthenaK uses CMake + Kokkos for performance portability and
supports Serial, OpenMP, CUDA, HIP, and SYCL backends.

Architecture::

    SimulationConfig  -->  athenak_config.py  -->  athinput (text)
                                                       |
    AthenaKSolver  <--  athenak_solver.py  <--  subprocess(athenak)
         |                                          |
    PlasmaSolverBase                          VTK output (binary)
                                                   |
                                             athenak_io.py (reader)

Build AthenaK before use::

    bash scripts/setup_athenak.sh   # clone + init submodules
    bash scripts/build_athenak.sh   # compile (auto-detects OpenMP)

Example::

    from dpf.athenak_wrapper import AthenaKSolver, is_available

    if is_available():
        solver = AthenaKSolver(config)
        state = solver.step(state, dt, current, voltage)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if AthenaK binary is available
_BINARY_PATHS = [
    Path(__file__).parents[3] / "external" / "athenak" / "bin" / "athenak",
    Path(__file__).parents[3] / "external" / "athenak" / "build_omp" / "src" / "athena",
    Path(__file__).parents[3] / "external" / "athenak" / "build_serial" / "src" / "athena",
]

_AVAILABLE = False
_BINARY_PATH: str | None = None

for _p in _BINARY_PATHS:
    if _p.is_file() and os.access(str(_p), os.X_OK):
        _BINARY_PATH = str(_p)
        _AVAILABLE = True
        logger.info("AthenaK binary found: %s", _BINARY_PATH)
        break

if not _AVAILABLE:
    logger.info("AthenaK binary not found — run scripts/build_athenak.sh")


def is_available() -> bool:
    """Return True if a compiled AthenaK binary is found and executable."""
    return _AVAILABLE


def get_binary_path() -> str | None:
    """Return path to the AthenaK binary, or None if not available."""
    return _BINARY_PATH


# Public API — always importable (graceful fallback)
from dpf.athenak_wrapper.athenak_config import generate_athenak_input  # noqa: E402
from dpf.athenak_wrapper.athenak_io import read_vtk_file  # noqa: E402
from dpf.athenak_wrapper.athenak_solver import AthenaKSolver  # noqa: E402

__all__ = [
    "AthenaKSolver",
    "generate_athenak_input",
    "get_binary_path",
    "is_available",
    "read_vtk_file",
]
