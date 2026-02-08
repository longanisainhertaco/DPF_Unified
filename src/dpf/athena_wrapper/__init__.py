"""Athena++ MHD solver wrapper for DPF Unified.

Provides a Python interface to the Princeton Athena++ C++ MHD code
via pybind11 bindings. The :class:`AthenaPPSolver` class implements
:class:`~dpf.core.bases.PlasmaSolverBase` so it can be used as a
drop-in replacement for the Python MHD solver.

Architecture::

    SimulationConfig  -->  athena_config.py  -->  athinput (text)
                                                      |
    AthenaPPSolver  <--  athena_engine.py  <--  _athena_core (C++)
         |                                          |
    PlasmaSolverBase                        pybind11 bindings
                                                    |
                                             Athena++ Mesh/MeshBlock

The C++ extension module ``_athena_core`` must be built before use::

    cd src/dpf/athena_wrapper/cpp
    mkdir build && cd build
    cmake .. -DATHENA_ROOT=<path-to-external/athena>
    make -j8

Example::

    from dpf.athena_wrapper import AthenaPPSolver, is_available

    if is_available():
        solver = AthenaPPSolver(config)
        state = solver.step(state, dt, current, voltage)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Attempt to import the compiled C++ extension
_athena_core = None
_AVAILABLE = False

try:
    import importlib
    _athena_core = importlib.import_module("dpf.athena_wrapper._athena_core")
    _AVAILABLE = True
    logger.info("Athena++ C++ backend loaded successfully")
except ImportError as exc:
    logger.info("Athena++ C++ backend not available: %s", exc)
    _AVAILABLE = False

# Public API — always importable (graceful fallback)
from dpf.athena_wrapper.athena_config import generate_athinput  # noqa: E402
from dpf.athena_wrapper.athena_engine import AthenaPPSolver  # noqa: E402


def is_available() -> bool:
    """Return True if the Athena++ C++ extension is compiled and importable."""
    return _AVAILABLE


__all__ = [
    "AthenaPPSolver",
    "generate_athinput",
    "is_available",
]
