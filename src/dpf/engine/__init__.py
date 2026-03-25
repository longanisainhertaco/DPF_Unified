"""DPF simulation engine package.

Backward-compatible re-export: ``from dpf.engine import SimulationEngine``
continues to work after the monolithic engine.py was decomposed into
focused sub-modules.
"""

from dpf.engine.core import SimulationEngine

__all__ = ["SimulationEngine"]
