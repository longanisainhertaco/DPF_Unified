"""Core abstract base classes and shared data structures.

Defines the interface contracts that all physics modules implement:
- ``CouplingState`` — plasma-circuit coupling dataclass
- ``PlasmaSolverBase`` — ABC for fluid/PIC plasma solvers
- ``CircuitSolverBase`` — ABC for circuit models
- ``DiagnosticsBase`` — ABC for diagnostics recorders
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class CouplingState:
    """Data exchanged between plasma solver and circuit solver each timestep.

    Attributes:
        Lp: Plasma inductance [H].
        emf: Electromotive force from plasma [V].
        current: Circuit current [A].
        voltage: Capacitor voltage [V].
        mutual_inductance: Mutual inductance [H].
        back_reaction: Back-reaction term on circuit from plasma dynamics.
        dL_dt: Rate of change of plasma inductance [H/s].
    """

    Lp: float = 0.0
    emf: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    mutual_inductance: float = 0.0
    back_reaction: float = 0.0
    dL_dt: float = 0.0


class PlasmaSolverBase(ABC):
    """Abstract base for all plasma solvers (fluid, PIC, hybrid)."""

    @abstractmethod
    def step(
        self,
        state: Dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
    ) -> Dict[str, np.ndarray]:
        """Advance the plasma state by one timestep.

        Args:
            state: Dictionary of field arrays (density, velocity, pressure, B, ...).
            dt: Timestep size [s].
            current: Circuit current [A] for boundary forcing.
            voltage: Capacitor voltage [V].

        Returns:
            Updated state dictionary.
        """

    def coupling_interface(self) -> CouplingState:
        """Return coupling quantities for the circuit solver."""
        return CouplingState()


class CircuitSolverBase(ABC):
    """Abstract base for circuit models."""

    @abstractmethod
    def step(
        self,
        coupling: CouplingState,
        back_emf: float,
        dt: float,
    ) -> CouplingState:
        """Advance the circuit by one timestep.

        Args:
            coupling: Current coupling state from plasma.
            back_emf: Back-EMF from plasma motion [V].
            dt: Timestep size [s].

        Returns:
            Updated coupling state.
        """


class DiagnosticsBase(ABC):
    """Abstract base for diagnostics recorders."""

    @abstractmethod
    def record(
        self,
        state: Dict[str, Any],
        time: float,
    ) -> None:
        """Record diagnostic quantities at the current timestep.

        Args:
            state: Simulation state dictionary.
            time: Current simulation time [s].
        """

    def finalize(self) -> None:
        """Clean up resources (close files, flush buffers)."""
