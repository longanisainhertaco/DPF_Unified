"""Core abstract base classes and shared data structures.

Defines the interface contracts that all physics modules implement:
- ``CouplingState`` — plasma-circuit coupling dataclass
- ``PlasmaSolverBase`` — ABC for fluid/PIC plasma solvers
- ``CircuitSolverBase`` — ABC for circuit models
- ``DiagnosticsBase`` — ABC for diagnostics recorders
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    """Result of a single simulation timestep.

    Attributes:
        time: Simulation time after this step [s].
        step: Step number after this step.
        dt: Timestep size used [s].
        current: Circuit current [A].
        voltage: Capacitor voltage [V].
        energy_conservation: E_total / E_initial.
        max_Te: Peak electron temperature [K].
        max_rho: Peak density [kg/m^3].
        Z_bar: Average ionization state.
        R_plasma: Plasma resistance [Ohm].
        eta_anomalous: Anomalous resistivity [Ohm*m].
        total_radiated_energy: Cumulative radiated energy [J].
        neutron_rate: Instantaneous DD neutron production rate [1/s].
        total_neutron_yield: Cumulative DD neutron yield (dimensionless count).
        finished: True when sim_time reached or max_steps exceeded.
    """

    time: float = 0.0
    step: int = 0
    dt: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    energy_conservation: float = 1.0
    max_Te: float = 0.0
    max_rho: float = 0.0
    Z_bar: float = 1.0
    R_plasma: float = 0.0
    eta_anomalous: float = 0.0
    total_radiated_energy: float = 0.0
    neutron_rate: float = 0.0
    total_neutron_yield: float = 0.0
    finished: bool = False


@dataclass
class CouplingState:
    """Data exchanged between plasma solver and circuit solver each timestep.

    Attributes:
        Lp: Plasma inductance [H].
        emf: Electromotive force from plasma [V].
        current: Circuit current [A].
        voltage: Capacitor voltage [V].
        dL_dt: Rate of change of plasma inductance [H/s].
        R_plasma: Plasma resistance [Ohm] (from Spitzer resistivity).
        Z_bar: Average ionization state (1.0 for fully ionized H).
    """

    Lp: float = 0.0
    emf: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    dL_dt: float = 0.0
    R_plasma: float = 0.0
    Z_bar: float = 1.0


class PlasmaSolverBase(ABC):
    """Abstract base for all plasma solvers (fluid, PIC, hybrid)."""

    @abstractmethod
    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
    ) -> dict[str, np.ndarray]:
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
        state: dict[str, Any],
        time: float,
    ) -> None:
        """Record diagnostic quantities at the current timestep.

        Args:
            state: Simulation state dictionary.
            time: Current simulation time [s].
        """

    def finalize(self) -> None:
        """Clean up resources (close files, flush buffers)."""
