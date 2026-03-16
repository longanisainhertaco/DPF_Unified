"""Energy balance tracking for MHD simulations.

Tracks the energy budget across all components:
- Magnetic (B^2 / 2mu_0)
- Kinetic (0.5 * rho * v^2)
- Thermal (p / (gamma - 1))
- Radiated (cumulative losses)
- Circuit (0.5 * C * V^2 + 0.5 * L * I^2)

Conservation error = |E_total(t) - E_total(0)| / E_total(0)

Usage:
    tracker = EnergyTracker(gamma=5/3)
    for each step:
        tracker.record(state, circuit, radiated_power, dt, cell_volume)
    report = tracker.conservation_report()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MU_0 = 4.0 * np.pi * 1e-7


@dataclass
class EnergySnapshot:
    """Energy components at a single timestep."""

    t: float
    E_magnetic: float    # B^2/(2*mu_0) integrated [J]
    E_kinetic: float     # 0.5*rho*v^2 integrated [J]
    E_thermal: float     # p/(gamma-1) integrated [J]
    E_radiated: float    # Cumulative radiated energy [J]
    E_circuit: float     # 0.5*C*V^2 + 0.5*L*I^2 [J]
    E_total: float       # Sum of all components


@dataclass
class EnergyReport:
    """Energy conservation report."""

    times: list[float] = field(default_factory=list)
    E_magnetic: list[float] = field(default_factory=list)
    E_kinetic: list[float] = field(default_factory=list)
    E_thermal: list[float] = field(default_factory=list)
    E_radiated: list[float] = field(default_factory=list)
    E_circuit: list[float] = field(default_factory=list)
    E_total: list[float] = field(default_factory=list)
    conservation_error: list[float] = field(default_factory=list)

    @property
    def max_conservation_error(self) -> float:
        return max(self.conservation_error) if self.conservation_error else 0.0

    @property
    def final_conservation_error(self) -> float:
        return self.conservation_error[-1] if self.conservation_error else 0.0

    @property
    def is_conserved(self) -> bool:
        """True if max error < 5% (acceptable for explicit MHD)."""
        return self.max_conservation_error < 0.05


class EnergyTracker:
    """Track energy balance during MHD simulation.

    Args:
        gamma: Adiabatic index (default 5/3).
    """

    def __init__(self, gamma: float = 5.0 / 3.0) -> None:
        self.gamma = gamma
        self._E0: float | None = None
        self._cumulative_radiated = 0.0
        self._report = EnergyReport()

    def compute_energies(
        self,
        state: dict[str, np.ndarray],
        cell_volume: float,
        C: float = 0.0,
        V_cap: float = 0.0,
        L_total: float = 0.0,
        I_current: float = 0.0,
    ) -> EnergySnapshot:
        """Compute all energy components from current state.

        Args:
            state: MHD state dict with rho, velocity, pressure, B.
            cell_volume: Volume per cell [m^3].
            C: Capacitance [F] for circuit energy.
            V_cap: Capacitor voltage [V].
            L_total: Total inductance [H].
            I_current: Current [A].

        Returns:
            EnergySnapshot with all components.
        """
        rho = state["rho"]
        p = state["pressure"]
        v = state.get("velocity")
        B = state.get("B")

        # Thermal energy: integral of p/(gamma-1) dV
        E_thermal = float(np.sum(p)) * cell_volume / (self.gamma - 1.0)

        # Kinetic energy: integral of 0.5*rho*v^2 dV
        E_kinetic = 0.0
        if v is not None:
            v_sq = np.sum(v**2, axis=0)  # |v|^2
            E_kinetic = 0.5 * float(np.sum(rho * v_sq)) * cell_volume

        # Magnetic energy: integral of B^2/(2*mu_0) dV
        E_magnetic = 0.0
        if B is not None:
            B_sq = np.sum(B**2, axis=0)  # |B|^2
            E_magnetic = float(np.sum(B_sq)) * cell_volume / (2.0 * MU_0)

        # Circuit energy: capacitor + inductor
        E_circuit = 0.5 * C * V_cap**2 + 0.5 * L_total * I_current**2

        E_total = E_thermal + E_kinetic + E_magnetic + self._cumulative_radiated + E_circuit

        return EnergySnapshot(
            t=0.0,  # Set by caller
            E_magnetic=E_magnetic,
            E_kinetic=E_kinetic,
            E_thermal=E_thermal,
            E_radiated=self._cumulative_radiated,
            E_circuit=E_circuit,
            E_total=E_total,
        )

    def record(
        self,
        state: dict[str, np.ndarray],
        t: float,
        dt: float,
        cell_volume: float,
        radiated_power: float = 0.0,
        C: float = 0.0,
        V_cap: float = 0.0,
        L_total: float = 0.0,
        I_current: float = 0.0,
    ) -> None:
        """Record energy at current timestep.

        Args:
            state: MHD state dict.
            t: Current time [s].
            dt: Timestep [s].
            cell_volume: Cell volume [m^3].
            radiated_power: Total radiated power this step [W].
            C, V_cap, L_total, I_current: Circuit parameters.
        """
        self._cumulative_radiated += radiated_power * dt

        snap = self.compute_energies(
            state, cell_volume, C, V_cap, L_total, I_current,
        )
        snap.t = t

        if self._E0 is None:
            self._E0 = snap.E_total

        # Conservation error
        if self._E0 > 0:
            error = abs(snap.E_total - self._E0) / self._E0
        else:
            error = 0.0

        self._report.times.append(t)
        self._report.E_magnetic.append(snap.E_magnetic)
        self._report.E_kinetic.append(snap.E_kinetic)
        self._report.E_thermal.append(snap.E_thermal)
        self._report.E_radiated.append(snap.E_radiated)
        self._report.E_circuit.append(snap.E_circuit)
        self._report.E_total.append(snap.E_total)
        self._report.conservation_error.append(error)

    def get_report(self) -> EnergyReport:
        return self._report

    def summary(self) -> str:
        r = self._report
        if not r.times:
            return "No energy data recorded"
        return (
            f"E_total: {r.E_total[0]:.2e} → {r.E_total[-1]:.2e} J | "
            f"Conservation: {r.max_conservation_error*100:.1f}% max error | "
            f"{'PASS' if r.is_conserved else 'FAIL'}"
        )
