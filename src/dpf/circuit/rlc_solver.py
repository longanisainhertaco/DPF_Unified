"""Distributed RLC circuit solver with plasma coupling.

Solves the circuit equation for a Dense Plasma Focus:
    L(t) * dI/dt + I * dL/dt + R * I = V_cap
    dV_cap/dt = -I / C

where L(t) = L0 + ESL + L_plasma(t) and the plasma inductance
L_plasma is computed from the plasma solver via CouplingState.

The solver uses a second-order implicit midpoint method for stability
with the stiff plasma inductance term.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dpf.constants import mu_0, pi
from dpf.core.bases import CircuitSolverBase, CouplingState

logger = logging.getLogger(__name__)


@dataclass
class CircuitState:
    """Internal state of the circuit solver."""

    current: float = 0.0        # Circuit current [A]
    voltage: float = 0.0        # Capacitor voltage [V]
    charge: float = 0.0         # Capacitor charge [C]
    energy_cap: float = 0.0     # Capacitor energy [J]
    energy_ind: float = 0.0     # Inductor energy [J]
    energy_res: float = 0.0     # Cumulative resistive dissipation [J]
    time: float = 0.0           # Current time [s]


class RLCSolver(CircuitSolverBase):
    """Second-order implicit RLC circuit solver for DPF.

    Parameters:
        C: Capacitance [F].
        V0: Initial voltage [V].
        L0: External inductance [H].
        R0: External resistance [Ohm].
        ESR: Equivalent series resistance [Ohm].
        ESL: Equivalent series inductance [H].
        anode_radius: Anode radius [m] (for plasma inductance estimate).
        cathode_radius: Cathode radius [m].
    """

    def __init__(
        self,
        C: float,
        V0: float,
        L0: float,
        R0: float = 0.0,
        ESR: float = 0.0,
        ESL: float = 0.0,
        anode_radius: float = 0.005,
        cathode_radius: float = 0.01,
    ) -> None:
        self.C = C
        self.L0 = L0
        self.R_total = R0 + ESR
        self.L_ext = L0 + ESL
        self.anode_radius = anode_radius
        self.cathode_radius = cathode_radius

        # Initial state
        self.state = CircuitState(
            voltage=V0,
            charge=C * V0,
            energy_cap=0.5 * C * V0**2,
        )

        logger.info(
            "RLCSolver initialized: C=%.2e F, V0=%.1f V, L0=%.2e H, R=%.2e Ohm",
            C, V0, L0, self.R_total,
        )

    @property
    def current(self) -> float:
        return self.state.current

    @property
    def voltage(self) -> float:
        return self.state.voltage

    def plasma_inductance_estimate(self, pinch_radius: float, length: float = 0.05) -> float:
        """Estimate plasma inductance from geometry.

        L_plasma = (mu_0 / 2pi) * length * ln(cathode_radius / pinch_radius)

        Args:
            pinch_radius: Current pinch radius [m].
            length: Plasma column length [m].

        Returns:
            Plasma inductance [H].
        """
        r = max(pinch_radius, 1e-6)  # Prevent log(0)
        return (mu_0 / (2 * pi)) * length * np.log(self.cathode_radius / r)

    def step(
        self,
        coupling: CouplingState,
        back_emf: float,
        dt: float,
    ) -> CouplingState:
        """Advance the circuit by one timestep using implicit midpoint.

        The circuit equation:
            (L_ext + Lp) * dI/dt = V_cap - R * I - I * dLp/dt - back_emf
            dV_cap/dt = -I / C

        Uses the implicit midpoint rule for second-order accuracy and
        unconditional stability.

        Args:
            coupling: Plasma coupling state (Lp, dL_dt from plasma solver).
            back_emf: Back-EMF from plasma motion [V].
            dt: Timestep [s].

        Returns:
            Updated coupling state with new current and voltage.
        """
        I_n = self.state.current
        V_n = self.state.voltage
        Lp = coupling.Lp
        dLp_dt = coupling.dL_dt
        L_total = self.L_ext + Lp
        R_eff = self.R_total

        # Implicit midpoint for I and V
        # At midpoint: I_mid = (I_n + I_{n+1}) / 2, V_mid = (V_n + V_{n+1}) / 2
        # V_{n+1} = V_n - (dt / C) * I_mid
        # L_total * (I_{n+1} - I_n) = dt * (V_mid - R_eff * I_mid - I_mid * dLp_dt - back_emf)
        #
        # Substituting V_{n+1} and I_mid:
        # L * (I1 - I0) = dt * ((V0 + V1)/2 - R*(I0+I1)/2 - (I0+I1)/2 * dLp_dt - back_emf)
        # Let alpha = dt / (2 * L_total)
        # I1 * (1 + alpha * (R_eff + dLp_dt) + alpha * dt / (2*C))
        #   = I0 * (1 - alpha * (R_eff + dLp_dt) - alpha * dt / (2*C))
        #     + 2 * alpha * (V0 - back_emf)
        # ... simplify below.

        alpha = dt / (2.0 * max(L_total, 1e-15))
        beta = alpha * dt / (2.0 * self.C)
        R_star = R_eff + dLp_dt

        denom = 1.0 + alpha * R_star + beta
        I_new = (
            I_n * (1.0 - alpha * R_star - beta)
            + 2.0 * alpha * (V_n - back_emf)
        ) / max(denom, 1e-30)

        # Update voltage
        I_mid = 0.5 * (I_n + I_new)
        V_new = V_n - (dt / self.C) * I_mid

        # Energy accounting
        self.state.energy_res += R_eff * I_mid**2 * dt
        self.state.energy_cap = 0.5 * self.C * V_new**2
        self.state.energy_ind = 0.5 * L_total * I_new**2

        # Update state
        self.state.current = I_new
        self.state.voltage = V_new
        self.state.charge = self.C * V_new
        self.state.time += dt

        return CouplingState(
            Lp=Lp,
            emf=back_emf,
            current=I_new,
            voltage=V_new,
            dL_dt=dLp_dt,
        )

    def total_energy(self) -> float:
        """Return total circuit energy (capacitor + inductor + resistive losses)."""
        return self.state.energy_cap + self.state.energy_ind + self.state.energy_res

    def initial_energy(self) -> float:
        """Return the initial energy stored in the capacitor."""
        return 0.5 * self.C * (self.state.charge / self.C) ** 2
