"""Distributed RLC circuit solver with plasma coupling.

Solves the circuit equation for a Dense Plasma Focus:
    L(t) * dI/dt + I * dL/dt + R * I = V_cap
    dV_cap/dt = -I / C

where L(t) = L0 + ESL + L_plasma(t) and the plasma inductance
L_plasma is computed from the plasma solver via CouplingState.

The solver uses a second-order implicit midpoint method for stability
with the stiff plasma inductance term.

Crowbar model: When enabled, the crowbar switch short-circuits the
capacitor bank (typically when V_cap crosses zero or at a fixed time).
Post-crowbar, the circuit becomes an L-R decay: I(t) = I_cb * exp(-R*t/L).

dL/dt computation: Uses 2nd-order central difference when sufficient
history is available (>= 3 points), falling back to 1st-order backward
difference for the first two steps.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

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
    energy_res: float = 0.0     # Cumulative resistive dissipation [J] (external only)
    energy_res_plasma: float = 0.0  # Cumulative plasma Ohmic dissipation [J] (tracked, not in total)
    time: float = 0.0           # Current time [s]
    crowbar_fired: bool = False     # Whether the crowbar switch has fired
    crowbar_fire_time: float = -1.0  # Time at which crowbar fired [s] (-1 = not fired)


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
        crowbar_enabled: Enable crowbar switch model.
        crowbar_mode: Trigger mode — 'voltage_zero' or 'fixed_time'.
        crowbar_time: Fixed trigger time [s] (only if mode='fixed_time').
        crowbar_resistance: Additional resistance from crowbar switch [Ohm].
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
        crowbar_enabled: bool = False,
        crowbar_mode: str = "voltage_zero",
        crowbar_time: float = 0.0,
        crowbar_resistance: float = 0.0,
    ) -> None:
        self.C = C
        self.L0 = L0
        self.R_total = R0 + ESR
        self.L_ext = L0 + ESL
        self.anode_radius = anode_radius
        self.cathode_radius = cathode_radius

        # Crowbar switch configuration
        self.crowbar_enabled = crowbar_enabled
        self.crowbar_mode = crowbar_mode
        self.crowbar_time = crowbar_time
        self.crowbar_resistance = crowbar_resistance

        # dL/dt history for 2nd-order central difference
        # Stores (time, L_plasma) tuples; max 3 entries needed
        self._Lp_history: deque[tuple[float, float]] = deque(maxlen=3)

        # Initial state
        self.state = CircuitState(
            voltage=V0,
            charge=C * V0,
            energy_cap=0.5 * C * V0**2,
        )

        logger.info(
            "RLCSolver initialized: C=%.2e F, V0=%.1f V, L0=%.2e H, R=%.2e Ohm, "
            "crowbar=%s (mode=%s)",
            C, V0, L0, self.R_total, crowbar_enabled, crowbar_mode,
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

    def compute_dLp_dt(self, Lp: float) -> float:
        """Compute dL_plasma/dt using the best available finite difference.

        Uses 2nd-order backward difference (BDF2) when >= 2 history points
        are available, otherwise falls back to 1st-order backward difference.

        BDF2 formula at t_n:
            dL/dt = (3*L_n - 4*L_{n-1} + L_{n-2}) / (2*dt)

        This is O(dt^2) accurate at the current time t_n, unlike the
        central difference (L_n - L_{n-2})/(2*dt) which is O(dt^2) at
        the midpoint t_{n-1} but only O(dt) at t_n.

        Args:
            Lp: Current plasma inductance [H].

        Returns:
            dL_plasma/dt [H/s].
        """
        n = len(self._Lp_history)

        if n == 0:
            return 0.0

        if n >= 2:
            # 2nd-order backward difference (BDF2)
            t_nm2, Lp_nm2 = self._Lp_history[-2]
            t_nm1, Lp_nm1 = self._Lp_history[-1]
            t_now = self.state.time
            dt1 = t_nm1 - t_nm2
            dt2 = t_now - t_nm1
            if dt1 > 0 and dt2 > 0:
                if abs(dt1 - dt2) < 1e-20 * max(dt1, dt2, 1e-30):
                    # Uniform spacing: use standard BDF2
                    dt = dt2
                    return (3.0 * Lp - 4.0 * Lp_nm1 + Lp_nm2) / (2.0 * dt)
                else:
                    # Non-uniform spacing: use general 2nd-order backward
                    r = dt2 / dt1
                    return (
                        (1.0 + 2.0 * r) / (1.0 + r) * Lp
                        - (1.0 + r) * Lp_nm1
                        + r**2 / (1.0 + r) * Lp_nm2
                    ) / dt2

        # 1st-order backward difference: (L_n - L_{n-1}) / (t_n - t_{n-1})
        t_nm1, Lp_nm1 = self._Lp_history[-1]
        dt_back = self.state.time - t_nm1
        if dt_back > 0:
            return (Lp - Lp_nm1) / dt_back

        return 0.0

    def _check_crowbar(self, V_old: float, V_new: float) -> bool:
        """Check if crowbar should trigger this step.

        Args:
            V_old: Capacitor voltage at start of step [V].
            V_new: Capacitor voltage at end of step [V].

        Returns:
            True if crowbar triggers this step.
        """
        if not self.crowbar_enabled or self.state.crowbar_fired:
            return False

        if self.crowbar_mode == "voltage_zero":
            # Trigger when V_cap changes sign (crosses zero from positive)
            return V_old > 0 and V_new <= 0
        elif self.crowbar_mode == "fixed_time":
            return self.state.time >= self.crowbar_time
        return False

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

        After crowbar fires, the capacitor is frozen (dV_cap/dt = 0) and
        the circuit decays as L-R with the crowbar switch resistance added.

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

        # Use 2nd-order dL/dt if coupling doesn't provide it, or if we have history
        if coupling.dL_dt != 0.0:
            dLp_dt = coupling.dL_dt
        else:
            dLp_dt = self.compute_dLp_dt(Lp)

        L_total = self.L_ext + Lp
        R_eff = self.R_total + coupling.R_plasma

        # Add crowbar resistance if already fired
        if self.state.crowbar_fired:
            R_eff += self.crowbar_resistance

        if self.state.crowbar_fired:
            # Post-crowbar: capacitor frozen, L-R decay
            # L * dI/dt = -R_eff * I - I * dLp/dt - back_emf
            # Implicit midpoint: L * (I1 - I0) / dt = -(R_eff + dLp_dt) * (I0 + I1)/2 - back_emf
            alpha = dt / (2.0 * max(L_total, 1e-15))
            R_star = R_eff + dLp_dt
            denom = 1.0 + alpha * R_star
            I_new = (
                I_n * (1.0 - alpha * R_star)
                - 2.0 * alpha * back_emf
            ) / max(denom, 1e-30)

            V_new = V_n  # Capacitor voltage frozen

        else:
            # Normal RLC operation: implicit midpoint method
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

            # Check if crowbar should trigger
            if self._check_crowbar(V_n, V_new):
                self.state.crowbar_fired = True
                self.state.crowbar_fire_time = self.state.time + dt
                V_new = 0.0  # Clamp voltage to zero at crowbar instant
                logger.info(
                    "Crowbar fired at t=%.3e s, I=%.1f A, V_cap=%.1f V",
                    self.state.time + dt, I_new, V_n,
                )

        # Energy accounting
        I_mid = 0.5 * (I_n + I_new)
        # Split resistive dissipation: external vs plasma
        R_ext = self.R_total
        if self.state.crowbar_fired:
            R_ext += self.crowbar_resistance
        self.state.energy_res += R_ext * I_mid**2 * dt
        # Plasma resistance dissipation tracked separately
        self.state.energy_res_plasma += coupling.R_plasma * I_mid**2 * dt
        self.state.energy_cap = 0.5 * self.C * V_new**2
        self.state.energy_ind = 0.5 * L_total * I_new**2

        # Update state
        self.state.current = I_new
        self.state.voltage = V_new
        self.state.charge = self.C * V_new
        self.state.time += dt

        # Record Lp history for 2nd-order dL/dt
        self._Lp_history.append((self.state.time, Lp))

        return CouplingState(
            Lp=Lp,
            emf=back_emf,
            current=I_new,
            voltage=V_new,
            dL_dt=dLp_dt,
        )

    def total_energy(self) -> float:
        """Return total circuit energy (capacitor + inductor + resistive losses).

        Note: energy_res_plasma is NOT included here because the MHD solver
        handles plasma Ohmic heating spatially (via eta_field). Including it
        would double-count the same physical dissipation.
        """
        return self.state.energy_cap + self.state.energy_ind + self.state.energy_res

    def initial_energy(self) -> float:
        """Return the initial energy stored in the capacitor."""
        return 0.5 * self.C * (self.state.charge / self.C) ** 2
