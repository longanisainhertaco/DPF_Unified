"""Pure MLX RLC circuit solver for Dense Plasma Focus.

Implements the same implicit midpoint method as src/dpf/circuit/rlc_solver.py
using 100% MLX operations with no numpy or CPU materialization.

All internal state is stored as Python floats to remain compatible with
mx.grad (MLX differentiates through Python scalar arithmetic when values
are extracted from mx scalars by the caller before passing here).

Circuit equation:
    L_total * dI/dt + I * dL/dt + R_eff * I = V_cap - back_emf
    dV_cap/dt = -I / C
    L_total = L0 + Lp
    R_eff = R0 + R_plasma

Implicit midpoint (2nd order, A-stable):
    alpha   = dt / (2 * L_total)
    beta    = alpha * dt / (2 * C)
    R_star  = R_eff + dLp_dt
    denom   = 1 + alpha * R_star + beta
    I_new   = (I_n * (1 - alpha*R_star - beta) + 2*alpha*(V_n - back_emf)) / denom
    V_new   = V_n - (dt/C) * (I_n + I_new) / 2

Crowbar (voltage_zero mode): when V_cap crosses zero, freeze capacitor
and decay as L-R:
    I_new = I_n * exp(-R_eff * dt / L_total)
    V_new = 0.0
"""

from __future__ import annotations

import math

import mlx.core as mx


def _implicit_midpoint_step(
    I_n: float,
    V_n: float,
    L_total: float,
    R_eff: float,
    C: float,
    dLp_dt: float,
    back_emf: float,
    dt: float,
) -> tuple[float, float]:
    """Implicit midpoint RLC step. Pure Python scalar arithmetic (mx.grad-safe)."""
    L_safe = max(L_total, 1e-15)
    alpha = dt / (2.0 * L_safe)
    beta = alpha * dt / (2.0 * C)
    R_star = R_eff + dLp_dt
    denom = max(1.0 + alpha * R_star + beta, 1e-30)
    I_new = (I_n * (1.0 - alpha * R_star - beta) + 2.0 * alpha * (V_n - back_emf)) / denom
    I_mid = 0.5 * (I_n + I_new)
    V_new = V_n - (dt / C) * I_mid
    return I_new, V_new


def _crowbar_step(
    I_n: float,
    R_eff: float,
    L_total: float,
    dt: float,
) -> tuple[float, float]:
    """Post-crowbar L-R exponential decay. Returns (I_new, V_new=0)."""
    L_safe = max(L_total, 1e-15)
    I_new = I_n * math.exp(-R_eff * dt / L_safe)
    return I_new, 0.0


class MLXCircuitSolver:
    """Pure MLX implicit midpoint RLC circuit solver for Dense Plasma Focus.

    Parameters:
        V0: Initial capacitor voltage [V].
        C: Capacitance [F].
        L0: External (stray) inductance [H].
        R0: External resistance [Ohm].
        crowbar_enabled: Enable crowbar switch model.
        crowbar_mode: Trigger mode — 'voltage_zero' or 'fixed_time'.
        crowbar_time: Fixed trigger time [s] (only used when mode='fixed_time').
        crowbar_resistance: Additional resistance from crowbar switch [Ohm].
    """

    def __init__(
        self,
        V0: float,
        C: float,
        L0: float,
        R0: float,
        crowbar_enabled: bool = False,
        crowbar_mode: str = "voltage_zero",
        crowbar_time: float = 0.0,
        crowbar_resistance: float = 0.0,
    ) -> None:
        self.C = C
        self.L0 = L0
        self.R0 = R0
        self.crowbar_enabled = crowbar_enabled
        self.crowbar_mode = crowbar_mode
        self.crowbar_time = crowbar_time
        self.crowbar_resistance = crowbar_resistance

        self._current: float = 0.0
        self._voltage: float = float(V0)
        self._time: float = 0.0
        self._crowbar_fired: bool = False

    @property
    def current(self) -> float:
        return self._current

    @property
    def voltage(self) -> float:
        return self._voltage

    def _should_fire_crowbar(self, V_old: float, V_new: float) -> bool:
        if not self.crowbar_enabled or self._crowbar_fired:
            return False
        if self.crowbar_mode == "voltage_zero":
            return V_old > 0.0 and V_new <= 0.0
        if self.crowbar_mode == "fixed_time":
            return self._time >= self.crowbar_time
        return False

    def step(
        self,
        Lp: float,
        dLp_dt: float,
        R_plasma: float,
        back_emf: float,
        dt: float,
    ) -> tuple[float, float]:
        """Advance circuit by dt. Returns (I_new, V_new).

        All arguments are Python floats extracted from mx scalars by the caller.
        Internal arithmetic uses Python scalars so mx.grad can differentiate
        through a calling function that builds Lp, dLp_dt etc. from mx arrays.

        Args:
            Lp: Plasma inductance [H].
            dLp_dt: Time derivative of plasma inductance [H/s].
            R_plasma: Plasma resistance [Ohm].
            back_emf: Back-EMF from plasma motion [V].
            dt: Timestep [s].

        Returns:
            (I_new, V_new): Updated current [A] and capacitor voltage [V].
        """
        I_n: float = self._current
        V_n: float = self._voltage
        L_total: float = self.L0 + Lp
        R_eff: float = self.R0 + R_plasma

        if self._crowbar_fired:
            R_eff += self.crowbar_resistance
            I_new, V_new = _crowbar_step(I_n, R_eff, L_total, dt)
        else:
            I_new, V_new = _implicit_midpoint_step(
                I_n, V_n, L_total, R_eff, self.C, dLp_dt, back_emf, dt
            )
            if self._should_fire_crowbar(V_n, V_new):
                self._crowbar_fired = True
                V_new = 0.0

        self._current = I_new
        self._voltage = V_new
        self._time += dt

        return I_new, V_new

    def reset(
        self,
        V0: float | None = None,
        I0: float = 0.0,
    ) -> None:
        """Reset circuit to initial conditions (useful for gradient sweeps).

        Args:
            V0: Reset voltage [V]. If None, uses construction V0.
            I0: Reset current [A]. Defaults to 0.
        """
        if V0 is not None:
            self._voltage = float(V0)
        self._current = float(I0)
        self._time = 0.0
        self._crowbar_fired = False

    def energy_capacitor(self) -> mx.array:
        """Capacitor energy as mx scalar [J]. Differentiable w.r.t. voltage."""
        V = mx.array(self._voltage)
        return mx.array(0.5 * self.C) * V * V

    def energy_inductor(self, Lp: float) -> mx.array:
        """Inductor energy as mx scalar [J]. Differentiable w.r.t. current."""
        I = mx.array(self._current)
        L_total = mx.array(self.L0 + Lp)
        return mx.array(0.5) * L_total * I * I
