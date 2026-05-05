"""Time-resolved neutron yield tracker for MHD simulations.

Accumulates thermonuclear and beam-target neutron yield at each timestep,
building a Y(t) curve that shows when fusion occurs during the discharge.

This enables:
- Identifying which phase produces the most neutrons
- Comparing thermonuclear vs beam-target contributions over time
- Correlating yield with pinch dynamics (compression, instability)

Usage:
    tracker = YieldTracker(ion_mass=3.34e-27)
    for each MHD step:
        tracker.accumulate(state, dt, I_current, V_pinch, cell_volume)
    result = tracker.get_result()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dpf.constants import k_B

logger = logging.getLogger(__name__)


@dataclass
class YieldTimepoint:
    """Neutron yield at a single timestep."""

    t: float                    # Time [s]
    dY_thermo: float            # Thermonuclear yield this step
    dY_bt: float                # Beam-target yield this step
    Y_thermo_cumulative: float  # Cumulative thermonuclear
    Y_bt_cumulative: float      # Cumulative beam-target
    T_peak_keV: float           # Peak ion temperature [keV]
    n_peak: float               # Peak ion density [m^-3]
    rho_ratio: float            # Peak density / initial density


@dataclass
class YieldResult:
    """Complete time-resolved yield from a simulation."""

    times: list[float] = field(default_factory=list)
    dY_thermo: list[float] = field(default_factory=list)
    dY_bt: list[float] = field(default_factory=list)
    Y_thermo_cumulative: list[float] = field(default_factory=list)
    Y_bt_cumulative: list[float] = field(default_factory=list)
    T_peak_keV: list[float] = field(default_factory=list)
    n_peak: list[float] = field(default_factory=list)

    @property
    def Y_total(self) -> float:
        if self.Y_thermo_cumulative and self.Y_bt_cumulative:
            return self.Y_thermo_cumulative[-1] + self.Y_bt_cumulative[-1]
        return 0.0

    @property
    def bt_fraction(self) -> float:
        total = self.Y_total
        if total > 0 and self.Y_bt_cumulative:
            return self.Y_bt_cumulative[-1] / total
        return 0.0

    @property
    def peak_yield_time(self) -> float:
        """Time of maximum instantaneous yield rate."""
        if not self.dY_thermo:
            return 0.0
        total_rate = [a + b for a, b in zip(self.dY_thermo, self.dY_bt, strict=False)]
        idx = int(np.argmax(total_rate))
        return self.times[idx]


class YieldTracker:
    """Accumulates neutron yield during an MHD simulation.

    Args:
        ion_mass: Ion mass [kg] (default: deuterium).
        rho0: Initial gas density [kg/m^3] for compression ratio.
    """

    def __init__(
        self,
        ion_mass: float = 3.34e-27,
        rho0: float = 1e-4,
    ) -> None:
        self.ion_mass = ion_mass
        self.rho0 = rho0
        self._Y_thermo = 0.0
        self._Y_bt = 0.0
        self._result = YieldResult()

    def accumulate(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        I_current: float = 0.0,
        V_pinch: float = 0.0,
        cell_volume: float = 1e-9,
        f_beam: float = 0.14,
        L_pinch: float = 0.0,
        tau_dwell: float = 0.0,
    ) -> None:
        """Accumulate yield from one MHD timestep.

        Args:
            state: MHD state dict with rho, pressure (and optionally Ti, Te).
            dt: Timestep [s].
            I_current: Circuit current [A] for beam-target.
            V_pinch: Pinch voltage [V] for beam-target.
            cell_volume: Cell volume [m^3].
            f_beam: Beam fraction for beam-target yield.
            L_pinch: Beam-target interaction length [m]. If 0, uses 1 cm default.
            tau_dwell: Pinch dwell time [s] for beam-target rate normalization.
                The Lee/Saw KR eq. 1 yield is per-shot; the wrapper divides
                by tau_dwell so that integrating dY_bt over the dwell window
                recovers the per-shot total. If 0, beam-target is suppressed
                (caller must supply a physical dwell time, typically 30-50 ns
                for PF-1000 / MJOLNIR-class devices).
        """
        rho = state["rho"]
        n_i = rho / self.ion_mass
        n_i_safe = np.maximum(n_i, 0.0)

        # Ion temperature
        if "Ti" in state:
            Ti = state["Ti"]
        else:
            Ti = state["pressure"] * self.ion_mass / (2.0 * np.maximum(rho, 1e-30) * k_B)
        Ti_safe = np.maximum(Ti, 1.0)

        # Peak values
        Ti_keV = float(np.max(Ti_safe)) * k_B / (1000.0 * 1.602e-19)
        n_peak = float(np.max(n_i_safe))
        _N_PEAK_MAX = 1.0e28
        n_peak_safe = min(n_peak, _N_PEAK_MAX)
        if n_peak > _N_PEAK_MAX:
            logger.debug("YieldTracker: n_peak %.2e exceeds physical cap, clamped to %.2e", n_peak, _N_PEAK_MAX)

        # Thermonuclear yield: dY = 0.25 * integral(n_D^2) * <sigma*v> * V_cell * dt
        # Correct volume integral: sum n_i^2 over all cells, not n_peak^2 * V_total
        dY_thermo = 0.0
        if Ti_keV > 0.1:
            try:
                from dpf.diagnostics.neutron_yield import dd_reactivity
                sigma_v = dd_reactivity(Ti_keV)
                n_sq_sum = float(np.sum(np.minimum(n_i_safe, _N_PEAK_MAX) ** 2))
                dY_step = 0.25 * n_sq_sum * float(cell_volume) * sigma_v * dt
                dY_thermo = float(min(dY_step, 1.0e50))
            except ImportError:
                pass

        # Beam-target yield. Lee/Saw KR eq. 1 [KR L4080-4087 p.18] is a
        # per-shot total, not a rate. The wrapper divides Yn_total by
        # tau_dwell so that integrating dY_bt over the dwell window
        # recovers Yn_total exactly. Without an explicit tau_dwell the
        # rate is zero (was: tau_transit ~ 1ns gave 30-50x overcount when
        # integrated over the ~30-50 ns pinch — MJOLNIR-2MJ 41x bug).
        dY_bt = 0.0
        if abs(V_pinch) > 1e3 and abs(I_current) > 1e3 and tau_dwell > 0.0:
            try:
                from dpf.diagnostics.beam_target import beam_target_yield_rate
                _L = L_pinch if L_pinch > 0 else 0.01  # fallback 1 cm
                bt_rate = beam_target_yield_rate(
                    abs(I_current), abs(V_pinch), n_peak_safe, _L,
                    f_beam=f_beam,
                    tau_dwell=tau_dwell,
                )
                dY_bt = bt_rate * dt
            except ImportError:
                pass
            except Exception as exc:
                logger.warning("beam_target_yield_rate failed: %s", exc)

        self._Y_thermo += dY_thermo
        self._Y_bt += dY_bt

        t = sum(self._result.times[-1:]) if self._result.times else 0.0
        t += dt

        self._result.times.append(t)
        self._result.dY_thermo.append(dY_thermo)
        self._result.dY_bt.append(dY_bt)
        self._result.Y_thermo_cumulative.append(self._Y_thermo)
        self._result.Y_bt_cumulative.append(self._Y_bt)
        self._result.T_peak_keV.append(Ti_keV)
        self._result.n_peak.append(n_peak)

    def get_result(self) -> YieldResult:
        """Return the accumulated yield result."""
        return self._result

    def summary(self) -> str:
        """Return a one-line summary."""
        r = self._result
        return (
            f"Y_total={r.Y_total:.2e} "
            f"(thermo={self._Y_thermo:.2e}, BT={self._Y_bt:.2e}, "
            f"BT%={r.bt_fraction*100:.0f}%)"
        )
