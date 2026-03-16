"""Multi-shot DPF simulation for high repetition rate studies (Challenge 10).

Models inter-shot physics:
1. Residual ionization from previous discharge
2. Gas heating: Ohmic deposition + conduction/radiation cooling
3. Electrode erosion accumulation (Cu/W mass per shot)
4. Fill gas replenishment (exponential approach to target pressure)
5. Impurity buildup and its effect on radiation losses
6. Yield degradation tracking

Usage:
    runner = MultiShotRunner(preset_name="pf1000", n_shots=10, rep_rate_hz=1.0)
    results = runner.run()

References:
    Soto et al., Phys. Plasmas 17:112702 (2010) — 100 Hz DPF.
    Lee & Saw, J. Fusion Energy 27:292 (2008) — rep-rate effects.
    Beg et al., Appl. Phys. Lett. 84:3500 (2004) — electrode erosion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_K_B = 1.380649e-23  # Boltzmann constant [J/K]


@dataclass
class ShotState:
    """State carried between consecutive DPF shots."""

    shot_number: int = 0
    residual_ionization_fraction: float = 0.0  # Z_bar / Z_max from previous shot
    gas_temperature_K: float = 300.0  # Fill gas temperature after cooling
    electrode_erosion_kg: float = 0.0  # Cumulative ablated mass
    impurity_fraction: float = 0.0  # High-Z impurity fraction in fill gas
    fill_pressure_Pa: float = 400.0  # Current fill pressure (changes with heating)
    target_pressure_Pa: float = 400.0  # Gas supply target pressure
    results: list[dict[str, Any]] = field(default_factory=list)
    yield_history: list[float] = field(default_factory=list)
    I_peak_history: list[float] = field(default_factory=list)
    pressure_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    erosion_history: list[float] = field(default_factory=list)
    impurity_history: list[float] = field(default_factory=list)


@dataclass
class MultiShotSummary:
    """Summary diagnostics for a multi-shot sequence."""

    n_shots: int
    total_yield: float  # Total neutron yield across all shots
    mean_yield: float  # Average per-shot yield
    yield_degradation: float  # Ratio: last shot yield / first shot yield
    mean_I_peak: float  # Average peak current [MA]
    final_temperature_K: float  # Gas temperature after last shot
    final_pressure_Pa: float  # Fill pressure after last shot
    total_erosion_kg: float  # Total electrode mass ablated
    final_impurity_fraction: float  # Final impurity level
    shots_until_50pct_yield: int  # How many shots before yield drops 50%
    rep_rate_hz: float
    device: str


class MultiShotRunner:
    """Run multiple consecutive DPF discharges with inter-shot physics.

    Args:
        preset_name: Device preset name.
        n_shots: Number of consecutive shots.
        rep_rate_hz: Repetition rate [Hz].
        sim_time_us: Simulation time per shot [us].
        gas_key: Fill gas species.
        electrode_material: "Cu" or "W" for erosion model.
    """

    # EMPIRICAL: ablation mass per unit energy at electrode surface
    ABLATION_EFFICIENCY = {"Cu": 5e-8, "W": 2e-8}  # kg/J

    # EMPIRICAL: gas cooling time constants
    COOLING_CONDUCTION_TAU_S = 0.05  # 50 ms conductive cooling to walls
    COOLING_RADIATION_TAU_S = 0.5  # 500 ms radiative cooling (slow at low T)

    # Gas replenishment time constant (depends on valve + pumping speed)
    GAS_REPLENISH_TAU_S = 0.2  # 200 ms for typical fast gas valve

    def __init__(
        self,
        preset_name: str = "pf1000",
        n_shots: int = 10,
        rep_rate_hz: float = 1.0,
        sim_time_us: float = 20.0,
        gas_key: str = "D2",
        electrode_material: str = "Cu",
        target_pressure_Pa: float | None = None,
        gas_replenish: bool = True,
        chamber_volume_m3: float = 1e-3,
    ) -> None:
        self.preset_name = preset_name
        self.n_shots = n_shots
        self.rep_rate_hz = rep_rate_hz
        self.sim_time_us = sim_time_us
        self.gas_key = gas_key
        self.electrode_material = electrode_material
        self.gas_replenish = gas_replenish
        self.chamber_volume_m3 = chamber_volume_m3
        self.inter_shot_dt = 1.0 / max(rep_rate_hz, 0.01)
        self._target_pressure_Pa = target_pressure_Pa

    def run(self, progress_fn: Any = None) -> tuple[ShotState, MultiShotSummary]:
        """Execute multi-shot sequence.

        Returns:
            Tuple of (ShotState, MultiShotSummary).
        """
        from app_engine import run_simulation_core

        state = ShotState()
        state.target_pressure_Pa = self._target_pressure_Pa or 400.0

        # Get initial pressure from preset if not specified
        try:
            from dpf.presets import get_preset
            preset = get_preset(self.preset_name)
            sc = preset.get("snowplow", {})
            if self._target_pressure_Pa is None:
                state.target_pressure_Pa = sc.get("fill_pressure_Pa", 400.0)
            state.fill_pressure_Pa = state.target_pressure_Pa
        except (ImportError, Exception):
            pass

        for i in range(self.n_shots):
            state.shot_number = i + 1
            if progress_fn:
                progress_fn(i / self.n_shots, desc=f"Shot {i + 1}/{self.n_shots}")

            pressure_torr = state.fill_pressure_Pa / 133.322

            try:
                result = run_simulation_core(
                    preset_name=self.preset_name,
                    sim_time_us=self.sim_time_us,
                    gas_key=self.gas_key,
                    pressure_torr=pressure_torr,
                )
            except Exception as exc:
                logger.warning("Shot %d failed: %s", i + 1, exc)
                result = {"error": str(exc), "shot": i + 1}
                state.results.append(result)
                state.yield_history.append(0.0)
                state.I_peak_history.append(0.0)
                continue

            # Record shot result
            result["shot_number"] = i + 1
            result["residual_ionization"] = state.residual_ionization_fraction
            result["impurity_fraction"] = state.impurity_fraction
            result["gas_temperature_K"] = state.gas_temperature_K

            Yn = result.get("neutron_yield", {}).get("Y_neutron", 0.0)
            I_pk = result.get("I_peak", 0.0)
            state.results.append(result)
            state.yield_history.append(float(Yn))
            state.I_peak_history.append(float(I_pk))

            # Inter-shot physics
            self._apply_inter_shot_physics(state, result)

            # Record histories
            state.pressure_history.append(state.fill_pressure_Pa)
            state.temperature_history.append(state.gas_temperature_K)
            state.erosion_history.append(state.electrode_erosion_kg)
            state.impurity_history.append(state.impurity_fraction)

            logger.info(
                "Shot %d/%d: I_peak=%.3f MA, Yn=%.2e, P=%.0f Pa, T=%.0f K, imp=%.1e",
                i + 1, self.n_shots,
                I_pk, Yn,
                state.fill_pressure_Pa,
                state.gas_temperature_K,
                state.impurity_fraction,
            )

        summary = self._compute_summary(state)
        return state, summary

    def _compute_summary(self, state: ShotState) -> MultiShotSummary:
        """Compute summary diagnostics from shot history."""
        yields = state.yield_history
        total_yield = sum(yields)
        mean_yield = total_yield / max(len(yields), 1)

        # Yield degradation: last / first (or 0 if first is 0)
        if len(yields) >= 2 and yields[0] > 0:
            degradation = yields[-1] / yields[0]
        else:
            degradation = 1.0

        # Shots until 50% yield loss
        shots_50pct = len(yields)  # Default: never
        if len(yields) >= 2 and yields[0] > 0:
            threshold = yields[0] * 0.5
            for i, y in enumerate(yields):
                if y < threshold:
                    shots_50pct = i + 1
                    break

        return MultiShotSummary(
            n_shots=len(yields),
            total_yield=total_yield,
            mean_yield=mean_yield,
            yield_degradation=degradation,
            mean_I_peak=float(np.mean(state.I_peak_history)) if state.I_peak_history else 0.0,
            final_temperature_K=state.gas_temperature_K,
            final_pressure_Pa=state.fill_pressure_Pa,
            total_erosion_kg=state.electrode_erosion_kg,
            final_impurity_fraction=state.impurity_fraction,
            shots_until_50pct_yield=shots_50pct,
            rep_rate_hz=self.rep_rate_hz,
            device=self.preset_name,
        )

    def _apply_inter_shot_physics(self, state: ShotState, result: dict) -> None:
        """Update inter-shot state based on discharge result.

        Models 5 physics processes between consecutive shots:
        1. Residual ionization decay (recombination)
        2. Gas heating from discharge + cooling (conduction + radiation)
        3. Gas replenishment (fast valve, exponential approach to target)
        4. Electrode erosion accumulation
        5. Impurity buildup from ablated electrode material
        """
        dt = self.inter_shot_dt

        # --- 1. Residual ionization ---
        # Radiative + 3-body recombination: tau_rec ~ 1/(alpha_R * n_e)
        # At n_e ~ 1e22 m^-3, alpha_R ~ 1e-18 m^3/s → tau ~ 1e-4 s = 100 us
        # At rep rates > 10 kHz, significant residual persists
        recomb_time_s = 1e-4  # EMPIRICAL: 100 us for ~1e22 m^-3
        decay = np.exp(-dt / recomb_time_s)
        state.residual_ionization_fraction = 0.5 * decay  # EMPIRICAL: 50% peak ionization

        # --- 2. Gas heating and cooling ---
        # Energy deposited in gas: ~10% of bank energy (rest is radiation + kinetic)
        E_deposited_J = result.get("E_bank_kJ", 0) * 1e3 * 0.1  # EMPIRICAL: 10%

        # Number density in chamber
        n_gas = state.fill_pressure_Pa / (_K_B * max(state.gas_temperature_K, 1.0))
        N_total = n_gas * self.chamber_volume_m3

        # Heating: dT = E / (1.5 * N * kB) for monatomic, (2.5 * N * kB) for diatomic
        dof = 5.0 if self.gas_key in ("D2", "H2", "N2") else 3.0  # degrees of freedom
        thermal_capacity = 0.5 * dof * N_total * _K_B
        dT = E_deposited_J / max(thermal_capacity, 1e-10)
        T_after_heating = state.gas_temperature_K + dT

        # Two-stage cooling: fast conduction to walls + slow radiation
        # Conduction: dominant at high T, brings gas toward wall temperature (~300 K)
        T_wall = 300.0
        cond_decay = np.exp(-dt / self.COOLING_CONDUCTION_TAU_S)
        T_after_cond = T_wall + (T_after_heating - T_wall) * cond_decay

        # Radiation: adds to cooling at high T (P_rad ~ T^0.5 for bremsstrahlung)
        rad_decay = np.exp(-dt / self.COOLING_RADIATION_TAU_S)
        T_final = T_wall + (T_after_cond - T_wall) * rad_decay

        state.gas_temperature_K = max(T_final, T_wall)

        # --- 3. Gas replenishment ---
        # Pressure drops from gas consumption + heating changes
        # Update pressure from ideal gas law at new temperature
        if T_after_heating > 0:
            P_heated = state.fill_pressure_Pa * T_after_heating / max(
                state.gas_temperature_K, 1.0
            )
        else:
            P_heated = state.fill_pressure_Pa

        # Fast gas valve: exponential approach to target pressure
        if self.gas_replenish:
            replenish_decay = np.exp(-dt / self.GAS_REPLENISH_TAU_S)
            state.fill_pressure_Pa = (
                state.target_pressure_Pa
                + (P_heated - state.target_pressure_Pa) * replenish_decay
            )
        else:
            state.fill_pressure_Pa = P_heated

        # Clamp pressure to physical range
        state.fill_pressure_Pa = max(state.fill_pressure_Pa, 10.0)  # Minimum 10 Pa

        # --- 4. Electrode erosion ---
        E_res_kJ_arr = result.get("E_res_kJ", [0])
        if isinstance(E_res_kJ_arr, (list, np.ndarray)) and len(E_res_kJ_arr) > 0:
            E_res_J = float(E_res_kJ_arr[-1]) * 1e3
        else:
            E_res_J = result.get("E_bank_kJ", 0) * 1e3 * 0.05  # EMPIRICAL: 5% goes to electrode
        eff = self.ABLATION_EFFICIENCY.get(self.electrode_material, 5e-8)
        dm = eff * E_res_J
        state.electrode_erosion_kg += dm

        # --- 5. Impurity fraction ---
        ion_mass_kg = 3.34e-27 if self.gas_key == "D2" else 6.64e-27
        n_fill = state.fill_pressure_Pa / (_K_B * max(state.gas_temperature_K, 1.0))
        fill_mass_kg = n_fill * ion_mass_kg * self.chamber_volume_m3
        if fill_mass_kg > 0:
            # Impurity fraction: ablated mass / total gas mass
            # Gas replenishment dilutes impurities
            if self.gas_replenish:
                # Fresh gas dilutes impurities
                dilution = 1.0 - np.exp(-dt / self.GAS_REPLENISH_TAU_S)
                state.impurity_fraction *= (1.0 - dilution * 0.5)  # EMPIRICAL: 50% dilution efficiency
            state.impurity_fraction = min(
                state.impurity_fraction + dm / max(fill_mass_kg, 1e-20), 1.0
            )
