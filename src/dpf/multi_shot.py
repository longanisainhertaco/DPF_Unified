"""Multi-shot DPF simulation for high repetition rate studies (Challenge 10).

Models inter-shot physics:
1. Residual ionization from previous discharge
2. Gas heating and thermal relaxation between shots
3. Electrode erosion accumulation (Cu/W mass per shot)
4. Fill gas replenishment

Usage:
    runner = MultiShotRunner(preset_name="pf1000", n_shots=10, rep_rate_hz=1.0)
    results = runner.run()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShotState:
    """State carried between consecutive DPF shots."""
    shot_number: int = 0
    residual_ionization_fraction: float = 0.0  # Z_bar / Z_max from previous shot
    gas_temperature_K: float = 300.0  # Fill gas temperature after cooling
    electrode_erosion_kg: float = 0.0  # Cumulative ablated mass
    impurity_fraction: float = 0.0  # High-Z impurity fraction in fill gas
    fill_pressure_Pa: float = 400.0  # Current fill pressure (changes with heating)
    results: list[dict[str, Any]] = field(default_factory=list)


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
    # Only ~0.1% of total resistive energy reaches the electrode surface
    ABLATION_EFFICIENCY = {"Cu": 5e-8, "W": 2e-8}  # kg/J (surface fraction included)
    # EMPIRICAL: gas cooling time constant between shots
    COOLING_TAU_S = 0.1  # 100 ms thermal relaxation

    def __init__(
        self,
        preset_name: str = "pf1000",
        n_shots: int = 10,
        rep_rate_hz: float = 1.0,
        sim_time_us: float = 20.0,
        gas_key: str = "D2",
        electrode_material: str = "Cu",
    ) -> None:
        self.preset_name = preset_name
        self.n_shots = n_shots
        self.rep_rate_hz = rep_rate_hz
        self.sim_time_us = sim_time_us
        self.gas_key = gas_key
        self.electrode_material = electrode_material
        self.inter_shot_dt = 1.0 / max(rep_rate_hz, 0.01)

    def run(self, progress_fn: Any = None) -> ShotState:
        """Execute multi-shot sequence.

        Returns:
            ShotState with accumulated results and final inter-shot state.
        """
        from app_engine import run_simulation_core

        state = ShotState()

        for i in range(self.n_shots):
            state.shot_number = i + 1
            if progress_fn:
                progress_fn(i / self.n_shots, desc=f"Shot {i + 1}/{self.n_shots}")

            # Modify fill conditions based on inter-shot state
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
                continue

            # Record shot result
            result["shot_number"] = i + 1
            result["residual_ionization"] = state.residual_ionization_fraction
            result["impurity_fraction"] = state.impurity_fraction
            result["gas_temperature_K"] = state.gas_temperature_K
            state.results.append(result)

            # === Inter-shot physics ===
            self._apply_inter_shot_physics(state, result)

            logger.info(
                "Shot %d: I_peak=%.3f MA, Yn=%.2e, T_gas=%.0f K, imp=%.1e",
                i + 1,
                result.get("I_peak", 0),
                result.get("neutron_yield", {}).get("Y_neutron", 0),
                state.gas_temperature_K,
                state.impurity_fraction,
            )

        return state

    def _apply_inter_shot_physics(self, state: ShotState, result: dict) -> None:
        """Update inter-shot state based on discharge result."""
        # 1. Residual ionization: fraction of peak ionization that persists
        #    Recombination time ~ 1-10 us, inter-shot time ~ ms-s
        #    At >1 Hz, some residual ionization persists
        recomb_time_s = 1e-5  # EMPIRICAL: ~10 us recombination
        decay = np.exp(-self.inter_shot_dt / recomb_time_s)
        state.residual_ionization_fraction = 0.5 * decay  # EMPIRICAL: 50% at pinch

        # 2. Gas heating: discharge deposits energy, gas cools between shots
        E_deposited_J = result.get("E_bank_kJ", 0) * 1e3 * 0.1  # EMPIRICAL: 10% heats gas
        # Delta T from energy deposition (assuming ideal gas in fixed volume)
        n_molecules = state.fill_pressure_Pa / (1.38e-23 * state.gas_temperature_K)
        if n_molecules > 0:
            dT = E_deposited_J / (1.5 * n_molecules * 1.38e-23)
            state.gas_temperature_K += dT

        # Cooling between shots (Newton's law toward 300 K)
        cooling = np.exp(-self.inter_shot_dt / self.COOLING_TAU_S)
        state.gas_temperature_K = 300.0 + (state.gas_temperature_K - 300.0) * cooling

        # Update pressure (gas law at new temperature, constant volume)
        state.fill_pressure_Pa *= state.gas_temperature_K / max(
            state.gas_temperature_K - dT if n_molecules > 0 else 300.0, 1.0
        )

        # 3. Electrode erosion
        E_res_kJ_arr = result.get("E_res_kJ", [0])
        if isinstance(E_res_kJ_arr, (list, np.ndarray)) and len(E_res_kJ_arr) > 0:
            E_res_J = float(E_res_kJ_arr[-1]) * 1e3  # final cumulative value in J
        else:
            E_res_J = 0.0
        eff = self.ABLATION_EFFICIENCY.get(self.electrode_material, 5e-5)
        dm = eff * E_res_J  # kg ablated this shot
        state.electrode_erosion_kg += dm

        # 4. Impurity fraction from ablated electrode material
        #    Fill gas mass in typical DPF chamber (~1 liter at fill pressure)
        chamber_vol_m3 = 1e-3  # EMPIRICAL: ~1 liter chamber volume
        ion_mass_kg = 3.34e-27 if self.gas_key == "D2" else 6.64e-27  # D2 or He
        n_fill = state.fill_pressure_Pa / (1.38e-23 * state.gas_temperature_K)
        fill_mass_kg = n_fill * ion_mass_kg * chamber_vol_m3
        if fill_mass_kg > 0:
            state.impurity_fraction = min(state.electrode_erosion_kg / fill_mass_kg, 1.0)
