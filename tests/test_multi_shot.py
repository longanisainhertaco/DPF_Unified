"""Tests for multi-shot DPF simulation (Challenge 10 deepening).

Tests cover:
- ShotState initialization and history tracking
- Inter-shot physics: ionization decay, cooling, replenishment, erosion
- MultiShotSummary computation
- Gas replenishment vs no-replenishment
- Edge cases
"""

from __future__ import annotations

import pytest

from dpf.multi_shot import MultiShotRunner, ShotState

# --- ShotState tests ---


class TestShotState:
    def test_default_state(self):
        state = ShotState()
        assert state.shot_number == 0
        assert state.residual_ionization_fraction == 0.0
        assert state.gas_temperature_K == 300.0
        assert state.electrode_erosion_kg == 0.0
        assert state.impurity_fraction == 0.0
        assert state.fill_pressure_Pa == 400.0
        assert len(state.yield_history) == 0
        assert len(state.I_peak_history) == 0

    def test_history_lists(self):
        state = ShotState()
        state.yield_history.append(1e8)
        state.I_peak_history.append(1.5)
        state.pressure_history.append(400.0)
        assert len(state.yield_history) == 1
        assert state.yield_history[0] == 1e8


# --- MultiShotRunner initialization ---


class TestMultiShotRunner:
    def test_default_init(self):
        runner = MultiShotRunner()
        assert runner.n_shots == 10
        assert runner.rep_rate_hz == 1.0
        assert runner.inter_shot_dt == 1.0
        assert runner.gas_replenish is True

    def test_custom_init(self):
        runner = MultiShotRunner(
            n_shots=5, rep_rate_hz=10.0,
            electrode_material="W",
            gas_replenish=False,
            chamber_volume_m3=2e-3,
        )
        assert runner.n_shots == 5
        assert runner.inter_shot_dt == pytest.approx(0.1)
        assert runner.electrode_material == "W"
        assert runner.gas_replenish is False
        assert runner.chamber_volume_m3 == 2e-3

    def test_high_rep_rate(self):
        runner = MultiShotRunner(rep_rate_hz=100.0)
        assert runner.inter_shot_dt == pytest.approx(0.01)


# --- Inter-shot physics (unit tests on _apply_inter_shot_physics) ---


class TestInterShotPhysics:
    def _make_runner(self, **kwargs) -> MultiShotRunner:
        defaults = {"n_shots": 1, "rep_rate_hz": 1.0}
        defaults.update(kwargs)
        return MultiShotRunner(**defaults)

    def _make_result(self, E_bank_kJ: float = 10.0, I_peak: float = 1.0) -> dict:
        return {
            "E_bank_kJ": E_bank_kJ,
            "I_peak": I_peak,
            "E_res_kJ": [E_bank_kJ * 0.3],  # 30% in resistance
        }

    def test_ionization_decays(self):
        """Residual ionization should decay between shots."""
        runner = self._make_runner(rep_rate_hz=1.0)
        state = ShotState()
        runner._apply_inter_shot_physics(state, self._make_result())
        # At 1 Hz (1s inter-shot), ionization should be ~0
        assert state.residual_ionization_fraction < 0.01

    def test_ionization_persists_at_high_rep_rate(self):
        """At high rep rate, some ionization should persist."""
        runner = self._make_runner(rep_rate_hz=10000.0)  # 10 kHz
        state = ShotState()
        runner._apply_inter_shot_physics(state, self._make_result())
        # At 10 kHz (0.1 ms), significant ionization persists
        assert state.residual_ionization_fraction > 0.1

    def test_gas_heats_then_cools(self):
        """Gas temperature should rise from discharge then cool toward wall."""
        runner = self._make_runner(rep_rate_hz=1.0)
        state = ShotState()
        state.gas_temperature_K = 300.0
        runner._apply_inter_shot_physics(state, self._make_result(E_bank_kJ=100.0))
        # After 1s cooling, temperature should be near wall temp
        assert state.gas_temperature_K < 500.0  # Cooled significantly
        assert state.gas_temperature_K >= 300.0  # But not below wall

    def test_gas_hotter_at_high_rep_rate(self):
        """Higher rep rate = less cooling time = hotter gas."""
        runner_slow = self._make_runner(rep_rate_hz=1.0)
        runner_fast = self._make_runner(rep_rate_hz=100.0)
        state_slow = ShotState()
        state_fast = ShotState()
        result = self._make_result(E_bank_kJ=100.0)
        runner_slow._apply_inter_shot_physics(state_slow, result)
        runner_fast._apply_inter_shot_physics(state_fast, result)
        assert state_fast.gas_temperature_K > state_slow.gas_temperature_K

    def test_gas_replenishment(self):
        """With replenishment, pressure should approach target."""
        runner = self._make_runner(rep_rate_hz=1.0)
        state = ShotState()
        state.fill_pressure_Pa = 200.0  # Low initial pressure
        state.target_pressure_Pa = 400.0
        runner._apply_inter_shot_physics(state, self._make_result(E_bank_kJ=1.0))
        # Pressure should be closer to target
        assert state.fill_pressure_Pa > 200.0

    def test_no_replenishment(self):
        """Without replenishment, pressure follows gas law only."""
        runner = self._make_runner(rep_rate_hz=1.0, gas_replenish=False)
        state = ShotState()
        state.fill_pressure_Pa = 400.0
        state.target_pressure_Pa = 400.0
        runner.gas_replenish = False
        runner._apply_inter_shot_physics(state, self._make_result(E_bank_kJ=100.0))
        # Pressure should change from heating, no valve correction
        # (exact value depends on heating/cooling balance)
        assert state.fill_pressure_Pa > 0

    def test_electrode_erosion_accumulates(self):
        """Erosion should increase with each shot."""
        runner = self._make_runner()
        state = ShotState()
        result = self._make_result(E_bank_kJ=50.0)
        runner._apply_inter_shot_physics(state, result)
        erosion_1 = state.electrode_erosion_kg
        runner._apply_inter_shot_physics(state, result)
        erosion_2 = state.electrode_erosion_kg
        assert erosion_2 > erosion_1 > 0

    def test_tungsten_less_erosion(self):
        """Tungsten should erode less than copper."""
        runner_cu = self._make_runner(electrode_material="Cu")
        runner_w = self._make_runner(electrode_material="W")
        state_cu = ShotState()
        state_w = ShotState()
        result = self._make_result(E_bank_kJ=50.0)
        runner_cu._apply_inter_shot_physics(state_cu, result)
        runner_w._apply_inter_shot_physics(state_w, result)
        assert state_w.electrode_erosion_kg < state_cu.electrode_erosion_kg

    def test_impurity_increases(self):
        """Impurity fraction should increase from erosion."""
        runner = self._make_runner()
        state = ShotState()
        state.fill_pressure_Pa = 400.0
        state.target_pressure_Pa = 400.0
        runner._apply_inter_shot_physics(state, self._make_result(E_bank_kJ=50.0))
        assert state.impurity_fraction > 0

    def test_impurity_bounded(self):
        """Impurity fraction should never exceed 1."""
        runner = self._make_runner()
        state = ShotState()
        state.fill_pressure_Pa = 400.0
        state.target_pressure_Pa = 400.0
        # Many shots
        for _ in range(100):
            runner._apply_inter_shot_physics(state, self._make_result(E_bank_kJ=500.0))
        assert state.impurity_fraction <= 1.0


# --- Summary computation ---


class TestMultiShotSummary:
    def test_summary_from_state(self):
        runner = MultiShotRunner(n_shots=3)
        state = ShotState()
        state.yield_history = [1e8, 8e7, 5e7]
        state.I_peak_history = [1.5, 1.4, 1.3]
        state.gas_temperature_K = 350.0
        state.fill_pressure_Pa = 380.0
        state.electrode_erosion_kg = 1e-9
        state.impurity_fraction = 0.001

        summary = runner._compute_summary(state)

        assert summary.n_shots == 3
        assert summary.total_yield == pytest.approx(2.3e8)
        assert summary.mean_yield == pytest.approx(2.3e8 / 3)
        assert summary.yield_degradation == pytest.approx(5e7 / 1e8)
        assert summary.final_temperature_K == 350.0
        assert summary.total_erosion_kg == 1e-9

    def test_shots_until_50pct(self):
        runner = MultiShotRunner(n_shots=5)
        state = ShotState()
        state.yield_history = [1e8, 8e7, 6e7, 4e7, 3e7]  # Drops below 50% at shot 4
        state.I_peak_history = [1.5] * 5

        summary = runner._compute_summary(state)
        assert summary.shots_until_50pct_yield == 4  # 4e7 < 5e7 (50% of 1e8)

    def test_no_degradation_when_constant(self):
        runner = MultiShotRunner(n_shots=3)
        state = ShotState()
        state.yield_history = [1e8, 1e8, 1e8]
        state.I_peak_history = [1.5, 1.5, 1.5]

        summary = runner._compute_summary(state)
        assert summary.yield_degradation == pytest.approx(1.0)
        assert summary.shots_until_50pct_yield == 3  # Never reaches 50%

    def test_empty_state(self):
        runner = MultiShotRunner(n_shots=0)
        state = ShotState()
        summary = runner._compute_summary(state)
        assert summary.n_shots == 0
        assert summary.total_yield == 0.0
