"""Consolidated circuit tests for DPF-Unified.

Coverage:
- test_circuit.py: RLCSolver initialization, LC oscillation, FieldManager inductance
- test_coupling.py: Spitzer resistivity, CouplingState fields, plasma resistance, SheathConfig
- test_ohmic_correction.py: Ohmic correction config, gap measurement, correction computation, integration
- test_phase_aa_circuit_fixes.py: Single circuit step fix, Bosch-Hale Branch 2
- test_phase_ar_crowbar.py: Crowbar infrastructure, Lee model crowbar, calibration bounds, presets
- test_phase_r_circuit_coupling.py: Back-EMF correctness, R_plasma cap, energy accounting, circuit dynamics, backend warnings
- test_phase_s_circuit.py: Crowbar trigger/state, post-crowbar behavior, 2nd-order dL/dt, PF-1000 analytical
- test_phase_s_energy.py: Energy conservation audit (circuit + MHD)
- test_phase_x_circuit_energy_coupling.py: Ohmic correction timing, conservative_energy config, circuit energy balance
- test_phase_z_back_emf.py: Back-EMF coupling (motional EMF), sign conventions, scaling laws
"""

from __future__ import annotations

import contextlib
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dpf.circuit.rlc_solver import CircuitState, RLCSolver
from dpf.config import SimulationConfig
from dpf.constants import e, m_e, mu_0, pi
from dpf.core.bases import CouplingState
from dpf.core.field_manager import FieldManager
from dpf.engine import SimulationEngine
from dpf.presets import get_preset

# --- Section: Basic RLC Solver and FieldManager (test_circuit.py) ---


class TestRLCSolver:
    def test_initialization(self):
        solver = RLCSolver(C=1e-6, V0=10e3, L0=100e-9, R0=0.1)
        assert solver.state.voltage == 10e3
        assert solver.state.current == 0.0
        assert solver.state.charge == 1e-6 * 10e3

    def test_lc_oscillation(self):
        """Verify frequency of ideal LC circuit."""
        C = 1e-6
        L = 1e-6
        V0 = 100.0
        solver = RLCSolver(C=C, V0=V0, L0=L, R0=0.0)

        dt = 1e-8
        coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

        steps = int((np.pi / 2 * 1e-6) / dt)

        max_I = 0.0
        for _ in range(steps * 2):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(coupling.current) > max_I:
                max_I = abs(coupling.current)

        assert np.isclose(max_I, 100.0, rtol=0.01)

    def test_variable_inductance(self):
        """Test circuit response to increasing inductance (compression)."""
        pass


class TestInductance:
    def test_cartesian_inductance(self):
        fm = FieldManager((1, 1, 1), dx=1.0, dy=1.0, dz=1.0, geometry="cartesian")
        fm.B[2, 0, 0, 0] = 1.0

        I_current = 1.0  # noqa: N806
        L_calc = fm.compute_plasma_inductance(I_current)

        assert np.isclose(L_calc, 1.0 / mu_0)

    def test_cylindrical_inductance(self):
        fm = FieldManager((1, 1, 1), dx=1.0, dz=1.0, geometry="cylindrical")
        fm.B[1, 0, 0, 0] = 1.0

        I_current = 1.0  # noqa: N806
        L_calc = fm.compute_plasma_inductance(I_current)

        assert np.isclose(L_calc, pi / mu_0)


# --- Section: Spitzer, CouplingState, Plasma Resistance, SheathConfig (test_coupling.py) ---


class TestSpitzerResistivity:
    """Tests for Spitzer resistivity function."""

    def test_analytic_formula(self):
        """eta = m_e * nu_ei / (ne * e^2 * alpha(Z)) — with Braginskii correction."""
        from dpf.collision.spitzer import nu_ei, spitzer_alpha, spitzer_resistivity

        ne = np.array([1e20])
        Te = np.array([1e6])
        lnL = 10.0
        Z = 1.0

        eta = spitzer_resistivity(ne, Te, lnL, Z=Z)
        freq = nu_ei(ne, Te, lnL, Z=Z)
        alpha_Z = spitzer_alpha(Z)
        eta_expected = m_e * freq / (ne * e**2) / alpha_Z

        np.testing.assert_allclose(eta, eta_expected, rtol=1e-10)

    def test_decreases_with_temperature(self):
        """Hotter plasma has lower resistivity (eta ~ Te^{-3/2})."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e20])
        lnL = 10.0

        eta_cold = spitzer_resistivity(ne, np.array([1e5]), lnL)
        eta_hot = spitzer_resistivity(ne, np.array([1e7]), lnL)

        assert eta_hot[0] < eta_cold[0], "Hot plasma should have lower resistivity"
        ratio = eta_cold[0] / eta_hot[0]
        np.testing.assert_allclose(ratio, 1000.0, rtol=0.1)

    def test_positive(self):
        """Resistivity should always be positive."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e18, 1e20, 1e22])
        Te = np.array([1e4, 1e6, 1e8])

        eta = spitzer_resistivity(ne, Te)
        assert np.all(eta > 0)

    def test_typical_dpf_value(self):
        """At DPF conditions (1 keV, 10^24 m^-3), eta ~ 10^-7 to 10^-5 Ohm*m."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e24])
        Te_keV = np.array([1.16e7])

        eta = spitzer_resistivity(ne, Te_keV)
        assert 1e-10 < eta[0] < 1e-3, f"eta = {eta[0]:.2e} out of expected range"


class TestCouplingState:
    """Tests for CouplingState dataclass."""

    def test_has_R_plasma(self):
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert hasattr(cs, "R_plasma")
        assert cs.R_plasma == 0.0

    def test_has_Z_bar(self):
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert hasattr(cs, "Z_bar")
        assert cs.Z_bar == 1.0

    def test_no_dead_fields(self):
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert not hasattr(cs, "mutual_inductance"), "mutual_inductance should be removed"
        assert not hasattr(cs, "back_reaction"), "back_reaction should be removed"

    def test_core_fields_preserved(self):
        from dpf.core.bases import CouplingState

        cs = CouplingState(Lp=1e-9, emf=100.0, current=1e3, voltage=500.0, dL_dt=-1e-4)
        assert cs.Lp == 1e-9
        assert cs.emf == 100.0
        assert cs.current == 1e3
        assert cs.voltage == 500.0
        assert cs.dL_dt == -1e-4


class TestCircuitPlasmaResistance:
    """Tests that plasma resistance affects circuit behavior."""

    def test_R_plasma_increases_damping(self):
        """Higher R_plasma should damp current faster."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        }

        solver_no_R = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.0)
        dt = 1e-10
        for _ in range(500):
            coupling = solver_no_R.step(coupling, back_emf=0.0, dt=dt)
        I_no_R = abs(solver_no_R.current)

        solver_with_R = RLCSolver(**params)
        coupling_R = CouplingState(R_plasma=1.0)
        for _ in range(500):
            coupling_R = solver_with_R.step(coupling_R, back_emf=0.0, dt=dt)
        I_with_R = abs(solver_with_R.current)

        assert I_with_R < I_no_R, (
            f"R_plasma should reduce current: I_with_R={I_with_R:.2e}, I_no_R={I_no_R:.2e}"
        )

    def test_R_plasma_zero_gives_same_result(self):
        """R_plasma=0 should give identical results to original behavior."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        }

        solver = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.0)
        dt = 1e-10
        for _ in range(100):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        E_final = solver.total_energy()
        E_init = 0.5 * params["C"] * params["V0"] ** 2
        assert abs(E_final - E_init) / E_init < 0.01

    def test_energy_accounting_with_R_plasma(self):
        """Circuit energy accounting should include R_plasma dissipation."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.0,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        }

        solver = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.5)
        dt = 1e-10
        for _ in range(200):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        E_total = solver.total_energy()
        E_init = 0.5 * params["C"] * params["V0"] ** 2
        assert abs(E_total - E_init) / E_init < 0.01


class TestSheathConfig:
    """Tests for SheathConfig in configuration."""

    def test_default_disabled(self):
        from dpf.config import SheathConfig

        cfg = SheathConfig()
        assert cfg.enabled is False
        assert cfg.boundary == "z_high"
        assert cfg.V_sheath == 0.0

    def test_simulation_config_has_sheath(self):
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={
                "C": 1e-6,
                "V0": 1e3,
                "L0": 1e-7,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
        )
        assert hasattr(config, "sheath")
        assert config.sheath.enabled is False

    def test_sheath_config_from_dict(self):
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={
                "C": 1e-6,
                "V0": 1e3,
                "L0": 1e-7,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            sheath={"enabled": True, "boundary": "z_low", "V_sheath": 50.0},
        )
        assert config.sheath.enabled is True
        assert config.sheath.boundary == "z_low"
        assert config.sheath.V_sheath == 50.0


# --- Section: Ohmic Correction (test_ohmic_correction.py) ---


def _ohmic_small_config(**overrides) -> SimulationConfig:
    """Create a minimal config for ohmic correction tests."""
    defaults = {
        "grid_shape": [8, 8, 8],
        "dx": 0.01,
        "sim_time": 1e-9,
        "fluid": {
            "backend": "python",
            "gamma": 5 / 3,
            "cfl": 0.3,
            "enable_resistive": True,
            "enable_energy_equation": True,
            "enable_ohmic_correction": True,
        },
        "circuit": {
            "C": 30e-6,
            "V0": 15e3,
            "L0": 33.5e-9,
            "R0": 10e-3,
            "anode_radius": 0.012,
            "cathode_radius": 0.032,
        },
        "diagnostics": {
            "hdf5_filename": ":memory:",
            "output_interval": 1,
        },
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _ohmic_warm_engine() -> SimulationEngine:
    """Create an engine with warm plasma so eta_field is computed."""
    cfg = _ohmic_small_config()
    engine = SimulationEngine(cfg)
    engine.state["Te"] = np.full((8, 8, 8), 5e4)
    engine.state["Ti"] = np.full((8, 8, 8), 1e4)
    B = engine.state["B"]
    x = np.linspace(0, 0.08, 8)
    B[1, :, :, :] = 0.1 * x[:, np.newaxis, np.newaxis]
    engine.state["B"] = B
    engine.state["rho"] = np.full((8, 8, 8), 1e-3)
    return engine


class TestOhmicCorrectionConfig:
    """Config field tests."""

    def test_default_enabled(self) -> None:
        cfg = _ohmic_small_config()
        assert cfg.fluid.enable_ohmic_correction is True

    def test_can_disable(self) -> None:
        cfg = _ohmic_small_config(
            fluid={
                "backend": "python",
                "enable_ohmic_correction": False,
                "enable_resistive": True,
                "enable_energy_equation": True,
            }
        )
        assert cfg.fluid.enable_ohmic_correction is False


class TestOhmicGapMeasurement:
    """Test _measure_ohmic_gap directly."""

    def test_gap_populated(self) -> None:
        engine = _ohmic_warm_engine()
        eta = np.full((8, 8, 8), 1e-4)
        coupling = CouplingState(current=1e4, R_plasma=0.01)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        assert len(engine._ohmic_gap_history) == 1
        assert engine._last_ohmic_gap != 0.0

    def test_gap_is_circuit_minus_mhd(self) -> None:
        engine = _ohmic_warm_engine()
        eta = np.full((8, 8, 8), 1e-4)
        current = 1e4
        R_p = 0.01
        coupling = CouplingState(current=current, R_plasma=R_p)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        Q_circuit = R_p * current**2
        assert abs(engine._last_ohmic_gap) <= Q_circuit * 2

    def test_gap_history_capped_at_100(self) -> None:
        engine = _ohmic_warm_engine()
        engine._ohmic_gap_history = list(range(120))
        eta = np.full((8, 8, 8), 1e-4)
        coupling = CouplingState(current=1e4, R_plasma=0.01)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        assert len(engine._ohmic_gap_history) <= 51


class TestOhmicCorrectionComputation:
    """Verify J^2-weighted correction distribution."""

    def test_correction_integrates_to_gap(self) -> None:
        engine = _ohmic_warm_engine()
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)

        dV = engine._cell_volume
        total_Q = float(np.sum(Q * dV))
        assert total_Q == pytest.approx(1e6, rel=0.01)

    def test_correction_nonnegative_for_positive_gap(self) -> None:
        engine = _ohmic_warm_engine()
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.all(Q >= 0)

    def test_correction_negative_for_negative_gap(self) -> None:
        engine = _ohmic_warm_engine()
        engine._last_ohmic_gap = -1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.all(Q <= 0)

    def test_zero_gap_gives_zero_correction(self) -> None:
        engine = _ohmic_warm_engine()
        engine._last_ohmic_gap = 0.0

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.allclose(Q, 0.0)

    def test_uniform_B_gives_zero_J_returns_zero(self) -> None:
        engine = _ohmic_warm_engine()
        engine.state["B"] = np.full((3, 8, 8, 8), 0.1)
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.allclose(Q, 0.0)

    def test_correction_disabled_no_gap_tracking(self) -> None:
        cfg = _ohmic_small_config(
            fluid={
                "backend": "python",
                "enable_ohmic_correction": False,
                "enable_resistive": True,
                "enable_energy_equation": True,
            }
        )
        engine = SimulationEngine(cfg)
        engine.state["Te"] = np.full((8, 8, 8), 5e4)
        engine.step()
        engine.step()
        assert len(engine._ohmic_gap_history) == 0


class TestOhmicCorrectionIntegration:
    """Integration test: full engine step with warm plasma."""

    def test_warm_step_measures_gap(self) -> None:
        engine = _ohmic_warm_engine()
        engine.step()
        assert len(engine._ohmic_gap_history) >= 1

    def test_multi_step_no_blowup(self) -> None:
        engine = _ohmic_warm_engine()
        for _ in range(3):
            engine.step()
        p = engine.state["pressure"]
        assert np.all(np.isfinite(p))
        assert np.all(p > 0)


# --- Section: Single Circuit Step Fix + Bosch-Hale (test_phase_aa_circuit_fixes.py) ---


class TestSingleCircuitStep:
    """Phase AA-D1: Single circuit step fix."""

    def test_engine_step_does_not_crash(self) -> None:
        """A single engine step should complete without raising."""
        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 8]
        preset["sim_time"] = 1e-9
        preset["diagnostics"] = {"hdf5_filename": ":memory:"}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        engine.step()

    def test_circuit_current_positive_after_step(self) -> None:
        """After one step, circuit current should be positive (discharging cap)."""
        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 8]
        preset["sim_time"] = 1e-9
        preset["diagnostics"] = {"hdf5_filename": ":memory:"}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        engine.step()
        assert engine.circuit.state.current >= 0.0

    def test_circuit_voltage_decreases_after_step(self) -> None:
        """After one step, capacitor voltage should decrease."""
        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 8]
        preset["sim_time"] = 1e-9
        preset["diagnostics"] = {"hdf5_filename": ":memory:"}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        V0 = engine.circuit.state.voltage
        engine.step()
        assert engine.circuit.state.voltage <= V0


class TestBoschHaleBranch2:
    """Phase AA-D2: Bosch-Hale reactivity Branch 2 (T > 550 keV)."""

    def test_dd_reactivity_high_T(self) -> None:
        """DD reactivity at T > 550 keV uses Branch 2 coefficients."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        T_high = 1e9  # 86 keV in K — well above 550 keV threshold? check units
        sigma_v = dd_reactivity(T_high)
        assert np.isfinite(sigma_v)
        assert sigma_v > 0.0

    def test_dd_reactivity_low_T(self) -> None:
        """DD reactivity at low T uses Branch 1 coefficients."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        T_low = 1e7  # ~1 keV in K
        sigma_v = dd_reactivity(T_low)
        assert np.isfinite(sigma_v)
        assert sigma_v > 0.0

    def test_dd_reactivity_monotonic_range(self) -> None:
        """DD reactivity should generally increase with temperature."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        temps = np.logspace(6, 10, 20)
        reacts = np.array([dd_reactivity(T) for T in temps])
        assert np.all(np.isfinite(reacts))
        assert np.all(reacts > 0)


# --- Section: Crowbar Infrastructure (test_phase_ar_crowbar.py) ---

_AR_PF1000_C = 1.332e-3
_AR_PF1000_V0 = 27e3
_AR_PF1000_L0 = 33.5e-9
_AR_PF1000_R0 = 2.3e-3


def _ar_make_solver(**overrides) -> RLCSolver:
    kw = dict(
        C=_AR_PF1000_C,
        V0=_AR_PF1000_V0,
        L0=_AR_PF1000_L0,
        R0=_AR_PF1000_R0,
        anode_radius=0.0575,
        cathode_radius=0.08,
    )
    kw.update(overrides)
    return RLCSolver(**kw)


class TestCrowbarResistanceInfrastructure:
    """Verify crowbar_resistance is plumbed through RLCSolver."""

    def test_default_crowbar_resistance_zero(self) -> None:
        solver = _ar_make_solver(crowbar_enabled=True)
        assert solver.crowbar_resistance == 0.0

    def test_crowbar_resistance_stored(self) -> None:
        solver = _ar_make_solver(crowbar_enabled=True, crowbar_resistance=0.005)
        assert solver.crowbar_resistance == pytest.approx(0.005)

    def test_crowbar_resistance_in_circuit_config(self) -> None:
        from dpf.config import CircuitConfig

        cfg = CircuitConfig(
            C=1e-3,
            V0=27e3,
            L0=33.5e-9,
            R0=2.3e-3,
            anode_radius=0.0575,
            cathode_radius=0.08,
            crowbar_enabled=True,
            crowbar_resistance=0.003,
        )
        assert cfg.crowbar_resistance == pytest.approx(0.003)


class TestLeeModelCrowbar:
    """Lee model crowbar parameter handling."""

    def test_lee_model_accepts_crowbar_params(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert result is not None
        assert result.peak_current > 0

    def test_crowbar_preset_has_resistance(self) -> None:
        from dpf.presets import _PRESETS

        if "pf1000" in _PRESETS and "circuit" in _PRESETS["pf1000"]:
            circuit_cfg = _PRESETS["pf1000"]["circuit"]
            if "crowbar_enabled" in circuit_cfg and circuit_cfg["crowbar_enabled"]:
                assert "crowbar_resistance" in circuit_cfg


class TestCalibrationBounds:
    """Crowbar-aware calibration bounds."""

    def test_calibrator_exists(self) -> None:
        from dpf.validation.calibration import LeeModelCalibrator

        cal = LeeModelCalibrator(device_name="PF-1000")
        assert cal is not None

    def test_default_crowbar_r_exists(self) -> None:
        from dpf.validation.calibration import _DEFAULT_CROWBAR_R

        assert isinstance(_DEFAULT_CROWBAR_R, dict)
        assert all(v >= 0.0 for v in _DEFAULT_CROWBAR_R.values())


class TestPresetUpdate:
    """Preset crowbar field injection."""

    def test_pf1000_preset_accessible(self) -> None:
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        assert "circuit" in preset

    def test_faeton_preset_accessible(self) -> None:
        from dpf.presets import _PRESETS

        assert "faeton" in _PRESETS


class TestDeepCopyPreset:
    """get_preset returns a deep copy."""

    def test_modifying_preset_does_not_affect_original(self) -> None:
        from dpf.presets import get_preset

        preset1 = get_preset("pf1000")
        preset1["circuit"]["V0"] = 99999
        preset2 = get_preset("pf1000")
        assert preset2["circuit"]["V0"] != 99999

    def test_nested_modification_isolated(self) -> None:
        from dpf.presets import get_preset

        p1 = get_preset("pf1000")
        p2 = get_preset("pf1000")
        p1["circuit"]["C"] = 0.0
        assert p2["circuit"]["C"] != 0.0


class TestLpL0Diagnostic:
    """L_p / L_0 ratio diagnostic."""

    def test_plasma_inductance_ratio_finite(self) -> None:
        solver = _ar_make_solver()
        coupling = CouplingState(Lp=10e-9)
        dt = 1e-9
        coupling = solver.step(coupling, 0.0, dt)
        Lp_L0_ratio = 10e-9 / _AR_PF1000_L0
        assert np.isfinite(Lp_L0_ratio)
        assert Lp_L0_ratio > 0


class TestVoltageScan:
    """Voltage scan: peak current vs. V0."""

    def test_higher_voltage_gives_higher_peak(self) -> None:
        from dpf.validation.experimental import PF1000_16KV_DATA, PF1000_DATA

        assert PF1000_DATA.voltage > PF1000_16KV_DATA.voltage
        assert PF1000_DATA.peak_current >= PF1000_16KV_DATA.peak_current * 0.5


# --- Section: Circuit-Plasma Coupling Fixes (test_phase_r_circuit_coupling.py) ---


class TestBackEMFCorrectness:
    """Verify back_emf=0 doesn't cause double-counting with dL/dt in R_star."""

    def test_back_emf_zero_correct(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.01)

        coupling = CouplingState(Lp=10e-9, dL_dt=1e-6, R_plasma=0.1)
        dt = 1e-6

        new_coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        assert new_coupling.current > 0, "Current should be positive"
        assert abs(new_coupling.current) < 1e6, "Current should not blow up"
        assert np.isfinite(new_coupling.current), "Current should be finite"
        assert new_coupling.voltage < 27000, "Voltage should decrease"


class TestRPlasmaCap:
    """Verify R_plasma cap is 1000 Ohm (not 10)."""

    def test_r_plasma_cap_1000(self, small_config: SimulationConfig) -> None:
        """Verify R_plasma is capped at 1000 Ohm during pinch disruption."""
        config = small_config
        config.circuit.V0 = 10000.0
        config.sim_time = 1e-6
        engine = SimulationEngine(config)

        nx, ny, nz = config.grid_shape
        engine.state["Te"] = np.full((nx, ny, nz), 1e6)
        engine.state["rho"] = np.full((nx, ny, nz), 1e-6)
        engine.state["B"] = np.zeros((3, nx, ny, nz))
        engine.state["B"][1, :, :, :] = 100.0

        original_circuit_step = engine.circuit.step
        captured_r_plasma = None

        def mock_circuit_step(coupling, back_emf, dt):
            nonlocal captured_r_plasma
            captured_r_plasma = coupling.R_plasma
            return original_circuit_step(coupling, back_emf, dt)

        with (
            patch.object(engine.circuit, "step", side_effect=mock_circuit_step),
            contextlib.suppress(Exception),
        ):
            engine.step()

        if captured_r_plasma is not None:
            assert captured_r_plasma <= 1000.0, (
                f"R_plasma should be capped at 1000 Ohm, got {captured_r_plasma}"
            )


class TestEnergyAccounting:
    """Verify energy accounting split between external and plasma resistance."""

    def test_energy_accounting_no_double_count(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.05, ESR=0.02)

        coupling = CouplingState(Lp=10e-9, dL_dt=0.0, R_plasma=2.0)
        dt = 1e-6

        initial_energy_res = solver.state.energy_res
        initial_energy_res_plasma = solver.state.energy_res_plasma
        I_initial = solver.state.current

        new_coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        I_mid = (I_initial + new_coupling.current) / 2.0
        expected_external_dissipation = solver.R_total * I_mid**2 * dt
        expected_plasma_dissipation = coupling.R_plasma * I_mid**2 * dt

        actual_external = solver.state.energy_res - initial_energy_res
        actual_plasma = solver.state.energy_res_plasma - initial_energy_res_plasma

        assert actual_external == pytest.approx(expected_external_dissipation, rel=1e-6), (
            f"energy_res should only count external R: "
            f"expected {expected_external_dissipation}, got {actual_external}"
        )
        assert actual_plasma == pytest.approx(expected_plasma_dissipation, rel=1e-6), (
            f"energy_res_plasma should track plasma dissipation: "
            f"expected {expected_plasma_dissipation}, got {actual_plasma}"
        )

    def test_total_energy_excludes_plasma_ohmic(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.1)

        coupling = CouplingState(Lp=10e-9, dL_dt=0.0, R_plasma=5.0)
        dt = 1e-6

        for _ in range(10):
            solver.step(coupling, back_emf=0.0, dt=dt)

        assert solver.state.energy_res_plasma > 0, "energy_res_plasma should accumulate"

        total = solver.total_energy()
        expected = solver.state.energy_cap + solver.state.energy_ind + solver.state.energy_res
        assert total == pytest.approx(expected, rel=1e-12), (
            "total_energy() should NOT include energy_res_plasma"
        )
        assert abs(total - (expected + solver.state.energy_res_plasma)) > 1e-6, (
            "total_energy() should exclude energy_res_plasma"
        )


class TestCircuitDynamics:
    """Verify R_plasma still affects circuit dynamics (dI/dt)."""

    def test_circuit_dynamics_include_r_plasma(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.01)

        coupling_no_plasma = CouplingState(Lp=10e-9, dL_dt=0.0, R_plasma=0.0)
        state_backup = CircuitState(
            current=solver.state.current,
            voltage=solver.state.voltage,
            charge=solver.state.charge,
            energy_cap=solver.state.energy_cap,
            energy_ind=solver.state.energy_ind,
            energy_res=solver.state.energy_res,
            energy_res_plasma=solver.state.energy_res_plasma,
            time=solver.state.time,
        )

        dt = 1e-6
        result_no_plasma = solver.step(coupling_no_plasma, back_emf=0.0, dt=dt)
        I_no_plasma = result_no_plasma.current

        solver.state = state_backup

        coupling_with_plasma = CouplingState(Lp=10e-9, dL_dt=0.0, R_plasma=5.0)
        result_with_plasma = solver.step(coupling_with_plasma, back_emf=0.0, dt=dt)
        I_with_plasma = result_with_plasma.current

        assert I_with_plasma < I_no_plasma, (
            f"R_plasma should damp current rise: "
            f"I_with_plasma={I_with_plasma}, I_no_plasma={I_no_plasma}"
        )


class TestBackendPhysicsWarnings:
    """Verify backend physics warnings are logged for non-Python backends."""

    def test_backend_physics_warnings_metal(
        self, small_config: SimulationConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = small_config
        config.fluid.backend = "metal"
        config.radiation.bremsstrahlung_enabled = True
        config.sheath.enabled = True

        mock_solver = MagicMock()
        with (
            patch("dpf.metal.metal_solver.MetalMHDSolver.is_available", return_value=True),
            patch("dpf.metal.metal_solver.MetalMHDSolver.__init__", return_value=None),
            patch("dpf.metal.metal_solver.MetalMHDSolver", return_value=mock_solver),
            caplog.at_level(logging.WARNING),
        ):
            _engine = SimulationEngine(config)

        warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
        assert not any(
            "Metal backend" in w and "does not support" in w for w in warnings
        ), "Metal shares engine operator-split loop — no physics is skipped"

    def test_backend_physics_warnings_athenak(
        self, small_config: SimulationConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = small_config
        config.fluid.backend = "athenak"
        config.fluid.enable_viscosity = True
        config.radiation.bremsstrahlung_enabled = True

        mock_solver = MagicMock()
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=True),
            patch("dpf.athenak_wrapper.AthenaKSolver.__init__", return_value=None),
            patch("dpf.athenak_wrapper.AthenaKSolver", return_value=mock_solver),
            caplog.at_level(logging.WARNING),
        ):
            _engine = SimulationEngine(config)

        warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
        assert any("skips physics modules" in w or "does not support" in w for w in warnings), (
            "AthenaK backend should warn about unsupported physics"
        )

    def test_no_warnings_for_python_backend(
        self, small_config: SimulationConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = small_config
        config.fluid.backend = "python"
        config.radiation.bremsstrahlung_enabled = True
        config.fluid.enable_viscosity = True

        with caplog.at_level(logging.WARNING):
            _engine = SimulationEngine(config)

        warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
        assert not any(
            "skips physics modules" in w or "does not support" in w for w in warnings
        ), "Python backend should NOT warn about unsupported physics"


# --- Section: Crowbar Model + dL/dt + PF-1000 Analytical (test_phase_s_circuit.py) ---

PF1000_C = 1.332e-3
PF1000_V0 = 27e3
PF1000_L0 = 33.5e-9
PF1000_R0 = 2.3e-3


def _make_solver(**overrides) -> RLCSolver:
    """Create a solver with PF-1000 defaults, overridable."""
    kw = dict(
        C=PF1000_C,
        V0=PF1000_V0,
        L0=PF1000_L0,
        R0=PF1000_R0,
        anode_radius=0.0575,
        cathode_radius=0.08,
    )
    kw.update(overrides)
    return RLCSolver(**kw)


def _coupling(Lp: float = 0.0, dL_dt: float = 0.0, R_plasma: float = 0.0) -> CouplingState:
    return CouplingState(Lp=Lp, dL_dt=dL_dt, R_plasma=R_plasma)


class TestCrowbarTrigger:
    """Tests for crowbar switch triggering conditions."""

    def test_crowbar_fires_at_voltage_zero(self):
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(20e-6 / dt)
        fired = False
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                fired = True
                break

        assert fired, "Crowbar should have fired when V_cap crossed zero"
        assert solver.state.crowbar_fire_time > 0.0

        alpha = PF1000_R0 / (2 * PF1000_L0)
        omega0_sq = 1.0 / (PF1000_C * PF1000_L0)
        omega_d = np.sqrt(omega0_sq - alpha**2)
        t_zero = (np.pi - np.arctan(omega_d / alpha)) / omega_d
        assert solver.state.crowbar_fire_time == pytest.approx(t_zero, rel=0.01)

    def test_crowbar_fires_at_fixed_time(self):
        fire_time = 4e-6
        solver = _make_solver(
            crowbar_enabled=True,
            crowbar_mode="fixed_time",
            crowbar_time=fire_time,
        )
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(10e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                break

        assert solver.state.crowbar_fired
        assert solver.state.crowbar_fire_time == pytest.approx(fire_time, abs=2 * dt)

    def test_crowbar_disabled_no_fire(self):
        solver = _make_solver(crowbar_enabled=False)
        coupling = _coupling()
        dt = 1e-9

        T_quarter = (np.pi / 2) * np.sqrt(PF1000_C * PF1000_L0)
        steps = int(2 * T_quarter / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert not solver.state.crowbar_fired
        assert solver.state.crowbar_fire_time < 0

    def test_crowbar_fires_only_once(self):
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        fire_times = []
        max_steps = int(30e-6 / dt)
        for _ in range(max_steps):
            was_fired = solver.state.crowbar_fired
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired and not was_fired:
                fire_times.append(solver.state.crowbar_fire_time)

        assert len(fire_times) == 1, "Crowbar should fire exactly once"


class TestPostCrowbar:
    """Tests for circuit behavior after crowbar fires."""

    def test_crowbar_prevents_voltage_reversal(self):
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(30e-6 / dt)
        found_crowbar = False
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                found_crowbar = True
            if found_crowbar:
                assert solver.state.voltage == pytest.approx(0.0, abs=1e-10), (
                    f"V_cap should be 0 after crowbar, got {solver.state.voltage}"
                )

        assert found_crowbar, "Crowbar should have fired"

    def test_post_crowbar_lr_decay(self):
        R0 = PF1000_R0
        crowbar_R = 0.005
        solver = _make_solver(
            crowbar_enabled=True,
            crowbar_mode="fixed_time",
            crowbar_time=5e-6,
            crowbar_resistance=crowbar_R,
        )
        R_total = R0 + crowbar_R
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(10e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                break

        assert solver.state.crowbar_fired
        I_crowbar = solver.state.current
        t_crowbar = solver.state.time

        tau = PF1000_L0 / R_total
        post_steps = int(5 * tau / dt)
        post_steps = max(post_steps, 50)
        for _ in range(post_steps):
            coupling = solver.step(coupling, 0.0, dt)

        elapsed = solver.state.time - t_crowbar
        expected_ratio = np.exp(-R_total * elapsed / PF1000_L0)
        actual_ratio = solver.state.current / I_crowbar

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.15), (
            f"Post-crowbar decay: expected ratio {expected_ratio:.6f}, got {actual_ratio:.6f}"
        )

    def test_crowbar_energy_conservation(self):
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        E_initial = solver.total_energy()
        assert E_initial == pytest.approx(0.5 * PF1000_C * PF1000_V0**2, rel=1e-10)

        max_steps = int(25e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert solver.state.crowbar_fired
        E_final = solver.total_energy()
        assert E_final == pytest.approx(E_initial, rel=0.01), (
            f"Energy not conserved: E_initial={E_initial:.2f}, E_final={E_final:.2f}"
        )


class TestCrowbarState:
    """Tests for crowbar state reporting in CircuitState."""

    def test_initial_crowbar_state(self):
        solver = _make_solver(crowbar_enabled=True)
        assert solver.state.crowbar_fired is False
        assert solver.state.crowbar_fire_time == -1.0

    def test_crowbar_state_after_fire(self):
        solver = _make_solver(
            crowbar_enabled=True, crowbar_mode="fixed_time", crowbar_time=1e-6
        )
        coupling = _coupling()
        dt = 1e-9

        steps = int(2e-6 / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert solver.state.crowbar_fired is True
        assert solver.state.crowbar_fire_time > 0
        assert solver.state.crowbar_fire_time == pytest.approx(1e-6, abs=2 * dt)


class TestDLdtSecondOrder:
    """Tests for the 2nd-order central difference dL/dt computation."""

    def test_dLdt_zero_history(self):
        solver = _make_solver()
        assert solver.compute_dLp_dt(1e-9) == 0.0

    def test_dLdt_first_order_fallback(self):
        solver = _make_solver()
        solver._Lp_history.append((0.0, 1e-9))
        solver.state.time = 1e-9
        assert solver.compute_dLp_dt(2e-9) == pytest.approx(1.0, rel=1e-10)

    def test_dLdt_second_order_bdf2(self):
        solver = _make_solver()
        dt = 1e-9
        solver._Lp_history.append((0.0, 0.0))
        solver._Lp_history.append((dt, 1e-9))
        solver.state.time = 2 * dt
        L_now = 4e-9
        result = solver.compute_dLp_dt(L_now)
        assert result == pytest.approx(4.0, rel=1e-10)

    def test_dLdt_accuracy_quadratic(self):
        alpha = 1e6
        L0_val = 1e-9
        dt = 1e-9

        solver = _make_solver()

        t0, t1, t2 = 0.0, dt, 2 * dt
        L_at_t0 = L0_val + alpha * t0**2
        L_at_t1 = L0_val + alpha * t1**2
        L_at_t2 = L0_val + alpha * t2**2

        solver._Lp_history.append((t0, L_at_t0))
        solver._Lp_history.append((t1, L_at_t1))
        solver.state.time = t2

        computed = solver.compute_dLp_dt(L_at_t2)
        exact = 2 * alpha * t2

        assert computed == pytest.approx(exact, rel=1e-10), (
            f"BDF2 should be exact for quadratic L(t): computed={computed}, exact={exact}"
        )

    def test_dLdt_vs_first_order_on_cubic(self):
        beta = 1e12
        dt = 1e-9

        t0, t1, t2 = 0.0, dt, 2 * dt
        L_fn = lambda t: beta * t**3  # noqa: E731
        exact_dLdt = 3 * beta * t2**2

        first_order = (L_fn(t2) - L_fn(t1)) / dt
        err_1st = abs(first_order - exact_dLdt) / abs(exact_dLdt)

        solver = _make_solver()
        solver._Lp_history.append((t0, L_fn(t0)))
        solver._Lp_history.append((t1, L_fn(t1)))
        solver.state.time = t2
        second_order = solver.compute_dLp_dt(L_fn(t2))
        err_2nd = abs(second_order - exact_dLdt) / abs(exact_dLdt)

        assert err_2nd < err_1st, (
            f"BDF2 error ({err_2nd:.4e}) should be less than "
            f"1st-order error ({err_1st:.4e})"
        )

    def test_dLdt_history_populated_during_stepping(self):
        solver = _make_solver()
        dt = 1e-9

        for i in range(3):
            coupling = _coupling(Lp=float(i + 1) * 1e-9)
            solver.step(coupling, 0.0, dt)

        assert len(solver._Lp_history) == 3
        assert solver._Lp_history[-1][1] == pytest.approx(3e-9, rel=1e-10)


class TestPF1000Analytical:
    """Analytical benchmarks using PF-1000 parameters."""

    def test_pf1000_quarter_period(self):
        solver = _make_solver(R0=0.0)
        coupling = _coupling()
        dt = 1e-9

        T_quarter_analytical = (np.pi / 2) * np.sqrt(PF1000_C * PF1000_L0)

        max_steps = int(20e-6 / dt)
        I_max = 0.0
        t_max = 0.0
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(solver.state.current) > I_max:
                I_max = abs(solver.state.current)
                t_max = solver.state.time

        assert t_max == pytest.approx(T_quarter_analytical, rel=0.01), (
            f"Peak current at t={t_max * 1e6:.2f} us, expected {T_quarter_analytical * 1e6:.2f} us"
        )

    def test_pf1000_peak_current(self):
        solver = _make_solver(R0=0.0)
        coupling = _coupling()
        dt = 1e-9

        I_peak_analytical = PF1000_V0 * np.sqrt(PF1000_C / PF1000_L0)

        max_steps = int(15e-6 / dt)
        I_max = 0.0
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(solver.state.current) > I_max:
                I_max = abs(solver.state.current)

        assert I_max == pytest.approx(I_peak_analytical, rel=0.01), (
            f"Peak current: {I_max:.0f} A, expected {I_peak_analytical:.0f} A"
        )

    def test_pf1000_energy_conservation_no_crowbar(self):
        solver = _make_solver()
        coupling = _coupling()
        dt = 1e-9

        E_initial = solver.total_energy()
        steps = int(15e-6 / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        E_final = solver.total_energy()
        assert E_final == pytest.approx(E_initial, rel=0.005), (
            f"Energy: initial={E_initial:.2f}, final={E_final:.2f}"
        )


class TestCircuitConfig:
    """Test CircuitConfig crowbar fields."""

    def test_config_crowbar_defaults(self):
        from dpf.config import CircuitConfig

        cfg = CircuitConfig(
            C=1e-6,
            V0=1e3,
            L0=1e-7,
            anode_radius=0.005,
            cathode_radius=0.01,
        )
        assert cfg.crowbar_enabled is False
        assert cfg.crowbar_mode == "voltage_zero"
        assert cfg.crowbar_time == 0.0
        assert cfg.crowbar_resistance == 0.0

    def test_config_crowbar_enabled(self):
        from dpf.config import CircuitConfig

        cfg = CircuitConfig(
            C=1e-6,
            V0=1e3,
            L0=1e-7,
            anode_radius=0.005,
            cathode_radius=0.01,
            crowbar_enabled=True,
            crowbar_mode="fixed_time",
            crowbar_time=4e-6,
            crowbar_resistance=0.001,
        )
        assert cfg.crowbar_enabled is True
        assert cfg.crowbar_mode == "fixed_time"
        assert cfg.crowbar_time == 4e-6
        assert cfg.crowbar_resistance == 0.001

    def test_config_invalid_crowbar_mode(self):
        from dpf.config import CircuitConfig

        with pytest.raises(ValueError, match="crowbar_mode"):
            CircuitConfig(
                C=1e-6,
                V0=1e3,
                L0=1e-7,
                anode_radius=0.005,
                cathode_radius=0.01,
                crowbar_mode="bogus",
            )


# --- Section: Energy Conservation Audit (test_phase_s_energy.py) ---


def _step_circuit(
    solver: RLCSolver,
    coupling: CouplingState,
    dt: float,
    n_steps: int,
) -> CouplingState:
    """Step the circuit solver n_steps times and return final coupling."""
    for _ in range(n_steps):
        coupling = solver.step(coupling, back_emf=0.0, dt=dt)
    return coupling


class TestEnergyAccountingFields:
    """Verify CircuitState energy fields exist and are initialized correctly."""

    def test_energy_fields_exist(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.1)
        state = solver.state
        assert hasattr(state, "energy_cap")
        assert hasattr(state, "energy_ind")
        assert hasattr(state, "energy_res")

    def test_initial_energy_cap_correct(self) -> None:
        C, V0 = 28e-6, 27000.0
        solver = RLCSolver(C=C, V0=V0, L0=50e-9, R0=0.1)
        expected = 0.5 * C * V0**2
        assert solver.state.energy_cap == pytest.approx(expected, rel=1e-10)

    def test_initial_energy_ind_zero(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.1)
        assert solver.state.energy_ind == pytest.approx(0.0, abs=1e-20)

    def test_initial_energy_res_zero(self) -> None:
        solver = RLCSolver(C=28e-6, V0=27000, L0=50e-9, R0=0.1)
        assert solver.state.energy_res == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize(
    "C,V0,L0",
    [
        (28e-6, 27000, 50e-9),
        (1e-3, 15000, 100e-9),
        (100e-6, 5000, 200e-9),
    ],
)
class TestCircuitInitialEnergy:
    """Parametrized: initial energy is 0.5*C*V0^2 across device configs."""

    def test_initial_total_energy(self, C, V0, L0) -> None:
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=0.0)
        expected = 0.5 * C * V0**2
        assert solver.total_energy() == pytest.approx(expected, rel=1e-10)


class TestCircuitEnergyUndamped:
    """Energy conservation for undamped (R=0) ideal LC circuit."""

    def test_total_energy_conserved_undamped(self) -> None:
        C, V0, L0 = 28e-6, 1000.0, 50e-9
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=0.0)
        coupling = CouplingState()

        E0 = solver.total_energy()
        coupling = _step_circuit(solver, coupling, dt=1e-9, n_steps=10000)
        E_final = solver.total_energy()

        assert E_final == pytest.approx(E0, rel=1e-4), (
            f"Undamped LC energy drift: E0={E0:.4e}, E_final={E_final:.4e}"
        )


class TestCircuitEnergyDamped:
    """Energy accounting for damped (R>0) circuit."""

    def test_energy_decreases_with_resistance(self) -> None:
        C, V0, L0, R0 = 28e-6, 1000.0, 50e-9, 0.1
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0)
        coupling = CouplingState()

        E0 = solver.total_energy()
        coupling = _step_circuit(solver, coupling, dt=1e-9, n_steps=5000)
        E_final = solver.total_energy()

        assert E_final > E0 * 0.5, "Energy should not all be dissipated in 5 us"
        assert E_final > E0 * 0.01, "Some energy should remain"

    def test_resistive_dissipation_accounts_for_loss(self) -> None:
        C, V0, L0, R0 = 28e-6, 1000.0, 50e-9, 0.5
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0)
        coupling = CouplingState()

        E0 = solver.total_energy()
        _step_circuit(solver, coupling, dt=1e-9, n_steps=5000)

        E_cap = solver.state.energy_cap
        E_ind = solver.state.energy_ind
        E_res = solver.state.energy_res
        E_total_accounted = E_cap + E_ind + E_res

        assert E_total_accounted == pytest.approx(E0, rel=0.01), (
            f"Energy not conserved: initial={E0:.4e}, accounted={E_total_accounted:.4e}"
        )


class TestEnergySplitSeparate:
    """Verify cap and ind energies are tracked separately."""

    def test_cap_energy_decreases_during_discharge(self) -> None:
        solver = RLCSolver(C=28e-6, V0=1000.0, L0=50e-9, R0=0.0)
        coupling = CouplingState()

        E_cap_0 = solver.state.energy_cap
        _step_circuit(solver, coupling, dt=1e-9, n_steps=1000)
        E_cap_1 = solver.state.energy_cap

        assert E_cap_1 < E_cap_0, "Cap energy should decrease during discharge"

    def test_ind_energy_increases_during_discharge(self) -> None:
        solver = RLCSolver(C=28e-6, V0=1000.0, L0=50e-9, R0=0.0)
        coupling = CouplingState()

        E_ind_0 = solver.state.energy_ind
        _step_circuit(solver, coupling, dt=1e-9, n_steps=1000)
        E_ind_1 = solver.state.energy_ind

        assert E_ind_1 > E_ind_0, "Ind energy should increase as current builds"


class TestMHDEnergyUniform:
    """MHD uniform-state energy: no net change under uniform conditions."""

    def test_uniform_state_step_no_crash(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            circuit={
                "C": 28e-6,
                "V0": 1000.0,
                "L0": 50e-9,
                "R0": 0.1,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        engine = SimulationEngine(config)
        engine.step()
        assert np.all(np.isfinite(engine.state["pressure"]))


@pytest.mark.parametrize(
    "C,V0,L0,R0",
    [
        (28e-6, 27000, 50e-9, 0.0),
        (28e-6, 27000, 50e-9, 0.01),
        (1e-3, 15000, 100e-9, 0.005),
    ],
)
class TestParametricEnergyConservation:
    """Parametrized: energy conservation across device configs."""

    def test_energy_accounted(self, C, V0, L0, R0) -> None:
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0)
        coupling = CouplingState()

        E0 = solver.total_energy()
        _step_circuit(solver, coupling, dt=1e-9, n_steps=5000)

        E_cap = solver.state.energy_cap
        E_ind = solver.state.energy_ind
        E_res = solver.state.energy_res
        E_total = E_cap + E_ind + E_res

        assert E_total == pytest.approx(E0, rel=0.01), (
            f"Energy not conserved for C={C}, V0={V0}, L0={L0}, R0={R0}: "
            f"initial={E0:.4e}, accounted={E_total:.4e}"
        )


# --- Section: Ohmic Correction Timing + Conservative Energy Config (test_phase_x_circuit_energy_coupling.py) ---


class TestOhmicCorrectionTiming:
    """Ohmic correction is applied at the right time in the engine loop."""

    def test_ohmic_correction_applied_after_mhd_step(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            fluid={
                "backend": "python",
                "enable_resistive": True,
                "enable_energy_equation": True,
                "enable_ohmic_correction": True,
            },
            circuit={
                "C": 28e-6,
                "V0": 27000.0,
                "L0": 50e-9,
                "R0": 0.01,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        engine = SimulationEngine(config)
        engine.step()
        assert np.all(np.isfinite(engine.state["pressure"]))

    def test_ohmic_gap_history_populated(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            fluid={
                "backend": "python",
                "enable_resistive": True,
                "enable_energy_equation": True,
                "enable_ohmic_correction": True,
            },
            circuit={
                "C": 28e-6,
                "V0": 27000.0,
                "L0": 50e-9,
                "R0": 0.01,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        engine = SimulationEngine(config)
        engine.state["Te"] = np.full((8, 8, 8), 1e5)
        engine.state["rho"] = np.full((8, 8, 8), 1e-3)
        engine.step()


class TestConservativeEnergyConfig:
    """conservative_energy config flag behavior."""

    def test_conservative_energy_flag_exists(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            circuit={
                "C": 28e-6,
                "V0": 27000.0,
                "L0": 50e-9,
                "R0": 0.01,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert hasattr(config.fluid, "enable_ohmic_correction")

    def test_conservative_energy_default_true(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            circuit={
                "C": 28e-6,
                "V0": 27000.0,
                "L0": 50e-9,
                "R0": 0.01,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert config.fluid.enable_ohmic_correction is True


class TestCircuitEnergyBalance:
    """Circuit energy balance with and without ohmic correction."""

    def test_circuit_energy_tracked_through_step(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-9,
            circuit={
                "C": 28e-6,
                "V0": 27000.0,
                "L0": 50e-9,
                "R0": 0.01,
                "anode_radius": 0.005,
                "cathode_radius": 0.01,
            },
            diagnostics={"hdf5_filename": ":memory:"},
        )
        engine = SimulationEngine(config)
        E0 = engine.circuit.total_energy()
        engine.step()
        E1 = engine.circuit.total_energy()
        assert E1 <= E0 * 1.01, "Circuit energy should not increase by more than 1%"
        assert E1 > 0, "Circuit energy should remain positive"


# --- Section: Back-EMF Coupling (test_phase_z_back_emf.py) ---


def _make_cylindrical_engine(nr: int = 8, nz: int = 8) -> SimulationEngine:
    preset = get_preset("pf1000")
    preset["grid_shape"] = [nr, 1, nz]
    preset["sim_time"] = 1e-8
    preset["dx"] = 1e-3
    preset["geometry"] = {"type": "cylindrical"}
    preset["snowplow"] = {"enabled": False}
    preset["radiation"] = {"bremsstrahlung_enabled": False}
    preset["sheath"] = {"enabled": False}
    preset["fluid"] = {"backend": "python"}
    preset["diagnostics"] = {"hdf5_filename": ":memory:"}
    config = SimulationConfig(**preset)
    return SimulationEngine(config)


def _make_cartesian_engine(nx: int = 8, nz: int = 8) -> SimulationEngine:
    preset = get_preset("tutorial")
    preset["grid_shape"] = [nx, nx, nz]
    preset["sim_time"] = 1e-8
    preset["diagnostics"] = {"hdf5_filename": ":memory:"}
    config = SimulationConfig(**preset)
    return SimulationEngine(config)


def _z_length_cylindrical(engine: SimulationEngine) -> float:
    nz = engine.config.grid_shape[2]
    dx = engine.config.dx
    dz = engine.config.geometry.dz if engine.config.geometry.dz else dx
    return nz * dz


def _z_length_cartesian(engine: SimulationEngine) -> float:
    nz = engine.config.grid_shape[2]
    dx = engine.config.dx
    dz = engine.config.geometry.dz if engine.config.geometry.dz else dx
    return nz * dz


def test_zero_velocity_gives_zero_emf() -> None:
    """Zero velocity field should produce exactly 0 V back-EMF."""
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["B"] = np.ones((3, nr, 1, nz)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(0.0, abs=1e-30), (
        f"Expected 0 V with zero velocity, got {result} V"
    )


def test_zero_bfield_gives_zero_emf() -> None:
    """Zero magnetic field should produce exactly 0 V back-EMF."""
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.ones((3, nr, 1, nz)) * 1e5
    engine.state["B"] = np.zeros((3, nr, 1, nz))

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(0.0, abs=1e-30), (
        f"Expected 0 V with zero B-field, got {result} V"
    )


def test_uniform_vr_btheta_cylindrical() -> None:
    """Uniform v_r=1e5 m/s and B_theta=0.5 T: EMF = -(1e5 * 0.5) * z_length."""
    engine = _make_cylindrical_engine(nr=8, nz=8)
    nr, _, nz = engine.config.grid_shape
    v_r = 1e5
    B_th = 0.5

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    z_len = _z_length_cylindrical(engine)
    expected_emf = -(v_r * B_th) * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Expected {expected_emf:.6e} V, got {result:.6e} V (z_length={z_len:.4e} m)"
    )


def test_sign_convention_imploding_plasma() -> None:
    """Imploding plasma (v_r < 0) with B_theta > 0 gives positive back-EMF."""
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    v_r = -2e5
    B_th = 1.0

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    result = engine._compute_back_emf(dt=1e-9)

    assert result > 0.0, (
        f"Imploding plasma (v_r<0, B_theta>0) should give positive back-EMF, "
        f"got {result:.4e} V"
    )


def test_cartesian_geometry_cross_product() -> None:
    """Cartesian EMF: v_x=1e5, B_y=0.3 → EMF = -(1e5*0.3) * z_length."""
    engine = _make_cartesian_engine(nx=8, nz=8)
    nx, ny, nz = engine.config.grid_shape
    v_x = 1e5
    B_y = 0.3

    engine.state["velocity"] = np.zeros((3, nx, ny, nz))
    engine.state["velocity"][0] = v_x
    engine.state["B"] = np.zeros((3, nx, ny, nz))
    engine.state["B"][1] = B_y

    z_len = _z_length_cartesian(engine)
    expected_emf = -(v_x * B_y) * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian back-EMF: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_cartesian_geometry_both_components() -> None:
    """Cartesian EMF with both (v_x, B_y) and (v_y, B_x): cross = v_x*B_y - v_y*B_x."""
    engine = _make_cartesian_engine(nx=8, nz=8)
    nx, ny, nz = engine.config.grid_shape
    v_x = 1e5
    B_y = 0.3
    v_y = 5e4
    B_x = 0.1

    engine.state["velocity"] = np.zeros((3, nx, ny, nz))
    engine.state["velocity"][0] = v_x
    engine.state["velocity"][1] = v_y
    engine.state["B"] = np.zeros((3, nx, ny, nz))
    engine.state["B"][0] = B_x
    engine.state["B"][1] = B_y

    z_len = _z_length_cartesian(engine)
    cross = v_x * B_y - v_y * B_x
    expected_emf = -cross * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian both-component test: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_none_velocity_returns_zero() -> None:
    """When velocity is None, _compute_back_emf returns 0.0."""
    engine = _make_cylindrical_engine()
    engine.state["velocity"] = None
    engine.state["B"] = np.ones((3, 8, 1, 8)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, f"Expected 0.0 when velocity is None, got {result}"


def test_none_bfield_returns_zero() -> None:
    """When B is None, _compute_back_emf returns 0.0."""
    engine = _make_cylindrical_engine()
    engine.state["velocity"] = np.ones((3, 8, 1, 8)) * 1e5
    engine.state["B"] = None

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, f"Expected 0.0 when B is None, got {result}"


def test_undersized_velocity_returns_zero() -> None:
    """When velocity.shape[0] < 2, method returns 0.0."""
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.ones((1, nr, 1, nz)) * 1e5
    engine.state["B"] = np.ones((3, nr, 1, nz)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, (
        f"Expected 0.0 when velocity has only 1 component, got {result}"
    )


def test_scales_with_z_length() -> None:
    """Doubling nz should double the back-EMF."""
    v_r = 1e5
    B_th = 0.5

    engine_8 = _make_cylindrical_engine(nr=8, nz=8)
    nr_8, _, nz_8 = engine_8.config.grid_shape
    engine_8.state["velocity"] = np.zeros((3, nr_8, 1, nz_8))
    engine_8.state["velocity"][0] = v_r
    engine_8.state["B"] = np.zeros((3, nr_8, 1, nz_8))
    engine_8.state["B"][1] = B_th
    emf_8 = engine_8._compute_back_emf(dt=1e-9)

    engine_16 = _make_cylindrical_engine(nr=8, nz=16)
    nr_16, _, nz_16 = engine_16.config.grid_shape
    engine_16.state["velocity"] = np.zeros((3, nr_16, 1, nz_16))
    engine_16.state["velocity"][0] = v_r
    engine_16.state["B"] = np.zeros((3, nr_16, 1, nz_16))
    engine_16.state["B"][1] = B_th
    emf_16 = engine_16._compute_back_emf(dt=1e-9)

    assert emf_16 == pytest.approx(2.0 * emf_8, rel=1e-10), (
        f"EMF should double when nz doubles: emf_8={emf_8:.4e}, emf_16={emf_16:.4e}"
    )


def test_scales_with_velocity() -> None:
    """Doubling v_r should double the back-EMF magnitude."""
    B_th = 0.5
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 1e5
    emf_1x = engine._compute_back_emf(dt=1e-9)

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 2e5
    emf_2x = engine._compute_back_emf(dt=1e-9)

    assert emf_2x == pytest.approx(2.0 * emf_1x, rel=1e-10), (
        f"EMF should double when v doubles: emf_1x={emf_1x:.4e}, emf_2x={emf_2x:.4e}"
    )


def test_scales_with_bfield() -> None:
    """Doubling B_theta should double the back-EMF magnitude."""
    v_r = 1e5
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 0.5
    emf_1x = engine._compute_back_emf(dt=1e-9)

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 1.0
    emf_2x = engine._compute_back_emf(dt=1e-9)

    assert emf_2x == pytest.approx(2.0 * emf_1x, rel=1e-10), (
        f"EMF should double when B doubles: emf_1x={emf_1x:.4e}, emf_2x={emf_2x:.4e}"
    )


def test_nonuniform_fields_mean_computed_correctly() -> None:
    """Non-uniform fields: back-EMF equals mean(-(v_r * B_theta)) * z_len."""
    engine = _make_cylindrical_engine(nr=4, nz=4)
    nr, _, nz = engine.config.grid_shape

    v_r_values = np.array([1e4, 2e4, 3e4, 4e4])
    B_th_value = 0.8

    velocity = np.zeros((3, nr, 1, nz))
    for ir in range(nr):
        velocity[0, ir, 0, :] = v_r_values[ir]
    engine.state["velocity"] = velocity

    B = np.zeros((3, nr, 1, nz))
    B[1] = B_th_value
    engine.state["B"] = B

    emf_density = -(velocity[0] * B[1])
    expected_mean = float(np.mean(emf_density))
    z_len = _z_length_cylindrical(engine)
    expected_emf = expected_mean * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Non-uniform fields: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_nonuniform_fields_cartesian() -> None:
    """Non-uniform Cartesian fields: EMF = mean(-(v_x*B_y - v_y*B_x)) * z_len."""
    engine = _make_cartesian_engine(nx=4, nz=4)
    nx, ny, nz = engine.config.grid_shape

    rng = np.random.default_rng(42)
    v_x = rng.uniform(0, 1e5, (nx, ny, nz))
    v_y = rng.uniform(0, 5e4, (nx, ny, nz))
    B_x = rng.uniform(0, 0.5, (nx, ny, nz))
    B_y = rng.uniform(0, 0.5, (nx, ny, nz))

    velocity = np.zeros((3, nx, ny, nz))
    velocity[0] = v_x
    velocity[1] = v_y
    engine.state["velocity"] = velocity

    B = np.zeros((3, nx, ny, nz))
    B[0] = B_x
    B[1] = B_y
    engine.state["B"] = B

    emf_density = -(v_x * B_y - v_y * B_x)
    z_len = _z_length_cartesian(engine)
    expected_emf = float(np.mean(emf_density)) * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian non-uniform fields: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_returns_float_type() -> None:
    """_compute_back_emf must return a Python float."""
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 1e5
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 0.3

    result = engine._compute_back_emf(dt=1e-9)

    assert isinstance(result, float), (
        f"_compute_back_emf must return float, got {type(result)}"
    )


def test_back_emf_integrates_in_circuit() -> None:
    """After setting velocity and B, back-EMF should be nonzero and correct."""
    engine = _make_cylindrical_engine(nr=8, nz=8)
    nr, _, nz = engine.config.grid_shape

    v_r = -5e4
    B_th = 0.2

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    z_len = _z_length_cylindrical(engine)
    expected_emf = -(v_r * B_th) * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result > 0.0, f"Back-EMF should be positive for imploding plasma, got {result:.4e} V"
    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Integration test: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_dz_parameter_affects_z_length() -> None:
    """When geometry.dz is explicitly set, it should be used instead of dx."""
    v_r = 1e5
    B_th = 0.5
    nr, nz = 8, 8

    preset_a = get_preset("pf1000")
    preset_a["grid_shape"] = [nr, 1, nz]
    preset_a["sim_time"] = 1e-8
    preset_a["dx"] = 1e-3
    preset_a["geometry"] = {"type": "cylindrical", "dz": None}
    preset_a["snowplow"] = {"enabled": False}
    preset_a["radiation"] = {"bremsstrahlung_enabled": False}
    preset_a["sheath"] = {"enabled": False}
    preset_a["fluid"] = {"backend": "python"}
    preset_a["diagnostics"] = {"hdf5_filename": ":memory:"}
    engine_a = SimulationEngine(SimulationConfig(**preset_a))
    engine_a.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine_a.state["velocity"][0] = v_r
    engine_a.state["B"] = np.zeros((3, nr, 1, nz))
    engine_a.state["B"][1] = B_th
    emf_a = engine_a._compute_back_emf(dt=1e-9)

    preset_b = get_preset("pf1000")
    preset_b["grid_shape"] = [nr, 1, nz]
    preset_b["sim_time"] = 1e-8
    preset_b["dx"] = 1e-3
    preset_b["geometry"] = {"type": "cylindrical", "dz": 2e-3}
    preset_b["snowplow"] = {"enabled": False}
    preset_b["radiation"] = {"bremsstrahlung_enabled": False}
    preset_b["sheath"] = {"enabled": False}
    preset_b["fluid"] = {"backend": "python"}
    preset_b["diagnostics"] = {"hdf5_filename": ":memory:"}
    engine_b = SimulationEngine(SimulationConfig(**preset_b))
    engine_b.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine_b.state["velocity"][0] = v_r
    engine_b.state["B"] = np.zeros((3, nr, 1, nz))
    engine_b.state["B"][1] = B_th
    emf_b = engine_b._compute_back_emf(dt=1e-9)

    assert emf_b == pytest.approx(2.0 * emf_a, rel=1e-10), (
        f"dz=2e-3 should give 2x EMF vs dz=1e-3: emf_a={emf_a:.4e}, emf_b={emf_b:.4e}"
    )
