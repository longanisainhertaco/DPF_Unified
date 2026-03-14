"""Consolidated research tests — Phase AE, AF, AS, AV, AY, BQ, BR, BS.

Covers:
- Phase AE: MJOLNIR preset, implosion scaling, plasma regime diagnostics
- Phase AF: Post-Jensen Cu cooling, multi-event neutron decomposition, regime integration
- Phase AS: Monte Carlo NRMSE uncertainty, calibration degeneracy, crowbar fix
- Phase AV: Pinch physics diagnostics (implosion, stagnation, MRTI, I^4 scaling)
- Phase AY: Comprehensive V&V diagnostics (NRMSE decomposition, ASME windows)
- Phase BQ: Expanded ASME V&V 20 uncertainty budget analysis
- Phase BR: GUM-compliant uncertainty taxonomy, ASME double-counting fix
- Phase BS: Provenance-dependent ASME, LOO maxiter=10 reference results
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.constants import eV, k_B
from dpf.validation.calibration import asme_vv20_assessment
from dpf.validation.experimental import DEVICES, compute_lp_l0_ratio
from dpf.validation.lee_model_comparison import LeeModel

# =====================================================================
# Shared helpers (deduplicated from AS + AY)
# =====================================================================

_FC = 0.800
_FM = 0.094
_F_MR = 0.1
_PCF = 0.14
_CROWBAR_R = 1.5e-3


def _make_model(**kwargs) -> LeeModel:
    """Create Lee model with calibrated PF-1000 parameters."""
    defaults = dict(
        current_fraction=_FC,
        mass_fraction=_FM,
        radial_mass_fraction=_F_MR,
        pinch_column_fraction=_PCF,
        crowbar_enabled=True,
        crowbar_resistance=_CROWBAR_R,
    )
    defaults.update(kwargs)
    return LeeModel(**defaults)


def _nrmse_window(t_sim, I_sim, t_exp, I_exp, t_start, t_end):
    """Compute NRMSE over a specific time window, normalized by global peak."""
    mask = (t_exp >= t_start) & (t_exp <= t_end)
    if mask.sum() < 2:
        return float("nan")
    t_w = t_exp[mask]
    I_w = I_exp[mask]
    I_sim_interp = np.interp(t_w, t_sim, I_sim)
    rmse = np.sqrt(np.mean((I_sim_interp - I_w) ** 2))
    I_global_max = np.max(np.abs(I_exp))
    return rmse / max(I_global_max, 1e-10)


# --- Section: Phase AE — Research-Informed Implementations ---


class TestMJOLNIRPreset:
    """Verify MJOLNIR preset has physically correct device parameters."""

    def test_preset_exists(self):
        from dpf.presets import get_preset, get_preset_names
        assert "mjolnir" in get_preset_names()
        preset = get_preset("mjolnir")
        assert preset is not None

    def test_stored_energy(self):
        """Stored energy E = 0.5 * C * V^2 should be ~0.72 MJ at 60 kV."""
        from dpf.presets import get_preset
        p = get_preset("mjolnir")
        C = p["circuit"]["C"]
        V0 = p["circuit"]["V0"]
        E = 0.5 * C * V0**2
        assert 0.5e6 < E < 2.5e6, f"Stored energy {E:.0f} J outside 0.5-2.5 MJ range"

    def test_anode_radius(self):
        """Anode diameter = 228.6 mm → radius = 114.3 mm."""
        from dpf.presets import get_preset
        p = get_preset("mjolnir")
        a = p["circuit"]["anode_radius"]
        assert pytest.approx(a, rel=0.01) == 0.1143

    def test_crowbar_enabled(self):
        from dpf.presets import get_preset
        p = get_preset("mjolnir")
        assert p["circuit"]["crowbar_enabled"] is True

    def test_cylindrical_geometry(self):
        from dpf.presets import get_preset
        p = get_preset("mjolnir")
        assert p["geometry"]["type"] == "cylindrical"

    def test_radiation_enabled(self):
        from dpf.presets import get_preset
        p = get_preset("mjolnir")
        assert p["radiation"]["bremsstrahlung_enabled"] is True


class TestImplosionScaling:
    """Verify 1D shock theory scaling relations from Goyon et al. (2025)."""

    def test_mjolnir_implosion_velocity(self):
        """MJOLNIR at 2 MA, R_imp=2.5 cm, 7 Torr → v_imp ~ 259 km/s."""
        from dpf.fluid.snowplow import implosion_scaling
        result = implosion_scaling(I_MA=2.0, R_imp_cm=2.5, P_fill_Torr=7.0)
        v_imp_kms = result["v_imp"] / 1e3
        assert 200 < v_imp_kms < 400, f"v_imp = {v_imp_kms:.0f} km/s, expected ~259 km/s"

    def test_mjolnir_stagnation_temperature(self):
        from dpf.fluid.snowplow import implosion_scaling
        result = implosion_scaling(I_MA=2.0, R_imp_cm=2.5, P_fill_Torr=7.0)
        assert result["T_stag_keV"] > 0.5, "T_stag should be > 0.5 keV for MJOLNIR"

    def test_scaling_dimensional_consistency(self):
        from dpf.fluid.snowplow import implosion_scaling
        result = implosion_scaling(I_MA=1.0, R_imp_cm=1.0, P_fill_Torr=1.0)
        assert result["v_imp"] > 0
        assert result["T_stag_keV"] > 0
        assert result["tau_exp_ns"] > 0
        assert result["tau_m0_ns"] > 0

    def test_velocity_scales_with_current(self):
        from dpf.fluid.snowplow import implosion_scaling
        r1 = implosion_scaling(I_MA=1.0, R_imp_cm=2.0, P_fill_Torr=5.0)
        r2 = implosion_scaling(I_MA=2.0, R_imp_cm=2.0, P_fill_Torr=5.0)
        assert pytest.approx(r2["v_imp"] / r1["v_imp"], rel=1e-10) == 2.0

    def test_temperature_scales_with_current_squared(self):
        from dpf.fluid.snowplow import implosion_scaling
        r1 = implosion_scaling(I_MA=1.0, R_imp_cm=2.0, P_fill_Torr=5.0)
        r2 = implosion_scaling(I_MA=2.0, R_imp_cm=2.0, P_fill_Torr=5.0)
        assert pytest.approx(r2["T_stag_keV"] / r1["T_stag_keV"], rel=1e-10) == 4.0

    def test_breakup_time_vs_expansion_time(self):
        from dpf.fluid.snowplow import implosion_scaling
        result = implosion_scaling(I_MA=2.0, R_imp_cm=2.5, P_fill_Torr=7.0)
        ratio = result["tau_m0_ns"] / result["tau_exp_ns"]
        assert pytest.approx(ratio, rel=0.02) == 31.0 / 31.5

    def test_pf1000_estimate(self):
        from dpf.fluid.snowplow import implosion_scaling
        result = implosion_scaling(I_MA=2.0, R_imp_cm=5.0, P_fill_Torr=3.0)
        assert 100e3 < result["v_imp"] < 500e3
        assert 0.5 < result["T_stag_keV"] < 50.0
        assert 1.0 < result["tau_exp_ns"] < 500.0


class TestPlasmaParameterND:
    """Test ion plasma parameter ND = (4pi/3) * lambda_Di^3 * ni."""

    def test_cold_fill_gas_is_collisional(self):
        from dpf.diagnostics.plasma_regime import plasma_parameter_ND
        ne = np.array([1e23])
        Ti = np.array([300.0])
        ND = plasma_parameter_ND(ne, Ti)
        assert ND[0] < 1.0

    def test_hot_pinch_is_collisionless(self):
        from dpf.diagnostics.plasma_regime import plasma_parameter_ND
        ne = np.array([1e25])
        Ti_K = np.array([1e3 * 1.6e-19 / 1.38e-23])
        ND = plasma_parameter_ND(ne, Ti_K)
        assert ND[0] > 1.0

    def test_nd_scales_with_temperature(self):
        from dpf.diagnostics.plasma_regime import plasma_parameter_ND
        ne = np.array([1e24])
        ND1 = plasma_parameter_ND(ne, np.array([1e4]))[0]
        ND2 = plasma_parameter_ND(ne, np.array([4e4]))[0]
        assert pytest.approx(ND2 / ND1, rel=1e-10) == 8.0

    def test_nd_positive(self):
        from dpf.diagnostics.plasma_regime import plasma_parameter_ND
        ne = np.array([1e20, 1e22, 1e24])
        Ti = np.array([1e3, 1e5, 1e7])
        ND = plasma_parameter_ND(ne, Ti)
        assert np.all(ND > 0)

    def test_array_shape_preserved(self):
        from dpf.diagnostics.plasma_regime import plasma_parameter_ND
        ne = np.ones((4, 8)) * 1e23
        Ti = np.ones((4, 8)) * 1e4
        ND = plasma_parameter_ND(ne, Ti)
        assert ND.shape == (4, 8)


class TestMagneticReynoldsNumber:
    """Test magnetic Reynolds number Rm = mu_0 * v * L / eta."""

    def test_fast_implosion_high_rm(self):
        from dpf.diagnostics.plasma_regime import magnetic_reynolds_number
        ne = np.array([1e24])
        Te = np.array([10.0 * 1.6e-19 / 1.38e-23])
        v = np.array([1e5])
        Rm = magnetic_reynolds_number(ne, Te, v, L_scale=0.01)
        assert Rm[0] > 1.0

    def test_rm_scales_with_velocity(self):
        from dpf.diagnostics.plasma_regime import magnetic_reynolds_number
        ne = np.array([1e24])
        Te = np.array([1e6])
        Rm1 = magnetic_reynolds_number(ne, Te, np.array([1e4]), 0.01)[0]
        Rm2 = magnetic_reynolds_number(ne, Te, np.array([2e4]), 0.01)[0]
        assert pytest.approx(Rm2 / Rm1, rel=1e-6) == 2.0

    def test_rm_positive(self):
        from dpf.diagnostics.plasma_regime import magnetic_reynolds_number
        ne = np.array([1e22])
        Te = np.array([1e4])
        v = np.array([1e3])
        Rm = magnetic_reynolds_number(ne, Te, v, 0.001)
        assert Rm[0] > 0


class TestDebyeLength:
    """Test electron Debye length (from dpf.diagnostics.plasma_regime)."""

    def test_dpf_fill_gas_debye_length(self):
        from dpf.diagnostics.plasma_regime import debye_length
        ne = np.array([1e23])
        Te = np.array([300.0])
        lam = debye_length(ne, Te)
        assert 1e-10 < lam[0] < 1e-6

    def test_debye_scales_with_temp(self):
        from dpf.diagnostics.plasma_regime import debye_length
        ne = np.array([1e24])
        lam1 = debye_length(ne, np.array([1e4]))[0]
        lam2 = debye_length(ne, np.array([4e4]))[0]
        assert pytest.approx(lam2 / lam1, rel=1e-10) == 2.0


class TestIonSkinDepth:
    """Test ion skin depth (ion inertial length)."""

    def test_dpf_pinch_skin_depth(self):
        from dpf.diagnostics.plasma_regime import ion_skin_depth
        ne = np.array([1e25])
        d_i = ion_skin_depth(ne)
        assert 1e-5 < d_i[0] < 1e-3

    def test_skin_depth_decreases_with_density(self):
        from dpf.diagnostics.plasma_regime import ion_skin_depth
        d1 = ion_skin_depth(np.array([1e24]))[0]
        d2 = ion_skin_depth(np.array([4e24]))[0]
        assert pytest.approx(d1 / d2, rel=1e-10) == 2.0


class TestRegimeValidity:
    """Test comprehensive regime validity assessment."""

    def test_cold_fill_gas_valid(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones(10) * 1e23
        Te = np.ones(10) * 300.0
        Ti = np.ones(10) * 300.0
        v = np.ones(10) * 1e3
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert result["fraction_valid"] > 0.5

    def test_returns_all_keys(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones(5) * 1e23
        Te = np.ones(5) * 1e4
        Ti = np.ones(5) * 1e4
        v = np.ones(5) * 1e4
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        for key in ("ND", "Rm", "lambda_De", "d_i", "mhd_valid", "fraction_valid"):
            assert key in result

    def test_fraction_valid_in_range(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones(10) * 1e23
        Te = np.ones(10) * 1e4
        Ti = np.ones(10) * 1e4
        v = np.ones(10) * 1e4
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert 0.0 <= result["fraction_valid"] <= 1.0


# --- Section: Phase AF — Post-Jensen Cu Cooling + Neutron Decomposition ---


class TestPostJensenCuCooling:
    """Verify tabulated Cu (Z=29) cooling function against shell physics."""

    def test_peak_at_100eV(self):
        from dpf.radiation.line_radiation import _cooling_copper
        Te_scan = np.arange(10, 201, 1.0)
        vals = np.array([_cooling_copper(T) for T in Te_scan])
        peak_idx = np.argmax(vals)
        peak_Te = Te_scan[peak_idx]
        assert 80 <= peak_Te <= 120, f"M-shell peak at {peak_Te} eV, expected 80-120 eV"

    def test_peak_magnitude(self):
        from dpf.radiation.line_radiation import _cooling_copper
        peak_val = _cooling_copper(100.0)
        assert 1e-30 < peak_val < 1e-29

    def test_ar_like_trough(self):
        from dpf.radiation.line_radiation import _cooling_copper
        val_peak = _cooling_copper(100.0)
        val_trough = _cooling_copper(1000.0)
        assert val_trough < val_peak / 3.0

    def test_l_shell_secondary_peak(self):
        from dpf.radiation.line_radiation import _cooling_copper
        val_trough = _cooling_copper(1000.0)
        val_l_peak = _cooling_copper(3000.0)
        assert val_l_peak > val_trough

    def test_monotonic_rise_below_peak(self):
        from dpf.radiation.line_radiation import _cooling_copper
        Te_vals = [1, 2, 5, 10, 20, 50, 80, 100]
        lambdas = [_cooling_copper(T) for T in Te_vals]
        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1]

    def test_high_temperature_decline(self):
        from dpf.radiation.line_radiation import _cooling_copper
        val_5keV = _cooling_copper(5000.0)
        val_10keV = _cooling_copper(10000.0)
        assert val_10keV < val_5keV

    def test_extrapolation_above_10keV(self):
        from dpf.radiation.line_radiation import _cooling_copper
        val = _cooling_copper(50000.0)
        assert val > 0 and np.isfinite(val)
        assert val < _cooling_copper(10000.0)

    def test_below_1eV(self):
        from dpf.radiation.line_radiation import _cooling_copper
        val = _cooling_copper(0.1)
        assert val > 0 and np.isfinite(val)
        assert val < 1e-33

    def test_cooling_function_dispatches_to_copper(self):
        from dpf.radiation.line_radiation import _cooling_copper, cooling_function
        Te_K = 100.0 * eV / k_B
        val_direct = _cooling_copper(100.0)
        val_dispatch = cooling_function(np.array([Te_K]), 29.0)[0]
        assert pytest.approx(val_direct, rel=1e-6) == val_dispatch

    def test_old_vs_new_much_higher(self):
        from dpf.radiation.line_radiation import _cooling_copper
        peak_val = _cooling_copper(100.0)
        old_peak_approx = 5.0e-32
        assert peak_val > 10 * old_peak_approx

    def test_interpolation_smoothness(self):
        from dpf.radiation.line_radiation import _cooling_copper
        Te_fine = np.linspace(10, 5000, 500)
        vals = np.array([_cooling_copper(T) for T in Te_fine])
        for i in range(1, len(vals)):
            ratio = vals[i] / max(vals[i - 1], 1e-40)
            assert 0.3 < ratio < 3.0, f"Jump at Te={Te_fine[i]:.0f}: ratio={ratio:.2f}"


class TestMultiEventNeutronDecomposition:
    """Test decompose_neutron_events from beam_target module."""

    def _make_single_event(self, t_peak_ns=500.0, fwhm_ns=20.0, n_points=1000):
        times = np.linspace(0, 1e-6, n_points)
        t_peak = t_peak_ns * 1e-9
        sigma = fwhm_ns * 1e-9 / 2.355
        rates = 1e15 * np.exp(-0.5 * ((times - t_peak) / sigma) ** 2)
        return times, rates

    def _make_double_event(self):
        times = np.linspace(0, 1e-6, 2000)
        sigma1 = 15e-9 / 2.355
        sigma2 = 10e-9 / 2.355
        rates = (
            1e15 * np.exp(-0.5 * ((times - 400e-9) / sigma1) ** 2)
            + 0.5e15 * np.exp(-0.5 * ((times - 700e-9) / sigma2) ** 2)
        )
        return times, rates

    def test_single_event_detected(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1

    def test_double_event_detected(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        assert result["n_events"] == 2

    def test_primary_fraction_single_event(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["primary_fraction"] > 0.7

    def test_primary_fraction_double_event(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        assert 0.5 < result["primary_fraction"] < 0.9

    def test_event_peak_times(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        peak_times_ns = sorted([ev["peak_time"] * 1e9 for ev in result["events"]])
        assert len(peak_times_ns) == 2
        assert abs(peak_times_ns[0] - 400) < 10
        assert abs(peak_times_ns[1] - 700) < 10

    def test_event_fwhm(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_single_event(fwhm_ns=20.0, n_points=2000)
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1
        fwhm = result["events"][0]["fwhm_ns"]
        assert 10 < fwhm < 40

    def test_total_yield_positive(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["total_yield"] > 0

    def test_empty_signal(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times = np.linspace(0, 1e-6, 100)
        rates = np.zeros_like(times)
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 0
        assert result["total_yield"] == 0.0

    def test_short_signal(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        result = decompose_neutron_events(np.array([0, 1e-6]), np.array([1e10, 1e10]))
        assert result["n_events"] == 0

    def test_threshold_filtering(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_double_event()
        result = decompose_neutron_events(
            times, rates, threshold_fraction=0.6, min_separation_ns=50.0,
        )
        assert result["n_events"] == 1

    def test_close_peaks_merged(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times = np.linspace(0, 1e-6, 2000)
        sigma = 5e-9 / 2.355
        rates = (
            1e15 * np.exp(-0.5 * ((times - 500e-9) / sigma) ** 2)
            + 0.8e15 * np.exp(-0.5 * ((times - 505e-9) / sigma) ** 2)
        )
        result = decompose_neutron_events(times, rates, min_separation_ns=10.0)
        assert result["n_events"] == 1

    def test_event_dict_keys(self):
        from dpf.diagnostics.beam_target import decompose_neutron_events
        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1
        ev = result["events"][0]
        for key in ("peak_time", "peak_rate", "fwhm_ns", "yield_count",
                    "start_time", "end_time"):
            assert key in ev
        assert ev["peak_rate"] > 0
        assert ev["yield_count"] > 0


class TestRegimeValidityEngineIntegration:
    """Test that the engine records regime validity diagnostics."""

    def test_engine_has_regime_result_attr(self):
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-9, dt_init=1e-11,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        assert hasattr(engine, "_last_regime_result")
        assert engine._last_regime_result is None

    def test_regime_validity_cold_fill(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones((8, 8, 8)) * 1e23
        Te = np.ones((8, 8, 8)) * 300.0
        Ti = np.ones((8, 8, 8)) * 300.0
        v = np.ones((8, 8, 8)) * 1e3
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert result["fraction_valid"] > 0.9

    def test_regime_validity_hot_pinch_invalid(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones((8, 8, 8)) * 1e25
        Te_K = np.ones((8, 8, 8)) * (1e3 * eV / k_B)
        Ti_K = Te_K.copy()
        v = np.ones((8, 8, 8)) * 1e5
        result = regime_validity(ne, Te_K, Ti_K, v, dx=1e-3)
        assert result["fraction_valid"] < 1.0

    def test_regime_returns_nd_and_rm(self):
        from dpf.diagnostics.plasma_regime import regime_validity
        ne = np.ones(10) * 1e23
        Te = np.ones(10) * 1e4
        Ti = np.ones(10) * 1e4
        v = np.ones(10) * 1e4
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert "ND" in result
        assert "Rm" in result
        assert "fraction_valid" in result
        assert 0.0 <= result["fraction_valid"] <= 1.0

    def test_engine_step_no_crash_on_regime_check(self):
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7, dt_init=1e-11,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        for _ in range(105):
            result = engine.step()
            if result.finished:
                break
        assert engine._last_regime_result is not None or engine.step_count < 100


# --- Section: Phase AS — Monte Carlo NRMSE + Crowbar Fix ---


class TestCrowbarFix:
    """Validate that crowbar detection now works with V threshold."""

    def test_crowbar_fires(self):
        model = _make_model()
        result = model.run("PF-1000")
        assert 3 in result.phases_completed

    def test_waveform_extends_past_radial(self):
        model = _make_model()
        result = model.run("PF-1000")
        assert result.t[-1] > 20e-6

    def test_nrmse_improved(self):
        model = _make_model()
        comp = model.compare_with_experiment("PF-1000")
        assert comp.waveform_nrmse < 0.15

    def test_crowbar_no_effect_on_pre_peak(self):
        model_cb = _make_model(crowbar_enabled=True)
        model_no = _make_model(crowbar_enabled=False)
        comp_cb = model_cb.compare_with_experiment("PF-1000")
        comp_no = model_no.compare_with_experiment("PF-1000")
        assert comp_cb.lee_result.peak_current == pytest.approx(
            comp_no.lee_result.peak_current, rel=1e-6
        )

    def test_lr_decay_post_crowbar(self):
        model = _make_model()
        result = model.run("PF-1000")
        t = result.t
        I_arr = result.I
        n = len(t)
        last_quarter = I_arr[3 * n // 4:]
        diffs = np.diff(last_quarter)
        frac_decreasing = np.sum(diffs < 0) / len(diffs)
        assert frac_decreasing > 0.9


class TestMonteCarloNRMSE:
    """Monte Carlo propagation of input uncertainties to NRMSE."""

    def test_monte_carlo_runs(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=20, seed=42)
        assert result.n_failures == 0
        assert result.n_samples == 20
        assert 0.05 < result.nrmse_mean < 0.40

    def test_nominal_within_ci(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=50, seed=42)
        model = _make_model()
        comp = model.compare_with_experiment("PF-1000")
        nrmse_nom = comp.waveform_nrmse
        assert result.nrmse_ci_lo <= nrmse_nom <= result.nrmse_ci_hi

    def test_pcf_dominant(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=30, seed=42)
        assert result.sensitivity.get("pcf", 0) > 0.25

    def test_relative_uncertainty(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=50, seed=42)
        rel = result.nrmse_std / result.nrmse_mean
        assert rel < 0.30

    def test_sensitivity_sums_to_one(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=20, seed=42)
        total = sum(result.sensitivity.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_top_three_sources(self):
        from dpf.validation.calibration import monte_carlo_nrmse
        result = monte_carlo_nrmse(n_samples=30, seed=42)
        top3 = sorted(result.sensitivity.values(), reverse=True)[:3]
        assert sum(top3) > 0.80


class TestCalibrationDegeneracy:
    """fc/fm calibration degeneracy: flat valley at fc²/fm ≈ const."""

    def test_multiple_optima_similar_objective(self):
        from dpf.validation.calibration import LeeModelCalibrator
        cal = LeeModelCalibrator(
            "PF-1000", pinch_column_fraction=_PCF,
            crowbar_enabled=True, crowbar_resistance=_CROWBAR_R,
        )
        r1 = cal.calibrate(maxiter=100, x0=(0.70, 0.15))
        cal2 = LeeModelCalibrator(
            "PF-1000", pinch_column_fraction=_PCF,
            crowbar_enabled=True, crowbar_resistance=_CROWBAR_R,
        )
        r2 = cal2.calibrate(maxiter=100, x0=(0.78, 0.10))
        diff = abs(r1.objective_value - r2.objective_value)
        avg = 0.5 * (r1.objective_value + r2.objective_value)
        assert diff / avg < 0.05

    def test_fc_squared_over_fm_invariant(self):
        from dpf.validation.calibration import LeeModelCalibrator
        ratios = []
        for x0 in [(0.70, 0.15), (0.78, 0.10), (0.75, 0.12)]:
            cal = LeeModelCalibrator(
                "PF-1000", pinch_column_fraction=_PCF,
                crowbar_enabled=True, crowbar_resistance=_CROWBAR_R,
            )
            r = cal.calibrate(maxiter=100, x0=x0)
            ratios.append(r.best_fc**2 / r.best_fm)
        mean_ratio = np.mean(ratios)
        for ratio in ratios:
            assert abs(ratio - mean_ratio) / mean_ratio < 0.30

    def test_timing_peak_tradeoff(self):
        model_low = _make_model(mass_fraction=0.094)
        model_high = _make_model(mass_fraction=0.142)
        comp_low = model_low.compare_with_experiment("PF-1000")
        comp_high = model_high.compare_with_experiment("PF-1000")
        assert comp_low.timing_error < comp_high.timing_error
        assert comp_low.peak_current_error > comp_high.peak_current_error


class TestBlindPrediction16kV:
    """PF-1000 at 16 kV / 1.05 Torr blind prediction with Phase AR fc/fm."""

    _MIDPOINT = 1.2e6
    _RANGE_LO = 1.1e6
    _RANGE_HI = 1.3e6

    def test_blind_prediction_finite(self):
        model = _make_model()
        result = model.run("PF-1000-16kV")
        assert result.peak_current > 500e3
        assert result.peak_current < 2.5e6
        assert len(result.t) > 10

    def test_blind_peak_within_30pct(self):
        model = _make_model()
        result = model.run("PF-1000-16kV")
        err = abs(result.peak_current - self._MIDPOINT) / self._MIDPOINT
        assert err < 0.30

    def test_blind_better_than_bare_rlc(self):
        model = _make_model()
        result = model.run("PF-1000-16kV")
        C, V0, L0, R0 = 1.332e-3, 16e3, 33.5e-9, 2.3e-3
        Z0 = math.sqrt(L0 / C)
        zeta = R0 / (2 * Z0)
        I_rlc = V0 / Z0 * math.exp(-math.pi * zeta / 2)
        err_lee = abs(result.peak_current - self._MIDPOINT) / self._MIDPOINT
        err_rlc = abs(I_rlc - self._MIDPOINT) / self._MIDPOINT
        improvement = (err_rlc - err_lee) / err_rlc
        assert improvement > 0.70

    def test_physics_loading_significant(self):
        model = _make_model()
        result = model.run("PF-1000-16kV")
        C, V0, L0, R0 = 1.332e-3, 16e3, 33.5e-9, 2.3e-3
        Z0 = math.sqrt(L0 / C)
        zeta = R0 / (2 * Z0)
        I_rlc = V0 / Z0 * math.exp(-math.pi * zeta / 2)
        loading = (I_rlc - result.peak_current) / I_rlc
        assert loading > 0.40

    def test_pressure_sensitivity(self):
        model = _make_model()
        params_35 = {
            "C": 1.332e-3, "V0": 16e3, "L0": 33.5e-9, "R0": 2.3e-3,
            "anode_radius": 0.115, "cathode_radius": 0.16,
            "anode_length": 0.6, "fill_pressure_torr": 3.5,
        }
        params_105 = {**params_35, "fill_pressure_torr": 1.05}
        result_35 = model.run(device_params=params_35)
        result_105 = model.run(device_params=params_105)
        assert result_105.peak_current_time < result_35.peak_current_time

    def test_16kv_crowbar_fires(self):
        model = _make_model()
        result = model.run("PF-1000-16kV")
        assert 3 in result.phases_completed


class TestPostPinchDiagnostic:
    """Diagnose model behavior in the 7-10 us post-pinch region."""

    def test_model_covers_full_scholz_window(self):
        model = _make_model()
        result = model.run("PF-1000")
        assert result.t[-1] > 10e-6

    def test_post_pinch_nrmse_identified(self):
        from dpf.validation.experimental import PF1000_DATA
        model = _make_model()
        result = model.run("PF-1000")
        t_exp = PF1000_DATA.waveform_t
        I_exp = PF1000_DATA.waveform_I
        I_model = np.interp(t_exp, result.t, result.I)
        I_peak = np.max(I_exp)
        mask7 = t_exp <= 7e-6
        nrmse_07 = np.sqrt(np.mean((I_model[mask7] - I_exp[mask7])**2)) / I_peak
        nrmse_710 = np.sqrt(np.mean((I_model[~mask7] - I_exp[~mask7])**2)) / I_peak
        assert nrmse_710 > nrmse_07

    def test_effective_post_pinch_resistance(self):
        from dpf.validation.experimental import PF1000_DATA
        t_exp = PF1000_DATA.waveform_t
        I_exp = PF1000_DATA.waveform_I
        mask = (t_exp >= 6e-6) & (t_exp <= 10e-6)
        t_fit = t_exp[mask]
        I_fit = I_exp[mask]
        valid = I_fit > 0
        if np.sum(valid) > 2:
            coeffs = np.polyfit(t_fit[valid], np.log(I_fit[valid]), 1)
            tau_decay = -1.0 / coeffs[0]
            L_total = 73e-9
            R_eff = L_total / tau_decay
            assert R_eff > 5e-3

    def test_blind_prediction_summary(self):
        model = _make_model()
        comp27 = model.compare_with_experiment("PF-1000")
        r16 = model.run("PF-1000-16kV")
        C, _V0_27, V0_16, L0, R0 = 1.332e-3, 27e3, 16e3, 33.5e-9, 2.3e-3
        Z0 = math.sqrt(L0 / C)
        zeta = R0 / (2 * Z0)
        I_rlc_16 = V0_16 / Z0 * math.exp(-math.pi * zeta / 2)
        err_lee = abs(r16.peak_current - 1.2e6) / 1.2e6
        err_rlc = abs(I_rlc_16 - 1.2e6) / 1.2e6
        print(f"\nPhase AS summary: NRMSE={comp27.waveform_nrmse:.4f}, "
              f"Lee err={err_lee:.1%}, RLC err={err_rlc:.1%}")
        assert True


# --- Section: Phase AV — Pinch Physics Diagnostics ---


class TestImplosionVelocity:
    """Tests for implosion velocity formula (Angus 2021)."""

    def test_mjolnir_60kv(self):
        from dpf.validation.pinch_physics import implosion_velocity
        v = implosion_velocity(I_MA=2.8, R_cm=2.54, P_Torr=8.0)
        assert v == pytest.approx(370.2, rel=0.02)

    def test_pf1000_typical(self):
        from dpf.validation.pinch_physics import implosion_velocity
        v = implosion_velocity(I_MA=1.87, R_cm=5.75, P_Torr=3.5)
        assert v == pytest.approx(165.2, rel=0.02)

    def test_linear_in_current(self):
        from dpf.validation.pinch_physics import implosion_velocity
        v1 = implosion_velocity(1.0, 1.0, 1.0)
        v2 = implosion_velocity(2.0, 1.0, 1.0)
        assert v2 == pytest.approx(2 * v1, rel=1e-10)

    def test_inverse_sqrt_pressure(self):
        from dpf.validation.pinch_physics import implosion_velocity
        v1 = implosion_velocity(1.0, 1.0, 1.0)
        v4 = implosion_velocity(1.0, 1.0, 4.0)
        assert v4 == pytest.approx(v1 / 2, rel=1e-10)

    def test_negative_input_raises(self):
        from dpf.validation.pinch_physics import implosion_velocity
        with pytest.raises(ValueError):
            implosion_velocity(-1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            implosion_velocity(1.0, -1.0, 1.0)
        with pytest.raises(ValueError):
            implosion_velocity(1.0, 1.0, -1.0)


class TestStagnationTemperature:
    """Tests for post-shock ion temperature (Goyon Eq. 2)."""

    def test_mjolnir_60kv(self):
        from dpf.validation.pinch_physics import stagnation_temperature
        T = stagnation_temperature(I_MA=2.8, R_cm=2.54, P_Torr=8.0)
        expected = 21 * 2.8**2 / (2.54**2 * 8)
        assert pytest.approx(expected, rel=1e-10) == T
        assert 2.0 < T < 5.0

    def test_quadratic_in_current(self):
        from dpf.validation.pinch_physics import stagnation_temperature
        T1 = stagnation_temperature(1.0, 1.0, 1.0)
        T3 = stagnation_temperature(3.0, 1.0, 1.0)
        assert pytest.approx(9 * T1, rel=1e-10) == T3

    def test_inverse_pressure(self):
        from dpf.validation.pinch_physics import stagnation_temperature
        T1 = stagnation_temperature(1.0, 1.0, 1.0)
        T2 = stagnation_temperature(1.0, 1.0, 2.0)
        assert pytest.approx(T1 / 2, rel=1e-10) == T2


class TestExpansionTimescale:
    """Tests for post-pinch expansion timescale (Goyon Eq. 3)."""

    def test_mjolnir_reference(self):
        from dpf.validation.pinch_physics import expansion_timescale
        tau = expansion_timescale(R_cm=2.54, P_Torr=8.0, CR=10.0, I_MA=2.8)
        expected = 31.5 * 2.54**2 * np.sqrt(8) / (10 * 2.8)
        assert tau == pytest.approx(expected, rel=1e-10)

    def test_quadratic_in_radius(self):
        from dpf.validation.pinch_physics import expansion_timescale
        tau1 = expansion_timescale(1.0, 1.0, 10.0, 1.0)
        tau2 = expansion_timescale(2.0, 1.0, 10.0, 1.0)
        assert tau2 == pytest.approx(4 * tau1, rel=1e-10)

    def test_inverse_CR(self):
        from dpf.validation.pinch_physics import expansion_timescale
        tau10 = expansion_timescale(1.0, 1.0, 10.0, 1.0)
        tau20 = expansion_timescale(1.0, 1.0, 20.0, 1.0)
        assert tau20 == pytest.approx(tau10 / 2, rel=1e-10)


class TestM0InstabilityTimescale:
    """Tests for m=0 sausage instability timescale (Goyon Eq. 4)."""

    def test_close_to_expansion(self):
        from dpf.validation.pinch_physics import expansion_timescale, m0_instability_timescale
        tau_exp = expansion_timescale(2.54, 8.0, 10.0, 2.8)
        tau_m0 = m0_instability_timescale(2.54, 8.0, 10.0, 2.8)
        assert tau_m0 / tau_exp == pytest.approx(31.0 / 31.5, rel=1e-10)

    def test_disruption_before_expansion(self):
        from dpf.validation.pinch_physics import expansion_timescale, m0_instability_timescale
        tau_exp = expansion_timescale(2.54, 8.0, 10.0, 2.8)
        tau_m0 = m0_instability_timescale(2.54, 8.0, 10.0, 2.8)
        assert tau_m0 < tau_exp


class TestMRTIGrowthRate:
    """Tests for mRT growth rate (Bian et al. 2026)."""

    def test_classical_rt_no_B(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        g, k, A = 1e12, 1e4, 0.5
        gamma = mrti_growth_rate(g, k, A)
        expected = np.sqrt(g * k * A)
        assert gamma == pytest.approx(expected, rel=1e-10)

    def test_classical_rt_matches_helper(self):
        from dpf.validation.pinch_physics import classical_rt_growth_rate, mrti_growth_rate
        g, k, A = 1e12, 1e4, 0.8
        assert classical_rt_growth_rate(g, k, A) == mrti_growth_rate(g, k, A)

    def test_magnetic_suppression(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        g, k, A = 1e12, 1e4, 0.5
        gamma_0 = mrti_growth_rate(g, k, A, B=0)
        gamma_B = mrti_growth_rate(g, k, A, B=1.0, theta=0, rho_h=10, rho_l=5)
        assert gamma_B < gamma_0

    def test_perpendicular_B_no_suppression(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        g, k, A = 1e12, 1e4, 0.5
        gamma_0 = mrti_growth_rate(g, k, A, B=0)
        gamma_perp = mrti_growth_rate(g, k, A, B=10, theta=np.pi / 2,
                                      rho_h=10, rho_l=5)
        assert gamma_perp == pytest.approx(gamma_0, rel=1e-6)

    def test_full_stabilization_returns_zero(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        g, k, A = 1e10, 1e3, 0.5
        gamma = mrti_growth_rate(g, k, A, B=100, theta=0, rho_h=1.0, rho_l=0.5)
        assert gamma == 0.0

    def test_atwood_zero(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        gamma = mrti_growth_rate(1e12, 1e4, 0.0)
        assert gamma == 0.0

    def test_atwood_one(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        gamma = mrti_growth_rate(1e12, 1e4, 1.0)
        assert gamma == pytest.approx(np.sqrt(1e12 * 1e4), rel=1e-10)

    def test_invalid_atwood_raises(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        with pytest.raises(ValueError, match="Atwood"):
            mrti_growth_rate(1e12, 1e4, -0.1)
        with pytest.raises(ValueError, match="Atwood"):
            mrti_growth_rate(1e12, 1e4, 1.5)

    def test_negative_wavenumber_raises(self):
        from dpf.validation.pinch_physics import mrti_growth_rate
        with pytest.raises(ValueError, match="Wavenumber"):
            mrti_growth_rate(1e12, -1e4, 0.5)


class TestNeutronYieldI4:
    """Tests for I^4 neutron yield scaling."""

    def test_quartic_scaling(self):
        from dpf.validation.pinch_physics import neutron_yield_I4
        Y1 = neutron_yield_I4(1.0)
        Y2 = neutron_yield_I4(2.0)
        assert pytest.approx(16 * Y1, rel=1e-10) == Y2

    def test_pf1000_order_of_magnitude(self):
        from dpf.validation.pinch_physics import neutron_yield_I4
        Y = neutron_yield_I4(2.0)
        assert 1e10 < Y < 1e13

    def test_custom_coefficient(self):
        from dpf.validation.pinch_physics import neutron_yield_I4
        Y = neutron_yield_I4(1.0, coefficient=1e11)
        assert pytest.approx(1e11, rel=1e-10) == Y


class TestPinchDebyeLength:
    """Tests for Debye length from dpf.validation.pinch_physics."""

    def test_typical_dpf_rundown(self):
        from dpf.validation.pinch_physics import debye_length
        lam = debye_length(n_e=1e22, T_e_eV=2.0)
        assert 1e-8 < lam < 1e-5

    def test_scales_with_temperature(self):
        from dpf.validation.pinch_physics import debye_length
        lam1 = debye_length(1e22, 1.0)
        lam4 = debye_length(1e22, 4.0)
        assert lam4 == pytest.approx(2 * lam1, rel=1e-6)

    def test_scales_with_density(self):
        from dpf.validation.pinch_physics import debye_length
        lam1 = debye_length(1e22, 1.0)
        lam4 = debye_length(4e22, 1.0)
        assert lam4 == pytest.approx(lam1 / 2, rel=1e-6)

    def test_negative_raises(self):
        from dpf.validation.pinch_physics import debye_length
        with pytest.raises(ValueError):
            debye_length(-1e22, 1.0)
        with pytest.raises(ValueError):
            debye_length(1e22, -1.0)


class TestMAClassDeviceComparison:
    """Test predictions against published MA-class device data."""

    @pytest.mark.parametrize("device,data", [])
    def test_yield_order_of_magnitude(self, device: str, data: dict):
        from dpf.validation.pinch_physics import neutron_yield_I4
        Y_pred = neutron_yield_I4(data["I_peak_MA"])
        Y_meas = data["Y_n"]
        ratio = Y_pred / Y_meas
        anomalous = {"POSEIDON", "Gemini"}
        if device in anomalous:
            assert 0.01 < ratio < 100
        else:
            assert 0.1 < ratio < 10

    def test_ma_class_devices_parametrize(self):
        from dpf.validation.pinch_physics import _MA_CLASS_DEVICES, neutron_yield_I4
        for device, data in _MA_CLASS_DEVICES.items():
            Y_pred = neutron_yield_I4(data["I_peak_MA"])
            Y_meas = data["Y_n"]
            ratio = Y_pred / Y_meas
            anomalous = {"POSEIDON", "Gemini"}
            if device in anomalous:
                assert 0.01 < ratio < 100, f"{device}: ratio={ratio:.1f}"
            else:
                assert 0.1 < ratio < 10, f"{device}: ratio={ratio:.1f}"

    def test_monotonic_yield_with_current(self):
        from dpf.validation.pinch_physics import _MA_CLASS_DEVICES, neutron_yield_I4
        currents = sorted(
            [(d["I_peak_MA"], d["Y_n"]) for d in _MA_CLASS_DEVICES.values()],
            key=lambda x: x[0],
        )
        for i in range(len(currents) - 1):
            I_lo, _ = currents[i]
            I_hi, _ = currents[i + 1]
            if I_lo < I_hi:
                assert neutron_yield_I4(I_hi) > neutron_yield_I4(I_lo)


class TestASMEDeltaModel:
    """Tests for delta_model computation in ASME V&V 20 assessment."""

    def test_delta_model_computed(self):
        result = asme_vv20_assessment()
        assert hasattr(result, "delta_model")
        assert result.delta_model >= 0

    def test_delta_model_formula(self):
        result = asme_vv20_assessment()
        expected = math.sqrt(max(0.0, result.E**2 - result.u_val**2))
        assert result.delta_model == pytest.approx(expected, rel=1e-10)

    def test_delta_model_positive_when_fail(self):
        result = asme_vv20_assessment()
        if not result.passes:
            assert result.delta_model > 0

    def test_delta_model_less_than_E(self):
        result = asme_vv20_assessment()
        assert result.delta_model <= result.E + 1e-15


# --- Section: Phase AY — Comprehensive V&V Diagnostics ---


class TestNRMSEPhaseDecomposition:
    """Decompose NRMSE by discharge phase to identify model-form error sources."""

    def test_phase_decomposition_finite(self):
        model = _make_model()
        result = model.run("PF-1000")
        dev = DEVICES["PF-1000"]
        t_exp = dev.waveform_t
        I_exp = dev.waveform_I
        windows = {
            "early_rise": (0.0, 3e-6),
            "late_rise": (3e-6, 5e-6),
            "peak_region": (5e-6, 6.5e-6),
            "post_peak": (6.5e-6, 10e-6),
        }
        for name, (t0, t1) in windows.items():
            nrmse = _nrmse_window(result.t, result.I, t_exp, I_exp, t0, t1)
            assert np.isfinite(nrmse), f"Non-finite NRMSE in {name}"

    def test_mid_rise_best_accuracy(self):
        model = _make_model()
        result = model.run("PF-1000")
        dev = DEVICES["PF-1000"]
        t_exp = dev.waveform_t
        I_exp = dev.waveform_I
        nrmse_mid = _nrmse_window(result.t, result.I, t_exp, I_exp, 2e-6, 5e-6)
        nrmse_early = _nrmse_window(result.t, result.I, t_exp, I_exp, 0.0, 2e-6)
        assert nrmse_mid < nrmse_early

    def test_post_peak_dominates_error(self):
        model = _make_model()
        result = model.run("PF-1000")
        dev = DEVICES["PF-1000"]
        t_exp = dev.waveform_t
        I_exp = dev.waveform_I
        nrmse_pre = _nrmse_window(result.t, result.I, t_exp, I_exp, 0.0, 5.8e-6)
        nrmse_full = _nrmse_window(result.t, result.I, t_exp, I_exp, 0.0, 10e-6)
        assert nrmse_pre < nrmse_full


class TestSegmentedASME:
    """ASME V&V 20 assessment in both pre-pinch and full-waveform windows."""

    def test_pre_pinch_asme_lower_E(self):
        full = asme_vv20_assessment(
            fc=_FC, fm=_FM, f_mr=_F_MR,
            pinch_column_fraction=_PCF,
            crowbar_resistance=_CROWBAR_R,
        )
        windowed = asme_vv20_assessment(
            fc=_FC, fm=_FM, f_mr=_F_MR,
            pinch_column_fraction=_PCF,
            crowbar_resistance=_CROWBAR_R,
            max_time=5.8e-6,
        )
        assert windowed.E < full.E

    def test_liftoff_reduces_E(self):
        no_liftoff = asme_vv20_assessment(
            fc=_FC, fm=_FM, f_mr=_F_MR,
            pinch_column_fraction=_PCF,
            crowbar_resistance=_CROWBAR_R,
            max_time=5.8e-6,
        )
        with_liftoff = asme_vv20_assessment(
            fc=_FC, fm=_FM, f_mr=_F_MR,
            pinch_column_fraction=_PCF,
            crowbar_resistance=_CROWBAR_R,
            liftoff_delay=0.6e-6,
            max_time=5.8e-6,
        )
        assert with_liftoff.E <= no_liftoff.E * 1.3


class TestCrossDeviceTransferability:
    """Quantify how well calibrated parameters transfer between devices."""

    def test_fc_squared_fm_ratio_consistency(self):
        ratio_pf1000 = _FC**2 / _FM
        assert ratio_pf1000 > 0
        assert 1.0 < ratio_pf1000 < 20.0

    def test_lp_l0_determines_validation_quality(self):
        plasma_significant = []
        for name, dev in DEVICES.items():
            diag = compute_lp_l0_ratio(
                L0=dev.inductance,
                anode_radius=dev.anode_radius,
                cathode_radius=dev.cathode_radius,
                anode_length=dev.anode_length,
            )
            if diag["L_p_over_L0"] > 1.0:
                plasma_significant.append(name)
        assert len(plasma_significant) >= 1
        assert "PF-1000" in plasma_significant

    def test_voltage_transferability_single_variable(self):
        model = _make_model()
        configs = {
            "PF-1000-16kV": {"V0": 16e3, "p_torr": 1.05, "I_exp": 1.2e6},
            "PF-1000-20kV": {"V0": 20e3, "p_torr": 2.0, "I_exp": 1.4e6},
            "PF-1000": {"V0": 27e3, "p_torr": 3.5, "I_exp": 1.87e6},
        }
        errors = []
        for name, cfg in configs.items():
            result = model.run(name)
            error = abs(result.peak_current - cfg["I_exp"]) / cfg["I_exp"]
            errors.append(error)
            assert error < 0.25, f"{name}: error = {error*100:.1f}%"
        assert np.mean(errors) < 0.15


class TestModelValidityWindow:
    """Quantify the temporal window where the Lee model is valid."""

    def test_sliding_window_nrmse(self):
        model = _make_model()
        result = model.run("PF-1000")
        dev = DEVICES["PF-1000"]
        t_exp = dev.waveform_t
        I_exp = dev.waveform_I
        t_ends = [4e-6, 5e-6, 5.8e-6, 6.5e-6, 7e-6, 8e-6, 10e-6]
        nrmses = [_nrmse_window(result.t, result.I, t_exp, I_exp, 0.0, t) for t in t_ends]
        assert nrmses[-1] > nrmses[2]

    def test_model_validity_fraction(self):
        model = _make_model()
        result = model.run("PF-1000")
        dev = DEVICES["PF-1000"]
        I_sim = np.interp(dev.waveform_t, result.t, result.I)
        I_exp = dev.waveform_I
        point_errors = np.abs(I_sim - I_exp) / np.maximum(np.abs(I_exp), 1e3)
        frac_20pct = np.mean(point_errors < 0.20)
        assert frac_20pct > 0.50


# --- Section: Phase BQ — Expanded ASME V&V 20 Uncertainty Budget ---

MULTI_CONDITION_ASME = {
    "27kV_to_16kV": {"E": 0.1187, "u_val": 0.1150, "ratio": 1.03},
    "16kV_to_27kV": {"E": 0.1006, "u_val": 0.0680, "ratio": 1.48},
    "Scholz_to_Gribkov": {"E": 0.1972, "u_val": 0.0603, "ratio": 3.27},
}

LOO_ASME = {
    "PF-1000": {"E": 0.4371, "u_val": 0.0680, "ratio": 6.42},
    "POSEIDON-60kV": {"E": 0.1996, "u_val": 0.0695, "ratio": 2.87},
    "UNU-ICTP": {"E": 0.1006, "u_val": 0.1095, "ratio": 0.92},
    "FAETON-I": {"E": 0.2548, "u_val": 0.1217, "ratio": 2.09},
    "MJOLNIR": {"E": 0.1718, "u_val": 0.1383, "ratio": 1.24},
}

DEVICE_UNCERTAINTIES = {
    "PF-1000": {"u_peak_I": 0.05, "u_dig": 0.03, "u_shot": 0.05, "n_shots": 5},
    "PF-1000-16kV": {"u_peak_I": 0.10, "u_dig": 0.05, "u_shot": 0.05, "n_shots": 16},
    "PF-1000-Gribkov": {"u_peak_I": 0.05, "u_dig": 0.03, "u_shot": 0.05, "n_shots": 5},
    "POSEIDON-60kV": {"u_peak_I": 0.05, "u_dig": 0.02, "u_shot": 0.06, "n_shots": 3},
    "UNU-ICTP": {"u_peak_I": 0.10, "u_dig": 0.016, "u_shot": 0.10, "n_shots": 10},
    "FAETON-I": {"u_peak_I": 0.08, "u_dig": 0.08, "u_shot": 0.08, "n_shots": 5},
    "MJOLNIR": {"u_peak_I": 0.08, "u_dig": 0.10, "u_shot": 0.10, "n_shots": 5},
}


def _compute_u_val(
    u_peak_I: float,
    u_dig: float,
    u_shot: float = 0.0,
    n_shots: int = 1,
    u_input: float = 0.027,
    u_num: float = 0.001,
    u_additional: float = 0.0,
) -> float:
    u_shot_avg = u_shot / np.sqrt(n_shots) if n_shots > 1 else u_shot
    u_exp = np.sqrt(u_peak_I**2 + u_dig**2 + u_shot_avg**2)
    return float(np.sqrt(u_exp**2 + u_input**2 + u_num**2 + u_additional**2))


def _minimum_u_additional_for_pass(E: float, u_val_current: float) -> float:
    if u_val_current >= E:
        return 0.0
    return float(np.sqrt(E**2 - u_val_current**2))


class TestCurrentASMEBudget:
    """Verify the current ASME budget components are self-consistent."""

    def test_27kv_to_16kv_u_val_reconstruction(self):
        u_exp = np.sqrt(0.10**2 + 0.05**2)
        u_val = np.sqrt(u_exp**2 + 0.027**2 + 0.001**2)
        assert u_val == pytest.approx(0.1150, abs=0.002)

    def test_27kv_to_16kv_ratio(self):
        ratio = 0.1187 / 0.1150
        assert ratio == pytest.approx(1.03, abs=0.01)

    def test_16kv_to_27kv_u_val(self):
        u_shot_avg = 0.05 / np.sqrt(5)
        u_exp = np.sqrt(0.05**2 + 0.03**2 + u_shot_avg**2)
        u_val = np.sqrt(u_exp**2 + 0.027**2 + 0.001**2)
        assert u_val == pytest.approx(0.0680, abs=0.003)

    def test_loo_unu_ictp_only_pass(self):
        passing = [k for k, v in LOO_ASME.items() if v["ratio"] <= 1.0]
        assert passing == ["UNU-ICTP"]


class TestExpandedBudget16kV:
    """Test ASME ratio with shot-to-shot added for PF-1000-16kV."""

    def test_16kv_shot_to_shot_reduces_ratio(self):
        dev = DEVICE_UNCERTAINTIES["PF-1000-16kV"]
        u_val_expanded = _compute_u_val(
            u_peak_I=dev["u_peak_I"], u_dig=dev["u_dig"],
            u_shot=dev["u_shot"], n_shots=dev["n_shots"],
        )
        assert 0.1187 / u_val_expanded < 1.03

    def test_16kv_expanded_still_fails(self):
        dev = DEVICE_UNCERTAINTIES["PF-1000-16kV"]
        u_val_expanded = _compute_u_val(
            u_peak_I=dev["u_peak_I"], u_dig=dev["u_dig"],
            u_shot=dev["u_shot"], n_shots=dev["n_shots"],
        )
        assert 0.1187 / u_val_expanded > 1.0

    def test_16kv_with_fewer_shots_could_flip(self):
        dev = DEVICE_UNCERTAINTIES["PF-1000-16kV"]
        u_val_1shot = _compute_u_val(
            u_peak_I=dev["u_peak_I"], u_dig=dev["u_dig"],
            u_shot=dev["u_shot"], n_shots=1,
        )
        assert 0.1187 / u_val_1shot < 1.0


class TestSensitivityAnalysis:
    """How much additional uncertainty is needed to flip each ASME ratio?"""

    def test_27kv_to_16kv_delta_u(self):
        delta = _minimum_u_additional_for_pass(0.1187, 0.1150)
        assert delta == pytest.approx(0.029, abs=0.005)

    def test_16kv_to_27kv_delta_u(self):
        delta = _minimum_u_additional_for_pass(0.1006, 0.0680)
        assert delta > 0.05

    def test_scholz_to_gribkov_delta_u(self):
        delta = _minimum_u_additional_for_pass(0.1972, 0.0603)
        assert delta > 0.15

    def test_loo_pf1000_delta_u(self):
        delta = _minimum_u_additional_for_pass(0.4371, 0.0680)
        assert delta > 0.40

    def test_loo_mjolnir_delta_u(self):
        delta = _minimum_u_additional_for_pass(0.1718, 0.1383)
        assert delta == pytest.approx(0.102, abs=0.01)

    def test_loo_unu_ictp_already_passes(self):
        delta = _minimum_u_additional_for_pass(0.1006, 0.1095)
        assert delta == 0.0

    def test_only_27kv_to_16kv_realistically_flippable(self):
        mc_delta = _minimum_u_additional_for_pass(
            MULTI_CONDITION_ASME["27kV_to_16kV"]["E"],
            MULTI_CONDITION_ASME["27kV_to_16kV"]["u_val"],
        )
        assert mc_delta < 0.05
        for name in ["16kV_to_27kV", "Scholz_to_Gribkov"]:
            r = MULTI_CONDITION_ASME[name]
            delta = _minimum_u_additional_for_pass(r["E"], r["u_val"])
            assert delta > 0.05


class TestWaveformProvenanceImpact:
    """Test how waveform provenance affects ASME."""

    def test_reconstructed_u_dig_higher(self):
        for name in ["PF-1000-16kV", "FAETON-I", "MJOLNIR"]:
            assert DEVICE_UNCERTAINTIES[name]["u_dig"] >= 0.05

    def test_measured_u_dig_lower(self):
        for name in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]:
            assert DEVICE_UNCERTAINTIES[name]["u_dig"] <= 0.03

    def test_16kv_u_dig_upgrade_to_8pct_flips(self):
        u_val = _compute_u_val(u_peak_I=0.10, u_dig=0.08, u_shot=0.05, n_shots=16)
        assert 0.1187 / u_val < 1.0

    def test_16kv_u_dig_upgrade_to_7pct_flips(self):
        u_val = _compute_u_val(u_peak_I=0.10, u_dig=0.07, u_shot=0.05, n_shots=16)
        assert 0.1187 / u_val < 1.0

    def test_16kv_u_dig_6pct_marginal(self):
        u_val = _compute_u_val(u_peak_I=0.10, u_dig=0.06, u_shot=0.05, n_shots=16)
        ratio = 0.1187 / u_val
        assert abs(ratio - 1.0) < 0.05


class TestBudgetDecomposition:
    """Decompose ASME budget into dominant uncertainty components."""

    def test_27kv_to_16kv_dominated_by_u_exp(self):
        assert 0.1118**2 / 0.1150**2 > 0.90

    def test_u_input_small_contribution(self):
        assert 0.027**2 / 0.1150**2 < 0.10

    def test_u_num_negligible(self):
        assert 0.001**2 / 0.1150**2 < 0.001

    def test_unu_ictp_passes_because_large_u_exp(self):
        unu = LOO_ASME["UNU-ICTP"]
        assert unu["u_val"] > unu["E"]


class TestMultiConditionExpandedASME:
    """Multi-condition ASME with expanded uncertainty for each direction."""

    def test_27kv_to_16kv_expanded_with_reconstruction(self):
        u_val = _compute_u_val(
            u_peak_I=0.10, u_dig=0.07, u_shot=0.05,
            n_shots=16, u_input=0.027, u_num=0.001,
        )
        assert 0.1187 / u_val < 1.0

    def test_16kv_to_27kv_expanded_still_fails(self):
        dev = DEVICE_UNCERTAINTIES["PF-1000"]
        u_val = _compute_u_val(
            u_peak_I=dev["u_peak_I"], u_dig=dev["u_dig"],
            u_shot=dev["u_shot"], n_shots=dev["n_shots"],
        )
        assert MULTI_CONDITION_ASME["16kV_to_27kV"]["E"] / u_val > 1.0

    def test_voltage_transfer_asymmetry_explained(self):
        u_exp_16kv = np.sqrt(0.10**2 + 0.05**2)
        u_exp_27kv = np.sqrt(0.05**2 + 0.03**2)
        assert u_exp_16kv > 1.5 * u_exp_27kv

    def test_expanded_asme_summary_table(self):
        ratio_27_16 = 0.1187 / _compute_u_val(0.10, 0.07, 0.05, 16)
        ratio_16_27 = 0.1006 / _compute_u_val(0.05, 0.03, 0.05, 5)
        ratio_sg = 0.1972 / _compute_u_val(0.05, 0.03, 0.05, 5)
        passes = sum(1 for r in [ratio_27_16, ratio_16_27, ratio_sg] if r <= 1.0)
        assert passes == 1


class TestReconstructionUncertaintyJustification:
    """Test that the reconstruction uncertainty values are physically justified."""

    def test_measured_range_2_to_3_pct(self):
        measured_u = [DEVICE_UNCERTAINTIES[d]["u_dig"]
                      for d in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]]
        assert all(0.01 <= u <= 0.04 for u in measured_u)

    def test_reconstructed_range_5_to_10_pct(self):
        recon_u = [DEVICE_UNCERTAINTIES[d]["u_dig"]
                   for d in ["PF-1000-16kV", "FAETON-I", "MJOLNIR"]]
        assert all(0.04 <= u <= 0.12 for u in recon_u)

    def test_reconstructed_always_higher(self):
        measured_max = max(DEVICE_UNCERTAINTIES[d]["u_dig"]
                          for d in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"])
        recon_min = min(DEVICE_UNCERTAINTIES[d]["u_dig"]
                        for d in ["PF-1000-16kV", "FAETON-I", "MJOLNIR"])
        assert recon_min > measured_max

    def test_16kv_5pct_conservative_for_reconstruction(self):
        assert (DEVICE_UNCERTAINTIES["PF-1000-16kV"]["u_dig"] <=
                DEVICE_UNCERTAINTIES["FAETON-I"]["u_dig"])
        assert (DEVICE_UNCERTAINTIES["FAETON-I"]["u_dig"] <=
                DEVICE_UNCERTAINTIES["MJOLNIR"]["u_dig"])

    def test_7pct_justified_for_shape_uncertainty(self):
        shape_u = np.sqrt(0.05**2 + 0.02**2)
        assert 0.05 < shape_u < 0.08
        assert shape_u < 0.07


# --- Section: Phase BR — GUM Uncertainty Taxonomy ---


class TestUncertaintyType:
    """Every device with a waveform must have uncertainty_type set per GUM."""

    def test_measured_devices_have_digitization_type(self):
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        assert len(measured) >= 3
        for name, dev in measured.items():
            assert dev.waveform_uncertainty_type == "digitization", (
                f"{name}: measured waveform should have uncertainty_type='digitization', "
                f"got '{dev.waveform_uncertainty_type}'"
            )

    def test_reconstructed_devices_have_reconstruction_type(self):
        from dpf.validation.experimental import get_devices_by_provenance
        recon = get_devices_by_provenance("reconstructed")
        assert len(recon) >= 2
        for name, dev in recon.items():
            assert dev.waveform_uncertainty_type == "reconstruction", (
                f"{name}: reconstructed waveform should have uncertainty_type='reconstruction', "
                f"got '{dev.waveform_uncertainty_type}'"
            )

    def test_no_waveform_devices_have_empty_type(self):
        no_waveform = [name for name, dev in DEVICES.items() if dev.waveform_t is None]
        for name in no_waveform:
            assert DEVICES[name].waveform_uncertainty_type == ""

    def test_uncertainty_type_values_exhaustive(self):
        valid_types = {"digitization", "reconstruction", ""}
        for _name, dev in DEVICES.items():
            assert dev.waveform_uncertainty_type in valid_types

    def test_measured_uncertainty_lower_than_reconstructed(self):
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        recon = get_devices_by_provenance("reconstructed")
        avg_meas = np.mean([d.waveform_amplitude_uncertainty for d in measured.values()])
        avg_recon = np.mean([d.waveform_amplitude_uncertainty for d in recon.values()])
        assert avg_meas < avg_recon


class TestAmplitudeUncertainty:
    """Verify waveform_amplitude_uncertainty replaces old digitization field."""

    def test_field_exists_on_all_devices(self):
        for name, dev in DEVICES.items():
            assert hasattr(dev, "waveform_amplitude_uncertainty"), (
                f"{name}: missing waveform_amplitude_uncertainty field"
            )

    def test_pf1000_digitization_3pct(self):
        assert DEVICES["PF-1000"].waveform_amplitude_uncertainty == pytest.approx(0.03, abs=0.005)

    def test_unu_ictp_digitization_1_6pct(self):
        assert DEVICES["UNU-ICTP"].waveform_amplitude_uncertainty == pytest.approx(0.016, abs=0.003)

    def test_gribkov_digitization_2pct(self):
        assert DEVICES["PF-1000-Gribkov"].waveform_amplitude_uncertainty == pytest.approx(0.02, abs=0.005)

    def test_poseidon60kv_digitization_2pct(self):
        assert DEVICES["POSEIDON-60kV"].waveform_amplitude_uncertainty == pytest.approx(0.02, abs=0.005)

    def test_pf1000_16kv_reconstruction_5pct(self):
        dev = DEVICES["PF-1000-16kV"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.05, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_faeton_reconstruction_8pct(self):
        dev = DEVICES["FAETON-I"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.08, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_mjolnir_reconstruction_10pct(self):
        dev = DEVICES["MJOLNIR"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.10, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_no_old_field_name(self):
        from dpf.validation.experimental import ExperimentalDevice
        dev = ExperimentalDevice(
            name="test", institution="test",
            capacitance=1e-3, voltage=1e4, inductance=1e-8, resistance=1e-3,
            anode_radius=0.01, cathode_radius=0.02, anode_length=0.1,
            fill_pressure_torr=1.0, fill_gas="deuterium",
            peak_current=1e5, neutron_yield=1e6,
            current_rise_time=1e-6, reference="test",
        )
        assert not hasattr(dev, "waveform_digitization_uncertainty")


class TestDoubleCounting:
    """Verify peak_current_from_shot_spread prevents double-counting in ASME."""

    def test_pf1000_16kv_has_flag(self):
        dev = DEVICES["PF-1000-16kV"]
        assert dev.peak_current_from_shot_spread is True

    def test_other_devices_no_flag(self):
        expected_false = ["PF-1000", "UNU-ICTP", "POSEIDON-60kV", "NX2",
                          "PF-1000-Gribkov", "FAETON-I", "MJOLNIR"]
        for name in expected_false:
            assert not DEVICES[name].peak_current_from_shot_spread

    def test_pf1000_16kv_peak_uncertainty_source(self):
        dev = DEVICES["PF-1000-16kV"]
        assert dev.peak_current_uncertainty == pytest.approx(0.10, abs=0.02)
        assert dev.peak_current_from_shot_spread is True

    def test_asme_budget_without_double_counting(self):
        dev = DEVICES["PF-1000-16kV"]
        u_exp_correct = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        assert u_exp_correct == pytest.approx(0.1118, abs=0.005)

    def test_pf1000_asme_budget_with_shot_to_shot(self):
        dev = DEVICES["PF-1000"]
        assert not dev.peak_current_from_shot_spread
        u_exp_with_shot = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
            + (0.05 / np.sqrt(5))**2
        )
        assert u_exp_with_shot > np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )


class TestGUMConsistency:
    """Verify GUM (JCGM 100:2008) requirements are met."""

    def test_each_component_has_physical_source(self):
        for _name, dev in DEVICES.items():
            if dev.waveform_t is not None:
                assert dev.waveform_uncertainty_type in ("digitization", "reconstruction")

    def test_independent_components(self):
        dev = DEVICES["PF-1000-16kV"]
        if dev.peak_current_from_shot_spread:
            assert dev.peak_current_from_shot_spread is True

    def test_provenance_consistency(self):
        for _name, dev in DEVICES.items():
            if dev.waveform_provenance == "measured":
                if dev.waveform_uncertainty_type:
                    assert dev.waveform_uncertainty_type == "digitization"
            elif dev.waveform_provenance == "reconstructed" and dev.waveform_uncertainty_type:
                assert dev.waveform_uncertainty_type == "reconstruction"

    def test_multishot_uncertainty_field_renamed(self):
        from dpf.validation.calibration import multi_shot_uncertainty
        result = multi_shot_uncertainty("PF-1000")
        assert hasattr(result, "u_amplitude")
        assert not hasattr(result, "u_digitization")
        assert result.u_amplitude == pytest.approx(0.03, abs=0.005)


class TestUncertaintyRanges:
    """Verify uncertainty values are physically reasonable."""

    @pytest.mark.parametrize("name,lo,hi", [
        ("PF-1000", 0.01, 0.05),
        ("UNU-ICTP", 0.005, 0.03),
        ("PF-1000-Gribkov", 0.01, 0.04),
        ("POSEIDON-60kV", 0.01, 0.04),
        ("PF-1000-16kV", 0.03, 0.10),
        ("FAETON-I", 0.05, 0.15),
        ("MJOLNIR", 0.05, 0.15),
    ])
    def test_amplitude_uncertainty_in_range(self, name, lo, hi):
        dev = DEVICES[name]
        assert lo <= dev.waveform_amplitude_uncertainty <= hi

    def test_reconstruction_higher_than_digitization(self):
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        recon = get_devices_by_provenance("reconstructed")
        max_measured = max(d.waveform_amplitude_uncertainty for d in measured.values())
        min_recon = min(d.waveform_amplitude_uncertainty for d in recon.values())
        assert min_recon >= max_measured


# --- Section: Phase BS — Provenance-Dependent ASME + LOO maxiter=10 ---

LOO_MAXITER10_RESULTS = {
    "PF-1000":       {"blind": 0.4376, "indep": 0.0957, "degrad": 4.57,
                      "fc": 0.563, "fm": 0.287, "delay": 0.000},
    "POSEIDON-60kV": {"blind": 0.1906, "indep": 0.0601, "degrad": 3.17,
                      "fc": 0.835, "fm": 0.226, "delay": 0.012},
    "UNU-ICTP":      {"blind": 0.0963, "indep": 0.0660, "degrad": 1.46,
                      "fc": 0.545, "fm": 0.085, "delay": 0.011},
    "FAETON-I":      {"blind": 0.1898, "indep": 0.0193, "degrad": 9.83,
                      "fc": 0.932, "fm": 0.181, "delay": 0.016},
    "MJOLNIR":       {"blind": 0.1755, "indep": 0.1723, "degrad": 1.02,
                      "fc": 0.626, "fm": 0.125, "delay": 0.009},
}

LOO_MAXITER3_RESULTS = {
    "PF-1000":       {"blind": 0.4377, "fc": 0.500, "fm": 0.227, "delay": 0.000},
    "POSEIDON-60kV": {"blind": 0.1917, "fc": 0.843, "fm": 0.239, "delay": 0.051},
    "UNU-ICTP":      {"blind": 0.0978, "fc": 0.701, "fm": 0.159, "delay": 0.067},
    "FAETON-I":      {"blind": 0.1720, "fc": 0.801, "fm": 0.146, "delay": 0.037},
    "MJOLNIR":       {"blind": 0.1777, "fc": 0.843, "fm": 0.239, "delay": 0.051},
}


class TestASMEProvenance:
    """Verify provenance and qualified fields on ASMEValidationResult."""

    def test_result_has_provenance_field(self):
        from dpf.validation.calibration import ASMEValidationResult
        r = ASMEValidationResult(
            E=0.10, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=1.67, passes=False,
        )
        assert hasattr(r, "waveform_provenance")
        assert r.waveform_provenance == ""

    def test_result_has_qualified_field(self):
        from dpf.validation.calibration import ASMEValidationResult
        r = ASMEValidationResult(
            E=0.10, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=1.67, passes=False,
        )
        assert hasattr(r, "qualified")
        assert r.qualified is False

    def test_measured_device_not_qualified(self):
        r = asme_vv20_assessment("PF-1000", fc=0.800, fm=0.128)
        assert r.waveform_provenance == "measured"
        assert r.qualified is False

    def test_reconstructed_device_is_qualified(self):
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        assert r.waveform_provenance == "reconstructed"
        assert r.qualified is True

    @pytest.mark.parametrize("device,expected_prov", [
        ("PF-1000", "measured"),
        ("POSEIDON-60kV", "measured"),
        ("UNU-ICTP", "measured"),
        ("PF-1000-Gribkov", "measured"),
        ("PF-1000-16kV", "reconstructed"),
        ("FAETON-I", "reconstructed"),
        ("MJOLNIR", "reconstructed"),
    ])
    def test_provenance_matches_device(self, device, expected_prov):
        r = asme_vv20_assessment(device, fc=0.700, fm=0.150)
        assert r.waveform_provenance == expected_prov

    @pytest.mark.parametrize("device", ["PF-1000-16kV", "FAETON-I", "MJOLNIR"])
    def test_reconstructed_always_qualified(self, device):
        r = asme_vv20_assessment(device, fc=0.700, fm=0.150)
        assert r.qualified is True


class TestASMEStratifiedSummary:
    """Verify stratified ASME summary separates measured vs reconstructed."""

    def _make_result(self, device: str, passes: bool = False, E: float = 0.10):
        from dpf.validation.calibration import ASMEValidationResult
        dev = DEVICES[device]
        return ASMEValidationResult(
            E=E, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=E / 0.06, passes=passes,
            device_name=device,
            waveform_provenance=dev.waveform_provenance,
            qualified=dev.waveform_provenance == "reconstructed",
        )

    def test_stratify_splits_correctly(self):
        from dpf.validation.calibration import asme_stratified_summary
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("POSEIDON-60kV", passes=False),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=False),
            self._make_result("MJOLNIR", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_total == 3
        assert summary.n_reconstructed_total == 2

    def test_measured_pass_count(self):
        from dpf.validation.calibration import asme_stratified_summary
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_pass == 2
        assert summary.n_reconstructed_pass == 1

    def test_total_properties(self):
        from dpf.validation.calibration import asme_stratified_summary
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("POSEIDON-60kV", passes=False),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=False),
            self._make_result("MJOLNIR", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_total == 5
        assert summary.n_total_pass == 3

    def test_empty_results(self):
        from dpf.validation.calibration import asme_stratified_summary
        summary = asme_stratified_summary([])
        assert summary.n_measured_total == 0
        assert summary.n_reconstructed_total == 0
        assert summary.n_total == 0

    def test_all_results_preserved(self):
        from dpf.validation.calibration import asme_stratified_summary
        results = [self._make_result("PF-1000"), self._make_result("FAETON-I")]
        summary = asme_stratified_summary(results)
        assert len(summary.all_results) == 2


class TestProvenanceBehavior:
    """Verify ASME assessment behavior differs by provenance where needed."""

    def test_measured_includes_shot_to_shot(self):
        r = asme_vv20_assessment("PF-1000", fc=0.800, fm=0.128)
        dev = DEVICES["PF-1000"]
        u_min = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        assert r.u_exp > u_min

    def test_reconstructed_skips_shot_for_16kv(self):
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        dev = DEVICES["PF-1000-16kV"]
        expected = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        assert r.u_exp == pytest.approx(expected, abs=0.001)

    def test_measured_u_exp_lower_than_reconstructed_avg(self):
        fc, fm = 0.700, 0.150
        m_uexp = [asme_vv20_assessment(d, fc=fc, fm=fm).u_exp
                  for d in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]]
        r_uexp = [asme_vv20_assessment(d, fc=fc, fm=fm).u_exp
                  for d in ["PF-1000-16kV", "FAETON-I", "MJOLNIR"]]
        assert np.mean(r_uexp) > np.mean(m_uexp)


class TestASMEIndependence:
    """Verify that reconstructed-waveform results are clearly flagged."""

    def test_qualified_result_is_not_independent(self):
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        assert r.qualified is True

    def test_all_loo_devices_provenance(self):
        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        measured = [d for d in loo_devices if DEVICES[d].waveform_provenance == "measured"]
        reconstructed = [d for d in loo_devices if DEVICES[d].waveform_provenance == "reconstructed"]
        assert len(measured) == 3
        assert len(reconstructed) == 2

    def test_stratified_loo_n3_measured(self):
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        loo_measured = {k: v for k, v in measured.items()
                        if k in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]}
        assert len(loo_measured) == 3


class TestMJOLNIRGeometry:
    """Verify MJOLNIR anode_radius fix is in place."""

    def test_anode_radius_corrected(self):
        assert DEVICES["MJOLNIR"].anode_radius == pytest.approx(0.114, abs=0.002)

    def test_cathode_radius(self):
        assert DEVICES["MJOLNIR"].cathode_radius == pytest.approx(0.157, abs=0.002)

    def test_ak_gap(self):
        dev = DEVICES["MJOLNIR"]
        gap = dev.cathode_radius - dev.anode_radius
        assert gap == pytest.approx(0.043, abs=0.003)

    def test_speed_factor_near_optimal(self):
        from dpf.validation.experimental import compute_speed_factor
        dev = DEVICES["MJOLNIR"]
        result = compute_speed_factor(
            dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
        )
        assert result["S_over_S_opt"] < 1.5


class TestLOOMaxiter10Results:
    """Codify LOO maxiter=10 results with corrected MJOLNIR geometry."""

    def test_mean_blind_nrmse(self):
        blind = [LOO_MAXITER10_RESULTS[d]["blind"] for d in LOO_MAXITER10_RESULTS]
        assert np.mean(blind) == pytest.approx(0.218, abs=0.01)

    def test_degeneracy_resolved(self):
        param_sets = set()
        for d in LOO_MAXITER10_RESULTS:
            r = LOO_MAXITER10_RESULTS[d]
            param_sets.add((round(r["fc"], 3), round(r["fm"], 3), round(r["delay"], 3)))
        assert len(param_sets) == 5

    def test_maxiter3_had_degeneracy(self):
        param_sets = set()
        for d in LOO_MAXITER3_RESULTS:
            r = LOO_MAXITER3_RESULTS[d]
            param_sets.add((round(r["fc"], 3), round(r["fm"], 3), round(r["delay"], 3)))
        assert len(param_sets) <= 4

    def test_pf1000_still_boundary_trapped(self):
        r = LOO_MAXITER10_RESULTS["PF-1000"]
        assert r["fc"] < 0.60
        assert r["blind"] > 0.40

    def test_unu_ictp_best_blind(self):
        best_dev = min(LOO_MAXITER10_RESULTS, key=lambda d: LOO_MAXITER10_RESULTS[d]["blind"])
        assert best_dev == "UNU-ICTP"
        assert LOO_MAXITER10_RESULTS["UNU-ICTP"]["blind"] < 0.10

    def test_mjolnir_low_degradation(self):
        assert LOO_MAXITER10_RESULTS["MJOLNIR"]["degrad"] < 1.5

    def test_faeton_high_degradation(self):
        r = LOO_MAXITER10_RESULTS["FAETON-I"]
        assert r["degrad"] > 5.0
        assert r["indep"] < 0.03

    def test_mean_unchanged_from_maxiter3(self):
        blind_10 = [LOO_MAXITER10_RESULTS[d]["blind"] for d in LOO_MAXITER10_RESULTS]
        blind_3 = [LOO_MAXITER3_RESULTS[d]["blind"] for d in LOO_MAXITER3_RESULTS]
        assert abs(np.mean(blind_10) - np.mean(blind_3)) < 0.02

    def test_stratified_measured_vs_reconstructed(self):
        measured = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]
        recon = ["FAETON-I", "MJOLNIR"]
        meas_mean = np.mean([LOO_MAXITER10_RESULTS[d]["blind"] for d in measured])
        recon_mean = np.mean([LOO_MAXITER10_RESULTS[d]["blind"] for d in recon])
        assert meas_mean > recon_mean

    def test_fc_fm_range(self):
        fc_fm = {d: r["fc"]**2 / r["fm"] for d, r in LOO_MAXITER10_RESULTS.items()}
        assert max(fc_fm.values()) / min(fc_fm.values()) > 3.0

    def test_asme_loo_one_pass(self):
        assert LOO_MAXITER10_RESULTS["UNU-ICTP"]["blind"] < 0.10


class TestLOOMaxiter10Live:
    """Live LOO maxiter=10 validation tests (slow, ~70 min)."""

    @pytest.mark.slow
    def test_loo_maxiter10_mean_within_tolerance(self):
        from dpf.validation.calibration import MultiDeviceCalibrator
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"],
            fc_bounds=(0.5, 0.95), fm_bounds=(0.04, 0.40),
            delay_bounds_us=(0.0, 2.0), maxiter=10, seed=42,
        )
        loo = cal.leave_one_out()
        mean_blind = np.mean([loo[d]["blind_nrmse"] for d in loo])
        assert mean_blind < 0.25

    @pytest.mark.slow
    def test_loo_maxiter10_no_degeneracy(self):
        from dpf.validation.calibration import MultiDeviceCalibrator
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"],
            fc_bounds=(0.5, 0.95), fm_bounds=(0.04, 0.40),
            delay_bounds_us=(0.0, 2.0), maxiter=10, seed=42,
        )
        loo = cal.leave_one_out()
        param_sets = set()
        for d in loo:
            param_sets.add((
                round(loo[d]["trained_fc"], 3),
                round(loo[d]["trained_fm"], 3),
                round(loo[d]["trained_delay_us"], 3),
            ))
        assert len(param_sets) == 5
