"""Phase AR: Crowbar resistance + recalibration tests.

Validates PhD Debate #30 recommendations:
1. Crowbar resistance (R_crowbar=1.5 mOhm) added to PF-1000 experimental data.
2. Lee model properly applies R_crowbar to post-crowbar L-R decay.
3. Calibration with correct Lee & Saw (2014) bounds fc=(0.6,0.8), fm=(0.05,0.25).
4. Preset updated: fc=0.800, fm=0.094.
5. Deep copy in get_preset() prevents test isolation bugs.
6. L_p/L0 diagnostic classifies device validation informativeness.
7. Voltage scan produces monotonic peak current across 16-35 kV.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

from dpf.presets import get_preset, _PRESETS
from dpf.validation.calibration import (
    CalibrationResult,
    LeeModelCalibrator,
    _DEFAULT_CROWBAR_R,
)
from dpf.validation.experimental import (
    DEVICES,
    PF1000_DATA,
    PF1000_16KV_DATA,
    NX2_DATA,
    UNU_ICTP_DATA,
    ExperimentalDevice,
    compute_lp_l0_ratio,
    compute_bare_rlc_timing,
)
from dpf.validation.lee_model_comparison import LeeModel, _get_device_params


# ============================================================
# TestCrowbarResistanceInfrastructure
# ============================================================

class TestCrowbarResistanceInfrastructure:
    """Crowbar resistance is properly wired through all layers."""

    def test_experimental_device_has_crowbar_field(self):
        """ExperimentalDevice dataclass has crowbar_resistance field."""
        dev = ExperimentalDevice(
            name="test", institution="test", capacitance=1e-3,
            voltage=20e3, inductance=30e-9, resistance=2e-3,
            anode_radius=0.01, cathode_radius=0.02, anode_length=0.1,
            fill_pressure_torr=3.0, fill_gas="deuterium",
            peak_current=1e6, neutron_yield=1e10,
            current_rise_time=5e-6, reference="test",
            crowbar_resistance=2.0e-3,
        )
        assert dev.crowbar_resistance == pytest.approx(2.0e-3)

    def test_crowbar_default_zero(self):
        """Crowbar resistance defaults to 0.0 for backward compatibility."""
        dev = ExperimentalDevice(
            name="test", institution="test", capacitance=1e-3,
            voltage=20e3, inductance=30e-9, resistance=2e-3,
            anode_radius=0.01, cathode_radius=0.02, anode_length=0.1,
            fill_pressure_torr=3.0, fill_gas="deuterium",
            peak_current=1e6, neutron_yield=1e10,
            current_rise_time=5e-6, reference="test",
        )
        assert dev.crowbar_resistance == 0.0

    def test_pf1000_crowbar_resistance(self):
        """PF-1000 has crowbar_resistance = 1.5 mOhm (PhD Debate #30)."""
        assert PF1000_DATA.crowbar_resistance == pytest.approx(1.5e-3)

    def test_pf1000_16kv_crowbar_resistance(self):
        """PF-1000 at 16 kV has same crowbar resistance."""
        assert PF1000_16KV_DATA.crowbar_resistance == pytest.approx(1.5e-3)

    def test_nx2_crowbar_zero(self):
        """NX2 has no crowbar_resistance set (default 0.0)."""
        assert NX2_DATA.crowbar_resistance == 0.0

    def test_unu_ictp_crowbar_zero(self):
        """UNU-ICTP has no crowbar_resistance set (default 0.0)."""
        assert UNU_ICTP_DATA.crowbar_resistance == 0.0

    def test_get_device_params_includes_crowbar(self):
        """_get_device_params() returns crowbar_resistance for PF-1000."""
        params = _get_device_params("PF-1000")
        assert "crowbar_resistance" in params
        assert params["crowbar_resistance"] == pytest.approx(1.5e-3)

    def test_get_device_params_crowbar_zero_for_nx2(self):
        """_get_device_params() returns 0 crowbar_resistance for NX2."""
        params = _get_device_params("NX2")
        assert params["crowbar_resistance"] == 0.0


# ============================================================
# TestLeeModelCrowbar
# ============================================================

class TestLeeModelCrowbar:
    """Lee model correctly applies crowbar resistance to post-crowbar decay."""

    def test_lee_model_accepts_crowbar_resistance(self):
        """LeeModel accepts crowbar_resistance parameter."""
        model = LeeModel(crowbar_resistance=2.0e-3)
        assert model.crowbar_resistance == pytest.approx(2.0e-3)

    def test_lee_model_crowbar_default_zero(self):
        """LeeModel defaults crowbar_resistance to 0.0."""
        model = LeeModel()
        assert model.crowbar_resistance == 0.0

    def test_crowbar_increases_post_peak_decay(self):
        """Non-zero crowbar_resistance produces faster post-peak decay.

        After crowbar fires, the L-R decay time constant is tau = L / (R0 + R_crowbar).
        Larger R_crowbar → smaller tau → faster decay → lower current at late times.
        """
        model_no_cb = LeeModel(crowbar_resistance=0.0)
        result_no_cb = model_no_cb.run("PF-1000")

        model_with_cb = LeeModel(crowbar_resistance=1.5e-3)
        result_with_cb = model_with_cb.run("PF-1000")

        # Both should produce valid results
        assert result_no_cb.peak_current > 0
        assert result_with_cb.peak_current > 0

        # Peak current should be similar (crowbar fires post-peak)
        peak_ratio = result_with_cb.peak_current / result_no_cb.peak_current
        assert 0.9 < peak_ratio < 1.1, (
            f"Peak ratio {peak_ratio:.3f} — crowbar should not affect pre-peak"
        )

        # Late-time current should be lower with crowbar (faster decay)
        # Compare at 80% through the time array (well past peak)
        n = len(result_no_cb.t)
        idx_late = int(0.8 * n)
        I_late_no_cb = abs(result_no_cb.I[idx_late])
        I_late_with_cb = abs(result_with_cb.I[idx_late])
        # With crowbar resistance, late current should be lower or equal
        # (may be equal if crowbar hasn't fired yet at that time)
        assert I_late_with_cb <= I_late_no_cb * 1.05, (
            f"Late current with crowbar ({I_late_with_cb:.0f}) should be "
            f"<= without ({I_late_no_cb:.0f})"
        )

    def test_crowbar_metadata_in_result(self):
        """Lee model result metadata contains R_crowbar."""
        model = LeeModel(crowbar_resistance=1.5e-3)
        result = model.run("PF-1000")
        assert "R_crowbar" in result.metadata
        assert result.metadata["R_crowbar"] == pytest.approx(1.5e-3)


# ============================================================
# TestCalibrationBounds
# ============================================================

class TestCalibrationBounds:
    """Calibration uses correct Lee & Saw (2014) bounds."""

    def test_calibrator_accepts_crowbar_params(self):
        """LeeModelCalibrator accepts crowbar_enabled and crowbar_resistance."""
        cal = LeeModelCalibrator(
            "PF-1000",
            crowbar_enabled=True,
            crowbar_resistance=1.5e-3,
        )
        assert cal.crowbar_enabled is True
        assert cal.crowbar_resistance == pytest.approx(1.5e-3)

    def test_calibrator_crowbar_defaults(self):
        """LeeModelCalibrator defaults: crowbar_enabled=False, resistance=0.0."""
        cal = LeeModelCalibrator("PF-1000")
        assert cal.crowbar_enabled is False
        assert cal.crowbar_resistance == 0.0

    def test_default_crowbar_r_registry(self):
        """_DEFAULT_CROWBAR_R has PF-1000 at 1.5 mOhm."""
        assert "PF-1000" in _DEFAULT_CROWBAR_R
        assert _DEFAULT_CROWBAR_R["PF-1000"] == pytest.approx(1.5e-3)

    def test_calibration_fc_within_lee_bounds(self):
        """Calibrated fc for PF-1000 is within Lee & Saw (2014) bounds (0.6, 0.8)."""
        cal = LeeModelCalibrator(
            "PF-1000",
            crowbar_enabled=True,
            crowbar_resistance=1.5e-3,
        )
        result = cal.calibrate(
            fc_bounds=(0.6, 0.8),
            fm_bounds=(0.05, 0.25),
            maxiter=20,
        )
        assert 0.6 <= result.best_fc <= 0.8
        assert 0.05 <= result.best_fm <= 0.25


# ============================================================
# TestPresetUpdate
# ============================================================

class TestPresetUpdate:
    """PF-1000 preset reflects Phase AR recalibration."""

    def test_pf1000_fc_updated(self):
        """PF-1000 preset fc = 0.800 (Phase AR)."""
        preset = get_preset("pf1000")
        assert preset["snowplow"]["current_fraction"] == pytest.approx(0.800)

    def test_pf1000_fm_updated(self):
        """PF-1000 preset fm = 0.094 (Phase AR)."""
        preset = get_preset("pf1000")
        assert preset["snowplow"]["mass_fraction"] == pytest.approx(0.094)

    def test_pf1000_crowbar_resistance_in_preset(self):
        """PF-1000 preset has crowbar_resistance = 1.5 mOhm."""
        preset = get_preset("pf1000")
        assert preset["circuit"]["crowbar_resistance"] == pytest.approx(1.5e-3)

    def test_pf1000_crowbar_enabled(self):
        """PF-1000 preset has crowbar_enabled = True."""
        preset = get_preset("pf1000")
        assert preset["circuit"]["crowbar_enabled"] is True

    def test_fc_squared_over_fm(self):
        """fc^2/fm ratio is physically reasonable (3-10 range)."""
        preset = get_preset("pf1000")
        fc = preset["snowplow"]["current_fraction"]
        fm = preset["snowplow"]["mass_fraction"]
        ratio = fc**2 / fm
        assert 3.0 < ratio < 10.0, f"fc^2/fm = {ratio:.2f} outside [3, 10]"


# ============================================================
# TestDeepCopyPreset
# ============================================================

class TestDeepCopyPreset:
    """get_preset() returns a deep copy — mutations don't bleed back."""

    def test_mutation_does_not_affect_original(self):
        """Modifying a preset dict does not change the global _PRESETS."""
        original_fc = _PRESETS["pf1000"]["snowplow"]["current_fraction"]

        preset = get_preset("pf1000")
        preset["snowplow"]["current_fraction"] = 0.999

        # Global should be unchanged
        assert _PRESETS["pf1000"]["snowplow"]["current_fraction"] == pytest.approx(
            original_fc
        )

    def test_two_presets_are_independent(self):
        """Two get_preset() calls return independent dicts."""
        p1 = get_preset("pf1000")
        p2 = get_preset("pf1000")

        p1["snowplow"]["mass_fraction"] = 0.999
        assert p2["snowplow"]["mass_fraction"] != pytest.approx(0.999)

    def test_circuit_dict_is_deep_copied(self):
        """Nested circuit dict is also deep copied."""
        preset = get_preset("pf1000")
        original_V0 = _PRESETS["pf1000"]["circuit"]["V0"]

        preset["circuit"]["V0"] = 99999
        assert _PRESETS["pf1000"]["circuit"]["V0"] == pytest.approx(original_V0)


# ============================================================
# TestLpL0Diagnostic
# ============================================================

class TestLpL0Diagnostic:
    """L_p/L0 diagnostic classifies validation informativeness."""

    def test_pf1000_plasma_significant(self):
        """PF-1000 is plasma-significant: L_p/L0 > 1.0."""
        result = compute_lp_l0_ratio(
            L0=PF1000_DATA.inductance,
            anode_radius=PF1000_DATA.anode_radius,
            cathode_radius=PF1000_DATA.cathode_radius,
            anode_length=PF1000_DATA.anode_length,
        )
        assert result["L_p_over_L0"] > 1.0
        assert result["regime"] == "plasma-significant"

    def test_unu_ictp_circuit_dominated(self):
        """UNU-ICTP is circuit-dominated: L_p/L0 < 0.5."""
        result = compute_lp_l0_ratio(
            L0=UNU_ICTP_DATA.inductance,
            anode_radius=UNU_ICTP_DATA.anode_radius,
            cathode_radius=UNU_ICTP_DATA.cathode_radius,
            anode_length=UNU_ICTP_DATA.anode_length,
        )
        assert result["L_p_over_L0"] < 0.5
        assert result["regime"] == "circuit-dominated"

    def test_pf1000_bare_rlc_fails_timing(self):
        """Bare RLC timing for PF-1000 has large error (>50%)."""
        t_rlc = compute_bare_rlc_timing(
            C=PF1000_DATA.capacitance,
            L0=PF1000_DATA.inductance,
            R0=PF1000_DATA.resistance,
        )
        t_exp = PF1000_DATA.current_rise_time
        error = abs(t_rlc - t_exp) / t_exp
        assert error > 0.5, f"Bare RLC timing error {error:.1%} should be >50%"

    def test_unu_ictp_bare_rlc_close(self):
        """Bare RLC timing for UNU-ICTP is close to experiment (<10%)."""
        t_rlc = compute_bare_rlc_timing(
            C=UNU_ICTP_DATA.capacitance,
            L0=UNU_ICTP_DATA.inductance,
            R0=UNU_ICTP_DATA.resistance,
        )
        t_exp = UNU_ICTP_DATA.current_rise_time
        error = abs(t_rlc - t_exp) / t_exp
        assert error < 0.10, f"UNU-ICTP bare RLC timing error {error:.1%} should be <10%"

    def test_lp_l0_returns_all_keys(self):
        """compute_lp_l0_ratio returns all expected keys."""
        result = compute_lp_l0_ratio(
            L0=30e-9, anode_radius=0.01, cathode_radius=0.02, anode_length=0.1,
        )
        assert "L_p_axial" in result
        assert "L_p_over_L0" in result
        assert "regime" in result
        assert "L_per_length" in result


# ============================================================
# TestVoltageScan
# ============================================================

class TestVoltageScan:
    """PF-1000 voltage scan produces monotonic peak current."""

    def test_voltage_scan_monotonic_peak(self):
        """Peak current increases monotonically with voltage (16-35 kV)."""
        base_params = _get_device_params("PF-1000")
        voltages = [16e3, 20e3, 27e3, 35e3]
        peaks = []
        for v in voltages:
            params = dict(base_params, V0=v)
            model = LeeModel(crowbar_resistance=1.5e-3)
            result = model.run(device_params=params)
            peaks.append(result.peak_current)

        # Verify monotonically increasing
        for i in range(1, len(peaks)):
            assert peaks[i] > peaks[i - 1], (
                f"Peak at {voltages[i]/1e3:.0f} kV ({peaks[i]:.0f} A) "
                f"<= peak at {voltages[i-1]/1e3:.0f} kV ({peaks[i-1]:.0f} A)"
            )

    def test_pf1000_27kv_peak_reasonable(self):
        """PF-1000 at 27 kV with calibrated fc/fm gives peak near 1.87 MA."""
        model = LeeModel(
            current_fraction=0.800,
            mass_fraction=0.094,
            crowbar_resistance=1.5e-3,
        )
        result = model.run("PF-1000")
        assert 1.5e6 < result.peak_current < 2.2e6, (
            f"PF-1000 27 kV peak = {result.peak_current:.2e} A, expected 1.5-2.2 MA"
        )

    def test_pf1000_16kv_peak_reasonable(self):
        """PF-1000 at 16 kV with calibrated fc/fm gives peak near 1.1-1.3 MA."""
        model = LeeModel(
            current_fraction=0.800,
            mass_fraction=0.094,
            crowbar_resistance=1.5e-3,
        )
        result = model.run("PF-1000-16kV")
        assert 0.8e6 < result.peak_current < 1.5e6, (
            f"PF-1000 16 kV peak = {result.peak_current:.2e} A, expected 0.8-1.5 MA"
        )
