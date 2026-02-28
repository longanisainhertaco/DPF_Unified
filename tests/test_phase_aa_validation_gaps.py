"""Phase AA: Validation module gap coverage.

Fills test coverage gaps identified during Phase Z/AA audit:

1. ``_find_first_peak()`` edge cases — length < 3, monotonic signal, multi-peak.
2. ``validate_current_waveform()`` — uncertainty dict keys, UNU-ICTP (no waveform),
   KeyError on unknown device, timing_error and peak_time_sim fields.
3. ``device_to_config_dict()`` — structure, grid shape calculation, circuit params.
4. ``LeeModelCalibrator._objective()`` — NaN waveform term excluded for no-waveform
   devices (key diagnostic for the fm calibration anomaly).
5. ``normalized_rmse()`` — partial time overlap (extrapolation clamp), zero-peak guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.validation.calibration import LeeModelCalibrator
from dpf.validation.experimental import (  # noqa: I001
    PF1000_DATA,
    _find_first_peak,
    device_to_config_dict,
    normalized_rmse,
    validate_current_waveform,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _sine_waveform(
    peak: float = 1.87e6,
    t_peak: float = 5.8e-6,
    n: int = 1000,
    t_end: float = 10e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DPF-like half-sine current waveform."""
    t = np.linspace(0.0, t_end, n)
    I_arr = peak * np.sin(np.pi * t / (2.0 * t_peak))
    I_arr = np.maximum(I_arr, 0.0)
    return t, I_arr


def _make_mock_comparison(
    peak_current_error: float = 0.10,
    timing_error: float = 0.05,
    waveform_nrmse: float = float("nan"),
) -> object:
    """Return a minimal object mimicking LeeModelComparison."""

    class _FakeComparison:
        pass

    c = _FakeComparison()
    c.peak_current_error = peak_current_error
    c.timing_error = timing_error
    c.waveform_nrmse = waveform_nrmse
    return c


# ===========================================================================
# TestFindFirstPeak
# ===========================================================================


class TestFindFirstPeak:
    """Direct tests for the _find_first_peak() helper."""

    def test_short_signal_returns_argmax(self) -> None:
        """For length-2 signal, falls back to argmax."""
        sig = np.array([0.3, 1.0])
        assert _find_first_peak(sig) == 1

    def test_single_element_returns_zero(self) -> None:
        """For length-1 signal, returns 0 (argmax of single element)."""
        sig = np.array([5.0])
        assert _find_first_peak(sig) == 0

    def test_monotonically_rising_returns_last_index(self) -> None:
        """Monotonically rising signal — no local peak found, fallback to argmax."""
        sig = np.arange(1.0, 11.0)  # [1,2,...,10]
        idx = _find_first_peak(sig)
        # Fallback gives argmax = index 9
        assert idx == 9

    def test_single_clear_peak(self) -> None:
        """Signal with one clear interior peak returns that peak index."""
        sig = np.array([0.0, 0.5, 1.0, 0.7, 0.3, 0.1])
        idx = _find_first_peak(sig)
        assert idx == 2

    def test_first_peak_returned_not_later_larger_one(self) -> None:
        """With two well-separated peaks, the FIRST is returned (not global max).

        The sustained-decline criterion requires 3+ consecutive declining
        points after the candidate to confirm a true peak (avoids
        phase-transition artifacts in DPF waveforms).
        """
        # First peak at index 3 (value 0.8), sustained decline [0.5, 0.3, 0.1],
        # then second peak at index 8 (value 1.0).
        sig = np.array([0.0, 0.3, 0.6, 0.8, 0.5, 0.3, 0.1, 0.5, 1.0, 0.6, 0.2])
        idx = _find_first_peak(sig)
        assert idx == 3, (
            f"Expected first peak at index 3, got {idx} (value={sig[idx]:.2f})"
        )

    def test_noise_spike_below_threshold_ignored(self) -> None:
        """A tiny early spike below min_prominence is not returned as first peak."""
        # Spike at index 1 = 4% of global_max (below 5% default threshold)
        sig = np.zeros(20)
        sig[1] = 0.04  # below threshold (global max will be 1.0)
        sig[10] = 1.0
        sig[11] = 0.9
        sig[12] = 0.7
        idx = _find_first_peak(sig)
        # The early tiny spike should be ignored; the real peak is at 10
        assert idx == 10

    def test_custom_min_prominence(self) -> None:
        """Custom min_prominence parameter filters peaks below that fraction."""
        # First peak at 0.3, second at 1.0
        sig = np.array([0.0, 0.1, 0.3, 0.2, 0.0, 0.5, 1.0, 0.8, 0.3])
        # With min_prominence=0.4, the first peak (0.3 = 30% of max=1.0) is excluded
        idx = _find_first_peak(sig, min_prominence=0.4)
        assert idx == 6, (
            f"With min_prominence=0.4, first peak should be at index 6, got {idx}"
        )

    def test_dpf_half_sine_returns_correct_index(self) -> None:
        """Typical DPF half-sine waveform — first peak is at expected index."""
        t, I_arr = _sine_waveform(peak=1.87e6, t_peak=5.8e-6, n=500, t_end=10e-6)
        idx = _find_first_peak(I_arr)
        # The peak of sin(pi*t/(2*5.8us)) at t in [0,10us] is at t=5.8us
        # For n=500 points over 10us: i_peak ~ 5.8/10 * 500 ≈ 290
        t_detected = t[idx]
        assert abs(t_detected - 5.8e-6) < 0.5e-6, (
            f"Detected peak at t={t_detected*1e6:.2f}us, expected ~5.8 us"
        )


# ===========================================================================
# TestValidateCurrentWaveformUncertainty
# ===========================================================================


class TestValidateCurrentWaveformUncertainty:
    """Tests for the uncertainty budget dict in validate_current_waveform()."""

    def test_uncertainty_key_present(self) -> None:
        """validate_current_waveform must return an 'uncertainty' key."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert "uncertainty" in metrics

    def test_uncertainty_dict_has_required_keys(self) -> None:
        """The uncertainty sub-dict must contain all 5 expected keys."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        u = metrics["uncertainty"]
        required = {
            "peak_current_exp_1sigma",
            "rise_time_exp_1sigma",
            "peak_current_combined_1sigma",
            "timing_combined_1sigma",
            "agreement_within_2sigma",
        }
        assert required.issubset(u.keys()), (
            f"Missing uncertainty keys: {required - set(u.keys())}"
        )

    def test_uncertainty_values_are_non_negative(self) -> None:
        """All scalar uncertainty values must be non-negative."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        u = metrics["uncertainty"]
        for key in ("peak_current_exp_1sigma", "rise_time_exp_1sigma",
                    "peak_current_combined_1sigma", "timing_combined_1sigma"):
            assert u[key] >= 0.0, f"Uncertainty key '{key}' = {u[key]} is negative"

    def test_agreement_within_2sigma_is_bool(self) -> None:
        """agreement_within_2sigma must be a bool."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert isinstance(metrics["uncertainty"]["agreement_within_2sigma"], bool)

    def test_perfect_match_agrees_within_2sigma(self) -> None:
        """A simulation matching the experimental peak within 1% passes 2-sigma check."""
        # PF-1000 peak = 1.87 MA, uncertainty = 5%; error 1% << 2*5% → agrees
        t = np.linspace(0, 10e-6, 1000)
        I_arr = 1.87e6 * np.sin(np.pi * t / (2.0 * 5.8e-6))
        I_arr = np.maximum(I_arr, 0.0)
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert metrics["uncertainty"]["agreement_within_2sigma"] is True

    def test_gross_mismatch_agreement_false(self) -> None:
        """A simulation 5× off experimental peak fails 2-sigma agreement check."""
        t = np.linspace(0, 10e-6, 500)
        I_arr = 5 * 1.87e6 * np.sin(np.pi * t / (2.0 * 5.8e-6))  # 5× too high
        I_arr = np.maximum(I_arr, 0.0)
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        # peak_current_error ~ 4.0 >> 2 * 0.05 = 0.10
        assert metrics["uncertainty"]["agreement_within_2sigma"] is False

    def test_combined_uncertainty_geq_experimental(self) -> None:
        """Combined uncertainty (quadrature) >= experimental-only uncertainty."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        u = metrics["uncertainty"]
        assert u["peak_current_combined_1sigma"] >= u["peak_current_exp_1sigma"]
        assert u["timing_combined_1sigma"] >= u["rise_time_exp_1sigma"]

    def test_exp_1sigma_matches_device_data(self) -> None:
        """peak_current_exp_1sigma must equal PF1000_DATA.peak_current_uncertainty."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        u = metrics["uncertainty"]
        assert u["peak_current_exp_1sigma"] == pytest.approx(
            PF1000_DATA.peak_current_uncertainty, rel=1e-9
        )


# ===========================================================================
# TestValidateCurrentWaveformCoverage
# ===========================================================================


class TestValidateCurrentWaveformCoverage:
    """Additional coverage tests for validate_current_waveform()."""

    def test_returns_timing_error(self) -> None:
        """Return dict must contain 'timing_error' key."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert "timing_error" in metrics
        assert isinstance(metrics["timing_error"], float)
        assert metrics["timing_error"] >= 0.0

    def test_returns_peak_time_sim(self) -> None:
        """Return dict must contain 'peak_time_sim' key with a positive value."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert "peak_time_sim" in metrics
        assert metrics["peak_time_sim"] > 0.0

    def test_unu_ictp_no_waveform(self) -> None:
        """UNU-ICTP (like NX2) has no digitized waveform → waveform_available=False."""
        t = np.linspace(0, 8e-6, 500)
        I_arr = 170e3 * np.sin(np.pi * t / (2.0 * 2.8e-6))
        I_arr = np.maximum(I_arr, 0.0)
        metrics = validate_current_waveform(t, I_arr, "UNU-ICTP")
        assert metrics["waveform_available"] is False
        assert math.isnan(metrics["waveform_nrmse"])

    def test_unknown_device_raises_keyerror(self) -> None:
        """KeyError is raised for a device_name not in DEVICES."""
        t, I_arr = _sine_waveform()
        with pytest.raises(KeyError):
            validate_current_waveform(t, I_arr, "FICTIONAL_DPF_DEVICE")

    def test_peak_current_sim_is_positive(self) -> None:
        """peak_current_sim must be > 0 for a non-trivial waveform."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert metrics["peak_current_sim"] > 0.0

    def test_peak_current_exp_matches_device_data(self) -> None:
        """peak_current_exp must match the registered PF-1000 experimental value."""
        t, I_arr = _sine_waveform()
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert metrics["peak_current_exp"] == pytest.approx(
            PF1000_DATA.peak_current, rel=1e-9
        )

    def test_timing_ok_for_well_matched_waveform(self) -> None:
        """A waveform peaking at the correct time should have timing_ok=True."""
        # Half-sine peaking exactly at experimental rise time (5.8 us)
        t = np.linspace(0, 10e-6, 2000)
        I_arr = 1.87e6 * np.sin(np.pi * t / (2.0 * 5.8e-6))
        I_arr = np.maximum(I_arr, 0.0)
        metrics = validate_current_waveform(t, I_arr, "PF-1000")
        assert metrics["timing_ok"] is True

    def test_all_devices_registered(self) -> None:
        """All three devices can be validated without error."""
        for device_name, peak_I, t_peak in [
            ("PF-1000", 1.87e6, 5.8e-6),
            ("NX2", 400e3, 1.8e-6),
            ("UNU-ICTP", 170e3, 2.8e-6),
        ]:
            t = np.linspace(0, 4 * t_peak, 500)
            I_arr = peak_I * np.sin(np.pi * t / (2.0 * t_peak))
            I_arr = np.maximum(I_arr, 0.0)
            metrics = validate_current_waveform(t, I_arr, device_name)
            assert "peak_current_error" in metrics


# ===========================================================================
# TestNormalizedRMSEEdgeCases
# ===========================================================================


class TestNormalizedRMSEEdgeCases:
    """Edge-case tests for normalized_rmse()."""

    def test_partial_overlap_clamps_via_interp(self) -> None:
        """When t_sim domain is smaller than t_exp, np.interp clamps at boundaries.

        normalized_rmse should return a finite, non-negative value — not NaN.
        """
        t_exp = np.linspace(0, 10e-6, 26)
        I_exp = 1.87e6 * np.sin(np.pi * t_exp / (2.0 * 5.8e-6))
        I_exp = np.maximum(I_exp, 0.0)

        # Simulated waveform covers only 0..5 us (partial overlap with t_exp=0..10 us)
        t_sim = np.linspace(0, 5e-6, 200)
        I_sim = 1.87e6 * np.sin(np.pi * t_sim / (2.0 * 5.8e-6))
        I_sim = np.maximum(I_sim, 0.0)

        result = normalized_rmse(t_sim, I_sim, t_exp, I_exp)
        assert math.isfinite(result), (
            f"normalized_rmse with partial overlap should be finite, got {result}"
        )
        assert result >= 0.0

    def test_zero_peak_guard(self) -> None:
        """When I_exp is all zeros, normalized_rmse must not divide by zero.

        The denominator guard ``max(I_peak_exp, 1e-300)`` prevents ZeroDivisionError.
        """
        t = np.linspace(0, 1e-6, 50)
        I_exp = np.zeros(50)
        I_sim = np.ones(50) * 1e3  # Some nonzero sim waveform

        result = normalized_rmse(t, I_sim, t, I_exp)
        assert math.isfinite(result), (
            f"normalized_rmse with zero I_exp should be finite, got {result}"
        )

    def test_coarse_sim_fine_exp(self) -> None:
        """Coarser t_sim interpolated onto finer t_exp returns a finite positive NRMSE."""
        t_exp = np.linspace(0, 10e-6, 200)  # Fine
        I_exp = 1.87e6 * np.sin(np.pi * t_exp / (2.0 * 5.8e-6))

        t_sim = np.linspace(0, 10e-6, 10)  # Very coarse
        I_sim = 1.87e6 * np.sin(np.pi * t_sim / (2.0 * 5.8e-6))

        result = normalized_rmse(t_sim, I_sim, t_exp, I_exp)
        assert math.isfinite(result)
        assert result >= 0.0

    def test_scale_factor_nrmse(self) -> None:
        """I_sim = 2 * I_exp → NRMSE = 1.0 (100% of peak)."""
        t = np.linspace(0, 10e-6, 100)
        I_exp = np.ones(100) * 1.87e6  # Constant 1.87 MA
        I_sim = 2.0 * I_exp

        result = normalized_rmse(t, I_sim, t, I_exp)
        assert result == pytest.approx(1.0, rel=1e-6), (
            f"2× scale factor should give NRMSE=1.0, got {result:.4f}"
        )


# ===========================================================================
# TestDeviceToConfigDict
# ===========================================================================


class TestDeviceToConfigDict:
    """Tests for device_to_config_dict() function."""

    @pytest.mark.parametrize("device_name", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_returns_dict(self, device_name: str) -> None:
        """device_to_config_dict returns a dict for all registered devices."""
        result = device_to_config_dict(device_name)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("device_name", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_required_keys_present(self, device_name: str) -> None:
        """Result contains all required top-level keys."""
        result = device_to_config_dict(device_name)
        required = {"grid_shape", "dx", "sim_time", "rho0", "T0", "ion_mass", "circuit"}
        assert required.issubset(result.keys()), (
            f"Missing keys for {device_name}: {required - set(result.keys())}"
        )

    @pytest.mark.parametrize("device_name", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_circuit_keys_present(self, device_name: str) -> None:
        """circuit sub-dict has C, V0, L0, R0, anode_radius, cathode_radius."""
        result = device_to_config_dict(device_name)
        circuit_required = {"C", "V0", "L0", "R0", "anode_radius", "cathode_radius"}
        assert circuit_required.issubset(result["circuit"].keys()), (
            f"Missing circuit keys for {device_name}: "
            f"{circuit_required - set(result['circuit'].keys())}"
        )

    def test_grid_shape_is_3_element_list(self) -> None:
        """grid_shape must be a 3-element list [nx, ny, nz]."""
        result = device_to_config_dict("PF-1000")
        gs = result["grid_shape"]
        assert isinstance(gs, list) and len(gs) == 3
        assert all(isinstance(n, int) and n > 0 for n in gs)

    def test_grid_shape_minimum_8_per_dim(self) -> None:
        """Each grid dimension must be at least 8 cells."""
        result = device_to_config_dict("PF-1000")
        assert all(n >= 8 for n in result["grid_shape"])

    def test_grid_shape_maximum_256_per_dim(self) -> None:
        """Each grid dimension must not exceed 256 cells."""
        result = device_to_config_dict("PF-1000")
        assert all(n <= 256 for n in result["grid_shape"])

    def test_dx_is_positive(self) -> None:
        """Grid spacing dx must be positive."""
        result = device_to_config_dict("PF-1000")
        assert result["dx"] > 0.0

    def test_sim_time_is_positive(self) -> None:
        """Simulation time must be positive."""
        result = device_to_config_dict("PF-1000")
        assert result["sim_time"] > 0.0

    def test_rho0_positive_from_fill_pressure(self) -> None:
        """Fill gas density rho0 is derived from pressure and must be positive."""
        result = device_to_config_dict("PF-1000")
        assert result["rho0"] > 0.0

    def test_circuit_values_match_device_data(self) -> None:
        """Circuit parameters must match PF1000_DATA."""
        result = device_to_config_dict("PF-1000")
        c = result["circuit"]
        assert c["C"] == pytest.approx(PF1000_DATA.capacitance, rel=1e-9)
        assert c["V0"] == pytest.approx(PF1000_DATA.voltage, rel=1e-9)
        assert c["L0"] == pytest.approx(PF1000_DATA.inductance, rel=1e-9)
        assert c["R0"] == pytest.approx(PF1000_DATA.resistance, rel=1e-9)

    def test_unknown_device_raises_keyerror(self) -> None:
        """device_to_config_dict raises KeyError for an unregistered device name."""
        with pytest.raises(KeyError):
            device_to_config_dict("FICTIONAL_DPF_777")

    def test_pf1000_rho0_order_of_magnitude(self) -> None:
        """PF-1000 fill density should be ~1e-4 kg/m^3 at 3.5 Torr deuterium."""
        result = device_to_config_dict("PF-1000")
        rho0 = result["rho0"]
        # Deuterium at 3.5 Torr, 300 K: rho ~ 3.7e-4 kg/m^3 order of magnitude
        assert 1e-5 < rho0 < 1e-2, (
            f"PF-1000 fill density {rho0:.2e} kg/m^3 out of expected range [1e-5, 1e-2]"
        )

    def test_sim_time_proportional_to_rise_time(self) -> None:
        """sim_time should be a few multiples of current_rise_time."""
        result = device_to_config_dict("PF-1000")
        # sim_time = 4 * current_rise_time
        expected_sim_time = 4.0 * PF1000_DATA.current_rise_time
        assert result["sim_time"] == pytest.approx(expected_sim_time, rel=1e-9)


# ===========================================================================
# TestCalibrationObjectiveNaNHandling
# ===========================================================================


class TestCalibrationObjectiveNaNHandling:
    """Tests for LeeModelCalibrator._objective() NaN waveform term handling.

    This is the key diagnostic for the fm calibration anomaly: when waveform_nrmse
    is NaN (device has no digitized waveform), the waveform term should be silently
    excluded from the objective, reducing effective DOF to 0.
    """

    def test_objective_excludes_nan_waveform_term(self, monkeypatch) -> None:
        """When waveform_nrmse is NaN, _objective uses only peak + timing terms.

        The objective value should equal peak_weight * peak_err + timing_weight * timing_err.
        """
        peak_err = 0.20
        timing_err = 0.10

        mock = _make_mock_comparison(
            peak_current_error=peak_err,
            timing_error=timing_err,
            waveform_nrmse=float("nan"),
        )

        cal = LeeModelCalibrator("NX2", peak_weight=0.4, timing_weight=0.3,
                                  waveform_weight=0.3)
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm, f_mr=None: mock)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        obj = cal._objective(np.array([0.7, 0.1]))

        # MED-3: weights renormalized when waveform unavailable
        total = 0.4 + 0.3  # peak_weight + timing_weight
        expected = (0.4 / total) * peak_err + (0.3 / total) * timing_err
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"Objective with NaN waveform: expected {expected:.4f}, got {obj:.4f}. "
            "Weights are renormalized when waveform is unavailable."
        )

    def test_objective_includes_finite_waveform_term(self, monkeypatch) -> None:
        """When waveform_nrmse is finite, _objective includes all 3 terms."""
        peak_err = 0.20
        timing_err = 0.10
        waveform_nrmse = 0.30

        mock = _make_mock_comparison(
            peak_current_error=peak_err,
            timing_error=timing_err,
            waveform_nrmse=waveform_nrmse,
        )

        cal = LeeModelCalibrator("PF-1000", peak_weight=0.4, timing_weight=0.3,
                                  waveform_weight=0.3)
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm, f_mr=None: mock)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        obj = cal._objective(np.array([0.7, 0.1]))

        expected = 0.4 * peak_err + 0.3 * timing_err + 0.3 * waveform_nrmse
        assert obj == pytest.approx(expected, rel=1e-9), (
            f"Objective with finite waveform: expected {expected:.4f}, got {obj:.4f}. "
            "All 3 terms should contribute."
        )

    def test_objective_zero_waveform_weight_skips_term(self, monkeypatch) -> None:
        """When waveform_weight=0, waveform term is skipped even if nrmse is finite."""
        peak_err = 0.15
        timing_err = 0.08
        waveform_nrmse = 0.50

        mock = _make_mock_comparison(
            peak_current_error=peak_err,
            timing_error=timing_err,
            waveform_nrmse=waveform_nrmse,
        )

        cal = LeeModelCalibrator("PF-1000", peak_weight=0.6, timing_weight=0.4,
                                  waveform_weight=0.0)
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm, f_mr=None: mock)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        obj = cal._objective(np.array([0.7, 0.1]))
        expected = 0.6 * peak_err + 0.4 * timing_err
        assert obj == pytest.approx(expected, rel=1e-9)

    def test_objective_clamps_fc_fm_to_bounds(self, monkeypatch) -> None:
        """_objective clamps params to bounds before calling _run_comparison."""
        received: dict[str, float] = {}

        def mock_comparison(fc: float, fm: float, f_mr: float | None = None) -> object:
            received["fc"] = fc
            received["fm"] = fm
            return _make_mock_comparison()

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", mock_comparison)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        # Pass out-of-bounds params: fc=0.1 < 0.5, fm=0.99 > 0.95
        cal._objective(np.array([0.1, 0.99]))

        assert received["fc"] == pytest.approx(0.5, abs=1e-9), (
            f"fc should be clamped to lower bound 0.5, got {received['fc']}"
        )
        assert received["fm"] == pytest.approx(0.95, abs=1e-9), (
            f"fm should be clamped to upper bound 0.95, got {received['fm']}"
        )

    def test_objective_evals_counter_incremented(self, monkeypatch) -> None:
        """_objective increments _n_evals on each call."""
        mock = _make_mock_comparison()

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm, f_mr=None: mock)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)
        cal._n_evals = 0

        cal._objective(np.array([0.7, 0.1]))
        cal._objective(np.array([0.7, 0.1]))

        assert cal._n_evals == 2

    def test_objective_returns_penalty_on_exception(self, monkeypatch) -> None:
        """_objective returns 10.0 (large penalty) when _run_comparison raises."""
        def _fail(fc: float, fm: float, f_mr: float | None = None) -> object:
            raise RuntimeError("LeeModel crashed")

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", _fail)
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)
        cal._n_evals = 0

        obj = cal._objective(np.array([0.7, 0.1]))
        assert obj == pytest.approx(10.0)
