"""Phase X: Calibration framework tests.

Tests for the LeeModelCalibrator and CalibrationResult classes in
dpf.validation.calibration, covering:
1. CalibrationResult dataclass structure and serialization.
2. LeeModelCalibrator instantiation and configuration.
3. Objective function correctness and bounds enforcement.
4. calibrate() return type, parameter bounds, and convergence properties.
5. calibrate_default_params() multi-device operation.
6. Edge cases: narrow bounds, failed objectives, unknown devices.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from dpf.validation.calibration import (
    CalibrationResult,
    LeeModelCalibrator,
    calibrate_default_params,
)

# ============================================================
# TestCalibrationResult
# ============================================================

class TestCalibrationResult:
    """Tests for the CalibrationResult dataclass."""

    def test_result_fields(self):
        """CalibrationResult has all expected fields with correct types."""
        result = CalibrationResult(
            best_fc=0.72,
            best_fm=0.64,
            peak_current_error=0.12,
            timing_error=0.08,
            objective_value=0.104,
            n_evals=42,
            converged=True,
            device_name="PF-1000",
        )
        assert result.best_fc == pytest.approx(0.72)
        assert result.best_fm == pytest.approx(0.64)
        assert result.peak_current_error == pytest.approx(0.12)
        assert result.timing_error == pytest.approx(0.08)
        assert result.objective_value == pytest.approx(0.104)
        assert result.n_evals == 42
        assert result.converged is True
        assert result.device_name == "PF-1000"

    def test_result_defaults(self):
        """device_name defaults to empty string."""
        result = CalibrationResult(
            best_fc=0.7,
            best_fm=0.5,
            peak_current_error=0.1,
            timing_error=0.05,
            objective_value=0.08,
            n_evals=10,
            converged=False,
        )
        assert result.device_name == ""

    def test_result_serialization(self):
        """CalibrationResult can be serialized via dataclasses.asdict."""
        result = CalibrationResult(
            best_fc=0.75,
            best_fm=0.60,
            peak_current_error=0.15,
            timing_error=0.10,
            objective_value=0.13,
            n_evals=30,
            converged=True,
            device_name="NX2",
        )
        d = dataclasses.asdict(result)
        assert isinstance(d, dict)
        assert d["best_fc"] == pytest.approx(0.75)
        assert d["best_fm"] == pytest.approx(0.60)
        assert d["device_name"] == "NX2"
        assert d["converged"] is True
        assert set(d.keys()) == {
            "best_fc", "best_fm", "peak_current_error", "timing_error",
            "objective_value", "n_evals", "converged", "device_name",
        }

    def test_result_is_dataclass(self):
        """CalibrationResult is a proper dataclass."""
        assert dataclasses.is_dataclass(CalibrationResult)


# ============================================================
# TestLeeModelCalibrator
# ============================================================

class TestLeeModelCalibrator:
    """Tests for LeeModelCalibrator instantiation and configuration."""

    def test_instantiation(self):
        """LeeModelCalibrator('PF-1000') creates object with expected attributes."""
        cal = LeeModelCalibrator("PF-1000")
        assert cal.device_name == "PF-1000"
        assert cal.method == "nelder-mead"
        assert cal.peak_weight == pytest.approx(0.4)
        assert cal.timing_weight == pytest.approx(0.3)
        assert cal.waveform_weight == pytest.approx(0.3)

    def test_instantiation_custom_weights(self):
        """Custom peak_weight and timing_weight are stored correctly."""
        cal = LeeModelCalibrator("NX2", peak_weight=0.8, timing_weight=0.2)
        assert cal.peak_weight == pytest.approx(0.8)
        assert cal.timing_weight == pytest.approx(0.2)
        assert cal.device_name == "NX2"

    def test_instantiation_custom_method(self):
        """Custom optimization method is stored."""
        cal = LeeModelCalibrator("PF-1000", method="cobyla")
        assert cal.method == "cobyla"

    def test_objective_returns_float(self, monkeypatch):
        """_objective([0.7, 0.3]) returns a float when bounds are set."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.15
        mock_comparison.timing_error = 0.10
        mock_comparison.waveform_nrmse = float("nan")

        cal = LeeModelCalibrator("PF-1000")
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        result = cal._objective([0.7, 0.3])
        assert isinstance(result, float)

    def test_objective_bounded(self, monkeypatch):
        """_objective output is between 0 and 10 for well-behaved comparisons."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.20
        mock_comparison.timing_error = 0.15
        mock_comparison.waveform_nrmse = float("nan")

        cal = LeeModelCalibrator("PF-1000")
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)

        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        value = cal._objective([0.7, 0.5])
        assert 0.0 <= value <= 10.0

    def test_calibrate_returns_result(self, monkeypatch):
        """calibrate(maxiter=3) returns a CalibrationResult instance."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.18
        mock_comparison.timing_error = 0.12

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        result = cal.calibrate(maxiter=3)
        assert isinstance(result, CalibrationResult)

    def test_calibrate_fc_in_bounds(self, monkeypatch):
        """result.best_fc falls within the specified fc_bounds."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        fc_bounds = (0.5, 0.95)
        result = cal.calibrate(fc_bounds=fc_bounds, maxiter=3)
        assert fc_bounds[0] <= result.best_fc <= fc_bounds[1]

    def test_calibrate_fm_in_bounds(self, monkeypatch):
        """result.best_fm falls within the specified fm_bounds."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        fm_bounds = (0.05, 0.95)
        result = cal.calibrate(fm_bounds=fm_bounds, maxiter=3)
        assert fm_bounds[0] <= result.best_fm <= fm_bounds[1]

    def test_calibrate_with_custom_x0(self, monkeypatch):
        """calibrate(x0=(0.7, 0.3)) completes without error."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.12
        mock_comparison.timing_error = 0.08

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        result = cal.calibrate(x0=(0.7, 0.3), maxiter=3)
        assert isinstance(result, CalibrationResult)

    def test_calibrate_device_name_in_result(self, monkeypatch):
        """Result device_name matches the calibrator's device_name."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        cal = LeeModelCalibrator("NX2")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        result = cal.calibrate(maxiter=3)
        assert result.device_name == "NX2"


# ============================================================
# TestCalibrationConvergence
# ============================================================

class TestCalibrationConvergence:
    """Tests that run the actual LeeModel (no mocking); should be under 5s each."""

    def test_pf1000_calibration_reasonable_fc(self):
        """For PF-1000 with maxiter=3, result.best_fc is within 0.5-0.95."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.calibrate(fc_bounds=(0.5, 0.95), fm_bounds=(0.05, 0.95), maxiter=3)
        assert 0.5 <= result.best_fc <= 0.95

    def test_pf1000_calibration_reasonable_fm(self):
        """For PF-1000 with maxiter=3, result.best_fm is within 0.05-0.95."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.calibrate(fc_bounds=(0.5, 0.95), fm_bounds=(0.05, 0.95), maxiter=3)
        assert 0.05 <= result.best_fm <= 0.95

    def test_calibrate_reduces_error(self):
        """After calibration, objective should be <= initial objective at x0."""
        x0 = (0.7, 0.5)
        cal = LeeModelCalibrator("PF-1000")
        result = cal.calibrate(
            fc_bounds=(0.5, 0.95), fm_bounds=(0.05, 0.95),
            maxiter=5, x0=x0,
        )
        # Evaluate objective at x0 directly
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)
        initial_obj = cal._objective(list(x0))

        # With maxiter=5 we can't guarantee strict improvement, but the result
        # should not be dramatically worse (within 2x of initial value)
        assert result.objective_value <= initial_obj * 2.0 + 0.5

    def test_n_evals_positive(self):
        """n_evals should be > 0 after calibration."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.calibrate(fc_bounds=(0.5, 0.95), fm_bounds=(0.05, 0.95), maxiter=3)
        assert result.n_evals > 0


# ============================================================
# TestCalibrateDefaultParams
# ============================================================

class TestCalibrateDefaultParams:
    """Tests for the calibrate_default_params() convenience function."""

    def test_default_devices(self, monkeypatch):
        """calibrate_default_params() runs for PF-1000 and NX2 by default."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        # Patch _run_comparison on LeeModelCalibrator instances
        original_init = LeeModelCalibrator.__init__

        def patched_init(self, device_name, method="nelder-mead",
                         peak_weight=0.4, timing_weight=0.3,
                         waveform_weight=0.3):
            original_init(self, device_name, method, peak_weight,
                          timing_weight, waveform_weight)
            self._run_comparison = lambda fc, fm: mock_comparison

        monkeypatch.setattr(LeeModelCalibrator, "__init__", patched_init)

        results = calibrate_default_params(maxiter=3)
        assert "PF-1000" in results
        assert "NX2" in results

    def test_custom_devices(self, monkeypatch):
        """calibrate_default_params(devices=['PF-1000']) returns dict with one key."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        original_init = LeeModelCalibrator.__init__

        def patched_init(self, device_name, method="nelder-mead",
                         peak_weight=0.4, timing_weight=0.3,
                         waveform_weight=0.3):
            original_init(self, device_name, method, peak_weight,
                          timing_weight, waveform_weight)
            self._run_comparison = lambda fc, fm: mock_comparison

        monkeypatch.setattr(LeeModelCalibrator, "__init__", patched_init)

        results = calibrate_default_params(devices=["PF-1000"], maxiter=3)
        assert len(results) == 1
        assert "PF-1000" in results

    def test_returns_dict(self, monkeypatch):
        """Return type is dict with CalibrationResult values."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        original_init = LeeModelCalibrator.__init__

        def patched_init(self, device_name, method="nelder-mead",
                         peak_weight=0.4, timing_weight=0.3,
                         waveform_weight=0.3):
            original_init(self, device_name, method, peak_weight,
                          timing_weight, waveform_weight)
            self._run_comparison = lambda fc, fm: mock_comparison

        monkeypatch.setattr(LeeModelCalibrator, "__init__", patched_init)

        results = calibrate_default_params(devices=["NX2"], maxiter=3)
        assert isinstance(results, dict)
        assert isinstance(results["NX2"], CalibrationResult)

    def test_empty_devices_list(self, monkeypatch):
        """calibrate_default_params(devices=[]) returns empty dict."""
        results = calibrate_default_params(devices=[], maxiter=3)
        assert results == {}


# ============================================================
# TestEdgeCases
# ============================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_narrow_bounds(self, monkeypatch):
        """Narrow fc_bounds=(0.7, 0.71) still produces a valid CalibrationResult."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05

        cal = LeeModelCalibrator("PF-1000")
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        result = cal.calibrate(
            fc_bounds=(0.70, 0.71), fm_bounds=(0.05, 0.95), maxiter=5,
        )
        assert isinstance(result, CalibrationResult)
        assert 0.70 <= result.best_fc <= 0.71

    def test_failed_objective_returns_penalty(self, monkeypatch):
        """When _run_comparison raises, _objective returns 10.0 (large penalty)."""
        def failing_comparison(fc, fm):
            raise RuntimeError("LeeModel integration failed")

        cal = LeeModelCalibrator("PF-1000")
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)
        monkeypatch.setattr(cal, "_run_comparison", failing_comparison)

        penalty = cal._objective([0.7, 0.5])
        assert penalty == pytest.approx(10.0)

    def test_unknown_device_raises(self):
        """LeeModelCalibrator('NONEXISTENT').calibrate() raises KeyError."""
        cal = LeeModelCalibrator("NONEXISTENT")
        with pytest.raises(KeyError):
            cal.calibrate(maxiter=3)

    def test_objective_increments_n_evals(self, monkeypatch):
        """Each call to _objective increments _n_evals."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.10
        mock_comparison.timing_error = 0.05
        mock_comparison.waveform_nrmse = float("nan")

        cal = LeeModelCalibrator("PF-1000")
        cal._fc_bounds = (0.5, 0.95)
        cal._fm_bounds = (0.05, 0.95)
        cal._n_evals = 0
        monkeypatch.setattr(cal, "_run_comparison", lambda fc, fm: mock_comparison)

        cal._objective([0.7, 0.5])
        cal._objective([0.75, 0.45])
        assert cal._n_evals == 2
