"""Phase Y: Cross-validation framework tests.

Tests for the CrossValidator and CrossValidationResult classes in
dpf.validation.calibration, covering:
1. CrossValidationResult dataclass construction and field access.
2. CrossValidator instantiation.
3. CrossValidator.validate() return type and result structure.
4. Generalization score range and formula verification.
5. Integration test: validate() with real LeeModel runs (slow).
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from dpf.validation.calibration import (
    CalibrationResult,
    CrossValidationResult,
    CrossValidator,
)


# ============================================================
# Helpers
# ============================================================

def _make_calibration_result(
    best_fc: float = 0.72,
    best_fm: float = 0.64,
    peak_current_error: float = 0.12,
    timing_error: float = 0.08,
    objective_value: float = 0.10,
    n_evals: int = 30,
    converged: bool = True,
    device_name: str = "PF-1000",
) -> CalibrationResult:
    """Return a fully populated CalibrationResult for use in unit tests."""
    return CalibrationResult(
        best_fc=best_fc,
        best_fm=best_fm,
        peak_current_error=peak_current_error,
        timing_error=timing_error,
        objective_value=objective_value,
        n_evals=n_evals,
        converged=converged,
        device_name=device_name,
    )


def _make_crossval_result(
    train_device: str = "PF-1000",
    test_device: str = "NX2",
    prediction_peak_error: float = 0.20,
    prediction_timing_error: float = 0.15,
) -> CrossValidationResult:
    """Return a CrossValidationResult constructed directly from known values."""
    avg_err = 0.5 * (prediction_peak_error + prediction_timing_error)
    generalization_score = max(1.0 - avg_err, 0.0)
    return CrossValidationResult(
        train_device=train_device,
        test_device=test_device,
        calibration=_make_calibration_result(device_name=train_device),
        prediction_peak_error=prediction_peak_error,
        prediction_timing_error=prediction_timing_error,
        generalization_score=generalization_score,
    )


# ============================================================
# TestCrossValidationResult
# ============================================================

class TestCrossValidationResult:
    """Tests for the CrossValidationResult dataclass."""

    def test_dataclass_fields(self):
        """CrossValidationResult can be constructed with all required fields."""
        cal_result = _make_calibration_result()
        cvr = CrossValidationResult(
            train_device="PF-1000",
            test_device="NX2",
            calibration=cal_result,
            prediction_peak_error=0.20,
            prediction_timing_error=0.15,
            generalization_score=0.825,
        )
        assert cvr.train_device == "PF-1000"
        assert cvr.test_device == "NX2"
        assert cvr.calibration is cal_result
        assert cvr.prediction_peak_error == pytest.approx(0.20)
        assert cvr.prediction_timing_error == pytest.approx(0.15)
        assert cvr.generalization_score == pytest.approx(0.825)

    def test_generalization_formula(self):
        """generalization_score == 1 - 0.5*(prediction_peak_error + prediction_timing_error).

        For prediction_peak_error=0.1, prediction_timing_error=0.3:
            generalization_score = 1 - 0.5*(0.1 + 0.3) = 1 - 0.2 = 0.8
        """
        avg_err = 0.5 * (0.1 + 0.3)
        expected_score = 1.0 - avg_err  # == 0.8

        cvr = CrossValidationResult(
            train_device="PF-1000",
            test_device="NX2",
            calibration=_make_calibration_result(),
            prediction_peak_error=0.1,
            prediction_timing_error=0.3,
            generalization_score=expected_score,
        )
        assert cvr.generalization_score == pytest.approx(0.8, abs=1e-9)

    def test_is_dataclass(self):
        """CrossValidationResult is a proper dataclass."""
        assert dataclasses.is_dataclass(CrossValidationResult)

    def test_asdict_serialization(self):
        """CrossValidationResult can be serialized via dataclasses.asdict."""
        cvr = _make_crossval_result(
            train_device="PF-1000",
            test_device="NX2",
            prediction_peak_error=0.20,
            prediction_timing_error=0.15,
        )
        d = dataclasses.asdict(cvr)
        assert isinstance(d, dict)
        assert d["train_device"] == "PF-1000"
        assert d["test_device"] == "NX2"
        assert d["prediction_peak_error"] == pytest.approx(0.20)
        assert d["prediction_timing_error"] == pytest.approx(0.15)
        # calibration should be a nested dict
        assert isinstance(d["calibration"], dict)
        assert "best_fc" in d["calibration"]

    def test_field_names(self):
        """CrossValidationResult has exactly the expected field names."""
        field_names = {f.name for f in dataclasses.fields(CrossValidationResult)}
        expected = {
            "train_device",
            "test_device",
            "calibration",
            "prediction_peak_error",
            "prediction_timing_error",
            "generalization_score",
        }
        assert field_names == expected

    def test_calibration_field_is_calibration_result(self):
        """calibration field holds a CalibrationResult instance."""
        cvr = _make_crossval_result()
        assert isinstance(cvr.calibration, CalibrationResult)

    def test_generalization_score_floored_at_zero(self):
        """generalization_score = max(1 - avg_error, 0.0) is never negative.

        When avg_prediction_error > 1.0, CrossValidator clamps to 0.
        Constructing a result manually with score=0.0 should still pass.
        """
        cvr = CrossValidationResult(
            train_device="PF-1000",
            test_device="NX2",
            calibration=_make_calibration_result(),
            prediction_peak_error=2.0,
            prediction_timing_error=2.0,
            generalization_score=0.0,  # clamped by CrossValidator
        )
        assert cvr.generalization_score == pytest.approx(0.0)


# ============================================================
# TestCrossValidator
# ============================================================

class TestCrossValidator:
    """Tests for the CrossValidator class."""

    def test_instantiation(self):
        """CrossValidator() creates without error and needs no arguments."""
        cv = CrossValidator()
        assert cv is not None

    def test_validate_method_exists(self):
        """CrossValidator has a validate() method."""
        cv = CrossValidator()
        assert hasattr(cv, "validate")
        assert callable(cv.validate)

    def test_validate_returns_result_type(self, monkeypatch):
        """validate() with mocked LeeModel returns a CrossValidationResult."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.18
        mock_comparison.timing_error = 0.10
        mock_comparison.waveform_nrmse = float("nan")

        # Patch LeeModelCalibrator._run_comparison so no real LeeModel runs
        from dpf.validation.calibration import LeeModelCalibrator

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )

        # Patch LeeModel.compare_with_experiment for the test-device prediction step
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)
        assert isinstance(result, CrossValidationResult)

    def test_validate_result_fields_present(self, monkeypatch):
        """The CrossValidationResult returned by validate() has all expected fields."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.15
        mock_comparison.timing_error = 0.12
        mock_comparison.waveform_nrmse = float("nan")

        from dpf.validation.calibration import LeeModelCalibrator
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )
        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)

        assert hasattr(result, "train_device")
        assert hasattr(result, "test_device")
        assert hasattr(result, "calibration")
        assert hasattr(result, "prediction_peak_error")
        assert hasattr(result, "prediction_timing_error")
        assert hasattr(result, "generalization_score")

    def test_validate_train_test_devices_recorded(self, monkeypatch):
        """validate() records train_device and test_device in the result."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.20
        mock_comparison.timing_error = 0.10
        mock_comparison.waveform_nrmse = float("nan")

        from dpf.validation.calibration import LeeModelCalibrator
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )
        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)

        assert result.train_device == "PF-1000"
        assert result.test_device == "NX2"

    def test_validate_generalization_score_range_mocked(self, monkeypatch):
        """generalization_score from validate() is clamped to [0, 1]."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.25
        mock_comparison.timing_error = 0.20
        mock_comparison.waveform_nrmse = float("nan")

        from dpf.validation.calibration import LeeModelCalibrator
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )
        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)

        assert 0.0 <= result.generalization_score <= 1.0

    def test_validate_generalization_formula_consistent(self, monkeypatch):
        """generalization_score == max(1 - 0.5*(peak_err + timing_err), 0)."""
        peak_err = 0.30
        timing_err = 0.20

        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = peak_err
        mock_comparison.timing_error = timing_err
        mock_comparison.waveform_nrmse = float("nan")

        from dpf.validation.calibration import LeeModelCalibrator
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )
        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)

        expected_score = max(1.0 - 0.5 * (peak_err + timing_err), 0.0)
        assert result.generalization_score == pytest.approx(expected_score, abs=1e-9)

    def test_validate_calibration_device_is_train(self, monkeypatch):
        """The calibration sub-result records the train_device name."""
        mock_comparison = MagicMock()
        mock_comparison.peak_current_error = 0.12
        mock_comparison.timing_error = 0.08
        mock_comparison.waveform_nrmse = float("nan")

        from dpf.validation.calibration import LeeModelCalibrator
        from dpf.validation import lee_model_comparison as lmc_mod

        monkeypatch.setattr(
            LeeModelCalibrator,
            "_run_comparison",
            lambda self, fc, fm: mock_comparison,
        )
        monkeypatch.setattr(
            lmc_mod.LeeModel,
            "compare_with_experiment",
            lambda self, device: mock_comparison,
        )

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=3)

        assert result.calibration.device_name == "PF-1000"


# ============================================================
# TestCrossValidatorIntegration (slow — runs real LeeModel)
# ============================================================

class TestCrossValidatorIntegration:
    """Integration tests that run the actual scipy optimizer + LeeModel.

    These are marked slow because each test invokes scipy.optimize.minimize
    with real Lee model evaluations.
    """

    @pytest.mark.slow
    def test_validate_returns_result(self):
        """CrossValidator().validate('PF-1000', 'NX2', maxiter=5) returns CrossValidationResult."""
        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=5)
        assert isinstance(result, CrossValidationResult)

    @pytest.mark.slow
    def test_generalization_score_range(self):
        """generalization_score from a real validate() run is in [0, 1]."""
        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=5)
        assert 0.0 <= result.generalization_score <= 1.0
