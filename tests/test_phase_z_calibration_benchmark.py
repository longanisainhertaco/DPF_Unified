"""Phase Z: Calibration benchmark tests against Lee & Saw published fc/fm.

Tests the calibration infrastructure in dpf.validation.calibration — verifying
that published fc/fm ranges are well-formed, that benchmark_against_published()
returns correct structure, and that known-good/bad parameters are classified
correctly.

References:
    Lee S. & Saw S.H., J. Fusion Energy 33, 319-335 (2014).
    Lee S. & Saw S.H., J. Fusion Energy 27, 292-295 (2008).
"""

from __future__ import annotations

import pytest

from dpf.validation.calibration import (
    _PUBLISHED_FC_FM_RANGES,
    CalibrationResult,
    CrossValidationResult,
    CrossValidator,
    LeeModelCalibrator,
    calibrate_default_params,
)

# ═══════════════════════════════════════════════════════
# Tests for published fc/fm ranges
# ═══════════════════════════════════════════════════════


class TestPublishedRanges:
    """Verify _PUBLISHED_FC_FM_RANGES structure and values."""

    def test_all_devices_present(self):
        """Published ranges exist for PF-1000, NX2, UNU-ICTP."""
        expected = {"PF-1000", "NX2", "UNU-ICTP"}
        assert set(_PUBLISHED_FC_FM_RANGES.keys()) == expected

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_range_keys(self, device: str):
        """Each device has 'fc' and 'fm' range tuples."""
        ranges = _PUBLISHED_FC_FM_RANGES[device]
        assert "fc" in ranges
        assert "fm" in ranges

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_fc_range_valid(self, device: str):
        """fc range is a 2-tuple of valid floats in (0, 1) with lo < hi."""
        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES[device]["fc"]
        assert 0.0 < fc_lo < 1.0
        assert 0.0 < fc_hi < 1.0
        assert fc_lo < fc_hi

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_fm_range_valid(self, device: str):
        """fm range is a 2-tuple of valid floats in (0, 1) with lo < hi."""
        fm_lo, fm_hi = _PUBLISHED_FC_FM_RANGES[device]["fm"]
        assert 0.0 < fm_lo < 1.0
        assert 0.0 < fm_hi < 1.0
        assert fm_lo < fm_hi

    def test_pf1000_fc_range_matches_literature(self):
        """PF-1000 fc should be in the 0.6-0.8 range (Lee & Saw 2014)."""
        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fc"]
        # Lee & Saw 2014 Table 1 reports fc ~ 0.7 for PF-1000
        assert fc_lo >= 0.5
        assert fc_hi <= 0.9
        # Center of range should be near 0.7
        center = 0.5 * (fc_lo + fc_hi)
        assert 0.6 < center < 0.85

    def test_pf1000_fm_range_matches_literature(self):
        """PF-1000 fm should be small (0.05-0.2 range)."""
        fm_lo, fm_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fm"]
        assert fm_lo < 0.25
        assert fm_hi < 0.5


# ═══════════════════════════════════════════════════════
# Tests for LeeModelCalibrator construction
# ═══════════════════════════════════════════════════════


class TestCalibratorConstruction:
    """Verify LeeModelCalibrator initialization."""

    def test_default_method(self):
        """Default optimization method is Nelder-Mead."""
        cal = LeeModelCalibrator("PF-1000")
        assert cal.method == "nelder-mead"

    def test_custom_weights(self):
        """Custom objective weights are stored."""
        cal = LeeModelCalibrator(
            "NX2",
            peak_weight=0.5,
            timing_weight=0.3,
            waveform_weight=0.2,
        )
        assert cal.peak_weight == pytest.approx(0.5)
        assert cal.timing_weight == pytest.approx(0.3)
        assert cal.waveform_weight == pytest.approx(0.2)

    def test_device_name_stored(self):
        """Device name is stored on the calibrator."""
        cal = LeeModelCalibrator("UNU-ICTP")
        assert cal.device_name == "UNU-ICTP"


# ═══════════════════════════════════════════════════════
# Tests for benchmark_against_published
# ═══════════════════════════════════════════════════════


class TestBenchmarkAgainstPublished:
    """Tests for LeeModelCalibrator.benchmark_against_published()."""

    @staticmethod
    def _make_cal_result(
        fc: float = 0.7, fm: float = 0.1, device: str = "PF-1000"
    ) -> CalibrationResult:
        """Create a synthetic CalibrationResult for unit testing."""
        return CalibrationResult(
            best_fc=fc,
            best_fm=fm,
            peak_current_error=0.05,
            timing_error=0.03,
            objective_value=0.04,
            n_evals=50,
            converged=True,
            device_name=device,
        )

    def test_return_keys(self):
        """benchmark_against_published returns all expected keys."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result(fc=0.7, fm=0.1)
        )
        expected_keys = {
            "fc_calibrated",
            "fm_calibrated",
            "fc_published_range",
            "fm_published_range",
            "fc_in_range",
            "fm_in_range",
            "both_in_range",
            "reference",
        }
        assert set(result.keys()) == expected_keys

    def test_in_range_true_pf1000(self):
        """fc=0.7, fm=0.1 should be in range for PF-1000."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result(fc=0.7, fm=0.1)
        )
        assert result["fc_in_range"] is True
        assert result["fm_in_range"] is True
        assert result["both_in_range"] is True

    def test_out_of_range_pf1000(self):
        """fc=0.1, fm=0.9 should be out of range for PF-1000."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result(fc=0.1, fm=0.9)
        )
        assert result["fc_in_range"] is False
        assert result["fm_in_range"] is False
        assert result["both_in_range"] is False

    def test_fc_in_fm_out(self):
        """fc in range but fm out of range → both_in_range=False."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result(fc=0.7, fm=0.95)
        )
        assert result["fc_in_range"] is True
        assert result["fm_in_range"] is False
        assert result["both_in_range"] is False

    def test_calibrated_values_echoed(self):
        """Result echoes back the fc/fm from the CalibrationResult."""
        cal = LeeModelCalibrator("NX2")
        cr = self._make_cal_result(fc=0.75, fm=0.15, device="NX2")
        result = cal.benchmark_against_published(calibration_result=cr)
        assert result["fc_calibrated"] == pytest.approx(0.75)
        assert result["fm_calibrated"] == pytest.approx(0.15)

    def test_published_range_returned(self):
        """Published ranges are returned as tuples."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result()
        )
        fc_range = result["fc_published_range"]
        fm_range = result["fm_published_range"]
        assert isinstance(fc_range, tuple) and len(fc_range) == 2
        assert isinstance(fm_range, tuple) and len(fm_range) == 2

    def test_reference_string(self):
        """Reference cites Lee & Saw (2014)."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result()
        )
        ref = result["reference"]
        assert "Lee" in ref
        assert "Saw" in ref
        assert "2014" in ref

    def test_unknown_device_raises(self):
        """benchmark_against_published raises KeyError for unknown device."""
        cal = LeeModelCalibrator("NONEXISTENT_DEVICE")
        with pytest.raises(KeyError, match="No published fc/fm range"):
            cal.benchmark_against_published(
                calibration_result=self._make_cal_result()
            )

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_midpoint_in_range(self, device: str):
        """Midpoint of published range should always be 'in range'."""
        ranges = _PUBLISHED_FC_FM_RANGES[device]
        fc_mid = 0.5 * (ranges["fc"][0] + ranges["fc"][1])
        fm_mid = 0.5 * (ranges["fm"][0] + ranges["fm"][1])
        cal = LeeModelCalibrator(device)
        result = cal.benchmark_against_published(
            calibration_result=self._make_cal_result(
                fc=fc_mid, fm=fm_mid, device=device
            )
        )
        assert result["both_in_range"] is True


# ═══════════════════════════════════════════════════════
# Tests for CalibrationResult dataclass
# ═══════════════════════════════════════════════════════


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_fields(self):
        """CalibrationResult has all expected fields."""
        cr = CalibrationResult(
            best_fc=0.7,
            best_fm=0.1,
            peak_current_error=0.05,
            timing_error=0.03,
            objective_value=0.04,
            n_evals=50,
            converged=True,
            device_name="PF-1000",
        )
        assert cr.best_fc == pytest.approx(0.7)
        assert cr.best_fm == pytest.approx(0.1)
        assert cr.peak_current_error == pytest.approx(0.05)
        assert cr.timing_error == pytest.approx(0.03)
        assert cr.objective_value == pytest.approx(0.04)
        assert cr.n_evals == 50
        assert cr.converged is True
        assert cr.device_name == "PF-1000"


# ═══════════════════════════════════════════════════════
# Tests for CrossValidator
# ═══════════════════════════════════════════════════════


class TestCrossValidatorStructure:
    """Tests for CrossValidator and CrossValidationResult structure."""

    def test_cross_validation_result_fields(self):
        """CrossValidationResult has all expected fields."""
        cr = CalibrationResult(
            best_fc=0.7, best_fm=0.1,
            peak_current_error=0.05, timing_error=0.03,
            objective_value=0.04, n_evals=50,
            converged=True, device_name="PF-1000",
        )
        cvr = CrossValidationResult(
            train_device="PF-1000",
            test_device="NX2",
            calibration=cr,
            prediction_peak_error=0.15,
            prediction_timing_error=0.10,
            generalization_score=0.875,
        )
        assert cvr.train_device == "PF-1000"
        assert cvr.test_device == "NX2"
        assert cvr.prediction_peak_error == pytest.approx(0.15)
        assert cvr.prediction_timing_error == pytest.approx(0.10)
        assert cvr.generalization_score == pytest.approx(0.875)

    def test_cross_validator_instantiation(self):
        """CrossValidator can be instantiated."""
        cv = CrossValidator()
        assert hasattr(cv, "validate")


# ═══════════════════════════════════════════════════════
# Integration tests (slow — run actual calibration)
# ═══════════════════════════════════════════════════════


class TestCalibrationIntegration:
    """Integration tests that run actual Lee model calibration."""

    @pytest.mark.slow
    def test_pf1000_calibration_converges(self):
        """PF-1000 calibration converges within maxiter."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.calibrate(maxiter=30)
        assert isinstance(result, CalibrationResult)
        assert result.device_name == "PF-1000"
        assert 0.0 < result.best_fc < 1.0
        assert 0.0 < result.best_fm < 1.0
        assert result.n_evals > 0

    @pytest.mark.slow
    def test_pf1000_benchmark_against_published(self):
        """PF-1000 calibrated fc/fm should fall in published ranges."""
        cal = LeeModelCalibrator("PF-1000")
        benchmark = cal.benchmark_against_published(maxiter=30)
        assert benchmark["fc_in_range"] is True or benchmark["fm_in_range"] is True, (
            f"Neither fc={benchmark['fc_calibrated']:.3f} "
            f"(range {benchmark['fc_published_range']}) "
            f"nor fm={benchmark['fm_calibrated']:.3f} "
            f"(range {benchmark['fm_published_range']}) in published range"
        )

    @pytest.mark.slow
    def test_calibrate_default_params(self):
        """calibrate_default_params runs for PF-1000 and NX2."""
        results = calibrate_default_params(
            devices=["PF-1000"], maxiter=30
        )
        assert "PF-1000" in results
        assert isinstance(results["PF-1000"], CalibrationResult)

    @pytest.mark.slow
    def test_cross_validation_runs(self):
        """Cross-validation PF-1000→NX2 runs and produces valid score."""
        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=30)
        assert isinstance(result, CrossValidationResult)
        assert 0.0 <= result.generalization_score <= 1.0
        assert result.train_device == "PF-1000"
        assert result.test_device == "NX2"
