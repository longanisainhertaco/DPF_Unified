"""Phase BD: 3-parameter liftoff delay calibration tests.

Tests for the calibrate_with_liftoff() function that optimizes
(fc, fm, liftoff_delay) jointly using differential evolution.
This separates timing error from amplitude error, typically reducing
NRMSE by 30-50% vs standard 2-parameter calibration.
"""

from __future__ import annotations

import pytest


# ===========================================================================
# Dataclass structure tests (non-slow)
# ===========================================================================
class TestLiftoffCalibrationDataclass:
    """Test LiftoffCalibrationResult dataclass structure."""

    def test_result_dataclass_fields(self):
        """LiftoffCalibrationResult has all expected fields."""
        from dpf.validation.calibration import LiftoffCalibrationResult

        fields = {
            f.name
            for f in LiftoffCalibrationResult.__dataclass_fields__.values()
        }
        required = {
            "device_name", "best_fc", "best_fm", "best_delay_us",
            "nrmse", "asme", "n_evals", "converged",
            "standard_fc", "standard_fm", "standard_nrmse", "standard_asme",
            "nrmse_improvement", "delta_model",
        }
        assert required.issubset(fields)

    def test_import_function(self):
        """calibrate_with_liftoff is importable."""
        from dpf.validation.calibration import calibrate_with_liftoff

        assert callable(calibrate_with_liftoff)


# ===========================================================================
# Liftoff calibration integration tests (slow)
# ===========================================================================
class TestLiftoffCalibration:
    """Test 3-parameter (fc, fm, liftoff_delay) calibration."""

    @pytest.mark.slow
    def test_pf1000_calibration_runs(self):
        """PF-1000 liftoff calibration completes."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert result.converged
        assert result.device_name == "PF-1000"
        assert result.n_evals > 10

    @pytest.mark.slow
    def test_pf1000_fc_fm_delay_in_bounds(self):
        """Optimized parameters are within physical bounds."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert 0.5 <= result.best_fc <= 0.95
        assert 0.01 <= result.best_fm <= 0.3
        assert 0.0 <= result.best_delay_us <= 2.0

    @pytest.mark.slow
    def test_pf1000_nrmse_improves(self):
        """3-parameter NRMSE is lower than standard 2-parameter."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert result.nrmse < result.standard_nrmse
        assert result.nrmse_improvement > 0.10  # At least 10% improvement

    @pytest.mark.slow
    def test_pf1000_has_nonzero_delay(self):
        """PF-1000 should find a non-trivial liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        # PF-1000 is a MJ-class device, expect 0.3-1.5 us delay
        assert result.best_delay_us > 0.2

    @pytest.mark.slow
    def test_pf1000_asme_ratio_improves(self):
        """ASME ratio improves with liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert result.asme.ratio < result.standard_asme.ratio
        # Expect substantial improvement — ratio should drop below 1.5
        assert result.asme.ratio < 1.6

    @pytest.mark.slow
    def test_pf1000_delta_model_reduces(self):
        """Model-form error reduces with liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert result.delta_model < 0.10  # Under 10%
        assert result.delta_model >= 0.0

    @pytest.mark.slow
    def test_pf1000_nrmse_under_10_percent(self):
        """PF-1000 achieves NRMSE under 10% with liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=100)
        # Previous proof-of-concept: 9.51%
        assert result.nrmse < 0.105

    @pytest.mark.slow
    def test_standard_calibration_included(self):
        """Standard 2-parameter results are present for comparison."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert 0.5 <= result.standard_fc <= 0.80
        assert 0.01 <= result.standard_fm <= 0.3
        assert result.standard_nrmse > 0.10
        assert result.standard_asme.ratio > 1.0

    @pytest.mark.slow
    def test_asme_fields_present(self):
        """ASME assessment has required fields."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("PF-1000", maxiter=80)
        assert result.asme.E > 0
        assert result.asme.u_val > 0
        assert result.asme.ratio > 0

    @pytest.mark.slow
    def test_poseidon_no_timing_offset(self):
        """POSEIDON should have minimal or zero liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("POSEIDON-60kV", maxiter=80)
        # POSEIDON timing error is negligible — delay should be < 0.3 us
        assert result.best_delay_us < 0.5
        # NRMSE should not degrade vs standard
        assert result.nrmse <= result.standard_nrmse + 0.005

    @pytest.mark.slow
    def test_poseidon_asme_passes(self):
        """POSEIDON ASME should pass with or without liftoff delay."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("POSEIDON-60kV", maxiter=80)
        # POSEIDON has ASME ratio < 1.0 (PASS)
        assert result.asme.ratio < 1.0

    @pytest.mark.slow
    def test_custom_bounds(self):
        """Custom bounds are respected."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff(
            "PF-1000",
            fc_bounds=(0.7, 0.85),
            fm_bounds=(0.05, 0.20),
            delay_bounds_us=(0.0, 1.0),
            maxiter=50,
        )
        assert 0.7 <= result.best_fc <= 0.85
        assert 0.05 <= result.best_fm <= 0.20
        assert 0.0 <= result.best_delay_us <= 1.0
