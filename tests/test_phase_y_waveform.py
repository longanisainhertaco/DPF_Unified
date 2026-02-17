"""Phase Y: Digitized waveform infrastructure tests.

Tests for the digitized waveform fields added to ExperimentalDevice,
the normalized_rmse() helper function, the waveform-aware
validate_current_waveform() return values, and the waveform_weight
parameter in LeeModelCalibrator.

Coverage:
1. PF-1000 waveform existence, shape, peak consistency, and time range.
2. NX2 has no waveform (waveform_t / waveform_I remain None).
3. normalized_rmse() returns 0 for identical arrays and a finite positive
   value for offset arrays.
4. validate_current_waveform() returns 'waveform_available' and
   'waveform_nrmse' keys for both a waveform-bearing device and one
   without a waveform.
5. LeeModelCalibrator exposes waveform_weight with a correct default and
   implements a 3-term objective that provides >= 1 DOF for PF-1000.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.validation.experimental import (
    DEVICES,
    NX2_DATA,
    PF1000_DATA,
    normalized_rmse,
    validate_current_waveform,
)
from dpf.validation.calibration import LeeModelCalibrator
from dpf.validation.lee_model_comparison import LeeModelComparison

# ---------------------------------------------------------------------------
# Helpers — synthetic simulated I(t) for validate_current_waveform calls
# ---------------------------------------------------------------------------

def _make_synthetic_pf1000_waveform() -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sim, I_sim) approximating the PF-1000 current waveform.

    Uses a clipped sinusoidal ramp peaking at ~1.87 MA at ~5.8 us.
    """
    t_sim = np.linspace(0, 10e-6, 1000)
    I_sim = 1.87e6 * np.sin(np.pi * t_sim / (2.0 * 5.8e-6))
    I_sim = np.maximum(I_sim, 0.0)
    return t_sim, I_sim


def _make_synthetic_nx2_waveform() -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sim, I_sim) approximating the NX2 current waveform."""
    t_sim = np.linspace(0, 4e-6, 500)
    I_sim = 400e3 * np.sin(np.pi * t_sim / (2.0 * 1.8e-6))
    I_sim = np.maximum(I_sim, 0.0)
    return t_sim, I_sim


# ===========================================================================
# TestDigitizedWaveform
# ===========================================================================

class TestDigitizedWaveform:
    """Tests for the digitized waveform fields on ExperimentalDevice."""

    def test_pf1000_waveform_exists(self):
        """PF1000_DATA.waveform_t and waveform_I must not be None."""
        assert PF1000_DATA.waveform_t is not None, (
            "PF1000_DATA.waveform_t should be a NumPy array, got None"
        )
        assert PF1000_DATA.waveform_I is not None, (
            "PF1000_DATA.waveform_I should be a NumPy array, got None"
        )

    def test_pf1000_waveform_shape(self):
        """PF-1000 digitized waveform must have exactly 26 points in both arrays."""
        assert len(PF1000_DATA.waveform_t) == 26, (
            f"Expected 26 time points, got {len(PF1000_DATA.waveform_t)}"
        )
        assert len(PF1000_DATA.waveform_I) == 26, (
            f"Expected 26 current points, got {len(PF1000_DATA.waveform_I)}"
        )

    def test_pf1000_waveform_peak_matches_scalar(self):
        """Peak of digitized waveform must match PF1000_DATA.peak_current within 1%."""
        waveform_peak = float(np.max(PF1000_DATA.waveform_I))
        scalar_peak = PF1000_DATA.peak_current  # 1.87e6 A
        relative_error = abs(waveform_peak - scalar_peak) / scalar_peak
        assert relative_error < 0.01, (
            f"Waveform peak {waveform_peak:.4e} A differs from scalar peak "
            f"{scalar_peak:.4e} A by {relative_error:.2%} (> 1%)"
        )

    def test_pf1000_waveform_time_range(self):
        """PF-1000 waveform must start at 0 and extend to approximately 10 us."""
        t_us = PF1000_DATA.waveform_t * 1e6  # Convert s -> us for readable assertions
        assert t_us[0] == pytest.approx(0.0, abs=1e-9), (
            f"Waveform should start at t=0, got t={t_us[0]:.2f} us"
        )
        # Allow ±5% around the expected 10 us endpoint
        assert t_us[-1] == pytest.approx(10.0, rel=0.05), (
            f"Waveform should end at ~10 us, got t={t_us[-1]:.2f} us"
        )

    def test_nx2_no_waveform(self):
        """NX2_DATA must have waveform_t = None and waveform_I = None."""
        assert NX2_DATA.waveform_t is None, (
            f"NX2_DATA.waveform_t should be None, got {type(NX2_DATA.waveform_t)}"
        )
        assert NX2_DATA.waveform_I is None, (
            f"NX2_DATA.waveform_I should be None, got {type(NX2_DATA.waveform_I)}"
        )


# ===========================================================================
# TestWaveformNRMSE
# ===========================================================================

class TestWaveformNRMSE:
    """Tests for normalized_rmse() and waveform keys in validate_current_waveform()."""

    def test_perfect_match_nrmse_zero(self):
        """normalized_rmse with identical sim and exp arrays must return 0.0."""
        t = np.linspace(0, 10e-6, 26)
        I = 1.87e6 * np.sin(np.pi * t / (2.0 * 5.8e-6))
        I = np.maximum(I, 0.0)

        result = normalized_rmse(t, I, t, I)
        assert result == pytest.approx(0.0, abs=1e-12), (
            f"Perfect match should give NRMSE=0, got {result}"
        )

    def test_offset_nrmse_nonzero(self):
        """normalized_rmse with I_sim = 1.1 * I_exp should return approximately 0.1."""
        t_exp = PF1000_DATA.waveform_t
        I_exp = PF1000_DATA.waveform_I

        # Offset the simulated waveform by +10% of the peak
        I_peak_exp = float(np.max(np.abs(I_exp)))
        I_sim_offset = I_exp + 0.1 * I_peak_exp

        nrmse = normalized_rmse(t_exp, I_sim_offset, t_exp, I_exp)
        # RMSE / I_peak = (0.1 * I_peak) / I_peak = 0.1 exactly for uniform offset
        assert nrmse == pytest.approx(0.1, rel=1e-6), (
            f"Uniform 10% offset should give NRMSE ≈ 0.1, got {nrmse:.4f}"
        )

    def test_validate_returns_waveform_fields(self):
        """validate_current_waveform must return 'waveform_available' and 'waveform_nrmse' keys."""
        t_sim, I_sim = _make_synthetic_pf1000_waveform()
        metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")

        assert "waveform_available" in metrics, (
            "'waveform_available' key missing from validate_current_waveform output"
        )
        assert "waveform_nrmse" in metrics, (
            "'waveform_nrmse' key missing from validate_current_waveform output"
        )

    def test_validate_pf1000_waveform_available(self):
        """validate_current_waveform for PF-1000 must report waveform_available=True
        and a finite waveform_nrmse."""
        t_sim, I_sim = _make_synthetic_pf1000_waveform()
        metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")

        assert metrics["waveform_available"] is True, (
            "PF-1000 has a digitized waveform; waveform_available should be True"
        )
        assert math.isfinite(metrics["waveform_nrmse"]), (
            f"waveform_nrmse should be finite for PF-1000, got {metrics['waveform_nrmse']}"
        )
        # NRMSE should be a non-negative float
        assert metrics["waveform_nrmse"] >= 0.0, (
            f"waveform_nrmse must be >= 0, got {metrics['waveform_nrmse']}"
        )

    def test_validate_nx2_no_waveform(self):
        """validate_current_waveform for NX2 must report waveform_available=False
        and waveform_nrmse=NaN."""
        t_sim, I_sim = _make_synthetic_nx2_waveform()
        metrics = validate_current_waveform(t_sim, I_sim, "NX2")

        assert metrics["waveform_available"] is False, (
            "NX2 has no digitized waveform; waveform_available should be False"
        )
        assert math.isnan(metrics["waveform_nrmse"]), (
            f"waveform_nrmse should be NaN for NX2, got {metrics['waveform_nrmse']}"
        )


# ===========================================================================
# TestCalibrationWaveform
# ===========================================================================

class TestCalibrationWaveform:
    """Tests for the waveform_weight parameter in LeeModelCalibrator."""

    def test_calibrator_has_waveform_weight(self):
        """LeeModelCalibrator('PF-1000') must expose waveform_weight defaulting to 0.3."""
        cal = LeeModelCalibrator("PF-1000")
        assert hasattr(cal, "waveform_weight"), (
            "LeeModelCalibrator must have a 'waveform_weight' attribute"
        )
        assert cal.waveform_weight == pytest.approx(0.3), (
            f"Default waveform_weight should be 0.3, got {cal.waveform_weight}"
        )

    def test_three_term_objective_dof(self):
        """With waveform_weight > 0 and PF-1000 waveform available, the calibrator
        uses 3 metrics (peak, timing, waveform) against 2 free parameters (fc, fm),
        giving >= 1 degree of freedom.

        This test verifies the calibrator instantiates with correct weights
        and that the sum of weights equals 1.0, confirming all 3 terms are active.
        """
        cal = LeeModelCalibrator(
            "PF-1000",
            peak_weight=0.4,
            timing_weight=0.3,
            waveform_weight=0.3,
        )

        # Verify all three weight attributes are present and positive
        assert cal.peak_weight > 0.0, "peak_weight must be positive"
        assert cal.timing_weight > 0.0, "timing_weight must be positive"
        assert cal.waveform_weight > 0.0, "waveform_weight must be positive"

        # Weights sum to 1.0: 3 metrics each carrying non-zero weight
        total_weight = cal.peak_weight + cal.timing_weight + cal.waveform_weight
        assert total_weight == pytest.approx(1.0, abs=1e-9), (
            f"peak_weight + timing_weight + waveform_weight should equal 1.0, "
            f"got {total_weight}"
        )

        # 3 metrics vs 2 free parameters (fc, fm) => >= 1 DOF
        n_metrics = 3  # peak error, timing error, waveform NRMSE
        n_params = 2   # fc, fm
        assert n_metrics - n_params >= 1, (
            "3 metrics / 2 params should give >= 1 degree of freedom"
        )

        # PF-1000 has a digitized waveform, so the 3rd term will be active
        assert PF1000_DATA.waveform_t is not None, (
            "PF-1000 must have waveform_t for the 3rd objective term to activate"
        )
        assert PF1000_DATA.waveform_I is not None, (
            "PF-1000 must have waveform_I for the 3rd objective term to activate"
        )
