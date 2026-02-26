"""Phase Z — Calibration Benchmark Tests.

Tests for LeeModelCalibrator.benchmark_against_published() and the
_PUBLISHED_FC_FM_RANGES registry in dpf.validation.calibration, covering:

1. _PUBLISHED_FC_FM_RANGES contains PF-1000, NX2, UNU-ICTP.
2. All range tuples are (lo, hi) with lo < hi and both in [0, 1].
3. benchmark_against_published() returns dict with all required keys.
4. benchmark_against_published() raises KeyError for unknown device.
5. Pre-computed calibration result is accepted without re-running optimizer.
6. fc_in_range is True when calibrated fc is inside published range.
7. fc_in_range is False when calibrated fc is outside published range.
8. fm_in_range is True when calibrated fm is inside published range.
9. fm_in_range is False when calibrated fm is outside published range.
10. both_in_range is True only when both fc and fm are in range.
11. reference string contains 'Lee' and a year.
"""

from __future__ import annotations

import re

import pytest

from dpf.validation.calibration import (
    _PUBLISHED_FC_FM_RANGES,
    CalibrationResult,
    LeeModelCalibrator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cal_result(
    fc: float = 0.72,
    fm: float = 0.10,
    device_name: str = "PF-1000",
) -> CalibrationResult:
    """Return a CalibrationResult with specified fc/fm."""
    return CalibrationResult(
        best_fc=fc,
        best_fm=fm,
        peak_current_error=0.05,
        timing_error=0.08,
        objective_value=0.06,
        n_evals=42,
        converged=True,
        device_name=device_name,
    )


# ===========================================================================
# TestPublishedRanges
# ===========================================================================


class TestPublishedRanges:
    """Tests for the _PUBLISHED_FC_FM_RANGES registry."""

    def test_pf1000_present(self) -> None:
        """PF-1000 must be in the registry."""
        assert "PF-1000" in _PUBLISHED_FC_FM_RANGES

    def test_nx2_present(self) -> None:
        """NX2 must be in the registry."""
        assert "NX2" in _PUBLISHED_FC_FM_RANGES

    def test_unu_ictp_present(self) -> None:
        """UNU-ICTP must be in the registry."""
        assert "UNU-ICTP" in _PUBLISHED_FC_FM_RANGES

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_fc_range_valid(self, device: str) -> None:
        """fc range must be (lo, hi) with 0 < lo < hi < 1."""
        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES[device]["fc"]
        assert 0.0 < fc_lo < fc_hi < 1.0, (
            f"Device {device}: fc range ({fc_lo}, {fc_hi}) invalid"
        )

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_fm_range_valid(self, device: str) -> None:
        """fm range must be (lo, hi) with 0 < lo < hi <= 1."""
        fm_lo, fm_hi = _PUBLISHED_FC_FM_RANGES[device]["fm"]
        assert 0.0 < fm_lo < fm_hi <= 1.0, (
            f"Device {device}: fm range ({fm_lo}, {fm_hi}) invalid"
        )

    def test_pf1000_fc_range_includes_0_7(self) -> None:
        """Published PF-1000 fc range must include 0.7 (Lee & Saw 2014)."""
        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fc"]
        assert fc_lo <= 0.70 <= fc_hi, (
            f"PF-1000 fc=0.70 (Lee & Saw 2014) not in range ({fc_lo}, {fc_hi})"
        )

    def test_pf1000_fm_range_includes_0_1(self) -> None:
        """Published PF-1000 fm range must include 0.10 (Lee & Saw 2014)."""
        fm_lo, fm_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fm"]
        assert fm_lo <= 0.10 <= fm_hi, (
            f"PF-1000 fm=0.10 (Lee & Saw 2014) not in range ({fm_lo}, {fm_hi})"
        )


# ===========================================================================
# TestBenchmarkAgainstPublished
# ===========================================================================


class TestBenchmarkAgainstPublished:
    """Tests for LeeModelCalibrator.benchmark_against_published()."""

    def test_raises_keyerror_for_unknown_device(self) -> None:
        """benchmark_against_published raises KeyError for unregistered device."""
        cal = LeeModelCalibrator("MY_FICTIONAL_DEVICE")
        with pytest.raises(KeyError):
            cal.benchmark_against_published(calibration_result=_make_cal_result())

    def test_returns_dict_with_required_keys(self) -> None:
        """Result contains all required keys."""
        cal = LeeModelCalibrator("PF-1000")
        result = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.72, fm=0.10, device_name="PF-1000"),
        )
        required = {
            "fc_calibrated",
            "fm_calibrated",
            "fc_published_range",
            "fm_published_range",
            "fc_in_range",
            "fm_in_range",
            "both_in_range",
            "reference",
        }
        assert required.issubset(result.keys())

    def test_fc_calibrated_matches_input(self) -> None:
        """fc_calibrated in result matches the CalibrationResult's best_fc."""
        cal = LeeModelCalibrator("PF-1000")
        cal_result = _make_cal_result(fc=0.73, fm=0.12)
        bench = cal.benchmark_against_published(calibration_result=cal_result)
        assert bench["fc_calibrated"] == pytest.approx(0.73)

    def test_fm_calibrated_matches_input(self) -> None:
        """fm_calibrated in result matches the CalibrationResult's best_fm."""
        cal = LeeModelCalibrator("PF-1000")
        cal_result = _make_cal_result(fc=0.73, fm=0.12)
        bench = cal.benchmark_against_published(calibration_result=cal_result)
        assert bench["fm_calibrated"] == pytest.approx(0.12)

    def test_fc_in_range_true_for_typical_pf1000(self) -> None:
        """fc=0.70 is within PF-1000 published range → fc_in_range=True."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.70, fm=0.10),
        )
        assert bench["fc_in_range"] is True

    def test_fc_in_range_false_for_low_fc(self) -> None:
        """fc=0.30 is below PF-1000 published range → fc_in_range=False."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.30, fm=0.10),
        )
        assert bench["fc_in_range"] is False

    def test_fm_in_range_true_for_typical_pf1000(self) -> None:
        """fm=0.10 is within PF-1000 published range → fm_in_range=True."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.70, fm=0.10),
        )
        assert bench["fm_in_range"] is True

    def test_fm_in_range_false_for_high_fm(self) -> None:
        """fm=0.90 is above PF-1000 published range → fm_in_range=False."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.70, fm=0.90),
        )
        assert bench["fm_in_range"] is False

    def test_both_in_range_true_when_both_valid(self) -> None:
        """both_in_range=True when fc and fm both within published ranges."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.70, fm=0.10),
        )
        assert bench["both_in_range"] is True

    def test_both_in_range_false_when_fc_out(self) -> None:
        """both_in_range=False when only fc is out of range."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.30, fm=0.10),
        )
        assert bench["both_in_range"] is False

    def test_both_in_range_false_when_fm_out(self) -> None:
        """both_in_range=False when only fm is out of range."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(fc=0.70, fm=0.90),
        )
        assert bench["both_in_range"] is False

    def test_reference_contains_lee(self) -> None:
        """reference string contains 'Lee' (proper attribution)."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(),
        )
        assert "Lee" in str(bench["reference"])

    def test_reference_contains_year(self) -> None:
        """reference string contains a 4-digit year."""
        cal = LeeModelCalibrator("PF-1000")
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(),
        )
        assert re.search(r"\d{4}", str(bench["reference"])) is not None, (
            f"No year found in reference: {bench['reference']}"
        )

    def test_published_range_values_match_registry(self) -> None:
        """fc_published_range and fm_published_range match _PUBLISHED_FC_FM_RANGES."""
        device = "NX2"
        cal = LeeModelCalibrator(device)
        bench = cal.benchmark_against_published(
            calibration_result=_make_cal_result(device_name=device),
        )
        expected_fc = _PUBLISHED_FC_FM_RANGES[device]["fc"]
        expected_fm = _PUBLISHED_FC_FM_RANGES[device]["fm"]
        assert bench["fc_published_range"] == expected_fc
        assert bench["fm_published_range"] == expected_fm
