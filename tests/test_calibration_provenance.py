"""Calibration provenance guardrails."""

from __future__ import annotations

from types import SimpleNamespace


def test_calibration_provenance_metadata_is_fail_closed() -> None:
    from dpf.validation import calibration_provenance_metadata

    metadata = calibration_provenance_metadata(
        device_name="PF-1000",
        preset="pf1000",
        optimizer="unit-test optimizer",
    )

    assert metadata["provenance_class"] == "optimized_parameter_fit"
    assert metadata["validation_status"] == "not_validation_evidence"
    assert metadata["result_label"] == "Calibration Fit"
    assert metadata["can_support_validation_claims"] is False
    assert "not experimental validation evidence" in metadata["source_authority_note"]


def test_auto_calibrate_output_carries_provenance(monkeypatch) -> None:
    import dpf.validation.calibration as calibration_module

    class FakeCalibrator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def calibrate(self, **kwargs):
            return SimpleNamespace(
                best_fc=0.7,
                best_fm=0.13,
                peak_current_error=0.04,
                timing_error=0.08,
                n_evals=3,
                converged=True,
            )

        def benchmark_against_published(self, result):
            return {
                "fc_published_range": (0.6, 0.8),
                "fm_published_range": (0.05, 0.2),
                "fc_in_range": True,
                "fm_in_range": True,
                "reference": "unit-test reference",
            }

    monkeypatch.setattr(calibration_module, "LeeModelCalibrator", FakeCalibrator)

    from app_calibrate import auto_calibrate

    result = auto_calibrate("pf1000")

    assert result["provenance_class"] == "optimized_parameter_fit"
    assert result["validation_status"] == "not_validation_evidence"
    assert result["result_label"] == "Calibration Fit"
    assert result["can_support_validation_claims"] is False
    assert result["calibration_provenance"]["preset"] == "pf1000"
    assert result["calibration_provenance"]["optimizer"] == "LeeModelCalibrator/Nelder-Mead"


def test_format_calibration_markdown_declares_non_validation_authority() -> None:
    from app_calibrate import format_calibration_markdown

    markdown = format_calibration_markdown({
        "best_fc": 0.7,
        "best_fm": 0.13,
        "I_peak_error": 0.04,
        "t_peak_error": 0.08,
        "n_evals": 3,
    })

    assert "optimized calibration fits are not validation evidence" in markdown
    assert "Reference validation requires accepted local" in markdown
