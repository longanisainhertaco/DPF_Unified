"""Calibration-result provenance labels.

These labels keep fitted parameters separate from experimental validation
evidence. A good optimizer result can tune a model for one comparison target,
but it is not itself a source-backed validation packet.
"""

from __future__ import annotations

from typing import Any

CALIBRATION_PROVENANCE_CLASS = "optimized_parameter_fit"
CALIBRATION_VALIDATION_STATUS = "not_validation_evidence"
CALIBRATION_RESULT_LABEL = "Calibration Fit"


def calibration_provenance_metadata(
    *,
    device_name: str = "",
    preset: str = "",
    optimizer: str = "",
    fitted_parameters: list[str] | tuple[str, ...] = ("fc", "fm"),
    reference_source: str = "",
) -> dict[str, Any]:
    """Return fail-closed provenance metadata for calibration outputs."""

    return {
        "provenance_class": CALIBRATION_PROVENANCE_CLASS,
        "validation_status": CALIBRATION_VALIDATION_STATUS,
        "result_label": CALIBRATION_RESULT_LABEL,
        "can_support_validation_claims": False,
        "artifact_role": "parameter_fit_not_validation",
        "device_name": device_name,
        "preset": preset,
        "optimizer": optimizer,
        "fitted_parameters": list(fitted_parameters),
        "reference_source": reference_source,
        "source_authority_note": (
            "Optimized calibration parameters are not experimental validation "
            "evidence. Reference validation requires accepted local "
            "KnowledgeReference evidence and same-scope validation packets."
        ),
    }
