"""Figure-backed observable candidate staging for first-principles gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_FIGURE_CANDIDATE_FIELDS = (
    "validation_scope",
    "channel",
    "source_path",
    "line_start",
    "line_end",
    "figure_id",
    "extraction_method",
    "digitization_hash",
    "uncertainty",
    "reviewer",
    "review_state",
    "scope_classification",
)

_ACCEPTED_SCOPE_CLASSIFICATION = "same_source_accepted"


def stage_figure_observable_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Stage a figure-derived observable without promoting acceptance.

    Figure digitization is high-risk evidence: OCR/plot extraction can be
    wrong, transfer-scope plots can be over-applied, and missing uncertainty can
    masquerade as precision. This helper makes that boundary explicit.
    """

    missing = [field for field in REQUIRED_FIGURE_CANDIDATE_FIELDS if field not in candidate]
    review_certificate = candidate.get("review_certificate")
    certificate_ok = (
        isinstance(review_certificate, Mapping)
        and review_certificate.get("accepted") is True
        and bool(review_certificate.get("reviewer"))
        and bool(review_certificate.get("certificate_hash"))
    )
    scope_classification = str(candidate.get("scope_classification", ""))

    blocking_reasons: list[str] = []
    if missing:
        blocking_reasons.append("figure_candidate_required_metadata_missing")
    if not certificate_ok:
        blocking_reasons.append("review_certificate_missing")
    if scope_classification != _ACCEPTED_SCOPE_CLASSIFICATION:
        blocking_reasons.append("scope_not_same_source_accepted")
    if candidate.get("review_state") != "reviewed":
        blocking_reasons.append("figure_candidate_not_reviewed")
    if not _uncertainty_declared(candidate.get("uncertainty")):
        blocking_reasons.append("uncertainty_missing_or_invalid")

    blocking_reasons = sorted(set(blocking_reasons))
    return {
        "status": (
            "blocked_figure_candidate_incomplete"
            if missing
            else "staged_figure_candidate_not_accepted"
        ),
        "validation_scope": candidate.get("validation_scope"),
        "channel": candidate.get("channel"),
        "source_path": candidate.get("source_path"),
        "line_start": candidate.get("line_start"),
        "line_end": candidate.get("line_end"),
        "figure_id": candidate.get("figure_id"),
        "extraction_method": candidate.get("extraction_method"),
        "digitization_hash": candidate.get("digitization_hash"),
        "uncertainty": candidate.get("uncertainty"),
        "reviewer": candidate.get("reviewer"),
        "review_state": candidate.get("review_state"),
        "scope_classification": scope_classification,
        "missing_required_fields": missing,
        "blocking_reasons": blocking_reasons,
        "accepted_observable_claim": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "acceptance_rule": (
            "Figure-backed observables require complete digitization metadata, "
            "declared uncertainty, reviewed same-source acceptance, and an "
            "accepted independent review certificate before any downstream "
            "acceptance claim can be considered."
        ),
    }


def _uncertainty_declared(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    return value is not None
