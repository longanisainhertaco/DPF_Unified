"""Cross-packet fail-closed acceptance shield for first-principles DPF gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_ACCEPTED_STATUSES = {"accepted", "passed", "ready"}
_REVIEW_CERTIFICATE_ACCEPTANCE_FLAGS = (
    "accepted_review_certificate",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
_REVIEW_CERTIFICATE_TOP_LEVEL_ACCEPTANCE_FLAGS = (
    *_REVIEW_CERTIFICATE_ACCEPTANCE_FLAGS,
    "emits_accepted_certificate",
)
_REQUIRED_REVIEW_CERTIFICATE_UNCERTAINTY_TERMS = {"measurement", "model", "numerical"}
_REQUIRED_REVIEW_CERTIFICATE_HASH_FIELDS = {
    "runtime_run_hash",
    "source_evidence_hash",
    "uq_packet_hash",
}
_REQUIRED_REVIEW_CERTIFICATE_REVIEW_FIELDS = {
    "reviewer_id",
    "reviewer_affiliation",
    "reviewed_at",
    "review_packet_hash",
    "review_status",
    "blocking_findings",
}


def build_first_principles_acceptance_shield(
    *,
    source_packet: Mapping[str, Any] | None = None,
    numerical_packet: Mapping[str, Any] | None = None,
    power_port_packet: Mapping[str, Any] | None = None,
    figure_packets: Sequence[Mapping[str, Any]] | None = None,
    uncertainty_packet: Mapping[str, Any] | None = None,
    review_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a final fail-closed acceptance shield over upstream packets.

    The shield does not promote acceptance. It exists to detect missing packets,
    staged-only/transfer-only evidence, and suspicious lower-layer acceptance
    flags that appear without an accepted packet status.
    """

    figures = list(figure_packets or ())
    blocking: list[str] = []
    anomalies: list[str] = []

    _packet_gate(
        "source_packet",
        source_packet,
        claim_keys=("accepted_source_claim",),
        blocking=blocking,
        anomalies=anomalies,
    )
    _packet_gate(
        "numerical_packet",
        numerical_packet,
        claim_keys=("can_support_numerical_acceptance",),
        blocking=blocking,
        anomalies=anomalies,
    )
    _packet_gate(
        "power_port_packet",
        power_port_packet,
        claim_keys=("accepted_power_port_claim",),
        blocking=blocking,
        anomalies=anomalies,
    )
    _packet_gate(
        "uncertainty_packet",
        uncertainty_packet,
        claim_keys=("accepted_uncertainty_claim", "can_support_uq_acceptance"),
        blocking=blocking,
        anomalies=anomalies,
    )

    if not figures:
        blocking.append("figure_packet_missing")
    for packet in figures:
        if packet.get("accepted_observable_claim") is not True:
            blocking.append("figure_packet_not_accepted")
        else:
            anomalies.append("figure_packet_claims_acceptance_in_phase4")
            blocking.append("phase4_no_packet_may_claim_acceptance")
        if packet.get("can_support_first_principles_acceptance") is True:
            anomalies.append("figure_packet_claims_first_principles_support")
            blocking.append("phase4_no_packet_may_claim_acceptance")

    review_status = _review_certificate_gate(
        review_certificate,
        blocking=blocking,
        anomalies=anomalies,
    )

    blocking = sorted(set(blocking))
    anomalies = sorted(set(anomalies))
    return {
        "status": "blocked_first_principles_acceptance",
        "accepted_first_principles_claim": False,
        "promotes_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "blocking_reasons": blocking,
        "claim_anomalies": anomalies,
        "packet_status": {
            "source_packet": _status(source_packet),
            "numerical_packet": _status(numerical_packet),
            "power_port_packet": _status(power_port_packet),
            "uncertainty_packet": _status(uncertainty_packet),
            "figure_packet_count": len(figures),
            **review_status,
        },
        "acceptance_rule": (
            "First-principles acceptance requires accepted same-source source "
            "packet, numerical-fidelity packet, circuit power-port packet, "
            "figure/observable packets, uncertainty budget, and independent "
            "review certificate. Transfer-only or staged-only evidence is "
            "insufficient."
        ),
    }


def _packet_gate(
    name: str,
    packet: Mapping[str, Any] | None,
    *,
    claim_keys: Sequence[str],
    blocking: list[str],
    anomalies: list[str],
) -> None:
    if packet is None:
        blocking.append(f"{name}_missing")
        return

    accepted_status = _is_accepted_status(packet.get("status"))
    if not accepted_status:
        blocking.append(f"{name}_not_accepted")

    for key in claim_keys:
        if packet.get(key) is True:
            anomalies.append(f"{name}_claims_acceptance_in_phase4")
            blocking.append("phase4_no_packet_may_claim_acceptance")
            if not accepted_status:
                anomalies.append(f"{name}_claims_acceptance_without_accepted_status")

    if packet.get("can_support_first_principles_acceptance") is True:
        anomalies.append(f"{name}_claims_first_principles_support_in_phase4")
        blocking.append("phase4_no_packet_may_claim_acceptance")
        if not accepted_status:
            anomalies.append(f"{name}_claims_first_principles_support_without_accepted_status")


def _review_certificate_gate(
    review_certificate: Mapping[str, Any] | None,
    *,
    blocking: list[str],
    anomalies: list[str],
) -> dict[str, Any]:
    """Fail closed around the Phase 7 review-certificate path.

    Phase 7-B is a hardening layer only. A certificate-shaped object may be
    complete enough for dry-run reporting, but this shield never treats an
    asserted ``accepted`` bit as acceptance authority. Final promotion requires a
    later explicit independent final-review surface, so forged or premature
    certificate flags are reported as anomalies while the returned acceptance
    status stays false.
    """

    if review_certificate is None:
        blocking.append("review_certificate_missing")
        return {"review_certificate_accepted": False, "review_certificate_row_count": 0}

    if review_certificate.get("accepted") is True:
        anomalies.append("review_certificate_claims_acceptance")
        blocking.append("review_certificate_not_final_review_accepted")
    else:
        blocking.append("review_certificate_not_accepted")

    boundary = review_certificate.get("acceptance_boundary")
    if isinstance(boundary, Mapping):
        for flag in _REVIEW_CERTIFICATE_TOP_LEVEL_ACCEPTANCE_FLAGS:
            if boundary.get(flag) is True:
                anomalies.append(f"review_certificate_boundary_claims_{flag}")
                blocking.append("review_certificate_acceptance_flags_not_false")
            elif boundary.get(flag) is not False:
                blocking.append("review_certificate_acceptance_flags_not_false")
    else:
        blocking.append("review_certificate_acceptance_boundary_missing")

    if review_certificate.get("phase6c_power_port_certification_scaffold"):
        blocking.append("review_certificate_upstream_phase6c_not_accepted")

    rows = review_certificate.get("review_certificate_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        blocking.append("review_certificate_rows_missing")
        return {"review_certificate_accepted": False, "review_certificate_row_count": 0}

    for row in rows:
        if not isinstance(row, Mapping):
            blocking.append("review_certificate_row_invalid")
            continue
        _review_certificate_row_gate(row, blocking=blocking, anomalies=anomalies)

    return {"review_certificate_accepted": False, "review_certificate_row_count": len(rows)}


def _review_certificate_row_gate(
    row: Mapping[str, Any],
    *,
    blocking: list[str],
    anomalies: list[str],
) -> None:
    for flag in _REVIEW_CERTIFICATE_ACCEPTANCE_FLAGS:
        if row.get(flag) is True:
            anomalies.append(f"review_certificate_row_claims_{flag}")
            blocking.append("review_certificate_acceptance_flags_not_false")
        elif row.get(flag) is not False:
            blocking.append("review_certificate_acceptance_flags_not_false")

    if row.get("certificate_status") == "blocked_review_certificate_incomplete":
        blocking.append("review_certificate_rows_incomplete")
    elif row.get("certificate_status") != "complete_not_accepted":
        blocking.append("review_certificate_invalid_certificate_status")

    if not _review_certificate_uncertainty_complete(row):
        blocking.append("review_certificate_incomplete_uncertainty_placeholders")
    if not _review_certificate_metrics_complete(row):
        blocking.append("review_certificate_incomplete_pass_fail_metrics")
    if not _review_certificate_negative_controls_complete(row):
        blocking.append("review_certificate_incomplete_negative_controls")
    if not _review_certificate_hashes_complete(row):
        blocking.append("review_certificate_incomplete_run_evidence_hashes")
    if not _review_certificate_review_complete(row):
        blocking.append("review_certificate_independent_review_missing")


def _review_certificate_uncertainty_complete(row: Mapping[str, Any]) -> bool:
    placeholders = row.get("uncertainty_placeholders")
    if (
        not isinstance(placeholders, Mapping)
        or set(placeholders) != _REQUIRED_REVIEW_CERTIFICATE_UNCERTAINTY_TERMS
    ):
        return False
    for term_name in _REQUIRED_REVIEW_CERTIFICATE_UNCERTAINTY_TERMS:
        term = placeholders.get(term_name)
        if not isinstance(term, Mapping):
            return False
        if term.get("term") != term_name or term.get("status") != "complete_not_accepted":
            return False
        if not isinstance(term.get("value"), int | float) or not term.get("unit"):
            return False
        if not _is_sha256_hex(term.get("evidence_hash")):
            return False
    return True


def _review_certificate_metrics_complete(row: Mapping[str, Any]) -> bool:
    metrics = row.get("pass_fail_metrics")
    if not isinstance(metrics, Mapping) or metrics.get("status") != "complete_not_accepted":
        return False
    tolerance = metrics.get("tolerance")
    return (
        bool(metrics.get("metric_id"))
        and metrics.get("result") is not None
        and isinstance(tolerance, Mapping)
        and isinstance(tolerance.get("value"), int | float)
        and bool(tolerance.get("unit"))
    )


def _review_certificate_negative_controls_complete(row: Mapping[str, Any]) -> bool:
    controls = row.get("negative_controls")
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)) or not controls:
        return False
    for control in controls:
        if not isinstance(control, Mapping):
            return False
        if control.get("status") != "complete_not_accepted":
            return False
        if not control.get("control_id") or not control.get("expected_result"):
            return False
        if not _is_sha256_hex(control.get("evidence_hash")):
            return False
    return True


def _review_certificate_hashes_complete(row: Mapping[str, Any]) -> bool:
    hashes = row.get("run_evidence_hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != _REQUIRED_REVIEW_CERTIFICATE_HASH_FIELDS:
        return False
    return all(_is_sha256_hex(hashes.get(field)) for field in _REQUIRED_REVIEW_CERTIFICATE_HASH_FIELDS)


def _review_certificate_review_complete(row: Mapping[str, Any]) -> bool:
    review = row.get("independent_review_placeholders")
    if not isinstance(review, Mapping) or set(review) != _REQUIRED_REVIEW_CERTIFICATE_REVIEW_FIELDS:
        return False
    if review.get("review_status") != "complete_not_accepted":
        return False
    return all(
        review.get(field) for field in _REQUIRED_REVIEW_CERTIFICATE_REVIEW_FIELDS if field != "blocking_findings"
    )


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _status(packet: Mapping[str, Any] | None) -> str:
    if packet is None:
        return "missing"
    return str(packet.get("status", "missing_status"))


def _is_accepted_status(value: object) -> bool:
    text = str(value or "").strip().lower()
    return any(text == status or text.startswith(f"{status}_") for status in _ACCEPTED_STATUSES)
