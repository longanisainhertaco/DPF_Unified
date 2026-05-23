#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE7A_REVIEW_CERTIFICATE_SKELETON_2026_05_22.json"
CANONICAL_PHASE6C_REF = "docs/SS12_P1_PHASE6C_POWER_PORT_CERTIFICATION_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE6C = ROOT / CANONICAL_PHASE6C_REF
REQUIRED_OBSERVABLE_IDS: tuple[str, ...] = (
    "crowbar_timing",
    "current_sheath_acceleration",
    "pinch_focus_dynamics",
    "magnetic_field_history",
    "temperature_distribution_history",
    "neutron_yield_timing_spectrum_anisotropy_detector_response",
)
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_review_certificate",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
TOP_LEVEL_ACCEPTANCE_FLAGS: tuple[str, ...] = (*ACCEPTANCE_FLAGS, "emits_accepted_certificate")
REQUIRED_MAPPING_FIELDS: tuple[str, ...] = (
    "observable_id",
    "runtime_output_field",
    "source_evidence_field",
    "comparison_field",
)
REQUIRED_UNCERTAINTY_TERMS: tuple[str, ...] = ("measurement", "model", "numerical")
REQUIRED_HASH_FIELDS: tuple[str, ...] = (
    "runtime_run_hash",
    "source_evidence_hash",
    "uq_packet_hash",
)
REQUIRED_REVIEW_FIELDS: tuple[str, ...] = (
    "reviewer_id",
    "reviewer_affiliation",
    "reviewed_at",
    "review_packet_hash",
    "review_status",
    "blocking_findings",
)
ALLOWED_CERTIFICATE_STATUS = {
    "blocked_review_certificate_incomplete",
    "complete_not_accepted",
}


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> Any:
    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite_constant)


def validate_scaffold(scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    _validate_phase6c_linkage(scaffold, issues)
    _validate_acceptance_boundary(scaffold, issues)

    rows = scaffold.get("review_certificate_rows")
    if not isinstance(rows, list):
        issues.append(_issue("review_certificate_rows_not_list", "review_certificate_rows must be a list"))
        return issues

    observable_ids = [str(row.get("observable_id")) for row in rows if isinstance(row, dict)]
    if len(observable_ids) != len(set(observable_ids)):
        issues.append(_issue("duplicate_observable_id", "review certificate observable ids must be unique"))
    if set(observable_ids) != set(REQUIRED_OBSERVABLE_IDS):
        issues.append(
            _issue(
                "required_observables_missing",
                "Phase 7-A certificate rows must exactly cover required observables",
                expected=sorted(REQUIRED_OBSERVABLE_IDS),
                actual=sorted(observable_ids),
            )
        )

    for row in rows:
        if not isinstance(row, dict):
            issues.append(_issue("review_certificate_row_not_object", "review certificate row must be an object"))
            continue
        _validate_row(row, issues)
    return issues


def _validate_phase6c_linkage(scaffold: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    phase6c_ref = str(scaffold.get("phase6c_power_port_certification_scaffold", ""))
    if phase6c_ref != CANONICAL_PHASE6C_REF:
        issues.append(
            _issue(
                "phase6c_scaffold_not_canonical",
                "Phase 7-A must pin the exact canonical Phase 6-C scaffold reference",
                expected=CANONICAL_PHASE6C_REF,
                actual=phase6c_ref,
            )
        )
        return
    try:
        phase6c = _load_json(CANONICAL_PHASE6C)
    except (OSError, ValueError) as exc:
        issues.append(
            _issue(
                "phase6c_scaffold_unreadable",
                "Phase 6-C scaffold cannot be loaded",
                error=str(exc),
            )
        )
        return
    boundary = phase6c.get("acceptance_boundary")
    if not isinstance(boundary, dict):
        issues.append(_issue("phase6c_acceptance_boundary_not_object", "Phase 6-C boundary must be an object"))
        return
    if any(boundary.get(flag) is not False for flag in ("promotes_acceptance", "can_support_first_principles_acceptance")):
        issues.append(_issue("phase6c_upstream_gate_promoted", "Phase 7-A refuses promoted upstream Phase 6-C gates"))


def _validate_acceptance_boundary(scaffold: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    boundary = scaffold.get("acceptance_boundary")
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_not_object", "acceptance_boundary must be an object"))
        return
    for flag in TOP_LEVEL_ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(
                _issue(
                    "top_level_acceptance_flag_not_false",
                    "top-level acceptance/certificate emission flags must be false in Phase 7-A",
                    flag=flag,
                )
            )


def _validate_row(row: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    observable_id = str(row.get("observable_id", "<missing>"))
    for flag in ACCEPTANCE_FLAGS:
        if row.get(flag) is not False:
            issues.append(
                _issue(
                    "row_acceptance_flag_not_false",
                    "row acceptance/promotion flags must be false in Phase 7-A",
                    observable_id=observable_id,
                    flag=flag,
                )
            )

    if row.get("certificate_status") not in ALLOWED_CERTIFICATE_STATUS:
        issues.append(
            _issue(
                "invalid_certificate_status",
                "certificate_status is not in the allowed set",
                observable_id=observable_id,
                status=row.get("certificate_status"),
            )
        )
    if row.get("certificate_status") == "complete_not_accepted":
        issues.append(
            _issue(
                "complete_certificate_blocked_by_phase7a",
                "Phase 7-A may define certificate slots only; it must not complete a certificate",
                observable_id=observable_id,
            )
        )

    _validate_mapping(row, observable_id, issues)
    placeholders_complete = _validate_uncertainty_placeholders(row, observable_id, issues)
    placeholders_complete = _validate_pass_fail_metrics(row, observable_id, issues) and placeholders_complete
    placeholders_complete = _validate_negative_controls(row, observable_id, issues) and placeholders_complete
    placeholders_complete = _validate_hashes(row, observable_id, issues) and placeholders_complete
    placeholders_complete = _validate_review_placeholders(row, observable_id, issues) and placeholders_complete

    if row.get("certificate_status") == "complete_not_accepted" and not placeholders_complete:
        issues.append(
            _issue(
                "complete_certificate_requires_no_placeholders",
                "complete certificate rows require all placeholder slots to be resolved first",
                observable_id=observable_id,
            )
        )


def _validate_mapping(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    mapping = row.get("output_field_mapping")
    if not isinstance(mapping, dict):
        issues.append(_issue("output_field_mapping_incomplete", "output_field_mapping must be an object", observable_id=observable_id))
        return False
    missing_or_empty = [field for field in REQUIRED_MAPPING_FIELDS if not mapping.get(field)]
    if mapping.get("observable_id") != observable_id:
        missing_or_empty.append("observable_id_matches_row")
    if missing_or_empty:
        issues.append(
            _issue(
                "output_field_mapping_incomplete",
                "output_field_mapping must name observable/runtime/source/comparison fields",
                observable_id=observable_id,
                missing=missing_or_empty,
            )
        )
        return False
    return True


def _validate_uncertainty_placeholders(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    placeholders = row.get("uncertainty_placeholders")
    if not isinstance(placeholders, dict) or set(placeholders) != set(REQUIRED_UNCERTAINTY_TERMS):
        issues.append(
            _issue(
                "uncertainty_placeholders_incomplete",
                "measurement/model/numerical uncertainty placeholders are required",
                observable_id=observable_id,
            )
        )
        return False
    complete = True
    for term_name in REQUIRED_UNCERTAINTY_TERMS:
        term = placeholders.get(term_name)
        if not isinstance(term, dict) or term.get("term") != term_name:
            complete = False
        elif term.get("status") == "placeholder_incomplete":
            complete = False
            if term.get("value") is not None or term.get("unit") is not None or term.get("evidence_hash") is not None:
                issues.append(
                    _issue(
                        "incomplete_uncertainty_placeholder_carries_values",
                        "incomplete uncertainty placeholders must not carry partial values",
                        observable_id=observable_id,
                        term=term_name,
                    )
                )
        elif term.get("status") != "complete_not_accepted":
            issues.append(
                _issue(
                    "invalid_uncertainty_placeholder_status",
                    "uncertainty placeholder status is not allowed",
                    observable_id=observable_id,
                    term=term_name,
                    status=term.get("status"),
                )
            )
            complete = False
    return complete


def _validate_pass_fail_metrics(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    metrics = row.get("pass_fail_metrics")
    if not isinstance(metrics, dict):
        issues.append(_issue("pass_fail_metrics_incomplete", "pass_fail_metrics must be an object", observable_id=observable_id))
        return False
    tolerance = metrics.get("tolerance")
    if not metrics.get("metric_id") or not isinstance(tolerance, dict):
        issues.append(_issue("pass_fail_metrics_incomplete", "metric id and tolerance slot are required", observable_id=observable_id))
        return False
    if metrics.get("status") == "placeholder_incomplete":
        if metrics.get("result") is not None or tolerance.get("value") is not None or tolerance.get("unit") is not None:
            issues.append(_issue("incomplete_metric_carries_values", "incomplete pass/fail metrics must not carry values", observable_id=observable_id))
        return False
    return metrics.get("status") == "complete_not_accepted"


def _validate_negative_controls(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    controls = row.get("negative_controls")
    if not isinstance(controls, list) or not controls:
        issues.append(_issue("negative_controls_missing", "at least one negative-control slot is required", observable_id=observable_id))
        return False
    complete = True
    for control in controls:
        if not isinstance(control, dict) or not control.get("control_id") or not control.get("expected_result"):
            issues.append(_issue("negative_control_incomplete", "negative controls require id and expected result", observable_id=observable_id))
            complete = False
            continue
        if control.get("status") == "placeholder_incomplete":
            complete = False
        elif control.get("status") != "complete_not_accepted":
            issues.append(_issue("invalid_negative_control_status", "negative-control status is not allowed", observable_id=observable_id, status=control.get("status")))
            complete = False
    return complete


def _validate_hashes(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    hashes = row.get("run_evidence_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_HASH_FIELDS):
        issues.append(_issue("run_evidence_hashes_incomplete", "runtime/source/UQ hash slots are required", observable_id=observable_id))
        return False
    if all(isinstance(hashes[field], str) and len(hashes[field]) == 64 for field in REQUIRED_HASH_FIELDS):
        return True
    if any(hashes[field] is not None for field in REQUIRED_HASH_FIELDS):
        issues.append(_issue("partial_run_evidence_hashes_forbidden", "hash slots must remain empty until all hashes exist", observable_id=observable_id))
    return False


def _validate_review_placeholders(row: dict[str, Any], observable_id: str, issues: list[dict[str, Any]]) -> bool:
    review = row.get("independent_review_placeholders")
    if not isinstance(review, dict) or set(review) != set(REQUIRED_REVIEW_FIELDS):
        issues.append(
            _issue(
                "independent_review_placeholders_incomplete",
                "independent review placeholder fields are required",
                observable_id=observable_id,
            )
        )
        return False
    if review.get("review_status") == "missing":
        return False
    if review.get("review_status") != "complete_not_accepted":
        issues.append(_issue("invalid_review_status", "review_status is not allowed", observable_id=observable_id, status=review.get("review_status")))
        return False
    return all(review.get(field) for field in REQUIRED_REVIEW_FIELDS if field != "blocking_findings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 7-A review-certificate skeleton")
    parser.add_argument("scaffold", nargs="?", default=str(DEFAULT_SCAFFOLD))
    args = parser.parse_args()
    scaffold_path = Path(args.scaffold)
    try:
        scaffold = _load_json(scaffold_path)
    except (OSError, ValueError) as exc:
        report = {
            "passed": False,
            "accepted_certificate_emitted": False,
            "issue_count": 1,
            "issues": [{"rule": "invalid_json_nonfinite_or_malformed", "message": str(exc)}],
            "scaffold": str(scaffold_path),
        }
        print(json.dumps(report, indent=2))
        return 1
    if not isinstance(scaffold, dict):
        report = {
            "passed": False,
            "accepted_certificate_emitted": False,
            "issue_count": 1,
            "issues": [{"rule": "scaffold_not_object", "message": "Phase 7-A scaffold JSON must be an object"}],
            "scaffold": str(scaffold_path),
        }
        print(json.dumps(report, indent=2))
        return 1
    issues = validate_scaffold(scaffold)
    report = {
        "passed": not issues,
        "accepted_certificate_emitted": False,
        "issue_count": len(issues),
        "issues": issues,
        "scaffold": str(scaffold_path),
    }
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
