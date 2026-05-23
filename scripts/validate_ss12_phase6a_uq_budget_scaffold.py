#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE5E_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE5E_SCAFFOLD_REF = "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json"
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_uq_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
UNCERTAINTY_TERMS: tuple[str, ...] = (
    "source_uncertainty",
    "digitization_uncertainty",
    "calibration_uncertainty",
    "numerical_uncertainty",
    "model_inadequacy_uncertainty",
)
ALLOWED_UQ_STATUS = {"blocked_digitization_not_reviewed", "complete_not_accepted"}
ALLOWED_REVIEW_STATUS = {"missing", "complete"}


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite_constant)


def _is_finite_nonnegative_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and value >= 0.0


def validate_scaffold(scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    boundary = scaffold.get("acceptance_boundary", {})
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_not_object", "acceptance_boundary must be an object"))
        boundary = {}
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(_issue("top_level_acceptance_flag_not_false", "top-level acceptance flag must be false", flag=flag))

    rows = scaffold.get("uq_budget_rows")
    if not isinstance(rows, list):
        issues.append(_issue("uq_budget_rows_not_list", "uq_budget_rows must be a list"))
        return issues
    ids = [str(row.get("id")) for row in rows if isinstance(row, dict)]
    if len(ids) != len(set(ids)):
        issues.append(_issue("duplicate_uq_budget_row_id", "UQ budget row ids must be unique"))
    packet_ids = [str(row.get("digitization_packet_id")) for row in rows if isinstance(row, dict)]
    if len(packet_ids) != len(set(packet_ids)):
        issues.append(_issue("duplicate_digitization_packet_id", "digitization packet ids must be unique in UQ rows"))
    for row in rows:
        if not isinstance(row, dict):
            issues.append(_issue("uq_budget_row_not_object", "UQ budget row must be an object"))
            continue
        _validate_row(row, issues)
    _validate_phase5e_linkage(scaffold, rows, issues)
    return issues


def _validate_phase5e_linkage(scaffold: dict[str, Any], rows: list[Any], issues: list[dict[str, Any]]) -> None:
    phase5e_ref = str(scaffold.get("phase5e_digitization_scaffold", ""))
    if phase5e_ref != CANONICAL_PHASE5E_SCAFFOLD_REF:
        issues.append(
            _issue(
                "phase5e_digitization_scaffold_not_canonical",
                "Phase 6-A UQ scaffold must link to the exact canonical Phase 5-E digitization scaffold reference",
                expected=CANONICAL_PHASE5E_SCAFFOLD_REF,
                actual=phase5e_ref,
            )
        )
        return
    phase5e_path = CANONICAL_PHASE5E_SCAFFOLD.resolve()
    try:
        phase5e = _load_json(phase5e_path)
    except (OSError, ValueError) as exc:
        issues.append(_issue("phase5e_digitization_scaffold_unreadable", "Phase 5-E digitization scaffold cannot be loaded", error=str(exc)))
        return
    packets = phase5e.get("digitization_packets")
    if not isinstance(packets, list):
        issues.append(_issue("phase5e_digitization_packets_not_list", "Phase 5-E digitization packets must be a list"))
        return
    phase5e_by_id = {str(packet.get("id")): packet for packet in packets if isinstance(packet, dict)}
    row_packet_ids = [str(row.get("digitization_packet_id")) for row in rows if isinstance(row, dict)]
    if set(row_packet_ids) != set(phase5e_by_id):
        issues.append(
            _issue(
                "uq_rows_do_not_match_phase5e_digitization_packets",
                "UQ rows must map one-to-one to Phase 5-E digitization packets",
                expected=sorted(phase5e_by_id),
                actual=sorted(row_packet_ids),
            )
        )
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        packet = phase5e_by_id[str(row.get("digitization_packet_id"))]
        if row.get("figure_source_id") != packet.get("figure_source_id") or row.get("crop_artifact_id") != packet.get("crop_artifact_id"):
            issues.append(
                _issue(
                    "uq_row_digitization_linkage_mismatch",
                    "UQ row figure/crop linkage must match referenced digitization packet",
                    row_id=row.get("id"),
                    digitization_packet_id=row.get("digitization_packet_id"),
                )
            )


def _validate_row(row: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    row_id = str(row.get("id", "<missing>"))
    for flag in ACCEPTANCE_FLAGS:
        if row.get(flag) is not False:
            issues.append(
                _issue("uq_acceptance_flag_not_false", "UQ acceptance/promotion flags must be false", row_id=row_id, flag=flag)
            )
    status = row.get("uq_status")
    if status not in ALLOWED_UQ_STATUS:
        issues.append(_issue("invalid_uq_status", "uq_status is not in the allowed set", row_id=row_id, status=status))
    if row.get("review_certificate_status") not in ALLOWED_REVIEW_STATUS:
        issues.append(_issue("invalid_review_certificate_status", "review certificate status is not in the allowed set", row_id=row_id))

    term_values = [row.get(term) for term in UNCERTAINTY_TERMS]
    terms_complete = all(_is_finite_nonnegative_number(value) for value in term_values)
    combined_complete = _is_finite_nonnegative_number(row.get("combined_uncertainty"))
    if status == "blocked_digitization_not_reviewed":
        for term in (*UNCERTAINTY_TERMS, "combined_uncertainty"):
            if row.get(term) is not None:
                issues.append(_issue("blocked_uq_row_uncertainty_forbidden", "blocked UQ rows must not carry uncertainty values", row_id=row_id, term=term))
    elif status == "complete_not_accepted":
        if not terms_complete or not combined_complete:
            issues.append(
                _issue(
                    "complete_uq_requires_all_uncertainty_terms",
                    "complete UQ rows require all uncertainty terms and combined uncertainty",
                    row_id=row_id,
                )
            )
        if row.get("review_certificate_status") != "complete":
            issues.append(_issue("complete_uq_requires_review_certificate", "complete UQ rows require review certificate", row_id=row_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 6-A UQ budget scaffold")
    parser.add_argument("scaffold", nargs="?", default=str(DEFAULT_SCAFFOLD))
    args = parser.parse_args()
    scaffold_path = Path(args.scaffold)
    try:
        scaffold = _load_json(scaffold_path)
    except ValueError as exc:
        report = {
            "passed": False,
            "issue_count": 1,
            "issues": [{"rule": "invalid_json_nonfinite_or_malformed", "message": str(exc)}],
            "scaffold": str(scaffold_path),
        }
        print(json.dumps(report, indent=2))
        return 1
    issues = validate_scaffold(scaffold)
    report = {"passed": not issues, "issue_count": len(issues), "issues": issues, "scaffold": str(scaffold_path)}
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
