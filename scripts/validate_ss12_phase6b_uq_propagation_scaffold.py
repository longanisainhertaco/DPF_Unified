#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE6A_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE6A_SCAFFOLD_REF = "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_propagation_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
ALLOWED_PROPAGATION_STATUS = {"blocked_uq_budget_incomplete", "complete_not_accepted"}
ALLOWED_REVIEW_STATUS = {"missing", "complete"}


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite_constant)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def validate_scaffold(scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    boundary = scaffold.get("acceptance_boundary", {})
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_not_object", "acceptance_boundary must be an object"))
        boundary = {}
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(_issue("top_level_acceptance_flag_not_false", "top-level acceptance flag must be false", flag=flag))
    packets = scaffold.get("propagation_packets")
    if not isinstance(packets, list):
        issues.append(_issue("propagation_packets_not_list", "propagation_packets must be a list"))
        return issues
    ids = [str(packet.get("id")) for packet in packets if isinstance(packet, dict)]
    if len(ids) != len(set(ids)):
        issues.append(_issue("duplicate_propagation_packet_id", "propagation packet ids must be unique"))
    uq_ids = [str(packet.get("uq_budget_row_id")) for packet in packets if isinstance(packet, dict)]
    if len(uq_ids) != len(set(uq_ids)):
        issues.append(_issue("duplicate_uq_budget_row_id", "UQ row ids must be unique in propagation packets"))
    for packet in packets:
        if not isinstance(packet, dict):
            issues.append(_issue("propagation_packet_not_object", "propagation packet must be an object"))
            continue
        _validate_packet(packet, issues)
    _validate_phase6a_linkage(scaffold, packets, issues)
    return issues


def _validate_phase6a_linkage(scaffold: dict[str, Any], packets: list[Any], issues: list[dict[str, Any]]) -> None:
    phase6a_ref = str(scaffold.get("phase6a_uq_budget_scaffold", ""))
    if phase6a_ref != CANONICAL_PHASE6A_SCAFFOLD_REF:
        issues.append(
            _issue(
                "phase6a_uq_budget_scaffold_not_canonical",
                "Phase 6-B propagation scaffold must link to the exact canonical Phase 6-A UQ scaffold reference",
                expected=CANONICAL_PHASE6A_SCAFFOLD_REF,
                actual=phase6a_ref,
            )
        )
        return
    try:
        phase6a = _load_json(CANONICAL_PHASE6A_SCAFFOLD)
    except (OSError, ValueError) as exc:
        issues.append(_issue("phase6a_uq_budget_scaffold_unreadable", "Phase 6-A UQ scaffold cannot be loaded", error=str(exc)))
        return
    rows = phase6a.get("uq_budget_rows")
    if not isinstance(rows, list):
        issues.append(_issue("phase6a_uq_budget_rows_not_list", "Phase 6-A UQ rows must be a list"))
        return
    rows_by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    packet_uq_ids = [str(packet.get("uq_budget_row_id")) for packet in packets if isinstance(packet, dict)]
    if set(packet_uq_ids) != set(rows_by_id):
        issues.append(
            _issue(
                "propagation_packets_do_not_match_phase6a_uq_rows",
                "Propagation packets must map one-to-one to Phase 6-A UQ rows",
                expected=sorted(rows_by_id),
                actual=sorted(packet_uq_ids),
            )
        )
        return
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        row = rows_by_id[str(packet.get("uq_budget_row_id"))]
        if packet.get("figure_source_id") != row.get("figure_source_id") or packet.get("digitization_packet_id") != row.get("digitization_packet_id"):
            issues.append(
                _issue(
                    "propagation_packet_uq_linkage_mismatch",
                    "Propagation packet figure/digitization linkage must match referenced UQ row",
                    packet_id=packet.get("id"),
                    uq_budget_row_id=packet.get("uq_budget_row_id"),
                )
            )


def _validate_packet(packet: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    packet_id = str(packet.get("id", "<missing>"))
    for flag in ACCEPTANCE_FLAGS:
        if packet.get(flag) is not False:
            issues.append(
                _issue("propagation_acceptance_flag_not_false", "propagation acceptance/promotion flags must be false", packet_id=packet_id, flag=flag)
            )
    status = packet.get("propagation_status")
    if status not in ALLOWED_PROPAGATION_STATUS:
        issues.append(_issue("invalid_propagation_status", "propagation_status is not in the allowed set", packet_id=packet_id, status=status))
    if packet.get("review_certificate_status") not in ALLOWED_REVIEW_STATUS:
        issues.append(_issue("invalid_review_certificate_status", "review certificate status is not in the allowed set", packet_id=packet_id))
    observable_ok = _is_finite_number(packet.get("propagated_observable"))
    uncertainty_value = packet.get("propagated_uncertainty")
    uncertainty_ok = (
        _is_finite_number(uncertainty_value) and uncertainty_value >= 0.0
        if isinstance(uncertainty_value, int | float) and not isinstance(uncertainty_value, bool)
        else False
    )
    if status == "blocked_uq_budget_incomplete":
        if packet.get("propagated_observable") is not None or packet.get("propagated_uncertainty") is not None:
            issues.append(_issue("blocked_propagation_values_forbidden", "blocked propagation packets must not carry propagated values", packet_id=packet_id))
    elif status == "complete_not_accepted":
        if not observable_ok or not uncertainty_ok:
            issues.append(_issue("complete_propagation_requires_finite_values", "complete propagation requires finite observable and nonnegative uncertainty", packet_id=packet_id))
        if packet.get("review_certificate_status") != "complete":
            issues.append(_issue("complete_propagation_requires_review_certificate", "complete propagation requires review certificate", packet_id=packet_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 6-B UQ propagation scaffold")
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
    if not isinstance(scaffold, dict):
        report = {
            "passed": False,
            "issue_count": 1,
            "issues": [{"rule": "scaffold_not_object", "message": "Phase 6-B scaffold JSON must be an object"}],
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
