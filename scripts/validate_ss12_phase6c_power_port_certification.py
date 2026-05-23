#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6C_POWER_PORT_CERTIFICATION_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE4B_ARTIFACT_REFS: tuple[str, ...] = (
    "src/dpf/first_principles/circuit_power_port.py",
    "tests/test_first_principles_circuit_power_port_phase4b.py",
    "docs/SS12_P1_PHASE4B_EVALUATE_LEARN_CONTINUE_2026_05_22.md",
)
CANONICAL_PHASE6B_SCAFFOLD_REF = "docs/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json"
CANONICAL_PHASE6B_SCAFFOLD = ROOT / CANONICAL_PHASE6B_SCAFFOLD_REF
REQUIRED_CERTIFICATION_ROW_IDS: tuple[str, ...] = (
    "cert_crowbar_timing",
    "cert_current_sheath_acceleration",
    "cert_pinch_focus_dynamics",
)
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_power_port_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
ALLOWED_CERTIFICATION_STATUS = {
    "blocked_certification_incomplete",
    "complete_not_accepted",
}
ALLOWED_POWER_PORT_STATUS = {
    "blocked_power_port_evidence_incomplete",
    "complete_not_accepted",
}
ALLOWED_UQ_STATUS = {
    "blocked_uq_budget_incomplete",
    "complete_not_accepted",
}
ALLOWED_REVIEW_STATUS = {"missing", "complete"}


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> Any:
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
            issues.append(
                _issue(
                    "top_level_acceptance_flag_not_false",
                    "top-level acceptance flag must be false",
                    flag=flag,
                )
            )

    _validate_phase4b_artifacts(scaffold, issues)
    phase6b_blocked = _validate_phase6b_linkage(scaffold, issues)

    rows = scaffold.get("power_port_certification_rows")
    if not isinstance(rows, list):
        issues.append(
            _issue("power_port_certification_rows_not_list", "power_port_certification_rows must be a list")
        )
        return issues

    row_ids = [str(row.get("id")) for row in rows if isinstance(row, dict)]
    if len(row_ids) != len(set(row_ids)):
        issues.append(_issue("duplicate_certification_row_id", "certification row ids must be unique"))
    if set(row_ids) != set(REQUIRED_CERTIFICATION_ROW_IDS):
        issues.append(
            _issue(
                "required_certification_rows_missing",
                "certification rows must exactly cover required Phase 6-C dynamics",
                expected=sorted(REQUIRED_CERTIFICATION_ROW_IDS),
                actual=sorted(row_ids),
            )
        )

    for row in rows:
        if not isinstance(row, dict):
            issues.append(_issue("certification_row_not_object", "certification row must be an object"))
            continue
        _validate_row(row, phase6b_blocked, issues)
    return issues


def _validate_phase4b_artifacts(scaffold: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    refs = scaffold.get("phase4b_circuit_power_port_artifacts")
    if refs != list(CANONICAL_PHASE4B_ARTIFACT_REFS):
        issues.append(
            _issue(
                "phase4b_artifacts_not_canonical",
                "Phase 6-C must pin exact canonical Phase 4-B power-port artifact references",
                expected=list(CANONICAL_PHASE4B_ARTIFACT_REFS),
                actual=refs,
            )
        )
        return
    for ref in CANONICAL_PHASE4B_ARTIFACT_REFS:
        if not (ROOT / ref).exists():
            issues.append(
                _issue(
                    "phase4b_artifact_missing",
                    "Canonical Phase 4-B artifact does not exist",
                    artifact=ref,
                )
            )


def _validate_phase6b_linkage(scaffold: dict[str, Any], issues: list[dict[str, Any]]) -> bool:
    phase6b_ref = str(scaffold.get("phase6b_uq_propagation_scaffold", ""))
    if phase6b_ref != CANONICAL_PHASE6B_SCAFFOLD_REF:
        issues.append(
            _issue(
                "phase6b_uq_propagation_scaffold_not_canonical",
                "Phase 6-C must link to the exact canonical Phase 6-B UQ propagation scaffold",
                expected=CANONICAL_PHASE6B_SCAFFOLD_REF,
                actual=phase6b_ref,
            )
        )
        return True
    try:
        phase6b = _load_json(CANONICAL_PHASE6B_SCAFFOLD)
    except (OSError, ValueError) as exc:
        issues.append(
            _issue(
                "phase6b_uq_propagation_scaffold_unreadable",
                "Phase 6-B UQ propagation scaffold cannot be loaded",
                error=str(exc),
            )
        )
        return True
    packets = phase6b.get("propagation_packets")
    if not isinstance(packets, list):
        issues.append(
            _issue("phase6b_propagation_packets_not_list", "Phase 6-B propagation packets must be a list")
        )
        return True
    return any(
        isinstance(packet, dict)
        and packet.get("propagation_status") == "blocked_uq_budget_incomplete"
        for packet in packets
    )


def _validate_row(
    row: dict[str, Any],
    phase6b_blocked: bool,
    issues: list[dict[str, Any]],
) -> None:
    row_id = str(row.get("id", "<missing>"))
    for flag in ACCEPTANCE_FLAGS:
        if row.get(flag) is not False:
            issues.append(
                _issue(
                    "certification_acceptance_flag_not_false",
                    "certification acceptance/promotion flags must be false",
                    row_id=row_id,
                    flag=flag,
                )
            )

    certification_status = row.get("certification_status")
    power_port_status = row.get("power_port_evidence_status")
    uq_status = row.get("uq_propagation_status")
    review_status = row.get("review_certificate_status")
    if certification_status not in ALLOWED_CERTIFICATION_STATUS:
        issues.append(
            _issue(
                "invalid_certification_status",
                "certification_status is not in the allowed set",
                row_id=row_id,
                status=certification_status,
            )
        )
    if power_port_status not in ALLOWED_POWER_PORT_STATUS:
        issues.append(
            _issue(
                "invalid_power_port_evidence_status",
                "power_port_evidence_status is not in the allowed set",
                row_id=row_id,
                status=power_port_status,
            )
        )
    if uq_status not in ALLOWED_UQ_STATUS:
        issues.append(
            _issue(
                "invalid_uq_propagation_status",
                "uq_propagation_status is not in the allowed set",
                row_id=row_id,
                status=uq_status,
            )
        )
    if review_status not in ALLOWED_REVIEW_STATUS:
        issues.append(
            _issue(
                "invalid_review_certificate_status",
                "review certificate status is not in the allowed set",
                row_id=row_id,
                status=review_status,
            )
        )

    observable = row.get("certified_observable")
    uncertainty = row.get("certified_uncertainty")
    observable_ok = _is_finite_number(observable)
    uncertainty_ok = (
        _is_finite_number(uncertainty)
        and isinstance(uncertainty, int | float)
        and not isinstance(uncertainty, bool)
        and uncertainty >= 0.0
    )
    is_blocked = certification_status == "blocked_certification_incomplete"
    if is_blocked and (observable is not None or uncertainty is not None):
        issues.append(
            _issue(
                "blocked_certification_values_forbidden",
                "blocked certification rows must not carry certified values",
                row_id=row_id,
            )
        )
    if certification_status == "complete_not_accepted":
        if power_port_status != "complete_not_accepted":
            issues.append(
                _issue(
                    "complete_certification_requires_power_port_evidence",
                    "complete certification requires complete non-accepted power-port evidence",
                    row_id=row_id,
                )
            )
        if uq_status != "complete_not_accepted":
            issues.append(
                _issue(
                    "complete_certification_requires_uq_propagation",
                    "complete certification requires complete non-accepted UQ propagation",
                    row_id=row_id,
                )
            )
        if not observable_ok or not uncertainty_ok:
            issues.append(
                _issue(
                    "complete_certification_requires_finite_values",
                    "complete certification requires finite observable and nonnegative uncertainty",
                    row_id=row_id,
                )
            )
        if review_status != "complete":
            issues.append(
                _issue(
                    "complete_certification_requires_review_certificate",
                    "complete certification requires a complete review certificate",
                    row_id=row_id,
                )
            )
    if phase6b_blocked and uq_status != "blocked_uq_budget_incomplete":
        issues.append(
            _issue(
                "phase6b_blocked_requires_row_uq_blocked",
                "Phase 6-B blocked propagation requires Phase 6-C certification rows to remain UQ-blocked",
                row_id=row_id,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 6-C power-port certification scaffold")
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
            "issues": [
                {
                    "rule": "scaffold_not_object",
                    "message": "Phase 6-C scaffold JSON must be an object",
                }
            ],
            "scaffold": str(scaffold_path),
        }
        print(json.dumps(report, indent=2))
        return 1
    issues = validate_scaffold(scaffold)
    report = {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "scaffold": str(scaffold_path),
    }
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
