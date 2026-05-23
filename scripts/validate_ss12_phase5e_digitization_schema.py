#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json"
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_digitization_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
ALLOWED_DIGITIZATION_STATUS = {"blocked_calibration_missing", "digitized_not_reviewed"}
ALLOWED_AXIS_CALIBRATION_STATUS = {"missing", "calibrated"}
ALLOWED_COMPLETION_STATUS = {"missing", "complete"}


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite_constant)


def validate_scaffold(scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    boundary = scaffold.get("acceptance_boundary", {})
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_not_object", "acceptance_boundary must be an object"))
        boundary = {}
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(_issue("top_level_acceptance_flag_not_false", "top-level acceptance flag must be false", flag=flag))

    packets = scaffold.get("digitization_packets")
    if not isinstance(packets, list):
        issues.append(_issue("digitization_packets_not_list", "digitization_packets must be a list"))
        return issues
    ids = [str(packet.get("id")) for packet in packets if isinstance(packet, dict)]
    if len(ids) != len(set(ids)):
        issues.append(_issue("duplicate_digitization_packet_id", "digitization packet ids must be unique"))

    for packet in packets:
        if not isinstance(packet, dict):
            issues.append(_issue("digitization_packet_not_object", "digitization packet must be an object"))
            continue
        _validate_packet(packet, issues)
    return issues


def _validate_packet(packet: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    packet_id = str(packet.get("id", "<missing>"))
    for flag in ACCEPTANCE_FLAGS:
        if packet.get(flag) is not False:
            issues.append(
                _issue(
                    "digitization_acceptance_flag_not_false",
                    "digitization packet acceptance/promotion flag must be false",
                    packet_id=packet_id,
                    flag=flag,
                )
            )

    digitized_series = packet.get("digitized_series")
    if not isinstance(digitized_series, list):
        issues.append(_issue("digitized_series_not_list", "digitized_series must be a list", packet_id=packet_id))
        digitized_series = []
    axis_status = packet.get("axis_calibration_status")
    axis_calibration = packet.get("axis_calibration")
    if digitized_series and axis_status != "calibrated":
        issues.append(
            _issue(
                "digitized_series_requires_axis_calibration",
                "digitized series requires calibrated axes",
                packet_id=packet_id,
            )
        )
    for point in digitized_series:
        if not _valid_digitized_point(point):
            issues.append(
                _issue(
                    "invalid_digitized_series_point",
                    "digitized series points must contain numeric x and y values",
                    packet_id=packet_id,
                    point=point,
                )
            )
            break
    if axis_status == "calibrated" and not _valid_axis_calibration(axis_calibration):
        issues.append(_issue("invalid_axis_calibration", "calibrated packets require two calibration points per axis", packet_id=packet_id))

    status = packet.get("digitization_status")
    if status not in ALLOWED_DIGITIZATION_STATUS:
        issues.append(_issue("invalid_digitization_status", "digitization_status is not in the allowed set", packet_id=packet_id, status=status))
    if axis_status not in ALLOWED_AXIS_CALIBRATION_STATUS:
        issues.append(_issue("invalid_axis_calibration_status", "axis_calibration_status is not in the allowed set", packet_id=packet_id, status=axis_status))
    if packet.get("uncertainty_budget_status") not in ALLOWED_COMPLETION_STATUS:
        issues.append(_issue("invalid_uncertainty_budget_status", "uncertainty_budget_status is not in the allowed set", packet_id=packet_id))
    if packet.get("review_certificate_status") not in ALLOWED_COMPLETION_STATUS:
        issues.append(_issue("invalid_review_certificate_status", "review_certificate_status is not in the allowed set", packet_id=packet_id))
    has_digitization_hash = isinstance(packet.get("digitization_hash"), str) and bool(HEX_SHA256.fullmatch(packet["digitization_hash"]))
    if packet.get("digitization_hash") is not None and not has_digitization_hash:
        issues.append(_issue("invalid_digitization_hash", "digitization_hash must be a 64-character hexadecimal sha256", packet_id=packet_id))
    if status != "blocked_calibration_missing":
        if not digitized_series or not has_digitization_hash or axis_status != "calibrated":
            issues.append(_issue("invalid_digitized_packet_state", "digitized status requires calibrated axes, series, and hash", packet_id=packet_id))
        if packet.get("uncertainty_budget_status") != "complete" or packet.get("review_certificate_status") != "complete":
            issues.append(
                _issue(
                    "digitized_packet_requires_uncertainty_and_review",
                    "digitized packets require uncertainty budget and review certificate before use",
                    packet_id=packet_id,
                )
            )
    else:
        if packet.get("digitization_hash") is not None:
            issues.append(_issue("blocked_packet_digitization_hash_forbidden", "blocked packets must not carry digitization hash", packet_id=packet_id))


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _valid_digitized_point(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) != {"x", "y"}:
        return False
    return _is_finite_number(value["x"]) and _is_finite_number(value["y"])


def _valid_axis_calibration(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for axis in ("x_axis", "y_axis"):
        axis_data = value.get(axis)
        if not isinstance(axis_data, dict):
            return False
        points = axis_data.get("points")
        if not isinstance(points, list) or len(points) < 2:
            return False
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                return False
            if not all(_is_finite_number(v) for v in point):
                return False
        if not isinstance(axis_data.get("unit"), str) or not axis_data["unit"]:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 5-E digitization scaffold")
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
