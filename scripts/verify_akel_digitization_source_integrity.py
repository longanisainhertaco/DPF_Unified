"""Verify Akel 2021 source parity and Fig. 1 digitization integrity.

This script is a pre-review guardrail for the Akel PF-1000 Fig. 1 current
waveform digitization. It verifies that the local PDF/markdown/JSON source
pair, figure crop, SVG overlay source, and draft packet hashes still line up
with the packet being reviewed.

The default mode accepts the current `blocked_by_review` state as an expected
blocker if all non-review integrity checks pass. Use `--require-accepted-review`
when an accepted review packet is expected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_kr_pdf_parity import verify_pair  # noqa: E402
from dpf.validation.digitization import (  # noqa: E402
    akel_fig1_draft_digitization_packet,
    digitization_verification_evidence,
    scientific_closure_digitization_queue,
    scientific_closure_digitization_status,
    sha256_file,
)
from dpf.validation.kr_targets import (  # noqa: E402
    pf1000_16kv_current_waveform_digitization_candidate_evidence,
)


REVIEW_ONLY_FAILURES = {
    "independent_review_missing",
    "independent_review_metadata_missing",
    "review_packet_hash_missing",
    "review_packet_hash_mismatch",
    "review_source_hash_mismatch",
    "review_figure_image_hash_mismatch",
    "review_task_id_mismatch",
    "review_scope_mismatch",
    "reviewer_missing",
    "review_date_missing",
    "review_notes_missing",
    "review_decision_not_accepted",
    "review_status_not_accepted",
}


AKEL_FIG1_TASK_ID = "akel_2021_fig1_current_waveform_shot_12581"


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    details: Mapping[str, Any] | None = None,
    fatal: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "fatal": bool(fatal),
            "details": dict(details or {}),
        }
    )


def _rel(path: str | Path) -> str:
    return str(path).replace(str(REPO_ROOT) + "/", "")


def _resolve(base_path: Path, raw_path: object) -> Path:
    return base_path / str(raw_path)


def _series_by_name(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    series = packet.get("digitized_series", [])
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes, bytearray)):
        return {}
    return {
        str(item.get("name")): item
        for item in series
        if isinstance(item, Mapping) and item.get("name")
    }


def _series_len(series: Mapping[str, Any]) -> int:
    x = series.get("x", [])
    y = series.get("y", [])
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes, bytearray)):
        return 0
    if not isinstance(y, Sequence) or isinstance(y, (str, bytes, bytearray)):
        return 0
    return min(len(x), len(y))


def _find_akel_pdf(base_path: Path, queue: Mapping[str, Any]) -> dict[str, Any]:
    expected_sha = str(queue.get("source_pdf_sha256", ""))
    candidates = [str(item) for item in queue.get("source_pdf_candidates", [])]
    matches: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    for candidate in candidates:
        path = base_path / candidate
        if not path.exists():
            checked.append({"path": candidate, "exists": False})
            continue
        actual_sha = sha256_file(path)
        record = {
            "path": candidate,
            "exists": True,
            "sha256": actual_sha,
            "matches_expected_sha256": actual_sha == expected_sha,
        }
        checked.append(record)
        if actual_sha == expected_sha:
            matches.append(record)
    return {
        "expected_sha256": expected_sha,
        "candidates_checked": checked,
        "matched_candidates": matches,
        "selected_path": matches[0]["path"] if matches else "",
    }


def _source_item_present(md_path: Path, source_lines: str) -> dict[str, Any]:
    if not md_path.exists():
        return {"passed": False, "reason": "markdown_missing"}
    lines = md_path.read_text().splitlines()
    try:
        start_s, end_s = str(source_lines).split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except ValueError:
        return {"passed": False, "reason": "source_lines_unparseable"}
    if start < 1 or end < start or end > len(lines):
        return {"passed": False, "reason": "source_lines_out_of_range"}
    excerpt = "\n".join(lines[start - 1 : end])
    required_fragments = [
        "Fig. 1",
        "PF1000",
        "16 kV",
        "1.2 Torr",
        "shot 12581",
    ]
    missing = [
        fragment
        for fragment in required_fragments
        if fragment.lower() not in excerpt.lower()
    ]
    return {
        "passed": not missing,
        "source_lines": source_lines,
        "missing_fragments": missing,
        "excerpt": excerpt,
    }


def _packet_point_count_report(
    packet: Mapping[str, Any],
    fig1_task: Mapping[str, Any],
) -> dict[str, Any]:
    series = _series_by_name(packet)
    expected: dict[str, int] = {}
    extraction_candidate = fig1_task.get("series_extraction_candidate", {})
    if isinstance(extraction_candidate, Mapping):
        measured = extraction_candidate.get("measured_current_candidate", {})
        computed = extraction_candidate.get("computed_current_candidate", {})
        if isinstance(measured, Mapping):
            expected["measured_current"] = int(measured.get("point_count", 0) or 0)
        if isinstance(computed, Mapping):
            expected["computed_current"] = int(computed.get("point_count", 0) or 0)

    actual = {name: _series_len(item) for name, item in series.items()}
    mismatches = {
        name: {"expected": count, "actual": actual.get(name, 0)}
        for name, count in expected.items()
        if actual.get(name, 0) != count
    }
    return {
        "passed": not mismatches,
        "expected_counts": expected,
        "actual_counts": actual,
        "mismatches": mismatches,
    }


def build_akel_digitization_source_integrity_report(
    *,
    base_path: str | Path = ".",
    check_pdf_text_parity: bool = True,
    require_accepted_review: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed integrity report for the local Akel Fig. 1 packet."""
    base = Path(base_path)
    checks: list[dict[str, Any]] = []
    queue = scientific_closure_digitization_queue()
    fig1_task = next(
        item
        for item in queue["items"]
        if isinstance(item, Mapping) and item.get("task_id") == AKEL_FIG1_TASK_ID
    )

    packet = akel_fig1_draft_digitization_packet(base_path=base)
    evidence = digitization_verification_evidence(packet, base_path=base)
    status = scientific_closure_digitization_status([packet], base_path=base)
    readiness = pf1000_16kv_current_waveform_digitization_candidate_evidence(
        packet,
        base_path=base,
    )

    source_path = _resolve(base, packet.get("source_path", ""))
    figure_path = _resolve(base, packet.get("figure_image_path", ""))
    verification = packet.get("verification", {})
    if not isinstance(verification, Mapping):
        verification = {}
    svg_path = _resolve(base, verification.get("overlay_residual_source_svg_path", ""))
    json_path = source_path.with_suffix(".json")

    source_sha = sha256_file(source_path) if source_path.exists() else ""
    figure_sha = sha256_file(figure_path) if figure_path.exists() else ""
    svg_sha = sha256_file(svg_path) if svg_path.exists() else ""

    _check(
        checks,
        "source_markdown_hash",
        source_path.exists() and source_sha == packet.get("source_sha256"),
        details={
            "path": _rel(source_path),
            "actual_sha256": source_sha,
            "expected_sha256": packet.get("source_sha256", ""),
        },
    )
    _check(
        checks,
        "figure_image_hash",
        figure_path.exists() and figure_sha == packet.get("figure_image_sha256"),
        details={
            "path": _rel(figure_path),
            "actual_sha256": figure_sha,
            "expected_sha256": packet.get("figure_image_sha256", ""),
        },
    )
    _check(
        checks,
        "overlay_source_svg_hash",
        svg_path.exists()
        and svg_sha == verification.get("overlay_residual_source_svg_sha256"),
        details={
            "path": _rel(svg_path),
            "actual_sha256": svg_sha,
            "expected_sha256": verification.get(
                "overlay_residual_source_svg_sha256",
                "",
            ),
        },
    )
    _check(
        checks,
        "draft_packet_hash",
        bool(packet.get("draft_packet_hash_verified")),
        details={
            "path": packet.get("draft_packet_path", ""),
            "actual_sha256": packet.get("draft_packet_sha256", ""),
            "expected_sha256": packet.get("draft_packet_expected_sha256", ""),
        },
    )
    _check(
        checks,
        "source_caption_line_window",
        _source_item_present(source_path, str(packet.get("source_lines", ""))).get(
            "passed",
        ),
        details=_source_item_present(source_path, str(packet.get("source_lines", ""))),
    )

    pdf_report = _find_akel_pdf(base, queue)
    _check(
        checks,
        "source_pdf_hash",
        bool(pdf_report["matched_candidates"]),
        details=pdf_report,
    )

    parity_report: dict[str, Any] = {
        "skipped": not check_pdf_text_parity,
        "reason": "disabled_by_caller" if not check_pdf_text_parity else "",
    }
    if check_pdf_text_parity:
        selected_pdf = str(pdf_report.get("selected_path", ""))
        if selected_pdf and json_path.exists():
            parity_report = verify_pair(base / selected_pdf, source_path, json_path)
            _check(
                checks,
                "pdf_markdown_json_text_parity",
                bool(parity_report["passed"]),
                details={
                    "pdf_path": parity_report["pdf_path"],
                    "markdown_path": parity_report["markdown_path"],
                    "json_path": parity_report["json_path"],
                    "pdf_pages": parity_report["pdf_pages"],
                    "failures": parity_report["failures"],
                },
            )
        else:
            _check(
                checks,
                "pdf_markdown_json_text_parity",
                False,
                details={
                    "selected_pdf": selected_pdf,
                    "json_path": _rel(json_path),
                    "json_exists": json_path.exists(),
                },
            )

    point_counts = _packet_point_count_report(packet, fig1_task)
    _check(checks, "series_point_counts", point_counts["passed"], details=point_counts)

    digitization_failures = set(
        str(item) for item in evidence.get("missing_or_failed_checks", [])
    )
    non_review_failures = sorted(digitization_failures - REVIEW_ONLY_FAILURES)
    review_failures = sorted(digitization_failures & REVIEW_ONLY_FAILURES)
    non_review_integrity_passed = not non_review_failures
    review_accepted = bool(evidence.get("passed"))
    expected_digitization_state_passed = (
        review_accepted if require_accepted_review else non_review_integrity_passed
    )
    _check(
        checks,
        "digitization_non_review_integrity",
        non_review_integrity_passed,
        details={
            "all_failures": sorted(digitization_failures),
            "non_review_failures": non_review_failures,
            "review_failures": review_failures,
        },
    )
    _check(
        checks,
        "expected_review_state",
        expected_digitization_state_passed,
        details={
            "require_accepted_review": require_accepted_review,
            "digitization_evidence_passed": review_accepted,
            "readiness_status": readiness.get("waveform_digitization_status"),
        },
    )

    fatal_failed_checks = [
        check["name"]
        for check in checks
        if check["fatal"] and not check["passed"]
    ]
    passed = not fatal_failed_checks
    accepted_for_validation = bool(evidence.get("passed")) and passed
    return {
        "passed": passed,
        "accepted_for_validation": accepted_for_validation,
        "validation_status": (
            "accepted"
            if accepted_for_validation
            else readiness.get("waveform_digitization_status", "blocked")
        ),
        "model_role": "akel_digitization_source_integrity_pre_review",
        "validation_scope": packet.get("validation_scope"),
        "task_id": packet.get("task_id"),
        "fatal_failed_checks": fatal_failed_checks,
        "checks": checks,
        "document_parity": parity_report,
        "digitization_evidence": evidence,
        "digitization_status": status,
        "readiness_evidence": readiness,
        "validity_notes": {
            "science_authority": (
                "This verifies local source and digitization integrity only. "
                "Science claims still require local KnowledgeReference support."
            ),
            "review_boundary": (
                "A blocked-by-review report may pass this script when all "
                "non-review integrity checks pass, but it remains "
                "accepted_for_validation=false."
            ),
            "simulation_boundary": (
                "This script does not compare a simulation waveform to Akel. "
                "It only protects source parity and digitization provenance."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path("."))
    parser.add_argument(
        "--skip-pdf-text-parity",
        action="store_true",
        help="Skip the PDF/markdown/JSON text parity check.",
    )
    parser.add_argument(
        "--require-accepted-review",
        action="store_true",
        help="Fail if the packet has not passed independent review.",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = build_akel_digitization_source_integrity_report(
        base_path=args.base_path,
        check_pdf_text_parity=not args.skip_pdf_text_parity,
        require_accepted_review=args.require_accepted_review,
    )
    payload = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
