"""Phase 5 figure-source manifest loading and staged packet construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dpf.first_principles.figure_candidate_staging import stage_figure_observable_candidate
from scripts.validate_ss12_phase5_figure_source_manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH = (
    ROOT / "docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json"
)


def load_phase5_figure_source_manifest(
    manifest_path: str | Path | None = DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load the Phase 5 figure-source manifest as JSON."""

    path = Path(manifest_path or DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH)
    return json.loads(path.read_text())


def build_phase5_figure_source_packets(
    *,
    manifest_path: str | Path | None = DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build staged non-promoting figure packets from the Phase 5 manifest.

    This helper intentionally does not digitize values and never promotes
    figure-derived evidence into acceptance. Its output feeds the Phase 4-D
    shield and later digitization/review steps.
    """

    path = Path(manifest_path or DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH)
    if not path.exists():
        return _blocked_packet("blocked_phase5_manifest_missing", "phase5_figure_source_manifest_missing")

    manifest = load_phase5_figure_source_manifest(path)
    validation_issues = validate_manifest(manifest, ROOT)
    if validation_issues:
        return _blocked_packet(
            "blocked_phase5_manifest_invalid",
            *sorted({str(issue.get("rule")) for issue in validation_issues}),
        )

    rows = manifest.get("figure_sources")
    if not isinstance(rows, list):
        return _blocked_packet("blocked_phase5_manifest_invalid", "phase5_figure_sources_not_list")

    staged_packets = [_stage_manifest_row(row, manifest) for row in rows if isinstance(row, dict)]
    accepted_rows = sum(
        1 for row in rows if isinstance(row, dict) and row.get("status") == "accepted"
    )
    blocking_reasons = sorted(
        {
            reason
            for packet in staged_packets
            for reason in packet.get("blocking_reasons", [])
        }
    )
    return {
        "status": "staged_phase5_figure_sources_not_accepted",
        "manifest_path": str(path),
        "manifest_id": manifest.get("manifest_id"),
        "validation_scope": manifest.get("validation_scope"),
        "accepted_figure_claim": False,
        "accepted_observable_claim": False,
        "promotes_acceptance": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "summary": {
            "total_rows": len(rows),
            "staged_packets": len(staged_packets),
            "accepted_rows": accepted_rows,
            "channels": sorted(
                {
                    str(row.get("channel"))
                    for row in rows
                    if isinstance(row, dict) and row.get("channel")
                }
            ),
        },
        "blocking_reasons": blocking_reasons,
        "staged_figure_packets": staged_packets,
        "acceptance_rule": (
            "Phase 5 figure-source packets are staged candidates only; "
            "digitized values require uncertainty, same-source classification, "
            "review certificate, and later certificate-stack validation before "
            "acceptance can even be considered."
        ),
    }


def _stage_manifest_row(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "validation_scope": manifest.get("validation_scope"),
        "channel": row.get("channel"),
        "source_path": row.get("source_path"),
        "line_start": row.get("line_start"),
        "line_end": row.get("line_end"),
        "figure_id": row.get("figure_id"),
        "extraction_method": row.get("extraction_method", "not_digitized_manifest_candidate"),
        "digitization_hash": row.get("digitization_hash", "pending_digitization"),
        "uncertainty": row.get("uncertainty", {"status": "pending"}),
        "reviewer": row.get("reviewer", "pending_review"),
        "review_state": row.get("review_state"),
        "scope_classification": row.get("scope_classification"),
    }
    staged = stage_figure_observable_candidate(candidate)
    staged["manifest_row_id"] = row.get("id")
    staged["manifest_status"] = row.get("status")
    staged["extraction_priority"] = row.get("extraction_priority")
    staged["extraction_target"] = row.get("extraction_target")
    staged["promotes_acceptance"] = False
    return staged


def _blocked_packet(status: str, *reasons: str) -> dict[str, Any]:
    return {
        "status": status,
        "accepted_figure_claim": False,
        "accepted_observable_claim": False,
        "promotes_acceptance": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "summary": {"total_rows": 0, "staged_packets": 0, "accepted_rows": 0},
        "blocking_reasons": list(reasons),
        "staged_figure_packets": [],
    }
