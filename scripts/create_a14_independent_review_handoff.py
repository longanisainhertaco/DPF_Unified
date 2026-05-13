"""Create the A14 independent-review handoff bundle.

The handoff is an external-review checklist and manifest. It does not promote
any draft packet to validation evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from dpf.validation import (
    a14_axis_calibration_draft_packets,
    a14_klir_fig2_timing_response_draft_packet,
    a14_springham_fig5_monoenergetic_draft_packet,
    a14_springham_fig5_gaussian_curve_draft_packet,
    a14_table_extraction_draft_packets,
    digitization_verification_evidence,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = REPO_ROOT / "docs" / "A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.json"
OUTPUT_MD = REPO_ROOT / "docs" / "A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.md"

REVIEW_FIELDS_REQUIRED = [
    "reviewer_identity_or_role",
    "review_date_utc",
    "reviewed_packet_path",
    "reviewed_packet_sha256",
    "reviewed_source_path",
    "reviewed_source_sha256",
    "reviewed_source_pdf_path",
    "reviewed_source_pdf_sha256",
    "reviewed_figure_image_sha256",
    "reviewed_crop_image_sha256",
    "review_status",
    "independent_review_count",
    "reviewer_notes",
]

REVIEW_CHECKLIST_REQUIRED = [
    "source_hash_matches_current_artifact",
    "pdf_hash_matches_current_artifact",
    "figure_or_crop_hash_matches_current_artifact",
    "source_lines_support_item_identity",
    "axis_or_table_structure_checked",
    "series_or_table_values_checked_against_source",
    "units_checked",
    "uncertainty_or_error_bar_handling_checked",
    "residual_or_table_transcription_quality_checked",
    "no_hidden_or_occluded_values_synthesized",
    "review_status_is_accepted_or_explicitly_rejected",
]


def _image_path(packet: dict[str, Any]) -> str:
    return str(packet.get("figure_image_path") or packet.get("crop_image_path") or "")


def _image_sha256(packet: dict[str, Any]) -> str:
    return str(
        packet.get("figure_image_sha256") or packet.get("crop_image_sha256") or ""
    )


def _review_item(
    *,
    packet: dict[str, Any],
    packet_path: str,
    packet_sha256: str,
    packet_kind: str,
    packet_index: int | None = None,
) -> dict[str, Any]:
    evidence = digitization_verification_evidence(packet)
    verification = packet.get("verification", {})
    verification_map = verification if isinstance(verification, dict) else {}
    series = packet.get("digitized_series", [])
    table_rows = packet.get("table_rows", [])
    reviewed_packet_sha256 = str(
        packet.get("draft_packet_item_sha256")
        or packet.get("draft_packet_sha256")
        or packet_sha256
    )
    review_metadata_template = {
        "task_id": packet["task_id"],
        "validation_scope": packet.get("validation_scope", ""),
        "reviewed_packet_sha256": reviewed_packet_sha256,
        "reviewed_source_sha256": packet.get("source_sha256", ""),
        "reviewed_source_pdf_sha256": packet.get("source_pdf_sha256", ""),
        "reviewer": "",
        "review_date": "",
        "review_notes": "",
        "decision": "accepted_or_rejected",
    }
    if packet.get("extraction_type") == "table":
        review_metadata_template["reviewed_crop_image_sha256"] = packet.get(
            "crop_image_sha256", ""
        )
    else:
        review_metadata_template["reviewed_figure_image_sha256"] = packet.get(
            "figure_image_sha256", ""
        )
    return {
        "task_id": packet["task_id"],
        "packet_kind": packet_kind,
        "packet_path": packet_path,
        "packet_sha256": reviewed_packet_sha256,
        "packet_bundle_sha256": packet_sha256,
        "packet_index": packet_index,
        "validation_scope": packet.get("validation_scope", ""),
        "source_path": packet.get("source_path", ""),
        "source_sha256": packet.get("source_sha256", ""),
        "source_pdf_path": packet.get("source_pdf_path", ""),
        "source_pdf_sha256": packet.get("source_pdf_sha256", ""),
        "source_lines": packet.get("source_lines", ""),
        "figure_or_table_id": packet.get("figure_id") or packet.get("table_id") or "",
        "page": packet.get("page"),
        "figure_or_crop_path": _image_path(packet),
        "figure_or_crop_sha256": _image_sha256(packet),
        "extraction_type": packet.get("extraction_type", ""),
        "extraction_status": packet.get("extraction_status", ""),
        "digitized_series_count": len(series) if isinstance(series, list) else 0,
        "table_row_count": len(table_rows) if isinstance(table_rows, list) else 0,
        "overlay_rms_residual_px": verification_map.get("overlay_rms_residual_px"),
        "overlay_max_residual_px": verification_map.get("overlay_max_residual_px"),
        "accepted_for_validation": bool(packet.get("accepted_for_validation", False)),
        "current_gate_passed": bool(evidence["passed"]),
        "current_gate_missing_or_failed_checks": list(
            evidence["missing_or_failed_checks"]
        ),
        "review_decision_template": {
            "review_status": "draft_unreviewed",
            "independent_review_count": 0,
            "accepted_for_validation": False,
            "review_metadata": review_metadata_template,
            "required_fields": REVIEW_FIELDS_REQUIRED,
            "required_checklist": REVIEW_CHECKLIST_REQUIRED,
        },
    }


def _axis_context_item(
    *,
    packet: dict[str, Any],
    packet_path: str,
    packet_sha256: str,
    packet_index: int,
) -> dict[str, Any]:
    return {
        "task_id": packet["task_id"],
        "packet_kind": "axis_calibration_context_not_digitization_packet",
        "packet_path": packet_path,
        "packet_sha256": packet_sha256,
        "packet_index": packet_index,
        "validation_scope": packet.get("validation_scope", ""),
        "source_path": packet.get("source_path", ""),
        "source_sha256": packet.get("source_sha256", ""),
        "source_pdf_path": packet.get("source_pdf_path", ""),
        "source_pdf_sha256": packet.get("source_pdf_sha256", ""),
        "source_lines": packet.get("source_lines", ""),
        "figure_or_table_id": packet.get("figure_id", ""),
        "page": packet.get("page"),
        "figure_or_crop_path": packet.get("figure_image_path", ""),
        "figure_or_crop_sha256": packet.get("figure_image_sha256", ""),
        "extraction_status": packet.get("extraction_status", ""),
        "visible_series": packet.get("visible_series", []),
        "axis_calibration_candidate": packet.get("axis_calibration_candidate", {}),
        "accepted_for_validation": False,
        "current_gate_status": (
            "context_only_axis_scaffold_no_digitized_series_or_residual"
        ),
    }


def build_handoff() -> dict[str, Any]:
    table_bundle = a14_table_extraction_draft_packets()
    axis_bundle = a14_axis_calibration_draft_packets()
    springham_packet = a14_springham_fig5_monoenergetic_draft_packet()
    springham_gaussian_packet = a14_springham_fig5_gaussian_curve_draft_packet()
    klir_fig2_packet = a14_klir_fig2_timing_response_draft_packet()

    review_items = [
        _review_item(
            packet=packet,
            packet_path=str(table_bundle["draft_packet_path"]),
            packet_sha256=str(table_bundle["draft_packet_sha256"]),
            packet_kind="table_draft_in_bundle",
            packet_index=index,
        )
        for index, packet in enumerate(table_bundle["packets"])
    ]
    review_items.append(
        _review_item(
            packet=springham_packet,
            packet_path=str(springham_packet["draft_packet_path"]),
            packet_sha256=str(springham_packet["draft_packet_sha256"]),
            packet_kind="figure_digitization_draft",
        )
    )
    review_items.append(
        _review_item(
            packet=springham_gaussian_packet,
            packet_path=str(springham_gaussian_packet["draft_packet_path"]),
            packet_sha256=str(springham_gaussian_packet["draft_packet_sha256"]),
            packet_kind="figure_digitization_draft",
        )
    )
    review_items.append(
        _review_item(
            packet=klir_fig2_packet,
            packet_path=str(klir_fig2_packet["draft_packet_path"]),
            packet_sha256=str(klir_fig2_packet["draft_packet_sha256"]),
            packet_kind="figure_digitization_draft",
        )
    )

    axis_context_items = [
        _axis_context_item(
            packet=packet,
            packet_path=str(axis_bundle["draft_packet_path"]),
            packet_sha256=str(axis_bundle["draft_packet_sha256"]),
            packet_index=index,
        )
        for index, packet in enumerate(axis_bundle["packets"])
    ]

    context_artifacts = [
        {
            "artifact_role": "crop_boundary_review",
            "path": "docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json",
        },
        {
            "artifact_role": "target_extraction_render_and_crop_inventory",
            "path": "docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.json",
        },
        {
            "artifact_role": "remaining_extraction_backlog",
            "path": "docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.md",
        },
        {
            "artifact_role": "table_draft_report",
            "path": "docs/A14_TABLE_EXTRACTION_DRAFTS_2026_05_11.md",
        },
        {
            "artifact_role": "axis_calibration_draft_report",
            "path": "docs/A14_AXIS_CALIBRATION_DRAFTS_2026_05_11.md",
        },
        {
            "artifact_role": "cikhardtova_fig6_extraction_blocker",
            "path": "docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.md",
        },
        {
            "artifact_role": "springham_fig5_digitization_report",
            "path": "docs/A14_SPRINGHAM_FIG5_DIGITIZATION_DRAFT_2026_05_11.md",
        },
    ]
    for artifact in context_artifacts:
        artifact_path = REPO_ROOT / str(artifact["path"])
        artifact["sha256"] = sha256_file(artifact_path)

    return {
        "model_role": "a14_independent_review_handoff",
        "generated_utc": datetime.now(UTC).isoformat(),
        "validation_gate": "digitization_verification_evidence",
        "accepted_for_validation_count": 0,
        "review_item_count": len(review_items),
        "axis_context_item_count": len(axis_context_items),
        "review_fields_required": REVIEW_FIELDS_REQUIRED,
        "review_checklist_required": REVIEW_CHECKLIST_REQUIRED,
        "status_boundary": (
            "This handoff is a review manifest only. It does not promote draft "
            "tables, axis scaffolds, or figure digitization packets to accepted "
            "validation evidence."
        ),
        "source_scope_guardrails": [
            "Use only the listed local KnowledgeReference and hashed local PDF artifacts.",
            "Do not accept packets if packet/source/PDF/crop hashes drift.",
            "Do not treat axis scaffolds as digitized data.",
            "Do not synthesize occluded curve segments or missing table entries.",
            "For table drafts, bind review to the per-table item hash and crop hash.",
            "Record rejected or correction-needed review outcomes explicitly.",
        ],
        "context_artifacts": context_artifacts,
        "review_items": review_items,
        "axis_context_items": axis_context_items,
    }


def _markdown(handoff: dict[str, Any]) -> str:
    lines = [
        "# A14 Independent Review Handoff",
        "",
        f"Generated UTC: `{handoff['generated_utc']}`",
        "",
        handoff["status_boundary"],
        "",
        "## Summary",
        "",
        f"- Reviewable draft digitization packets: {handoff['review_item_count']}",
        f"- Axis-calibration context scaffolds: {handoff['axis_context_item_count']}",
        f"- Accepted for validation: {handoff['accepted_for_validation_count']}",
        f"- Validation gate: `{handoff['validation_gate']}`",
        "",
        "## Required Review Fields",
        "",
    ]
    lines.extend(f"- `{field}`" for field in handoff["review_fields_required"])
    lines.extend(["", "## Required Review Checklist", ""])
    lines.extend(f"- `{item}`" for item in handoff["review_checklist_required"])
    lines.extend(["", "## Review Items", ""])
    for item in handoff["review_items"]:
        lines.extend(
            [
                f"### {item['task_id']}",
                "",
                f"- Packet path: `{item['packet_path']}`",
                f"- Packet SHA-256: `{item['packet_sha256']}`",
                f"- Packet bundle SHA-256: `{item['packet_bundle_sha256']}`",
                f"- Item: `{item['figure_or_table_id']}` on page `{item['page']}`",
                f"- Source: `{item['source_path']}`",
                f"- Source lines: `{item['source_lines']}`",
                f"- Figure/crop: `{item['figure_or_crop_path']}`",
                "- Current gate missing checks: "
                + ", ".join(f"`{check}`" for check in item["current_gate_missing_or_failed_checks"]),
                f"- Accepted for validation: {item['accepted_for_validation']}",
                "",
            ]
        )
    lines.extend(["## Axis Context Items", ""])
    for item in handoff["axis_context_items"]:
        lines.extend(
            [
                f"- `{item['task_id']}`: `{item['figure_or_table_id']}` from "
                f"`{item['source_path']}`; status `{item['current_gate_status']}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    handoff = build_handoff()
    OUTPUT_JSON.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(handoff))
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "review_item_count": handoff["review_item_count"],
                "axis_context_item_count": handoff["axis_context_item_count"],
                "accepted_for_validation_count": handoff[
                    "accepted_for_validation_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
