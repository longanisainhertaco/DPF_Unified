"""Create the A14 crop-boundary review inventory.

This report is a workbench QA artifact. It records crop hashes, review status,
and next actions for target-extraction crops, but it does not accept any crop or
digitized data for validation.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT_PATH = REPO_ROOT / "docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.json"
OUTPUT_JSON_PATH = REPO_ROOT / "docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json"
OUTPUT_MD_PATH = REPO_ROOT / "docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.md"


DRAFT_TABLE_IDS = {
    ("springham-2021-zrbe-activation", "Table 1"),
    ("springham-2021-zrbe-activation", "Table 2"),
    ("catenacci-2020-neutron-tomography", "Table I"),
    ("catenacci-2020-neutron-tomography", "Table II"),
    ("catenacci-2020-neutron-tomography", "Table III"),
    ("catenacci-2020-neutron-tomography", "Table IV"),
}


VISUAL_QA = {
    ("cikhardtova-2015-linear-density", "Fig. 1"): (
        "boundary_ready_for_draft_extraction",
        "Axes, curves, ticks, legend, and caption are visible.",
    ),
    ("cikhardtova-2015-linear-density", "Fig. 2"): (
        "manual_review_required",
        "Interferogram imagery appears complete but is not direct axis extraction.",
    ),
    ("cikhardtova-2015-linear-density", "Fig. 3"): (
        "boundary_ready_for_draft_extraction",
        "Both 3D density panels and caption are visible; axes are small.",
    ),
    ("cikhardtova-2015-linear-density", "Fig. 4"): (
        "boundary_ready_for_draft_extraction",
        "Both 3D density panels and caption are visible; axes are small.",
    ),
    ("cikhardtova-2015-linear-density", "Fig. 5"): (
        "boundary_ready_for_draft_extraction",
        "Recropped plot now preserves the plot body, axes, and caption.",
    ),
    ("cikhardtova-2015-linear-density", "Fig. 6"): (
        "boundary_ready_for_draft_extraction",
        "Clean 2D plot with axes, legend, curves, and caption.",
    ),
    ("szydlowski-2004-fast-ion-neutron", "Fig. 1"): (
        "manual_review_required",
        "Complete diagram crop; geometry extraction is manual, not axis digitization.",
    ),
    ("szydlowski-2004-fast-ion-neutron", "Fig. 2"): (
        "boundary_ready_for_draft_extraction",
        "Time traces and side images are included for timing extraction.",
    ),
    ("szydlowski-2004-fast-ion-neutron", "Fig. 3"): (
        "boundary_ready_for_draft_extraction",
        "Clean spectrum plot with axes, ticks, legend, and caption.",
    ),
    ("szydlowski-2004-fast-ion-neutron", "Fig. 4"): (
        "manual_review_required",
        "Complete image-panel crop; visual track examples, not numeric curves.",
    ),
    ("szydlowski-2004-fast-ion-neutron", "Fig. 5"): (
        "boundary_ready_for_draft_extraction",
        "Clean angular-distribution plot with axes, legend, and data points.",
    ),
    ("klir-2011-tof-detector", "Fig. 1"): (
        "manual_review_required",
        "Recropped detector diagram is complete; geometry extraction is manual.",
    ),
    ("klir-2011-tof-detector", "Fig. 2"): (
        "boundary_ready_for_draft_extraction",
        "Clean calibration plot with axes, curves, error bars, and caption.",
    ),
    ("klir-2011-tof-detector", "Fig. 3"): (
        "boundary_ready_for_draft_extraction",
        "Recropped time-response plot removes next-section text.",
    ),
    ("klir-2011-tof-detector", "Fig. 4"): (
        "boundary_ready_for_draft_extraction",
        "Recropped PMT-delay plot removes stray text and watermark.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 1"): (
        "boundary_ready_for_draft_extraction",
        "Both plotted panels, axes, legends, and caption are visible.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 2"): (
        "manual_review_required",
        "Complete apparatus/layout crop; mostly geometric extraction.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 3"): (
        "manual_review_required",
        "Complete block diagram; not a numeric plot/table.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 4"): (
        "boundary_ready_for_draft_extraction",
        "Spectrum plot is visible with axes, curves, and caption.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 5"): (
        "boundary_ready_for_draft_extraction",
        "Clean calibration curve with axes, legend, and caption.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 6"): (
        "boundary_ready_for_draft_extraction",
        "Multi-panel plot is fully visible; dense but extractable.",
    ),
    ("springham-2021-zrbe-activation", "Fig. 7"): (
        "boundary_ready_for_draft_extraction",
        "Three pressure-sweep panels visible with axes and legends.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 1"): (
        "manual_review_required",
        "Recropped shadow-bar diagram is complete; geometry extraction is manual.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 2"): (
        "boundary_ready_for_draft_extraction",
        "Recropped calibration/subtraction panels preserve axes, legends, and caption.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 3"): (
        "manual_review_required",
        "Complete experimental setup diagram; mostly manual geometry extraction.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 4"): (
        "boundary_ready_for_draft_extraction",
        "Four 3D reconstructions visible; axes/caption present though small.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 5"): (
        "boundary_ready_for_draft_extraction",
        "Two curve panels visible with axes and caption.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 6"): (
        "boundary_ready_for_draft_extraction",
        "Time profiles visible with axes and explanatory caption.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 7"): (
        "manual_review_required",
        "Tall dense reconstruction panels need manual review before extraction.",
    ),
    ("catenacci-2020-neutron-tomography", "Fig. 8"): (
        "boundary_ready_for_draft_extraction",
        "Clean time trace with axes, dashed markers, and caption.",
    ),
}


RECOMMENDED_NEXT_AXIS_CALIBRATION = [
    {
        "source_slug": "cikhardtova-2015-linear-density",
        "figure_id": "Fig. 6",
        "reason": "Clean 2D numeric plot with multiple labeled traces.",
    },
    {
        "source_slug": "klir-2011-tof-detector",
        "figure_id": "Fig. 2",
        "reason": "Clean response-versus-voltage calibration plot with error bars.",
    },
    {
        "source_slug": "springham-2021-zrbe-activation",
        "figure_id": "Fig. 5",
        "reason": "Clean calibration curve with axes and legend fully visible.",
    },
]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _entry_for_crop(task: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
    crop_path = Path(crop["path"])
    crop_sha256 = _sha256_file(crop_path)
    crop_hash_matches = crop_sha256 == crop["sha256"]
    source_slug = task["slug"]
    figure_id = crop["figure_id"]
    is_table = figure_id.startswith("Table")
    is_draft_extracted_table = (source_slug, figure_id) in DRAFT_TABLE_IDS
    qa_status, qa_reason = VISUAL_QA.get(
        (source_slug, figure_id),
        (
            "pending_boundary_review",
            "No visual QA status has been recorded for this crop yet.",
        ),
    )

    if is_draft_extracted_table:
        status = "draft_extracted_review_blocked"
        checklist = {
            "source_hash_recorded": True,
            "crop_file_exists": crop_path.exists(),
            "crop_hash_matches_report": crop_hash_matches,
            "axes_or_table_visible": "table_body_transcribed_from_crop",
            "caption_or_title_visible": "title_visible_or_source_line_recorded",
            "legend_visible_or_not_applicable": "not_applicable_table",
            "units_visible_or_not_applicable": "units_transcribed_where_present",
            "trace_or_table_body_visible": "table_body_transcribed",
            "requires_numeric_extraction": False,
            "requires_independent_review": True,
        }
        next_action = (
            "Run independent review on the table draft packet; do not use for "
            "validation until digitization_verification_evidence() passes."
        )
        notes = (
            "Table has a draft packet in "
            "KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json."
        )
    else:
        status = qa_status
        checklist = {
            "source_hash_recorded": True,
            "crop_file_exists": crop_path.exists(),
            "crop_hash_matches_report": crop_hash_matches,
            "axes_or_table_visible": (
                "visually_ready"
                if status == "boundary_ready_for_draft_extraction"
                else "manual_or_adjustment_review_needed"
            ),
            "caption_or_title_visible": (
                "visually_ready"
                if status == "boundary_ready_for_draft_extraction"
                else "manual_or_adjustment_review_needed"
            ),
            "legend_visible_or_not_applicable": (
                "visually_ready_or_not_applicable"
                if status == "boundary_ready_for_draft_extraction"
                else "manual_or_adjustment_review_needed"
            ),
            "units_visible_or_not_applicable": (
                "visually_ready_or_not_applicable"
                if status == "boundary_ready_for_draft_extraction"
                else "manual_or_adjustment_review_needed"
            ),
            "trace_or_table_body_visible": (
                "visually_ready"
                if status == "boundary_ready_for_draft_extraction"
                else "manual_or_adjustment_review_needed"
            ),
            "requires_numeric_extraction": True,
            "requires_independent_review": True,
        }
        if status == "boundary_ready_for_draft_extraction":
            next_action = (
                "Record axis calibration and draft numeric arrays, then "
                "measure residuals and request independent review."
            )
        elif status == "manual_review_required":
            next_action = (
                "Perform manual geometry/image interpretation review before "
                "deciding whether numeric extraction is appropriate."
            )
        elif status == "crop_adjustment_needed":
            next_action = (
                "Adjust the crop boundary and regenerate the target extraction "
                "report before digitizing."
            )
        else:
            next_action = (
                "Review crop boundaries visually, then record axis/table "
                "calibration before extracting numeric arrays."
            )
        notes = qa_reason

    return {
        "source_slug": source_slug,
        "source_md": task["source_md"],
        "source_md_sha256": task["source_md_sha256"],
        "source_pdf": task["source_pdf"],
        "source_pdf_sha256": task["source_pdf_sha256"],
        "figure_id": figure_id,
        "page": crop["page"],
        "crop_path": _repo_relative(crop_path),
        "crop_sha256": crop_sha256,
        "source_report_crop_sha256": crop["sha256"],
        "width_px": crop["width_px"],
        "height_px": crop["height_px"],
        "dpi": crop["dpi"],
        "pdf_rect_points": crop["pdf_rect_points"],
        "required_data": crop["required_data"],
        "extraction_kind": "table" if is_table else "figure",
        "boundary_review_status": status,
        "visual_qa_basis": "read_only_subagent_visual_review_2026_05_11",
        "visual_qa_reason": qa_reason,
        "checklist": checklist,
        "next_action": next_action,
        "accepted_for_validation": False,
        "validation_gate": "digitization_verification_evidence",
        "notes": notes,
    }


def build_report() -> dict[str, Any]:
    source_report = json.loads(SOURCE_REPORT_PATH.read_text())
    entries = [
        _entry_for_crop(task, crop)
        for task in source_report["tasks"]
        for crop in task["crop_candidates"]
    ]
    status_counts = Counter(entry["boundary_review_status"] for entry in entries)
    kind_counts = Counter(entry["extraction_kind"] for entry in entries)
    return {
        "report_id": "A14_CROP_BOUNDARY_REVIEW_2026_05_11",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_boundary": (
            "Local KnowledgeReference records, hashed local intake PDFs, and "
            "generated crop workbench files only. This report is QA status; it "
            "is not accepted validation evidence."
        ),
        "source_report_path": _repo_relative(SOURCE_REPORT_PATH),
        "source_report_sha256": _sha256_file(SOURCE_REPORT_PATH),
        "total_crop_count": len(entries),
        "figure_crop_count": kind_counts["figure"],
        "table_crop_count": kind_counts["table"],
        "accepted_for_validation_count": 0,
        "boundary_review_status_counts": dict(sorted(status_counts.items())),
        "recommended_next_axis_calibration_crops": RECOMMENDED_NEXT_AXIS_CALIBRATION,
        "review_entries": entries,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A14 Crop-Boundary Review",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "This is a crop-boundary QA and planning artifact only. It is not "
        "accepted validation evidence, and no figure/table data may support "
        "simulation validation until a packet passes "
        "`digitization_verification_evidence()` with independent review.",
        "",
        "## Summary",
        "",
        f"- Source report: `{report['source_report_path']}`",
        f"- Source report SHA-256: `{report['source_report_sha256']}`",
        f"- Total crops: {report['total_crop_count']}",
        f"- Figure crops: {report['figure_crop_count']}",
        f"- Table crops: {report['table_crop_count']}",
        f"- Accepted for validation: {report['accepted_for_validation_count']}",
        "",
        "Boundary status counts:",
        "",
    ]
    for status, count in report["boundary_review_status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "Recommended next axis-calibration crops:", ""])
    for item in report["recommended_next_axis_calibration_crops"]:
        lines.append(
            f"- `{item['source_slug']}` {item['figure_id']}: {item['reason']}"
        )
    lines.extend(
        [
            "",
            "## Entries",
            "",
            "| Source | Item | Page | Status | Crop | Next action |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for entry in report["review_entries"]:
        lines.append(
            "| "
            f"{entry['source_slug']} | "
            f"{entry['figure_id']} | "
            f"{entry['page']} | "
            f"`{entry['boundary_review_status']}` | "
            f"`{entry['crop_path']}` | "
            f"{entry['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- `draft_extracted_review_blocked` means a draft table packet exists, "
            "but it remains unusable for validation until independent review "
            "accepts the exact source/crop/table packet.",
            "- `boundary_ready_for_draft_extraction` means a visual QA pass found "
            "the crop boundary suitable for draft calibration/extraction, not "
            "that the figure data are accepted.",
            "- `manual_review_required` means the crop is mainly diagram/image "
            "content or otherwise needs human interpretation before numeric "
            "digitization.",
            "- `crop_adjustment_needed` means the crop must be fixed and "
            "regenerated before extraction.",
            "- `pending_boundary_review` means only the crop file and hash were "
            "inventoried.",
            "- `accepted_for_validation_count` must stay `0` until accepted "
            "digitization packets exist.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUTPUT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD_PATH.write_text(_markdown(report))
    print(
        "wrote",
        _repo_relative(OUTPUT_JSON_PATH),
        _repo_relative(OUTPUT_MD_PATH),
        "crops=",
        report["total_crop_count"],
    )


if __name__ == "__main__":
    main()
