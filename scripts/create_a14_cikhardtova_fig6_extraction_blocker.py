"""Create the A14 Cikhardtova Fig. 6 extraction-blocker report."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = (
    REPO_ROOT / "docs" / "A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.json"
)
OUTPUT_MD = (
    REPO_ROOT / "docs" / "A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.md"
)

SOURCE_PATH = "KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md"
SOURCE_PDF_PATH = (
    "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
    "cikhardtova2015.pdf"
)
FIGURE_IMAGE_PATH = (
    "KnowledgeReference/figures/target-extraction/2026-05-11/"
    "cikhardtova-2015-linear-density/crops/page-03-fig-6.png"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report() -> dict[str, Any]:
    return {
        "model_role": "a14_extraction_blocker_report",
        "task_id": "a14_cikhardtova_2015_fig6_linear_density_extraction_blocker",
        "validation_scope": "a14_cikhardtova_2015_linear_density_motion",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_path": SOURCE_PATH,
        "source_sha256": sha256_file(REPO_ROOT / SOURCE_PATH),
        "source_pdf_path": SOURCE_PDF_PATH,
        "source_pdf_sha256": sha256_file(REPO_ROOT / SOURCE_PDF_PATH),
        "source_lines": "200-222",
        "figure_image_path": FIGURE_IMAGE_PATH,
        "figure_image_sha256": sha256_file(REPO_ROOT / FIGURE_IMAGE_PATH),
        "figure_id": "Fig. 6",
        "page": 3,
        "accepted_for_validation": False,
        "draft_extraction_status": "blocked_manual_curve_separation_required",
        "blocked_reason": (
            "Five monochrome line styles overlap and nearly merge across the "
            "same z-axis intervals. A quick point-pick pass could mislabel "
            "series and would not be defensible as a draft numeric packet."
        ),
        "visible_series": [
            {"name": "-5 ns", "style": "black dotted"},
            {"name": "+25 ns", "style": "black dashed"},
            {"name": "+55 ns", "style": "black solid"},
            {"name": "+85 ns", "style": "black dash-dot"},
            {"name": "+95 ns", "style": "black long-dashed"},
        ],
        "axis_context": {
            "x": {
                "quantity": "z_axis",
                "unit": "mm",
                "visible_range": [10.0, 50.0],
            },
            "y": {
                "quantity": "linear_density",
                "unit": "per_mm_context_from_source_lines_208_218",
                "visible_range": [5.0e17, 5.0e18],
            },
        },
        "required_next_steps": [
            "perform manual or vector-assisted curve separation for all five series",
            "record per-series pixel picks with line-style labels",
            "measure round-trip residuals for every extracted series",
            "document uncertainty from overlapping/merged curve regions",
            "submit the resulting packet for independent review before validation use",
        ],
        "validation_gate": "digitization_verification_evidence",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A14 Cikhardtova Fig. 6 Extraction Blocker",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "This report records why Cikhardtova 2015 Fig. 6 was not converted into "
        "a numeric draft packet in this pass.",
        "",
        "## Status",
        "",
        f"- Task: `{report['task_id']}`",
        f"- Source: `{report['source_path']}`",
        f"- Source lines: `{report['source_lines']}`",
        f"- Figure crop: `{report['figure_image_path']}`",
        f"- Draft extraction status: `{report['draft_extraction_status']}`",
        f"- Accepted for validation: {report['accepted_for_validation']}",
        "",
        "## Blocker",
        "",
        report["blocked_reason"],
        "",
        "## Required Next Steps",
        "",
    ]
    lines.extend(f"- {step}" for step in report["required_next_steps"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(report))
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "accepted_for_validation": report["accepted_for_validation"],
                "draft_extraction_status": report["draft_extraction_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
