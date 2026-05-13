"""Create A14 axis-calibration draft packets for selected clean figure crops.

These packets record source-bound crop hashes and approximate raster frame
calibration metadata. They intentionally contain no digitized curve arrays and
are not validation evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = (
    REPO_ROOT
    / "KnowledgeReference"
    / "digitization"
    / "a14-2026-05-11-axis-calibration-draft-packets.json"
)
OUTPUT_MD = REPO_ROOT / "docs" / "A14_AXIS_CALIBRATION_DRAFTS_2026_05_11.md"


@dataclass(frozen=True)
class AxisCalibrationDraft:
    task_id: str
    validation_scope: str
    source_path: str
    source_pdf_path: str
    source_lines: str
    crop_image_path: str
    figure_id: str
    page: int
    required_data: str
    plot_frame_px: dict[str, list[float]]
    axis_calibration_candidate: dict[str, Any]
    visible_series: list[dict[str, Any]]
    extraction_notes: list[str]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_pair(stem: str, pdf_name: str) -> tuple[str, str]:
    return (
        f"KnowledgeReference/{stem}",
        f"downloaded_books_papers/Research Papers/2026-05-11-user-ingest/{pdf_name}",
    )


def calibration_drafts() -> list[AxisCalibrationDraft]:
    cikhardtova_md, cikhardtova_pdf = _source_pair(
        "cikhardtova-plazma-indd-9dfed6c0.md",
        "cikhardtova2015.pdf",
    )
    klir_md, klir_pdf = _source_pair(
        "fusion-neutron-detector-for-time-of-flight-measurements-in-z-pinch-"
        "and-plasma-focus-214fbdae.md",
        "klir2011.pdf",
    )
    springham_md, springham_pdf = _source_pair(
        "nuclear-inst-and-methods-in-physics-research-a-988-2021-164830-"
        "bc8edab3.md",
        "springham2021.pdf",
    )
    return [
        AxisCalibrationDraft(
            task_id="a14_cikhardtova_2015_fig6_axis_calibration_draft",
            validation_scope="a14_cikhardtova_2015_linear_density_motion",
            source_path=cikhardtova_md,
            source_pdf_path=cikhardtova_pdf,
            source_lines="200-222",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "cikhardtova-2015-linear-density/crops/page-03-fig-6.png"
            ),
            figure_id="Fig. 6",
            page=3,
            required_data="temporal_evolution_linear_density_curve",
            plot_frame_px={"x": [179.0, 774.0], "y": [14.0, 459.0]},
            axis_calibration_candidate={
                "x": {
                    "quantity": "z_axis",
                    "unit": "mm",
                    "visible_tick_labels": [10, 20, 30, 40, 50],
                    "data_range": [10.0, 50.0],
                    "pixel_range_approx": [179.0, 774.0],
                    "calibration_status": "draft_approximate_raster_frame",
                },
                "y": {
                    "quantity": "linear_density",
                    "unit": "per_mm_context_from_source_lines_208_218",
                    "visible_tick_labels": [
                        "5,00E+017",
                        "1,00E+018",
                        "1,50E+018",
                        "2,00E+018",
                        "2,50E+018",
                        "3,00E+018",
                        "3,50E+018",
                        "4,00E+018",
                        "4,50E+018",
                        "5,00E+018",
                    ],
                    "data_range": [5.0e17, 5.0e18],
                    "pixel_range_approx": [459.0, 14.0],
                    "calibration_status": "draft_approximate_raster_frame",
                },
            },
            visible_series=[
                {"name": "-5 ns", "style": "black dotted"},
                {"name": "+25 ns", "style": "black dashed"},
                {"name": "+55 ns", "style": "black solid"},
                {"name": "+85 ns", "style": "black dash-dot"},
                {"name": "+95 ns", "style": "black long-dashed"},
            ],
            extraction_notes=[
                "Assisted/manual extraction required because monochrome line "
                "styles overlap and several traces nearly merge.",
                "Decimal-comma scientific notation from the axis should be "
                "preserved in review metadata.",
                "Source lines 208-218 provide per-mm context and axial-motion "
                "interpretation; the crop axis itself labels only linear density.",
            ],
        ),
        AxisCalibrationDraft(
            task_id="a14_klir_2011_fig2_axis_calibration_draft",
            validation_scope="a14_klir_2011_tof_detector_response",
            source_path=klir_md,
            source_pdf_path=klir_pdf,
            source_lines="172-209",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "klir-2011-tof-detector/crops/page-03-fig-2.png"
            ),
            figure_id="Fig. 2",
            page=3,
            required_data="pmt_response_fwhm_and_rise_time_vs_voltage",
            plot_frame_px={"x": [218.0, 808.0], "y": [102.0, 539.0]},
            axis_calibration_candidate={
                "x": {
                    "quantity": "pmt_voltage",
                    "unit": "kV",
                    "visible_tick_labels": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
                    "data_range": [1.0, 2.4],
                    "pixel_range_approx": [218.0, 808.0],
                    "calibration_status": "draft_visible_tick_range",
                },
                "y": {
                    "quantity": "time_response",
                    "unit": "ns",
                    "visible_tick_labels": [1, 2, 3, 4, 5],
                    "data_range": [1.0, 5.0],
                    "pixel_range_approx": [539.0, 102.0],
                    "calibration_status": "draft_visible_tick_range",
                },
            },
            visible_series=[
                {"name": "FWHM", "style": "black curve with error bars"},
                {"name": "Rise time", "style": "black curve with error bars"},
            ],
            extraction_notes=[
                "Manual seed points are required because both series are black "
                "and error bars intersect the guide curves.",
                "Caption states error bars indicate +/-2 sigma uncertainty.",
                "Source lines 172-209 bind Fig. 2 to PMT voltage, time "
                "response, and detector timing context.",
            ],
        ),
        AxisCalibrationDraft(
            task_id="a14_springham_2021_fig5_axis_calibration_draft",
            validation_scope="a14_springham_2021_zrbe_activation_response",
            source_path=springham_md,
            source_pdf_path=springham_pdf,
            source_lines="546-616",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "springham-2021-zrbe-activation/crops/page-06-fig-5.png"
            ),
            figure_id="Fig. 5",
            page=6,
            required_data="mcnp5_response_effective_energy_vs_zr_be_count_ratio",
            plot_frame_px={"x": [506.0, 1408.0], "y": [15.0, 740.0]},
            axis_calibration_candidate={
                "x": {
                    "quantity": "zr_to_be_detector_count_ratio",
                    "unit": "dimensionless",
                    "visible_tick_labels": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
                    "data_range": [0.0, 0.14],
                    "pixel_range_approx": [506.0, 1408.0],
                    "calibration_status": "draft_approximate_raster_frame",
                },
                "y": {
                    "quantity": "effective_neutron_energy",
                    "unit": "MeV",
                    "visible_tick_labels": [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6],
                    "data_range": [2.2, 3.6],
                    "pixel_range_approx": [740.0, 15.0],
                    "calibration_status": "draft_approximate_raster_frame",
                },
            },
            visible_series=[
                {"name": "mono-energetic neutrons", "style": "blue open-circle curve"},
                {
                    "name": "Gaussian peak neutrons (200 keV FWHM)",
                    "style": "black curve",
                },
                {
                    "name": "Gaussian peak neutrons (400 keV FWHM)",
                    "style": "red curve",
                },
            ],
            extraction_notes=[
                "Good candidate for automated color-assisted extraction, with "
                "manual handling around open-circle markers.",
                "Legend and title boxes occlude portions of the plot area; "
                "do not synthesize hidden curve segments.",
                "Source lines 561-579 state that only the mono-energetic "
                "curve was used for subsequent effective-energy calculations.",
            ],
        ),
    ]


def _packet(draft: AxisCalibrationDraft) -> dict[str, Any]:
    source_path = REPO_ROOT / draft.source_path
    pdf_path = REPO_ROOT / draft.source_pdf_path
    crop_path = REPO_ROOT / draft.crop_image_path
    return {
        "task_id": draft.task_id,
        "validation_scope": draft.validation_scope,
        "source_path": draft.source_path,
        "source_sha256": sha256_file(source_path),
        "source_pdf_path": draft.source_pdf_path,
        "source_pdf_sha256": sha256_file(pdf_path),
        "source_lines": draft.source_lines,
        "figure_image_path": draft.crop_image_path,
        "figure_image_sha256": sha256_file(crop_path),
        "figure_id": draft.figure_id,
        "page": draft.page,
        "extraction_type": "figure",
        "extraction_status": "axis_calibration_draft_no_series",
        "required_data": draft.required_data,
        "plot_frame_px": draft.plot_frame_px,
        "axis_calibration_candidate": draft.axis_calibration_candidate,
        "visible_series": draft.visible_series,
        "digitized_series": [],
        "accepted_for_validation": False,
        "verification": {
            "independent_review_count": 0,
            "review_status": "draft_unreviewed",
            "accepted_for_validation": False,
            "residual_status": "not_measured",
        },
        "extraction_notes": draft.extraction_notes,
    }


def build_bundle() -> dict[str, Any]:
    packets = [_packet(draft) for draft in calibration_drafts()]
    return {
        "model_role": "a14_axis_calibration_draft_packets",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_boundary": (
            "Local KnowledgeReference records, local source PDFs, and generated "
            "crop PNGs only. These are draft axis-calibration scaffolds, not "
            "digitized arrays or validation evidence."
        ),
        "packet_count": len(packets),
        "accepted_for_validation_count": 0,
        "packets": packets,
    }


def _markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# A14 Axis-Calibration Drafts",
        "",
        f"Generated UTC: `{bundle['generated_utc']}`",
        "",
        "These packets record source-bound crop hashes and draft axis/frame "
        "metadata for the first three A14 figure extraction candidates. They "
        "contain no digitized series arrays, no residuals, and no independent "
        "review acceptance.",
        "",
        "## Summary",
        "",
        f"- Draft packets: {bundle['packet_count']}",
        f"- Accepted for validation: {bundle['accepted_for_validation_count']}",
        "",
        "| Task | Figure | Source lines | Status | Visible series |",
        "| --- | --- | --- | --- | --- |",
    ]
    for packet in bundle["packets"]:
        series = ", ".join(item["name"] for item in packet["visible_series"])
        lines.append(
            "| "
            f"`{packet['task_id']}` | "
            f"{packet['figure_id']} | "
            f"`{packet['source_lines']}` | "
            f"`{packet['extraction_status']}` | "
            f"{series} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- These packets are calibration scaffolds only.",
            "- Pixel ranges are approximate raster-frame metadata and must be "
            "refined during numeric extraction.",
            "- Hidden/occluded curve segments must not be synthesized.",
            "- Validation use requires digitized arrays, residual evidence, and "
            "accepted independent review through `digitization_verification_evidence()`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    bundle = build_bundle()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(bundle))
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "packet_count": bundle["packet_count"],
                "accepted_for_validation_count": bundle[
                    "accepted_for_validation_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
