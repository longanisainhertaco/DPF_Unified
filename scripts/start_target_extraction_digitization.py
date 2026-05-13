#!/usr/bin/env python3
"""Start KR target extraction and figure digitization workbench artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import fitz


RUN_ID = "2026-05-11"
REPORT_STEM = "TARGET_EXTRACTION_DIGITIZATION_2026_05_11"


@dataclass(frozen=True)
class CropTask:
    figure_id: str
    page: int
    rect: tuple[float, float, float, float]
    required_data: str


@dataclass(frozen=True)
class SourceTask:
    slug: str
    title: str
    target_function: str
    source_md: str
    source_pdf: str
    pages: tuple[int, ...]
    figures: tuple[str, ...]
    target_groups: tuple[str, ...]
    status: str
    crop_tasks: tuple[CropTask, ...] = ()


SOURCE_TASKS: tuple[SourceTask, ...] = (
    SourceTask(
        slug="cikhardtova-2015-linear-density",
        title="Cikhardtova 2015 PF-1000 shot 9881 linear-density motion",
        target_function="pf1000_cikhardtova_linear_density_motion_targets",
        source_md="KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md",
        source_pdf=(
            "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
            "cikhardtova2015.pdf"
        ),
        pages=(2, 3),
        figures=("Fig. 1", "Fig. 2", "Figs. 3-6"),
        target_groups=("phase_timing", "spatial_density", "uncertainty"),
        status="target_record_started_page_rendered_crop_pending",
        crop_tasks=(
            CropTask(
                figure_id="Fig. 1",
                page=2,
                rect=(66.0, 70.0, 296.0, 252.0),
                required_data="shot_9881_sxr_hxr_neutron_didt_timing_trace",
            ),
            CropTask(
                figure_id="Fig. 2",
                page=2,
                rect=(60.0, 558.0, 536.0, 780.0),
                required_data="interferograms_at_minus5_plus55_plus95_ns",
            ),
            CropTask(
                figure_id="Fig. 3",
                page=3,
                rect=(60.0, 70.0, 536.0, 264.0),
                required_data="linear_density_profiles_minus5_and_plus25_ns",
            ),
            CropTask(
                figure_id="Fig. 4",
                page=3,
                rect=(60.0, 270.0, 536.0, 456.0),
                required_data="linear_density_profiles_plus55_and_plus85_ns",
            ),
            CropTask(
                figure_id="Fig. 5",
                page=3,
                rect=(58.0, 550.0, 296.0, 785.0),
                required_data="linear_density_profile_plus95_ns",
            ),
            CropTask(
                figure_id="Fig. 6",
                page=3,
                rect=(301.0, 606.0, 536.0, 780.0),
                required_data="temporal_evolution_linear_density_curve",
            ),
        ),
    ),
    SourceTask(
        slug="szydlowski-2004-fast-ion-neutron",
        title="Szydlowski 2004 PF-1000 fast ions and neutrons",
        target_function="pf1000_szydlowski_fast_ion_neutron_targets",
        source_md="KnowledgeReference/doi-10-1016-j-vacuum-2004-07-040-6de67a98.md",
        source_pdf=(
            "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
            "szydlowski2004.pdf"
        ),
        pages=(2, 3, 4),
        figures=("Fig. 1", "Fig. 2", "Fig. 3", "Figs. 4-5"),
        target_groups=(
            "neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ),
        status="target_record_started_page_rendered_crop_pending",
        crop_tasks=(
            CropTask(
                figure_id="Fig. 1",
                page=2,
                rect=(42.0, 500.0, 258.0, 666.0),
                required_data="pf1000_activation_scintillator_geometry",
            ),
            CropTask(
                figure_id="Fig. 2",
                page=3,
                rect=(42.0, 492.0, 258.0, 664.0),
                required_data="xray_neutron_time_sequence",
            ),
            CropTask(
                figure_id="Fig. 3",
                page=3,
                rect=(282.0, 330.0, 497.0, 496.0),
                required_data="upstream_neutron_energy_spectrum",
            ),
            CropTask(
                figure_id="Fig. 4",
                page=3,
                rect=(282.0, 500.0, 497.0, 664.0),
                required_data="cr39_ion_track_examples",
            ),
            CropTask(
                figure_id="Fig. 5",
                page=4,
                rect=(42.0, 90.0, 258.0, 242.0),
                required_data="fast_deuteron_angular_distribution",
            ),
        ),
    ),
    SourceTask(
        slug="klir-2011-tof-detector",
        title="Klir 2011 ToF detector response calibration",
        target_function="klir_2011_tof_detector_response_targets",
        source_md=(
            "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-"
            "measurements-in-z-pinch-and-plasma-focus-214fbdae.md"
        ),
        source_pdf=(
            "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
            "klir2011.pdf"
        ),
        pages=(3, 4, 5, 6),
        figures=("Fig. 1", "Fig. 2", "Fig. 3", "Fig. 4"),
        target_groups=("neutron_detector_response", "uncertainty"),
        status="target_record_started_page_rendered_crop_pending",
        crop_tasks=(
            CropTask(
                figure_id="Fig. 1",
                page=3,
                rect=(45.0, 45.0, 610.0, 300.0),
                required_data="tof_detector_scintillator_pmt_geometry",
            ),
            CropTask(
                figure_id="Fig. 2",
                page=3,
                rect=(300.0, 545.0, 610.0, 765.0),
                required_data="pmt_response_fwhm_and_rise_time_vs_voltage",
            ),
            CropTask(
                figure_id="Fig. 3",
                page=4,
                rect=(45.0, 50.0, 310.0, 288.0),
                required_data="single_neutron_time_response_trace",
            ),
            CropTask(
                figure_id="Fig. 4",
                page=4,
                rect=(45.0, 565.0, 310.0, 760.0),
                required_data="pmt_delay_vs_voltage_curve",
            ),
        ),
    ),
    SourceTask(
        slug="springham-2021-zrbe-activation",
        title="Springham 2021 NX3 Zr/Be activation energy and anisotropy",
        target_function="nx3_springham_zrbe_activation_targets",
        source_md=(
            "KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-"
            "2021-164830-bc8edab3.md"
        ),
        source_pdf=(
            "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
            "springham2021.pdf"
        ),
        pages=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        figures=(
            "Fig. 1",
            "Fig. 2",
            "Fig. 3",
            "Fig. 4",
            "Fig. 5",
            "Fig. 6",
            "Fig. 7",
            "Tables 1-2",
        ),
        target_groups=(
            "neutron_yield",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ),
        status="target_record_started_page_rendered_crop_pending",
        crop_tasks=(
            CropTask(
                figure_id="Fig. 1",
                page=2,
                rect=(35.0, 55.0, 290.0, 470.0),
                required_data="activation_cross_sections_and_dd_neutron_energy_curve",
            ),
            CropTask(
                figure_id="Fig. 2",
                page=4,
                rect=(35.0, 55.0, 560.0, 310.0),
                required_data="nx3_geometry_zrbe_activation_detector_layout",
            ),
            CropTask(
                figure_id="Fig. 3",
                page=4,
                rect=(120.0, 315.0, 475.0, 625.0),
                required_data="pulse_processing_and_data_acquisition_diagram",
            ),
            CropTask(
                figure_id="Fig. 4",
                page=5,
                rect=(35.0, 50.0, 560.0, 295.0),
                required_data="zr_activation_bgo_pha_spectrum_and_sca_window",
            ),
            CropTask(
                figure_id="Fig. 5",
                page=6,
                rect=(35.0, 55.0, 560.0, 335.0),
                required_data="mcnp5_response_effective_energy_vs_zr_be_count_ratio",
            ),
            CropTask(
                figure_id="Table 1",
                page=7,
                rect=(35.0, 55.0, 290.0, 210.0),
                required_data="shot_count_by_deuterium_pressure_and_obstacle_state",
            ),
            CropTask(
                figure_id="Fig. 6",
                page=8,
                rect=(35.0, 55.0, 560.0, 460.0),
                required_data="shot_to_shot_5mbar_yield_anisotropy_and_energy_series",
            ),
            CropTask(
                figure_id="Table 2",
                page=9,
                rect=(120.0, 55.0, 475.0, 245.0),
                required_data="four_mbar_zrbe_activation_summary_table",
            ),
            CropTask(
                figure_id="Fig. 7",
                page=9,
                rect=(0.0, 255.0, 575.0, 685.0),
                required_data="pressure_sweep_yield_energy_and_anisotropy_curves",
            ),
        ),
    ),
    SourceTask(
        slug="catenacci-2020-neutron-tomography",
        title="Catenacci 2020 DPF neutron time-energy tomography",
        target_function="nnss_dpf_neutron_time_energy_tomography_targets",
        source_md=(
            "KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-"
            "energy-spectrum-from-a-dense-plasma-focus-b78f1154.md"
        ),
        source_pdf=(
            "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
            "catenacci2020.pdf"
        ),
        pages=(3, 4, 5, 6, 7),
        figures=("Fig. 1", "Fig. 2", "Fig. 3", "Figs. 4-8", "Tables I-IV"),
        target_groups=("neutron_timing", "neutron_spectrum", "neutron_detector_response", "uncertainty"),
        status="target_record_started_page_rendered_crop_pending",
        crop_tasks=(
            CropTask(
                figure_id="Fig. 1",
                page=3,
                rect=(45.0, 70.0, 545.0, 288.0),
                required_data="shadow_bar_system_configuration",
            ),
            CropTask(
                figure_id="Fig. 2",
                page=4,
                rect=(45.0, 50.0, 570.0, 282.0),
                required_data="fg_sb_shadow_bar_calibration_and_subtraction_pulses",
            ),
            CropTask(
                figure_id="Table I",
                page=4,
                rect=(0.0, 280.0, 310.0, 375.0),
                required_data="shadow_bar_detector_scale_factors",
            ),
            CropTask(
                figure_id="Table II",
                page=4,
                rect=(300.0, 280.0, 585.0, 430.0),
                required_data="shadow_bar_detector_distance_and_angle_table",
            ),
            CropTask(
                figure_id="Fig. 3",
                page=5,
                rect=(45.0, 55.0, 305.0, 390.0),
                required_data="nnss_dpf_tomography_experimental_setup_diagram",
            ),
            CropTask(
                figure_id="Table III",
                page=5,
                rect=(320.0, 55.0, 565.0, 175.0),
                required_data="trial_peak_energy_and_total_neutron_yield_table",
            ),
            CropTask(
                figure_id="Fig. 4",
                page=6,
                rect=(45.0, 50.0, 565.0, 490.0),
                required_data="time_energy_spectrum_reconstruction_trials_1_to_4",
            ),
            CropTask(
                figure_id="Fig. 5",
                page=6,
                rect=(45.0, 510.0, 305.0, 645.0),
                required_data="double_pinch_energy_probability_curves",
            ),
            CropTask(
                figure_id="Fig. 6",
                page=6,
                rect=(305.0, 510.0, 565.0, 680.0),
                required_data="close_range_and_shadow_bar_pinch_profiles",
            ),
            CropTask(
                figure_id="Fig. 7",
                page=7,
                rect=(45.0, 50.0, 305.0, 565.0),
                required_data="scatter_corrected_uncorrected_difference_reconstruction",
            ),
            CropTask(
                figure_id="Fig. 8",
                page=7,
                rect=(310.0, 55.0, 565.0, 240.0),
                required_data="neutron_gamma_pulse_arrival_trace",
            ),
            CropTask(
                figure_id="Table IV",
                page=7,
                rect=(305.0, 245.0, 570.0, 385.0),
                required_data="predicted_vs_reconstructed_max_energy_table",
            ),
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_pages(
    pdf_path: Path,
    *,
    pages: Iterable[int],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(pdf_path) as document:
        for page_number in pages:
            if page_number < 1 or page_number > document.page_count:
                rendered.append(
                    {
                        "page": page_number,
                        "status": "page_out_of_range",
                        "pdf_page_count": document.page_count,
                    }
                )
                continue
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = output_dir / f"page-{page_number:02d}.png"
            pixmap.save(output_path)
            rendered.append(
                {
                    "page": page_number,
                    "status": "rendered_crop_pending",
                    "path": output_path.as_posix(),
                    "sha256": sha256_file(output_path),
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                    "dpi": dpi,
                }
            )
    return rendered


def _safe_filename(value: str) -> str:
    return (
        value.lower()
        .replace(".", "")
        .replace("/", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )


def crop_figures(
    pdf_path: Path,
    *,
    crop_tasks: Iterable[CropTask],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, object]] = []
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(pdf_path) as document:
        for crop_task in crop_tasks:
            if crop_task.page < 1 or crop_task.page > document.page_count:
                crops.append(
                    {
                        "figure_id": crop_task.figure_id,
                        "page": crop_task.page,
                        "status": "page_out_of_range",
                        "pdf_page_count": document.page_count,
                    }
                )
                continue
            page = document.load_page(crop_task.page - 1)
            rect = fitz.Rect(crop_task.rect) & page.rect
            output_path = (
                output_dir
                / f"page-{crop_task.page:02d}-{_safe_filename(crop_task.figure_id)}.png"
            )
            pixmap = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
            pixmap.save(output_path)
            crops.append(
                {
                    "figure_id": crop_task.figure_id,
                    "page": crop_task.page,
                    "status": "crop_candidate_unreviewed",
                    "path": output_path.as_posix(),
                    "sha256": sha256_file(output_path),
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                    "dpi": dpi,
                    "pdf_rect_points": list(crop_task.rect),
                    "required_data": crop_task.required_data,
                    "accepted_for_validation": False,
                    "acceptance_boundary": (
                        "Manual crop candidate only. It is not accepted "
                        "digitization evidence until figure/table extraction "
                        "arrays, residuals, and independent review are recorded."
                    ),
                }
            )
    return crops


def build_report(repo_root: Path, *, render: bool, dpi: int) -> dict[str, object]:
    workbench_root = (
        repo_root / "KnowledgeReference" / "figures" / "target-extraction" / RUN_ID
    )
    tasks: list[dict[str, object]] = []
    for task in SOURCE_TASKS:
        source_md = repo_root / task.source_md
        source_pdf = repo_root / task.source_pdf
        task_record: dict[str, object] = {
            "slug": task.slug,
            "title": task.title,
            "target_function": task.target_function,
            "source_md": task.source_md,
            "source_md_sha256": sha256_file(source_md) if source_md.exists() else None,
            "source_pdf": task.source_pdf,
            "source_pdf_sha256": sha256_file(source_pdf) if source_pdf.exists() else None,
            "pages": list(task.pages),
            "figures": list(task.figures),
            "target_groups": list(task.target_groups),
            "status": task.status,
            "digitization_gate": "digitization_verification_evidence",
            "accepted_for_validation": False,
            "next_actions": [],
        }
        if render and source_pdf.exists():
            task_record["rendered_pages"] = render_pages(
                source_pdf,
                pages=task.pages,
                output_dir=workbench_root / task.slug,
                dpi=dpi,
            )
            task_record["crop_candidates"] = crop_figures(
                source_pdf,
                crop_tasks=task.crop_tasks,
                output_dir=workbench_root / task.slug / "crops",
                dpi=dpi,
            )
        elif render:
            task_record["rendered_pages"] = [
                {
                    "status": "source_pdf_missing",
                    "source_pdf": task.source_pdf,
                }
            ]
            task_record["crop_candidates"] = [
                {
                    "status": "source_pdf_missing",
                    "source_pdf": task.source_pdf,
                }
            ] if task.crop_tasks else []
        else:
            task_record["rendered_pages"] = []
            task_record["crop_candidates"] = []
        if task_record["crop_candidates"]:
            first_action = "review crop boundaries and adjust any missing axes or captions"
        else:
            first_action = "crop cited figures or tables from rendered pages"
        task_record["next_actions"] = [
            first_action,
            "record axis calibration or table structure",
            "extract numeric arrays with units",
            "measure overlay or extraction residuals where applicable",
            "record independent review before validation use",
        ]
        tasks.append(task_record)

    accepted_count = sum(1 for task in tasks if task["accepted_for_validation"])
    rendered_count = sum(
        1
        for task in tasks
        for page in task["rendered_pages"]  # type: ignore[index]
        if page.get("status") == "rendered_crop_pending"
    )
    crop_count = sum(
        1
        for task in tasks
        for crop in task["crop_candidates"]  # type: ignore[index]
        if crop.get("status") == "crop_candidate_unreviewed"
    )
    return {
        "report_id": REPORT_STEM,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_boundary": (
            "Local KnowledgeReference markdown records and hashed local intake "
            "source PDFs only. Rendered pages and crops are workbench artifacts, "
            "not accepted validation evidence."
        ),
        "accepted_for_validation_count": accepted_count,
        "source_count": len(tasks),
        "rendered_page_count": rendered_count,
        "crop_candidate_count": crop_count,
        "tasks": tasks,
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Target Extraction and Digitization Start",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "This file starts target extraction and digitization work only. The source of",
        "truth remains local `KnowledgeReference/` files. Rendered pages are",
        "workbench artifacts and are not accepted validation evidence.",
        "",
        "## Summary",
        "",
        f"- Sources started: `{report['source_count']}`",
        f"- Rendered pages: `{report['rendered_page_count']}`",
        f"- Crop candidates: `{report['crop_candidate_count']}`",
        f"- Accepted validation packets: `{report['accepted_for_validation_count']}`",
        "",
        "## Source Tasks",
        "",
    ]
    for task in report["tasks"]:  # type: ignore[index]
        lines.extend(
            [
                f"### {task['title']}",
                "",
                f"- Target function: `{task['target_function']}`",
                f"- Source markdown: `{task['source_md']}`",
                f"- Source markdown SHA-256: `{task['source_md_sha256']}`",
                f"- Source PDF: `{task['source_pdf']}`",
                f"- Source PDF SHA-256: `{task['source_pdf_sha256']}`",
                f"- Status: `{task['status']}`",
                f"- Accepted for validation: `{task['accepted_for_validation']}`",
                f"- Target groups: `{', '.join(task['target_groups'])}`",
                f"- Figures/tables queued: `{', '.join(task['figures'])}`",
                "",
                "Rendered workbench pages:",
                "",
            ]
        )
        rendered_pages = task["rendered_pages"]
        if not rendered_pages:
            lines.append("- None")
        else:
            for page in rendered_pages:  # type: ignore[assignment]
                if page.get("status") == "rendered_crop_pending":
                    lines.append(
                        "- Page {page}: `{path}` SHA-256 `{sha}`".format(
                            page=page["page"],
                            path=page["path"],
                            sha=page["sha256"],
                        )
                    )
                else:
                    lines.append(f"- `{page}`")
        lines.extend(
            [
                "",
                "Crop candidates:",
                "",
            ]
        )
        crop_candidates = task["crop_candidates"]
        if not crop_candidates:
            lines.append("- None")
        else:
            for crop in crop_candidates:  # type: ignore[assignment]
                if crop.get("status") == "crop_candidate_unreviewed":
                    lines.append(
                        (
                            "- {figure}, page {page}: `{path}` SHA-256 `{sha}`; "
                            "status `{status}`; accepted_for_validation "
                            "`{accepted}`; required data `{required_data}`; "
                            "size `{width}x{height}` px"
                        ).format(
                            figure=crop["figure_id"],
                            page=crop["page"],
                            path=crop["path"],
                            sha=crop["sha256"],
                            status=crop["status"],
                            accepted=crop["accepted_for_validation"],
                            required_data=crop["required_data"],
                            width=crop["width_px"],
                            height=crop["height_px"],
                        )
                    )
                else:
                    lines.append(f"- `{crop}`")
        lines.extend(
            [
                "",
                "Next actions:",
                "",
            ]
        )
        for action in task["next_actions"]:
            lines.append(f"- {action}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--dpi", type=int, default=250, help="Render DPI")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Write the report without rendering page images",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report = build_report(repo_root, render=not args.no_render, dpi=args.dpi)
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    json_path = docs_dir / f"{REPORT_STEM}.json"
    md_path = docs_dir / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown_report(report))
    print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
