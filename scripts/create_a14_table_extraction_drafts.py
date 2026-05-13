#!/usr/bin/env python3
"""Create draft table-extraction packets for the 2026-05-11 A14 lane.

These packets are source-bound transcription drafts.  They are intentionally
not accepted validation evidence until independent review metadata is added and
the generic digitization verification gate passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = (
    REPO_ROOT
    / "KnowledgeReference"
    / "digitization"
    / "a14-2026-05-11-table-draft-packets.json"
)
OUTPUT_MD = REPO_ROOT / "docs" / "A14_TABLE_EXTRACTION_DRAFTS_2026_05_11.md"


@dataclass(frozen=True)
class TableDraft:
    task_id: str
    validation_scope: str
    source_path: str
    source_pdf_path: str
    source_lines: str
    crop_image_path: str
    table_id: str
    page: int
    required_data: str
    table_rows: list[dict[str, Any]]
    digitized_series: list[dict[str, Any]]
    notes: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _springham_source() -> tuple[str, str]:
    return (
        "KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-"
        "2021-164830-bc8edab3.md",
        "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
        "springham2021.pdf",
    )


def _catenacci_source() -> tuple[str, str]:
    return (
        "KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-"
        "energy-spectrum-from-a-dense-plasma-focus-b78f1154.md",
        "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
        "catenacci2020.pdf",
    )


def _series(
    name: str,
    x: list[float],
    y: list[float],
    *,
    x_unit: str,
    y_unit: str,
    note: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "x": x,
        "y": y,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "note": note,
    }


def table_drafts() -> list[TableDraft]:
    springham_md, springham_pdf = _springham_source()
    catenacci_md, catenacci_pdf = _catenacci_source()
    springham_table1_rows = [
        {
            "d2_pressure_mbar": 1.5,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": None,
            "op_shots": None,
        },
        {
            "d2_pressure_mbar": 2.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": 2,
            "op_shots": 32,
        },
        {
            "d2_pressure_mbar": 3.0,
            "w_o_op_series": 3,
            "w_o_op_shots": 48,
            "op_series": 3,
            "op_shots": 48,
        },
        {
            "d2_pressure_mbar": 4.0,
            "w_o_op_series": 4,
            "w_o_op_shots": 64,
            "op_series": 3,
            "op_shots": 48,
        },
        {
            "d2_pressure_mbar": 5.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 28,
            "op_series": 2,
            "op_shots": 32,
        },
        {
            "d2_pressure_mbar": 6.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": 3,
            "op_shots": 48,
        },
        {
            "d2_pressure_mbar": 7.0,
            "w_o_op_series": 4,
            "w_o_op_shots": 56,
            "op_series": None,
            "op_shots": None,
        },
        {
            "d2_pressure_mbar": 8.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": None,
            "op_shots": None,
        },
        {
            "d2_pressure_mbar": 9.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": None,
            "op_shots": None,
        },
        {
            "d2_pressure_mbar": 10.0,
            "w_o_op_series": 2,
            "w_o_op_shots": 32,
            "op_series": None,
            "op_shots": None,
        },
    ]
    springham_table2_rows = [
        {
            "condition": "w_o_op_0deg",
            "shot_count": 64,
            "mean_corrected_zr_count": 2634.0,
            "mean_corrected_be_count": 75404.0,
            "mean_psa_yield": 1.37e9,
            "max_psa_yield": 4.05e9,
            "effective_neutron_energy_mev": 2.82,
            "group_delta_effective_energy_mev": 0.33,
            "group_anisotropy_anbe": 3.06,
        },
        {
            "condition": "w_o_op_90deg",
            "shot_count": 64,
            "mean_corrected_zr_count": 300.3,
            "mean_corrected_be_count": 17090.0,
            "mean_psa_yield": 0.449e9,
            "max_psa_yield": 1.18e9,
            "effective_neutron_energy_mev": 2.49,
            "group_delta_effective_energy_mev": 0.33,
            "group_anisotropy_anbe": 3.06,
        },
        {
            "condition": "op_0deg",
            "shot_count": 48,
            "mean_corrected_zr_count": 518.1,
            "mean_corrected_be_count": 12870.0,
            "mean_psa_yield": 2.34e8,
            "max_psa_yield": 9.26e8,
            "effective_neutron_energy_mev": 2.83,
            "group_delta_effective_energy_mev": 0.28,
            "group_anisotropy_anbe": 1.49,
        },
        {
            "condition": "op_90deg",
            "shot_count": 48,
            "mean_corrected_zr_count": 134.4,
            "mean_corrected_be_count": 6312.0,
            "mean_psa_yield": 1.57e8,
            "max_psa_yield": 5.31e8,
            "effective_neutron_energy_mev": 2.55,
            "group_delta_effective_energy_mev": 0.28,
            "group_anisotropy_anbe": 1.49,
        },
    ]
    catenacci_table1_rows = [
        {"detector": "SB - 10 m", "scale_factor": 1.13},
        {"detector": "SB - 14 m", "scale_factor": 0.84},
        {"detector": "SB - 18 m", "scale_factor": 0.98},
        {"detector": "SB - 22 m", "scale_factor": 1.07},
    ]
    catenacci_table2_rows = [
        {"detector": "FG - 10 m", "distance_to_pinch_m": 10.29, "angle_deg": 72.19},
        {"detector": "SB - 10 m", "distance_to_pinch_m": 10.29, "angle_deg": 76.06},
        {"detector": "FG - 14 m", "distance_to_pinch_m": 14.24, "angle_deg": 71.63},
        {"detector": "SB - 14 m", "distance_to_pinch_m": 14.29, "angle_deg": 70.56},
        {"detector": "FG - 18 m", "distance_to_pinch_m": 18.27, "angle_deg": 67.52},
        {"detector": "SB - 18 m", "distance_to_pinch_m": 18.30, "angle_deg": 66.52},
        {"detector": "FG - 22 m", "distance_to_pinch_m": 22.25, "angle_deg": 63.76},
        {"detector": "SB - 22 m", "distance_to_pinch_m": 22.27, "angle_deg": 64.77},
    ]
    catenacci_table3_rows = [
        {
            "trial": 1,
            "main_pinch_energy_mev": 2.42,
            "second_pinch_energy_mev": None,
            "neutron_yield_1e10": 3.72,
        },
        {
            "trial": 2,
            "main_pinch_energy_mev": 2.47,
            "second_pinch_energy_mev": None,
            "neutron_yield_1e10": 4.28,
        },
        {
            "trial": 3,
            "main_pinch_energy_mev": 2.41,
            "second_pinch_energy_mev": 2.59,
            "neutron_yield_1e10": 4.38,
        },
        {
            "trial": 4,
            "main_pinch_energy_mev": 2.48,
            "second_pinch_energy_mev": 2.48,
            "neutron_yield_1e10": 4.60,
        },
    ]
    catenacci_table4_rows = [
        {
            "trial": 1,
            "predicted_max_energy_mev": 2.90,
            "reconstructed_max_energy_mev": 2.89,
        },
        {
            "trial": 2,
            "predicted_max_energy_mev": 3.00,
            "reconstructed_max_energy_mev": 3.04,
        },
        {
            "trial": 3,
            "predicted_max_energy_mev": 3.01,
            "reconstructed_max_energy_mev": 3.04,
        },
        {
            "trial": 4,
            "predicted_max_energy_mev": 3.00,
            "reconstructed_max_energy_mev": 3.01,
        },
    ]

    return [
        TableDraft(
            task_id="a14_springham_2021_table1_shot_counts",
            validation_scope="nx3_springham_2021_zrbe_activation",
            source_path=springham_md,
            source_pdf_path=springham_pdf,
            source_lines="709-724",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "springham-2021-zrbe-activation/crops/page-07-table-1.png"
            ),
            table_id="Table 1",
            page=7,
            required_data="shot_count_by_deuterium_pressure_and_obstacle_state",
            table_rows=springham_table1_rows,
            digitized_series=[
                _series(
                    "w_o_op_shot_count_by_pressure",
                    [row["d2_pressure_mbar"] for row in springham_table1_rows],
                    [row["w_o_op_shots"] for row in springham_table1_rows],
                    x_unit="mbar",
                    y_unit="shots",
                    note="Obstacle-free shot counts by D2 pressure.",
                ),
                _series(
                    "op_shot_count_by_pressure",
                    [
                        row["d2_pressure_mbar"]
                        for row in springham_table1_rows
                        if row["op_shots"] is not None
                    ],
                    [
                        row["op_shots"]
                        for row in springham_table1_rows
                        if row["op_shots"] is not None
                    ],
                    x_unit="mbar",
                    y_unit="shots",
                    note="Obstacle-plate shot counts where a table entry exists.",
                ),
            ],
            notes="Dash entries are stored as null in table_rows and omitted from numeric series.",
        ),
        TableDraft(
            task_id="a14_springham_2021_table2_four_mbar_activation_summary",
            validation_scope="nx3_springham_2021_zrbe_activation",
            source_path=springham_md,
            source_pdf_path=springham_pdf,
            source_lines="1047-1054",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "springham-2021-zrbe-activation/crops/page-09-table-2.png"
            ),
            table_id="Table 2",
            page=9,
            required_data="four_mbar_zrbe_activation_summary_table",
            table_rows=springham_table2_rows,
            digitized_series=[
                _series(
                    "mean_corrected_zr_count_by_condition",
                    [0.0, 1.0, 2.0, 3.0],
                    [row["mean_corrected_zr_count"] for row in springham_table2_rows],
                    x_unit="condition_index",
                    y_unit="count",
                    note="Condition order is w_o_op_0deg, w_o_op_90deg, op_0deg, op_90deg.",
                ),
                _series(
                    "mean_corrected_be_count_by_condition",
                    [0.0, 1.0, 2.0, 3.0],
                    [row["mean_corrected_be_count"] for row in springham_table2_rows],
                    x_unit="condition_index",
                    y_unit="count",
                    note="Condition order is w_o_op_0deg, w_o_op_90deg, op_0deg, op_90deg.",
                ),
                _series(
                    "effective_neutron_energy_by_condition",
                    [0.0, 1.0, 2.0, 3.0],
                    [
                        row["effective_neutron_energy_mev"]
                        for row in springham_table2_rows
                    ],
                    x_unit="condition_index",
                    y_unit="MeV",
                    note="Condition order is w_o_op_0deg, w_o_op_90deg, op_0deg, op_90deg.",
                ),
            ],
            notes="Grouped delta-E and anisotropy values are retained in table_rows.",
        ),
        TableDraft(
            task_id="a14_catenacci_2020_table_i_shadow_bar_scale_factors",
            validation_scope="nnss_catenacci_2020_neutron_time_energy_tomography",
            source_path=catenacci_md,
            source_pdf_path=catenacci_pdf,
            source_lines="448-456",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "catenacci-2020-neutron-tomography/crops/page-04-table-i.png"
            ),
            table_id="Table I",
            page=4,
            required_data="shadow_bar_detector_scale_factors",
            table_rows=catenacci_table1_rows,
            digitized_series=[
                _series(
                    "shadow_bar_scale_factor_by_nominal_distance",
                    [10.0, 14.0, 18.0, 22.0],
                    [row["scale_factor"] for row in catenacci_table1_rows],
                    x_unit="m",
                    y_unit="scale_factor",
                    note="Nominal detector distance parsed from detector label.",
                ),
            ],
            notes="Scale factor values are dimensionless.",
        ),
        TableDraft(
            task_id="a14_catenacci_2020_table_ii_detector_positions",
            validation_scope="nnss_catenacci_2020_neutron_time_energy_tomography",
            source_path=catenacci_md,
            source_pdf_path=catenacci_pdf,
            source_lines="481-489",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "catenacci-2020-neutron-tomography/crops/page-04-table-ii.png"
            ),
            table_id="Table II",
            page=4,
            required_data="shadow_bar_detector_distance_and_angle_table",
            table_rows=catenacci_table2_rows,
            digitized_series=[
                _series(
                    "distance_to_pinch_by_detector_index",
                    list(range(len(catenacci_table2_rows))),
                    [row["distance_to_pinch_m"] for row in catenacci_table2_rows],
                    x_unit="detector_index",
                    y_unit="m",
                    note="Detector labels are preserved in table_rows.",
                ),
                _series(
                    "angle_from_axial_by_detector_index",
                    list(range(len(catenacci_table2_rows))),
                    [row["angle_deg"] for row in catenacci_table2_rows],
                    x_unit="detector_index",
                    y_unit="deg",
                    note="Detector labels are preserved in table_rows.",
                ),
            ],
            notes="Angles are measured from the axial direction.",
        ),
        TableDraft(
            task_id="a14_catenacci_2020_table_iii_peak_energy_yield",
            validation_scope="nnss_catenacci_2020_neutron_time_energy_tomography",
            source_path=catenacci_md,
            source_pdf_path=catenacci_pdf,
            source_lines="549-557",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "catenacci-2020-neutron-tomography/crops/page-05-table-iii.png"
            ),
            table_id="Table III",
            page=5,
            required_data="trial_peak_energy_and_total_neutron_yield_table",
            table_rows=catenacci_table3_rows,
            digitized_series=[
                _series(
                    "main_pinch_peak_energy_by_trial",
                    [1.0, 2.0, 3.0, 4.0],
                    [row["main_pinch_energy_mev"] for row in catenacci_table3_rows],
                    x_unit="trial",
                    y_unit="MeV",
                    note="Main-pinch peak energy for each trial.",
                ),
                _series(
                    "neutron_yield_by_trial",
                    [1.0, 2.0, 3.0, 4.0],
                    [row["neutron_yield_1e10"] for row in catenacci_table3_rows],
                    x_unit="trial",
                    y_unit="1e10_neutrons",
                    note="Table label gives total neutron yield in 4pi.",
                ),
            ],
            notes="Dash entries for second pinch are stored as null in table_rows.",
        ),
        TableDraft(
            task_id="a14_catenacci_2020_table_iv_max_energy_comparison",
            validation_scope="nnss_catenacci_2020_neutron_time_energy_tomography",
            source_path=catenacci_md,
            source_pdf_path=catenacci_pdf,
            source_lines="668-676",
            crop_image_path=(
                "KnowledgeReference/figures/target-extraction/2026-05-11/"
                "catenacci-2020-neutron-tomography/crops/page-07-table-iv.png"
            ),
            table_id="Table IV",
            page=7,
            required_data="predicted_vs_reconstructed_max_energy_table",
            table_rows=catenacci_table4_rows,
            digitized_series=[
                _series(
                    "predicted_max_energy_by_trial",
                    [1.0, 2.0, 3.0, 4.0],
                    [row["predicted_max_energy_mev"] for row in catenacci_table4_rows],
                    x_unit="trial",
                    y_unit="MeV",
                    note="Maximum energy predicted from Eq. 9.",
                ),
                _series(
                    "reconstructed_max_energy_by_trial",
                    [1.0, 2.0, 3.0, 4.0],
                    [
                        row["reconstructed_max_energy_mev"]
                        for row in catenacci_table4_rows
                    ],
                    x_unit="trial",
                    y_unit="MeV",
                    note="Tomographically reconstructed maximum energy.",
                ),
            ],
            notes="This table is a cross-check of maximum energy, not a full spectrum extraction.",
        ),
    ]


def build_packet(draft: TableDraft) -> dict[str, Any]:
    source = REPO_ROOT / draft.source_path
    pdf = REPO_ROOT / draft.source_pdf_path
    crop = REPO_ROOT / draft.crop_image_path
    return {
        "task_id": draft.task_id,
        "validation_scope": draft.validation_scope,
        "source_path": draft.source_path,
        "source_sha256": sha256_file(source),
        "source_pdf_path": draft.source_pdf_path,
        "source_pdf_sha256": sha256_file(pdf),
        "source_lines": draft.source_lines,
        "crop_image_path": draft.crop_image_path,
        "crop_image_sha256": sha256_file(crop),
        "table_id": draft.table_id,
        "page": draft.page,
        "extraction_type": "table",
        "extraction_status": "draft_unreviewed",
        "required_data": draft.required_data,
        "table_rows": draft.table_rows,
        "digitized_series": draft.digitized_series,
        "transcription_basis": (
            "Values were transcribed from the local crop image generated from "
            "the hashed local PDF and cross-referenced against local KR text "
            "line ranges where text extraction preserved captions."
        ),
        "notes": draft.notes,
        "verification": {
            "independent_review_count": 0,
            "review_status": "draft_unreviewed",
            "accepted_for_validation": False,
            "review_required": True,
        },
    }


def build_payload() -> dict[str, Any]:
    packets = [build_packet(draft) for draft in table_drafts()]
    return {
        "model_role": "a14_table_extraction_draft_packets",
        "created_date": "2026-05-11",
        "source_of_truth_rule": (
            "Only local KnowledgeReference markdown and hashed local PDFs/crops "
            "are used. These drafts are not accepted validation evidence."
        ),
        "packet_count": len(packets),
        "accepted_for_validation_count": 0,
        "packets": packets,
        "next_required_steps": [
            "independent table transcription review",
            "review metadata bound to current packet/source/crop hashes",
            "digitization_verification_evidence pass",
            "mapping into same-scope validation targets only after acceptance",
        ],
    }


def write_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A14 Table Extraction Drafts",
        "",
        "Source guardrail: these are draft table transcriptions from local KR",
        "sources and hashed local crop images. They are not accepted validation",
        "evidence until independent review metadata is added and",
        "`digitization_verification_evidence()` passes.",
        "",
        f"- Packet count: `{payload['packet_count']}`",
        f"- Accepted for validation: `{payload['accepted_for_validation_count']}`",
        "",
        "## Draft Packets",
        "",
    ]
    for packet in payload["packets"]:
        lines.extend(
            [
                f"### {packet['task_id']}",
                "",
                f"- Scope: `{packet['validation_scope']}`",
                f"- Source: `{packet['source_path']}`",
                f"- Source lines: `{packet['source_lines']}`",
                f"- Source SHA-256: `{packet['source_sha256']}`",
                f"- Local PDF SHA-256: `{packet['source_pdf_sha256']}`",
                f"- Crop: `{packet['crop_image_path']}`",
                f"- Crop SHA-256: `{packet['crop_image_sha256']}`",
                f"- Table/page: `{packet['table_id']}`, page `{packet['page']}`",
                f"- Required data: `{packet['required_data']}`",
                f"- Extraction status: `{packet['extraction_status']}`",
                f"- Series count: `{len(packet['digitized_series'])}`",
                f"- Row count: `{len(packet['table_rows'])}`",
                f"- Verification status: `{packet['verification']['review_status']}`",
                "",
            ]
        )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    OUTPUT_MD.write_text(text)
    return text


def main() -> None:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(payload)
    print(json.dumps({"json": OUTPUT_JSON.as_posix(), "markdown": OUTPUT_MD.as_posix()}))


if __name__ == "__main__":
    main()
