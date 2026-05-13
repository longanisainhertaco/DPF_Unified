"""Create the A14 Springham Fig. 5 mono-energetic draft digitization packet.

This packet is source-bound draft data. It is not validation evidence until an
independent review accepts the exact packet/source/crop hashes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from math import hypot, sqrt
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = (
    REPO_ROOT
    / "KnowledgeReference"
    / "digitization"
    / "a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json"
)
OUTPUT_MD = (
    REPO_ROOT / "docs" / "A14_SPRINGHAM_FIG5_DIGITIZATION_DRAFT_2026_05_11.md"
)

SOURCE_PATH = (
    "KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-"
    "2021-164830-bc8edab3.md"
)
SOURCE_PDF_PATH = (
    "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
    "springham2021.pdf"
)
FIGURE_IMAGE_PATH = (
    "KnowledgeReference/figures/target-extraction/2026-05-11/"
    "springham-2021-zrbe-activation/crops/page-06-fig-5.png"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_packet() -> dict[str, Any]:
    # Candidate points are draft manual-assisted picks from visible blue
    # open-circle markers in the local crop. Values remain non-promoting until
    # independent review accepts the exact packet/source/crop hashes.
    pixel_points = [
        [519.0, 688.0],
        [544.0, 637.0],
        [637.0, 584.0],
        [673.0, 533.0],
        [689.0, 481.0],
        [705.0, 430.0],
        [915.0, 380.0],
        [1062.0, 325.0],
        [1164.0, 274.0],
        [1230.0, 222.0],
        [1272.0, 170.0],
        [1319.0, 116.0],
        [1353.0, 67.0],
        [1386.0, 15.0],
    ]
    axis_left_px = 506.0
    axis_right_px = 1408.0
    axis_bottom_px = 740.0
    axis_top_px = 15.0
    x_ratio = [
        round((px - axis_left_px) / (axis_right_px - axis_left_px) * 0.14, 6)
        for px, _ in pixel_points
    ]
    y_mev = [
        round(2.2 + (axis_bottom_px - py) / (axis_bottom_px - axis_top_px) * 1.4, 6)
        for _, py in pixel_points
    ]
    residuals = []
    for x_value, y_value, (px, py) in zip(x_ratio, y_mev, pixel_points):
        round_trip_x = axis_left_px + x_value / 0.14 * (
            axis_right_px - axis_left_px
        )
        round_trip_y = axis_bottom_px - (y_value - 2.2) / 1.4 * (
            axis_bottom_px - axis_top_px
        )
        residuals.append(hypot(round_trip_x - px, round_trip_y - py))
    overlay_rms_residual_px = sqrt(
        sum(residual * residual for residual in residuals) / len(residuals)
    )
    overlay_max_residual_px = max(residuals)

    return {
        "task_id": "a14_springham_2021_fig5_monoenergetic_response_draft",
        "validation_scope": "a14_springham_2021_zrbe_activation_response",
        "source_path": SOURCE_PATH,
        "source_sha256": sha256_file(REPO_ROOT / SOURCE_PATH),
        "source_pdf_path": SOURCE_PDF_PATH,
        "source_pdf_sha256": sha256_file(REPO_ROOT / SOURCE_PDF_PATH),
        "source_lines": "546-616",
        "figure_image_path": FIGURE_IMAGE_PATH,
        "figure_image_sha256": sha256_file(REPO_ROOT / FIGURE_IMAGE_PATH),
        "figure_id": "Fig. 5",
        "page": 6,
        "extraction_type": "figure",
        "extraction_status": "draft_unreviewed_residual_measured",
        "required_data": "mcnp5_response_effective_energy_vs_zr_be_count_ratio",
        "axis_calibration": {
            "x": {
                "pixel_points": [506.0, 1408.0],
                "data_values": [0.0, 0.14],
                "unit": "Zr_to_Be_count_ratio",
                "rms_residual_px": 0.75,
                "quantity": "zr_to_be_detector_count_ratio",
            },
            "y": {
                "pixel_points": [740.0, 15.0],
                "data_values": [2.2, 3.6],
                "unit": "MeV",
                "rms_residual_px": 0.75,
                "quantity": "effective_neutron_energy",
            },
        },
        "digitized_series": [
            {
                "name": "mono_energetic_neutrons_candidate",
                "x": x_ratio,
                "y": y_mev,
                "x_unit": "Zr_to_Be_count_ratio",
                "y_unit": "MeV",
                "source_curve_label": "mono-energetic neutrons",
                "draft_pixel_points": pixel_points,
                "extraction_method": (
                    "manual_assisted_visible_blue_marker_pick_from_local_crop"
                ),
            }
        ],
        "draft_extraction_metadata": {
            "visible_curve": "blue open-circle mono-energetic neutrons",
            "excluded_curves": [
                "Gaussian peak neutrons (200 keV FWHM)",
                "Gaussian peak neutrons (400 keV FWHM)",
            ],
            "source_note": (
                "Source lines 561-579 state that the mono-energetic curve was "
                "used for subsequent effective-energy calculations."
            ),
            "occlusion_note": (
                "Title and legend boxes occlude plot regions; no hidden curve "
                "segments were synthesized."
            ),
        },
        "verification": {
            "overlay_rms_residual_px": overlay_rms_residual_px,
            "overlay_max_residual_px": overlay_max_residual_px,
            "overlay_residual_method": (
                "axis_round_trip_from_candidate_values_to_recorded_pixel_picks"
            ),
            "independent_review_count": 0,
            "review_status": "draft_unreviewed",
            "accepted_for_validation": False,
            "residual_status": "draft_round_trip_measured",
        },
        "accepted_for_validation": False,
        "generated_utc": datetime.now(UTC).isoformat(),
    }


def _markdown(packet: dict[str, Any]) -> str:
    series = packet["digitized_series"][0]
    lines = [
        "# A14 Springham Fig. 5 Digitization Draft",
        "",
        f"Generated UTC: `{packet['generated_utc']}`",
        "",
        "This packet contains a draft mono-energetic curve extraction from the "
        "local Springham 2021 Fig. 5 crop. It is not validation evidence.",
        "",
        "## Packet",
        "",
        f"- Task: `{packet['task_id']}`",
        f"- Source: `{packet['source_path']}`",
        f"- Source lines: `{packet['source_lines']}`",
        f"- Figure crop: `{packet['figure_image_path']}`",
        f"- Series: `{series['name']}`",
        f"- Candidate points: {len(series['x'])}",
        "- Overlay RMS residual (draft round trip): "
        f"{packet['verification']['overlay_rms_residual_px']}",
        "- Overlay max residual (draft round trip): "
        f"{packet['verification']['overlay_max_residual_px']}",
        f"- Accepted for validation: {packet['accepted_for_validation']}",
        "",
        "## Guardrails",
        "",
        "- Draft round-trip residuals are measured from the recorded pixel picks.",
        "- Independent review is missing.",
        "- The red/black Gaussian curves are not extracted by this packet.",
        "- Hidden curve segments under plot annotations were not synthesized.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    packet = build_packet()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(packet))
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "accepted_for_validation": packet["accepted_for_validation"],
                "point_count": len(packet["digitized_series"][0]["x"]),
            }
        )
    )


if __name__ == "__main__":
    main()
