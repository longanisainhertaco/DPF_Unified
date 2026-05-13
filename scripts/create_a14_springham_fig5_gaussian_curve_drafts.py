"""Create A14 Springham Fig. 5 Gaussian-curve draft digitization packets."""

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
    / "a14-2026-05-11-springham-fig5-gaussian-curves-draft-packet.json"
)
OUTPUT_MD = (
    REPO_ROOT / "docs" / "A14_SPRINGHAM_FIG5_GAUSSIAN_CURVES_DRAFT_2026_05_11.md"
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

AXIS_LEFT_PX = 506.0
AXIS_RIGHT_PX = 1408.0
AXIS_BOTTOM_PX = 740.0
AXIS_TOP_PX = 15.0
X_MIN = 0.0
X_MAX = 0.14
Y_MIN = 2.2
Y_MAX = 3.6


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_to_data(pixel_points: list[list[float]]) -> tuple[list[float], list[float]]:
    x_values = [
        round(
            X_MIN
            + (px - AXIS_LEFT_PX) / (AXIS_RIGHT_PX - AXIS_LEFT_PX) * (X_MAX - X_MIN),
            6,
        )
        for px, _ in pixel_points
    ]
    y_values = [
        round(
            Y_MIN
            + (AXIS_BOTTOM_PX - py)
            / (AXIS_BOTTOM_PX - AXIS_TOP_PX)
            * (Y_MAX - Y_MIN),
            6,
        )
        for _, py in pixel_points
    ]
    return x_values, y_values


def _round_trip_residuals(
    x_values: list[float],
    y_values: list[float],
    pixel_points: list[list[float]],
) -> list[float]:
    residuals = []
    for x_value, y_value, (px, py) in zip(x_values, y_values, pixel_points):
        round_trip_x = AXIS_LEFT_PX + (x_value - X_MIN) / (X_MAX - X_MIN) * (
            AXIS_RIGHT_PX - AXIS_LEFT_PX
        )
        round_trip_y = AXIS_BOTTOM_PX - (y_value - Y_MIN) / (Y_MAX - Y_MIN) * (
            AXIS_BOTTOM_PX - AXIS_TOP_PX
        )
        residuals.append(hypot(round_trip_x - px, round_trip_y - py))
    return residuals


def _series(
    *,
    name: str,
    source_curve_label: str,
    pixel_points: list[list[float]],
) -> dict[str, Any]:
    x_values, y_values = _pixel_to_data(pixel_points)
    residuals = _round_trip_residuals(x_values, y_values, pixel_points)
    return {
        "name": name,
        "source_curve_label": source_curve_label,
        "x": x_values,
        "y": y_values,
        "x_unit": "Zr_to_Be_count_ratio",
        "y_unit": "MeV",
        "draft_pixel_points": pixel_points,
        "extraction_method": "manual_assisted_visible_curve_pick_from_local_crop",
        "overlay_rms_residual_px": sqrt(
            sum(residual * residual for residual in residuals) / len(residuals)
        ),
        "overlay_max_residual_px": max(residuals),
    }


def build_packet() -> dict[str, Any]:
    red_pixel_points = [
        [545.0, 716.0],
        [585.0, 643.0],
        [635.0, 576.0],
        [705.0, 502.0],
        [780.0, 443.0],
        [860.0, 398.0],
        [940.0, 358.0],
        [1020.0, 320.0],
        [1100.0, 280.0],
        [1180.0, 231.0],
        [1260.0, 170.0],
        [1330.0, 91.0],
        [1388.0, 20.0],
    ]
    black_pixel_points = [
        [545.0, 668.0],
        [585.0, 621.0],
        [635.0, 571.0],
        [690.0, 487.0],
        [750.0, 427.0],
        [820.0, 399.0],
        [900.0, 370.0],
        [980.0, 344.0],
        [1060.0, 314.0],
        [1140.0, 278.0],
        [1220.0, 224.0],
        [1300.0, 139.0],
        [1368.0, 46.0],
    ]
    digitized_series = [
        _series(
            name="gaussian_peak_200kev_fwhm_candidate",
            source_curve_label="Gaussian peak neutrons (200 keV FWHM)",
            pixel_points=black_pixel_points,
        ),
        _series(
            name="gaussian_peak_400kev_fwhm_candidate",
            source_curve_label="Gaussian peak neutrons (400 keV FWHM)",
            pixel_points=red_pixel_points,
        ),
    ]
    all_residuals = [
        residual
        for series in digitized_series
        for residual in _round_trip_residuals(
            series["x"], series["y"], series["draft_pixel_points"]
        )
    ]
    return {
        "task_id": "a14_springham_2021_fig5_gaussian_curves_draft",
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
        "required_data": "mcnp5_gaussian_response_effective_energy_vs_zr_be_count_ratio",
        "axis_calibration": {
            "x": {
                "pixel_points": [AXIS_LEFT_PX, AXIS_RIGHT_PX],
                "data_values": [X_MIN, X_MAX],
                "unit": "Zr_to_Be_count_ratio",
                "rms_residual_px": 0.75,
                "quantity": "zr_to_be_detector_count_ratio",
            },
            "y": {
                "pixel_points": [AXIS_BOTTOM_PX, AXIS_TOP_PX],
                "data_values": [Y_MIN, Y_MAX],
                "unit": "MeV",
                "rms_residual_px": 0.75,
                "quantity": "effective_neutron_energy",
            },
        },
        "digitized_series": digitized_series,
        "draft_extraction_metadata": {
            "visible_curves": [
                "black curve: Gaussian peak neutrons (200 keV FWHM)",
                "red curve: Gaussian peak neutrons (400 keV FWHM)",
            ],
            "excluded_curve": "blue open-circle mono-energetic neutrons",
            "source_note": (
                "Source lines 561-579 state that Fig. 5 shows mono-energetic, "
                "200 keV FWHM, and 400 keV FWHM MCNP5 response curves."
            ),
            "occlusion_note": (
                "Sampled points are restricted to visible curve segments; no "
                "hidden curve segments under annotations were synthesized."
            ),
        },
        "verification": {
            "overlay_rms_residual_px": sqrt(
                sum(residual * residual for residual in all_residuals)
                / len(all_residuals)
            ),
            "overlay_max_residual_px": max(all_residuals),
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
    lines = [
        "# A14 Springham Fig. 5 Gaussian-Curve Digitization Draft",
        "",
        f"Generated UTC: `{packet['generated_utc']}`",
        "",
        "This packet contains draft Gaussian-curve extractions from the local "
        "Springham 2021 Fig. 5 crop. It is not validation evidence.",
        "",
        "## Packet",
        "",
        f"- Task: `{packet['task_id']}`",
        f"- Source: `{packet['source_path']}`",
        f"- Source lines: `{packet['source_lines']}`",
        f"- Figure crop: `{packet['figure_image_path']}`",
        f"- Series count: {len(packet['digitized_series'])}",
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
        "- The blue mono-energetic curve is handled by a separate packet.",
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
                "series_count": len(packet["digitized_series"]),
            }
        )
    )


if __name__ == "__main__":
    main()
