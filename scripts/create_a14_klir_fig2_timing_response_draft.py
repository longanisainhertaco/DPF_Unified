"""Create the A14 Klir Fig. 2 timing-response draft digitization packet."""

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
    / "a14-2026-05-11-klir-fig2-timing-response-draft-packet.json"
)
OUTPUT_MD = (
    REPO_ROOT / "docs" / "A14_KLIR_FIG2_TIMING_RESPONSE_DRAFT_2026_05_11.md"
)

SOURCE_PATH = (
    "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-measurements-"
    "in-z-pinch-and-plasma-focus-214fbdae.md"
)
SOURCE_PDF_PATH = (
    "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/"
    "klir2011.pdf"
)
FIGURE_IMAGE_PATH = (
    "KnowledgeReference/figures/target-extraction/2026-05-11/"
    "klir-2011-tof-detector/crops/page-03-fig-2.png"
)

AXIS_LEFT_PX = 255.0
AXIS_RIGHT_PX = 771.0
AXIS_BOTTOM_PX = 539.0
AXIS_TOP_PX = 102.0
X_MIN = 1.0
X_MAX = 2.4
Y_MIN = 0.0
Y_MAX = 5.0
X_VALUES_KV = [1.0, 1.1, 1.2, 1.4, 1.6, 1.9, 2.2, 2.4]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_y_to_ns(pixel_points: list[list[float]]) -> list[float]:
    return [
        round(
            Y_MIN
            + (AXIS_BOTTOM_PX - py)
            / (AXIS_BOTTOM_PX - AXIS_TOP_PX)
            * (Y_MAX - Y_MIN),
            6,
        )
        for _, py in pixel_points
    ]


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
    y_values = _pixel_y_to_ns(pixel_points)
    residuals = _round_trip_residuals(X_VALUES_KV, y_values, pixel_points)
    return {
        "name": name,
        "source_curve_label": source_curve_label,
        "x": X_VALUES_KV,
        "y": y_values,
        "x_unit": "kV",
        "y_unit": "ns",
        "draft_pixel_points": pixel_points,
        "extraction_method": "manual_assisted_visible_curve_pick_from_local_crop",
        "overlay_rms_residual_px": sqrt(
            sum(residual * residual for residual in residuals) / len(residuals)
        ),
        "overlay_max_residual_px": max(residuals),
    }


def build_packet() -> dict[str, Any]:
    fwhm_pixel_points = [
        [255.0, 195.0],
        [292.0, 213.0],
        [329.0, 230.0],
        [403.0, 252.0],
        [476.0, 277.0],
        [587.0, 299.0],
        [697.0, 313.0],
        [771.0, 321.0],
    ]
    rise_time_pixel_points = [
        [255.0, 296.0],
        [292.0, 322.0],
        [329.0, 338.0],
        [403.0, 358.0],
        [476.0, 379.0],
        [587.0, 400.0],
        [697.0, 409.0],
        [771.0, 414.0],
    ]
    digitized_series = [
        _series(
            name="fwhm_candidate",
            source_curve_label="FWHM",
            pixel_points=fwhm_pixel_points,
        ),
        _series(
            name="rise_time_candidate",
            source_curve_label="Rise time",
            pixel_points=rise_time_pixel_points,
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
        "task_id": "a14_klir_2011_fig2_timing_response_draft",
        "validation_scope": "a14_klir_2011_tof_detector_response",
        "source_path": SOURCE_PATH,
        "source_sha256": sha256_file(REPO_ROOT / SOURCE_PATH),
        "source_pdf_path": SOURCE_PDF_PATH,
        "source_pdf_sha256": sha256_file(REPO_ROOT / SOURCE_PDF_PATH),
        "source_lines": "172-209",
        "figure_image_path": FIGURE_IMAGE_PATH,
        "figure_image_sha256": sha256_file(REPO_ROOT / FIGURE_IMAGE_PATH),
        "figure_id": "Fig. 2",
        "page": 3,
        "extraction_type": "figure",
        "extraction_status": "draft_unreviewed_residual_measured",
        "required_data": "pmt_response_fwhm_and_rise_time_vs_voltage",
        "axis_calibration": {
            "x": {
                "pixel_points": [AXIS_LEFT_PX, AXIS_RIGHT_PX],
                "data_values": [X_MIN, X_MAX],
                "unit": "kV",
                "rms_residual_px": 0.75,
                "quantity": "pmt_voltage",
            },
            "y": {
                "pixel_points": [AXIS_BOTTOM_PX, AXIS_TOP_PX],
                "data_values": [Y_MIN, Y_MAX],
                "unit": "ns",
                "rms_residual_px": 0.75,
                "quantity": "time_response",
            },
        },
        "digitized_series": digitized_series,
        "draft_extraction_metadata": {
            "visible_curves": [
                "FWHM",
                "Rise time",
            ],
            "source_note": (
                "Source lines 172-209 identify Fig. 2 as PMT response FWHM "
                "and rise time versus operating voltage; the caption says "
                "error bars indicate +/-2 sigma uncertainty."
            ),
            "error_bar_status": (
                "Curve centerlines are sampled; numeric error-bar extents are "
                "not extracted in this packet."
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
        "# A14 Klir Fig. 2 Timing-Response Digitization Draft",
        "",
        f"Generated UTC: `{packet['generated_utc']}`",
        "",
        "This packet contains draft FWHM and rise-time curve extractions from "
        "the local Klir 2011 Fig. 2 crop. It is not validation evidence.",
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
        "- Error-bar magnitudes are not digitized in this packet.",
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
