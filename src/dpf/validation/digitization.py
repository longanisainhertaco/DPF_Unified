"""Verification helpers for digitized KnowledgeReference figures and tables.

Digitized data can support validation only when the source document, extracted
figure image, axis calibration, extracted arrays, and review status are all
traceable.  This module audits that packet; it does not perform interactive
digitization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path


_AKEL_2021_SOURCE_PATH = (
    "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
)
_AKEL_2021_SOURCE_SHA256 = (
    "31a68fe51d3ccc5b8181392ae18f66245d0b0926784371fb53eaf2306674cf7a"
)
_AKEL_2021_PDF_SHA256 = (
    "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
)
_AKEL_2021_PDF_CANDIDATES = [
    "archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf",
    "archive_reference_OLD/references/papers/archive/akel-2021-pf1000-neutron-yield.pdf",
]
_AKEL_FIG1_DRAFT_PACKET_PATH = (
    "KnowledgeReference/digitization/"
    "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
)
_AKEL_FIG1_DRAFT_PACKET_SHA256 = (
    "abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7"
)
_AKEL_REQUIRED_PACKET_FIELDS = [
    "task_id",
    "source_path",
    "source_sha256",
    "source_pdf_sha256",
    "source_lines",
    "figure_image_path",
    "figure_image_sha256",
    "figure_id",
    "page",
    "axis_calibration",
    "digitized_series",
    "verification",
]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def akel_fig1_draft_digitization_packet(
    base_path: str | Path = ".",
) -> dict[str, object]:
    """Load the draft Akel Fig. 1 digitization packet with hash metadata."""
    packet_path = Path(base_path) / _AKEL_FIG1_DRAFT_PACKET_PATH
    payload = json.loads(packet_path.read_text())
    actual_sha256 = sha256_file(packet_path)
    payload["draft_packet_path"] = _AKEL_FIG1_DRAFT_PACKET_PATH
    payload["draft_packet_sha256"] = actual_sha256
    payload["draft_packet_expected_sha256"] = _AKEL_FIG1_DRAFT_PACKET_SHA256
    payload["draft_packet_hash_verified"] = (
        actual_sha256 == _AKEL_FIG1_DRAFT_PACKET_SHA256
    )
    return payload


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _numbers(value: object) -> list[float]:
    if not _is_sequence(value):
        return []
    numbers: list[float] = []
    for item in value:
        try:
            number = float(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
        if not isfinite(number):
            return []
        numbers.append(number)
    return numbers


def _path_is_knowledge_reference(raw_path: str) -> bool:
    parts = Path(raw_path).parts
    return bool(parts) and parts[0] == "KnowledgeReference" and ".." not in parts


def _akel_common_digitization_fields() -> dict[str, object]:
    return {
        "validation_scope": "pf1000_16kv_2021_akel",
        "source_path": _AKEL_2021_SOURCE_PATH,
        "source_sha256": _AKEL_2021_SOURCE_SHA256,
        "source_pdf_sha256": _AKEL_2021_PDF_SHA256,
        "source_pdf_candidates": list(_AKEL_2021_PDF_CANDIDATES),
        "source_markdown_pdf_text_parity_passed": True,
        "digitization_gate": "digitization_verification_evidence",
        "required_digitization_packet_fields": list(_AKEL_REQUIRED_PACKET_FIELDS),
        "requires_independent_review": True,
        "figure_image_status": "not_extracted",
        "rendering_tool_candidates": ["pdftoppm", "pdfimages"],
        "suggested_page_render_command": (
            "pdftoppm -png -r 300 -f <page> -l <page> <source_pdf> "
            "<output_prefix>"
        ),
        "done_condition": (
            "Render the cited figure page, crop only the cited figure, extract "
            "the required series, pass digitization_verification_evidence(), "
            "then attach the arrays to a same-scope KR target or evidence "
            "comparator."
        ),
        "model_use_boundary": (
            "This queue is planning metadata only. It does not validate a "
            "simulation until the cited figure data are digitized and the "
            "verification gate passes."
        ),
    }


def _akel_current_waveform_digitization_task(
    *,
    task_id: str,
    figure_id: str,
    source_lines: str,
    figure_caption: str,
    page: int,
    shot: int,
    pressure_torr: float,
    figure_image_status: str = "not_extracted",
    figure_image_path: str | None = None,
    figure_image_sha256: str | None = None,
    axis_calibration_candidate: Mapping[str, object] | None = None,
    series_extraction_candidate: Mapping[str, object] | None = None,
    draft_digitization_packet_status: str | None = None,
    draft_digitization_packet_path: str | None = None,
    draft_digitization_packet_sha256: str | None = None,
    extraction_note: str | None = None,
) -> dict[str, object]:
    task = _akel_common_digitization_fields()
    task.update(
        {
            "task_id": task_id,
            "priority": 1,
            "group": "circuit_waveform",
            "source_lines": source_lines,
            "figure_id": figure_id,
            "figure_caption": figure_caption,
            "page": page,
            "shot": shot,
            "pressure_torr": pressure_torr,
            "figure_image_status": figure_image_status,
            "x_quantity": "time",
            "x_unit": "us",
            "y_quantity": "discharge_current",
            "y_unit": "kA_or_MA_as_axis_labeled",
            "required_series": [
                "measured_current",
                "computed_current",
            ],
            "target_after_digitization": "pf1000_16kv_current_waveform_targets",
            "scientific_gap_closed_if_verified": (
                "same-shot current waveform arrays for PF-1000 16 kV Akel "
                "waveform validation"
            ),
        }
    )
    if figure_image_path:
        task["figure_image_path"] = figure_image_path
    if figure_image_sha256:
        task["figure_image_sha256"] = figure_image_sha256
    if axis_calibration_candidate:
        task["axis_calibration_candidate"] = dict(axis_calibration_candidate)
    if series_extraction_candidate:
        task["series_extraction_candidate"] = dict(series_extraction_candidate)
    if draft_digitization_packet_status:
        task["draft_digitization_packet_status"] = draft_digitization_packet_status
    if draft_digitization_packet_path:
        task["draft_digitization_packet_path"] = draft_digitization_packet_path
    if draft_digitization_packet_sha256:
        task["draft_digitization_packet_sha256"] = draft_digitization_packet_sha256
    if extraction_note:
        task["extraction_note"] = extraction_note
    return task


def _akel_yield_digitization_task(
    *,
    task_id: str,
    figure_id: str,
    source_lines: str,
    figure_caption: str,
    page: int,
    pressure_torr: float,
) -> dict[str, object]:
    task = _akel_common_digitization_fields()
    task.update(
        {
            "task_id": task_id,
            "priority": 2,
            "group": "neutron_yield",
            "source_lines": source_lines,
            "figure_id": figure_id,
            "figure_caption": figure_caption,
            "page": page,
            "pressure_torr": pressure_torr,
            "x_quantity": "shot_index_or_shot_number",
            "x_unit": "shot",
            "y_quantity": "neutron_yield",
            "y_unit": "neutrons_per_shot",
            "required_series": [
                "measured_neutron_yield",
                "computed_neutron_yield",
            ],
            "table_backed_scalar_target": "pf1000_16kv_shot_table_2021_akel",
            "target_after_digitization": "pf1000_16kv_akel_table_targets",
            "scientific_gap_closed_if_verified": (
                "plot-level cross-check of the typed Akel scalar neutron-yield "
                "table, not detector-response closure"
            ),
        }
    )
    return task


def scientific_closure_digitization_queue() -> dict[str, object]:
    """Return local figure digitization tasks required for closure planning."""
    tasks = [
        _akel_current_waveform_digitization_task(
            task_id="akel_2021_fig1_current_waveform_shot_12581",
            figure_id="Fig. 1",
            source_lines="294-295",
            figure_caption=(
                "Computed and measured currents of the PF1000 at 16 kV, "
                "1.2 Torr of deuterium (shot 12581)."
            ),
            page=3,
            shot=12581,
            pressure_torr=1.20,
            figure_image_status="extracted_not_digitized",
            figure_image_path=(
                "KnowledgeReference/figures/"
                "akel-2021-fig1-current-waveform-shot-12581.png"
            ),
            figure_image_sha256=(
                "4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e"
            ),
            axis_calibration_candidate={
                "x": {
                    "pixel_points": [112.5, 997.3],
                    "data_values": [0.0, 10.0],
                    "unit": "us",
                    "basis": "OCR and vector axis ticks from the extracted panel",
                },
                "y": {
                    "pixel_points": [617.0, 53.2],
                    "data_values": [0.0, 1400.0],
                    "unit": "kA",
                    "basis": "OCR and vector axis ticks from the extracted panel",
                },
            },
            series_extraction_candidate={
                "basis": (
                    "Current pdftocairo page-3 SVG exposes separable vector "
                    "paths for the Fig. 1 trace candidates."
                ),
                "plot_bbox_svg_points": {
                    "x0": 336.15,
                    "y0": 468.32,
                    "x1": 551.18,
                    "y1": 603.27,
                },
                "measured_current_candidate": {
                    "svg_path_range": "filled black paths 1987-2280",
                    "point_count": 294,
                    "x_us_range": [0.02, 9.98],
                    "y_kA_range": [11.0, 1261.0],
                },
                "computed_current_candidate": {
                    "svg_path_range": "black stroke paths 1942-1975",
                    "point_count": 34,
                    "x_us_range": [0.01, 10.01],
                    "y_kA_range": [22.0, 1265.0],
                },
                "legend_text_exclusion": (
                    "filled black paths 2345-2411 are legend glyphs in the "
                    "white legend box, not trace data"
                ),
                "acceptance_boundary": (
                    "Candidate path separation is not an accepted digitization "
                    "packet. Accepted evidence still requires exported arrays, "
                    "overlay residuals, and independent review."
                ),
            },
            draft_digitization_packet_status="draft_unreviewed",
            draft_digitization_packet_path=_AKEL_FIG1_DRAFT_PACKET_PATH,
            draft_digitization_packet_sha256=_AKEL_FIG1_DRAFT_PACKET_SHA256,
            extraction_note=(
                "Figure panel cropped from the parity-verified local Akel PDF "
                "page 3 render at 300 dpi. This is not accepted digitized "
                "waveform evidence until measured/computed series arrays, "
                "overlay residuals, and independent review are supplied."
            ),
        ),
        _akel_current_waveform_digitization_task(
            task_id="akel_2021_fig2_current_waveform_shot_12584",
            figure_id="Fig. 2",
            source_lines="296-297",
            figure_caption=(
                "Computed and measured currents of the PF1000 at 16 kV, "
                "1.2 Torr of deuterium (shot 12584)."
            ),
            page=3,
            shot=12584,
            pressure_torr=1.20,
        ),
        _akel_current_waveform_digitization_task(
            task_id="akel_2021_fig3_current_waveform_shot_12592",
            figure_id="Fig. 3",
            source_lines="298-299",
            figure_caption=(
                "Computed and measured currents of the PF1000 at 16 kV, "
                "1.05 Torr of deuterium (shot 12592)."
            ),
            page=3,
            shot=12592,
            pressure_torr=1.05,
        ),
        _akel_current_waveform_digitization_task(
            task_id="akel_2021_fig4_current_waveform_shot_12604",
            figure_id="Fig. 4",
            source_lines="300-301",
            figure_caption=(
                "Computed and measured currents of the PF1000 at 16 kV, "
                "1.05 Torr of deuterium (shot 12604)."
            ),
            page=3,
            shot=12604,
            pressure_torr=1.05,
        ),
        _akel_yield_digitization_task(
            task_id="akel_2021_fig5_neutron_yield_1p20_torr",
            figure_id="Fig. 5",
            source_lines="916",
            figure_caption=(
                "Calculated and measured neutron yields of the PF1000 at "
                "16 kV, 1.2 Torr of deuterium."
            ),
            page=5,
            pressure_torr=1.20,
        ),
        _akel_yield_digitization_task(
            task_id="akel_2021_fig6_neutron_yield_1p05_torr",
            figure_id="Fig. 6",
            source_lines="917",
            figure_caption=(
                "Calculated and measured neutron yields of the PF1000 at "
                "16 kV, 1.05 Torr of deuterium."
            ),
            page=5,
            pressure_torr=1.05,
        ),
    ]
    return {
        "model_role": "scientific_closure_digitization_queue",
        "source_of_truth_rule": (
            "Only local KnowledgeReference markdown is source-of-truth "
            "scientific evidence. Local PDFs are parity references for figure "
            "rendering, and digitized arrays become evidence only after the "
            "verification gate passes."
        ),
        "validation_scope": "pf1000_16kv_2021_akel",
        "source_path": _AKEL_2021_SOURCE_PATH,
        "source_sha256": _AKEL_2021_SOURCE_SHA256,
        "source_pdf_sha256": _AKEL_2021_PDF_SHA256,
        "source_pdf_candidates": list(_AKEL_2021_PDF_CANDIDATES),
        "queue_status": "open",
        "items": tasks,
        "summary": {
            "task_count": len(tasks),
            "priority_1_count": sum(1 for task in tasks if task["priority"] == 1),
            "extracted_figure_count": sum(
                1
                for task in tasks
                if str(task.get("figure_image_status")) != "not_extracted"
            ),
            "not_extracted_figure_count": sum(
                1
                for task in tasks
                if str(task.get("figure_image_status")) == "not_extracted"
            ),
            "draft_digitization_packet_count": sum(
                1 for task in tasks if task.get("draft_digitization_packet_path")
            ),
            "figure_count": len({task["figure_id"] for task in tasks}),
            "groups": sorted({str(task["group"]) for task in tasks}),
            "not_yet_evidence": True,
        },
    }


def _coerce_packets_by_task(
    packets: Mapping[str, object] | Sequence[object] | None,
) -> tuple[dict[str, Mapping[str, object]], list[str]]:
    if packets is None:
        return {}, []
    invalid_packets: list[str] = []
    packets_by_task: dict[str, Mapping[str, object]] = {}
    if isinstance(packets, Mapping):
        if "task_id" in packets:
            task_id = str(packets.get("task_id", ""))
            if task_id:
                packets_by_task[task_id] = packets
            else:
                invalid_packets.append("packet_missing_task_id")
            return packets_by_task, invalid_packets
        for key, packet in packets.items():
            if isinstance(packet, Mapping):
                task_id = str(packet.get("task_id") or key)
                packets_by_task[task_id] = packet
            else:
                invalid_packets.append(str(key))
        return packets_by_task, invalid_packets
    if _is_sequence(packets):
        for index, packet in enumerate(packets):
            if not isinstance(packet, Mapping):
                invalid_packets.append(f"packet_{index}_not_mapping")
                continue
            task_id = str(packet.get("task_id", ""))
            if not task_id:
                invalid_packets.append(f"packet_{index}_missing_task_id")
                continue
            packets_by_task[task_id] = packet
        return packets_by_task, invalid_packets
    return {}, ["packets_not_mapping_or_sequence"]


def _packet_series_names(packet: Mapping[str, object]) -> set[str]:
    digitized_series = packet.get("digitized_series", [])
    if not _is_sequence(digitized_series):
        return set()
    return {
        str(series.get("name"))
        for series in digitized_series
        if isinstance(series, Mapping) and str(series.get("name", ""))
    }


def _task_packet_failures(
    packet: Mapping[str, object],
    task: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    for key in task.get("required_digitization_packet_fields", []):
        if not packet.get(str(key)):
            failures.append(f"{key}_missing")

    for key in (
        "task_id",
        "source_path",
        "source_sha256",
        "source_pdf_sha256",
        "source_lines",
        "figure_id",
        "page",
    ):
        packet_value = str(packet.get(key, ""))
        task_value = str(task.get(key, ""))
        if task_value and packet_value != task_value:
            failures.append(f"{key}_mismatch")

    required_series = {str(name) for name in task.get("required_series", [])}
    packet_series = _packet_series_names(packet)
    for missing_series in sorted(required_series - packet_series):
        failures.append(f"missing_required_series:{missing_series}")
    return sorted(set(failures))


def scientific_closure_digitization_status(
    packets: Mapping[str, object] | Sequence[object] | None = None,
    *,
    base_path: str | Path = ".",
) -> dict[str, object]:
    """Evaluate digitization packets against the local closure queue."""
    queue = scientific_closure_digitization_queue()
    packets_by_task, invalid_packets = _coerce_packets_by_task(packets)
    task_statuses: list[dict[str, object]] = []
    accepted_count = 0
    failed_count = 0
    open_count = 0

    for task in queue["items"]:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task["task_id"])
        packet = packets_by_task.get(task_id)
        if packet is None:
            open_count += 1
            task_statuses.append(
                {
                    "task_id": task_id,
                    "status": "open",
                    "passed": False,
                    "group": task["group"],
                    "figure_id": task["figure_id"],
                    "figure_image_status": task.get("figure_image_status", ""),
                    "figure_image_path": task.get("figure_image_path", ""),
                    "figure_image_sha256": task.get("figure_image_sha256", ""),
                    "draft_digitization_packet_status": task.get(
                        "draft_digitization_packet_status",
                        "",
                    ),
                    "draft_digitization_packet_path": task.get(
                        "draft_digitization_packet_path",
                        "",
                    ),
                    "draft_digitization_packet_sha256": task.get(
                        "draft_digitization_packet_sha256",
                        "",
                    ),
                    "missing_or_failed_checks": ["digitization_packet_missing"],
                }
            )
            continue
        evidence = digitization_verification_evidence(packet, base_path=base_path)
        task_failures = _task_packet_failures(packet, task)
        missing_or_failed = sorted(
            set(evidence["missing_or_failed_checks"]) | set(task_failures)
        )
        accepted = bool(evidence["passed"]) and not missing_or_failed
        if accepted:
            accepted_count += 1
            status = "accepted"
        else:
            failed_count += 1
            status = "failed"
        task_statuses.append(
            {
                "task_id": task_id,
                "status": status,
                "passed": accepted,
                "group": task["group"],
                "figure_id": task["figure_id"],
                "figure_image_status": task.get("figure_image_status", ""),
                "figure_image_path": task.get("figure_image_path", ""),
                "figure_image_sha256": task.get("figure_image_sha256", ""),
                "draft_digitization_packet_status": task.get(
                    "draft_digitization_packet_status",
                    "",
                ),
                "draft_digitization_packet_path": task.get(
                    "draft_digitization_packet_path",
                    "",
                ),
                "draft_digitization_packet_sha256": task.get(
                    "draft_digitization_packet_sha256",
                    "",
                ),
                "missing_or_failed_checks": missing_or_failed,
                "evidence": evidence,
            }
        )

    expected_task_ids = {
        str(task["task_id"])
        for task in queue["items"]
        if isinstance(task, Mapping)
    }
    extra_packet_ids = sorted(set(packets_by_task) - expected_task_ids)
    task_count = len(task_statuses)
    return {
        "model_role": "scientific_closure_digitization_status",
        "validation_scope": queue["validation_scope"],
        "source_path": queue["source_path"],
        "queue_complete": accepted_count == task_count and not invalid_packets,
        "accepted_task_count": accepted_count,
        "failed_task_count": failed_count,
        "open_task_count": open_count,
        "task_count": task_count,
        "invalid_packets": sorted(invalid_packets),
        "extra_packet_ids": extra_packet_ids,
        "task_statuses": task_statuses,
        "missing_or_failed_tasks": [
            status["task_id"]
            for status in task_statuses
            if status["status"] != "accepted"
        ],
    }


def _check_hash(
    packet: Mapping[str, object],
    *,
    base_path: Path,
    path_key: str,
    hash_key: str,
    required: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    raw_path = str(packet.get(path_key, ""))
    expected_hash = str(packet.get(hash_key, ""))
    if not raw_path:
        return (not required, [path_key] if required else [])
    if not _path_is_knowledge_reference(raw_path):
        return False, [f"{path_key}_not_knowledge_reference"]
    if not expected_hash:
        return False, [hash_key]
    path = base_path / raw_path
    if not path.exists() or not path.is_file():
        return False, [f"{path_key}_missing"]
    if sha256_file(path) != expected_hash:
        failures.append(f"{hash_key}_mismatch")
    return not failures, failures


def _axis_calibration_passed(
    calibration: object,
    axis: str,
    *,
    max_axis_residual_px: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not isinstance(calibration, Mapping):
        return False, [f"{axis}_axis_calibration_missing"]
    pixels = _numbers(calibration.get("pixel_points"))
    values = _numbers(calibration.get("data_values"))
    residual = calibration.get("rms_residual_px", 0.0)
    try:
        residual_value = float(residual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        residual_value = float("inf")
    if len(pixels) < 2 or len(values) < 2:
        failures.append(f"{axis}_axis_needs_two_calibration_points")
    if len(pixels) != len(values):
        failures.append(f"{axis}_axis_calibration_length_mismatch")
    if len(set(pixels)) < 2:
        failures.append(f"{axis}_axis_pixel_points_not_distinct")
    if len(set(values)) < 2:
        failures.append(f"{axis}_axis_data_values_not_distinct")
    if not isfinite(residual_value) or residual_value > max_axis_residual_px:
        failures.append(f"{axis}_axis_residual_too_large")
    if not str(calibration.get("unit", "")):
        failures.append(f"{axis}_axis_unit_missing")
    return not failures, failures


def _series_passed(
    series: object,
    *,
    min_points: int,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not isinstance(series, Mapping):
        return False, ["digitized_series_record_invalid"]
    x = _numbers(series.get("x"))
    y = _numbers(series.get("y"))
    if len(x) < min_points or len(y) < min_points:
        failures.append("digitized_series_too_short")
    if len(x) != len(y):
        failures.append("digitized_series_length_mismatch")
    if not str(series.get("x_unit", "")):
        failures.append("digitized_series_x_unit_missing")
    if not str(series.get("y_unit", "")):
        failures.append("digitized_series_y_unit_missing")
    if not str(series.get("name", "")):
        failures.append("digitized_series_name_missing")
    return not failures, failures


def digitization_verification_evidence(
    packet: Mapping[str, object],
    *,
    base_path: str | Path = ".",
    min_points: int = 3,
    max_axis_residual_px: float = 1.0,
    max_overlay_residual_px: float = 2.0,
) -> dict[str, object]:
    """Audit a digitized data packet for one-for-one source traceability."""
    failures: list[str] = []
    base = Path(base_path)

    source_path = str(packet.get("source_path", ""))
    source_ok, source_failures = _check_hash(
        packet,
        base_path=base,
        path_key="source_path",
        hash_key="source_sha256",
        required=True,
    )
    if not source_ok:
        failures.extend(source_failures)

    extraction_type = str(packet.get("extraction_type", "figure"))
    if extraction_type == "figure":
        figure_ok, figure_failures = _check_hash(
            packet,
            base_path=base,
            path_key="figure_image_path",
            hash_key="figure_image_sha256",
            required=True,
        )
        if not figure_ok:
            failures.extend(figure_failures)
    elif extraction_type != "table":
        failures.append("extraction_type_unknown")

    if not str(packet.get("figure_id") or packet.get("table_id") or ""):
        failures.append("source_item_id_missing")
    if not packet.get("page"):
        failures.append("source_page_missing")

    axis_calibration = packet.get("axis_calibration", {})
    if extraction_type == "figure":
        for axis in ("x", "y"):
            calibration = (
                axis_calibration.get(axis)
                if isinstance(axis_calibration, Mapping)
                else None
            )
            ok, axis_failures = _axis_calibration_passed(
                calibration,
                axis,
                max_axis_residual_px=max_axis_residual_px,
            )
            if not ok:
                failures.extend(axis_failures)

    digitized_series = packet.get("digitized_series", [])
    if not _is_sequence(digitized_series) or not digitized_series:
        failures.append("digitized_series_missing")
    elif _is_sequence(digitized_series):
        for series in digitized_series:
            ok, series_failures = _series_passed(series, min_points=min_points)
            if not ok:
                failures.extend(series_failures)

    verification = packet.get("verification", {})
    if not isinstance(verification, Mapping):
        failures.append("verification_block_missing")
        verification = {}
    try:
        overlay_residual = float(verification.get("overlay_rms_residual_px", 0.0))
    except (TypeError, ValueError):
        overlay_residual = float("inf")
    if extraction_type == "figure" and (
        not isfinite(overlay_residual)
        or overlay_residual > max_overlay_residual_px
    ):
        failures.append("overlay_residual_too_large")
    try:
        independent_review_count = int(
            verification.get("independent_review_count", 0) or 0
        )
    except (TypeError, ValueError):
        independent_review_count = 0
    if independent_review_count < 1:
        failures.append("independent_review_missing")
    if verification.get("review_status") != "accepted":
        failures.append("review_status_not_accepted")

    missing_or_failed = sorted(set(failures))
    return {
        "passed": not missing_or_failed,
        "validation_tier": "digitization_provenance",
        "model_role": "digitized_source_verification_audit",
        "source": source_path,
        "source_sha256": packet.get("source_sha256", ""),
        "figure_image_path": packet.get("figure_image_path", ""),
        "figure_image_sha256": packet.get("figure_image_sha256", ""),
        "figure_id": packet.get("figure_id", ""),
        "table_id": packet.get("table_id", ""),
        "page": packet.get("page", ""),
        "extraction_type": extraction_type,
        "n_series": len(digitized_series) if _is_sequence(digitized_series) else 0,
        "missing_or_failed_checks": missing_or_failed,
        "validity_notes": {
            "claim_scope": (
                "A digitized series can support validation only for the cited "
                "source item and only after this one-for-one provenance audit "
                "passes."
            ),
            "audit_role": (
                "This audit checks traceability and digitization quality; it "
                "does not validate a simulation against the digitized data."
            ),
        },
    }
