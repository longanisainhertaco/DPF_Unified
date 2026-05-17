"""GV verified-shot workbook waveform extraction.

The local GV bundle contains experimental current waveform columns in Excel
workbooks plus Gratton-Vargas reduced-model output. This module extracts only
the workbook experimental columns into fail-closed first-principles target
packets. It does not run ``GV.exe`` and does not treat GV output as physics
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from dpf.first_principles.source_targets import GV_ROOT, GV_VERIFIED_SHOTS

_CELL_RE = re.compile(r"^([A-Z]+)([0-9]+)$")
_SUPPORTED_SERIES = {"preferred", "raw", "measured", "smoothed"}


def extract_gv_current_waveform_packet(
    shot_id: str,
    *,
    series: str = "preferred",
    root: str | Path = GV_ROOT,
    require_hash_match: bool = True,
) -> dict[str, Any]:
    """Extract a candidate current-waveform packet from a verified GV workbook.

    The returned packet can seed engineering comparators only. It remains
    blocked for first-principles validation until the raw artifact or verified
    extract is promoted into ``KnowledgeReference/`` and reviewed with
    uncertainty.
    """

    row = _gv_shot_row(shot_id)
    selected = _select_waveform_columns(row, series=series)
    workbook_path = Path(root) / str(row["xlsx_file"])
    actual_sha256 = _sha256_file(workbook_path)
    expected_sha256 = str(row["xlsx_sha256"])
    hash_matches = actual_sha256 == expected_sha256
    if require_hash_match and not hash_matches:
        raise ValueError(
            f"GV workbook hash mismatch for {shot_id}: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )

    time_values, current_values = _read_xlsx_numeric_pair(
        workbook_path,
        time_col=selected["time_col"],
        current_col=selected["current_col"],
    )
    if not time_values:
        raise ValueError(f"no waveform points extracted for GV shot {shot_id}")

    monotonic = _is_monotonic_non_decreasing(time_values)
    series_payload = {
        "x": time_values,
        "y": current_values,
        "x_unit": "us",
        "y_unit": "kA",
    }
    series_sha256 = _sha256_json(series_payload)
    packet = {
        "task_id": f"gv_{row['shot_id']}_current_waveform_candidate",
        "validation_scope": f"gv_verified_{row['shot_id']}",
        "device": row["device"],
        "shot_id": row["shot_id"],
        "shot_note": row["shot_note"],
        "source_status": "user_verified_local_download_not_knowledge_reference_promoted",
        "accepted_for_first_principles_validation": False,
        "can_seed_engineering_comparator": True,
        "source_paths": {
            "workbook": str(workbook_path),
            "input_deck": str(Path(root) / str(row["input_file"])),
            "gv_reduced_model_output": str(Path(root) / str(row["txt_file"])),
        },
        "source_hashes": {
            "workbook_expected_sha256": expected_sha256,
            "workbook_actual_sha256": actual_sha256,
            "workbook_hash_matches_expected": hash_matches,
            "input_deck_sha256": row["input_sha256"],
            "gv_reduced_model_output_sha256": row["txt_sha256"],
        },
        "workbook": {
            "sheet": "Sheet1",
            "time_column": selected["time_col"],
            "current_column": selected["current_col"],
            "series_kind": selected["series_kind"],
            "note": row["workbook_note"],
        },
        "digitized_series": [
            {
                "name": selected["series_name"],
                **series_payload,
                "point_count": len(time_values),
                "series_sha256": series_sha256,
            }
        ],
        "summary": {
            "point_count": len(time_values),
            "time_min_us": min(time_values),
            "time_max_us": max(time_values),
            "current_min_kA": min(current_values),
            "current_max_kA": max(current_values),
            "monotonic_time_non_decreasing": monotonic,
            "contains_negative_current": any(value < 0.0 for value in current_values),
        },
        "gv_baseline_policy": {
            "gv_txt_may_be_used_as_reduced_model_baseline": True,
            "gv_txt_may_support_first_principles_closure": False,
            "experimental_columns_are_separated_from_gv_output": True,
        },
        "verification": {
            "review_status": "candidate_not_reviewed",
            "independent_review_count": 0,
            "per_point_uncertainty_status": "missing",
            "knowledge_reference_promotion_status": "not_promoted",
            "accepted_for_validation": False,
        },
        "missing_for_first_principles_acceptance": [
            "knowledge_reference_promotion",
            "per_point_time_uncertainty",
            "per_point_current_uncertainty",
            "independent_review",
            "output_mapping",
            "comparator_metric_and_tolerance",
            "startup_bvp",
            "spatial_density_field_temperature_history",
            "mechanism_separated_neutron_history",
            "detector_response_and_uq",
        ],
    }
    packet["packet_sha256"] = _sha256_json(packet)
    return packet


def extract_all_gv_current_waveform_packets(
    *,
    series: str = "preferred",
    root: str | Path = GV_ROOT,
    require_hash_match: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Extract candidate waveform packets for every unique GV verified shot."""

    return tuple(
        extract_gv_current_waveform_packet(
            str(row["shot_id"]),
            series=series,
            root=root,
            require_hash_match=require_hash_match,
        )
        for row in GV_VERIFIED_SHOTS
    )


def gv_waveform_packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a small summary for UI/manifest surfaces without full arrays."""

    return {
        "task_id": packet["task_id"],
        "validation_scope": packet["validation_scope"],
        "device": packet["device"],
        "shot_id": packet["shot_id"],
        "accepted_for_first_principles_validation": packet[
            "accepted_for_first_principles_validation"
        ],
        "series": packet["digitized_series"][0]["name"],
        "point_count": packet["summary"]["point_count"],
        "time_range_us": [
            packet["summary"]["time_min_us"],
            packet["summary"]["time_max_us"],
        ],
        "current_range_kA": [
            packet["summary"]["current_min_kA"],
            packet["summary"]["current_max_kA"],
        ],
        "packet_sha256": packet["packet_sha256"],
        "missing_for_first_principles_acceptance": list(
            packet["missing_for_first_principles_acceptance"]
        ),
    }


def _select_waveform_columns(row: dict[str, Any], *, series: str) -> dict[str, str]:
    if series not in _SUPPORTED_SERIES:
        allowed = ", ".join(sorted(_SUPPORTED_SERIES))
        raise ValueError(f"series must be one of {allowed}")

    columns = row["waveform_columns"]
    if series in {"preferred", "raw"} and {
        "raw_time_us",
        "raw_current_kA",
    } <= set(columns):
        return {
            "time_col": str(columns["raw_time_us"]),
            "current_col": str(columns["raw_current_kA"]),
            "series_kind": "raw_workbook_experimental_waveform",
            "series_name": "raw_measured_current",
        }
    if series in {"preferred", "measured"} and {
        "time_us",
        "current_kA",
    } <= set(columns):
        return {
            "time_col": str(columns["time_us"]),
            "current_col": str(columns["current_kA"]),
            "series_kind": "workbook_experimental_waveform",
            "series_name": "measured_current",
        }
    if series in {"preferred", "smoothed"} and {
        "smoothed_time_us",
        "smoothed_current_kA",
    } <= set(columns):
        return {
            "time_col": str(columns["smoothed_time_us"]),
            "current_col": str(columns["smoothed_current_kA"]),
            "series_kind": "smoothed_workbook_experimental_waveform",
            "series_name": "smoothed_measured_current",
        }
    raise ValueError(f"GV shot {row['shot_id']} has no {series!r} waveform columns")


def _read_xlsx_numeric_pair(
    path: Path,
    *,
    time_col: str,
    current_col: str,
) -> tuple[list[float], list[float]]:
    wanted = {time_col.upper(), current_col.upper()}
    rows: dict[int, dict[str, float]] = {}
    with zipfile.ZipFile(path) as zf:
        with zf.open("xl/worksheets/sheet1.xml") as sheet:
            for _, elem in ET.iterparse(sheet, events=("end",)):
                if _local_name(elem.tag) != "c":
                    continue
                ref = elem.attrib.get("r", "")
                parsed = _parse_cell_ref(ref)
                if parsed is None:
                    elem.clear()
                    continue
                col, row_number = parsed
                if col not in wanted:
                    elem.clear()
                    continue
                value = _numeric_cell_value(elem)
                if value is not None and math.isfinite(value):
                    rows.setdefault(row_number, {})[col] = value
                elem.clear()

    time_values: list[float] = []
    current_values: list[float] = []
    for row_number in sorted(rows):
        row = rows[row_number]
        if time_col.upper() in row and current_col.upper() in row:
            time_values.append(row[time_col.upper()])
            current_values.append(row[current_col.upper()])
    return time_values, current_values


def _numeric_cell_value(elem: ET.Element) -> float | None:
    if elem.attrib.get("t") in {"s", "str", "inlineStr"}:
        return None
    text = None
    for child in elem:
        if _local_name(child.tag) == "v":
            text = child.text
            break
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_cell_ref(ref: str) -> tuple[str, int] | None:
    match = _CELL_RE.match(ref.upper())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _gv_shot_row(shot_id: str) -> dict[str, Any]:
    normalized = str(shot_id).strip().lower()
    for row in GV_VERIFIED_SHOTS:
        if str(row["shot_id"]).lower() == normalized:
            return row
    allowed = ", ".join(str(row["shot_id"]) for row in GV_VERIFIED_SHOTS)
    raise ValueError(f"unknown GV verified shot {shot_id!r}; expected one of {allowed}")


def _is_monotonic_non_decreasing(values: list[float]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
