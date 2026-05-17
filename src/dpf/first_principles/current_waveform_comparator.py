"""Engineering current-waveform comparators for first-principles runs.

The routines in this module bind user-verified current-waveform candidate
targets to solver outputs without using the target as a fit, drive, or
reduced-model closure. They produce engineering telemetry only and always fail
closed for scientific acceptance.

Local source-truth routing:
    docs/FIRST_PRINCIPLES_GV_SHOT_INFO_TRIAGE_2026_05_16.md
    docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dpf.first_principles.gv_waveforms import extract_gv_current_waveform_packet

CURRENT_WAVEFORM_COMPARATOR_SOURCE_REFERENCES = {
    "gv_triage": "docs/FIRST_PRINCIPLES_GV_SHOT_INFO_TRIAGE_2026_05_16.md",
    "source_truth_index": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json",
}
CURRENT_WAVEFORM_COMPARATOR_SOURCE_STATUS = (
    "engineering_comparator_for_user_verified_targets_not_validation"
)
CURRENT_WAVEFORM_COMPARATOR_CAN_SUPPORT_FIRST_PRINCIPLES_ACCEPTANCE = False

STATUS_COMPUTED = "engineering_current_waveform_comparison_not_validation"
STATUS_NO_TARGET = "blocked_current_waveform_target_not_bound"
STATUS_NO_SIMULATION_CURRENT = "blocked_simulated_current_history_missing"
STATUS_NO_OVERLAP = "blocked_current_waveform_time_overlap_missing"


def build_engineering_current_waveform_comparator(
    *,
    declared_scope: str,
    device_name: str | None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    simulation_telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a non-promoting current waveform comparison packet.

    The only currently implemented target source is the user-verified GV
    workbook packet. That target can seed engineering comparison, but it cannot
    drive the first-principles solver and cannot close acceptance gates.
    """

    target_packet, target_status = _load_first_gv_current_target(validation_targets)
    if target_packet is None:
        return _blocked_packet(
            status=STATUS_NO_TARGET,
            declared_scope=declared_scope,
            device_name=device_name,
            reason=target_status,
        )

    sim_series = _simulation_current_series(simulation_telemetry)
    if not sim_series["time_us"]:
        return _blocked_packet(
            status=STATUS_NO_SIMULATION_CURRENT,
            declared_scope=declared_scope,
            device_name=device_name,
            reason="simulation telemetry did not expose circuit current_history",
            target_packet=target_packet,
        )

    target_series = target_packet["digitized_series"][0]
    target_time_us = np.asarray(target_series["x"], dtype=float)
    target_current_kA = np.asarray(target_series["y"], dtype=float)
    sim_time_us = np.asarray(sim_series["time_us"], dtype=float)
    sim_current_kA = np.asarray(sim_series["current_A"], dtype=float) * 1.0e-3

    finite_target = np.isfinite(target_time_us) & np.isfinite(target_current_kA)
    finite_sim = np.isfinite(sim_time_us) & np.isfinite(sim_current_kA)
    target_time_us = target_time_us[finite_target]
    target_current_kA = target_current_kA[finite_target]
    sim_time_us = sim_time_us[finite_sim]
    sim_current_kA = sim_current_kA[finite_sim]

    if target_time_us.size == 0 or sim_time_us.size == 0:
        return _blocked_packet(
            status=STATUS_NO_OVERLAP,
            declared_scope=declared_scope,
            device_name=device_name,
            reason="target or simulation series had no finite samples",
            target_packet=target_packet,
        )

    order = np.argsort(target_time_us)
    target_time_us = target_time_us[order]
    target_current_kA = target_current_kA[order]

    target_min = float(target_time_us[0])
    target_max = float(target_time_us[-1])
    overlap = (sim_time_us >= target_min) & (sim_time_us <= target_max)
    if not np.any(overlap):
        return _blocked_packet(
            status=STATUS_NO_OVERLAP,
            declared_scope=declared_scope,
            device_name=device_name,
            reason="simulation current-history times do not overlap target waveform",
            target_packet=target_packet,
            sim_series=sim_series,
        )

    overlap_time_us = sim_time_us[overlap]
    overlap_sim_kA = sim_current_kA[overlap]
    overlap_target_kA = np.interp(
        overlap_time_us,
        target_time_us,
        target_current_kA,
    )
    residual_kA = overlap_sim_kA - overlap_target_kA
    target_duration_us = max(target_max - target_min, 0.0)
    simulation_duration_us = (
        float(np.max(sim_time_us) - np.min(sim_time_us)) if sim_time_us.size else 0.0
    )
    temporal_coverage_fraction = (
        0.0
        if target_duration_us <= 0.0
        else min(1.0, simulation_duration_us / target_duration_us)
    )
    full_target_peak_kA = _peak_abs(target_current_kA)
    overlap_target_peak_kA = _peak_abs(overlap_target_kA)
    sim_peak_kA = _peak_abs(overlap_sim_kA)

    return {
        "status": STATUS_COMPUTED,
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "engineering_comparison_only_do_not_validate",
        "target_packet": _target_summary(target_packet),
        "output_mapping": {
            "solver_observable": "simulation.circuit.current_history[].current_A",
            "target_observable": "gv_waveform_packet.digitized_series[0].y",
            "time_observable": "gv_waveform_packet.digitized_series[0].x",
            "solver_current_unit": "A",
            "target_current_unit": "kA",
            "unit_conversion": "solver_current_A * 1e-3 -> kA",
            "time_alignment_policy": (
                "absolute workbook time in microseconds; no phase shift, fit, "
                "or waveform-derived drive is applied"
            ),
        },
        "series_counts": {
            "simulation_points": int(sim_time_us.size),
            "target_points": int(target_time_us.size),
            "overlap_points": int(overlap_time_us.size),
        },
        "time_ranges_us": {
            "simulation": [float(np.min(sim_time_us)), float(np.max(sim_time_us))],
            "target": [target_min, target_max],
            "overlap": [
                float(np.min(overlap_time_us)),
                float(np.max(overlap_time_us)),
            ],
        },
        "metrics": {
            "mae_kA": float(np.mean(np.abs(residual_kA))),
            "rmse_kA": float(np.sqrt(np.mean(residual_kA * residual_kA))),
            "max_abs_error_kA": float(np.max(np.abs(residual_kA))),
            "mean_signed_error_kA": float(np.mean(residual_kA)),
            "simulation_overlap_peak_abs_current_kA": sim_peak_kA,
            "target_overlap_peak_abs_current_kA": overlap_target_peak_kA,
            "target_full_peak_abs_current_kA": full_target_peak_kA,
            "peak_abs_current_error_fraction_vs_overlap": _fractional_error(
                sim_peak_kA,
                overlap_target_peak_kA,
            ),
            "peak_abs_current_error_fraction_vs_full_target": _fractional_error(
                sim_peak_kA,
                full_target_peak_kA,
            ),
            "temporal_coverage_fraction_of_target": temporal_coverage_fraction,
        },
        "sampled_series": {
            "time_us": [float(value) for value in overlap_time_us],
            "simulation_current_kA": [float(value) for value in overlap_sim_kA],
            "target_current_kA": [float(value) for value in overlap_target_kA],
            "residual_kA": [float(value) for value in residual_kA],
        },
        "first_principles_policy": {
            "experimental_waveform_used_as_drive": False,
            "experimental_waveform_used_as_fit": False,
            "reduced_model_used": False,
            "comparison_changes_solver_state": False,
        },
        "missing_for_acceptance": _missing_for_acceptance(
            temporal_coverage_fraction=temporal_coverage_fraction,
        ),
        "can_support_first_principles_acceptance": False,
    }


def _load_first_gv_current_target(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    for target in validation_targets:
        if str(target.get("observable", "")) != "current_waveform":
            continue
        source_reference = target.get("source_reference")
        if not isinstance(source_reference, Mapping):
            continue
        record_id = str(source_reference.get("record_id", ""))
        if not record_id.startswith("gv:") or not record_id.endswith(":workbook"):
            continue
        shot_id = record_id.split(":", 2)[1]
        try:
            return extract_gv_current_waveform_packet(shot_id), "loaded"
        except Exception as exc:  # pragma: no cover - exercised when local bundle is absent.
            return None, f"gv waveform packet load failed: {exc}"
    return None, "no GV current_waveform validation target was declared"


def _simulation_current_series(
    simulation_telemetry: Mapping[str, Any],
) -> dict[str, list[float]]:
    circuit = simulation_telemetry.get("circuit")
    if not isinstance(circuit, Mapping):
        return {"time_us": [], "current_A": []}
    history = circuit.get("current_history")
    if not isinstance(history, list):
        return {"time_us": [], "current_A": []}
    time_us: list[float] = []
    current_A: list[float] = []
    for sample in history:
        if not isinstance(sample, Mapping):
            continue
        try:
            time_us.append(float(sample["time_us"]))
            current_A.append(float(sample["current_A"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {"time_us": time_us, "current_A": current_A}


def _blocked_packet(
    *,
    status: str,
    declared_scope: str,
    device_name: str | None,
    reason: str,
    target_packet: Mapping[str, Any] | None = None,
    sim_series: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packet = {
        "status": status,
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "engineering_comparison_blocked_do_not_validate",
        "reason": reason,
        "first_principles_policy": {
            "experimental_waveform_used_as_drive": False,
            "experimental_waveform_used_as_fit": False,
            "reduced_model_used": False,
            "comparison_changes_solver_state": False,
        },
        "missing_for_acceptance": _missing_for_acceptance(
            temporal_coverage_fraction=0.0,
        ),
        "can_support_first_principles_acceptance": False,
    }
    if target_packet is not None:
        packet["target_packet"] = _target_summary(target_packet)
    if sim_series is not None:
        packet["simulation_series_counts"] = {
            "simulation_points": len(sim_series.get("time_us", ())),
        }
    return packet


def _target_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    summary = packet.get("summary", {})
    series = packet.get("digitized_series", [{}])[0]
    return {
        "task_id": packet.get("task_id"),
        "validation_scope": packet.get("validation_scope"),
        "device": packet.get("device"),
        "shot_id": packet.get("shot_id"),
        "series": series.get("name"),
        "packet_sha256": packet.get("packet_sha256"),
        "accepted_for_first_principles_validation": packet.get(
            "accepted_for_first_principles_validation",
        ),
        "can_seed_engineering_comparator": packet.get(
            "can_seed_engineering_comparator",
        ),
        "point_count": summary.get("point_count"),
        "time_range_us": [
            summary.get("time_min_us"),
            summary.get("time_max_us"),
        ],
        "current_range_kA": [
            summary.get("current_min_kA"),
            summary.get("current_max_kA"),
        ],
        "source_status": packet.get("source_status"),
    }


def _missing_for_acceptance(*, temporal_coverage_fraction: float) -> list[str]:
    missing = [
        "knowledge_reference_promotion_or_reviewed_extract",
        "per_point_time_uncertainty",
        "per_point_current_uncertainty",
        "independent_review",
        "accepted_output_mapping",
        "accepted_metric_and_tolerance",
        "uq_propagation",
        "negative_control",
        "certificate_gate",
    ]
    if temporal_coverage_fraction < 0.95:
        missing.append("whole_shot_temporal_coverage")
    return missing


def _peak_abs(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _fractional_error(predicted: float, target: float) -> float | None:
    if abs(target) <= 0.0:
        return None
    return float(abs(predicted - target) / abs(target))
