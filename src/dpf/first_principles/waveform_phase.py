"""Fail-closed waveform and phase evidence packets for first-principles runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WAVEFORM_PHASE_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "108-142",
        "role": "pf1000_akel_waveform_diagnostic_and_timing_context",
    },
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "218-295,318-333",
        "role": "pf1000_akel_measured_current_figures_and_phase_scalars",
    },
    {
        "path": "docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md",
        "lines": "146-172,198-215,224-249",
        "role": "local_digitization_and_phase_queue_status",
    },
    {
        "path": (
            "KnowledgeReference/digitization/"
            "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
        ),
        "lines": "710-787",
        "role": "draft_waveform_packet_not_acceptance_evidence",
    },
    {
        "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
        "lines": "75-79",
        "role": "acceptance_gate_requirements",
    },
)

REQUIRED_WAVEFORM_PHASE_CHANNELS = (
    "accepted_digitized_current_waveform",
    "accepted_current_derivative_or_dip_trace",
    "time_axis_calibration",
    "current_axis_calibration",
    "per_point_waveform_uncertainty",
    "figure_source_hashes",
    "independent_review_accepted",
    "breakdown_to_derivative_dip_time",
    "derivative_dip_zero_time_definition",
    "current_dip_timing_and_depth",
    "axial_phase_timing",
    "radial_phase_timing",
    "pinch_phase_timing",
    "phase_semantics",
    "production_output_mapping",
    "comparator_metric_and_tolerance",
    "uq_budget",
)

PF1000_AKEL_TEXT_SUPPORTED_WAVEFORM_PHASE_CHANNELS = (
    "voltage_current_derivative_and_current_traces_measured",
    "derivative_dip_zero_time_definition",
    "breakdown_to_derivative_dip_time",
    "constriction_timing_text",
    "secondary_plasmoid_decay_timing_text",
    "channel_timing_uncertainty_text",
    "measured_current_figures_exist",
    "current_fit_through_dip_text",
    "peak_current_scalar",
    "pinch_current_scalar",
    "pinch_duration_scalar",
)

BLOCKING_WAVEFORM_PHASE_CHANNELS = (
    "accepted_digitized_current_waveform",
    "accepted_current_derivative_or_dip_trace",
    "per_point_waveform_uncertainty",
    "independent_review_accepted",
    "current_dip_timing_and_depth",
    "axial_phase_timing",
    "radial_phase_timing",
    "pinch_phase_timing",
    "production_output_mapping",
    "comparator_metric_and_tolerance",
    "uq_budget",
)

DRAFT_AKEL_FIG1_PACKET_STATUS = {
    "path": (
        "KnowledgeReference/digitization/"
        "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
    ),
    "measured_candidate_count": 294,
    "computed_candidate_count": 34,
    "overlay_rms_residual_px": 0.213455189,
    "review_status": "draft",
    "independent_review_count": 0,
    "accepted_for_validation": False,
}

REQUIRED_REVIEW_CHANNELS = (
    "source_pdf_hash",
    "figure_crop_hash",
    "axis_calibration_review",
    "series_extraction_review",
    "overlay_residual_review",
    "per_point_uncertainty_review",
    "independent_reviewer_metadata",
    "review_status_accepted",
)


def build_waveform_phase_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_channels: tuple[str, ...] | list[str] = (),
    same_scope_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting waveform and phase evidence packet."""

    accepted = {str(channel) for channel in accepted_channels}
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted.update(target_channels)
    if _looks_like_pf1000_akel_scope(declared_scope, device_name):
        text_supported = set(PF1000_AKEL_TEXT_SUPPORTED_WAVEFORM_PHASE_CHANNELS)
    else:
        text_supported = set()

    missing = set(REQUIRED_WAVEFORM_PHASE_CHANNELS) - accepted
    missing.update(BLOCKING_WAVEFORM_PHASE_CHANNELS)

    return {
        "status": "blocked_waveform_phase_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_current_waveform_or_phase_acceptance",
        "acceptance_gate": (
            "draft_or_text_waveform_evidence_cannot_support_validation_until "
            "digitized_traces_uncertainty_comparator_and_review_pass"
        ),
        "required_channels": list(REQUIRED_WAVEFORM_PHASE_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "waveform_phase_channel_status": _channel_statuses(
            required_channels=REQUIRED_WAVEFORM_PHASE_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "draft_digitization_packet_status": dict(DRAFT_AKEL_FIG1_PACKET_STATUS),
        "required_review_channels": list(REQUIRED_REVIEW_CHANNELS),
        "waveform_phase_target_policy": {
            "draft_digitization_can_seed_engineering_reference": True,
            "draft_digitization_can_support_acceptance": False,
            "text_timing_scalars_can_support_acceptance": False,
            "accepted_targets_must_match_declared_scope": True,
            "accepted_targets_require_per_point_uncertainty": True,
            "accepted_targets_require_independent_review": True,
        },
        "negative_test_policy": {
            "draft_waveform_promotion_rejection_required": True,
            "text_timing_scalar_promotion_rejection_required": True,
            "missing_per_point_uncertainty_rejection_required": True,
            "missing_independent_review_rejection_required": True,
            "mismatched_scope_waveform_rejection_required": True,
            "missing_output_mapping_or_tolerance_rejection_required": True,
        },
        "validation_target_scope_decisions": target_decisions,
        "source_references": list(WAVEFORM_PHASE_SOURCE_REFS),
        "same_scope_source_status": (
            None if same_scope_source is None else same_scope_source.get("status")
        ),
        "validation_target_count": len(validation_targets),
        "can_support_first_principles_acceptance": False,
    }


def _accepted_channels_from_targets(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    declared_scope: str,
    device_name: str | None,
) -> tuple[set[str], list[dict[str, Any]]]:
    accepted: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for target in validation_targets:
        status = str(target.get("status", ""))
        observable = str(target.get("observable", "")).strip()
        name = str(target.get("name", observable or "unnamed_target"))
        if status not in {
            "accepted_same_scope_source",
            "reviewed_same_scope_source",
            "accepted",
        }:
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "not_accepted_waveform_phase_status",
            })
            continue
        if not _target_scope_matches(target, declared_scope, device_name):
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "rejected_missing_or_mismatched_scope_metadata",
            })
            continue
        if observable in {
            "current_waveform",
            "current_dip",
            "phase_timing",
        }:
            accepted.add(_observable_to_channel(observable))
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_waveform_phase_target_channel",
            })
    return accepted, decisions


def _channel_statuses(
    *,
    required_channels: tuple[str, ...],
    accepted: set[str],
    text_supported: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in required_channels:
        if channel in accepted:
            statuses[channel] = "accepted_waveform_phase"
        elif channel in text_supported:
            statuses[channel] = "text_supported_reference_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _observable_to_channel(observable: str) -> str:
    return {
        "current_waveform": "accepted_digitized_current_waveform",
        "current_dip": "current_dip_timing_and_depth",
        "phase_timing": "phase_semantics",
    }[observable]


def _target_scope_matches(
    target: Mapping[str, Any],
    declared_scope: str,
    device_name: str | None,
) -> bool:
    target_scope = str(
        target.get("declared_scope")
        or target.get("validation_scope")
        or target.get("scope")
        or ""
    ).strip()
    if target_scope:
        return _normalized_scope(target_scope) == _normalized_scope(declared_scope)

    source_reference = target.get("source_reference")
    if isinstance(source_reference, Mapping):
        haystack = " ".join(
            str(source_reference.get(key, ""))
            for key in ("record_id", "role", "path")
        ).lower()
        if _looks_like_pf1000_akel_scope(declared_scope, device_name):
            return (
                "akel" in haystack
                and ("12581" in haystack or "16kv" in haystack or "16_kv" in haystack)
            )
    return False


def _normalized_scope(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _looks_like_pf1000_akel_scope(
    declared_scope: str,
    device_name: str | None,
) -> bool:
    haystack = f"{declared_scope} {device_name or ''}".lower()
    return "pf1000" in haystack or "pf-1000" in haystack or "akel" in haystack
