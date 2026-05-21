"""Fail-closed comparator and uncertainty-matrix packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.first_principles.channel_state import (
    AKEL_16KV_SHOT_MARKERS,
    looks_like_pf1000_akel_16kv_scope,
)

COMPARATOR_UQ_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "120-139,862-889",
        "role": "pf1000_akel_scalar_yield_timing_and_series_uncertainty_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "1018-1040,1042-1089,1214-1266",
        "role": "numerical_resolution_parameter_sensitivity_and_yield_uncertainty_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md"
        ),
        "lines": "32-53,337-351,390-427,518-526",
        "role": "time_energy_spectrum_detector_model_and_scatter_subtraction_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "175-204,277-284",
        "role": "direct_scattered_neutron_and_anisotropy_comparator_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "lines": "529-604",
        "role": "mechanism_timing_spectrum_anisotropy_error_bar_context",
    },
    {
        "path": "docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md",
        "lines": "224-289",
        "role": "local_queue_required_current_phase_neutron_temperature_uncertainty_channels",
    },
)

REQUIRED_OBSERVABLE_GROUPS = (
    "current_waveform",
    "current_dip",
    "phase_timing",
    "spatial_density",
    "magnetic_em_field",
    "temperature",
    "field_coupling",
    "neutron_scalar_yield",
    "neutron_timing",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_activation_response",
    "numerical_fidelity",
    "physics_fidelity",
)

REQUIRED_COMPARATOR_UQ_CHANNELS = (
    "accepted_same_scope_target_registry",
    "source_hashes_and_review_status",
    "output_field_mapping_by_observable",
    "unit_conversion_and_coordinate_mapping",
    "time_alignment_policy",
    "comparator_metric_by_observable",
    "comparator_tolerance_by_observable",
    "measurement_uncertainty_by_observable",
    "model_uncertainty_by_observable",
    "numerical_uncertainty_by_observable",
    "closure_sensitivity_uncertainty",
    "detector_response_uncertainty",
    "shot_to_shot_uncertainty_or_scope_rule",
    "uq_propagation_method",
    "pass_fail_rule_by_observable",
    "negative_control_cases",
    "requirement_links",
    "artifact_links_and_hashes",
    "independent_review_certificate",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "scalar_neutron_yield_uncertainty_text",
    "channel_timing_uncertainty_text",
    "shot_series_yield_range_text",
    "scintillator_detector_geometry_text",
    "activation_counter_calibration_text",
)

TRANSFER_RULE_REQUIRED_CHANNELS = (
    "source_scope_identity",
    "target_scope_identity",
    "changed_device_or_shot_parameters",
    "observable_transfer_equations_or_bounds",
    "metric_transfer_or_rejection_rule",
    "tolerance_transfer_or_rejection_rule",
    "uncertainty_inflation_rule",
    "review_certificate",
    "negative_test_cross_scope_promotion",
)

OTHER_SCOPE_SOURCE_GROUPS = (
    {
        "name": "hybrid_pic_fluid_resolution_sensitivity",
        "scope_mismatch": "2D axisymmetric compact/LLNL-like source, not PF-1000/Akel shot 12581.",
        "usable_for": "numerical resolution and closure-sensitivity matrix requirements",
    },
    {
        "name": "tof_tomography_detector_response",
        "scope_mismatch": "NNSS deuterium DPF detector setup, not PF-1000/Akel shot 12581.",
        "usable_for": "detector-forward spectrum comparator and scatter-subtraction requirements",
    },
    {
        "name": "pf1000_full_energy_anisotropy",
        "scope_mismatch": "PF-1000 full-energy 450-500 kJ source, not Akel 16 kV shot 12581.",
        "usable_for": "direct/scattered neutron and angular-comparator requirements",
    },
    {
        "name": "mjolnir_ma_class_mechanism_uq",
        "scope_mismatch": "MA/MJ-class MJOLNIR source, not PF-1000/Akel shot 12581.",
        "usable_for": "mechanism timing, spectrum, anisotropy, and error-bar requirements",
    },
)


def build_comparator_uq_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_channels: tuple[str, ...] | list[str] = (),
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting comparator/UQ matrix packet."""

    accepted = {str(channel) for channel in accepted_channels}
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted.update(target_channels)
    # Codex S9 P1-1: only the Akel 16 kV / shot-12581 revision receives the
    # Akel text-supported channels (non-acceptance reference).  Every other
    # scope -- including the full-energy scope pf1000_full_energy_27_to_40_kv --
    # gets an empty set (fail-closed; selected-scope-only until KR supplies
    # selected-scope records).
    text_supported = (
        set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
        if looks_like_pf1000_akel_16kv_scope(declared_scope, device_name)
        else set()
    )
    missing = set(REQUIRED_COMPARATOR_UQ_CHANNELS) - accepted
    missing.update(REQUIRED_COMPARATOR_UQ_CHANNELS)

    return {
        "status": "blocked_comparator_uq_matrix_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_comparator_or_uq_acceptance",
        "acceptance_gate": (
            "text_uncertainty_other_scope_sensitivity_and_partial_targets_cannot_"
            "support_comparator_acceptance_until_same_scope_targets_output_mapping_"
            "metrics_tolerances_uq_propagation_reviews_and_negative_controls_pass"
        ),
        "required_observable_groups": list(REQUIRED_OBSERVABLE_GROUPS),
        "required_channels": list(REQUIRED_COMPARATOR_UQ_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "comparator_uq_channel_status": _channel_statuses(
            required_channels=REQUIRED_COMPARATOR_UQ_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "observable_group_status": _observable_group_statuses(accepted),
        "upstream_packet_statuses": _upstream_statuses(upstream_packets),
        "upstream_acceptance_gate": _upstream_acceptance_gate(upstream_packets),
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "cross_scope_policy": {
            "status": "blocked_without_reviewed_transfer_rule",
            "required_transfer_rule_channels": list(TRANSFER_RULE_REQUIRED_CHANNELS),
            "other_scope_sources_usable_for": "requirements_or_schema_only",
            "can_use_other_scope_for_acceptance": False,
        },
        "validation_target_scope_decisions": target_decisions,
        "source_references": list(COMPARATOR_UQ_SOURCE_REFS),
        "validation_target_count": len(validation_targets),
        "can_support_comparator_acceptance": False,
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
                "decision": "not_accepted_comparator_uq_status",
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
        if observable in REQUIRED_OBSERVABLE_GROUPS:
            accepted.add(f"accepted_{observable}_target")
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_comparator_uq_target_channel",
            })
        else:
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "ignored_unmapped_comparator_observable",
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
            statuses[channel] = "accepted_comparator_uq"
        elif channel in text_supported:
            statuses[channel] = "text_supported_reference_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _observable_group_statuses(accepted: set[str]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for group in REQUIRED_OBSERVABLE_GROUPS:
        target_channel = f"accepted_{group}_target"
        statuses[group] = {
            "accepted_target_present": target_channel in accepted,
            "required_target_channel": target_channel,
            "decision": "blocked_until_full_comparator_uq_channels_pass",
        }
    return statuses


def _upstream_statuses(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    for name, packet in (upstream_packets or {}).items():
        statuses[str(name)] = (
            None if packet.get("status") is None else str(packet["status"])
        )
    return statuses


def _upstream_acceptance_gate(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    statuses = _upstream_statuses(upstream_packets)
    blocking = {
        name: status
        for name, status in statuses.items()
        if status is None or not _accepted_status(status)
    }
    return {
        "status": "blocked_by_upstream_packets" if blocking else "ready",
        "blocking_upstream_packets": blocking,
        "acceptance_rule": "all_upstream_packets_must_be_accepted_not_candidate_or_blocked",
    }


def _accepted_status(status: str) -> bool:
    return status.startswith("accepted") or status in {"ready", "passed"}


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
        if looks_like_pf1000_akel_16kv_scope(declared_scope, device_name):
            return "akel" in haystack and any(
                marker in haystack for marker in AKEL_16KV_SHOT_MARKERS
            )
    return False


def _normalized_scope(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
