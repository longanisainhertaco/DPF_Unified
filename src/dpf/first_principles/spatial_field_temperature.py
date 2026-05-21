"""Fail-closed spatial, field, and temperature evidence packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.first_principles.channel_state import (
    AKEL_16KV_SHOT_MARKERS,
    looks_like_pf1000_akel_16kv_scope,
)

SPATIAL_FIELD_TEMPERATURE_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "585-607",
        "role": "pf1000_akel_lee_output_density_geometry_scalars",
    },
    {
        "path": (
            "KnowledgeReference/"
            "sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md"
        ),
        "lines": "27-33,129-176",
        "role": "pf1000_density_diagnostic_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        ),
        "lines": "1459-1522",
        "role": "pf1000_magnetic_probe_and_pcs_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md"
        ),
        "lines": "21-65,180-195",
        "role": "pf1000_density_imaging_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "optical-spectroscopy-of-freepropagating-plasma-and-its-interaction-with-tungsten-targets-i-3a20181e.md"
        ),
        "lines": "119-160",
        "role": "pf1000_spectroscopy_temperature_limitation_other_scope",
    },
    {
        "path": "docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md",
        "lines": "266-289",
        "role": "spatial_temperature_and_uncertainty_queue_status",
    },
)

REQUIRED_SPATIAL_FIELD_TEMPERATURE_CHANNELS = (
    "accepted_same_scope_density_history",
    "density_diagnostic_geometry",
    "density_registration_and_interpolation",
    "density_uncertainty",
    "accepted_same_scope_magnetic_field_history",
    "accepted_same_scope_electric_field_history",
    "field_probe_geometry_and_calibration",
    "field_uncertainty",
    "accepted_same_scope_electron_temperature_history",
    "accepted_same_scope_ion_temperature_or_distribution",
    "temperature_diagnostic_model",
    "temperature_uncertainty",
    "output_field_mapping",
    "comparator_metric_and_tolerance",
    "source_review_certificate",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "lee_output_maximum_pinch_density_scalar",
    "lee_output_pinch_radius_scalar",
    "lee_output_pinch_length_scalar",
    "lee_output_velocity_scalars",
)

BLOCKING_SPATIAL_FIELD_TEMPERATURE_CHANNELS = (
    "accepted_same_scope_density_history",
    "density_uncertainty",
    "accepted_same_scope_magnetic_field_history",
    "accepted_same_scope_electric_field_history",
    "field_uncertainty",
    "accepted_same_scope_electron_temperature_history",
    "accepted_same_scope_ion_temperature_or_distribution",
    "temperature_uncertainty",
    "output_field_mapping",
    "comparator_metric_and_tolerance",
    "source_review_certificate",
)

TRANSFER_RULE_REQUIRED_CHANNELS = (
    "source_scope_identity",
    "target_scope_identity",
    "changed_device_or_shot_parameters",
    "observable_transfer_equations_or_bounds",
    "diagnostic_response_transfer_bounds",
    "uncertainty_inflation_rule",
    "review_certificate",
    "negative_test_cross_scope_promotion",
)

OTHER_SCOPE_SOURCE_GROUPS = (
    {
        "name": "pf1000_interferometry_density_other_campaign",
        "scope_mismatch": "PF-1000 interferometry shot uses different pressure/yield/scope than Akel 16 kV shot 12581.",
        "usable_for": "density diagnostic schema and registration requirements",
    },
    {
        "name": "pf1000_pcs_magnetic_probe_other_campaign",
        "scope_mismatch": "PCS magnetic-probe shots are 20-27 kV full-energy PF-1000, not Akel 16 kV.",
        "usable_for": "magnetic-field and current-density diagnostic requirements",
    },
    {
        "name": "pf1000_spectroscopy_other_campaign",
        "scope_mismatch": "Spectroscopy source is 21-27 kV PF-1000 plasma stream/target work, not Akel shot 12581.",
        "usable_for": "density/temperature diagnostic limitations and spectroscopy schema",
    },
)


def build_spatial_field_temperature_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_channels: tuple[str, ...] | list[str] = (),
    same_scope_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting spatial/field/temperature evidence packet."""

    accepted = {str(channel) for channel in accepted_channels}
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted.update(target_channels)
    if looks_like_pf1000_akel_16kv_scope(declared_scope, device_name):
        # Akel 16 kV / shot-12581 revision: Akel Lee-output text scalars are
        # non-acceptance engineering reference only.
        text_supported = set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
    else:
        # Codex S9 P1-1: every other scope -- including the full-energy scope
        # pf1000_full_energy_27_to_40_kv -- receives no Akel reference
        # channels (fail-closed; selected-scope-only until KR supplies records).
        text_supported = set()

    missing = set(REQUIRED_SPATIAL_FIELD_TEMPERATURE_CHANNELS) - accepted
    missing.update(BLOCKING_SPATIAL_FIELD_TEMPERATURE_CHANNELS)

    return {
        "status": "blocked_spatial_field_temperature_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_spatial_field_temperature_acceptance",
        "acceptance_gate": (
            "lee_output_scalars_and_other_scope_diagnostics_cannot_support_"
            "spatial_field_temperature_validation_until_same_scope_fields_"
            "diagnostic_models_uncertainty_and_review_pass"
        ),
        "required_channels": list(REQUIRED_SPATIAL_FIELD_TEMPERATURE_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "spatial_field_temperature_channel_status": _channel_statuses(
            required_channels=REQUIRED_SPATIAL_FIELD_TEMPERATURE_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "cross_scope_policy": {
            "status": "blocked_without_reviewed_transfer_rule",
            "required_transfer_rule_channels": list(TRANSFER_RULE_REQUIRED_CHANNELS),
            "other_scope_sources_usable_for": "requirements_or_schema_only",
            "can_use_other_scope_for_acceptance": False,
        },
        "validation_target_scope_decisions": target_decisions,
        "source_references": list(SPATIAL_FIELD_TEMPERATURE_SOURCE_REFS),
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
                "decision": "not_accepted_spatial_field_temperature_status",
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
        if observable in _OBSERVABLE_TO_CHANNEL:
            accepted.add(_OBSERVABLE_TO_CHANNEL[observable])
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_spatial_field_temperature_target_channel",
            })
        else:
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "ignored_unmapped_spatial_field_temperature_observable",
            })
    return accepted, decisions


_OBSERVABLE_TO_CHANNEL = {
    "density_spatial_history": "accepted_same_scope_density_history",
    "magnetic_field_history": "accepted_same_scope_magnetic_field_history",
    "electric_field_history": "accepted_same_scope_electric_field_history",
    "electron_temperature_history": "accepted_same_scope_electron_temperature_history",
    "ion_temperature_history": "accepted_same_scope_ion_temperature_or_distribution",
    "ion_distribution_history": "accepted_same_scope_ion_temperature_or_distribution",
}


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
            statuses[channel] = "accepted_spatial_field_temperature"
        elif channel in text_supported:
            statuses[channel] = "text_supported_reference_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


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
