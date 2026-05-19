"""Fail-closed same-scope source packets for first-principles DPF claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SAME_SCOPE_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "108-142,256-333,862-889",
        "role": "pf1000_akel_16kv_current_yield_reference_candidate",
    },
    {
        "path": (
            "KnowledgeReference/"
            "sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md"
        ),
        "lines": "129-131,162-176",
        "role": "pf1000_spatial_density_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "121-137,175-177,269-275",
        "role": "pf1000_neutron_transport_detector_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "1220-1266",
        "role": "hybrid_pic_architecture_order_of_magnitude_other_scope",
    },
)

REQUIRED_SAME_SCOPE_CHANNELS = (
    "declared_validation_scope",
    "device_geometry_and_electrode_dimensions",
    "bank_circuit_drive",
    "gas_species_pressure_temperature",
    "accepted_digitized_current_waveform",
    "startup_breakdown_preionization",
    "density_spatial_history",
    "em_field_history",
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
    "neutron_scalar_yield",
    "neutron_timing_history",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response_and_calibration",
    "uncertainty_budget",
    "source_review_certificate",
    "cross_scope_transfer_rule_or_rejection_tests",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "device_geometry_and_electrode_dimensions",
    "bank_circuit_drive",
    "gas_species_pressure_temperature",
    "peak_current_scalar",
    "pinch_geometry_lee_output",
    "neutron_scalar_yield",
    "neutron_detector_layout_text",
    "timing_uncertainty_text",
)

BLOCKING_SAME_SCOPE_CHANNELS = (
    "accepted_digitized_current_waveform",
    "startup_breakdown_preionization",
    "density_spatial_history",
    "em_field_history",
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
    "neutron_timing_history",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response_and_calibration",
    "uncertainty_budget",
    "source_review_certificate",
    "cross_scope_transfer_rule_or_rejection_tests",
)

TRANSFER_RULE_REQUIRED_CHANNELS = (
    "source_scope_identity",
    "target_scope_identity",
    "changed_device_or_shot_parameters",
    "observable_transfer_equations_or_bounds",
    "uncertainty_inflation_rule",
    "review_certificate",
    "negative_test_cross_scope_promotion",
)

OTHER_SCOPE_SOURCE_GROUPS = (
    {
        "name": "pf1000_interferometry_density_other_campaign",
        "scope_mismatch": "PF-1000 interferometry shot is not the Akel 16 kV, 1.05-1.2 Torr shot set.",
        "usable_for": "diagnostic requirement and density-history schema",
    },
    {
        "name": "pf1000_anisotropy_detector_other_campaign",
        "scope_mismatch": "PF-1000 anisotropy work is full-scale 450-500 kJ at 3.5 Torr, not Akel 16 kV.",
        "usable_for": "detector-response and neutron-transport requirement schema",
    },
    {
        "name": "hybrid_pic_non_pf1000_order_of_magnitude",
        "scope_mismatch": "Hybrid PIC/fluid source is 2-D/non-hollow and order-of-magnitude, not PF-1000/Akel.",
        "usable_for": "architecture and closure-gap requirements",
    },
)


def build_same_scope_source_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_same_scope_channels: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return a non-promoting packet describing same-scope source availability."""

    accepted = {str(channel) for channel in accepted_same_scope_channels}
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted.update(target_channels)
    declared = bool(str(declared_scope).strip()) and declared_scope != "not_declared"
    if declared:
        accepted.add("declared_validation_scope")

    if _looks_like_pf1000_akel_scope(declared_scope, device_name):
        text_supported = set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
    else:
        text_supported = set()

    missing = set(REQUIRED_SAME_SCOPE_CHANNELS) - accepted
    missing.update(BLOCKING_SAME_SCOPE_CHANNELS)
    if not declared:
        missing.add("declared_validation_scope")

    return {
        "status": "blocked_same_scope_source_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_promote_whole_shot_first_principles_claim",
        "scope_policy": (
            "single_declared_scope_only_no_cross_device_shot_or_configuration_mix "
            "without_reviewed_transfer_rule"
        ),
        "required_channels": list(REQUIRED_SAME_SCOPE_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "accepted_same_scope_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "same_scope_channel_status": _channel_statuses(
            required_channels=REQUIRED_SAME_SCOPE_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "same_scope_target_policy": {
            "text_supported_channels_can_seed_engineering_reference": True,
            "text_supported_channels_can_support_acceptance": False,
            "accepted_targets_must_match_declared_scope": True,
            "accepted_targets_must_include_review_certificate": True,
            "cross_scope_targets_require_reviewed_transfer_rule": True,
        },
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "cross_scope_policy": {
            "status": "blocked_without_reviewed_transfer_rule",
            "required_transfer_rule_channels": list(TRANSFER_RULE_REQUIRED_CHANNELS),
            "other_scope_sources_usable_for": "requirements_or_schema_only",
            "can_use_other_scope_for_acceptance": False,
        },
        "acceptance_gate": (
            "text_supported_pf1000_akel_scalars_and_other_scope_diagnostics_"
            "cannot_support_whole_shot_acceptance_until_all_same_scope_targets_"
            "current_startup_density_fields_temperatures_neutrons_detector_uq_"
            "review_and_cross_scope_rejection_tests_pass"
        ),
        "negative_test_policy": {
            "text_reference_promotion_rejection_required": True,
            "other_scope_diagnostic_promotion_rejection_required": True,
            "mismatched_shot_or_pressure_rejection_required": True,
            "missing_review_certificate_rejection_required": True,
            "missing_uncertainty_budget_rejection_required": True,
            "cross_scope_without_transfer_rule_rejection_required": True,
        },
        "validation_target_scope_decisions": target_decisions,
        "source_references": list(SAME_SCOPE_SOURCE_REFS),
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
                "decision": "not_accepted_same_scope_status",
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
        if observable:
            evidence_type = str(target.get("evidence_type", "")).strip()
            if evidence_type == "lee_model_output" and observable in BLOCKING_SAME_SCOPE_CHANNELS:
                # Lee model outputs are NOT independent measurements.
                # They cannot satisfy blocking same-scope channels.
                # WP-N7 §6.1, N7-NEG-10.
                decisions.append({
                    "target": name,
                    "observable": observable,
                    "status": status,
                    "decision": "rejected_lee_model_output_not_independent_measurement",
                    "evidence_type": evidence_type,
                })
                continue
            accepted.add(observable)
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_same_scope_target_channel",
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
            statuses[channel] = "accepted_same_scope"
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
