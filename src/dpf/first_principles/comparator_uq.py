"""Fail-closed comparator and uncertainty-matrix packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    accepted.update(_accepted_channels_from_targets(validation_targets))
    text_supported = (
        set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
        if _looks_like_pf1000_akel_scope(declared_scope, device_name)
        else set()
    )
    missing = set(REQUIRED_COMPARATOR_UQ_CHANNELS) - accepted
    missing.update(REQUIRED_COMPARATOR_UQ_CHANNELS)

    return {
        "status": "blocked_comparator_uq_matrix_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_comparator_or_uq_acceptance",
        "required_observable_groups": list(REQUIRED_OBSERVABLE_GROUPS),
        "required_channels": list(REQUIRED_COMPARATOR_UQ_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "upstream_packet_statuses": _upstream_statuses(upstream_packets),
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "source_references": list(COMPARATOR_UQ_SOURCE_REFS),
        "validation_target_count": len(validation_targets),
        "can_support_comparator_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _accepted_channels_from_targets(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
) -> set[str]:
    accepted: set[str] = set()
    for target in validation_targets:
        status = str(target.get("status", ""))
        if status not in {
            "accepted_same_scope_source",
            "reviewed_same_scope_source",
            "accepted",
        }:
            continue
        observable = str(target.get("observable", "")).strip()
        if observable in REQUIRED_OBSERVABLE_GROUPS:
            accepted.add(f"accepted_{observable}_target")
    return accepted


def _upstream_statuses(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    for name, packet in (upstream_packets or {}).items():
        statuses[str(name)] = (
            None if packet.get("status") is None else str(packet["status"])
        )
    return statuses


def _looks_like_pf1000_akel_scope(
    declared_scope: str,
    device_name: str | None,
) -> bool:
    haystack = f"{declared_scope} {device_name or ''}".lower()
    return "pf1000" in haystack or "pf-1000" in haystack or "akel" in haystack
