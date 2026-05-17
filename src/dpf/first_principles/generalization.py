"""Fail-closed generalization packets for first-principles DPF claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GENERALIZATION_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "88-89,133-135,172,713",
        "role": "general_dpf_claim_requires_second_scope_full_evidence_repeat",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md",
        "lines": "86-91,130-157",
        "role": "source_truth_index_second_scope_material_and_missing_same_scope_set",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md"
        ),
        "lines": "87-118,135-156",
        "role": "llnl_180ka_fully_kinetic_geometry_current_beam_and_yield_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "58-63,685-711,952-990,1018-1040,1083-1089,1240-1263",
        "role": "llnl_like_hybrid_pic_architecture_and_uncertainty_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "lines": "101-145,174-211,429-430,480-487,575-608,748-764",
        "role": "mjolnir_device_diagnostic_mhd_kinetic_and_mechanism_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "121-137,175-204,269-284,432-438",
        "role": "pf1000_full_energy_anisotropy_direct_scattered_and_tof_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md"
        ),
        "lines": "130-174",
        "role": "pf1000_full_energy_interferometry_density_and_pinch_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md"
        ),
        "lines": "46-55,64-78",
        "role": "faeton_i_second_device_voltage_yield_anisotropy_and_ion_beam_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md"
        ),
        "lines": "232-239,305-320,331-354,403-438,526-898,917-957,976-1006,1170-1202",
        "role": "soto_2010_cchen_device_configs_aggregate_targets_and_scaling_matrix",
    },
    {
        "path": "KnowledgeReference/original-research-f7894f85.md",
        "lines": "30-45,108-170,200-215,269-288,300-323,365-394",
        "role": "user_validated_ir_mpf100_second_scope_deck_and_neutron_context",
    },
    {
        "path": "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md",
        "lines": "38-41,66-91,103-147,161-200,210-237,291-295",
        "role": "user_validated_compact_chinese_dpf_second_scope_deck_and_tof_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
        ),
        "lines": "1196-1209,1382-1496,1697-1702,1857-1901,3367-3439",
        "role": "user_validated_historical_dpf_startup_diagnostic_design_scope",
    },
)

REQUIRED_GENERALIZATION_CHANNELS = (
    "accepted_primary_scope_certificate",
    "declared_second_scope",
    "second_scope_device_geometry",
    "second_scope_drive_waveform",
    "second_scope_startup_packet",
    "second_scope_power_port_packet",
    "second_scope_dimensionality_handoff_packet",
    "second_scope_physics_closure_packet",
    "second_scope_density_field_temperature_packet",
    "second_scope_neutron_authority_packet",
    "second_scope_detector_response_uq",
    "second_scope_comparator_uq_packet",
    "second_scope_numerical_fidelity_packet",
    "second_scope_certificate",
    "no_hidden_pf1000_akel_assumptions",
    "device_parameterization_schema",
    "scale_transition_or_nondimensionalization_review",
    "regression_against_primary_scope",
    "source_review_certificate",
    "cross_scope_negative_tests",
)

CANDIDATE_SECOND_SCOPES = (
    {
        "scope_id": "soto2010_cchen_pf400j_pf50j_speed_nanofocus_matrix",
        "device_family": "CCHEN PF-400J/PF-50J/SPEED/Nanofocus",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "machine_geometry_and_bank_configuration",
            "aggregate_peak_current_pressure_and_energy_context",
            "pf400j_pf50j_neutron_yield_density_line_density_and_energy_context",
            "pf400j_angular_distribution_context",
            "cross_device_drive_and_energy_density_scaling_matrix",
            "pinch_radius_density_magnetic_field_velocity_and_temperature_sanity_bounds",
        ),
        "blocks": (
            "table targets are not yet extracted into accepted typed evidence",
            "aggregate observations do not provide a complete same-shot time-history packet",
            "requires independent FP-1 through FP-14 packet chain and certificate for the selected second scope",
        ),
    },
    {
        "scope_id": "pf1000_full_energy_anisotropy_450_500kj_3p5torr",
        "device_family": "PF-1000",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "full_energy_operating_point",
            "interferometry_density_and_pinch_context",
            "tld_bonner_sphere_anisotropy",
            "silver_activation_reference",
            "tof_direct_scattered_separation_context",
            "vessel_transport_mcnp_context",
        ),
        "blocks": (
            "not the Akel 16 kV shot 12581 scope",
            "needs full FP-1 through FP-14 packet extraction and review",
        ),
    },
    {
        "scope_id": "faeton_i_100kv_second_device_scope",
        "device_family": "FAETON-I",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "100kv_direct_charged_operating_context",
            "current_sheath_and_voltage_context",
            "shot_table_current_factor_voltage_yield_context",
            "neutron_yield_and_anisotropy_context",
            "pmt_scintillator_neutron_energy_context",
            "fast_faraday_cup_ion_beam_context",
        ),
        "blocks": (
            "source currently leans on Lee-model comparisons rather than a complete first-principles certificate",
            "requires typed extraction into FP-1 through FP-14 gates and independent review",
        ),
    },
    {
        "scope_id": "ir_mpf_100_115kj_second_device_scope",
        "device_family": "IR-MPF-100",
        "source_status": "user_validated_source_target_material_not_acceptance",
        "source_supported_channels": (
            "144uf_40kv_115kj_bank",
            "1p224ma_theoretical_current_and_120nh_total_inductance",
            "mather_geometry_anode_cathode_insulator_deck",
            "current_voltage_hard_xray_and_activation_counter_context",
            "preliminary_neutron_yield_and_double_pinch_context",
        ),
        "blocks": (
            "measured waveforms are not digitized with uncertainties",
            "neutron data are scalar/preliminary and not mechanism separated",
            "requires full FP-1 through FP-14 packet extraction and review",
        ),
    },
    {
        "scope_id": "compact_chinese_dpf_400ka_second_device_scope",
        "device_family": "compact Mather-type DPF neutron source",
        "source_status": "user_validated_source_target_material_not_acceptance",
        "source_supported_channels": (
            "40uf_10_20kv_bank_and_400ka_current_context",
            "17mm_anode_40mm_outer_radius_8rod_cathode_geometry",
            "tof_neutron_detector_and_fwhm_context",
            "pressure_yield_and_voltage_threshold_context",
        ),
        "blocks": (
            "translation and visual table review still required for target extraction",
            "pressure-yield and current waveform curves are not digitized",
            "requires full FP-1 through FP-14 packet extraction and review",
        ),
    },
    {
        "scope_id": "llnl_180ka_kinetic_or_hybrid_reference",
        "device_family": "LLNL compact DPF-like",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "electrode_geometry_context",
            "180ka_current_context",
            "measured_high_energy_beam_context",
            "neutron_yield_scale_context",
            "hybrid_pic_resolution_and_closure_sensitivity_context",
        ),
        "blocks": (
            "hybrid source is 2D axisymmetric and LLNL-like",
            "public same-scope experimental packet is incomplete",
        ),
    },
    {
        "scope_id": "mjolnir_60kv_735kj_9torr_mechanism_scope",
        "device_family": "MJOLNIR",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "device_and_diagnostic_layout",
            "transmission_line_circuit_model_context",
            "mhd_to_kinetic_modeling_context",
            "60kv_735kj_operating_point",
            "neutron_pulse_shape_spectrum_and_anisotropy_context",
            "activation_detector_angular_ratio_context",
        ),
        "blocks": (
            "requires typed extraction into the same FP-1 through FP-14 gates",
            "requires review before use as a validation scope",
        ),
    },
    {
        "scope_id": "pf1000_akel_other_shot_or_pressure_series",
        "device_family": "PF-1000/Akel",
        "source_status": "candidate_requirement_material_not_acceptance",
        "source_supported_channels": (
            "shot_series_scalar_yield_context",
            "pressure_series_context",
            "detector_layout_context",
        ),
        "blocks": (
            "same machine does not prove device generality by itself",
            "needs shot-specific waveform, spatial, neutron, detector, and UQ packets",
        ),
    },
)

BLOCKING_UPSTREAM_STATUSES = (
    "blocked_first_principles_certificate_not_available",
    "blocked_comparator_uq_matrix_not_available",
    "blocked_mechanism_separated_neutron_authority_not_available",
    "blocked_spatial_field_temperature_packet_not_available",
    "blocked_waveform_phase_packet_not_available",
    "blocked_same_scope_source_packet_not_available",
    "candidate_engineering_closure_packet_not_validation",
    "candidate_engineering_dimensionality_handoff_not_validation",
    "candidate_engineering_power_port_not_validation",
)

SECOND_SCOPE_REQUIRED_GATE_IDS = (
    "FP-1",
    "FP-2",
    "FP-3",
    "FP-4",
    "FP-5",
    "FP-6",
    "FP-7",
    "FP-8",
    "FP-9",
    "FP-10",
    "FP-11",
    "FP-12",
    "FP-13",
    "FP-14",
)


def build_generalized_dpf_machine_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting packet for a generalized DPF-machine claim."""

    accepted = {str(channel) for channel in accepted_channels}
    missing = set(REQUIRED_GENERALIZATION_CHANNELS) - accepted
    missing.update(REQUIRED_GENERALIZATION_CHANNELS)
    upstream_statuses = _upstream_statuses(upstream_packets)
    upstream_blockers = {
        name: status
        for name, status in upstream_statuses.items()
        if _status_blocks_generalization(status)
    }

    return {
        "status": "blocked_generalized_dpf_machine_path_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_claim_general_dpf_first_principles_tool",
        "release_label": "single_scope_engineering_candidate_not_generalized",
        "required_channels": list(REQUIRED_GENERALIZATION_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "generalization_channel_status": _generalization_channel_statuses(
            accepted=accepted,
            missing=missing,
        ),
        "claim_policy": {
            "single_scope_is_not_generalized": True,
            "accepted_primary_scope_certificate_required": True,
            "accepted_second_scope_certificate_required": True,
            "second_scope_must_repeat_full_fp_1_through_fp_14_chain": True,
            "cross_scope_transfer_or_scale_review_required": True,
        },
        "required_second_scope_gate_ids": list(SECOND_SCOPE_REQUIRED_GATE_IDS),
        "candidate_second_scopes": [
            _coerce_candidate_scope(scope) for scope in CANDIDATE_SECOND_SCOPES
        ],
        "candidate_second_scope_decisions": [
            _candidate_second_scope_decision(scope) for scope in CANDIDATE_SECOND_SCOPES
        ],
        "upstream_packet_statuses": upstream_statuses,
        "upstream_generalization_blockers": upstream_blockers,
        "upstream_acceptance_gate": _upstream_acceptance_gate(upstream_statuses),
        "source_references": list(GENERALIZATION_SOURCE_REFS),
        "can_claim_generalized_dpf_machine": False,
        "can_support_first_principles_acceptance": False,
    }


def _upstream_statuses(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    for name, packet in (upstream_packets or {}).items():
        statuses[str(name)] = (
            None if packet.get("status") is None else str(packet["status"])
        )
    return statuses


def _generalization_channel_statuses(
    *,
    accepted: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_GENERALIZATION_CHANNELS:
        if channel in accepted:
            statuses[channel] = "accepted_generalization_channel"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _upstream_acceptance_gate(
    upstream_statuses: Mapping[str, str | None],
) -> dict[str, Any]:
    blockers = {
        name: status
        for name, status in upstream_statuses.items()
        if _status_blocks_generalization(status)
    }
    return {
        "status": "blocked_by_primary_scope_packets" if blockers else "ready",
        "blocking_upstream_packets": blockers,
        "acceptance_rule": (
            "primary_scope_certificate_and_all_prior_packets_must_be_accepted_"
            "before_generalization_can_start"
        ),
    }


def _status_blocks_generalization(status: str | None) -> bool:
    if status is None:
        return True
    normalized = status.strip().lower()
    if normalized in BLOCKING_UPSTREAM_STATUSES:
        return True
    return (
        normalized.startswith("blocked")
        or normalized.startswith("candidate")
        or normalized == "not_validation"
    )


def _coerce_candidate_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope_id": str(scope["scope_id"]),
        "device_family": str(scope["device_family"]),
        "source_status": str(scope["source_status"]),
        "source_supported_channels": list(scope["source_supported_channels"]),
        "blocks": list(scope["blocks"]),
    }


def _candidate_second_scope_decision(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope_id": str(scope["scope_id"]),
        "device_family": str(scope["device_family"]),
        "decision": "candidate_requirement_material_not_acceptance",
        "required_gate_ids": list(SECOND_SCOPE_REQUIRED_GATE_IDS),
        "must_write_independent_certificate": True,
        "source_supported_channels": list(scope["source_supported_channels"]),
        "blocks": list(scope["blocks"]),
    }
