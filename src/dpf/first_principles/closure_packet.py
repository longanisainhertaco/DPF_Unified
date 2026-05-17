"""Fail-closed physics-closure packets for first-principles DPF runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CLOSURE_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "431-619,1210-1280",
        "role": "hybrid_closure_equations_and_limitations",
    },
    {
        "path": (
            "KnowledgeReference/"
            "unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md"
        ),
        "lines": "277-293,333-369",
        "role": "eos_radiation_material_and_pinch_scope",
    },
    {
        "path": "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json",
        "lines": "57-62",
        "role": "pf1000_two_temperature_heat_flux_ionization_equation_structure",
    },
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "2996-3020,general_formulary_support",
        "role": "plasma_formula_units_transport_thermal_equilibration_radiation_support",
    },
)

REQUIRED_EFFECTS = (
    "eos_thermodynamics",
    "ionization_charge_state",
    "single_two_temperature_energy",
    "electrical_thermal_transport",
    "radiation_losses",
    "impurity_electrode_ablation",
    "hall_flr_kinetic_scope",
    "three_d_instabilities",
    "restrike_anomalous_resistance",
    "beam_target_coupling",
)

REQUIRED_CLOSURE_PACKET_CHANNELS = (
    "effect_id",
    "classification",
    "source_equations_or_bound",
    "symbol_map",
    "units",
    "validity_regime",
    "implementation_reference",
    "verification_tests",
    "sensitivity_or_uq",
    "nondominance_or_claim_impact",
    "review_status",
)


def build_physics_closure_packet(
    *,
    include_hall: bool,
    electron_energy_present: bool,
    kinetic_yield_present: bool,
    collisions_enabled: bool,
    electron_heat_flux_present: bool = False,
    electron_equilibration_audit_present: bool = False,
    ionization_charge_state_present: bool = False,
    source_backed_transport_present: bool = False,
    dimensionality: Mapping[str, Any] | None = None,
    community_formula_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-promoting closure status for all required physics effects."""
    effects = {
        "eos_thermodynamics": _effect(
            "blocked",
            implemented=False,
            missing=("qEOS_or_tabular_EOS", "low_density_validity", "verification_tests"),
            claim_impact="whole_shot_thermodynamics_and_pressure_authority_blocked",
        ),
        "ionization_charge_state": _effect(
            "candidate" if ionization_charge_state_present else "blocked",
            implemented=ionization_charge_state_present,
            missing=(
                "accepted_ionization_recombination_model"
                if ionization_charge_state_present
                else "ionization_recombination_model",
                "accepted_charge_state_transport"
                if ionization_charge_state_present
                else "charge_state_transport",
                "startup_link",
                "accepted_neutral_particle_source_coupling"
                if ionization_charge_state_present
                else "neutral_particle_source_coupling",
                "accepted_conductivity_eos_charge_state_feedback"
                if ionization_charge_state_present
                else "conductivity_eos_charge_state_feedback",
            ),
            claim_impact="breakdown_sheath_and_resistivity_authority_blocked",
        ),
        "single_two_temperature_energy": _effect(
            "candidate" if electron_energy_present else "blocked",
            implemented=electron_energy_present,
            missing=(
                "accepted_electron_heat_flux"
                if electron_heat_flux_present
                else "electron_heat_flux",
                "accepted_electron_ion_collisional_coupling"
                if electron_equilibration_audit_present
                else "electron_ion_collisional_coupling",
                "temperature_diagnostic_validation",
                "hall_pressure_sensitivity_uq",
            ),
            claim_impact="hall_pressure_and_yield_authority_blocked",
        ),
        "electrical_thermal_transport": _effect(
            "candidate",
            implemented=True,
            missing=(
                "accepted_transport_validity_regime"
                if source_backed_transport_present
                else "transport_validity_regime",
                "accepted_ohmic_cfl_nondominance"
                if source_backed_transport_present
                else "ohmic_cfl_nondominance",
                "accepted_thermal_conduction_closure"
                if electron_heat_flux_present
                else "thermal_conduction_closure",
                "sensitivity_uq",
            ),
            claim_impact="field_current_coupling_remains_engineering_candidate",
        ),
        "radiation_losses": _effect(
            "blocked",
            implemented=False,
            missing=("loss_model_or_bound", "opacity_or_diffusion_decision", "energy_ledger"),
            claim_impact="radiating_gas_or_high_z_claims_blocked",
        ),
        "impurity_electrode_ablation": _effect(
            "blocked",
            implemented=False,
            missing=("ablation_source_model", "impurity_transport", "electrode_material_uq"),
            claim_impact="waveform_pinch_radiation_neutron_impurity_effects_blocked",
        ),
        "hall_flr_kinetic_scope": _effect(
            "candidate" if include_hall else "blocked",
            implemented=include_hall,
            missing=(
                "electron_temperature_authority",
                "flr_validity_or_handoff",
                "kinetic_interval_review",
            ),
            claim_impact="late_pinch_and_acceleration_authority_blocked",
        ),
        "three_d_instabilities": _effect(
            "candidate",
            implemented=True,
            missing=("accepted_m_mode_evidence", "same_scope_3d_instability_packet"),
            claim_impact="kink_fragmentation_and_lifetime_authority_blocked",
        ),
        "restrike_anomalous_resistance": _effect(
            "blocked",
            implemented=False,
            missing=("restrike_model", "anomalous_resistivity_model", "post_pinch_scope"),
            claim_impact="current_dip_and_post_pinch_claims_blocked",
        ),
        "beam_target_coupling": _effect(
            "candidate" if kinetic_yield_present else "blocked",
            implemented=kinetic_yield_present,
            missing=(
                "mechanism_separation",
                "ion_distribution_transport_stopping",
                "spectrum_anisotropy_detector_response",
                "uq",
            ),
            claim_impact="total_neutron_yield_authority_blocked",
        ),
    }
    if not collisions_enabled:
        effects["electrical_thermal_transport"]["missing_channels"].append(
            "accepted_collision_parameterization"
        )
    missing_effects = [
        key
        for key, record in effects.items()
        if record["can_support_first_principles_acceptance"] is False
    ]
    active_candidate_closures = [
        key
        for key, record in effects.items()
        if record["status"] == "candidate" and record["implemented"]
    ]
    return {
        "status": "candidate_engineering_closure_packet_not_validation",
        "decision": "do_not_promote_without_complete_physics_closure_matrix",
        "required_effects": list(REQUIRED_EFFECTS),
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "effects": effects,
        "closure_matrix_status_by_effect": {
            key: record["status"] for key, record in effects.items()
        },
        "closure_effect_status": _closure_effect_statuses(effects),
        "missing_or_unaccepted_effects": missing_effects,
        "candidate_runtime_channels": _candidate_runtime_channels(
            include_hall=include_hall,
            ionization_charge_state_present=ionization_charge_state_present,
            source_backed_transport_present=source_backed_transport_present,
            electron_energy_present=electron_energy_present,
            electron_heat_flux_present=electron_heat_flux_present,
            electron_equilibration_audit_present=electron_equilibration_audit_present,
            kinetic_yield_present=kinetic_yield_present,
            collisions_enabled=collisions_enabled,
            dimensionality=dimensionality,
            community_formula_audit=community_formula_audit,
        ),
        "active_candidate_closures": active_candidate_closures,
        "community_formula_audit": _community_formula_audit_packet(
            community_formula_audit
        ),
        "community_formula_audit_policy": {
            "optional_audit_can_support_acceptance": False,
            "local_source_truth_remains_required": True,
            "missing_or_failed_audit_blocks_engineering_run": False,
            "outside_tolerance_audit_requires_review": True,
        },
        "active_closure_policy": {
            "candidate_closures_can_run_engineering_cases": True,
            "candidate_closures_can_support_acceptance": False,
            "active_candidate_closures": active_candidate_closures,
            "required_promotion_path": (
                "each_active_or_bounded_out_effect_needs_source_equations_symbol_"
                "map_units_validity_implementation_tests_sensitivity_uq_claim_"
                "impact_and_review"
            ),
        },
        "dimensionality_acceptance_gate": _dimensionality_acceptance_gate(
            dimensionality
        ),
        "acceptance_gate": (
            "candidate_transport_ohm_electron_energy_hall_instability_and_yield_"
            "closures_cannot_support_physics_acceptance_until_every_required_"
            "effect_is_implemented_validated_or_bounded_out_with_source_equations_"
            "units_validity_tests_sensitivity_uq_claim_impact_hashes_and_review"
        ),
        "negative_test_policy": {
            "missing_effect_rejection_required": True,
            "candidate_closure_promotion_rejection_required": True,
            "hall_pressure_without_electron_temperature_rejection_required": True,
            "total_yield_without_mechanism_separation_rejection_required": True,
            "radiation_or_ablation_absent_claim_rejection_required": True,
            "anomalous_resistance_or_restrike_claim_rejection_required": True,
            "closure_sensitivity_uq_missing_rejection_required": True,
        },
        "source_references": list(CLOSURE_SOURCE_REFS),
        "dimensionality_status": None if dimensionality is None else dimensionality.get("status"),
        "source_model_limitations": (
            []
            if dimensionality is None
            else list(dimensionality.get("source_model_limitations", ()))
        ),
        "can_support_first_principles_acceptance": False,
    }


def _effect(
    status: str,
    *,
    implemented: bool,
    missing: tuple[str, ...],
    claim_impact: str,
) -> dict[str, Any]:
    missing_set = set(missing)
    return {
        "status": status,
        "implemented": implemented,
        "classification": status if status != "candidate" else "candidate_only",
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "missing_channels": list(missing),
        "channel_status": _effect_channel_statuses(
            implemented=implemented,
            missing=missing_set,
        ),
        "claim_impact": claim_impact,
        "review_status": "not_reviewed_for_acceptance",
        "can_support_first_principles_acceptance": False,
    }


def _effect_channel_statuses(
    *,
    implemented: bool,
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_CLOSURE_PACKET_CHANNELS:
        if channel in {"effect_id", "classification"}:
            statuses[channel] = "present_non_accepting_metadata"
        elif channel == "implementation_reference" and implemented:
            statuses[channel] = "candidate_implementation_reference_not_acceptance"
        elif channel == "review_status":
            statuses[channel] = "not_reviewed_for_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "missing_or_unaccepted"
    return statuses


def _closure_effect_statuses(
    effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "status": str(record["status"]),
            "classification": str(record["classification"]),
            "implemented": bool(record["implemented"]),
            "claim_impact": str(record["claim_impact"]),
            "missing_channels": list(record["missing_channels"]),
            "review_status": str(record["review_status"]),
            "can_support_first_principles_acceptance": False,
        }
        for key, record in effects.items()
    }


def _dimensionality_acceptance_gate(
    dimensionality: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if dimensionality is None:
        return {
            "status": "blocked_dimensionality_packet_missing",
            "blocking_source_model_limitations": [],
        }
    limitations = list(dimensionality.get("source_model_limitations", ()))
    return {
        "status": "blocked_by_dimensionality_or_handoff_packet",
        "dimensionality_status": dimensionality.get("status"),
        "blocking_source_model_limitations": limitations,
        "can_accept_closure_without_dimensionality_acceptance": False,
    }


def _candidate_runtime_channels(
    *,
    include_hall: bool,
    ionization_charge_state_present: bool,
    source_backed_transport_present: bool,
    electron_energy_present: bool,
    electron_heat_flux_present: bool,
    electron_equilibration_audit_present: bool,
    kinetic_yield_present: bool,
    collisions_enabled: bool,
    dimensionality: Mapping[str, Any] | None,
    community_formula_audit: Mapping[str, Any] | None,
) -> list[str]:
    channels: set[str] = set()
    channels.add("candidate_electrical_transport_source_terms")
    if include_hall:
        channels.add("candidate_hall_term_enabled")
    if ionization_charge_state_present:
        channels.add("candidate_ionization_charge_state_transport")
    if source_backed_transport_present:
        channels.add("candidate_source_backed_partial_ionized_conductivity")
    if electron_energy_present:
        channels.add("candidate_electron_energy_source_terms")
    if electron_heat_flux_present:
        channels.add("candidate_braginskii_electron_heat_flux")
    if electron_equilibration_audit_present:
        channels.add("candidate_electron_ion_equilibration_audit")
    if kinetic_yield_present:
        channels.add("candidate_kinetic_yield_history")
    if collisions_enabled:
        channels.add("candidate_collision_stage_enabled")
    if _community_formula_audit_available(community_formula_audit):
        channels.add("candidate_plasmapy_community_formula_audit")
    if dimensionality is not None:
        channels.add("candidate_dimensionality_packet_linked")
        for channel in dimensionality.get("candidate_runtime_channels", ()):
            if str(channel).startswith("candidate_"):
                channels.add(f"dimensionality_{channel}")
    return sorted(channels)


def _community_formula_audit_packet(
    community_formula_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if community_formula_audit is None:
        return {
            "status": "community_formula_audit_not_requested",
            "can_support_first_principles_acceptance": False,
        }
    packet = dict(community_formula_audit)
    packet["can_support_first_principles_acceptance"] = False
    return packet


def _community_formula_audit_available(
    community_formula_audit: Mapping[str, Any] | None,
) -> bool:
    if community_formula_audit is None:
        return False
    status = str(community_formula_audit.get("status", ""))
    return status.startswith("community_formula_audit_") and "unavailable" not in status
