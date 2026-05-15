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
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "general_formulary_support",
        "role": "plasma_formula_units_transport_radiation_support",
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
    dimensionality: Mapping[str, Any] | None = None,
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
            "blocked",
            implemented=False,
            missing=("ionization_recombination_model", "charge_state_transport", "startup_link"),
            claim_impact="breakdown_sheath_and_resistivity_authority_blocked",
        ),
        "single_two_temperature_energy": _effect(
            "candidate" if electron_energy_present else "blocked",
            implemented=electron_energy_present,
            missing=(
                "electron_heat_flux",
                "electron_ion_collisional_coupling",
                "temperature_diagnostic_validation",
                "hall_pressure_sensitivity_uq",
            ),
            claim_impact="hall_pressure_and_yield_authority_blocked",
        ),
        "electrical_thermal_transport": _effect(
            "candidate",
            implemented=True,
            missing=(
                "transport_validity_regime",
                "ohmic_cfl_nondominance",
                "thermal_conduction_closure",
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
    return {
        "status": "candidate_engineering_closure_packet_not_validation",
        "decision": "do_not_promote_without_complete_physics_closure_matrix",
        "required_effects": list(REQUIRED_EFFECTS),
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "effects": effects,
        "closure_matrix_status_by_effect": {
            key: record["status"] for key, record in effects.items()
        },
        "missing_or_unaccepted_effects": missing_effects,
        "candidate_runtime_channels": _candidate_runtime_channels(
            include_hall=include_hall,
            electron_energy_present=electron_energy_present,
            kinetic_yield_present=kinetic_yield_present,
            collisions_enabled=collisions_enabled,
            dimensionality=dimensionality,
        ),
        "active_candidate_closures": [
            key
            for key, record in effects.items()
            if record["status"] == "candidate" and record["implemented"]
        ],
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
    return {
        "status": status,
        "implemented": implemented,
        "classification": status if status != "candidate" else "candidate_only",
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "missing_channels": list(missing),
        "claim_impact": claim_impact,
        "review_status": "not_reviewed_for_acceptance",
        "can_support_first_principles_acceptance": False,
    }


def _candidate_runtime_channels(
    *,
    include_hall: bool,
    electron_energy_present: bool,
    kinetic_yield_present: bool,
    collisions_enabled: bool,
    dimensionality: Mapping[str, Any] | None,
) -> list[str]:
    channels: set[str] = set()
    channels.add("candidate_electrical_transport_scaffold")
    if include_hall:
        channels.add("candidate_hall_term_enabled")
    if electron_energy_present:
        channels.add("candidate_electron_energy_scaffold")
    if kinetic_yield_present:
        channels.add("candidate_kinetic_yield_history")
    if collisions_enabled:
        channels.add("candidate_collision_stage_enabled")
    if dimensionality is not None:
        channels.add("candidate_dimensionality_packet_linked")
        for channel in dimensionality.get("candidate_runtime_channels", ()):
            if str(channel).startswith("candidate_"):
                channels.add(f"dimensionality_{channel}")
    return sorted(channels)
