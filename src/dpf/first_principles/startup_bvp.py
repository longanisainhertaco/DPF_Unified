"""Fail-closed startup BVP packets for first-principles DPF runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STARTUP_BVP_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "613-735,1219-1220",
        "role": "end_of_rundown_sheath_initialization_and_quasineutral_limitations",
    },
    {
        "path": (
            "KnowledgeReference/"
            "unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md"
        ),
        "lines": "245-392,555-585",
        "role": "breakdown_liftoff_sequence_and_pic_to_mhd_startup_handoff_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "sand2009-6373-b93aec67.md"
        ),
        "lines": "151-163,317-352,360-369,470-475,682-690",
        "role": "user_validated_alegra_pic_startup_import_and_3d_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md"
        ),
        "lines": "520-590,1488-1545",
        "role": "insulator_breakdown_pressure_preionization_and_startup_controls",
    },
    {
        "path": (
            "KnowledgeReference/"
            "effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md"
        ),
        "lines": "452-670",
        "role": "pressure_regime_insulator_length_sheath_mass_and_velocity_context",
    },
    {
        "path": "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md",
        "lines": "56-74",
        "role": "pf1000_surface_discharge_avalanche_streamer_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
        ),
        "lines": "506-514,579-653,1545-1640,3372-3380",
        "role": "user_validated_insulator_breakdown_symmetry_and_conditioning_design",
    },
    {
        "path": (
            "KnowledgeReference/"
            "high-power-laser-and-particle-beams-d1758d55.md"
        ),
        "lines": "103-147,161-176,180-200,210-237",
        "role": "user_validated_compact_mather_geometry_tof_and_neutron_pulse_context",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md",
        "lines": "1-118",
        "role": "local_source_truth_startup_blocker_contract",
    },
)

ACCEPTED_STARTUP_MODES = (
    "imported_pic_sheath_state",
    "surface_breakdown_bvp",
)

ENGINEERING_ONLY_STARTUP_MODES = (
    "source_backed_end_rundown_sheath",
    "plasma_injection_startup",
)

REJECTED_STARTUP_MODES = (
    "seeded_layer",
    "source_backed_candidate_uniform",
    "source_backed_profile",
)

REQUIRED_STARTUP_CHANNELS = (
    "device_geometry_and_insulator",
    "gas_species_pressure_temperature",
    "bank_voltage_and_early_circuit",
    "breakdown_or_flashover_model",
    "preionization_state",
    "surface_material_secondary_emission",
    "pressure_regime_classifier",
    "initial_density_ionization_charge_state",
    "initial_current_density_distribution",
    "initial_velocity_distribution",
    "initial_electric_field",
    "initial_magnetic_field",
    "electron_temperature_initial",
    "ion_temperature_initial",
    "initial_resistivity_or_conductivity",
    "sheath_liftoff_and_handoff_interval",
    "charge_current_divb_energy_consistency",
    "source_paths_hashes_units_and_review",
)

MODE_REQUIRED_PAYLOADS = {
    "imported_pic_sheath_state": (
        "mesh_mapping",
        "particles",
        "electron_density",
        "ion_density",
        "electron_temperature",
        "ion_temperature",
        "velocity",
        "electric_field",
        "magnetic_field",
        "current_density",
        "charge_consistency",
        "boundary_labels",
        "source_references",
        "hashes",
        "units",
        "conservation_checks",
    ),
    "surface_breakdown_bvp": (
        "surface_flashover_equations",
        "secondary_emission_or_material_model",
        "avalanche_streamer_closure",
        "preionization_model",
        "pressure_regime_classifier",
        "electrode_insulator_boundary_data",
        "verification_tests",
    ),
    "source_backed_end_rundown_sheath": (
        "hybrid_pic_fluid_initial_sheath_values",
        "explicit_end_of_rundown_scope",
        "breakdown_liftoff_exclusion",
    ),
    "plasma_injection_startup": (
        "source_backed_density_distribution",
        "source_backed_velocity_distribution",
        "device_scope_limitation",
    ),
    "seeded_layer": (),
    "source_backed_candidate_uniform": (),
    "source_backed_profile": (),
}

CANDIDATE_INPUT_TO_REQUIRED_CHANNEL = {
    "candidate_device_geometry": "device_geometry_and_insulator",
    "candidate_insulator_geometry": "device_geometry_and_insulator",
    "candidate_gas_species_pressure_temperature": "gas_species_pressure_temperature",
    "candidate_bank_voltage_and_initial_circuit": "bank_voltage_and_early_circuit",
    "candidate_initial_density": "initial_density_ionization_charge_state",
    "candidate_initial_electron_temperature": "electron_temperature_initial",
    "candidate_initial_ion_temperature": "ion_temperature_initial",
    "candidate_initial_electric_field": "initial_electric_field",
    "candidate_initial_magnetic_field": "initial_magnetic_field",
    "candidate_civ_paschen_breakdown_audit": "breakdown_or_flashover_model",
    "candidate_civ_paschen_initial_ionization": "initial_density_ionization_charge_state",
    "candidate_civ_paschen_liftoff_delay": "sheath_liftoff_and_handoff_interval",
    "candidate_source_backed_end_rundown_sheath_profile": (
        "initial_density_ionization_charge_state"
    ),
    "candidate_source_backed_sheath_velocity_distribution": (
        "initial_velocity_distribution"
    ),
}


def build_startup_bvp_packet(
    startup: Mapping[str, Any],
    *,
    device: Mapping[str, Any] | None = None,
    gas: Mapping[str, Any] | None = None,
    circuit: Mapping[str, Any] | None = None,
    candidate_breakdown_audit: Mapping[str, Any] | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return a startup packet that rejects non-source-backed whole-shot starts."""

    mode = str(startup.get("mode", "not_declared"))
    evidence_status = str(startup.get("evidence_status", "not_reviewed"))
    accepted = {str(channel) for channel in accepted_channels}
    accepted.update(str(channel) for channel in startup.get("accepted_channels", ()))
    startup_payload_review = _startup_payload_review(
        mode=mode,
        startup=startup,
        evidence_status=evidence_status,
    )
    if startup_payload_review["channel_acceptance_eligible"]:
        accepted.update(startup_payload_review["accepted_channels"])
    candidate_inputs = _candidate_input_channels(
        startup=startup,
        device=device,
        gas=gas,
        circuit=circuit,
    )
    candidate_inputs.update(_candidate_breakdown_channels(candidate_breakdown_audit))
    missing = set(REQUIRED_STARTUP_CHANNELS) - accepted
    missing.update(str(channel) for channel in startup.get("missing_channels", ()))

    whole_shot_requested = bool(startup.get("can_support_whole_shot_acceptance"))
    mode_is_accepted = mode in ACCEPTED_STARTUP_MODES
    reviewed = evidence_status in {"reviewed", "accepted", "accepted_same_scope_source"}
    payload_acceptance_eligible = bool(
        startup_payload_review["channel_acceptance_eligible"]
    )
    can_support = (
        whole_shot_requested
        and mode_is_accepted
        and reviewed
        and not missing
        and payload_acceptance_eligible
    )
    status = (
        "accepted_startup_bvp_packet"
        if can_support
        else _blocked_status_for_mode(mode)
    )

    return {
        "status": status,
        "mode": mode,
        "evidence_status": evidence_status,
        "source_scope": startup.get("source_scope", "not_declared"),
        "decision": (
            "startup_packet_can_support_whole_shot_first_principles"
            if can_support
            else "do_not_promote_startup_to_whole_shot_first_principles"
        ),
        "startup_mode_class": _mode_class(mode),
        "accepted_modes": list(ACCEPTED_STARTUP_MODES),
        "engineering_only_modes": list(ENGINEERING_ONLY_STARTUP_MODES),
        "rejected_modes": list(REJECTED_STARTUP_MODES),
        "required_channels": list(REQUIRED_STARTUP_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "startup_channel_status": _startup_channel_statuses(
            accepted=accepted,
            missing=missing,
            candidate_inputs=candidate_inputs,
        ),
        "candidate_input_channels": sorted(candidate_inputs),
        "startup_payload_review": startup_payload_review,
        "candidate_breakdown_audit": _candidate_breakdown_audit_packet(
            candidate_breakdown_audit
        ),
        "candidate_input_policy": {
            "candidate_inputs_can_seed_engineering_runs": True,
            "candidate_inputs_can_support_whole_shot_acceptance": False,
            "required_promotion_path": (
                "reviewed_imported_pic_sheath_state_or_source_backed_"
                "surface_breakdown_bvp"
            ),
        },
        "mode_required_payload": list(MODE_REQUIRED_PAYLOADS.get(mode, ())),
        "mode_payload_status": _mode_payload_status(mode, accepted),
        "startup_mode_status": _startup_mode_statuses(
            current_mode=mode,
            can_support=can_support,
        ),
        "source_references": list(STARTUP_BVP_SOURCE_REFS),
        "acceptance_gate": (
            "engineering_end_rundown_seeded_or_text_startup_cannot_support_"
            "whole_shot_first_principles_until_reviewed_imported_pic_state_or_"
            "source_backed_surface_breakdown_bvp_payload_channels_hashes_"
            "consistency_tests_and_review_pass"
        ),
        "negative_test_policy": {
            "seeded_layer_rejection_required": True,
            "uniform_or_profile_startup_rejection_required": True,
            "end_rundown_whole_shot_rejection_required": True,
            "unreviewed_imported_pic_state_rejection_required": True,
            "missing_field_particle_payload_rejection_required": True,
            "startup_payload_channel_overclaim_rejection_required": True,
            "civ_paschen_scaffold_promotion_rejection_required": True,
            "cross_scope_startup_promotion_rejection_required": True,
        },
        "whole_shot_startup_blocked": not can_support,
        "can_support_whole_shot_acceptance": can_support,
        "can_support_first_principles_acceptance": can_support,
    }


def _startup_channel_statuses(
    *,
    accepted: set[str],
    missing: set[str],
    candidate_inputs: set[str],
) -> dict[str, str]:
    candidate_required = {
        CANDIDATE_INPUT_TO_REQUIRED_CHANNEL[channel]
        for channel in candidate_inputs
        if channel in CANDIDATE_INPUT_TO_REQUIRED_CHANNEL
    }
    statuses: dict[str, str] = {}
    for channel in REQUIRED_STARTUP_CHANNELS:
        if channel in accepted:
            statuses[channel] = "accepted_startup_channel_declared"
        elif channel in candidate_required:
            statuses[channel] = "candidate_input_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _startup_payload_review(
    *,
    mode: str,
    startup: Mapping[str, Any],
    evidence_status: str,
) -> dict[str, Any]:
    payload = startup.get("startup_payload")
    if payload is None:
        payload = startup.get("payload")
    required_payload = tuple(MODE_REQUIRED_PAYLOADS.get(mode, ()))
    if not isinstance(payload, Mapping) or not payload:
        return {
            "status": "startup_payload_not_supplied",
            "mode": mode,
            "required_payload": list(required_payload),
            "payload_field_status": {
                field: "missing_payload" for field in required_payload
            },
            "accepted_channels": [],
            "missing_payload_fields": list(required_payload),
            "missing_required_startup_channels": list(REQUIRED_STARTUP_CHANNELS),
            "channel_acceptance_eligible": False,
            "can_support_whole_shot_acceptance": False,
            "can_support_first_principles_acceptance": False,
        }

    payload_mode = str(payload.get("mode", payload.get("payload_mode", mode)))
    payload_evidence_status = str(payload.get("evidence_status", evidence_status))
    reviewed = payload_evidence_status in {
        "reviewed",
        "accepted",
        "accepted_same_scope_source",
    }
    declared_scope = str(startup.get("source_scope", "not_declared"))
    payload_scope = str(payload.get("source_scope", declared_scope))
    scope_matches = (
        declared_scope in {"not_declared", payload_scope}
        or payload_scope == "not_declared"
    )
    payload_status = {
        field: (
            "payload_channel_present"
            if _payload_has_channel(payload, field)
            else "missing_payload_channel"
        )
        for field in required_payload
    }
    missing_payload = [
        field
        for field, status in payload_status.items()
        if status == "missing_payload_channel"
    ]
    accepted_channels = {
        str(channel) for channel in payload.get("accepted_channels", ())
    }
    missing_startup_channels = sorted(
        set(REQUIRED_STARTUP_CHANNELS) - accepted_channels
    )
    payload_can_support = bool(payload.get("can_support_whole_shot_acceptance"))
    mode_matches = payload_mode == mode
    eligible = (
        mode in ACCEPTED_STARTUP_MODES
        and mode_matches
        and reviewed
        and scope_matches
        and payload_can_support
        and not missing_payload
        and not missing_startup_channels
    )
    if eligible:
        status = "reviewed_startup_payload_complete"
    elif mode not in ACCEPTED_STARTUP_MODES:
        status = "startup_payload_for_nonaccepted_mode_not_promoting"
    elif not reviewed:
        status = "startup_payload_unreviewed"
    elif not mode_matches:
        status = "startup_payload_mode_mismatch"
    elif not scope_matches:
        status = "startup_payload_scope_mismatch"
    elif missing_payload or missing_startup_channels:
        status = "startup_payload_incomplete"
    elif not payload_can_support:
        status = "startup_payload_nonpromoting"
    else:
        status = "startup_payload_blocked"
    return {
        "status": status,
        "mode": mode,
        "payload_mode": payload_mode,
        "evidence_status": payload_evidence_status,
        "source_scope": declared_scope,
        "payload_source_scope": payload_scope,
        "required_payload": list(required_payload),
        "payload_field_status": payload_status,
        "accepted_channels": sorted(accepted_channels),
        "missing_payload_fields": missing_payload,
        "missing_required_startup_channels": missing_startup_channels,
        "mode_matches": mode_matches,
        "source_scope_matches": scope_matches,
        "payload_declares_whole_shot_support": payload_can_support,
        "channel_acceptance_eligible": eligible,
        "can_support_whole_shot_acceptance": eligible,
        "can_support_first_principles_acceptance": eligible,
        "acceptance_rule": (
            "accepted startup modes require a reviewed same-scope payload, all "
            "mode-required payload channels, all required startup channels, "
            "source references, hashes, units, and conservation checks"
        ),
    }


def _payload_has_channel(payload: Mapping[str, Any], field: str) -> bool:
    if payload.get(field) is not None:
        return True
    channels = payload.get("payload_channels")
    if isinstance(channels, Mapping) and channels.get(field) is not None:
        return True
    return isinstance(channels, (list, tuple, set)) and field in channels


def _mode_payload_status(mode: str, accepted: set[str]) -> dict[str, str]:
    return {
        payload: (
            "accepted_payload_channel_declared"
            if payload in accepted
            else "missing_or_unreviewed_payload"
        )
        for payload in MODE_REQUIRED_PAYLOADS.get(mode, ())
    }


def _startup_mode_statuses(
    *,
    current_mode: str,
    can_support: bool,
) -> dict[str, dict[str, Any]]:
    all_modes = (
        ACCEPTED_STARTUP_MODES
        + ENGINEERING_ONLY_STARTUP_MODES
        + REJECTED_STARTUP_MODES
    )
    statuses: dict[str, dict[str, Any]] = {}
    for mode in all_modes:
        if mode in REJECTED_STARTUP_MODES:
            status = "rejected_for_accepted_first_principles_claims"
            decision = "must_fail_acceptance_gate"
        elif mode in ENGINEERING_ONLY_STARTUP_MODES:
            status = "engineering_candidate_not_whole_shot"
            decision = "usable_for_engineering_or_narrowed_handoff_only"
        elif mode == "surface_breakdown_bvp":
            status = "accepted_only_after_complete_source_bvp_payload_and_review"
            decision = "blocked_until_payload_channels_and_review_pass"
        else:
            status = "accepted_only_after_complete_reviewed_imported_state"
            decision = "blocked_until_particles_fields_currents_hashes_and_review_pass"
        if mode == current_mode and can_support:
            status = "accepted_startup_bvp_packet"
            decision = "can_support_whole_shot_startup_claim"
        statuses[mode] = {
            "current_mode": mode == current_mode,
            "mode_class": _mode_class(mode),
            "required_payload": list(MODE_REQUIRED_PAYLOADS.get(mode, ())),
            "status": status,
            "decision": decision,
            "can_support_acceptance_without_complete_payload": False,
        }
    if current_mode not in statuses:
        statuses[current_mode] = {
            "current_mode": True,
            "mode_class": "unknown",
            "required_payload": [],
            "status": "unknown_startup_mode_blocks_acceptance",
            "decision": "must_fail_acceptance_gate",
            "can_support_acceptance_without_complete_payload": False,
        }
    return statuses


def _candidate_input_channels(
    *,
    startup: Mapping[str, Any],
    device: Mapping[str, Any] | None,
    gas: Mapping[str, Any] | None,
    circuit: Mapping[str, Any] | None,
) -> set[str]:
    channels: set[str] = set()
    if device and _has_all(
        device,
        "anode_radius_m",
        "cathode_radius_m",
        "anode_length_m",
    ):
        channels.add("candidate_device_geometry")
    if device and _has_all(device, "insulator_length_m"):
        channels.add("candidate_insulator_geometry")
    if gas and _has_all(gas, "species", "pressure_Pa", "temperature_K"):
        channels.add("candidate_gas_species_pressure_temperature")
    if circuit and _has_any(circuit, "voltage_V", "initial_current_A", "charge_C"):
        channels.add("candidate_bank_voltage_and_initial_circuit")
    if startup.get("background_density_m3") is not None:
        channels.add("candidate_initial_density")
    if startup.get("electron_temperature_K") is not None:
        channels.add("candidate_initial_electron_temperature")
    if startup.get("ion_temperature_K") is not None:
        channels.add("candidate_initial_ion_temperature")
    if startup.get("initial_electric_field_V_m") is not None:
        channels.add("candidate_initial_electric_field")
    if startup.get("initial_magnetic_field_T") is not None:
        channels.add("candidate_initial_magnetic_field")
    if startup.get("source_scope") is not None:
        channels.add("candidate_startup_source_scope")
    payload = startup.get("startup_payload")
    if isinstance(payload, Mapping) and payload.get("profile_type") == "annular_axial_sheath":
        channels.add("candidate_source_backed_end_rundown_sheath_profile")
        if payload.get("sheath_drift_velocity_m_s") is not None:
            channels.add("candidate_source_backed_sheath_velocity_distribution")
    return channels


def _candidate_breakdown_channels(
    candidate_breakdown_audit: Mapping[str, Any] | None,
) -> set[str]:
    if not candidate_breakdown_audit:
        return set()
    if candidate_breakdown_audit.get("can_support_first_principles_acceptance") is True:
        return set()
    status = str(candidate_breakdown_audit.get("status", ""))
    if status != "candidate_civ_paschen_breakdown_audit_engineering_only":
        return set()
    channels = {"candidate_civ_paschen_breakdown_audit"}
    breakdown = candidate_breakdown_audit.get("breakdown")
    if (
        isinstance(breakdown, Mapping)
        and breakdown.get("initial_ionization_fraction") is not None
    ):
        channels.add("candidate_civ_paschen_initial_ionization")
    liftoff = candidate_breakdown_audit.get("liftoff")
    if (
        isinstance(liftoff, Mapping)
        and liftoff.get("candidate_liftoff_delay_s") is not None
    ):
        channels.add("candidate_civ_paschen_liftoff_delay")
    return channels


def _candidate_breakdown_audit_packet(
    candidate_breakdown_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not candidate_breakdown_audit:
        return {
            "status": "candidate_breakdown_audit_not_supplied",
            "can_support_whole_shot_acceptance": False,
            "can_support_first_principles_acceptance": False,
        }
    packet = dict(candidate_breakdown_audit)
    packet["can_support_whole_shot_acceptance"] = False
    packet["can_support_first_principles_acceptance"] = False
    return packet


def _has_any(mapping: Mapping[str, Any], *keys: str) -> bool:
    return any(mapping.get(key) is not None for key in keys)


def _has_all(mapping: Mapping[str, Any], *keys: str) -> bool:
    return all(mapping.get(key) is not None for key in keys)


def _blocked_status_for_mode(mode: str) -> str:
    if mode in REJECTED_STARTUP_MODES:
        return "rejected_startup_mode_for_first_principles"
    return "blocked_startup_bvp_packet_not_available"


def _mode_class(mode: str) -> str:
    if mode in ACCEPTED_STARTUP_MODES:
        return "accepted_only_with_complete_reviewed_payload"
    if mode in ENGINEERING_ONLY_STARTUP_MODES:
        return "engineering_only"
    if mode in REJECTED_STARTUP_MODES:
        return "rejected_for_accepted_claims"
    return "unknown"
