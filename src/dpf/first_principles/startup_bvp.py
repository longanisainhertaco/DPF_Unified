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


def build_startup_bvp_packet(
    startup: Mapping[str, Any],
    *,
    device: Mapping[str, Any] | None = None,
    gas: Mapping[str, Any] | None = None,
    circuit: Mapping[str, Any] | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return a startup packet that rejects non-source-backed whole-shot starts."""

    mode = str(startup.get("mode", "not_declared"))
    evidence_status = str(startup.get("evidence_status", "not_reviewed"))
    accepted = {str(channel) for channel in accepted_channels}
    accepted.update(str(channel) for channel in startup.get("accepted_channels", ()))
    candidate_inputs = _candidate_input_channels(
        startup=startup,
        device=device,
        gas=gas,
        circuit=circuit,
    )
    missing = set(REQUIRED_STARTUP_CHANNELS) - accepted
    missing.update(str(channel) for channel in startup.get("missing_channels", ()))

    whole_shot_requested = bool(startup.get("can_support_whole_shot_acceptance"))
    mode_is_accepted = mode in ACCEPTED_STARTUP_MODES
    reviewed = evidence_status in {"reviewed", "accepted", "accepted_same_scope_source"}
    can_support = (
        whole_shot_requested
        and mode_is_accepted
        and reviewed
        and not missing
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
        "candidate_input_channels": sorted(candidate_inputs),
        "mode_required_payload": list(MODE_REQUIRED_PAYLOADS.get(mode, ())),
        "source_references": list(STARTUP_BVP_SOURCE_REFS),
        "whole_shot_startup_blocked": not can_support,
        "can_support_whole_shot_acceptance": can_support,
        "can_support_first_principles_acceptance": can_support,
    }


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
    return channels


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
