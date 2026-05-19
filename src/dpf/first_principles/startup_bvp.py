"""Fail-closed startup BVP packets for first-principles DPF runs.

This module carries two layers:

* ``build_startup_bvp_packet`` -- the mode-level acceptance gate that decides
  whether a declared startup mode (imported PIC sheath, surface breakdown BVP,
  seeded layer, ...) may support a whole-shot first-principles claim.
* ``build_startup_channel_packet`` -- the S3.4 typed per-channel startup packet
  required by ``docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md``
  section "S3.4 Startup BVP Packet". It enumerates every whole-shot startup
  channel as a typed ``StartupChannel`` record carrying its status, exact local
  source references, units, symbol map, input dependencies, output fields,
  blocker reason, and effect on the first-principles claim.

The per-channel statuses are fixed by the WP-N2 research packet
``docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md``,
which classified all 11 physics channels as ``candidate`` and ``0`` as
``supported`` for a DPF-specific startup BVP. No channel is promoted to
``computed`` here: the local corpus does not supply a DPF-specific closure for
any startup channel, so the typed packet always reports
``can_support_first_principles_acceptance = False``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
        "startup_channel_packet": build_startup_packet().as_dict(),
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


# ===========================================================================
# S3.4 typed startup channel packet
# ===========================================================================
#
# Handoff: docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md
#          section "S3.4 Startup BVP Packet".
# Research basis (channel status fixed here): WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md
#   -> all channels candidate or blocked; 0 supported; 0 computed.
#
# A startup channel may only carry status "computed" when the local corpus
# supplies a DPF-specific closure that produces the runtime field. WP-N2 found
# none, so every channel below is "candidate" or "blocked". Promotion to
# "computed" requires a cited DPF-specific source for that channel and is NOT
# permitted from inferred formulae or back-solved end-state results.

# Allowed typed-channel status values. "supported" is reserved for a future
# DPF-specific source closure and is intentionally unused by the registry.
STARTUP_CHANNEL_STATUSES = ("computed", "candidate", "blocked")

# Statuses that count toward first-principles startup authority. Per WP-N2 no
# channel is "computed", so the typed packet's authority is always blocked.
STARTUP_CHANNEL_AUTHORITY_STATUSES = ("computed",)


@dataclass(frozen=True)
class StartupSourceRef:
    """Exact local source reference for a startup channel claim."""

    path: str
    lines: str
    equation_or_figure: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "lines": self.lines,
            "equation_or_figure": self.equation_or_figure,
        }


@dataclass(frozen=True)
class StartupChannel:
    """Typed startup BVP channel record.

    Fields are the S3.4 handoff-required packet fields: channel ID, status,
    source refs, units, symbol map, input dependencies, output fields, blocker
    reason, and the effect of the channel on the first-principles claim.
    """

    channel_id: str
    status: str
    source_refs: tuple[StartupSourceRef, ...]
    units: Mapping[str, str]
    symbol_map: Mapping[str, str]
    input_dependencies: tuple[str, ...]
    output_fields: tuple[str, ...]
    blocker_reason: str
    blocker_id: str
    first_principles_claim_effect: str
    missing_parameter_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STARTUP_CHANNEL_STATUSES:
            raise ValueError(
                f"startup channel '{self.channel_id}' has invalid status "
                f"'{self.status}'; allowed: {STARTUP_CHANNEL_STATUSES}"
            )
        if self.status == "blocked" and not self.blocker_id:
            raise ValueError(
                f"blocked startup channel '{self.channel_id}' must carry a "
                "blocker_id"
            )
        if self.status != "blocked" and not self.source_refs:
            raise ValueError(
                f"non-blocked startup channel '{self.channel_id}' must carry "
                "at least one source reference"
            )

    @property
    def supports_first_principles(self) -> bool:
        return self.status in STARTUP_CHANNEL_AUTHORITY_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "status": self.status,
            "source_refs": [ref.as_dict() for ref in self.source_refs],
            "units": dict(self.units),
            "symbol_map": dict(self.symbol_map),
            "input_dependencies": list(self.input_dependencies),
            "output_fields": list(self.output_fields),
            "blocker_reason": self.blocker_reason,
            "blocker_id": self.blocker_id,
            "missing_parameter_ids": list(self.missing_parameter_ids),
            "first_principles_claim_effect": self.first_principles_claim_effect,
            "supports_first_principles_acceptance": (
                self.supports_first_principles
            ),
        }


@dataclass(frozen=True)
class StartupPacket:
    """Typed S3.4 startup BVP packet.

    Aggregates the per-channel ``StartupChannel`` records and exposes whether
    the startup channels can support a first-principles acceptance claim. Per
    WP-N2 this is always ``False``: no channel reaches ``computed`` status.
    """

    channels: tuple[StartupChannel, ...]
    status: str
    blocker_ids: tuple[str, ...]
    can_support_first_principles_acceptance: bool
    requirement_ids: tuple[str, ...] = field(
        default=("DPF-PHYS-010", "DPF-PHYS-017", "DPF-PHYS-021")
    )

    @property
    def channels_by_id(self) -> dict[str, StartupChannel]:
        return {channel.channel_id: channel for channel in self.channels}

    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in STARTUP_CHANNEL_STATUSES}
        for channel in self.channels:
            counts[channel.status] += 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet_type": "first_principles_startup_bvp_channel_packet",
            "requirement_ids": list(self.requirement_ids),
            "status": self.status,
            "source_authority": (
                "KnowledgeReference and tracked verified extracts only"
            ),
            "research_basis": (
                "docs/external_team_submissions/"
                "2026_05_18_three_sprint_blocker_packet/sprint_3/"
                "WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md"
            ),
            "channel_status_counts": self.status_counts(),
            "channels": [channel.as_dict() for channel in self.channels],
            "channels_blocking_startup_authority": sorted(
                channel.channel_id
                for channel in self.channels
                if not channel.supports_first_principles
            ),
            "blocker_ids": list(self.blocker_ids),
            "forbidden_startup_inputs": list(FORBIDDEN_STARTUP_INPUTS),
            "can_support_first_principles_acceptance": (
                self.can_support_first_principles_acceptance
            ),
            "first_principles_claim_effect": (
                "startup authority blocked: no startup channel reaches "
                "computed status; seeded startup stays rejected"
            ),
        }


# Forbidden startup inputs (S3.4 handoff "Forbidden"). These are recorded in
# the typed packet so a reviewer sees them; none may ever produce a "computed"
# channel.
FORBIDDEN_STARTUP_INPUTS = (
    "arbitrary_seed_density_as_accepted_startup",
    "back_solve_initial_condition_from_published_end_state_results",
    "silent_fallback_to_engineering_defaults_in_first_principles_mode",
)

# Missing-parameter inventory from WP-N2 section 4. Each entry holds a channel
# at "candidate" until a DPF-specific local source supplies it.
STARTUP_MISSING_PARAMETERS: dict[str, str] = {
    "M1": (
        "DPF surface-flashover BVP closure (insulator-surface physics) -- no "
        "reviewed equation set produces initial E/B/J/n/Te/Ti along the "
        "insulator from applied voltage"
    ),
    "M2": (
        "alpha, gamma, sigma_i0, eta, beta_ep, R_ph numerical values for "
        "D2/H2/Ne/Ar at DPF voltages"
    ),
    "M3": (
        "secondary-emission coefficients for DPF materials (Cu anode, "
        "pyrex/alumina insulator)"
    ),
    "M4": "quantitative preionization seed-density / ionization-fraction model",
    "M5": (
        "closed flashover delay / voltage / striation-to-uniform timescale "
        "model"
    ),
    "M6": "breakdown-phase initial E and J field distributions",
    "M7": "start-of-shot density/species and Te/Ti fields",
    "M8": (
        "numerical handoff-interval definition and a same-device reviewed PIC "
        "import payload"
    ),
    "M9": "DPF-specific homogeneous-field validity for the Te relation Eq. (4)",
}


def _ref(path: str, lines: str, equation_or_figure: str) -> StartupSourceRef:
    return StartupSourceRef(
        path=f"KnowledgeReference/{path}",
        lines=lines,
        equation_or_figure=equation_or_figure,
    )


# Source-file shorthands used by the registry below.
_KR_BREAKDOWN_NOBLE = (
    "the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-"
    "of-noble-gases-3.md"
)
_KR_CURRENT_SHEATH = (
    "effect-of-current-sheath-initiation-on-the-radial-collapse-and-"
    "energetic-particle-accelera-b2e95b88.md"
)
_KR_VERSATILE = (
    "the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-"
    "applications.md"
)
_KR_GRIBKOV = "gribkov-2007-pf1000-jphysd-part2.md"
_KR_DESIGN_CONSTRUCTION = (
    "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
)
_KR_HYBRID_PIC = "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
_KR_SAND2009 = "sand2009-6373-b93aec67.md"
_KR_ALEGRA = (
    "unlimited-release-printed-september-2009-alegra-hedp-simulations-of-"
    "the-dense-plasma-focus.md"
)
_KR_GAS_DISCHARGE = (
    "theory-and-finite-element-simulation-methodology-of-gas-discharge-"
    "plasmas.md"
)


def _build_startup_channel_registry() -> tuple[StartupChannel, ...]:
    """Return the 13 typed startup channels from the S3.4 handoff.

    Status per channel is fixed by WP-N2: every channel is ``candidate`` (the
    corpus supplies a relation or qualitative basis) -- none is ``computed``,
    and none is ``blocked`` because no channel is fully source-empty. The
    whole-shot startup is blocked as a whole because no channel reaches
    ``computed``.
    """

    return (
        StartupChannel(
            channel_id="gas_and_fill_conditions",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_VERSATILE,
                    "527-534,569-573",
                    "prose: filamentary (high p) vs uniform (low p) "
                    "breakdown; optimum pressure ~10 mbar",
                ),
                _ref(
                    _KR_HYBRID_PIC,
                    "661-690",
                    "prose + Table 1: prefill n0=6.7e22 m^-3, T1~0.026 eV "
                    "(end-of-rundown scope)",
                ),
            ),
            units={
                "fill_pressure_Pa": "Pa",
                "fill_temperature_K": "K",
                "background_number_density_m3": "m^-3",
            },
            symbol_map={
                "p0": "fill gas pressure",
                "T1": "background gas temperature",
                "n0": "background number density",
            },
            input_dependencies=(
                "gas.species",
                "gas.pressure_Pa",
                "gas.temperature_K",
            ),
            output_fields=(
                "fill_pressure_Pa",
                "fill_temperature_K",
                "species_name",
            ),
            blocker_reason=(
                "corpus supplies pressure-regime guidance and an "
                "end-of-rundown prefill density, but no start-of-discharge "
                "fill-condition closure for the whole-shot startup BVP"
            ),
            blocker_id="STARTUP-BVP-CH01-FILL-NO-DPF-CLOSURE",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M7",),
        ),
        StartupChannel(
            channel_id="breakdown_paschen_or_alternative",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_BREAKDOWN_NOBLE,
                    "130-133,150-156,196-203",
                    "Eq. (1) first Townsend coefficient; Eq. (3) alpha(Te); "
                    "Eq. (6) Townsend / Eq. (7) streamer breakdown",
                ),
                _ref(
                    _KR_CURRENT_SHEATH,
                    "631-639",
                    "prose: canonical Paschen feedback 'should no longer "
                    "apply' to DPFs",
                ),
            ),
            units={
                "alpha": "1/m",
                "gamma": "dimensionless",
                "d": "m",
                "breakdown_onset_time_s": "s",
            },
            symbol_map={
                "alpha": "primary (impact) ionization coefficient",
                "gamma": "secondary electron emission coefficient",
                "d": "inter-electrode / insulator-surface path length",
                "x": "distance along the field",
            },
            input_dependencies=(
                "device.insulator_length_m",
                "device.insulator_material",
                "gas.species",
                "gas.pressure_Pa",
                "circuit.voltage_V",
            ),
            output_fields=(
                "breakdown_onset_time_s",
                "breakdown_mechanism",
                "breakdown_path_length_m",
            ),
            blocker_reason=(
                "corpus explicitly states the canonical Paschen/Townsend "
                "feedback loop does not describe DPF insulator breakdown; no "
                "reviewed DPF surface-flashover BVP closure exists locally"
            ),
            blocker_id="STARTUP-BVP-CH02-BREAKDOWN-PASCHEN-CONTRADICTED-FOR-DPF",
            first_principles_claim_effect=(
                "candidate input only; corpus contradicts canonical Paschen "
                "for DPFs, so cannot support first-principles acceptance"
            ),
            missing_parameter_ids=("M1", "M2"),
        ),
        StartupChannel(
            channel_id="preionization",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_VERSATILE,
                    "1490-1530",
                    "prose: Ni-63 and U-238 preionizer measured yield deltas "
                    "(+25%, +50+/-5%); uniform-sheet mechanism",
                ),
            ),
            units={
                "preionization_seed_density_m3": "m^-3",
                "preionization_ionization_fraction": "dimensionless",
            },
            symbol_map={
                "n_seed": "preionization seed electron number density",
                "f_ion": "preionization ionization fraction",
            },
            input_dependencies=(
                "preionizer.model_or_none",
                "device.insulator_length_m",
            ),
            output_fields=(
                "preionization_seed_density_m3",
                "preionization_ionization_fraction",
            ),
            blocker_reason=(
                "corpus documents preionization as an experimental "
                "intervention with measured yield deltas, not a quantitative "
                "seed-density initial-condition generator"
            ),
            blocker_id="STARTUP-BVP-CH03-PREIONIZATION-NO-QUANTITATIVE-MODEL",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M4",),
        ),
        StartupChannel(
            channel_id="flashover",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_DESIGN_CONSTRUCTION,
                    "512-514,583-589",
                    "prose: breakdown along the insulator determines the "
                    "sheath; radial striated light pattern",
                ),
                _ref(
                    _KR_VERSATILE,
                    "538-541",
                    "prose: Kies ~100 J/cm^2 upper-limit insulator-surface "
                    "energy density",
                ),
            ),
            units={
                "flashover_complete_time_s": "s",
                "surface_energy_density_J_cm2": "J/cm^2",
                "striation_to_uniform_time_s": "s",
            },
            symbol_map={
                "w_surface": "insulator-surface energy density",
                "t_flashover": "surface-flashover completion time",
            },
            input_dependencies=(
                "device.insulator_length_m",
                "device.insulator_outer_radius_m",
                "device.insulator_material",
                "circuit.voltage_V",
            ),
            output_fields=(
                "flashover_complete_time_s",
                "surface_energy_density_J_cm2",
                "striation_to_uniform_time_s",
            ),
            blocker_reason=(
                "corpus gives a qualitative surface-discharge description and "
                "an upper-limit energy-density estimate, not a closed "
                "flashover-delay / flashover-voltage model"
            ),
            blocker_id="STARTUP-BVP-CH04-FLASHOVER-NO-CLOSED-DELAY-MODEL",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M1", "M5"),
        ),
        StartupChannel(
            channel_id="secondary_emission",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_BREAKDOWN_NOBLE,
                    "113-117",
                    "prose: gamma definition and range 1e-3..1e-8 "
                    "(electrode-material dependent)",
                ),
                _ref(
                    _KR_GAS_DISCHARGE,
                    "1187-1196",
                    "prose: secondary-emission electron flux boundary "
                    "condition Gamma_e = gamma * c_p * |u_p|",
                ),
            ),
            units={
                "secondary_emission_coefficient_gamma": "dimensionless",
                "Gamma_e_mol_m2_s": "mol m^-2 s^-1",
                "c_p_mol_m3": "mol m^-3",
                "u_p_m_s": "m s^-1",
            },
            symbol_map={
                "gamma": "secondary electron emission coefficient",
                "Gamma_e": "secondary electron number flux",
                "c_p": "positive-ion molar concentration",
                "u_p": "ion drift speed",
            },
            input_dependencies=(
                "device.insulator_material",
                "electrode.material",
                "surface.ion_flux",
            ),
            output_fields=(
                "secondary_emission_coefficient_gamma",
                "Gamma_e_mol_m2_s",
            ),
            blocker_reason=(
                "the cited gamma ranges are generic gas-discharge values; the "
                "corpus supplies no DPF-material (Cu / pyrex / alumina) "
                "ion-induced electron-emission yields"
            ),
            blocker_id="STARTUP-BVP-CH05-SECONDARY-EMISSION-NO-DPF-MATERIAL-GAMMA",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M2", "M3"),
        ),
        StartupChannel(
            channel_id="photoemission",
            status="blocked",
            source_refs=(),
            units={
                "photoemission_electron_flux_mol_m2_s": "mol m^-2 s^-1",
            },
            symbol_map={
                "Gamma_ph": "photoemission electron number flux",
            },
            input_dependencies=(
                "surface.photon_flux",
                "device.insulator_material",
            ),
            output_fields=("photoemission_electron_flux_mol_m2_s",),
            blocker_reason=(
                "no local KnowledgeReference source supplies a DPF "
                "photoemission boundary model; WP-N2 did not enumerate a "
                "photoemission channel and the corpus search returned no "
                "DPF-specific photoemission closure"
            ),
            blocker_id="STARTUP-BVP-CH06-PHOTOEMISSION-NO-LOCAL-SOURCE",
            first_principles_claim_effect=(
                "blocked: no local source; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=(),
        ),
        StartupChannel(
            channel_id="surface_plasma",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_DESIGN_CONSTRUCTION,
                    "583-589,601-607,632-651",
                    "prose: surface-discharge current path; inverse-pinch "
                    "force F = i dL x B; 1-microsecond uniformization",
                ),
                _ref(
                    _KR_GRIBKOV,
                    "62-66",
                    "prose: first stage surface discharge along the "
                    "cylindrical insulator",
                ),
            ),
            units={
                "surface_sheet_thickness_m": "m",
                "surface_sheet_conductivity_S_m": "S m^-1",
            },
            symbol_map={
                "F": "per-segment magnetic force i dL x B",
                "sigma_s": "surface-plasma sheet conductivity",
            },
            input_dependencies=(
                "flashover.output",
                "circuit.current_A",
            ),
            output_fields=(
                "surface_plasma_mask",
                "surface_sheet_thickness_m",
                "surface_sheet_conductivity_S_m",
            ),
            blocker_reason=(
                "corpus describes the surface plasma qualitatively only; no "
                "closed surface-plasma field set (density, thickness, "
                "conductivity, temperature) is provided"
            ),
            blocker_id="STARTUP-BVP-CH07-SURFACE-PLASMA-NO-CLOSED-FIELD-SET",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M1",),
        ),
        StartupChannel(
            channel_id="initial_e_b_j",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_HYBRID_PIC,
                    "748-757",
                    "Eq. (34) B_theta = mu I / (2 pi r); Eq. (35) external "
                    "circuit equation (implosion-phase relations)",
                ),
                _ref(
                    _KR_DESIGN_CONSTRUCTION,
                    "508-512,601-605",
                    "prose + Fig. 2a: radial applied-voltage E-field; surface "
                    "current density J at breakdown",
                ),
            ),
            units={
                "initial_E_field_V_m": "V m^-1",
                "initial_B_field_T": "T",
                "initial_J_A_m2": "A m^-2",
            },
            symbol_map={
                "B_theta": "azimuthal magnetic field",
                "I": "circuit current",
                "r": "radius",
                "mu": "permeability",
            },
            input_dependencies=(
                "circuit.voltage_V",
                "circuit.inductance_H",
                "circuit.capacitance_F",
                "circuit.resistance_ohm",
                "device.geometry",
            ),
            output_fields=(
                "initial_E_field_V_m",
                "initial_B_field_T",
                "initial_J_A_m2",
            ),
            blocker_reason=(
                "Eq. (34)/(35) are implosion-phase circuit/boundary "
                "relations, not breakdown-phase initial fields; no closed "
                "source-derived initial E and J distribution exists locally"
            ),
            blocker_id="STARTUP-BVP-CH08-INITIAL-FIELDS-NO-BREAKDOWN-PHASE-SET",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M6",),
        ),
        StartupChannel(
            channel_id="species_and_charge_state",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_HYBRID_PIC,
                    "661-690",
                    "prose + Table 1: deuterium species; end-of-rundown "
                    "n0=6.7e22 m^-3, ns,0=3.3e23 m^-3",
                ),
            ),
            units={
                "total_number_density_m3": "m^-3",
                "ion_density_m3": "m^-3",
                "electron_density_m3": "m^-3",
                "charge_state": "dimensionless",
            },
            symbol_map={
                "n0": "background number density",
                "ns0": "current-sheath number density",
                "Z": "ion charge state",
            },
            input_dependencies=(
                "gas.species",
                "gas.pressure_Pa",
                "breakdown.ionization",
            ),
            output_fields=(
                "total_number_density_m3",
                "ion_density_m3",
                "electron_density_m3",
                "species_name",
                "charge_state",
            ),
            blocker_reason=(
                "corpus supplies only end-of-rundown handoff densities; "
                "start-of-discharge species/charge-state fields for the "
                "breakdown BVP are absent"
            ),
            blocker_id="STARTUP-BVP-CH09-SPECIES-NO-START-OF-SHOT-FIELDS",
            first_principles_claim_effect=(
                "candidate input only; end-of-rundown scope, cannot support "
                "first-principles whole-shot acceptance"
            ),
            missing_parameter_ids=("M7",),
        ),
        StartupChannel(
            channel_id="ionization_recombination_status",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_BREAKDOWN_NOBLE,
                    "130-133,150-156",
                    "Eq. (1) first Townsend coefficient; Eq. (3) alpha(Te)",
                ),
                _ref(
                    _KR_GAS_DISCHARGE,
                    "256-292",
                    "Eq. (6) electron source term R_e and coefficient "
                    "definitions (alpha, eta, beta_ep, R0, R_ph)",
                ),
                _ref(
                    _KR_CURRENT_SHEATH,
                    "656-662",
                    "prose: ionization path length Liz(P), Liz/Li = 2.4 at "
                    "optimal pressure",
                ),
            ),
            units={
                "ionization_fraction": "dimensionless",
                "ionization_source_rate_mol_m3_s": "mol m^-3 s^-1",
                "alpha": "1/m",
                "beta_ep_m3_s": "m^3 s^-1",
            },
            symbol_map={
                "R_e": "electron hydrodynamic source term",
                "alpha": "primary ionization coefficient",
                "eta": "attachment coefficient",
                "beta_ep": "electron-ion recombination coefficient",
                "R_ph": "photoionization rate",
            },
            input_dependencies=(
                "breakdown.alpha_packet",
                "gas.species",
                "gas.pressure_Pa",
            ),
            output_fields=(
                "ionization_fraction",
                "ionization_source_rate_mol_m3_s",
            ),
            blocker_reason=(
                "R_e is a generic gas-discharge hydrodynamic source term; the "
                "corpus supplies no DPF-specific alpha(E/p), beta_ep, R_ph and "
                "the Liz/Li=2.4 ratio is an empirical fit, not a closure"
            ),
            blocker_id="STARTUP-BVP-CH10-IONIZATION-NO-DPF-COEFFICIENTS",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M2",),
        ),
        StartupChannel(
            channel_id="electron_and_ion_temperature",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_BREAKDOWN_NOBLE,
                    "160-167,169-172",
                    "Eq. (4) Te = xi lambda e U/d (homogeneous-field only); "
                    "Eq. (5) mean free-electron energy",
                ),
                _ref(
                    _KR_CURRENT_SHEATH,
                    "643-655",
                    "prose: initial plasma 'a few eV'; Te ~ 4 eV analysis "
                    "assumption",
                ),
            ),
            units={
                "electron_temperature_K": "K",
                "ion_temperature_K": "K",
                "lambda_m": "m",
            },
            symbol_map={
                "Te": "free-electron temperature",
                "Ti": "ion temperature",
                "xi": "thermalization form factor",
                "lambda": "electron mean free path",
                "delta": "electron-mass to gas-molar-mass ratio",
            },
            input_dependencies=(
                "initial_e_b_j.field",
                "gas.species",
                "breakdown.mean_free_path",
            ),
            output_fields=(
                "electron_temperature_K",
                "ion_temperature_K",
            ),
            blocker_reason=(
                "Eq. (4) is valid only for a homogeneous/pseudo-homogeneous "
                "field; the DPF coaxial gap is inhomogeneous, and Te ~ 4 eV "
                "is an analysis assumption, not a DPF-valid closure"
            ),
            blocker_id="STARTUP-BVP-CH11-TEMPERATURE-NO-DPF-VALID-RELATION",
            first_principles_claim_effect=(
                "candidate input only; homogeneous-field relation invalid for "
                "the DPF gap, cannot support first-principles acceptance"
            ),
            missing_parameter_ids=("M7", "M9"),
        ),
        StartupChannel(
            channel_id="sheath_surface_liftoff",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_HYBRID_PIC,
                    "607-614,703-708",
                    "prose + Fig. 4: Yee staggered grid placement; "
                    "end-of-rundown sheath thickness 0.15-0.20 cm",
                ),
                _ref(
                    _KR_SAND2009,
                    "317-323",
                    "prose + Figure 1: DPF phases breakdown, lift-off, "
                    "run-down, pinch",
                ),
            ),
            units={
                "sheath_thickness_m": "m",
                "sheath_drift_velocity_m_s": "m s^-1",
                "sheath_density_m3": "m^-3",
            },
            symbol_map={
                "delta_sheath": "sheath axial thickness",
                "vd": "sheath axial drift velocity",
            },
            input_dependencies=(
                "surface_plasma.output",
                "end_of_rundown_handoff.engineering_mode",
            ),
            output_fields=(
                "sheath_mask",
                "sheath_thickness_m",
                "sheath_drift_velocity_m_s",
                "sheath_density_m3",
            ),
            blocker_reason=(
                "corpus supplies an end-of-rundown measured sheath thickness, "
                "not a closed breakdown-BVP sheath-surface initial state "
                "(mask, thickness, density, conductivity, velocity)"
            ),
            blocker_id="STARTUP-BVP-CH12-SHEATH-NO-BREAKDOWN-BVP-STATE",
            first_principles_claim_effect=(
                "candidate input only; end-of-rundown scope, cannot support "
                "first-principles whole-shot acceptance"
            ),
            missing_parameter_ids=("M1", "M7"),
        ),
        StartupChannel(
            channel_id="handoff_interval_into_3d_solver",
            status="candidate",
            source_refs=(
                _ref(
                    _KR_SAND2009,
                    "470-475,682-690",
                    "prose: arbitrary '1 eV thin layer' seed; ALEGRA imports "
                    "PIC ion/electron densities, temperatures, B-field",
                ),
                _ref(
                    _KR_ALEGRA,
                    "268-272",
                    "prose: ALEGRA capability to import PIC-derived data to "
                    "initiate MHD simulations",
                ),
            ),
            units={
                "handoff_start_time_s": "s",
                "handoff_end_time_s": "s",
                "handoff_tolerance_s": "s",
            },
            symbol_map={
                "t_start": "start of the voltage discharge",
                "t_handoff": "sheath-liftoff complete; MHD rundown begins",
            },
            input_dependencies=(
                "breakdown.output",
                "flashover.output",
                "sheath_surface_liftoff.output",
                "mhd_solver.readiness",
            ),
            output_fields=(
                "handoff_start_time_s",
                "handoff_end_time_s",
                "handoff_tolerance_s",
                "handoff_mode",
            ),
            blocker_reason=(
                "corpus supplies no numerical handoff-interval definition "
                "(t_start, t_handoff, tolerance) and no same-device reviewed "
                "PIC import payload; the '1 eV thin layer' seed is called "
                "arbitrary by the source itself"
            ),
            blocker_id="STARTUP-BVP-CH13-HANDOFF-NO-NUMERICAL-DEFINITION",
            first_principles_claim_effect=(
                "candidate input only; cannot support first-principles "
                "acceptance"
            ),
            missing_parameter_ids=("M8",),
        ),
    )


STARTUP_BVP_CHANNELS: tuple[StartupChannel, ...] = _build_startup_channel_registry()


def build_startup_packet() -> StartupPacket:
    """Return the typed S3.4 startup BVP channel packet.

    The packet enumerates every whole-shot startup channel as a typed
    ``StartupChannel`` record. Per the WP-N2 research packet no channel is
    source-``computed`` for a DPF-specific startup BVP closure, so the packet's
    ``can_support_first_principles_acceptance`` is always ``False`` and every
    non-``computed`` channel contributes a blocker ID that blocks startup
    authority.
    """

    channels = STARTUP_BVP_CHANNELS
    blocking_channels = tuple(
        channel for channel in channels if not channel.supports_first_principles
    )
    blocker_ids = tuple(channel.blocker_id for channel in blocking_channels)
    can_support = len(blocking_channels) == 0
    status = (
        "startup_channel_packet_supports_first_principles"
        if can_support
        else "blocked_startup_channel_packet_no_computed_channel"
    )
    return StartupPacket(
        channels=channels,
        status=status,
        blocker_ids=blocker_ids,
        can_support_first_principles_acceptance=can_support,
    )
