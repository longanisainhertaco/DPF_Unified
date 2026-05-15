"""Fail-closed dimensionality and handoff packets for first-principles runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DIMENSIONALITY_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "246-320,468-606,1210-1230,1270-1280",
        "role": "hybrid_pic_fluid_architecture_and_limitations",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md"
        ),
        "lines": "352-380,474-546,614-624",
        "role": "3d_mhd_rundown_and_pic_handoff",
    },
    {
        "path": "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md",
        "lines": "34-80,152-168",
        "role": "fully_kinetic_beam_target_authority",
    },
)

REQUIRED_HANDOFF_CHANNELS = (
    "claim_mode",
    "geometry_dimensionality",
    "active_solver_family",
    "claim_boundary",
    "resolved_m_modes",
    "maxwell_fields",
    "kinetic_ion_distribution",
    "electron_model",
    "source_model_limitation_registry",
    "mhd_to_kinetic_state_transfer",
    "current_charge_consistency",
    "energy_momentum_transfer",
    "accepted_electron_energy_closure",
    "kinetic_electron_or_scope_bound",
    "mechanism_separated_neutron_history",
    "same_scope_3d_evidence",
    "source_references",
)

BLOCKED_OBSERVABLES = (
    "unrestricted_whole_shot_authority",
    "m1_kink_and_higher_azimuthal_modes",
    "pinch_fragmentation_lifetime",
    "electron_kinetic_instabilities",
    "beam_target_neutron_authority",
    "detector_response_and_uq",
)

CLAIM_MODES = (
    {
        "mode": "bounded_axisymmetric_mhd_claim",
        "status": "interim_only",
        "allowed_claim": "pre-breakdown comparator or narrowed macroscopic scope",
    },
    {
        "mode": "validated_3d_mhd_rundown_claim",
        "status": "requires_review",
        "allowed_claim": "3D electrode-geometry and rundown behavior only",
    },
    {
        "mode": "mhd_to_kinetic_handoff_claim",
        "status": "required_for_mhd_plus_pic_path",
        "allowed_claim": "handoff only after transferred state and conservation evidence pass",
    },
    {
        "mode": "validated_3d_hybrid_pic_fluid_claim",
        "status": "target_engineering_path_blocked_for_acceptance",
        "allowed_claim": "3D hybrid engineering candidate until source-equivalence and same-scope evidence pass",
    },
    {
        "mode": "fully_kinetic_pinch_claim",
        "status": "required_for_unrestricted_beam_target_authority_if_hybrid_limits_dominate",
        "allowed_claim": "beam-target authority only with kinetic electrons/ions and detector/UQ packet",
    },
)

SOURCE_MODEL_LIMITATIONS = (
    "source_hybrid_reference_axisymmetric_m0_not_full_3d",
    "quasineutral_electron_fluid_no_debye_sheath_microphysics",
    "kinetic_ions_only_no_kinetic_electrons",
    "separate_electron_energy_not_accepted",
    "m1_kink_and_higher_m_modes_not_accepted",
    "no_reviewed_mhd_to_kinetic_state_transfer",
    "no_same_scope_3d_pf1000_akel_validation_packet",
)


def build_dimensionality_handoff_packet(
    *,
    grid_shape: tuple[int, int, int] | list[int],
    run_mode: str,
    startup: Mapping[str, Any] | None = None,
    power_port: Mapping[str, Any] | None = None,
    simulation_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-promoting dimensionality and kinetic-handoff metadata."""
    shape = tuple(int(v) for v in grid_shape)
    is_3d_grid = len(shape) == 3 and all(v > 1 for v in shape)
    missing = {
        "same_scope_3d_validation_packet",
        "source_equivalence_review",
        "mhd_to_kinetic_handoff_state",
        "claim_boundary_and_valid_interval",
        "separate_electron_energy_authority",
        "kinetic_electron_or_bounded_out_electron_kinetics",
        "source_model_limitation_review",
        "fully_kinetic_or_reviewed_beam_target_interval",
        "mechanism_separated_detector_uq_packet",
    }
    if not is_3d_grid:
        missing.add("true_3d_grid")
    if startup and startup.get("whole_shot_startup_blocked") is True:
        missing.add("accepted_startup_interval")
    if power_port and power_port.get("can_support_first_principles_acceptance") is False:
        missing.add("accepted_power_port_packet")

    return {
        "status": "candidate_engineering_dimensionality_handoff_not_validation",
        "claim_mode": "validated_3d_hybrid_pic_fluid_claim_candidate",
        "allowed_claim": "engineering_3d_hybrid_em_pic_fluid_candidate_only",
        "decision": "do_not_claim_unrestricted_whole_shot_dimensionality_authority",
        "claim_modes": [dict(mode) for mode in CLAIM_MODES],
        "active_solver_family": run_mode,
        "geometry_dimensionality": "3d" if is_3d_grid else "not_true_3d",
        "grid_shape": list(shape),
        "resolved_m_modes": "cartesian_3d_grid_present_but_m_mode_evidence_not_accepted",
        "claim_boundary": "not_accepted_for_whole_shot_or_unrestricted_neutron_authority",
        "mhd_to_kinetic_handoff": "not_present",
        "handoff_required_for_observables": [
            "nonthermal_ion_tail",
            "beam_target_neutron_yield",
            "neutron_spectrum",
            "neutron_anisotropy",
            "detector_response_and_uq",
            "late_pinch_electron_kinetic_instabilities",
        ],
        "electron_model": "quasineutral_fluid_candidate",
        "kinetic_ion_model": "pic_macroparticle_candidate",
        "kinetic_electron_model": "not_present",
        "source_model_limitations": list(SOURCE_MODEL_LIMITATIONS),
        "blocked_observables": list(BLOCKED_OBSERVABLES),
        "required_channels": list(REQUIRED_HANDOFF_CHANNELS),
        "missing_acceptance_channels": sorted(missing),
        "candidate_runtime_channels": _candidate_runtime_channels(
            is_3d_grid=is_3d_grid,
            run_mode=run_mode,
            startup=startup,
            power_port=power_port,
            simulation_telemetry=simulation_telemetry,
        ),
        "source_references": list(DIMENSIONALITY_SOURCE_REFS),
        "can_support_first_principles_acceptance": False,
    }


def _candidate_runtime_channels(
    *,
    is_3d_grid: bool,
    run_mode: str,
    startup: Mapping[str, Any] | None,
    power_port: Mapping[str, Any] | None,
    simulation_telemetry: Mapping[str, Any] | None,
) -> list[str]:
    channels: set[str] = set()
    if is_3d_grid:
        channels.add("candidate_true_3d_grid")
    if "hybrid" in run_mode and "pic" in run_mode:
        channels.add("candidate_hybrid_pic_fluid_solver_family")
    if startup is not None:
        channels.add("candidate_startup_packet_linked")
    if power_port is not None:
        channels.add("candidate_power_port_packet_linked")
    if simulation_telemetry:
        if str(simulation_telemetry.get("status", "")).startswith(
            "candidate_engineering_"
        ):
            channels.add("candidate_package_native_simulation_telemetry")
        last_step = simulation_telemetry.get("last_step")
        if isinstance(last_step, Mapping):
            if last_step.get("source_ordered_loop") is not None:
                channels.add("candidate_source_ordered_loop")
            if last_step.get("electron_energy") is not None:
                channels.add("candidate_separate_electron_energy_scaffold")
            if last_step.get("kinetic_yield") is not None:
                channels.add("candidate_kinetic_yield_history")
    return sorted(channels)
