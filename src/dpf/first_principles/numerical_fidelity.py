"""Fail-closed numerical-fidelity packets for first-principles DPF runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpf.first_principles.channel_state import (
    ACCEPTED,
    BLOCKED_MISSING_SOURCE,
    ChannelState,
    channel_state_map,
    channel_state_summary,
)

NUMERICAL_FIDELITY_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "243-260",
        "role": "first_principles_numerical_verification_test_surfaces",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "410-424,609-645,1018-1110",
        "role": "hybrid_pic_yield_resolution_cfl_marder_and_parameter_sensitivity_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "particle-simulation-of-plasmas-review-and-advances-6d7355ba.md"
        ),
        "lines": "456-530,671-705,744-755",
        "role": "pic_yee_maxwell_courant_and_charge_conservation_method_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md"
        ),
        "lines": "55-90,429-500",
        "role": "finite_volume_constrained_transport_resistive_timestep_and_convergence_context",
    },
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "2444-2576",
        "role": "shock_fixture_physics_context",
    },
)

REQUIRED_NUMERICAL_TEST_SURFACES = (
    "finite_volume_shock_behavior",
    "cylindrical_source_terms",
    "maxwell_yee_update_and_courant_limit",
    "divergence_b_control",
    "gauss_law_or_charge_current_continuity",
    "resistive_diffusion",
    "joule_heating_and_total_energy",
    "circuit_power_port_coupling",
    "particle_push_and_current_deposition",
    "mesh_and_timestep_convergence",
    "restart_reproducibility",
    "backend_and_precision_parity",
    "limiter_zero_acceptance",
)

REQUIRED_NUMERICAL_FIDELITY_CHANNELS = (
    "test_surface_registry",
    "source_backed_numerical_method_map",
    "analytic_or_manufactured_reference_solutions",
    "mesh_family_definitions",
    "timestep_family_definitions",
    "norms_by_test_surface",
    "tolerances_by_test_surface",
    "observed_order_or_monotonic_convergence",
    "finite_volume_shock_packet",
    "cylindrical_source_term_packet",
    "maxwell_yee_courant_packet",
    "divergence_b_packet",
    "gauss_law_or_continuity_packet",
    "resistive_diffusion_packet",
    "joule_heating_energy_packet",
    "circuit_power_port_numerical_packet",
    "particle_push_deposition_packet",
    "mesh_timestep_convergence_packet",
    "restart_reproducibility_packet",
    "backend_precision_parity_packet",
    "limiter_zero_packet",
    "same_scope_numerical_observable_mapping",
    "artifact_links_and_hashes",
    "negative_tests_for_failed_tolerance",
    "independent_review_certificate",
)

EXISTING_TEST_SURFACES = (
    {
        "path": "tests/test_maxwell_3d_field_core.py",
        "surface": "maxwell_yee_update_and_divergence_diagnostics",
        "status": "candidate_component_test_not_acceptance",
    },
    {
        "path": "tests/test_marder_correction.py",
        "surface": "gauss_law_marder_residual_and_nondominance",
        "status": "candidate_component_test_not_acceptance",
    },
    {
        "path": "tests/test_pic_current_source_port.py",
        "surface": "pic_current_to_yee_edge_mapping",
        "status": "candidate_component_test_not_acceptance",
    },
    {
        "path": "tests/test_hybrid_3d_loop.py",
        "surface": "hybrid_pic_field_loop_and_candidate_telemetry",
        "status": "candidate_component_test_not_acceptance",
    },
    {
        "path": "tests/test_circuit_magnetic_boundary.py",
        "surface": "external_circuit_magnetic_boundary_slice",
        "status": "candidate_component_test_not_acceptance",
    },
    {
        "path": "tests/test_first_principles_runner.py",
        "surface": "package_native_manifest_and_conservation_telemetry",
        "status": "candidate_component_test_not_acceptance",
    },
)

CANDIDATE_TEST_SURFACE_COVERAGE = (
    {
        "surface": "finite_volume_shock_behavior",
        "candidate_artifacts": (
            "legacy_mhd_verification_surfaces_not_attached_to_package_native_fp4",
        ),
        "coverage_status": "legacy_candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "cylindrical_source_terms",
        "candidate_artifacts": (
            "legacy_cylindrical_source_term_tests_not_attached_to_package_native_fp4",
        ),
        "coverage_status": "legacy_candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "maxwell_yee_update_and_courant_limit",
        "candidate_artifacts": ("tests/test_maxwell_3d_field_core.py",),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "divergence_b_control",
        "candidate_artifacts": (
            "tests/test_maxwell_3d_field_core.py",
            "tests/test_marder_correction.py",
        ),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "gauss_law_or_charge_current_continuity",
        "candidate_artifacts": (
            "tests/test_marder_correction.py",
            "tests/test_pic_current_source_port.py",
        ),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "resistive_diffusion",
        "candidate_artifacts": (
            "legacy_resistive_diffusion_surfaces_not_attached_to_package_native_fp4",
        ),
        "coverage_status": "legacy_candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "joule_heating_and_total_energy",
        "candidate_artifacts": (
            "tests/test_hybrid_3d_loop.py",
            "tests/test_first_principles_runner.py",
        ),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "circuit_power_port_coupling",
        "candidate_artifacts": (
            "tests/test_circuit_magnetic_boundary.py",
            "tests/test_first_principles_runner.py",
        ),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
    {
        "surface": "particle_push_and_current_deposition",
        "candidate_artifacts": (
            "tests/test_pic_current_source_port.py",
            "tests/test_hybrid_3d_loop.py",
        ),
        "coverage_status": "candidate_component_coverage_not_acceptance",
    },
)

REQUIRED_UPSTREAM_NUMERICAL_PACKETS = (
    "startup_bvp",
    "limiter_readiness",
    "power_port",
    "dimensionality_handoff",
    "physics_closure",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE3_TRANSFER_MATRIX_PATH = (
    _REPO_ROOT / "docs/SS12_P1_PHASE3_TRANSFER_CANDIDATE_MATRIX_2026_05_22.json"
)


def build_numerical_fidelity_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    conservation: Mapping[str, Any] | None = None,
    simulation_telemetry: Mapping[str, Any] | None = None,
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
    phase3_transfer_matrix_path: str | Path | None = DEFAULT_PHASE3_TRANSFER_MATRIX_PATH,
) -> dict[str, Any]:
    """Return a non-promoting numerical-fidelity gate packet."""

    accepted = {str(channel) for channel in accepted_channels}
    # Canonical per-channel states (Codex S7-A7): a channel is either
    # ``accepted`` or ``blocked_missing_source``; it can never be both
    # accepted and missing.
    channel_states = _numerical_channel_states(accepted)
    state_summary = channel_state_summary(channel_states)
    missing = set(state_summary["missing_acceptance_channels"])
    phase3_transfer_linkage = load_phase3_transfer_candidate_linkage(
        phase3_transfer_matrix_path
    )

    return {
        "status": "blocked_numerical_fidelity_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_accept_numerical_fidelity",
        "required_test_surfaces": list(REQUIRED_NUMERICAL_TEST_SURFACES),
        "required_channels": list(REQUIRED_NUMERICAL_FIDELITY_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "channel_states": channel_state_map(channel_states),
        "channel_state_summary": state_summary,
        "numerical_channel_status": _numerical_channel_statuses(
            accepted=accepted,
            missing=missing,
        ),
        "test_surface_status": _test_surface_statuses(
            accepted,
            upstream_packets=upstream_packets,
        ),
        "candidate_runtime_channels": _candidate_runtime_channels(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
            upstream_packets=upstream_packets,
        ),
        "phase3_transfer_candidate_linkage": phase3_transfer_linkage,
        "phase4a_transfer_linkage_gate": _phase4a_transfer_linkage_gate(
            phase3_transfer_linkage
        ),
        "runtime_observations": _runtime_observations(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        ),
        "existing_test_surfaces": [dict(surface) for surface in EXISTING_TEST_SURFACES],
        "upstream_packet_statuses": _upstream_statuses(upstream_packets),
        "upstream_acceptance_gate": _upstream_acceptance_gate(upstream_packets),
        "source_references": list(NUMERICAL_FIDELITY_SOURCE_REFS),
        "acceptance_gate": (
            "candidate_component_tests_and_runtime_diagnostics_cannot_support_"
            "numerical_acceptance_until_all_test_surfaces_have_source_backed_"
            "references_norms_tolerances_convergence_limiter_zero_backend_scope_"
            "artifact_hashes_negative_tests_and_review"
        ),
        "validation_rule": (
            "No numerical packet can pass with unspecified tolerances, missing "
            "convergence evidence, hidden limiter activity, or absent review."
        ),
        "negative_test_policy": {
            "failed_tolerance_negative_required": True,
            "hidden_limiter_regression_required": True,
            "backend_precision_fallback_rejection_required": True,
            "gauss_or_divergence_residual_failure_required": True,
            "conservation_residual_failure_required": True,
            "restart_mismatch_required": True,
            "candidate_component_promotion_rejection_required": True,
        },
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def load_phase3_transfer_candidate_linkage(
    matrix_path: str | Path | None = DEFAULT_PHASE3_TRANSFER_MATRIX_PATH,
) -> dict[str, Any]:
    """Load Phase 3 transfer candidates as non-accepting linkage metadata."""

    if matrix_path is None:
        return _blocked_transfer_linkage(
            None,
            status="blocked_transfer_matrix_path_not_declared",
            reason="phase3_transfer_matrix_path_not_declared",
        )

    path = Path(matrix_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        return _blocked_transfer_linkage(
            path,
            status="blocked_transfer_matrix_missing",
            reason="phase3_transfer_matrix_missing",
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return _blocked_transfer_linkage(
            path,
            status="blocked_transfer_matrix_invalid_json",
            reason="phase3_transfer_matrix_invalid_json",
        )

    if not isinstance(raw, Mapping):
        return _blocked_transfer_linkage(
            path,
            status="blocked_transfer_matrix_invalid_schema",
            reason="phase3_transfer_matrix_not_object",
        )

    boundary = _mapping(raw.get("acceptance_boundary"))
    boundary_is_non_promoting = (
        boundary.get("promotes_acceptance") is False
        and boundary.get("can_fill_same_scope_channel") is False
        and boundary.get("requires_transfer_rule_review") is True
    )
    raw_candidates = raw.get("transfer_candidates")
    if not isinstance(raw_candidates, list):
        return _blocked_transfer_linkage(
            path,
            status="blocked_transfer_matrix_invalid_schema",
            reason="phase3_transfer_candidates_not_list",
        )

    transfer_candidates = [
        _transfer_candidate_summary(row)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    blocking_reasons: list[str] = []
    if not boundary_is_non_promoting:
        blocking_reasons.append("phase3_transfer_boundary_not_non_promoting")
    if len(transfer_candidates) != len(raw_candidates):
        blocking_reasons.append("phase3_transfer_candidate_row_not_object")
    if not transfer_candidates:
        blocking_reasons.append("phase3_transfer_candidates_missing")

    status = (
        "loaded_transfer_candidates_non_promoting"
        if not blocking_reasons
        else "blocked_transfer_matrix_invalid_non_promotion_contract"
    )
    return {
        "status": status,
        "matrix_path": _display_path(path),
        "matrix_id": raw.get("matrix_id"),
        "validation_scope": raw.get("validation_scope"),
        "same_source_matrix": raw.get("same_source_matrix"),
        "acceptance_boundary": dict(boundary),
        "transfer_candidate_channels": sorted(
            {row["channel"] for row in transfer_candidates}
        ),
        "accepted_source_channels": [],
        "transfer_candidates": transfer_candidates,
        "transfer_candidate_count": len(transfer_candidates),
        "global_blockers": list(raw.get("global_blockers", ())),
        "all_transfer_candidates_non_promoting": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "promotes_acceptance": False,
        "can_fill_same_scope_channel": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _blocked_transfer_linkage(
    path: Path | None,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "matrix_path": None if path is None else _display_path(path),
        "matrix_id": None,
        "validation_scope": None,
        "same_source_matrix": None,
        "acceptance_boundary": {
            "promotes_acceptance": False,
            "can_fill_same_scope_channel": False,
            "requires_transfer_rule_review": True,
        },
        "transfer_candidate_channels": [],
        "accepted_source_channels": [],
        "transfer_candidates": [],
        "transfer_candidate_count": 0,
        "global_blockers": [],
        "all_transfer_candidates_non_promoting": False,
        "blocking_reasons": [reason],
        "promotes_acceptance": False,
        "can_fill_same_scope_channel": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _transfer_candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "channel": str(row.get("channel", "unknown_channel")),
        "status": str(row.get("status", "unknown_status")),
        "source_path": row.get("source_path"),
        "line_start": row.get("line_start"),
        "line_end": row.get("line_end"),
        "scope_assessment": row.get("scope_assessment"),
        "source_channel_role": "transfer_candidate_not_accepted_channel",
        "requires_transfer_rule_review": True,
        "promotes_acceptance": False,
        "can_fill_same_scope_channel": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _phase4a_transfer_linkage_gate(
    linkage: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(linkage.get("status"))
    transfer_matrix_available = status != "blocked_transfer_matrix_missing"
    if status == "blocked_transfer_matrix_missing":
        gate_status = "blocked_by_missing_phase3_transfer_matrix"
    elif status == "loaded_transfer_candidates_non_promoting":
        gate_status = "transfer_candidates_linked_non_promoting_packet_still_blocked"
    else:
        gate_status = "blocked_by_invalid_phase3_transfer_matrix"
    return {
        "status": gate_status,
        "transfer_matrix_available": transfer_matrix_available,
        "transfer_candidate_count": int(linkage.get("transfer_candidate_count", 0) or 0),
        "accepted_source_channel_count": len(linkage.get("accepted_source_channels", ())),
        "all_transfer_candidates_non_promoting": bool(
            linkage.get("all_transfer_candidates_non_promoting")
        ),
        "blocking_reasons": list(linkage.get("blocking_reasons", ())),
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _numerical_channel_states(accepted: set[str]) -> dict[str, ChannelState]:
    """Map every required numerical-fidelity channel onto a canonical state.

    A channel is ``accepted`` only when the deck declares it; otherwise it is
    ``blocked_missing_source`` -- no numerical reference/test artifact exists.
    """

    return {
        channel: (ACCEPTED if channel in accepted else BLOCKED_MISSING_SOURCE)
        for channel in REQUIRED_NUMERICAL_FIDELITY_CHANNELS
    }


def _numerical_channel_statuses(
    *,
    accepted: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_NUMERICAL_FIDELITY_CHANNELS:
        if channel in accepted:
            statuses[channel] = "accepted_numerical_channel_declared"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _test_surface_statuses(
    accepted: set[str],
    *,
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    coverage = {str(item["surface"]): item for item in CANDIDATE_TEST_SURFACE_COVERAGE}
    statuses: dict[str, dict[str, Any]] = {}
    limiter_zero = _mapping((upstream_packets or {}).get("experimental_limiter_zero_probe"))
    for surface in REQUIRED_NUMERICAL_TEST_SURFACES:
        surface_coverage = coverage.get(surface)
        declared_accepted = surface in accepted
        candidate_artifacts = list(
            surface_coverage.get("candidate_artifacts", ())
            if surface_coverage is not None
            else ()
        )
        if declared_accepted:
            status = "accepted_test_surface_declared_packet_still_blocked"
        elif (
            surface == "limiter_zero_acceptance"
            and limiter_zero.get("zero_acceptance_blockers_observed") is True
        ):
            status = "candidate_runtime_limiter_zero_observed_not_acceptance"
            candidate_artifacts.append("runtime_experimental_limiter_zero_probe")
        elif surface_coverage is not None:
            status = str(surface_coverage["coverage_status"])
        else:
            status = "missing_or_blocked"
        statuses[surface] = {
            "status": status,
            "candidate_artifacts": candidate_artifacts,
            "acceptance_rule": (
                "requires_source_backed_reference_solution_norm_tolerance_"
                "convergence_artifacts_hashes_limiter_zero_scope_and_review"
            ),
            "can_support_numerical_acceptance": False,
        }
    return statuses


def _runtime_observations(
    *,
    conservation: Mapping[str, Any] | None,
    simulation_telemetry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observations: dict[str, Any] = {
        "candidate_only": True,
        "accepted_numerical_packet": False,
        "tolerance_claim": False,
        "convergence_claim": False,
    }
    if conservation:
        observations["conservation_finite_state"] = conservation.get("finite_state")
        observations["final_max_abs_div_B_T_per_m"] = conservation.get(
            "final_max_abs_div_B_T_per_m"
        )
        observations["relative_tracked_total_energy_change"] = conservation.get(
            "relative_tracked_total_energy_change"
        )
    if simulation_telemetry:
        observations["simulation_status"] = simulation_telemetry.get("status")
        observations["n_steps_completed"] = simulation_telemetry.get(
            "n_steps_completed"
        )
        observations["has_circuit_boundary_runtime"] = (
            simulation_telemetry.get("circuit") is not None
        )
    return observations


def _candidate_runtime_channels(
    *,
    conservation: Mapping[str, Any] | None,
    simulation_telemetry: Mapping[str, Any] | None,
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> list[str]:
    channels: set[str] = set()
    if conservation:
        if conservation.get("status") == "engineering_candidate_conservation_telemetry_not_validation":
            channels.add("candidate_conservation_telemetry")
        if conservation.get("final_max_abs_div_B_T_per_m") is not None:
            channels.add("candidate_divergence_b_diagnostic")
        if conservation.get("relative_tracked_total_energy_change") is not None:
            channels.add("candidate_tracked_total_energy_delta")
    if simulation_telemetry:
        if str(simulation_telemetry.get("status", "")).startswith(
            "candidate_engineering_"
        ):
            channels.add("candidate_hybrid_pic_3d_simulation")
        if simulation_telemetry.get("circuit") is not None:
            channels.add("candidate_circuit_coupled_run")
        last_step = simulation_telemetry.get("last_step")
        if isinstance(last_step, Mapping):
            if last_step.get("source_ordered_loop") is not None:
                channels.add("candidate_source_ordered_loop")
            if last_step.get("electron_energy") is not None:
                channels.add("candidate_electron_energy_source_update")
            if last_step.get("kinetic_yield") is not None:
                channels.add("candidate_kinetic_yield_history")
    limiter_zero = _mapping((upstream_packets or {}).get("experimental_limiter_zero_probe"))
    if limiter_zero.get("zero_acceptance_blockers_observed") is True:
        channels.add("candidate_limiter_zero_no_applied_blockers_observed")
    power_port = _mapping((upstream_packets or {}).get("power_port"))
    residual_budget = _mapping(power_port.get("candidate_power_residual_budget"))
    if residual_budget.get("full_completed_step_j_dot_e_integral_available") is True:
        channels.add("candidate_full_completed_step_j_dot_e_integral")
    if residual_budget.get("cumulative_terminal_active_port_work_J") is not None:
        channels.add("candidate_cumulative_terminal_i_udpf_work")
    if (
        residual_budget.get("full_completed_step_active_port_integral_available")
        is True
    ):
        channels.add("candidate_full_completed_step_terminal_i_udpf_integral")
    return sorted(channels)


def _upstream_statuses(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    for name, packet in (upstream_packets or {}).items():
        statuses[str(name)] = (
            None if packet.get("status") is None else str(packet["status"])
        )
    return statuses


def _upstream_acceptance_gate(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    statuses = _upstream_statuses(upstream_packets)
    blockers = {
        name: statuses.get(name)
        for name in REQUIRED_UPSTREAM_NUMERICAL_PACKETS
        if not _status_is_accepted(statuses.get(name))
    }
    return {
        "status": (
            "blocked_by_upstream_packets"
            if blockers
            else "upstream_packets_accepted_numerical_channels_still_required"
        ),
        "required_upstream_packets": list(REQUIRED_UPSTREAM_NUMERICAL_PACKETS),
        "blocking_upstream_packets": blockers,
        "all_required_upstream_packets_accepted": not blockers,
    }


def _status_is_accepted(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().lower()
    return normalized.startswith("accepted") or normalized in {"passed", "ready"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
