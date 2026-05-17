"""Fail-closed numerical-fidelity packets for first-principles DPF runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def build_numerical_fidelity_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    conservation: Mapping[str, Any] | None = None,
    simulation_telemetry: Mapping[str, Any] | None = None,
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting numerical-fidelity gate packet."""

    accepted = {str(channel) for channel in accepted_channels}
    missing = set(REQUIRED_NUMERICAL_FIDELITY_CHANNELS) - accepted
    missing.update(REQUIRED_NUMERICAL_FIDELITY_CHANNELS)

    return {
        "status": "blocked_numerical_fidelity_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_accept_numerical_fidelity",
        "required_test_surfaces": list(REQUIRED_NUMERICAL_TEST_SURFACES),
        "required_channels": list(REQUIRED_NUMERICAL_FIDELITY_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "numerical_channel_status": _numerical_channel_statuses(
            accepted=accepted,
            missing=missing,
        ),
        "test_surface_status": _test_surface_statuses(accepted),
        "candidate_runtime_channels": _candidate_runtime_channels(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
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


def _test_surface_statuses(accepted: set[str]) -> dict[str, dict[str, Any]]:
    coverage = {str(item["surface"]): item for item in CANDIDATE_TEST_SURFACE_COVERAGE}
    statuses: dict[str, dict[str, Any]] = {}
    for surface in REQUIRED_NUMERICAL_TEST_SURFACES:
        surface_coverage = coverage.get(surface)
        declared_accepted = surface in accepted
        if declared_accepted:
            status = "accepted_test_surface_declared_packet_still_blocked"
        elif surface_coverage is not None:
            status = str(surface_coverage["coverage_status"])
        else:
            status = "missing_or_blocked"
        statuses[surface] = {
            "status": status,
            "candidate_artifacts": list(
                surface_coverage.get("candidate_artifacts", ())
                if surface_coverage is not None
                else ()
            ),
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
        observations["conservation_passed"] = conservation.get("passed")
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
