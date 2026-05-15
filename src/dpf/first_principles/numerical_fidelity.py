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
        "candidate_runtime_channels": _candidate_runtime_channels(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        ),
        "existing_test_surfaces": [dict(surface) for surface in EXISTING_TEST_SURFACES],
        "upstream_packet_statuses": _upstream_statuses(upstream_packets),
        "source_references": list(NUMERICAL_FIDELITY_SOURCE_REFS),
        "validation_rule": (
            "No numerical packet can pass with unspecified tolerances, missing "
            "convergence evidence, hidden limiter activity, or absent review."
        ),
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


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
