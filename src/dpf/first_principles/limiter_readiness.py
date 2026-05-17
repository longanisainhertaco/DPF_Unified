"""Fail-closed limiter-readiness packets for first-principles DPF runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)

LIMITER_READINESS_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "61-63,159-160,210-239",
        "role": "first_principles_limiter_registry_and_limiter_zero_contract",
    },
    {
        "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
        "lines": "64",
        "role": "hidden_engineering_limiters_must_block_accepted_claims",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "410-424,582-605,1046-1067",
        "role": "marder_and_ohmic_cfl_limiter_sensitivity_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md"
        ),
        "lines": "326,471-500,952",
        "role": "slope_limiter_and_resistive_timestep_method_context",
    },
)

REQUIRED_LIMITER_READINESS_CHANNELS = (
    "active_path_limiter_inventory",
    "limiter_event_schema",
    "code_path_and_field_mapping",
    "classification_by_limiter",
    "activation_count_by_limiter",
    "before_after_minmax_by_limiter",
    "nonfinite_count_by_limiter",
    "source_or_method_justification",
    "readiness_effect_by_limiter",
    "source_backed_physical_bounds",
    "verified_numerical_method_bounds",
    "zero_acceptance_blocker_full_run",
    "full_horizon_run_manifest",
    "backend_precision_fallback_inventory",
    "fallback_rejection_tests",
    "synthetic_acceptance_blocker_negative_test",
    "app_only_runner_rejection_test",
    "artifact_links_and_hashes",
    "independent_review_certificate",
)

KNOWN_LIMITER_FAMILIES = (
    {
        "family": "state_mutating_floor_cap_clip",
        "examples": (
            "density_floor",
            "temperature_floor_or_cap",
            "pressure_floor",
            "velocity_or_current_clip",
            "back_emf_clip",
        ),
        "acceptance_rule": "blocked unless source-backed physical bound or verified numerical method is attached",
    },
    {
        "family": "method_limiter_or_stability_guard",
        "examples": (
            "finite_volume_slope_limiter",
            "ohmic_cfl_conductivity_limiter",
            "marder_correction",
            "resistive_timestep_constraint",
        ),
        "acceptance_rule": "candidate only until nondominance, convergence, and sensitivity evidence are accepted",
    },
    {
        "family": "repair_or_fallback",
        "examples": (
            "nonfinite_state_repair",
            "backend_precision_fallback",
            "unsupported_physics_fallback",
            "surrogate_or_reduced_model_fallback",
        ),
        "acceptance_rule": "blocks accepted first-principles claims",
    },
)


def build_limiter_readiness_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    conservation: Mapping[str, Any] | None = None,
    simulation_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting limiter/readiness packet."""

    accepted = {str(channel) for channel in accepted_channels}
    missing = set(REQUIRED_LIMITER_READINESS_CHANNELS) - accepted
    missing.update(REQUIRED_LIMITER_READINESS_CHANNELS)
    limiter_zero_probe = build_experimental_limiter_zero_probe_packet(
        declared_scope=declared_scope,
        device_name=device_name or "not_declared",
        simulation_telemetry=simulation_telemetry,
    )

    return {
        "status": "blocked_limiter_readiness_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_mark_limiter_zero_or_physical_bounds_accepted",
        "required_channels": list(REQUIRED_LIMITER_READINESS_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "limiter_channel_status": _limiter_channel_statuses(
            accepted=accepted,
            missing=missing,
        ),
        "known_limiter_families": [dict(family) for family in KNOWN_LIMITER_FAMILIES],
        "limiter_family_status": _limiter_family_statuses(),
        "candidate_runtime_channels": _candidate_runtime_channels(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        ),
        "runtime_observations": _runtime_observations(
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        ),
        "runtime_limiter_zero_probe": limiter_zero_probe,
        "source_references": list(LIMITER_READINESS_SOURCE_REFS),
        "acceptance_gate": (
            "candidate_runtime_telemetry_cannot_support_limiter_zero_acceptance_"
            "until_active_path_inventory_full_horizon_zero_blocker_run_fallback_"
            "rejection_tests_hashes_and_review_pass"
        ),
        "validation_rule": (
            "Accepted first-principles readiness requires a complete active-path "
            "limiter inventory and a full-horizon run with zero acceptance-blocking "
            "activations; missing ledger evidence blocks acceptance."
        ),
        "negative_test_policy": {
            "synthetic_acceptance_blocker_required": True,
            "app_only_runner_rejection_required": True,
            "fallback_rejection_required": True,
            "hidden_limiter_regression_required": True,
        },
        "can_support_limiter_zero_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _limiter_channel_statuses(
    *,
    accepted: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_LIMITER_READINESS_CHANNELS:
        if channel in accepted:
            statuses[channel] = "accepted_limiter_readiness_channel"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _limiter_family_statuses() -> dict[str, dict[str, Any]]:
    return {
        str(family["family"]): {
            "examples": list(family["examples"]),
            "acceptance_rule": str(family["acceptance_rule"]),
            "status": "requires_inventory_and_review",
        }
        for family in KNOWN_LIMITER_FAMILIES
    }


def _runtime_observations(
    *,
    conservation: Mapping[str, Any] | None,
    simulation_telemetry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observations: dict[str, Any] = {
        "candidate_only": True,
        "full_horizon_run": False,
        "zero_acceptance_blocker_claim": False,
    }
    if conservation:
        observations["conservation_passed"] = conservation.get("passed")
        observations["final_max_abs_div_B_T_per_m"] = conservation.get(
            "final_max_abs_div_B_T_per_m"
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
        if conservation.get("passed") is True:
            channels.add("candidate_finite_conservation_snapshot")
        if conservation.get("final_max_abs_div_B_T_per_m") is not None:
            channels.add("candidate_divergence_b_reported")
    if simulation_telemetry:
        if str(simulation_telemetry.get("status", "")).startswith(
            "candidate_engineering_"
        ):
            channels.add("candidate_package_native_runtime")
        if simulation_telemetry.get("circuit") is not None:
            channels.add("candidate_circuit_boundary_runtime")
        last_step = simulation_telemetry.get("last_step")
        if isinstance(last_step, Mapping):
            if last_step.get("source_ordered_loop") is not None:
                channels.add("candidate_source_ordered_loop_runtime")
            if last_step.get("electron_energy") is not None:
                channels.add("candidate_electron_energy_runtime")
            if last_step.get("kinetic_yield") is not None:
                channels.add("candidate_kinetic_yield_runtime")
    return sorted(channels)
