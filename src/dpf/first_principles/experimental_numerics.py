"""Experimental runtime numerical audit packets for first-principles shots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from dpf.first_principles.experimental_shot import stable_vacuum_cfl_dt_s
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)

EXPERIMENTAL_NUMERICAL_RUNTIME_AUDIT_STATUS = (
    "experimental_numerical_runtime_audit_not_validation"
)
EXPERIMENTAL_NUMERICAL_FAMILY_STATUS = (
    "experimental_numerical_family_probe_not_validation"
)
EXPERIMENTAL_REPRODUCIBILITY_STATUS = (
    "experimental_reproducibility_probe_not_validation"
)

EXPERIMENTAL_NUMERICAL_RUNTIME_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_3D_HYBRID_PIC_REVIEW_2026_05_14.md",
        "lines": "39,70-80",
        "role": "marder_conductivity_limiter_nondominance_requirements",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md",
        "lines": "50-53",
        "role": "hybrid_pic_loop_cfl_limiter_and_sensitivity_scope",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md",
        "lines": "48-55,63-70",
        "role": "closure_and_transport_sources_plus_remaining_blocks",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "186-188",
        "role": "limiter_zero_and_numerical_fidelity_completion_gates",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "410-424,468-606,1030-1068",
        "role": "marder_ohmic_cfl_refinement_and_sensitivity_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "particle-simulation-of-plasmas-review-and-advances-6d7355ba.md"
        ),
        "lines": "456-530,671-705,744-755",
        "role": "yee_courant_and_charge_conservation_method_context",
    },
)

MISSING_NUMERICAL_AUDIT_EVIDENCE = (
    "mesh_family_runs_same_physics_same_outputs",
    "timestep_family_runs_same_physics_same_outputs",
    "observable_norms_and_tolerances",
    "observed_order_or_monotonic_convergence",
    "restart_reproducibility_hashes",
    "backend_precision_parity_matrix",
    "full_horizon_limiter_inventory_and_zero_acceptance_blocker_run",
    "marder_and_ohmic_cfl_nondominance_sensitivity",
    "accepted_energy_power_residual_budget",
    "independent_engineering_review_decision",
)

NUMERICAL_FAMILY_OBSERVABLES = (
    "final_time_s",
    "final_field_energy_J",
    "final_particle_count",
    "final_circuit_current_A",
    "relative_tracked_total_energy_change",
    "final_max_abs_div_B_T_per_m",
    "candidate_cumulative_neutrons",
)


def build_experimental_numerical_runtime_audit_packet(
    *,
    declared_scope: str,
    device_name: str,
    simulation_telemetry: Mapping[str, Any] | None,
    conservation: Mapping[str, Any] | None,
    duration_plan: Mapping[str, Any] | None,
    limiter_readiness: Mapping[str, Any] | None = None,
    numerical_fidelity: Mapping[str, Any] | None = None,
    grid_spacing_m: Sequence[float] | None = None,
    dt_s: float | None = None,
    vacuum_cfl: float = 0.95,
) -> dict[str, Any]:
    """Return a non-promoting runtime audit for experimental shot attempts."""

    simulation = _mapping(simulation_telemetry)
    conservation_packet = _mapping(conservation)
    duration = _mapping(duration_plan)
    limiter = _mapping(limiter_readiness)
    numerical = _mapping(numerical_fidelity)
    spacing = _spacing_tuple(grid_spacing_m) or _spacing_tuple(
        duration.get("grid_spacing_m")
    )
    resolved_dt = _first_float(dt_s, duration.get("dt_s"), simulation.get("dt_s"))
    stable_dt = _first_float(duration.get("stable_vacuum_dt_s"))
    if stable_dt is None and spacing is not None:
        stable_dt = stable_vacuum_cfl_dt_s(spacing, cfl=vacuum_cfl)
    final_time_s = _optional_float(simulation.get("final_time_s"))
    target_time_s = _optional_float(simulation.get("target_time_s"))
    history = _history_summary(simulation.get("history_summary"))
    last_step = _mapping(simulation.get("last_step"))
    limiter_zero_probe = build_experimental_limiter_zero_probe_packet(
        declared_scope=declared_scope,
        device_name=device_name,
        simulation_telemetry=simulation,
    )

    return {
        "status": EXPERIMENTAL_NUMERICAL_RUNTIME_AUDIT_STATUS,
        "declared_scope": declared_scope,
        "device_name": device_name,
        "run_intent": "source_truth_runtime_numerical_troubleshooting",
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "audit_scope": "runtime_observations_not_acceptance",
            "validation_promotion_allowed": False,
        },
        "runtime_horizon": _runtime_horizon(
            simulation=simulation,
            final_time_s=final_time_s,
            target_time_s=target_time_s,
        ),
        "courant_budget": _courant_budget(
            spacing=spacing,
            dt_s=resolved_dt,
            stable_dt_s=stable_dt,
            duration_plan=duration,
        ),
        "conservation_snapshot": _conservation_snapshot(conservation_packet),
        "divergence_snapshot": _divergence_snapshot(conservation_packet, history),
        "history_observations": history,
        "last_step_limiter_observations": _last_step_limiter_observations(last_step),
        "full_horizon_limiter_activation_summary": simulation.get(
            "limiter_activation_summary"
        ),
        "limiter_zero_probe": limiter_zero_probe,
        "limiter_gate_snapshot": {
            "status": limiter.get("status"),
            "can_support_limiter_zero_acceptance": limiter.get(
                "can_support_limiter_zero_acceptance"
            ),
            "missing_acceptance_channel_count": _list_len(
                limiter.get("missing_acceptance_channels")
            ),
        },
        "numerical_gate_snapshot": {
            "status": numerical.get("status"),
            "can_support_numerical_acceptance": numerical.get(
                "can_support_numerical_acceptance"
            ),
            "missing_acceptance_channel_count": _list_len(
                numerical.get("missing_acceptance_channels")
            ),
        },
        "restart_reproducibility": {
            "available": False,
            "status": "missing_restart_reproducibility_packet",
            "required": [
                "checkpoint_write_read_same_state",
                "continued_run_matches_uninterrupted_run",
                "artifact_hashes_for_state_and_diagnostics",
            ],
        },
        "mesh_timestep_convergence": {
            "available": False,
            "status": "missing_mesh_timestep_convergence_family",
            "required": [
                "coarse_medium_fine_grid_family",
                "dt_refinement_family",
                "same_observable_norms",
                "tolerances_and_acceptance_rule",
            ],
        },
        "source_audit_findings": {
            "supported_guidance": [
                "explicit full-EM/PIC runtime needs Yee/Courant accounting",
                "Marder/Gauss-law control and Ohmic CFL limiting are method limiters",
                "source sensitivity examples do not transfer automatically to PF-1000/GV",
            ],
            "same_scope_acceptance_evidence_found": False,
            "missing_for_engineering_signoff": list(
                MISSING_NUMERICAL_AUDIT_EVIDENCE
            ),
        },
        "troubleshooting_priority": _troubleshooting_priority(
            final_time_s=final_time_s,
            target_time_s=target_time_s,
            conservation=conservation_packet,
            duration_plan=duration,
            history=history,
        ),
        "source_references": list(EXPERIMENTAL_NUMERICAL_RUNTIME_SOURCE_REFS),
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "runtime_audit_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def build_experimental_numerical_family_packet(
    *,
    family_kind: str,
    case_payloads: Sequence[Mapping[str, Any]],
    declared_scope: str,
    device_name: str,
) -> dict[str, Any]:
    """Summarize a non-promoting mesh/timestep family run."""

    cases = [_case_summary(index=index, payload=payload) for index, payload in enumerate(case_payloads)]
    comparisons = _case_comparisons(cases)
    completed = [case for case in cases if case["duration_request_satisfied"] is True]
    finite = [case for case in cases if case["finite_state_all"] is True]
    missing = list(MISSING_NUMERICAL_AUDIT_EVIDENCE)
    return {
        "status": EXPERIMENTAL_NUMERICAL_FAMILY_STATUS,
        "family_kind": str(family_kind),
        "declared_scope": declared_scope,
        "device_name": device_name,
        "run_intent": "experimental_mesh_timestep_troubleshooting_family",
        "case_count": len(cases),
        "duration_satisfied_case_count": len(completed),
        "finite_case_count": len(finite),
        "cases": cases,
        "pairwise_comparisons": comparisons,
        "observable_names": list(NUMERICAL_FAMILY_OBSERVABLES),
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "family_outputs_are_troubleshooting_only": True,
            "validation_promotion_allowed": False,
        },
        "convergence_decision": {
            "status": "not_assessed_no_accepted_tolerances",
            "observed_deltas_available": bool(comparisons),
            "tolerance_claim": False,
            "can_support_numerical_acceptance": False,
        },
        "missing_for_engineering_signoff": missing,
        "next_required_actions": [
            "select_engineering_observables_and_units",
            "define_source_reviewed_norms_and_tolerances",
            "run_at_least_three_ordered_mesh_or_timestep_levels",
            "attach_restart_reproducibility_hashes",
            "attach_limiter_activation_counts_for_every_case",
        ],
        "source_references": list(EXPERIMENTAL_NUMERICAL_RUNTIME_SOURCE_REFS),
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "family_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def build_experimental_reproducibility_packet(
    *,
    run_payloads: Sequence[Mapping[str, Any]],
    declared_scope: str,
    device_name: str,
) -> dict[str, Any]:
    """Summarize deterministic reruns without promoting restart acceptance."""

    runs = [
        _reproducibility_run_summary(index=index, payload=payload)
        for index, payload in enumerate(run_payloads)
    ]
    comparisons = _reproducibility_comparisons(runs)
    hashes = [
        str(run["state_observable_hash_sha256"])
        for run in runs
        if run.get("state_observable_hash_sha256")
    ]
    all_hashes_identical = len(hashes) >= 2 and len(set(hashes)) == 1
    finite_runs = [run for run in runs if run["finite_state_all"] is True]
    completed_runs = [
        run for run in runs if run["duration_request_satisfied"] is True
    ]
    return {
        "status": EXPERIMENTAL_REPRODUCIBILITY_STATUS,
        "declared_scope": declared_scope,
        "device_name": device_name,
        "run_intent": "experimental_deterministic_rerun_troubleshooting",
        "run_count": len(runs),
        "finite_run_count": len(finite_runs),
        "duration_satisfied_run_count": len(completed_runs),
        "all_state_observable_hashes_identical": all_hashes_identical,
        "runs": runs,
        "pairwise_comparisons": comparisons,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "reproducibility_outputs_are_troubleshooting_only": True,
            "validation_promotion_allowed": False,
        },
        "deterministic_rerun": {
            "available": len(runs) >= 2,
            "hash_material": [
                "simulation_terminal_scalars",
                "history_summary",
                "last_step_telemetry",
                "conservation_telemetry",
                "terminal_state_fingerprint",
                "terminal_packet_statuses",
            ],
            "all_hashes_identical": all_hashes_identical,
            "decision": (
                "deterministic_rerun_matched_not_restart_acceptance"
                if all_hashes_identical
                else "deterministic_rerun_mismatch_or_insufficient_runs"
            ),
        },
        "checkpoint_restart": {
            "available": False,
            "status": "missing_first_principles_checkpoint_restart_packet",
            "required": [
                "write_complete_field_particle_closure_circuit_state",
                "read_checkpoint_into_package_native_first_principles_runner",
                "continue_from_checkpoint_at_multiple_offsets",
                "compare_against_uninterrupted_run_hashes",
            ],
        },
        "continued_run_equivalence": {
            "available": False,
            "status": "missing_split_run_continuation_packet",
            "required": [
                "split_same_deck_into_A_plus_B_steps",
                "preserve_lagged_field_work_and_circuit_sequence_state",
                "match_uninterrupted_terminal_fields_particles_and_history",
            ],
        },
        "missing_for_engineering_signoff": [
            "checkpoint_write_read_same_state",
            "continued_run_matches_uninterrupted_run",
            "multiple_restart_offsets",
            "manifest_hash_for_each_checkpoint",
            "accepted_tolerances_for_nonbitwise_backends",
            *[
                item
                for item in MISSING_NUMERICAL_AUDIT_EVIDENCE
                if item != "restart_reproducibility_hashes"
            ],
        ],
        "source_references": [
            *EXPERIMENTAL_NUMERICAL_RUNTIME_SOURCE_REFS,
            {
                "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
                "lines": "87-88",
                "role": "checkpoint_restart_deterministic_comparison_requirement",
            },
            {
                "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
                "lines": "140-144,412-414",
                "role": "restart_reproducibility_finish_line_and_acceptance_fields",
            },
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_"
                    "2026_05_15.md"
                ),
                "lines": "60-85",
                "role": "numerical_fidelity_packet_restart_hash_requirements",
            },
        ],
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "deterministic_rerun_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def _case_summary(index: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    simulation = _mapping(payload.get("simulation"))
    conservation = _mapping(payload.get("conservation_telemetry"))
    numerics = _mapping(payload.get("experimental_numerics"))
    courant = _mapping(numerics.get("courant_budget"))
    circuit = _mapping(simulation.get("circuit"))
    last_step = _mapping(simulation.get("last_step"))
    kinetic_yield = _mapping(last_step.get("kinetic_yield"))
    finite_state = _mapping(simulation.get("finite_state"))
    limiter_summary = _mapping(simulation.get("limiter_activation_summary"))
    limiter_zero_probe = _mapping(
        payload.get("limiter_zero_probe")
        or _mapping(payload.get("telemetry_packets")).get(
            "experimental_limiter_zero_probe"
        )
        or _mapping(numerics).get("limiter_zero_probe")
    )
    state_fingerprint = _mapping(simulation.get("state_fingerprint"))
    observables = {
        "final_time_s": _optional_float(simulation.get("final_time_s")),
        "final_field_energy_J": _optional_float(
            simulation.get("final_field_energy_J")
        ),
        "final_particle_count": _optional_float(payload.get("final_particle_count")),
        "final_circuit_current_A": _optional_float(circuit.get("final_current_A")),
        "relative_tracked_total_energy_change": _optional_float(
            conservation.get("relative_tracked_total_energy_change")
        ),
        "final_max_abs_div_B_T_per_m": _optional_float(
            conservation.get("final_max_abs_div_B_T_per_m")
        ),
        "candidate_cumulative_neutrons": _optional_float(
            kinetic_yield.get("cumulative_neutrons")
        ),
    }
    return {
        "case_index": int(index),
        "case_label": str(payload.get("case_label", f"case_{index}")),
        "case_family_axis": payload.get("case_family_axis"),
        "deck_name": _mapping(payload.get("deck")).get("name"),
        "grid_shape": payload.get("grid_shape"),
        "dt_s": _optional_float(payload.get("dt_s")),
        "n_steps": _optional_int(payload.get("n_steps")),
        "n_steps_completed": _optional_int(payload.get("n_steps_completed")),
        "target_time_s": _optional_float(payload.get("target_time_s")),
        "duration_request_satisfied": payload.get("duration_request_satisfied"),
        "termination_reason": payload.get("termination_reason"),
        "finite_state_all": finite_state.get("all_finite"),
        "retained_step_result_count": _optional_int(
            simulation.get("retained_step_result_count")
        ),
        "limiter_activation_summary": {
            "status": limiter_summary.get("status"),
            "steps_observed": _optional_int(limiter_summary.get("steps_observed")),
            "activation_counts": limiter_summary.get("activation_counts"),
            "max_observed": limiter_summary.get("max_observed"),
        },
        "limiter_zero_probe": {
            "status": limiter_zero_probe.get("status"),
            "zero_acceptance_blockers_observed": limiter_zero_probe.get(
                "zero_acceptance_blockers_observed"
            ),
            "total_acceptance_blocking_activations": _optional_int(
                limiter_zero_probe.get("total_acceptance_blocking_activations")
            ),
            "review_required": limiter_zero_probe.get("review_required"),
        },
        "state_fingerprint": {
            "status": state_fingerprint.get("status"),
            "sha256": state_fingerprint.get("sha256"),
            "included_state_count": _optional_int(
                state_fingerprint.get("included_state_count")
            ),
            "particle_count": _optional_int(state_fingerprint.get("particle_count")),
        },
        "courant": {
            "dt_within_vacuum_cfl": courant.get("dt_within_vacuum_cfl"),
            "dt_to_stable_vacuum_dt_ratio": _optional_float(
                courant.get("dt_to_stable_vacuum_dt_ratio")
            ),
        },
        "observables": observables,
        "can_support_first_principles_acceptance": False,
    }


def _reproducibility_run_summary(
    index: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _case_summary(index, payload)
    summary["case_label"] = str(payload.get("case_label", f"rerun_{index}"))
    material = _reproducibility_hash_material(payload)
    summary["state_observable_hash_sha256"] = _stable_payload_hash(material)
    summary["hash_material_keys"] = sorted(material)
    return summary


def _reproducibility_comparisons(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = _case_comparisons(runs)
    for comparison, left, right in zip(
        comparisons,
        runs,
        runs[1:],
        strict=False,
    ):
        left_hash = left.get("state_observable_hash_sha256")
        right_hash = right.get("state_observable_hash_sha256")
        comparison["state_observable_hashes_match"] = (
            bool(left_hash) and left_hash == right_hash
        )
        comparison["left_state_observable_hash_sha256"] = left_hash
        comparison["right_state_observable_hash_sha256"] = right_hash
        comparison["acceptance_decision"] = "not_checkpoint_restart_acceptance"
    return comparisons


def _reproducibility_hash_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    simulation = _mapping(payload.get("simulation"))
    conservation = _mapping(payload.get("conservation_telemetry"))
    telemetry_packets = _mapping(payload.get("telemetry_packets"))
    packet_statuses = {
        key: _mapping(value).get("status")
        for key, value in sorted(telemetry_packets.items())
        if isinstance(value, Mapping)
    }
    return {
        "deck_name": _mapping(payload.get("deck")).get("name"),
        "grid_shape": payload.get("grid_shape"),
        "dt_s": payload.get("dt_s"),
        "n_steps": payload.get("n_steps"),
        "n_steps_completed": payload.get("n_steps_completed"),
        "target_time_s": payload.get("target_time_s"),
        "duration_request_satisfied": payload.get("duration_request_satisfied"),
        "termination_reason": payload.get("termination_reason"),
        "simulation_terminal": {
            key: simulation.get(key)
            for key in (
                "status",
                "final_time_s",
                "n_particles_initial",
                "n_particles_final",
                "initial_field_energy_J",
                "final_field_energy_J",
                "finite_state",
                "circuit",
                "limiter_activation_summary",
                "state_fingerprint",
            )
        },
        "history_summary": simulation.get("history_summary"),
        "last_step": simulation.get("last_step"),
        "conservation_telemetry": conservation,
        "packet_statuses": packet_statuses,
        "final_particle_count": payload.get("final_particle_count"),
        "final_field_energy_J": payload.get("final_field_energy_J"),
    }


def _stable_payload_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload,
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _case_comparisons(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(cases, cases[1:], strict=False):
        left_observables = _mapping(left.get("observables"))
        right_observables = _mapping(right.get("observables"))
        deltas: dict[str, dict[str, float | None]] = {}
        for name in NUMERICAL_FAMILY_OBSERVABLES:
            left_value = _optional_float(left_observables.get(name))
            right_value = _optional_float(right_observables.get(name))
            absolute = (
                None
                if left_value is None or right_value is None
                else right_value - left_value
            )
            relative = (
                None
                if absolute is None or left_value in (None, 0.0)
                else absolute / abs(left_value)
            )
            deltas[name] = {
                "left": left_value,
                "right": right_value,
                "absolute_delta": absolute,
                "relative_delta": relative,
            }
        comparisons.append({
            "left_case_label": left.get("case_label"),
            "right_case_label": right.get("case_label"),
            "observable_deltas": deltas,
            "tolerance_claim": False,
            "acceptance_decision": "not_assessed_no_accepted_tolerances",
        })
    return comparisons


def _runtime_horizon(
    *,
    simulation: Mapping[str, Any],
    final_time_s: float | None,
    target_time_s: float | None,
) -> dict[str, Any]:
    coverage = (
        None
        if final_time_s is None or target_time_s in (None, 0.0)
        else final_time_s / target_time_s
    )
    return {
        "simulation_status": simulation.get("status"),
        "n_steps_requested": _optional_int(simulation.get("n_steps_requested")),
        "n_steps_completed": _optional_int(simulation.get("n_steps_completed")),
        "final_time_s": final_time_s,
        "target_time_s": target_time_s,
        "target_coverage_fraction": coverage,
        "duration_request_satisfied": simulation.get("duration_request_satisfied"),
        "termination_reason": simulation.get("termination_reason"),
        "retained_step_result_count": _optional_int(
            simulation.get("retained_step_result_count")
        ),
        "history_stride": _optional_int(simulation.get("history_stride")),
        "max_step_results": _optional_int(simulation.get("max_step_results")),
        "finite_state": simulation.get("finite_state"),
    }


def _courant_budget(
    *,
    spacing: tuple[float, float, float] | None,
    dt_s: float | None,
    stable_dt_s: float | None,
    duration_plan: Mapping[str, Any],
) -> dict[str, Any]:
    ratio = None if dt_s is None or stable_dt_s in (None, 0.0) else dt_s / stable_dt_s
    return {
        "status": "candidate_runtime_vacuum_cfl_audit",
        "grid_spacing_m": None if spacing is None else list(spacing),
        "dt_s": dt_s,
        "stable_vacuum_dt_s": stable_dt_s,
        "dt_to_stable_vacuum_dt_ratio": ratio,
        "dt_within_vacuum_cfl": (
            None if dt_s is None or stable_dt_s is None else dt_s <= stable_dt_s
        ),
        "steps_required_current_dt": duration_plan.get(
            "steps_required_current_dt"
        ),
        "steps_required_vacuum_cfl_dt": duration_plan.get(
            "steps_required_vacuum_cfl_dt"
        ),
        "claim_limit": (
            "vacuum CFL compliance is necessary runtime evidence only; it is not "
            "mesh convergence, limiter nondominance, or validation"
        ),
    }


def _conservation_snapshot(conservation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": conservation.get("status"),
        "finite_snapshot": conservation.get("passed"),
        "delta_tracked_total_energy_J": _optional_float(
            conservation.get("delta_tracked_total_energy_J")
        ),
        "relative_tracked_total_energy_change": _optional_float(
            conservation.get("relative_tracked_total_energy_change")
        ),
        "initial_tracked_total_energy_J": _nested_optional_float(
            conservation,
            "initial",
            "tracked_total_energy_J",
        ),
        "final_tracked_total_energy_J": _nested_optional_float(
            conservation,
            "final",
            "tracked_total_energy_J",
        ),
        "tolerance_claim": False,
    }


def _divergence_snapshot(
    conservation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "final_max_abs_div_B_T_per_m": _optional_float(
            conservation.get("final_max_abs_div_B_T_per_m")
        ),
        "history_max_abs_div_B_T_per_m": history.get(
            "max_abs_div_B_T_per_m_max"
        ),
        "tolerance_claim": False,
        "requires_gauss_or_divergence_nondominance": True,
    }


def _last_step_limiter_observations(last_step: Mapping[str, Any]) -> dict[str, Any]:
    field_step = _mapping(last_step.get("field_step"))
    conductivity = _mapping(field_step.get("conductivity"))
    marder = _mapping(field_step.get("marder"))
    electron_energy = _mapping(last_step.get("electron_energy"))
    heat_flux = _mapping(electron_energy.get("heat_flux"))
    concerns: list[str] = []
    if conductivity:
        concerns.append("conductivity_blend_and_ohmic_cfl_requires_nondominance")
    if marder:
        concerns.append("marder_correction_requires_nondominance")
    if str(heat_flux.get("status", "")).startswith("blocked"):
        concerns.append("heat_flux_subcycle_or_stability_block_reported")
    if electron_energy:
        concerns.append("electron_temperature_floor_requires_limiter_inventory")
    return {
        "conductivity_status": conductivity.get("status"),
        "conductivity_cfl_limited_fraction": _optional_float(
            conductivity.get("cfl_limited_fraction")
        ),
        "conductivity_sigma_cfl_S_m": _optional_float(
            conductivity.get("sigma_cfl_S_m")
        ),
        "marder_status": marder.get("status"),
        "marder_factor_m2": _optional_float(marder.get("marder_factor_m2")),
        "marder_residual_after": _optional_float(
            marder.get("residual_l2_after")
        ),
        "electron_energy_status": electron_energy.get("status"),
        "heat_flux_status": heat_flux.get("status"),
        "active_runtime_concerns": concerns,
        "limiter_zero_claim": False,
    }


def _history_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {
            "sample_count": 0,
            "time_monotonic_non_decreasing": None,
            "finite_numeric_samples": None,
        }
    samples = [_mapping(item) for item in value if isinstance(item, Mapping)]
    times = [_optional_float(item.get("time_s")) for item in samples]
    finite_values = [
        number
        for item in samples
        for number in (_numeric_values(item))
    ]
    return {
        "sample_count": len(samples),
        "first_time_s": _first_non_none(times),
        "last_time_s": _last_non_none(times),
        "time_monotonic_non_decreasing": _monotonic_non_decreasing(times),
        "finite_numeric_samples": all(isfinite(number) for number in finite_values),
        "particle_count_min": _series_min(samples, "n_particles"),
        "particle_count_max": _series_max(samples, "n_particles"),
        "field_energy_J_min": _series_min(samples, "field_energy_J"),
        "field_energy_J_max": _series_max(samples, "field_energy_J"),
        "max_abs_div_B_T_per_m_max": _series_max(
            samples,
            "max_abs_div_B_T_per_m",
        ),
        "electron_temperature_max_K": _series_max(
            samples,
            "electron_temperature_max_K",
        ),
        "cumulative_neutrons_max": _series_max(samples, "cumulative_neutrons"),
        "cumulative_neutrons_non_decreasing": _monotonic_non_decreasing(
            [_optional_float(item.get("cumulative_neutrons")) for item in samples]
        ),
    }


def _troubleshooting_priority(
    *,
    final_time_s: float | None,
    target_time_s: float | None,
    conservation: Mapping[str, Any],
    duration_plan: Mapping[str, Any],
    history: Mapping[str, Any],
) -> list[str]:
    priorities: list[str] = []
    if final_time_s is None or target_time_s is None or final_time_s < target_time_s:
        priorities.append("extend_runtime_to_requested_target_with_stable_dt")
    if conservation.get("passed") is not True:
        priorities.append("stop_and_fix_nonfinite_or_missing_conservation_snapshot")
    if duration_plan.get("dt_within_vacuum_cfl") is not True:
        priorities.append("reduce_dt_or_refine_grid_until_vacuum_cfl_passes")
    if history.get("sample_count", 0) < 2:
        priorities.append("retain_runtime_history_for_trend_and_restart_checks")
    priorities.extend(
        [
            "run_mesh_and_timestep_family_without_physics_changes",
            "add_restart_reproducibility_artifact",
            "instrument_limiter_activation_counts_before_acceptance",
        ]
    )
    return priorities


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _spacing_tuple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 3:
        return None
    spacing = tuple(float(item) for item in value)
    if any(item <= 0.0 for item in spacing):
        return None
    return spacing  # type: ignore[return-value]


def _first_float(*values: Any) -> float | None:
    for value in values:
        converted = _optional_float(value)
        if converted is not None:
            return converted
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_optional_float(
    mapping: Mapping[str, Any],
    first_key: str,
    second_key: str,
) -> float | None:
    nested = mapping.get(first_key)
    if not isinstance(nested, Mapping):
        return None
    return _optional_float(nested.get(second_key))


def _list_len(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _numeric_values(mapping: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for value in mapping.values():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _series_min(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [_optional_float(item.get(key)) for item in samples]
    finite_values = [value for value in values if value is not None]
    return None if not finite_values else min(finite_values)


def _series_max(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [_optional_float(item.get(key)) for item in samples]
    finite_values = [value for value in values if value is not None]
    return None if not finite_values else max(finite_values)


def _monotonic_non_decreasing(values: list[float | None]) -> bool | None:
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return None
    return all(
        next_value >= value
        for value, next_value in zip(present, present[1:], strict=False)
    )


def _first_non_none(values: list[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _last_non_none(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None
