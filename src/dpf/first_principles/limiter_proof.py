"""Experimental limiter-zero proof packets for first-principles 3-D runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPERIMENTAL_LIMITER_PROOF_STATUS = "experimental_limiter_zero_probe_not_validation"

LIMITER_PROOF_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "186-188,210-239",
        "role": "limiter_zero_and_numerical_fidelity_completion_gates",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.md",
        "lines": "45-76",
        "role": "limiter_inventory_and_zero_blocker_evidence_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "410-424,582-605,1046-1067",
        "role": "marder_ohmic_cfl_method_limiter_context",
    },
)

ACCEPTANCE_BLOCKING_LIMITER_COUNTS = (
    "conductivity_ohmic_cfl_limited_steps",
    "conductivity_density_blend_applied_steps",
    "marder_dominant_correction_steps",
    "electron_temperature_floor_contact_steps",
    "blocked_heat_flux_steps",
)

METHOD_REVIEW_LIMITER_COUNTS = (
    "marder_correction_steps",
    "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps",
)


def build_experimental_limiter_zero_probe_packet(
    *,
    declared_scope: str,
    device_name: str,
    simulation_telemetry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify full-horizon limiter telemetry without promoting acceptance."""

    simulation = _mapping(simulation_telemetry)
    summary = _mapping(simulation.get("limiter_activation_summary"))
    counts = _int_mapping(summary.get("activation_counts"))
    observed_steps = _optional_int(summary.get("steps_observed"))
    completed_steps = _optional_int(simulation.get("n_steps_completed"))
    final_time_s = _optional_float(simulation.get("final_time_s"))
    target_time_s = _optional_float(simulation.get("target_time_s"))
    finite_state = _finite_state_all(simulation)
    blocking_counts = {
        name: int(counts.get(name, 0))
        for name in ACCEPTANCE_BLOCKING_LIMITER_COUNTS
    }
    method_review_counts = {
        name: int(counts.get(name, 0)) for name in METHOD_REVIEW_LIMITER_COUNTS
    }
    max_observed = dict(_mapping(summary.get("max_observed")))
    method_limiter_decisions = _method_limiter_decisions(
        method_review_counts=method_review_counts,
        blocking_counts=blocking_counts,
        max_observed=max_observed,
    )
    total_blocking = sum(blocking_counts.values())
    inventory_complete = (
        observed_steps is not None
        and completed_steps is not None
        and observed_steps == completed_steps
    )
    target_satisfied = (
        target_time_s is None
        or (final_time_s is not None and final_time_s >= target_time_s)
    )
    zero_blockers_observed = (
        inventory_complete
        and total_blocking == 0
        and finite_state is True
        and target_satisfied
    )
    review_required = _review_required(
        blocking_counts=blocking_counts,
        method_review_counts=method_review_counts,
        method_limiter_decisions=method_limiter_decisions,
        inventory_complete=inventory_complete,
        finite_state=finite_state,
        target_satisfied=target_satisfied,
    )

    return {
        "status": EXPERIMENTAL_LIMITER_PROOF_STATUS,
        "declared_scope": declared_scope,
        "device_name": device_name,
        "run_intent": "experimental_full_horizon_limiter_zero_troubleshooting",
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "limiter_packet_is_runtime_observation_only": True,
            "validation_promotion_allowed": False,
        },
        "runtime_horizon": {
            "n_steps_completed": completed_steps,
            "steps_observed": observed_steps,
            "inventory_complete_for_completed_steps": inventory_complete,
            "final_time_s": final_time_s,
            "target_time_s": target_time_s,
            "target_time_satisfied": target_satisfied,
            "finite_state_all": finite_state,
        },
        "activation_counts": counts,
        "acceptance_blocking_counts": blocking_counts,
        "method_review_counts": method_review_counts,
        "method_limiter_decisions": method_limiter_decisions,
        "total_acceptance_blocking_activations": total_blocking,
        "zero_acceptance_blockers_observed": zero_blockers_observed,
        "max_observed": max_observed,
        "review_required": review_required,
        "source_references": list(LIMITER_PROOF_SOURCE_REFS),
        "acceptance_state": {
            "can_support_limiter_zero_acceptance": False,
            "can_support_first_principles_acceptance": False,
            "validated": False,
            "review_decision": "experimental_limiter_runtime_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def _review_required(
    *,
    blocking_counts: Mapping[str, int],
    method_review_counts: Mapping[str, int],
    method_limiter_decisions: Mapping[str, Mapping[str, Any]],
    inventory_complete: bool,
    finite_state: bool | None,
    target_satisfied: bool,
) -> list[str]:
    required: list[str] = []
    if not inventory_complete:
        required.append("complete_full_horizon_limiter_inventory")
    if finite_state is not True:
        required.append("finite_state_all_steps")
    if not target_satisfied:
        required.append("target_time_horizon_satisfied")
    for name, count in blocking_counts.items():
        if count > 0:
            required.append(f"resolve_acceptance_blocking_{name}")
    for name, count in method_review_counts.items():
        if name == "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps":
            if count > 0:
                required.append("review_unapplied_raw_ohmic_cfl_exceedance")
            continue
        marder_nondominant = (
            name == "marder_correction_steps"
            and _mapping(method_limiter_decisions.get("marder_correction")).get(
                "nondominant_observed"
            )
            is True
        )
        if count > 0 and not marder_nondominant:
            required.append(f"review_method_limiter_nondominance_{name}")
    required.extend(
        [
            "attach_source_backed_physical_bounds_or_method_proofs",
            "run_backend_precision_parity_matrix",
            "independent_engineering_review_decision",
        ]
    )
    return required


def _method_limiter_decisions(
    *,
    method_review_counts: Mapping[str, int],
    blocking_counts: Mapping[str, int],
    max_observed: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    marder_steps = int(method_review_counts.get("marder_correction_steps", 0))
    dominant_steps = int(blocking_counts.get("marder_dominant_correction_steps", 0))
    relative = _optional_float(max_observed.get("marder_relative_correction_linf"))
    threshold = _optional_float(max_observed.get("marder_nondominance_threshold"))
    nondominant = (
        marder_steps > 0
        and dominant_steps == 0
        and relative is not None
        and threshold is not None
        and relative <= threshold
    )
    return {
        "marder_correction": {
            "method_limiter": True,
            "steps_observed": marder_steps,
            "dominant_correction_steps": dominant_steps,
            "max_relative_correction_linf": relative,
            "nondominance_threshold": threshold,
            "nondominant_observed": nondominant,
            "status": (
                "candidate_method_limiter_nondominant_observed"
                if nondominant
                else "candidate_method_limiter_requires_review"
            ),
            "can_support_limiter_zero_acceptance": False,
        },
        "conductivity_ohmic_cfl_raw_exceedance": {
            "method_limiter": True,
            "steps_observed": int(
                method_review_counts.get(
                    "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps",
                    0,
                )
            ),
            "applied_limiter_steps": int(
                blocking_counts.get("conductivity_ohmic_cfl_limited_steps", 0)
            ),
            "max_raw_exceedance_fraction": _optional_float(
                max_observed.get("conductivity_cfl_limited_fraction")
            ),
            "status": (
                "candidate_raw_explicit_ohmic_cfl_exceedance_observed_not_applied"
                if int(
                    method_review_counts.get(
                        "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps",
                        0,
                    )
                )
                > 0
                and int(blocking_counts.get("conductivity_ohmic_cfl_limited_steps", 0))
                == 0
                else "candidate_explicit_ohmic_cfl_limit_requires_review"
            ),
            "can_support_limiter_zero_acceptance": False,
        },
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    converted: dict[str, int] = {}
    for key, item in value.items():
        try:
            converted[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return converted


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


def _finite_state_all(simulation: Mapping[str, Any]) -> bool | None:
    finite_state = simulation.get("finite_state")
    if isinstance(finite_state, Mapping):
        value = finite_state.get("all_finite")
        return bool(value) if value is not None else None
    return None
