"""Experimental split-continuation probe for first-principles 3-D runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from dpf.fields import HybridPIC3DSimulationResult
from dpf.first_principles.runner import (
    FirstPrinciples3DDeck,
    build_first_principles_3d_session,
    run_first_principles_3d_deck,
)

EXPERIMENTAL_SPLIT_CONTINUATION_STATUS = (
    "experimental_split_continuation_probe_not_restart_acceptance"
)


def build_experimental_split_continuation_packet(
    *,
    deck: Mapping[str, Any] | object | None,
    split_after_steps: int,
) -> dict[str, Any]:
    """Compare uninterrupted N-step run against A+B live continuation."""

    fixed_deck = _fixed_step_deck(deck)
    if split_after_steps <= 0 or split_after_steps >= fixed_deck.n_steps:
        raise ValueError("split_after_steps must satisfy 0 < split < n_steps")

    uninterrupted = run_first_principles_3d_deck(fixed_deck)
    session = build_first_principles_3d_session(fixed_deck)
    first_segment = session.run_segment(split_after_steps)
    second_segment = session.run_segment(fixed_deck.n_steps - split_after_steps)
    full_summary = _simulation_summary(
        uninterrupted.result,
        total_steps=uninterrupted.result.telemetry.n_steps_completed,
    )
    split_summary = _simulation_summary(
        second_segment,
        total_steps=session.completed_steps,
    )
    comparisons = _observable_comparisons(full_summary, split_summary)
    fingerprint_match = (
        full_summary["state_fingerprint_sha256"]
        == split_summary["state_fingerprint_sha256"]
    )
    observables_match = all(
        item["absolute_delta"] in (0.0, None) for item in comparisons.values()
    )
    return {
        "status": EXPERIMENTAL_SPLIT_CONTINUATION_STATUS,
        "run_intent": "experimental_live_split_continuation_troubleshooting",
        "deck_name": fixed_deck.validation_scope,
        "device_name": fixed_deck.device_name,
        "total_steps": fixed_deck.n_steps,
        "split_after_steps": int(split_after_steps),
        "first_segment_steps_completed": (
            first_segment.telemetry.n_steps_completed
        ),
        "second_segment_steps_completed": (
            second_segment.telemetry.n_steps_completed
        ),
        "split_total_steps_completed": session.completed_steps,
        "state_fingerprints_match": fingerprint_match,
        "tracked_observables_match_exactly": observables_match,
        "uninterrupted": full_summary,
        "split_continuation": split_summary,
        "observable_comparisons": comparisons,
        "continuation_state": {
            "first_segment": first_segment.telemetry.continuation_state,
            "second_segment": second_segment.telemetry.continuation_state,
            "lagged_field_work_preserved_into_second_segment": (
                first_segment.telemetry.continuation_state is not None
                and second_segment.telemetry.continuation_state is not None
                and bool(
                    first_segment.telemetry.continuation_state.get(
                        "has_lagged_field_work"
                    )
                )
                and bool(
                    second_segment.telemetry.continuation_state.get(
                        "has_lagged_field_work"
                    )
                )
            ),
        },
        "checkpoint_restart": {
            "available": False,
            "status": "live_same_process_continuation_only",
            "required": [
                "load_npz_checkpoint_into_new_first_principles_runner",
                "restore_lagged_field_work_and_predictor_corrector_state",
                "restore_kinetic_yield_history_state",
                "compare_checkpoint_restart_against_uninterrupted_run",
            ],
        },
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "split_continuation_outputs_are_troubleshooting_only": True,
            "validation_promotion_allowed": False,
        },
        "source_references": [
            {
                "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
                "lines": "412-414",
                "role": "restart_reproducibility_acceptance_fields",
            },
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_"
                    "2026_05_15.md"
                ),
                "lines": "60-85",
                "role": "restart_reproducibility_packet_requirement",
            },
            {
                "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
                "lines": "87-88",
                "role": "deterministic_checkpoint_restart_requirement",
            },
        ],
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "split_continuation_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def _fixed_step_deck(deck: Mapping[str, Any] | object | None) -> FirstPrinciples3DDeck:
    resolved = FirstPrinciples3DDeck.from_deck(deck)
    values = asdict(resolved)
    values["target_time_s"] = None
    return FirstPrinciples3DDeck.from_deck(values)


def _simulation_summary(
    result: HybridPIC3DSimulationResult,
    *,
    total_steps: int,
) -> dict[str, Any]:
    telemetry = result.telemetry.to_dict()
    circuit = telemetry.get("circuit") if isinstance(telemetry, Mapping) else None
    if not isinstance(circuit, Mapping):
        circuit = {}
    last_step = telemetry.get("last_step") if isinstance(telemetry, Mapping) else None
    kinetic_yield = {}
    if isinstance(last_step, Mapping) and isinstance(
        last_step.get("kinetic_yield"),
        Mapping,
    ):
        kinetic_yield = dict(last_step["kinetic_yield"])
    state_fingerprint = telemetry.get("state_fingerprint")
    if not isinstance(state_fingerprint, Mapping):
        state_fingerprint = {}
    continuation = telemetry.get("continuation_state")
    if not isinstance(continuation, Mapping):
        continuation = {}
    return {
        "status": telemetry.get("status"),
        "n_steps_completed": int(total_steps),
        "final_time_s": _optional_float(continuation.get("total_time_s"))
        or _optional_float(telemetry.get("final_time_s")),
        "state_fingerprint_sha256": state_fingerprint.get("sha256"),
        "final_particle_count": telemetry.get("n_particles_final"),
        "final_field_energy_J": telemetry.get("final_field_energy_J"),
        "final_circuit_current_A": circuit.get("final_current_A"),
        "candidate_cumulative_neutrons": kinetic_yield.get("cumulative_neutrons"),
        "finite_state_all": (
            telemetry.get("finite_state", {}).get("all_finite")
            if isinstance(telemetry.get("finite_state"), Mapping)
            else None
        ),
        "continuation_state": dict(continuation),
    }


def _observable_comparisons(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, dict[str, float | None]]:
    names = (
        "final_time_s",
        "final_particle_count",
        "final_field_energy_J",
        "final_circuit_current_A",
        "candidate_cumulative_neutrons",
    )
    comparisons: dict[str, dict[str, float | None]] = {}
    for name in names:
        left_value = _optional_float(left.get(name))
        right_value = _optional_float(right.get(name))
        absolute = (
            None
            if left_value is None or right_value is None
            else right_value - left_value
        )
        comparisons[name] = {
            "left": left_value,
            "right": right_value,
            "absolute_delta": absolute,
        }
    return comparisons


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
