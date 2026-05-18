"""Experimental checkpoint-loaded restart probe for first-principles 3-D runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dpf.fields import HybridPIC3DSimulationResult
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)
from dpf.first_principles.runner import (
    FirstPrinciples3DDeck,
    build_first_principles_3d_session,
    run_first_principles_3d_deck,
)
from dpf.first_principles.state_checkpoint import (
    load_checkpoint_into_first_principles_3d_session,
    write_simulation_state_checkpoint_roundtrip,
)

EXPERIMENTAL_CHECKPOINT_RESTART_STATUS = (
    "experimental_checkpoint_restart_probe_not_validation"
)


def build_experimental_checkpoint_restart_packet(
    *,
    deck: Mapping[str, Any] | object | None,
    split_after_steps: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Compare uninterrupted N-step run against checkpoint-loaded continuation."""

    fixed_deck = _fixed_step_deck(deck)
    if split_after_steps <= 0 or split_after_steps >= fixed_deck.n_steps:
        raise ValueError("split_after_steps must satisfy 0 < split < n_steps")

    uninterrupted = run_first_principles_3d_deck(fixed_deck)
    writer_session = build_first_principles_3d_session(fixed_deck)
    first_segment = writer_session.run_segment(split_after_steps)
    checkpoint_packet = write_simulation_state_checkpoint_roundtrip(
        simulation=first_segment,
        checkpoint_path=checkpoint_path,
        deck=fixed_deck,
    )
    loaded_session = load_checkpoint_into_first_principles_3d_session(
        checkpoint_path=checkpoint_path,
        deck=fixed_deck,
    )
    second_segment = loaded_session.run_segment(
        fixed_deck.n_steps - split_after_steps
    )
    full_summary = _simulation_summary(
        uninterrupted.result,
        declared_scope=fixed_deck.validation_scope,
        device_name=fixed_deck.device_name,
        total_steps=uninterrupted.result.telemetry.n_steps_completed,
    )
    restart_summary = _simulation_summary(
        second_segment,
        declared_scope=fixed_deck.validation_scope,
        device_name=fixed_deck.device_name,
        total_steps=loaded_session.completed_steps,
    )
    comparisons = _observable_comparisons(full_summary, restart_summary)
    fingerprint_match = (
        full_summary["state_fingerprint_sha256"]
        == restart_summary["state_fingerprint_sha256"]
    )
    observables_match = all(
        item["absolute_delta"] in (0.0, None) for item in comparisons.values()
    )
    return {
        "status": EXPERIMENTAL_CHECKPOINT_RESTART_STATUS,
        "run_intent": "experimental_checkpoint_loaded_restart_troubleshooting",
        "deck_name": fixed_deck.validation_scope,
        "device_name": fixed_deck.device_name,
        "total_steps": fixed_deck.n_steps,
        "split_after_steps": int(split_after_steps),
        "checkpoint_path": str(checkpoint_path),
        "first_segment_steps_completed": first_segment.telemetry.n_steps_completed,
        "restart_segment_steps_completed": second_segment.telemetry.n_steps_completed,
        "restart_total_steps_completed": loaded_session.completed_steps,
        "state_fingerprints_match": fingerprint_match,
        "tracked_observables_match_exactly": observables_match,
        "checkpoint_roundtrip": checkpoint_packet,
        "uninterrupted": full_summary,
        "checkpoint_restart": restart_summary,
        "observable_comparisons": comparisons,
        "restart_state": {
            "loaded_completed_steps": loaded_session.completed_steps,
            "lagged_field_work_loaded": loaded_session.lagged_field_work is not None,
            "previous_total_current_loaded": (
                loaded_session.simulator.loop.field_stepper.previous_total_current_A_m2
                is not None
            ),
            "kinetic_yield_state_loaded": (
                loaded_session.simulator.loop.kinetic_yield_history is not None
                and loaded_session.simulator.loop.kinetic_yield_history.time_s > 0.0
            ),
            "second_segment_continuation_state": (
                second_segment.telemetry.continuation_state
            ),
        },
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "checkpoint_restart_outputs_are_troubleshooting_only": True,
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
                "role": "checkpoint_restart_packet_requirement",
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
            "review_decision": "checkpoint_restart_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def build_experimental_checkpoint_restart_family_packet(
    *,
    deck: Mapping[str, Any] | object | None,
    split_after_steps: tuple[int, ...],
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Run checkpoint-loaded restart probes across multiple split offsets."""

    fixed_deck = _fixed_step_deck(deck)
    output_dir = Path(checkpoint_dir)
    cases: list[dict[str, Any]] = []
    for split in split_after_steps:
        if split <= 0 or split >= fixed_deck.n_steps:
            raise ValueError("each split offset must satisfy 0 < split < n_steps")
        packet = build_experimental_checkpoint_restart_packet(
            deck=fixed_deck,
            split_after_steps=split,
            checkpoint_path=output_dir
            / f"checkpoint_restart_split_{int(split):04d}.npz",
        )
        cases.append(packet)
    matching = [
        case
        for case in cases
        if case["state_fingerprints_match"] is True
        and case["tracked_observables_match_exactly"] is True
    ]
    return {
        "status": "experimental_checkpoint_restart_family_probe_not_validation",
        "run_intent": "experimental_multi_offset_checkpoint_restart_family",
        "deck_name": fixed_deck.validation_scope,
        "device_name": fixed_deck.device_name,
        "total_steps": fixed_deck.n_steps,
        "split_after_steps": [int(item) for item in split_after_steps],
        "case_count": len(cases),
        "matching_case_count": len(matching),
        "all_cases_match": len(cases) > 0 and len(matching) == len(cases),
        "checkpoint_dir": str(output_dir),
        "cases": cases,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "restart_family_outputs_are_troubleshooting_only": True,
            "validation_promotion_allowed": False,
        },
        "remaining_for_acceptance": [
            "whole_shot_horizon_restart_family",
            "accepted_nonbitwise_tolerances_if_backend_changes",
            "limiter_zero_interpretation_for_every_restart_case",
            "independent_engineering_review",
        ],
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "checkpoint_restart_family_probe_only",
        },
        "can_support_first_principles_acceptance": False,
    }


EXPERIMENTAL_SEGMENTED_RUN_STATUS = (
    "experimental_segmented_checkpointed_run_not_validation"
)


def build_experimental_segmented_run_packet(
    *,
    deck: Mapping[str, Any] | object | None,
    segment_steps: int,
    checkpoint_dir: str | Path,
    verify_against_uninterrupted: bool = True,
) -> dict[str, Any]:
    """Run a fixed-step horizon as checkpointed segments of ``segment_steps``.

    The horizon ``deck.n_steps`` is advanced in segments.  At every segment
    boundary the live session state is written to a metadata-tagged checkpoint
    and reloaded through ``load_checkpoint_into_first_principles_3d_session``
    -- whose loader validation gate fails closed on any grid/circuit/closure/
    species mismatch.  Because each segment's checkpoint round-trip restores
    fields, particles, previous current, circuit state, electron/ion energy,
    ionization state, kinetic-yield history, and lagged field work, the
    segmented horizon is equivalent to one uninterrupted run at the same step
    sequence.

    Cumulative ledgers (volume J.E work, terminal active-port work, limiter
    activations) are accumulated across segments here, because per-``run``
    telemetry resets its cumulative counters at each segment start.  History
    capping caps retained samples only; cumulative counters cover the full
    horizon.
    """

    fixed_deck = _fixed_step_deck(deck)
    total_steps = int(fixed_deck.n_steps)
    if int(segment_steps) != segment_steps or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    segment_steps = int(segment_steps)
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_first_principles_3d_session(fixed_deck)
    cumulative = {
        "cumulative_j_dot_e_work_J": 0.0,
        "cumulative_j_dot_e_step_count": 0,
        "cumulative_active_port_work_J": 0.0,
        "cumulative_active_port_step_count": 0,
    }
    limiter_steps_observed = 0
    segments: list[dict[str, Any]] = []
    last_segment: HybridPIC3DSimulationResult | None = None
    completed = 0
    segment_index = 0
    while completed < total_steps:
        this_segment_steps = min(segment_steps, total_steps - completed)
        result = session.run_segment(this_segment_steps)
        last_segment = result
        telemetry = result.telemetry
        cumulative["cumulative_j_dot_e_work_J"] += float(
            telemetry.cumulative_j_dot_e_work_J or 0.0
        )
        cumulative["cumulative_j_dot_e_step_count"] += int(
            telemetry.cumulative_j_dot_e_step_count
        )
        cumulative["cumulative_active_port_work_J"] += float(
            telemetry.cumulative_active_port_work_J or 0.0
        )
        cumulative["cumulative_active_port_step_count"] += int(
            telemetry.cumulative_active_port_step_count
        )
        if isinstance(telemetry.limiter_activation_summary, Mapping):
            limiter_steps_observed += int(
                telemetry.limiter_activation_summary.get("steps_observed", 0)
            )
        completed += int(telemetry.n_steps_completed)

        # Boundary checkpoint: write live state and reload through the
        # fail-closed loader.  This both exercises the validated loader on the
        # real horizon and proves the checkpoint carries every state channel.
        checkpoint_path = output_dir / f"segment_{segment_index:04d}.npz"
        roundtrip = write_simulation_state_checkpoint_roundtrip(
            simulation=result,
            checkpoint_path=checkpoint_path,
            deck=fixed_deck,
        )
        if completed < total_steps:
            session = load_checkpoint_into_first_principles_3d_session(
                checkpoint_path=checkpoint_path,
                deck=fixed_deck,
            )
        segments.append({
            "segment_index": segment_index,
            "segment_steps_requested": this_segment_steps,
            "segment_steps_completed": int(telemetry.n_steps_completed),
            "total_steps_completed_after_segment": completed,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_write_read_hashes_match": roundtrip[
                "write_read_hashes_match"
            ],
            "continuation_state": telemetry.continuation_state,
            "segment_cumulative_j_dot_e_work_J": (
                telemetry.cumulative_j_dot_e_work_J
            ),
            "segment_cumulative_j_dot_e_step_count": (
                telemetry.cumulative_j_dot_e_step_count
            ),
        })
        segment_index += 1

    if last_segment is None:  # pragma: no cover - total_steps > 0 guaranteed
        raise RuntimeError("segmented run produced no segments")

    segmented_summary = _simulation_summary(
        last_segment,
        declared_scope=fixed_deck.validation_scope,
        device_name=fixed_deck.device_name,
        total_steps=completed,
    )

    equivalence: dict[str, Any] = {
        "verified_against_uninterrupted": bool(verify_against_uninterrupted),
    }
    if verify_against_uninterrupted:
        uninterrupted = run_first_principles_3d_deck(fixed_deck)
        uninterrupted_summary = _simulation_summary(
            uninterrupted.result,
            declared_scope=fixed_deck.validation_scope,
            device_name=fixed_deck.device_name,
            total_steps=uninterrupted.result.telemetry.n_steps_completed,
        )
        comparisons = _observable_comparisons(
            uninterrupted_summary,
            segmented_summary,
        )
        equivalence.update({
            "uninterrupted": uninterrupted_summary,
            "observable_comparisons": comparisons,
            "state_fingerprints_match": (
                uninterrupted_summary["state_fingerprint_sha256"]
                == segmented_summary["state_fingerprint_sha256"]
            ),
            "tracked_observables_match_exactly": all(
                item["absolute_delta"] in (0.0, None)
                for item in comparisons.values()
            ),
        })

    return {
        "status": EXPERIMENTAL_SEGMENTED_RUN_STATUS,
        "run_intent": "experimental_segmented_checkpointed_long_run",
        "deck_name": fixed_deck.validation_scope,
        "device_name": fixed_deck.device_name,
        "total_steps": total_steps,
        "segment_steps": segment_steps,
        "segment_count": len(segments),
        "total_steps_completed": completed,
        "checkpoint_dir": str(output_dir),
        "all_segment_checkpoint_roundtrips_match": all(
            seg["checkpoint_write_read_hashes_match"] is True for seg in segments
        ),
        "cumulative_ledgers": {
            **cumulative,
            "limiter_steps_observed": limiter_steps_observed,
            "ledger_status": (
                "candidate_cumulative_segmented_ledger_not_validation"
            ),
            "covers_full_horizon": limiter_steps_observed == total_steps,
        },
        "segments": segments,
        "segmented_run": segmented_summary,
        "equivalence": equivalence,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "segmented_run_outputs_are_troubleshooting_only": True,
            "validation_promotion_allowed": False,
        },
        "source_references": [
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_EXTERNAL_TEAM_AUDIT_AND_NEXT_"
                    "INSTRUCTIONS_2026_05_18.md"
                ),
                "lines": "508-540",
                "role": "wp_n4_segmented_run_and_loader_validation_requirement",
            },
            {
                "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
                "lines": "412-414",
                "role": "restart_reproducibility_acceptance_fields",
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
            "review_decision": "segmented_checkpointed_run_probe_only",
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
    declared_scope: str,
    device_name: str,
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
    limiter_zero_probe = build_experimental_limiter_zero_probe_packet(
        declared_scope=declared_scope,
        device_name=device_name,
        simulation_telemetry=telemetry,
    )
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
        "limiter_zero_probe": {
            "status": limiter_zero_probe.get("status"),
            "zero_acceptance_blockers_observed": limiter_zero_probe.get(
                "zero_acceptance_blockers_observed"
            ),
            "total_acceptance_blocking_activations": limiter_zero_probe.get(
                "total_acceptance_blocking_activations"
            ),
            "review_required": limiter_zero_probe.get("review_required"),
        },
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
