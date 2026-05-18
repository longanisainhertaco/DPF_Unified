from __future__ import annotations

from dpf.first_principles.experimental_shot import stable_ohmic_cfl_dt_s
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)


def test_stable_ohmic_cfl_dt_uses_explicit_relaxation_limit() -> None:
    dt_s = stable_ohmic_cfl_dt_s(
        10.0,
        ohmic_cfl_safety=0.5,
        cfl=0.8,
    )

    assert dt_s == 0.8 * 0.5 * 8.8541878128e-12 / 10.0


def test_limiter_zero_probe_classifies_clean_runtime_without_promotion() -> None:
    packet = build_experimental_limiter_zero_probe_packet(
        declared_scope="unit_scope",
        device_name="unit_device",
        simulation_telemetry={
            "n_steps_completed": 2,
            "final_time_s": 2.0e-13,
            "target_time_s": 2.0e-13,
            "finite_state": {"all_finite": True},
            "limiter_activation_summary": {
                "steps_observed": 2,
                "activation_counts": {
                    "conductivity_ohmic_cfl_limited_steps": 0,
                    "conductivity_density_blend_applied_steps": 0,
                    "marder_correction_steps": 0,
                    "marder_dominant_correction_steps": 0,
                    "electron_temperature_floor_contact_steps": 0,
                    "blocked_heat_flux_steps": 0,
                },
                "max_observed": {},
            },
        },
    )

    assert packet["status"] == "experimental_limiter_zero_probe_not_validation"
    assert packet["runtime_horizon"]["inventory_complete_for_completed_steps"] is True
    assert packet["runtime_horizon"]["target_time_satisfied"] is True
    assert packet["zero_acceptance_blockers_observed"] is True
    assert packet["total_acceptance_blocking_activations"] == 0
    assert packet["acceptance_state"]["can_support_limiter_zero_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_limiter_zero_probe_blocks_acceptance_blocking_activation() -> None:
    packet = build_experimental_limiter_zero_probe_packet(
        declared_scope="unit_scope",
        device_name="unit_device",
        simulation_telemetry={
            "n_steps_completed": 3,
            "final_time_s": 3.0e-13,
            "target_time_s": 3.0e-13,
            "finite_state": {"all_finite": True},
            "limiter_activation_summary": {
                "steps_observed": 3,
                "activation_counts": {
                    "marder_dominant_correction_steps": 1,
                    "marder_correction_steps": 3,
                },
                "max_observed": {
                    "marder_relative_correction_linf": 2.0,
                },
            },
        },
    )

    assert packet["zero_acceptance_blockers_observed"] is False
    assert packet["total_acceptance_blocking_activations"] == 1
    assert (
        "resolve_acceptance_blocking_marder_dominant_correction_steps"
        in packet["review_required"]
    )
    assert (
        "review_method_limiter_nondominance_marder_correction_steps"
        in packet["review_required"]
    )


def test_limiter_zero_probe_records_marder_nondominance_observation() -> None:
    packet = build_experimental_limiter_zero_probe_packet(
        declared_scope="unit_scope",
        device_name="unit_device",
        simulation_telemetry={
            "n_steps_completed": 3,
            "final_time_s": 3.0e-13,
            "target_time_s": 3.0e-13,
            "finite_state": {"all_finite": True},
            "limiter_activation_summary": {
                "steps_observed": 3,
                "activation_counts": {
                    "marder_correction_steps": 3,
                    "marder_dominant_correction_steps": 0,
                },
                "max_observed": {
                    "marder_relative_correction_linf": 0.1,
                    "marder_nondominance_threshold": 0.5,
                },
            },
        },
    )

    decision = packet["method_limiter_decisions"]["marder_correction"]
    assert decision["status"] == "candidate_method_limiter_nondominant_observed"
    assert decision["nondominant_observed"] is True
    assert (
        "review_method_limiter_nondominance_marder_correction_steps"
        not in packet["review_required"]
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_unapplied_raw_ohmic_cfl_exceedance_is_review_not_blocker() -> None:
    packet = build_experimental_limiter_zero_probe_packet(
        declared_scope="unit_scope",
        device_name="unit_device",
        simulation_telemetry={
            "n_steps_completed": 4,
            "final_time_s": 4.0e-13,
            "target_time_s": 4.0e-13,
            "finite_state": {"all_finite": True},
            "limiter_activation_summary": {
                "steps_observed": 4,
                "activation_counts": {
                    "conductivity_ohmic_cfl_limited_steps": 0,
                    "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps": 4,
                    "marder_correction_steps": 4,
                    "marder_dominant_correction_steps": 0,
                },
                "max_observed": {
                    "conductivity_cfl_limited_fraction": 1.0,
                    "marder_relative_correction_linf": 0.1,
                    "marder_nondominance_threshold": 0.5,
                },
            },
        },
    )

    assert packet["total_acceptance_blocking_activations"] == 0
    assert packet["zero_acceptance_blockers_observed"] is True
    assert packet["method_review_counts"][
        "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps"
    ] == 4
    assert (
        packet["method_limiter_decisions"][
            "conductivity_ohmic_cfl_raw_exceedance"
        ]["status"]
        == "candidate_raw_explicit_ohmic_cfl_exceedance_observed_not_applied"
    )
    assert "review_unapplied_raw_ohmic_cfl_exceedance" in packet["review_required"]
