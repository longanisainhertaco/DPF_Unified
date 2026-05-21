import numpy as np
import pytest

from dpf.fields import CircuitMagneticBoundaryDrive, Maxwell3DGrid
from dpf.fields.hybrid_simulator import _circuit_udpf_for_step as _udpf_for_step
from dpf.first_principles import (
    minimal_engineering_deck,
    pf1000_akel_16kv_engineering_deck,
)
from dpf.first_principles import (
    run_first_principles_3d_deck as package_runner,
)
from dpf.first_principles.power_port import build_engineering_power_port_packet
from dpf.first_principles.runner import (
    ENGINEERING_CANDIDATE_STATUS,
    HybridEMPicFluidRun,
    build_first_principles_3d_session,
    run_first_principles_3d_deck,
)


def test_power_port_candidate_residual_budget_tracks_sign_hypotheses() -> None:
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_volume_j_dot_e",
                "circuit_step": {
                    "current_A": 2.0,
                    "udpf_V": -3.0,
                },
            },
        },
        conservation={
            "delta_tracked_total_energy_J": 1.0,
            "initial": {"tracked_total_energy_J": 10.0},
            "final": {
                "tracked_total_energy_J": 11.0,
                "magnetic_energy_J": 2.0,
            },
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": 1.0,
            "cumulative_j_dot_e_step_count": 1,
            "cumulative_active_port_work_J": -3.0,
            "cumulative_active_port_step_count": 1,
            "udpf_source_counts": {"candidate_lagged_volume_j_dot_e": 1},
            "history_stride": 1,
            "history_summary": [
                {
                    "j_dot_e_power_W": 2.0,
                },
            ],
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": 2.0,
                        "domain": "full_cartesian_grid_volume",
                    },
                },
            },
        },
    )

    budget = packet["candidate_power_residual_budget"]
    assert budget["available"] is True
    assert budget["integrated_volume_j_dot_e_work_J"] == 1.0
    assert (
        budget["integrated_volume_j_dot_e_work_source"]
        == "simulator_cumulative_all_completed_steps"
    )
    assert budget["full_completed_step_j_dot_e_integral_available"] is True
    assert budget["cumulative_terminal_active_port_work_J"] == -3.0
    assert budget["full_completed_step_active_port_integral_available"] is True
    assert budget["delta_plus_active_port_work_J"] == -2.0
    assert budget["active_port_plus_integrated_j_dot_e_work_J"] == -2.0
    assert budget["udpf_source_counts"] == {"candidate_lagged_volume_j_dot_e": 1}
    assert budget["retained_volume_j_dot_e_work_J"] == 1.0
    assert budget["delta_minus_integrated_j_dot_e_work_J"] == 0.0
    assert budget["delta_minus_retained_j_dot_e_work_J"] == 0.0
    assert budget["delta_plus_retained_j_dot_e_work_J"] == 2.0
    assert budget["full_retained_history_available"] is True
    assert budget["accepted_residual_tolerance"] == "not_attached"
    assert budget["can_support_power_port_acceptance"] is False
    assert "candidate_power_residual_budget" in packet["candidate_runtime_channels"]
    assert (
        "candidate_full_completed_step_terminal_i_udpf_integral"
        in packet["candidate_runtime_channels"]
    )


def test_power_port_packet_recognizes_auluck_j_dot_e_source_sign_candidate() -> None:
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {
                    "current_A": 2.0,
                    "udpf_V": 5.0,
                },
            },
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -5.0,
            "cumulative_j_dot_e_step_count": 1,
            "cumulative_active_port_work_J": 5.0,
            "cumulative_active_port_step_count": 1,
            "udpf_source_counts": {
                "candidate_lagged_auluck_volume_j_dot_e": 1,
            },
            "history_stride": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -10.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    },
                },
            },
        },
    )

    assert packet["active_load_relation"] == (
        "lagged_auluck_volume_j_dot_e_voltage_not_accepted"
    )
    assert packet["active_load_decision"][
        "candidate_lagged_volume_j_dot_e_is_active_load"
    ] is True
    assert packet["candidate_power_residual_budget"][
        "active_port_plus_integrated_j_dot_e_work_J"
    ] == 0.0
    assert packet["can_support_first_principles_acceptance"] is False


def test_power_port_stage0_packets_defer_sigma_line_voltage_driver() -> None:
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {
                    "current_A": 4.0,
                    "udpf_V": 6.0,
                },
            },
        },
        conservation={
            "delta_tracked_total_energy_J": 3.0,
            "initial": {
                "electric_energy_J": 1.0,
                "magnetic_energy_J": 2.0,
                "tracked_total_energy_J": 20.0,
            },
            "final": {
                "electric_energy_J": 2.0,
                "magnetic_energy_J": 4.0,
                "tracked_total_energy_J": 23.0,
            },
        },
        simulation_telemetry={
            "dt_s": 0.25,
            "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -6.0,
            "cumulative_j_dot_e_step_count": 1,
            "cumulative_active_port_work_J": 6.0,
            "cumulative_active_port_step_count": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -24.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    },
                },
            },
        },
    )

    assert set(packet["stage0_packet_ids"]) == {
        "power_port_source_review",
        "power_port_domain_review",
        "power_port_sign_review",
        "power_port_time_centering_review",
        "power_port_energy_ledger_review",
        "negative_test_plan",
    }
    assert packet["stage0_packet_scaffolds"]["power_port_source_review"][
        "deferred_or_unverified_by_local_sources"
    ] == ["Sigma quasi-TEM line-voltage driver"]
    sigma = packet["power_port_operator_comparison"]["operators"][
        "sigma_quasi_tem_line_voltage"
    ]
    assert sigma["source_status"] == "not_verified_in_local_dpf_source"
    assert sigma["can_be_accepted_now"] is False
    assert packet["sigma_quasi_tem_line_voltage_operator"][
        "disallowed_runtime_use"
    ] == "accepted_or_primary_circuit_driver"
    ledger = packet["candidate_stage0_energy_ledger"]["terms"]
    assert ledger["terminal_port_work_J"] == 6.0
    assert ledger["volume_j_dot_e_work_J"] == -6.0
    assert ledger["stored_em_energy_delta_J"] == 3.0
    assert ledger["wall_poynting_flux_excluding_declared_port_J"] is None


def test_first_principles_3d_session_matches_uninterrupted_split_run() -> None:
    deck = {
        "n_steps": 4,
        "grid_shape": (4, 4, 4),
        "dt_s": 1.0e-13,
        "background_density_m3": 1.0e21,
        "density_floor_m3": 1.0e21,
        "apply_circuit_boundary": True,
        "history_stride": 1,
        "max_step_results": 4,
    }

    uninterrupted = run_first_principles_3d_deck(deck)
    session = build_first_principles_3d_session(deck)
    first_segment = session.run_segment(2)
    second_segment = session.run_segment(2)

    assert session.completed_steps == 4
    assert first_segment.telemetry.continuation_state["total_steps_completed"] == 2
    assert second_segment.telemetry.continuation_state["total_steps_completed"] == 4
    assert second_segment.telemetry.state_fingerprint["sha256"] == (
        uninterrupted.result.telemetry.state_fingerprint["sha256"]
    )
    assert second_segment.telemetry.final_field_energy_J == (
        uninterrupted.result.telemetry.final_field_energy_J
    )
    assert second_segment.telemetry.n_particles_final == (
        uninterrupted.result.telemetry.n_particles_final
    )
    assert second_segment.telemetry.circuit["final_current_A"] == (
        uninterrupted.result.telemetry.circuit["final_current_A"]
    )
    assert (
        second_segment.telemetry.continuation_state["has_lagged_field_work"]
        is True
    )


def test_first_principles_3d_session_adaptive_validity_reaches_short_target() -> None:
    session = build_first_principles_3d_session(
        pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    )

    packet = session.run_adaptive_validity(
        target_time_s=2.0e-13,
        max_steps=4,
    )

    assert packet["status"] == "candidate_adaptive_validity_run_not_validation"
    assert packet["duration_request_satisfied"] is True
    assert packet["termination_reason"] == "target_time_reached"
    assert packet["accepted_step_count"] == 2
    assert packet["rejected_step_count"] == 0
    assert packet["limiter_activation_summary"]["steps_observed"] == 2
    assert packet["state_fingerprint"]["sha256"]


def test_run_first_principles_3d_deck_returns_candidate_manifest_and_telemetry() -> (
    None
):
    result = run_first_principles_3d_deck(
        {
            "n_steps": 2,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "background_density_m3": 1.0e21,
            "density_floor_m3": 1.0e21,
            "apply_circuit_boundary": True,
        }
    )

    assert result.status == ENGINEERING_CANDIDATE_STATUS
    assert result.reduced_models_used is False
    assert result.can_support_first_principles_acceptance is False
    assert result.result.telemetry.n_steps_completed == 2
    assert result.result.telemetry.retained_step_result_count == 2
    assert result.result.telemetry.history_stride == 1
    assert result.result.telemetry.stop_reason == "completed_step_budget"
    assert result.result.telemetry.circuit is not None
    assert result.result.telemetry.circuit["current_history"][0]["sample"] == "initial"
    assert result.result.telemetry.circuit["current_history"][-1]["sample"] == "post_step"
    assert result.result.telemetry.last_step is not None
    assert result.result.telemetry.last_step["electron_energy"]["status"] == (
        "candidate_engineering_electron_energy_closure"
    )
    assert (
        result.result.telemetry.last_step["electron_energy"]["heat_flux"]["status"]
        == "candidate_braginskii_anisotropic_heat_flux_applied"
    )
    assert (
        result.result.telemetry.last_step["electron_energy"]["equilibration_audit"][
            "status"
        ]
        == "candidate_nrl_equal_temperature_equilibration_audit"
    )
    assert result.result.telemetry.last_step["ionization_charge_state"]["status"] == (
        "candidate_deuterium_charge_state_transport"
    )
    assert (
        result.result.telemetry.last_step["ionization_charge_state"]["particle_source"][
            "status"
        ]
        == "candidate_ionization_pic_particle_source"
    )
    assert result.result.telemetry.last_step["source_backed_transport"]["status"] == (
        "candidate_source_backed_partial_ionized_conductivity"
    )
    assert (
        result.result.telemetry.last_step["field_step"]["conductivity"][
            "density_blend_applied"
        ]
        is False
    )
    assert result.result.telemetry.last_step["kinetic_yield"]["status"] == (
        "candidate_engineering_kinetic_yield_history"
    )
    assert result.telemetry["simulation"]["status"] == (
        "candidate_engineering_3d_hybrid_pic_simulation"
    )
    assert result.telemetry["engineering_current_waveform_comparison"]["status"] == (
        "blocked_current_waveform_target_not_bound"
    )
    assert (
        result.validation_packet["engineering_current_waveform_comparison_status"]
        == "blocked_current_waveform_target_not_bound"
    )
    assert result.telemetry["hybrid_pic_3d_readiness"]["status"] == "blocked"
    assert result.telemetry["hybrid_pic_3d_readiness"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "kinetic_ion_pic_push_deposition" in result.telemetry[
        "hybrid_pic_3d_readiness"
    ]["missing_capabilities"]
    assert result.validation_packet["hybrid_pic_3d_readiness_status"] == "blocked"
    assert "same_scope_3d_validation_packet" in result.validation_packet[
        "hybrid_pic_3d_missing_capabilities"
    ]
    assert result.manifest["validation_status"] == "not_validation"
    assert result.manifest["scientific_status"] == ENGINEERING_CANDIDATE_STATUS
    assert result.manifest["reduced_models_used"] is False
    assert result.telemetry["startup"]["mode"] == "source_backed_end_rundown_sheath"
    assert result.telemetry["startup"]["status"] == (
        "blocked_startup_bvp_packet_not_available"
    )
    assert result.telemetry["pic_particle_loading"]["status"] == (
        "candidate_density_normalized_pic_loading_not_validation"
    )
    assert result.manifest["candidate_evidence"]["pic_particle_loading_packet"] == (
        result.telemetry["pic_particle_loading"]
    )
    assert result.manifest["candidate_evidence"][
        "hybrid_pic_3d_readiness_packet"
    ] == result.telemetry["hybrid_pic_3d_readiness"]
    assert result.telemetry["startup"]["decision"] == (
        "do_not_promote_startup_to_whole_shot_first_principles"
    )
    assert result.telemetry["startup"]["whole_shot_startup_blocked"] is True
    assert (
        "breakdown_or_flashover_model"
        in result.telemetry["startup"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["startup"]["startup_channel_status"][
            "breakdown_or_flashover_model"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["startup"]["candidate_breakdown_audit"]["status"]
        == "candidate_civ_paschen_breakdown_audit_unavailable"
    )
    assert "surface_breakdown_bvp" in result.telemetry["startup"]["accepted_modes"]
    assert (
        result.telemetry["startup"]["startup_mode_status"][
            "source_backed_end_rundown_sheath"
        ]["status"]
        == "engineering_candidate_not_whole_shot"
    )
    assert (
        result.telemetry["startup"]["mode_payload_status"][
            "breakdown_liftoff_exclusion"
        ]
        == "missing_or_unreviewed_payload"
    )
    assert (
        result.telemetry["startup"]["candidate_input_policy"][
            "candidate_inputs_can_support_whole_shot_acceptance"
        ]
        is False
    )
    assert (
        result.telemetry["startup"]["negative_test_policy"][
            "end_rundown_whole_shot_rejection_required"
        ]
        is True
    )
    assert result.telemetry["limiter_readiness"]["status"] == (
        "blocked_limiter_readiness_packet_not_available"
    )
    assert (
        result.telemetry["limiter_readiness"]["can_support_limiter_zero_acceptance"]
        is False
    )
    assert (
        "active_path_limiter_inventory"
        in result.telemetry["limiter_readiness"]["missing_acceptance_channels"]
    )
    assert (
        "zero_acceptance_blocker_full_run"
        in result.telemetry["limiter_readiness"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["limiter_readiness"]["limiter_channel_status"][
            "zero_acceptance_blocker_full_run"
        ]
        == "missing_or_blocked"
    )
    assert (
        "candidate_finite_conservation_snapshot"
        in result.telemetry["limiter_readiness"]["candidate_runtime_channels"]
    )
    assert (
        result.telemetry["limiter_readiness"]["runtime_observations"]["candidate_only"]
        is True
    )
    assert (
        result.telemetry["limiter_readiness"]["limiter_family_status"][
            "repair_or_fallback"
        ]["status"]
        == "requires_inventory_and_review"
    )
    assert result.telemetry["power_port"]["authority_contract"] == (
        "field_power_required"
    )
    assert (
        result.telemetry["power_port"]["can_support_first_principles_acceptance"]
        is False
    )
    assert result.telemetry["power_port"]["accepted_load_power_source"] == "none"
    assert result.telemetry["power_port"]["diagnostic_field_inductance_H"] > 0.0
    assert result.telemetry["power_port"]["magnetic_energy_inductance_authority"] == (
        "diagnostic_only_not_circuit_load"
    )
    assert (
        result.telemetry["power_port"]["power_port_channel_status"]["terminal_current"]
        == "candidate_runtime_only_not_acceptance"
    )
    assert (
        result.telemetry["power_port"]["power_port_channel_status"][
            "poynting_power_or_j_dot_e"
        ]
        == "candidate_runtime_only_not_acceptance"
    )
    assert result.telemetry["power_port"]["j_dot_e_power_W"] is not None
    assert result.telemetry["power_port"]["j_dot_e_domain"] == (
        "resolved_plasma_current_carrying_cells"
    )
    assert (
        result.telemetry["power_port"]["energy_ledger_status"]["magnetic_energy"][
            "status"
        ]
        == "candidate_runtime_only_not_acceptance"
    )
    assert result.telemetry["power_port"]["active_load_relation"] == (
        "lagged_volume_j_dot_e_voltage_not_accepted"
    )
    assert result.telemetry["power_port"]["active_load_decision"]["decision"] == (
        "candidate_lagged_field_power_load_not_accepted"
    )
    assert (
        result.telemetry["power_port"]["active_load_decision"][
            "diagnostic_relations_do_not_define_load"
        ]
        is True
    )
    assert result.telemetry["power_port"]["acceptance_gate"].startswith(
        "terminal_current_voltage_and_energy_ledger_candidates_cannot_support"
    )
    assert (
        result.telemetry["power_port"]["negative_test_policy"][
            "diagnostic_inductance_as_load_rejection_required"
        ]
        is True
    )
    assert (
        "candidate_diagnostic_field_inductance"
        in result.telemetry["power_port"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_volume_j_dot_e_power"
        in result.telemetry["power_port"]["candidate_runtime_channels"]
    )
    assert (
        result.telemetry["power_port"]["active_load_decision"][
            "candidate_volume_j_dot_e_is_not_active_load"
        ]
        is False
    )
    assert (
        result.telemetry["power_port"]["active_load_decision"][
            "candidate_lagged_volume_j_dot_e_is_active_load"
        ]
        is True
    )
    assert (
        result.telemetry["power_port"]["power_port_step_records"][0][
            "residual_interpretation"
        ]
        == "tracked_energy_delta_not_accepted_power_port_residual"
    )
    residual_budget = result.telemetry["power_port"]["candidate_power_residual_budget"]
    assert residual_budget["status"] == "candidate_power_residual_budget_not_validation"
    assert residual_budget["available"] is True
    assert residual_budget["tracked_energy_delta_J"] is not None
    assert residual_budget["integrated_volume_j_dot_e_work_J"] is not None
    assert (
        residual_budget["integrated_volume_j_dot_e_work_source"]
        == "simulator_cumulative_all_completed_steps"
    )
    assert (
        residual_budget["full_completed_step_j_dot_e_integral_available"]
        is True
    )
    assert (
        residual_budget["cumulative_terminal_active_port_work_J"]
        is not None
    )
    assert (
        residual_budget["full_completed_step_active_port_integral_available"]
        is True
    )
    assert (
        residual_budget["cumulative_terminal_active_port_step_count"]
        == result.telemetry["simulation"]["n_steps_completed"]
    )
    assert residual_budget["udpf_source_counts"]
    assert residual_budget["retained_volume_j_dot_e_work_J"] is not None
    assert residual_budget["accepted_residual_tolerance"] == "not_attached"
    assert residual_budget["can_support_power_port_acceptance"] is False
    assert (
        "candidate_power_residual_budget"
        in result.telemetry["power_port"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_full_completed_step_terminal_i_udpf_integral"
        in result.telemetry["power_port"]["candidate_runtime_channels"]
    )
    assert (
        result.telemetry["power_port"]["residual_policy"][
            "candidate_power_residual_budget_available"
        ]
        is True
    )
    assert (
        result.telemetry["power_port"]["power_port_step_records"][0]["j_dot_e_power_W"]
        is not None
    )
    assert (
        result.telemetry["power_port"]["power_port_step_records"][0][
            "active_load_relation"
        ]
        == "lagged_volume_j_dot_e_voltage_not_accepted"
    )
    assert (
        result.telemetry["power_port"]["power_port_step_records"][0][
            "can_support_first_principles_acceptance"
        ]
        is False
    )
    assert (
        "poynting_or_j_dot_e_power_integral"
        in result.telemetry["power_port"]["missing_acceptance_channels"]
    )
    assert (
        "startup_handoff_interval"
        in result.telemetry["power_port"]["missing_acceptance_channels"]
    )
    assert result.telemetry["dimensionality_handoff"]["geometry_dimensionality"] == "3d"
    assert result.telemetry["dimensionality_handoff"]["decision"] == (
        "do_not_claim_unrestricted_whole_shot_dimensionality_authority"
    )
    assert result.telemetry["dimensionality_handoff"]["allowed_claim"] == (
        "engineering_3d_hybrid_em_pic_fluid_candidate_only"
    )
    assert (
        result.telemetry["dimensionality_handoff"][
            "can_support_first_principles_acceptance"
        ]
        is False
    )
    assert (
        result.telemetry["dimensionality_handoff"]["handoff_channel_status"][
            "geometry_dimensionality"
        ]
        == "candidate_true_3d_grid_not_acceptance"
    )
    assert (
        result.telemetry["dimensionality_handoff"]["handoff_channel_status"][
            "mhd_to_kinetic_state_transfer"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["dimensionality_handoff"]["claim_mode_status"][
            "validated_3d_hybrid_pic_fluid_claim"
        ]["decision"]
        == "engineering_candidate_only"
    )
    assert (
        result.telemetry["dimensionality_handoff"]["source_model_limitation_status"][
            "source_hybrid_reference_axisymmetric_m0_not_full_3d"
        ]["status"]
        == "blocks_unrestricted_whole_shot_acceptance"
    )
    assert (
        result.telemetry["dimensionality_handoff"]["handoff_observable_status"][
            "beam_target_neutron_yield"
        ]["requires_mechanism_separated_neutron_authority"]
        is True
    )
    assert (
        result.telemetry["dimensionality_handoff"]["upstream_acceptance_gate"]["status"]
        == "blocked_by_upstream_packets"
    )
    assert result.telemetry["dimensionality_handoff"]["acceptance_gate"].startswith(
        "true_3d_runtime_channels_cannot_support_unrestricted"
    )
    assert (
        result.telemetry["dimensionality_handoff"]["negative_test_policy"][
            "mhd_only_pinch_neutron_rejection_required"
        ]
        is True
    )
    assert (
        "mhd_to_kinetic_handoff_state"
        in result.telemetry["dimensionality_handoff"]["missing_acceptance_channels"]
    )
    assert (
        "candidate_true_3d_grid"
        in result.telemetry["dimensionality_handoff"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_separate_electron_energy_source_terms"
        in result.telemetry["dimensionality_handoff"]["candidate_runtime_channels"]
    )
    assert (
        "source_hybrid_reference_axisymmetric_m0_not_full_3d"
        in result.telemetry["dimensionality_handoff"]["source_model_limitations"]
    )
    assert "fully_kinetic_pinch_claim" in {
        mode["mode"]
        for mode in result.telemetry["dimensionality_handoff"]["claim_modes"]
    }
    assert result.telemetry["physics_closure"]["status"] == (
        "candidate_engineering_closure_packet_not_validation"
    )
    assert result.telemetry["physics_closure"]["decision"] == (
        "do_not_promote_without_complete_physics_closure_matrix"
    )
    assert (
        result.telemetry["physics_closure"]["can_support_first_principles_acceptance"]
        is False
    )
    assert (
        "source_equations_or_bound"
        in result.telemetry["physics_closure"]["required_packet_channels"]
    )
    assert (
        result.telemetry["physics_closure"]["effects"]["eos_thermodynamics"][
            "channel_status"
        ]["source_equations_or_bound"]
        == "missing_or_unaccepted"
    )
    assert (
        "candidate_electron_energy_source_terms"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_braginskii_electron_heat_flux"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_electron_ion_equilibration_audit"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_ionization_charge_state_transport"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_source_backed_partial_ionized_conductivity"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        "ionization_charge_state"
        in result.telemetry["physics_closure"]["active_candidate_closures"]
    )
    assert (
        "accepted_ionization_recombination_model"
        in result.telemetry["physics_closure"]["effects"]["ionization_charge_state"][
            "missing_channels"
        ]
    )
    assert (
        "accepted_charge_state_transport"
        in result.telemetry["physics_closure"]["effects"]["ionization_charge_state"][
            "missing_channels"
        ]
    )
    assert (
        "accepted_neutral_particle_source_coupling"
        in result.telemetry["physics_closure"]["effects"]["ionization_charge_state"][
            "missing_channels"
        ]
    )
    assert (
        "accepted_transport_validity_regime"
        in result.telemetry["physics_closure"]["effects"][
            "electrical_thermal_transport"
        ]["missing_channels"]
    )
    assert (
        "accepted_thermal_conduction_closure"
        in result.telemetry["physics_closure"]["effects"][
            "electrical_thermal_transport"
        ]["missing_channels"]
    )
    assert (
        "accepted_electron_heat_flux"
        in result.telemetry["physics_closure"]["effects"][
            "single_two_temperature_energy"
        ]["missing_channels"]
    )
    assert (
        "accepted_electron_ion_collisional_coupling"
        in result.telemetry["physics_closure"]["effects"][
            "single_two_temperature_energy"
        ]["missing_channels"]
    )
    assert (
        "candidate_kinetic_yield_history"
        in result.telemetry["physics_closure"]["candidate_runtime_channels"]
    )
    assert (
        result.telemetry["physics_closure"]["community_formula_audit"]["dependency"]
        == "plasmapy"
    )
    assert (
        result.telemetry["physics_closure"]["community_formula_audit"][
            "source_truth_policy"
        ]["local_knowledge_reference_remains_authority"]
        is True
    )
    assert (
        result.telemetry["physics_closure"]["community_formula_audit_policy"][
            "optional_audit_can_support_acceptance"
        ]
        is False
    )
    assert (
        "single_two_temperature_energy"
        in result.telemetry["physics_closure"]["active_candidate_closures"]
    )
    assert (
        result.telemetry["physics_closure"]["active_closure_policy"][
            "candidate_closures_can_support_acceptance"
        ]
        is False
    )
    assert (
        result.telemetry["physics_closure"]["closure_effect_status"][
            "beam_target_coupling"
        ]["classification"]
        == "candidate_only"
    )
    assert (
        result.telemetry["physics_closure"]["dimensionality_acceptance_gate"]["status"]
        == "blocked_by_dimensionality_or_handoff_packet"
    )
    assert result.telemetry["physics_closure"]["acceptance_gate"].startswith(
        "candidate_transport_ohm_electron_energy_hall_instability"
    )
    assert (
        result.telemetry["physics_closure"]["negative_test_policy"][
            "total_yield_without_mechanism_separation_rejection_required"
        ]
        is True
    )
    assert (
        "eos_thermodynamics"
        in result.telemetry["physics_closure"]["missing_or_unaccepted_effects"]
    )
    assert (
        result.telemetry["physics_closure"]["effects"]["eos_thermodynamics"][
            "required_packet_channels"
        ]
        == result.telemetry["physics_closure"]["required_packet_channels"]
    )
    assert (
        result.telemetry["physics_closure"]["closure_matrix_status_by_effect"][
            "radiation_losses"
        ]
        == "blocked"
    )
    assert (
        result.telemetry["physics_closure"]["effects"]["beam_target_coupling"]["status"]
        == "candidate"
    )
    assert result.telemetry["same_scope_source"]["status"] == (
        "blocked_same_scope_source_packet_not_available"
    )
    assert (
        result.telemetry["same_scope_source"]["can_support_first_principles_acceptance"]
        is False
    )
    assert (
        "accepted_digitized_current_waveform"
        in result.telemetry["same_scope_source"]["missing_acceptance_channels"]
    )
    assert result.telemetry["waveform_phase"]["status"] == (
        "blocked_waveform_phase_packet_not_available"
    )
    assert (
        result.telemetry["waveform_phase"]["can_support_first_principles_acceptance"]
        is False
    )
    assert (
        result.telemetry["waveform_phase"]["draft_digitization_packet_status"][
            "accepted_for_validation"
        ]
        is False
    )
    assert (
        result.telemetry["waveform_phase"]["draft_digitization_packet_status"][
            "review_status"
        ]
        == "draft"
    )
    assert (
        "review_status_accepted"
        in result.telemetry["waveform_phase"]["required_review_channels"]
    )
    assert (
        "accepted_current_derivative_or_dip_trace"
        in result.telemetry["waveform_phase"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["waveform_phase"]["waveform_phase_channel_status"][
            "accepted_digitized_current_waveform"
        ]
        == "missing_or_blocked"
    )
    assert result.telemetry["spatial_field_temperature"]["status"] == (
        "blocked_spatial_field_temperature_packet_not_available"
    )
    assert (
        result.telemetry["spatial_field_temperature"][
            "can_support_first_principles_acceptance"
        ]
        is False
    )
    assert (
        "accepted_same_scope_density_history"
        in result.telemetry["spatial_field_temperature"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["spatial_field_temperature"][
            "spatial_field_temperature_channel_status"
        ]["accepted_same_scope_density_history"]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["spatial_field_temperature"]["cross_scope_policy"][
            "can_use_other_scope_for_acceptance"
        ]
        is False
    )
    assert result.telemetry["neutron_authority"]["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert (
        result.telemetry["neutron_authority"]["can_support_first_principles_acceptance"]
        is False
    )
    assert (
        "mechanism_separated_yield_channels"
        in result.telemetry["neutron_authority"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["neutron_authority"]["neutron_authority_channel_status"][
            "mechanism_separated_yield_channels"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["neutron_authority"]["cross_scope_policy"][
            "can_use_other_scope_for_acceptance"
        ]
        is False
    )
    assert (
        result.telemetry["neutron_authority"]["mechanism_separation_policy"][
            "total_yield_is_not_authoritative_without_separate_mechanisms"
        ]
        is True
    )
    assert (
        "candidate_pic_ion_neutron_yield_history"
        in result.telemetry["neutron_authority"]["candidate_runtime_channels"]
    )
    assert result.telemetry["comparator_uq"]["status"] == (
        "blocked_comparator_uq_matrix_not_available"
    )
    assert (
        result.telemetry["comparator_uq"]["can_support_first_principles_acceptance"]
        is False
    )
    assert (
        "output_field_mapping_by_observable"
        in result.telemetry["comparator_uq"]["missing_acceptance_channels"]
    )
    assert (
        "pass_fail_rule_by_observable"
        in result.telemetry["comparator_uq"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["comparator_uq"]["comparator_uq_channel_status"][
            "pass_fail_rule_by_observable"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["comparator_uq"]["observable_group_status"][
            "neutron_spectrum"
        ]["decision"]
        == "blocked_until_full_comparator_uq_channels_pass"
    )
    assert (
        result.telemetry["comparator_uq"]["cross_scope_policy"][
            "can_use_other_scope_for_acceptance"
        ]
        is False
    )
    assert (
        result.telemetry["comparator_uq"]["upstream_packet_statuses"][
            "neutron_authority"
        ]
        == "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert (
        result.telemetry["comparator_uq"]["upstream_acceptance_gate"]["status"]
        == "blocked_by_upstream_packets"
    )
    assert result.telemetry["numerical_fidelity"]["status"] == (
        "blocked_numerical_fidelity_packet_not_available"
    )
    assert (
        result.telemetry["numerical_fidelity"]["can_support_numerical_acceptance"]
        is False
    )
    assert (
        "mesh_timestep_convergence_packet"
        in result.telemetry["numerical_fidelity"]["missing_acceptance_channels"]
    )
    assert (
        "limiter_zero_packet"
        in result.telemetry["numerical_fidelity"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["numerical_fidelity"]["numerical_channel_status"][
            "mesh_timestep_convergence_packet"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["numerical_fidelity"]["test_surface_status"][
            "maxwell_yee_update_and_courant_limit"
        ]["status"]
        == "candidate_component_coverage_not_acceptance"
    )
    assert (
        "tests/test_maxwell_3d_field_core.py"
        in result.telemetry["numerical_fidelity"]["test_surface_status"][
            "maxwell_yee_update_and_courant_limit"
        ]["candidate_artifacts"]
    )
    assert (
        result.telemetry["numerical_fidelity"]["test_surface_status"][
            "mesh_and_timestep_convergence"
        ]["status"]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["numerical_fidelity"]["runtime_observations"]["candidate_only"]
        is True
    )
    assert (
        result.telemetry["numerical_fidelity"]["runtime_observations"][
            "tolerance_claim"
        ]
        is False
    )
    assert (
        result.telemetry["numerical_fidelity"]["negative_test_policy"][
            "candidate_component_promotion_rejection_required"
        ]
        is True
    )
    assert (
        result.telemetry["numerical_fidelity"]["upstream_acceptance_gate"]["status"]
        == "blocked_by_upstream_packets"
    )
    assert (
        "candidate_conservation_telemetry"
        in result.telemetry["numerical_fidelity"]["candidate_runtime_channels"]
    )
    assert (
        "candidate_full_completed_step_terminal_i_udpf_integral"
        in result.telemetry["numerical_fidelity"]["candidate_runtime_channels"]
    )
    assert (
        result.telemetry["numerical_fidelity"]["upstream_packet_statuses"][
            "limiter_readiness"
        ]
        == "blocked_limiter_readiness_packet_not_available"
    )
    assert (
        "candidate_divergence_b_diagnostic"
        in result.telemetry["numerical_fidelity"]["candidate_runtime_channels"]
    )
    assert result.telemetry["certificate_gate"]["status"] == (
        "blocked_first_principles_certificate_not_available"
    )
    assert (
        result.telemetry["certificate_gate"]["can_write_accepted_certificate"] is False
    )
    assert (
        "run_manifest_hash"
        in result.telemetry["certificate_gate"]["missing_acceptance_channels"]
    )
    assert (
        "negative_test_missing_uq"
        in result.telemetry["certificate_gate"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["certificate_gate"]["certificate_channel_status"][
            "negative_test_missing_uq"
        ]
        == "missing_or_blocked"
    )
    assert result.telemetry["certificate_gate"]["release_decision"] == (
        "do_not_release_first_principles_claim"
    )
    assert (
        result.telemetry["certificate_gate"]["upstream_packet_statuses"][
            "comparator_uq"
        ]
        == "blocked_comparator_uq_matrix_not_available"
    )
    assert (
        result.telemetry["certificate_gate"]["upstream_packet_statuses"][
            "numerical_fidelity"
        ]
        == "blocked_numerical_fidelity_packet_not_available"
    )
    assert (
        result.telemetry["certificate_gate"]["upstream_packet_statuses"][
            "limiter_readiness"
        ]
        == "blocked_limiter_readiness_packet_not_available"
    )
    assert (
        result.telemetry["certificate_gate"]["upstream_packet_acceptance_matrix"][
            "comparator_uq_packet_accepted"
        ]["accepted_for_certificate"]
        is False
    )
    assert (
        result.telemetry["certificate_gate"]["negative_test_matrix"][
            "negative_test_missing_uq"
        ]["decision"]
        == "missing_required_negative_test"
    )
    assert result.telemetry["generalization"]["status"] == (
        "blocked_generalized_dpf_machine_path_not_available"
    )
    assert (
        result.telemetry["generalization"]["can_claim_generalized_dpf_machine"] is False
    )
    assert (
        "accepted_primary_scope_certificate"
        in result.telemetry["generalization"]["missing_acceptance_channels"]
    )
    assert (
        "second_scope_certificate"
        in result.telemetry["generalization"]["missing_acceptance_channels"]
    )
    assert (
        result.telemetry["generalization"]["generalization_channel_status"][
            "second_scope_certificate"
        ]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["generalization"]["claim_policy"][
            "single_scope_is_not_generalized"
        ]
        is True
    )
    assert (
        result.telemetry["generalization"]["required_second_scope_gate_ids"][-1]
        == "FP-14"
    )
    assert (
        result.telemetry["generalization"]["upstream_packet_statuses"][
            "certificate_gate"
        ]
        == "blocked_first_principles_certificate_not_available"
    )
    assert (
        result.telemetry["generalization"]["upstream_packet_statuses"][
            "numerical_fidelity"
        ]
        == "blocked_numerical_fidelity_packet_not_available"
    )
    assert (
        result.telemetry["generalization"]["upstream_packet_statuses"][
            "limiter_readiness"
        ]
        == "blocked_limiter_readiness_packet_not_available"
    )
    assert "mjolnir_60kv_735kj_9torr_mechanism_scope" in {
        scope["scope_id"]
        for scope in result.telemetry["generalization"]["candidate_second_scopes"]
    }
    assert "faeton_i_100kv_second_device_scope" in {
        scope["scope_id"]
        for scope in result.telemetry["generalization"]["candidate_second_scopes"]
    }
    assert (
        result.manifest["metadata"]["deck"]["startup"]["mode"]
        == "source_backed_end_rundown_sheath"
    )
    assert (
        result.manifest["metadata"]["deck"]["fluid"]["initial_ionization_fraction"]
        == 0.01
    )
    assert (
        result.manifest["candidate_evidence"]["first_principles_3d_runner"]["status"]
        == ENGINEERING_CANDIDATE_STATUS
    )
    assert result.manifest["candidate_evidence"]["startup_bvp_packet"] == (
        result.telemetry["startup"]
    )
    assert result.manifest["candidate_evidence"]["power_port_packet"] == (
        result.telemetry["power_port"]
    )
    assert result.manifest["candidate_evidence"]["limiter_readiness_packet"] == (
        result.telemetry["limiter_readiness"]
    )
    assert result.manifest["candidate_evidence"]["dimensionality_handoff_packet"] == (
        result.telemetry["dimensionality_handoff"]
    )
    assert result.manifest["candidate_evidence"]["physics_closure_packet"] == (
        result.telemetry["physics_closure"]
    )
    assert result.manifest["candidate_evidence"]["same_scope_source_packet"] == (
        result.telemetry["same_scope_source"]
    )
    assert result.manifest["candidate_evidence"]["waveform_phase_packet"] == (
        result.telemetry["waveform_phase"]
    )
    assert (
        result.manifest["candidate_evidence"]["spatial_field_temperature_packet"]
        == result.telemetry["spatial_field_temperature"]
    )
    assert result.manifest["candidate_evidence"]["neutron_authority_packet"] == (
        result.telemetry["neutron_authority"]
    )
    assert result.manifest["candidate_evidence"]["comparator_uq_packet"] == (
        result.telemetry["comparator_uq"]
    )
    assert result.manifest["candidate_evidence"]["numerical_fidelity_packet"] == (
        result.telemetry["numerical_fidelity"]
    )
    assert result.manifest["candidate_evidence"]["certificate_gate_packet"] == (
        result.telemetry["certificate_gate"]
    )
    assert result.manifest["candidate_evidence"]["generalization_packet"] == (
        result.telemetry["generalization"]
    )
    assert result.conservation_telemetry["status"] == (
        "engineering_candidate_conservation_telemetry_not_validation"
    )
    assert np.isfinite(
        result.conservation_telemetry["relative_tracked_total_energy_change"]
    )
    assert result.validation_packet["status"] == "not_validation"
    assert result.validation_packet["startup_bvp_status"] == (
        "blocked_startup_bvp_packet_not_available"
    )
    assert result.validation_packet["dimensionality_handoff_status"] == (
        "candidate_engineering_dimensionality_handoff_not_validation"
    )
    assert (
        "mhd_to_kinetic_handoff_state"
        in result.validation_packet[
            "dimensionality_handoff_missing_acceptance_channels"
        ]
    )
    assert (
        "beam_target_neutron_authority"
        in result.validation_packet["dimensionality_handoff_blocked_observables"]
    )
    assert (
        result.validation_packet[
            "dimensionality_handoff_can_support_first_principles_acceptance"
        ]
        is False
    )
    assert result.validation_packet["can_support_first_principles_acceptance"] is False


def test_first_principles_runner_loads_pic_particles_from_deck_density() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "grid_spacing_m": (1.0e-3, 1.0e-3, 1.0e-3),
            "dt_s": 1.0e-13,
            "background_density_m3": 2.0e20,
            "initial_ionization_fraction": 0.25,
            "apply_circuit_boundary": False,
        }
    )

    loading = result.telemetry["pic_particle_loading"]
    expected_ions = 64 * 2.0e20 * 0.25 * 1.0e-9
    assert loading["loading_policy"] == (
        "six_stream_zero_mean_thermal_moment_quadrature_per_active_cell"
    )
    assert loading["active_loaded_cells"] == 64
    assert loading["macroparticles_loaded"] == 384
    assert loading["velocity_quadrature_directions_per_cell"] == 6
    assert loading["initial_ion_density_m3"] == 5.0e19
    assert np.isclose(loading["macro_particle_weight_min"], 5.0e10 / 6.0)
    assert np.isclose(loading["macro_particle_weight_max"], 5.0e10 / 6.0)
    assert loading["ion_thermal_speed_m_s"] > 0.0
    assert np.isclose(loading["represented_physical_ions"], expected_ions)
    assert result.result.telemetry.n_particles_initial == 384
    assert (
        result.telemetry["candidate_evidence"][
            "density_normalized_pic_particle_loading"
        ]["status"]
        == "candidate"
    )
    assert loading["can_support_first_principles_acceptance"] is False


def test_first_principles_runner_does_not_create_zero_weight_initial_particles() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "grid_spacing_m": (1.0e-3, 1.0e-3, 1.0e-3),
            "dt_s": 1.0e-13,
            "background_density_m3": 2.0e20,
            "initial_ionization_fraction": 0.0,
            "apply_circuit_boundary": False,
        }
    )

    loading = result.telemetry["pic_particle_loading"]
    assert loading["active_loaded_cells"] == 64
    assert loading["macroparticles_loaded"] == 0
    assert loading["represented_physical_ions"] == 0.0
    assert result.result.telemetry.n_particles_initial == 0


def test_package_exports_first_principles_3d_runner() -> None:
    assert package_runner is run_first_principles_3d_deck


def test_hybrid_em_pic_fluid_run_can_disable_optional_circuit_boundary() -> None:
    grid = Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    runner = HybridEMPicFluidRun(
        {
            "grid": grid,
            "n_steps": 1,
            "dt_s": 1.0e-13,
            "apply_circuit_boundary": False,
        }
    )

    result = runner.run()

    assert result.status == ENGINEERING_CANDIDATE_STATUS
    assert result.result.circuit is None
    assert result.result.telemetry.circuit is None
    assert result.telemetry["grid_spacing_m"] == [1.0e-3, 1.0e-3, 1.0e-3]
    assert result.telemetry["power_port"]["active_load_relation"] == (
        "no_active_circuit_boundary"
    )
    assert (
        result.telemetry["power_port"]["power_port_channel_status"]["terminal_current"]
        == "missing_or_blocked"
    )
    assert (
        result.telemetry["power_port"]["active_load_decision"]["active_load_relation"]
        == "no_active_circuit_boundary"
    )
    assert (
        "terminal_current"
        in result.telemetry["power_port"]["missing_acceptance_channels"]
    )
    assert result.manifest["validation_status"] == "not_validation"


def test_hybrid_em_pic_fluid_run_accepts_supplied_circuit_boundary() -> None:
    grid = Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    boundary = CircuitMagneticBoundaryDrive(grid)
    result = run_first_principles_3d_deck(
        {
            "grid": grid,
            "n_steps": 1,
            "dt_s": 1.0e-13,
            "apply_circuit_boundary": True,
        },
        circuit_boundary=boundary,
    )

    assert result.result.circuit is not None
    assert result.result.telemetry.circuit is not None
    assert result.result.telemetry.circuit["last"]["boundary"]["faces_updated"] > 0
    assert result.manifest["can_support_first_principles_acceptance"] is False


def test_first_principles_runner_reports_low_current_p_over_i_feedback_blocker() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 2,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "background_density_m3": 1.0e21,
            "density_floor_m3": 1.0e21,
            "apply_circuit_boundary": True,
            "circuit_udpf_mode": "lagged_auluck_volume_j_dot_e",
            "circuit_state": {"current_A": 0.0, "charge_C": 0.0},
            "circuit_feedback_min_current_A": 1.0,
            "history_stride": 1,
            "max_step_results": 2,
        }
    )

    last = result.result.telemetry.circuit["last"]
    feedback = last["low_current_feedback"]
    assert feedback["status"] == (
        "blocked_low_current_p_over_i_singularity_not_validation"
    )
    assert feedback["p_over_i_formula_active_candidate"] is True
    assert feedback["low_current_threshold_hit"] is True
    assert feedback["singularity_blocked_this_step"] is True
    assert result.telemetry["power_port"]["low_current_p_over_i_singularity"][
        "status"
    ] == "blocked_low_current_p_over_i_singularity_not_validation"
    assert result.result.telemetry.udpf_source_counts[
        "input_sequence_fallback_low_current"
    ] == 1


def test_first_principles_runner_applies_candidate_boundary_policy() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (5, 5, 5),
            "dt_s": 1.0e-13,
            "apply_circuit_boundary": False,
            "pml_cells": 1,
            "pml_strength": 0.25,
            "particle_absorption_enabled": True,
        }
    )

    boundary_policy = result.telemetry["boundary_policy"]
    assert boundary_policy["status"] == (
        "candidate_engineering_boundary_policy_not_validation"
    )
    assert boundary_policy["pml_cells"] == 1
    assert boundary_policy["pml_strength"] == 0.25
    assert boundary_policy["particle_absorption_enabled"] is True
    assert (
        boundary_policy["field_boundary_runtime"][
            "maxwell_core_receives_boundary_policy"
        ]
        is True
    )
    assert (
        boundary_policy["particle_boundary_runtime"]["absorbing_boundary_enabled"]
        is True
    )
    assert boundary_policy["can_support_first_principles_acceptance"] is False

    last_step = result.result.telemetry.last_step
    assert last_step is not None
    assert last_step["particle_boundaries"]["status"] == (
        "candidate_engineering_particle_absorption"
    )
    assert (
        result.manifest["metadata"]["deck"]["first_principles_3d"]["boundary_policy"][
            "pml_cells"
        ]
        == 1
    )
    assert result.manifest["candidate_evidence"]["boundary_policy_packet"] == (
        boundary_policy
    )
    assert "pml_conductor_particle_boundaries" in result.telemetry["candidate_evidence"]


def test_first_principles_runner_projects_candidate_conductor_mask_from_package_deck() -> (
    None
):
    from dpf.first_principles import pf1000_akel_16kv_engineering_deck

    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    result = run_first_principles_3d_deck(deck)

    boundary_policy = result.telemetry["boundary_policy"]
    conductor_mask = boundary_policy["conductor_mask"]
    assert conductor_mask["status"] == (
        "candidate_engineering_conductor_mask_not_validation"
    )
    assert conductor_mask["mask_source"] == (
        "candidate_pf1000_rod_hollow_projection"
    )
    assert conductor_mask["conductor_mask_status"] == "candidate_geometry_mask"
    assert conductor_mask["conductor_mask_mode"] == ("pf1000_rod_hollow_projection")
    assert conductor_mask["device_cathode_rod_count"] == 12
    assert conductor_mask["pf1000_geometry_features"]["cathode_rods_projected"] is True
    assert (
        conductor_mask["pf1000_geometry_features"][
            "hollow_anode_inner_radius_supplied"
        ]
        is False
    )
    assert conductor_mask["conductor_cells_active"] > 0
    assert (
        boundary_policy["field_boundary_runtime"]["conductor_e_zero_candidate"] is True
    )
    assert boundary_policy["can_support_first_principles_acceptance"] is False
    assert (
        result.manifest["metadata"]["deck"]["first_principles_3d"]["boundary_policy"][
            "conductor_mask_mode"
        ]
        == "pf1000_rod_hollow_projection"
    )


def test_pf1000_runner_emits_source_locked_deck_diff_packet() -> None:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    result = run_first_principles_3d_deck(deck)

    packet = result.telemetry["deck_diff"]
    assert packet["status"] == "candidate_source_locked_deck_match_not_validation"
    assert packet["deck_lock"] == "pf1000_akel_16kv_1p2torr_shot_12581"
    assert packet["mismatch_keys"] == []
    assert packet["comparisons"]["circuit_voltage_V"]["status"] == (
        "source_locked_match_not_validation"
    )
    assert packet["comparisons"]["gas_pressure_Pa"]["status"] == (
        "source_locked_match_not_validation"
    )
    assert result.manifest["candidate_evidence"]["deck_diff_packet"] == packet


def test_pf1000_candidate_breakdown_profile_seeds_insulator_layer_only() -> None:
    from dpf.first_principles import pf1000_akel_16kv_engineering_deck

    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    result = run_first_principles_3d_deck(deck)

    startup = result.telemetry["startup"]
    assert startup["status"] == "rejected_startup_mode_for_first_principles"
    assert startup["mode"] == "seeded_layer"
    loading = result.telemetry["pic_particle_loading"]
    profile = loading["initial_plasma_profile"]
    assert profile["profile_type"] == "annular_axial_sheath"
    assert profile["ionization_fraction_min"] == 0.0
    assert profile["ionization_fraction_max"] == pytest.approx(0.1)
    assert "preaccelerated_current_sheath" in profile["applied_regions"]
    assert "background_prefill" not in profile["applied_regions"]
    assert loading["active_cells_with_positive_ion_weight"] > 0
    assert loading["active_cells_with_positive_ion_weight"] < loading["active_loaded_cells"]


def test_first_principles_3d_runner_rejects_invalid_step_count() -> None:
    runner = HybridEMPicFluidRun({"n_steps": 0})

    try:
        runner.run()
    except ValueError as exc:
        assert "n_steps must be a positive integer" in str(exc)
    else:
        raise AssertionError("expected invalid n_steps to fail closed")


def test_first_principles_3d_runner_rejects_invalid_boundary_policy() -> None:
    runner = HybridEMPicFluidRun({"n_steps": 1, "pml_cells": -1})

    try:
        runner.run()
    except ValueError as exc:
        assert "pml_cells must be a non-negative integer" in str(exc)
    else:
        raise AssertionError("expected invalid pml_cells to fail closed")


def test_first_principles_3d_runner_carries_startup_policy_from_package_deck() -> None:
    deck = minimal_engineering_deck(n_steps=1, shape=(6, 6, 10))
    result = run_first_principles_3d_deck(deck)

    startup = result.telemetry["startup"]
    assert startup["mode"] == "source_backed_end_rundown_sheath"
    assert startup["can_support_whole_shot_acceptance"] is False
    assert startup["status"] == "blocked_startup_bvp_packet_not_available"
    assert "breakdown_model" in startup["declared_startup_missing_channels"]
    assert "breakdown_or_flashover_model" in startup["missing_acceptance_channels"]
    assert (
        "candidate_source_backed_end_rundown_sheath_profile"
        in startup["candidate_input_channels"]
    )
    assert (
        "candidate_source_backed_sheath_velocity_distribution"
        in startup["candidate_input_channels"]
    )
    loading = result.telemetry["pic_particle_loading"]
    profile = loading["initial_plasma_profile"]
    assert profile["status"] == (
        "candidate_source_backed_annular_sheath_profile_not_validation"
    )
    assert profile["profile_type"] == "annular_axial_sheath"
    assert "background_prefill" in profile["applied_regions"]
    assert "preaccelerated_current_sheath" in profile["applied_regions"]
    assert loading["initial_ion_density_m3"] is None
    assert loading["initial_ion_density_max_m3"] == 3.3e23
    assert profile["max_abs_plasma_drift_m_s"] == 1.1e5
    assert result.manifest["metadata"]["deck"]["startup"] == startup


def test_first_principles_runner_rejects_seeded_startup_for_acceptance() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "seeded_layer",
            "startup_evidence_status": "reviewed",
            "startup_can_support_whole_shot_acceptance": False,
            "startup_missing_channels": (),
        }
    )

    startup = result.telemetry["startup"]
    assert startup["status"] == "rejected_startup_mode_for_first_principles"
    assert startup["startup_mode_class"] == "rejected_for_accepted_claims"
    assert startup["startup_mode_status"]["seeded_layer"]["status"] == (
        "rejected_for_accepted_first_principles_claims"
    )
    assert startup["startup_mode_status"]["seeded_layer"]["decision"] == (
        "must_fail_acceptance_gate"
    )
    assert startup["acceptance_gate"].startswith(
        "engineering_end_rundown_seeded_or_text_startup_cannot_support"
    )
    assert startup["negative_test_policy"]["seeded_layer_rejection_required"] is True
    assert startup["whole_shot_startup_blocked"] is True
    assert startup["can_support_first_principles_acceptance"] is False
    assert (
        result.telemetry["certificate_gate"]["upstream_packet_statuses"]["startup_bvp"]
        == "rejected_startup_mode_for_first_principles"
    )


def test_first_principles_runner_marks_pf1000_akel_same_scope_as_blocked() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "device_name": "PF1000 Akel shot 12581 reference candidate",
            "validation_scope": "pf1000_akel_16kv_1p2torr_shot_12581",
            "apply_circuit_boundary": False,
        }
    )

    packet = result.telemetry["same_scope_source"]
    assert packet["declared_scope"] == "pf1000_akel_16kv_1p2torr_shot_12581"
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert "neutron_scalar_yield" in packet["text_supported_reference_channels"]
    assert (
        "device_geometry_and_electrode_dimensions"
        in packet["text_supported_reference_channels"]
    )
    assert (
        packet["same_scope_channel_status"]["device_geometry_and_electrode_dimensions"]
        == "text_supported_reference_only_not_acceptance"
    )
    assert "neutron_spectrum" in packet["missing_acceptance_channels"]
    assert (
        "cross_scope_transfer_rule_or_rejection_tests"
        in packet["missing_acceptance_channels"]
    )
    assert packet["cross_scope_policy"]["can_use_other_scope_for_acceptance"] is False
    assert packet["cross_scope_policy"]["other_scope_sources_usable_for"] == (
        "requirements_or_schema_only"
    )
    assert (
        packet["same_scope_target_policy"][
            "text_supported_channels_can_support_acceptance"
        ]
        is False
    )
    assert packet["acceptance_gate"].startswith(
        "text_supported_pf1000_akel_scalars_and_other_scope_diagnostics"
    )
    assert (
        packet["negative_test_policy"][
            "other_scope_diagnostic_promotion_rejection_required"
        ]
        is True
    )
    assert packet["decision"] == "do_not_promote_whole_shot_first_principles_claim"

    waveform_phase = result.telemetry["waveform_phase"]
    assert waveform_phase["status"] == "blocked_waveform_phase_packet_not_available"
    assert (
        "breakdown_to_derivative_dip_time"
        in waveform_phase["text_supported_reference_channels"]
    )
    assert (
        "pinch_duration_scalar" in waveform_phase["text_supported_reference_channels"]
    )
    assert (
        "accepted_digitized_current_waveform"
        in waveform_phase["missing_acceptance_channels"]
    )
    assert (
        waveform_phase["draft_digitization_packet_status"]["independent_review_count"]
        == 0
    )
    assert (
        waveform_phase["waveform_phase_target_policy"][
            "draft_digitization_can_support_acceptance"
        ]
        is False
    )
    assert (
        waveform_phase["negative_test_policy"][
            "draft_waveform_promotion_rejection_required"
        ]
        is True
    )
    assert waveform_phase["acceptance_gate"].startswith(
        "draft_or_text_waveform_evidence_cannot_support_validation"
    )
    assert waveform_phase["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    spatial = result.telemetry["spatial_field_temperature"]
    assert spatial["status"] == (
        "blocked_spatial_field_temperature_packet_not_available"
    )
    assert (
        "lee_output_maximum_pinch_density_scalar"
        in spatial["text_supported_reference_channels"]
    )
    assert (
        "lee_output_maximum_pinch_density_scalar"
        in spatial["text_supported_not_acceptance_channels"]
    )
    assert (
        "accepted_same_scope_magnetic_field_history"
        in spatial["missing_acceptance_channels"]
    )
    assert spatial["cross_scope_policy"]["other_scope_sources_usable_for"] == (
        "requirements_or_schema_only"
    )
    assert spatial["acceptance_gate"].startswith(
        "lee_output_scalars_and_other_scope_diagnostics_cannot_support"
    )
    assert spatial["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    neutron = result.telemetry["neutron_authority"]
    assert neutron["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert (
        "measured_scalar_yield_shot_12581"
        in neutron["text_supported_reference_channels"]
    )
    assert (
        "measured_scalar_yield_shot_12581"
        in neutron["text_supported_not_acceptance_channels"]
    )
    assert (
        "silver_activation_total_yield_measurement"
        in neutron["text_supported_reference_channels"]
    )
    assert (
        "accepted_beam_target_yield_history" in neutron["missing_acceptance_channels"]
    )
    assert "neutron_anisotropy_angular_yield" in neutron["missing_acceptance_channels"]
    assert neutron["mechanism_separation_policy"][
        "scalar_yield_agreement_usable_for"
    ] == ("baseline_comparison_only")
    assert neutron["acceptance_gate"].startswith(
        "scalar_yield_reduced_model_text_and_other_scope_neutron_diagnostics"
    )
    assert neutron["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    comparator = result.telemetry["comparator_uq"]
    assert comparator["status"] == "blocked_comparator_uq_matrix_not_available"
    assert (
        "scalar_neutron_yield_uncertainty_text"
        in comparator["text_supported_reference_channels"]
    )
    assert (
        "scalar_neutron_yield_uncertainty_text"
        in comparator["text_supported_not_acceptance_channels"]
    )
    assert (
        "channel_timing_uncertainty_text"
        in comparator["text_supported_reference_channels"]
    )
    assert (
        "comparator_metric_by_observable" in comparator["missing_acceptance_channels"]
    )
    assert comparator["cross_scope_policy"]["other_scope_sources_usable_for"] == (
        "requirements_or_schema_only"
    )
    assert comparator["acceptance_gate"].startswith(
        "text_uncertainty_other_scope_sensitivity_and_partial_targets"
    )
    assert comparator["upstream_packet_statuses"]["same_scope_source"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    numerical = result.telemetry["numerical_fidelity"]
    assert numerical["status"] == "blocked_numerical_fidelity_packet_not_available"
    assert "maxwell_yee_courant_packet" in numerical["missing_acceptance_channels"]
    assert "backend_precision_parity_packet" in numerical["missing_acceptance_channels"]
    assert numerical["numerical_channel_status"]["backend_precision_parity_packet"] == (
        "missing_or_blocked"
    )
    assert (
        numerical["test_surface_status"]["finite_volume_shock_behavior"]["status"]
        == "legacy_candidate_component_coverage_not_acceptance"
    )
    assert (
        numerical["test_surface_status"]["limiter_zero_acceptance"]["status"]
        == "candidate_runtime_limiter_zero_observed_not_acceptance"
    )
    assert (
        "runtime_experimental_limiter_zero_probe"
        in numerical["test_surface_status"]["limiter_zero_acceptance"][
            "candidate_artifacts"
        ]
    )
    assert numerical["acceptance_gate"].startswith(
        "candidate_component_tests_and_runtime_diagnostics_cannot_support"
    )
    assert (
        numerical["upstream_acceptance_gate"]["blocking_upstream_packets"][
            "limiter_readiness"
        ]
        == "blocked_limiter_readiness_packet_not_available"
    )
    assert numerical["upstream_packet_statuses"]["power_port"] == (
        "candidate_engineering_power_port_not_validation"
    )

    limiter = result.telemetry["limiter_readiness"]
    assert limiter["status"] == "blocked_limiter_readiness_packet_not_available"
    assert "full_horizon_run_manifest" in limiter["missing_acceptance_channels"]
    assert (
        "backend_precision_fallback_inventory" in limiter["missing_acceptance_channels"]
    )
    assert limiter["negative_test_policy"]["hidden_limiter_regression_required"] is True
    assert limiter["acceptance_gate"].startswith(
        "candidate_runtime_telemetry_cannot_support_limiter_zero_acceptance"
    )

    certificate = result.telemetry["certificate_gate"]
    assert certificate["status"] == (
        "blocked_first_principles_certificate_not_available"
    )
    assert certificate["release_label"] == (
        "engineering_candidate_not_releasable_for_first_principles_claim"
    )
    assert (
        certificate["acceptance_policy"][
            "draft_candidate_blocked_or_rejected_packets_block_release"
        ]
        is True
    )
    assert "same_scope_source" in certificate["upstream_certificate_blockers"]
    assert "comparator_uq_packet_accepted" in certificate["missing_acceptance_channels"]
    assert (
        certificate["upstream_packet_acceptance_matrix"][
            "same_scope_source_packet_accepted"
        ]["decision"]
        == "missing_or_blocking_upstream_packet"
    )

    generalization = result.telemetry["generalization"]
    assert generalization["status"] == (
        "blocked_generalized_dpf_machine_path_not_available"
    )
    candidate_scope_ids = {
        scope["scope_id"] for scope in generalization["candidate_second_scopes"]
    }
    assert "pf1000_full_energy_anisotropy_450_500kj_3p5torr" in candidate_scope_ids
    assert "soto2010_cchen_pf400j_pf50j_speed_nanofocus_matrix" in candidate_scope_ids
    assert "llnl_180ka_kinetic_or_hybrid_reference" in candidate_scope_ids
    candidate_decisions = {
        scope["scope_id"]: scope
        for scope in generalization["candidate_second_scope_decisions"]
    }
    assert (
        candidate_decisions["soto2010_cchen_pf400j_pf50j_speed_nanofocus_matrix"][
            "decision"
        ]
        == "candidate_requirement_material_not_acceptance"
    )
    assert (
        candidate_decisions["llnl_180ka_kinetic_or_hybrid_reference"]["decision"]
        == "candidate_requirement_material_not_acceptance"
    )
    assert (
        candidate_decisions["llnl_180ka_kinetic_or_hybrid_reference"][
            "must_write_independent_certificate"
        ]
        is True
    )
    assert (
        "no_hidden_pf1000_akel_assumptions"
        in generalization["missing_acceptance_channels"]
    )
    assert generalization["upstream_acceptance_gate"]["status"] == (
        "blocked_by_primary_scope_packets"
    )
    assert generalization["upstream_packet_statuses"]["certificate_gate"] == (
        "blocked_first_principles_certificate_not_available"
    )


# --- WP-1 / SSR-006 power-port negative tests ----------------------------


def test_wp1_sign_reversal_breaks_active_port_vs_j_dot_e_residual() -> None:
    """Negative test: a sign-flipped U_DPF must make active-port work and
    integrated J.E DISAGREE (residual non-zero); neither path may claim
    acceptance.  Auluck Eq.1 fixes U_DPF = -J.E_integral / I; reversed sign
    is wrong physics and the ledger must expose it."""
    correct = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {"current_A": 2.0, "udpf_V": 5.0},
            }
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -5.0,
            "cumulative_j_dot_e_step_count": 1,
            "cumulative_active_port_work_J": 5.0,
            "cumulative_active_port_step_count": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -10.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    }
                }
            },
        },
    )
    reversed_sign = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {"current_A": 2.0, "udpf_V": -5.0},
            }
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -5.0,
            "cumulative_j_dot_e_step_count": 1,
            "cumulative_active_port_work_J": -5.0,
            "cumulative_active_port_step_count": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -10.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    }
                }
            },
        },
    )
    rb_ok = correct["candidate_power_residual_budget"]
    rb_bad = reversed_sign["candidate_power_residual_budget"]
    # Correct sign: active_port + integrated_j_dot_e cancels to 0.
    assert rb_ok["active_port_plus_integrated_j_dot_e_work_J"] == pytest.approx(0.0)
    # Reversed sign: residual is non-zero and large.
    assert abs(rb_bad["active_port_plus_integrated_j_dot_e_work_J"]) > 1.0
    # Neither path may claim acceptance.
    assert correct["can_support_first_principles_acceptance"] is False
    assert reversed_sign["can_support_first_principles_acceptance"] is False
    assert reversed_sign["active_load_decision"]["can_support_power_port_acceptance"] is False


def test_wp1_domain_corruption_is_flagged_by_domain_review() -> None:
    """Negative test: a J.E integral from an undeclared/unmasked domain must
    not pass the domain review; the corrupted domain string must be surfaced
    verbatim, and the interface domain must stay not_declared."""
    corrupt = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {"current_A": 3.0, "udpf_V": 4.0},
            }
        },
        simulation_telemetry={
            "dt_s": 0.25,
            "n_steps_completed": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -12.0,
                        "domain": "unmasked_full_grid_including_source_interface",
                    }
                }
            },
        },
    )
    dom = corrupt["stage0_packet_scaffolds"]["power_port_domain_review"]
    assert dom["status"] == "blocked_domain_packet_not_available"
    assert dom["can_support_power_port_acceptance"] is False
    # The corrupted domain string is surfaced verbatim, not silently normalized.
    assert dom["declared_runtime_domain"] == (
        "unmasked_full_grid_including_source_interface"
    )
    assert corrupt["interface_surface_or_volume_domain"] == "not_declared"


def test_wp1_time_centering_downgrade_stays_non_accepted() -> None:
    """Negative test: begin-step (uncentered) time-centering must keep the
    time-centering review non-accepted; the runtime metadata only carries a
    begin-step candidate."""
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {"current_A": 2.0, "udpf_V": 5.0},
            }
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -10.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    }
                }
            },
        },
    )
    tc = packet["stage0_packet_scaffolds"]["power_port_time_centering_review"]
    assert tc["status"] == "candidate_time_centering_packet_not_validation"
    assert tc["can_support_power_port_acceptance"] is False
    assert packet["time_centering"] == "candidate_runner_step_metadata_only"
    assert tc["runtime_time_centering"] == "begin_step_or_retained_step_metadata"


def test_wp1_low_current_p_over_i_singularity_unit_blocked() -> None:
    """Negative test (unit): at |I| <= min_current_A the P/I feedback must
    fall back to input sequence with the low-current tag; 1/I is never taken."""
    udpf, source = _udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e",
        input_udpf_V=7.0,
        lagged_field_work={"j_dot_e_power_W": -10.0},
        current_A=0.0,
        min_current_A=1.0,
    )
    assert udpf == pytest.approx(7.0)
    assert source == "input_sequence_fallback_low_current"


def test_wp1_sigma_line_voltage_is_rejected_as_driver() -> None:
    """Negative test: the Sigma/quasi-TEM line-voltage operator must be
    DEFERRED everywhere and must never be an accepted/primary circuit driver."""
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                "circuit_step": {"current_A": 2.0, "udpf_V": 5.0},
            }
        },
        simulation_telemetry={
            "dt_s": 0.5,
            "n_steps_completed": 1,
            "last_step": {
                "field_step": {
                    "field_work": {
                        "j_dot_e_power_W": -10.0,
                        "domain": "resolved_plasma_current_carrying_cells",
                    }
                }
            },
        },
    )
    sigma_op = packet["sigma_quasi_tem_line_voltage_operator"]
    assert sigma_op["status"] == "deferred_sigma_quasi_tem_driver_not_source_verified"
    assert sigma_op["allowed_runtime_use"] == "exploratory_diagnostic_only"
    assert sigma_op["disallowed_runtime_use"] == "accepted_or_primary_circuit_driver"
    assert sigma_op["can_support_power_port_acceptance"] is False
    cmp_op = packet["power_port_operator_comparison"]
    assert cmp_op["decision"] == (
        "do_not_replace_active_driver_with_sigma_line_voltage"
    )
    assert (
        cmp_op["operators"]["sigma_quasi_tem_line_voltage"]["source_status"]
        == "not_verified_in_local_dpf_source"
    )
    assert cmp_op["operators"]["sigma_quasi_tem_line_voltage"]["can_be_accepted_now"] is False
    from dpf.first_principles.deck import FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES
    assert not any("sigma" in m for m in FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES)


def test_wp1_auluck_mode_first_step_falls_back_without_singularity() -> None:
    """Negative test (G2): on the first step lagged_field_work is None; the
    Auluck mode must fall back to the input sequence with the first-step tag,
    never computing 0/0."""
    udpf, source = _udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e",
        input_udpf_V=3.0,
        lagged_field_work=None,
        current_A=0.0,
        min_current_A=1.0,
    )
    assert udpf == pytest.approx(3.0)
    assert source == "input_sequence_fallback_first_step"


def test_wp1_default_circuit_udpf_mode_does_not_silently_take_p_over_i() -> None:
    """Negative test (G6): a deck with no explicit circuit_udpf_mode must use
    the conservative default, which must NOT be the source-sign Auluck mode.
    Guards against a 1/I path being inherited without disclosure."""
    from dpf.first_principles.deck import ClosurePolicy
    default_mode = ClosurePolicy().circuit_udpf_mode
    assert default_mode == "lagged_volume_j_dot_e"
    assert default_mode != "lagged_auluck_volume_j_dot_e"


def test_wp1_auluck_mode_does_not_clip_negative_j_dot_e() -> None:
    """Hardening test: Auluck mode must pass signed negative J.E straight
    through as -power_W/I; it must NOT route to the blocked-clip fallback tag.
    Asserts Auluck 2021 Eq.1 sign contract (KR auluck-2021:173-200)."""
    udpf, source = _udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e",
        input_udpf_V=99.0,
        lagged_field_work={"j_dot_e_power_W": -8.0},
        current_A=4.0,
        min_current_A=1.0,
    )
    assert udpf == pytest.approx(2.0)   # -(-8.0) / 4.0
    assert source == "candidate_lagged_auluck_volume_j_dot_e"
    assert "negative" not in source


def test_first_principles_runner_propagates_long_run_history_controls() -> None:
    result = run_first_principles_3d_deck(
        {
            "n_steps": 5,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "apply_circuit_boundary": False,
            "history_stride": 2,
            "max_step_results": 1,
            "target_time_s": 3.0e-13,
        }
    )

    simulation = result.telemetry["simulation"]
    assert simulation["n_steps_completed"] == 3
    assert simulation["stop_reason"] == "target_time_reached"
    assert simulation["duration_request_satisfied"] is True
    assert simulation["retained_step_result_count"] == 1
    assert simulation["history_stride"] == 2
    assert simulation["max_step_results"] == 1
    assert simulation["history_summary"][-1]["step_index"] == 2
    assert result.telemetry["n_steps"] == 5
    assert result.telemetry["n_steps_completed"] == 3
    assert result.manifest["metadata"]["deck"]["first_principles_3d"][
        "history_stride"
    ] == 2
    assert result.conservation_telemetry["n_steps"] == 3


# ---------------------------------------------------------------------------
# WS-C: Package-Native 3-D Acceptance Contract — negative-test suite
#
# Spec: docs/SPRINT7_FIRST_PRINCIPLES_RUNTIME_CONTRACT_INSTRUCTIONS_2026_05_20.md
# §Workstream C Audit
# Gate under test: dpf.validation.hybrid_pic_3d.hybrid_pic_3d_readiness_status
# ---------------------------------------------------------------------------

def _all_accepted_evidence() -> dict:
    """Build a minimal fully-accepted evidence dict for every capability."""
    from dpf.validation.hybrid_pic_3d import HYBRID_PIC_3D_CAPABILITY_IDS
    return {
        cap_id: {"passed": True, "status": "accepted"}
        for cap_id in HYBRID_PIC_3D_CAPABILITY_IDS
    }


def test_wsc_candidate_record_cannot_produce_acceptance() -> None:
    """WS-C negative: a candidate-only evidence record (passed=True but
    status='candidate') must never unlock can_support_first_principles_acceptance.
    Guardrail: candidate telemetry is visible but must be rejected."""
    from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

    evidence = _all_accepted_evidence()
    # Downgrade two capabilities to candidate status — not in _ACCEPTED_STATUSES
    for cap_id in list(evidence)[:2]:
        evidence[cap_id] = {"passed": True, "status": "candidate"}

    result = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "cartesian_3d",
        "hybrid_pic_3d_evidence": evidence,
    })

    assert result["can_support_first_principles_acceptance"] is False
    assert result["status"] == "blocked"
    # Both downgraded capabilities must appear in missing_capabilities
    for cap_id in list(_all_accepted_evidence())[:2]:
        assert cap_id in result["missing_capabilities"]


def test_wsc_missing_top_level_contract_key_cannot_produce_acceptance() -> None:
    """WS-C negative: a result dict that omits hybrid_pic_3d_evidence entirely
    must block acceptance — the gate must fail closed, not open."""
    from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

    # No hybrid_pic_3d_evidence key at all
    result = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "cartesian_3d",
    })

    assert result["can_support_first_principles_acceptance"] is False
    assert result["status"] == "blocked"
    # With no evidence supplied every capability must be missing
    from dpf.validation.hybrid_pic_3d import HYBRID_PIC_3D_CAPABILITY_IDS
    for cap_id in HYBRID_PIC_3D_CAPABILITY_IDS:
        assert cap_id in result["missing_capabilities"]


def test_wsc_wrong_backend_label_cannot_produce_acceptance() -> None:
    """WS-C negative: geometry_dimensionality='cylindrical_2d' must block
    acceptance even when every PIC capability record is fully accepted.
    Guards against a 2-D/cylindrical run masquerading as a 3-D acceptance."""
    from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

    result = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "cylindrical_2d",
        "hybrid_pic_3d_evidence": _all_accepted_evidence(),
    })

    assert result["can_support_first_principles_acceptance"] is False
    assert result["status"] == "blocked"
    assert "explicit_3d_geometry" in result["missing_capabilities"]
    # Dimensionality must be faithfully round-tripped
    assert result["geometry_dimensionality"] == "cylindrical_2d"


def test_wsc_missing_same_scope_3d_validation_cannot_produce_acceptance() -> None:
    """WS-C negative: omitting the same_scope_3d_validation_packet capability
    from accepted evidence must block acceptance even when all other 13
    capabilities are fully accepted — same-scope 3-D validation is not optional."""
    from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

    evidence = _all_accepted_evidence()
    # Remove the same-scope 3-D validation capability
    del evidence["same_scope_3d_validation_packet"]

    result = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "cartesian_3d",
        "hybrid_pic_3d_evidence": evidence,
    })

    assert result["can_support_first_principles_acceptance"] is False
    assert result["status"] == "blocked"
    assert "same_scope_3d_validation_packet" in result["missing_capabilities"]
