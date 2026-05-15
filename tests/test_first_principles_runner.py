import numpy as np

from dpf.fields import CircuitMagneticBoundaryDrive, Maxwell3DGrid
from dpf.first_principles import run_first_principles_3d_deck as package_runner
from dpf.first_principles.runner import (
    ENGINEERING_CANDIDATE_STATUS,
    HybridEMPicFluidRun,
    run_first_principles_3d_deck,
)


def test_run_first_principles_3d_deck_returns_candidate_manifest_and_telemetry() -> None:
    result = run_first_principles_3d_deck({
        "n_steps": 2,
        "grid_shape": (4, 4, 4),
        "dt_s": 1.0e-13,
        "apply_circuit_boundary": True,
    })

    assert result.status == ENGINEERING_CANDIDATE_STATUS
    assert result.reduced_models_used is False
    assert result.can_support_first_principles_acceptance is False
    assert result.result.telemetry.n_steps_completed == 2
    assert result.result.telemetry.circuit is not None
    assert result.result.telemetry.last_step is not None
    assert result.result.telemetry.last_step["electron_energy"]["status"] == (
        "candidate_engineering_electron_energy_closure"
    )
    assert result.result.telemetry.last_step["kinetic_yield"]["status"] == (
        "candidate_engineering_kinetic_yield_history"
    )
    assert result.telemetry["simulation"]["status"] == (
        "candidate_engineering_3d_hybrid_pic_simulation"
    )
    assert result.manifest["validation_status"] == "not_validation"
    assert result.manifest["scientific_status"] == ENGINEERING_CANDIDATE_STATUS
    assert result.manifest["reduced_models_used"] is False
    assert result.telemetry["startup"]["mode"] == "source_backed_end_rundown_sheath"
    assert result.telemetry["startup"]["status"] == (
        "blocked_startup_bvp_packet_not_available"
    )
    assert result.telemetry["startup"]["decision"] == (
        "do_not_promote_startup_to_whole_shot_first_principles"
    )
    assert result.telemetry["startup"]["whole_shot_startup_blocked"] is True
    assert "breakdown_or_flashover_model" in result.telemetry["startup"][
        "missing_acceptance_channels"
    ]
    assert "surface_breakdown_bvp" in result.telemetry["startup"]["accepted_modes"]
    assert result.telemetry["limiter_readiness"]["status"] == (
        "blocked_limiter_readiness_packet_not_available"
    )
    assert result.telemetry["limiter_readiness"][
        "can_support_limiter_zero_acceptance"
    ] is False
    assert "active_path_limiter_inventory" in result.telemetry[
        "limiter_readiness"
    ]["missing_acceptance_channels"]
    assert "zero_acceptance_blocker_full_run" in result.telemetry[
        "limiter_readiness"
    ]["missing_acceptance_channels"]
    assert "candidate_finite_conservation_snapshot" in result.telemetry[
        "limiter_readiness"
    ]["candidate_runtime_channels"]
    assert result.telemetry["power_port"]["authority_contract"] == (
        "field_power_required"
    )
    assert result.telemetry["power_port"]["can_support_first_principles_acceptance"] is False
    assert result.telemetry["power_port"]["accepted_load_power_source"] == "none"
    assert result.telemetry["power_port"]["diagnostic_field_inductance_H"] > 0.0
    assert result.telemetry["power_port"]["magnetic_energy_inductance_authority"] == (
        "diagnostic_only_not_circuit_load"
    )
    assert "candidate_diagnostic_field_inductance" in result.telemetry["power_port"][
        "candidate_runtime_channels"
    ]
    assert result.telemetry["power_port"]["power_port_step_records"][0][
        "residual_interpretation"
    ] == "tracked_energy_delta_not_accepted_power_port_residual"
    assert result.telemetry["power_port"]["power_port_step_records"][0][
        "can_support_first_principles_acceptance"
    ] is False
    assert "poynting_or_j_dot_e_power_integral" in result.telemetry["power_port"][
        "missing_acceptance_channels"
    ]
    assert "startup_handoff_interval" in result.telemetry["power_port"][
        "missing_acceptance_channels"
    ]
    assert result.telemetry["dimensionality_handoff"]["geometry_dimensionality"] == "3d"
    assert result.telemetry["dimensionality_handoff"]["decision"] == (
        "do_not_claim_unrestricted_whole_shot_dimensionality_authority"
    )
    assert result.telemetry["dimensionality_handoff"]["allowed_claim"] == (
        "engineering_3d_hybrid_em_pic_fluid_candidate_only"
    )
    assert result.telemetry["dimensionality_handoff"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "mhd_to_kinetic_handoff_state" in result.telemetry["dimensionality_handoff"][
        "missing_acceptance_channels"
    ]
    assert "candidate_true_3d_grid" in result.telemetry["dimensionality_handoff"][
        "candidate_runtime_channels"
    ]
    assert "candidate_separate_electron_energy_scaffold" in result.telemetry[
        "dimensionality_handoff"
    ]["candidate_runtime_channels"]
    assert "source_hybrid_reference_axisymmetric_m0_not_full_3d" in result.telemetry[
        "dimensionality_handoff"
    ]["source_model_limitations"]
    assert "fully_kinetic_pinch_claim" in {
        mode["mode"] for mode in result.telemetry["dimensionality_handoff"]["claim_modes"]
    }
    assert result.telemetry["physics_closure"]["status"] == (
        "candidate_engineering_closure_packet_not_validation"
    )
    assert result.telemetry["physics_closure"]["decision"] == (
        "do_not_promote_without_complete_physics_closure_matrix"
    )
    assert result.telemetry["physics_closure"]["can_support_first_principles_acceptance"] is False
    assert "source_equations_or_bound" in result.telemetry["physics_closure"][
        "required_packet_channels"
    ]
    assert "candidate_electron_energy_scaffold" in result.telemetry["physics_closure"][
        "candidate_runtime_channels"
    ]
    assert "candidate_kinetic_yield_history" in result.telemetry["physics_closure"][
        "candidate_runtime_channels"
    ]
    assert "single_two_temperature_energy" in result.telemetry["physics_closure"][
        "active_candidate_closures"
    ]
    assert "eos_thermodynamics" in result.telemetry["physics_closure"][
        "missing_or_unaccepted_effects"
    ]
    assert result.telemetry["physics_closure"]["effects"]["eos_thermodynamics"][
        "required_packet_channels"
    ] == result.telemetry["physics_closure"]["required_packet_channels"]
    assert result.telemetry["physics_closure"]["closure_matrix_status_by_effect"][
        "radiation_losses"
    ] == "blocked"
    assert result.telemetry["physics_closure"]["effects"]["beam_target_coupling"][
        "status"
    ] == "candidate"
    assert result.telemetry["same_scope_source"]["status"] == (
        "blocked_same_scope_source_packet_not_available"
    )
    assert result.telemetry["same_scope_source"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "accepted_digitized_current_waveform" in result.telemetry[
        "same_scope_source"
    ]["missing_acceptance_channels"]
    assert result.telemetry["waveform_phase"]["status"] == (
        "blocked_waveform_phase_packet_not_available"
    )
    assert result.telemetry["waveform_phase"][
        "can_support_first_principles_acceptance"
    ] is False
    assert result.telemetry["waveform_phase"]["draft_digitization_packet_status"][
        "accepted_for_validation"
    ] is False
    assert result.telemetry["waveform_phase"]["draft_digitization_packet_status"][
        "review_status"
    ] == "draft"
    assert "review_status_accepted" in result.telemetry["waveform_phase"][
        "required_review_channels"
    ]
    assert "accepted_current_derivative_or_dip_trace" in result.telemetry[
        "waveform_phase"
    ]["missing_acceptance_channels"]
    assert result.telemetry["waveform_phase"]["waveform_phase_channel_status"][
        "accepted_digitized_current_waveform"
    ] == "missing_or_blocked"
    assert result.telemetry["spatial_field_temperature"]["status"] == (
        "blocked_spatial_field_temperature_packet_not_available"
    )
    assert result.telemetry["spatial_field_temperature"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "accepted_same_scope_density_history" in result.telemetry[
        "spatial_field_temperature"
    ]["missing_acceptance_channels"]
    assert result.telemetry["spatial_field_temperature"][
        "spatial_field_temperature_channel_status"
    ]["accepted_same_scope_density_history"] == "missing_or_blocked"
    assert result.telemetry["spatial_field_temperature"]["cross_scope_policy"][
        "can_use_other_scope_for_acceptance"
    ] is False
    assert result.telemetry["neutron_authority"]["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert result.telemetry["neutron_authority"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "mechanism_separated_yield_channels" in result.telemetry[
        "neutron_authority"
    ]["missing_acceptance_channels"]
    assert "candidate_pic_ion_neutron_yield_history" in result.telemetry[
        "neutron_authority"
    ]["candidate_runtime_channels"]
    assert result.telemetry["comparator_uq"]["status"] == (
        "blocked_comparator_uq_matrix_not_available"
    )
    assert result.telemetry["comparator_uq"][
        "can_support_first_principles_acceptance"
    ] is False
    assert "output_field_mapping_by_observable" in result.telemetry[
        "comparator_uq"
    ]["missing_acceptance_channels"]
    assert "pass_fail_rule_by_observable" in result.telemetry["comparator_uq"][
        "missing_acceptance_channels"
    ]
    assert result.telemetry["comparator_uq"]["upstream_packet_statuses"][
        "neutron_authority"
    ] == "blocked_mechanism_separated_neutron_authority_not_available"
    assert result.telemetry["numerical_fidelity"]["status"] == (
        "blocked_numerical_fidelity_packet_not_available"
    )
    assert result.telemetry["numerical_fidelity"][
        "can_support_numerical_acceptance"
    ] is False
    assert "mesh_timestep_convergence_packet" in result.telemetry[
        "numerical_fidelity"
    ]["missing_acceptance_channels"]
    assert "limiter_zero_packet" in result.telemetry["numerical_fidelity"][
        "missing_acceptance_channels"
    ]
    assert "candidate_conservation_telemetry" in result.telemetry[
        "numerical_fidelity"
    ]["candidate_runtime_channels"]
    assert result.telemetry["numerical_fidelity"]["upstream_packet_statuses"][
        "limiter_readiness"
    ] == "blocked_limiter_readiness_packet_not_available"
    assert "candidate_divergence_b_diagnostic" in result.telemetry[
        "numerical_fidelity"
    ]["candidate_runtime_channels"]
    assert result.telemetry["certificate_gate"]["status"] == (
        "blocked_first_principles_certificate_not_available"
    )
    assert result.telemetry["certificate_gate"]["can_write_accepted_certificate"] is False
    assert "run_manifest_hash" in result.telemetry["certificate_gate"][
        "missing_acceptance_channels"
    ]
    assert "negative_test_missing_uq" in result.telemetry["certificate_gate"][
        "missing_acceptance_channels"
    ]
    assert result.telemetry["certificate_gate"]["upstream_packet_statuses"][
        "comparator_uq"
    ] == "blocked_comparator_uq_matrix_not_available"
    assert result.telemetry["certificate_gate"]["upstream_packet_statuses"][
        "numerical_fidelity"
    ] == "blocked_numerical_fidelity_packet_not_available"
    assert result.telemetry["certificate_gate"]["upstream_packet_statuses"][
        "limiter_readiness"
    ] == "blocked_limiter_readiness_packet_not_available"
    assert result.telemetry["generalization"]["status"] == (
        "blocked_generalized_dpf_machine_path_not_available"
    )
    assert result.telemetry["generalization"][
        "can_claim_generalized_dpf_machine"
    ] is False
    assert "accepted_primary_scope_certificate" in result.telemetry[
        "generalization"
    ]["missing_acceptance_channels"]
    assert "second_scope_certificate" in result.telemetry["generalization"][
        "missing_acceptance_channels"
    ]
    assert result.telemetry["generalization"]["upstream_packet_statuses"][
        "certificate_gate"
    ] == "blocked_first_principles_certificate_not_available"
    assert result.telemetry["generalization"]["upstream_packet_statuses"][
        "numerical_fidelity"
    ] == "blocked_numerical_fidelity_packet_not_available"
    assert result.telemetry["generalization"]["upstream_packet_statuses"][
        "limiter_readiness"
    ] == "blocked_limiter_readiness_packet_not_available"
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
    assert result.manifest["candidate_evidence"]["first_principles_3d_runner"][
        "status"
    ] == ENGINEERING_CANDIDATE_STATUS
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
    assert result.manifest["candidate_evidence"][
        "spatial_field_temperature_packet"
    ] == result.telemetry["spatial_field_temperature"]
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
    assert "mhd_to_kinetic_handoff_state" in result.validation_packet[
        "dimensionality_handoff_missing_acceptance_channels"
    ]
    assert "beam_target_neutron_authority" in result.validation_packet[
        "dimensionality_handoff_blocked_observables"
    ]
    assert result.validation_packet[
        "dimensionality_handoff_can_support_first_principles_acceptance"
    ] is False
    assert result.validation_packet["can_support_first_principles_acceptance"] is False


def test_package_exports_first_principles_3d_runner() -> None:
    assert package_runner is run_first_principles_3d_deck


def test_hybrid_em_pic_fluid_run_can_disable_optional_circuit_boundary() -> None:
    grid = Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    runner = HybridEMPicFluidRun({
        "grid": grid,
        "n_steps": 1,
        "dt_s": 1.0e-13,
        "apply_circuit_boundary": False,
    })

    result = runner.run()

    assert result.status == ENGINEERING_CANDIDATE_STATUS
    assert result.result.circuit is None
    assert result.result.telemetry.circuit is None
    assert result.telemetry["grid_spacing_m"] == [1.0e-3, 1.0e-3, 1.0e-3]
    assert result.telemetry["power_port"]["active_load_relation"] == (
        "no_active_circuit_boundary"
    )
    assert "terminal_current" in result.telemetry["power_port"][
        "missing_acceptance_channels"
    ]
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


def test_first_principles_3d_runner_rejects_invalid_step_count() -> None:
    runner = HybridEMPicFluidRun({"n_steps": 0})

    try:
        runner.run()
    except ValueError as exc:
        assert "n_steps must be a positive integer" in str(exc)
    else:
        raise AssertionError("expected invalid n_steps to fail closed")


def test_first_principles_3d_runner_carries_startup_policy_from_package_deck() -> None:
    from dpf.first_principles import minimal_engineering_deck

    deck = minimal_engineering_deck(n_steps=1, shape=(4, 4, 4))
    result = run_first_principles_3d_deck(deck)

    startup = result.telemetry["startup"]
    assert startup["mode"] == "source_backed_end_rundown_sheath"
    assert startup["can_support_whole_shot_acceptance"] is False
    assert startup["status"] == "blocked_startup_bvp_packet_not_available"
    assert "breakdown_model" in startup["declared_startup_missing_channels"]
    assert "breakdown_or_flashover_model" in startup["missing_acceptance_channels"]
    assert result.manifest["metadata"]["deck"]["startup"] == startup


def test_first_principles_runner_rejects_seeded_startup_for_acceptance() -> None:
    result = run_first_principles_3d_deck({
        "n_steps": 1,
        "grid_shape": (4, 4, 4),
        "dt_s": 1.0e-13,
        "startup_mode": "seeded_layer",
        "startup_evidence_status": "reviewed",
        "startup_can_support_whole_shot_acceptance": False,
        "startup_missing_channels": (),
    })

    startup = result.telemetry["startup"]
    assert startup["status"] == "rejected_startup_mode_for_first_principles"
    assert startup["startup_mode_class"] == "rejected_for_accepted_claims"
    assert startup["whole_shot_startup_blocked"] is True
    assert startup["can_support_first_principles_acceptance"] is False
    assert result.telemetry["certificate_gate"]["upstream_packet_statuses"][
        "startup_bvp"
    ] == "rejected_startup_mode_for_first_principles"


def test_first_principles_runner_marks_pf1000_akel_same_scope_as_blocked() -> None:
    result = run_first_principles_3d_deck({
        "n_steps": 1,
        "grid_shape": (4, 4, 4),
        "dt_s": 1.0e-13,
        "device_name": "PF1000 Akel shot 12581 reference candidate",
        "validation_scope": "pf1000_akel_16kv_1p2torr_shot_12581",
        "apply_circuit_boundary": False,
    })

    packet = result.telemetry["same_scope_source"]
    assert packet["declared_scope"] == "pf1000_akel_16kv_1p2torr_shot_12581"
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert "neutron_scalar_yield" in packet["text_supported_reference_channels"]
    assert "device_geometry_and_electrode_dimensions" in packet[
        "text_supported_reference_channels"
    ]
    assert packet["same_scope_channel_status"][
        "device_geometry_and_electrode_dimensions"
    ] == "text_supported_reference_only_not_acceptance"
    assert "neutron_spectrum" in packet["missing_acceptance_channels"]
    assert "cross_scope_transfer_rule_or_rejection_tests" in packet[
        "missing_acceptance_channels"
    ]
    assert packet["cross_scope_policy"]["can_use_other_scope_for_acceptance"] is False
    assert packet["cross_scope_policy"]["other_scope_sources_usable_for"] == (
        "requirements_or_schema_only"
    )
    assert packet["decision"] == "do_not_promote_whole_shot_first_principles_claim"

    waveform_phase = result.telemetry["waveform_phase"]
    assert waveform_phase["status"] == "blocked_waveform_phase_packet_not_available"
    assert "breakdown_to_derivative_dip_time" in waveform_phase[
        "text_supported_reference_channels"
    ]
    assert "pinch_duration_scalar" in waveform_phase["text_supported_reference_channels"]
    assert "accepted_digitized_current_waveform" in waveform_phase[
        "missing_acceptance_channels"
    ]
    assert waveform_phase["draft_digitization_packet_status"]["independent_review_count"] == 0
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
    assert "lee_output_maximum_pinch_density_scalar" in spatial[
        "text_supported_reference_channels"
    ]
    assert "lee_output_maximum_pinch_density_scalar" in spatial[
        "text_supported_not_acceptance_channels"
    ]
    assert "accepted_same_scope_magnetic_field_history" in spatial[
        "missing_acceptance_channels"
    ]
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
    assert "measured_scalar_yield_shot_12581" in neutron[
        "text_supported_reference_channels"
    ]
    assert "silver_activation_total_yield_measurement" in neutron[
        "text_supported_reference_channels"
    ]
    assert "accepted_beam_target_yield_history" in neutron[
        "missing_acceptance_channels"
    ]
    assert "neutron_anisotropy_angular_yield" in neutron[
        "missing_acceptance_channels"
    ]
    assert neutron["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    comparator = result.telemetry["comparator_uq"]
    assert comparator["status"] == "blocked_comparator_uq_matrix_not_available"
    assert "scalar_neutron_yield_uncertainty_text" in comparator[
        "text_supported_reference_channels"
    ]
    assert "channel_timing_uncertainty_text" in comparator[
        "text_supported_reference_channels"
    ]
    assert "comparator_metric_by_observable" in comparator[
        "missing_acceptance_channels"
    ]
    assert comparator["upstream_packet_statuses"]["same_scope_source"] == (
        "blocked_same_scope_source_packet_not_available"
    )

    numerical = result.telemetry["numerical_fidelity"]
    assert numerical["status"] == "blocked_numerical_fidelity_packet_not_available"
    assert "maxwell_yee_courant_packet" in numerical["missing_acceptance_channels"]
    assert "backend_precision_parity_packet" in numerical[
        "missing_acceptance_channels"
    ]
    assert numerical["upstream_packet_statuses"]["power_port"] == (
        "candidate_engineering_power_port_not_validation"
    )

    limiter = result.telemetry["limiter_readiness"]
    assert limiter["status"] == "blocked_limiter_readiness_packet_not_available"
    assert "full_horizon_run_manifest" in limiter["missing_acceptance_channels"]
    assert "backend_precision_fallback_inventory" in limiter[
        "missing_acceptance_channels"
    ]

    certificate = result.telemetry["certificate_gate"]
    assert certificate["status"] == (
        "blocked_first_principles_certificate_not_available"
    )
    assert certificate["release_label"] == (
        "engineering_candidate_not_releasable_for_first_principles_claim"
    )
    assert "same_scope_source" in certificate["upstream_certificate_blockers"]
    assert "comparator_uq_packet_accepted" in certificate[
        "missing_acceptance_channels"
    ]

    generalization = result.telemetry["generalization"]
    assert generalization["status"] == (
        "blocked_generalized_dpf_machine_path_not_available"
    )
    candidate_scope_ids = {
        scope["scope_id"] for scope in generalization["candidate_second_scopes"]
    }
    assert "pf1000_full_energy_anisotropy_450_500kj_3p5torr" in candidate_scope_ids
    assert "llnl_180ka_kinetic_or_hybrid_reference" in candidate_scope_ids
    assert "no_hidden_pf1000_akel_assumptions" in generalization[
        "missing_acceptance_channels"
    ]
    assert generalization["upstream_packet_statuses"]["certificate_gate"] == (
        "blocked_first_principles_certificate_not_available"
    )
