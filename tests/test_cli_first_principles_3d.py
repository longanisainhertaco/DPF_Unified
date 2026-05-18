"""CLI coverage for the package-native 3-D first-principles runner."""

from __future__ import annotations

import json

from click.testing import CliRunner


def test_first_principles_3d_default_deck_writes_json(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "first_principles_3d.json"

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["tool"] == "dpf first-principles-3d"
    assert payload["runner"] == "dpf.fields.HybridPIC3DSimulator"
    assert payload["command_status"] == (
        "package_native_first_principles_3d_engineering_run"
    )
    assert payload["deck"]["source"] == "built_in"
    assert payload["deck"]["name"] == (
        "pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate"
    )
    assert payload["deck"]["device_name"] == (
        "PF-1000/Akel shot 12581 engineering candidate"
    )
    assert payload["deck"]["boundary_policy"]["pml_cells"] == 1
    assert payload["deck"]["boundary_policy"]["conductor_mask_mode"] == (
        "pf1000_rod_hollow_projection"
    )
    assert payload["deck"]["circuit_udpf_mode"] == "lagged_volume_j_dot_e"
    assert payload["deck"]["circuit"]["capacitance_F"] == 1.332e-3
    assert payload["deck"]["circuit"]["voltage_V"] == 1.6e4
    assert payload["deck"]["circuit"]["inductance_H"] == 25.0e-9
    assert payload["deck"]["circuit"]["resistance_ohm"] == 6.1e-3
    assert payload["deck"]["circuit"]["initial_current_A"] == 0.0
    assert payload["deck"]["circuit"]["initial_charge_C"] == 0.0
    assert payload["deck"]["circuit"]["circuit_feedback_min_current_A"] == 1.0
    assert payload["deck"]["gas_pressure_Pa"] == 1.2 * 133.32236842105263
    assert payload["deck"]["background_density_m3"] > 1.0e22
    assert payload["deck"]["density_floor_m3"] == payload["deck"][
        "background_density_m3"
    ]
    assert payload["deck"]["marder_factor_scale"] == 0.0
    assert payload["deck"]["startup_profile_type"] == "annular_axial_sheath"
    assert payload["boundary_policy"]["particle_absorption_enabled"] is True
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    assert payload["validation_packet"]["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )
    assert payload["simulation"]["status"] == (
        "candidate_engineering_3d_hybrid_pic_simulation"
    )
    assert payload["simulation"]["retained_step_result_count"] == 2
    assert payload["engineering_firm_dossier"]["runtime_scope"][
        "n_steps_completed"
    ] == 2
    assert payload["engineering_firm_dossier"]["runtime_scope"][
        "retained_step_result_count"
    ] == 2
    assert payload["telemetry_packets"]["physics_closure"]["community_formula_audit"][
        "dependency"
    ] == "plasmapy"
    assert payload["engineering_firm_dossier"]["status"] == (
        "engineering_firm_experimental_test_dossier_not_validation"
    )
    assert payload["engineering_firm_dossier"]["runtime_scope"][
        "first_principles_only_enforced"
    ] is True
    assert payload["engineering_firm_dossier"]["active_blocker_count"] > 0
    assert "PlasmaPy optional community-formulary audit packet when installed" in (
        payload["engineering_firm_dossier"]["observable_surfaces"]
    )
    assert "Package-native 3-D first-principles engineering candidate" in result.output
    assert "current_waveform_comparison:" in result.output
    assert "blocker_count:" in result.output


def test_first_principles_3d_user_validated_deck_presets_are_non_promoting() -> None:
    from dpf.cli.main import cli

    cases = [
        (
            "ir_mpf_100",
            "ir_mpf_100_20kv_1p9torr_engineering_candidate",
            "IR-MPF-100 20 kV / 1.9 Torr engineering candidate",
        ),
        (
            "compact_chinese_dpf",
            "compact_chinese_dpf_20kv_580pa_engineering_candidate",
            "Compact Chinese Mather DPF 20 kV / 580 Pa engineering candidate",
        ),
        (
            "willenborg_hendricks",
            "willenborg_hendricks_19kv_1torr_engineering_candidate",
            "Willenborg/Hendricks DPF 19 kV / 1 Torr engineering candidate",
        ),
        (
            "gv_pf24_krakow_16092202",
            "gv_pf24_krakow_16092202_engineering_candidate",
            (
                "PF-24-KRAKOW pf24_krakow_16092202 GV verified-shot "
                "engineering candidate"
            ),
        ),
    ]

    for preset, deck_id, device_name in cases:
        result = CliRunner().invoke(
            cli,
            [
                "first-principles-3d",
                "--deck-preset",
                preset,
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["deck"]["source"] == f"built_in:{preset}"
        assert payload["deck"]["name"] == deck_id
        assert payload["deck"]["device_name"] == device_name
        assert payload["deck"]["background_density_m3"] > 0.0
        assert payload["deck"]["density_floor_m3"] == payload["deck"][
            "background_density_m3"
        ]
        assert payload["deck"]["marder_factor_scale"] == 0.0
        assert payload["scientific_status"] == "engineering_candidate_not_validation"
        assert payload["reduced_models_used"] is False
        assert payload["can_support_first_principles_acceptance"] is False


def test_first_principles_3d_runtime_overrides_do_not_promote_builtin_deck() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck-preset",
            "gv_pf24_krakow_16092202",
            "--steps",
            "3",
            "--dt-s",
            "2e-13",
            "--history-stride",
            "2",
            "--max-step-results",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deck"]["source"] == "built_in:gv_pf24_krakow_16092202"
    assert payload["deck"]["steps"] == 3
    assert payload["deck"]["dt_s"] == 2.0e-13
    assert payload["deck"]["history_stride"] == 2
    assert payload["deck"]["max_step_results"] == 1
    assert payload["n_steps"] == 3
    assert payload["n_steps_completed"] == 3
    assert payload["dt_s"] == 2.0e-13
    assert payload["history_stride"] == 2
    assert payload["max_step_results"] == 1
    assert payload["simulation"]["retained_step_result_count"] == 1
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    assert payload["engineering_current_waveform_comparison"]["status"] == (
        "engineering_current_waveform_comparison_not_validation"
    )
    assert payload["can_support_first_principles_acceptance"] is False


def test_first_principles_3d_target_time_stops_before_step_budget() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--steps",
            "5",
            "--target-time-s",
            "2e-13",
            "--history-stride",
            "2",
            "--max-step-results",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_steps"] == 5
    assert payload["n_steps_completed"] == 2
    assert payload["termination_reason"] == "target_time_reached"
    assert payload["duration_request_satisfied"] is True
    assert payload["simulation"]["retained_step_result_count"] == 1
    assert payload["engineering_firm_dossier"]["runtime_scope"][
        "simulated_time_s"
    ] == payload["simulation"]["final_time_s"]


def test_experimental_whole_shot_writes_engineering_review_packet(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_whole_shot.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-whole-shot",
            "--steps",
            "3",
            "--target-time-s",
            "1e-6",
            "--history-stride",
            "2",
            "--max-step-results",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_whole_shot"]
    numerics = payload["experimental_numerics"]
    limiter_probe = numerics["limiter_zero_probe"]
    module_names = set(packet["candidate_module_names"])
    duration_plan = packet["duration_plan"]
    assert payload["tool"] == "dpf experimental-whole-shot"
    assert payload["command_status"] == (
        "experimental_whole_shot_engineering_candidate_run"
    )
    assert packet["status"] == "experimental_whole_shot_candidate_not_validation"
    assert packet["run_intent"] == "experimental_whole_shot_engineering_review"
    assert packet["execution_policy"]["candidate_physics_can_run_engineering_cases"]
    assert packet["execution_policy"]["candidate_physics_can_support_acceptance"] is False
    assert packet["duration_request"]["requested_duration_s"] == 1.0e-6
    assert packet["duration_request"]["duration_request_satisfied"] is False
    assert duration_plan["steps_required_current_dt"] == 10_000_000
    assert duration_plan["current_step_budget_satisfies_target"] is False
    assert duration_plan["steps_required_vacuum_cfl_dt"] < (
        duration_plan["steps_required_current_dt"]
    )
    assert duration_plan["vacuum_cfl_step_budget_satisfies_target"] is False
    assert duration_plan["stable_vacuum_dt_s"] > payload["dt_s"]
    assert packet["whole_shot_duration_reached"] is False
    assert packet["can_run_experimental_shot_attempt"] is True
    assert packet["acceptance_state"]["validated"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert numerics["status"] == "experimental_numerical_runtime_audit_not_validation"
    assert numerics["courant_budget"]["dt_within_vacuum_cfl"] is True
    limiter_summary = numerics["full_horizon_limiter_activation_summary"]
    assert limiter_summary["status"] == (
        "experimental_full_horizon_limiter_inventory_not_validation"
    )
    assert limiter_summary["steps_observed"] == payload["n_steps_completed"]
    assert "activation_counts" in limiter_summary
    assert limiter_probe["status"] == "experimental_limiter_zero_probe_not_validation"
    assert limiter_probe["runtime_horizon"][
        "inventory_complete_for_completed_steps"
    ] is True
    assert limiter_probe["can_support_first_principles_acceptance"] is False
    fingerprint = payload["simulation"]["state_fingerprint"]
    assert fingerprint["status"] == (
        "experimental_terminal_state_fingerprint_not_restart_acceptance"
    )
    assert len(fingerprint["sha256"]) == 64
    assert numerics["restart_reproducibility"]["available"] is False
    assert numerics["mesh_timestep_convergence"]["available"] is False
    assert numerics["source_audit_findings"][
        "same_scope_acceptance_evidence_found"
    ] is False
    assert numerics["can_support_first_principles_acceptance"] is False
    assert payload["can_support_first_principles_acceptance"] is False
    assert "startup_breakdown_liftoff_audit" in module_names
    assert "pf1000_rod_hollow_conductor_projection" in module_names
    assert "volume_j_dot_e_power_accounting" in module_names
    assert "electrical_transport_source_terms" in module_names
    assert "experimental_whole_shot" in payload["telemetry_packets"]
    assert "experimental_numerics" in payload["telemetry_packets"]
    assert "experimental_limiter_zero_probe" in payload["telemetry_packets"]
    assert payload["manifest"]["candidate_evidence"][
        "experimental_whole_shot_packet"
    ]["status"] == "experimental_whole_shot_candidate_not_validation"
    assert payload["manifest"]["candidate_evidence"][
        "experimental_limiter_zero_probe_packet"
    ]["status"] == "experimental_limiter_zero_probe_not_validation"
    assert payload["manifest"]["candidate_evidence"][
        "experimental_numerical_runtime_audit_packet"
    ]["status"] == "experimental_numerical_runtime_audit_not_validation"
    assert packet["engineer_review_packet_required"] is True
    assert packet["active_blockers"]
    assert "Experimental whole-shot engineering candidate" in result.output
    assert "duration_request_satisfied: False" in result.output
    assert "required_steps_current_dt:" in result.output


def test_experimental_whole_shot_auto_step_budget_can_reach_small_target(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_whole_shot_auto.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-whole-shot",
            "--dt-policy",
            "vacuum-cfl",
            "--vacuum-cfl",
            "0.25",
            "--target-time-s",
            "1e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "5",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    plan = payload["experimental_whole_shot"]["duration_plan"]
    numerics = payload["experimental_numerics"]
    assert payload["tool"] == "dpf experimental-whole-shot"
    assert payload["n_steps"] == 1
    assert payload["n_steps_completed"] == 1
    assert payload["duration_request_satisfied"] is True
    assert plan["steps_required_current_dt"] == 1
    assert plan["current_step_budget_satisfies_target"] is True
    assert plan["dt_within_vacuum_cfl"] is True
    assert numerics["courant_budget"]["dt_within_vacuum_cfl"] is True
    assert numerics["courant_budget"]["dt_to_stable_vacuum_dt_ratio"] <= 1.0
    assert payload["can_support_first_principles_acceptance"] is False


def test_experimental_whole_shot_auto_step_budget_guard_blocks_large_run() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "experimental-whole-shot",
            "--dt-policy",
            "vacuum-cfl",
            "--target-time-s",
            "1e-6",
            "--auto-step-budget",
            "--max-auto-steps",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert "auto step budget would require" in result.output


def test_experimental_numerical_family_reports_timestep_deltas(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_numerical_family.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-numerical-family",
            "--family",
            "timestep",
            "--target-time-s",
            "1e-13",
            "--timestep-scales",
            "1,0.5",
            "--max-auto-steps",
            "5",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_numerical_family"]
    assert payload["tool"] == "dpf experimental-numerical-family"
    assert payload["command_status"] == (
        "experimental_numerical_family_engineering_candidate_run"
    )
    assert packet["status"] == "experimental_numerical_family_probe_not_validation"
    assert packet["family_kind"] == "timestep"
    assert packet["case_count"] == 2
    assert packet["duration_satisfied_case_count"] == 2
    assert len(packet["pairwise_comparisons"]) == 1
    assert packet["convergence_decision"]["tolerance_claim"] is False
    assert packet["convergence_decision"]["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["cases"][0]["case_family_axis"]["kind"] == "timestep"
    assert packet["cases"][1]["dt_s"] < packet["cases"][0]["dt_s"]
    assert payload["can_support_first_principles_acceptance"] is False
    assert "Experimental numerical family engineering candidate" in result.output


def test_experimental_reproducibility_reports_matching_rerun_hashes(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_reproducibility.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-reproducibility",
            "--target-time-s",
            "1e-13",
            "--repeat-count",
            "2",
            "--max-auto-steps",
            "5",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_reproducibility"]
    assert payload["tool"] == "dpf experimental-reproducibility"
    assert payload["command_status"] == (
        "experimental_reproducibility_engineering_candidate_run"
    )
    assert packet["status"] == "experimental_reproducibility_probe_not_validation"
    assert packet["run_count"] == 2
    assert packet["deterministic_rerun"]["available"] is True
    assert packet["all_state_observable_hashes_identical"] is True
    assert packet["pairwise_comparisons"][0]["state_observable_hashes_match"] is True
    assert len(packet["runs"][0]["state_fingerprint"]["sha256"]) == 64
    assert packet["checkpoint_restart"]["available"] is False
    assert packet["continued_run_equivalence"]["available"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "Experimental reproducibility engineering candidate" in result.output


def test_experimental_limiter_proof_reports_runtime_zero_probe(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_limiter_proof.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-limiter-proof",
            "--target-time-s",
            "1e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "5",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_limiter_zero_probe"]
    assert payload["tool"] == "dpf experimental-limiter-proof"
    assert payload["command_status"] == (
        "experimental_limiter_proof_engineering_candidate_run"
    )
    assert packet["status"] == "experimental_limiter_zero_probe_not_validation"
    assert packet["runtime_horizon"][
        "inventory_complete_for_completed_steps"
    ] is True
    assert packet["runtime_horizon"]["target_time_satisfied"] is True
    assert "acceptance_blocking_counts" in packet
    assert packet["acceptance_state"]["can_support_limiter_zero_acceptance"] is False
    assert payload["telemetry_packets"]["experimental_limiter_zero_probe"] == packet
    assert payload["telemetry_packets"]["limiter_readiness"][
        "runtime_limiter_zero_probe"
    ]["status"] == "experimental_limiter_zero_probe_not_validation"
    assert payload["experimental_numerics"]["limiter_zero_probe"][
        "status"
    ] == "experimental_limiter_zero_probe_not_validation"
    assert payload["can_support_first_principles_acceptance"] is False
    assert "Experimental limiter-zero engineering candidate" in result.output


def test_experimental_limiter_proof_combined_cfl_clears_ohmic_limit_for_short_target(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_limiter_proof_combined_cfl.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-limiter-proof",
            "--target-time-s",
            "5e-14",
            "--auto-step-budget",
            "--max-auto-steps",
            "40",
            "--dt-policy",
            "combined-cfl",
            "--history-stride",
            "1",
            "--max-step-results",
            "3",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_limiter_zero_probe"]
    counts = packet["acceptance_blocking_counts"]
    assert payload["duration_request_satisfied"] is True
    assert payload["dt_s"] < 1.0e-13
    assert counts["conductivity_ohmic_cfl_limited_steps"] == 0
    assert packet["total_acceptance_blocking_activations"] == 0
    assert packet["zero_acceptance_blockers_observed"] is True
    marder_decision = packet["method_limiter_decisions"]["marder_correction"]
    assert marder_decision["steps_observed"] == 0
    assert marder_decision["nondominant_observed"] is False
    assert "marder_dominant_correction_steps" not in packet["review_required"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_experimental_state_checkpoint_writes_roundtrip_packet(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_state_checkpoint.json"
    checkpoint = tmp_path / "terminal_state_checkpoint.npz"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-state-checkpoint",
            "--target-time-s",
            "1e-13",
            "--max-auto-steps",
            "5",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--checkpoint-output",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_state_checkpoint"]
    assert checkpoint.exists()
    assert payload["tool"] == "dpf experimental-state-checkpoint"
    assert payload["command_status"] == (
        "experimental_state_checkpoint_engineering_candidate_run"
    )
    assert packet["status"] == (
        "experimental_state_checkpoint_roundtrip_not_restart_acceptance"
    )
    assert packet["write_read_hashes_match"] is True
    assert len(packet["write_content_sha256"]) == 64
    assert packet["write_content_sha256"] == packet["read_content_sha256"]
    assert packet["restart_acceptance"][
        "can_restart_live_runner_from_checkpoint"
    ] is False
    assert packet["restart_acceptance"]["continued_run_equivalence_available"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "Experimental state-checkpoint engineering candidate" in result.output


def test_experimental_split_continuation_matches_uninterrupted_run(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_split_continuation.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-split-continuation",
            "--steps",
            "4",
            "--split-after-steps",
            "2",
            "--history-stride",
            "1",
            "--max-step-results",
            "4",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_split_continuation"]
    assert payload["tool"] == "dpf experimental-split-continuation"
    assert payload["command_status"] == (
        "experimental_split_continuation_engineering_candidate_run"
    )
    assert packet["status"] == (
        "experimental_split_continuation_probe_not_restart_acceptance"
    )
    assert packet["total_steps"] == 4
    assert packet["split_after_steps"] == 2
    assert packet["state_fingerprints_match"] is True
    assert packet["tracked_observables_match_exactly"] is True
    assert packet["continuation_state"][
        "lagged_field_work_preserved_into_second_segment"
    ] is True
    assert packet["checkpoint_restart"]["available"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "Experimental split-continuation engineering candidate" in result.output


def test_experimental_checkpoint_restart_matches_uninterrupted_run(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_checkpoint_restart.json"
    checkpoint = tmp_path / "restart_checkpoint.npz"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-checkpoint-restart",
            "--steps",
            "4",
            "--split-after-steps",
            "2",
            "--history-stride",
            "1",
            "--max-step-results",
            "4",
            "--checkpoint-output",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_checkpoint_restart"]
    assert checkpoint.exists()
    assert payload["tool"] == "dpf experimental-checkpoint-restart"
    assert payload["command_status"] == (
        "experimental_checkpoint_restart_engineering_candidate_run"
    )
    assert packet["status"] == "experimental_checkpoint_restart_probe_not_validation"
    assert packet["total_steps"] == 4
    assert packet["split_after_steps"] == 2
    assert packet["checkpoint_roundtrip"]["write_read_hashes_match"] is True
    assert packet["state_fingerprints_match"] is True
    assert packet["tracked_observables_match_exactly"] is True
    assert packet["restart_state"]["lagged_field_work_loaded"] is True
    assert packet["restart_state"]["previous_total_current_loaded"] is True
    assert packet["restart_state"]["kinetic_yield_state_loaded"] is True
    assert packet["can_support_first_principles_acceptance"] is False
    assert "Experimental checkpoint-restart engineering candidate" in result.output


def test_experimental_checkpoint_restart_family_reports_multi_offset_matches(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_checkpoint_restart_family.json"
    checkpoint_dir = tmp_path / "restart_family"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-checkpoint-restart-family",
            "--steps",
            "4",
            "--split-after-steps",
            "1,2",
            "--history-stride",
            "1",
            "--max-step-results",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_checkpoint_restart_family"]
    assert payload["tool"] == "dpf experimental-checkpoint-restart-family"
    assert payload["command_status"] == (
        "experimental_checkpoint_restart_family_engineering_candidate_run"
    )
    assert packet["status"] == (
        "experimental_checkpoint_restart_family_probe_not_validation"
    )
    assert packet["case_count"] == 2
    assert packet["matching_case_count"] == 2
    assert packet["all_cases_match"] is True
    assert [case["split_after_steps"] for case in packet["cases"]] == [1, 2]
    assert all(case["state_fingerprints_match"] for case in packet["cases"])
    assert all(
        case["tracked_observables_match_exactly"] for case in packet["cases"]
    )
    assert all(
        case["uninterrupted"]["limiter_zero_probe"]["status"]
        == "experimental_limiter_zero_probe_not_validation"
        for case in packet["cases"]
    )
    assert all(
        case["checkpoint_restart"]["limiter_zero_probe"]["status"]
        == "experimental_limiter_zero_probe_not_validation"
        for case in packet["cases"]
    )
    assert checkpoint_dir.exists()
    assert packet["can_support_first_principles_acceptance"] is False
    assert "Experimental checkpoint-restart family engineering candidate" in (
        result.output
    )


def test_experimental_checkpoint_restart_family_auto_step_budget_reaches_target(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "experimental_checkpoint_restart_family_auto.json"
    checkpoint_dir = tmp_path / "restart_family_auto"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-checkpoint-restart-family",
            "--target-time-s",
            "2e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "5",
            "--split-after-steps",
            "1",
            "--history-stride",
            "1",
            "--max-step-results",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    packet = payload["experimental_checkpoint_restart_family"]
    assert packet["total_steps"] == 2
    assert packet["split_after_steps"] == [1]
    assert packet["case_count"] == 1
    assert packet["matching_case_count"] == 1
    assert packet["all_cases_match"] is True
    assert packet["cases"][0]["uninterrupted"]["final_time_s"] >= 2.0e-13
    assert packet["can_support_first_principles_acceptance"] is False


def test_experimental_checkpoint_restart_family_auto_step_budget_guard_blocks_large_run() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "experimental-checkpoint-restart-family",
            "--target-time-s",
            "1e-6",
            "--auto-step-budget",
            "--max-auto-steps",
            "2",
            "--split-after-steps",
            "1",
            "--checkpoint-dir",
            "restart_family",
        ],
    )

    assert result.exit_code != 0
    assert "auto step budget would require" in result.output


def test_first_principles_3d_reads_deck_json_and_prints_json(tmp_path) -> None:
    from dpf.cli.main import cli

    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps({
        "name": "tiny_test_deck",
        "steps": 1,
        "grid_shape": [4, 4, 4],
        "dt_s": 1.0e-13,
        "apply_circuit_boundary": False,
        "pml_cells": 1,
        "pml_strength": 0.2,
        "particle_absorption_enabled": True,
    }))

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck",
            str(deck),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool"] == "dpf first-principles-3d"
    assert payload["deck"]["source"] == str(deck)
    assert payload["deck"]["name"] == "tiny_test_deck"
    assert payload["deck"]["steps"] == 1
    assert payload["deck"]["grid_shape"] == [4, 4, 4]
    assert payload["deck"]["apply_circuit_boundary"] is False
    assert payload["deck"]["circuit_udpf_mode"] == "lagged_volume_j_dot_e"
    assert payload["deck"]["boundary_policy"]["pml_cells"] == 1
    assert payload["deck"]["boundary_policy"]["pml_strength"] == 0.2
    assert payload["boundary_policy"]["particle_absorption_enabled"] is True
    assert payload["n_steps"] == 1


def test_first_principles_3d_runtime_overrides_compact_json_deck(tmp_path) -> None:
    from dpf.cli.main import cli

    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps({
        "name": "tiny_override_deck",
        "steps": 1,
        "grid_shape": [4, 4, 4],
        "dt_s": 1.0e-13,
        "apply_circuit_boundary": False,
        "pml_cells": 0,
        "pml_strength": 0.0,
        "particle_absorption_enabled": False,
    }))

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck",
            str(deck),
            "--steps",
            "2",
            "--dt-s",
            "3e-13",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deck"]["name"] == "tiny_override_deck"
    assert payload["deck"]["steps"] == 2
    assert payload["deck"]["dt_s"] == 3.0e-13
    assert payload["n_steps"] == 2
    assert payload["dt_s"] == 3.0e-13
