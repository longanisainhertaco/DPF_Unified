from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from dpf.cli.main import cli
from dpf.first_principles import (
    bank_energy_J,
    build_experimental_inverse_calibration_packet,
    build_experimental_inverse_parameter_packet,
    build_source_bounded_candidate_grid_from_parameter_scales,
    classify_inverse_calibration_results,
    current_implied_inductance_H,
    ideal_lc_peak_current_A,
    ideal_lc_quarter_cycle_s,
    score_current_history_against_targets,
)

GV_ROOT = Path("/Users/anthonyzamora/Downloads/GV")


def test_inverse_parameter_algebra_helpers_are_first_principles_circuit_identities() -> None:
    assert bank_energy_J(40.0e-6, 20.0e3) == pytest.approx(8.0e3)
    assert current_implied_inductance_H(
        capacitance_F=40.0e-6,
        voltage_V=20.0e3,
        peak_current_A=400.0e3,
    ) == pytest.approx(100.0e-9)
    assert ideal_lc_peak_current_A(
        capacitance_F=40.0e-6,
        voltage_V=20.0e3,
        inductance_H=100.0e-9,
    ) == pytest.approx(400.0e3)
    assert ideal_lc_quarter_cycle_s(
        capacitance_F=43.5e-6,
        inductance_H=100.0e-9,
    ) == pytest.approx(0.5 * math.pi * math.sqrt(43.5e-6 * 100.0e-9))


def test_source_bounded_candidate_grid_supports_parameter_specific_scales() -> None:
    candidates = build_source_bounded_candidate_grid_from_parameter_scales(
        baseline_parameters={"inductance": 2.0, "resistance": 3.0},
        parameter_names=("inductance", "resistance"),
        parameter_scale_values={
            "inductance": (0.75, 1.0),
            "resistance": (1.0, 2.0, 4.0),
        },
    )

    assert len(candidates) == 6
    assert candidates[0]["parameter_values"]["inductance"] == pytest.approx(1.5)
    assert candidates[0]["parameter_values"]["resistance"] == pytest.approx(3.0)
    assert candidates[-1]["parameter_factors"]["inductance"] == pytest.approx(1.0)
    assert candidates[-1]["parameter_factors"]["resistance"] == pytest.approx(4.0)
    assert candidates[-1]["parameter_values"]["inductance"] == pytest.approx(2.0)
    assert candidates[-1]["parameter_values"]["resistance"] == pytest.approx(12.0)


def test_inverse_parameter_packet_keeps_nonunique_physics_closed() -> None:
    packet = build_experimental_inverse_parameter_packet(
        scope="may15",
        include_gv_waveforms=False,
    )

    assert packet["status"] == "experimental_inverse_parameter_completion_not_validation"
    assert packet["source_policy"]["reduced_models_used"] is False
    assert packet["deck_completion_policy"]["may_fill_experimental_decks"] is True
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["status_counts"]["direct_algebraic_inference"] > 0
    assert packet["status_counts"]["underdetermined_requires_additional_observable"] > 0

    compact = packet["machines"]["compact_chinese_dpf_2018"]
    deck_L = compact["deck_fill_candidates"]["inductance_H"]
    assert deck_L["status"] == "direct_algebraic_inference"
    assert deck_L["value"] == pytest.approx(100.0e-9)
    assert compact["unresolved_parameters"]["resistance_ohm"]["status"] == (
        "underdetermined_requires_additional_observable"
    )

    ir = packet["machines"]["ir_mpf_100_salehizadeh_2012"]
    check = ir["consistency_checks"]["source_L_vs_source_theoretical_peak_current"]
    assert check["status"] == "contradiction_or_scope_mismatch"
    assert check["calculated_value"] != pytest.approx(check["source_value"])


def test_inverse_parameter_packet_includes_pf1000_hollow_anode_gap() -> None:
    packet = build_experimental_inverse_parameter_packet(
        scope="pf1000",
        include_gv_waveforms=False,
    )

    pf1000 = packet["machines"]["pf1000_akel_16kv_shot_12581"]
    assert pf1000["known_parameters"]["capacitance_F"]["value"] == pytest.approx(
        1.332e-3
    )
    assert pf1000["derived_parameters"]["bank_energy_J"]["value"] == pytest.approx(
        170496.0
    )
    assert pf1000["derived_parameters"]["cathode_outer_radius_m"]["value"] == pytest.approx(
        0.20
    )
    assert pf1000["unresolved_parameters"]["hollow_anode_inner_radius_m"]["status"] == (
        "underdetermined_requires_additional_observable"
    )
    assert packet["unresolved_parameter_count"] >= 1


@pytest.mark.skipif(
    not (GV_ROOT / "PF-24-KRAKOW-16092202.xlsx").exists(),
    reason="verified local GV workbook bundle is not available",
)
def test_inverse_parameter_packet_uses_gv_waveform_as_candidate_not_drive() -> None:
    packet = build_experimental_inverse_parameter_packet(
        scope="gv",
        include_gv_waveforms=True,
    )

    pf24 = packet["machines"]["gv_pf24_krakow_16092202"]
    peak = pf24["derived_parameters"]["waveform_peak_current_A"]
    assert peak["status"] == "waveform_derived_candidate"
    assert peak["value"] == pytest.approx(401.6e3)
    assert packet["source_policy"]["measured_waveforms_used_as_drive"] is False
    assert packet["source_policy"]["gv_reduced_model_output_used"] is False
    assert pf24["deck_fill_candidates"]["inductance_H"]["status"] == "known_source_value"


def test_experimental_inverse_parameters_cli_outputs_nonpromoting_packet() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-inverse-parameters",
            "--scope",
            "may15",
            "--no-include-gv-waveforms",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["machine_count"] == 3
    assert payload["status"] == "experimental_inverse_parameter_completion_not_validation"
    assert payload["can_support_first_principles_acceptance"] is False
    assert "compact_chinese_dpf_2018" in payload["machines"]


def test_experimental_machine_shot_family_cli_runs_source_backed_decks() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-machine-shot-family",
            "--scope",
            "may15",
            "--target-time-s",
            "2e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "10",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
            "--no-include-gv-waveforms",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "experimental_machine_shot_family_not_validation"
    assert payload["case_count"] == 3
    assert payload["completed_case_count"] == 3
    assert payload["duration_satisfied_case_count"] == 3
    assert payload["inverse_parameter_summary"]["machine_count"] == 3
    assert payload["can_support_first_principles_acceptance"] is False


def test_experimental_machine_shot_family_cli_records_step_cap_blocks() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-machine-shot-family",
            "--scope",
            "pf1000",
            "--target-time-s",
            "1e-6",
            "--dt-policy",
            "combined-cfl",
            "--auto-step-budget",
            "--max-auto-steps",
            "10",
            "--no-include-gv-waveforms",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["case_count"] == 1
    assert payload["completed_case_count"] == 0
    assert payload["blocked_case_count"] == 1
    assert payload["cases"][0]["case_status"] == "blocked_before_run"
    assert "auto step budget would require" in payload["cases"][0]["blocking_reason"]


def test_inverse_calibration_scoring_and_identifiability_are_explicit() -> None:
    scoring = score_current_history_against_targets(
        current_history=[
            {"time_s": 0.0, "current_A": 0.0},
            {"time_s": 1.0e-9, "current_A": 9.0e5},
        ],
        target_observables={"peak_current_A": 1.0e6},
    )

    assert scoring["status"] == "scored_against_source_observables"
    assert scoring["metrics"]["peak_current_relative_error"] == pytest.approx(0.1)
    assert scoring["metrics"]["peak_at_final_sample"] is True

    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_low",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 0.75},
                },
                "scoring": {"usable": True, "score": 0.2},
            },
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_best",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {"usable": True, "score": 0.1},
            },
        ),
        parameter_names=("inductance",),
        accepted_fit_score_threshold=0.2,
    )

    assert classification["status"] == "uniquely_inferred_on_candidate_grid"
    assert classification["best_candidate_id"] == "candidate_best"
    assert classification["can_conclude_unique_parameters"] is True


def test_inverse_calibration_does_not_claim_unique_without_fit_tolerance() -> None:
    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_best",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.33,
                    "metrics": {"peak_at_final_sample": False},
                },
            },
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_other",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.5},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.50,
                    "metrics": {"peak_at_final_sample": False},
                },
            },
        ),
        parameter_names=("inductance",),
    )

    assert classification["status"] == (
        "separated_candidate_grid_without_accepted_fit_tolerance"
    )
    assert classification["best_candidate_grid_separated"] is True
    assert classification["fit_score_within_accepted_threshold"] is None
    assert classification["can_conclude_unique_parameters"] is False


def test_inverse_calibration_blocks_unique_when_fit_score_exceeds_threshold() -> None:
    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_best",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.33,
                    "metrics": {"peak_at_final_sample": False},
                },
            },
        ),
        parameter_names=("inductance",),
        accepted_fit_score_threshold=0.2,
    )

    assert classification["status"] == (
        "separated_candidate_grid_but_fit_score_exceeds_threshold"
    )
    assert classification["fit_score_within_accepted_threshold"] is False
    assert classification["can_conclude_unique_parameters"] is False


def test_inverse_calibration_ignores_nonfinite_completed_candidate_scores() -> None:
    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "finite_state_all_finite": False,
                "candidate": {
                    "candidate_id": "candidate_nonfinite",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.0,
                    "metrics": {"peak_at_final_sample": False},
                },
            },
        ),
        parameter_names=("inductance",),
    )

    assert classification["status"] == "no_conclusion_no_usable_completed_candidates"
    assert classification["can_conclude_unique_parameters"] is False


def test_inverse_calibration_packet_flags_invariant_parameter_surfaces() -> None:
    packet = build_experimental_inverse_calibration_packet(
        declared_scope="unit",
        device_name="unit",
        target_observables={"peak_current_A": 1.0},
        parameter_names=("pressure",),
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "finite_state_all_finite": True,
                "candidate": {
                    "candidate_id": "candidate_low",
                    "baseline_parameters": {"pressure": 10.0},
                    "parameter_values": {"pressure": 7.5},
                    "parameter_factors": {"pressure": 0.75},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {
                        "simulation_peak_current_A": 100.0,
                        "simulation_peak_time_s": 1.0e-6,
                        "waveform_nrmse_fraction": 0.2,
                    },
                },
            },
            {
                "case_status": "completed_engineering_candidate_run",
                "finite_state_all_finite": True,
                "candidate": {
                    "candidate_id": "candidate_high",
                    "baseline_parameters": {"pressure": 10.0},
                    "parameter_values": {"pressure": 12.5},
                    "parameter_factors": {"pressure": 1.25},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {
                        "simulation_peak_current_A": 100.0,
                        "simulation_peak_time_s": 1.0e-6,
                        "waveform_nrmse_fraction": 0.2,
                    },
                },
            },
        ),
    )

    pressure = packet["parameter_sensitivity"]["parameters"]["pressure"]
    assert pressure["status"] == "no_observed_effect_on_scored_metrics"
    assert pressure["can_infer_parameter_sensitivity"] is False
    assert pressure["groups"][0]["metric_ranges"]["score"]["range"] == 0.0


def test_inverse_calibration_packet_separates_runtime_from_scored_sensitivity() -> None:
    packet = build_experimental_inverse_calibration_packet(
        declared_scope="unit",
        device_name="unit",
        target_observables={"peak_current_A": 1.0},
        parameter_names=("pressure",),
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "finite_state_all_finite": True,
                "candidate": {
                    "candidate_id": "candidate_low",
                    "baseline_parameters": {"pressure": 10.0},
                    "parameter_values": {"pressure": 7.5},
                    "parameter_factors": {"pressure": 0.75},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {
                        "simulation_peak_current_A": 100.0,
                        "simulation_peak_time_s": 1.0e-6,
                        "waveform_nrmse_fraction": 0.2,
                    },
                },
                "plasma_loading_summary": {
                    "j_dot_e_power_W_max_abs": 1.0,
                    "j_dot_e_energy_trapezoid_J": 1.0e-6,
                    "field_energy_delta_J": 1.0e-8,
                    "field_energy_J_final": 2.0e-8,
                    "circuit_current_A_final": 100.0,
                },
            },
            {
                "case_status": "completed_engineering_candidate_run",
                "finite_state_all_finite": True,
                "candidate": {
                    "candidate_id": "candidate_high",
                    "baseline_parameters": {"pressure": 10.0},
                    "parameter_values": {"pressure": 12.5},
                    "parameter_factors": {"pressure": 1.25},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {
                        "simulation_peak_current_A": 100.0,
                        "simulation_peak_time_s": 1.0e-6,
                        "waveform_nrmse_fraction": 0.2,
                    },
                },
                "plasma_loading_summary": {
                    "j_dot_e_power_W_max_abs": 2.0,
                    "j_dot_e_energy_trapezoid_J": 2.0e-6,
                    "field_energy_delta_J": 2.0e-8,
                    "field_energy_J_final": 4.0e-8,
                    "circuit_current_A_final": 100.0,
                },
            },
        ),
    )

    pressure = packet["parameter_sensitivity"]["parameters"]["pressure"]
    group = pressure["groups"][0]
    assert pressure["status"] == (
        "observed_effect_on_candidate_runtime_metrics_not_scored_current"
    )
    assert pressure["scored_effect_observed_group_count"] == 0
    assert pressure["runtime_effect_observed_group_count"] == 1
    assert group["metric_ranges"]["score"]["range"] == 0.0
    assert group["runtime_metric_ranges"]["j_dot_e_power_W_max_abs"]["range"] == 1.0
    assert group["effect_observed_on_scored_metrics"] is False
    assert group["effect_observed_on_runtime_metrics"] is True


def test_inverse_calibration_scoring_can_use_nonpromoting_waveform_shape() -> None:
    scoring = score_current_history_against_targets(
        current_history=[
            {"time_us": 0.0, "current_A": 0.0},
            {"time_us": 1.0, "current_A": 1.0e3},
            {"time_us": 2.0, "current_A": 0.0},
        ],
        target_observables={
            "waveform": {
                "time_us": [0.0, 1.0, 2.0],
                "current_kA": [0.0, 1.1, 0.0],
                "series_sha256": "candidate-series",
            },
            "minimum_waveform_coverage_fraction": 0.95,
            "minimum_waveform_overlap_points": 3,
        },
    )

    assert scoring["status"] == "scored_against_source_observables"
    assert scoring["metrics"]["waveform_status"] == "scored_waveform_shape_overlap"
    assert scoring["metrics"]["waveform_score_included"] is True
    assert scoring["metrics"]["waveform_coverage_horizon_limited"] is False
    assert scoring["metrics"]["waveform_overlap_point_count"] == 3
    assert scoring["metrics"]["waveform_nrmse_fraction"] == pytest.approx(
        0.052486388108147805
    )


def test_inverse_calibration_waveform_coverage_excludes_pretrigger_baseline() -> None:
    scoring = score_current_history_against_targets(
        current_history=[
            {"time_us": 0.0, "current_A": 0.0},
            {"time_us": 3.0, "current_A": 1.0e3},
            {"time_us": 6.0, "current_A": 0.0},
        ],
        target_observables={
            "waveform": {
                "time_us": [-0.5, 0.0, 3.0, 6.0],
                "current_kA": [0.0, 0.0, 1.0, 0.0],
            },
            "minimum_waveform_coverage_fraction": 0.95,
            "minimum_waveform_overlap_points": 3,
        },
    )

    assert scoring["metrics"]["waveform_target_time_range_us"] == [-0.5, 6.0]
    assert scoring["metrics"]["waveform_scored_time_range_us"] == [0.0, 6.0]
    assert scoring["metrics"]["waveform_pretrigger_time_excluded_from_coverage_us"] == 0.5
    assert scoring["metrics"]["waveform_temporal_coverage_fraction_of_target"] == 1.0
    assert scoring["metrics"]["waveform_coverage_horizon_limited"] is False


def test_inverse_calibration_refuses_unique_claim_when_best_peak_hits_horizon() -> None:
    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_best",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {"peak_at_final_sample": True},
                },
            },
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_other",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.25},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.2,
                    "metrics": {"peak_at_final_sample": False},
                },
            },
        ),
        parameter_names=("inductance",),
    )

    assert classification["status"] == "horizon_limited_requires_longer_run"
    assert classification["horizon_limited_candidate_ids"] == ["candidate_best"]
    assert classification["horizon_limited_reasons"] == {
        "candidate_best": ["peak_at_final_sample"]
    }
    assert classification["can_conclude_unique_parameters"] is False


def test_inverse_calibration_refuses_unique_claim_when_waveform_is_truncated() -> None:
    classification = classify_inverse_calibration_results(
        candidate_results=(
            {
                "case_status": "completed_engineering_candidate_run",
                "candidate": {
                    "candidate_id": "candidate_best",
                    "baseline_parameters": {"inductance": 1.0},
                    "parameter_values": {"inductance": 1.0},
                },
                "scoring": {
                    "usable": True,
                    "score": 0.1,
                    "metrics": {
                        "peak_at_final_sample": False,
                        "waveform_coverage_horizon_limited": True,
                    },
                },
            },
        ),
        parameter_names=("inductance",),
    )

    assert classification["status"] == "horizon_limited_requires_longer_run"
    assert classification["horizon_limited_reasons"] == {
        "candidate_best": ["waveform_temporal_coverage_below_required"]
    }
    assert classification["can_conclude_unique_parameters"] is False


def test_experimental_inverse_calibration_cli_fits_candidate_grid() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-inverse-calibration",
            "--deck-preset",
            "compact_chinese_dpf",
            "--parameters",
            "inductance",
            "--candidate-scales",
            "0.75,1",
            "--target-time-s",
            "2e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "10",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "experimental_inverse_calibration_not_validation"
    assert payload["candidate_count"] == 2
    assert payload["completed_candidate_count"] == 2
    assert payload["target_observables"]["peak_current_A"] == pytest.approx(400.0e3)
    assert payload["identifiability"]["status"] in {
        "uniquely_inferred_on_candidate_grid",
        "range_constrained_on_candidate_grid",
        "underdetermined_or_correlated_on_candidate_grid",
        "horizon_limited_requires_longer_run",
        "separated_candidate_grid_without_accepted_fit_tolerance",
        "separated_candidate_grid_but_fit_score_exceeds_threshold",
    }
    evidence = payload["candidate_results"][0]["runtime_evidence_packets"]
    assert evidence["startup"]["can_support_first_principles_acceptance"] is False
    assert evidence["power_port"]["status"] == (
        "candidate_engineering_power_port_not_validation"
    )
    assert evidence["experimental_limiter_zero_probe"]["can_support_first_principles_acceptance"] is False
    assert payload["can_support_first_principles_acceptance"] is False


def test_experimental_inverse_calibration_cli_accepts_parameter_specific_scales() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-inverse-calibration",
            "--deck-preset",
            "gv_pf24_krakow_16092202",
            "--parameters",
            "inductance,resistance",
            "--candidate-scales",
            "1",
            "--parameter-scale",
            "inductance=0.75,1",
            "--parameter-scale",
            "resistance=1,2,4",
            "--target-time-s",
            "2e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "10",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["candidate_count"] == 6
    assert payload["parameter_names"] == ["inductance", "resistance"]
    assert payload["runtime_policy"]["parameter_scale_values"] == {
        "inductance": [0.75, 1.0],
        "resistance": [1.0, 2.0, 4.0],
    }
    assert payload["candidate_results"][0]["candidate"]["parameter_factors"] == {
        "inductance": 0.75,
        "resistance": 1.0,
    }


def test_experimental_inverse_calibration_pressure_updates_startup_density() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "experimental-inverse-calibration",
            "--deck-preset",
            "gv_pf24_krakow_16092202",
            "--parameters",
            "pressure",
            "--candidate-scales",
            "1",
            "--parameter-scale",
            "pressure=0.75,1.25",
            "--target-time-s",
            "2e-13",
            "--auto-step-budget",
            "--max-auto-steps",
            "10",
            "--history-stride",
            "1",
            "--max-step-results",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    densities = [
        item["runtime_coupling"]["pressure_density_coupling"][
            "background_density_m3"
        ]
        for item in payload["candidate_results"]
    ]
    assert densities[0] > 1.0e22
    assert densities[1] > densities[0]
    assert payload["candidate_results"][0]["runtime_coupling"][
        "pressure_density_coupling"
    ]["formula"] == "n = pressure_Pa / (k_B * gas_temperature_K)"
    loading = payload["candidate_results"][0]["plasma_loading_summary"]
    assert loading["status"] == "experimental_plasma_loading_observability_not_validation"
    assert loading["history_point_count"] >= 1
    assert loading["j_dot_e_power_W_max_abs"] is not None
    assert loading["cumulative_j_dot_e_work_J"] is not None
    assert loading["cumulative_j_dot_e_step_count"] is not None
    assert loading["circuit_current_A_final"] is not None
    assert loading["circuit_udpf_source_counts"]
    assert "circuit_terminal_voltage_V_final_step" in loading
    assert loading["electron_density_m3_max_retained"] is not None
    assert loading["source_backed_sigma_S_m_max_retained"] is not None
    assert loading["conductivity_ohmic_cfl_limit_applied_counts"]
    assert loading["electric_update_scheme_counts"]
    assert loading["ohm_time_centering_theta_max_retained"] is not None
    assert "electron_heat_flux_status_counts" in loading
    assert "electron_heat_flux_terminal_status" in loading
    assert "electron_heat_flux_required_subcycles_max" in loading
    assert "conductivity_ohmic_cfl_limit_applied_terminal" in loading
    assert "electric_update_scheme_terminal" in loading
    assert "ohm_time_centering_theta_terminal" in loading
    pressure = payload["parameter_sensitivity"]["parameters"]["pressure"]
    group = pressure["groups"][0]
    assert "runtime_metric_ranges" in group
    assert "j_dot_e_power_W_max_abs" in group["runtime_metric_ranges"]
    assert "circuit_active_power_W_final_step" in group["runtime_metric_ranges"]
    assert "source_backed_sigma_S_m_max_retained" in group["runtime_metric_ranges"]
