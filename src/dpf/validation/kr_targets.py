"""KnowledgeReference-backed validation targets.

These targets are extracted from the local KnowledgeReference corpus. They are
not pass/fail results by themselves; they define what a simulation output must
be compared against before predictive-readiness evidence can be claimed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def lee_snowplow_phase_semantics_targets() -> dict[str, object]:
    """Return KR-backed Lee/RADPF phase semantics for tier-2 targets."""
    source = (
        "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-"
        "s-lee-and-s-h-saw-part-1-basic-course.md"
    )
    return {
        "target_id": "lee_radpf_phase_semantics_course",
        "device": "generic Lee/RADPF",
        "model_role": "kr_phase_semantics_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "current_trace_fit_features": "886-891",
            "radial_rollover_and_dip": "14922-14936",
            "example_phase_timing": "16239-16244",
            "gross_compression_semantics": "16298-16304",
        },
        "phase_semantics": {
            "axial": (
                "Axial phase is fitted from current rise, rounding, and peak "
                "current before radial dip fitting."
            ),
            "radial": (
                "Radial phase starts as the current trace rolls over and is "
                "fitted through the current dip."
            ),
            "pinch": (
                "Pinch/gross compression follows shock-axis arrival and "
                "reflected-shock/piston interaction."
            ),
        },
        "required_for_full_tier2": ["axial", "radial", "pinch"],
        "validation_note": (
            "This target defines Lee/RADPF phase semantics. It is not a "
            "device-specific timing dataset by itself."
        ),
    }


def lee_course_nx2_neon_phase_timing_example_targets() -> dict[str, object]:
    """Return a typed Lee-course NX2 neon phase-timing example target."""
    source = (
        "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-"
        "s-lee-and-s-h-saw-part-1-basic-course.md"
    )
    return {
        "target_id": "lee_course_nx2_neon_phase_timing_example",
        "device": "NX2",
        "model_role": "kr_phase_timing_example_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "phase_endpoint_times": "1938-1958",
            "radial_axis_and_pinch_timing": "1978-1994",
            "course_context": "2038-2048",
        },
        "shot_context": {
            "device": "NX2",
            "voltage_kV": 11.0,
            "fill_pressure_torr": 2.6,
            "fill_gas": "neon",
            "context": "RADPF worksheet example with fitted model parameters",
        },
        "phase_timing": {
            "axial_end_time_us": 1.172,
            "radial_start_time_us": 1.172,
            "radial_end_time_us": 1.407,
            "radial_duration_us": 0.235,
            "pinch_start_time_us": 1.38,
            "pinch_end_time_us": 1.407,
            "pinch_duration_ns": 26.2,
            "radial_shock_axis_time_after_radial_start_ns": 178.0,
            "reflected_shock_piston_time_after_radial_start_ns": 210.0,
        },
        "missing_for_predictive_tier2": [
            "deuterium_device_match",
            "measured_current_trace_source",
            "experimental_phase_timing_uncertainty",
        ],
        "validation_note": (
            "This is a typed Lee/RADPF course example for NX2 neon, useful for "
            "phase endpoint semantics and parser tests. It is not a same-shot "
            "experimental deuterium target and cannot close predictive tier 2."
        ),
    }


def lee_radpf_theory_model_scope_targets() -> dict[str, object]:
    """Return KR-backed Lee/RADPF theoretical model-scope targets."""
    source = "KnowledgeReference/lee_radpf_theory.md"
    return {
        "target_id": "lee_radpf_theory_model_scope_2008",
        "device": "generic Lee/RADPF",
        "model_role": "kr_reduced_model_scope_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "coupled_circuit_motion_equations": "13-166",
            "normalization_and_axial_scaling": "169-209,710-718",
            "snowplow_to_slug_model_transition": "1268-1296",
            "radial_current_mass_factors_and_elongation": "1401-1403,1443-1450",
            "radial_scaling_times": "2312-2387",
            "delay_and_reflected_shock_assumptions": "3292-3364",
            "radiation_and_self_absorption_limits": "3763-4004",
            "neutron_yield_terms_and_calibration": "4048-4104",
            "instability_and_expanded_column": "5323-5344",
        },
        "phase_semantics": {
            "axial": (
                "snowplow current sheath trajectory and speed are used for "
                "axial motion and a reasonable current profile"
            ),
            "radial": (
                "radial phase replaces infinitesimally thin snowplow behavior "
                "with a slug model: a shock front opens space for the magnetic "
                "piston current sheath"
            ),
            "reflected_shock": (
                "reflected shock starts when the inward radial shock reaches "
                "the axis and is assigned a constant fraction of the on-axis "
                "inward radial shock speed"
            ),
            "slow_compression": (
                "slow-compression and instability behavior are reduced-model "
                "extensions, not a first-principles kinetic pinch model"
            ),
            "expanded_column": (
                "pinch breakup is modeled as an expanded current column with "
                "uniform anode-to-cathode current"
            ),
        },
        "current_waveform_targets": {
            "external_circuit_and_sheath_motion_coupled": True,
            "plasma_resistance_ignored_for_em_drive_approximation": True,
            "tube_voltage_inductive_component_only_in_axial_and_radial_phases": True,
            "current_sheath_motion_and_position_affect_circuit_equation": True,
            "anomalous_resistance_can_create_unphysical_negative_voltage_spike": True,
            "piston_motion_frozen_or_inward_when_instability_resistance_introduced": True,
        },
        "phase_timing": {
            "alpha_interpreted_as_electrical_time_over_axial_transit_time": True,
            "alpha1_interpreted_as_axial_transit_over_radial_transit_time": True,
            "characteristic_axial_to_radial_time_ratio_typical": 40.0,
            "axial_transit_time_characteristically_over_radial_shock_time": 20.0,
            "reflected_shock_speed_fraction_of_on_axis_radial_shock_speed": 0.3,
            "communication_delay_time_expression": "(rp - rs) / SDS",
        },
        "temperature_targets": {
            "axial_temperature_only_deduced_from_trajectory_and_speed": True,
            "axial_snowplow_no_density_information": True,
            "radial_shock_temperature_computed_from_shock_speed": True,
            "slow_compression_temperature_estimated_from_energy_balance": True,
            "deuterium_radiation_collapse_critical_current_MA": 1.6,
            "neon_argon_line_radiation_can_reduce_critical_current_below_kA": 100.0,
        },
        "radiation_model_targets": {
            "spitzer_resistivity_used": True,
            "bremsstrahlung_loss_term": True,
            "recombination_loss_term": True,
            "line_loss_term": True,
            "plasma_self_absorption_correction": True,
            "volumetric_to_surface_emission_transition": True,
            "radiation_collapse_can_drive_radius_rapidly_to_small_values": True,
        },
        "neutron_yield_model_targets": {
            "thermonuclear_term_uses_density_volume_sigma_v_and_time": True,
            "beam_target_term_is_phenomenological": True,
            "beam_deuterons_from_diode_action_near_anode": True,
            "beam_voltage_proportional_to_Vmax": True,
            "beam_energy_used_for_cross_section_is_3_times_Vmax": True,
            "code_Vmax_kV_order_range": [20.0, 50.0],
            "experiment_beam_energy_keV_responsible_range": [50.0, 150.0],
            "small_lower_voltage_machine_beam_energy_keV_range": [30.0, 60.0],
            "empirical_yield_fit": "Yn = 9e10 * Ipinch^3.8",
            "empirical_fit_current_range_MA": [0.1, 1.0],
            "calibration_point_current_MA": 0.5,
            "calibration_point_yield_neutrons": 7.0e9,
        },
        "model_scope_limits": {
            "reduced_model_not_first_principles_kinetic_closure": True,
            "beam_target_yield_calibrated_to_experiment_not_predicted_ab_initio": True,
            "axial_snowplow_cannot_supply_density_without_extra_mechanisms": True,
            "radial_snowplow_singularity_avoided_by_slug_model_assumption": True,
            "instability_resistance_not_self_consistent_mhd_or_kinetic_model": True,
        },
        "uncertainty": {
            "source_is_theoretical_basis_not_same_shot_validation_dataset": True,
            "empirical_neutron_calibration_collapses_multiple_experiments": True,
            "line_extraction_contains_equation_formatting_artifacts": True,
            "no_digitized_current_or_neutron_trace_uncertainty": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "measured_current_trace",
            "measured_voltage_trace",
            "per_point_trace_uncertainty",
        ],
        "missing_for_full_tier2": [
            "same_device_phase_endpoint_measurements",
            "shock_front_and_piston_position_measurements",
            "instability_breakup_timing",
        ],
        "missing_for_full_tier4": [
            "measured_density_profile",
            "measured_temperature_profile",
            "measured_magnetic_field_profile",
            "radiation_transport_validation",
        ],
        "missing_for_full_tier5": [
            "mechanism_separated_neutron_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "detector_response_model",
            "independent_beam_target_calibration_validation",
        ],
        "validation_note": (
            "This Lee/RADPF theory source is a high-value reduced-model scope "
            "constraint. It defines what the Lee model assumes and calibrates, "
            "including the phenomenological beam-target yield term, but it is "
            "not a same-shot experimental validation dataset and cannot close "
            "predictive end-to-end DPF accuracy."
        ),
    }


def lee_2014_radiative_model_review_targets() -> dict[str, object]:
    """Return KR-backed Lee 2014 radiative-model equation targets."""
    source = "KnowledgeReference/lee-2014-plasma-focus-radiative-model.md"
    return {
        "target_id": "lee_2014_radiative_model_review",
        "device": "generic Lee 5-phase plasma focus",
        "model_role": "kr_radiative_lee_model_equation_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "citation_status_and_scope": "6-14",
            "phase_summary": "18-24",
            "axial_and_radial_equations": "30-88",
            "reflected_shock_equations": "88-117",
            "pinch_radiation_equations": "119-190",
            "closure_and_phase_termination": "196-200",
            "constants_and_conventions": "204-208",
            "provenance_notes": "212-216",
        },
        "phase_model_targets": {
            "phase_count": 5,
            "phases": [
                "axial",
                "radial_inward_shock",
                "radial_reflected_shock",
                "slow_compression_or_pinch",
                "expanded_column",
            ],
            "type_2_optional_phase_4a_anomalous_resistance": True,
            "axial_phase_snowplow_circuit_coupled": True,
            "radial_phase_slug_model": True,
            "slow_compression_includes_radiative_terms": True,
            "expanded_column_attains_anode_radius": True,
        },
        "current_waveform_targets": {
            "axial_motion_coupled_to_circuit_equation": True,
            "radial_circuit_equation_includes_dynamic_inductance_terms": True,
            "radial_phase_closed_equation_set": ["14", "15", "17", "19"],
            "reflected_shock_closed_equation_set": ["34", "35", "36", "37"],
            "mass_swept_factor_f_m": True,
            "current_factor_f_c": True,
            "radial_mass_factor_f_mr": True,
            "drive_or_speed_factor": "(I0/a) / sqrt(rho)",
        },
        "phase_timing": {
            "axial_phase_ends_when_current_sheath_reaches_anode_end": True,
            "radial_inward_phase_ends_when_shock_reaches_axis": True,
            "reflected_shock_speed_fraction_of_on_axis_inward_shock": 0.3,
            "reflected_shock_phase_ends_when_rs_reaches_piston": True,
            "pinch_phase_ends_after_small_disturbance_transit_time": True,
            "axial_normalized_time_increment": 0.001,
            "radial_normalized_time_increment": 0.00001,
        },
        "temperature_targets": {
            "reflected_shock_temperature_jump_factor_near": 2.0,
            "bennett_temperature_equation": "T = mu * I^2 * f_c^2 / (8*pi^2*k*DN*a^2*f_mr)",
            "gamma_H_D_T_He": 5.0 / 3.0,
            "gamma_molecular_gas": 7.0 / 5.0,
            "gamma_strongly_ionising_argon_example": 1.15,
        },
        "radiation_model_targets": {
            "joule_heating_term": "dQ_J/dt = R * I^2 * f_c^2",
            "spitzer_resistance_equation": "R = 1290 * Z * z_f / (pi * r_p^2 * T^(3/2))",
            "bremsstrahlung_loss_term": True,
            "line_radiation_loss_term": True,
            "total_power_gain_loss": "dQ/dt = dQ_J/dt + dQ_B/dt + dQ_L/dt",
            "self_absorption_correction": True,
            "surface_emission_transition": True,
            "radiation_collapse_from_severe_loss": True,
            "deuterium_radiation_collapse_critical_current_MA": 1.6,
            "neon_argon_critical_current_below_kA": 100.0,
            "corona_model_polynomials_for_non_light_gases": True,
            "line_radiation_gases": ["Ne", "N", "O", "Ar", "Kr", "Xe"],
        },
        "neutron_context": {
            "pinch_dynamics_and_yields_section_context": True,
            "no_neutron_yield_equation_extracted_in_local_markdown": True,
            "no_neutron_observable_targets": True,
        },
        "model_scope_limits": {
            "equation_review_not_same_shot_validation_dataset": True,
            "source_transcribes_pages_4_to_9_only": True,
            "pinch_equations_51_52_53_not_transcribed_in_local_extract": True,
            "no_digitized_current_voltage_trace": True,
            "no_density_temperature_diagnostic_comparison": True,
            "no_neutron_measurement_comparison": True,
        },
        "uncertainty": {
            "no_experimental_error_bars": True,
            "no_numerical_convergence_study_in_extract": True,
            "equation_transcription_contains_interpretation_notes": True,
            "not_a_calibration_dataset": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "measured_current_trace",
            "measured_voltage_trace",
            "waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "measured_phase_endpoint_times",
            "shock_and_piston_trajectory_diagnostics",
            "pinch_duration_uncertainty",
        ],
        "missing_for_full_tier4": [
            "measured_temperature_profile",
            "measured_density_profile",
            "measured_radiated_power_trace",
            "radiation_transport_validation",
        ],
        "missing_for_full_tier5": [
            "neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This peer-reviewed Lee 2014 source is a high-value equation and "
            "scope target for the radiative five-phase Lee model. It supports "
            "implementation checks for reduced-model phases and radiation-loss "
            "terms, but it is not an experimental predictive-validation packet."
        ),
    }


def pf1000_16kv_shot12581_phase_targets() -> dict[str, object]:
    """Return partial PF-1000 phase timing targets for shot 12581."""
    source = "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    return {
        "target_id": "pf1000_16kv_shot12581_phase_2021_akel",
        "validation_scope": "pf1000_16kv_2021_akel",
        "device": "PF-1000",
        "shot": "12581",
        "model_role": "kr_partial_phase_timing_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "device_and_diagnostics": "111-124",
            "derivative_dip_marker": "132-137",
            "lee_fit_procedure": "219-235",
            "shot_context_and_fit": "250-285",
            "table_context": "332-346",
        },
        "shot_context": {
            "bank_energy_kJ": 170.5,
            "voltage_kV": 16.0,
            "fill_pressure_torr": 1.2,
            "capacitance_uF": 1332.0,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.1,
            "anode_radius_cm": 11.55,
            "cathode_radius_cm": 16.0,
            "anode_length_cm": 48.0,
        },
        "phase_timing": {
            "current_dip_end_time_us": 8.0,
            "pinch_duration_ns": 212.0,
            "derivative_dip_from_breakdown_us": 7.0,
            "channel_timing_uncertainty_ns": [3.0, 5.0],
        },
        "phase_semantics": {
            "axial_phase_mass_swept_factor": "fm",
            "axial_phase_current_factor": "fc",
            "radial_phase_mass_swept_factor": "fmr",
            "radial_phase_current_factor": "fcr",
            "current_waveform_fit_drives_phase_dynamics": True,
            "pinch_duration_reported": True,
            "scope_note": (
                "These are Lee-model fitted phase semantics for PF-1000 16 kV "
                "shots; they are not complete measured axial/radial phase "
                "endpoint timings."
            ),
        },
        "lee_fit_parameters": {
            "fm": 0.17,
            "fc": 0.70,
            "fmr": 0.26,
            "fcr": 0.75,
        },
        "derived_outputs": {
            "peak_current_kA": 1165.0,
            "pinch_current_kA": 523.0,
            "axial_speed_cm_per_us": 10.5,
            "shock_speed_cm_per_us": 22.0,
            "piston_speed_cm_per_us": 18.0,
            "final_pinch_radius_cm": 2.3,
            "pinch_length_cm": 18.2,
            "vmax_kV": 30.0,
        },
        "missing_for_full_tier2": [
            "axial_rundown_end_time",
            "radial_transit_duration",
        ],
        "validation_note": (
            "This PF-1000 record can check current-dip/pinch timing and pinch "
            "duration only. It is partial tier-2 evidence until axial and "
            "radial transit targets are extracted for the same shot."
        ),
    }


def pf1000_16kv_current_waveform_targets() -> dict[str, object]:
    """Return typed PF-1000 16 kV measured-current waveform targets."""
    source = "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    return {
        "target_id": "pf1000_16kv_current_waveform_2021_akel",
        "validation_scope": "pf1000_16kv_2021_akel",
        "device": "PF-1000",
        "model_role": "kr_circuit_waveform_target",
        "validation_tier": 1,
        "source": source,
        "source_lines": {
            "device_and_current_measurement_context": "111-124",
            "lee_current_trace_fit_method": "217-236",
            "shot_12581_fit_context": "247-285",
            "measured_current_waveform_figures": "294-300",
            "table_context": "332-346",
        },
        "shot_context": {
            "bank_energy_kJ": 170.5,
            "voltage_kV": 16.0,
            "fill_pressure_torr_range": [1.05, 1.2],
            "capacitance_uF": 1332.0,
            "static_inductance_nH": 25.0,
            "anode_radius_cm": 11.55,
            "cathode_radius_cm": 16.0,
            "anode_length_cm": 48.0,
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "peak_current_kA_range": [1100.0, 1300.0],
            "shot_12581_peak_current_kA": 1165.0,
            "shot_12581_pinch_current_kA": 523.0,
            "fit_valid_until": "end_of_current_dip",
            "figures": ["1", "2", "3", "4"],
        },
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "per_point_current_uncertainty",
            "per_point_timing_uncertainty",
        ],
        "validation_note": (
            "The KR source establishes measured PF-1000 16 kV current-waveform "
            "targets and fit context, but this target does not yet include "
            "digitized point-by-point trace data. It can guide extraction and "
            "audit provenance, not close tier-1 waveform validation by itself."
        ),
    }


def pf1000_16kv_current_waveform_digitization_candidate_evidence(
    packet: Mapping[str, object] | None = None,
    *,
    target: dict[str, object] | None = None,
    base_path: str | Path = ".",
) -> dict[str, object]:
    """Report current-waveform digitization readiness for Akel Fig. 1.

    This is a data-readiness check. It does not compare a simulation waveform
    against the digitized trace and therefore does not close tier-1 validation.
    """
    from dpf.validation.digitization import (
        akel_fig1_draft_digitization_packet,
        digitization_verification_evidence,
        scientific_closure_digitization_status,
    )

    target = target or pf1000_16kv_current_waveform_targets()
    packet_source = "provided_packet"
    if packet is None:
        try:
            packet = akel_fig1_draft_digitization_packet(base_path=base_path)
            packet_source = "default_local_draft_packet"
        except FileNotFoundError:
            packet_source = "missing_default_local_draft_packet"

    task_id = "akel_2021_fig1_current_waveform_shot_12581"
    required_series = {"measured_current", "computed_current"}
    review_only_failures = {
        "independent_review_missing",
        "independent_review_metadata_missing",
        "review_packet_hash_missing",
        "review_packet_hash_mismatch",
        "review_source_hash_mismatch",
        "review_figure_image_hash_mismatch",
        "review_task_id_mismatch",
        "review_scope_mismatch",
        "reviewer_missing",
        "review_date_missing",
        "review_notes_missing",
        "review_decision_not_accepted",
        "review_status_not_accepted",
    }

    if packet is None:
        status_report = scientific_closure_digitization_status(None, base_path=base_path)
        packet_series: set[str] = set()
        task_status = {
            "status": "open",
            "missing_or_failed_checks": ["digitization_packet_missing"],
        }
    else:
        digitization_verification_evidence(packet, base_path=base_path)
        status_report = scientific_closure_digitization_status(
            [packet],
            base_path=base_path,
        )
        digitized_series = packet.get("digitized_series", [])
        packet_series = {
            str(series.get("name"))
            for series in digitized_series
            if isinstance(series, Mapping)
        } if isinstance(digitized_series, Sequence) else set()
        task_status = next(
            (
                item
                for item in status_report.get("task_statuses", [])
                if isinstance(item, Mapping) and item.get("task_id") == task_id
            ),
            {
                "status": "missing",
                "missing_or_failed_checks": ["digitization_task_status_missing"],
            },
        )

    missing_or_failed = sorted(
        str(check) for check in task_status.get("missing_or_failed_checks", [])
    )
    if task_status.get("status") == "accepted":
        readiness_status = "accepted_digitization_available"
    elif packet is None:
        readiness_status = "blocked_by_missing_packet"
    elif set(missing_or_failed).issubset(review_only_failures):
        readiness_status = "blocked_by_review"
    elif "overlay_residual_too_large" in missing_or_failed:
        readiness_status = "blocked_by_overlay_residual"
    else:
        readiness_status = "blocked_by_digitization_quality"

    verification = packet.get("verification", {}) if isinstance(packet, Mapping) else {}
    if not isinstance(verification, Mapping):
        verification = {}

    return {
        "passed": readiness_status == "accepted_digitization_available",
        "waveform_digitization_status": readiness_status,
        "target": "pf1000_16kv_current_waveform_digitization",
        "validation_scope": target.get("validation_scope", target["target_id"]),
        "model_role": "kr_current_waveform_digitization_readiness",
        "validation_tier": 1,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "task_id": task_id,
            "packet_source": packet_source,
            "required_series_present": required_series.issubset(packet_series),
            "available_series": sorted(packet_series),
            "missing_or_failed_checks": missing_or_failed,
            "overlay_rms_residual_px": verification.get("overlay_rms_residual_px"),
            "overlay_residual_status": verification.get("overlay_residual_status"),
            "review_status": verification.get("review_status"),
            "independent_review_count": verification.get(
                "independent_review_count",
            ),
            "digitization_status": status_report,
            "missing_for_full_tier1": [
                "independent_digitization_review",
                "packet_tied_review_metadata",
                "review_status_accepted",
                "per_point_current_uncertainty",
                "per_point_timing_uncertainty",
            ],
        },
        "validity_notes": {
            "data_readiness_only": (
                "This evidence only reports whether same-scope waveform "
                "digitization is accepted. It does not compare a simulation "
                "waveform against the trace."
            ),
            "draft_boundary": (
                "A draft packet with measured overlay residuals remains "
                "blocked until independent review accepts it."
            ),
        },
    }


def _digitized_series_by_name(packet: Mapping[str, object], name: str) -> Mapping[str, object] | None:
    digitized_series = packet.get("digitized_series", [])
    if not isinstance(digitized_series, Sequence):
        return None
    for series in digitized_series:
        if isinstance(series, Mapping) and series.get("name") == name:
            return series
    return None


def _unit_scale(unit: str, *, quantity: str) -> float:
    normalized = unit.strip().lower()
    if quantity == "time":
        if normalized in {"us", "microsecond", "microseconds"}:
            return 1.0
        if normalized in {"s", "sec", "second", "seconds"}:
            return 1.0e6
    if quantity == "current":
        if normalized in {"ma", "megaamp", "megaamps"}:
            return 1.0
        if normalized in {"ka", "kiloamp", "kiloamps"}:
            return 1.0e-3
        if normalized in {"a", "amp", "amps"}:
            return 1.0e-6
    raise ValueError(f"unsupported {quantity} unit: {unit!r}")


def _array_in_units(values: Sequence[object], unit: str, *, quantity: str) -> np.ndarray:
    scale = _unit_scale(unit, quantity=quantity)
    array = np.asarray(values, dtype=float) * scale
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{quantity} array must contain at least three finite points")
    return array


def _waveform_dip_metrics(time_us: np.ndarray, current_MA: np.ndarray) -> dict[str, object]:
    peak_index = int(np.argmax(current_MA))
    peak_current = float(current_MA[peak_index])
    if peak_index >= current_MA.size - 2 or peak_current <= 0.0:
        return {"dip_present": False, "reason": "no_post_peak_dip_window"}
    post_peak = current_MA[peak_index + 1 :]
    dip_index = int(np.argmin(post_peak)) + peak_index + 1
    dip_current = float(current_MA[dip_index])
    dip_depth_fraction = (peak_current - dip_current) / max(abs(peak_current), 1.0e-30)
    return {
        "dip_present": dip_depth_fraction > 0.05,
        "peak_time_us": float(time_us[peak_index]),
        "peak_current_MA": peak_current,
        "dip_time_us": float(time_us[dip_index]),
        "dip_current_MA": dip_current,
        "dip_depth_fraction": float(dip_depth_fraction),
    }


def pf1000_16kv_current_waveform_comparison_candidate_evidence(
    simulation_time: Sequence[object],
    simulation_current: Sequence[object],
    packet: Mapping[str, object] | None = None,
    *,
    target: dict[str, object] | None = None,
    base_path: str | Path = ".",
    time_unit: str = "us",
    current_unit: str = "MA",
    uncertainty: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare a simulated current waveform to accepted Akel Fig. 1 data.

    Draft or review-blocked digitization packets return a blocked status before
    any waveform metric is computed.
    """
    target = target or pf1000_16kv_current_waveform_targets()
    readiness = pf1000_16kv_current_waveform_digitization_candidate_evidence(
        packet,
        target=target,
        base_path=base_path,
    )
    if not readiness.get("passed"):
        return {
            "passed": False,
            "waveform_comparison_status": readiness["waveform_digitization_status"],
            "metrics_computed": False,
            "validation_scope": readiness["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": readiness["source"],
            "source_lines": readiness["source_lines"],
            "details": {
                "digitization_readiness": readiness,
                "missing_for_full_tier1": ["accepted_same_scope_digitized_waveform"],
            },
        }

    if packet is None:
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_missing_packet",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {"missing_for_full_tier1": ["digitization_packet_missing"]},
        }

    if packet.get("validation_scope") != target["validation_scope"]:
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_scope_mismatch",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {
                "packet_validation_scope": packet.get("validation_scope"),
                "target_validation_scope": target["validation_scope"],
            },
        }

    uncertainty = uncertainty or {}
    required_uncertainty = {"current_MA", "time_us"}
    missing_uncertainty = sorted(required_uncertainty - set(uncertainty))
    if missing_uncertainty:
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_missing_uncertainty",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {
                "missing_uncertainty": missing_uncertainty,
                "required_uncertainty": sorted(required_uncertainty),
            },
        }

    measured = _digitized_series_by_name(packet, "measured_current")
    if measured is None:
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_missing_measured_series",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {"missing_series": "measured_current"},
        }

    try:
        measured_t_us = _array_in_units(
            measured.get("x", []), str(measured.get("x_unit", "")), quantity="time"
        )
        measured_i_MA = _array_in_units(
            measured.get("y", []), str(measured.get("y_unit", "")), quantity="current"
        )
        sim_t_us = _array_in_units(simulation_time, time_unit, quantity="time")
        sim_i_MA = _array_in_units(simulation_current, current_unit, quantity="current")
    except ValueError as exc:
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_malformed_waveform",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {"error": str(exc)},
        }

    if sim_t_us.min() > measured_t_us.min() or sim_t_us.max() < measured_t_us.max():
        return {
            "passed": False,
            "waveform_comparison_status": "blocked_by_time_range",
            "metrics_computed": False,
            "validation_scope": target["validation_scope"],
            "model_role": "kr_current_waveform_comparison_candidate",
            "validation_tier": 1,
            "source": target["source"],
            "source_lines": target["source_lines"],
            "details": {
                "simulation_time_us_range": [float(sim_t_us.min()), float(sim_t_us.max())],
                "measured_time_us_range": [float(measured_t_us.min()), float(measured_t_us.max())],
            },
        }

    sim_interp_MA = np.interp(measured_t_us, sim_t_us, sim_i_MA)
    rms_error = float(np.sqrt(np.mean((sim_interp_MA - measured_i_MA) ** 2)))
    normalization = max(float(np.ptp(measured_i_MA)), 1.0e-30)
    waveform_nrmse = rms_error / normalization

    measured_dip = _waveform_dip_metrics(measured_t_us, measured_i_MA)
    simulated_dip = _waveform_dip_metrics(measured_t_us, sim_interp_MA)
    waveform_nrmse_tolerance = float(uncertainty.get("waveform_nrmse_tolerance", 0.25))
    dip_depth_tolerance = float(uncertainty.get("dip_depth_fraction_tolerance", 0.20))
    dip_timing_tolerance = float(uncertainty.get("dip_timing_us_tolerance", 0.75))

    metric_failures: list[str] = []
    if waveform_nrmse > waveform_nrmse_tolerance:
        metric_failures.append("waveform_nrmse_too_large")
    if not measured_dip.get("dip_present"):
        metric_failures.append("measured_current_dip_missing")
    if not simulated_dip.get("dip_present"):
        metric_failures.append("simulated_current_dip_missing")
    if measured_dip.get("dip_present") and simulated_dip.get("dip_present"):
        dip_depth_error = abs(
            float(simulated_dip["dip_depth_fraction"])
            - float(measured_dip["dip_depth_fraction"])
        )
        dip_timing_error_us = abs(
            float(simulated_dip["dip_time_us"]) - float(measured_dip["dip_time_us"])
        )
        if dip_depth_error > dip_depth_tolerance:
            metric_failures.append("current_dip_depth_error_too_large")
        if dip_timing_error_us > dip_timing_tolerance:
            metric_failures.append("current_dip_timing_error_too_large")
    else:
        dip_depth_error = None
        dip_timing_error_us = None

    return {
        "passed": not metric_failures,
        "waveform_comparison_status": "passed" if not metric_failures else "failed",
        "metrics_computed": True,
        "validation_scope": target["validation_scope"],
        "model_role": "kr_current_waveform_comparison_candidate",
        "validation_tier": 1,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "waveform_nrmse": waveform_nrmse,
            "waveform_nrmse_tolerance": waveform_nrmse_tolerance,
            "measured_dip": measured_dip,
            "simulated_dip": simulated_dip,
            "dip_depth_error": dip_depth_error,
            "dip_depth_fraction_tolerance": dip_depth_tolerance,
            "dip_timing_error_us": dip_timing_error_us,
            "dip_timing_us_tolerance": dip_timing_tolerance,
            "missing_or_failed_checks": metric_failures,
            "uncertainty": dict(uncertainty),
        },
    }


def _pf1000_16kv_akel_shot_table_rows() -> list[dict[str, float | int]]:
    """Return merged Akel 2021 Table 1 and Table 2 rows."""
    return [
        {
            "shot": 12581,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.1,
            "peak_current_kA": 1165.0,
            "pinch_current_kA": 523.0,
            "fm": 0.17,
            "fc": 0.70,
            "fmr": 0.26,
            "fcr": 0.75,
            "axial_speed_cm_per_us": 10.5,
            "shock_speed_cm_per_us": 22.2,
            "piston_speed_cm_per_us": 18.0,
            "pinch_density_1e23_per_m3": 1.7,
            "pinch_radius_cm": 2.40,
            "pinch_length_cm": 18.2,
            "computed_neutron_yield_n": 6.14e9,
            "measured_neutron_yield_n": 6.1e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12582,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.2,
            "peak_current_kA": 1303.0,
            "pinch_current_kA": 389.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.10,
            "fcr": 0.43,
            "axial_speed_cm_per_us": 11.1,
            "shock_speed_cm_per_us": 24.6,
            "piston_speed_cm_per_us": 18.6,
            "pinch_density_1e23_per_m3": 0.8,
            "pinch_radius_cm": 2.17,
            "pinch_length_cm": 16.6,
            "computed_neutron_yield_n": 1.51e9,
            "measured_neutron_yield_n": 1.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12583,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.0,
            "peak_current_kA": 1335.0,
            "pinch_current_kA": 390.0,
            "fm": 0.21,
            "fc": 0.70,
            "fmr": 0.10,
            "fcr": 0.42,
            "axial_speed_cm_per_us": 10.8,
            "shock_speed_cm_per_us": 24.6,
            "piston_speed_cm_per_us": 18.7,
            "pinch_density_1e23_per_m3": 0.8,
            "pinch_radius_cm": 2.17,
            "pinch_length_cm": 16.6,
            "computed_neutron_yield_n": 1.54e9,
            "measured_neutron_yield_n": 1.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12584,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.3,
            "peak_current_kA": 1228.0,
            "pinch_current_kA": 404.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.10,
            "fcr": 0.50,
            "axial_speed_cm_per_us": 10.5,
            "shock_speed_cm_per_us": 26.1,
            "piston_speed_cm_per_us": 20.4,
            "pinch_density_1e23_per_m3": 0.7,
            "pinch_radius_cm": 2.24,
            "pinch_length_cm": 16.6,
            "computed_neutron_yield_n": 1.55e9,
            "measured_neutron_yield_n": 1.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12586,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.1,
            "peak_current_kA": 1237.0,
            "pinch_current_kA": 365.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.08,
            "fcr": 0.43,
            "axial_speed_cm_per_us": 10.7,
            "shock_speed_cm_per_us": 25.9,
            "piston_speed_cm_per_us": 19.8,
            "pinch_density_1e23_per_m3": 0.6,
            "pinch_radius_cm": 2.18,
            "pinch_length_cm": 16.3,
            "computed_neutron_yield_n": 1.0e9,
            "measured_neutron_yield_n": 1.0e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12587,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.5,
            "peak_current_kA": 1135.0,
            "pinch_current_kA": 413.0,
            "fm": 0.17,
            "fc": 0.70,
            "fmr": 0.13,
            "fcr": 0.57,
            "axial_speed_cm_per_us": 10.4,
            "shock_speed_cm_per_us": 24.0,
            "piston_speed_cm_per_us": 19.0,
            "pinch_density_1e23_per_m3": 0.9,
            "pinch_radius_cm": 2.28,
            "pinch_length_cm": 17.0,
            "computed_neutron_yield_n": 1.83e9,
            "measured_neutron_yield_n": 1.8e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12588,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.0,
            "peak_current_kA": 1285.0,
            "pinch_current_kA": 312.0,
            "fm": 0.24,
            "fc": 0.70,
            "fmr": 0.03,
            "fcr": 0.35,
            "axial_speed_cm_per_us": 9.8,
            "shock_speed_cm_per_us": 35.7,
            "piston_speed_cm_per_us": 27.6,
            "pinch_density_1e23_per_m3": 0.2,
            "pinch_radius_cm": 2.13,
            "pinch_length_cm": 15.4,
            "computed_neutron_yield_n": 3.2e8,
            "measured_neutron_yield_n": 3.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12589,
            "fill_pressure_torr": 1.20,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.3,
            "peak_current_kA": 1321.0,
            "pinch_current_kA": 312.0,
            "fm": 0.22,
            "fc": 0.70,
            "fmr": 0.03,
            "fcr": 0.33,
            "axial_speed_cm_per_us": 10.5,
            "shock_speed_cm_per_us": 35.3,
            "piston_speed_cm_per_us": 26.9,
            "pinch_density_1e23_per_m3": 0.3,
            "pinch_radius_cm": 2.10,
            "pinch_length_cm": 15.3,
            "computed_neutron_yield_n": 3.3e8,
            "measured_neutron_yield_n": 3.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12590,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.4,
            "peak_current_kA": 1314.0,
            "pinch_current_kA": 368.0,
            "fm": 0.24,
            "fc": 0.70,
            "fmr": 0.10,
            "fcr": 0.40,
            "axial_speed_cm_per_us": 10.4,
            "shock_speed_cm_per_us": 24.1,
            "piston_speed_cm_per_us": 18.2,
            "pinch_density_1e23_per_m3": 0.7,
            "pinch_radius_cm": 2.15,
            "pinch_length_cm": 16.5,
            "computed_neutron_yield_n": 12.0e8,
            "measured_neutron_yield_n": 10.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12592,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.1,
            "peak_current_kA": 1151.0,
            "pinch_current_kA": 438.0,
            "fm": 0.18,
            "fc": 0.70,
            "fmr": 0.17,
            "fcr": 0.60,
            "axial_speed_cm_per_us": 10.8,
            "shock_speed_cm_per_us": 23.9,
            "piston_speed_cm_per_us": 18.9,
            "pinch_density_1e23_per_m3": 1.0,
            "pinch_radius_cm": 2.30,
            "pinch_length_cm": 17.4,
            "computed_neutron_yield_n": 2.53e9,
            "measured_neutron_yield_n": 2.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12593,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.9,
            "peak_current_kA": 1192.0,
            "pinch_current_kA": 262.0,
            "fm": 0.22,
            "fc": 0.70,
            "fmr": 0.03,
            "fcr": 0.30,
            "axial_speed_cm_per_us": 10.2,
            "shock_speed_cm_per_us": 31.2,
            "piston_speed_cm_per_us": 23.3,
            "pinch_density_1e23_per_m3": 0.2,
            "pinch_radius_cm": 2.05,
            "pinch_length_cm": 15.3,
            "computed_neutron_yield_n": 1.7e8,
            "measured_neutron_yield_n": 1.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12594,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.4,
            "peak_current_kA": 1224.0,
            "pinch_current_kA": 332.0,
            "fm": 0.22,
            "fc": 0.70,
            "fmr": 0.04,
            "fcr": 0.40,
            "axial_speed_cm_per_us": 10.4,
            "shock_speed_cm_per_us": 35.7,
            "piston_speed_cm_per_us": 27.9,
            "pinch_density_1e23_per_m3": 0.3,
            "pinch_radius_cm": 2.18,
            "pinch_length_cm": 15.6,
            "computed_neutron_yield_n": 4.2e8,
            "measured_neutron_yield_n": 4.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12595,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.6,
            "peak_current_kA": 1269.0,
            "pinch_current_kA": 391.0,
            "fm": 0.21,
            "fc": 0.70,
            "fmr": 0.12,
            "fcr": 0.45,
            "axial_speed_cm_per_us": 11.0,
            "shock_speed_cm_per_us": 24.8,
            "piston_speed_cm_per_us": 18.9,
            "pinch_density_1e23_per_m3": 0.8,
            "pinch_radius_cm": 2.19,
            "pinch_length_cm": 16.8,
            "computed_neutron_yield_n": 1.53e9,
            "measured_neutron_yield_n": 1.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12596,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.5,
            "peak_current_kA": 1131.0,
            "pinch_current_kA": 434.0,
            "fm": 0.18,
            "fc": 0.70,
            "fmr": 0.12,
            "fcr": 0.62,
            "axial_speed_cm_per_us": 10.6,
            "shock_speed_cm_per_us": 28.2,
            "piston_speed_cm_per_us": 22.8,
            "pinch_density_1e23_per_m3": 0.7,
            "pinch_radius_cm": 2.33,
            "pinch_length_cm": 16.9,
            "computed_neutron_yield_n": 1.85e9,
            "measured_neutron_yield_n": 1.8e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12597,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.1,
            "peak_current_kA": 1231.0,
            "pinch_current_kA": 449.0,
            "fm": 0.20,
            "fc": 0.70,
            "fmr": 0.15,
            "fcr": 0.57,
            "axial_speed_cm_per_us": 10.8,
            "shock_speed_cm_per_us": 25.8,
            "piston_speed_cm_per_us": 20.4,
            "pinch_density_1e23_per_m3": 0.9,
            "pinch_radius_cm": 2.30,
            "pinch_length_cm": 17.2,
            "computed_neutron_yield_n": 2.64e9,
            "measured_neutron_yield_n": 2.6e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12598,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 6.0,
            "peak_current_kA": 1169.0,
            "pinch_current_kA": 302.0,
            "fm": 0.20,
            "fc": 0.70,
            "fmr": 0.04,
            "fcr": 0.37,
            "axial_speed_cm_per_us": 10.6,
            "shock_speed_cm_per_us": 32.0,
            "piston_speed_cm_per_us": 24.5,
            "pinch_density_1e23_per_m3": 0.3,
            "pinch_radius_cm": 2.13,
            "pinch_length_cm": 15.6,
            "computed_neutron_yield_n": 3.1e8,
            "measured_neutron_yield_n": 3.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12599,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.3,
            "peak_current_kA": 1223.0,
            "pinch_current_kA": 304.0,
            "fm": 0.21,
            "fc": 0.70,
            "fmr": 0.04,
            "fcr": 0.35,
            "axial_speed_cm_per_us": 10.6,
            "shock_speed_cm_per_us": 32.0,
            "piston_speed_cm_per_us": 24.3,
            "pinch_density_1e23_per_m3": 0.3,
            "pinch_radius_cm": 2.11,
            "pinch_length_cm": 15.6,
            "computed_neutron_yield_n": 3.1e8,
            "measured_neutron_yield_n": 3.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12600,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.1,
            "peak_current_kA": 1328.0,
            "pinch_current_kA": 296.0,
            "fm": 0.24,
            "fc": 0.70,
            "fmr": 0.04,
            "fcr": 0.30,
            "axial_speed_cm_per_us": 10.8,
            "shock_speed_cm_per_us": 30.5,
            "piston_speed_cm_per_us": 22.6,
            "pinch_density_1e23_per_m3": 0.3,
            "pinch_radius_cm": 2.05,
            "pinch_length_cm": 15.5,
            "computed_neutron_yield_n": 3.3e8,
            "measured_neutron_yield_n": 3.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12601,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.8,
            "peak_current_kA": 1263.0,
            "pinch_current_kA": 348.0,
            "fm": 0.22,
            "fc": 0.70,
            "fmr": 0.06,
            "fcr": 0.40,
            "axial_speed_cm_per_us": 10.7,
            "shock_speed_cm_per_us": 31.7,
            "piston_speed_cm_per_us": 24.5,
            "pinch_density_1e23_per_m3": 0.4,
            "pinch_radius_cm": 2.17,
            "pinch_length_cm": 15.9,
            "computed_neutron_yield_n": 6.2e8,
            "measured_neutron_yield_n": 6.0e8,
            "measured_neutron_yield_uncertainty_n": 0.2e8,
        },
        {
            "shot": 12602,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.2,
            "peak_current_kA": 1214.0,
            "pinch_current_kA": 422.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.14,
            "fcr": 0.53,
            "axial_speed_cm_per_us": 11.1,
            "shock_speed_cm_per_us": 24.8,
            "piston_speed_cm_per_us": 19.3,
            "pinch_density_1e23_per_m3": 0.9,
            "pinch_radius_cm": 2.26,
            "pinch_length_cm": 17.1,
            "computed_neutron_yield_n": 2.11e9,
            "measured_neutron_yield_n": 2.0e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12603,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 5.3,
            "peak_current_kA": 1207.0,
            "pinch_current_kA": 598.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.35,
            "fcr": 0.85,
            "axial_speed_cm_per_us": 11.0,
            "shock_speed_cm_per_us": 23.6,
            "piston_speed_cm_per_us": 19.5,
            "pinch_density_1e23_per_m3": 1.9,
            "pinch_radius_cm": 2.46,
            "pinch_length_cm": 18.8,
            "computed_neutron_yield_n": 11.1e9,
            "measured_neutron_yield_n": 11.2e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12604,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.6,
            "peak_current_kA": 1261.0,
            "pinch_current_kA": 415.0,
            "fm": 0.20,
            "fc": 0.70,
            "fmr": 0.18,
            "fcr": 0.48,
            "axial_speed_cm_per_us": 11.2,
            "shock_speed_cm_per_us": 21.1,
            "piston_speed_cm_per_us": 15.9,
            "pinch_density_1e23_per_m3": 1.2,
            "pinch_radius_cm": 2.20,
            "pinch_length_cm": 17.5,
            "computed_neutron_yield_n": 2.54e9,
            "measured_neutron_yield_n": 2.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12605,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.7,
            "peak_current_kA": 1241.0,
            "pinch_current_kA": 489.0,
            "fm": 0.19,
            "fc": 0.70,
            "fmr": 0.27,
            "fcr": 0.61,
            "axial_speed_cm_per_us": 11.4,
            "shock_speed_cm_per_us": 20.9,
            "piston_speed_cm_per_us": 16.3,
            "pinch_density_1e23_per_m3": 1.7,
            "pinch_radius_cm": 2.30,
            "pinch_length_cm": 18.3,
            "computed_neutron_yield_n": 5.52e9,
            "measured_neutron_yield_n": 5.5e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
        {
            "shot": 12606,
            "fill_pressure_torr": 1.05,
            "static_inductance_nH": 25.0,
            "short_circuit_resistance_mohm": 4.5,
            "peak_current_kA": 1268.0,
            "pinch_current_kA": 457.0,
            "fm": 0.20,
            "fc": 0.70,
            "fmr": 0.24,
            "fcr": 0.54,
            "axial_speed_cm_per_us": 11.2,
            "shock_speed_cm_per_us": 20.4,
            "piston_speed_cm_per_us": 15.5,
            "pinch_density_1e23_per_m3": 1.6,
            "pinch_radius_cm": 2.25,
            "pinch_length_cm": 18.0,
            "computed_neutron_yield_n": 4.20e9,
            "measured_neutron_yield_n": 4.1e9,
            "measured_neutron_yield_uncertainty_n": 0.2e9,
        },
    ]


def _pf1000_16kv_akel_pressure_summaries(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for pressure in (1.20, 1.05):
        pressure_rows = [
            row for row in rows
            if math.isclose(float(row["fill_pressure_torr"]), pressure)
        ]
        count = len(pressure_rows)
        summaries[f"{pressure:.2f}_torr"] = {
            "shot_count": count,
            "mean_peak_current_kA": sum(
                float(row["peak_current_kA"]) for row in pressure_rows
            ) / count,
            "mean_pinch_current_kA": sum(
                float(row["pinch_current_kA"]) for row in pressure_rows
            ) / count,
            "mean_computed_neutron_yield_n": sum(
                float(row["computed_neutron_yield_n"]) for row in pressure_rows
            ) / count,
            "mean_measured_neutron_yield_n": sum(
                float(row["measured_neutron_yield_n"]) for row in pressure_rows
            ) / count,
        }
    return summaries


def pf1000_16kv_akel_table_targets() -> dict[str, object]:
    """Return typed PF-1000 16 kV scalar/yield table targets from Akel 2021."""
    source = "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    rows = _pf1000_16kv_akel_shot_table_rows()
    return {
        "target_id": "pf1000_16kv_shot_table_2021_akel",
        "validation_scope": "pf1000_16kv_2021_akel",
        "device": "PF-1000",
        "model_role": "kr_scalar_current_pinch_yield_table_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "fit_and_yield_context": "306-327",
            "table_1_current_and_fit_rows": "330-583",
            "table_2_pinch_and_yield_rows": "584-837",
        },
        "shot_context": {
            "bank_energy_kJ": 170.5,
            "voltage_kV": 16.0,
            "fill_pressures_torr": [1.20, 1.05],
            "working_gas": "deuterium",
            "shot_count": len(rows),
        },
        "table_extraction_verification": {
            "table_1_row_count": 24,
            "table_2_row_count": 24,
            "merged_row_count": len(rows),
            "table_1_table_2_shot_ids_match": True,
            "source_markdown_pdf_parity_verified": True,
            "source_pdf_sha256": (
                "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
            ),
            "parity_verifier": "scripts/verify_kr_pdf_parity.py",
        },
        "current_waveform_targets": {
            "table_scalar_current_available": True,
            "shot_count": len(rows),
            "peak_current_kA_range": [
                min(float(row["peak_current_kA"]) for row in rows),
                max(float(row["peak_current_kA"]) for row in rows),
            ],
            "pinch_current_kA_range": [
                min(float(row["pinch_current_kA"]) for row in rows),
                max(float(row["pinch_current_kA"]) for row in rows),
            ],
            "fit_parameters_by_shot_available": True,
        },
        "pinch_geometry_targets": {
            "axial_speed_cm_per_us_available": True,
            "shock_speed_cm_per_us_available": True,
            "piston_speed_cm_per_us_available": True,
            "pinch_density_1e23_per_m3_available": True,
            "pinch_radius_cm_available": True,
            "pinch_length_cm_available": True,
        },
        "neutron_yield_targets": {
            "computed_and_measured_per_shot_available": True,
            "yield_unit": "neutrons_per_shot",
            "measured_yield_uncertainty_per_row_available": True,
            "pressure_group_summaries": _pf1000_16kv_akel_pressure_summaries(rows),
        },
        "uncertainty": {
            "measured_neutron_yield_uncertainty_per_row_available": True,
            "measured_neutron_yield_uncertainty_n_range": [
                min(
                    float(row["measured_neutron_yield_uncertainty_n"])
                    for row in rows
                ),
                max(
                    float(row["measured_neutron_yield_uncertainty_n"])
                    for row in rows
                ),
            ],
            "shot_to_shot_variation_explicitly_discussed": True,
            "uncertainty_scope": (
                "Printed scalar measured-yield uncertainty and shot-to-shot "
                "variation context only."
            ),
            "missing_uncertainty_components": [
                "digitized_current_trace_uncertainty",
                "systematic_detector_response_uncertainty",
                "model_form_uncertainty",
                "input_parameter_covariance",
                "blind_prediction_acceptance_rule",
            ],
        },
        "partial_target_groups": [
            "uncertainty",
        ],
        "reported_parameter_means": {
            "fm": 0.20,
            "fc": 0.70,
            "fmr": 0.12,
            "fcr": 0.48,
        },
        "shot_rows": rows,
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "per_point_current_uncertainty",
            "per_point_timing_uncertainty",
        ],
        "missing_for_predictive_neutron_yield_validation": [
            "independent_detector_response_model_for_each_shot",
            "shot_resolved_systematic_yield_uncertainty",
            "predeclared_acceptance_tolerance_for_blind_prediction",
        ],
        "missing_for_full_tier5": [
            "neutron_pulse_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "Akel 2021 Tables 1 and 2 now provide a row-level PF-1000 16 kV "
            "scalar target set for fitted current parameters, pinch geometry, "
            "and computed/measured neutron yield. This is stronger than figure "
            "context alone, but it still cannot close waveform validation or "
            "high-fidelity neutron prediction without digitized traces, detector "
            "response, spectrum, anisotropy, and timing evidence."
        ),
    }


def pf1000_full_energy_phase_context_targets() -> dict[str, object]:
    """Return PF-1000 full-energy phase context from KR paper I."""
    source = "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md"
    return {
        "target_id": "pf1000_full_energy_phase_context_2007_gribkov",
        "validation_scope": "pf1000_full_energy_2007_gribkov_scholz",
        "device": "PF-1000",
        "model_role": "kr_partial_phase_timing_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "diagnostics_and_temporal_resolution": "36-50",
            "generic_dpf_phase_semantics": "59-145",
            "operating_regime_and_neutron_pulses": "642-678",
            "compression_current_dip_temperature_density": "720-750",
            "confinement_time_context": "760-780",
            "upper_energy_summary": "1238-1260",
        },
        "shot_context": {
            "device": "PF-1000",
            "bank_energy_kJ_max": 850.0,
            "fill_pressure_torr_range": [2.0, 4.0],
            "discharge_current_MA_range": [2.5, 3.0],
            "working_gas": "deuterium",
            "configuration": "long cathode rods with smooth flush-mounted anode head",
        },
        "phase_semantics": {
            "breakdown": "surface discharge along the cylindrical insulator",
            "inverse_pinch": "plasma sheath expands from the insulator to cathode bars",
            "rundown": (
                "MHD plasma-current sheath accelerates along the anode and "
                "toward the chamber axis."
            ),
            "first_compression": (
                "shock convergence and maximum plasma compression on the Z-axis "
                "with current, dense-plasma, and bright-plasma pinch structures."
            ),
            "current_abruption": (
                "post-compression disturbance associated with the later, usually "
                "larger neutron pulse."
            ),
        },
        "phase_timing": {
            "max_compression_before_current_dip_ns": 100.0,
            "max_compression_after_current_max_us": 2.0,
            "pinch_confinement_time_ns": 150.0,
            "neutron_pulse_fwhm_ns": 150.0,
            "implosion_speed_cm_per_s_range": [2.0e7, 5.0e7],
        },
        "missing_for_full_tier2": [
            "digitized_current_and_derivative_traces",
            "breakdown_to_rundown_absolute_timing",
            "radial_transit_start_and_end_times",
            "per-shot phase timing uncertainty",
        ],
        "validation_note": (
            "Paper I gives PF-1000 full-energy phase semantics and several timing "
            "relations, including compression relative to current maximum and "
            "current dip. It is still a partial phase-timing target because the "
            "KR text does not provide complete digitized phase endpoints for a "
            "single shot."
        ),
    }


def pf1000_full_energy_neutron_spatial_targets() -> dict[str, object]:
    """Return PF-1000 full-energy neutron/spatial context from KR paper II."""
    source = "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md"
    return {
        "target_id": "pf1000_full_energy_neutron_spatial_2007_scholz",
        "validation_scope": "pf1000_full_energy_2007_gribkov_scholz",
        "device": "PF-1000",
        "model_role": "kr_multi_observable_partial_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "detectors_and_calibration": "320-386",
            "current_yield_anisotropy": "392-467",
            "pulse_width_scatter_limits": "465-532",
            "tof_spectrum_timing": "530-565,585-635",
            "pinch_spatial_estimates": "716-820",
            "current_waveform_optimization_limits": "1467-1558",
        },
        "shot_context": {
            "device": "PF-1000",
            "bank_energy_kJ": 810.0,
            "shot_3121_bank_energy_MJ": 0.810,
            "shot_3121_pressure_Pa": 465.0,
            "shot_3121_voltage_kV": 35.0,
            "fill_pressure_torr_range": [2.0, 4.0],
            "working_gas": "deuterium",
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "total_current_typical_MA_range": [2.5, 2.6],
            "total_current_best_MA": 3.0,
            "expected_scaling_current_MA": 6.0,
            "estimated_average_pinch_current_MA": 2.0,
            "current_plateau_after_us": 3.5,
            "current_peculiarity_or_dip_time_us": 8.0,
            "anode_shortening_predicted_current_gain_fraction_min": 0.30,
        },
        "spatial_density_targets": {
            "first_compression_ion_density_cm3": 0.8e19,
            "second_neutron_pulse_ion_density_cm3": 2.0e18,
        },
        "magnetic_field_targets": {
            "first_compression_azimuthal_Bmax_MG": 2.0,
            "compressed_Bz_order_G": 1.0e5,
            "Bmax_estimate_current_MA": 2.0,
            "Bmax_estimate_radius_cm": 0.45,
        },
        "spatial_geometry_targets": {
            "first_neutron_pulse_pinch_radius_cm": 0.45,
            "first_neutron_pulse_pinch_height_cm": 10.0,
            "second_neutron_pulse_radius_cm": 1.0,
            "pinch_confinement_time_ns": 150.0,
            "effective_expansion_velocity_cm_per_s_max": 0.5e7,
        },
        "temperature_targets": {
            "first_compression_estimated_ion_temperature_keV": 1.3,
            "second_neutron_pulse_effective_ion_temperature_min_keV": 4.0,
            "direct_ion_temperature_measured": False,
        },
        "event_sequence": [
            {
                "event": "first_neutron_pulse",
                "mechanism": "first_compression",
                "timing_note": (
                    "After time-of-flight correction, first HXR and neutron "
                    "pulses nearly coincide and align with maximum compression."
                ),
                "fwhm_ns": 150.0,
                "required": True,
            },
            {
                "event": "second_neutron_pulse",
                "mechanism": "post_abruption_beam_target_or_second_compression",
                "timing_note": (
                    "Second neutron pulse is usually four to ten times larger "
                    "and peaks later than the second HXR pulse."
                ),
                "second_to_first_amplitude_ratio_range": [4.0, 10.0],
                "required": True,
            },
        ],
        "detector_tof": {
            "detector_distance_m": 7.0,
            "hxr_time_of_flight_ns": 23.3,
            "neutron_time_of_flight_2p45_MeV_ns": 323.0,
        },
        "neutron_yield_targets": {
            "yield_range_neutrons_per_shot": [5.0e10, 2.0e11],
            "max_yield_neutrons_per_shot": 6.0e11,
            "shot_3121_activation_anisotropy_available": True,
            "bubble_detector_cross_check_angle_deg": 90.0,
            "same_scope_detector_response_required_for_predictive_yield": True,
        },
        "activation_requirements": {
            "activation_counter_materials": ["silver", "indium"],
            "bubble_detector_cross_check_angle_deg": 90.0,
            "silver_counter_angles_deg": [0.0, 30.0, 60.0, 90.0, 150.0],
            "calibration_source": "AmBe neutron source inside the DPF chamber",
        },
        "tof_requirements": {
            "scintillator_pm_distance_m": 7.0,
            "observed_angles_deg": [0.0, 90.0],
            "hxr_time_of_flight_ns": 23.3,
            "neutron_time_of_flight_2p45_MeV_ns": 323.0,
        },
        "response_model_requirements": [
            "activation_counter_calibration",
            "time_of_flight_correction",
            "xray_neutron_delay_correction",
            "room_scatter_or_background_assessment",
            "detector_tail_distortion_assessment",
        ],
        "spectral_targets": {
            "first_pulse": "neutron spectrum centered at 2.45 MeV",
            "second_pulse": (
                "head-on neutrons have higher energy than side-on neutrons"
            ),
        },
        "anisotropy_targets": {
            "shot_3121_Y0_over_Y90": 1.8,
            "shot_3121_Y180_over_Y90": 0.65,
            "yield_range_neutrons_per_shot": [5.0e10, 2.0e11],
            "max_yield_neutrons_per_shot": 6.0e11,
        },
        "uncertainty": {
            "bubble_detector_relative_lower_at_90deg": 0.30,
            "scattered_neutron_environment_affects_tail_shapes": True,
            "direct_ion_temperature_measured": False,
            "pinch_current_directly_measured": False,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "per_point_current_uncertainty",
            "per_point_timing_uncertainty",
            "direct_pinch_current_measurement",
        ],
        "missing_for_full_tier4": [
            "direct_experimental_temperature_diagnostic",
            "same-shot calibrated magnetic-field map",
            "same-shot density uncertainty",
        ],
        "missing_for_full_tier5": [
            "neutron_field_transport_or_room_scatter_response_model",
            "digitized_neutron_pulse_traces",
            "digitized_neutron_spectra",
            "yield_calibration_uncertainty",
            "fast_ion_distribution_uncertainty",
        ],
        "validation_note": (
            "Paper II gives a high-value PF-1000 full-energy target bundle for "
            "current, neutron timing, neutron anisotropy, spatial density, "
            "magnetic-field estimates, and temperature estimates. It is not a "
            "closed validation dataset: current traces and neutron histories "
            "are not digitized here, the pinch current and ion temperature were "
            "not directly measured, and the source explicitly identifies room "
            "scatter as a limit on neutron-pulse interpretation."
        ),
    }


def pf1000_cikhardtova_linear_density_motion_targets() -> dict[str, object]:
    """Return PF-1000 shot 9881 linear-density and motion targets."""
    source = "KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md"
    return {
        "target_id": "pf1000_linear_density_motion_2015_cikhardtova",
        "validation_scope": "pf1000_shot9881_linear_density_2015_cikhardtova",
        "device": "PF-1000",
        "model_role": "kr_interferometry_linear_density_motion_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "diagnostics_and_timing_uncertainty": "90-110",
            "shot_and_interferogram_context": "112-123",
            "linear_density_formula": "124-140",
            "profile_motion_description": "141-158",
            "linear_density_and_axial_lobule": "200-218",
            "summary_velocity_targets": "241-254",
        },
        "shot_context": {
            "shot": 9881,
            "time_zero": "minimum_of_current_derivative",
            "laser_wavelength_nm": 527.0,
            "laser_pulse_duration_ns_less_than": 1.0,
            "interferometer": "Mach-Zehnder",
            "beam_count": 15,
            "beam_interval_ns_range": [10.0, 20.0],
            "probe_window_ns": 210.0,
            "scintillator_detector_count": 3,
            "scintillator_detector_distance_m": 7.0,
            "sxr_energy_keV_range": [0.7, 15.0],
        },
        "density_formula_targets": {
            "delta_x_mm": 0.5,
            "delta_z_mm": 1.0,
            "laser_wavelength_nm": 527.0,
            "linear_density_per_shifted_fringe_coefficient": 2.1e15,
            "source_formula": "N(x,z) = delta * 2.1e15",
            "acceptance_boundary": (
                "The coefficient is copied from the local KR text. Profile "
                "curves from Figs. 3-6 still require figure digitization and "
                "independent review before per-point validation use."
            ),
        },
        "phase_timing": {
            "interferogram_times_ns": [-5.0, 55.0, 95.0],
            "linear_density_profile_times_ns": [-5.0, 25.0, 55.0, 85.0, 95.0],
            "timing_uncertainty_ns_range": [2.0, 3.0],
        },
        "spatial_density_targets": {
            "linear_density_at_95ns_z10_to_34_per_mm": 0.8e18,
            "linear_density_z_range_mm": [10.0, 34.0],
            "lobule_z_positions_mm_by_time_ns": {
                "-5": 25.0,
                "25": 30.0,
                "55": 35.0,
            },
        },
        "spatial_motion_targets": {
            "zipper_velocity_m_per_s_range": [5.0e5, 1.5e6],
            "axial_lobule_velocity_m_per_s": 1.5e5,
            "axial_lobule_velocity_uncertainty_m_per_s": 0.3e5,
            "mean_implosion_velocity_m_per_s": 2.2e5,
            "mean_implosion_velocity_uncertainty_m_per_s": 0.4e5,
            "expansion_velocity_m_per_s_range": [0.4e5, 1.2e5],
        },
        "digitization_requirements": [
            {
                "figure_id": "Fig. 2",
                "required_data": "interferogram_geometry_at_-5_55_95_ns",
                "status": "page_render_needed",
            },
            {
                "figure_id": "Figs. 3-6",
                "required_data": "linear_density_profiles_vs_radius_and_z",
                "status": "page_render_needed",
            },
        ],
        "uncertainty": {
            "timing_uncertainty_ns_range": [2.0, 3.0],
            "implosion_velocity_uncertainty_m_per_s": 0.4e5,
            "axial_lobule_velocity_uncertainty_m_per_s": 0.3e5,
            "missing_profile_uncertainty": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_density",
            "uncertainty",
        ],
        "missing_for_full_tier4": [
            "digitized_linear_density_profiles_from_figures_3_to_6",
            "interferogram_geometry_extraction_from_figure_2",
            "per_profile_linear_density_uncertainty",
            "independent_reviewed_digitization_packet",
        ],
        "validation_note": (
            "Cikhardtova 2015 supplies PF-1000 shot-9881 interferometry "
            "motion and line-density targets. This record starts extraction "
            "but does not promote figure curves until a digitization packet "
            "passes the local review gate."
        ),
    }


def pf1000_szydlowski_fast_ion_neutron_targets() -> dict[str, object]:
    """Return PF-1000 fast-ion and neutron targets from Szydlowski 2004."""
    source = "KnowledgeReference/doi-10-1016-j-vacuum-2004-07-040-6de67a98.md"
    return {
        "target_id": "pf1000_fast_ion_neutron_2004_szydlowski",
        "validation_scope": "pf1000_full_energy_fast_ion_neutron_2004_szydlowski",
        "device": "PF-1000",
        "model_role": "kr_fast_ion_neutron_spectrum_anisotropy_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "device_geometry_and_energy": "90-112",
            "neutron_and_fast_ion_diagnostics": "113-140",
            "yield_and_anisotropy_summary": "141-161",
            "neutron_pulse_and_spectrum": "172-204",
            "cr39_fast_ion_context": "205-227",
        },
        "shot_context": {
            "device": "PF-1000",
            "energy_level_kJ_range": [266.0, 1064.0],
            "voltage_kV_range": [20.0, 40.0],
            "anode_length_mm": 447.0,
            "anode_diameter_mm": 244.0,
            "insulator_diameter_mm": 244.0,
            "insulator_length_mm": 113.0,
            "cathode_rod_count": 24,
            "cathode_rod_diameter_mm": 32.0,
            "cathode_rod_length_mm": 600.0,
            "cathode_cylinder_diameter_mm": 368.0,
            "capacitance_source_text": (
                "1332 mF in the extracted KR text; review source PDF glyph "
                "before using as a circuit numeric target."
            ),
        },
        "activation_requirements": {
            "silver_activation_counter_count": 4,
            "activation_counter_angles": "different angles to electrode axis",
            "tof_scintillator_distances_m": {
                "downstream": 6.5,
                "upstream": [40.0, 85.0],
            },
            "cr39_distance_from_electrode_end_mm": 550.0,
            "cr39_angle_support": "semicircular",
            "al_filter_thicknesses_present": True,
        },
        "neutron_yield_targets": {
            "regular_neutron_emission_neutrons_per_shot_range": [1.0e10, 1.0e11],
            "yield_increases_then_decreases_with_voltage": True,
        },
        "anisotropy_targets": {
            "coefficient_definition": "Yn(30deg) / Yn(90deg)",
            "coefficient_at_133_Pa": 1.4,
            "coefficient_at_665_Pa_less_than": 1.2,
            "trend_with_pressure": "decreases_with_increasing_fill_pressure",
        },
        "neutron_timing": {
            "neutron_pulse_count": "two_or_three",
            "pulse_spacing_source_text": (
                "about 2 ms apart in the extracted KR text; review the PDF "
                "before using this as an absolute timing value."
            ),
            "neutron_prepulses_start_before_xray": True,
        },
        "spectral_targets": {
            "upstream_spectrum_peak_MeV_range": [2.2, 2.3],
            "dd_reference_energy_MeV": 2.45,
            "direction": "upstream",
        },
        "fast_ion_targets": {
            "cr39_shot_fill_pressure_Pa": 252.7,
            "crater_density_per_mm2_range": [1.0e3, 1.0e5],
            "al_filter_threshold_source_text": {
                "uncovered": "energy480keV OCR text; requires PDF glyph review",
                "1.5_mm_al": "faster than 330 keV",
                "4_mm_al": "energy4580keV OCR text; requires PDF glyph review",
            },
        },
        "uncertainty": {
            "ocr_glyph_review_required_for_capacitance_and_ion_thresholds": True,
            "digitized_spectrum_uncertainty_missing": True,
            "activation_calibration_uncertainty_missing": True,
        },
        "partial_target_groups": [
            "neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier5": [
            "digitized_neutron_spectrum_from_figure_3",
            "digitized_anisotropy_or_yield_curves",
            "activation_counter_calibration_uncertainty",
            "tof_detector_response_uncertainty",
            "pdf_review_of_ocr_suspect_units",
        ],
        "validation_note": (
            "Szydlowski 2004 starts a PF-1000 full-energy neutron/fast-ion "
            "target bundle. OCR-suspect capacitance, pulse-spacing, and ion "
            "threshold glyphs are deliberately quarantined as source text until "
            "PDF review and digitization are complete."
        ),
    }


def pf1000_krasa_vessel_scatter_anisotropy_targets() -> dict[str, object]:
    """Return PF-1000 vessel-scatter and wall-geometry targets from Krasa 2008."""
    source = (
        "KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-"
        "caused-by-the-plasma-focus-vessel-527cc533.md"
    )
    return {
        "target_id": "pf1000_vessel_scatter_anisotropy_2008_krasa",
        "validation_scope": "pf1000_full_energy_vessel_scatter_2008_krasa",
        "device": "PF-1000",
        "model_role": "kr_vessel_scatter_detector_response_geometry_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "vessel_geometry": "113-118",
            "measurement_scope_and_scatter_role": "121-130",
            "operating_point_and_electrodes": "132-140",
            "anisotropy_operating_conditions": "269-275",
            "tof_scatter_transform": "276-301",
        },
        "vessel_geometry": {
            "material": "stainless_steel",
            "vessel_length_m": 3.2,
            "vacuum_chamber_length_m": 2.5,
            "vacuum_chamber_diameter_m": 1.4,
            "average_wall_thickness_m": 0.010,
            "collector_ring_diameter_m": 2.5,
            "cables_not_sketch_count": 288,
        },
        "shot_context": {
            "bank_energy_kJ_range": [450.0, 500.0],
            "fill_pressure_torr": 3.5,
            "anode_diameter_m": 0.231,
            "anode_length_m": 0.600,
            "outer_electrode_count": 12,
            "outer_electrode_material": "stainless_steel",
            "outer_electrode_diameter_m": 0.080,
            "outer_electrode_radius_m": 0.200,
            "max_neutron_yield_per_shot_approx": 3.5e11,
        },
        "scatter_transport_targets": {
            "uses_mcnp4c_vessel_scatter_model": True,
            "measures_tld600h_tld700h_bonner_sphere_anisotropy": True,
            "direct_and_scattered_neutron_groups_determined": True,
            "direct_scattered_tof_separation_required": True,
            "tof_signal_kernel": "S(L,t) proportional to L^2/t^5 * f(L/t)",
            "velocity_distribution_kernel": "f(v) proportional to v^-5 * S(L/v)",
        },
        "partial_target_groups": [
            "neutron_anisotropy",
            "neutron_spectrum",
            "neutron_detector_response",
            "spatial_geometry",
            "uncertainty",
        ],
        "missing_for_full_tier5": [
            "digitized_figure_4_group_energy_spectra",
            "digitized_figure_5_vessel_scatter_anisotropy_curve",
            "mcnp_geometry_deck_or_equivalent_transport_mesh",
            "detector_position_table_and_uncertainty_budget",
            "same_scope_transfer_rule_for_akel_16kv",
        ],
        "validation_note": (
            "Krasa 2008 target-extracts PF-1000 vessel material/thickness "
            "and the direct/scattered neutron separation requirement.  It is "
            "full-energy PF-1000 vessel-scatter evidence and does not validate "
            "the Akel 16 kV shot without a reviewed transfer rule."
        ),
    }


def klir_2011_tof_detector_response_targets() -> dict[str, object]:
    """Return ToF detector response targets from Klir et al. 2011."""
    source = (
        "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-"
        "measurements-in-z-pinch-and-plasma-focus-214fbdae.md"
    )
    return {
        "target_id": "tof_detector_response_2011_klir",
        "validation_scope": "tof_detector_response_2011_klir",
        "device": "PF-1000 / z-pinch detector calibration",
        "model_role": "kr_neutron_detector_response_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "use_scope_and_sensitivity": "78-102",
            "scintillator_and_pmt_design": "118-138",
            "shielding_and_pf1000_stand": "154-170",
            "temporal_resolution": "171-198",
            "pmt_delay_calibration": "199-207",
        },
        "detector_use_scope": {
            "neutron_yield_range_per_shot": [1.0e6, 1.0e13],
            "single_neutron_sensitivity_energy_MeV_range": [1.8, 3.0],
            "absolute_yield_calibration_energy_MeV": 2.45,
            "used_at_pf1000": True,
        },
        "scintillator_targets": {
            "material": "Saint Gobain BC-408",
            "rise_time_ns": 0.9,
            "fwhm_ns": 2.5,
            "decay_time_ns": 2.1,
            "peak_emission_nm": 425.0,
            "density_g_per_cc": 1.032,
            "hydrogen_to_carbon_atomic_ratio": 1.104,
            "front_diameter_mm": 45.0,
            "thickness_mm": 50.0,
            "thickness_approximately_2p45MeV_neutron_mean_free_path": True,
        },
        "pmt_targets": {
            "assembly": "Hamamatsu H1949-51",
            "tube": "R1828-01",
            "rise_time_ns_at_2p5_kV": 1.3,
            "peak_cathode_sensitivity_nm": 420.0,
            "effective_photocathode_diameter_mm": 46.0,
            "dynamic_range_greater_than": 1.0e6,
        },
        "response_timing_targets": {
            "acquisition_system_response_ns_less_than": 1.0,
            "neutron_transit_uncertainty_through_50mm_scintillator_ns": 1.0,
            "typical_voltage_kV": 1.4,
            "oscilloscope_bandwidth_MHz": 500.0,
            "single_neutron_signal_fwhm_ns": 5.7,
            "single_neutron_signal_fwhm_uncertainty_ns_2sigma": 0.6,
            "rise_time_ns": 2.9,
            "rise_time_uncertainty_ns": 0.2,
            "fall_time_ns": 8.0,
            "fall_time_uncertainty_ns": 1.0,
            "two_neutron_resolvable_shift_ns": 5.5,
            "fwhm_ns_at_1p9_kV": 5.3,
            "nonlinear_anode_current_mA_above": 250.0,
        },
        "timing_calibration": {
            "pmt_delay_uncertainty_ns_less_than": 1.0,
            "pmt_to_pmt_delay_difference_ns_at_gt_1kV_less_equal": 2.0,
        },
        "uncertainty": {
            "fwhm_uncertainty_is_plus_minus_2sigma": True,
            "digitized_voltage_response_curve_missing": True,
            "absolute_sensitivity_curve_extraction_missing": True,
        },
        "partial_target_groups": [
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier5": [
            "digitized_fig2_voltage_response_curve",
            "digitized_fig4_pmt_delay_curve",
            "single_neutron_sensitivity_curve_or_table",
            "same_detector_geometry_mapping_to_target_experiment",
        ],
        "validation_note": (
            "Klir 2011 provides detector-response guardrails needed before "
            "ToF neutron timing or spectrum evidence can be treated as "
            "predictive validation. It is a detector calibration target, not "
            "a DPF simulation result by itself."
        ),
    }


def nx3_springham_zrbe_activation_targets() -> dict[str, object]:
    """Return NX3 Zr/Be activation neutron-energy and anisotropy targets."""
    source = (
        "KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-"
        "2021-164830-bc8edab3.md"
    )
    return {
        "target_id": "nx3_zrbe_activation_2021_springham",
        "validation_scope": "nx3_zrbe_activation_2021_springham",
        "device": "NX3",
        "model_role": "kr_activation_neutron_energy_anisotropy_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_targets": "36-60",
            "dd_kinematics_and_prior_energy_ranges": "61-106",
            "activation_ratio_method": "154-316",
            "device_geometry_and_bank": "379-409",
            "counting_interval": "410-420",
        },
        "shot_context": {
            "bank_energy_kJ": 7.2,
            "charge_voltage_kV": 12.0,
            "peak_discharge_current_kA_approx": 600.0,
            "fill_pressure_mbar_values": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "anode_radius_mm": 26.0,
            "anode_length_mm": 126.0,
            "cathode_rod_count": 6,
            "cathode_rod_diameter_mm": 12.0,
            "cathode_radius_mm": 56.0,
            "bank_capacitance_uF": 100.0,
            "design_quarter_period_us_at_5mbar": 3.0,
        },
        "activation_requirements": {
            "detector_pair_materials": ["zirconium", "beryllium"],
            "detector_angles_deg": [0.0, 90.0],
            "mcnp5_response_relationship_required": True,
            "be_reaction_threshold_MeV": 0.67,
            "zr_reaction_threshold_MeV": 2.35,
            "counting_interval_s": 3.0,
            "target_midpoint_distances_mm": {
                "RBe0": 296.0,
                "RBe90": 318.0,
                "RZr0": 320.0,
                "RZr90": 342.0,
            },
        },
        "neutron_yield_targets": {
            "highest_yield_neutrons_per_shot_approx": 1.0e9,
            "highest_yield_pressure_mbar": 5.0,
            "typical_neutron_burst_duration_ns_approx": 100.0,
        },
        "spectral_targets": {
            "effective_energy_MeV_at_0deg_approx": 2.8,
            "effective_energy_MeV_at_90deg_approx": 2.5,
            "neutron_energy_range_MeV_approx": [2.1, 3.1],
            "beam_target_constants_MeV": {
                "A": 2.451,
                "B": 1.226,
            },
        },
        "anisotropy_targets": {
            "fluence_anisotropy_AnBe_range": [2.5, 4.5],
            "mean_fluence_anisotropy_declines_with_pressure": True,
            "energy_anisotropy_delta_En_nearly_constant_with_pressure": True,
            "obstacle_distance_cm": 6.0,
            "obstacle_reduces_yield_and_fluence_anisotropy": True,
            "obstacle_slightly_increases_effective_energy": True,
        },
        "mechanism_targets": {
            "beam_target_model_consistent": True,
            "thermonuclear_contribution_negligible": True,
            "gyrating_particle_contribution_negligible": True,
        },
        "uncertainty": {
            "thick_target_monte_carlo_response_required": True,
            "counting_statistics_and_detector_efficiency_needed": True,
            "cross_section_uncertainty_needed": True,
        },
        "partial_target_groups": [
            "neutron_yield",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier5": [
            "digitized_pressure_sweep_tables_or_curves",
            "mcnp_response_curve_packet",
            "per_shot_count_uncertainties",
            "activation_cross_section_uncertainty_budget",
        ],
        "validation_note": (
            "Springham 2021 is an activation-detector neutron energy and "
            "anisotropy target for NX3. It supports mechanism and detector "
            "guardrails, but it is not same-scope PF-1000 evidence."
        ),
    }


def nnss_dpf_neutron_time_energy_tomography_targets() -> dict[str, object]:
    """Return NNSS DPF neutron time-energy tomography targets."""
    source = (
        "KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-"
        "energy-spectrum-from-a-dense-plasma-focus-b78f1154.md"
    )
    return {
        "target_id": "nnss_dpf_neutron_time_energy_tomography_2020_catenacci",
        "validation_scope": "nnss_dpf_neutron_tomography_2020_catenacci",
        "device": "NNSS DPF",
        "model_role": "kr_neutron_time_energy_tomography_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_validation_scope": "27-58",
            "tof_model_formulation": "166-246",
            "detector_model_and_parameter_count": "327-381",
            "shadow_bar_subtraction": "388-463",
            "detector_geometry_and_close_range": "471-526",
            "reconstruction_results": "527-588",
            "scatter_bias_and_energy_check": "589-694",
            "conclusion_and_detector_specs": "710-770",
        },
        "tomography_model_targets": {
            "time_energy_distribution": "f(t,E)",
            "uses_close_range_creation_time_profile": True,
            "energy_grid_MeV_range": [1.45, 3.45],
            "typical_energy_bin_count_range": [25, 30],
            "typical_time_interval_count": 3,
            "shadow_bar_system_count": 4,
            "data_points_vs_unknowns_overdetermined": True,
        },
        "detector_geometry": {
            "shadow_bar_pair_distances_m": [10.0, 14.0, 18.0, 22.0],
            "shadow_bar_detector_separations_cm": [60.0, 27.0, 31.0, 39.0],
            "shadow_bar_angles_deg_range_from_axis": [63.0, 76.0],
            "close_range_detector_distance_cm": 25.0,
            "close_range_sampling_GS_per_s": 2.5,
            "close_range_bandwidth_GHz": 1.0,
            "close_range_impulse_response_ns_less_than": 2.0,
        },
        "neutron_timing_targets": {
            "single_and_double_pinch_reconstruction_supported": True,
            "double_pinch_separation_ns_less_than": 50.0,
            "trial3_close_range_pulse_width_ns": 150.0,
            "trial3_first_detector_10m_pulse_width_ns_greater_than": 500.0,
        },
        "spectral_targets": {
            "peak_energies_near_MeV": 2.45,
            "trial3_second_pinch_higher_energy_density_MeV_around": 2.7,
            "energy_resolution_estimated_finer_than_keV": 100.0,
            "uncorrected_scatter_biases_density_below_MeV": 2.0,
            "scatter_correction_max_relative_difference_fraction": 0.23,
        },
        "response_model_requirements": [
            "foreground_shadowed_detector_scaling",
            "shadow_bar_background_subtraction",
            "detector_efficiency_and_area_terms",
            "close_range_detector_time_profile_constraint",
            "gamma_neutron_arrival_energy_cross_check",
        ],
        "uncertainty": {
            "figure_and_table_numeric_extraction_needed": True,
            "scatter_background_correction_required": True,
            "detector_impulse_response_needed": True,
        },
        "partial_target_groups": [
            "neutron_timing",
            "neutron_spectrum",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier5": [
            "digitized_fig4_time_energy_reconstructions",
            "digitized_fig5_double_pinch_energy_curves",
            "digitized_fig6_close_range_and_shadow_bar_waveforms",
            "digitized_fig7_scatter_subtraction_comparison",
            "tables_i_to_iv_numeric_extraction",
            "per_trial_uncertainty_budget",
        ],
        "validation_note": (
            "Catenacci 2020 starts a neutron time-energy tomography target "
            "for an NNSS DPF. It is method and detector-response evidence for "
            "Tier 5 design, not same-scope PF-1000 validation."
        ),
    }


def deuterium_argon_admixture_neutron_targets() -> dict[str, object]:
    """Return KR-backed D-Ar admixture neutron-yield target context."""
    source = (
        "KnowledgeReference/regular-article-deuterium-argon-admixture-for-"
        "plasma-focus-neutron-generation-muhammad-luqman.md"
    )
    return {
        "target_id": "deuterium_argon_admixture_neutron_2026_omar",
        "device": "Mather-type 2.7 kJ PF",
        "model_role": "kr_gas_admixture_neutron_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_outcomes": "44-58",
            "apparatus_and_diagnostics": "124-139",
            "gas_mixture_method": "139-193",
            "lee_current_fit_method": "220-235",
            "focus_time_and_shot_selection": "245-249",
            "table_2_and_3_context": "344-408",
            "pinch_voltage_current_uncertainty": "410-441",
            "best_yield_current_energy": "509-527",
            "energy_temperature_limits": "584-600",
            "summary_and_limits": "660-733",
        },
        "shot_context": {
            "device_type": "Mather",
            "capacitance_uF": 30.0,
            "charging_voltage_kV_max": 14.0,
            "reported_total_discharge_energy_kJ": 2.7,
            "total_fill_pressure_mbar": 4.0,
            "anode_diameter_mm": 19.0,
            "anode_effective_length_mm": 160.0,
            "cathode_rods": 6,
            "cathode_radius_mm": 32.0,
            "argon_mass_percentages": [10.0, 30.0, 50.0, 60.0, 70.0],
            "shots_per_mixture": 30,
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "measured_voltage_available": True,
            "rogowski_conversion_kA_per_V": 36.0,
            "voltage_probe_calibration_factor": 0.71,
            "lee_model_fit_to_measured_current": True,
            "measured_current_drop_at_pinch": True,
        },
        "phase_timing": {
            "plasma_focus_time_us_range": [2.7, 3.3],
            "pure_deuterium_voltage_spike_fwhm_ns": 93.0,
            "pure_deuterium_voltage_spike_fwhm_uncertainty_ns": 10.0,
            "ten_percent_argon_voltage_spike_fwhm_ns": 44.0,
            "ten_percent_argon_voltage_spike_fwhm_uncertainty_ns": 13.0,
            "pure_deuterium_computed_pinch_lifetime_ns": 4.90,
            "ten_percent_argon_computed_pinch_lifetime_ns": 4.92,
        },
        "neutron_yield_targets": {
            "pure_deuterium_average_neutrons_per_shot": 4.7e6,
            "pure_deuterium_neutrons_per_shot_std": 0.8e6,
            "fifty_percent_argon_average_neutrons_per_shot": 3.0e7,
            "fifty_percent_argon_neutrons_per_shot_std": 0.6e7,
            "fifty_percent_argon_recorded_neutrons_per_shot": 3.9e7,
            "yield_enhancement_over_pure_deuterium_min": 5.0,
        },
        "pinch_energy_targets": {
            "pure_deuterium_energy_into_pinch_J": 83.0,
            "pure_deuterium_energy_into_pinch_std_J": 17.0,
            "fifty_percent_argon_energy_into_pinch_J": 139.0,
            "fifty_percent_argon_energy_into_pinch_std_J": 16.0,
            "argon_50_percent_energy_gain_fraction_min": 0.60,
        },
        "pinch_current_targets": {
            "pure_deuterium_computed_pinch_current_kA": 138.0,
            "fifty_percent_argon_computed_pinch_current_kA": 144.0,
            "sixty_to_seventy_percent_argon_computed_pinch_current_kA": 144.0,
        },
        "temperature_targets": {
            "computed_ion_temperature_order_keV": 1.0,
            "trend_with_argon_mass": "decreases_with_increasing_argon_mass",
            "direct_temperature_measured": False,
        },
        "activation_requirements": {
            "activation_material": "indium",
            "counter_distance_cm": 28.0,
            "counter_axis": "z_axis",
            "calibration_factor_neutrons_per_count": 8.22e4,
            "paraffin_thickness_cm": 7.0,
            "decay_half_life_s": 14.1,
        },
        "response_model_requirements": [
            "indium_activation_calibration",
            "moderator_geometry",
            "activation_branching_fraction",
            "counter_solid_angle",
            "shot_to_shot_variation",
        ],
        "uncertainty": {
            "uncertainty_statistic": "standard_deviation",
            "shot_to_shot_variation_used_for_error_bars": True,
            "pure_deuterium_breakdown_voltage_kV": 12.1,
            "pure_deuterium_breakdown_voltage_std_kV": 1.6,
            "fifty_percent_argon_breakdown_voltage_kV": 11.8,
            "fifty_percent_argon_breakdown_voltage_std_kV": 1.8,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_temperature",
            "neutron_detector_response",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "digitized_voltage_trace_points",
            "per_point_current_uncertainty",
            "per_point_voltage_uncertainty",
        ],
        "missing_for_full_tier2": [
            "absolute_axial_rundown_start_end_times",
            "radial_transit_endpoints",
            "direct_observed_pinch_lifetime",
        ],
        "missing_for_full_tier4": [
            "direct_experimental_ion_temperature_diagnostic",
            "density_profile",
            "magnetic_field_or_em_probe_trace",
        ],
        "missing_for_full_tier5": [
            "time_resolved_neutron_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "activation_response_uncertainty_budget",
        ],
        "validation_note": (
            "This source is valuable for gas-admixture neutron-yield and "
            "activation-detector targets with shot-to-shot uncertainty. It "
            "does not close end-to-end DPF validation because the waveforms are "
            "not digitized here, neutron data are time integrated, and ion "
            "temperature is computed from Lee-model fitting rather than "
            "directly measured."
        ),
    }


def ff1_focus_fusion_plasmoid_targets() -> dict[str, object]:
    """Return KR-backed FF-1 plasmoid and p-B11 context targets."""
    source = (
        "KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-"
        "fusion-with-the-dense-plasma-focus.md"
    )
    return {
        "target_id": "ff1_focus_fusion_plasmoid_2023_lerner",
        "device": "FF-1 / FF-2B",
        "model_role": "kr_advanced_fuel_plasmoid_context_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "p_b11_constraints": "66-98",
            "dpf_plasmoid_review_context": "121-164",
            "bostick_nardi_instability_sequence": "169-198",
            "yield_plateau_context": "233-250",
            "plasmoid_model_scalings": "259-323",
            "qmf_simulation_limits": "420-433",
            "ff1_device_and_diagnostics": "770-800",
            "ion_beam_energy_transfer": "808-849",
            "confined_ion_energy_tof": "856-929",
            "density_and_nst_estimates": "930-1065",
            "impurity_measurement": "1159-1197",
            "current_challenges_and_projections": "1245-1367",
        },
        "shot_context": {
            "device": "FF-1",
            "later_device_name": "FF-2B",
            "capacitance_uF": 113.0,
            "capacitor_count_total": 12,
            "routine_capacitor_count": 8,
            "charge_voltage_kV_range": [24.0, 40.0],
            "charge_voltage_kV_max": 45.0,
            "stored_energy_kJ_max": 115.0,
            "peak_current_MA": 1.0,
            "rise_time_us": 1.8,
            "anode_radius_cm": 2.8,
            "cathode_radius_cm": 5.0,
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "main_current_rogowski": True,
            "beam_rogowski_coils": 2,
            "reported_peak_current_MA": 1.0,
            "best_pinch_inductance_increase_nH": 10.0,
            "current_oscillation_frequency_MHz_range": [14.0, 40.0],
        },
        "spatial_density_targets": {
            "visible_light_plasmoid_volume_cm3": 1.2e-3,
            "minimum_ion_density_cm3": 3.0e19,
            "maximum_ion_density_cm3": 4.0e19,
            "fill_density_cm3": 1.6e18,
            "compression_factor": 20.0,
        },
        "magnetic_field_targets": {
            "qmf_required_B_for_p_GG": 14.0,
            "qmf_required_B_for_alpha_GG": 3.5,
            "qmf_required_B_for_boron11_GG": 1.3,
            "model_example_core_field_GG": 10.0,
            "filament_density_and_B_amplification_factor_range": [10.0, 20.0],
        },
        "spatial_geometry_targets": {
            "plasmoid_radius_observed_um": 250.0,
            "projected_filament_radius_um": 50.0,
            "qmf_simulation_minimum_radius_um": 14.0,
        },
        "temperature_targets": {
            "confined_ion_energy_best_keV": 240.0,
            "confined_ion_energy_uncertainty_keV": 20.0,
            "ten_shot_mean_ion_energy_keV": 124.0,
            "viscous_heating_predicted_Ti_keV": 113.0,
            "p_b11_average_ion_energy_requirement_keV_min": 100.0,
            "direct_temperature_measured": False,
        },
        "event_sequence": [
            {
                "event": "plasmoid_formation",
                "mechanism": "filament_convergence_and_kink",
                "timing_note": "ICCD timing matched the observed ion beam to plasmoid formation.",
                "required": True,
            },
            {
                "event": "ion_and_electron_beam_release",
                "mechanism": "magnetic_field_change_induced_electric_field",
                "timing_note": "Ion and electron beams release energy trapped in the plasmoid.",
                "required": True,
            },
        ],
        "detector_tof": {
            "neutron_tof_detector_distances_m": [11.5, 17.5],
            "scintillator_pm_time_resolution_ns": 1.0,
            "iccd_minimum_exposure_ns": 0.2,
        },
        "spectral_targets": {
            "neutron_tof_mean_ion_energy_keV": 240.0,
            "ion_beam_mean_energy_MeV": 3.0,
            "confined_ion_energy_exceeds_keV": 200.0,
        },
        "anisotropy_targets": {
            "bubble_detector_result": (
                "axial and horizontal bubble detectors supported isotropic "
                "neutron distribution for confined-ion interpretation"
            ),
        },
        "neutron_yield_targets": {
            "best_2016_neutron_yield": 2.5e11,
            "best_2016_neutron_yield_uncertainty": 0.25e11,
            "best_2016_fusion_energy_J": 0.2,
            "best_2016_fusion_energy_uncertainty_J": 0.02,
            "best_2016_device_energy_kJ": 60.0,
            "best_2016_wall_plug_efficiency": 3.3e-6,
        },
        "activation_requirements": {
            "silver_activation_neutron_detector": True,
            "bubble_detectors": True,
            "scintillator_pm_tubes": 5,
            "time_integrated_optical_spectrometer": True,
        },
        "response_model_requirements": [
            "silver_activation_calibration",
            "bubble_detector_angular_response",
            "scintillator_neutron_tof_response",
            "xray_neutron_signal_separation",
            "beam_rogowski_response",
        ],
        "advanced_fuel_context": {
            "fuel": "p-B11",
            "secondary_neutron_energy_fraction": 0.002,
            "nst_keV_s_per_m3": 3.4e20,
            "nst_uncertainty_keV_s_per_m3": 0.8e20,
            "p_b11_projection_is_validated": False,
            "qmf_simulation_dimension": "0-D uniform sphere",
            "qmf_simulation_limitation": "not fully realistic",
        },
        "impurity_targets": {
            "beryllium_deposition_current_sheath_ug_per_shot": 6.0,
            "beryllium_deposition_current_sheath_uncertainty_ug_per_shot": 1.0,
            "beryllium_deposition_post_pinch_ug_per_shot": 14.0,
            "beryllium_deposition_post_pinch_uncertainty_ug_per_shot": 2.0,
            "gas_in_current_sheath_mg": 3.3,
            "be_impurity_mass_fraction_max": 0.002,
            "be_impurity_ion_fraction_max": 0.0005,
            "zeff_nominal": 1.004,
            "zeff_all_deposition_assigned_to_pinch": 1.012,
        },
        "uncertainty": {
            "best_ion_energy_uncertainty_keV": 20.0,
            "best_neutron_yield_relative_uncertainty": 0.10,
            "nst_relative_uncertainty": 0.8 / 3.4,
            "impurity_deposition_uncertainty_reported": True,
            "shot_to_shot_impurity_variation_noted": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_main_current_trace",
            "per_point_current_uncertainty",
            "switch_timing_jitter",
        ],
        "missing_for_full_tier4": [
            "direct_density_field_measurement_at_neutron_emission",
            "direct_magnetic_field_measurement_inside_plasmoid",
            "direct_ion_temperature_diagnostic",
        ],
        "missing_for_full_tier5": [
            "time_resolved_neutron_birth_history",
            "calibrated detector_response_model",
            "shot_series_distribution_for_record_yield",
            "p_b11 experimental neutron_alpha_yield_measurement",
        ],
        "validation_note": (
            "This KR target is valuable for FF-1 plasmoid, neutron yield, "
            "confined ion energy, density, n-tau-T, impurity, and p-B11 "
            "context. It does not validate p-B11 net energy: the p-B11 portions "
            "are constraints, projections, or reduced simulations, and the "
            "target lacks digitized waveforms and full detector-response "
            "uncertainty."
        ),
    }


def lee_drive_parameter_speed_enhancement_targets() -> dict[str, object]:
    """Return KR-backed Lee drive-parameter and speed-enhancement targets."""
    source = (
        "KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-"
        "enhancing-the-neutron-yield.md"
    )
    return {
        "target_id": "lee_drive_parameter_speed_enhancement_2003",
        "device": "generic neutron-optimized Mather DPF",
        "model_role": "kr_scaling_regime_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "abstract_speed_scaling": "17-28",
            "axial_radial_model": "32-102",
            "speed_parameter_definition": "194-197",
            "pinch_size_lifetime_scaling": "201-239",
            "drive_parameter_survey": "249-333",
            "speed_temperature_density_context": "337-346",
            "yield_scaling": "351-405",
            "speed_limit_and_enhancement": "411-445",
            "speed_enhancement_figure": "480-481",
        },
        "phase_semantics": {
            "axial": "snowplow model for axial phase",
            "radial": "slug model for radial phase",
            "pinch": (
                "pinch phase ends when instabilities disrupt the compressed "
                "plasma column"
            ),
        },
        "phase_timing": {
            "deuterium_minimum_radius_over_anode_radius": 0.12,
            "deuterium_maximum_length_over_anode_radius": 0.8,
            "deuterium_radial_shock_transit_s_per_m": 5.0e-6,
            "deuterium_pinch_lifetime_s_per_m": 2.0e-6,
            "neon_minimum_radius_over_anode_radius": 0.04,
            "neon_maximum_length_over_anode_radius": 0.8,
            "neon_radial_shock_transit_s_per_m": 4.0e-6,
            "neon_pinch_lifetime_s_per_m": 1.0e-6,
        },
        "drive_parameter_targets": {
            "definition": "Ip / a / sqrt(p_D2)",
            "mean_kA_per_cm_per_sqrt_torr": 89.0,
            "standard_deviation_kA_per_cm_per_sqrt_torr": 7.7,
            "rounded_target_kA_per_cm_per_sqrt_torr": 89.0,
            "rounded_uncertainty_kA_per_cm_per_sqrt_torr": 8.0,
            "survey_energy_range_kJ": [3.0, 280.0],
            "typical_axial_speed_cm_per_us": 10.0,
            "typical_radial_speed_cm_per_us": 25.0,
        },
        "temperature_targets": {
            "constant_drive_parameter_implies_nearly_constant_temperature": True,
            "small_focus_ion_temperature_keV": 1.0,
            "beam_energy_keV": 50.0,
            "speed_enhancement_temperature_range_keV": [1.0, 10.0],
        },
        "neutron_yield_scaling_targets": {
            "constant_speed_thermonuclear_scaling": "Y ~ I^4",
            "speed_enhanced_thermonuclear_scaling": "Yth ~ I^4 * v_axial^4",
            "speed_enhanced_beam_target_scaling": "Ybt ~ I^4.5 * v_axial^-1.5",
            "fixed_anode_radius_thermonuclear_limit": "Yth ~ I^8",
            "fixed_anode_radius_beam_target_limit": "Ybt ~ I^3",
            "small_focus_thermonuclear_fraction": 0.15,
            "small_focus_beam_target_fraction": 0.85,
        },
        "operational_limits": {
            "quality_deterioration_axial_speed_cm_per_us": 10.0,
            "magnetic_reynolds_transition_speed_cm_per_us": 5.0,
            "magnetic_reynolds_speed_power": 4.0,
            "stepped_anode_speed_cm_per_us": 15.0,
            "argon_krypton_axial_phase_speed_cm_per_us_range": [15.0, 20.0],
        },
        "uncertainty": {
            "drive_parameter_standard_deviation_fraction": 7.7 / 89.0,
            "actual_shot_data_uncertainty_noted": True,
            "non_thermal_neutron_fraction_requires_mechanism_separation": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "uncertainty",
        ],
        "missing_for_predictive_tier2": [
            "device_specific_current_trace",
            "device_specific_pressure_and_geometry",
            "shot_specific_axial_radial_endpoint_times",
        ],
        "missing_for_full_tier4": [
            "device_specific_temperature_measurement",
            "same_scope_density_measurement",
            "same_scope_magnetic_field_measurement",
        ],
        "missing_for_full_tier5": [
            "time_resolved_neutron_history",
            "mechanism_separated_neutron_yield_measurement",
            "detector_response_model",
        ],
        "validation_note": (
            "This source is a generic scaling/regime target. It supports "
            "drive-parameter, phase-scaling, and speed-enhancement checks, but "
            "it is not a same-device experimental validation packet and cannot "
            "close predictive readiness by itself."
        ),
    }


def rawat_dpf_operating_envelope_targets() -> dict[str, object]:
    """Return generic DPF operating-envelope targets from Rawat 2015 review."""
    source = (
        "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
        "alternative-fusion-source-to-versatile-high-energy-4.md"
    )
    return {
        "target_id": "rawat_dpf_operating_envelope_2015",
        "device": "generic DPF",
        "model_role": "kr_operating_envelope_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_and_scope": "52-74",
            "fusion_scaling_and_mechanism_limits": "109-134",
            "high_energy_density_regime": "253-268",
            "device_layout_and_phase_dynamics": "275-313",
            "optimized_device_universality": "319-355",
            "repeatability_and_conditioning": "383-408",
            "pressure_and_application_context": "720-722,749-754,791-802",
        },
        "phase_semantics": {
            "breakdown": "discharge across insulator sleeve at closed electrode end",
            "current_sheath_formation": (
                "well-defined plasma sheath forms after breakdown"
            ),
            "axial_acceleration": (
                "J x B driven current sheath acceleration along the electrode"
            ),
            "radial_acceleration": (
                "rollover at anode end followed by radial collapse to a hot, "
                "dense pinch column"
            ),
            "instability_disruption": (
                "pinched column breaks up through m=0 and m=1 instabilities"
            ),
        },
        "phase_timing": {
            "current_sheath_formation_ns_range": [100.0, 500.0],
            "quarter_period_ns_range": [500.0, 3000.0],
            "typical_axial_speed_cm_per_us_range": [4.0, 6.0],
            "optimized_axial_speed_cm_per_us_range": [2.0, 10.0],
            "radial_speed_multiplier_over_axial_range": [2.0, 2.5],
            "radiation_and_particle_duration_ns_order": [10.0, 500.0],
        },
        "spatial_density_targets": {
            "pinch_plasma_density_m3_range": [5.0e24, 1.0e26],
            "dpf_energy_density_J_per_m3_range": [1.2e10, 9.5e10],
            "hed_classification_context_J_per_m3_range": [1.0e10, 1.0e11],
        },
        "temperature_targets": {
            "axial_end_electron_temperature_eV": 100.0,
            "axial_end_ion_temperature_eV": 300.0,
            "fast_shock_temperature_eV_order": 100.0,
            "pinch_temperature_keV_range": [0.2, 2.0],
            "pinch_ion_temperature_keV_range": [0.3, 1.5],
            "reflected_shock_magnetic_compression_temperature_keV_range": [1.0, 2.0],
            "direct_temperature_measured": False,
        },
        "energetic_particle_targets": {
            "electron_energy_keV_order_range": [10.0, 300.0],
            "ion_energy_keV_to_MeV_range": [10.0, 3000.0],
            "forward_ion_cone_half_angle_deg": 20.0,
            "xray_photon_energy_eV_range": [100.0, 3.0e5],
        },
        "neutron_mechanism_targets": {
            "yield_storage_energy_scaling": "Yn ~ E0^2",
            "beam_target_mechanism_dominant_in_experiments": True,
            "thermonuclear_and_beam_target_joint_contribution": True,
            "neutron_anisotropy_axis_greater_than_perpendicular": True,
        },
        "operational_context": {
            "capacitor_charge_voltage_kV_typical_range": [10.0, 30.0],
            "efficient_operation_pressure": "few mbar",
            "shot_conditioning_required": True,
            "conditioning_shot_count_order": "few to few tens",
            "current_or_voltage_probe_used_for_focus_efficiency": True,
        },
        "uncertainty": {
            "review_not_same_shot_dataset": True,
            "shot_to_shot_variation_noted": True,
            "conditioning_needed_for_repeatability": True,
            "digitized_waveforms_not_provided": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_density",
            "spatial_temperature",
            "neutron_timing",
            "neutron_anisotropy",
            "uncertainty",
        ],
        "missing_for_full_tier2": [
            "device_specific_current_trace",
            "shot_specific_phase_endpoint_times",
            "phase_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "same_scope_density_profile",
            "same_scope_magnetic_field_measurement",
            "direct_temperature_diagnostic",
        ],
        "missing_for_full_tier5": [
            "time_resolved_neutron_history",
            "neutron_spectrum",
            "absolute_detector_response_model",
            "mechanism_separated_yield_uncertainty",
        ],
        "validation_note": (
            "This Rawat review target is a generic DPF operating envelope and "
            "mechanism sanity check. It should reject simulations that miss "
            "basic DPF scales, but it cannot validate a predictive end-to-end "
            "device model without same-scope measured waveforms, spatial "
            "diagnostics, neutron histories, and uncertainty."
        ),
    }


def auluck_gpf_scaling_theory_targets() -> dict[str, object]:
    """Return KR-backed Generalized Plasma Focus scaling-theory targets."""
    source = (
        "KnowledgeReference/the-generalized-plasma-focus-problem-and-its-"
        "application-to-space-propulsion-s-k-h-auluck.md"
    )
    return {
        "target_id": "auluck_gpf_scaling_theory_2023",
        "device": "Generalized Plasma Focus / tapered-anode concept",
        "model_role": "kr_scaling_theory_target",
        "validation_tier": 2,
        "source": source,
        "source_lines": {
            "abstract_and_gpf_definition": "29-48",
            "fusion_scaling_and_model_limits": "92-150",
            "gpf_scaling_scope": "152-188",
            "propulsion_concept_scope_limit": "190-221",
            "propagation_delay_and_liftoff": "2123-2244",
            "ionization_stability_and_pressure_limit": "2249-2417",
            "example_parameter_derivation": "5367-5531",
            "laboratory_example_numbers": "5796-5845,5957-5978",
            "validation_requirements": "6248-6336",
            "summary_values": "6346-6369",
        },
        "phase_semantics": {
            "formation_delay": (
                "The just-formed plasma remains near the initial surface while "
                "current rises and energy is spent dissociating and ionizing gas."
            ),
            "supersonic_liftoff": (
                "The sheath begins propagation only after sufficient magnetic "
                "pressure and ionization stability are achieved."
            ),
            "gpf_surface": (
                "A reference GV surface represents the interface between "
                "current-free and current-bearing plasma regions."
            ),
            "tapered_anode_switch": (
                "A modified plasma focus transports current behind the sheath "
                "to an axial wire or tube fuel element."
            ),
        },
        "phase_timing": {
            "lift_off_fraction_of_charge_example": 0.10,
            "lift_off_reduced_time_example": 0.426395,
            "quarter_period_us_laboratory_example": 8.45,
            "wire_surface_travel_time_ns": 8.4,
            "radial_alfven_transit_time_ns": 17.0,
            "wire_explosion_timescale_ps": 3.0,
            "magnetic_field_rise_time_ns": 40.0,
        },
        "scaling_theory_targets": {
            "requires_first_principles_scaling_for_untested_configurations": True,
            "conventional_dpf_fusion_scaling_failure_observed": True,
            "lee_model_requires_experimental_current_waveform_fit": True,
            "rgv_model_requires_experimental_current_waveform_fit": True,
            "formation_delay_explicitly_addressed": True,
            "calculated_temporal_current_profile": True,
            "validity_claim": "ballpark estimates for GPF-conforming concepts",
            "space_propulsion_feasibility_claimed": False,
        },
        "power_density_amplification_context": {
            "storage_volume_scale": "several cubic meters",
            "chamber_volume_scale": "few litres",
            "plasma_volume_scale": "few cubic millimetres",
            "storage_time_scale_s": 10.0,
            "delivery_time_scale_us": 10.0,
            "pinch_delivery_time_scale_ns": 10.0,
            "example_power_density_amplification": 9000.0,
            "example_I_over_r_peak": 100.0,
        },
        "spatial_density_targets": {
            "hydrogen_fill_density_kg_per_m3_example": 0.00342,
            "hydrogen_pressure_mbar_example": 43.0,
            "wire_inner_radius_mm_example": 0.025,
            "gas_distribution_nonuniformity_affects_inductance": True,
        },
        "magnetic_field_targets": {
            "example_wire_surface_B_initial_T": 20.0,
            "example_wire_surface_B_final_T": 200.0,
            "example_wire_current_kA": 80.0,
            "example_wire_current_density_A_per_m2": 1.8e12,
            "azimuthal_magnetic_field_drives_gpf": True,
        },
        "propulsion_example_targets": {
            "charging_voltage_kV_example": 20.0,
            "capacitance_uF_example": 43.0,
            "current_scale_kA_example": 160.0,
            "stored_energy_kJ_example": 8.6,
            "jet_alfven_velocity_m_per_s_example": 1450.0,
            "impulse_kg_m_per_s_example": 0.002,
            "unuictp_scale_laboratory_facility_sufficient": True,
        },
        "validation_requirements": [
            "measure_voltage_across_plasma",
            "measure_current_through_plasma",
            "calculate_and_compare_inductance_variation",
            "repeat_for_multiple_anode_profiles",
            "measure_jet_momentum_and_velocity",
            "verify_absolute_magnitude_and_scaling",
            "measure_wall_energy_deposition_for_dynamic_hohlraum_variant",
            "validate_gas_distribution_and_breakdown_strategy",
            "test_deuterium-filled tube neutron emission separately",
        ],
        "uncertainty": {
            "theory_requires_baseline_laboratory_validation": True,
            "space_propulsion_concept_is_illustrative_not_feasibility_claim": True,
            "conventional_fusion_mechanism_not_fully_understood": True,
            "neutron_scaling_failure_has_no_workaround_in_conventional_dpf": True,
            "gas_distribution_corrections_needed": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "uncertainty",
        ],
        "missing_for_predictive_tier2": [
            "measured_current_waveform_for_the_modified_device",
            "measured_voltage_waveform_for_the_modified_device",
            "experimental_liftoff_time",
            "profile_sweep_validation_data",
        ],
        "missing_for_full_tier4": [
            "measured_density_distribution_along_gpf_path",
            "measured_magnetic_field_distribution",
            "wire_or_tube_plasma_temperature_diagnostic",
        ],
        "missing_for_full_tier5": [
            "deuterium_tube_neutron_yield_measurement",
            "neutron_timing_spectrum_anisotropy",
            "detector_response_and_uncertainty",
            "validated_MAGLIF_or_dynamic_hohlraum_coupling",
        ],
        "validation_note": (
            "This Auluck source supplies first-principles scaling constraints "
            "and validation requirements for a Generalized Plasma Focus, not "
            "a completed experimental benchmark. It strengthens scope control "
            "by making conventional neutron-scaling failure and the required "
            "laboratory validation explicit."
        ),
    }


def auluck_neutron_yield_scaling_failure_targets() -> dict[str, object]:
    """Return KR-backed Auluck neutron-yield scaling failure targets."""
    source = (
        "KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-"
        "dense-plasma-focus-s-k-h-auluck-international.md"
    )
    return {
        "target_id": "auluck_neutron_yield_scaling_failure_2023",
        "device": "generic DPF",
        "model_role": "kr_scaling_failure_theory_target",
        "validation_tier": 3,
        "source": source,
        "source_lines": {
            "scaling_failure_inference": "13-19",
            "insulator_radius_yield_scaling": "21-35",
            "small_device_liftoff_tests": "36-43",
            "references_context": "46-144",
        },
        "scaling_failure_targets": {
            "failure_inferred_from_large_devices_abruptly_underperforming_above_voltage": True,
            "failed_device_must_be_tweaked_to_drive_parameter_limits": True,
            "generalized_optimization_criteria_required": True,
            "insulator_outer_radius_to_anode_radius_ratio_symbol": "R_I_tilde",
            "yield_inverse_power_of_insulator_radius_ratio": 5.0,
            "typical_insulator_radius_ratio": 1.0,
            "proposed_reduced_insulator_radius_ratio": 0.4,
            "claimed_yield_increase_orders_if_all_conditions_met": 2.0,
            "insulator_in_shadow_of_anode": True,
        },
        "phase_timing": {
            "liftoff_time_measurement_is_primary_test": True,
            "liftoff_time_should_correlate_with_drive_parameter": True,
            "liftoff_time_should_correlate_with_insulator_radius": True,
            "source_references_relation_12_not_extracted": True,
        },
        "field_context": {
            "drive_parameter_limit_controls_scaling_failure_argument": True,
            "operating_pressure_range_changes_with_insulator_radius": True,
            "add_on_insulator_pressure_range_test": True,
            "insulator_outer_radius_less_than_anode_radius_test": True,
        },
        "neutron_yield_targets": {
            "neutron_measurement_not_primary_small_device_test": True,
            "small_devices_can_test_scaling_failure_without_neutron_measurements": True,
            "reaction_yield_relation_17_not_extracted": True,
            "yield_scaling_claim_is_theory_not_validation_dataset": True,
        },
        "model_scope_limits": {
            "local_extract_only_conclusion_and_references": True,
            "equations_12_and_17_not_available_in_markdown": True,
            "no_experimental_liftoff_dataset": True,
            "no_neutron_yield_dataset": True,
            "requires_all_optimization_conditions_simultaneously": True,
        },
        "uncertainty": {
            "no_error_bars": True,
            "no_device_table": True,
            "no_digitized_drive_parameter_sweep": True,
            "no_pressure_range_sweep_data": True,
            "no_neutron_yield_uncertainty": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_magnetic_or_em",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier2": [
            "measured_liftoff_time_vs_drive_parameter",
            "measured_liftoff_time_vs_insulator_radius",
            "phase_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "drive_parameter_sweep_dataset",
            "operating_pressure_range_vs_insulator_radius",
            "field_or_current_sheath_measurement",
        ],
        "missing_for_full_tier5": [
            "same_device_neutron_yield_vs_insulator_radius",
            "neutron_yield_vs_drive_parameter",
            "detector_response",
            "uncertainty_budget",
            "full_equation_context_from_reingested_pdf",
        ],
        "validation_note": (
            "This local extract supports only a narrow scaling-failure "
            "constraint: do not claim neutron-yield scaling without checking "
            "drive-parameter and insulator-radius limits. The proposed "
            "near-term validation is lift-off timing and operating-pressure "
            "range, not neutron yield, and the body equations/data needed for "
            "quantitative validation are missing from the markdown."
        ),
    }


def auluck_circuit_element_poynting_targets() -> dict[str, object]:
    """Return KR-backed DPF circuit-element/Poynting-theorem targets."""
    source = "KnowledgeReference/auluck-2021-dpf-circuit-element.md"
    return {
        "target_id": "auluck_circuit_element_poynting_2021",
        "device": "PF-1000 / generic DPF circuit element",
        "model_role": "kr_circuit_field_coupling_target",
        "validation_tier": 3,
        "source": source,
        "source_lines": {
            "abstract": "26-41",
            "diagnostic_and_interpreted_inductance": "44-70",
            "pf1000_probe_and_interferometry_context": "72-119",
            "neon_delay_and_dynamo_motivation": "121-149",
            "poynting_circuit_element_basis": "151-201",
            "finite_boundary_context": "211-224",
            "inductance_capacitance_resistance_terms": "762-786,788-910",
            "anomalous_impedance_and_dynamo": "950-1019",
            "conclusion": "1021-1045",
        },
        "current_waveform_targets": {
            "dI_dt_diagnostic_standard": True,
            "voltage_diagnostic_standard": True,
            "current_derivative_dip_indicates_operation": True,
            "voltage_spike_indicates_operation": True,
            "dip_and_voltage_spike_correlate_with_neutron_yield": True,
            "voltage_spike_time_correlated_not_simultaneous_with_dI_dt_minimum": True,
            "interpreted_inductance_monotonic_in_pf1000": True,
            "scalar_time_varying_inductance_is_incomplete": True,
        },
        "phase_timing": {
            "pf1000_interferogram_interval_ns_range": [10.0, 15.0],
            "pf1000_example_probe_times_ns": [-68.0, -38.0, 22.0],
            "diagnostic_propagation_delay_ns_range": [10.0, 20.0],
            "diagnostic_external_path_length_m": 2.0,
            "deuterium_dI_dt_minimum_axis_arrival_simultaneous_within_error": True,
            "neon_dI_dt_minimum_after_column_breakup_ns_min": 200.0,
        },
        "spatial_density_targets": {
            "current_carrying_layer_thickness_cm_range": [1.6, 2.6],
            "sheath_average_velocity_m_per_s": 2.1e5,
            "sheath_velocity_shot_to_shot_fraction": 0.25,
            "density_drop_orders_min": 2.0,
            "density_drop_radial_width_mm_max": 1.0,
        },
        "magnetic_field_targets": {
            "pf1000_probe_radii_mm": [40.0, 13.0, 0.0],
            "probe_height_above_anode_mm": 10.0,
            "axial_magnetic_field_detected": True,
            "poloidal_magnetic_field_required_post_stagnation": True,
            "three_dimensional_magnetic_structure_required": True,
            "all_three_magnetic_components_contribute_to_plasma_inductance": True,
            "motional_dynamo_amplifies_seed_field": True,
        },
        "field_coupling_requirements": {
            "terminal_voltage_from_integral_minus_J_dot_E_over_I": True,
            "circuit_power_must_account_for_all_chamber_processes": True,
            "plasma_inductance_from_total_magnetic_energy_incomplete": True,
            "motional_impedance_differs_from_dLdt_inductance_term": True,
            "anomalous_impedance_needed_for_unaccounted_terms": True,
            "term_with_velocity_along_B_has_no_circuit_analog": True,
            "post_breakup_current_streamlines_can_form_quasi_closed_paths": True,
        },
        "neutron_context": {
            "axial_magnetic_field_correlation_with_neutron_emission": True,
            "dI_dt_voltage_transient_parameters_correlate_with_yield": True,
            "no_neutron_yield_target_provided": True,
        },
        "uncertainty": {
            "sheath_velocity_shot_to_shot_variation_fraction": 0.25,
            "diagnostic_timing_propagation_delay_must_be_accounted": True,
            "source_is_theory_interpretation_not_same_shot_dataset": True,
            "requires_3d_field_and_velocity_diagnostics_for_closure": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_dI_dt_trace",
            "digitized_voltage_trace",
            "per_point_trace_uncertainty",
            "instrument_response_for_2m_signal_path",
        ],
        "missing_for_full_tier2": [
            "same_shot_current_front_axis_arrival_time",
            "same_shot_plasma_breakup_time",
            "same_shot_voltage_spike_time",
        ],
        "missing_for_full_tier3": [
            "volume_integrated_J_dot_E_from_3d_fields",
            "magnetic_energy_decomposition_by_component",
            "circuit_energy_balance_with_anomalous_impedance",
        ],
        "missing_for_full_tier4": [
            "3d_magnetic_field_measurement",
            "3d_velocity_field_measurement",
            "density_uncertainty_for_current_carrying_layer",
        ],
        "missing_for_full_tier5": [
            "mechanism_linked_neutron_timing",
            "neutron_yield_response_to_anomalous_impedance",
            "detector_response_and_anisotropy",
        ],
        "validation_note": (
            "This source is a circuit-field coupling constraint. It prevents "
            "post-stagnation DPF behavior from being reduced to a scalar "
            "time-varying inductance unless the missing 3D magnetic, velocity, "
            "and Poynting-power terms are explicitly bounded or modeled."
        ),
    }


def pfz200_hybrid_xpinch_proton_neutron_targets() -> dict[str, object]:
    """Return KR-backed PFZ-200 hybrid X-pinch proton/neutron targets."""
    source = "KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-plasma-focus-2.md"
    return {
        "target_id": "pfz200_hybrid_xpinch_proton_neutron_2026_novotny",
        "device": "PFZ-200 hybrid X-pinch",
        "model_role": "kr_hybrid_xpinch_particle_source_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract": "59-69",
            "device_and_geometry": "134-149",
            "proton_and_neutron_diagnostics": "151-183",
            "plasma_visualization": "204-220",
            "neutron_campaigns_and_timing": "223-269",
            "proton_imaging_filters": "271-305",
            "five_mm_gap_yield_context": "308-317",
            "three_mm_gap_source_size": "410-459",
            "pinhole_source_context": "463-475",
            "proton_spectra": "609-645,681-704",
        },
        "shot_context": {
            "device": "PFZ-200",
            "configuration": "deuterium gas hybrid X-pinch driven by small DPF",
            "stored_energy_kJ": 3.0,
            "discharge_current_kA_min": 200.0,
            "rise_time_us": 1.6,
            "peak_current_rise_time_us_max": 2.0,
            "average_current_derivative_kA_per_ns": 0.1,
            "deuterium_pressure_Pa": 360.0,
            "anode_length_mm": 150.0,
            "anode_diameter_mm": 25.0,
            "cathode_rods": 12,
            "ak_gaps_mm": [3.0, 5.0],
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "measured_current_derivative_available": True,
            "rogowski_coil_location": "plasma focus collector",
            "hybrid_gap_does_not_change_max_current_waveform": True,
        },
        "phase_timing": {
            "neutron_production_fwhm_ns": {
                "hybrid_3mm_gap": [20.0, 7.0],
                "hybrid_5mm_gap_table": [27.0, 8.0],
                "hybrid_5mm_gap_text": [26.0, 6.0],
                "unmodified_dpf": [38.0, 9.0],
            },
            "neutron_timing_reference": "relative to beginning of HXR emission",
        },
        "spatial_geometry_targets": {
            "unmodified_cylindrical_anode_column_length_mm": 10.0,
            "conical_anode_column_length_mm": 6.0,
            "hybrid_5mm_gap_pinch_length_mm_range": [3.0, 4.0],
            "hybrid_3mm_gap_source_size_mm_range": [0.8, 2.0],
            "hybrid_3mm_gap_single_shot_height_mm": [1.3, 0.2],
            "hybrid_3mm_gap_single_shot_width_mm": [1.4, 0.3],
            "hybrid_3mm_gap_conclusion_diameter_mm_range": [1.1, 1.5],
        },
        "neutron_yield_targets": {
            "typical_hybrid_neutrons_per_shot_range": [1.0e7, 1.0e8],
            "hybrid_max_neutrons_per_shot": 1.0e8,
            "hybrid_average_after_first_shot_ignored": 6.0e7,
            "unmodified_dpf_typical_neutrons_per_shot_range": [1.0e8, 1.5e8],
            "hybrid_5mm_gap_neutrons_per_shot_range": [2.0e7, 9.0e7],
            "single_shot_3mm_neutrons": 4.0e7,
            "accumulated_3mm_neutrons": 2.0e8,
        },
        "proton_source_targets": {
            "hybrid_3mm_gap_proton_yield_order": 1.0e7,
            "single_shot_proton_yield": 2.0e7,
            "multiple_shot_accumulated_proton_yield": 9.0e7,
            "dominant_proton_energy_MeV": 3.2,
            "maximum_proton_energy_MeV": 3.6,
            "deuteron_energy_for_3p2MeV_proton_keV": 300.0,
            "maximum_deuteron_energy_MeV": 1.3,
            "proton_energy_angle_deg": 81.0,
        },
        "event_sequence": [
            {
                "event": "hybrid_xpinch_neutron_pulse",
                "mechanism": "DD fusion in localized hybrid X-pinch source",
                "fwhm_ns": 20.0,
                "required": True,
            },
        ],
        "activation_requirements": {
            "silver_activation_counter": True,
            "sac_model": "Los Alamos",
            "ntof_detector_distances_m": [0.30, 0.35, 2.5, 4.3],
            "scintillator": "BC-408",
            "scintillator_diameter_in": 2.0,
            "scintillator_length_cm": 5.0,
            "lead_shielding_cm": 5.0,
        },
        "response_model_requirements": [
            "silver_activation_calibration",
            "ntof_detector_response",
            "hxr_neutron_discrimination",
            "lead_shielding_effect",
            "proton_neutron_anisotropy_correction",
            "cr39_track_overlap_correction",
        ],
        "diagnostic_requirements": {
            "schlieren_laser_wavelength_nm": 532.0,
            "schlieren_pulse_duration_ns": [2.0, 0.5],
            "schlieren_laser_energy_mJ_max": 87.0,
            "cr39_etch_solution": "6 M NaOH",
            "cr39_etch_time_h": 6.0,
            "cr39_etch_temperature_C": 80.0,
            "pinhole_diameter_um": 900.0,
            "aluminum_filter_um_nominal": 70.0,
            "proton_filter_threshold_MeV": 2.75,
        },
        "uncertainty": {
            "neutron_fwhm_uncertainties_reported": True,
            "source_size_uncertainties_reported": True,
            "grid_thickness_mm": [0.50, 0.06],
            "shot_to_shot_source_fluctuation_noted": True,
            "sac_yield_may_be_overestimated_by_anisotropy": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "neutron_timing",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_current_derivative_trace",
            "per_point_current_uncertainty",
        ],
        "missing_for_full_tier2": [
            "ordinary_dpf_axial_radial_phase_endpoints",
            "current_trace_to_neutron_pulse_alignment",
        ],
        "missing_for_full_tier5": [
            "complete neutron_detector_response_model",
            "absolute proton_neutron_anisotropy",
            "time_resolved proton emission history",
            "ordinary_dpf_validation_scope",
        ],
        "validation_note": (
            "This target is for a modified hybrid X-pinch load on PFZ-200. It "
            "is useful for localized DD proton/neutron source diagnostics and "
            "time-of-flight response checks, but it is not an ordinary DPF "
            "end-to-end validation packet."
        ),
    }


def llnl_fully_kinetic_dpf_targets() -> dict[str, object]:
    """Return KR-backed LLNL fully kinetic DPF simulation targets."""
    source = (
        "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-"
        "z-pinch-8.md"
    )
    return {
        "target_id": "llnl_fully_kinetic_dpf_2012_schmidt",
        "device": "LLNL DPF kinetic benchmark",
        "model_role": "kr_kinetic_fidelity_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract": "20-32",
            "dpf_phase_and_kinetic_need": "35-66",
            "pic_setup": "70-99",
            "current_impedance": "102-112",
            "fields_beams_and_temperature": "113-142",
            "neutron_yield_comparison": "143-151",
            "summary": "155-159",
            "figure_context": "242-274",
        },
        "shot_context": {
            "device": "LLNL DPF",
            "configuration": "2D cylindrical r-z fully kinetic DPF run-in/pinch benchmark",
            "fill_gas": "deuterium",
            "fill_pressure_torr": 1.0,
            "steady_state_current_kA": 180.0,
            "initial_voltage_drop_kV": 4.0,
            "incoming_voltage_ramp_ns": 10.0,
        },
        "simulation_context": {
            "code": "LSP",
            "method": "implicit particle-in-cell",
            "compared_models": [
                "fluid",
                "hybrid kinetic ions/fluid electrons",
                "fully kinetic",
            ],
            "coordinates": "2D cylindrical r,z",
            "anode_length_cm": 5.0,
            "anode_outer_radius_cm": 1.0,
            "anode_inner_radius_cm": 0.5,
            "cathode_radius_cm": 1.5,
            "domain_radius_cm": 1.5,
            "domain_length_cm": 10.0,
            "grid_r_by_z": [322, 151],
            "initial_sheath_width_mm": 1.0,
            "neutral_density_cm3": 6.7e16,
            "sheath_density_cm3": 3.3e17,
            "sweep_efficiency_fraction": 0.25,
            "initial_time_step_ns": 2.4e-4,
            "final_kinetic_time_step_ns": 8.5e-6,
        },
        "phase_semantics": {
            "run_down": (
                "Plasma sheath is pushed down the inner electrode by J x B, "
                "ionizing and sweeping neutral gas."
            ),
            "run_in": (
                "After the sheath reaches the anode end, it collapses radially "
                "inward."
            ),
            "pinch": (
                "Axis implosion creates high-density plasma and high-energy "
                "electron/ion beams, x-rays, and neutrons in deuterium."
            ),
        },
        "current_waveform_targets": {
            "fully_kinetic_current_dip_kA": 15.0,
            "fully_kinetic_current_dip_fraction": 0.08,
            "experimental_current_dip_kA_max_near_1_torr": 40.0,
            "plasma_impedance_initial_ohm": 0.020,
            "plasma_impedance_pinch_ohm": 1.0,
        },
        "field_context": {
            "ez_field_complex_structure_required": True,
            "forward_ion_beams": True,
            "reverse_electron_beams": True,
            "weaker_reverse_ion_beam_allowed": True,
            "field_probe_location_r_cm": 0.05,
            "field_probe_location_z_cm": 5.0,
            "field_sample_interval_time_steps": 2,
            "pre_pinch_frequency_response_GHz": "few",
            "pinch_frequency_response_multiple_GHz": 4.0,
            "lower_hybrid_frequency_range_GHz": [10.0, 20.0],
            "lower_hybrid_drift_instability_implicated": True,
        },
        "temperature_targets": {
            "fully_kinetic_hot_pinch_ion_temperature_keV": 12.0,
            "fully_kinetic_hot_pinch_electron_temperature_keV": 3.0,
            "hybrid_max_ion_tail_keV": 200.0,
            "direct_experimental_temperature_measured": False,
        },
        "spectral_targets": {
            "fully_kinetic_predicts_ion_energy_MeV_min": 1.0,
            "kj_class_dpf_measured_ion_energy_MeV_max": 8.0,
            "llnl_measured_ion_beam_energy_keV_min": 400.0,
            "hybrid_ion_tail_keV_max": 200.0,
        },
        "neutron_yield_targets": {
            "fully_kinetic_neutrons_per_shot": 0.86e7,
            "llnl_experimental_neutrons_per_shot_max_at_180kA": 2.0e7,
            "hybrid_neutrons_per_shot": 3.6e4,
            "fluid_neutrons_per_shot": 0.0,
            "beam_target_dominance_expected_at_low_current": True,
        },
        "event_sequence": [
            {
                "event": "run_down",
                "mechanism": "J x B sheath acceleration and neutral sweep-up",
                "required": True,
            },
            {
                "event": "run_in",
                "mechanism": "radial sheath collapse after anode-end arrival",
                "required": True,
            },
            {
                "event": "pinch",
                "mechanism": "axis implosion with high-density plasma",
                "required": True,
            },
            {
                "event": "beam_output",
                "mechanism": "kinetic ion/electron beam formation",
                "required": True,
            },
        ],
        "uncertainty": {
            "shot_to_shot_current_dip_variation_noted": True,
            "experimental_yield_context_is_upper_bound": True,
            "digitized_waveform_uncertainty_provided": False,
            "model_comparison_not_experimental_uncertainty_budget": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "digitized_voltage_trace_points",
            "per_point_current_voltage_uncertainty",
        ],
        "missing_for_full_tier4": [
            "direct_experimental_density_map",
            "direct_experimental_temperature_map",
            "direct_experimental_magnetic_or_electric_field_map",
            "three_dimensional_effect_validation",
            "validated_collision_ionization_and_anomalous_resistivity_models",
        ],
        "missing_for_full_tier5": [
            "neutron_detector_response_model",
            "time_resolved_experimental_neutron_spectrum",
            "mechanism_separated_experimental_neutron_history",
            "shot_ensemble_uncertainty",
            "three_dimensional_kinetic_validation",
        ],
        "validation_note": (
            "This KR source establishes that fully kinetic physics is necessary "
            "to reproduce MeV ions and approximate neutron yield in this "
            "low-current DPF benchmark. It is a simulation-to-experiment "
            "context target, not a direct experimental dataset, and cannot "
            "close predictive readiness by itself."
        ),
    }


def nstec_3d_mhd_rundown_targets() -> dict[str, object]:
    """Return KR-backed NSTec/Gemini 3D-MHD rundown benchmark targets."""
    source = (
        "KnowledgeReference/fully-three-dimensional-simulation-and-modeling-"
        "of-a-dense-plasma-focus.md"
    )
    return {
        "target_id": "nstec_3d_mhd_rundown_2014_meehan",
        "device": "NSTec / Gemini DPF",
        "model_role": "kr_3d_mhd_rundown_benchmark_target",
        "validation_tier": 3,
        "source": source,
        "source_lines": {
            "abstract": "27-48",
            "mhd_motivation_and_3d_need": "86-124",
            "device_geometry_and_bank": "140-183",
            "faraday_current_diagnostic": "184-247",
            "shot_repeatability": "267-279",
            "alegra_setup_eos_and_circuit": "280-313,339-350",
            "mhd_scope_limits_and_startup": "351-380",
            "two_dimensional_limits": "381-421",
            "density_floor_and_3d_flow": "432-511",
            "current_and_rundown_comparison": "514-566",
            "figure_current_comparison": "587-599",
            "conclusion": "600-624",
        },
        "shot_context": {
            "device": "NSTec / Gemini DPF",
            "fill_gas": "deuterium",
            "nominal_fill_pressure_torr_about": 7.0,
            "comparison_voltage_kV": 37.5,
            "comparison_pressure_torr": 7.28,
            "repeat_shots": 37,
            "bank_capacitance_uF": 432.0,
            "bank_voltage_kV_max": 70.0,
            "bank_energy_MJ_max": 1.0,
            "coaxial_cables": 36,
            "rail_gap_switches": 8,
            "cathode_bars": 24,
            "cathode_bar_thickness_in": 0.375,
            "cathode_height_in": 30.75,
            "cathode_inside_diameter_in": 6.0,
            "anode_outer_diameter_in": 4.0,
            "anode_height_above_ground_in": 23.6,
            "vacuum_chamber_diameter_in": 12.0,
            "insulator_material": "Pyrex",
            "insulator_thickness_in": 0.5,
            "insulator_height_in": 8.63,
        },
        "phase_semantics": {
            "inverse_pinch": (
                "Gas expands outward from anode to cathode bars after Marx "
                "breakdown and shock formation."
            ),
            "run_down": (
                "Once gas touches the cathode, plasma moves up the anode "
                "until Z-pinch at the anode top."
            ),
            "mhd_scope": (
                "Inverse pinch and rundown are the phases treated as "
                "approximately governed by MHD in this source."
            ),
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "diagnostic": "Faraday rotator",
            "faraday_loop_turns": 5.25,
            "faraday_loop_calibration_factor_dependent": False,
            "repeat_shot_current_profiles_nearly_identical": True,
            "measured_peak_current_MA": 2.17,
            "two_dimensional_peak_current_MA": 2.08,
            "two_dimensional_peak_current_error_fraction": 0.04,
            "three_dimensional_peak_current_MA": 1.82,
            "three_dimensional_peak_current_error_fraction": 0.16,
            "series_inductance_nH_nominal": 25.0,
            "two_dimensional_tweaked_series_inductance_nH": 28.2,
            "tweaked_inductance_has_experimental_justification": False,
        },
        "phase_timing": {
            "rundown_end_definition": "maximum derivative of current profile",
            "experimental_rundown_time_us": 6.96,
            "three_dimensional_rundown_time_us": 6.69,
            "three_dimensional_rundown_error_fraction_max": 0.04,
            "two_dimensional_rundown_time_us": 5.59,
            "two_dimensional_rundown_error_fraction_about": 0.20,
            "three_dimensional_better_for_rundown": True,
        },
        "field_context": {
            "code": "ALEGRA-MHD",
            "method": "finite-element ALE resistive MHD",
            "external_circuit_solver_coupled": True,
            "two_dimensional_systematic_lower_inductance": True,
            "two_dimensional_systematic_higher_current": True,
            "two_dimensional_unrealistically_fast_rundown": True,
            "three_dimensional_larger_inductance_from_cathode_bar_flow": True,
            "three_dimensional_lower_peak_current": True,
            "three_dimensional_longer_rundown": True,
            "helical_asymmetric_or_off_axis_currents_require_3d": True,
        },
        "spatial_density_targets": {
            "density_floor_kg_per_m3": 2.5e-4,
            "density_floor_suppresses_lmd_conductivity": True,
            "density_snapshot_time_us_about": 3.0,
            "three_dimensional_plasma_flows_around_cathode_bars": True,
            "three_dimensional_plasma_slightly_slower_and_less_dense": True,
        },
        "temperature_targets": {
            "startup_hot_gas_layer_temperature_K": 1.0e6,
            "startup_layer_stabilized_temperature_K": 1.0e4,
            "startup_stabilization_time_ns_about": 20.0,
            "conductivity_floor_temperature_eV_about": 1.0,
            "direct_experimental_temperature_measured": False,
        },
        "model_scope_limits": {
            "breakdown_not_covered_by_mhd": True,
            "hot_start_layer_is_artificial": True,
            "near_z_pinch_mhd_unphysical": True,
            "missing_kinetic_instability_resistivity": True,
            "pic_transfer_required_near_pinch": True,
            "empirical_models_require_existing_similar_data": True,
        },
        "neutron_validation_context": {
            "neutron_yield_not_measured_in_this_target": True,
            "be_activation_detector_named_for_deuterium": True,
            "pr_activation_detector_named_for_dt": True,
            "maximum_current_and_rundown_are_prerequisites_for_yield_prediction": True,
        },
        "uncertainty": {
            "repeat_shot_consistency_context_shots": 37,
            "peak_current_error_fractions_reported": True,
            "rundown_error_fractions_reported": True,
            "digitized_waveform_uncertainty_provided": False,
            "faraday_verdet_uncertainty_provided": False,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_faraday_current_traces",
            "per_point_current_uncertainty",
            "verdet_constant_uncertainty",
            "per_shot_peak_current_statistics",
        ],
        "missing_for_full_tier2": [
            "inverse_pinch_start_time",
            "inverse_pinch_end_time",
            "independent_z_pinch_time_diagnostic",
            "rundown_time_uncertainty",
        ],
        "missing_for_full_tier3": [
            "mesh_resolution_and_convergence_record",
            "circuit_energy_conservation_record",
            "field_derived_inductance_validation",
            "backend_parity_or_restart_reproducibility",
        ],
        "missing_for_full_tier4": [
            "direct_experimental_density_map",
            "direct_experimental_temperature_map",
            "direct_experimental_magnetic_or_electric_field_map",
        ],
        "missing_for_full_tier5": [
            "measured_neutron_yield",
            "activation_detector_response",
            "neutron_timing_spectrum_anisotropy",
            "beam_target_or_pic_pinch_closure",
        ],
        "validation_note": (
            "This target supports 3D-MHD rundown and current-waveform "
            "benchmarking against Faraday current data. It explicitly does "
            "not validate the late Z-pinch, kinetic resistivity, neutron "
            "production, or detector response."
        ),
    }


def alegra_hedp_dpf_mhd_validation_targets() -> dict[str, object]:
    """Return KR-backed ALEGRA-HEDP DPF MHD benchmark and limit targets."""
    source = (
        "KnowledgeReference/unlimited-release-printed-september-2009-alegra-"
        "hedp-simulations-of-the-dense-plasma-focus.md"
    )
    return {
        "target_id": "alegra_hedp_dpf_mhd_validation_2009_kueny",
        "device": "Bernard Long / Bernard Short / Tallboy",
        "model_role": "kr_mhd_validation_limit_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_and_scope": "118-131",
            "operation_regime": "251-261",
            "pinch_physics_limits": "265-301",
            "targeted_experiments": "305-326",
            "alegra_physics_capabilities": "331-341",
            "setup_eos_resolution_and_seed_limits": "347-387",
            "bernard_long_results": "399-459",
            "bernard_short_and_tallboy_results": "470-522",
            "comparison_table": "523-547,590-597",
            "discussion_and_next_steps": "549-577",
        },
        "phase_semantics": {
            "sheath_creation": "gas breakdown along the insulator",
            "lift_off": "self-magnetic-field driven lift-off of a millimeter-scale sheath",
            "run_down": "plasma sheath acceleration down the coaxial cavity",
            "collapse": "anode-tip rollover into a hot, dense pinch",
            "early_pinch_stop": (
                "MHD calculations are stopped when charge separation and "
                "instabilities make the MHD approximation fail."
            ),
        },
        "phase_timing": {
            "pinch_lifetime_ns_range_bernard_long_experiment": [20.0, 50.0],
            "alegra_post_axis_mhd_valid_time_ns_order": 1.0,
            "device_voltage_kV_range": [20.0, 50.0],
            "gas_fill_torr_range": [3.0, 10.0],
            "current_MA_range": [0.6, 1.8],
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "digitized_current_traces_available_in_target": False,
            "bernard_long_peak_current_experiment_MA": 0.6,
            "bernard_long_peak_current_alegra_MA_range": [0.5, 0.6],
            "bernard_short_peak_current_experiment_MA": 1.5,
            "bernard_short_peak_current_alegra_MA": 1.5,
            "tallboy_peak_current_experiment_MA": 2.3,
            "tallboy_peak_current_alegra_MA": 1.8,
        },
        "shot_context": {
            "bernard_long": {
                "capacitance_uF": 135.0,
                "charging_voltage_kV": 20.0,
                "stored_energy_kJ": 27.0,
                "stray_inductance_nH_estimated": 27.0,
                "resistance_mohm_estimated": 3.3,
                "fill_pressure_torr": 3.0,
                "anode_length_cm": 25.0,
            },
            "bernard_short": {
                "capacitance_uF": 120.0,
                "charging_voltage_kV": 40.0,
                "stored_energy_kJ": 96.0,
                "resistance_mohm_estimated": 3.3,
                "fill_pressure_torr": 10.0,
                "anode_length_cm": 15.0,
            },
            "tallboy": {
                "capacitance_uF": 216.0,
                "charging_voltage_kV": 50.0,
                "stored_energy_kJ": 270.0,
                "inductance_nH": 50.0,
                "resistance_mohm_assumed": 3.3,
                "anode_length_cm": 66.0,
            },
        },
        "spatial_density_targets": {
            "generic_pinch_width_mm": 1.0,
            "generic_pinch_length_mm_order": 1.0,
            "generic_pinch_density_cm3_range": [1.0e19, 1.0e20],
            "bernard_long_experiment_pinch_density_cm3_range": [1.0e18, 5.0e19],
            "bernard_long_alegra_pinch_number_density_cm3": 1.4e19,
            "bernard_long_alegra_peak_mass_density_kg_per_m3": 0.045,
            "bernard_long_closed_case_peak_mass_density_kg_per_m3": 0.079,
        },
        "temperature_targets": {
            "generic_pinch_temperature_eV_min_order": 100.0,
            "bernard_long_experiment_pre_pinch_ion_temperature_eV": 300.0,
            "bernard_long_experiment_pinch_ion_temperature_eV": 700.0,
            "bernard_long_alegra_pre_pinch_ion_temperature_eV_range": [250.0, 650.0],
            "bernard_long_alegra_pre_pinch_electron_temperature_eV_range": [200.0, 360.0],
            "bernard_long_alegra_pinch_ion_temperature_keV": 9.0,
            "computed_pinch_temperatures_high_unresolved": True,
        },
        "field_context": {
            "magnetics_modeled": True,
            "lumped_circuit_model": True,
            "lsp_magnetic_field_import_capability_incomplete": True,
            "cathode_bar_3d_fields_required": True,
        },
        "neutron_yield_targets": {
            "bernard_long_experiment_neutrons": 1.5e9,
            "bernard_long_alegra_thermonuclear_neutrons": 1.2e5,
            "bernard_short_experiment_neutrons": 3.0e10,
            "bernard_short_alegra_thermonuclear_neutrons": 1.5e6,
            "tallboy_experiment_neutrons": 3.5e11,
            "tallboy_alegra_thermonuclear_neutrons": 3.7e7,
            "mhd_expected_to_underpredict_total_yield": True,
        },
        "mhd_scope_limits": {
            "mhd_can_model_only_thermonuclear_component": True,
            "nonthermal_neutron_mechanisms_required": True,
            "charge_separation_breaks_mhd": True,
            "instabilities_break_mhd_after_early_pinch": True,
            "two_dimensional_model_misses_cathode_bars": True,
            "two_dimensional_model_misses_filamentation": True,
            "three_dimensional_mhd_required_before_particle_followup": True,
            "pic_to_mhd_sheath_import_needed": True,
        },
        "numerical_model_limits": {
            "sesame_lowest_meaningful_density_kg_per_m3": 0.01,
            "initial_fill_below_sesame_consistency": True,
            "qEOS_deuterium_used": True,
            "tabular_qEOS_and_full_qEOS_differ_little": True,
            "cell_size_mm": 0.5,
            "seed_ionized_gas_temperature_eV": 1.0,
            "seed_layer_arbitrary": True,
            "radiation_diffusion_not_used": True,
            "thermal_emission_model_used": True,
        },
        "uncertainty": {
            "bernard_geometry_inputs_partly_estimated": True,
            "circuit_LR_values_estimated_for_bernard_cases": True,
            "tallboy_comparison_under_review": True,
            "digitized_traces_not_in_target": True,
            "neutron_detector_response_not_in_target": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_traces",
            "per_point_current_uncertainty",
            "validated_circuit_LR_for_bernard_cases",
            "resolved_tallboy_current_discrepancy",
        ],
        "missing_for_full_tier2": [
            "digitized_lift_off_rundown_collapse_timing",
            "same-shot_phase_endpoint_uncertainty",
            "validated_3d_cathode_bar_geometry",
        ],
        "missing_for_full_tier4": [
            "direct_same-shot_temperature_uncertainty",
            "direct_same-shot_density_uncertainty",
            "validated_magnetic_field_import_or_measurement",
            "resolution_convergence_for_pinch_temperature",
        ],
        "missing_for_full_tier5": [
            "nonthermal_beam_target_neutron_model",
            "neutron_timing_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response_model",
            "kinetic_or_pic_followup_after_mhd_breakdown",
        ],
        "validation_note": (
            "This report is valuable because peak current and early-pinch MHD "
            "observables agree for the Bernard devices, while total neutron "
            "yield is underpredicted by orders of magnitude exactly as the KR "
            "source expects. It should validate early MHD/circuit behavior, "
            "not end-to-end neutron prediction."
        ),
    }


def blagoev_electric_flux_diagnostic_targets() -> dict[str, object]:
    """Return KR-backed DPF electric-flux diagnostic targets."""
    source = (
        "KnowledgeReference/measurement-of-electric-flux-emission-a-new-"
        "diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md"
    )
    return {
        "target_id": "blagoev_electric_flux_diagnostic_2025",
        "device": "University of Sofia plasma focus",
        "model_role": "kr_formation_symmetry_diagnostic_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_and_predictive_modeling_limit": "30-44",
            "dpf_phase_context_and_singularity": "47-62,83-95",
            "formation_modeling_motivation": "102-116",
            "electric_flux_theory_and_probe_basis": "117-170",
            "probe_hardware_and_circuit_context": "174-194",
            "calibration_and_baseline_symmetry": "367-412",
            "sofia_device_and_anode_context": "426-455",
            "phase_interpretation_and_shot_examples": "466-497",
            "shot_667_and_conclusion": "506-522,538-542",
        },
        "shot_context": {
            "device": "University of Sofia plasma focus",
            "configuration": "Mather type",
            "stored_energy_kJ": 3.0,
            "capacitance_uF": 20.0,
            "charging_voltage_kV_max": 40.0,
            "anode_material_initial": "hollow copper tube",
            "anode_diameter_cm": 2.0,
            "anode_length_cm": 14.5,
            "cathode_rods": 6,
            "cathode_rod_diameter_cm": 0.8,
            "cathode_rod_length_cm": 16.0,
            "cathode_circle_radius_cm": 3.5,
            "vacuum_chamber_inner_diameter_cm": 15.5,
            "vacuum_chamber_height_cm": 35.0,
            "operating_gases": ["air", "argon", "deuterium"],
            "tested_shots": [
                {
                    "shot": 665,
                    "fill_gas": "argon",
                    "fill_pressure_torr": 0.95,
                    "charging_voltage_kV": 19.0,
                },
                {
                    "shot": 668,
                    "fill_gas": "argon",
                    "fill_pressure_torr": 0.83,
                    "charging_voltage_kV": 19.1,
                },
                {
                    "shot": 667,
                    "fill_gas": "argon",
                    "fill_pressure_torr": 0.77,
                    "charging_voltage_kV": 19.0,
                    "reference_singularity_time_us": 3.03,
                },
            ],
        },
        "current_waveform_targets": {
            "current_derivative_singularity_is_phase_reference": True,
            "current_maximum_signifies_end_of_rundown": True,
            "radial_phase_between_current_maximum_and_singularity": True,
            "rogowski_signal_can_be_electric_flux_contaminated": True,
            "d_dot_oscillations_frequency_phase_correlated_with_dI_dt": True,
            "sampling_interval_ns": 1.0,
            "smoothing_points": 10,
        },
        "phase_timing": {
            "formation_phase_detected_by_early_d_dot_signals": True,
            "rundown_phase_signals_should_have_similar_shape_and_magnitude": True,
            "radial_phase_signals_differ_considerably": True,
            "lower_pressure_shot_has_earlier_singularity": True,
            "shot_667_reference_singularity_time_us": 3.03,
            "surface_discharge_can_begin_earlier_near_one_probe": True,
        },
        "electric_flux_diagnostic_targets": {
            "three_symmetric_identical_d_dot_probes": True,
            "probes_measure_electric_flux_from_finite_plasma": True,
            "central_pin_of_sma_adapter_is_floating_conductor": True,
            "coax_terminated_ohms_both_ends": 50.0,
            "probe_support_capacity": 6,
            "used_probe_channels": ["CH2", "CH3", "CH4"],
            "electric_flux_asymmetry_indicates_plasma_formation_asymmetry": True,
            "d_dot_probe_immune_to_magnetic_field_in_source_interpretation": True,
            "motional_electric_field_dominates_rundown": True,
        },
        "calibration_targets": {
            "baseline_symmetric_test_uses_central_conductor": True,
            "voltage_divider_resistances_ohm": [1306.0, 13.2],
            "applied_voltage_kV_max": 5.34,
            "integrated_d_dot_max_within_percent_of_mean": 3.0,
            "capacitance_C1_pF_estimate": 0.006,
            "baseline_symmetry_deviation_is_environment_property": True,
        },
        "hardware_fault_targets": {
            "hollow_copper_anode_hidden_deformation_detected": True,
            "pinch_still_formed_despite_anode_deformation": True,
            "electric_flux_emission_depends_on_anode_geometry": True,
            "solid_brass_electrode_replacement_used_for_normal_operation": True,
            "asymmetrical_magnetic_impulse_can_amplify_distortion": True,
        },
        "modeling_scope_limits": {
            "formation_process_atomic_radiative_multiscale": True,
            "predictive_modeling_of_formation_reported_very_difficult": True,
            "diagnostic_provides_symmetry_duration_dynamics_not_full_state": True,
            "not_a_neutron_yield_validation_dataset": True,
        },
        "uncertainty": {
            "baseline_integrated_signal_spread_percent": 3.0,
            "sampling_interval_ns": 1.0,
            "moving_average_points": 10,
            "preliminary_results_only": True,
            "no_digitized_probe_waveforms_in_target": True,
            "no_per_point_probe_uncertainty": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_magnetic_or_em",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_dI_dt_trace",
            "digitized_d_dot_waveforms",
            "per_point_waveform_uncertainty",
            "rogowski_electric_flux_pickup_characterization",
        ],
        "missing_for_full_tier2": [
            "absolute_current_maximum_time_for_each_shot",
            "absolute_singularity_time_for_each_shot",
            "independent phase endpoint diagnostic",
        ],
        "missing_for_full_tier4": [
            "calibrated_azimuthal_electric_flux_field_reconstruction",
            "simultaneous_density_temperature_magnetic_field_profiles",
            "probe_transfer_function_uncertainty",
        ],
        "missing_for_full_tier5": [
            "neutron_yield_for_same_shots",
            "neutron_timing_spectrum_anisotropy",
            "detector_response_coupling_to_neutron_outputs",
        ],
        "validation_note": (
            "This source gives a formation-symmetry and electric-flux "
            "diagnostic constraint for DPF startup and rundown. It should be "
            "used to reject models that ignore azimuthal formation asymmetry "
            "or Rogowski electric-flux pickup, but it is not an end-to-end "
            "neutron or pinch-validation dataset."
        ),
    }


def auluck_poloidal_magnetic_field_targets() -> dict[str, object]:
    """Return KR-backed poloidal/axial magnetic-field dynamo targets."""
    source = "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md"
    return {
        "target_id": "auluck_poloidal_magnetic_field_2024",
        "device": "generic / small DPF",
        "model_role": "kr_poloidal_field_dynamo_scope_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "magnetic_probe_limits_and_indirect_evidence": "80-110",
            "scope_and_simple_dynamo_hypothesis": "113-141",
            "ohm_law_geomagnetic_seed_and_Rm": "145-163",
            "gpf_scaling_context": "166-192",
            "gv_surface_and_current_context": "260-292",
            "flux_function_and_field_equations": "317-390,415-456",
            "dynamo_implications": "478-528",
            "current_loss_and_external_field_test": "529-590",
            "nikulin_torque_observation": "595-604,760-763",
        },
        "phase_semantics": {
            "plasma_formation": "generic plasma focus formation phase",
            "plasma_propagation": (
                "GV-surface propagation with scaled variables moving at the "
                "scaling velocity"
            ),
            "approach_to_axis": (
                "generic plasma focus approach-to-symmetry-axis phase"
            ),
        },
        "current_waveform_targets": {
            "mhd_codes_neglecting_dynamo_can_overestimate_observed_current": True,
            "apparent_current_loss_may_be_azimuthal_circulating_current": True,
            "lee_radial_current_fraction_should_vary_with_external_axial_field": True,
            "monitor_current_derivative_and_integrated_current_for_test": True,
            "equivalent_loop_voltage_can_include_geomagnetic_term": True,
            "geomagnetic_term_independent_of_charging_voltage": True,
        },
        "magnetic_field_targets": {
            "poloidal_flux_emission_outside_plasma_is_indirect_axial_field_evidence": True,
            "direct_point_measurement_of_axial_B_inside_plasma_is_rejected": True,
            "magnetic_probe_spatial_resolution_mm_range": [1.0, 2.0],
            "magnetic_probe_alters_plasma_and_current_source": True,
            "faraday_rotation_abel_transform_not_available_for_axial_component": True,
            "simple_dynamo_seeded_by_geomagnetic_field": True,
            "curved_plasma_armature_drives_azimuthal_electric_field": True,
            "zero_resistivity_limit_assumed": True,
            "hall_term_neglected_as_model_assumption": True,
            "magnetic_reynolds_number_assumed_much_greater_than_one": True,
            "exponential_poloidal_field_growth_possible_with_positive_separation_constant": True,
        },
        "field_context": {
            "gpf_scaling_coordinates_by_anode_radius": True,
            "gpf_scaling_density_by_fill_gas_mass_density": True,
            "gpf_scaling_magnetic_field": "B0 = mu0 * I(t) / (2*pi*a*r_tilde)",
            "gpf_scaling_velocity_from_B0_and_fill_density": True,
            "gv_solution_mather_surfaces_resemble_experiment": True,
            "velocity_defined_normal_to_gv_surface_in_armature_region": True,
            "flux_function_U_evolves_by_hamilton_jacobi_form": True,
            "electric_field_Etheta_proportional_to_hamiltonian": True,
            "axial_B_from_radial_derivative_of_flux_function": True,
            "radial_B_from_axial_derivative_of_flux_function": True,
            "azimuthal_current_density_depends_on_flux_spatial_structure": True,
        },
        "experimental_test_requirements": {
            "uniform_axial_field_over_entire_small_dpf_volume": True,
            "helmholtz_coil_dc_source_variable_polarity": True,
            "external_field_amplitude_max_times_local_geomagnetic": 2.0,
            "monitor_poloidal_magnetic_flux_emission": True,
            "look_for_variation_near_geomagnetic_null": True,
            "nonuniform_high_field_has_no_interpretive_value": True,
            "excessively_high_field_disturbs_plasma_armature": True,
        },
        "neutron_context": {
            "gemini_neutron_yield_control_question_is_hypothesis": True,
            "neutron_fluence_anisotropy_control_question_is_hypothesis": True,
            "no_neutron_yield_target_provided": True,
        },
        "supporting_observation": {
            "nikulin_cone_plasma_focus_energy_kJ": 2.5,
            "cone_twisted_not_radially_imploded": True,
            "pure_azimuthal_magnetic_field_cannot_produce_torque": True,
            "torque_demonstrates_poloidal_field_must_exist_in_source_argument": True,
        },
        "modeling_scope_limits": {
            "letter_is_first_tentative_step": True,
            "not_quantitative_calculation_formalism": True,
            "boundary_value_problem_not_fully_solved": True,
            "some_shots_show_flux_increase_and_some_do_not": True,
            "indirect_evidence_not_direct_point_measurement": True,
        },
        "uncertainty": {
            "source_proposes_experimental_test_not_completed_validation": True,
            "no_digitized_flux_signal_in_target": True,
            "no_uncertainty_on_geomagnetic_field_response": True,
            "nikulin_observation_is_qualitative_supporting_evidence": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_magnetic_or_em",
            "neutron_anisotropy",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "measured_current_trace_under_external_field_sweep",
            "per_point_current_uncertainty",
        ],
        "missing_for_full_tier2": [
            "phase_resolved_poloidal_flux_signal_timing",
            "current_derivative_singularity_time_for_field_sweep",
        ],
        "missing_for_full_tier4": [
            "calibrated_poloidal_flux_signal",
            "external_axial_field_sweep_dataset",
            "radial_current_fraction_vs_external_field",
            "3d_magnetic_field_reconstruction",
        ],
        "missing_for_full_tier5": [
            "same_shot_neutron_yield_vs_external_field",
            "same_shot_neutron_anisotropy_vs_external_field",
            "detector_response_and_uncertainty",
        ],
        "validation_note": (
            "This Auluck letter adds a strong KR constraint that DPF models "
            "must not assume purely toroidal magnetic structure by default. It "
            "defines a proposed external-field test and circuit-current "
            "implication, but it does not provide the completed quantitative "
            "field-sweep validation dataset."
        ),
    }


def wante_nitrogen_ion_irradiation_targets() -> dict[str, object]:
    """Return KR-backed UNU/ICTP nitrogen-ion DPF irradiation targets."""
    source = (
        "KnowledgeReference/regular-article-nitrogen-ion-irradiation-of-"
        "carbon-thin-lms-using-a-dense-plasma-focus-enhanced.md"
    )
    return {
        "target_id": "wante_nitrogen_ion_irradiation_2025",
        "device": "UNU/ICTP PF",
        "model_role": "kr_ion_beam_material_processing_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_ion_beam_and_material_results": "47-60",
            "device_table_and_geometry": "170-205",
            "faraday_cup_and_acceleration_context": "207-240",
            "tof_energy_and_operating_pressure": "241-257,268-305",
            "pf_characteristics_and_lee_fit": "348-365,374-375",
            "edx_impurity_and_doping_response": "419-462",
            "conclusion_material_response": "605-637",
        },
        "shot_context": {
            "device": "UNU/ICTP PF",
            "stored_energy_kJ_nominal": 3.0,
            "operated_energy_kJ": 2.54,
            "capacitance_uF": 30.0,
            "charging_voltage_kV": 13.0,
            "inductance_nH": 156.0,
            "resistance_mohm": 21.4,
            "anode_radius_cm": 0.95,
            "cathode_radius_cm": 3.2,
            "anode_length_cm": 16.0,
            "anode_diameter_cm": 1.9,
            "cathode_rods": 6,
            "cathode_rod_length_cm": 16.0,
            "cathode_rod_diameter_cm": 0.95,
            "insulator_material": "Pyrex",
            "insulator_thickness_cm": 2.0,
            "insulator_length_cm": 5.0,
            "insulator_diameter_cm": 2.6,
            "fill_gas": "nitrogen",
            "nitrogen_purity_fraction": 0.99999,
            "optimal_pressure_mbar": 1.5,
            "initial_vacuum_mbar": 5.0e-3,
            "preliminary_shots_for_stable_pinch": 4,
            "irradiation_shot_counts": [6, 12, 24],
            "shot_interval_min": 5.0,
            "sample_distance_from_anode_cm": 38.0,
        },
        "current_waveform_targets": {
            "current_voltage_and_ion_signals_captured": True,
            "oscilloscope": "Yokogawa DL7480",
            "lee_model_fit_to_measured_current": True,
            "lee_fit_fm": 0.03,
            "lee_fit_fc": 0.7,
            "lee_fit_fmr": 0.18,
            "lee_fit_fcr": 0.85,
            "current_voltage_waveform_figures_not_digitized_in_target": True,
        },
        "phase_timing": {
            "pinch_formation_indicated_by_xray_signal": True,
            "xray_peak_t0_aligns_with_voltage_peak": True,
            "ion_peak_t1_almost_simultaneous_with_xray_emission": True,
            "ion_tof_definition": "t_ion_peak - t_xray_peak",
            "flight_path_cm": 38.0,
            "stable_pinch_after_initial_shots": 4,
        },
        "spatial_density_targets": {
            "pinch_particle_density_m3_range_context": [1.0e18, 1.0e20],
            "source_context_not_same_shot_profile": True,
        },
        "temperature_targets": {
            "pinch_temperature_K_order_context": 1.0e6,
            "source_context_not_same_shot_profile": True,
        },
        "ion_beam_targets": {
            "measured_nitrogen_ion_energy_keV": 72.40,
            "lee_model_nitrogen_ion_energy_keV": 71.0,
            "ion_flux_m2_s": 7.2e27,
            "ion_fluence_m2": 6.4e19,
            "faraday_cup_mode": "biased ion collector",
            "faraday_cup_bias_V": -45.0,
            "typical_pf_ion_energy_range": "tens_keV_to_MeV",
            "ion_flux_and_fluence_from_current_waveform_fit": True,
        },
        "application_response_targets": {
            "nitrogen_doping_percent_by_shots": {
                "6": 7.06,
                "12": 5.96,
                "24": 7.93,
            },
            "nitrogen_doping_rate_percent_per_shot": {
                "6": 1.18,
                "12": 0.50,
                "24": 0.33,
            },
            "nitrogen_incorporation_non_linear_with_shot_count": True,
            "copper_impurity_from_anode_ablation_increases_with_dose": True,
            "max_copper_impurity_percent_24_shots": 2.11,
            "fluorine_content_percent_as_deposited": 12.06,
            "fluorine_content_percent_min_after_irradiation": 4.94,
            "crystallite_size_nm_as_deposited": 6.27,
            "crystallite_size_nm_24_shots": 11.16,
            "xrd_new_peak_degrees_24_shots": [52.0, 76.0],
            "interlayer_spacing_nm_as_deposited": 0.37,
            "interlayer_spacing_nm_24_shots": 0.340,
        },
        "modeling_scope_limits": {
            "material_processing_target_not_neutron_validation": True,
            "ion_beam_energy_is_faraday_cup_tof_observable": True,
            "lee_fit_supports_current_and_ion_energy_only_for_this_scope": True,
            "surface_response_depends_on_material_target_and_shot_history": True,
        },
        "uncertainty": {
            "no_digitized_current_voltage_or_ion_waveforms": True,
            "no_per_point_waveform_uncertainty": True,
            "no_faraday_cup_transfer_function_uncertainty": True,
            "material_response_not_unique_machine_state_validation": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_voltage_trace",
            "digitized_ion_signal",
            "per_point_trace_uncertainty",
        ],
        "missing_for_full_tier2": [
            "absolute_xray_peak_time",
            "absolute_ion_peak_time",
            "pinch_duration",
            "phase_endpoint_uncertainty",
        ],
        "missing_for_full_tier4": [
            "same_shot_density_profile",
            "same_shot_temperature_profile",
            "same_shot_magnetic_field_profile",
            "faraday_cup_response_uncertainty",
        ],
        "missing_for_full_tier5": [
            "neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This source supplies useful UNU/ICTP nitrogen-ion beam and "
            "material-processing constraints, including a Lee-fit ion-energy "
            "comparison. It does not validate end-to-end DPF neutron, pinch, "
            "or high-fidelity plasma prediction."
        ),
    }


def demina_dpf_material_damage_targets() -> dict[str, object]:
    """Return KR-backed DPF material-irradiation damage targets."""
    source = (
        "KnowledgeReference/application-of-a-plasma-accelerator-of-the-dense-"
        "plasma-focus-type-in-simulation-of-radiation.md"
    )
    return {
        "target_id": "demina_dpf_material_damage_apdm4",
        "device": "PF-5M / PF-6 / PF-1000",
        "model_role": "kr_material_response_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_device_and_flux_scope": "24-43",
            "purpose_and_materials": "47-61",
            "radiation_conditions": "67-84",
            "tungsten_damage_and_erosion": "90-150,180-223,381-386",
            "cfc_damage_and_erosion": "152-166,225-262,388-393",
            "redeposition_and_compounds": "289-320",
            "conclusions": "322-342",
        },
        "device_context": {
            "devices": ["PF-5M", "PF-6", "PF-1000"],
            "PF_5M_bank_energy_kJ": 5.0,
            "PF_6_bank_energy_kJ": 7.0,
            "PF_1000_bank_energy_MJ": 1.2,
            "PF_1000_experiment_stored_energy_kJ_approx": 600.0,
            "working_gas": "deuterium",
            "PF_1000_initial_pressure_Pa": 470.0,
            "W_and_W_CFC_pulses": 10,
            "CFC_SiC_pulses": 5,
            "samples_positioned_at_different_distances_from_anode": True,
        },
        "radiation_environment_targets": {
            "power_flux_density_W_cm2_range": [1.0e7, 1.0e10],
            "pulse_duration_us_range": [0.2, 1.0],
            "plasma_and_fast_ion_electron_streams": True,
            "reactor_wall_surface_proxy": True,
            "ipc_reactor_level_heat_flux_context": True,
            "mpc_reactor_level_heat_flux_exceeded": True,
        },
        "material_targets": {
            "sintered_W_specimen_mm": [10.0, 10.0, 2.0],
            "W_CFC_specimen_mm": [25.0, 25.0, 10.0],
            "W_CFC_W_cylinder_diameter_mm": 12.0,
            "CFC_SiC_specimen_mm": [10.0, 10.0, 8.0],
            "CFC_SiC_volume_fraction_percent_cases": [8.0, 40.0],
            "materials": ["tungsten", "W/CFC", "CFC/SiC"],
        },
        "tungsten_damage_targets": {
            "melting_and_evaporation": True,
            "wavelike_surface_relief": True,
            "nanoscale_cellular_structure_at_W_cm2": 1.0e10,
            "intergranular_and_transgranular_microcracks_above_W_cm2": 1.0e8,
            "microcrack_penetration_um_order": 10.0,
            "bubble_size_um_order": 1.0,
            "bubble_gas_contains_implanted_deuterium": True,
            "erosion_depth_per_pulse_um_by_condition": {
                "0.8e8_W_cm2": 0.0123,
                "1.0e8_W_cm2": 0.0267,
                "2.0e8_W_cm2": 0.0696,
                "4.0e8_W_cm2": 0.0390,
                "1e10_ion_1e9_plasma_W_cm2": 2.05,
            },
            "highest_power_density_erosion_depth_um_approx": 2.0,
        },
        "cfc_damage_targets": {
            "W_droplets_and_elongated_ridges_on_CFC": True,
            "W_CFC_irradiation_power_flux_W_cm2": 3.0e10,
            "normal_fiber_orientation_evaporates_more": True,
            "parallel_fiber_orientation_has_lower_erosion_rate": True,
            "CFC_8SiC_evaporated_layer_um_per_shot_at_1e9_W_cm2": 2.6,
            "CFC_40SiC_evaporated_layer_um_per_shot_at_1e9_W_cm2": 1.9,
            "CFC_SiC_shots": 5,
        },
        "redeposition_targets": {
            "implanted_working_gas_and_structural_elements": True,
            "observed_elements_on_W": ["Cu", "O", "Fe", "Cr"],
            "observed_elements_on_CFC_SiC": ["Fe", "Cr", "Si", "Cu"],
            "steel_holder_sources_Fe_Cr": True,
            "copper_anode_source_Cu": True,
            "surface_layer_compounds": ["Fe2C", "Fe5C2", "Cu4Si", "(Cr,Fe)7C3"],
            "surface_compounds_affect_deuterium_tritium_diffusion": True,
        },
        "model_scope_limits": {
            "material_damage_target_not_core_dpf_machine_validation": True,
            "does_not_provide_current_voltage_waveforms": True,
            "does_not_provide_incident_particle_spectrum": True,
            "does_not_provide_same_shot_density_temperature_or_field_profiles": True,
            "does_not_provide_neutron_yield_timing_spectrum_or_anisotropy": True,
            "sample_response_depends_on_geometry_distance_and_material_history": True,
        },
        "uncertainty": {
            "no_per_shot_heat_flux_uncertainty": True,
            "no_erosion_measurement_uncertainty": True,
            "no_sample_distance_table_in_extract": True,
            "no_shot_to_shot_reproducibility_metrics": True,
        },
        "missing_for_material_response_validation": [
            "digitized_incident_flux_history",
            "particle_energy_spectrum",
            "sample_distance_by_condition",
            "surface_temperature_history",
            "erosion_uncertainty",
            "shot_to_shot_repeatability",
        ],
        "missing_for_full_dpf_validation": [
            "current_voltage_waveforms",
            "phase_timing",
            "same_shot_spatial_profiles",
            "neutron_observables",
            "uncertainty_budget",
        ],
        "validation_note": (
            "This source is useful for bounding DPF-driven material damage and "
            "radiation-load applications. It does not validate the DPF plasma "
            "state, circuit coupling, phase dynamics, or neutron production."
        ),
    }


def altarabulsi_deuteron_beam_fluence_targets() -> dict[str, object]:
    """Return KR-backed Lee-model deuteron beam fluence targets."""
    source = (
        "KnowledgeReference/original-deuteron-beam-fluence-emitted-from-"
        "dense-plasma-focus.md"
    )
    return {
        "target_id": "altarabulsi_deuteron_beam_fluence_2024",
        "device": "PF-1000 / MPEF-12 kJ / PF-2.7 kJ",
        "model_role": "kr_ion_beam_fluence_validation_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_scope_and_orders": "30-42",
            "ion_beam_context_and_diagnostics": "52-132",
            "lee_fluence_equations": "181-288",
            "device_parameters_table": "300-389,1461-1480",
            "current_waveform_fitting": "390-489",
            "computed_and_measured_fluence_discussion": "523-628",
            "table3_fluence_comparisons": "701-780,1515-1545",
            "distance_and_application_scaling": "631-688,880-907",
            "summary_pf1000_value": "909-943",
        },
        "shot_context": {
            "model_code": "RADPFV6.16FIB",
            "working_gas": "deuterium",
            "fitted_devices": ["PF-1000", "MPEF-12 kJ", "PF-2.7 kJ"],
            "comparison_metric": "deuteron beam fluence",
            "fluence_pressure_range_Torr_general": [1.0, 15.0],
            "storage_energy_range_kJ_general": [0.2, 863.0],
            "pinch_exit_fluence_order_ions_m2": 1.0e20,
            "distance_14cm_fluence_order_ions_m2": 1.0e19,
            "material_processing_application_scope": True,
        },
        "device_parameter_targets": {
            "PF-1000": {
                "stored_energy_kJ": 863.1,
                "L0_nH": 33.5,
                "C0_uF": 1332.0,
                "r0_mohm": 6.3,
                "cathode_radius_cm": 16.0,
                "anode_radius_cm": 11.5,
                "anode_effective_length_cm": 60.0,
                "fm": 0.142,
                "fc": 0.7,
                "fmr": 0.2,
                "fcr": 0.6,
                "V0_kV": 36.0,
                "fit_pressure_Torr": 3.5,
            },
            "MPEF-12 kJ": {
                "stored_energy_kJ": 9.7,
                "L0_nH": 65.0,
                "C0_uF": 40.0,
                "r0_mohm": 1.0,
                "cathode_radius_cm": 5.5,
                "anode_radius_cm": 3.0,
                "anode_effective_length_cm": 11.5,
                "fm": 0.09,
                "fc": 0.7,
                "fmr": 0.1,
                "fcr": 0.8,
                "V0_kV": 22.0,
                "fit_pressure_Torr": 3.0,
            },
            "PF-2.7 kJ": {
                "stored_energy_kJ": 2.7,
                "L0_nH": 110.0,
                "C0_uF": 30.0,
                "r0_mohm": 22.0,
                "cathode_radius_cm": 3.2,
                "anode_radius_cm": 0.95,
                "anode_effective_length_cm": 22.0,
                "fm": 0.8,
                "fc": 0.8,
                "fmr": 0.5,
                "fcr": 0.8,
                "V0_kV": 13.5,
                "fit_pressure_Torr": 0.15,
            },
        },
        "current_waveform_targets": {
            "measured_current_waveforms_fit_before_fluence_comparison": True,
            "fitting_done_to_end_of_pinch_phase": True,
            "mpef_12kj_fit_pressure_Torr": 3.0,
            "mpef_12kj_end_of_pinch_time_us_approx": 2.08,
            "mass_and_current_factors_adjusted": True,
            "L0_and_r0_may_be_adjusted": True,
            "current_trace_is_gross_performance_indicator": True,
            "digitized_waveform_points_available_in_target": False,
        },
        "phase_timing": {
            "mpef_12kj_end_of_pinch_time_us_approx": 2.08,
            "fit_scope_up_to_end_of_pinch": True,
            "post_pinch_divergence_not_considered": True,
            "post_radial_phase_lacks_significant_ion_beam_acceleration": True,
        },
        "spatial_density_targets": {
            "pinch_volume_used_for_beam_density": True,
            "pinch_radius_and_length_from_lee_model": True,
            "beam_fluence_distance_model_conic_spreading": True,
            "background_gas_energy_loss_neglected_for_distance_estimates": True,
        },
        "temperature_targets": {
            "ion_energy_context_keV_range": [100.0, 1000.0],
            "ion_pulse_duration_context_ns_order": 10.0,
            "temperature_not_directly_validated_in_target": True,
        },
        "ion_beam_formula_targets": {
            "beam_flux_equals_density_times_velocity": True,
            "fluence_equals_flux_times_pinch_lifetime": True,
            "beam_kinetic_energy_fraction_of_pinch_inductive_energy_fe": 0.14,
            "beam_energy_fraction_of_stored_energy_range": [0.03, 0.06],
            "plasma_diode_voltage_equals_three_Vmax": True,
        },
        "fluence_comparison_targets": {
            "PF-1000": {
                "distance_cm": 14.0,
                "fit_pressure_Torr": 3.5,
                "rows": [
                    {"pressure_Torr": 0.5, "sim_ions_m2": 7.3e19, "exp_ions_m2": 7.5e19},
                ],
                "only_one_measured_pressure_available": True,
            },
            "MPEF-12 kJ": {
                "distance_cm": 14.0,
                "fit_pressure_Torr": 3.0,
                "rows": [
                    {"pressure_Torr": 0.76, "sim_ions_m2": 5.5e18, "exp_ions_m2": 5.57e18, "exp_sigma_ions_m2": 0.84e18},
                    {"pressure_Torr": 1.49, "sim_ions_m2": 5.9e18, "exp_ions_m2": 5.79e18, "exp_sigma_ions_m2": 0.81e18},
                    {"pressure_Torr": 2.24, "sim_ions_m2": 6.5e18, "exp_ions_m2": 6.53e18, "exp_sigma_ions_m2": 0.78e18},
                    {"pressure_Torr": 3.0, "sim_ions_m2": 7.5e18, "exp_ions_m2": 7.05e18, "exp_sigma_ions_m2": 0.70e18},
                    {"pressure_Torr": 4.5, "sim_ions_m2": 6.8e18, "exp_ions_m2": 6.68e18, "exp_sigma_ions_m2": 0.82e18},
                    {"pressure_Torr": 6.0, "sim_ions_m2": 5.6e18, "exp_ions_m2": 5.75e18, "exp_sigma_ions_m2": 0.81e18},
                    {"pressure_Torr": 7.5, "sim_ions_m2": 6.0e18, "exp_ions_m2": 6.15e18, "exp_sigma_ions_m2": 0.94e18},
                ],
            },
            "PF-2.7 kJ": {
                "distance_cm": 40.0,
                "fit_pressure_Torr": 0.15,
                "rows": [
                    {"pressure_Torr": 0.075, "sim_ions_m2": 3.99e15, "exp_ions_m2": 3.89e15, "exp_sigma_ions_m2": 0.48e15},
                    {"pressure_Torr": 0.15, "sim_ions_m2": 4.94e15, "exp_ions_m2": 4.95e15, "exp_sigma_ions_m2": 0.25e15},
                    {"pressure_Torr": 0.225, "sim_ions_m2": 4.14e15, "exp_ions_m2": 4.06e15, "exp_sigma_ions_m2": 0.21e15},
                    {"pressure_Torr": 0.375, "sim_ions_m2": 3.68e15, "exp_ions_m2": 3.77e15, "exp_sigma_ions_m2": 0.16e15},
                    {"pressure_Torr": 0.6, "sim_ions_m2": 1.77e15, "exp_ions_m2": 1.86e15, "exp_sigma_ions_m2": 0.11e15},
                ],
            },
            "agreement_within_reported_errors_claimed": True,
        },
        "distance_scaling_targets": {
            "PF-24_pressure_Torr": 11.0,
            "PF-24_pinch_exit_fluence_ions_m2": 3.87e20,
            "fluence_decreases_with_distance": True,
            "PF-24_flux_pinch_exit_ions_m2_s": 8.7e27,
            "PF-24_flux_26cm_ions_m2_s": 2.61e26,
            "PF-24_energy_flux_pinch_exit_W_m2": 1.37e14,
            "PF-24_energy_flux_26cm_W_m2": 4.09e12,
            "PF-24_damage_factor_pinch_exit_W_m2_s05": 2.88e10,
            "PF-24_damage_factor_26cm_W_m2_s05": 8.63e8,
        },
        "model_scope_limits": {
            "ion_beam_fluence_target_not_neutron_validation": True,
            "computed_values_depend_on_current_waveform_fit": True,
            "pinch_exit_values_are_model_outputs": True,
            "distance_model_assumes_conic_spreading": True,
            "background_gas_energy_loss_neglected": True,
            "PF1000_has_only_one_measured_fluence_point": True,
        },
        "uncertainty": {
            "table3_MPEF12_and_PF27_include_exp_error_bars": True,
            "PF1000_measured_value_has_approximate_marker_no_exp_sigma": True,
            "no_raw_detector_response_in_target": True,
            "no_digitized_current_waveform_points": True,
            "no_shot_to_shot_distribution_beyond_table_errors": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_waveforms",
            "digitized_voltage_waveforms",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "full_phase_endpoint_times_for_all_devices",
            "phase_endpoint_uncertainty",
            "same_shot_phase_timing_for_each_fluence_measurement",
        ],
        "missing_for_full_tier4": [
            "same_shot_pinch_radius",
            "same_shot_pinch_length",
            "same_shot_density_profile",
            "same_shot_temperature_profile",
            "beam_divergence_measurement",
        ],
        "missing_for_full_tier5": [
            "raw_fluence_detector_response",
            "raw_detector_calibration",
            "same_shot_neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This Altarabulsi 2024 source is a useful KR target for Lee-code "
            "deuteron beam fluence after current-waveform fitting, including "
            "reported measured-vs-computed fluence with error bars for MPEF-12 "
            "and PF-2.7 kJ. It remains an ion-beam/material-processing target, "
            "not neutron validation, and lacks raw current/voltage traces, "
            "detector response, same-shot spatial diagnostics, and complete "
            "uncertainty propagation."
        ),
    }


def kiai_double_dpf_icf_concept_targets() -> dict[str, object]:
    """Return KR-backed double-DPF/ICF concept and roadmap targets."""
    source = (
        "KnowledgeReference/2025-double-3mj-dense-plasma-focus-"
        "thermonuclear-icf.md"
    )
    return {
        "target_id": "kiai_double_dpf_icf_concept_2025",
        "device": "conceptual double 3 MJ DPF",
        "model_role": "kr_concept_roadmap_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_concept_and_validation_status": "16-40",
            "conceptual_configuration": "83-106",
            "full_scale_parameter_table": "392-453",
            "energy_coupling_and_engineering_context": "464-530",
            "simplified_hts_power_projection": "942-996",
            "pellet_replacement_cycle": "997-1013",
            "compressed_pellet_and_hts_theory": "1200-1239,1275-1321,1349-1363,1373-1419",
            "experimental_validation_roadmap": "1478-1514",
            "prototype_parameter_table": "1519-1584",
        },
        "shot_context": {
            "configuration": "two coaxial DPF devices driving a DT pellet",
            "concept_is_theoretical_not_built_device": True,
            "fuel_pellet": "deuterium-tritium",
            "full_scale_total_bank_energy_MJ": 6.0,
            "full_scale_bank_energy_MJ_each": 3.0,
            "hts_magnetic_lenses": True,
            "proposed_laboratory_prototype_energy_kJ": 30.0,
            "future_laboratory_validation_required": True,
        },
        "conceptual_full_scale_parameters": {
            "fill_gas": "deuterium",
            "pressure_torr": 10.0,
            "impedance_mohm": 12.5,
            "peak_circuit_current_MA": 20.0,
            "maximum_charging_voltage_kV": 200.0,
            "capacitor_bank_uF": 150.0,
            "stored_bank_energy_MJ_total": 6.0,
            "stored_bank_energy_MJ_each": 3.0,
            "inductance_nH": 35.0,
            "circuit_period_us": 17.5,
            "anode_radius_cm": 15.0,
            "anode_length_cm": 80.0,
            "cathode_radius_cm": 22.5,
            "axial_speed_cm_per_us": 29.5,
            "radial_speed_cm_per_us": 42.4,
            "pinch_radius_cm": 1.8,
            "pinch_lifetime_ns_each_dpf": 300.0,
            "pinch_length_cm": 12.0,
            "cathode_to_anode_radius_ratio": 1.5,
            "current_loss_factor": 0.7,
            "mass_sweep_factor_fm": 0.13,
            "induced_voltage_MV": 20.0,
        },
        "prototype_30kj_parameters": {
            "anode_length_cm": 15.0,
            "anode_radius_cm": 2.5,
            "operating_voltage_kV_range": [50.0, 60.0],
            "capacitor_bank_uF": 500.0,
            "stored_energy_kJ": 30.0,
            "plasma_density_ions_m3": 6.0e25,
            "fusion_neutron_yield_per_shot": 1.0e10,
            "pinch_efficiency_fraction_range": [0.20, 0.30],
            "peak_current_MA_at_50kV": 3.54,
            "peak_current_MA_at_60kV": 4.24,
            "maximum_pinch_current_MA_range": [0.71, 1.06],
            "pinch_radius_mm": 3.0,
            "pinch_length_cm": 2.0,
            "pinch_lifetime_ns": 50.0,
            "deuteron_number_density_m3": 6.0e25,
            "thermonuclear_neutron_yield_per_shot": 1.0e10,
        },
        "current_waveform_targets": {
            "full_scale_peak_circuit_current_MA": 20.0,
            "full_scale_current_loss_factor": 0.7,
            "full_scale_mass_sweep_factor_fm": 0.13,
            "prototype_peak_current_MA_range": [3.54, 4.24],
            "prototype_maximum_pinch_current_MA_range": [0.71, 1.06],
            "digitized_current_trace_available": False,
            "circuit_waveform_is_parameter_projection": True,
        },
        "phase_timing": {
            "full_scale_circuit_period_us": 17.5,
            "full_scale_pinch_lifetime_ns_each_dpf": 300.0,
            "prototype_pinch_lifetime_ns": 50.0,
            "capacitor_charge_time_s_range": [300.0, 600.0],
            "shot_rate_per_hour_range": [6.0, 12.0],
            "ideal_pellet_replacement_time_s_max": 10.0,
            "example_pellet_motion_speed_m_per_s": 0.1,
            "synchronized_dual_dpf_timing_not_validated": True,
        },
        "spatial_density_targets": {
            "prototype_plasma_density_ions_m3": 6.0e25,
            "prototype_deuteron_density_m3": 6.0e25,
            "compressed_pellet_density_m3": 4.5e32,
            "bohm_scaling_density_m3": 1.0e22,
            "densities_are_model_or_design_values": True,
        },
        "temperature_targets": {
            "pellet_ignition_temperature_keV_range": [10.0, 20.0],
            "bohm_scaling_temperature_keV": 15.0,
            "alpha_mfp_temperature_cases_keV": [1.0, 5.0, 10.0],
            "temperatures_are_model_or_design_values": True,
        },
        "magnetic_field_targets": {
            "hts_field_T_range": [10.0, 15.0],
            "compressed_pellet_case_field_T": 15.0,
            "bohm_scaling_field_T": 5.0,
            "rebco_temperature_K_options": [77.0, 20.0],
            "cryogenic_heat_load_W": 10.0,
            "cryocooler_electrical_W_per_cooling_W_at_4K": 300.0,
            "hts_field_map_not_validated": True,
        },
        "neutron_yield_targets": {
            "prototype_projected_neutrons_per_shot": 1.0e10,
            "prototype_projected_thermonuclear_neutrons_per_shot": 1.0e10,
            "yield_values_are_projection_not_measurement": True,
        },
        "power_projection_targets": {
            "driver_efficiency": 0.3,
            "thermal_to_electric_efficiency": 0.4,
            "input_power_MW": 3.0,
            "hts_projection_by_field_T": {
                "10": {
                    "gain": 1.5e4,
                    "fusion_power_MW": 13.5,
                    "electric_power_MW": 5.4,
                },
                "12": {
                    "gain": 2.0e4,
                    "fusion_power_MW": 18.0,
                    "electric_power_MW": 7.2,
                },
                "15": {
                    "gain": 2.8e4,
                    "fusion_power_MW": 25.2,
                    "electric_power_MW": 10.08,
                },
            },
            "without_hts_fusion_power_MW": 25.0,
            "without_hts_electric_power_MW": 10.0,
            "with_hts_fusion_power_MW": 75.0,
            "with_hts_electric_power_MW": 30.0,
            "extreme_pellet_projection_fusion_power_PW": 3.61,
            "extreme_pellet_projection_electric_power_TW": 613.0,
            "all_power_values_are_simplified_model_outputs": True,
        },
        "experimental_roadmap_requirements": {
            "stage_1_single_30kj_prototype": [
                "hts_enhanced_plasma_confinement",
                "energy_transfer_efficiency",
                "neutron_yield",
                "plasma_stability",
            ],
            "stage_2_double_30kj_dpf": [
                "synchronized_operation",
                "dt_pellet_compression_and_acceleration",
                "dual_beam_energy_transfer",
                "fusion_yield_vs_single_dpf",
            ],
            "stage_3_full_scale_fusion_testing": [
                "ignition_temperature_10_to_20_keV",
                "density_diagnostics",
                "energy_retention",
                "fusion_output",
            ],
            "required_diagnostics": [
                "plasma_diagnostics",
                "neutron_yield_measurements",
                "high_speed_imaging",
            ],
        },
        "model_scope_limits": {
            "theoretical_proposal_not_validated_experiment": True,
            "simplified_power_assumptions": True,
            "full_mhd_simulation_and_precise_efficiency_parameters_required": True,
            "no_measured_dt_pellet_coupling_dataset": True,
            "no_measured_full_scale_double_dpf_neutron_dataset": True,
        },
        "uncertainty": {
            "no_digitized_current_voltage_waveforms": True,
            "no_phase_timing_uncertainties": True,
            "no_profile_uncertainties": True,
            "no_neutron_detector_response": True,
            "scale_up_from_30kj_to_6mj_unvalidated": True,
            "dt_pellet_ignition_claim_requires_experimental_validation": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_voltage_trace",
            "measured_circuit_waveform_for_30kj_prototype",
            "measured_circuit_waveform_for_6mj_double_dpf",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "measured_single_30kj_phase_times",
            "measured_double_30kj_synchronization_timing",
            "measured_full_scale_dual_dpf_phase_times",
            "pinch_duration_uncertainty",
        ],
        "missing_for_full_tier4": [
            "measured_density_profile",
            "measured_temperature_profile",
            "measured_hts_field_map",
            "same_shot_pellet_coupling_diagnostics",
            "profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "measured_neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "dt_pellet_experiment",
            "energy_accounting_with_uncertainty",
            "validated_30kj_to_6mj_scaling",
        ],
        "validation_note": (
            "This Kiai 2025 Scientific Reports source is useful as a "
            "high-level double-DPF/ICF concept and validation roadmap. Its "
            "6 MJ, HTS, pellet, neutron-yield, and power numbers are design "
            "or simplified-model projections; they must not be treated as "
            "validated end-to-end predictive simulation targets until the "
            "three-stage experimental program supplies measured waveforms, "
            "profiles, neutron diagnostics, detector response, and uncertainty."
        ),
    }


def wang_metallic_vapor_interferometry_targets() -> dict[str, object]:
    """Return KR-backed DPF-16 metallic-vapor interferometry targets."""
    source = (
        "KnowledgeReference/observation-of-the-metallic-vapor-from-a-plasma-"
        "focus-wang-xinxin-3-yang-jinji-department-of.md"
    )
    return {
        "target_id": "wang_metallic_vapor_interferometry_1999",
        "device": "DPF-16",
        "model_role": "kr_interferometry_material_vapor_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_and_claim": "15-21",
            "dpf_context": "27-71",
            "device_and_gas_setup": "92-121",
            "interferogram_timing_and_phase_interpretation": "123-178",
            "metallic_vapor_evidence_and_delay": "178-214",
            "higher_pressure_interferograms": "219-228",
            "conclusion": "230-234,266-271",
        },
        "shot_context": {
            "device": "DPF-16",
            "stored_energy_kJ": 16.0,
            "charging_voltage_kV": 20.0,
            "peak_current_kA": 380.0,
            "configuration": "Mather type",
            "working_gas": "hydrogen",
            "working_pressure_Pa_range": [70.0, 650.0],
            "typical_interferogram_pressure_Pa": 200.0,
            "higher_pressure_interferogram_Pa": 330.0,
        },
        "geometry": {
            "anode_material": "oxygen-free copper",
            "anode_diameter_mm": 66.0,
            "anode_length_mm": 265.0,
            "cathode_length_mm": 265.0,
            "target_material": "tungsten",
            "target_diameter_mm": 10.0,
            "target_height_mm": 6.0,
            "field_of_view_diameter_mm": 60.0,
        },
        "phase_timing": {
            "reference_time_definition": "pinch spike in dI/dt waveform",
            "t0_corresponds_to_maximum_compression_above_anode": True,
            "compression_times_ns": [-200.0, -140.0, -60.0],
            "minimal_radius_time_ns": 0.0,
            "expansion_begins_time_ns": 40.0,
            "post_focus_expansion_time_ns": 200.0,
            "metallic_vapor_visible_time_ns": 280.0,
            "metallic_vapor_delay_from_pinch_ns_about": 280.0,
            "higher_pressure_vapor_times_ns": [220.0, 300.0],
        },
        "spatial_density_targets": {
            "laser_differential_interferometer_records_ps_evolution": True,
            "high_density_volume_emerges_from_anode_target": True,
            "high_density_volume_not_plasma_sheath": True,
            "metallic_vapor_visibility_is_qualitative_interferogram_target": True,
            "no_electron_density_inversion_in_extract": True,
        },
        "temperature_targets": {
            "generic_dpf_temperature_K_context": 1.0e7,
            "source_context_not_same_shot_temperature_diagnostic": True,
        },
        "xray_material_process_targets": {
            "metallic_vapor_observed_first_time_claim": True,
            "vapor_from_target_surface_by_intense_electron_beam": True,
            "target_erosion_after_many_shots_confirms_evaporation": True,
            "high_density_volume_absent_with_hollow_anode": True,
            "hard_xray_emission_after_focus_over_context": "several_hundred_ns",
            "observation_supports_xray_emission_physical_process": True,
        },
        "model_scope_limits": {
            "material_vapor_target_not_neutron_validation": True,
            "interferograms_are_visual_qualitative_evidence": True,
            "no_line_integrated_density_values_extracted": True,
            "no_xray_spectrum_or_absolute_xray_yield": True,
            "no_beam_current_or_energy_measurement": True,
        },
        "uncertainty": {
            "no_digitized_interferogram_phase_shift": True,
            "no_density_uncertainty": True,
            "no_timing_uncertainty": True,
            "no_current_waveform_uncertainty": True,
            "no_xray_detector_response": True,
        },
        "partial_target_groups": [
            "phase_timing",
            "spatial_density",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_dIdt_trace",
            "current_trace",
            "voltage_trace",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "absolute_current_trace_timebase",
            "pinch_spike_uncertainty",
            "interferogram_gate_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "line_integrated_density_map",
            "electron_density_inversion",
            "temperature_diagnostic",
            "vapor_species_spectroscopy",
            "same_shot_profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "xray_time_history",
            "xray_spectrum",
            "electron_beam_energy",
            "electron_beam_current",
            "neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This Wang/Yang source is useful for the timing and qualitative "
            "spatial appearance of metallic vapor above a DPF anode target. "
            "It constrains anode-material vapor and X-ray-process hypotheses, "
            "but it is not a quantitative plasma, X-ray, or neutron validation "
            "packet because the local extract lacks digitized interferometry, "
            "density inversion, spectra, beam diagnostics, detector response, "
            "and uncertainty."
        ),
    }


def esaulov_2d_mhrdr_dpf_targets() -> dict[str, object]:
    """Return KR-backed Esaulov 2D MHRDR DPF modeling targets."""
    source = "KnowledgeReference/esaulov_2003_2d_mhd_dpf.md"
    return {
        "target_id": "esaulov_2d_mhrdr_dpf_2003",
        "device": "LANL Begay DPF",
        "model_role": "kr_2d_multitemperature_mhd_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract": "26-49",
            "mhd_model_and_phases": "79-121",
            "begay_device_parameters": "200-257",
            "current_sheath_formation": "272-314",
            "current_sheath_acceleration": "334-459",
            "z_pinch_and_neutron_rate": "461-529",
            "axis_history_and_figure_scales": "623-660,664-724",
            "neutron_rate_mechanism_and_conclusion": "727-791",
            "radial_slice_scales": "830-879",
        },
        "phase_semantics": {
            "breakdown_and_sheath_formation": (
                "initially ionized region forms along the insulator and bridges "
                "the electrodes"
            ),
            "magnetic_flux_penetration": (
                "magnetic flux penetrates the weakly ionized region, heating "
                "and ionizing plasma until sheath detachment"
            ),
            "acceleration": (
                "Lorentz force and magnetic pressure accelerate a complex "
                "snowplow current sheath"
            ),
            "collapse": (
                "2D current-sheath collapse starts near the inner electrode "
                "surface and propagates upward along z"
            ),
        },
        "shot_context": {
            "device": "LANL Begay plasma focus",
            "configuration": "Mather type",
            "inner_electrode_radius_cm": 1.18,
            "outer_electrode_radius_cm": 3.65,
            "inner_electrode_length_cm": 15.7,
            "fill_gas": "deuterium",
            "fill_pressure_torr": 1.0,
            "capacitance_uF": 36.4,
            "charging_voltage_kV": 14.0,
            "series_inductance_nH": 178.0,
        },
        "current_waveform_targets": {
            "self_consistent_circuit_mhd_solution": True,
            "current_sheath_current_kA_range_during_acceleration": [50.0, 100.0],
            "electrode_voltage_drop_kV_range_during_acceleration": [1.0, 2.0],
            "figure_1_current_axis_kA_max": 160.0,
            "digitized_current_trace_available_in_target": False,
        },
        "phase_timing": {
            "formation_time_us_examples": [0.9, 2.0],
            "acceleration_slice_times_us": [1.0, 2.0],
            "collapse_pressure_contour_times_us": [2.6, 2.65],
            "local_neutron_rate_peak_times_us": [2.74, 2.92],
            "radial_slice_times_us": [2.72, 2.90],
            "generic_focus_duration_ns_range": [100.0, 150.0],
        },
        "spatial_density_targets": {
            "abstract_density_cm3_min": 1.0e19,
            "axis_history_density_ug_per_cm3_scale_max": 3.0,
            "electron_density_axis_history_cm3_scale_max": 1.0e18,
            "moving_sheath_density_ug_per_cm3_scale_max": 0.5,
            "pinch_region_z_cm": 18.1,
            "pinch_region_height_above_anode_cm": 2.5,
        },
        "temperature_targets": {
            "abstract_temperature_keV_order": 1.0,
            "axis_history_temperature_keV_scale_max": 5.0,
            "moving_sheath_temperature_eV_scale_max": 20.0,
            "electron_temperature_lower_than_ion_temperature": True,
            "shock_compression_dominates_acceleration_temperature": True,
        },
        "field_context": {
            "code": "MHRDR",
            "model": "2D multi-temperature MHD",
            "ion_electron_radiation_temperatures": True,
            "electron_ion_thermal_conduction": True,
            "resistive_diffusion": True,
            "radiation_diffusion": True,
            "lorentz_force_and_shock_hydrodynamics": True,
            "external_circuit_self_consistent": True,
            "magnetic_piston_snowplow_structure": True,
            "moving_sheath_B_T_scale_max": 2.0,
        },
        "neutron_yield_targets": {
            "dt_high_yield_context_neutrons_per_pulse_range": [1.0e11, 1.0e12],
            "dd_yield_computed_with_maxwell_average_cross_sections": True,
            "local_neutron_rate_peaks_causes": [
                "high_density",
                "high_ion_temperature_after_hydromagnetic_shock_collapse",
            ],
            "space_integrated_neutron_rate_s_inverse_scale": 1.0e14,
        },
        "mhd_scope_limits": {
            "high_pressure_thermal_mechanism_assumption": True,
            "beam_target_mechanisms_not_primary_in_this_target": True,
            "radiation_energy_negligible_for_begay_case": True,
            "radiation_can_matter_at_22_torr_3MA": True,
            "two_dimensional_effects_govern_collapse": True,
            "qualitative_and_quantitative_agreement_claimed_without_error_bars": True,
        },
        "uncertainty": {
            "source_has_no_digitized_waveforms": True,
            "source_has_no_error_bars": True,
            "figure_scales_are_used_as_context_only": True,
            "agreement_claim_not_same_shot_validation_packet": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_voltage_trace",
            "per_point_trace_uncertainty",
        ],
        "missing_for_full_tier2": [
            "experimental_phase_endpoint_times",
            "phase_timing_uncertainty",
            "breakdown_to_liftoff_observation",
        ],
        "missing_for_full_tier4": [
            "same-shot_density_profile",
            "same-shot_temperature_profile",
            "same-shot_magnetic_field_profile",
            "profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "absolute_neutron_yield",
            "neutron_time_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "detector_response_model",
            "beam_target_or_kinetic_neutron_component",
        ],
        "validation_note": (
            "This Esaulov source is useful for Begay-device parameters and "
            "2D multi-temperature MHD phase/neutron-rate context. It is not a "
            "complete validation packet because the KR extract lacks digitized "
            "experimental traces, uncertainty, detector response, and "
            "mechanism-separated neutron validation."
        ),
    }


def ou_foi_2d_dpf_simulation_targets() -> dict[str, object]:
    """Return KR-backed FOI 2D DPF MHD simulation targets."""
    source = "KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md"
    return {
        "target_id": "ou_foi_2d_dpf_simulation_2024",
        "device": "LLNL DPF reference / FOI parametric DPF",
        "model_role": "kr_2d_mhd_parameter_sweep_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "model_context_and_foi_scope": "14-34",
            "mhd_equations_and_boundary_current": "35-100",
            "llnl_reference_and_sheath_interpretation": "101-113,134-237",
            "comparison_with_llnl_images": "247-252,288-301",
            "parameter_definitions_and_pressure_sweep": "254-282,303-326",
            "current_sweep": "283-287,330-389",
            "anode_radius_and_gap_sweeps": "335-349,391-471",
            "analytic_velocity_check_and_conclusion": "482-521",
        },
        "model_context": {
            "code": "FOI",
            "electron_inertia_ignored": True,
            "simplified_ohm_law_closes_maxwell": True,
            "electromagnetic_solver": "TVD-CP",
            "fluid_solver": "RTVD",
            "gas_model": "adiabatic single-phase ideal gas",
            "swept_region_treated_as_vacuum_high_resistivity": True,
            "other_region_low_resistivity": True,
            "electrodes_fixed_velocity_zero": True,
            "courant_number": 0.5,
            "self_emission_high_density_region_proxy_for_sheath": True,
        },
        "current_waveform_targets": {
            "boundary_Bphi_formula": "mu * I / (2*pi*r)",
            "current_formula": "Imax * sin(2*pi*f*t)",
            "sine_current_used_as_discharge_approximation": True,
            "quarter_period_ns": 135.0,
            "not_a_measured_current_trace": True,
        },
        "llnl_reference_case": {
            "anode_diameter_cm": 15.2,
            "cathode_anode_gap_cm": 4.3,
            "peak_current_MA": 2.5,
            "fill_pressure_Pa": 2926.0,
            "morphology_agrees_with_optical_framing_images": True,
            "timing_difference_large": True,
        },
        "phase_timing": {
            "llnl_run_down_time_us": 3.9,
            "llnl_run_in_time_us": 6.2,
            "llnl_pinch_time_us": 7.4,
            "llnl_breakup_time_us": 7.4,
            "column_cutoff_time_ns_less_than": 100.0,
            "current_amplitude_MA_cases": [1.5, 2.0, 2.5, 3.0, 3.5],
            "pinch_time_ns_by_current_MA": {
                "1.5": 188.99,
                "2.0": 155.08,
                "2.5": 135.65,
                "3.0": 123.40,
                "3.5": 114.29,
            },
            "pinch_current_MA_by_current_amplitude_MA": {
                "1.5": 1.213,
                "2.0": 1.946,
                "2.5": 2.500,
                "3.0": 2.973,
                "3.5": 3.399,
            },
            "extend_anode_to_match_pinch_with_current_peak": True,
        },
        "spatial_density_targets": {
            "compression_ratio_definition": "rho_max / rho0",
            "higher_compression_ratio_means_smaller_column_radius": True,
            "higher_compression_ratio_means_higher_column_temperature_and_density": True,
            "pressure_sweep_Pa": [133.0, 665.0, 1330.0, 1995.0, 2660.0],
            "sheath_velocity_above_m_per_s": 1.0e5,
            "low_pressure_pinch_before_current_peak_suboptimal_yield": True,
            "high_pressure_pinch_after_current_peak_suboptimal_yield": True,
        },
        "temperature_targets": {
            "temperature_inferred_from_compression_quality_not_diagnostic": True,
            "neutral_gas_heating_and_ionization_in_high_density_region": True,
            "line_emission_from_high_density_region_context": True,
        },
        "magnetic_field_targets": {
            "azimuthal_magnetic_field_between_electrodes": True,
            "near_anode_B_stronger_due_to_inverse_radius": True,
            "sheath_velocity_proportional_to_current": True,
            "sheath_velocity_inverse_to_sqrt_pressure": True,
            "sheath_velocity_inverse_to_anode_radius": True,
            "anode_radius_mm_cases": [30.0, 35.0, 40.0, 45.0, 50.0],
            "cathode_anode_gap_mm_cases": [15.0, 20.0, 25.0, 30.0, 35.0],
            "gap_little_effect_on_near_anode_axial_motion": True,
        },
        "neutron_context": {
            "pressure_timing_controls_neutron_yield_qualitatively": True,
            "current_increase_improves_compression_but_not_alone_optimal": True,
            "no_neutron_yield_values_extracted": True,
            "no_neutron_detector_model": True,
        },
        "model_scope_limits": {
            "simulation_vs_llnl_morphology_agrees_but_timing_differs": True,
            "ideal_gas_single_phase_model": True,
            "no_measured_waveform_fit": True,
            "no_quantitative_error_bars": True,
            "parametric_design_guidance_not_validation_packet": True,
        },
        "uncertainty": {
            "no_timing_uncertainty": True,
            "no_density_uncertainty": True,
            "no_temperature_uncertainty": True,
            "no_neutron_yield_uncertainty": True,
            "figure_values_partly_read_from_local_text": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "measured_current_trace",
            "measured_voltage_trace",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "quantitative_llnl_frame_time_alignment",
            "phase_endpoint_uncertainty",
            "breakdown_and_liftoff_time",
        ],
        "missing_for_full_tier4": [
            "measured_density_profile",
            "measured_temperature_profile",
            "measured_magnetic_field_profile",
            "same_shot_profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "measured_neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This FOI source provides useful 2D MHD design trends for pressure, "
            "current amplitude, anode radius, and cathode-anode gap, plus an "
            "LLNL morphology comparison. It is not an end-to-end validation "
            "packet because the timing mismatch is large and the extract lacks "
            "measured waveforms, uncertainty, spatial diagnostics, and neutron "
            "outputs."
        ),
    }


def sun_two_temperature_mhd_motion_targets() -> dict[str, object]:
    """Return KR-backed two-temperature MHD DPF motion-process targets."""
    source = (
        "KnowledgeReference/2025-theoretical-and-numerical-studies-on-motion-"
        "process-of-dense-plasma-focus.md"
    )
    return {
        "target_id": "sun_two_temperature_mhd_motion_2025",
        "device": "UNU / UDMPF1 / PF-1000 parameter study",
        "model_role": "kr_two_temperature_mhd_motion_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_scope": "28-34,1159-1194",
            "dpf_processes_and_model_limits": "83-172",
            "external_circuit": "423-481",
            "unu_geometry_and_boundaries": "484-548",
            "model_validation": "552-623",
            "unu_motion_process": "626-755",
            "parameter_laws": "755-832,914-996",
            "conclusion": "1000-1017",
        },
        "model_context": {
            "model": "two-temperature nonideal MHD coupled to external RLC circuit",
            "electron_ion_thermal_nonequilibrium": True,
            "braginskii_transport_coefficients": True,
            "resistive_effects": True,
            "external_circuit_coupled": True,
            "lee_model_sem_empirical_requires_current_fit": True,
            "mhd_useful_for_macroscopic_sheath_current_instability_and_parameter_effects": True,
            "mhd_cannot_self_consistently_resolve_high_energy_beams_or_neutron_production": True,
        },
        "shot_context": {
            "reference_device": "UNU",
            "comparison_devices": ["UNU", "UDMPF1"],
            "generic_current_range": "hundreds_kA_to_MA",
            "generic_pulse_width": "microseconds",
            "generic_temperature_K_order": 1.0e7,
            "generic_speed_m_s_range": [1.0e5, 1.0e6],
            "generic_density_m3_range": [1.0e23, 1.0e26],
            "historic_large_device_dd_yield_per_pulse_range": [1.0e11, 1.0e12],
            "historic_large_device_dt_yield_per_pulse_above": 1.0e13,
            "china_first_dpf_energy_kJ": 40.0,
            "china_first_dpf_average_neutron_yield_per_pulse": 1.0e9,
        },
        "current_waveform_targets": {
            "UNU_V0_kV": 15.0,
            "UNU_C0_uF": 30.0,
            "UNU_L0_nH": 110.0,
            "UNU_r0_mohm": 12.0,
            "UPF_from_magnetic_flux_derivative": True,
            "UNU_current_compared_to_lee_model_and_experiment": True,
            "UNU_voltage_compared_to_lee_model_and_experiment": True,
            "digitized_current_voltage_points_not_available": True,
        },
        "geometry": {
            "UNU_anode_radius_cm": 0.95,
            "UNU_cathode_radius_cm": 3.2,
            "UNU_cathode_anode_gap_cm": 2.25,
            "UNU_anode_length_cm": 16.0,
            "UNU_cathode_length_cm": 25.0,
            "axisymmetric_2d_model": True,
            "solid_electrode_regions_excluded": True,
        },
        "phase_timing": {
            "typical_processes": [
                "gas_breakdown",
                "run_down",
                "run_in",
                "pinch",
            ],
            "UNU_axial_phase_us_range": [0.0, 2.5],
            "UNU_radial_implosion_us_range": [2.78, 2.90],
            "pinch_time_us_approx": 2.8,
            "axial_phase_should_match_quarter_period": True,
            "quarter_period_matching_transfers_more_energy_to_pinch": True,
        },
        "spatial_density_targets": {
            "UNU_background_density_m3": 2.4e23,
            "UNU_background_pressure_Torr": 3.5,
            "UNU_inlet_sheath_density_multiplier": 4.0,
            "radial_implosion_density_m3_approx": 1.0e24,
            "radial_implosion_density_increases_by_order_of_magnitude": True,
            "high_temperature_high_density_focus_forms_on_axis": True,
        },
        "temperature_targets": {
            "UNU_background_temperature_eV": 1.0,
            "UNU_inlet_sheath_temperature_eV": 2.0,
            "axial_phase_ion_temperature_eV_range": [1.0, 100.0],
            "radial_implosion_ion_temperature_keV_approx": 1.0,
            "pinch_temperature_increases_by_two_orders": True,
        },
        "magnetic_field_targets": {
            "B_boundary_formula": "mu0 * I / (2*pi*r)",
            "lorentz_force_accelerates_sheath_axially": True,
            "inner_electrode_sheath_moves_faster_due_to_inverse_radius_B": True,
            "magnetic_field_strength_increases_during_radial_implosion": True,
            "radial_implosion_lorentz_force_grows_rapidly": True,
            "sheath_speed_up_to_km_s": 90.0,
        },
        "parameter_scaling_targets": {
            "large_dpf_current_saturates_when_increasing_C_or_decreasing_L": True,
            "increasing_circuit_voltage_more_effective_for_large_dpf_current": True,
            "anode_to_cathode_radius_ratio_should_be_as_small_as_possible": True,
            "c_ratio_cases_for_pf1000": [1.4, 1.8, 2.2, 2.6],
            "c_from_3_to_1p5_impedance_reduction_factor": 2.7,
            "c_from_2_to_1p3_impedance_reduction_factor": 2.6,
            "lower_c_increases_peak_and_pinch_current": True,
        },
        "validation_comparison": {
            "UNU_current_voltage_match_lee_model_fit_to_experiment": True,
            "UDMPF1_radial_trajectory_matches_experiment": True,
            "no_error_metric_in_extract": True,
            "no_digitized_trace_in_extract": True,
        },
        "neutron_context": {
            "mhd_not_for_quantitative_neutron_yield": True,
            "pinch_stage_determines_instability_and_beam_target_neutrons": True,
            "no_neutron_yield_target_extracted_for_model_validation": True,
        },
        "model_scope_limits": {
            "gas_breakdown_not_detailed_validation": True,
            "fully_ionized_initial_plasma_assumption": True,
            "neutral_gas_requires_multifluid_model_not_used": True,
            "no_self_consistent_high_energy_particle_beams": True,
            "no_self_consistent_neutron_production": True,
            "comparisons_are_qualitative_or_plot_based_without_error_bars": True,
        },
        "uncertainty": {
            "no_digitized_current_voltage_traces": True,
            "no_radial_trajectory_error_bars": True,
            "no_density_profile_uncertainty": True,
            "no_temperature_profile_uncertainty": True,
            "no_neutron_uncertainty": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_voltage_trace",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "digitized_phase_endpoint_times",
            "radial_trajectory_points",
            "phase_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "measured_density_profile",
            "measured_temperature_profile",
            "measured_magnetic_field_profile",
            "same_shot_profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "self_consistent_beam_model",
            "self_consistent_neutron_production",
            "measured_neutron_yield",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This Sun 2025 source is a valuable two-temperature MHD motion "
            "target with external circuit coupling and benchmark comparisons "
            "to UNU/UDMPF1. It strengthens macroscopic DPF dynamics and design "
            "scaling, but explicitly does not close high-energy particle-beam "
            "or neutron-yield prediction."
        ),
    }


def beresnyak_hawk_3d_mhd_targets() -> dict[str, object]:
    """Return KR-backed HAWK 3D MHD model-scope targets."""
    source = "KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md"
    return {
        "target_id": "beresnyak_hawk_3d_mhd_2018",
        "device": "NRL HAWK DPF",
        "model_role": "kr_3d_mhd_model_scope_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_and_scope": "26-42",
            "dpf_mhd_validity_and_hawk_purpose": "91-147",
            "athena_and_circuit_setup": "155-199",
            "geometry_and_injected_plasma": "202-218",
            "pinch_time_drive_parameter_and_resolution": "223-253",
            "mhd_collapse_and_thermal_yield_metric": "255-301",
            "hall_mhd_result": "305-333",
            "particle_acceleration_and_conclusion": "338-389,393-398",
        },
        "shot_context": {
            "facility": "Naval Research Laboratory HAWK",
            "generator_current_kA": 665.0,
            "generator_rise_time_us": 1.2,
            "generator_inductance_nH": 720.0,
            "generator_high_impedance": True,
            "local_plasma_injection_with_plasma_guns": True,
            "evacuated_interelectrode_space": True,
            "not_uniform_neutral_gas_fill": True,
            "plasma_is_assumed_fully_ionized_deuterium": True,
        },
        "circuit_model": {
            "series_inductance_nH": 720.0,
            "series_resistance_ohm": 0.15,
            "capacitance_uF": 1.07,
            "initial_capacitor_voltage_kV": 640.0,
            "initial_current_A": 0.0,
            "simulation_input_current_and_dIdt": True,
            "boundary_Bphi_from_current": True,
            "boundary_velocity_gradient_from_dIdt": True,
            "device_voltage_from_integrated_electric_field": True,
            "circuit_solver_first_order_euler": True,
            "athena_timestep_ns_below": 0.1,
        },
        "current_waveform_targets": {
            "target_density_current_close_to_short_circuit": True,
            "target_density_device_voltage_kV_below": 10.0,
            "short_circuit_sine_period_us": 5.2,
            "pinch_near_current_peak": True,
            "digitized_current_voltage_trace_available": False,
        },
        "phase_timing": {
            "pinch_time_us_at_density_3e16_cm3": 0.95,
            "pinch_formation_figure_time_us_around": 1.0,
            "shock_turns_around_anode_corner_before_axial_pinch": True,
            "axial_instabilities_emerge_after_collapse": True,
        },
        "geometry": {
            "anode_radius_cm": 6.33,
            "anode_length_cm": 4.0,
            "cathode_radius_cm": 8.57,
            "initial_density_high_to_low_ratio": 2.0,
            "background_density_fraction_of_rho0": 0.25,
            "azimuthal_mode_numbers": [0, 3, 6],
            "density_distribution_from_plasma_gun_measurements": True,
        },
        "spatial_density_targets": {
            "characteristic_mass_density_g_per_cc": 1.0e-7,
            "characteristic_number_density_cm3": 3.0e16,
            "thermal_yield_total_max_density_cm3": 9.0e15,
            "equivalent_gas_fill_fraction_parameter": 0.2,
            "collapse_quality_m3_m6_about_same": True,
            "nonaxisymmetric_collapse_still_reaches_axis": True,
        },
        "temperature_targets": {
            "particle_injection_temperature_keV": 1.0,
            "maximum_plasma_temperature_keV_around": 3.0,
            "most_volume_temperature_much_lower": True,
            "temperature_used_with_density_as_collapse_quality_metric": True,
        },
        "field_context": {
            "mhd_code": "Athena",
            "athena_3d_eulerian_unsplit_finite_volume": True,
            "shock_capturing_riemann_solvers": True,
            "resistive_diffusion_for_magnetic_field": True,
            "optional_nonideal_terms": [
                "viscosity",
                "resistivity",
                "Hall",
                "ambipolar_diffusion",
            ],
            "ideal_electric_field": "E = v cross B / c",
            "grid_resolution_examples": {
                "nx": 480,
                "ny": 480,
                "nz": 288,
            },
        },
        "magnetic_field_targets": {
            "hall_term_breaks_mirror_symmetry": True,
            "positive_polarity_center_conductor_anode": True,
            "negative_polarity_center_conductor_cathode": True,
            "hall_positive_polarity_faster_tighter_pinch_near_anode": True,
            "spitzer_resistivity_no_qualitative_dynamic_change": True,
            "qualitative_agreement_current_disruption_near_anode": True,
        },
        "thermal_neutron_metric": {
            "thermal_neutron_yield_used_as_collapse_metric": True,
            "thermal_fusion_thought_subdominant_in_dpf": True,
            "not_projected_hawk_yield": True,
            "thermal_neutron_rate_increases_with_decreasing_density": True,
            "total_thermal_yield_max_density_cm3": 9.0e15,
        },
        "particle_acceleration_targets": {
            "test_particle_code": "Hephaestus",
            "particles_evolved_in_ideal_mhd_fields": True,
            "bohm_scattering_assumed": True,
            "initial_distribution": "isotropic Maxwellian",
            "injection_temperature_keV": 1.0,
            "stochastic_power_law_tail_cutoff_keV_around": 200.0,
            "stochastic_energy_below_generator_voltage_by_factor": 3.0,
            "stochastic_distribution_largely_isotropic": True,
            "dpf_high_energy_tail_usually_axial_beam": True,
            "stochastic_acceleration_may_inject_current_disruption_stage": True,
        },
        "model_scope_limits": {
            "mhd_valid_until_current_disruption": True,
            "strong_nonideal_electric_fields_not_modeled": True,
            "final_current_disruption_stage_future_work": True,
            "thermal_neutron_count_lower_than_perfect_collapse_models": True,
            "beam_target_neutrons_likely_dominate_actual_yield": True,
            "current_disruption_voltages_not_captured": True,
            "not_experimental_validation_packet": True,
        },
        "uncertainty": {
            "no_measured_hawk_dpf_current_trace_in_target": True,
            "no_measured_hawk_dpf_voltage_trace_in_target": True,
            "no_measured_hawk_neutron_yield_in_target": True,
            "no_detector_response_or_calibration_uncertainty": True,
            "no_quantitative_error_bars_for_qualitative_agreement": True,
            "initial_density_distribution_approximated_from_measurements": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "measured_current_trace",
            "measured_voltage_trace",
            "digitized_simulation_current_trace",
            "digitized_simulation_voltage_trace",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "measured_pinching_time",
            "measured_current_disruption_time",
            "phase_endpoint_uncertainty",
        ],
        "missing_for_full_tier4": [
            "measured_density_profile",
            "measured_temperature_profile",
            "measured_magnetic_field_profile",
            "same_shot_spatial_profile_uncertainty",
        ],
        "missing_for_full_tier5": [
            "measured_neutron_yield",
            "neutron_time_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "beam_target_current_disruption_model",
            "nonideal_electric_field_acceleration",
        ],
        "validation_note": (
            "This Beresnyak 2018 source is valuable for HAWK-specific 3D "
            "MHD scope, circuit coupling, nonaxisymmetric injected plasma, "
            "Hall-MHD qualitative behavior, and stochastic ion-acceleration "
            "bounds. It is not an end-to-end validation target because the "
            "experiments were planned, current disruption was not modeled, "
            "and measured HAWK DPF waveform, profile, neutron, detector, and "
            "uncertainty data are absent from the local extract."
        ),
    }


def narkis_kr_doped_dpf_mhd_targets() -> dict[str, object]:
    """Return KR-backed Kr-doped Gemini-like DPF radiation-MHD targets."""
    source = "KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md"
    return {
        "target_id": "narkis_kr_doped_dpf_mhd_2021",
        "device": "Gemini-like DPF",
        "model_role": "kr_2d_radiation_mhd_scope_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_scope_and_limits": "59-69",
            "introduction_and_mhd_scope": "72-127",
            "hydra_geometry_and_mesh": "129-143,194-196",
            "circuit_and_initial_conditions": "144-181",
            "dopant_eos_and_radiation_transport": "182-189",
            "sheath_dynamics_and_width": "191-231,251-261,319-329",
            "temperature_density_table": "272-299,330-401",
            "radiation_and_peak_neutron_results": "406-460",
            "summary_and_missing_physics": "488-518",
        },
        "shot_context": {
            "facility_context": "Gemini DPF at Nevada Test Site",
            "simulation_code": "HYDRA",
            "geometry": "quasi-2D R-Z with one azimuthal cell",
            "peak_current_MA_range": [2.0, 3.0],
            "peak_current_MA_max_considered": 3.2,
            "working_gas": "deuterium",
            "krypton_dopant_volume_fraction_cases": [0.0, 0.001, 0.01],
            "charging_voltage_kV_cases": [35.0, 40.0, 45.0, 50.0],
            "experimental_current_data_available_kV": [35.0, 40.0],
        },
        "geometry": {
            "anode_radius_cm": 7.62,
            "cathode_radius_cm": 10.16,
            "anode_length_cm_without_cap": 43.18,
            "cathode_length_cm": 59.18,
            "hemispherical_anode_cap_radius_cm": 7.62,
            "rundown_mesh_resolution_mm": [1.0, 1.0],
            "near_cap_mesh_resolution_um": [200.0, 200.0],
            "compute_time_hours_range": [24.0, 48.0],
            "compute_cores": 288,
        },
        "current_waveform_targets": {
            "rlc_circuit_model": True,
            "R_mohm": 1.4,
            "L_nH": 40.0,
            "C_uF": 432.0,
            "fill_pressure_scale_factor": 0.75,
            "resistance_treated_as_free_parameter": True,
            "matched_implosion_times_and_peak_currents_as_sanity_check": True,
            "strict_experimental_current_trace_fit_not_done": True,
            "short_circuit_load_fit_recommended_for_strict_comparison": True,
            "experimental_traces_unavailable_for_45_and_50_kV": True,
        },
        "initial_conditions": {
            "uniform_ionization_assumed": True,
            "boundary_z_cm": 1.0,
            "right_region_density_ug_cm3_range": [0.9, 2.3],
            "right_region_temperature_eV": 1.0,
            "initial_pressure_range": "few_to_several_Torr",
            "left_region_density_fraction": 0.1,
            "left_region_temperature_multiplier": 10.0,
            "mhd_density_floor_fraction_q0": 0.4,
            "high_voltage_initial_densities_scaled_with_voltage_squared": True,
        },
        "phase_timing": {
            "table_radius_on_anode_surface_mm": 5.0,
            "time_us_by_dopant_and_voltage": {
                "0% Kr": {"35": 5.874, "40": 6.519, "45": 6.521, "50": 6.524},
                "0.1% Kr": {"35": 6.421, "40": 6.285, "45": 6.287, "50": 6.287},
                "1.0% Kr": {"35": 5.725, "40": 6.521, "45": 6.524, "50": 6.525},
            },
            "thermalization_at_axis_duration_ns": "few",
            "thermonuclear_neutron_pulse_duration_ns": "few",
            "kinetic_and_experimental_neutron_pulse_duration_ns": "tens_to_hundreds",
        },
        "spatial_density_targets": {
            "density_table_units": "1e18 cm^-3",
            "ion_density_by_dopant_and_voltage": {
                "0% Kr": {"35": 2.852, "40": 4.944, "45": 6.184, "50": 7.607},
                "0.1% Kr": {"35": 4.377, "40": 5.251, "45": 6.522, "50": 7.938},
                "1.0% Kr": {"35": 4.347, "40": 9.917, "45": 12.91, "50": 15.87},
            },
            "sheath_width_0_and_0p1_percent_start_mm_range": [7.0, 10.0],
            "sheath_width_0_and_0p1_percent_narrowing_factor_range": [3.0, 4.0],
            "sheath_width_1_percent_narrowing_factor_at_pinch": 2.0,
            "species_separation_not_modeled": True,
        },
        "temperature_targets": {
            "table_temperature_units": "eV",
            "ion_temperature_by_dopant_and_voltage": {
                "0% Kr": {"35": 340.9, "40": 231.4, "45": 237.5, "50": 245.9},
                "0.1% Kr": {"35": 266.5, "40": 284.4, "45": 267.6, "50": 250.6},
                "1.0% Kr": {"35": 361.1, "40": 172.3, "45": 178.3, "50": 156.0},
            },
            "electron_temperature_by_dopant_and_voltage": {
                "0% Kr": {"35": 276.0, "40": 240.2, "45": 252.1, "50": 265.5},
                "0.1% Kr": {"35": 180.0, "40": 199.3, "45": 200.4, "50": 199.1},
                "1.0% Kr": {"35": 107.1, "40": 91.4, "45": 100.6, "50": 98.5},
            },
            "peak_temperature_keV_by_dopant": {
                "0% Kr": 6.7,
                "0.1% Kr": 8.3,
                "1.0% Kr": 12.6,
            },
            "two_temperature_persists_for_0p1_and_1_percent_Kr": True,
        },
        "magnetic_field_targets": {
            "bulk_sheath_dynamics_from_radiation_mhd": True,
            "j_cross_b_force_drives_runin": True,
            "solid_cathode_quasi_2d_mass_flow_matches_recent_experiments": True,
            "instability_growth_not_quantified_prior_to_pinch": True,
            "3d_instability_and_kinetic_seeding_not_captured": True,
        },
        "radiation_targets": {
            "LEOS_mixed_deuterium_krypton": True,
            "multigroup_radiation_diffusion": True,
            "krypton_increases_radiative_losses": True,
            "radiative_losses_1_percent_vs_0_percent_multiplier_range": [2.0, 3.0],
            "radiative_losses_0p1_percent_vs_0_percent_increase_fraction_range": [
                0.20,
                0.60,
            ],
            "higher_dopant_narrows_sheath": True,
        },
        "neutron_yield_targets": {
            "thermonuclear_yield_order_range": [1.0e9, 1.0e10],
            "thermonuclear_yield_increases_with_krypton_dopant": True,
            "experimental_total_yield_order_at_2MA": 1.0e11,
            "thermonuclear_fraction_context": 0.01,
            "dd_total_context_neutrons": 1.0e11,
            "dt_total_context_neutrons": 1.0e12,
            "power_law_exponents_all_points": {
                "0% Kr": 5.726,
                "0.1% Kr": 4.643,
                "1.0% Kr": 4.859,
                "overall": 5.102,
            },
            "power_law_exponents_excluding_35kV": {
                "0% Kr": 5.403,
                "0.1% Kr": 4.852,
                "1.0% Kr": 4.012,
                "overall": 4.754,
            },
            "max_dNdt_35kV_neutrons_per_ns": {
                "0% Kr": 1.1e9,
                "0.1% Kr": 2.4e9,
                "1.0% Kr": 1.8e9,
            },
            "yield_drop_at_3MA_seen_experimentally_not_reproduced": True,
        },
        "model_scope_limits": {
            "fully_kinetic_required_for_total_neutron_yield": True,
            "mhd_cannot_capture_beam_target_neutron_production": True,
            "initiation_and_termination_are_fundamentally_kinetic": True,
            "breakdown_physics_neglected": True,
            "2d_mhd_likely_overestimates_pinch_tightness": True,
            "3d_and_fully_kinetic_confirmation_required": True,
            "strict_current_trace_fit_not_performed": True,
            "species_separation_not_accounted": True,
        },
        "uncertainty": {
            "circuit_resistance_free_parameter": True,
            "initial_fill_pressure_scale_factor_used": True,
            "experimental_data_only_35_and_40kV": True,
            "no_detector_response_in_target": True,
            "no_beam_target_uncertainty": True,
            "no_3d_instability_uncertainty": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "spatial_temperature",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "strict_digitized_experimental_current_trace_fit",
            "short_circuit_load_fit",
            "voltage_trace_comparison",
            "per_point_waveform_uncertainty",
        ],
        "missing_for_full_tier2": [
            "measured_implosion_times_for_all_voltage_dopant_cases",
            "measured_pinching_times",
            "breakdown_timing_model",
            "phase_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "same_shot_density_profile",
            "same_shot_temperature_profile",
            "same_shot_magnetic_field_profile",
            "3d_instability_measurement",
            "species_separation_diagnostic",
        ],
        "missing_for_full_tier5": [
            "beam_target_neutron_model",
            "fully_kinetic_stagnation_model",
            "neutron_detector_response",
            "neutron_spectrum",
            "neutron_anisotropy",
            "experimental_neutron_pulse_width_match",
            "3d_kinetic_kr_doped_validation",
        ],
        "validation_note": (
            "This Narkis/Hahn 2021 source is a strong KR constraint on "
            "radiation-MHD scope for Kr-doped, Gemini-like DPF simulations: "
            "Kr narrows the sheath, increases radiative losses, and increases "
            "thermonuclear yield in 2D MHD. It also explicitly blocks using "
            "2D MHD to claim total predictive neutron yield because breakdown, "
            "3D instability, species separation, kinetic stagnation, and "
            "beam-target production are absent or only bounded."
        ),
    }


def faeton_i_high_voltage_dpf_targets() -> dict[str, object]:
    """Return KR-backed partial FAETON-I high-voltage DPF targets."""
    source = (
        "KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-"
        "radiation-output-of-a-100-kv-plasma-focus-device.md"
    )
    return {
        "target_id": "faeton_i_high_voltage_dpf_2025",
        "device": "FAETON-I",
        "model_role": "kr_high_voltage_dpf_validation_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "table_3_current_factors_yields": "11-46,191-195",
            "current_factor_interpretation": "47-55",
            "ntof_faraday_gamma_diagnostics": "56-62",
            "conclusion_device_yields_diagnostics": "63-73",
            "restrike_vmax_interpretation": "74-85",
            "scaling_and_dt_projection_limits": "90-99",
        },
        "shot_context": {
            "device": "FAETON-I",
            "direct_charged_voltage_kV": 100.0,
            "deuterium_shots_recorded_min": 1100,
            "consecutive_shots_without_gas_refill": 5,
            "max_voltage_shot_deuterium_fraction": 0.992,
            "max_voltage_shot_krypton_fraction": 0.008,
            "max_voltage_shot_pressure_torr": 12.0,
            "source_extract_is_references_conclusion_and_table_only": True,
        },
        "current_waveform_targets": {
            "radial_current_factor_good_sheath_threshold": 0.7,
            "radial_current_factor_exceptional_range": [0.8, 0.9],
            "poor_shot_current_factor_example": 0.4,
            "average_voltage_spike_for_good_sheath_kV": 100.0,
            "peak_inductive_voltage_preferred_indicator_high_voltage_pf": True,
            "current_dip_severity_not_good_indicator_when_restrikes_truncate_dip": True,
            "restrike_severity_does_not_control_peak_voltage_or_yield": True,
            "voltage_peak_pre_stagnation_and_dynamics_induced": True,
            "table_3_shots": [
                {
                    "shot": 1062,
                    "fcr": 0.4,
                    "fcr2": 0.35,
                    "Vp_kV": 37.3,
                    "Yn_code": 2.77e9,
                    "Yn_measured": 3.0e9,
                },
                {
                    "shot": 1036,
                    "fcr": 0.72,
                    "fcr2": 0.35,
                    "Vp_kV": 101.4,
                    "Yn_code": 2.54e10,
                    "Yn_measured": 2.21e10,
                },
                {
                    "shot": 1027,
                    "fcr": 0.8,
                    "fcr2": 0.58,
                    "Vp_kV": 160.5,
                    "Yn_code": 5.5e10,
                    "Yn_measured": 5.44e10,
                },
                {
                    "shot": 895,
                    "fcr": 0.9,
                    "fcr2": 0.7,
                    "Vp_kV": 194.0,
                    "Yn_code": 4.1e10,
                    "Yn_measured": 6.0e10,
                },
            ],
        },
        "phase_timing": {
            "voltage_spike_before_stagnation": True,
            "first_piston_induced_neutron_pulse_before_stagnation": True,
            "second_pinch_instability_neutron_pulse_afterward": True,
            "restrikes_expected_to_reduce_second_pulse": True,
            "absolute_phase_times_available_in_extract": False,
        },
        "event_sequence": [
            {
                "event": "pre_stagnation_voltage_spike",
                "mechanism": "dynamics_induced_piston_voltage",
                "relative_time": "before_stagnation",
                "required": True,
            },
            {
                "event": "first_neutron_pulse",
                "mechanism": "piston_induced_beam_target",
                "relative_time": "before_stagnation",
                "required": True,
            },
            {
                "event": "second_neutron_pulse",
                "mechanism": "pinch_instability",
                "relative_time": "after_first_pulse",
                "required": False,
            },
        ],
        "neutron_yield_targets": {
            "five_shot_consistent_dd_yield": 2.5e10,
            "exceptional_dd_yield_max": 8.0e10,
            "max_voltage_shot_dd_yield": 6.0e10,
            "good_fcr_yield_about": 2.5e10,
            "table_3_measured_yields": {
                "1062": 3.0e9,
                "1036": 2.21e10,
                "1027": 5.44e10,
                "895": 6.0e10,
            },
            "dd_yield_outperforms_universal_scaling_law": True,
            "dynamics_induced_yield_predominant_in_source_interpretation": True,
        },
        "spectral_targets": {
            "dd_neutron_energy_peak_MeV": 2.5,
            "dd_neutron_energy_uncertainty_MeV": 0.3,
            "fast_deuteron_energy_keV_about": 350.0,
            "gamma_energy_detected_MeV_min": 3.0,
        },
        "anisotropy_targets": {
            "forward_on_axis_factor": 1.6,
        },
        "response_model_requirements": {
            "pmt_scintillator_distances_m": [5.0, 10.0, 20.0, 40.0],
            "ntof_distance_m": 40.0,
            "lead_shielding_cm": 30.0,
            "neutron_spectrum_from_pulse_shape_analysis_required": True,
            "faraday_cup_ion_energy_from_xray_to_ion_tof": True,
            "gamma_detection_response_above_3MeV_required": True,
        },
        "projections_not_validation_targets": {
            "faeton_x_dt_yield_projection_low": {
                "charging_voltage_kV": 65.0,
                "stored_energy_MJ": 1.0,
                "peak_current_MA": 4.0,
                "dt_14p1MeV_neutron_yield": 2.0e14,
            },
            "faeton_x_dt_yield_projection_high": {
                "charging_voltage_kV": 150.0,
                "stored_energy_MJ": 5.0,
                "peak_current_MA": 7.0,
                "dt_14p1MeV_neutron_yield": 2.0e15,
            },
            "dt_projection_is_not_validated_by_faeton_i_dd_data": True,
        },
        "uncertainty": {
            "neutron_energy_uncertainty_MeV": 0.3,
            "full_body_not_available_in_local_markdown": True,
            "digitized_current_voltage_traces_not_available": True,
            "full_shot_dataset_available_only_by_request": True,
            "detector_response_calibration_uncertainty_not_in_extract": True,
            "lee_model_two_step_fit_supported_as_reported_fit_not_physics_law": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "neutron_timing",
            "neutron_spectrum",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace",
            "digitized_voltage_trace",
            "per_point_trace_uncertainty",
            "instrument_response_for_voltage_and_current_diagnostics",
        ],
        "missing_for_full_tier2": [
            "absolute_axial_radial_stagnation_phase_times",
            "neutron_pulse_absolute_timing_history",
            "same_shot_phase_endpoint_uncertainty",
        ],
        "missing_for_full_tier4": [
            "same_shot_density_profile",
            "same_shot_temperature_profile",
            "same_shot_magnetic_field_profile",
        ],
        "missing_for_full_tier5": [
            "full_detector_response_model",
            "yield_calibration_uncertainty",
            "full_neutron_time_histories",
            "full_neutron_spectra",
            "complete_shot_dataset",
            "independent_dt_projection_validation",
        ],
        "validation_note": (
            "This local FAETON-I markdown extraction supplies a useful high-"
            "voltage DPF current-factor, voltage, yield, spectrum, anisotropy, "
            "and diagnostic target, but only from a table/conclusion extract. "
            "It cannot close full predictive validation without digitized "
            "waveforms, phase times, detector response, calibration "
            "uncertainty, and the full shot dataset. D-T yield values are "
            "recorded only as projections, not validation targets."
        ),
    }


def mjolnir_high_low_parasitic_current_targets() -> dict[str, object]:
    """Return KR-backed MJOLNIR high/low-yield parasitic-current targets."""
    source = "KnowledgeReference/goyon-2022-mjolnir-high-low.md"
    return {
        "target_id": "mjolnir_high_low_parasitic_current_2022_goyon",
        "validation_scope": "mjolnir_high_low_parasitic_current_2022_goyon",
        "device": "MJOLNIR",
        "model_role": "kr_parasitic_current_yield_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract": "25-41",
            "motivation_and_diagnostics_context": "44-131",
            "device_and_circuit": "144-216",
            "diagnostics": "217-245",
            "electrodes_and_fill": "246-286",
            "pic_model": "287-311",
            "snowplow_model": "312-332",
            "phase_shape_and_pic_density": "341-394",
            "conditioning_rundown": "397-486",
            "runin_velocity": "488-557",
            "current_dip_model": "565-645",
            "voltage_and_didt_yield": "646-701",
            "pic_parasitic_yield": "713-781",
            "pressure_scaling": "783-832",
            "conclusion": "836-908",
        },
        "shot_context": {
            "device": "MJOLNIR",
            "purpose": "single-pulse flash neutron radiography",
            "one_MJ_configuration_energy_MJ": 1.0,
            "two_MJ_configuration_energy_MJ": 2.0,
            "one_MJ_marx_towers": 3,
            "two_MJ_marx_towers": 6,
            "one_MJ_marx_modules": 12,
            "two_MJ_marx_modules": 24,
            "module_capacitors_uF_each": 34.0,
            "one_MJ_transmission_cables": 84,
            "two_MJ_transmission_cables": 168,
            "one_MJ_peak_current_MA_at_100kV": 2.5,
            "two_MJ_commissioned_peak_current_MA_at_70kV": 3.25,
            "one_MJ_highest_neutron_yield": 3.8e11,
            "two_MJ_highest_neutron_yield": 4.1e11,
            "one_MJ_lumped_capacitance_uF": 204.0,
            "one_MJ_lumped_inductance_nH": 67.4,
            "one_MJ_lumped_resistance_mohm": 12.5,
            "two_MJ_estimated_capacitance_uF": 408.0,
            "two_MJ_estimated_inductance_nH": 46.7,
            "two_MJ_estimated_resistance_mohm": 6.3,
            "deuterium_fill_pressure_torr_range": [8.0, 24.0],
            "anode_diameters_cm": [15.0, 23.0],
            "anode_exposed_lengths_cm_range": [18.0, 25.0],
            "anode_hollow_radii_cm": [3.8, 0.9, 1.9],
            "ak_gap_cm": 4.3,
            "cathode_rods": 24,
            "insulator_material": "MACOR",
            "insulator_exposed_length_cm": 4.6,
        },
        "diagnostic_requirements": {
            "be_activation_yield_detector": True,
            "main_head_rogowski_current": True,
            "anode_cathode_voltage_probe": True,
            "fiber_coupled_photodiodes_for_rundown_timing": True,
            "visible_framing_camera_frames": 16,
            "visible_framing_camera_exposure_ns": 3.0,
            "voltage_probe_resistor_kV": 100.0,
            "voltage_probe_resistor_ohm": 955.5,
            "voltage_probe_rogowski_nH": 8.0,
            "voltage_probe_bandwidth_MHz": 50.0,
            "voltage_probe_frequency_corrected_MHz": 200.0,
            "copper_shield_thickness_mm": 1.0,
            "kapton_layers": 10,
            "kapton_layer_thickness_in": 0.003,
        },
        "simulation_context": {
            "pic_code": "CHICAGO",
            "circuit_code": "BERTHA",
            "fluid_stage_time_step_ps": 10.0,
            "pre_kinetic_time_step_fs": 100.0,
            "kinetic_stage_time_step_fs": 100.0,
            "cell_size_um_range": [25.0, 300.0],
            "kinetic_duration_after_ion_beam_ns": 100.0,
            "dd_fusion_package": True,
            "binary_collisions": True,
            "snowplow_pistons": 3,
            "snowplow_anode_shape_multiplier_required": True,
            "snowplow_parasitic_current_path_enabled": True,
        },
        "phase_semantics": {
            "run_down": "plasma sheath traverses the anode before run-in",
            "run_in": "plasma sheath implodes radially toward smaller radius",
            "stagnation": "plasma column reaches minimum radius",
            "expansion": "post-stagnation radial expansion",
            "second_compression": "nonuniform second implosion after expansion",
            "column_breakup": "late discontinuity or breakup of plasma column",
        },
        "current_waveform_targets": {
            "current_trace_diagnostic": "main head Rogowski coil",
            "voltage_trace_diagnostic": "anode-cathode voltage probe",
            "low_yield_signature": "smaller current drop during implosion",
            "current_dip_slope_correlates_with_yield": True,
            "voltage_spike_correlates_with_yield": True,
            "high_yield_voltage_spike_kV_about": 180.0,
            "nominal_dataset_shots": 48,
            "nominal_dataset_pressure_torr": 16.0,
            "nominal_dataset_voltage_kV_range": [90.0, 100.0],
            "conditioning_current_path_inductance_nH": 7.0,
            "conditioning_current_path_resistance_mohm": 7.0,
            "high_yield_current_path_time_after_stagnation_ns": 50.0,
            "low_yield_conditioning_path_time_before_stagnation_ns": 200.0,
            "medium_yield_conditioning_path_time_before_stagnation_ns": 50.0,
            "lowest_nominal_path_time_before_implosion_ns_min": 150.0,
            "voltage_trace_divergence_before_best_max_voltage_ns_about": 200.0,
        },
        "phase_timing": {
            "run_down_speed_scaling": "I / (r * sqrt(rho))",
            "run_down_velocity_conditioning_stabilizes_after_shots_about": 5,
            "yield_conditioning_rises_over_shots_min": 10,
            "simple_rebuild_rundown_stabilizes_after_shots_about": 2,
            "simple_rebuild_yield_stabilizes_after_shots_about": 4,
            "runin_velocity_dataset_voltage_kV": 60.0,
            "runin_velocity_dataset_pressure_torr": 8.0,
            "runin_velocity_dataset_configuration_MJ": 2.0,
            "runin_average_last_radius_cm": 1.5,
            "high_yield_shots_have_higher_runin_velocity": True,
            "stabilized_rundown_and_runin_are_necessary_not_sufficient": True,
        },
        "spatial_density_targets": {
            "pic_density_phases": [
                "run-in",
                "stagnation",
                "expansion",
                "break-up",
            ],
            "beam_generation_location_z_cm_about": -1.0,
            "dense_target_location_z_cm_range": [1.0, 2.0],
            "beam_generation_location_stochastic": True,
            "target_location_anode_shape_influenced": True,
        },
        "field_context": {
            "parasitic_current_paths_divert_current_from_main_sheath": True,
            "rBtheta_diagnostic_used_in_pic": True,
            "constant_rBtheta_indicates_no_regional_current_flow": True,
            "tip_or_insulator_current_path_cases": True,
            "current_path_net_resistance_mohm": 25.0,
            "current_path_introduced_before_implosion_ns": 100.0,
            "late_tip_path_time_before_stagnation_ns": 10.0,
            "magnetic_energy_downstream_correlates_with_beam_energy": True,
        },
        "neutron_yield_targets": {
            "mjolnir_max_neutrons_per_pulse": 4.1e11,
            "mjolnir_max_peak_current_MA": 3.3,
            "nominal_yield_fluctuation_factor_about": 2.0,
            "conditioning_or_offnormal_yield_variation_order_of_magnitude": True,
            "april_2019_second_day_yield_range": [1.2e10, 9.2e10],
            "highest_yield_no_intentional_parasitic_path": True,
            "lower_yield_with_insulator_parasitic_path": True,
            "lowest_yield_with_anode_tip_parasitic_path": True,
            "higher_pressure_yield_degradation_observed": True,
            "high_pressure_current_dip_reproducibility_degrades": True,
        },
        "event_sequence": [
            {
                "event": "conditioning",
                "mechanism": "run-down velocity and yield rise after hardware exposure",
                "required": True,
            },
            {
                "event": "parasitic_path_before_stagnation",
                "mechanism": "alternate low-inductance current path reduces current dip",
                "required": True,
            },
            {
                "event": "beam_target_neutron_generation",
                "mechanism": "ion beam into downstream dense target",
                "required": True,
            },
        ],
        "activation_requirements": {
            "be_activation_detector": True,
            "absolute_response_details_in_source": False,
        },
        "response_model_requirements": [
            "be_activation_calibration",
            "rogowski_response",
            "voltage_probe_transfer_function",
            "photodiode_timing_response",
            "framing_camera_timing_response",
        ],
        "uncertainty": {
            "nominal_shot_to_shot_yield_fluctuation_factor_about": 2.0,
            "rundown_velocity_error_bars_random_only": True,
            "systematic_light_collection_error_nearly_doubles_velocity_error": True,
            "current_dip_dataset_shot_count": 48,
            "digitized_trace_uncertainty_provided": False,
            "activation_detector_uncertainty_provided": False,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_density",
            "spatial_magnetic_or_em",
            "neutron_timing",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_rogowski_current_traces",
            "digitized_voltage_probe_traces",
            "per_point_current_voltage_uncertainty",
        ],
        "missing_for_full_tier2": [
            "shot_resolved_rundown_and_runin_time_table",
            "stagnation_time_uncertainty",
            "direct_plasma_sheath_current_measurement",
        ],
        "missing_for_full_tier4": [
            "direct_experimental_density_map",
            "direct_experimental_temperature_map",
            "direct_experimental_magnetic_field_map",
            "impurity_spectroscopy_measurements",
        ],
        "missing_for_full_tier5": [
            "be_activation_detector_response_model",
            "time_resolved_neutron_history",
            "neutron_spectrum",
            "neutron_anisotropy",
            "shot_resolved_yield_uncertainty",
        ],
        "validation_note": (
            "This target captures MJOLNIR high/low-yield evidence for "
            "parasitic current paths. It is a mechanism and diagnostic target, "
            "not a complete same-shot neutron validation packet."
        ),
    }


def mjolnir_first_experiments_targets() -> dict[str, object]:
    """Return KR-backed MJOLNIR first-experiments validation context."""
    source = (
        "KnowledgeReference/ieee-trans-plas-sci-paper-first-experiments-and-"
        "radiographs-on-the-megajoule-neutron-imaging.md"
    )
    return {
        "target_id": "mjolnir_first_experiments_2021_offermann",
        "validation_scope": "mjolnir_1mj_first_experiments_2021",
        "device": "MJOLNIR",
        "model_role": "kr_partial_mjolnir_campaign_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract": "37-62",
            "device_and_circuit": "131-158",
            "diagnostics": "167-259",
            "simulation_benchmarking": "260-317",
            "yield_and_anode_campaign": "318-360",
            "radiograph_and_timing": "366-405",
            "light_gate_velocity": "406-464",
            "current_restrike_interpretation": "466-496",
            "anode_optimization_and_limits": "497-559",
            "conclusion_and_future_work": "560-592",
        },
        "shot_context": {
            "device": "MJOLNIR",
            "campaign": "first 1 MJ pulsed-power configuration",
            "first_plasmas_date": "2018-08-30",
            "first_neutrons_date": "2018-08-31",
            "stored_energy_MJ_max": 1.0,
            "erected_voltage_kV_max": 100.0,
            "charge_voltage_kV_each_polarity": 50.0,
            "peak_current_MA_max": 2.5,
            "high_voltage_plasma_shots": 436,
            "gas_fill_pressure_torr_range": [4.0, 30.0],
            "best_yield_pressure_torr_range": [4.0, 16.0],
            "working_gas": "deuterium",
        },
        "geometry_targets": {
            "anode_diameter_cm": 15.2,
            "anode_exposed_length_cm_range": [18.3, 22.1],
            "anode_hollow_radii_cm": [0.9, 3.8],
            "anode_cathode_gap_cm": 4.3,
            "macor_insulator_exposed_length_cm": 4.6,
            "initial_implosion_radius_cm": 4.3,
            "optimized_implosion_radius_cm": 1.4,
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "rogowski_between_transmission_plate_and_electrodes": True,
            "rogowski_frequency_corrected": True,
            "lumped_capacitance_uF": 204.0,
            "lumped_inductance_nH": 67.4,
            "lumped_resistance_mohm": 12.5,
            "peak_current_MA_at_100kV": 2.5,
            "current_dip_depth_correlates_with_yield": True,
            "restrike_model_required_to_match_current_traces": True,
        },
        "phase_semantics": {
            "flashover": "gas fill flashes over across the insulator",
            "rundown": "plasma sheath accelerates down the coaxial gun",
            "runin": "plasma sheath accelerates inward toward the axis",
            "pinch": "plasma current sheath pinches on axis and breaks up",
            "restrike": "current is diverted away from the pinch region",
        },
        "phase_timing": {
            "optical_light_gates_measure_sheath_arrival": True,
            "run_down_speed_scales_inverse_sqrt_pressure": True,
            "run_down_speed_scales_with_voltage": True,
            "model_camera_timing_agreement_fraction_about": 0.015,
            "pinch_jitter_for_operating_parameters_ns_min": 10.0,
            "poor_yield_shots_show_slower_runin": True,
        },
        "neutron_yield_targets": {
            "first_yield_neutrons_per_pulse_at_60kV": 3.1e10,
            "max_yield_neutrons_per_pulse": 3.8e11,
            "radiograph_shots": 5,
            "radiograph_total_yield_neutrons": 1.1e12,
            "radiograph_detector_distance_m": 2.6,
            "radiograph_flux_at_detector_n_per_cm2": 1.3e6,
        },
        "event_sequence": [
            {
                "event": "xray_pulse",
                "mechanism": "pinch x-ray emission used for neutron-camera timing",
                "required": True,
            },
            {
                "event": "neutron_pulse",
                "mechanism": "DPF neutron pulse timed with nToF for camera gate",
                "required": True,
            },
            {
                "event": "early_restrike",
                "mechanism": "early current diversion associated with poor yield",
                "required": True,
            },
        ],
        "activation_requirements": {
            "be_activation_detector_main_yield": True,
            "be_detector_sandia_ibl_calibrated": True,
            "be_detector_angle_deg_range": [45.0, 50.0],
            "be_detector_distance_m_range": [1.33, 1.65],
            "distance_changed_to_2p63m_for_inverse_square_check": True,
            "two_pmt_yields_agree_within_fraction": 0.02,
        },
        "tof_requirements": {
            "ntof_used_for_xray_and_neutron_arrival": True,
            "second_ntof_used_for_xray_timing": True,
            "mcp_gate_confirmed_by_incremental_timing_scan": True,
            "lead_plate_discriminated_xray_vs_neutron_timing": True,
        },
        "response_model_requirements": [
            "activation_calibration_matching_scatter_environment",
            "be_detector_position_and_distance_correction",
            "neutron_camera_scintillator_fiber_response",
            "collimator_and_camera_box_direct_hit_correction",
            "ntof_based_gate_timing",
            "anisotropy_characterization_for_radiograph_flux",
        ],
        "uncertainty": {
            "scatter_contribution_checked_by_inverse_square_series": True,
            "be_detector_two_pmts_within_fraction": 0.02,
            "anisotropy_quantitative_characterization_available": False,
            "optical_light_gate_window_coating_limits_availability": True,
            "current_trace_cable_reflection_can_influence_didt": True,
            "neutron_source_size_not_yet_confirmed": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "neutron_timing",
            "neutron_anisotropy",
            "neutron_detector_response",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_frequency_corrected_current_traces",
            "per_point_current_uncertainty",
            "voltage_trace_digitization_and_uncertainty",
        ],
        "missing_for_full_tier2": [
            "digitized_light_gate_arrival_times",
            "digitized_runin_velocity_points",
            "shot_by_shot_phase_endpoint_uncertainty",
        ],
        "missing_for_full_tier5": [
            "quantitative_neutron_anisotropy_characterization",
            "digitized_ntof_traces",
            "complete_neutron_camera_response_model",
            "shot_by_shot_yield_uncertainty_table",
        ],
        "validation_note": (
            "This paper is a high-value MJOLNIR campaign target for circuit, "
            "phase, neutron-yield, and diagnostic context. It remains partial "
            "because most waveform, light-gate, nToF, and uncertainty data are "
            "reported through figures or qualitative comparisons rather than "
            "digitized point-by-point validation series."
        ),
    }


def pf400j_xray_inference_targets() -> dict[str, object]:
    """Return KR-backed PF-400J x-ray diagnostic inference targets."""
    source = (
        "KnowledgeReference/inference-of-x-ray-emission-from-a-plasma-focus-"
        "discharge-comparison-between-characteristic.md"
    )
    return {
        "target_id": "pf400j_xray_inference_2020_orellana",
        "device": "PF-400J",
        "model_role": "kr_xray_diagnostic_inference_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract": "38-56",
            "phase_and_diagnostic_context": "87-188",
            "device_and_conditions": "190-220",
            "electrical_diagnostics": "221-235",
            "xray_detector_and_campaign": "236-290",
            "pinch_voltage_scatter": "393-430",
            "feature_selection": "461-565",
            "ml_results_and_limits": "628-681",
            "conclusion": "774-823",
            "normalization": "837-882",
            "cnn_signal_shape": "922-952",
        },
        "shot_context": {
            "device": "PF-400J",
            "architecture": "Mather-type small plasma focus",
            "fill_gas": "hydrogen",
            "fill_pressure_mbar": 9.0,
            "charging_voltage_kV": 26.0,
            "stored_energy_J": 287.0,
            "capacitance_nF": 850.0,
            "external_inductance_nH": 39.0,
            "external_resistance_mohm": 42.0,
            "quarter_period_ns": 291.0,
            "anode_material": "steel",
            "anode_effective_length_mm": 13.0,
            "insulator_material": "alumina",
            "insulator_length_mm": 23.0,
            "cathode_bars_used": False,
            "spark_gap_gas": "nitrogen",
            "spark_gap_pressure_bar": 0.5,
            "recorded_discharges": 959,
            "measurement_campaigns": 2,
        },
        "phase_semantics": {
            "breakdown": "gas breakdown and current-sheath formation over insulator",
            "rundown": "current sheath moves along the anode by magnetic force",
            "compression": "current sheath compresses ionized gas at anode end",
            "pinch": "maximum compression produces dense plasma column",
            "disruption": "pinch disruption produces axial shock",
            "jet": "plasma jets emitted from the anode top",
        },
        "current_waveform_targets": {
            "rogowski_current_didt_available": True,
            "inductive_loop_current_didt_available": True,
            "voltage_divider_available": True,
            "vivaldi_antenna_em_burst_available": True,
            "gas_breakdown_marker": "abrupt voltage-divider fall with Rogowski dI/dt rise",
            "pinch_time_marker": "pinch observed in Rogowski coil",
            "voltage_at_pinch_kV_high_pmt_signal_range": [10.0, 14.0],
            "voltage_at_pinch_kV_highest_bins_range": [16.0, 20.0],
            "scatter_bins_kV": 2.0,
            "signal_parameters": [
                "breakdown_voltage",
                "pinch_voltage",
                "breakdown_to_pinch_time",
                "pinch_current",
                "pinch_didt",
                "vivaldi_breakdown_transient_energy",
                "vivaldi_pinch_transient_energy",
                "vivaldi_pinch_fft",
            ],
        },
        "phase_timing": {
            "breakdown_to_pinch_parameter": "tbp",
            "timing_uncertainty_source": "cable length from voltage divider and Rogowski sensors",
            "timing_uncertainty_ns_order": "few",
            "proper_timing_needed_for_peak_current_at_focus": True,
        },
        "field_context": {
            "vivaldi_antenna_distance_m": 0.25,
            "inductive_loop_distance_mm": 2.0,
            "em_burst_near_and_far_field_components": True,
            "remote_noninvasive_diagnostic_supported": True,
            "high_frequency_oscillations_carry_discharge_information": True,
        },
        "xray_detector_targets": {
            "hard_xray_energy_keV_range": [1.0, 300.0],
            "average_absorbed_dose_mGy_per_shot": 2.4,
            "detector": "scintillator-photomultiplier",
            "scintillator": "BC-408",
            "pmt_model": "Hamamatsu R1828-01",
            "aluminum_casing_thickness_mm": 5.0,
            "system_response_energy_keV_min": 20.0,
            "pmt_bias_kV": 1.4,
            "pmt_linearity_threshold_mA": 500.0,
            "pmt_linearity_max_deviation_fraction": 0.05,
            "sc_pmt_distance_from_anode_m": 0.54,
            "lead_filter_thickness_mm": 4.0,
            "lead_filter_cutoff_keV": 250.0,
            "normalized_standard_deviation_R_used": True,
        },
        "data_acquisition": {
            "tds_648a_bandwidth_GHz": 1.0,
            "tds_648a_sampling_GSamples_per_s": 5.0,
            "tds_648a_channels": [
                "Rogowski",
                "voltage_divider",
                "unblocked_SC_PMT",
                "blocked_SC_PMT",
            ],
            "ni_pxi_bandwidth_GHz": 3.0,
            "ni_pxi_sampling_GSamples_per_s": 6.25,
            "ni_pxi_vertical_bits": 8,
            "ni_pxi_channels": ["Vivaldi", "inductive_loop_sensor"],
            "entire_signal_samples": 5625,
            "entire_signal_window_ns": 900.0,
            "cnn_matrix_shape": [75, 75],
        },
        "ml_inference_targets": {
            "task": "classify x-ray detector R-value class from electrical/EM signals",
            "training_fraction": 0.80,
            "validation_fraction": 0.20,
            "training_realizations": 40,
            "best_feature_set": [
                "breakdown_voltage",
                "pinch_voltage",
                "pinch_current",
                "pinch_didt",
                "breakdown_to_pinch_time",
                "vivaldi_pinch_fft",
            ],
            "entire_signal_no_significant_improvement": True,
            "best_models_include": ["gradient_boost", "2D_CNN"],
            "random_guessing_probability_below_percent": 1.0,
            "specificity_best_percent_range": [40.0, 60.0],
            "false_positives_and_false_negatives_reported": True,
        },
        "uncertainty": {
            "pinch_voltage_scatter_high": True,
            "r_value_distribution_biased_to_low_emission": True,
            "larger_campaign_recommended_for_balanced_classes": True,
            "antenna_normalization_position_sensitive": True,
            "cross_device_normalization_unresolved": True,
            "digitized_trace_uncertainty_provided": False,
            "absolute_xray_yield_uncertainty_provided": False,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_magnetic_or_em",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_voltage_antenna_detector_traces",
            "per_channel_calibration_uncertainty",
            "per_point_timing_uncertainty",
        ],
        "missing_for_full_tier2": [
            "shot_resolved_breakdown_to_pinch_table",
            "pinch_time_uncertainty_distribution",
            "current_sheath_position_diagnostic",
        ],
        "missing_for_full_tier4": [
            "absolute_xray_spectrum",
            "detector_response_function",
            "direct_density_temperature_magnetic_field_diagnostics",
            "cross_device_validation_of_ml_normalization",
        ],
        "missing_for_full_tier5": [
            "deuterium_or_dt_neutron_target",
            "neutron_yield_measurement",
            "neutron_detector_response",
        ],
        "validation_note": (
            "This PF-400J source supports x-ray diagnostic inference and "
            "electrical/EM feature selection. It is hydrogen, hundreds-of-"
            "joules, and x-ray focused; it must not be used as neutron-yield "
            "validation for a deuterium DPF."
        ),
    }


def pf1000_16kv_phase_candidate_evidence_from_history(
    times_s: Sequence[float],
    phases: Sequence[object],
    *,
    target: dict[str, object] | None = None,
    pinch_time_tolerance_us: float = 0.5,
    pinch_duration_relative_tolerance: float = 0.50,
) -> dict[str, object]:
    """Compare phase labels with partial PF-1000 shot-12581 timing targets."""
    target = target or pf1000_16kv_shot12581_phase_targets()
    times = np.asarray(times_s, dtype=float)
    phase_labels = [
        str(phase).strip().lower().replace("-", "_").replace(" ", "_")
        for phase in phases
    ]
    n = min(times.size, len(phase_labels))
    times = times[:n]
    phase_labels = phase_labels[:n]

    pinch_labels = {"pinch", "reflected", "post_pinch", "stagnation"}
    pinch_indices = [
        idx for idx, phase in enumerate(phase_labels)
        if phase in pinch_labels and math.isfinite(float(times[idx]))
    ]

    timing = target["phase_timing"]
    target_dip_time_s = float(timing["current_dip_end_time_us"]) * 1.0e-6
    target_duration_s = float(timing["pinch_duration_ns"]) * 1.0e-9

    pinch_time_s = None
    pinch_duration_s = None
    pinch_time_error = math.inf
    pinch_duration_error = math.inf
    if pinch_indices:
        pinch_time_s = float(times[pinch_indices[0]])
        pinch_time_error = abs(pinch_time_s - target_dip_time_s)
        start = pinch_indices[0]
        end = start
        for idx in pinch_indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                break
        if end > start:
            pinch_duration_s = float(times[end] - times[start])
            if target_duration_s > 0.0:
                pinch_duration_error = (
                    abs(pinch_duration_s - target_duration_s) / target_duration_s
                )

    pinch_time_passed = pinch_time_error <= pinch_time_tolerance_us * 1.0e-6
    duration_passed = (
        pinch_duration_s is not None
        and pinch_duration_error <= pinch_duration_relative_tolerance
    )

    return {
        "passed": False,
        "phases": {
            "axial": False,
            "radial": False,
            "pinch": pinch_time_passed and duration_passed,
        },
        "target": "pf1000_16kv_shot12581_phase",
        "validation_scope": target.get("validation_scope", target["target_id"]),
        "model_role": "simulation_to_kr_partial_phase_target_comparison",
        "validation_tier": 2,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "pinch_time_s": pinch_time_s,
            "target_current_dip_end_time_s": target_dip_time_s,
            "pinch_time_absolute_error_s": pinch_time_error,
            "pinch_time_tolerance_s": pinch_time_tolerance_us * 1.0e-6,
            "pinch_duration_s": pinch_duration_s,
            "target_pinch_duration_s": target_duration_s,
            "pinch_duration_relative_error": pinch_duration_error,
            "pinch_duration_relative_tolerance": pinch_duration_relative_tolerance,
            "missing_for_full_tier2": target["missing_for_full_tier2"],
        },
        "validity_notes": {
            "partial_phase_target": (
                "This evidence is intentionally not a tier-2 pass because the "
                "KR record does not provide all axial, radial, and pinch timing "
                "targets required by the predictive-readiness gate."
            ),
        },
    }


def pf1000_16kv_derived_output_candidate_evidence(
    observables: dict[str, object],
    *,
    target: dict[str, object] | None = None,
    relative_tolerance: float = 0.25,
) -> dict[str, object]:
    """Compare PF-1000 16 kV Lee-derived outputs against KR targets.

    This remains candidate evidence because the KR record gives fitted Lee
    outputs and partial timing, not full measured axial/radial phase targets.
    """
    target = target or pf1000_16kv_shot12581_phase_targets()
    targets = target["derived_outputs"]
    normalized = {
        str(key).strip().lower(): value
        for key, value in observables.items()
    }
    aliases = {
        "peak_current_kA": ("peak_current_ka", "i_peak_ka"),
        "pinch_current_kA": ("pinch_current_ka", "i_pinch_ka"),
        "axial_speed_cm_per_us": ("axial_speed_cm_per_us", "va_cm_per_us"),
        "shock_speed_cm_per_us": ("shock_speed_cm_per_us", "vs_cm_per_us"),
        "piston_speed_cm_per_us": ("piston_speed_cm_per_us", "vp_cm_per_us"),
        "final_pinch_radius_cm": ("final_pinch_radius_cm", "pinch_radius_cm", "rp_cm"),
        "pinch_length_cm": ("pinch_length_cm", "zp_cm"),
        "vmax_kV": ("vmax_kv", "v_max_kv"),
    }

    observed_outputs: dict[str, float] = {}
    relative_errors: dict[str, float] = {}
    output_passes: dict[str, bool] = {}
    missing_outputs: list[str] = []
    for name, target_value in targets.items():
        observed_value = None
        for alias in aliases.get(name, (name.lower(),)):
            if alias in normalized:
                try:
                    observed_value = float(normalized[alias])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    observed_value = None
                break
        if observed_value is None or not math.isfinite(observed_value):
            missing_outputs.append(name)
            output_passes[name] = False
            relative_errors[name] = math.inf
            continue
        target_float = float(target_value)
        observed_outputs[name] = observed_value
        if target_float == 0.0:
            error = 0.0 if observed_value == 0.0 else math.inf
        else:
            error = abs(observed_value - target_float) / abs(target_float)
        relative_errors[name] = error
        output_passes[name] = error <= relative_tolerance

    phases = {
        "axial": output_passes.get("axial_speed_cm_per_us", False),
        "radial": (
            output_passes.get("shock_speed_cm_per_us", False)
            and output_passes.get("piston_speed_cm_per_us", False)
        ),
        "pinch": (
            output_passes.get("pinch_current_kA", False)
            and output_passes.get("final_pinch_radius_cm", False)
            and output_passes.get("pinch_length_cm", False)
        ),
    }

    return {
        "passed": False,
        "phases": phases,
        "output_passes": output_passes,
        "target": "pf1000_16kv_shot12581_derived_outputs",
        "validation_scope": target.get("validation_scope", target["target_id"]),
        "model_role": "simulation_to_kr_partial_phase_dynamics_comparison",
        "validation_tier": 2,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "observed_outputs": observed_outputs,
            "target_outputs": targets,
            "relative_errors": relative_errors,
            "relative_tolerance": relative_tolerance,
            "missing_outputs": missing_outputs,
            "missing_for_full_tier2": target["missing_for_full_tier2"],
        },
        "validity_notes": {
            "partial_derived_output_target": (
                "These are fitted Lee-code outputs from the PF-1000 KR record. "
                "They can audit production observables, but they do not replace "
                "full measured axial, radial, and pinch timing validation."
            ),
        },
    }


def _normalized_observable_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _akel_prediction_rows(
    predictions: Mapping[object, object] | Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    if isinstance(predictions, Mapping):
        rows_value = predictions.get("shot_rows")
        if isinstance(rows_value, Sequence) and not isinstance(
            rows_value,
            (str, bytes, bytearray),
        ):
            return _akel_prediction_rows([
                row for row in rows_value if isinstance(row, Mapping)
            ])

        keyed_rows: dict[int, Mapping[str, object]] = {}
        for shot, value in predictions.items():
            if isinstance(value, Mapping):
                try:
                    shot_id = int(shot)
                except (TypeError, ValueError):
                    shot_value = value.get("shot")
                    try:
                        shot_id = int(shot_value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                keyed_rows[shot_id] = value
        if keyed_rows:
            return keyed_rows
        if "shot" in predictions:
            try:
                return {int(predictions["shot"]): predictions}
            except (TypeError, ValueError):
                return {}
        return {}

    rows: dict[int, Mapping[str, object]] = {}
    for row in predictions:
        if not isinstance(row, Mapping):
            continue
        try:
            rows[int(row["shot"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _akel_observed_float(
    row: Mapping[str, object],
    aliases: Sequence[str],
) -> float | None:
    normalized = {
        _normalized_observable_key(key): value
        for key, value in row.items()
    }
    for alias in aliases:
        value = normalized.get(_normalized_observable_key(alias))
        if value is None:
            continue
        try:
            observed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return observed if math.isfinite(observed) else None
    return None


def pf1000_16kv_akel_table_candidate_evidence(
    predictions: Mapping[object, object] | Sequence[Mapping[str, object]],
    *,
    target: dict[str, object] | None = None,
    required_fields: Sequence[str] | None = None,
    relative_tolerance: float = 0.25,
) -> dict[str, object]:
    """Compare shot-resolved scalar predictions with Akel 2021 table rows."""
    target = target or pf1000_16kv_akel_table_targets()
    target_rows = {
        int(row["shot"]): row
        for row in target["shot_rows"]  # type: ignore[index]
        if isinstance(row, Mapping)
    }
    observed_rows = _akel_prediction_rows(predictions)
    fields = tuple(required_fields or (
        "peak_current_kA",
        "pinch_current_kA",
        "axial_speed_cm_per_us",
        "shock_speed_cm_per_us",
        "piston_speed_cm_per_us",
        "pinch_density_1e23_per_m3",
        "pinch_radius_cm",
        "pinch_length_cm",
        "neutron_yield_n",
    ))
    target_field = {
        "neutron_yield_n": "measured_neutron_yield_n",
    }
    aliases = {
        "peak_current_kA": ("peak_current_kA", "i_peak_kA", "ipeak_kA"),
        "pinch_current_kA": ("pinch_current_kA", "i_pinch_kA", "ipinch_kA"),
        "axial_speed_cm_per_us": ("axial_speed_cm_per_us", "va_cm_per_us"),
        "shock_speed_cm_per_us": ("shock_speed_cm_per_us", "vs_cm_per_us"),
        "piston_speed_cm_per_us": ("piston_speed_cm_per_us", "vp_cm_per_us"),
        "pinch_density_1e23_per_m3": (
            "pinch_density_1e23_per_m3",
            "ni_1e23_per_m3",
        ),
        "pinch_radius_cm": ("pinch_radius_cm", "rp_cm", "final_pinch_radius_cm"),
        "pinch_length_cm": ("pinch_length_cm", "zp_cm"),
        "neutron_yield_n": (
            "neutron_yield_n",
            "predicted_neutron_yield_n",
            "computed_neutron_yield_n",
            "yield_n",
            "yn",
        ),
    }

    missing_shots = sorted(set(target_rows) - set(observed_rows))
    extra_shots = sorted(set(observed_rows) - set(target_rows))
    shot_results: list[dict[str, object]] = []
    field_passes = {field: True for field in fields}
    field_errors: dict[str, list[float]] = {field: [] for field in fields}
    missing_fields: dict[str, list[int]] = {field: [] for field in fields}
    yield_uncertainty_normalized_errors: list[float] = []

    for shot in sorted(target_rows):
        row_target = target_rows[shot]
        observed_row = observed_rows.get(shot)
        field_results: dict[str, dict[str, object]] = {}
        row_passed = observed_row is not None
        for field in fields:
            canonical_target_field = target_field.get(field, field)
            target_value = float(row_target[canonical_target_field])
            observed_value = (
                _akel_observed_float(observed_row, aliases.get(field, (field,)))
                if observed_row is not None else None
            )
            if observed_value is None:
                field_passes[field] = False
                missing_fields[field].append(shot)
                field_results[field] = {
                    "passed": False,
                    "target_value": target_value,
                    "observed_value": None,
                    "relative_error": math.inf,
                }
                row_passed = False
                continue
            relative_error = (
                0.0 if target_value == 0.0 and observed_value == 0.0
                else math.inf if target_value == 0.0
                else abs(observed_value - target_value) / abs(target_value)
            )
            passed = relative_error <= relative_tolerance
            if not passed:
                field_passes[field] = False
                row_passed = False
            field_errors[field].append(relative_error)
            field_results[field] = {
                "passed": passed,
                "target_value": target_value,
                "observed_value": observed_value,
                "relative_error": relative_error,
            }
            if field == "neutron_yield_n":
                uncertainty = float(
                    row_target.get("measured_neutron_yield_uncertainty_n", math.nan)
                )
                absolute_error = abs(observed_value - target_value)
                normalized_uncertainty_error = (
                    absolute_error / uncertainty
                    if math.isfinite(uncertainty) and uncertainty > 0.0
                    else math.inf
                )
                field_results[field].update({
                    "absolute_error": absolute_error,
                    "measured_uncertainty_n": uncertainty,
                    "measurement_uncertainty_normalized_error": (
                        normalized_uncertainty_error
                    ),
                })
                yield_uncertainty_normalized_errors.append(
                    normalized_uncertainty_error
                )
        shot_results.append({
            "shot": shot,
            "passed": row_passed,
            "fields": field_results,
        })

    all_rows_present = not missing_shots
    passed = all_rows_present and all(field_passes.values())
    max_relative_errors = {
        field: max(errors) if errors else math.inf
        for field, errors in field_errors.items()
    }

    return {
        "passed": passed,
        "target": "pf1000_16kv_shot_table",
        "validation_scope": target["validation_scope"],
        "model_role": "simulation_to_kr_akel_table_scalar_yield_comparison",
        "validation_tier": 5,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "validated_features": {"yield": passed},
        "required_fields": list(fields),
        "relative_tolerance": relative_tolerance,
        "row_count": {
            "required": len(target_rows),
            "provided": len(set(observed_rows) & set(target_rows)),
        },
        "field_passes": field_passes,
        "max_relative_errors": max_relative_errors,
        "max_measurement_uncertainty_normalized_error": (
            max(yield_uncertainty_normalized_errors)
            if yield_uncertainty_normalized_errors else math.inf
        ),
        "missing_shots": missing_shots,
        "extra_shots": extra_shots,
        "missing_fields": {
            field: shots for field, shots in missing_fields.items() if shots
        },
        "shot_results": shot_results,
        "validity_notes": {
            "scalar_table_only": (
                "This evidence compares scalar table rows. It does not validate "
                "digitized current waveforms, phase-transition timing, neutron "
                "pulse timing, spectrum, anisotropy, or detector response."
            ),
            "tolerance_origin": (
                "The relative tolerance is an explicit software acceptance "
                "threshold supplied to this function, not a source-reported "
                "predictive validation criterion."
            ),
        },
    }


def mjolnir_neutron_timing_targets() -> dict[str, object]:
    """Return KR-backed neutron mechanism/timing targets for MJOLNIR."""
    source = (
        "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
        "dense-plasma-focus-z-pinch-5.md"
    )
    return {
        "target_id": "mjolnir_neutron_timing_2025_goyon",
        "device": "MJOLNIR",
        "model_role": "kr_validation_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "mechanisms": "405-448",
            "tof_and_smoothing": "474-530",
            "spectrum_anisotropy": "548-616",
        },
        "shot_context": {
            "charge_voltage_kV": 60.0,
            "stored_energy_kJ": 735.0,
            "average_peak_current_MA": 2.8,
            "stagnation_current_MA": 2.1,
        },
        "event_sequence": [
            {
                "event": "stagnation",
                "mechanism": "thermonuclear",
                "relative_time_ns": 0.0,
                "required": True,
            },
            {
                "event": "first_disruption",
                "mechanism": "beam_target",
                "relative_time_ns": 5.0,
                "timing_note": "MHD-kinetic simulated burst about 5 ns after stagnation.",
                "required": True,
            },
            {
                "event": "first_beam_target_measurement_correlation",
                "mechanism": "beam_target",
                "relative_time_ns": 10.0,
                "timing_note": (
                    "Synthetic pulse shape predicts the first beam-target event "
                    "about 10 ns after stagnation, consistent with measurements."
                ),
                "required": True,
            },
            {
                "event": "later_disruptions",
                "mechanism": "beam_target",
                "relative_time_ns": None,
                "timing_note": "Later beam-target peaks correspond to pinch disruptions.",
                "required": False,
            },
        ],
        "detector_tof": {
            "neutron_energy_MeV": 2.45,
            "arrival_delay_ns_first_detector": 96.0,
        },
        "spectral_targets": {
            "thermonuclear": "narrow spectrum around 2.45 MeV",
            "beam_target": "broader spectrum with high-energy neutrons up to about 5 MeV",
        },
        "anisotropy_targets": {
            "low_yield": "on-axis/off-axis measurements within about 10 percent error bar",
            "high_yield": (
                "on-axis activation can be about 60-100 percent higher than off-axis, "
                "depending on reaction channel"
            ),
        },
        "validation_note": (
            "A simulation should compare neutron birth history, beam-target timing, "
            "spectrum broadening, and anisotropy trends against these targets before "
            "claiming tier-5 neutron mechanism/timing validation."
        ),
    }


def mjolnir_stagnation_temperature_targets() -> dict[str, object]:
    """Return KR-backed MJOLNIR stagnation-temperature target context."""
    source = (
        "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
        "dense-plasma-focus-z-pinch-5.md"
    )
    return {
        "target_id": "mjolnir_stagnation_temperature_2025_goyon",
        "validation_scope": "mjolnir_neutron_timing_2025_goyon",
        "device": "MJOLNIR",
        "model_role": "kr_spatial_temperature_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "stagnation_temperature_scaling": "293-316",
            "several_kev_context": "405-416",
            "xray_filter_context": "459-461",
        },
        "temperature_targets": {
            "stagnation_temperature_scaling_reference_keV": 21.0,
            "temperature_order": "several_keV",
            "temperature_quantity": "average_stagnation_temperature_Tst",
            "temperature_definition": "(Te + Ti) / 2",
            "working_gas": "deuterium",
        },
        "missing_for_full_tier4": [
            "direct_experimental_temperature_diagnostic",
            "experimental_temperature_uncertainty",
            "same_scope_density_and_magnetic_field_targets",
        ],
        "validation_note": (
            "KR provides a shock-theory/MHD-kinetic temperature context for "
            "MJOLNIR stagnation and reports several-keV stagnation conditions. "
            "This is useful target context for same-scope audits, but it is not "
            "a direct experimental temperature diagnostic and remains partial "
            "for tier 4."
        ),
    }


def mjolnir_neutron_spectrum_evidence(
    thermonuclear_energies_MeV: Sequence[float],
    beam_target_energies_MeV: Sequence[float],
    *,
    target: dict[str, object] | None = None,
    thermonuclear_center_MeV: float = 2.45,
    thermonuclear_mean_tolerance_MeV: float = 0.15,
    thermonuclear_std_max_MeV: float = 0.30,
    beam_high_energy_min_MeV: float = 3.0,
    beam_high_energy_max_MeV: float = 5.5,
) -> dict[str, object]:
    """Compare mechanism-separated neutron energies with MJOLNIR targets."""
    target = target or mjolnir_neutron_timing_targets()
    thermo = np.asarray(thermonuclear_energies_MeV, dtype=float)
    beam = np.asarray(beam_target_energies_MeV, dtype=float)
    thermo = thermo[np.isfinite(thermo)]
    beam = beam[np.isfinite(beam)]

    thermo_passed = False
    beam_passed = False
    thermo_mean = math.nan
    thermo_std = math.nan
    beam_max = math.nan
    beam_std = math.nan

    if thermo.size >= 2:
        thermo_mean = float(np.mean(thermo))
        thermo_std = float(np.std(thermo))
        thermo_passed = (
            abs(thermo_mean - thermonuclear_center_MeV)
            <= thermonuclear_mean_tolerance_MeV
            and thermo_std <= thermonuclear_std_max_MeV
        )

    if beam.size >= 2:
        beam_max = float(np.max(beam))
        beam_std = float(np.std(beam))
        beam_passed = (
            beam_high_energy_min_MeV <= beam_max <= beam_high_energy_max_MeV
            and beam_std > thermo_std
        )

    passed = thermo_passed and beam_passed
    return {
        "passed": passed,
        "validated_features": {"spectrum": passed},
        "diagnostics": {"spectrum": passed},
        "mechanisms": {
            "thermonuclear": thermo_passed,
            "beam_target": beam_passed,
        },
        "target": "mjolnir_neutron_spectrum",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 5,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "thermonuclear": {
                "mean_MeV": thermo_mean,
                "std_MeV": thermo_std,
                "target_center_MeV": thermonuclear_center_MeV,
                "mean_tolerance_MeV": thermonuclear_mean_tolerance_MeV,
                "std_max_MeV": thermonuclear_std_max_MeV,
                "n_samples": int(thermo.size),
                "passed": thermo_passed,
            },
            "beam_target": {
                "max_MeV": beam_max,
                "std_MeV": beam_std,
                "high_energy_min_MeV": beam_high_energy_min_MeV,
                "high_energy_max_MeV": beam_high_energy_max_MeV,
                "n_samples": int(beam.size),
                "passed": beam_passed,
            },
        },
        "validity_notes": {
            "spectrum_scope": (
                "MJOLNIR KR expects a narrow thermonuclear spectrum around "
                "2.45 MeV and broader beam-target spectrum extending to high "
                "energies. This helper checks those mechanism-separated "
                "spectral features; it does not test angular anisotropy."
            ),
        },
    }


def mjolnir_neutron_anisotropy_evidence(
    on_axis_yield: float,
    off_axis_yield: float,
    *,
    yield_regime: str = "high_yield",
    target: dict[str, object] | None = None,
    low_yield_relative_tolerance: float = 0.10,
    high_yield_ratio_range: tuple[float, float] = (0.60, 1.00),
) -> dict[str, object]:
    """Compare on-axis/off-axis neutron yield anisotropy with MJOLNIR KR."""
    target = target or mjolnir_neutron_timing_targets()
    on_axis = float(on_axis_yield)
    off_axis = float(off_axis_yield)
    if not math.isfinite(on_axis) or not math.isfinite(off_axis) or off_axis <= 0.0:
        anisotropy_ratio = math.inf
    else:
        anisotropy_ratio = on_axis / off_axis - 1.0

    regime = str(yield_regime).strip().lower().replace("-", "_").replace(" ", "_")
    if regime == "low_yield":
        passed = abs(anisotropy_ratio) <= low_yield_relative_tolerance
        target_description = target["anisotropy_targets"]["low_yield"]
        tolerance_details: dict[str, object] = {
            "absolute_ratio_tolerance": low_yield_relative_tolerance,
        }
    else:
        low, high = high_yield_ratio_range
        passed = low <= anisotropy_ratio <= high
        target_description = target["anisotropy_targets"]["high_yield"]
        tolerance_details = {
            "target_ratio_range": [low, high],
        }

    return {
        "passed": passed,
        "validated_features": {"anisotropy": passed},
        "diagnostics": {"anisotropy": passed},
        "target": "mjolnir_neutron_anisotropy",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 5,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "yield_regime": regime,
            "on_axis_yield": on_axis,
            "off_axis_yield": off_axis,
            "on_axis_excess_ratio": anisotropy_ratio,
            "target_description": target_description,
            **tolerance_details,
        },
        "validity_notes": {
            "anisotropy_scope": (
                "MJOLNIR KR compares on-axis and off-axis neutron activation. "
                "This helper validates that angular trend only; it does not "
                "replace timing or spectral validation."
            ),
        },
    }


def mjolnir_neutron_detector_response_targets() -> dict[str, object]:
    """Return KR-backed MJOLNIR detector/activation response targets."""
    source = (
        "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
        "dense-plasma-focus-z-pinch-5.md"
    )
    return {
        "target_id": "mjolnir_neutron_detector_response_2025_goyon",
        "validation_scope": "mjolnir_neutron_timing_2025_goyon",
        "device": "MJOLNIR",
        "model_role": "kr_detector_response_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "activation_diagnostics": "132-149",
            "tof_and_timing_diagnostics": "160-168",
            "synthetic_detector_response": "449-509",
            "activation_anisotropy": "595-607",
        },
        "activation_requirements": {
            "reactions": ["Be", "Y", "Br"],
            "be_absolute_calibrated": True,
            "labr_y_cross_calibrated_to_be": True,
            "anisotropy_angles_deg": [10.0, 70.0],
            "be_reference_angle_deg": 45.0,
        },
        "tof_requirements": {
            "scintillator_distances_m": [2.2, 6.6],
            "relative_timing_precision_ns_max": 1.0,
            "neutron_energy_MeV": 2.45,
            "arrival_delay_ns_first_detector": 96.0,
        },
        "response_model_requirements": [
            "propagation_widening",
            "detector_temporal_response",
            "xray_peak_cotiming",
            "beam_target_energy_spread",
            "room_scatter_or_background_assessment",
        ],
        "validation_note": (
            "KR reports absolutely/cross-calibrated activation channels, timed "
            "scintillator TOF detectors, synthetic detector response broadening, "
            "and unresolved room/equipment scattering. A high-fidelity neutron "
            "claim should therefore carry detector and activation response "
            "evidence in addition to neutron birth-history evidence."
        ),
    }


def mcalpine_dpf_nrta_mcnp_targets() -> dict[str, object]:
    """Return KR-backed DPF NRTA MCNP application targets."""
    source = (
        "KnowledgeReference/monte-carlo-simulations-of-neutron-resonance-"
        "transmission-analysis-with-the-dense-plasma-focus.md"
    )
    return {
        "target_id": "mcalpine_dpf_nrta_mcnp_2014",
        "device": "LLNL DPF / NRTA application model",
        "model_role": "kr_neutron_application_response_target",
        "validation_tier": 5,
        "source": source,
        "source_lines": {
            "abstract_and_source_value": "32-42",
            "llnl_dpf_source_context": "45-117",
            "nrta_energy_and_tof_concept": "119-142",
            "mcnp_transport_setup": "144-165",
            "methods_source_and_pulses": "168-204",
            "tof_and_eng_comparison_results": "250-302",
            "material_discrimination_results": "303-339",
            "conclusions": "342-384",
            "future_work": "387-406",
        },
        "dpf_source_context": {
            "fusion_reaction": "D-D",
            "source_neutron_energy_MeV": 2.45,
            "llnl_dpf_yield_neutrons_about": 1.0e7,
            "llnl_simulated_pulse_duration_ns_range": [20.0, 60.0],
            "generic_dpf_stored_energy_range_J": [10.0, 1.0e6],
            "generic_neutron_yield_per_pulse_range": [1.0e4, 1.0e13],
            "generic_neutron_pulse_duration_ns_range": [10.0, 100.0],
            "llnl_working_gas": "deuterium",
            "dt_mixture_possible": True,
            "kinetic_simulations_used_to_inform_design_context": True,
        },
        "nrta_targets": {
            "resonance_energy_eV_range": [1.0, 50.0],
            "target_nuclide_examples": ["235U", "239Pu"],
            "absorption_notches_identify_isotopes": True,
            "notch_amplitudes_indicate_relative_amounts": True,
            "tof_required_for_low_energy_neutron_spectrum": True,
            "applications": ["arms_control", "safeguards", "snm_assay"],
        },
        "mcnp_setup": {
            "transport_code": "MCNP",
            "source_model": "monoenergetic isotropic point source",
            "moderator_material": "polyethylene",
            "moderator_thickness_cm": 3.0,
            "detector_distance_m": 2.0,
            "detector_assumed": "3He",
            "detector_efficiency_postprocess_scaling": "sigma_abs proportional to 1/v",
            "inspection_object_volume_cm3_about": 180.0,
            "inspection_object_has_steel_cladding": True,
            "passive_background_ignored": True,
            "interrogation_time_s_under": 1.0,
            "source_particles_per_simulation": 1.0e10,
        },
        "nrta_tof_context": {
            "dpf_gaussian_fwhm_ns": 20.0,
            "eng_trapezoidal_pulse_us": 4.0,
            "tof_spectrum_compared_to_true_detector_energy_spectrum": True,
            "tof_broadens_resonances_slightly": True,
            "tof_reproduces_resonance_locations": True,
        },
        "application_results": {
            "dpf_resolves_resonances_undetectable_with_eng": True,
            "eng_time_for_comparable_resolvable_measurement": "about_a_day",
            "dpf_time_for_comparable_measurement": "single_pulse",
            "materials_compared": [
                "depleted_uranium",
                "highly_enriched_uranium",
                "plutonium",
                "lead",
            ],
            "distinguishes_snm_from_lead": True,
            "distinguishes_uranium_from_plutonium": True,
            "distinguishes_heu_from_depleted_uranium": True,
            "common_elements_without_resonances_in_range": ["N", "O", "C", "Fe"],
        },
        "nrta_tof_requirements": {
            "short_pulse_needed_for_nrta": True,
            "compact_flight_path_m": 2.0,
            "moderator_thickness_cm": 3.0,
            "source_pulse_fwhm_ns": 20.0,
            "energy_range_eV": [1.0, 50.0],
        },
        "nrta_response_model_requirements": [
            "direct_detector_response_in_transport",
            "room_and_surroundings_geometry",
            "minimum_neutron_yield_analysis",
            "experimental_nrta_benchmark",
            "novel_detector_material_response",
        ],
        "model_scope_limits": {
            "application_transport_target_not_dpf_plasma_validation": True,
            "dpf_source_not_self_consistently_simulated": True,
            "source_assumed_monoenergetic_isotropic_point": True,
            "detector_response_only_post_processed": True,
            "no_current_voltage_or_phase_diagnostics": True,
            "no_neutron_birth_spectrum_or_anisotropy_from_dpf": True,
            "experiments_required_for_confidence": True,
        },
        "uncertainty": {
            "no_experimental_nrta_uncertainty": True,
            "passive_background_ignored": True,
            "room_scatter_not_included": True,
            "minimum_yield_not_determined": True,
            "detector_response_not_directly_simulated": True,
        },
        "missing_for_nrta_validation": [
            "self_consistent_dpf_neutron_source_model",
            "neutron_birth_spectrum",
            "source_anisotropy",
            "direct_detector_response_transport",
            "experimental_nrta_benchmark",
            "minimum_yield_validation",
        ],
        "missing_for_full_dpf_validation": [
            "current_voltage_waveforms",
            "phase_timing",
            "same_shot_spatial_profiles",
            "mechanism_separated_neutron_history",
            "source_spectrum_anisotropy",
            "validated_detector_response",
        ],
        "validation_note": (
            "This LLNL report is useful for DPF-enabled NRTA application and "
            "TOF detector-response requirements. It should not be read as "
            "validation of the DPF plasma model, because MCNP uses a simplified "
            "point neutron source and future work explicitly calls for "
            "experiments and direct detector-response modeling."
        ),
    }


def _token_set(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    elif isinstance(value, Sequence):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return {
        item.strip().lower().replace("-", "_").replace(" ", "_")
        for item in items
        if str(item).strip()
    }


def _reaction_set(value: object) -> set[str]:
    aliases = {
        "beryllium": "be",
        "be": "be",
        "yttrium": "y",
        "y": "y",
        "bromine": "br",
        "br": "br",
    }
    return {
        aliases[token]
        for token in _token_set(value)
        if token in aliases
    }


def _numeric_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = value
    else:
        values = [value]
    numbers: list[float] = []
    for item in values:
        try:
            number = float(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _contains_numeric(
    values: Sequence[float],
    target: float,
    *,
    absolute_tolerance: float,
) -> bool:
    return any(abs(value - target) <= absolute_tolerance for value in values)


def mjolnir_neutron_detector_response_evidence(
    detector_response: dict[str, object],
    *,
    target: dict[str, object] | None = None,
    angle_tolerance_deg: float = 2.0,
    distance_tolerance_m: float = 0.15,
) -> dict[str, object]:
    """Audit detector/activation response evidence against MJOLNIR KR."""
    target = target or mjolnir_neutron_detector_response_targets()
    reactions = _reaction_set(detector_response.get("activation_reactions"))
    angles = _numeric_list(detector_response.get("activation_detector_angles_deg"))
    angles.extend(_numeric_list(detector_response.get("anisotropy_angles_deg")))
    angles.extend(_numeric_list(detector_response.get("reference_detector_angle_deg")))
    angles.extend(_numeric_list(detector_response.get("be_reference_angle_deg")))
    distances = _numeric_list(detector_response.get("tof_distances_m"))
    distances.extend(_numeric_list(detector_response.get("scintillator_distances_m")))

    response_terms = _token_set(detector_response.get("response_terms"))
    response_terms.update(_token_set(detector_response.get("modeled_effects")))
    response_aliases = {
        "propagation_tof": "propagation_widening",
        "time_of_flight": "propagation_widening",
        "tof": "propagation_widening",
        "temporal_response": "detector_temporal_response",
        "detector_response": "detector_temporal_response",
        "xray_cotiming": "xray_peak_cotiming",
        "x_ray_peak_cotiming": "xray_peak_cotiming",
        "energy_spread": "beam_target_energy_spread",
        "beam_energy_spread": "beam_target_energy_spread",
        "room_scatter": "room_scatter_or_background_assessment",
        "background_scatter": "room_scatter_or_background_assessment",
        "scattering_background": "room_scatter_or_background_assessment",
    }
    response_terms = {
        response_aliases.get(term, term)
        for term in response_terms
    }
    for bool_key, canonical in (
        ("propagation_widening_modeled", "propagation_widening"),
        ("detector_temporal_response_modeled", "detector_temporal_response"),
        ("xray_peak_cotiming_modeled", "xray_peak_cotiming"),
        ("beam_target_energy_spread_modeled", "beam_target_energy_spread"),
        ("room_scatter_or_background_assessed", "room_scatter_or_background_assessment"),
        ("scattering_background_modeled", "room_scatter_or_background_assessment"),
    ):
        if detector_response.get(bool_key) is True:
            response_terms.add(canonical)

    activation_requirements = target["activation_requirements"]
    tof_requirements = target["tof_requirements"]
    required_reactions = _reaction_set(activation_requirements["reactions"])
    required_response_terms = set(target["response_model_requirements"])
    timing_precision = detector_response.get("relative_timing_precision_ns")
    if timing_precision is None:
        timing_precision = detector_response.get("detector_timing_precision_ns")
    try:
        timing_precision_ns = float(timing_precision)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        timing_precision_ns = math.inf

    activation_reactions_passed = required_reactions.issubset(reactions)
    activation_angles_passed = all(
        _contains_numeric(angles, angle, absolute_tolerance=angle_tolerance_deg)
        for angle in activation_requirements["anisotropy_angles_deg"]
    ) and _contains_numeric(
        angles,
        float(activation_requirements["be_reference_angle_deg"]),
        absolute_tolerance=angle_tolerance_deg,
    )
    calibration_passed = (
        detector_response.get("be_absolute_calibrated") is True
        and (
            detector_response.get("labr_y_cross_calibrated_to_be") is True
            or detector_response.get("activation_cross_calibrated_to_be") is True
        )
    )
    tof_distances_passed = all(
        _contains_numeric(distances, distance, absolute_tolerance=distance_tolerance_m)
        for distance in tof_requirements["scintillator_distances_m"]
    )
    timing_precision_passed = (
        math.isfinite(timing_precision_ns)
        and timing_precision_ns <= float(tof_requirements["relative_timing_precision_ns_max"])
    )
    response_terms_passed = required_response_terms.issubset(response_terms)
    checks = {
        "activation_reactions": activation_reactions_passed,
        "activation_angles": activation_angles_passed,
        "activation_calibration": calibration_passed,
        "tof_distances": tof_distances_passed,
        "relative_timing_precision": timing_precision_passed,
        "response_model_terms": response_terms_passed,
    }
    passed = all(checks.values())
    missing_response_terms = sorted(required_response_terms - response_terms)

    return {
        "passed": passed,
        "validated_features": {"detector_response": passed},
        "diagnostics": {
            "activation_response": (
                activation_reactions_passed
                and activation_angles_passed
                and calibration_passed
            ),
            "tof_response": tof_distances_passed and timing_precision_passed,
            "synthetic_response_model": response_terms_passed,
        },
        "target": "mjolnir_neutron_detector_response",
        "validation_scope": target["validation_scope"],
        "model_role": "simulation_to_kr_detector_response_audit",
        "validation_tier": 5,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "checks": checks,
            "activation_reactions": sorted(reactions),
            "activation_angles_deg": sorted(angles),
            "tof_distances_m": sorted(distances),
            "relative_timing_precision_ns": timing_precision_ns,
            "response_terms": sorted(response_terms),
            "missing_response_terms": missing_response_terms,
        },
        "validity_notes": {
            "detector_response_scope": (
                "This evidence audits exported detector and activation response "
                "metadata. It does not substitute for neutron birth timing, "
                "spectrum, or anisotropy validation, and it remains false unless "
                "room/equipment scattering or equivalent background response is "
                "modeled or explicitly assessed."
            ),
        },
    }


def pf1000_spatial_pinch_targets() -> dict[str, object]:
    """Return KR-backed PF-1000 radiating-pinch geometry targets."""
    source = "KnowledgeReference/scholz-2006-pf1000-mega-joule.md"
    return {
        "target_id": "pf1000_spatial_pinch_2006_scholz",
        "device": "PF-1000",
        "model_role": "kr_validation_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "density_proxy_diagnostic": "333-346",
            "radiating_pinch_geometry": "375-383",
            "shot_context": "420",
            "neutron_xray_timing": "400-410",
            "summary_limits": "477-489",
        },
        "shot_context": {
            "fill_pressure_hPa": 4.0,
            "bank_energy_kJ": 734.0,
            "current_MA": 1.66,
        },
        "density_proxy_diagnostic": {
            "instrument": "filtered high-speed frame camera",
            "wavelength_nm": 589.0,
            "emission_basis": "bremsstrahlung intensity proportional to electron density squared",
        },
        "radiating_pinch_geometry": {
            "minimum_diameter_mm": 5.0,
            "radiating_length_cm": 5.0,
            "dense_sphere_diameter_cm": 1.0,
            "dense_sphere_z_range_cm": [6.0, 8.0],
            "dense_sphere_lifetime_ns_range": [30.0, 50.0],
            "plasma_column_evolution_ns": 200.0,
        },
        "emission_timing": {
            "xray_neutron_start_before_xray_peak_ns": [20.0, 30.0],
            "first_neutron_fwhm_ns": [50.0, 70.0],
            "second_neutron_fwhm_ns": [70.0, 100.0],
            "second_to_first_neutron_pulse_ratio": [3.0, 10.0],
        },
        "validation_note": (
            "This target supports density-proxy spatial validation from gated "
            "emission geometry. Full tier-4 spatial validation still also needs "
            "magnetic-field and temperature evidence."
        ),
    }


def pf1000_spatial_pinch_evidence_from_geometry(
    geometry: dict[str, object],
    *,
    target: dict[str, object] | None = None,
    diameter_tolerance: float = 0.50,
    length_tolerance: float = 0.50,
) -> dict[str, object]:
    """Compare synthetic radiating-pinch geometry with the PF-1000 KR target."""
    target = target or pf1000_spatial_pinch_targets()
    target_geometry = target["radiating_pinch_geometry"]
    target_diameter = float(target_geometry["minimum_diameter_mm"])
    target_length = float(target_geometry["radiating_length_cm"])

    measured_diameter = geometry.get("diameter_mm")
    measured_length = geometry.get("length_cm")
    has_region = geometry.get("has_radiating_region") is True

    diameter_error = math.inf
    length_error = math.inf
    if measured_diameter is not None and target_diameter > 0.0:
        diameter_error = abs(float(measured_diameter) - target_diameter) / target_diameter
    if measured_length is not None and target_length > 0.0:
        length_error = abs(float(measured_length) - target_length) / target_length

    density_passed = (
        has_region
        and diameter_error <= diameter_tolerance
        and length_error <= length_tolerance
    )
    return {
        "passed": density_passed,
        "diagnostics": {
            "density": density_passed,
        },
        "quantity_relative_errors": {
            "density_proxy_diameter": diameter_error,
            "density_proxy_length": length_error,
        },
        "target": "pf1000_spatial_pinch",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 4,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "measured_geometry": dict(geometry),
            "target_geometry": target_geometry,
            "diameter_tolerance": diameter_tolerance,
            "length_tolerance": length_tolerance,
            "missing_for_full_tier4": ["magnetic_field", "temperature"],
        },
        "validity_notes": {
            "density_proxy": (
                "The PF-1000 target is based on bremsstrahlung image geometry, "
                "not a calibrated density field."
            ),
            "tier4_limit": (
                "This evidence can cover the density part of spatial validation "
                "only; it must be combined with magnetic-field and temperature "
                "diagnostics before predictive-readiness tier 4 can pass."
            ),
        },
    }


def pf1000_interferometry_density_targets() -> dict[str, object]:
    """Return KR-backed PF-1000 interferometry density profile targets."""
    source = "KnowledgeReference/malir-2024-interferometry-dpf.md"
    return {
        "target_id": "pf1000_interferometry_density_2024_malir",
        "device": "PF-1000",
        "model_role": "kr_validation_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "device_and_diagnostic": "190-205",
            "shot_context": "208-239",
            "profile_selection": "301-330",
            "density_profile_features": "331-348",
            "density_uncertainty": "381-397",
            "comparison_limits": "945-990",
        },
        "shot_context": {
            "shot_13317": {
                "fill_pressure_Torr": 0.9,
                "initial_density_cm3": 2.9e16,
                "peak_current_MA": 1.5,
                "interferograms": 14,
            },
            "shot_13328": {
                "fill_pressure_Torr": 0.75,
                "initial_density_cm3": 2.4e16,
                "peak_current_MA": 1.3,
                "interferograms": 15,
            },
            "voltage_insulator_kV_min": 30.0,
            "observation_window_cm": [10.0, 15.0],
            "profile_height_above_anode_cm": 1.0,
            "profile_band_width_mm": 6.0,
        },
        "density_profile_targets": {
            "shot_13317": {
                "peak_density_cm3": 2.5e18,
                "peak_radius_cm": 0.5,
            },
            "shot_13328": {
                "peak_density_cm3": 2.0e18,
                "peak_radius_cm": 0.5,
            },
            "minimum_radius_cm": 1.0,
            "shock_width_cm": 1.0,
            "pre_axis_density_rise_cm3": [5.0e17, 1.0e18],
        },
        "uncertainty": {
            "relative_error_far_from_axis": 0.20,
            "axis_or_fringe_position_error_mm_max": 3.0,
            "near_axis_error_note": (
                "Relative error is larger closer than about 2.5 mm to the axis."
            ),
        },
        "validation_note": (
            "This target supports density-profile spatial validation from "
            "PF-1000 Mach-Zehnder interferometry at about 1 cm above the anode. "
            "It is a density-only tier-4 component and must be combined with "
            "same-scope magnetic-field and temperature evidence for full tier 4."
        ),
    }


def pf1000_interferometry_density_evidence_from_profile(
    radius_cm: Sequence[float],
    electron_density_cm3: Sequence[float],
    *,
    shot: str = "13328",
    target: dict[str, object] | None = None,
    peak_density_tolerance: float = 0.30,
    peak_radius_tolerance_cm: float = 0.50,
) -> dict[str, object]:
    """Compare a radial electron-density profile with PF-1000 KR targets."""
    target = target or pf1000_interferometry_density_targets()
    radii = np.asarray(radius_cm, dtype=float)
    densities = np.asarray(electron_density_cm3, dtype=float)
    n = min(radii.size, densities.size)
    radii = radii[:n]
    densities = densities[:n]
    finite = np.isfinite(radii) & np.isfinite(densities) & (densities >= 0.0)
    radii = radii[finite]
    densities = densities[finite]

    shot_key = f"shot_{str(shot).replace(' ', '').replace('-', '')}"
    profile_targets = target["density_profile_targets"]
    if shot_key not in profile_targets:
        shot_key = "shot_13328"
    shot_target = profile_targets[shot_key]

    peak_density = math.nan
    peak_radius = math.nan
    peak_density_error = math.inf
    peak_radius_error_cm = math.inf
    if densities.size:
        peak_idx = int(np.argmax(densities))
        peak_density = float(densities[peak_idx])
        peak_radius = float(radii[peak_idx])
        target_density = float(shot_target["peak_density_cm3"])
        target_radius = float(shot_target["peak_radius_cm"])
        if target_density > 0.0:
            peak_density_error = abs(peak_density - target_density) / target_density
        peak_radius_error_cm = abs(peak_radius - target_radius)

    density_passed = (
        densities.size >= 3
        and peak_density_error <= peak_density_tolerance
        and peak_radius_error_cm <= peak_radius_tolerance_cm
    )
    return {
        "passed": density_passed,
        "diagnostics": {
            "density": density_passed,
        },
        "quantity_relative_errors": {
            "electron_density_peak": peak_density_error,
        },
        "target": "pf1000_interferometry_density_profile",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 4,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "shot_key": shot_key,
            "n_profile_points": int(densities.size),
            "peak_density_cm3": peak_density,
            "target_peak_density_cm3": float(shot_target["peak_density_cm3"]),
            "peak_density_relative_error": peak_density_error,
            "peak_density_tolerance": peak_density_tolerance,
            "peak_radius_cm": peak_radius,
            "target_peak_radius_cm": float(shot_target["peak_radius_cm"]),
            "peak_radius_error_cm": peak_radius_error_cm,
            "peak_radius_tolerance_cm": peak_radius_tolerance_cm,
            "profile_height_above_anode_cm": target["shot_context"][
                "profile_height_above_anode_cm"
            ],
            "profile_band_width_mm": target["shot_context"]["profile_band_width_mm"],
            "uncertainty": target["uncertainty"],
            "missing_for_full_tier4": ["magnetic_field", "temperature"],
        },
        "validity_notes": {
            "density_profile_scope": (
                "This evidence compares radial electron-density profile features "
                "at the selected PF-1000 interferometry height. It does not "
                "validate magnetic-field or temperature structure."
            ),
            "uncertainty_scope": (
                "The KR source reports about 20 percent relative density error "
                "away from the axis and larger near-axis uncertainty; this "
                "helper uses loose feature tolerances and reports the KR "
                "uncertainty metadata."
            ),
        },
    }


def llnl_12kj_em_fluctuation_targets() -> dict[str, object]:
    """Return KR-backed EM fluctuation targets for the LLNL 1.2 kJ DPF."""
    source = (
        "KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-"
        "simulations-with-experimental-measurements.md"
    )
    return {
        "target_id": "llnl_12kj_em_fluctuation_2014_schmidt",
        "device": "LLNL 1.2 kJ DPF",
        "model_role": "kr_validation_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "rf_probe_setup": "120-122",
            "frequency_comparison": "156-164",
            "field_and_lhdi_context": "168-170",
            "figure_summary": "234-238",
        },
        "diagnostic": {
            "instrument": "EM pick-up probe",
            "measurement_bandwidth_GHz": 5.0,
            "simulated_probe": "synthetic Ez probe 0.75 cm from axis",
        },
        "frequency_targets": {
            "high_quality_pinch_band_GHz": [3.0, 4.0],
            "poor_pinch_upper_activity_GHz": 3.0,
            "simulated_strongest_band_GHz": [3.0, 4.0],
        },
        "field_context": {
            "simulated_pinch_field_T": [10.0, 40.0],
            "lower_hybrid_frequency_GHz": [4.6, 18.0],
        },
        "validation_note": (
            "This target supports EM/magnetic fluctuation validation. Full "
            "tier-4 spatial validation also needs density and temperature "
            "diagnostics for the same claimed run scope."
        ),
    }


def _dominant_frequency_ghz(
    times_s: Sequence[float],
    signal: Sequence[float],
) -> tuple[float | None, float]:
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    n = min(times.size, values.size)
    if n < 4:
        return None, 0.0
    times = times[:n]
    values = values[:n]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        return None, 0.0

    dt = float(np.median(np.diff(times)))
    if dt <= 0.0:
        return None, 0.0
    centered = values - float(np.mean(values))
    if float(np.max(np.abs(centered))) <= 0.0:
        return None, 0.0

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(n, dt)
    if spectrum.size <= 1 or float(np.sum(spectrum[1:])) <= 0.0:
        return None, 0.0
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    total_power = float(np.sum(spectrum[1:]))
    return float(freqs[peak_idx] / 1.0e9), total_power


def llnl_12kj_em_fluctuation_evidence_from_signal(
    times_s: Sequence[float],
    signal: Sequence[float],
    *,
    target: dict[str, object] | None = None,
    min_band_power_fraction: float = 0.40,
) -> dict[str, object]:
    """Compare an EM probe signal with the LLNL 1.2 kJ DPF RF target."""
    target = target or llnl_12kj_em_fluctuation_targets()
    band = target["frequency_targets"]["high_quality_pinch_band_GHz"]
    band_low, band_high = float(band[0]), float(band[1])

    times = np.asarray(times_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    n = min(times.size, values.size)
    dominant_ghz, total_power = _dominant_frequency_ghz(times[:n], values[:n])
    band_power_fraction = 0.0
    if dominant_ghz is not None and total_power > 0.0 and n >= 4:
        dt = float(np.median(np.diff(times[:n])))
        centered = values[:n] - float(np.mean(values[:n]))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        freqs_ghz = np.fft.rfftfreq(n, dt) / 1.0e9
        band_mask = (freqs_ghz >= band_low) & (freqs_ghz <= band_high)
        band_power_fraction = float(np.sum(spectrum[band_mask]) / total_power)

    magnetic_passed = (
        dominant_ghz is not None
        and band_low <= dominant_ghz <= band_high
        and band_power_fraction >= min_band_power_fraction
    )
    band_center = 0.5 * (band_low + band_high)
    half_width = 0.5 * (band_high - band_low)
    relative_error = (
        abs(float(dominant_ghz) - band_center) / half_width
        if dominant_ghz is not None and half_width > 0.0 else math.inf
    )
    return {
        "passed": magnetic_passed,
        "diagnostics": {
            "magnetic_field": magnetic_passed,
        },
        "quantity_relative_errors": {
            "magnetic_field_rf_frequency": relative_error,
        },
        "target": "llnl_12kj_em_fluctuation",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 4,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "dominant_frequency_GHz": dominant_ghz,
            "target_band_GHz": [band_low, band_high],
            "band_power_fraction": band_power_fraction,
            "min_band_power_fraction": min_band_power_fraction,
            "missing_for_full_tier4": ["density", "temperature"],
        },
        "validity_notes": {
            "magnetic_proxy": (
                "The target is an EM fluctuation comparison, not a calibrated "
                "3D magnetic-field map."
            ),
            "tier4_limit": (
                "This evidence can cover the magnetic-field/EM fluctuation "
                "part of spatial validation only."
            ),
        },
    }


def uofsi_argon_temperature_targets() -> dict[str, object]:
    """Return KR-backed UofS-I argon DPF electron-temperature targets."""
    source = (
        "KnowledgeReference/a-thesis-submitted-to-the-college-of-graduate-"
        "and-postdoctoral-studies-in-partial-fulllment-of-the.md"
    )
    return {
        "target_id": "uofsi_argon_temperature_thesis_2020",
        "validation_scope": "uofsi_argon_temperature_thesis_2020",
        "device": "UofS-I DPF",
        "model_role": "kr_partial_xray_temperature_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "abstract_device_temperature": "98-116",
            "device_description": "1569-1599",
            "device_operation_parameters": "2057-2099",
            "plasma_dynamics": "2146-2173",
            "lee_fit_and_phase_context": "2216-2293",
            "temperature_measurement": "2464-2479",
            "conclusions": "2718-2767",
        },
        "shot_context": {
            "device": "UofS-I DPF",
            "device_type": "Mather",
            "stored_energy_kJ": 1.0,
            "capacitance_uF": 5.0,
            "charging_voltage_kV": 20.0,
            "power_supply_voltage_kV_max": 30.0,
            "working_gas": "argon",
            "optimum_pressure_mTorr_range": [100.0, 200.0],
            "lee_optimum_pressure_mTorr": 150.0,
        },
        "current_waveform_targets": {
            "measured_current_available": True,
            "current_diagnostic": "Pearson Rogowski coil",
            "lee_model_fit_to_current_waveform": True,
            "lee_mass_factor_axial": 0.046,
            "lee_current_factor_axial": 0.7,
            "lee_mass_factor_radial": 0.31,
            "lee_current_factor_radial": 0.7,
            "typical_current_dip_kA": 25.0,
            "typical_voltage_spike_kV": 20.0,
        },
        "phase_timing": {
            "axial_acceleration_duration_us": 1.15,
            "pinch_time_us_about": 1.3,
            "typical_current_rise_time_us": 1.2,
            "current_dip_marks_pinch": True,
            "voltage_spike_marks_pinch": True,
        },
        "temperature_targets": {
            "diagnostic": "soft x-ray double-filter ratio method",
            "argon_electron_temperature_average_keV": 5.7,
            "argon_electron_temperature_uncertainty_keV": 0.7,
            "argon_electron_temperature_range_keV": [5.5, 7.5],
            "table_temperature_min_keV": 4.35,
            "table_temperature_max_keV": 6.6,
            "assumed_electron_density_cm3": 1.0e18,
        },
        "diagnostic_requirements": {
            "soft_xray_pin_diode_filters": True,
            "hard_xray_detector": True,
            "electron_beam_charge_collector": True,
            "ion_beam_faraday_cup": True,
            "anode_voltage_probe": True,
        },
        "uncertainty": {
            "temperature_uncertainty_keV": 0.7,
            "electron_density_assumed_not_measured": True,
            "argon_line_emission_accuracy_not_quantitatively_assessed": True,
            "correlations_reported_as_linear_without_digitized_fit_tables": True,
        },
        "partial_target_groups": [
            "circuit_waveform",
            "phase_timing",
            "spatial_temperature",
            "uncertainty",
        ],
        "missing_for_full_tier1": [
            "digitized_current_trace_points",
            "per_point_current_uncertainty",
            "voltage_trace_digitization_and_uncertainty",
        ],
        "missing_for_full_tier2": [
            "digitized_phase_endpoint_trace",
            "shot_resolved_phase_timing_uncertainty",
        ],
        "missing_for_full_tier4": [
            "same_scope_density_measurement",
            "same_scope_magnetic_field_measurement",
            "time_resolved_temperature_profile",
        ],
        "validation_note": (
            "This target adds a direct soft-x-ray-filter electron-temperature "
            "measurement for a 1 kJ argon Mather DPF. It remains partial for "
            "end-to-end spatial validation because the thesis does not provide "
            "same-scope density and magnetic-field measurements or digitized "
            "waveform/temperature traces."
        ),
    }


def dpf_pinch_temperature_targets() -> dict[str, object]:
    """Return DPF pinch temperature-regime targets from the KR review."""
    source = (
        "KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-"
        "for-diverse-applications.md"
    )
    return {
        "target_id": "dpf_pinch_temperature_review_regime",
        "device": "generic DPF",
        "model_role": "kr_regime_target",
        "validation_tier": 4,
        "source": source,
        "source_lines": {
            "pinch_density_temperature": "354-362",
            "xray_temperature": "368-372",
            "density_magnetic_field": "374-377",
        },
        "temperature_targets": {
            "pinch_temperature_min_keV": 1.0,
            "reflected_shock_temperature_keV": 2.0,
            "thermal_xray_temperature_range_keV": [0.4, 4.0],
            "ion_temperature_nominal_keV": 1.0,
        },
        "context_targets": {
            "pinch_diameter_mm": [1.0, 2.0],
            "pinch_density_min_cm3": 1.0e19,
            "compressed_magnetic_field_min_T": 100.0,
        },
        "validation_note": (
            "This is a broad DPF regime target from a review, useful for "
            "temperature sanity checks. It is not a device-specific spatial "
            "validation dataset by itself."
        ),
    }


def dpf_pinch_temperature_evidence(
    *,
    ion_temperature_keV: float | None = None,
    electron_temperature_keV: float | None = None,
    xray_temperature_keV: float | None = None,
    target: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build temperature-regime evidence from simulated pinch temperatures."""
    target = target or dpf_pinch_temperature_targets()
    temp_targets = target["temperature_targets"]
    xray_low, xray_high = temp_targets["thermal_xray_temperature_range_keV"]
    values = {
        "ion_temperature_keV": ion_temperature_keV,
        "electron_temperature_keV": electron_temperature_keV,
        "xray_temperature_keV": xray_temperature_keV,
    }
    finite_values = {
        name: float(value)
        for name, value in values.items()
        if value is not None and math.isfinite(float(value))
    }
    passes = {
        name: xray_low <= value <= xray_high
        for name, value in finite_values.items()
    }
    temperature_passed = bool(passes) and any(passes.values())
    closest_error = math.inf
    band_center = 0.5 * (float(xray_low) + float(xray_high))
    half_width = 0.5 * (float(xray_high) - float(xray_low))
    for value in finite_values.values():
        closest_error = min(closest_error, abs(value - band_center) / half_width)

    return {
        "passed": temperature_passed,
        "diagnostics": {
            "temperature": temperature_passed,
        },
        "quantity_relative_errors": {
            "temperature_regime": closest_error,
        },
        "target": "dpf_pinch_temperature_regime",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_regime_target_comparison",
        "validation_tier": 4,
        "source": target["source"],
        "source_lines": target["source_lines"],
        "details": {
            "target_id": target["target_id"],
            "simulated_temperatures_keV": finite_values,
            "component_passes": passes,
            "target_temperature_range_keV": [float(xray_low), float(xray_high)],
            "missing_for_full_tier4": ["density", "magnetic_field"],
        },
        "validity_notes": {
            "regime_target": (
                "The KR source gives broad DPF pinch temperature ranges, not "
                "a device-specific calibrated temperature diagnostic."
            ),
            "tier4_limit": (
                "This evidence can cover temperature-regime consistency only."
            ),
        },
    }


def _sequence_from_history(history: object, name: str) -> Sequence[float]:
    if hasattr(history, name):
        value = getattr(history, name)
    elif isinstance(history, dict):
        value = history.get(name, [])
    else:
        value = []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return []


def _history_times_s(history: object) -> np.ndarray:
    times = _sequence_from_history(history, "times")
    if not times and isinstance(history, dict):
        times = _sequence_from_history(history, "times_s")
    if times:
        return np.asarray(times, dtype=float)
    if isinstance(history, dict):
        times_us = _sequence_from_history(history, "times_us")
        if times_us:
            return np.asarray(times_us, dtype=float) * 1.0e-6
    return np.asarray([], dtype=float)


def _rate_values(history: object, name: str) -> np.ndarray:
    values = _sequence_from_history(history, name)
    return np.asarray(values, dtype=float)


def _peak_time_s(times: np.ndarray, values: np.ndarray) -> float | None:
    if times.size == 0 or values.size == 0:
        return None
    n = min(times.size, values.size)
    if n == 0:
        return None
    values = values[:n]
    if not np.any(np.isfinite(values)) or float(np.nanmax(values)) <= 0.0:
        return None
    return float(times[:n][int(np.nanargmax(values))])


def _local_peak_times_s(
    times: np.ndarray,
    values: np.ndarray,
    *,
    threshold_fraction: float,
    min_separation_ns: float,
) -> list[float]:
    n = min(times.size, values.size)
    if n < 3:
        peak = _peak_time_s(times, values)
        return [] if peak is None else [peak]

    times = times[:n]
    values = values[:n]
    if not np.any(np.isfinite(values)) or float(np.nanmax(values)) <= 0.0:
        return []

    smoothed = np.array(values, dtype=float)
    smoothed[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3.0
    threshold = threshold_fraction * float(np.nanmax(smoothed))

    candidates: list[int] = []
    for idx in range(1, n - 1):
        if (
            smoothed[idx] > smoothed[idx - 1]
            and smoothed[idx] >= smoothed[idx + 1]
            and smoothed[idx] >= threshold
        ):
            candidates.append(idx)

    if not candidates:
        peak = _peak_time_s(times, values)
        return [] if peak is None else [peak]

    min_separation_s = min_separation_ns * 1.0e-9
    peaks: list[int] = []
    for idx in candidates:
        if peaks and times[idx] - times[peaks[-1]] < min_separation_s:
            if smoothed[idx] > smoothed[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)
    return [float(times[idx]) for idx in peaks]


def mjolnir_neutron_timing_evidence_from_history(
    history: object,
    *,
    stagnation_time_s: float | None = None,
    target: dict[str, object] | None = None,
    thermonuclear_tolerance_ns: float = 2.0,
    beam_target_tolerance_ns: float = 2.5,
    beam_target_reference_ns: float = 5.0,
    require_measurement_correlation: bool = False,
    measurement_correlation_tolerance_ns: float = 3.0,
    peak_threshold_fraction: float = 0.1,
    min_peak_separation_ns: float = 3.0,
) -> dict[str, object]:
    """Compare a mechanism-separated neutron history with the MJOLNIR target.

    The input can be a ``YieldResult`` or a ``yield_time_resolved`` dictionary.
    This helper does not infer that a run is validated from the target alone;
    it requires simulated thermonuclear and beam-target time histories.
    """
    target = target or mjolnir_neutron_timing_targets()
    times = _history_times_s(history)
    thermo = _rate_values(history, "dY_thermo")
    beam = _rate_values(history, "dY_bt")

    thermo_peak = _peak_time_s(times, thermo)
    inferred_stagnation = False
    if stagnation_time_s is None and thermo_peak is not None:
        stagnation_time_s = thermo_peak
        inferred_stagnation = True

    mechanism_passes = {"thermonuclear": False, "beam_target": False}
    details: dict[str, object] = {
        "target_id": target["target_id"],
        "source": target["source"],
        "source_lines": target["source_lines"],
        "stagnation_time_s": stagnation_time_s,
        "stagnation_time_inferred_from_thermonuclear_peak": inferred_stagnation,
        "beam_target_reference_ns": beam_target_reference_ns,
        "require_measurement_correlation": require_measurement_correlation,
    }

    if thermo_peak is not None and stagnation_time_s is not None:
        thermo_relative_ns = (thermo_peak - stagnation_time_s) * 1.0e9
        thermo_error_ns = abs(thermo_relative_ns)
        mechanism_passes["thermonuclear"] = (
            thermo_error_ns <= thermonuclear_tolerance_ns
        )
        details["thermonuclear"] = {
            "peak_time_s": thermo_peak,
            "relative_time_ns": thermo_relative_ns,
            "target_relative_time_ns": 0.0,
            "absolute_error_ns": thermo_error_ns,
            "tolerance_ns": thermonuclear_tolerance_ns,
            "passed": mechanism_passes["thermonuclear"],
        }
    else:
        details["thermonuclear"] = {
            "passed": False,
            "limitation": "missing thermonuclear neutron history or stagnation time",
        }

    beam_peaks = _local_peak_times_s(
        times,
        beam,
        threshold_fraction=peak_threshold_fraction,
        min_separation_ns=min_peak_separation_ns,
    )
    if beam_peaks and stagnation_time_s is not None:
        relative_beam_peaks_ns = [
            (peak - stagnation_time_s) * 1.0e9
            for peak in beam_peaks
        ]
        beam_errors = [
            abs(relative_ns - beam_target_reference_ns)
            for relative_ns in relative_beam_peaks_ns
        ]
        best_idx = int(np.argmin(beam_errors))
        beam_passed = beam_errors[best_idx] <= beam_target_tolerance_ns

        measurement_passed = True
        measurement_error_ns = None
        if require_measurement_correlation:
            measurement_errors = [
                abs(relative_ns - 10.0)
                for relative_ns in relative_beam_peaks_ns
            ]
            measurement_error_ns = min(measurement_errors)
            measurement_passed = (
                measurement_error_ns <= measurement_correlation_tolerance_ns
            )

        mechanism_passes["beam_target"] = beam_passed and measurement_passed
        details["beam_target"] = {
            "peak_times_s": beam_peaks,
            "relative_peak_times_ns": relative_beam_peaks_ns,
            "target_relative_time_ns": beam_target_reference_ns,
            "best_absolute_error_ns": beam_errors[best_idx],
            "tolerance_ns": beam_target_tolerance_ns,
            "measurement_correlation_error_ns": measurement_error_ns,
            "measurement_correlation_tolerance_ns": (
                measurement_correlation_tolerance_ns
                if require_measurement_correlation else None
            ),
            "passed": mechanism_passes["beam_target"],
        }
    else:
        details["beam_target"] = {
            "passed": False,
            "limitation": "missing beam-target neutron history or stagnation time",
        }

    passed = all(mechanism_passes.values())
    timing_relative_errors = {}
    for mechanism, detail in details.items():
        if not isinstance(detail, dict):
            continue
        error = detail.get("absolute_error_ns") or detail.get("best_absolute_error_ns")
        tolerance = detail.get("tolerance_ns")
        if error is not None and tolerance:
            timing_relative_errors[mechanism] = float(error) / float(tolerance)

    return {
        "passed": passed,
        "mechanisms": mechanism_passes,
        "timing_relative_errors": timing_relative_errors,
        "target": "mjolnir_neutron_timing",
        "validation_scope": target["target_id"],
        "model_role": "simulation_to_kr_target_comparison",
        "validation_tier": 5,
        "source": target["source"],
        "details": details,
        "validity_notes": {
            "history_requirement": (
                "Requires mechanism-separated neutron histories; scalar yield "
                "totals are insufficient for this KR timing validation."
            ),
            "inferred_stagnation": (
                "If no independent stagnation time is supplied, the "
                "thermonuclear peak is used as the stagnation marker and the "
                "evidence remains weaker than a diagnostic-timed comparison."
            ),
        },
    }


_KR_TARGET_FACTORIES = (
    lee_snowplow_phase_semantics_targets,
    lee_course_nx2_neon_phase_timing_example_targets,
    lee_radpf_theory_model_scope_targets,
    lee_2014_radiative_model_review_targets,
    pf1000_16kv_current_waveform_targets,
    pf1000_16kv_shot12581_phase_targets,
    pf1000_16kv_akel_table_targets,
    pf1000_full_energy_phase_context_targets,
    pf1000_full_energy_neutron_spatial_targets,
    pf1000_cikhardtova_linear_density_motion_targets,
    pf1000_szydlowski_fast_ion_neutron_targets,
    pf1000_krasa_vessel_scatter_anisotropy_targets,
    klir_2011_tof_detector_response_targets,
    nx3_springham_zrbe_activation_targets,
    nnss_dpf_neutron_time_energy_tomography_targets,
    deuterium_argon_admixture_neutron_targets,
    ff1_focus_fusion_plasmoid_targets,
    lee_drive_parameter_speed_enhancement_targets,
    rawat_dpf_operating_envelope_targets,
    auluck_gpf_scaling_theory_targets,
    auluck_neutron_yield_scaling_failure_targets,
    auluck_circuit_element_poynting_targets,
    blagoev_electric_flux_diagnostic_targets,
    auluck_poloidal_magnetic_field_targets,
    wante_nitrogen_ion_irradiation_targets,
    demina_dpf_material_damage_targets,
    altarabulsi_deuteron_beam_fluence_targets,
    kiai_double_dpf_icf_concept_targets,
    wang_metallic_vapor_interferometry_targets,
    pfz200_hybrid_xpinch_proton_neutron_targets,
    llnl_fully_kinetic_dpf_targets,
    nstec_3d_mhd_rundown_targets,
    alegra_hedp_dpf_mhd_validation_targets,
    esaulov_2d_mhrdr_dpf_targets,
    ou_foi_2d_dpf_simulation_targets,
    sun_two_temperature_mhd_motion_targets,
    beresnyak_hawk_3d_mhd_targets,
    narkis_kr_doped_dpf_mhd_targets,
    faeton_i_high_voltage_dpf_targets,
    mjolnir_high_low_parasitic_current_targets,
    mjolnir_first_experiments_targets,
    pf400j_xray_inference_targets,
    mjolnir_neutron_timing_targets,
    mjolnir_stagnation_temperature_targets,
    mjolnir_neutron_detector_response_targets,
    mcalpine_dpf_nrta_mcnp_targets,
    pf1000_spatial_pinch_targets,
    pf1000_interferometry_density_targets,
    llnl_12kj_em_fluctuation_targets,
    uofsi_argon_temperature_targets,
    dpf_pinch_temperature_targets,
)

_END_TO_END_TARGET_GROUPS = (
    "circuit_waveform",
    "phase_semantics",
    "phase_timing",
    "spatial_density",
    "spatial_magnetic_or_em",
    "spatial_temperature",
    "neutron_yield",
    "neutron_timing",
    "neutron_spectrum",
    "neutron_anisotropy",
    "neutron_detector_response",
    "uncertainty",
)

_TARGET_SEMANTIC_MARKERS = {
    "lee_radpf_phase_semantics_course": ("radial phase", "current dip"),
    "lee_course_nx2_neon_phase_timing_example": (
        "axial phase ends",
        "radial phase ends",
        "pinch phase",
    ),
    "lee_radpf_theory_model_scope_2008": (
        "generating equations",
        "slug model",
        "beam-target",
        "radiation collapse",
    ),
    "lee_2014_radiative_model_review": (
        "j fusion energy",
        "phase 4a",
        "bremsstrahlung",
        "1.6 ma",
    ),
    "pf1000_16kv_current_waveform_2021_akel": (
        "measured current",
        "current wave",
    ),
    "pf1000_16kv_shot12581_phase_2021_akel": (
        "current dip",
        "pinch duration",
    ),
    "pf1000_16kv_shot_table_2021_akel": (
        "computed and measured neutron yields",
        "ipeak",
        "pinch density",
        "shot-to-shot variation",
    ),
    "pf1000_full_energy_phase_context_2007_gribkov": (
        "compression",
        "current dip",
        "neutron pulse",
    ),
    "pf1000_full_energy_neutron_spatial_2007_scholz": (
        "rogowski",
        "2.45 mev",
        "temperature",
        "bubble",
    ),
    "pf1000_linear_density_motion_2015_cikhardtova": (
        "linear density",
        "527 nm",
        "zippering",
        "timing of different",
    ),
    "pf1000_fast_ion_neutron_2004_szydlowski": (
        "silver activation",
        "2.2",
        "cr-39",
        "anisotropy",
    ),
    "pf1000_vessel_scatter_anisotropy_2008_krasa": (
        "stainless steel",
        "10 mm",
        "scattered",
        "tof",
    ),
    "tof_detector_response_2011_klir": (
        "bc-408",
        "5.7",
        "2.45 mev",
        "pmt delay",
    ),
    "nx3_zrbe_activation_2021_springham": (
        "zr/be",
        "2.8 mev",
        "anisotropy",
        "straightforward beam",
    ),
    "nnss_dpf_neutron_time_energy_tomography_2020_catenacci": (
        "shadow bar",
        "energy spectrum",
        "100 kev",
        "scatter",
    ),
    "deuterium_argon_admixture_neutron_2026_omar": (
        "argon",
        "indium",
        "neutron yield",
        "standard deviation",
    ),
    "ff1_focus_fusion_plasmoid_2023_lerner": (
        "p-b11",
        "plasmoid",
        "neutrons",
        "impurity",
    ),
    "lee_drive_parameter_speed_enhancement_2003": (
        "drive parameter",
        "neutron yield",
        "standard deviation",
    ),
    "rawat_dpf_operating_envelope_2015": (
        "high energy density",
        "current sheath",
        "few mbar",
        "shot to shot",
    ),
    "auluck_gpf_scaling_theory_2023": (
        "generalized plasma focus",
        "scaling theory",
        "neutron yield scaling",
        "43 millibar",
        "200 t",
    ),
    "auluck_neutron_yield_scaling_failure_2023": (
        "failure of neutron yield scaling",
        "inverse fifth power",
        "lift-off time",
        "0.4",
    ),
    "auluck_circuit_element_poynting_2021": (
        "poynting",
        "anomalous impedance",
        "3-d magnetic field",
        "current derivative",
    ),
    "blagoev_electric_flux_diagnostic_2025": (
        "electric flux",
        "d-dot",
        "3 kj",
        "singularity",
    ),
    "auluck_poloidal_magnetic_field_2024": (
        "poloidal magnetic field",
        "geomagnetic",
        "helmholtz",
        "nikulin",
    ),
    "wante_nitrogen_ion_irradiation_2025": (
        "unu/ictp",
        "faraday cup",
        "72.40 kev",
        "1.5 mbar",
    ),
    "demina_dpf_material_damage_apdm4": (
        "pf-1000",
        "470 pa",
        "microcracks",
        "cfc",
    ),
    "altarabulsi_deuteron_beam_fluence_2024": (
        "deuteron beam fluence",
        "radpfv6.16fib",
        "mpef-12",
        "7.05",
    ),
    "kiai_double_dpf_icf_concept_2025": (
        "double-dpf",
        "high-temperature superconducting",
        "30 kj",
        "75 mw",
    ),
    "wang_metallic_vapor_interferometry_1999": (
        "metallic vapor",
        "laser differential interferometer",
        "dpf-16",
        "280 ns",
    ),
    "pfz200_hybrid_xpinch_proton_neutron_2026_novotny": (
        "hybrid x-pinch",
        "cr-39",
        "neutron production",
        "rogowski",
    ),
    "llnl_fully_kinetic_dpf_2012_schmidt": (
        "fully kinetic",
        "neutron yield",
        "lower hybrid",
        "current dips",
    ),
    "nstec_3d_mhd_rundown_2014_meehan": (
        "faraday",
        "rundown",
        "2.17 ma",
        "lower density",
    ),
    "alegra_hedp_dpf_mhd_validation_2009_kueny": (
        "alegra-hedp",
        "mhd",
        "qeos",
        "tallboy",
    ),
    "esaulov_2d_mhrdr_dpf_2003": (
        "mhrdr",
        "maxwell average",
        "begay",
        "snowplow",
    ),
    "ou_foi_2d_dpf_simulation_2024": (
        "foi",
        "llnl",
        "2.5",
        "188.99",
    ),
    "sun_two_temperature_mhd_motion_2025": (
        "two-temperature",
        "braginskii",
        "90",
        "neutron",
    ),
    "beresnyak_hawk_3d_mhd_2018": (
        "hawk",
        "athena",
        "640-kv",
        "200 kev",
    ),
    "narkis_kr_doped_dpf_mhd_2021": (
        "hydra",
        "1.0% kr",
        "12.6kev",
        "beam-target",
    ),
    "faeton_i_high_voltage_dpf_2025": (
        "faeton-i",
        "fcr",
        "faraday cup",
        "anisotropy",
    ),
    "mjolnir_high_low_parasitic_current_2022_goyon": (
        "parasitic current",
        "rogowski",
        "current dip",
        "4.1",
    ),
    "mjolnir_first_experiments_2021_offermann": (
        "mjolnir",
        "rogowski",
        "3.8",
    ),
    "pf400j_xray_inference_2020_orellana": (
        "x rays",
        "bc-408",
        "rogowski",
        "959",
    ),
    "mjolnir_neutron_timing_2025_goyon": (
        "thermonuclear",
        "beam",
    ),
    "mjolnir_stagnation_temperature_2025_goyon": (
        "temperature",
        "kev",
        "mjolnir",
    ),
    "mjolnir_neutron_detector_response_2025_goyon": (
        "activation",
        "scintillator",
        "response",
    ),
    "mcalpine_dpf_nrta_mcnp_2014": (
        "mcnp",
        "2.45 mev",
        "3 cm",
        "single pulse",
    ),
    "pf1000_spatial_pinch_2006_scholz": (
        "camera",
        "pinch",
    ),
    "pf1000_interferometry_density_2024_malir": (
        "density",
        "interferometer",
    ),
    "llnl_12kj_em_fluctuation_2014_schmidt": (
        "probe",
        "ghz",
    ),
    "uofsi_argon_temperature_thesis_2020": (
        "5.7",
        "rogowski",
        "argon",
    ),
    "dpf_pinch_temperature_review_regime": (
        "temperature",
        "kev",
    ),
}


def _flatten_source_lines(
    source_lines: object,
    *,
    prefix: str = "",
) -> list[dict[str, str]]:
    if isinstance(source_lines, str):
        line_range = source_lines.strip()
        return (
            [{"source_line_key": prefix, "source_lines": line_range}]
            if line_range else []
        )
    if isinstance(source_lines, Mapping):
        items: list[dict[str, str]] = []
        for key, value in source_lines.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_source_lines(value, prefix=child_prefix))
        return items
    if isinstance(source_lines, Sequence) and not isinstance(
        source_lines,
        (str, bytes, bytearray),
    ):
        items = []
        for idx, value in enumerate(source_lines):
            child_prefix = f"{prefix}[{idx}]" if prefix else str(idx)
            items.extend(_flatten_source_lines(value, prefix=child_prefix))
        return items
    return []


def _resolve_kr_source(source: str) -> Path | None:
    if not source.startswith("KnowledgeReference/"):
        return None
    path = Path(source)
    if path.is_absolute():
        return path if path.is_file() else None
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return None


def _line_ranges_from_items(
    line_items: Sequence[Mapping[str, str]],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in line_items:
        for part in str(item.get("source_lines", "")).split(","):
            tokens = part.strip().split("-", 1)
            if not tokens or not tokens[0].strip().isdigit():
                continue
            start = int(tokens[0])
            end = int(tokens[1]) if len(tokens) > 1 and tokens[1].strip().isdigit() else start
            if start <= end:
                ranges.append((start, end))
    return ranges


def _source_excerpt_for_line_items(
    source: str,
    line_items: Sequence[Mapping[str, str]],
) -> str:
    path = _resolve_kr_source(source)
    if path is None:
        return ""
    ranges = _line_ranges_from_items(line_items)
    if not ranges:
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpts: list[str] = []
    for start, end in ranges:
        lo = max(start - 1, 0)
        hi = min(end, len(lines))
        if lo < hi:
            excerpts.extend(lines[lo:hi])
    return "\n".join(excerpts).lower()


def kr_validation_target_manifest() -> list[dict[str, object]]:
    """Return all currently coded KR validation targets with source metadata."""
    manifest: list[dict[str, object]] = []
    for factory in _KR_TARGET_FACTORIES:
        target = factory()
        line_items = _flatten_source_lines(target.get("source_lines"))
        manifest.append({
            "target_id": target.get("target_id", ""),
            "device": target.get("device", ""),
            "validation_scope": target.get(
                "validation_scope",
                target.get("target_id", ""),
            ),
            "validation_tier": target.get("validation_tier", ""),
            "model_role": target.get("model_role", ""),
            "source": target.get("source", ""),
            "source_line_items": line_items,
            "n_source_line_items": len(line_items),
        })
    return manifest


def kr_validation_target_semantic_audit() -> dict[str, object]:
    """Check that target line windows contain expected domain markers."""
    target_records: list[dict[str, object]] = []
    for target in kr_validation_target_manifest():
        target_id = str(target.get("target_id", ""))
        markers = _TARGET_SEMANTIC_MARKERS.get(target_id, ())
        line_items = [
            item for item in target.get("source_line_items", [])
            if isinstance(item, Mapping)
        ]
        excerpt = _source_excerpt_for_line_items(
            str(target.get("source", "")),
            line_items,
        )
        missing = [
            marker for marker in markers
            if marker.lower() not in excerpt
        ]
        target_records.append({
            "target_id": target_id,
            "source": target.get("source", ""),
            "markers": list(markers),
            "missing_markers": missing,
            "passed": bool(markers) and not missing,
        })

    failed = [
        str(record["target_id"])
        for record in target_records
        if record.get("passed") is not True
    ]
    return {
        "passed": not failed and bool(target_records),
        "model_role": "kr_validation_target_semantic_audit",
        "targets": target_records,
        "missing_or_failed_targets": failed,
        "validity_notes": {
            "scope": (
                "This is a lightweight guard that cited KR line windows contain "
                "expected domain terms. It is not a replacement for human "
                "scientific review of the extracted target values."
            ),
        },
    }


def kr_validation_target_source_audit() -> dict[str, object]:
    """Audit local source authority for every coded KR validation target."""
    from dpf.validation.quality_assessment import source_authority_evidence

    targets: list[dict[str, object]] = []
    for target in kr_validation_target_manifest():
        source = str(target.get("source", ""))
        line_items = target.get("source_line_items", [])
        line_ranges = [
            str(item.get("source_lines", ""))
            for item in line_items
            if isinstance(item, Mapping)
        ]
        evidence = source_authority_evidence(
            validation_scope=str(target.get("validation_scope", "")),
            sources=[source for _ in line_ranges],
            source_lines=line_ranges,
            provenance="kr_extracted",
        )
        targets.append({
            **target,
            "source_authority_passed": evidence["passed"],
            "source_authority": evidence,
        })

    missing = [
        str(target.get("target_id", ""))
        for target in targets
        if target.get("source_authority_passed") is not True
    ]
    return {
        "passed": not missing and bool(targets),
        "model_role": "kr_validation_target_source_audit",
        "targets": targets,
        "missing_or_invalid_targets": missing,
        "validity_notes": {
            "scope": (
                "This audit proves only that coded KR targets cite existing "
                "local files and valid line ranges. It does not prove that the "
                "line contents semantically support each extracted target."
            ),
        },
    }


def _typed_observable_groups(target: Mapping[str, object]) -> set[str]:
    groups: set[str] = set()
    if "current_waveform_targets" in target:
        groups.add("circuit_waveform")
    if "phase_semantics" in target:
        groups.add("phase_semantics")
    if "phase_timing" in target:
        groups.add("phase_timing")
    if (
        "density_profile_targets" in target
        or "density_proxy_diagnostic" in target
        or "spatial_density_targets" in target
    ):
        groups.add("spatial_density")
    if (
        "frequency_targets" in target
        or "field_context" in target
        or "magnetic_field_targets" in target
        or "electric_flux_diagnostic_targets" in target
    ):
        groups.add("spatial_magnetic_or_em")
    if "temperature_targets" in target:
        groups.add("spatial_temperature")
    if "neutron_yield_targets" in target:
        groups.add("neutron_yield")
    if "event_sequence" in target or "detector_tof" in target:
        groups.add("neutron_timing")
    if "spectral_targets" in target:
        groups.add("neutron_spectrum")
    if "anisotropy_targets" in target:
        groups.add("neutron_anisotropy")
    if (
        "activation_requirements" in target
        or "tof_requirements" in target
        or "response_model_requirements" in target
    ):
        groups.add("neutron_detector_response")
    if "uncertainty" in target:
        groups.add("uncertainty")
    return groups


def _partial_groups_for_target(target: Mapping[str, object]) -> set[str]:
    partial: set[str] = set()
    explicit_partial = target.get("partial_target_groups", ())
    if isinstance(explicit_partial, Sequence) and not isinstance(
        explicit_partial,
        (str, bytes, bytearray),
    ):
        partial.update(str(group) for group in explicit_partial)
    if target.get("missing_for_full_tier1"):
        partial.add("circuit_waveform")
    if (
        target.get("missing_for_full_tier2")
        or target.get("missing_for_predictive_tier2")
    ):
        partial.add("phase_timing")
    if target.get("missing_for_predictive_neutron_yield_validation"):
        partial.add("neutron_yield")
    if target.get("missing_for_full_tier4") and "temperature_targets" in target:
        partial.add("spatial_temperature")
    if target.get("missing_for_full_tier5"):
        groups = _typed_observable_groups(target)
        partial.update(
            group for group in groups
            if group.startswith("neutron_")
        )
    if str(target.get("device", "")).strip().lower() == "generic dpf":
        partial.add("spatial_temperature")
    return partial


def _target_scope_blocker_items(target: Mapping[str, object], group: str) -> list[str]:
    blockers: list[str] = []

    def add_items(value: object) -> None:
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            blockers.extend(str(item) for item in value if str(item).strip())

    if group == "circuit_waveform":
        add_items(target.get("missing_for_full_tier1"))
    if group == "phase_timing":
        add_items(target.get("missing_for_full_tier2"))
        add_items(target.get("missing_for_predictive_tier2"))

    tier4_items: list[str] = []
    value = target.get("missing_for_full_tier4")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tier4_items = [str(item) for item in value if str(item).strip()]
    if group == "spatial_density":
        blockers.extend(item for item in tier4_items if "density" in item.lower())
    if group == "spatial_magnetic_or_em":
        blockers.extend(
            item for item in tier4_items
            if "magnetic" in item.lower() or "field" in item.lower()
        )
    if group == "spatial_temperature":
        blockers.extend(item for item in tier4_items if "temperature" in item.lower())

    tier5_items: list[str] = []
    value = target.get("missing_for_full_tier5")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tier5_items = [str(item) for item in value if str(item).strip()]
    if group == "neutron_timing":
        blockers.extend(
            item for item in tier5_items
            if "pulse" in item.lower()
            or "history" in item.lower()
            or "timing" in item.lower()
                or "trace" in item.lower()
        )
    if group == "neutron_yield":
        add_items(target.get("missing_for_predictive_neutron_yield_validation"))
        blockers.extend(
            item for item in tier5_items
            if "yield" in item.lower()
            or "calibration" in item.lower()
            or "activation" in item.lower()
            or "response" in item.lower()
            or "uncertainty" in item.lower()
        )
    if group == "neutron_spectrum":
        blockers.extend(
            item for item in tier5_items
            if "spectrum" in item.lower() or "spectra" in item.lower()
        )
    if group == "neutron_anisotropy":
        blockers.extend(
            item for item in tier5_items
            if "fast_ion" in item.lower() or "anisotropy" in item.lower()
        )
    if group == "neutron_detector_response":
        blockers.extend(
            item for item in tier5_items
            if "response" in item.lower()
            or "transport" in item.lower()
            or "scatter" in item.lower()
            or "detector" in item.lower()
        )
    if group == "uncertainty":
        uncertainty = target.get("uncertainty")
        if isinstance(uncertainty, Mapping):
            add_items(uncertainty.get("missing_uncertainty_components"))
        for item in (
            list(blockers)
            + tier4_items
            + tier5_items
        ):
            if "uncertainty" in item.lower():
                blockers.append(item)

    return sorted(dict.fromkeys(blockers))


def _scope_closure_blocker_records(
    scope_targets: Sequence[Mapping[str, object]],
    present_groups: set[str],
    missing_groups: Sequence[str],
    partial_groups: set[str],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for group in missing_groups:
        records[group] = {
            "group": group,
            "status": "missing_group",
            "data_availability": "absent_from_same_scope_targets",
            "target_ids": [],
            "sources": [],
            "missing_items": [f"same_scope_{group}_target"],
            "required_data_to_complete": [f"same_scope_{group}_target"],
            "reason": "No KR target in this validation scope covers this group.",
        }

    for group in sorted(partial_groups):
        record = records.setdefault(
            group,
            {
                "group": group,
                "status": "partial_group",
                "data_availability": "partial_only_in_same_scope_targets",
                "target_ids": [],
                "sources": [],
                "missing_items": [],
                "required_data_to_complete": [],
                "reason": (
                    "KR target exists in this validation scope, but extraction "
                    "or the source itself lacks the observables needed for "
                    "full predictive validation."
                ),
            },
        )
        for target in scope_targets:
            if group not in _typed_observable_groups(target):
                continue
            target_ids = record["target_ids"]
            if isinstance(target_ids, list):
                target_ids.append(str(target.get("target_id", "")))
            sources = record["sources"]
            if isinstance(sources, list):
                sources.append(str(target.get("source", "")))
            missing_items = record["missing_items"]
            if isinstance(missing_items, list):
                missing_items.extend(_target_scope_blocker_items(target, group))

        for key in ("target_ids", "sources", "missing_items"):
            values = record.get(key, [])
            if isinstance(values, list):
                record[key] = sorted(dict.fromkeys(value for value in values if value))
        if not record.get("missing_items"):
            record["missing_items"] = [f"complete_same_scope_{group}_evidence"]
        record["required_data_to_complete"] = list(record["missing_items"])

    return records


def kr_validation_target_coverage_report() -> dict[str, object]:
    """Report typed end-to-end observable coverage in coded KR targets."""
    group_records = {
        group: {
            "group": group,
            "target_ids": [],
            "status": "missing",
            "limitations": [],
        }
        for group in _END_TO_END_TARGET_GROUPS
    }

    for factory in _KR_TARGET_FACTORIES:
        target = factory()
        target_id = str(target.get("target_id", ""))
        for group in _typed_observable_groups(target):
            if group not in group_records:
                continue
            record = group_records[group]
            target_ids = record["target_ids"]
            if isinstance(target_ids, list):
                target_ids.append(target_id)
            record["status"] = "present"
            if group == "phase_timing" and target.get("missing_for_full_tier2"):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "phase timing target does not include all axial, radial, "
                        "and pinch timing observables required for full tier 2"
                    )
            if group == "phase_timing" and target.get("missing_for_predictive_tier2"):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "phase timing example lacks same-shot deuterium "
                        "experimental targets and uncertainty"
                    )
            if group == "circuit_waveform" and target.get("missing_for_full_tier1"):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "current waveform target lacks digitized trace points "
                        "and per-point uncertainty needed for full tier 1"
                    )
            if group == "spatial_temperature" and target.get("missing_for_full_tier4"):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "temperature target lacks direct experimental diagnostic, "
                        "uncertainty, or same-scope density and magnetic-field targets"
                    )
            if (
                group in _partial_groups_for_target(target)
                and group not in {
                    "phase_timing",
                    "circuit_waveform",
                    "spatial_temperature",
                }
            ):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "target is explicitly marked partial by KR extraction "
                        "limitations"
                    )
            if (
                group in {"spatial_density", "spatial_magnetic_or_em", "spatial_temperature"}
                and str(target.get("device", "")).strip().lower() == "generic dpf"
            ):
                record["status"] = "partial"
                limitations = record["limitations"]
                if isinstance(limitations, list):
                    limitations.append(
                        "generic DPF regime target is not a same-device spatial "
                        "validation target"
                    )

    missing_or_partial = [
        group for group, record in group_records.items()
        if record["status"] != "present"
    ]
    return {
        "passed": not missing_or_partial,
        "model_role": "kr_validation_target_coverage_report",
        "required_groups": list(_END_TO_END_TARGET_GROUPS),
        "groups": list(group_records.values()),
        "missing_or_partial_groups": missing_or_partial,
        "validity_notes": {
            "scope": (
                "This report tracks whether typed KR targets exist for the "
                "end-to-end validation plan. It does not claim same-scope "
                "closure or simulation agreement."
            ),
        },
    }


def kr_validation_same_scope_target_report() -> dict[str, object]:
    """Report whether any one KR validation scope covers all target groups."""
    scopes: dict[str, dict[str, object]] = {}
    for factory in _KR_TARGET_FACTORIES:
        target = factory()
        validation_scope = str(
            target.get("validation_scope") or target.get("target_id") or ""
        )
        if not validation_scope:
            continue
        record = scopes.setdefault(
            validation_scope,
            {
                "validation_scope": validation_scope,
                "device": target.get("device", ""),
                "target_ids": [],
                "targets": [],
                "present_groups": set(),
                "partial_groups": set(),
            },
        )
        target_ids = record["target_ids"]
        targets = record["targets"]
        present_groups = record["present_groups"]
        partial_groups = record["partial_groups"]
        if isinstance(target_ids, list):
            target_ids.append(target.get("target_id", ""))
        if isinstance(targets, list):
            targets.append(target)
        if isinstance(present_groups, set):
            present_groups.update(_typed_observable_groups(target))
        if isinstance(partial_groups, set):
            partial_groups.update(_partial_groups_for_target(target))

    scope_records: list[dict[str, object]] = []
    for scope, record in scopes.items():
        present_groups = set(record.get("present_groups", set()))
        partial_groups = set(record.get("partial_groups", set())) & present_groups
        missing_groups = [
            group for group in _END_TO_END_TARGET_GROUPS if group not in present_groups
        ]
        passed = not missing_groups and not partial_groups
        target_records = [
            target for target in record.get("targets", [])
            if isinstance(target, Mapping)
        ]
        closure_blockers = _scope_closure_blocker_records(
            target_records,
            present_groups,
            missing_groups,
            partial_groups,
        )
        scope_records.append({
            "validation_scope": scope,
            "device": record.get("device", ""),
            "target_ids": list(record.get("target_ids", [])),
            "present_groups": sorted(present_groups),
            "missing_groups": missing_groups,
            "partial_groups": sorted(partial_groups),
            "closure_blockers": closure_blockers,
            "closure_blocker_groups": sorted(closure_blockers),
            "passed": passed,
            "status": "complete" if passed else "incomplete",
        })

    scope_records.sort(
        key=lambda item: (
            len(item["missing_groups"]) + len(item["partial_groups"]),  # type: ignore[arg-type]
            str(item["validation_scope"]),
        )
    )
    passed_scopes = [
        str(record["validation_scope"])
        for record in scope_records
        if record.get("passed") is True
    ]
    widest_scope_records = sorted(
        scope_records,
        key=lambda item: (
            len(item["missing_groups"]),  # type: ignore[arg-type]
            len(item["partial_groups"]),  # type: ignore[arg-type]
            str(item["validation_scope"]),
        ),
    )
    widest_available_scope = (
        widest_scope_records[0] if widest_scope_records else None
    )
    if passed_scopes:
        next_same_scope_steps = [
            "At least one same-scope KR validation packet is complete; use it "
            "for simulation-to-target comparison before making predictive claims.",
        ]
    elif isinstance(widest_available_scope, Mapping) and not widest_available_scope.get(
        "missing_groups",
    ):
        next_same_scope_steps = [
            "Use the widest same-scope packet as the closure path because it "
            "has all required groups present.",
            "Replace partial target groups with complete KR-backed evidence: "
            f"{', '.join(str(group) for group in widest_available_scope.get('partial_groups', []))}.",
            "Use the scope closure_blockers records as the exact extraction or "
            "KR-absence checklist for each partial group.",
            "Keep predictive readiness blocked for this scope until those "
            "partial groups have digitized traces, uncertainty, and same-shot "
            "diagnostic support.",
        ]
    else:
        next_same_scope_steps = [
            "No same-scope packet has all required groups present; add missing "
            "KR-backed target groups for the best available device/scope.",
            "Keep predictive readiness blocked until one device/shot/scope has "
            "complete circuit, phase, spatial, neutron, and uncertainty evidence.",
        ]
    return {
        "passed": bool(passed_scopes),
        "model_role": "kr_validation_same_scope_target_report",
        "required_groups": list(_END_TO_END_TARGET_GROUPS),
        "scopes": scope_records,
        "passed_scopes": passed_scopes,
        "best_available_scope": scope_records[0] if scope_records else None,
        "widest_available_scope": widest_available_scope,
        "next_same_scope_steps": next_same_scope_steps,
        "validity_notes": {
            "scope": (
                "A predictive end-to-end run needs targets from one compatible "
                "device, shot, or validation scope. Cross-device target "
                "coverage is insufficient."
            ),
        },
    }
