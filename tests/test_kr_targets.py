"""Tests for KnowledgeReference-backed validation targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dpf.validation import (
    akel_fig1_draft_digitization_packet,
    alegra_hedp_dpf_mhd_validation_targets,
    altarabulsi_deuteron_beam_fluence_targets,
    auluck_circuit_element_poynting_targets,
    auluck_gpf_scaling_theory_targets,
    auluck_neutron_yield_scaling_failure_targets,
    auluck_poloidal_magnetic_field_targets,
    beresnyak_hawk_3d_mhd_targets,
    blagoev_electric_flux_diagnostic_targets,
    deuterium_argon_admixture_neutron_targets,
    demina_dpf_material_damage_targets,
    dpf_pinch_temperature_evidence,
    dpf_pinch_temperature_targets,
    esaulov_2d_mhrdr_dpf_targets,
    faeton_i_high_voltage_dpf_targets,
    ff1_focus_fusion_plasmoid_targets,
    kr_validation_same_scope_target_report,
    kr_validation_target_coverage_report,
    kr_validation_target_manifest,
    kr_validation_target_semantic_audit,
    kr_validation_target_source_audit,
    klir_2011_tof_detector_response_targets,
    kiai_double_dpf_icf_concept_targets,
    lee_2014_radiative_model_review_targets,
    lee_course_nx2_neon_phase_timing_example_targets,
    lee_drive_parameter_speed_enhancement_targets,
    lee_radpf_theory_model_scope_targets,
    lee_snowplow_phase_semantics_targets,
    llnl_12kj_em_fluctuation_evidence_from_signal,
    llnl_12kj_em_fluctuation_targets,
    llnl_fully_kinetic_dpf_targets,
    mcalpine_dpf_nrta_mcnp_targets,
    mjolnir_first_experiments_targets,
    mjolnir_neutron_anisotropy_evidence,
    mjolnir_neutron_detector_response_evidence,
    mjolnir_neutron_detector_response_targets,
    mjolnir_high_low_parasitic_current_targets,
    mjolnir_neutron_spectrum_evidence,
    mjolnir_neutron_timing_evidence_from_history,
    mjolnir_neutron_timing_targets,
    mjolnir_stagnation_temperature_targets,
    narkis_kr_doped_dpf_mhd_targets,
    nstec_3d_mhd_rundown_targets,
    ou_foi_2d_dpf_simulation_targets,
    sun_two_temperature_mhd_motion_targets,
    pf1000_16kv_akel_table_candidate_evidence,
    pf1000_16kv_akel_table_targets,
    pf1000_16kv_derived_output_candidate_evidence,
    pf1000_16kv_current_waveform_comparison_candidate_evidence,
    pf1000_16kv_current_waveform_digitization_candidate_evidence,
    pf1000_16kv_current_waveform_targets,
    pf1000_16kv_phase_candidate_evidence_from_history,
    pf1000_16kv_shot12581_phase_targets,
    pf1000_cikhardtova_linear_density_motion_targets,
    pf1000_full_energy_neutron_spatial_targets,
    pf1000_full_energy_phase_context_targets,
    pf1000_interferometry_density_evidence_from_profile,
    pf1000_szydlowski_fast_ion_neutron_targets,
    pf1000_interferometry_density_targets,
    pf400j_xray_inference_targets,
    pf1000_spatial_pinch_evidence_from_geometry,
    pf1000_spatial_pinch_targets,
    nnss_dpf_neutron_time_energy_tomography_targets,
    nx3_springham_zrbe_activation_targets,
    pfz200_hybrid_xpinch_proton_neutron_targets,
    predictive_readiness_report,
    rawat_dpf_operating_envelope_targets,
    sha256_file,
    uofsi_argon_temperature_targets,
    validation_tier_report,
    wante_nitrogen_ion_irradiation_targets,
    wang_metallic_vapor_interferometry_targets,
)


def _write_bytes(path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _accepted_pf1000_waveform_packet(tmp_path, *, validation_scope="pf1000_16kv_2021_akel"):
    repo_root = Path(__file__).resolve().parents[1]
    source_path = _write_bytes(
        tmp_path / "KnowledgeReference" / "radiation-physics-and-chemistry-188-2021-109633.md",
        (
            repo_root
            / "KnowledgeReference"
            / "radiation-physics-and-chemistry-188-2021-109633.md"
        ).read_bytes(),
    )
    figure_path = _write_bytes(
        tmp_path
        / "KnowledgeReference"
        / "figures"
        / "akel-2021-fig1-current-waveform-shot-12581.png",
        (
            repo_root
            / "KnowledgeReference"
            / "figures"
            / "akel-2021-fig1-current-waveform-shot-12581.png"
        ).read_bytes(),
    )
    packet_sha = "synthetic-accepted-waveform-packet"
    return {
        "task_id": "akel_2021_fig1_current_waveform_shot_12581",
        "validation_scope": validation_scope,
        "packet_sha256": packet_sha,
        "source_path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "source_sha256": sha256_file(source_path),
        "source_pdf_sha256": (
            "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
        ),
        "source_lines": "294-295",
        "figure_image_path": (
            "KnowledgeReference/figures/"
            "akel-2021-fig1-current-waveform-shot-12581.png"
        ),
        "figure_image_sha256": sha256_file(figure_path),
        "figure_id": "Fig. 1",
        "page": 3,
        "extraction_type": "figure",
        "axis_calibration": {
            "x": {
                "pixel_points": [0.0, 600.0],
                "data_values": [0.0, 6.0],
                "unit": "us",
                "rms_residual_px": 0.2,
            },
            "y": {
                "pixel_points": [400.0, 20.0],
                "data_values": [0.0, 1.4],
                "unit": "MA",
                "rms_residual_px": 0.2,
            },
        },
        "digitized_series": [
            {
                "name": "measured_current",
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
                "x_unit": "us",
                "y_unit": "MA",
            },
            {
                "name": "computed_current",
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [0.0, 0.25, 0.75, 1.18, 1.02, 0.63, 0.72],
                "x_unit": "us",
                "y_unit": "MA",
            },
        ],
        "verification": {
            "overlay_rms_residual_px": 0.4,
            "independent_review_count": 1,
            "review_status": "accepted",
            "review_metadata": {
                "reviewed_packet_sha256": packet_sha,
                "reviewed_source_sha256": sha256_file(source_path),
                "reviewed_figure_image_sha256": sha256_file(figure_path),
                "task_id": "akel_2021_fig1_current_waveform_shot_12581",
                "validation_scope": validation_scope,
                "reviewer": "independent-reviewer",
                "review_date": "2026-05-08",
                "review_notes": "Independent review accepted the source-bound packet.",
                "decision": "accepted",
            },
        },
    }


def test_mjolnir_neutron_timing_target_metadata():
    target = mjolnir_neutron_timing_targets()

    assert target["target_id"] == "mjolnir_neutron_timing_2025_goyon"
    assert target["device"] == "MJOLNIR"
    assert target["model_role"] == "kr_validation_target"
    assert target["validation_tier"] == 5
    assert target["source"].startswith("KnowledgeReference/")
    assert "neutron-generation-dynamics" in target["source"]
    assert target["source_lines"]["mechanisms"] == "405-448"
    assert target["source_lines"]["tof_and_smoothing"] == "474-530"
    assert target["source_lines"]["spectrum_anisotropy"] == "548-616"


def test_mjolnir_shot_context_matches_kr_values():
    target = mjolnir_neutron_timing_targets()
    context = target["shot_context"]

    assert context["charge_voltage_kV"] == 60.0
    assert context["stored_energy_kJ"] == 735.0
    assert context["average_peak_current_MA"] == 2.8
    assert context["stagnation_current_MA"] == 2.1


def test_mjolnir_neutron_event_sequence_contains_required_mechanisms():
    target = mjolnir_neutron_timing_targets()
    events = {
        (event["event"], event["mechanism"]): event
        for event in target["event_sequence"]
    }

    stagnation = events[("stagnation", "thermonuclear")]
    assert stagnation["relative_time_ns"] == 0.0
    assert stagnation["required"] is True

    disruption = events[("first_disruption", "beam_target")]
    assert disruption["relative_time_ns"] == 5.0
    assert disruption["required"] is True

    measurement = events[("first_beam_target_measurement_correlation", "beam_target")]
    assert measurement["relative_time_ns"] == 10.0
    assert measurement["required"] is True


def test_mjolnir_detector_spectrum_and_anisotropy_targets():
    target = mjolnir_neutron_timing_targets()

    assert target["detector_tof"]["neutron_energy_MeV"] == 2.45
    assert target["detector_tof"]["arrival_delay_ns_first_detector"] == 96.0
    assert "2.45 MeV" in target["spectral_targets"]["thermonuclear"]
    assert "5 MeV" in target["spectral_targets"]["beam_target"]
    assert "10 percent" in target["anisotropy_targets"]["low_yield"]
    assert "60-100 percent" in target["anisotropy_targets"]["high_yield"]


def _complete_mjolnir_detector_response() -> dict[str, object]:
    return {
        "activation_reactions": ["Be", "Y", "Br"],
        "activation_detector_angles_deg": [10.0, 45.0, 70.0],
        "be_absolute_calibrated": True,
        "labr_y_cross_calibrated_to_be": True,
        "tof_distances_m": [2.2, 6.6],
        "relative_timing_precision_ns": 1.0,
        "response_terms": [
            "propagation_widening",
            "detector_temporal_response",
            "xray_peak_cotiming",
            "beam_target_energy_spread",
            "room_scatter_or_background_assessment",
        ],
    }


def test_kr_validation_target_manifest_lists_coded_targets():
    manifest = kr_validation_target_manifest()
    target_ids = {str(target["target_id"]) for target in manifest}

    assert "lee_radpf_phase_semantics_course" in target_ids
    assert "lee_course_nx2_neon_phase_timing_example" in target_ids
    assert "lee_radpf_theory_model_scope_2008" in target_ids
    assert "lee_2014_radiative_model_review" in target_ids
    assert "pf1000_16kv_current_waveform_2021_akel" in target_ids
    assert "pf1000_16kv_shot12581_phase_2021_akel" in target_ids
    assert "pf1000_16kv_shot_table_2021_akel" in target_ids
    assert "pf1000_full_energy_phase_context_2007_gribkov" in target_ids
    assert "pf1000_full_energy_neutron_spatial_2007_scholz" in target_ids
    assert "deuterium_argon_admixture_neutron_2026_omar" in target_ids
    assert "ff1_focus_fusion_plasmoid_2023_lerner" in target_ids
    assert "lee_drive_parameter_speed_enhancement_2003" in target_ids
    assert "auluck_gpf_scaling_theory_2023" in target_ids
    assert "auluck_neutron_yield_scaling_failure_2023" in target_ids
    assert "auluck_circuit_element_poynting_2021" in target_ids
    assert "blagoev_electric_flux_diagnostic_2025" in target_ids
    assert "auluck_poloidal_magnetic_field_2024" in target_ids
    assert "wante_nitrogen_ion_irradiation_2025" in target_ids
    assert "demina_dpf_material_damage_apdm4" in target_ids
    assert "altarabulsi_deuteron_beam_fluence_2024" in target_ids
    assert "kiai_double_dpf_icf_concept_2025" in target_ids
    assert "wang_metallic_vapor_interferometry_1999" in target_ids
    assert "pfz200_hybrid_xpinch_proton_neutron_2026_novotny" in target_ids
    assert "llnl_fully_kinetic_dpf_2012_schmidt" in target_ids
    assert "nstec_3d_mhd_rundown_2014_meehan" in target_ids
    assert "alegra_hedp_dpf_mhd_validation_2009_kueny" in target_ids
    assert "esaulov_2d_mhrdr_dpf_2003" in target_ids
    assert "ou_foi_2d_dpf_simulation_2024" in target_ids
    assert "sun_two_temperature_mhd_motion_2025" in target_ids
    assert "beresnyak_hawk_3d_mhd_2018" in target_ids
    assert "narkis_kr_doped_dpf_mhd_2021" in target_ids
    assert "faeton_i_high_voltage_dpf_2025" in target_ids
    assert "mjolnir_high_low_parasitic_current_2022_goyon" in target_ids
    assert "mjolnir_first_experiments_2021_offermann" in target_ids
    assert "pf400j_xray_inference_2020_orellana" in target_ids
    assert "rawat_dpf_operating_envelope_2015" in target_ids
    assert "mjolnir_neutron_timing_2025_goyon" in target_ids
    assert "mjolnir_stagnation_temperature_2025_goyon" in target_ids
    assert "mcalpine_dpf_nrta_mcnp_2014" in target_ids
    assert "pf1000_interferometry_density_2024_malir" in target_ids
    assert "uofsi_argon_temperature_thesis_2020" in target_ids
    assert all(str(target["source"]).startswith("KnowledgeReference/") for target in manifest)
    assert all(int(target["n_source_line_items"]) > 0 for target in manifest)


def test_kr_validation_target_source_audit_passes_for_local_targets():
    audit = kr_validation_target_source_audit()

    assert audit["passed"] is True
    assert audit["missing_or_invalid_targets"] == []
    assert len(audit["targets"]) == len(kr_validation_target_manifest())
    assert all(
        target["source_authority_passed"] is True
        for target in audit["targets"]
    )


def test_kr_validation_target_semantic_audit_passes_for_coded_targets():
    audit = kr_validation_target_semantic_audit()

    assert audit["passed"] is True
    assert audit["missing_or_failed_targets"] == []
    targets = {target["target_id"]: target for target in audit["targets"]}
    assert targets["pf1000_16kv_current_waveform_2021_akel"]["passed"] is True
    assert targets["mjolnir_neutron_detector_response_2025_goyon"]["passed"] is True


def test_kr_validation_target_coverage_report_lists_remaining_groups():
    report = kr_validation_target_coverage_report()
    groups = {record["group"]: record for record in report["groups"]}

    assert report["passed"] is False
    assert groups["circuit_waveform"]["status"] == "partial"
    assert "pf1000_16kv_current_waveform_2021_akel" in (
        groups["circuit_waveform"]["target_ids"]
    )
    assert "pf1000_16kv_shot_table_2021_akel" in (
        groups["circuit_waveform"]["target_ids"]
    )
    assert groups["phase_timing"]["status"] == "partial"
    assert "pf1000_16kv_shot12581_phase_2021_akel" in (
        groups["phase_timing"]["target_ids"]
    )
    assert "lee_course_nx2_neon_phase_timing_example" in (
        groups["phase_timing"]["target_ids"]
    )
    assert "pf1000_full_energy_phase_context_2007_gribkov" in (
        groups["phase_timing"]["target_ids"]
    )
    assert "deuterium_argon_admixture_neutron_2026_omar" in (
        groups["phase_timing"]["target_ids"]
    )
    assert groups["neutron_yield"]["status"] == "partial"
    assert "pf1000_16kv_shot_table_2021_akel" in (
        groups["neutron_yield"]["target_ids"]
    )
    assert "pf1000_full_energy_neutron_spatial_2007_scholz" in (
        groups["neutron_yield"]["target_ids"]
    )
    assert groups["neutron_detector_response"]["status"] == "present"
    assert groups["spatial_temperature"]["status"] == "partial"
    assert "pf1000_full_energy_neutron_spatial_2007_scholz" in (
        groups["spatial_temperature"]["target_ids"]
    )
    assert "mjolnir_stagnation_temperature_2025_goyon" in (
        groups["spatial_temperature"]["target_ids"]
    )
    assert "circuit_waveform" in report["missing_or_partial_groups"]
    assert "phase_timing" in report["missing_or_partial_groups"]


def test_kr_validation_same_scope_target_report_requires_one_scope():
    report = kr_validation_same_scope_target_report()

    assert report["passed"] is False
    assert report["passed_scopes"] == []
    best = report["best_available_scope"]
    assert isinstance(best, dict)
    assert best["status"] == "incomplete"
    assert best["missing_groups"]
    widest = report["widest_available_scope"]
    assert isinstance(widest, dict)
    assert widest["validation_scope"] == "pf1000_full_energy_2007_gribkov_scholz"
    assert widest["missing_groups"] == []
    assert "circuit_waveform" in widest["partial_groups"]
    assert "neutron_yield" in widest["partial_groups"]
    assert report["next_same_scope_steps"][0].startswith(
        "Use the widest same-scope packet"
    )
    scopes = {scope["validation_scope"]: scope for scope in report["scopes"]}
    mjolnir = scopes["mjolnir_neutron_timing_2025_goyon"]
    assert "neutron_detector_response" in mjolnir["present_groups"]
    assert "spatial_temperature" in mjolnir["present_groups"]
    assert "spatial_temperature" in mjolnir["partial_groups"]
    assert "circuit_waveform" in mjolnir["missing_groups"]
    pf1000 = scopes["pf1000_full_energy_2007_gribkov_scholz"]
    assert "phase_semantics" in pf1000["present_groups"]
    assert "phase_timing" in pf1000["present_groups"]
    assert "circuit_waveform" in pf1000["present_groups"]
    assert "spatial_magnetic_or_em" in pf1000["present_groups"]
    assert "neutron_detector_response" in pf1000["present_groups"]
    assert "neutron_yield" in pf1000["present_groups"]
    assert "neutron_detector_response" not in pf1000["missing_groups"]
    assert "neutron_detector_response" in pf1000["partial_groups"]
    assert "neutron_yield" in pf1000["partial_groups"]
    assert "spatial_temperature" in pf1000["partial_groups"]
    blockers = pf1000["closure_blockers"]
    assert blockers["circuit_waveform"]["data_availability"] == (
        "partial_only_in_same_scope_targets"
    )
    assert "digitized_current_trace_points" in (
        blockers["circuit_waveform"]["missing_items"]
    )
    assert blockers["circuit_waveform"]["required_data_to_complete"] == (
        blockers["circuit_waveform"]["missing_items"]
    )
    assert "radial_transit_start_and_end_times" in (
        blockers["phase_timing"]["missing_items"]
    )
    assert "direct_experimental_temperature_diagnostic" in (
        blockers["spatial_temperature"]["missing_items"]
    )
    assert "neutron_field_transport_or_room_scatter_response_model" in (
        blockers["neutron_detector_response"]["missing_items"]
    )
    assert "yield_calibration_uncertainty" in (
        blockers["neutron_yield"]["missing_items"]
    )
    assert "fast_ion_distribution_uncertainty" in (
        blockers["uncertainty"]["missing_items"]
    )
    assert blockers["neutron_detector_response"]["sources"] == [
        "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md"
    ]
    akel = scopes["pf1000_16kv_2021_akel"]
    assert "phase_semantics" in akel["present_groups"]
    assert "uncertainty" in akel["present_groups"]
    assert "phase_semantics" not in akel["missing_groups"]
    assert "uncertainty" not in akel["missing_groups"]
    assert "phase_timing" in akel["partial_groups"]
    assert "uncertainty" in akel["partial_groups"]
    assert "systematic_detector_response_uncertainty" in (
        akel["closure_blockers"]["uncertainty"]["missing_items"]
    )
    assert mjolnir["closure_blockers"]["circuit_waveform"][
        "data_availability"
    ] == "absent_from_same_scope_targets"
    assert mjolnir["closure_blockers"]["circuit_waveform"][
        "required_data_to_complete"
    ] == ["same_scope_circuit_waveform_target"]


def test_mjolnir_stagnation_temperature_target_is_partial_context():
    target = mjolnir_stagnation_temperature_targets()

    assert target["target_id"] == "mjolnir_stagnation_temperature_2025_goyon"
    assert target["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"
    assert target["validation_tier"] == 4
    assert target["source_lines"]["stagnation_temperature_scaling"] == "293-316"
    assert target["temperature_targets"]["stagnation_temperature_scaling_reference_keV"] == 21.0
    assert "direct_experimental_temperature_diagnostic" in (
        target["missing_for_full_tier4"]
    )


def test_rawat_operating_envelope_is_generic_review_target():
    target = rawat_dpf_operating_envelope_targets()

    assert target["target_id"] == "rawat_dpf_operating_envelope_2015"
    assert target["device"] == "generic DPF"
    assert "rawat" in target["validation_note"].lower()
    assert target["phase_timing"]["current_sheath_formation_ns_range"] == [
        100.0,
        500.0,
    ]
    assert target["phase_timing"]["optimized_axial_speed_cm_per_us_range"] == [
        2.0,
        10.0,
    ]
    assert target["phase_timing"]["radial_speed_multiplier_over_axial_range"] == [
        2.0,
        2.5,
    ]
    assert target["spatial_density_targets"]["pinch_plasma_density_m3_range"] == [
        5.0e24,
        1.0e26,
    ]
    assert target["temperature_targets"]["pinch_temperature_keV_range"] == [
        0.2,
        2.0,
    ]
    assert target["operational_context"]["capacitor_charge_voltage_kV_typical_range"] == [
        10.0,
        30.0,
    ]
    assert target["uncertainty"]["shot_to_shot_variation_noted"] is True
    assert "device_specific_current_trace" in target["missing_for_full_tier2"]


def test_auluck_gpf_scaling_theory_target_marks_validation_limits():
    target = auluck_gpf_scaling_theory_targets()

    assert target["target_id"] == "auluck_gpf_scaling_theory_2023"
    assert target["device"] == "Generalized Plasma Focus / tapered-anode concept"
    assert target["scaling_theory_targets"][
        "conventional_dpf_fusion_scaling_failure_observed"
    ] is True
    assert target["scaling_theory_targets"]["space_propulsion_feasibility_claimed"] is False
    assert target["phase_timing"]["quarter_period_us_laboratory_example"] == 8.45
    assert target["phase_timing"]["wire_surface_travel_time_ns"] == 8.4
    assert target["spatial_density_targets"]["hydrogen_pressure_mbar_example"] == 43.0
    assert target["magnetic_field_targets"]["example_wire_surface_B_final_T"] == 200.0
    assert target["propulsion_example_targets"]["stored_energy_kJ_example"] == 8.6
    assert target["propulsion_example_targets"]["impulse_kg_m_per_s_example"] == 0.002
    assert "measure_voltage_across_plasma" in target["validation_requirements"]
    assert "profile_sweep_validation_data" in target["missing_for_predictive_tier2"]
    assert "deuterium_tube_neutron_yield_measurement" in target["missing_for_full_tier5"]


def test_auluck_neutron_yield_scaling_failure_target_requires_liftoff_tests():
    target = auluck_neutron_yield_scaling_failure_targets()

    assert target["target_id"] == "auluck_neutron_yield_scaling_failure_2023"
    assert target["scaling_failure_targets"][
        "yield_inverse_power_of_insulator_radius_ratio"
    ] == 5.0
    assert target["scaling_failure_targets"]["typical_insulator_radius_ratio"] == 1.0
    assert target["scaling_failure_targets"][
        "proposed_reduced_insulator_radius_ratio"
    ] == 0.4
    assert target["scaling_failure_targets"][
        "claimed_yield_increase_orders_if_all_conditions_met"
    ] == 2.0
    assert target["phase_timing"]["liftoff_time_measurement_is_primary_test"] is True
    assert target["neutron_yield_targets"][
        "small_devices_can_test_scaling_failure_without_neutron_measurements"
    ] is True
    assert target["model_scope_limits"][
        "equations_12_and_17_not_available_in_markdown"
    ] is True
    assert "measured_liftoff_time_vs_drive_parameter" in (
        target["missing_for_full_tier2"]
    )


def test_alegra_hedp_mhd_validation_target_keeps_neutron_limit_explicit():
    target = alegra_hedp_dpf_mhd_validation_targets()

    assert target["target_id"] == "alegra_hedp_dpf_mhd_validation_2009_kueny"
    assert target["current_waveform_targets"]["bernard_short_peak_current_alegra_MA"] == 1.5
    assert target["current_waveform_targets"]["tallboy_peak_current_alegra_MA"] == 1.8
    assert target["neutron_yield_targets"]["bernard_long_experiment_neutrons"] == 1.5e9
    assert target["neutron_yield_targets"]["bernard_long_alegra_thermonuclear_neutrons"] == 1.2e5
    assert target["neutron_yield_targets"]["tallboy_experiment_neutrons"] == 3.5e11
    assert target["neutron_yield_targets"]["tallboy_alegra_thermonuclear_neutrons"] == 3.7e7
    assert target["mhd_scope_limits"]["mhd_can_model_only_thermonuclear_component"] is True
    assert target["mhd_scope_limits"]["three_dimensional_mhd_required_before_particle_followup"] is True
    assert target["numerical_model_limits"]["qEOS_deuterium_used"] is True
    assert target["numerical_model_limits"]["seed_ionized_gas_temperature_eV"] == 1.0
    assert "nonthermal_beam_target_neutron_model" in target["missing_for_full_tier5"]


def test_auluck_circuit_element_target_requires_3d_poynting_terms():
    target = auluck_circuit_element_poynting_targets()

    assert target["target_id"] == "auluck_circuit_element_poynting_2021"
    assert target["current_waveform_targets"]["dI_dt_diagnostic_standard"] is True
    assert target["current_waveform_targets"]["scalar_time_varying_inductance_is_incomplete"] is True
    assert target["phase_timing"]["pf1000_example_probe_times_ns"] == [
        -68.0,
        -38.0,
        22.0,
    ]
    assert target["phase_timing"]["neon_dI_dt_minimum_after_column_breakup_ns_min"] == 200.0
    assert target["spatial_density_targets"]["current_carrying_layer_thickness_cm_range"] == [
        1.6,
        2.6,
    ]
    assert target["magnetic_field_targets"]["pf1000_probe_radii_mm"] == [
        40.0,
        13.0,
        0.0,
    ]
    assert target["field_coupling_requirements"][
        "plasma_inductance_from_total_magnetic_energy_incomplete"
    ] is True
    assert target["field_coupling_requirements"]["anomalous_impedance_needed_for_unaccounted_terms"] is True
    assert "volume_integrated_J_dot_E_from_3d_fields" in target["missing_for_full_tier3"]


def test_blagoev_electric_flux_diagnostic_records_formation_symmetry_limits():
    target = blagoev_electric_flux_diagnostic_targets()

    assert target["target_id"] == "blagoev_electric_flux_diagnostic_2025"
    assert target["shot_context"]["stored_energy_kJ"] == 3.0
    assert target["shot_context"]["capacitance_uF"] == 20.0
    assert target["shot_context"]["charging_voltage_kV_max"] == 40.0
    assert target["shot_context"]["cathode_rods"] == 6
    assert target["current_waveform_targets"][
        "radial_phase_between_current_maximum_and_singularity"
    ] is True
    assert target["current_waveform_targets"]["sampling_interval_ns"] == 1.0
    assert target["phase_timing"]["shot_667_reference_singularity_time_us"] == 3.03
    assert target["electric_flux_diagnostic_targets"][
        "three_symmetric_identical_d_dot_probes"
    ] is True
    assert target["calibration_targets"]["integrated_d_dot_max_within_percent_of_mean"] == 3.0
    assert target["calibration_targets"]["applied_voltage_kV_max"] == 5.34
    assert target["hardware_fault_targets"][
        "hollow_copper_anode_hidden_deformation_detected"
    ] is True
    assert target["modeling_scope_limits"][
        "not_a_neutron_yield_validation_dataset"
    ] is True
    assert "digitized_d_dot_waveforms" in target["missing_for_full_tier1"]


def test_auluck_poloidal_magnetic_field_target_keeps_test_incomplete():
    target = auluck_poloidal_magnetic_field_targets()

    assert target["target_id"] == "auluck_poloidal_magnetic_field_2024"
    assert target["current_waveform_targets"][
        "mhd_codes_neglecting_dynamo_can_overestimate_observed_current"
    ] is True
    assert target["magnetic_field_targets"]["magnetic_probe_spatial_resolution_mm_range"] == [
        1.0,
        2.0,
    ]
    assert target["magnetic_field_targets"][
        "simple_dynamo_seeded_by_geomagnetic_field"
    ] is True
    assert target["magnetic_field_targets"][
        "hall_term_neglected_as_model_assumption"
    ] is True
    assert target["field_context"][
        "gpf_scaling_magnetic_field"
    ] == "B0 = mu0 * I(t) / (2*pi*a*r_tilde)"
    assert target["experimental_test_requirements"][
        "external_field_amplitude_max_times_local_geomagnetic"
    ] == 2.0
    assert target["supporting_observation"][
        "nikulin_cone_plasma_focus_energy_kJ"
    ] == 2.5
    assert target["modeling_scope_limits"]["boundary_value_problem_not_fully_solved"] is True
    assert "external_axial_field_sweep_dataset" in target["missing_for_full_tier4"]


def test_wante_nitrogen_ion_irradiation_target_is_not_neutron_validation():
    target = wante_nitrogen_ion_irradiation_targets()

    assert target["target_id"] == "wante_nitrogen_ion_irradiation_2025"
    assert target["shot_context"]["stored_energy_kJ_nominal"] == 3.0
    assert target["shot_context"]["operated_energy_kJ"] == 2.54
    assert target["shot_context"]["capacitance_uF"] == 30.0
    assert target["shot_context"]["charging_voltage_kV"] == 13.0
    assert target["shot_context"]["optimal_pressure_mbar"] == 1.5
    assert target["phase_timing"]["flight_path_cm"] == 38.0
    assert target["current_waveform_targets"]["lee_fit_fc"] == 0.7
    assert target["current_waveform_targets"]["lee_fit_fcr"] == 0.85
    assert target["ion_beam_targets"]["measured_nitrogen_ion_energy_keV"] == 72.40
    assert target["ion_beam_targets"]["lee_model_nitrogen_ion_energy_keV"] == 71.0
    assert target["ion_beam_targets"]["ion_flux_m2_s"] == 7.2e27
    assert target["ion_beam_targets"]["ion_fluence_m2"] == 6.4e19
    assert target["application_response_targets"][
        "nitrogen_doping_percent_by_shots"
    ]["24"] == 7.93
    assert target["application_response_targets"][
        "copper_impurity_from_anode_ablation_increases_with_dose"
    ] is True
    assert target["modeling_scope_limits"][
        "material_processing_target_not_neutron_validation"
    ] is True
    assert "neutron_yield" in target["missing_for_full_tier5"]


def test_demina_dpf_material_damage_target_bounds_application_response_only():
    target = demina_dpf_material_damage_targets()

    assert target["target_id"] == "demina_dpf_material_damage_apdm4"
    assert target["device_context"]["PF_1000_bank_energy_MJ"] == 1.2
    assert target["device_context"]["PF_1000_experiment_stored_energy_kJ_approx"] == 600.0
    assert target["device_context"]["PF_1000_initial_pressure_Pa"] == 470.0
    assert target["radiation_environment_targets"][
        "power_flux_density_W_cm2_range"
    ] == [1.0e7, 1.0e10]
    assert target["radiation_environment_targets"]["pulse_duration_us_range"] == [
        0.2,
        1.0,
    ]
    assert target["tungsten_damage_targets"][
        "intergranular_and_transgranular_microcracks_above_W_cm2"
    ] == 1.0e8
    assert target["tungsten_damage_targets"][
        "erosion_depth_per_pulse_um_by_condition"
    ]["1e10_ion_1e9_plasma_W_cm2"] == 2.05
    assert target["cfc_damage_targets"][
        "CFC_8SiC_evaporated_layer_um_per_shot_at_1e9_W_cm2"
    ] == 2.6
    assert target["redeposition_targets"]["observed_elements_on_W"] == [
        "Cu",
        "O",
        "Fe",
        "Cr",
    ]
    assert target["model_scope_limits"][
        "material_damage_target_not_core_dpf_machine_validation"
    ] is True
    assert "current_voltage_waveforms" in target["missing_for_full_dpf_validation"]


def test_altarabulsi_deuteron_beam_fluence_target_matches_table3():
    target = altarabulsi_deuteron_beam_fluence_targets()

    assert target["target_id"] == "altarabulsi_deuteron_beam_fluence_2024"
    assert target["shot_context"]["model_code"] == "RADPFV6.16FIB"
    assert target["device_parameter_targets"]["PF-1000"]["stored_energy_kJ"] == 863.1
    assert target["device_parameter_targets"]["MPEF-12 kJ"]["V0_kV"] == 22.0
    assert target["device_parameter_targets"]["PF-2.7 kJ"]["fit_pressure_Torr"] == 0.15
    assert target["current_waveform_targets"][
        "mpef_12kj_end_of_pinch_time_us_approx"
    ] == 2.08
    assert target["ion_beam_formula_targets"][
        "beam_kinetic_energy_fraction_of_pinch_inductive_energy_fe"
    ] == 0.14
    pf1000_rows = target["fluence_comparison_targets"]["PF-1000"]["rows"]
    assert pf1000_rows[0]["pressure_Torr"] == 0.5
    assert pf1000_rows[0]["sim_ions_m2"] == 7.3e19
    assert pf1000_rows[0]["exp_ions_m2"] == 7.5e19
    mpef_rows = target["fluence_comparison_targets"]["MPEF-12 kJ"]["rows"]
    assert mpef_rows[3]["pressure_Torr"] == 3.0
    assert mpef_rows[3]["sim_ions_m2"] == 7.5e18
    assert mpef_rows[3]["exp_ions_m2"] == 7.05e18
    assert mpef_rows[3]["exp_sigma_ions_m2"] == 0.70e18
    pf27_rows = target["fluence_comparison_targets"]["PF-2.7 kJ"]["rows"]
    assert pf27_rows[1]["sim_ions_m2"] == 4.94e15
    assert pf27_rows[1]["exp_ions_m2"] == 4.95e15
    assert target["model_scope_limits"][
        "ion_beam_fluence_target_not_neutron_validation"
    ] is True
    assert "raw_fluence_detector_response" in target["missing_for_full_tier5"]


def test_kiai_double_dpf_icf_concept_target_is_projection_not_validation():
    target = kiai_double_dpf_icf_concept_targets()

    assert target["target_id"] == "kiai_double_dpf_icf_concept_2025"
    assert target["shot_context"]["concept_is_theoretical_not_built_device"] is True
    assert target["conceptual_full_scale_parameters"][
        "stored_bank_energy_MJ_total"
    ] == 6.0
    assert target["conceptual_full_scale_parameters"][
        "stored_bank_energy_MJ_each"
    ] == 3.0
    assert target["conceptual_full_scale_parameters"][
        "peak_circuit_current_MA"
    ] == 20.0
    assert target["conceptual_full_scale_parameters"][
        "pinch_lifetime_ns_each_dpf"
    ] == 300.0
    assert target["prototype_30kj_parameters"]["stored_energy_kJ"] == 30.0
    assert target["prototype_30kj_parameters"]["operating_voltage_kV_range"] == [
        50.0,
        60.0,
    ]
    assert target["prototype_30kj_parameters"][
        "fusion_neutron_yield_per_shot"
    ] == 1.0e10
    assert target["magnetic_field_targets"]["hts_field_T_range"] == [10.0, 15.0]
    assert target["power_projection_targets"]["with_hts_fusion_power_MW"] == 75.0
    assert target["model_scope_limits"][
        "theoretical_proposal_not_validated_experiment"
    ] is True
    assert "dt_pellet_experiment" in target["missing_for_full_tier5"]


def test_wang_metallic_vapor_target_records_interferometry_scope():
    target = wang_metallic_vapor_interferometry_targets()

    assert target["target_id"] == "wang_metallic_vapor_interferometry_1999"
    assert target["device"] == "DPF-16"
    assert target["shot_context"]["stored_energy_kJ"] == 16.0
    assert target["shot_context"]["charging_voltage_kV"] == 20.0
    assert target["shot_context"]["peak_current_kA"] == 380.0
    assert target["shot_context"]["working_pressure_Pa_range"] == [70.0, 650.0]
    assert target["geometry"]["anode_diameter_mm"] == 66.0
    assert target["geometry"]["target_diameter_mm"] == 10.0
    assert target["phase_timing"]["metallic_vapor_visible_time_ns"] == 280.0
    assert target["phase_timing"]["higher_pressure_vapor_times_ns"] == [
        220.0,
        300.0,
    ]
    assert target["spatial_density_targets"][
        "laser_differential_interferometer_records_ps_evolution"
    ] is True
    assert target["xray_material_process_targets"][
        "high_density_volume_absent_with_hollow_anode"
    ] is True
    assert target["model_scope_limits"][
        "material_vapor_target_not_neutron_validation"
    ] is True
    assert "electron_beam_energy" in target["missing_for_full_tier5"]


def test_esaulov_2d_mhrdr_target_records_begay_device_and_limits():
    target = esaulov_2d_mhrdr_dpf_targets()

    assert target["target_id"] == "esaulov_2d_mhrdr_dpf_2003"
    assert target["shot_context"]["inner_electrode_radius_cm"] == 1.18
    assert target["shot_context"]["fill_pressure_torr"] == 1.0
    assert target["shot_context"]["capacitance_uF"] == 36.4
    assert target["shot_context"]["charging_voltage_kV"] == 14.0
    assert target["current_waveform_targets"]["current_sheath_current_kA_range_during_acceleration"] == [
        50.0,
        100.0,
    ]
    assert target["phase_timing"]["local_neutron_rate_peak_times_us"] == [
        2.74,
        2.92,
    ]
    assert target["field_context"]["ion_electron_radiation_temperatures"] is True
    assert target["neutron_yield_targets"]["dd_yield_computed_with_maxwell_average_cross_sections"] is True
    assert target["mhd_scope_limits"]["beam_target_mechanisms_not_primary_in_this_target"] is True
    assert "beam_target_or_kinetic_neutron_component" in target["missing_for_full_tier5"]


def test_ou_foi_2d_dpf_target_records_parameter_sweeps_and_limits():
    target = ou_foi_2d_dpf_simulation_targets()

    assert target["target_id"] == "ou_foi_2d_dpf_simulation_2024"
    assert target["model_context"]["code"] == "FOI"
    assert target["model_context"]["courant_number"] == 0.5
    assert target["llnl_reference_case"]["anode_diameter_cm"] == 15.2
    assert target["llnl_reference_case"]["peak_current_MA"] == 2.5
    assert target["llnl_reference_case"]["fill_pressure_Pa"] == 2926.0
    assert target["current_waveform_targets"]["quarter_period_ns"] == 135.0
    assert target["phase_timing"]["llnl_run_down_time_us"] == 3.9
    assert target["phase_timing"]["current_amplitude_MA_cases"] == [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]
    assert target["phase_timing"]["pinch_time_ns_by_current_MA"]["1.5"] == 188.99
    assert target["phase_timing"]["pinch_current_MA_by_current_amplitude_MA"][
        "2.5"
    ] == 2.500
    assert target["spatial_density_targets"]["pressure_sweep_Pa"] == [
        133.0,
        665.0,
        1330.0,
        1995.0,
        2660.0,
    ]
    assert target["magnetic_field_targets"]["anode_radius_mm_cases"] == [
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
    ]
    assert target["model_scope_limits"][
        "simulation_vs_llnl_morphology_agrees_but_timing_differs"
    ] is True
    assert "measured_neutron_yield" in target["missing_for_full_tier5"]


def test_sun_two_temperature_mhd_motion_target_records_unu_motion_scope():
    target = sun_two_temperature_mhd_motion_targets()

    assert target["target_id"] == "sun_two_temperature_mhd_motion_2025"
    assert target["current_waveform_targets"]["UNU_V0_kV"] == 15.0
    assert target["current_waveform_targets"]["UNU_C0_uF"] == 30.0
    assert target["current_waveform_targets"]["UNU_L0_nH"] == 110.0
    assert target["current_waveform_targets"]["UNU_r0_mohm"] == 12.0
    assert target["geometry"]["UNU_anode_radius_cm"] == 0.95
    assert target["geometry"]["UNU_cathode_radius_cm"] == 3.2
    assert target["phase_timing"]["UNU_axial_phase_us_range"] == [0.0, 2.5]
    assert target["phase_timing"]["UNU_radial_implosion_us_range"] == [
        2.78,
        2.90,
    ]
    assert target["spatial_density_targets"]["UNU_background_density_m3"] == 2.4e23
    assert target["temperature_targets"][
        "radial_implosion_ion_temperature_keV_approx"
    ] == 1.0
    assert target["magnetic_field_targets"]["sheath_speed_up_to_km_s"] == 90.0
    assert target["parameter_scaling_targets"]["c_ratio_cases_for_pf1000"] == [
        1.4,
        1.8,
        2.2,
        2.6,
    ]
    assert target["model_scope_limits"]["no_self_consistent_neutron_production"] is True
    assert "self_consistent_neutron_production" in target["missing_for_full_tier5"]


def test_beresnyak_hawk_3d_mhd_target_is_model_scope_not_validation():
    target = beresnyak_hawk_3d_mhd_targets()

    assert target["target_id"] == "beresnyak_hawk_3d_mhd_2018"
    assert target["shot_context"]["generator_current_kA"] == 665.0
    assert target["shot_context"]["generator_rise_time_us"] == 1.2
    assert target["shot_context"]["generator_inductance_nH"] == 720.0
    assert target["circuit_model"]["initial_capacitor_voltage_kV"] == 640.0
    assert target["geometry"]["anode_radius_cm"] == 6.33
    assert target["geometry"]["cathode_radius_cm"] == 8.57
    assert target["spatial_density_targets"][
        "characteristic_number_density_cm3"
    ] == 3.0e16
    assert target["phase_timing"]["pinch_time_us_at_density_3e16_cm3"] == 0.95
    assert target["current_waveform_targets"][
        "target_density_device_voltage_kV_below"
    ] == 10.0
    assert target["temperature_targets"][
        "maximum_plasma_temperature_keV_around"
    ] == 3.0
    assert target["particle_acceleration_targets"][
        "stochastic_power_law_tail_cutoff_keV_around"
    ] == 200.0
    assert target["model_scope_limits"][
        "current_disruption_voltages_not_captured"
    ] is True
    assert "beam_target_current_disruption_model" in target["missing_for_full_tier5"]


def test_narkis_kr_doped_mhd_target_blocks_total_yield_claims():
    target = narkis_kr_doped_dpf_mhd_targets()

    assert target["target_id"] == "narkis_kr_doped_dpf_mhd_2021"
    assert target["shot_context"]["simulation_code"] == "HYDRA"
    assert target["shot_context"]["peak_current_MA_range"] == [2.0, 3.0]
    assert target["shot_context"]["krypton_dopant_volume_fraction_cases"] == [
        0.0,
        0.001,
        0.01,
    ]
    assert target["geometry"]["anode_radius_cm"] == 7.62
    assert target["current_waveform_targets"]["R_mohm"] == 1.4
    assert target["current_waveform_targets"]["C_uF"] == 432.0
    assert target["phase_timing"]["time_us_by_dopant_and_voltage"][
        "0.1% Kr"
    ]["40"] == 6.285
    assert target["spatial_density_targets"][
        "ion_density_by_dopant_and_voltage"
    ]["1.0% Kr"]["50"] == 15.87
    assert target["temperature_targets"]["peak_temperature_keV_by_dopant"][
        "1.0% Kr"
    ] == 12.6
    assert target["neutron_yield_targets"][
        "thermonuclear_yield_order_range"
    ] == [1.0e9, 1.0e10]
    assert target["neutron_yield_targets"]["max_dNdt_35kV_neutrons_per_ns"][
        "0.1% Kr"
    ] == 2.4e9
    assert target["model_scope_limits"][
        "mhd_cannot_capture_beam_target_neutron_production"
    ] is True
    assert "beam_target_neutron_model" in target["missing_for_full_tier5"]


def test_faeton_i_high_voltage_target_records_partial_validation_packet():
    target = faeton_i_high_voltage_dpf_targets()

    assert target["target_id"] == "faeton_i_high_voltage_dpf_2025"
    assert target["device"] == "FAETON-I"
    assert target["shot_context"]["direct_charged_voltage_kV"] == 100.0
    assert target["shot_context"]["deuterium_shots_recorded_min"] == 1100
    assert target["current_waveform_targets"][
        "radial_current_factor_good_sheath_threshold"
    ] == 0.7
    assert target["current_waveform_targets"]["table_3_shots"][2]["shot"] == 1027
    assert target["current_waveform_targets"]["table_3_shots"][2]["Yn_measured"] == 5.44e10
    assert target["current_waveform_targets"]["table_3_shots"][3]["Vp_kV"] == 194.0
    assert target["phase_timing"]["voltage_spike_before_stagnation"] is True
    assert target["neutron_yield_targets"]["exceptional_dd_yield_max"] == 8.0e10
    assert target["spectral_targets"]["dd_neutron_energy_peak_MeV"] == 2.5
    assert target["spectral_targets"]["dd_neutron_energy_uncertainty_MeV"] == 0.3
    assert target["anisotropy_targets"]["forward_on_axis_factor"] == 1.6
    assert target["response_model_requirements"]["pmt_scintillator_distances_m"] == [
        5.0,
        10.0,
        20.0,
        40.0,
    ]
    assert target["projections_not_validation_targets"][
        "dt_projection_is_not_validated_by_faeton_i_dd_data"
    ] is True
    assert "digitized_current_trace" in target["missing_for_full_tier1"]
    assert "independent_dt_projection_validation" in target["missing_for_full_tier5"]


def test_lee_course_nx2_phase_timing_example_metadata():
    target = lee_course_nx2_neon_phase_timing_example_targets()

    assert target["target_id"] == "lee_course_nx2_neon_phase_timing_example"
    assert target["validation_tier"] == 2
    assert target["source_lines"]["phase_endpoint_times"] == "1938-1958"
    assert target["phase_timing"]["axial_end_time_us"] == 1.172
    assert target["phase_timing"]["radial_duration_us"] == 0.235
    assert target["phase_timing"]["pinch_duration_ns"] == 26.2
    assert "experimental_phase_timing_uncertainty" in (
        target["missing_for_predictive_tier2"]
    )


def test_lee_radpf_theory_model_scope_records_reduced_model_limits():
    target = lee_radpf_theory_model_scope_targets()

    assert target["target_id"] == "lee_radpf_theory_model_scope_2008"
    assert target["phase_semantics"]["radial"].startswith("radial phase replaces")
    assert target["current_waveform_targets"][
        "external_circuit_and_sheath_motion_coupled"
    ] is True
    assert target["phase_timing"][
        "characteristic_axial_to_radial_time_ratio_typical"
    ] == 40.0
    assert target["phase_timing"][
        "reflected_shock_speed_fraction_of_on_axis_radial_shock_speed"
    ] == 0.3
    assert target["temperature_targets"][
        "deuterium_radiation_collapse_critical_current_MA"
    ] == 1.6
    assert target["radiation_model_targets"]["line_loss_term"] is True
    assert target["neutron_yield_model_targets"]["code_Vmax_kV_order_range"] == [
        20.0,
        50.0,
    ]
    assert target["neutron_yield_model_targets"][
        "empirical_yield_fit"
    ] == "Yn = 9e10 * Ipinch^3.8"
    assert target["neutron_yield_model_targets"][
        "calibration_point_yield_neutrons"
    ] == 7.0e9
    assert target["model_scope_limits"][
        "beam_target_yield_calibrated_to_experiment_not_predicted_ab_initio"
    ] is True
    assert "independent_beam_target_calibration_validation" in (
        target["missing_for_full_tier5"]
    )


def test_lee_2014_radiative_model_review_records_equation_scope():
    target = lee_2014_radiative_model_review_targets()

    assert target["target_id"] == "lee_2014_radiative_model_review"
    assert target["phase_model_targets"]["phase_count"] == 5
    assert target["phase_model_targets"]["type_2_optional_phase_4a_anomalous_resistance"] is True
    assert target["current_waveform_targets"]["radial_phase_closed_equation_set"] == [
        "14",
        "15",
        "17",
        "19",
    ]
    assert target["phase_timing"][
        "reflected_shock_speed_fraction_of_on_axis_inward_shock"
    ] == 0.3
    assert target["phase_timing"]["radial_normalized_time_increment"] == 0.00001
    assert target["temperature_targets"]["reflected_shock_temperature_jump_factor_near"] == 2.0
    assert target["radiation_model_targets"][
        "deuterium_radiation_collapse_critical_current_MA"
    ] == 1.6
    assert target["radiation_model_targets"]["neon_argon_critical_current_below_kA"] == 100.0
    assert target["neutron_context"]["no_neutron_observable_targets"] is True
    assert "measured_radiated_power_trace" in target["missing_for_full_tier4"]


def test_pf1000_current_waveform_target_metadata():
    target = pf1000_16kv_current_waveform_targets()

    assert target["target_id"] == "pf1000_16kv_current_waveform_2021_akel"
    assert target["validation_scope"] == "pf1000_16kv_2021_akel"
    assert target["validation_tier"] == 1
    assert target["source"] == "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    assert target["source_lines"]["measured_current_waveform_figures"] == "294-300"
    assert target["current_waveform_targets"]["measured_current_available"] is True
    assert target["current_waveform_targets"]["peak_current_kA_range"] == [
        1100.0,
        1300.0,
    ]
    assert "digitized_current_trace_points" in target["missing_for_full_tier1"]


def test_pf1000_current_waveform_digitization_candidate_reports_review_blocker():
    packet = akel_fig1_draft_digitization_packet()

    evidence = pf1000_16kv_current_waveform_digitization_candidate_evidence(packet)

    assert evidence["passed"] is False
    assert evidence["waveform_digitization_status"] == "blocked_by_review"
    assert evidence["validation_scope"] == "pf1000_16kv_2021_akel"
    assert evidence["details"]["required_series_present"] is True
    assert evidence["details"]["available_series"] == [
        "computed_current",
        "measured_current",
    ]
    assert evidence["details"]["missing_or_failed_checks"] == [
        "independent_review_missing",
        "review_status_not_accepted",
    ]
    assert evidence["details"]["overlay_rms_residual_px"] == 0.213455189
    assert evidence["details"]["review_status"] == "draft"
    assert evidence["details"]["independent_review_count"] == 0
    assert "per_point_current_uncertainty" in (
        evidence["details"]["missing_for_full_tier1"]
    )


def test_pf1000_current_waveform_comparison_blocks_draft_without_metrics():
    packet = akel_fig1_draft_digitization_packet()

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0],
        [0.0, 0.5, 1.0],
        packet,
    )

    assert evidence["passed"] is False
    assert evidence["waveform_comparison_status"] == "blocked_by_review"
    assert evidence["metrics_computed"] is False


def test_pf1000_current_waveform_comparison_requires_uncertainty(tmp_path):
    packet = _accepted_pf1000_waveform_packet(tmp_path)

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
        packet,
        base_path=tmp_path,
    )

    assert evidence["passed"] is False
    assert evidence["waveform_comparison_status"] == "blocked_by_missing_uncertainty"
    assert evidence["metrics_computed"] is False


def test_pf1000_current_waveform_comparison_blocks_stale_review_without_metrics(tmp_path):
    packet = _accepted_pf1000_waveform_packet(tmp_path)
    packet["verification"]["review_metadata"][
        "reviewed_packet_sha256"
    ] = "stale-packet-hash"

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is False
    assert evidence["waveform_comparison_status"] == "blocked_by_review"
    assert evidence["metrics_computed"] is False
    checks = evidence["details"]["digitization_readiness"]["details"][
        "missing_or_failed_checks"
    ]
    assert checks == ["review_packet_hash_mismatch"]


def test_pf1000_current_waveform_comparison_blocks_malformed_review_without_metrics(
    tmp_path,
):
    packet = _accepted_pf1000_waveform_packet(tmp_path)
    review = packet["verification"]["review_metadata"]
    review["reviewer"] = ""
    review["review_notes"] = ""

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is False
    assert evidence["waveform_comparison_status"] == "blocked_by_review"
    assert evidence["metrics_computed"] is False
    checks = evidence["details"]["digitization_readiness"]["details"][
        "missing_or_failed_checks"
    ]
    assert checks == ["review_notes_missing", "reviewer_missing"]


def test_pf1000_current_waveform_comparison_passes_same_scope_trace(tmp_path):
    packet = _accepted_pf1000_waveform_packet(tmp_path)

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is True
    assert evidence["waveform_comparison_status"] == "passed"
    assert evidence["metrics_computed"] is True
    assert evidence["details"]["waveform_nrmse"] == 0.0
    assert evidence["details"]["simulated_dip"]["dip_present"] is True


def test_pf1000_current_waveform_comparison_rejects_cross_scope_packet(tmp_path):
    packet = _accepted_pf1000_waveform_packet(
        tmp_path,
        validation_scope="pf1000_27kv_full_energy",
    )

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.7],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is False
    assert evidence["waveform_comparison_status"] == "blocked_by_scope_mismatch"
    assert evidence["metrics_computed"] is False


def test_pf1000_current_waveform_comparison_fails_distorted_waveform(tmp_path):
    packet = _accepted_pf1000_waveform_packet(tmp_path)

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.1, 0.2, 0.25, 0.2, 0.15, 0.1],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is False
    assert "waveform_nrmse_too_large" in evidence["details"]["missing_or_failed_checks"]


def test_pf1000_current_waveform_comparison_fails_missing_dip(tmp_path):
    packet = _accepted_pf1000_waveform_packet(tmp_path)

    evidence = pf1000_16kv_current_waveform_comparison_candidate_evidence(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        packet,
        base_path=tmp_path,
        uncertainty={"current_MA": 0.03, "time_us": 0.02},
    )

    assert evidence["passed"] is False
    assert "simulated_current_dip_missing" in (
        evidence["details"]["missing_or_failed_checks"]
    )


def test_pf1000_akel_table_target_metadata():
    target = pf1000_16kv_akel_table_targets()

    assert target["target_id"] == "pf1000_16kv_shot_table_2021_akel"
    assert target["validation_scope"] == "pf1000_16kv_2021_akel"
    assert target["validation_tier"] == 5
    assert target["source_lines"]["table_1_current_and_fit_rows"] == "330-583"
    assert target["source_lines"]["table_2_pinch_and_yield_rows"] == "584-837"
    assert target["table_extraction_verification"]["merged_row_count"] == 24
    assert target["table_extraction_verification"][
        "table_1_table_2_shot_ids_match"
    ] is True
    assert target["table_extraction_verification"][
        "source_markdown_pdf_parity_verified"
    ] is True

    rows = {int(row["shot"]): row for row in target["shot_rows"]}
    assert len(rows) == 24
    assert rows[12581]["peak_current_kA"] == 1165.0
    assert rows[12581]["fmr"] == 0.26
    assert rows[12581]["pinch_radius_cm"] == 2.40
    assert rows[12603]["measured_neutron_yield_n"] == 11.2e9
    assert rows[12590]["measured_neutron_yield_uncertainty_n"] == 0.2e8

    summaries = target["neutron_yield_targets"]["pressure_group_summaries"]
    assert summaries["1.20_torr"]["shot_count"] == 8
    assert summaries["1.05_torr"]["shot_count"] == 16
    assert np.isclose(
        summaries["1.20_torr"]["mean_computed_neutron_yield_n"],
        1.78e9,
        rtol=0.01,
    )
    assert np.isclose(
        summaries["1.05_torr"]["mean_measured_neutron_yield_n"],
        2.29e9,
        rtol=0.01,
    )
    assert target["uncertainty"]["measured_neutron_yield_uncertainty_n_range"] == [
        2.0e7,
        2.0e8,
    ]
    assert "systematic_detector_response_uncertainty" in (
        target["uncertainty"]["missing_uncertainty_components"]
    )
    assert "uncertainty" in target["partial_target_groups"]
    assert "digitized_current_trace_points" in target["missing_for_full_tier1"]
    assert "neutron_detector_response" in target["missing_for_full_tier5"]


def test_pf1000_akel_table_candidate_evidence_passes_complete_scalar_rows():
    target = pf1000_16kv_akel_table_targets()
    predictions = []
    for row in target["shot_rows"]:
        predictions.append({
            "shot": row["shot"],
            "peak_current_kA": row["peak_current_kA"],
            "pinch_current_kA": row["pinch_current_kA"],
            "axial_speed_cm_per_us": row["axial_speed_cm_per_us"],
            "shock_speed_cm_per_us": row["shock_speed_cm_per_us"],
            "piston_speed_cm_per_us": row["piston_speed_cm_per_us"],
            "pinch_density_1e23_per_m3": row["pinch_density_1e23_per_m3"],
            "pinch_radius_cm": row["pinch_radius_cm"],
            "pinch_length_cm": row["pinch_length_cm"],
            "predicted_neutron_yield_n": row["measured_neutron_yield_n"],
        })

    evidence = pf1000_16kv_akel_table_candidate_evidence(predictions, target=target)

    assert evidence["passed"] is True
    assert evidence["validation_scope"] == "pf1000_16kv_2021_akel"
    assert evidence["validated_features"] == {"yield": True}
    assert evidence["row_count"] == {"required": 24, "provided": 24}
    assert evidence["missing_shots"] == []
    assert evidence["field_passes"]["neutron_yield_n"] is True
    assert evidence["max_measurement_uncertainty_normalized_error"] == 0.0
    rows = {result["shot"]: result for result in evidence["shot_results"]}
    assert rows[12603]["fields"]["neutron_yield_n"]["relative_error"] == 0.0
    assert rows[12603]["fields"]["neutron_yield_n"]["measured_uncertainty_n"] == (
        2.0e8
    )
    assert rows[12603]["fields"]["neutron_yield_n"][
        "measurement_uncertainty_normalized_error"
    ] == 0.0


def test_pf1000_akel_table_candidate_evidence_rejects_missing_or_bad_rows():
    target = pf1000_16kv_akel_table_targets()
    predictions = {
        int(row["shot"]): {
            "peak_current_kA": row["peak_current_kA"],
            "pinch_current_kA": row["pinch_current_kA"],
            "axial_speed_cm_per_us": row["axial_speed_cm_per_us"],
            "shock_speed_cm_per_us": row["shock_speed_cm_per_us"],
            "piston_speed_cm_per_us": row["piston_speed_cm_per_us"],
            "pinch_density_1e23_per_m3": row["pinch_density_1e23_per_m3"],
            "pinch_radius_cm": row["pinch_radius_cm"],
            "pinch_length_cm": row["pinch_length_cm"],
            "neutron_yield_n": row["measured_neutron_yield_n"],
        }
        for row in target["shot_rows"]
    }
    predictions.pop(12581)
    predictions[12603]["neutron_yield_n"] = 1.0e9

    evidence = pf1000_16kv_akel_table_candidate_evidence(predictions, target=target)

    assert evidence["passed"] is False
    assert evidence["missing_shots"] == [12581]
    assert evidence["row_count"] == {"required": 24, "provided": 23}
    assert evidence["field_passes"]["neutron_yield_n"] is False
    assert 12581 in evidence["missing_fields"]["peak_current_kA"]
    assert evidence["max_relative_errors"]["neutron_yield_n"] > 0.5
    assert evidence["max_measurement_uncertainty_normalized_error"] > 40.0


def test_mjolnir_detector_response_target_metadata():
    target = mjolnir_neutron_detector_response_targets()

    assert target["target_id"] == "mjolnir_neutron_detector_response_2025_goyon"
    assert target["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"
    assert target["source_lines"]["activation_diagnostics"] == "132-149"
    assert target["source_lines"]["synthetic_detector_response"] == "449-509"
    assert target["activation_requirements"]["reactions"] == ["Be", "Y", "Br"]
    assert target["tof_requirements"]["scintillator_distances_m"] == [2.2, 6.6]


def test_mcalpine_dpf_nrta_mcnp_target_is_application_response_scope():
    target = mcalpine_dpf_nrta_mcnp_targets()

    assert target["target_id"] == "mcalpine_dpf_nrta_mcnp_2014"
    assert target["dpf_source_context"]["source_neutron_energy_MeV"] == 2.45
    assert target["dpf_source_context"]["llnl_dpf_yield_neutrons_about"] == 1.0e7
    assert target["dpf_source_context"]["llnl_simulated_pulse_duration_ns_range"] == [
        20.0,
        60.0,
    ]
    assert target["nrta_targets"]["resonance_energy_eV_range"] == [1.0, 50.0]
    assert target["mcnp_setup"]["moderator_thickness_cm"] == 3.0
    assert target["mcnp_setup"]["detector_distance_m"] == 2.0
    assert target["mcnp_setup"]["source_particles_per_simulation"] == 1.0e10
    assert target["nrta_tof_context"]["dpf_gaussian_fwhm_ns"] == 20.0
    assert target["nrta_tof_context"]["eng_trapezoidal_pulse_us"] == 4.0
    assert target["application_results"]["dpf_time_for_comparable_measurement"] == "single_pulse"
    assert target["model_scope_limits"]["dpf_source_not_self_consistently_simulated"] is True
    assert "experimental_nrta_benchmark" in target["missing_for_nrta_validation"]


def test_mjolnir_detector_response_evidence_requires_full_response_model():
    evidence = mjolnir_neutron_detector_response_evidence(
        _complete_mjolnir_detector_response(),
    )

    assert evidence["passed"] is True
    assert evidence["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"
    assert evidence["diagnostics"]["activation_response"] is True
    assert evidence["diagnostics"]["tof_response"] is True
    assert evidence["diagnostics"]["synthetic_response_model"] is True


def test_mjolnir_detector_response_evidence_rejects_unassessed_scatter():
    response = _complete_mjolnir_detector_response()
    response["response_terms"] = [
        "propagation_widening",
        "detector_temporal_response",
        "xray_peak_cotiming",
        "beam_target_energy_spread",
    ]

    evidence = mjolnir_neutron_detector_response_evidence(response)

    assert evidence["passed"] is False
    assert evidence["diagnostics"]["synthetic_response_model"] is False
    assert "room_scatter_or_background_assessment" in (
        evidence["details"]["missing_response_terms"]
    )


def test_kr_target_is_not_predictive_readiness_evidence_by_itself():
    target = mjolnir_neutron_timing_targets()
    result = {
        "I_peak": 1.0,
        "n_steps": 20,
        "neutron_mechanism_timing_validation": target,
    }

    readiness = predictive_readiness_report(result)
    assert "passed" not in target
    assert readiness.ready is False
    assert (
        "Neutron yield/mechanism/timing/spectrum/anisotropy validation"
        in readiness.missing_evidence
    )


def test_lee_phase_semantics_target_metadata():
    target = lee_snowplow_phase_semantics_targets()

    assert target["target_id"] == "lee_radpf_phase_semantics_course"
    assert target["validation_tier"] == 2
    assert target["source"].startswith("KnowledgeReference/")
    assert target["source_lines"]["radial_rollover_and_dip"] == "14922-14936"
    assert target["required_for_full_tier2"] == ["axial", "radial", "pinch"]


def test_pf1000_16kv_phase_target_metadata():
    target = pf1000_16kv_shot12581_phase_targets()

    assert target["target_id"] == "pf1000_16kv_shot12581_phase_2021_akel"
    assert target["validation_scope"] == "pf1000_16kv_2021_akel"
    assert target["device"] == "PF-1000"
    assert target["shot"] == "12581"
    assert target["validation_tier"] == 2
    assert target["shot_context"]["voltage_kV"] == 16.0
    assert target["lee_fit_parameters"]["fmr"] == 0.26
    assert target["phase_semantics"]["axial_phase_mass_swept_factor"] == "fm"
    assert target["phase_semantics"]["radial_phase_current_factor"] == "fcr"
    assert target["phase_timing"]["current_dip_end_time_us"] == 8.0
    assert target["phase_timing"]["pinch_duration_ns"] == 212.0
    assert "axial_rundown_end_time" in target["missing_for_full_tier2"]


def test_pf1000_full_energy_phase_context_target_metadata():
    target = pf1000_full_energy_phase_context_targets()

    assert target["target_id"] == "pf1000_full_energy_phase_context_2007_gribkov"
    assert target["validation_scope"] == "pf1000_full_energy_2007_gribkov_scholz"
    assert target["source"] == "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md"
    assert target["shot_context"]["bank_energy_kJ_max"] == 850.0
    assert target["shot_context"]["discharge_current_MA_range"] == [2.5, 3.0]
    assert "first_compression" in target["phase_semantics"]
    assert target["phase_timing"]["max_compression_before_current_dip_ns"] == 100.0
    assert target["phase_timing"]["pinch_confinement_time_ns"] == 150.0
    assert "radial_transit_start_and_end_times" in target["missing_for_full_tier2"]


def test_pf1000_full_energy_neutron_spatial_target_metadata():
    target = pf1000_full_energy_neutron_spatial_targets()

    assert target["target_id"] == "pf1000_full_energy_neutron_spatial_2007_scholz"
    assert target["validation_scope"] == "pf1000_full_energy_2007_gribkov_scholz"
    assert target["source"] == "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md"
    assert target["shot_context"]["bank_energy_kJ"] == 810.0
    assert target["shot_context"]["shot_3121_voltage_kV"] == 35.0
    assert target["current_waveform_targets"]["total_current_typical_MA_range"] == [
        2.5,
        2.6,
    ]
    assert target["current_waveform_targets"]["estimated_average_pinch_current_MA"] == 2.0
    assert target["spatial_density_targets"]["first_compression_ion_density_cm3"] == 0.8e19
    assert target["magnetic_field_targets"]["first_compression_azimuthal_Bmax_MG"] == 2.0
    assert target["temperature_targets"]["direct_ion_temperature_measured"] is False
    assert target["activation_requirements"]["activation_counter_materials"] == [
        "silver",
        "indium",
    ]
    assert target["tof_requirements"]["scintillator_pm_distance_m"] == 7.0
    assert "room_scatter_or_background_assessment" in (
        target["response_model_requirements"]
    )
    assert target["anisotropy_targets"]["shot_3121_Y0_over_Y90"] == 1.8
    assert target["uncertainty"]["bubble_detector_relative_lower_at_90deg"] == 0.30
    assert "neutron_field_transport_or_room_scatter_response_model" in (
        target["missing_for_full_tier5"]
    )


def test_pf1000_cikhardtova_linear_density_motion_target_metadata():
    target = pf1000_cikhardtova_linear_density_motion_targets()

    assert target["target_id"] == "pf1000_linear_density_motion_2015_cikhardtova"
    assert target["validation_scope"] == (
        "pf1000_shot9881_linear_density_2015_cikhardtova"
    )
    assert target["source"] == "KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md"
    assert target["source_lines"]["linear_density_formula"] == "124-140"
    assert target["shot_context"]["shot"] == 9881
    assert target["density_formula_targets"][
        "linear_density_per_shifted_fringe_coefficient"
    ] == 2.1e15
    assert target["phase_timing"]["timing_uncertainty_ns_range"] == [2.0, 3.0]
    assert target["spatial_motion_targets"]["zipper_velocity_m_per_s_range"] == [
        5.0e5,
        1.5e6,
    ]
    assert target["spatial_motion_targets"]["mean_implosion_velocity_m_per_s"] == 2.2e5
    assert "spatial_density" in target["partial_target_groups"]
    assert "digitized_linear_density_profiles_from_figures_3_to_6" in (
        target["missing_for_full_tier4"]
    )


def test_pf1000_szydlowski_fast_ion_neutron_target_metadata():
    target = pf1000_szydlowski_fast_ion_neutron_targets()

    assert target["target_id"] == "pf1000_fast_ion_neutron_2004_szydlowski"
    assert target["validation_scope"] == (
        "pf1000_full_energy_fast_ion_neutron_2004_szydlowski"
    )
    assert target["source_lines"]["device_geometry_and_energy"] == "90-112"
    assert target["shot_context"]["energy_level_kJ_range"] == [266.0, 1064.0]
    assert "review source PDF glyph" in target["shot_context"]["capacitance_source_text"]
    assert target["activation_requirements"]["silver_activation_counter_count"] == 4
    assert target["neutron_yield_targets"][
        "regular_neutron_emission_neutrons_per_shot_range"
    ] == [1.0e10, 1.0e11]
    assert target["anisotropy_targets"]["coefficient_at_133_Pa"] == 1.4
    assert target["anisotropy_targets"]["coefficient_at_665_Pa_less_than"] == 1.2
    assert target["spectral_targets"]["upstream_spectrum_peak_MeV_range"] == [2.2, 2.3]
    assert target["fast_ion_targets"]["crater_density_per_mm2_range"] == [
        1.0e3,
        1.0e5,
    ]
    assert "pdf_review_of_ocr_suspect_units" in target["missing_for_full_tier5"]


def test_klir_tof_detector_response_target_metadata():
    target = klir_2011_tof_detector_response_targets()

    assert target["target_id"] == "tof_detector_response_2011_klir"
    assert target["source_lines"]["temporal_resolution"] == "171-198"
    assert target["detector_use_scope"]["neutron_yield_range_per_shot"] == [
        1.0e6,
        1.0e13,
    ]
    assert target["scintillator_targets"]["material"] == "Saint Gobain BC-408"
    assert target["scintillator_targets"]["thickness_mm"] == 50.0
    assert target["pmt_targets"]["assembly"] == "Hamamatsu H1949-51"
    assert target["response_timing_targets"]["single_neutron_signal_fwhm_ns"] == 5.7
    assert target["response_timing_targets"][
        "single_neutron_signal_fwhm_uncertainty_ns_2sigma"
    ] == 0.6
    assert target["timing_calibration"]["pmt_delay_uncertainty_ns_less_than"] == 1.0
    assert "neutron_detector_response" in target["partial_target_groups"]
    assert "digitized_fig2_voltage_response_curve" in target["missing_for_full_tier5"]


def test_springham_zrbe_activation_target_metadata():
    target = nx3_springham_zrbe_activation_targets()

    assert target["target_id"] == "nx3_zrbe_activation_2021_springham"
    assert target["validation_scope"] == "nx3_zrbe_activation_2021_springham"
    assert target["source_lines"]["abstract_targets"] == "36-60"
    assert target["shot_context"]["bank_energy_kJ"] == 7.2
    assert target["shot_context"]["fill_pressure_mbar_values"] == [
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
    assert target["activation_requirements"]["detector_angles_deg"] == [0.0, 90.0]
    assert target["neutron_yield_targets"]["highest_yield_pressure_mbar"] == 5.0
    assert target["spectral_targets"]["effective_energy_MeV_at_0deg_approx"] == 2.8
    assert target["anisotropy_targets"]["fluence_anisotropy_AnBe_range"] == [2.5, 4.5]
    assert target["mechanism_targets"]["beam_target_model_consistent"] is True
    assert target["mechanism_targets"]["thermonuclear_contribution_negligible"] is True
    assert "mcnp_response_curve_packet" in target["missing_for_full_tier5"]


def test_catenacci_time_energy_tomography_target_metadata():
    target = nnss_dpf_neutron_time_energy_tomography_targets()

    assert target["target_id"] == (
        "nnss_dpf_neutron_time_energy_tomography_2020_catenacci"
    )
    assert target["validation_scope"] == "nnss_dpf_neutron_tomography_2020_catenacci"
    assert target["source_lines"]["shadow_bar_subtraction"] == "388-463"
    assert target["tomography_model_targets"]["energy_grid_MeV_range"] == [1.45, 3.45]
    assert target["tomography_model_targets"]["typical_energy_bin_count_range"] == [
        25,
        30,
    ]
    assert target["detector_geometry"]["shadow_bar_pair_distances_m"] == [
        10.0,
        14.0,
        18.0,
        22.0,
    ]
    assert target["detector_geometry"]["close_range_detector_distance_cm"] == 25.0
    assert target["neutron_timing_targets"]["double_pinch_separation_ns_less_than"] == 50.0
    assert target["spectral_targets"]["energy_resolution_estimated_finer_than_keV"] == 100.0
    assert target["spectral_targets"][
        "scatter_correction_max_relative_difference_fraction"
    ] == 0.23
    assert "digitized_fig4_time_energy_reconstructions" in (
        target["missing_for_full_tier5"]
    )


def test_deuterium_argon_admixture_neutron_target_metadata():
    target = deuterium_argon_admixture_neutron_targets()

    assert target["target_id"] == "deuterium_argon_admixture_neutron_2026_omar"
    assert target["validation_tier"] == 5
    assert "deuterium-argon-admixture" in target["source"]
    assert target["shot_context"]["capacitance_uF"] == 30.0
    assert target["shot_context"]["total_fill_pressure_mbar"] == 4.0
    assert target["current_waveform_targets"]["rogowski_conversion_kA_per_V"] == 36.0
    assert target["phase_timing"]["plasma_focus_time_us_range"] == [2.7, 3.3]
    assert (
        target["neutron_yield_targets"][
            "fifty_percent_argon_average_neutrons_per_shot"
        ]
        == 3.0e7
    )
    assert (
        target["pinch_energy_targets"]["fifty_percent_argon_energy_into_pinch_J"]
        == 139.0
    )
    assert target["activation_requirements"]["activation_material"] == "indium"
    assert target["activation_requirements"]["calibration_factor_neutrons_per_count"] == 8.22e4
    assert target["temperature_targets"]["direct_temperature_measured"] is False
    assert target["uncertainty"]["uncertainty_statistic"] == "standard_deviation"
    assert "time_resolved_neutron_history" in target["missing_for_full_tier5"]


def test_ff1_focus_fusion_plasmoid_target_metadata():
    target = ff1_focus_fusion_plasmoid_targets()

    assert target["target_id"] == "ff1_focus_fusion_plasmoid_2023_lerner"
    assert target["device"] == "FF-1 / FF-2B"
    assert target["validation_tier"] == 5
    assert "focus-fusion-overview" in target["source"]
    assert target["shot_context"]["capacitance_uF"] == 113.0
    assert target["shot_context"]["stored_energy_kJ_max"] == 115.0
    assert target["current_waveform_targets"]["best_pinch_inductance_increase_nH"] == 10.0
    assert target["spatial_density_targets"]["minimum_ion_density_cm3"] == 3.0e19
    assert target["magnetic_field_targets"]["qmf_required_B_for_p_GG"] == 14.0
    assert target["temperature_targets"]["confined_ion_energy_best_keV"] == 240.0
    assert target["temperature_targets"]["confined_ion_energy_uncertainty_keV"] == 20.0
    assert target["detector_tof"]["neutron_tof_detector_distances_m"] == [11.5, 17.5]
    assert target["neutron_yield_targets"]["best_2016_neutron_yield"] == 2.5e11
    assert target["advanced_fuel_context"]["p_b11_projection_is_validated"] is False
    assert target["advanced_fuel_context"]["nst_keV_s_per_m3"] == 3.4e20
    assert target["impurity_targets"]["zeff_nominal"] == 1.004
    assert "p_b11 experimental neutron_alpha_yield_measurement" in (
        target["missing_for_full_tier5"]
    )


def test_lee_drive_parameter_speed_enhancement_target_metadata():
    target = lee_drive_parameter_speed_enhancement_targets()

    assert target["target_id"] == "lee_drive_parameter_speed_enhancement_2003"
    assert target["model_role"] == "kr_scaling_regime_target"
    assert target["source"] == (
        "KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-"
        "enhancing-the-neutron-yield.md"
    )
    assert target["phase_timing"]["deuterium_minimum_radius_over_anode_radius"] == 0.12
    assert target["phase_timing"]["deuterium_maximum_length_over_anode_radius"] == 0.8
    assert target["phase_timing"]["deuterium_pinch_lifetime_s_per_m"] == 2.0e-6
    assert (
        target["drive_parameter_targets"]["mean_kA_per_cm_per_sqrt_torr"]
        == 89.0
    )
    assert (
        target["drive_parameter_targets"][
            "standard_deviation_kA_per_cm_per_sqrt_torr"
        ]
        == 7.7
    )
    assert target["temperature_targets"]["small_focus_ion_temperature_keV"] == 1.0
    assert (
        target["neutron_yield_scaling_targets"][
            "speed_enhanced_thermonuclear_scaling"
        ]
        == "Yth ~ I^4 * v_axial^4"
    )
    assert target["operational_limits"]["quality_deterioration_axial_speed_cm_per_us"] == 10.0
    assert "device_specific_current_trace" in target["missing_for_predictive_tier2"]


def test_pfz200_hybrid_xpinch_proton_neutron_target_metadata():
    target = pfz200_hybrid_xpinch_proton_neutron_targets()

    assert target["target_id"] == "pfz200_hybrid_xpinch_proton_neutron_2026_novotny"
    assert target["device"] == "PFZ-200 hybrid X-pinch"
    assert target["source"] == (
        "KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-"
        "plasma-focus-2.md"
    )
    assert target["shot_context"]["stored_energy_kJ"] == 3.0
    assert target["shot_context"]["discharge_current_kA_min"] == 200.0
    assert target["shot_context"]["deuterium_pressure_Pa"] == 360.0
    assert target["current_waveform_targets"]["measured_current_available"] is True
    assert target["phase_timing"]["neutron_production_fwhm_ns"]["hybrid_3mm_gap"] == [
        20.0,
        7.0,
    ]
    assert target["phase_timing"]["neutron_production_fwhm_ns"]["unmodified_dpf"] == [
        38.0,
        9.0,
    ]
    assert target["spatial_geometry_targets"]["hybrid_3mm_gap_conclusion_diameter_mm_range"] == [
        1.1,
        1.5,
    ]
    assert target["neutron_yield_targets"]["hybrid_average_after_first_shot_ignored"] == 6.0e7
    assert target["proton_source_targets"]["maximum_proton_energy_MeV"] == 3.6
    assert target["proton_source_targets"]["maximum_deuteron_energy_MeV"] == 1.3
    assert target["activation_requirements"]["ntof_detector_distances_m"] == [
        0.30,
        0.35,
        2.5,
        4.3,
    ]
    assert target["uncertainty"]["sac_yield_may_be_overestimated_by_anisotropy"] is True
    assert "ordinary_dpf_validation_scope" in target["missing_for_full_tier5"]


def test_llnl_fully_kinetic_dpf_target_metadata():
    target = llnl_fully_kinetic_dpf_targets()

    assert target["target_id"] == "llnl_fully_kinetic_dpf_2012_schmidt"
    assert target["device"] == "LLNL DPF kinetic benchmark"
    assert target["model_role"] == "kr_kinetic_fidelity_target"
    assert target["validation_tier"] == 5
    assert "fully-kinetic-simulations" in target["source"]
    assert target["source_lines"]["pic_setup"] == "70-99"
    assert target["shot_context"]["steady_state_current_kA"] == 180.0
    assert target["simulation_context"]["code"] == "LSP"
    assert target["simulation_context"]["grid_r_by_z"] == [322, 151]
    assert target["current_waveform_targets"]["fully_kinetic_current_dip_kA"] == 15.0
    assert target["current_waveform_targets"]["experimental_current_dip_kA_max_near_1_torr"] == 40.0
    assert target["field_context"]["lower_hybrid_frequency_range_GHz"] == [
        10.0,
        20.0,
    ]
    assert target["temperature_targets"]["fully_kinetic_hot_pinch_ion_temperature_keV"] == 12.0
    assert target["spectral_targets"]["fully_kinetic_predicts_ion_energy_MeV_min"] == 1.0
    assert target["neutron_yield_targets"]["fully_kinetic_neutrons_per_shot"] == 0.86e7
    assert target["neutron_yield_targets"]["hybrid_neutrons_per_shot"] == 3.6e4
    assert "three_dimensional_kinetic_validation" in target["missing_for_full_tier5"]


def test_nstec_3d_mhd_rundown_target_metadata():
    target = nstec_3d_mhd_rundown_targets()

    assert target["target_id"] == "nstec_3d_mhd_rundown_2014_meehan"
    assert target["device"] == "NSTec / Gemini DPF"
    assert target["model_role"] == "kr_3d_mhd_rundown_benchmark_target"
    assert target["validation_tier"] == 3
    assert "fully-three-dimensional" in target["source"]
    assert target["source_lines"]["current_and_rundown_comparison"] == "514-566"
    assert target["shot_context"]["comparison_voltage_kV"] == 37.5
    assert target["shot_context"]["comparison_pressure_torr"] == 7.28
    assert target["shot_context"]["repeat_shots"] == 37
    assert target["current_waveform_targets"]["diagnostic"] == "Faraday rotator"
    assert target["current_waveform_targets"]["faraday_loop_turns"] == 5.25
    assert target["current_waveform_targets"]["measured_peak_current_MA"] == 2.17
    assert target["current_waveform_targets"]["two_dimensional_peak_current_MA"] == 2.08
    assert target["current_waveform_targets"]["three_dimensional_peak_current_MA"] == 1.82
    assert target["phase_timing"]["experimental_rundown_time_us"] == 6.96
    assert target["phase_timing"]["three_dimensional_rundown_time_us"] == 6.69
    assert target["phase_timing"]["two_dimensional_rundown_time_us"] == 5.59
    assert target["spatial_density_targets"]["density_floor_kg_per_m3"] == 2.5e-4
    assert target["temperature_targets"]["startup_hot_gas_layer_temperature_K"] == 1.0e6
    assert target["model_scope_limits"]["near_z_pinch_mhd_unphysical"] is True
    assert target["neutron_validation_context"]["neutron_yield_not_measured_in_this_target"] is True
    assert "digitized_faraday_current_traces" in target["missing_for_full_tier1"]


def test_mjolnir_high_low_parasitic_current_target_metadata():
    target = mjolnir_high_low_parasitic_current_targets()

    assert target["target_id"] == "mjolnir_high_low_parasitic_current_2022_goyon"
    assert target["device"] == "MJOLNIR"
    assert target["model_role"] == "kr_parasitic_current_yield_target"
    assert target["validation_tier"] == 5
    assert target["source"] == "KnowledgeReference/goyon-2022-mjolnir-high-low.md"
    assert target["source_lines"]["current_dip_model"] == "565-645"
    assert target["shot_context"]["one_MJ_peak_current_MA_at_100kV"] == 2.5
    assert target["shot_context"]["two_MJ_commissioned_peak_current_MA_at_70kV"] == 3.25
    assert target["shot_context"]["two_MJ_highest_neutron_yield"] == 4.1e11
    assert target["diagnostic_requirements"]["main_head_rogowski_current"] is True
    assert target["diagnostic_requirements"]["voltage_probe_resistor_ohm"] == 955.5
    assert target["simulation_context"]["pic_code"] == "CHICAGO"
    assert target["simulation_context"]["circuit_code"] == "BERTHA"
    assert target["current_waveform_targets"]["high_yield_voltage_spike_kV_about"] == 180.0
    assert target["current_waveform_targets"]["conditioning_current_path_inductance_nH"] == 7.0
    assert target["current_waveform_targets"]["low_yield_conditioning_path_time_before_stagnation_ns"] == 200.0
    assert target["phase_timing"]["runin_velocity_dataset_pressure_torr"] == 8.0
    assert target["spatial_density_targets"]["dense_target_location_z_cm_range"] == [
        1.0,
        2.0,
    ]
    assert target["field_context"]["current_path_net_resistance_mohm"] == 25.0
    assert target["neutron_yield_targets"]["nominal_yield_fluctuation_factor_about"] == 2.0
    assert target["activation_requirements"]["be_activation_detector"] is True
    assert "time_resolved_neutron_history" in target["missing_for_full_tier5"]


def test_mjolnir_first_experiments_target_metadata():
    target = mjolnir_first_experiments_targets()

    assert target["target_id"] == "mjolnir_first_experiments_2021_offermann"
    assert target["validation_scope"] == "mjolnir_1mj_first_experiments_2021"
    assert target["source"].startswith("KnowledgeReference/ieee-trans")
    assert target["shot_context"]["stored_energy_MJ_max"] == 1.0
    assert target["shot_context"]["peak_current_MA_max"] == 2.5
    assert target["shot_context"]["high_voltage_plasma_shots"] == 436
    assert target["geometry_targets"]["anode_diameter_cm"] == 15.2
    assert target["current_waveform_targets"]["lumped_capacitance_uF"] == 204.0
    assert target["current_waveform_targets"]["lumped_inductance_nH"] == 67.4
    assert target["phase_timing"]["model_camera_timing_agreement_fraction_about"] == 0.015
    assert target["neutron_yield_targets"]["max_yield_neutrons_per_pulse"] == 3.8e11
    assert target["activation_requirements"]["be_detector_angle_deg_range"] == [
        45.0,
        50.0,
    ]
    assert "anisotropy_characterization_for_radiograph_flux" in (
        target["response_model_requirements"]
    )
    assert "quantitative_neutron_anisotropy_characterization" in (
        target["missing_for_full_tier5"]
    )


def test_pf400j_xray_inference_target_metadata():
    target = pf400j_xray_inference_targets()

    assert target["target_id"] == "pf400j_xray_inference_2020_orellana"
    assert target["device"] == "PF-400J"
    assert target["model_role"] == "kr_xray_diagnostic_inference_target"
    assert target["validation_tier"] == 4
    assert "inference-of-x-ray-emission" in target["source"]
    assert target["source_lines"]["xray_detector_and_campaign"] == "236-290"
    assert target["shot_context"]["fill_gas"] == "hydrogen"
    assert target["shot_context"]["charging_voltage_kV"] == 26.0
    assert target["shot_context"]["stored_energy_J"] == 287.0
    assert target["shot_context"]["capacitance_nF"] == 850.0
    assert target["shot_context"]["recorded_discharges"] == 959
    assert target["current_waveform_targets"]["voltage_at_pinch_kV_high_pmt_signal_range"] == [
        10.0,
        14.0,
    ]
    assert target["phase_timing"]["timing_uncertainty_ns_order"] == "few"
    assert target["field_context"]["vivaldi_antenna_distance_m"] == 0.25
    assert target["xray_detector_targets"]["scintillator"] == "BC-408"
    assert target["xray_detector_targets"]["system_response_energy_keV_min"] == 20.0
    assert target["xray_detector_targets"]["lead_filter_cutoff_keV"] == 250.0
    assert target["data_acquisition"]["entire_signal_samples"] == 5625
    assert target["data_acquisition"]["cnn_matrix_shape"] == [75, 75]
    assert "vivaldi_pinch_fft" in target["ml_inference_targets"]["best_feature_set"]
    assert target["ml_inference_targets"]["entire_signal_no_significant_improvement"] is True
    assert "neutron_yield_measurement" in target["missing_for_full_tier5"]


def test_pf1000_phase_candidate_evidence_remains_partial_tier_two():
    times_s = np.array([0.0, 7.5e-6, 8.0e-6, 8.106e-6, 8.212e-6, 8.4e-6])
    phases = ["rundown", "radial", "pinch", "pinch", "pinch", "post"]

    evidence = pf1000_16kv_phase_candidate_evidence_from_history(times_s, phases)

    assert evidence["passed"] is False
    assert evidence["validation_scope"] == "pf1000_16kv_2021_akel"
    assert evidence["phases"]["pinch"] is True
    assert evidence["phases"]["axial"] is False
    assert evidence["details"]["pinch_duration_relative_error"] < 0.01
    tiers = {tier.level: tier for tier in validation_tier_report({
        "has_snowplow": True,
        "snowplow_validation": evidence,
    })}
    assert tiers[2].status == "partial"


def test_pf1000_phase_candidate_rejects_wrong_pinch_time():
    times_s = np.array([0.0, 6.0e-6, 6.1e-6, 6.2e-6])
    phases = ["rundown", "pinch", "pinch", "pinch"]

    evidence = pf1000_16kv_phase_candidate_evidence_from_history(times_s, phases)

    assert evidence["passed"] is False
    assert evidence["phases"]["pinch"] is False


def test_pf1000_derived_output_candidate_compares_lee_outputs():
    evidence = pf1000_16kv_derived_output_candidate_evidence({
        "peak_current_kA": 1165.0,
        "pinch_current_kA": 523.0,
        "axial_speed_cm_per_us": 10.5,
        "shock_speed_cm_per_us": 22.0,
        "piston_speed_cm_per_us": 18.0,
        "final_pinch_radius_cm": 2.3,
        "pinch_length_cm": 18.2,
        "vmax_kV": 30.0,
    })

    assert evidence["passed"] is False
    assert evidence["validation_scope"] == "pf1000_16kv_2021_akel"
    assert evidence["phases"] == {"axial": True, "radial": True, "pinch": True}
    assert evidence["output_passes"]["vmax_kV"] is True
    assert evidence["details"]["missing_outputs"] == []


def test_pf1000_derived_output_candidate_reports_missing_outputs():
    evidence = pf1000_16kv_derived_output_candidate_evidence({
        "peak_current_kA": 1165.0,
    })

    assert evidence["passed"] is False
    assert evidence["phases"]["axial"] is False
    assert "axial_speed_cm_per_us" in evidence["details"]["missing_outputs"]
    assert "pinch_current_kA" in evidence["details"]["missing_outputs"]


def _gaussian(times_s: np.ndarray, center_ns: float, width_ns: float, scale: float) -> np.ndarray:
    center_s = center_ns * 1.0e-9
    sigma_s = width_ns * 1.0e-9
    return scale * np.exp(-0.5 * ((times_s - center_s) / sigma_s) ** 2)


def test_mjolnir_timing_evidence_passes_with_kr_like_history():
    times_s = np.arange(90.0, 121.0, 1.0) * 1.0e-9
    history = {
        "times_s": times_s,
        "dY_thermo": _gaussian(times_s, 100.0, 1.5, 1.0e8),
        "dY_bt": (
            _gaussian(times_s, 105.0, 1.0, 5.0e7)
            + _gaussian(times_s, 110.0, 1.0, 4.0e7)
        ),
    }

    evidence = mjolnir_neutron_timing_evidence_from_history(
        history,
        stagnation_time_s=100.0e-9,
        require_measurement_correlation=True,
    )

    assert evidence["passed"] is True
    assert evidence["mechanisms"]["thermonuclear"] is True
    assert evidence["mechanisms"]["beam_target"] is True
    assert evidence["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"
    assert np.allclose(
        evidence["details"]["beam_target"]["relative_peak_times_ns"],
        [5.0, 10.0],
    )
    assert evidence["source"].startswith("KnowledgeReference/")

    tiers = {tier.level: tier for tier in validation_tier_report({
        "neutron_mechanism_timing_validation": evidence,
    })}
    assert tiers[5].status == "decomposed_estimate"

    spectrum = mjolnir_neutron_spectrum_evidence(
        thermonuclear_energies_MeV=[2.42, 2.45, 2.48],
        beam_target_energies_MeV=[2.6, 3.4, 4.2, 4.9],
    )
    tiers = {tier.level: tier for tier in validation_tier_report({
        "neutron_mechanism_timing_validation": evidence,
        "neutron_spectrum_validation": spectrum,
    })}
    assert tiers[5].status == "decomposed_estimate"

    anisotropy = mjolnir_neutron_anisotropy_evidence(
        on_axis_yield=1.8,
        off_axis_yield=1.0,
        yield_regime="high_yield",
    )
    yield_evidence = {
        "passed": True,
        "validated_features": {"yield": True},
        "validation_scope": "mjolnir_neutron_timing_2025_goyon",
        "source": evidence["source"],
        "source_lines": "548-616",
        "validation_tier": 5,
        "model_role": "simulation_to_kr_target_comparison",
    }
    tiers = {tier.level: tier for tier in validation_tier_report({
        "neutron_yield_validation": yield_evidence,
        "neutron_mechanism_timing_validation": evidence,
        "neutron_spectrum_validation": spectrum,
        "neutron_anisotropy_validation": anisotropy,
    })}
    assert tiers[5].status == "decomposed_estimate"


def test_mjolnir_spectrum_evidence_requires_narrow_thermo_and_broad_beam():
    evidence = mjolnir_neutron_spectrum_evidence(
        thermonuclear_energies_MeV=[2.42, 2.45, 2.48],
        beam_target_energies_MeV=[2.6, 3.4, 4.2, 4.9],
    )

    assert evidence["passed"] is True
    assert evidence["validated_features"]["spectrum"] is True
    assert evidence["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"
    assert evidence["mechanisms"]["thermonuclear"] is True
    assert evidence["mechanisms"]["beam_target"] is True


def test_mjolnir_spectrum_evidence_rejects_thermal_like_beam():
    evidence = mjolnir_neutron_spectrum_evidence(
        thermonuclear_energies_MeV=[2.42, 2.45, 2.48],
        beam_target_energies_MeV=[2.40, 2.45, 2.50],
    )

    assert evidence["passed"] is False
    assert evidence["mechanisms"]["thermonuclear"] is True
    assert evidence["mechanisms"]["beam_target"] is False


def test_mjolnir_anisotropy_evidence_accepts_high_yield_on_axis_excess():
    evidence = mjolnir_neutron_anisotropy_evidence(
        on_axis_yield=1.8,
        off_axis_yield=1.0,
        yield_regime="high_yield",
    )

    assert evidence["passed"] is True
    assert evidence["validated_features"]["anisotropy"] is True
    assert evidence["validation_scope"] == "mjolnir_neutron_timing_2025_goyon"


def test_mjolnir_anisotropy_evidence_rejects_wrong_high_yield_trend():
    evidence = mjolnir_neutron_anisotropy_evidence(
        on_axis_yield=1.05,
        off_axis_yield=1.0,
        yield_regime="high_yield",
    )

    assert evidence["passed"] is False
    assert evidence["validated_features"]["anisotropy"] is False


def test_mjolnir_timing_evidence_fails_without_beam_target_history():
    times_s = np.arange(90.0, 121.0, 1.0) * 1.0e-9
    history = {
        "times_s": times_s,
        "dY_thermo": _gaussian(times_s, 100.0, 1.5, 1.0e8),
        "dY_bt": np.zeros_like(times_s),
    }

    evidence = mjolnir_neutron_timing_evidence_from_history(
        history,
        stagnation_time_s=100.0e-9,
    )

    assert evidence["passed"] is False
    assert evidence["mechanisms"]["thermonuclear"] is True
    assert evidence["mechanisms"]["beam_target"] is False


def test_mjolnir_timing_evidence_can_infer_stagnation_but_marks_it():
    times_s = np.arange(90.0, 121.0, 1.0) * 1.0e-9
    history = {
        "times_s": times_s,
        "dY_thermo": _gaussian(times_s, 100.0, 1.5, 1.0e8),
        "dY_bt": _gaussian(times_s, 105.0, 1.0, 5.0e7),
    }

    evidence = mjolnir_neutron_timing_evidence_from_history(history)

    assert evidence["passed"] is True
    assert (
        evidence["details"]["stagnation_time_inferred_from_thermonuclear_peak"]
        is True
    )


def test_pf1000_spatial_pinch_target_metadata():
    target = pf1000_spatial_pinch_targets()

    assert target["target_id"] == "pf1000_spatial_pinch_2006_scholz"
    assert target["device"] == "PF-1000"
    assert target["validation_tier"] == 4
    assert target["source"] == "KnowledgeReference/scholz-2006-pf1000-mega-joule.md"
    assert target["source_lines"]["density_proxy_diagnostic"] == "333-346"
    assert target["source_lines"]["radiating_pinch_geometry"] == "375-383"
    assert target["shot_context"]["fill_pressure_hPa"] == 4.0
    assert target["shot_context"]["bank_energy_kJ"] == 734.0
    assert target["shot_context"]["current_MA"] == 1.66
    geometry = target["radiating_pinch_geometry"]
    assert geometry["minimum_diameter_mm"] == 5.0
    assert geometry["radiating_length_cm"] == 5.0
    assert geometry["dense_sphere_lifetime_ns_range"] == [30.0, 50.0]


def test_pf1000_spatial_pinch_geometry_evidence_covers_density_only():
    geometry = {
        "has_radiating_region": True,
        "diameter_mm": 5.2,
        "length_cm": 4.8,
        "diagnostic_role": "density_proxy_bremsstrahlung_spatial_geometry",
    }

    evidence = pf1000_spatial_pinch_evidence_from_geometry(geometry)

    assert evidence["passed"] is True
    assert evidence["diagnostics"]["density"] is True
    assert evidence["details"]["missing_for_full_tier4"] == [
        "magnetic_field",
        "temperature",
    ]

    tiers = {tier.level: tier for tier in validation_tier_report({
        "spatial_validation": evidence,
    })}
    assert tiers[4].status == "not_validated"


def test_pf1000_spatial_pinch_geometry_evidence_fails_large_mismatch():
    geometry = {
        "has_radiating_region": True,
        "diameter_mm": 20.0,
        "length_cm": 1.0,
    }

    evidence = pf1000_spatial_pinch_evidence_from_geometry(geometry)

    assert evidence["passed"] is False
    assert evidence["diagnostics"]["density"] is False


def test_pf1000_interferometry_density_target_metadata():
    target = pf1000_interferometry_density_targets()

    assert target["target_id"] == "pf1000_interferometry_density_2024_malir"
    assert target["device"] == "PF-1000"
    assert target["validation_tier"] == 4
    assert target["source"] == "KnowledgeReference/malir-2024-interferometry-dpf.md"
    assert target["source_lines"]["device_and_diagnostic"] == "190-205"
    assert target["source_lines"]["density_profile_features"] == "331-348"
    assert target["shot_context"]["shot_13328"]["fill_pressure_Torr"] == 0.75
    assert target["density_profile_targets"]["shot_13328"]["peak_density_cm3"] == 2.0e18
    assert target["density_profile_targets"]["shot_13317"]["peak_density_cm3"] == 2.5e18
    assert target["uncertainty"]["relative_error_far_from_axis"] == 0.20


def test_pf1000_interferometry_density_evidence_covers_density_only():
    radius_cm = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    density_cm3 = np.array([0.4e18, 1.1e18, 2.05e18, 1.4e18, 0.7e18])

    evidence = pf1000_interferometry_density_evidence_from_profile(
        radius_cm,
        density_cm3,
        shot="13328",
    )

    assert evidence["passed"] is True
    assert evidence["diagnostics"]["density"] is True
    assert evidence["details"]["shot_key"] == "shot_13328"
    assert evidence["details"]["missing_for_full_tier4"] == [
        "magnetic_field",
        "temperature",
    ]

    tiers = {tier.level: tier for tier in validation_tier_report({
        "spatial_validation": evidence,
    })}
    assert tiers[4].status == "not_validated"


def test_pf1000_interferometry_density_evidence_rejects_wrong_peak():
    radius_cm = np.array([0.0, 0.5, 1.0])
    density_cm3 = np.array([0.1e18, 0.5e18, 0.4e18])

    evidence = pf1000_interferometry_density_evidence_from_profile(
        radius_cm,
        density_cm3,
        shot="13328",
    )

    assert evidence["passed"] is False
    assert evidence["diagnostics"]["density"] is False


def test_llnl_em_fluctuation_target_metadata():
    target = llnl_12kj_em_fluctuation_targets()

    assert target["target_id"] == "llnl_12kj_em_fluctuation_2014_schmidt"
    assert target["validation_tier"] == 4
    assert target["source"].startswith("KnowledgeReference/")
    assert target["source_lines"]["rf_probe_setup"] == "120-122"
    assert target["frequency_targets"]["high_quality_pinch_band_GHz"] == [3.0, 4.0]
    assert target["field_context"]["simulated_pinch_field_T"] == [10.0, 40.0]


def test_llnl_em_fluctuation_evidence_detects_3_to_4_ghz_band():
    sample_rate_hz = 40.0e9
    times_s = np.arange(0.0, 40.0e-9, 1.0 / sample_rate_hz)
    signal = np.sin(2.0 * np.pi * 3.5e9 * times_s)

    evidence = llnl_12kj_em_fluctuation_evidence_from_signal(times_s, signal)

    assert evidence["passed"] is True
    assert evidence["diagnostics"]["magnetic_field"] is True
    assert 3.0 <= evidence["details"]["dominant_frequency_GHz"] <= 4.0

    tiers = {tier.level: tier for tier in validation_tier_report({
        "spatial_validation": evidence,
    })}
    assert tiers[4].status == "not_validated"


def test_llnl_em_fluctuation_evidence_rejects_low_frequency_signal():
    sample_rate_hz = 40.0e9
    times_s = np.arange(0.0, 40.0e-9, 1.0 / sample_rate_hz)
    signal = np.sin(2.0 * np.pi * 1.0e9 * times_s)

    evidence = llnl_12kj_em_fluctuation_evidence_from_signal(times_s, signal)

    assert evidence["passed"] is False
    assert evidence["diagnostics"]["magnetic_field"] is False


def test_uofsi_argon_temperature_target_metadata():
    target = uofsi_argon_temperature_targets()

    assert target["target_id"] == "uofsi_argon_temperature_thesis_2020"
    assert target["device"] == "UofS-I DPF"
    assert target["shot_context"]["capacitance_uF"] == 5.0
    assert target["shot_context"]["charging_voltage_kV"] == 20.0
    assert target["current_waveform_targets"]["lee_mass_factor_axial"] == 0.046
    assert target["phase_timing"]["axial_acceleration_duration_us"] == 1.15
    assert target["temperature_targets"]["argon_electron_temperature_average_keV"] == 5.7
    assert target["temperature_targets"]["argon_electron_temperature_uncertainty_keV"] == 0.7
    assert target["uncertainty"]["electron_density_assumed_not_measured"] is True
    assert "same_scope_magnetic_field_measurement" in target["missing_for_full_tier4"]


def test_dpf_pinch_temperature_target_metadata():
    target = dpf_pinch_temperature_targets()

    assert target["target_id"] == "dpf_pinch_temperature_review_regime"
    assert target["model_role"] == "kr_regime_target"
    assert target["source"].startswith("KnowledgeReference/")
    assert target["source_lines"]["pinch_density_temperature"] == "354-362"
    assert target["source_lines"]["xray_temperature"] == "368-372"
    assert target["temperature_targets"]["pinch_temperature_min_keV"] == 1.0
    assert target["temperature_targets"]["thermal_xray_temperature_range_keV"] == [
        0.4,
        4.0,
    ]
    assert target["context_targets"]["compressed_magnetic_field_min_T"] == 100.0


def test_dpf_pinch_temperature_evidence_covers_temperature_only():
    evidence = dpf_pinch_temperature_evidence(ion_temperature_keV=1.2)

    assert evidence["passed"] is True
    assert evidence["diagnostics"]["temperature"] is True
    assert evidence["details"]["component_passes"]["ion_temperature_keV"] is True
    assert evidence["details"]["missing_for_full_tier4"] == [
        "density",
        "magnetic_field",
    ]

    tiers = {tier.level: tier for tier in validation_tier_report({
        "spatial_validation": evidence,
    })}
    assert tiers[4].status == "not_validated"


def test_dpf_pinch_temperature_evidence_rejects_out_of_regime_temperature():
    evidence = dpf_pinch_temperature_evidence(ion_temperature_keV=0.05)

    assert evidence["passed"] is False
    assert evidence["diagnostics"]["temperature"] is False
