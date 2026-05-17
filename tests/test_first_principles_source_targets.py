from __future__ import annotations

import pytest

from dpf.first_principles import (
    build_dimensionality_handoff_packet,
    build_generalized_dpf_machine_packet,
    build_mechanism_separated_neutron_packet,
    build_startup_bvp_packet,
    gv_verified_shot_targets,
    may15_second_scope_engineering_decks,
    may15_user_validated_source_targets,
    may16_validated_thesis_source_targets,
    run_first_principles_3d_deck,
)


def test_may15_user_validated_targets_are_source_accepted_but_nonpromoting() -> None:
    packet = may15_user_validated_source_targets()

    assert packet["source_status"] == "all_eight_user_verified_validated_research_sources"
    assert packet["accepted_for_whole_shot_first_principles"] is False
    assert packet["source_ingestion_ledger"]["promoted_count"] == 6
    assert packet["source_ingestion_ledger"]["already_represented_count"] == 2
    assert packet["source_ingestion_ledger"]["parity_failed_count"] == 0

    deck_targets = packet["device_deck_targets"]
    assert set(deck_targets) == {
        "ir_mpf_100_salehizadeh_2012",
        "compact_chinese_dpf_2018",
        "willenborg_hendricks_1977_startup_design",
    }

    for target in deck_targets.values():
        assert target["accepted_for_validation"] is False
        assert "missing_for_first_principles_acceptance" in target


def test_gv_verified_shot_targets_are_waveform_candidates_not_authority() -> None:
    packet = gv_verified_shot_targets()

    assert packet["source_status"] == (
        "user_verified_local_download_not_knowledge_reference_promoted"
    )
    assert packet["accepted_for_whole_shot_first_principles"] is False
    assert packet["shot_count"] == 8
    assert "current_kA" in packet["reduced_model_output_columns"]

    targets = packet["shot_targets"]
    pf24 = targets["pf24_krakow_16092202"]
    assert pf24["geometry_mm"]["anode_radius"] == pytest.approx(31.0)
    assert pf24["circuit"]["capacitance_uF"] == pytest.approx(115.2)
    assert pf24["gas"]["fitted_pressure_torr"] == pytest.approx(1.1)
    assert pf24["experimental_waveform"]["columns"]["time_us"] == "L"
    assert pf24["experimental_waveform"]["columns"]["current_kA"] == "M"
    assert pf24["experimental_waveform"]["status"] == (
        "user_verified_workbook_candidate_not_comparator_bound"
    )
    assert pf24["gv_baseline"]["accepted_as_first_principles_closure"] is False
    assert "startup_bvp" in pf24["missing_for_first_principles_acceptance"]

    pf360 = targets["pf360_20140122_7"]
    assert pf360["experimental_waveform"]["columns"]["raw_time_us"] == "AC"
    assert pf360["experimental_waveform"]["columns"]["raw_current_kA"] == "AD"


def test_may16_validated_thesis_targets_are_nonpromoting_source_candidates() -> None:
    packet = may16_validated_thesis_source_targets()

    assert packet["source_status"] == (
        "all_seven_user_verified_validated_documents_promoted_to_knowledge_reference"
    )
    assert packet["document_count"] == 7
    assert packet["accepted_for_whole_shot_first_principles"] is False
    assert packet["source_ingestion_ledger"]["promoted_count"] == 7
    assert packet["source_ingestion_ledger"]["failed_count"] == 0
    assert "typed target extraction with units and uncertainty" in packet["what_it_does_not_close"]

    targets = packet["document_targets"]
    assert set(targets) == {
        "arwinder_2015_comparative_pf_machines",
        "talebitaher_2012_nx2_coded_aperture_imaging",
        "saw_1990_current_stepped_z_pinch",
        "serban_1995_anode_geometry_focus_characteristics",
        "rafique_2000_deuterium_pf_compression_radiation",
        "verma_2010_miniature_repetitive_pf_neutron_source",
        "avaria_2022_bayesian_sheath_diagnostics",
    }

    for target in targets.values():
        assert target["accepted_for_validation"] is False
        assert target["accepted_for_whole_shot_first_principles"] is False
        assert target["source_status"] == (
            "user_verified_validated_knowledge_reference_promoted"
        )
        assert target["knowledge_reference"]["markdown"].startswith("KnowledgeReference/")
        assert target["target_extraction_required"]


def test_may16_targets_map_to_first_principles_blockers_without_authority() -> None:
    packet = may16_validated_thesis_source_targets()
    targets = packet["document_targets"]

    saw = targets["saw_1990_current_stepped_z_pinch"]
    assert saw["text_status"] == "ocr_sidecar_created_from_scanned_pdf"
    assert "FP-5" in saw["useful_gate_ids"]
    assert "DPF geometry or whole-shot DPF validation" in saw["not_authority_for"]
    assert any("gamma-varying model equations" in item for item in saw["target_extraction_required"])

    talebitaher = targets["talebitaher_2012_nx2_coded_aperture_imaging"]
    assert "FP-12" in talebitaher["useful_gate_ids"]
    assert "FP-13" in talebitaher["useful_gate_ids"]
    assert any("beryllium activation" in item for item in talebitaher["candidate_facts"])
    assert any("detector geometry" in item for item in talebitaher["target_extraction_required"])

    rafique = targets["rafique_2000_deuterium_pf_compression_radiation"]
    assert "FP-12" in rafique["useful_gate_ids"]
    assert any("80 keV to 250 keV" in item for item in rafique["candidate_facts"])
    assert "accepted PF-1000/Akel neutron mechanism" in rafique["not_authority_for"]

    avaria = targets["avaria_2022_bayesian_sheath_diagnostics"]
    assert "FP-5" in avaria["useful_gate_ids"]
    assert "FP-11" in avaria["useful_gate_ids"]
    assert any("Stark-broadened H-alpha" in item for item in avaria["candidate_facts"])
    assert "PF-1000/Akel same-scope density validation" in avaria["not_authority_for"]

    coverage = packet["gate_coverage_candidates"]
    assert "avaria_2022_bayesian_sheath_diagnostics" in coverage["FP-5_startup_bvp"]
    assert "talebitaher_2012_nx2_coded_aperture_imaging" in coverage["FP-13_comparator_uq"]
    assert "arwinder_2015_comparative_pf_machines" in coverage["FP-15_generalization"]


def test_ir_mpf_100_source_deck_values_are_typed_and_source_scoped() -> None:
    target = may15_user_validated_source_targets()["device_deck_targets"][
        "ir_mpf_100_salehizadeh_2012"
    ]

    assert target["source"] == "KnowledgeReference/original-research-f7894f85.md"
    assert target["source_lines"]["bank_and_current"] == "108-160"
    assert target["circuit"]["capacitance_F"] == pytest.approx(144.0e-6)
    assert target["circuit"]["capacitor_count"] == 24
    assert target["circuit"]["capacitance_each_F"] == pytest.approx(6.0e-6)
    assert target["circuit"]["maximum_voltage_V"] == pytest.approx(40.0e3)
    assert target["circuit"]["maximum_stored_energy_J"] == pytest.approx(115.0e3)
    assert target["circuit"]["theoretical_peak_current_A"] == pytest.approx(1.224e6)
    assert target["circuit"]["total_inductance_H"] == pytest.approx(120.0e-9)
    assert target["geometry"]["anode_radius_m"] == pytest.approx(6.25e-2)
    assert target["geometry"]["cathode_radius_m"] == pytest.approx(1.02e-1)
    assert target["geometry"]["insulator_length_m"] == pytest.approx(5.0e-2)
    assert target["diagnostic_geometry"][
        "silver_activation_counter_distance_from_anode_top_m"
    ] == pytest.approx(1.30)
    assert target["neutron_targets"]["yield_neutrons_per_shot_at_29kJ_1p9torr"] == pytest.approx(
        1.5e9
    )
    assert target["design_formula_context_not_closure"][
        "accepted_as_active_first_principles_closure"
    ] is False


def test_compact_chinese_dpf_source_deck_preserves_tof_and_yield_targets() -> None:
    target = may15_user_validated_source_targets()["device_deck_targets"][
        "compact_chinese_dpf_2018"
    ]

    assert target["source"] == "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md"
    assert target["source_lines"]["simulation_and_measurement"] == "180-200,210-232"
    assert target["circuit"]["capacitance_total_F"] == pytest.approx(40.0e-6)
    assert target["circuit"]["delivered_current_A_approx"] == pytest.approx(400.0e3)
    assert target["geometry"]["anode_radius_m"] == pytest.approx(17.0e-3)
    assert target["geometry"]["inner_electrode_length_m"] == pytest.approx(15.0e-2)
    assert target["geometry"]["outer_electrode_length_m"] == pytest.approx(16.0e-2)
    assert target["geometry"]["cathode_rod_count"] == 8
    assert target["geometry"]["insulator_inner_diameter_m"] == pytest.approx(36.0e-3)
    assert target["diagnostics"]["neutron_detector_angle_deg"] == pytest.approx(90.0)
    assert target["diagnostics"]["tof_2p45MeV_neutron_speed_m_s"] == pytest.approx(2.16e7)
    assert target["operating_targets"]["optimum_pressure_Pa_range"] == [550.0, 600.0]
    assert target["operating_targets"]["average_dd_neutron_yield_per_pulse_min"] == pytest.approx(
        5.0e8
    )
    assert target["operating_targets"]["neutron_pulse_fwhm_s"] == pytest.approx(40.0e-9)


def test_willenborg_hendricks_startup_target_encodes_breakdown_and_diagnostics() -> None:
    target = may15_user_validated_source_targets()["device_deck_targets"][
        "willenborg_hendricks_1977_startup_design"
    ]

    assert "design-and-construction-of-a-dense-plasma-focus-device" in target["source"]
    assert target["source_lines"]["operation_gates"] == "3367-3439"
    assert target["circuit"]["capacitance_F"] == pytest.approx(43.5e-6)
    assert target["circuit"]["capacitance_status"] == "inferred_from_three_14p5uF_capacitors"
    assert target["circuit"]["stored_energy_J_at_20kV"] == pytest.approx(8.7e3)
    assert target["geometry"]["outer_electrode_rod_count"] == 8
    assert target["gas"]["working_pressure_torr_range"] == [0.1, 10.0]
    assert target["startup_constraints"]["conditioned_insulator_required"] is True
    assert target["startup_constraints"]["conditioning_shots_max"] == 15
    assert target["startup_constraints"]["focus_delay_s_approx"] == pytest.approx(2.8e-6)
    assert target["switch"]["jitter_s"] == pytest.approx(20.0e-9)
    assert target["diagnostic_targets"]["xray_detector_model"] == "100-PIN-125"
    assert target["diagnostic_targets"]["voltage_probe_divider_ratio"] == pytest.approx(800.0)


def test_architecture_blocker_targets_keep_neutron_and_3d_authority_closed() -> None:
    targets = may15_user_validated_source_targets()["architecture_blocker_targets"]
    alegra = targets["sandia_alegra_hedp_2009"]
    gribkov = targets["gribkov_applications_2015"]

    assert alegra["accepted_for_validation"] is False
    assert alegra["blocker_facts"]["mhd_thermonuclear_neutron_yield_below_total_observed"] is True
    assert alegra["blocker_facts"]["first_principles_startup_requires_pic_breakdown_import_or_bvp"] is True
    assert alegra["numeric_context"]["dpf_current_range_A"] == [0.6e6, 1.8e6]
    assert "pic_to_mhd_startup_import" in alegra["required_capabilities"]

    assert gribkov["accepted_for_validation"] is False
    assert gribkov["mechanism_context"]["current_abruption_ps_scale"] is True
    assert gribkov["mechanism_context"]["bank_to_fast_ion_beam_efficiency_approx"] == pytest.approx(
        0.10
    )
    assert gribkov["large_machine_constraints"]["simple_clr_circuit_is_insufficient_for_very_large_dpf"] is True
    assert gribkov["large_machine_constraints"]["mhd_plus_telegraph_equations_required_for_final_stage"] is True


def test_first_principles_gates_reference_may15_validated_sources() -> None:
    startup = build_startup_bvp_packet(
        {"mode": "source_backed_end_rundown_sheath"},
    )
    dimensionality = build_dimensionality_handoff_packet(
        grid_shape=(4, 4, 4),
        run_mode="hybrid_em_pic_fluid",
        startup=startup,
    )
    neutron = build_mechanism_separated_neutron_packet(
        declared_scope="pf1000_akel_16kv_1p2torr_shot_12581",
        device_name="PF-1000/Akel",
    )
    generalization = build_generalized_dpf_machine_packet(
        declared_scope="pf1000_akel_16kv_1p2torr_shot_12581",
    )

    startup_paths = {item["path"] for item in startup["source_references"]}
    assert "KnowledgeReference/sand2009-6373-b93aec67.md" in startup_paths
    assert (
        "KnowledgeReference/"
        "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
        in startup_paths
    )
    assert "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md" in (
        startup_paths
    )
    assert startup["can_support_first_principles_acceptance"] is False

    dimensionality_paths = {
        item["path"] for item in dimensionality["source_references"]
    }
    assert "KnowledgeReference/sand2009-6373-b93aec67.md" in dimensionality_paths
    assert dimensionality["can_support_first_principles_acceptance"] is False

    neutron_paths = {item["path"] for item in neutron["source_references"]}
    assert "KnowledgeReference/sand2009-6373-b93aec67.md" in neutron_paths
    assert (
        "KnowledgeReference/"
        "open-access-proceedings-journal-of-physics-conference-series-ed196711.md"
        in neutron_paths
    )
    assert "KnowledgeReference/original-research-f7894f85.md" in neutron_paths
    assert "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md" in (
        neutron_paths
    )
    assert neutron["can_support_first_principles_acceptance"] is False

    generalization_paths = {
        item["path"] for item in generalization["source_references"]
    }
    assert "KnowledgeReference/original-research-f7894f85.md" in generalization_paths
    assert "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md" in (
        generalization_paths
    )
    assert (
        "KnowledgeReference/"
        "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
        in generalization_paths
    )
    assert generalization["can_claim_generalized_dpf_machine"] is False
    assert generalization["can_support_first_principles_acceptance"] is False


def test_may15_second_scope_decks_execute_as_engineering_candidates() -> None:
    for deck in may15_second_scope_engineering_decks(n_steps=1, shape=(4, 4, 4)):
        result = run_first_principles_3d_deck(deck)

        assert result.reduced_models_used is False
        assert result.can_support_first_principles_acceptance is False
        assert result.manifest["validation_status"] == "not_validation"
        assert result.manifest["metadata"]["deck"]["geometry"]["device_name"] == (
            deck.device.name
        )
        assert result.telemetry["startup"]["whole_shot_startup_blocked"] is True
        assert result.telemetry["generalization"]["can_claim_generalized_dpf_machine"] is False
