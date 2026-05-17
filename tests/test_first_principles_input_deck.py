from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dpf.first_principles import (
    FirstPrinciplesInputDeck,
    compact_chinese_dpf_engineering_deck,
    gv_verified_engineering_deck,
    gv_verified_engineering_decks,
    ir_mpf_100_engineering_deck,
    load_first_principles_input_deck,
    may15_second_scope_engineering_decks,
    pf1000_akel_16kv_engineering_deck,
    willenborg_hendricks_engineering_deck,
)

SOURCE_SHA = "a" * 64
TARGET_SHA = "b" * 64


def minimal_3d_deck_payload() -> dict[str, object]:
    return {
        "deck_id": "minimal-3d",
        "description": "Minimal package-native 3D first-principles deck",
        "device_geometry": {
            "coordinate_system": "cartesian_3d",
            "anode_radius_m": 0.01,
            "cathode_radius_m": 0.03,
            "anode_length_m": 0.05,
            "cathode_length_m": 0.10,
            "source_reference_ids": ["pf1000_geometry"],
        },
        "circuit": {
            "capacitance_F": 1.332e-3,
            "initial_voltage_V": 16_000.0,
            "static_inductance_H": 25.0e-9,
            "static_resistance_ohm": 2.3e-3,
            "source_reference_ids": ["pf1000_geometry"],
        },
        "gas": {
            "fill_pressure_Pa": 466.6,
            "fill_temperature_K": 300.0,
            "species": [
                {
                    "name": "D2",
                    "atomic_mass_amu": 2.014,
                    "charge_state": 0.0,
                    "number_fraction": 1.0,
                    "source_reference_ids": ["pf1000_geometry"],
                }
            ],
        },
        "grid": {
            "dimensionality": "3d",
            "coordinate_system": "cartesian",
            "shape": [4, 4, 8],
            "spacing_m": [0.001, 0.001, 0.001],
            "field_layout": "staggered_yee",
        },
        "startup_policy": {
            "initialization": "source_backed_profile",
            "magnetic_seed": "none",
            "preionization_fraction": 0.0,
            "require_source_backed_initial_conditions": True,
            "source_reference_ids": ["pf1000_geometry"],
        },
        "closure_policy": {
            "field_equations": "resistive_mhd",
            "resistivity": "spitzer",
            "equation_of_state": "ideal_gas",
            "radiation": "bremsstrahlung",
            "ionization": "fixed_charge_state",
            "source_reference_ids": ["pf1000_geometry"],
        },
        "boundary_policy": {
            "pml_cells": 1,
            "pml_strength": 0.25,
            "particle_absorption_enabled": True,
            "open_boundary": True,
            "conductor_mask_status": "not_supplied",
            "source_reference_ids": ["pf1000_geometry"],
        },
        "diagnostic_policy": {
            "sample_interval_steps": 2,
            "record_current": True,
            "record_voltage": True,
            "record_fields": True,
            "record_energy_balance": True,
            "output_format": "json",
        },
        "source_references": [
            {
                "source_id": "pf1000_geometry",
                "path": "KnowledgeReference/pf1000/geometry.md",
                "sha256": SOURCE_SHA,
                "title": "PF-1000 geometry source packet",
                "source_scope": "pf1000_16kv_2021_akel",
            }
        ],
        "validation_target_references": [
            {
                "target_id": "pf1000_current_trace",
                "observable": "current_waveform",
                "validation_scope": "pf1000_16kv_2021_akel",
                "source_reference_id": "pf1000_geometry",
                "target_path": "KnowledgeReference/pf1000/current_trace.json",
                "target_sha256": TARGET_SHA,
                "status": "blocked_by_review",
            }
        ],
    }


def test_minimal_3d_first_principles_deck_loads_from_dict_json_and_file(tmp_path) -> None:
    payload = minimal_3d_deck_payload()

    deck = FirstPrinciplesInputDeck.from_dict(payload)

    assert deck.schema_version == "dpf.first_principles.input_deck.v1"
    assert deck.grid.dimensionality == "3d"
    assert deck.grid.shape == (4, 4, 8)
    assert deck.diagnostics.history_stride == 2
    assert deck.diagnostics.max_step_results == 256
    assert deck.startup.mode == "source_backed_profile"
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert deck.closures.circuit_udpf_mode == "lagged_volume_j_dot_e"
    assert deck.boundaries.pml_cells == 1
    assert deck.boundaries.pml_strength == pytest.approx(0.25)
    assert deck.boundaries.particle_absorption_enabled is True
    assert deck.boundaries.conductor_mask_mode == "none"
    assert deck.boundaries.can_support_first_principles_acceptance is False
    assert deck.source_references[0].path == "KnowledgeReference/pf1000/geometry.md"
    assert deck.source_references[0].sha256 == SOURCE_SHA
    assert deck.validation_target_references[0].target_sha256 == TARGET_SHA

    json_round_trip = FirstPrinciplesInputDeck.from_json(deck.to_json())
    assert json_round_trip == deck

    deck_path = tmp_path / "deck.json"
    deck_path.write_text(json.dumps(payload), encoding="utf-8")
    file_loaded = load_first_principles_input_deck(deck_path)
    assert file_loaded == deck

    mapping_loaded = load_first_principles_input_deck(deck.to_dict())
    assert mapping_loaded == deck


def test_first_principles_deck_rejects_top_level_reduced_model_authority_fields() -> None:
    payload = minimal_3d_deck_payload()
    payload["lee_fit_factors"] = {"fc": 0.7, "fm": 0.1}

    with pytest.raises(ValidationError, match="Reduced-model authority fields"):
        FirstPrinciplesInputDeck.from_dict(payload)


def test_first_principles_deck_rejects_nested_snowplow_closure_fields() -> None:
    payload = minimal_3d_deck_payload()
    closure_policy = payload["closure_policy"]
    assert isinstance(closure_policy, dict)
    closure_policy["snowplow_closure"] = {"current_factor": 0.8}

    with pytest.raises(ValidationError, match="Reduced-model authority fields"):
        FirstPrinciplesInputDeck.from_dict(payload)


def test_first_principles_deck_rejects_unknown_source_reference_ids() -> None:
    payload = minimal_3d_deck_payload()
    payload["startup_policy"] = {
        "initialization": "source_backed_profile",
        "source_reference_ids": ["missing-source"],
    }

    with pytest.raises(ValidationError, match="unknown source_reference_id"):
        FirstPrinciplesInputDeck.from_dict(payload)


def test_startup_policy_preserves_source_truth_blocker_fields() -> None:
    payload = minimal_3d_deck_payload()
    payload["startup_policy"] = {
        "mode": "source_backed_end_rundown_sheath",
        "background_density_m3": 6.7e22,
        "electron_temperature_K": 7.2e5,
        "ion_temperature_K": 7.2e5,
        "initial_electric_field_V_m": [2.0e5, 0.0, 0.0],
        "initial_magnetic_field_T": [0.0, 0.0, 0.4],
        "particle_weight": 2.0e8,
        "evidence_status": "engineering_candidate_not_whole_shot",
        "source_scope": "end_of_rundown_sheath",
        "can_support_whole_shot_acceptance": False,
        "required_channels": ["current_density", "electric_field", "magnetic_field"],
        "missing_channels": ["breakdown_model", "sheath_liftoff"],
        "source_reference_ids": ["pf1000_geometry"],
    }

    deck = FirstPrinciplesInputDeck.from_dict(payload)

    assert deck.startup.mode == "source_backed_end_rundown_sheath"
    assert deck.startup.background_density_m3 == pytest.approx(6.7e22)
    assert deck.startup.initial_magnetic_field_T == (0.0, 0.0, 0.4)
    assert deck.startup.evidence_status == "engineering_candidate_not_whole_shot"
    assert deck.startup.source_scope == "end_of_rundown_sheath"
    assert deck.startup.whole_shot_startup_blocked is True
    assert deck.startup.required_channels == (
        "current_density",
        "electric_field",
        "magnetic_field",
    )
    assert deck.startup.missing_channels == ("breakdown_model", "sheath_liftoff")


def test_seeded_layer_startup_cannot_claim_whole_shot_acceptance() -> None:
    payload = minimal_3d_deck_payload()
    payload["startup_policy"] = {
        "mode": "seeded_layer",
        "can_support_whole_shot_acceptance": True,
        "source_reference_ids": ["pf1000_geometry"],
    }

    with pytest.raises(ValidationError, match="cannot support accepted whole-shot"):
        FirstPrinciplesInputDeck.from_dict(payload)


def test_imported_pic_startup_requires_review_before_acceptance() -> None:
    payload = minimal_3d_deck_payload()
    payload["startup_policy"] = {
        "mode": "imported_pic_sheath_state",
        "evidence_status": "candidate",
        "can_support_whole_shot_acceptance": True,
        "source_reference_ids": ["pf1000_geometry"],
    }

    with pytest.raises(ValidationError, match="only after evidence_status='reviewed'"):
        FirstPrinciplesInputDeck.from_dict(payload)


def test_surface_breakdown_bvp_with_missing_channels_remains_blocked() -> None:
    payload = minimal_3d_deck_payload()
    payload["startup_policy"] = {
        "mode": "surface_breakdown_bvp",
        "evidence_status": "candidate",
        "can_support_whole_shot_acceptance": True,
        "missing_channels": ["surface_flashover_closure"],
        "source_reference_ids": ["pf1000_geometry"],
    }

    deck = FirstPrinciplesInputDeck.from_dict(payload)

    assert deck.startup.mode == "surface_breakdown_bvp"
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert deck.startup.whole_shot_startup_blocked is True


def test_pf1000_akel_engineering_deck_is_source_scoped_and_nonpromoting() -> None:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=2, shape=(5, 5, 5))

    assert deck.deck_id == "pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate"
    assert deck.device.name == "PF-1000/Akel shot 12581 engineering candidate"
    assert deck.device.anode_radius_m == pytest.approx(0.1155)
    assert deck.device.cathode_radius_m == pytest.approx(0.16)
    assert deck.device.anode_length_m == pytest.approx(0.48)
    assert deck.device.insulator_length_m == pytest.approx(0.085)
    assert deck.device.cathode_rod_count == 12
    assert deck.device.cathode_rod_diameter_m == pytest.approx(0.080)
    assert deck.device.insulator_material == "alumina"
    assert deck.circuit.capacitance_F == pytest.approx(1.332e-3)
    assert deck.circuit.voltage_V == pytest.approx(1.6e4)
    assert deck.circuit.inductance_H == pytest.approx(25.0e-9)
    assert deck.circuit.resistance_ohm == pytest.approx(6.1e-3)
    assert deck.circuit.initial_charge_C == pytest.approx(0.0)
    assert deck.gas.pressure_Pa == pytest.approx(1.2 * 133.32236842105263)
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert deck.closures.circuit_udpf_mode == "lagged_volume_j_dot_e"
    assert deck.boundaries.pml_cells == 1
    assert deck.boundaries.particle_absorption_enabled is True
    assert deck.boundaries.conductor_mask_status == "candidate_geometry_mask"
    assert deck.boundaries.conductor_mask_mode == "pf1000_rod_hollow_projection"
    assert "surface_flashover_closure" in deck.startup.missing_channels
    assert deck.validation_targets[0].status == "blocked_by_review"
    assert deck.scientific_status == "engineering_candidate_not_validation"
    assert deck.source_references[0].path == (
        "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    )


def test_may15_second_scope_engineering_decks_are_runnable_nonpromoting_decks() -> None:
    decks = may15_second_scope_engineering_decks(n_steps=1, shape=(5, 5, 5))

    assert [deck.deck_id for deck in decks] == [
        "ir_mpf_100_20kv_1p9torr_engineering_candidate",
        "compact_chinese_dpf_20kv_580pa_engineering_candidate",
        "willenborg_hendricks_19kv_1torr_engineering_candidate",
    ]

    for deck in decks:
        assert deck.scientific_status == "engineering_candidate_not_validation"
        assert deck.startup.can_support_whole_shot_acceptance is False
        assert deck.boundaries.conductor_mask_status == "candidate_geometry_mask"
        assert deck.boundaries.conductor_mask_mode == "axisymmetric_coaxial_projection"
        assert deck.validation_targets
        assert deck.source_references[0].sha256
        assert FirstPrinciplesInputDeck.from_json(deck.to_json()) == deck


def test_gv_verified_engineering_deck_uses_verified_shot_values_without_promoting() -> None:
    deck = gv_verified_engineering_deck(
        "pf24_krakow_16092202",
        n_steps=1,
        shape=(5, 5, 5),
    )

    assert deck.deck_id == "gv_pf24_krakow_16092202_engineering_candidate"
    assert deck.device.anode_radius_m == pytest.approx(31.0e-3)
    assert deck.device.cathode_radius_m == pytest.approx(49.0e-3)
    assert deck.device.anode_length_m == pytest.approx(172.0e-3)
    assert deck.device.insulator_length_m == pytest.approx(40.0e-3)
    assert deck.circuit.capacitance_F == pytest.approx(115.2e-6)
    assert deck.circuit.inductance_H == pytest.approx(21.0e-9)
    assert deck.circuit.resistance_ohm == pytest.approx(22.0e-3)
    assert deck.circuit.voltage_V == pytest.approx(16.0e3)
    assert deck.circuit.initial_charge_C == pytest.approx(0.0)
    assert deck.gas.pressure_Pa == pytest.approx(1.1 * 133.32236842105263)
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert "neutron_mechanism_separation" in deck.startup.missing_channels
    assert deck.validation_targets[0].status == (
        "user_verified_waveform_candidate_not_comparator_bound"
    )
    assert deck.validation_targets[1].status == (
        "reduced_model_baseline_not_first_principles_closure"
    )
    assert "Downloads/GV/Gvinp-PF-24-KRAKOW-16092202.inp" in (
        deck.source_references[0].path
    )
    assert deck.source_references[1].role == "gv_verified_workbook_waveform_candidate"
    assert FirstPrinciplesInputDeck.from_json(deck.to_json()) == deck


def test_gv_verified_engineering_decks_cover_all_unique_verified_shots() -> None:
    decks = gv_verified_engineering_decks(n_steps=1, shape=(5, 5, 5))

    assert len(decks) == 8
    assert {deck.deck_id for deck in decks} >= {
        "gv_lpp_ff1_05_23_16_1_engineering_candidate",
        "gv_pf24_krakow_16092202_engineering_candidate",
        "gv_pf360_20140122_7_engineering_candidate",
        "gv_gemini_rog_i005_20130716_engineering_candidate",
        "gv_onesys_rog01004_20051208_engineering_candidate",
    }
    for deck in decks:
        assert deck.scientific_status == "engineering_candidate_not_validation"
        assert deck.startup.can_support_whole_shot_acceptance is False
        assert deck.validation_targets


def test_ir_mpf_100_engineering_deck_uses_validated_source_values() -> None:
    deck = ir_mpf_100_engineering_deck(n_steps=1, shape=(5, 5, 5))

    assert deck.device.anode_radius_m == pytest.approx(6.25e-2)
    assert deck.device.cathode_radius_m == pytest.approx(1.02e-1)
    assert deck.device.anode_length_m == pytest.approx(2.2e-1)
    assert deck.device.insulator_length_m == pytest.approx(5.0e-2)
    assert deck.circuit.capacitance_F == pytest.approx(144.0e-6)
    assert deck.circuit.voltage_V == pytest.approx(2.0e4)
    assert deck.circuit.inductance_H == pytest.approx(120.0e-9)
    assert deck.circuit.resistance_ohm == pytest.approx(5.0e-3)
    assert deck.circuit.initial_charge_C == pytest.approx(0.0)
    assert deck.gas.pressure_Pa == pytest.approx(1.9 * 133.32236842105263)
    assert "measured_current_waveform_digitization" in deck.startup.missing_channels
    assert deck.source_references[0].path == "KnowledgeReference/original-research-f7894f85.md"


def test_compact_chinese_dpf_engineering_deck_marks_inferred_circuit_nonaccepting() -> None:
    deck = compact_chinese_dpf_engineering_deck(n_steps=1, shape=(5, 5, 5))

    assert deck.device.anode_radius_m == pytest.approx(17.0e-3)
    assert deck.device.cathode_radius_m == pytest.approx(40.0e-3)
    assert deck.device.insulator_length_m == pytest.approx(40.0e-3)
    assert deck.circuit.capacitance_F == pytest.approx(40.0e-6)
    assert deck.circuit.voltage_V == pytest.approx(2.0e4)
    assert deck.circuit.inductance_H == pytest.approx(100.0e-9)
    assert deck.circuit.initial_charge_C == pytest.approx(0.0)
    assert deck.gas.pressure_Pa == pytest.approx(580.0)
    assert "translation_review" in deck.startup.missing_channels
    assert "inferred_circuit_inductance" in deck.startup.source_scope


def test_willenborg_hendricks_engineering_deck_uses_startup_bvp_mode_but_stays_blocked() -> None:
    deck = willenborg_hendricks_engineering_deck(n_steps=1, shape=(5, 5, 5))

    assert deck.startup.mode == "surface_breakdown_bvp"
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert deck.device.anode_radius_m == pytest.approx((1.78 * 0.0254) / 2.0)
    assert deck.circuit.capacitance_F == pytest.approx(43.5e-6)
    assert deck.circuit.voltage_V == pytest.approx(1.9e4)
    assert deck.circuit.inductance_H == pytest.approx(100.0e-9)
    assert deck.gas.pressure_Pa == pytest.approx(133.32236842105263)
    assert "surface_flashover_equations" in deck.startup.missing_channels
    assert "modern_device_scope_review" in deck.startup.missing_channels
