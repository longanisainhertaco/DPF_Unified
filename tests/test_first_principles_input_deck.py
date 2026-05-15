from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dpf.first_principles import (
    FirstPrinciplesInputDeck,
    load_first_principles_input_deck,
    pf1000_akel_16kv_engineering_deck,
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
    assert deck.startup.mode == "source_backed_profile"
    assert deck.startup.can_support_whole_shot_acceptance is False
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
    assert deck.circuit.capacitance_F == pytest.approx(1.332e-3)
    assert deck.circuit.voltage_V == pytest.approx(1.6e4)
    assert deck.circuit.inductance_H == pytest.approx(25.0e-9)
    assert deck.circuit.resistance_ohm == pytest.approx(6.1e-3)
    assert deck.gas.pressure_Pa == pytest.approx(1.2 * 133.32236842105263)
    assert deck.startup.can_support_whole_shot_acceptance is False
    assert "surface_flashover_closure" in deck.startup.missing_channels
    assert deck.validation_targets[0].status == "blocked_by_review"
    assert deck.scientific_status == "engineering_candidate_not_validation"
    assert deck.source_references[0].path == (
        "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    )
