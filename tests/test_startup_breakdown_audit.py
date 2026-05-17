from __future__ import annotations

import pytest

from dpf.first_principles import (
    build_candidate_startup_breakdown_audit,
    pf1000_akel_16kv_engineering_deck,
)
from dpf.first_principles.runner import FirstPrinciples3DDeck
from dpf.first_principles.startup_bvp import build_startup_bvp_packet


def test_candidate_startup_breakdown_audit_computes_pf1000_without_promotion() -> None:
    audit = build_candidate_startup_breakdown_audit(
        device={
            "device_name": "PF-1000/Akel",
            "anode_radius_m": 0.1155,
            "cathode_radius_m": 0.16,
            "anode_length_m": 0.48,
            "insulator_length_m": 0.0,
        },
        gas={"species": "D", "pressure_Pa": 1.2 * 133.32236842105263, "temperature_K": 300.0},
        circuit={"voltage_V": 16.0e3, "initial_current_A": 0.0},
        startup={"initial_magnetic_field_T": (0.0, 0.0, 0.0)},
    )

    assert audit["status"] == "candidate_civ_paschen_breakdown_audit_engineering_only"
    assert audit["can_support_first_principles_acceptance"] is False
    assert audit["can_support_whole_shot_acceptance"] is False
    assert audit["input_summary"]["gas_species_effective"] == "D2"
    assert audit["input_summary"]["paschen_path_policy"] == (
        "radial_gap_fallback_because_insulator_length_missing_or_zero"
    )
    assert audit["breakdown"]["applied_voltage_V"] == pytest.approx(16.0e3)
    assert audit["breakdown"]["breakdown_time_s"] > 0.0
    assert audit["liftoff"]["candidate_liftoff_delay_s"] > audit["breakdown"]["breakdown_time_s"]
    assert audit["liftoff"]["can_support_handoff_acceptance"] is False


def test_startup_bvp_packet_treats_breakdown_audit_as_candidate_only() -> None:
    audit = build_candidate_startup_breakdown_audit(
        device={
            "device_name": "PF-1000/Akel",
            "anode_radius_m": 0.1155,
            "cathode_radius_m": 0.16,
            "anode_length_m": 0.48,
            "insulator_length_m": 0.0,
        },
        gas={"species": "D2", "pressure_Pa": 160.0, "temperature_K": 300.0},
        circuit={"voltage_V": 16.0e3, "initial_current_A": 0.0},
        startup={"initial_magnetic_field_T": (0.0, 0.0, 0.0)},
    )

    packet = build_startup_bvp_packet(
        {
            "mode": "source_backed_end_rundown_sheath",
            "evidence_status": "engineering_candidate_not_whole_shot",
            "can_support_whole_shot_acceptance": False,
        },
        candidate_breakdown_audit=audit,
    )

    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["whole_shot_startup_blocked"] is True
    assert "candidate_civ_paschen_breakdown_audit" in packet["candidate_input_channels"]
    assert "candidate_civ_paschen_liftoff_delay" in packet["candidate_input_channels"]
    assert packet["startup_channel_status"]["breakdown_or_flashover_model"] == (
        "candidate_input_only_not_acceptance"
    )
    assert packet["startup_channel_status"]["sheath_liftoff_and_handoff_interval"] == (
        "candidate_input_only_not_acceptance"
    )
    assert packet["candidate_breakdown_audit"]["status"] == (
        "candidate_civ_paschen_breakdown_audit_engineering_only"
    )


def test_package_runner_attaches_startup_breakdown_audit_to_pf1000_deck() -> None:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    startup_packet = FirstPrinciples3DDeck.from_deck(deck).startup_packet()

    assert startup_packet["whole_shot_startup_blocked"] is True
    assert startup_packet["can_support_first_principles_acceptance"] is False
    assert startup_packet["candidate_breakdown_audit"]["status"] == (
        "candidate_civ_paschen_breakdown_audit_engineering_only"
    )
    assert startup_packet["candidate_breakdown_audit"]["input_summary"][
        "device_name"
    ] == "PF-1000/Akel shot 12581 engineering candidate"
    assert startup_packet["startup_channel_status"]["breakdown_or_flashover_model"] == (
        "candidate_input_only_not_acceptance"
    )
