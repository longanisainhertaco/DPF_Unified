from __future__ import annotations

from dpf.first_principles.circuit_power_port import build_circuit_power_port_packet
from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
)

PF1000_SCOPE = "pf1000_full_energy_27_to_40_kv"


def test_phase4b_bank_parameters_alone_remain_blocked() -> None:
    packet = build_circuit_power_port_packet(
        validation_scope=PF1000_SCOPE,
        bank_parameters={
            "C0_F": 1.332e-3,
            "L0_H": 8.9e-9,
            "E0_J": 1.064e6,
        },
    )

    assert packet["status"] == "blocked_circuit_power_port_not_accepted"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["bank_circuit_transfer_candidate"]["present"] is True
    assert packet["accepted_power_port_claim"] is False
    assert "waveform_or_power_history_missing" in packet["blocking_reasons"]
    assert "review_certificate_missing" in packet["blocking_reasons"]


def test_phase4b_density_weighted_or_metadata_only_coupling_remains_blocked() -> None:
    packet = build_circuit_power_port_packet(
        validation_scope=PF1000_SCOPE,
        coupling_result={
            "t_us": [0.0, 0.1, 0.2],
            "Lp_mhd_nH": [1.0, 1.4, 2.1],
            "coupling_interval_authority": "density_weighted_mhd",
        },
    )

    assert packet["accepted_power_port_claim"] is False
    assert packet["field_coupling_evidence"]["passed"] is False
    assert "field_coupling_packet_not_passed" in packet["blocking_reasons"]
    assert "density_weighted_or_metadata_only_coupling" in packet["blocking_reasons"]


def test_phase4b_waveform_without_power_port_conventions_remains_blocked() -> None:
    coupled = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope=PF1000_SCOPE,
    )
    packet = build_circuit_power_port_packet(
        validation_scope=PF1000_SCOPE,
        coupling_result={"circuit_coupled_energy_verification": coupled},
        waveform_packet={"source_path": "KnowledgeReference/example.md"},
    )

    assert packet["accepted_power_port_claim"] is False
    assert packet["field_coupling_evidence"]["passed"] is False
    assert "sign_convention_missing" in packet["blocking_reasons"]
    assert "time_centering_missing" in packet["blocking_reasons"]
    assert "poynting_or_j_dot_e_residual_review_missing" in packet["blocking_reasons"]
    assert "review_certificate_missing" in packet["blocking_reasons"]


def test_phase4b_unreviewed_power_port_residual_remains_blocked() -> None:
    coupled = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope=PF1000_SCOPE,
        interval_labels=["field_derived_candidate"],
    )
    packet = build_circuit_power_port_packet(
        validation_scope=PF1000_SCOPE,
        coupling_result={"circuit_coupled_energy_verification": coupled},
        waveform_packet={
            "source_path": "KnowledgeReference/example.md",
            "figure_id": "synthetic",
            "extraction_method": "unit_test",
            "digitization_hash": "abc123",
            "uncertainty": {"relative": 0.05},
            "sign_convention": "positive_poynting_power_is_load_absorbed_power",
            "time_centering": "implicit_midpoint",
            "poynting_or_j_dot_e_residual": {"passed": True},
        },
    )

    assert packet["accepted_power_port_claim"] is False
    assert "poynting_or_j_dot_e_residual_review_missing" in packet["blocking_reasons"]
    assert packet["waveform_review_status"]["can_support_acceptance"] is False


def test_phase4b_transfer_candidate_matrix_links_without_promotion() -> None:
    packet = build_circuit_power_port_packet(validation_scope=PF1000_SCOPE)

    linkage = packet["phase3_transfer_candidate_linkage"]
    assert linkage["promotes_acceptance"] is False
    assert "power_circuit_coupling" in linkage["transfer_candidate_channels"]
    assert packet["accepted_power_port_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_phase4b_missing_transfer_matrix_is_explicit_blocker(tmp_path) -> None:
    missing_matrix = tmp_path / "missing_transfer_matrix.json"

    packet = build_circuit_power_port_packet(
        validation_scope=PF1000_SCOPE,
        phase3_transfer_matrix_path=missing_matrix,
    )

    assert packet["phase3_transfer_candidate_linkage"]["status"] == "blocked_transfer_matrix_missing"
    assert "transfer_candidate_linkage_not_loaded" in packet["blocking_reasons"]
    assert packet["accepted_power_port_claim"] is False
