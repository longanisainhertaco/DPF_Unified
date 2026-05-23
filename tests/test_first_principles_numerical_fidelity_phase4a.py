from __future__ import annotations

import json
from pathlib import Path

from dpf.first_principles.channel_state import BLOCKED_MISSING_SOURCE
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet

PF1000_SCOPE = "pf1000_full_energy_27_to_40_kv"
PF1000_DEVICE = "PF-1000"


def test_phase4a_packet_links_phase3_transfer_candidates_without_accepting_channels() -> None:
    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )

    linkage = packet["phase3_transfer_candidate_linkage"]

    assert linkage["status"] == "loaded_transfer_candidates_non_promoting"
    assert linkage["matrix_id"] == "ss12_p1_phase3_transfer_candidate_matrix"
    assert linkage["accepted_source_channels"] == []
    assert "current_waveform" in linkage["transfer_candidate_channels"]
    assert "power_circuit_coupling" in linkage["transfer_candidate_channels"]
    assert set(linkage["transfer_candidate_channels"]).isdisjoint(packet["accepted_channels"])
    assert all(row["promotes_acceptance"] is False for row in linkage["transfer_candidates"])
    assert all(row["can_fill_same_scope_channel"] is False for row in linkage["transfer_candidates"])
    assert all(
        row["source_channel_role"] == "transfer_candidate_not_accepted_channel"
        for row in linkage["transfer_candidates"]
    )
    assert packet["phase4a_transfer_linkage_gate"]["all_transfer_candidates_non_promoting"] is True
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_phase4a_transfer_candidate_name_overlap_does_not_promote_channel(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "phase3-transfer-matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_id": "test_transfer_matrix",
                "validation_scope": PF1000_SCOPE,
                "acceptance_boundary": {
                    "promotes_acceptance": False,
                    "can_fill_same_scope_channel": False,
                    "requires_transfer_rule_review": True,
                },
                "transfer_candidates": [
                    {
                        "channel": "mesh_timestep_convergence_packet",
                        "status": "transfer_candidate",
                        "source_path": "KnowledgeReference/example.md",
                        "line_start": 1,
                        "line_end": 1,
                        "scope_assessment": "Deliberate name overlap for non-promotion.",
                    }
                ],
                "global_blockers": ["Transfer candidates cannot promote acceptance."],
            }
        )
    )

    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        phase3_transfer_matrix_path=matrix_path,
    )

    assert "mesh_timestep_convergence_packet" in packet[
        "phase3_transfer_candidate_linkage"
    ]["transfer_candidate_channels"]
    assert "mesh_timestep_convergence_packet" not in packet["accepted_channels"]
    assert (
        packet["channel_states"]["mesh_timestep_convergence_packet"]
        == BLOCKED_MISSING_SOURCE.value
    )
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_phase4a_missing_transfer_matrix_path_keeps_packet_blocked(tmp_path: Path) -> None:
    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        phase3_transfer_matrix_path=tmp_path / "missing-phase3-transfer-matrix.json",
    )

    linkage = packet["phase3_transfer_candidate_linkage"]
    gate = packet["phase4a_transfer_linkage_gate"]

    assert linkage["status"] == "blocked_transfer_matrix_missing"
    assert linkage["transfer_candidate_channels"] == []
    assert linkage["accepted_source_channels"] == []
    assert "phase3_transfer_matrix_missing" in linkage["blocking_reasons"]
    assert gate["status"] == "blocked_by_missing_phase3_transfer_matrix"
    assert gate["transfer_matrix_available"] is False
    assert packet["status"] == "blocked_numerical_fidelity_packet_not_available"
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False
