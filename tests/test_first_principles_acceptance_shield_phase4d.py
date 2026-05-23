from __future__ import annotations

from dpf.first_principles.acceptance_shield import build_first_principles_acceptance_shield
from dpf.first_principles.circuit_power_port import build_circuit_power_port_packet
from dpf.first_principles.figure_candidate_staging import stage_figure_observable_candidate
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet

PF1000_SCOPE = "pf1000_full_energy_27_to_40_kv"


def test_phase4d_transfer_and_staged_packets_do_not_pass_acceptance_shield() -> None:
    shield = build_first_principles_acceptance_shield(
        source_packet={"status": "target_extraction_candidate", "accepted_source_claim": False},
        numerical_packet=build_numerical_fidelity_packet(declared_scope=PF1000_SCOPE),
        power_port_packet=build_circuit_power_port_packet(validation_scope=PF1000_SCOPE),
        figure_packets=[
            stage_figure_observable_candidate({
                "validation_scope": PF1000_SCOPE,
                "channel": "current_waveform",
                "source_path": "KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md",
                "line_start": 169,
                "line_end": 178,
                "figure_id": "Fig. 6",
                "extraction_method": "manual_digitization_pending_review",
                "digitization_hash": "sha256:abc123",
                "uncertainty": {"relative": 0.15},
                "reviewer": "pending",
                "review_state": "reviewed",
                "scope_classification": "transfer_candidate",
                "review_certificate": {"accepted": True, "reviewer": "x", "certificate_hash": "sha256:def"},
            })
        ],
        uncertainty_packet={"status": "blocked_uncertainty_budget_missing"},
        review_certificate={"accepted": False},
    )

    assert shield["status"] == "blocked_first_principles_acceptance"
    assert shield["accepted_first_principles_claim"] is False
    assert shield["can_support_first_principles_acceptance"] is False
    assert "source_packet_not_accepted" in shield["blocking_reasons"]
    assert "numerical_packet_not_accepted" in shield["blocking_reasons"]
    assert "power_port_packet_not_accepted" in shield["blocking_reasons"]
    assert "figure_packet_not_accepted" in shield["blocking_reasons"]
    assert "uncertainty_packet_not_accepted" in shield["blocking_reasons"]
    assert "review_certificate_not_accepted" in shield["blocking_reasons"]


def test_phase4d_rejects_packet_that_claims_acceptance_but_lacks_required_status() -> None:
    shield = build_first_principles_acceptance_shield(
        source_packet={"status": "target_extraction_candidate", "accepted_source_claim": True},
        numerical_packet={"status": "blocked_numerical_fidelity_packet_not_available", "can_support_numerical_acceptance": True},
        power_port_packet={"status": "blocked_circuit_power_port_not_accepted", "accepted_power_port_claim": True},
        uncertainty_packet={"status": "candidate", "accepted_uncertainty_claim": True},
        review_certificate={"accepted": True},
    )

    assert shield["accepted_first_principles_claim"] is False
    assert shield["claim_anomalies"]
    assert "source_packet_claims_acceptance_without_accepted_status" in shield["claim_anomalies"]
    assert "numerical_packet_claims_acceptance_without_accepted_status" in shield["claim_anomalies"]
    assert "power_port_packet_claims_acceptance_without_accepted_status" in shield["claim_anomalies"]
    assert "uncertainty_packet_claims_acceptance_without_accepted_status" in shield["claim_anomalies"]


def test_phase4d_missing_packets_are_fail_closed() -> None:
    shield = build_first_principles_acceptance_shield()

    assert shield["status"] == "blocked_first_principles_acceptance"
    assert shield["accepted_first_principles_claim"] is False
    assert set(shield["blocking_reasons"]) >= {
        "source_packet_missing",
        "numerical_packet_missing",
        "power_port_packet_missing",
        "uncertainty_packet_missing",
        "review_certificate_missing",
    }


def test_phase4d_flags_lower_layer_claims_even_with_accepted_like_status() -> None:
    shield = build_first_principles_acceptance_shield(
        source_packet={"status": "accepted_source_packet", "accepted_source_claim": True},
        numerical_packet={"status": "passed_numerical_packet", "can_support_numerical_acceptance": True},
        power_port_packet={"status": "ready_power_port", "accepted_power_port_claim": True},
        uncertainty_packet={"status": "accepted_uq", "accepted_uncertainty_claim": True},
        figure_packets=[{"accepted_observable_claim": True}],
        review_certificate={"accepted": True},
    )

    assert shield["accepted_first_principles_claim"] is False
    assert shield["blocking_reasons"]
    assert "phase4_no_packet_may_claim_acceptance" in shield["blocking_reasons"]
    assert "source_packet_claims_acceptance_in_phase4" in shield["claim_anomalies"]
    assert "numerical_packet_claims_acceptance_in_phase4" in shield["claim_anomalies"]
    assert "power_port_packet_claims_acceptance_in_phase4" in shield["claim_anomalies"]
    assert "uncertainty_packet_claims_acceptance_in_phase4" in shield["claim_anomalies"]
