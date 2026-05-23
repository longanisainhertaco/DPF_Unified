from __future__ import annotations

import copy
import json
from pathlib import Path

from dpf.first_principles.acceptance_shield import build_first_principles_acceptance_shield

ROOT = Path(__file__).resolve().parents[1]
PHASE7A_SCAFFOLD = ROOT / "docs/SS12_P1_PHASE7A_REVIEW_CERTIFICATE_SKELETON_2026_05_22.json"


def _phase7a_certificate() -> dict:
    return json.loads(PHASE7A_SCAFFOLD.read_text())


def _complete_looking_packet(status: str = "accepted") -> dict:
    return {
        "status": status,
        "accepted_source_claim": False,
        "can_support_numerical_acceptance": False,
        "accepted_power_port_claim": False,
        "accepted_uncertainty_claim": False,
        "can_support_uq_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def test_phase7b_review_certificate_skeleton_is_blocked_by_final_shield() -> None:
    shield = build_first_principles_acceptance_shield(
        source_packet=_complete_looking_packet(),
        numerical_packet=_complete_looking_packet(),
        power_port_packet=_complete_looking_packet(),
        figure_packets=[{"status": "accepted", "accepted_observable_claim": False}],
        uncertainty_packet=_complete_looking_packet(),
        review_certificate=_phase7a_certificate(),
    )

    assert shield["accepted_first_principles_claim"] is False
    assert shield["promotes_acceptance"] is False
    assert shield["can_support_first_principles_acceptance"] is False
    assert "review_certificate_not_accepted" in shield["blocking_reasons"]
    assert "review_certificate_incomplete_uncertainty_placeholders" in shield["blocking_reasons"]
    assert "review_certificate_incomplete_negative_controls" in shield["blocking_reasons"]
    assert "review_certificate_incomplete_run_evidence_hashes" in shield["blocking_reasons"]
    assert "review_certificate_independent_review_missing" in shield["blocking_reasons"]
    assert "review_certificate_upstream_phase6c_not_accepted" in shield["blocking_reasons"]
    assert shield["packet_status"]["review_certificate_accepted"] is False
    assert shield["packet_status"]["review_certificate_row_count"] == 6


def test_phase7b_forged_certificate_acceptance_flags_are_anomalies_not_promotion() -> None:
    certificate = _phase7a_certificate()
    certificate["accepted"] = True
    certificate["acceptance_boundary"]["accepted_review_certificate"] = True
    certificate["acceptance_boundary"]["promotes_acceptance"] = True
    certificate["acceptance_boundary"]["can_support_first_principles_acceptance"] = True
    certificate["acceptance_boundary"]["emits_accepted_certificate"] = True
    row = certificate["review_certificate_rows"][0]
    row["accepted_review_certificate"] = True
    row["accepted_runtime_claim"] = True
    row["promotes_acceptance"] = True
    row["can_support_first_principles_acceptance"] = True

    shield = build_first_principles_acceptance_shield(
        source_packet=_complete_looking_packet(),
        numerical_packet=_complete_looking_packet(),
        power_port_packet=_complete_looking_packet(),
        figure_packets=[{"status": "accepted", "accepted_observable_claim": False}],
        uncertainty_packet=_complete_looking_packet(),
        review_certificate=certificate,
    )

    assert shield["accepted_first_principles_claim"] is False
    assert shield["promotes_acceptance"] is False
    assert shield["can_support_first_principles_acceptance"] is False
    assert "review_certificate_acceptance_flags_not_false" in shield["blocking_reasons"]
    assert "review_certificate_claims_acceptance" in shield["claim_anomalies"]
    assert "review_certificate_boundary_claims_promotes_acceptance" in shield["claim_anomalies"]
    assert "review_certificate_row_claims_accepted_review_certificate" in shield["claim_anomalies"]
    assert shield["packet_status"]["review_certificate_accepted"] is False


def test_phase7b_complete_not_accepted_certificate_still_cannot_promote_without_explicit_final_review() -> None:
    certificate = _phase7a_certificate()
    for row in certificate["review_certificate_rows"]:
        row["certificate_status"] = "complete_not_accepted"
        for term in row["uncertainty_placeholders"].values():
            term["status"] = "complete_not_accepted"
            term["value"] = 0.1
            term["unit"] = "normalized"
            term["evidence_hash"] = "a" * 64
        row["pass_fail_metrics"]["status"] = "complete_not_accepted"
        row["pass_fail_metrics"]["tolerance"]["value"] = 0.2
        row["pass_fail_metrics"]["tolerance"]["unit"] = "normalized"
        row["pass_fail_metrics"]["result"] = "within_tolerance"
        for control in row["negative_controls"]:
            control["status"] = "complete_not_accepted"
            control["evidence_hash"] = "b" * 64
        row["run_evidence_hashes"] = {
            "runtime_run_hash": "c" * 64,
            "source_evidence_hash": "d" * 64,
            "uq_packet_hash": "e" * 64,
        }
        row["independent_review_placeholders"] = {
            "reviewer_id": "reviewer-red-team",
            "reviewer_affiliation": "independent",
            "reviewed_at": "2026-05-22T12:30:00Z",
            "review_packet_hash": "f" * 64,
            "review_status": "complete_not_accepted",
            "blocking_findings": [],
        }
    promoted_certificate = copy.deepcopy(certificate)
    promoted_certificate["accepted"] = True

    shield = build_first_principles_acceptance_shield(
        source_packet=_complete_looking_packet(),
        numerical_packet=_complete_looking_packet(),
        power_port_packet=_complete_looking_packet(),
        figure_packets=[{"status": "accepted", "accepted_observable_claim": False}],
        uncertainty_packet=_complete_looking_packet(),
        review_certificate=promoted_certificate,
    )

    assert shield["accepted_first_principles_claim"] is False
    assert shield["promotes_acceptance"] is False
    assert "review_certificate_not_final_review_accepted" in shield["blocking_reasons"]
    assert "review_certificate_claims_acceptance" in shield["claim_anomalies"]
    assert shield["packet_status"]["review_certificate_accepted"] is False
