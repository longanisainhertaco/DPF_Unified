from __future__ import annotations

from dpf.first_principles.figure_candidate_staging import stage_figure_observable_candidate

PF1000_SCOPE = "pf1000_full_energy_27_to_40_kv"


def _base_candidate() -> dict[str, object]:
    return {
        "validation_scope": PF1000_SCOPE,
        "channel": "current_waveform",
        "source_path": "KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md",
        "line_start": 169,
        "line_end": 178,
        "figure_id": "Fig. 6",
        "extraction_method": "manual_digitization_pending_review",
        "digitization_hash": "sha256:abc123",
        "uncertainty": {"relative": 0.15, "basis": "digitizer_repeatability"},
        "reviewer": "independent_reviewer_pending",
        "review_state": "candidate_pending_review",
        "scope_classification": "transfer_candidate",
    }


def test_phase4c_complete_figure_candidate_stages_without_acceptance() -> None:
    packet = stage_figure_observable_candidate(_base_candidate())

    assert packet["status"] == "staged_figure_candidate_not_accepted"
    assert packet["accepted_observable_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["missing_required_fields"] == []
    assert "review_certificate_missing" in packet["blocking_reasons"]
    assert "scope_not_same_source_accepted" in packet["blocking_reasons"]


def test_phase4c_missing_digitization_metadata_is_blocked() -> None:
    candidate = _base_candidate()
    del candidate["digitization_hash"]
    del candidate["uncertainty"]
    del candidate["reviewer"]

    packet = stage_figure_observable_candidate(candidate)

    assert packet["status"] == "blocked_figure_candidate_incomplete"
    assert packet["accepted_observable_claim"] is False
    assert packet["missing_required_fields"] == [
        "digitization_hash",
        "uncertainty",
        "reviewer",
    ]


def test_phase4c_reviewed_transfer_candidate_still_does_not_promote_acceptance() -> None:
    candidate = _base_candidate()
    candidate["review_state"] = "reviewed"
    candidate["review_certificate"] = {
        "accepted": True,
        "reviewer": "test-reviewer",
        "certificate_hash": "sha256:def456",
    }

    packet = stage_figure_observable_candidate(candidate)

    assert packet["accepted_observable_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "scope_not_same_source_accepted" in packet["blocking_reasons"]


def test_phase4c_same_source_requires_review_certificate_and_explicit_acceptance() -> None:
    candidate = _base_candidate()
    candidate["scope_classification"] = "same_source_candidate"
    candidate["review_state"] = "reviewed"

    packet = stage_figure_observable_candidate(candidate)

    assert packet["accepted_observable_claim"] is False
    assert "review_certificate_missing" in packet["blocking_reasons"]
    assert "scope_not_same_source_accepted" in packet["blocking_reasons"]
