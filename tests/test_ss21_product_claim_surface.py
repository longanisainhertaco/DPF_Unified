"""SS21 product-claim surface and release-decision guardrails."""

from __future__ import annotations

from pathlib import Path

SS21_MEMO = Path("docs/SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md")
README = Path("README.md")


def test_ss21_release_decision_memo_records_honest_blocked_release() -> None:
    memo = SS21_MEMO.read_text(encoding="utf-8")

    required_phrases = [
        "Release decision: HONEST-BLOCKED / SOURCE-GATED PREVIEW",
        "accepted_runtime_claim=false",
        "can_support_first_principles_acceptance=false",
        "promotes_acceptance=false",
        "not yet an end-to-end predictive DPF simulator",
        "engineering probe",
        "source-gated preview",
        "SS20 dry-run outcome",
        "Post-review fix/reverify status",
        "independent focused review PASS",
        "review-approved",
    ]
    for phrase in required_phrases:
        assert phrase in memo

    banned_phrases = [
        "Release decision: ACCEPTED",
        "production first-principles certificate accepted",
        "full-3D acceptance claim authorized",
        "publication-grade validated simulator",
    ]
    for phrase in banned_phrases:
        assert phrase not in memo


def test_readme_exposes_ss21_release_posture_without_overclaiming() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Release posture" in readme
    assert "HONEST-BLOCKED / SOURCE-GATED PREVIEW" in readme
    assert "SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md" in readme
    assert "accepted_runtime_claim=false" in readme
    assert "can_support_first_principles_acceptance=false" in readme
    assert "promotes_acceptance=false" in readme

    banned_phrases = [
        "publication-grade validated simulator",
        "accepted first-principles simulator",
        "full-3D accepted",
        "production certificate accepted",
    ]
    for phrase in banned_phrases:
        assert phrase not in readme


def test_public_claim_surface_has_required_guardrail_terms() -> None:
    files = [
        Path("README.md"),
        Path("app.py"),
        Path("app_validation.py"),
        Path("docs/V_AND_V_SUMMARY.md"),
        Path("docs/joss-paper-draft.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    required_terms = [
        "source-gated",
        "not validation evidence",
        "not scientific validation",
        "not yet an end-to-end predictive DPF simulator",
    ]
    for term in required_terms:
        assert term in combined

    banned_terms = [
        "publication-grade accuracy",
        "VALIDATED against 7+ published devices",
        "97x demonstrated",
        "validated against published experimental data for seven DPF devices",
        "Release decision: ACCEPTED",
    ]
    for term in banned_terms:
        assert term not in combined
