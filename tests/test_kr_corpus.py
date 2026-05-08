"""Tests for KnowledgeReference corpus inventory and review status."""

from __future__ import annotations

from dpf.validation import (
    kr_corpus_inventory,
    kr_corpus_review_decisions,
    kr_corpus_review_status,
    kr_unreviewed_dpf_source_triage,
)


def test_kr_corpus_inventory_counts_local_source_tree():
    inventory = kr_corpus_inventory()

    assert inventory["passed"] is True
    assert inventory["model_role"] == "kr_corpus_inventory"
    assert inventory["total_files"] >= inventory["md_files"]
    assert inventory["total_files"] >= inventory["json_files"]
    assert inventory["md_files"] > 0
    assert inventory["dpf_named_md_files"] > 0
    assert inventory["dpf_content_md_files"] >= inventory["dpf_named_md_files"]
    assert inventory["dpf_relevant_md_files"] >= inventory["dpf_named_md_files"]
    assert ".md" in inventory["extension_counts"]


def test_kr_corpus_review_status_remains_open_until_validation_coverage_closed():
    status = kr_corpus_review_status()

    assert status["passed"] is False
    assert status["model_role"] == "kr_corpus_review_status"
    assert status["coded_target_count"] > 0
    assert status["unique_coded_target_source_count"] > 0
    assert status["reviewed_dpf_named_md_files"] == (
        status["corpus_counts"]["dpf_named_md_files"]
    )
    assert status["unreviewed_dpf_named_md_files"] == []
    assert status["reviewed_dpf_relevant_md_files"] == (
        status["corpus_counts"]["dpf_relevant_md_files"]
    )
    assert status["unreviewed_dpf_relevant_md_files"] == []
    assert status["target_coverage_passed"] is False
    assert status["same_scope_passed"] is False
    assert "phase_timing" in status["target_missing_or_partial_groups"]
    assert status["reviewed_by_coded_target_files"] > 0
    assert status["reviewed_by_explicit_decision_files"] >= 0
    assert "explicit_review_decisions" in status
    assert status["next_ratcheting_steps"][0].startswith(
        "DPF-relevant KnowledgeReference markdown review is complete"
    )
    assert "circuit_waveform" in status["next_ratcheting_steps"][1]


def test_kr_corpus_review_decisions_track_duplicate_sources():
    decisions = kr_corpus_review_decisions()

    assert decisions
    duplicates = [
        decision for decision in decisions
        if decision["status"] == "duplicate"
    ]
    assert duplicates
    assert all(
        str(decision["source"]).startswith("KnowledgeReference/")
        for decision in duplicates
    )
    assert all(
        str(decision["canonical_source"]).startswith("KnowledgeReference/")
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md"
        and decision["canonical_source"]
        == (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
            "dense-plasma-focus-z-pinch-5.md"
        )
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
            "alternative-fusion-source-to-versatile-high-energy.md"
        )
        and decision["canonical_source"]
        == (
            "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
            "alternative-fusion-source-to-versatile-high-energy-4.md"
        )
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == "KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md"
        and decision["canonical_source"]
        == "KnowledgeReference/goyon-2022-mjolnir-high-low.md"
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/measurement-of-electric-flux-emission-a-new-"
            "diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v-4.md"
        )
        and decision["canonical_source"]
        == (
            "KnowledgeReference/measurement-of-electric-flux-emission-a-new-"
            "diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md"
        )
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus-5.md"
        and decision["canonical_source"]
        == "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md"
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/double-3-mj-dense-plasma-focus-for-"
            "thermonuclear-drive-inertial-confinement-fusion-5.md"
        )
        and decision["canonical_source"]
        == (
            "KnowledgeReference/2025-double-3mj-dense-plasma-focus-"
            "thermonuclear-icf.md"
        )
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/double-3-mj-dense-plasma-focus-for-"
            "thermonuclear-drive-inertial-confinement-fusion.md"
        )
        and decision["canonical_source"]
        == (
            "KnowledgeReference/2025-double-3mj-dense-plasma-focus-"
            "thermonuclear-icf.md"
        )
        for decision in duplicates
    )
    assert any(
        decision["source"]
        == "KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md"
        and decision["canonical_source"]
        == "KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md"
        for decision in duplicates
    )
    insufficient = [
        decision for decision in decisions
        if decision["status"] == "insufficient_extractable_validation_data"
    ]
    assert any(
        decision["source"]
        == "KnowledgeReference/modification-and-numerical-modelling-of-dense-plasma-focus.md"
        for decision in insufficient
    )
    assert any(
        decision["source"]
        == "KnowledgeReference/auluck-2022-dpf-theory-part1.md"
        for decision in insufficient
    )
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/dimensions-and-lifetime-of-the-plasma-focus-"
            "pinch-plasma-science-ieee-transactions-on-2.md"
        )
        for decision in insufficient
    )
    assert all(decision["reason"] for decision in insufficient)
    non_scientific = [
        decision for decision in decisions
        if decision["status"] == "non_scientific_frontend_guide"
    ]
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/building-a-sci-fi-themed-dense-plasma-focus-"
            "simulation-front-end-in-unity.md"
        )
        for decision in non_scientific
    )
    assert all(decision["canonical_source"] == "" for decision in non_scientific)
    assert all(decision["reason"] for decision in non_scientific)
    corrections = [
        decision for decision in decisions
        if decision["status"] == "correction_only"
    ]
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/2023-correction-to-focus-fusion-overview-of-"
            "progress-towards-p-b11-fusion-with-the.md"
        )
        and decision["canonical_source"]
        == (
            "KnowledgeReference/focus-fusion-overview-of-progress-towards-"
            "p-b11-fusion-with-the-dense-plasma-focus.md"
        )
        for decision in corrections
    )
    assert all(decision["reason"] for decision in corrections)
    acronym_collisions = [
        decision for decision in decisions
        if decision["status"] == "non_dpf_acronym_collision"
    ]
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/dpf-bi-rrt-an-improved-path-planning-"
            "algorithm-for-complex-3d-environments-with-adaptive-sampling.md"
        )
        for decision in acronym_collisions
    )
    assert all(decision["canonical_source"] == "" for decision in acronym_collisions)
    assert all(decision["reason"] for decision in acronym_collisions)
    software_summaries = [
        decision for decision in decisions
        if decision["status"] == "non_scientific_software_performance_summary"
    ]
    assert any(
        decision["source"]
        == (
            "KnowledgeReference/optimization-and-development-of-a-dense-"
            "plasma-focus-simulator.md"
        )
        for decision in software_summaries
    )
    assert all(decision["canonical_source"] == "" for decision in software_summaries)
    assert all(decision["reason"] for decision in software_summaries)


def test_unreviewed_dpf_source_triage_prioritizes_remaining_sources():
    triage = kr_unreviewed_dpf_source_triage(max_hits_per_category=1)

    assert triage["passed"] is True
    assert triage["model_role"] == "kr_unreviewed_dpf_source_triage"
    assert triage["unreviewed_dpf_relevant_md_files"] == 0
    assert triage["records"] == []
    assert all(count == 0 for count in triage["category_counts"].values())
