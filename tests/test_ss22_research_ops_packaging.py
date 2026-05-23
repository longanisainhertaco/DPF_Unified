"""SS22 research/ops packaging and long-run roadmap guardrails."""

from __future__ import annotations

import re
from pathlib import Path

SS22_MEMO = Path("docs/SS22_RESEARCH_OPS_PACKAGING_STATUS_2026_05_23.md")
RUNBOOK = Path("docs/SS22_RESEARCH_OPS_RUNBOOK_2026_05_23.md")
EVIDENCE_INDEX = Path("docs/SS22_EVIDENCE_INDEX_2026_05_23.md")
ROADMAP = Path("docs/SS22_LONG_RUN_RESEARCH_ROADMAP_2026_05_23.md")
FUTURE_BACKLOG = Path("docs/SS22_FUTURE_SPRINT_QUEUE_2026_05_23.md")

SS22_DOCS = [SS22_MEMO, RUNBOOK, EVIDENCE_INDEX, ROADMAP, FUTURE_BACKLOG]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ss22_required_package_files_exist_and_keep_release_fail_closed() -> None:
    for path in SS22_DOCS:
        assert path.exists(), f"missing SS22 packaging artifact: {path}"

    combined = "\n".join(_read(path) for path in SS22_DOCS)
    required_phrases = [
        "HONEST-BLOCKED / SOURCE-GATED PREVIEW",
        "accepted_runtime_claim=false",
        "can_support_first_principles_acceptance=false",
        "promotes_acceptance=false",
        "Retrieval is not authority",
        "local source authority only",
        "No corpus/PDF/symlink normalization",
    ]
    for phrase in required_phrases:
        assert phrase in combined

    banned_phrases = [
        "Release decision: ACCEPTED",
        "accepted_runtime_claim=true",
        "can_support_first_principles_acceptance=true",
        "promotes_acceptance=true",
        "publication-grade validated simulator",
        "end-to-end predictive DPF simulator is accepted",
    ]
    for phrase in banned_phrases:
        assert phrase not in combined


def test_ss22_status_memo_records_required_verification_and_elc() -> None:
    memo = _read(SS22_MEMO)

    required_sections = [
        "## Verification matrix",
        "docs render/link scan",
        "board orphan check",
        "claim scan",
        "final status memo",
        "## Evaluate / Learn / Continue",
        "Evaluate:",
        "Learn:",
        "Continue:",
    ]
    for section in required_sections:
        assert section in memo

    assert "review_result: PASS" in memo
    assert "review_artifact: `/tmp/dpf_claude_bridge_t_f9ba10c9_2026-05-23T071705.990612Z0000.txt`" in memo
    assert "active_child_review_task: none" in memo
    assert "active_ss22_work_excluding_current_fix_reverify: 0" in memo
    assert "orphaned_active_work: 0" in memo


def test_ss22_fix_reverify_closes_review_lane_without_promoting_acceptance() -> None:
    memo = _read(SS22_MEMO)

    required_phrases = [
        "fix/reverify result: review PASS consumed",
        "t_ac939060=done",
        "t_f9ba10c9=done",
        "t_d78850af=running during verification",
        "No reviewer fixes were required beyond recording the review PASS and re-running package verification.",
    ]
    for phrase in required_phrases:
        assert phrase in memo

    forbidden_phrases = [
        "accepted_runtime_claim=true",
        "can_support_first_principles_acceptance=true",
        "promotes_acceptance=true",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in memo


def test_ss22_evidence_index_classifies_artifacts_without_promoting_acceptance() -> None:
    index = _read(EVIDENCE_INDEX)

    required_rows = [
        "SS14 PF-1000 same-scope source packet",
        "SS16 startup BVP evidence packet",
        "SS17 spatial/thermodynamic validation packets",
        "SS18 neutron diagnostic validation stack",
        "SS19 UQ/comparator/certificate pipeline",
        "SS20 full integration acceptance dry-run",
        "SS21 product claim surface and release decision",
    ]
    for row in required_rows:
        assert row in index

    required_classifications = [
        "candidate / blocked",
        "synthetic wiring only",
        "review-approved honest-blocked wording only",
        "not validation evidence",
    ]
    for classification in required_classifications:
        assert classification in index


def test_ss22_runbook_and_roadmap_bound_resource_and_scope_risks() -> None:
    runbook = _read(RUNBOOK)
    roadmap = _read(ROADMAP)
    backlog = _read(FUTURE_BACKLOG)
    combined = "\n".join([runbook, roadmap, backlog])

    required_terms = [
        "Resource contention guard",
        "Scope explosion guard",
        "Claim drift guard",
        "lightweight inventory before heavy jobs",
        "future sprint queue",
        "same-scope evidence",
        "review certificate",
        "publication packet status: deferred",
    ]
    for term in required_terms:
        assert term in combined


def test_ss22_markdown_links_resolve_for_package_docs() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)#][^)]+)\)")
    missing: list[str] = []
    for path in SS22_DOCS:
        text = _read(path)
        for target in link_pattern.findall(text):
            if "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path}:{target}")

    assert missing == []
