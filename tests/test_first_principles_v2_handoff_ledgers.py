from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

BLOCKER_LEDGER = (
    DOCS / "FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv"
)
SOURCE_LEDGER = (
    DOCS / "FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv"
)
USER_SUPPLIED_INTAKE = DOCS / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json"
V2_HANDOFF = (
    DOCS / "FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_V2_2026_05_20.md"
)
WSE_SOURCE_PACKETS = (
    DOCS / "extractions" / "SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md"
)
RTM_CSV = DOCS / "SRS_TRACEABILITY_MATRIX.csv"
RTM_JSON = DOCS / "SRS_TRACEABILITY_MATRIX.json"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_srs_traceability.py"

# Sprint 4 audit-handoff commit. Rows last verified at that commit and not
# touched by Sprint 7 still carry it; it is a historical anchor, not the
# current HEAD.
SPRINT4_COMMIT = "8e6b5e9"
# Sprint 7 runtime-contract HEAD. Rows corrected by the Sprint 7 multi-agent
# audit / Super-Sprint 8 WS0 ledger repair carry this commit.
SPRINT7_COMMIT = "35bb1a9"

# Blocker IDs whose ledger rows were re-verified by the Sprint 7 multi-agent
# audit (findings S7-A1, S7-A2, S7-A4) and Super-Sprint 8 WS0. These rows must
# carry the Sprint 7 commit, not the stale Sprint 4 commit.
SPRINT7_REVERIFIED_BLOCKERS = {
    "STARTUP-BVP-CH03",
    "STARTUP-BVP-CH04",
    "STARTUP-BVP-CH07",
    "STARTUP-BVP-CH08",
    "CLOSURE-BLK-BRAG-001",
    "SAME-SCOPE-COMPARATOR-DECISION",
}
# Source IDs re-verified by the same audit pass.
SPRINT7_REVERIFIED_SOURCES = {
    "bennett_2017_startup",
    "braginskii_1965_transport",
}

BLOCKER_HEADER = [
    "blocker_id",
    "current_repo_status",
    "corrected_status",
    "source_or_acquisition",
    "exact_path_or_full_citation",
    "line_or_page_range",
    "scope_tag",
    "runtime_claim_allowed",
    "remaining_action",
    "parent_blocker_id",
    "child_source_id",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "last_verified_commit",
]

SOURCE_HEADER = [
    "source_id",
    "priority",
    "source",
    "resolves_blockers",
    "already_in_kr",
    "on_disk_path",
    "external_required",
    "notes",
    "last_verified_commit",
]

# Sprint 7 / Super-Sprint 8 WS0 normalized blocker status distribution.
# Bennett rows moved kr_promotion_recommended -> on_disk_line_page_verified...
# (S7-A1); Braginskii moved pdf_present... -> target_extracted... (S7-A2);
# the comparator decision moved existing_kr_source_supported ->
# scope_governance_decision_pending (S7-A4).
EXPECTED_BLOCKER_STATUS_COUNTS = {
    "existing_kr_source_supported": 3,
    "existing_kr_target_extraction_pending": 4,
    "on_disk_line_page_verified_kr_promotion_required": 4,
    "target_extracted_source_supported_pending_equation_extraction_and_review": 1,
    "scope_governance_decision_pending": 1,
    "external_acquisition_required": 13,
    "dependency_blocked": 1,
    "absent_from_literature": 4,
}


def _read_csv(path: Path) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.reader(fh))
    assert raw_rows, f"{path} is empty"
    header = raw_rows[0]
    rows = raw_rows[1:]
    bad_rows = [
        (line_no, len(row))
        for line_no, row in enumerate(rows, start=2)
        if len(row) != len(header)
    ]
    assert not bad_rows, (
        f"{path}: rows with wrong field counts for {len(header)}-column header: "
        f"{bad_rows}"
    )
    records = [dict(zip(header, row, strict=True)) for row in rows]
    return header, rows, records


def test_blocker_resolution_ledger_is_uniform_and_fail_closed() -> None:
    header, _, records = _read_csv(BLOCKER_LEDGER)

    assert header == BLOCKER_HEADER
    assert len(records) == 31
    assert len({row["blocker_id"] for row in records}) == 31
    assert Counter(row["corrected_status"] for row in records) == Counter(
        EXPECTED_BLOCKER_STATUS_COUNTS
    )

    for row in records:
        assert row["runtime_claim_allowed"] == "false"
        assert row["accepted_runtime_claim"] == "false"
        assert row["can_support_first_principles_acceptance"] == "false"
        assert row["last_verified_commit"] in {SPRINT4_COMMIT, SPRINT7_COMMIT}
        assert "..." not in row["exact_path_or_full_citation"]


def test_blocker_ledger_commit_pins_are_per_row_not_a_stale_global() -> None:
    """Sprint 7 / Super-Sprint 8 WS0 (audit finding S7-A3).

    The ledger test must not lock every row to the stale Sprint 4 commit.
    Rows the Sprint 7 multi-agent audit re-verified must carry the Sprint 7
    commit; untouched rows keep the Sprint 4 commit.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    for blocker_id in SPRINT7_REVERIFIED_BLOCKERS:
        assert blocker_id in by_id, f"missing Sprint 7 blocker row: {blocker_id}"
        assert by_id[blocker_id]["last_verified_commit"] == SPRINT7_COMMIT, (
            f"{blocker_id} must be re-pinned to the Sprint 7 commit "
            f"{SPRINT7_COMMIT}, not the stale Sprint 4 commit"
        )

    untouched = [
        row
        for row in records
        if row["blocker_id"] not in SPRINT7_REVERIFIED_BLOCKERS
    ]
    assert untouched, "expected non-Sprint-7 rows to remain"
    for row in untouched:
        assert row["last_verified_commit"] == SPRINT4_COMMIT, (
            f"{row['blocker_id']} was not re-verified by Sprint 7; it must keep "
            f"the Sprint 4 commit {SPRINT4_COMMIT}"
        )

    # At least one row of each commit pin must exist, proving the test no
    # longer assumes a single global commit.
    pins = {row["last_verified_commit"] for row in records}
    assert pins == {SPRINT4_COMMIT, SPRINT7_COMMIT}


def test_bennett_2017_blocker_rows_are_kr_promotion_required_not_promotion_recommended() -> None:
    """Sprint 7 audit finding S7-A1.

    Bennett 2017 is on-disk line/page verified but NOT KR-authoritative. The
    four startup-BVP blocker rows it backs must carry the corrected status
    `on_disk_line_page_verified_kr_promotion_required`, never the over-promoted
    `kr_promotion_recommended` or `target_extracted...` states, and runtime
    consumption must remain disallowed.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    bennett_rows = (
        "STARTUP-BVP-CH03",
        "STARTUP-BVP-CH04",
        "STARTUP-BVP-CH07",
        "STARTUP-BVP-CH08",
    )
    for blocker_id in bennett_rows:
        row = by_id[blocker_id]
        assert (
            row["corrected_status"]
            == "on_disk_line_page_verified_kr_promotion_required"
        ), f"{blocker_id} must reflect the S7-A1 Bennett correction"
        assert row["corrected_status"] != "kr_promotion_recommended"
        assert "target_extracted" not in row["corrected_status"]
        assert row["child_source_id"] == "bennett_2017_startup"
        assert row["runtime_claim_allowed"] == "false"
        assert row["accepted_runtime_claim"] == "false"
        assert row["can_support_first_principles_acceptance"] == "false"
        # The remaining action must require canonical KR ingestion first.
        assert "KR" in row["remaining_action"] or "kr" in row["remaining_action"]


def test_braginskii_blocker_row_is_target_extracted_pending_equation_extraction() -> None:
    """Sprint 7 audit finding S7-A2.

    Braginskii Table 2 Z=1 cells are render-verified and target-extracted, but
    Eqs. 4.30-4.45 and the five review-required cells stay blocked. The blocker
    row must advance past the stale `pdf_present_needs_rendered_page_or_ocr_
    verification` status to the equation-extraction-pending status while
    keeping all acceptance flags false.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    row = by_id["CLOSURE-BLK-BRAG-001"]
    assert (
        row["corrected_status"]
        == "target_extracted_source_supported_pending_equation_extraction_and_review"
    )
    assert row["corrected_status"] != "pdf_present_needs_rendered_page_or_ocr_verification"
    assert row["accepted_runtime_claim"] == "false"
    assert row["can_support_first_principles_acceptance"] == "false"
    assert row["runtime_claim_allowed"] == "false"
    # The corrected row must point at the target-extraction doc and keep the
    # equation block flagged.
    citation = row["exact_path_or_full_citation"]
    assert "BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md" in citation
    assert "4.30-4.45" in row["line_or_page_range"]
    assert "4.30-4.45" in row["remaining_action"]


def test_same_scope_comparator_decision_is_control_plane_governance_not_kr_authority() -> None:
    """Sprint 7 audit finding S7-A4.

    `SAME-SCOPE-COMPARATOR-DECISION` cites an in-repo scope decision memo, not
    a KR scientific source. It must be a control-plane governance row
    (`scope_governance_decision_pending`) and must not be counted as a KR
    scientific source-supported row.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    row = by_id["SAME-SCOPE-COMPARATOR-DECISION"]
    assert row["corrected_status"] == "scope_governance_decision_pending"
    assert row["corrected_status"] != "existing_kr_source_supported"
    # Its citation is an in-repo decision memo, never a KnowledgeReference path.
    citation = row["exact_path_or_full_citation"]
    assert "FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md" in citation
    assert "KnowledgeReference/" not in citation
    assert row["accepted_runtime_claim"] == "false"
    assert row["can_support_first_principles_acceptance"] == "false"

    # No KR-source-supported row may cite the scope decision memo as a source.
    kr_supported = [
        candidate
        for candidate in records
        if candidate["corrected_status"] == "existing_kr_source_supported"
    ]
    for candidate in kr_supported:
        assert (
            "FIRST_PRINCIPLES_SCOPE_DECISION_MEMO" not in candidate[
                "exact_path_or_full_citation"
            ]
        ), (
            f"{candidate['blocker_id']} counts a governance memo as KR "
            "scientific authority"
        )


def test_source_acquisition_ledger_has_expected_rows_and_external_gate() -> None:
    header, _, records = _read_csv(SOURCE_LEDGER)

    assert header == SOURCE_HEADER
    assert len(records) == 31
    assert len({row["source_id"] for row in records}) == 31

    true_external_p1_p2 = [
        row
        for row in records
        if row["priority"] in {"P1", "P2"} and row["external_required"] == "true"
    ]
    assert len(true_external_p1_p2) == 12

    for row in records:
        assert row["last_verified_commit"] in {SPRINT4_COMMIT, SPRINT7_COMMIT}
        assert row["already_in_kr"] in {"true", "false"}
        assert row["external_required"] in {"true", "false"}


def test_source_ledger_sprint7_reverified_rows_carry_sprint7_commit() -> None:
    """Sprint 7 audit finding S7-A3 (source-ledger half).

    The Bennett and Braginskii source rows were re-verified during the Sprint 7
    multi-agent audit; they must carry the Sprint 7 commit. Both remain
    `already_in_kr=false` because neither is canonical KR markdown yet.
    """
    _, _, records = _read_csv(SOURCE_LEDGER)
    by_id = {row["source_id"]: row for row in records}

    for source_id in SPRINT7_REVERIFIED_SOURCES:
        assert source_id in by_id, f"missing Sprint 7 source row: {source_id}"
        assert by_id[source_id]["last_verified_commit"] == SPRINT7_COMMIT
        # Sprint 7 did not perform KR ingestion for either source.
        assert by_id[source_id]["already_in_kr"] == "false"


def test_wse_source_packet_target_extraction_claims_match_kr_ledger_state() -> None:
    """Super-Sprint 8 WS0 regression gate (audit finding S7-A1).

    Fails if any Sprint 7 WS-E source packet claims target extraction for a
    source whose source-acquisition-ledger row says `already_in_kr=false`.
    A source that is not in KR cannot be a target-extracted KR record; the
    correct state for such a source is on-disk line/page verification with
    KR promotion still required.
    """
    _, _, source_records = _read_csv(SOURCE_LEDGER)
    not_in_kr = {
        row["source_id"]
        for row in source_records
        if row["already_in_kr"] == "false"
    }
    # Bennett must be in the not-in-KR set for this gate to be meaningful.
    assert "bennett_2017_startup" in not_in_kr

    packet_text = WSE_SOURCE_PACKETS.read_text(encoding="utf-8")

    # The WS-E packet section for each not-in-KR source must not assert an
    # unqualified target-extraction status. The qualified Braginskii status
    # `target_extracted_source_supported_pending_equation_extraction_and_review`
    # is allowed (its Table 2 cells are genuinely extracted); the bare
    # `target_extracted_source_supported` claim and the prose label
    # "primary (target-extracted)" are not allowed for a not-in-KR source.
    #
    # Bennett (already_in_kr=false, not target-extracted at all) is checked by
    # source-id-anchored phrasing.
    forbidden_bennett_phrases = [
        # bare target-extraction claim lumping Bennett in
        "(Braginskii Table 2 and Bennett 2017) are\n`target_extracted_source_supported`",
        "Bennett 2017 startup | CH03/CH04/CH07/CH08 | primary (target-extracted)",
    ]
    offending = [
        phrase for phrase in forbidden_bennett_phrases if phrase in packet_text
    ]
    assert not offending, (
        "WS-E packet claims target extraction for Bennett 2017 while its "
        f"source-ledger row has already_in_kr=false: {offending}"
    )

    # Bennett's own packet section must explicitly state KR promotion is
    # required and that it is not KR-authoritative.
    assert "on_disk_line_page_verified_kr_promotion_required" in packet_text
    assert "bennett_2017_startup.already_in_kr=false" in packet_text

    # Generic invariant: the unqualified token must never appear without its
    # `_pending_` qualifier for a not-in-KR source. The packet only references
    # the two primary sources, both of which are not-in-KR, so the bare token
    # must not appear anywhere as a status claim.
    for line in packet_text.splitlines():
        stripped = line.strip()
        if "target_extracted_source_supported" not in stripped:
            continue
        # Allowed: the qualified pending status.
        cleaned = stripped.replace(
            "target_extracted_source_supported_pending_"
            "equation_extraction_and_review",
            "",
        )
        cleaned = cleaned.replace(
            "target_extracted_source_supported_pending_", ""
        )
        assert "target_extracted_source_supported" not in cleaned, (
            "WS-E packet still carries an unqualified "
            f"target_extracted_source_supported claim: {stripped!r}"
        )


def test_committed_rtm_exports_match_fresh_render(tmp_path: Path) -> None:
    """Super-Sprint 8 WS0 RTM drift gate (audit finding S7-A5).

    Renders the SRS traceability matrix fresh into a temp directory using the
    exact `scripts/export_srs_traceability.py` CLI entry point, then asserts
    that the committed CSV and JSON exports are byte-identical to the fresh
    render. A read-only drift check: it never writes the committed files.
    """
    assert RTM_CSV.exists(), f"committed RTM CSV missing: {RTM_CSV}"
    assert RTM_JSON.exists(), f"committed RTM JSON missing: {RTM_JSON}"

    fresh_csv = tmp_path / "rtm.csv"
    fresh_json = tmp_path / "rtm.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--baseline",
            "docs/DPF_REQUIREMENTS_BASELINE.md",
            "--csv",
            str(fresh_csv),
            "--json",
            str(fresh_json),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"export_srs_traceability.py failed: {result.stderr or result.stdout}"
    )

    assert RTM_JSON.read_text(encoding="utf-8") == fresh_json.read_text(
        encoding="utf-8"
    ), (
        "committed SRS_TRACEABILITY_MATRIX.json has drifted from a fresh "
        "export; run scripts/export_srs_traceability.py"
    )
    assert RTM_CSV.read_text(encoding="utf-8") == fresh_csv.read_text(
        encoding="utf-8"
    ), (
        "committed SRS_TRACEABILITY_MATRIX.csv has drifted from a fresh "
        "export; run scripts/export_srs_traceability.py"
    )


def test_sprint6_pf1000_geometry_source_transitions_are_not_absent() -> None:
    _, _, blocker_records = _read_csv(BLOCKER_LEDGER)
    blockers = {row["blocker_id"]: row for row in blocker_records}

    insulator = blockers["PF1000-BLK-015"]
    assert insulator["corrected_status"] == "existing_kr_source_supported"
    assert insulator["accepted_runtime_claim"] == "false"
    assert insulator["can_support_first_principles_acceptance"] == "false"
    assert "source_available_revision_not_mapped" in insulator["current_repo_status"]
    assert "recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md" in (
        insulator["exact_path_or_full_citation"]
    )

    _, _, source_records = _read_csv(SOURCE_LEDGER)
    sources = {row["source_id"]: row for row in source_records}
    assert sources["scholz_2001_recent_progress_pf1000_hardware"][
        "already_in_kr"
    ] == "true"
    assert sources["scholz_2000_pf1000_device"]["already_in_kr"] == "true"
    assert sources["scholz_2000_pf1000_device"]["external_required"] == "false"


def test_user_supplied_intake_sources_are_represented_in_source_ledger() -> None:
    _, _, source_records = _read_csv(SOURCE_LEDGER)
    source_ids = {row["source_id"] for row in source_records}
    intake = json.loads(USER_SUPPLIED_INTAKE.read_text())

    assert not intake["failed"]
    assert {
        "scholz_2001_recent_progress_pf1000_hardware",
        "bruzzone_bernal_2001_lhi_interface",
        "scholz_2000_pf1000_device",
        "herold_1989_poseidon_pf360_context",
        "scholz_1999_foam_liner_context",
        "loarer_2007_gas_balance_context",
        "shakya_2015_pf1000_pf400_lee_context",
        "scholz_gribkov_2007_part2",
        "gribkov_malaquias_2006_dmp_applications_context",
    } <= source_ids

    non_failed_count = len(intake["promoted"]) + len(intake["skipped_existing"])
    sprint6_rows = [
        row
        for row in source_records
        if row["source_id"]
        in {
            "scholz_2001_recent_progress_pf1000_hardware",
            "bruzzone_bernal_2001_lhi_interface",
            "scholz_2000_pf1000_device",
            "herold_1989_poseidon_pf360_context",
            "scholz_1999_foam_liner_context",
            "loarer_2007_gas_balance_context",
            "shakya_2015_pf1000_pf400_lee_context",
            "scholz_gribkov_2007_part2",
            "gribkov_malaquias_2006_dmp_applications_context",
        }
    ]
    assert len(sprint6_rows) == non_failed_count
    for row in sprint6_rows:
        assert row["already_in_kr"] == "true"
        assert row["external_required"] == "false"


def test_v2_handoff_no_longer_contains_superseded_count_claims() -> None:
    text = V2_HANDOFF.read_text(encoding="utf-8")

    forbidden = [
        "source-acquisition_row_count = 19",
        "6 neutron + 1 thermonuclear-prefactor",
        "Sources counted: 19",
        "The 11 P1+P2 external acquisitions",
        "existing_kr_review_pending",
        "31 blockers, 19 sources",
        "source-acquisition_row_count = 23",
        "23 visible source-acquisition rows",
    ]
    for phrase in forbidden:
        assert phrase not in text

    assert (
        "docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv"
        in text
    )
    assert (
        "docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv"
        in text
    )
