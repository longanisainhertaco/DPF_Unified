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

# Ledger commit-pin tiers. The ledger uses three pins, proving the test no
# longer assumes a single stale global commit:
#  - Sprint 4 audit-handoff commit: rows last verified there and untouched
#    since.
#  - Sprint 7 runtime-contract HEAD: rows corrected by the Sprint 7
#    multi-agent audit / Super-Sprint 8 Phase A WS0 ledger repair.
#  - Super-Sprint 8 Phase A commit: rows the Phase B physics workstreams
#    (WS4 Bennett startup, WS5 Braginskii Z=1 transport) advanced on top of
#    the Phase A foundation.
SPRINT4_COMMIT = "8e6b5e9"
SPRINT7_COMMIT = "35bb1a9"
PHASEA_COMMIT = "bd5be3a"
KNOWN_LEDGER_COMMITS = {SPRINT4_COMMIT, SPRINT7_COMMIT, PHASEA_COMMIT}

# Blocker IDs the Sprint 8 Phase B physics workstreams re-verified on the
# Phase A foundation: WS4 wired the four Bennett startup channels and WS5
# wired the Braginskii Z=1 transport candidate. These rows must carry the
# Phase A commit.
PHASEB_REVERIFIED_BLOCKERS = {
    "STARTUP-BVP-CH03",
    "STARTUP-BVP-CH04",
    "STARTUP-BVP-CH07",
    "STARTUP-BVP-CH08",
    "CLOSURE-BLK-BRAG-001",
}
# The same-scope comparator governance row was last touched by the Sprint 7
# audit (Phase A WS0) and not by Phase B; it keeps the Sprint 7 commit.
SPRINT7_ONLY_BLOCKERS = {
    "SAME-SCOPE-COMPARATOR-DECISION",
}
# Source IDs the Phase B physics workstreams re-verified.
PHASEB_REVERIFIED_SOURCES = {
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

# Sprint 8 Phase B normalized blocker status distribution. Phase B advanced
# two groups on top of the Phase A WS0 state:
#  - WS4 ingested Bennett 2017 into canonical KR markdown and wired the four
#    startup channels, so the Bennett rows moved
#    on_disk_line_page_verified_kr_promotion_required ->
#    source_backed_runtime_candidate.
#  - WS5 render-verified Braginskii Eqs. 4.30-4.45 and wired the Z=1 transport
#    candidate, so CLOSURE-BLK-BRAG-001 moved
#    target_extracted_source_supported_pending_equation_extraction_and_review
#    -> equations_4_30_to_4_45_render_verified_z1_transport_wired_as_candidate_
#    acceptance_blocked.
# All acceptance flags remain false; these are candidate (engineering)
# transitions, not accepted physics.
EXPECTED_BLOCKER_STATUS_COUNTS = {
    "existing_kr_source_supported": 3,
    "existing_kr_target_extraction_pending": 4,
    "source_backed_runtime_candidate": 4,
    "equations_4_30_to_4_45_render_verified_z1_transport_wired_as_candidate_acceptance_blocked": 1,
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
        assert row["last_verified_commit"] in KNOWN_LEDGER_COMMITS
        assert "..." not in row["exact_path_or_full_citation"]


def test_blocker_ledger_commit_pins_are_per_row_not_a_stale_global() -> None:
    """Sprint 7 audit finding S7-A3, carried forward through Sprint 8 Phase B.

    The ledger test must not lock every row to one stale global commit. There
    are now three pin tiers: Sprint 4 (`8e6b5e9`) for untouched rows, Sprint 7
    (`35bb1a9`) for the comparator-governance row, and Sprint 8 Phase A
    (`bd5be3a`) for the five Phase B physics rows.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    for blocker_id in PHASEB_REVERIFIED_BLOCKERS:
        assert blocker_id in by_id, f"missing Phase B blocker row: {blocker_id}"
        assert by_id[blocker_id]["last_verified_commit"] == PHASEA_COMMIT, (
            f"{blocker_id} must be re-pinned to the Phase A commit "
            f"{PHASEA_COMMIT} after the Phase B physics wiring"
        )

    for blocker_id in SPRINT7_ONLY_BLOCKERS:
        assert blocker_id in by_id, f"missing Sprint 7 blocker row: {blocker_id}"
        assert by_id[blocker_id]["last_verified_commit"] == SPRINT7_COMMIT, (
            f"{blocker_id} was last touched by the Sprint 7 audit; it must "
            f"keep the Sprint 7 commit {SPRINT7_COMMIT}"
        )

    touched = PHASEB_REVERIFIED_BLOCKERS | SPRINT7_ONLY_BLOCKERS
    untouched = [row for row in records if row["blocker_id"] not in touched]
    assert untouched, "expected untouched Sprint 4 rows to remain"
    for row in untouched:
        assert row["last_verified_commit"] == SPRINT4_COMMIT, (
            f"{row['blocker_id']} was not re-verified after Sprint 4; it must "
            f"keep the Sprint 4 commit {SPRINT4_COMMIT}"
        )

    # All three pin tiers must be present, proving the test does not assume a
    # single global commit.
    pins = {row["last_verified_commit"] for row in records}
    assert pins == KNOWN_LEDGER_COMMITS


def test_bennett_2017_blocker_rows_are_source_backed_runtime_candidates() -> None:
    """Sprint 8 WS4 (carries forward audit finding S7-A1).

    Bennett 2017 was ingested as canonical KR markdown in Sprint 8 WS4 and the
    four startup-BVP blocker rows it backs are now `source_backed_runtime_
    candidate`. They are no longer `on_disk_line_page_verified_kr_promotion_
    required` (Phase A WS0) nor the over-promoted `kr_promotion_recommended`.
    A candidate is engineering evidence only: all acceptance flags stay false.
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
        assert row["corrected_status"] == "source_backed_runtime_candidate", (
            f"{blocker_id} must reflect the WS4 Bennett KR-ingestion transition"
        )
        assert row["corrected_status"] != "kr_promotion_recommended"
        assert (
            row["corrected_status"]
            != "on_disk_line_page_verified_kr_promotion_required"
        )
        assert row["child_source_id"] == "bennett_2017_startup"
        assert row["runtime_claim_allowed"] == "false"
        assert row["accepted_runtime_claim"] == "false"
        assert row["can_support_first_principles_acceptance"] == "false"
        # The row now cites the canonical KR markdown record.
        assert (
            "KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md"
            in row["exact_path_or_full_citation"]
        )


def test_braginskii_blocker_row_is_z1_transport_candidate_acceptance_blocked() -> None:
    """Sprint 8 WS5 (carries forward audit finding S7-A2).

    Braginskii Table 2 Z=1 cells and Eqs. 4.30-4.45 are render-verified, and
    Sprint 8 WS5 wired the Z=1 parallel transport as a candidate closure. The
    blocker row must advance past the Phase A
    `target_extracted_source_supported_pending_equation_extraction_and_review`
    status to the Z=1-transport-wired status, with acceptance still blocked
    (numerical-fidelity, same-scope comparator, and certificate gates remain),
    keeping all acceptance flags false.
    """
    _, _, records = _read_csv(BLOCKER_LEDGER)
    by_id = {row["blocker_id"]: row for row in records}

    row = by_id["CLOSURE-BLK-BRAG-001"]
    assert row["corrected_status"] == (
        "equations_4_30_to_4_45_render_verified_z1_transport_wired_"
        "as_candidate_acceptance_blocked"
    )
    assert (
        row["corrected_status"]
        != "target_extracted_source_supported_pending_equation_extraction_and_review"
    )
    assert (
        row["corrected_status"]
        != "pdf_present_needs_rendered_page_or_ocr_verification"
    )
    assert row["accepted_runtime_claim"] == "false"
    assert row["can_support_first_principles_acceptance"] == "false"
    assert row["runtime_claim_allowed"] == "false"
    # The row must still point at the target-extraction doc and keep the five
    # review-required cells plus the acceptance gates flagged.
    citation = row["exact_path_or_full_citation"]
    assert "BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md" in citation
    assert "4.30-4.45" in row["line_or_page_range"]
    action = row["remaining_action"]
    assert "review-required" in action
    assert "certificate" in action


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
        assert row["last_verified_commit"] in KNOWN_LEDGER_COMMITS
        assert row["already_in_kr"] in {"true", "false"}
        assert row["external_required"] in {"true", "false"}


def test_source_ledger_phaseb_reverified_rows_carry_phasea_commit() -> None:
    """Sprint 8 Phase B (source-ledger half; carries forward S7-A3).

    The Bennett and Braginskii source rows were re-verified by the Phase B
    physics workstreams; they must carry the Phase A commit. Sprint 8 WS4
    ingested Bennett 2017 as canonical KR markdown, so `bennett_2017_startup`
    is now `already_in_kr=true`. `braginskii_1965_transport` keeps
    `already_in_kr=false`: the full Braginskii 1965 paper is not ingested as a
    KR markdown record (only the Table 2 target extraction is).
    """
    _, _, records = _read_csv(SOURCE_LEDGER)
    by_id = {row["source_id"]: row for row in records}

    for source_id in PHASEB_REVERIFIED_SOURCES:
        assert source_id in by_id, f"missing Phase B source row: {source_id}"
        assert by_id[source_id]["last_verified_commit"] == PHASEA_COMMIT

    assert by_id["bennett_2017_startup"]["already_in_kr"] == "true"
    assert by_id["braginskii_1965_transport"]["already_in_kr"] == "false"


def test_wse_source_packet_target_extraction_claims_match_kr_ledger_state() -> None:
    """Regression gate (audit finding S7-A1), ledger-driven negative control.

    Fails if any Sprint 7 WS-E source packet asserts an unqualified
    target-extraction status for a source whose source-acquisition-ledger row
    currently says `already_in_kr=false`. A source that is not in KR cannot be
    a target-extracted KR record.

    This stays a true negative-control as the ledger evolves. Sprint 8 WS4
    ingested Bennett 2017 into KR (`bennett_2017_startup.already_in_kr=true`),
    so a Bennett target-extraction claim is now consistent and exempt. The
    Braginskii full-paper source row (`braginskii_1965_transport`) keeps
    `already_in_kr=false`, so the gate still meaningfully guards it: the WS-E
    Braginskii section may only use the qualified
    `target_extracted_source_supported_pending_...` status, never the bare
    `target_extracted_source_supported` claim.
    """
    _, _, source_records = _read_csv(SOURCE_LEDGER)
    not_in_kr = {
        row["source_id"]
        for row in source_records
        if row["already_in_kr"] == "false"
    }
    # The gate is only meaningful while at least one WS-E primary source is
    # still not in KR. Braginskii's full-paper source row is the live guard.
    assert "braginskii_1965_transport" in not_in_kr

    packet_text = WSE_SOURCE_PACKETS.read_text(encoding="utf-8")

    # Generic invariant: the WS-E packet must never carry an unqualified
    # `target_extracted_source_supported` status claim. Both WS-E primary
    # sources trace to not-in-KR-equivalent full-paper handling at the time
    # the packet was written; only the qualified pending status is allowed.
    for line in packet_text.splitlines():
        stripped = line.strip()
        if "target_extracted_source_supported" not in stripped:
            continue
        # Strip the allowed qualified pending status, then any remaining
        # `_pending_` form, and assert nothing unqualified survives.
        cleaned = stripped.replace(
            "target_extracted_source_supported_pending_"
            "equation_extraction_and_review",
            "",
        )
        cleaned = cleaned.replace(
            "target_extracted_source_supported_pending_", ""
        )
        assert "target_extracted_source_supported" not in cleaned, (
            "WS-E packet carries an unqualified "
            f"target_extracted_source_supported claim: {stripped!r}"
        )

    # The summary table must not label a not-in-KR source as a bare
    # "primary (target-extracted)" lane without a qualifier.
    assert (
        "Braginskii 1965 Table 2 | CLOSURE-BLK-BRAG-001 | primary "
        "(target-extracted) |" not in packet_text
    ), (
        "WS-E summary table labels Braginskii as a bare "
        "'primary (target-extracted)' lane while its full-paper source row "
        "is still already_in_kr=false"
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
