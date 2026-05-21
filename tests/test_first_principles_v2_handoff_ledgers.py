from __future__ import annotations

import csv
import json
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

EXPECTED_BLOCKER_STATUS_COUNTS = {
    "existing_kr_source_supported": 4,
    "existing_kr_target_extraction_pending": 4,
    "kr_promotion_recommended": 4,
    "pdf_present_needs_rendered_page_or_ocr_verification": 1,
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
        assert row["last_verified_commit"] == "8e6b5e9"
        assert "..." not in row["exact_path_or_full_citation"]


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
        assert row["last_verified_commit"] == "8e6b5e9"
        assert row["already_in_kr"] in {"true", "false"}
        assert row["external_required"] in {"true", "false"}


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
