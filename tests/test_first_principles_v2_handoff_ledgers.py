from __future__ import annotations

import csv
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
    "existing_kr_source_supported": 3,
    "existing_kr_target_extraction_pending": 4,
    "kr_promotion_recommended": 4,
    "pdf_present_needs_rendered_page_or_ocr_verification": 1,
    "external_acquisition_required": 13,
    "dependency_blocked": 1,
    "absent_from_literature": 5,
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
    records = [dict(zip(header, row)) for row in rows]
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
    assert len(records) == 23
    assert len({row["source_id"] for row in records}) == 23

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


def test_v2_handoff_no_longer_contains_superseded_count_claims() -> None:
    text = V2_HANDOFF.read_text(encoding="utf-8")

    forbidden = [
        "source-acquisition_row_count = 19",
        "6 neutron + 1 thermonuclear-prefactor",
        "Sources counted: 19",
        "The 11 P1+P2 external acquisitions",
        "existing_kr_review_pending",
        "31 blockers, 19 sources",
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
