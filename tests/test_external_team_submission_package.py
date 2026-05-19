"""RC-3 (test half): CSV schema tests for the 2026-05-18 three-sprint blocker packet.

Verifies that every row in each of the five submission CSVs has exactly the
number of fields declared in that file's header row.  Uses only the stdlib
``csv`` module -- no quoting assumptions, no external dependencies.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

PACKET_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "external_team_submissions"
    / "2026_05_18_three_sprint_blocker_packet"
)

CSV_FILES = [
    "BLOCKER_MATRIX.csv",
    "CLAIMS_LEDGER.csv",
    "SOURCE_PACKET_INDEX.csv",
    "TEST_MAP.csv",
    "ARTIFACT_HASHES.csv",
]


@pytest.mark.parametrize("csv_name", CSV_FILES)
def test_csv_all_rows_match_header_field_count(csv_name: str) -> None:
    """Every data row in the submission CSV must have exactly as many fields as
    the header row.  A mismatch means a comma-containing field was not quoted,
    which breaks machine-readable review tooling."""
    csv_path = PACKET_DIR / csv_name
    assert csv_path.exists(), f"submission CSV not found: {csv_path}"

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    assert rows, f"{csv_name} is empty"
    header_width = len(rows[0])
    bad_rows: list[tuple[int, int]] = [
        (line_number, len(row))
        for line_number, row in enumerate(rows[1:], start=2)
        if len(row) != header_width
    ]
    assert not bad_rows, (
        f"{csv_name}: header has {header_width} fields but rows "
        f"{bad_rows} have wrong field counts"
    )
