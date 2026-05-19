"""RC-3 (test half): CSV schema tests for the 2026-05-18 three-sprint blocker packet.

Verifies that every row in each of the five submission CSVs has exactly the
number of fields declared in that file's header row.  Uses only the stdlib
``csv`` module -- no quoting assumptions, no external dependencies.

Also includes package-consistency tests (Next Instruction 2):
  - No packet markdown may reference PENDING.md (the placeholder was removed).
  - README.md and THREE_SPRINT_FINAL_SUMMARY.md must agree on Sprint 2 status.
  - CHANGELOG.md must mention every commit hash present in git log 76480b0..HEAD.
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

PACKET_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "external_team_submissions"
    / "2026_05_18_three_sprint_blocker_packet"
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_FILES = [
    "BLOCKER_MATRIX.csv",
    "CLAIMS_LEDGER.csv",
    "SOURCE_PACKET_INDEX.csv",
    "TEST_MAP.csv",
    "ARTIFACT_HASHES.csv",
]

# The base commit anchoring the changelog coverage requirement.
_CHANGELOG_BASE = "76480b0"


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


# ---------------------------------------------------------------------------
# Package-consistency tests
# ---------------------------------------------------------------------------


def _collect_packet_markdown() -> list[Path]:
    """Return all .md files under the packet directory (recursive)."""
    return list(PACKET_DIR.rglob("*.md"))


def test_no_packet_markdown_references_pending_md() -> None:
    """No markdown in the packet may reference sprint_2/PENDING.md.

    sprint_2/PENDING.md was removed when the Sprint 2 proposal docs were added
    (commit bd840f4).  Any surviving reference means a doc was not updated.
    """
    bad: list[tuple[Path, int, str]] = []
    for md_path in _collect_packet_markdown():
        text = md_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "PENDING.md" not in line or "sprint_2" not in line:
                continue
            # Historical changelog/patch-scope lines that record the deletion
            # are acceptable — they document *why* PENDING.md no longer exists.
            # Reject only live references: lines that treat PENDING.md as a
            # navigation target ("See sprint_2/PENDING.md", links, etc.) without
            # also noting it was removed/deleted.
            stripped = line.strip().lower()
            is_historical = any(
                kw in stripped
                for kw in ("removed", "deleted", "superseded", "no longer")
            )
            if not is_historical:
                bad.append((md_path, lineno, line.strip()))
    assert not bad, (
        "Packet markdown still references the deleted sprint_2/PENDING.md:\n"
        + "\n".join(f"  {p}:{n}: {l}" for p, n, l in bad)
    )


def test_readme_and_summary_agree_on_sprint2_status() -> None:
    """README.md must not contradict THREE_SPRINT_FINAL_SUMMARY.md on Sprint 2.

    README.md says Sprint 2 proposals exist (Sprint 2 outcome section).
    THREE_SPRINT_FINAL_SUMMARY.md must not say Sprint 2 is 'pending' or
    'deferred' while README says proposals were delivered.
    """
    readme_path = PACKET_DIR / "README.md"
    summary_path = PACKET_DIR / "THREE_SPRINT_FINAL_SUMMARY.md"

    assert readme_path.exists(), f"README.md not found: {readme_path}"
    assert summary_path.exists(), (
        f"THREE_SPRINT_FINAL_SUMMARY.md not found: {summary_path}"
    )

    readme_text = readme_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")

    # README must mention Sprint 2 proposals being delivered.
    readme_has_sprint2_proposals = bool(
        re.search(
            r"sprint.2.*(proposal|deliver|WP.N1B|WP.N4B)",
            readme_text,
            re.IGNORECASE,
        )
    )
    assert readme_has_sprint2_proposals, (
        "README.md does not mention Sprint 2 proposals being delivered; "
        "if proposals were removed, update this test."
    )

    # If README says Sprint 2 proposals exist, the summary must NOT say
    # Sprint 2 is simply 'pending' or 'deferred' without qualification.
    # The summary is allowed to say Sprint 2 *implementation* is pending, but
    # must acknowledge proposals were delivered.
    # Forbidden pattern: a section heading whose body says only "pending" with
    # a reference to PENDING.md and no mention of proposals/proposals delivered.
    forbidden = re.search(
        r"Sprint 2.*\bpending\b.*sprint_2/PENDING\.md",
        summary_text,
        re.IGNORECASE | re.DOTALL,
    )
    assert not forbidden, (
        "THREE_SPRINT_FINAL_SUMMARY.md says Sprint 2 is pending and points to "
        "sprint_2/PENDING.md, but README.md says Sprint 2 proposals are "
        "delivered.  Update the summary to remove the PENDING.md reference."
    )

    # Summary must mention Sprint 2 proposals (not just 'pending').
    summary_has_proposals = bool(
        re.search(
            r"sprint.2.*(proposal|deliver|WP.N1B|WP.N4B)",
            summary_text,
            re.IGNORECASE,
        )
    )
    assert summary_has_proposals, (
        "THREE_SPRINT_FINAL_SUMMARY.md does not mention Sprint 2 proposals or "
        "WP-N1B/WP-N4B while README.md says they were delivered."
    )


def test_changelog_covers_all_commits_since_base() -> None:
    """CHANGELOG.md must mention every commit hash in git log 76480b0..HEAD~1.

    HEAD is exempt: the changelog is committed *in* a commit and cannot list its
    own hash, so the final changelog-wrapper commit is structurally unlistable
    (the same constraint Codex audit F4 accepted for AUDIT_COMMANDS.md). Every
    commit before HEAD must be present; omitting one means the changelog does
    not account for that change and the packet is non-self-describing.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"{_CHANGELOG_BASE}..HEAD~1"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"git not available or command failed: {exc}")

    commit_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not commit_lines:
        pytest.skip("No commits found since base; nothing to check.")

    # Extract short hashes (first token on each line).
    hashes_in_log: list[str] = [line.split()[0] for line in commit_lines]

    changelog_path = PACKET_DIR / "CHANGELOG.md"
    assert changelog_path.exists(), f"CHANGELOG.md not found: {changelog_path}"
    changelog_text = changelog_path.read_text(encoding="utf-8")

    missing: list[str] = [h for h in hashes_in_log if h not in changelog_text]
    assert not missing, (
        "CHANGELOG.md omits the following commit hashes from "
        f"git log {_CHANGELOG_BASE}..HEAD:\n  " + "\n  ".join(missing)
    )
