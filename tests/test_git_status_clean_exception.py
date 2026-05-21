"""Unit tests for the Sprint 9 WS9-0 narrow PDF-symlink typechange exception.

Tests verify that _is_excused_pdf_typechange and _classify_git_status_lines
accept ONLY ` T ` typechange lines inside the known PDF reference directories
and reject everything else.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import helpers from the audit script without executing __main__
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_mod = importlib.import_module("run_codex_periodic_audit")
_is_excused = _mod._is_excused_pdf_typechange
_classify = _mod._classify_git_status_lines


# ---------------------------------------------------------------------------
# _is_excused_pdf_typechange
# ---------------------------------------------------------------------------


class TestIsExcusedPdfTypechange:
    """Narrow acceptance: only ` T ` lines in known PDF dirs are excused."""

    # --- should be excused ---

    def test_downloaded_books_papers_research_quoted(self) -> None:
        line = ' T "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/buneman1959.pdf"'
        assert _is_excused(line) is True

    def test_downloaded_books_papers_papers_dir(self) -> None:
        line = " T downloaded_books_papers/papers/mhd-numerics/foo.pdf"
        assert _is_excused(line) is True

    def test_tmp_pdfs_dir(self) -> None:
        line = " T tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf"
        assert _is_excused(line) is True

    def test_deeply_nested_under_downloaded(self) -> None:
        line = " T downloaded_books_papers/a/b/c/deep.pdf"
        assert _is_excused(line) is True

    # --- must NOT be excused: wrong XY code ---

    def test_modified_M_not_excused(self) -> None:
        line = " M downloaded_books_papers/Research Papers/foo.pdf"
        assert _is_excused(line) is False

    def test_deleted_D_not_excused(self) -> None:
        line = " D downloaded_books_papers/papers/old.pdf"
        assert _is_excused(line) is False

    def test_untracked_not_excused(self) -> None:
        line = "?? downloaded_books_papers/Research Papers/new.pdf"
        assert _is_excused(line) is False

    def test_staged_typechange_not_excused(self) -> None:
        # staged typechange is "T " not " T "
        line = "T  downloaded_books_papers/Research Papers/staged.pdf"
        assert _is_excused(line) is False

    def test_added_not_excused(self) -> None:
        line = " A downloaded_books_papers/papers/added.pdf"
        assert _is_excused(line) is False

    # --- must NOT be excused: wrong directory ---

    def test_T_outside_pdf_dirs_src(self) -> None:
        line = " T src/physics/axial_model.py"
        assert _is_excused(line) is False

    def test_T_outside_pdf_dirs_tests(self) -> None:
        line = " T tests/test_foo.py"
        assert _is_excused(line) is False

    def test_T_outside_pdf_dirs_scripts(self) -> None:
        line = " T scripts/run_codex_periodic_audit.py"
        assert _is_excused(line) is False

    def test_T_outside_pdf_dirs_root(self) -> None:
        line = " T CHANGELOG.md"
        assert _is_excused(line) is False

    def test_T_outside_pdf_dirs_partial_match(self) -> None:
        # path starts with similar prefix but is not in the allowed list
        line = " T downloaded_books_papers_extra/foo.pdf"
        assert _is_excused(line) is False

    def test_T_outside_pdf_dirs_references(self) -> None:
        # KnowledgeReference and references/papers are NOT in the exception
        line = " T references/papers/mhd-numerics/lee2019.pdf"
        assert _is_excused(line) is False


# ---------------------------------------------------------------------------
# _classify_git_status_lines
# ---------------------------------------------------------------------------


class TestClassifyGitStatusLines:
    """classify() correctly splits lines into (excused, real_dirty)."""

    def test_all_excused(self) -> None:
        lines = [
            ' T "downloaded_books_papers/Research Papers/buneman1959.pdf"',
            " T tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf",
            " T downloaded_books_papers/papers/mhd-numerics/lee.pdf",
        ]
        excused, real_dirty = _classify(lines)
        assert len(excused) == 3
        assert real_dirty == []

    def test_all_real_dirty(self) -> None:
        lines = [
            " M src/physics/axial.py",
            "?? scripts/new_script.py",
            " D tests/test_old.py",
        ]
        excused, real_dirty = _classify(lines)
        assert excused == []
        assert len(real_dirty) == 3

    def test_mixed_excused_and_real_dirty(self) -> None:
        lines = [
            ' T "downloaded_books_papers/Research Papers/buneman1959.pdf"',
            " M src/physics/axial.py",
            " T tmp/pdfs/batch/paper.pdf",
            "?? docs/NEW_UNDOCUMENTED_FILE.md",
        ]
        excused, real_dirty = _classify(lines)
        assert len(excused) == 2
        assert len(real_dirty) == 2

    def test_T_outside_pdf_dir_is_real_dirty(self) -> None:
        lines = [" T src/physics/coupling.py"]
        excused, real_dirty = _classify(lines)
        assert excused == []
        assert real_dirty == [" T src/physics/coupling.py"]

    def test_empty_input(self) -> None:
        excused, real_dirty = _classify([])
        assert excused == []
        assert real_dirty == []

    def test_145_pdf_lines_all_excused(self) -> None:
        """Simulate the real 145-file scenario — all must be excused."""
        lines = [
            f' T "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/paper{i:03d}.pdf"'
            for i in range(144)
        ] + [" T tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf"]
        assert len(lines) == 145
        excused, real_dirty = _classify(lines)
        assert len(excused) == 145
        assert real_dirty == []

    def test_145_pdf_plus_one_modified_fails(self) -> None:
        """Adding a single real dirty line to the 145 must produce a failure."""
        lines = [
            f' T "downloaded_books_papers/Research Papers/2026-05-11-user-ingest/paper{i:03d}.pdf"'
            for i in range(145)
        ] + [" M src/mhd_runner.py"]
        excused, real_dirty = _classify(lines)
        assert len(excused) == 145
        assert real_dirty == [" M src/mhd_runner.py"]
