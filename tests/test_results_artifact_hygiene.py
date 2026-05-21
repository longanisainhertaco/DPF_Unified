"""Tests for the active results artifact hygiene audit.

Verifies that:
  1. Zero active (non-archive) result artifacts contain forbidden stale patterns.
  2. A temp non-archive file with a forbidden pattern is correctly flagged.
  3. A temp file placed under an archive_* directory is correctly ignored.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the scan function from the audit script
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_mod = importlib.import_module("verify_active_results_artifact_hygiene")
scan_active_results = _mod.scan_active_results

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Positive test: live repo results/ must be clean
# ---------------------------------------------------------------------------


class TestLiveRepoResultsClean:
    """The live results/ directory must contain no active stale artifacts."""

    def test_no_active_forbidden_patterns(self) -> None:
        issues = scan_active_results(ROOT)
        if issues:
            detail = "\n".join(
                f"  {i['file']}: pattern={i['pattern']!r} lines={i['lines']}"
                for i in issues
            )
            pytest.fail(
                f"Found {len(issues)} active result artifact(s) with forbidden "
                f"same-scope/LLNL-like patterns:\n{detail}"
            )


# ---------------------------------------------------------------------------
# Negative test: temp non-archive file with forbidden pattern must be flagged
# ---------------------------------------------------------------------------


class TestNonArchiveFileIsFlagged:
    """A forbidden pattern in a non-archive result file must produce an issue."""

    def test_non_archive_json_with_forbidden_pattern_is_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        stale_file = results_dir / "probe_active_2026_05_21.json"
        payload = {
            "selected": "same_scope_3d_validation_packet",
            "note": "this is a synthetic stale artifact for test purposes",
        }
        stale_file.write_text(json.dumps(payload, indent=2))

        issues = scan_active_results(tmp_path)
        assert len(issues) >= 1, "Expected at least one issue for the stale file"
        matching = [i for i in issues if "same_scope_3d_validation_packet" in i["pattern"]]
        assert matching, "Expected an issue reporting same_scope_3d_validation_packet"
        assert matching[0]["file"] == "results/probe_active_2026_05_21.json"

    def test_non_archive_json_with_llnl_pattern_is_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        stale_file = results_dir / "scope_stale_2026_05_21.json"
        payload = {
            "source_scope": "llnl_like_180ka_axisymmetric_hybrid_pic",
            "note": "synthetic stale artifact",
        }
        stale_file.write_text(json.dumps(payload, indent=2))

        issues = scan_active_results(tmp_path)
        matching = [
            i for i in issues if "llnl_like_180ka_axisymmetric_hybrid_pic" in i["pattern"]
        ]
        assert matching, "Expected an issue reporting llnl_like_180ka_axisymmetric_hybrid_pic"
        assert matching[0]["file"] == "results/scope_stale_2026_05_21.json"


# ---------------------------------------------------------------------------
# Archive exclusion test: file under archive_* must be silently ignored
# ---------------------------------------------------------------------------


class TestArchiveFileIsIgnored:
    """A forbidden pattern inside an archive_* directory must be ignored."""

    def test_archive_dir_file_not_flagged(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        archive_dir = results_dir / "archive_stale_pre_ss11_2026_05_21"
        archive_dir.mkdir(parents=True)
        archived_file = archive_dir / "experimental_old_artifact_2026_05_16.json"
        payload = {
            "selected": "same_scope_3d_validation_packet",
            "source_scope": "llnl_like_180ka_axisymmetric_hybrid_pic",
        }
        archived_file.write_text(json.dumps(payload, indent=2))

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            f"Files under archive_* must be excluded from the active scan; "
            f"got issues: {issues}"
        )

    def test_nested_archive_dir_file_not_flagged(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        nested_archive = results_dir / "archive_stale_pre_ssr_2026_05_18"
        nested_archive.mkdir(parents=True)
        nested_file = nested_archive / "some_old_probe.json"
        payload = {"cap": "same_scope_3d_validation_packet"}
        nested_file.write_text(json.dumps(payload, indent=2))

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "Nested archive_* files must be excluded from the scan"
        )

    def test_clean_active_file_alongside_archive_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        archive_dir = results_dir / "archive_stale_pre_ss11_2026_05_21"
        archive_dir.mkdir(parents=True)
        # Archive file (forbidden pattern — must be ignored)
        (archive_dir / "old.json").write_text(
            '{"source_scope": "llnl_like_180ka_axisymmetric_hybrid_pic"}'
        )
        # Clean active file (no forbidden patterns — must not produce issues)
        (results_dir / "clean_active_probe.json").write_text(
            '{"status": "ok", "result": "clean"}'
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "A clean active file alongside an archived file must produce no issues"
        )
