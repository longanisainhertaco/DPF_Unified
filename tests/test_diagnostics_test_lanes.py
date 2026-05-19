from __future__ import annotations

from pathlib import Path

import pytest

from dpf.diagnostics.test_lanes import (
    diagnostics_test_lane_counts,
    diagnostics_test_lane_entries,
    diagnostics_test_lane_for_file,
    diagnostics_test_lane_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
PYPROJECT = ROOT / "pyproject.toml"
REGISTERED_MARKERS = {
    "diagnostics_engineering",
    "diagnostics_synthetic",
    "diagnostics_source_component",
    "diagnostics_source_blocked",
    "diagnostics_validation",
}


def test_diagnostics_test_lane_manifest_files_exist() -> None:
    manifest = diagnostics_test_lane_manifest()
    assert manifest

    for entry in manifest:
        assert (TESTS_DIR / entry["test_file"]).exists(), entry["test_file"]
        assert entry["validation_status"] == "not_validation_evidence"
        assert entry["can_support_validation_claims"] is False
        assert entry["blockers"]


def test_diagnostics_test_lanes_have_no_source_backed_validation_yet() -> None:
    for entry in diagnostics_test_lane_entries():
        assert entry.lane != "source-backed-validation"
        assert "diagnostics_validation" not in entry.markers
        assert entry.can_support_validation_claims is False


def test_diagnostics_test_lane_counts_split_nonvalidation_work() -> None:
    counts = diagnostics_test_lane_counts()
    assert counts["engineering-smoke"] >= 1
    assert counts["synthetic-only"] >= 1
    assert counts["source-component-check"] >= 1
    assert counts["source-blocked"] >= 1
    assert "source-backed-validation" not in counts


def test_diagnostics_pytest_markers_are_registered() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    for marker in REGISTERED_MARKERS:
        assert f'"{marker}:' in pyproject


def test_collection_hook_marks_diagnostics_lane(request: pytest.FixtureRequest) -> None:
    entry = diagnostics_test_lane_for_file("test_diagnostics_test_lanes.py")
    assert entry is not None

    marker_names = {marker.name for marker in request.node.iter_markers()}
    assert set(entry.markers) <= marker_names
    assert "diagnostics_validation" not in marker_names
