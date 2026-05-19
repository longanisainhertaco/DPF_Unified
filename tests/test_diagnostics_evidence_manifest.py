from __future__ import annotations

import ast
from pathlib import Path

from dpf.diagnostics.evidence_manifest import (
    diagnostics_evidence_by_module,
    diagnostics_evidence_entries,
    diagnostics_evidence_manifest,
    diagnostics_manifest_entry,
    diagnostics_manifest_status_counts,
)

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = ROOT / "src" / "dpf" / "diagnostics"
EXCLUDED_MODULES = {"__init__", "evidence_manifest", "test_lanes"}
ALLOWED_CLASSES = {
    "accepted",
    "blocked-by-review",
    "missing",
    "engineering-probe",
    "synthetic-only",
}


def _public_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            symbols.add(node.name)
    return symbols


def test_manifest_entries_fail_closed_for_validation_claims() -> None:
    manifest = diagnostics_evidence_manifest()
    assert manifest

    for entry in manifest:
        assert entry["evidence_class"] in ALLOWED_CLASSES
        assert entry["evidence_class"] != "accepted"
        assert entry["validation_status"] != "accepted"
        assert entry["can_support_validation_claims"] is False
        assert entry["source_status"]
        assert entry["blockers"]


def test_manifest_covers_every_diagnostics_module_and_public_symbol() -> None:
    by_module = diagnostics_evidence_by_module()
    module_files = [
        path
        for path in sorted(DIAGNOSTICS_DIR.glob("*.py"))
        if path.stem not in EXCLUDED_MODULES
    ]

    assert module_files
    assert {path.stem for path in module_files} == set(by_module)

    for path in module_files:
        expected = _public_symbols(path)
        covered: set[str] = set()
        for entry in by_module[path.stem]:
            covered.update(entry["symbols"])
        assert expected <= covered, path.stem


def test_status_counts_include_conservative_evidence_lanes() -> None:
    counts = diagnostics_manifest_status_counts()
    assert counts["blocked-by-review"] >= 1
    assert counts["engineering-probe"] >= 1
    assert counts["missing"] >= 1
    assert counts["synthetic-only"] >= 1
    assert "accepted" not in counts


def test_specific_diagnostics_guardrail_entries() -> None:
    entries = {entry.entry_id: entry for entry in diagnostics_evidence_entries()}

    beam_tracker = entries["DIA-BEAM-TRACKER"]
    assert beam_tracker.module == "beam_tracker"
    assert beam_tracker.evidence_class == "engineering-probe"
    assert beam_tracker.can_support_validation_claims is False

    hdf5 = diagnostics_manifest_entry("DIA-HDF5")
    assert hdf5["module"] == "hdf5_writer"
    assert "rough max_div_B" in hdf5["output"]
    assert hdf5["validation_status"] == "not_validation_evidence"

    pb11 = entries["DIA-PB11"]
    assert pb11.module == "pb11_yield"
    assert pb11.evidence_class == "missing"
    assert "source_missing" in pb11.source_status
