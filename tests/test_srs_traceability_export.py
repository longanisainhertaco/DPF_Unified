from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_srs_traceability.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_srs_traceability", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_requirements_baseline_exports_import_ready_matrix(tmp_path: Path):
    exporter = _load_exporter()
    baseline = ROOT / "docs" / "DPF_REQUIREMENTS_BASELINE.md"

    rows = exporter.parse_requirements_baseline(baseline)
    matrix = exporter.build_traceability_matrix(rows, baseline)

    assert matrix["schema"] == "dpf.srs.traceability.v1"
    assert matrix["doorstop_status"] == "staged_not_imported"
    assert matrix["summary"]["requirement_count"] == len(rows)
    assert matrix["summary"]["p0_or_p1_without_verification"] == []

    by_id = {row.req_id: row for row in rows}
    assert by_id["DPF-VV-001"].status == "implemented"
    assert by_id["DPF-PHYS-004"].status == "blocked"
    assert by_id["DPF-DATA-009"].source_section == "Deferred Or Scope-Decision Requirements"
    assert "inspection" in by_id["DPF-REL-002"].verification_methods

    json_path = tmp_path / "matrix.json"
    csv_path = tmp_path / "matrix.csv"
    exporter.write_json(matrix, json_path)
    exporter.write_csv(rows, csv_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["summary"]["requirement_count"] == len(rows)
    csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert csv_header.startswith("id,priority,requirement,owner,status")


def test_requirements_baseline_ids_are_unique_and_p0_p1_rows_are_verifiable():
    exporter = _load_exporter()
    rows = exporter.parse_requirements_baseline(ROOT / "docs" / "DPF_REQUIREMENTS_BASELINE.md")

    ids = [row.req_id for row in rows]
    assert len(ids) == len(set(ids))

    p0_p1_rows = [row for row in rows if row.priority in {"P0", "P1"}]
    assert p0_p1_rows
    assert all(row.verification_methods for row in p0_p1_rows)
    assert all(row.evidence for row in rows if row.status == "implemented")
