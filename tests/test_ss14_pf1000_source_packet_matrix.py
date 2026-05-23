from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/SS14_PF1000_SAME_SCOPE_SOURCE_PACKET_MATRIX_2026_05_23.json"
_SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_validator = importlib.import_module("validate_ss14_pf1000_source_packet_matrix")
validate_matrix = _validator.validate_matrix

REQUIRED_CHANNELS = [
    "geometry",
    "bank_circuit",
    "gas_fill",
    "current_waveform",
    "startup",
    "density_history",
    "em_field_history",
    "temperature_or_distribution_history",
    "neutron_scalar_yield",
    "neutron_timing",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response",
    "uncertainty_budget",
    "review_certificate",
]


def test_ss14_matrix_expands_pf1000_full_energy_channels_fail_closed() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    assert matrix["packet_id"] == "ss14_pf1000_same_scope_source_packet_expansion"
    assert matrix["validation_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert matrix["kb_support_path"] == "/Users/anthonyzamora/Desktop/heliosmatrix_kb"
    assert matrix["authority_rule"] == "local_line_cited_sources_only_retrieval_is_not_authority"
    assert matrix["acceptance_boundary"] == {
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }

    channels = matrix["channels"]
    assert [row["channel"] for row in channels] == REQUIRED_CHANNELS
    assert not any(row["status"] == "accepted" for row in channels)
    assert all(row["blocked_reason"] for row in channels if row["status"] != "accepted")

    channels_by_name = {row["channel"]: row for row in channels}
    assert len(channels_by_name["geometry"]["observables"]) >= 4
    assert len(channels_by_name["bank_circuit"]["observables"]) >= 4
    assert len(channels_by_name["density_history"]["observables"]) >= 2
    assert len(channels_by_name["neutron_timing"]["observables"]) >= 2
    assert len(channels_by_name["detector_response"]["observables"]) >= 2
    assert channels_by_name["review_certificate"]["status"] == "blocked"


def test_ss14_source_refs_resolve_and_quotes_match_exact_line_windows() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    checked_refs = 0
    for channel in matrix["channels"]:
        for observable in channel.get("observables", []):
            for ref in observable.get("source_refs", []):
                source_path = ROOT / ref["source_path"]
                assert source_path.exists(), ref
                lines = source_path.read_text(errors="ignore").splitlines()
                start = ref["line_start"]
                end = ref["line_end"]
                assert 1 <= start <= end <= len(lines), ref
                extracted = " ".join(line.strip() for line in lines[start - 1 : end])
                assert ref["quote"] == extracted, (ref, extracted)
                checked_refs += 1

    assert checked_refs >= 20


def test_ss14_transfer_rows_are_non_promoting_candidates_or_rejections() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    transfer_rows = matrix["transfer_rows"]

    assert transfer_rows
    assert {row["status"] for row in transfer_rows} <= {"candidate", "cross_scope_candidate", "rejected"}
    assert not any(row.get("promotes_acceptance") for row in transfer_rows)
    assert all(row["blocked_reason"] for row in transfer_rows)
    assert any(row["status"] == "cross_scope_candidate" for row in transfer_rows)
    assert any("PF400" in row["blocked_reason"] or "PF-400" in row["blocked_reason"] for row in transfer_rows)


def _write_mutated_matrix(tmp_path: Path, mutator: object) -> Path:
    matrix = json.loads(MATRIX_PATH.read_text())
    mutator(matrix)  # type: ignore[operator]
    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(matrix, indent=2))
    return target


def _rules(path: Path) -> set[str]:
    return {issue["rule"] for issue in validate_matrix(path, ROOT)}


def test_validator_accepts_live_ss14_matrix() -> None:
    assert validate_matrix(MATRIX_PATH, ROOT) == []


def test_validator_flags_acceptance_promotion_and_accepted_rows(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["acceptance_boundary"]["promotes_acceptance"] = True
        matrix["channels"][0]["status"] = "accepted"

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "acceptance_flag_not_false" in rules
    assert "accepted_row_forbidden_in_ss14" in rules


def test_validator_flags_fabricated_quote(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["channels"][0]["observables"][0]["source_refs"][0]["quote"] = "fabricated"

    assert "source_ref_quote_mismatch" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_rejects_non_knowledge_reference_source_refs(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        ref = matrix["channels"][0]["observables"][0]["source_refs"][0]
        ref["source_path"] = "docs/DPF_UNIFIED_FULL_PROJECT_KANBAN_PLAN_2026_05_22.md"
        ref["line_start"] = 1
        ref["line_end"] = 1
        ref["quote"] = "# DPF Unified Full Project Kanban Plan"

    assert "source_ref_not_knowledge_reference" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_rejects_overwide_source_line_windows(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        ref = matrix["channels"][0]["observables"][0]["source_refs"][0]
        ref["line_start"] = 81
        ref["line_end"] = 105

    assert "source_ref_line_window_too_wide" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_flags_promoting_transfer_row(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["transfer_rows"][0]["promotes_acceptance"] = True

    assert "transfer_row_promotes_acceptance" in _rules(_write_mutated_matrix(tmp_path, mutate))
