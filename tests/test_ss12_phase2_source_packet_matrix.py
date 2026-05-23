from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json"
_SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_validator = importlib.import_module("validate_ss12_phase2_source_packet_matrix")
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


def test_phase2_extracted_matrix_is_fail_closed_and_complete() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    assert matrix["validation_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert matrix["selected_source_scope"] == "pf1000_szydlowski_2001_large_electrode_full_energy_source"
    assert "24-rod geometry is not asserted" in matrix["source_scope_note"]
    assert matrix["acceptance_boundary"] == {
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }

    channels = matrix["channels"]
    assert [row["channel"] for row in channels] == REQUIRED_CHANNELS
    assert not any(row["status"] == "accepted" for row in channels)
    assert all(row["blocked_reason"] for row in channels if row["status"] != "accepted")

    for row in channels:
        if row["status"] == "accepted":
            assert row["scope_match"] == "same_scope"
            assert row["observables"]


def test_phase2_extracted_matrix_source_refs_resolve_and_quotes_match() -> None:
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
                # OCR line wrapping and superscript normalization are allowed; require each
                # substantive quote token to occur in the cited range. Hyphenated OCR line
                # breaks such as "elec- trodes" are normalized before comparison.
                quote = ref["quote"].replace("^", "")
                normalized = extracted.replace("×", "x").replace("- ", "")
                normalized = " ".join(normalized.split())
                normalized_quote = quote.replace("×", "x").replace("- ", "")
                normalized_quote = " ".join(normalized_quote.split())
                assert normalized_quote == normalized, (ref, extracted)
                checked_refs += 1

    assert checked_refs >= 10


def _write_mutated_matrix(tmp_path: Path, mutator: object) -> Path:
    matrix = json.loads(MATRIX_PATH.read_text())
    mutator(matrix)  # type: ignore[operator]
    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(matrix, indent=2))
    return target


def _rules(path: Path) -> set[str]:
    return {issue["rule"] for issue in validate_matrix(path, ROOT)}


def test_validator_accepts_live_extracted_matrix() -> None:
    assert validate_matrix(MATRIX_PATH, ROOT) == []


def test_validator_flags_missing_required_channel(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["channels"] = [
            row for row in matrix["channels"] if row["channel"] != "review_certificate"
        ]

    assert "missing_required_channel" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_flags_truthy_acceptance_boundary(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["acceptance_boundary"]["accepted_runtime_claim"] = True

    assert "acceptance_flag_not_false" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_rejects_accepted_row_without_same_scope_reviewed_refs(
    tmp_path: Path,
) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        row = matrix["channels"][0]
        row["status"] = "accepted"
        row["scope_match"] = "same_scope_candidate"
        row["observables"][0]["source_refs"][0]["review_status"] = "target_extraction_candidate"

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "accepted_row_forbidden_in_phase2" in rules
    assert "accepted_row_not_same_scope" in rules
    assert "accepted_source_ref_not_reviewed" in rules


def test_validator_flags_unresolved_source_ref_and_bad_line_range(
    tmp_path: Path,
) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        first_ref = matrix["channels"][0]["observables"][0]["source_refs"][0]
        first_ref["source_path"] = "KnowledgeReference/does-not-exist-for-ss12.md"
        second_ref = matrix["channels"][0]["observables"][1]["source_refs"][0]
        second_ref["line_start"] = 999999
        second_ref["line_end"] = 1000000

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "source_ref_missing" in rules
    assert "source_ref_line_range_invalid" in rules


def test_validator_rejects_phase2_accepted_row_even_with_same_scope_reviewed_refs(
    tmp_path: Path,
) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        row = matrix["channels"][0]
        row["status"] = "accepted"
        row["scope_match"] = "same_scope"
        row["observables"][0]["source_refs"][0]["review_status"] = "accepted"

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "accepted_row_forbidden_in_phase2" in rules


def test_validator_rejects_source_ref_that_escapes_repo(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["channels"][0]["observables"][0]["source_refs"][0]["source_path"] = (
            "KnowledgeReference/../../outside-source.md"
        )

    assert "source_ref_outside_repo" in _rules(_write_mutated_matrix(tmp_path, mutate))
