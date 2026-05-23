from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/SS17_SPATIAL_THERMO_VALIDATION_PACKET_MATRIX_2026_05_23.json"
_SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_validator = importlib.import_module("validate_ss17_spatial_thermo_packet_matrix")
validate_matrix = _validator.validate_matrix

REQUIRED_CHANNELS = [
    "density_emission_geometry",
    "phase_timing",
    "em_field_history",
    "temperature_or_distribution_history",
    "comparator_stubs",
    "uncertainty_annotations",
    "review_certificate",
]


def test_ss17_matrix_builds_spatial_thermo_packets_fail_closed() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    assert matrix["packet_id"] == "ss17_spatial_thermo_validation_packets"
    assert matrix["validation_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert matrix["source_scope"] == "PF-1000 full-energy / upper-energy local KnowledgeReference line-cited evidence; HeliosMatrix KB used for discovery only"
    assert matrix["authority_rule"] == "local_line_cited_sources_only_retrieval_is_not_authority"
    assert matrix["acceptance_boundary"] == {
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }

    channels = matrix["channels"]
    assert [row["channel"] for row in channels] == REQUIRED_CHANNELS
    assert not any(row["status"] == "accepted" for row in channels)
    assert all(row["blocked_reason"] for row in channels)

    channels_by_name = {row["channel"]: row for row in channels}
    assert len(channels_by_name["density_emission_geometry"]["observables"]) >= 3
    assert len(channels_by_name["phase_timing"]["observables"]) >= 2
    assert len(channels_by_name["em_field_history"]["observables"]) >= 2
    assert len(channels_by_name["temperature_or_distribution_history"]["observables"]) >= 2
    assert len(channels_by_name["comparator_stubs"]["comparators"]) >= 4
    assert channels_by_name["review_certificate"]["status"] == "blocked"


def test_ss17_comparator_stubs_reject_scalar_only_acceptance_shortcuts() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    comparator_channel = next(row for row in matrix["channels"] if row["channel"] == "comparator_stubs")

    required_stubs = {
        "density_field_geometry_comparator",
        "em_field_history_comparator",
        "temperature_distribution_comparator",
        "phase_timing_comparator",
    }
    stubs = {row["name"]: row for row in comparator_channel["comparators"]}
    assert required_stubs <= set(stubs)

    for stub in stubs.values():
        assert stub["implementation_status"] == "stub_blocked_by_missing_reviewed_inputs"
        assert stub["accepts_scalar_only_input"] is False
        assert stub["promotes_acceptance"] is False
        assert stub["requires_uncertainty"] is True
        assert stub["requires_review_certificate"] is True
        assert stub["blocked_reason"]
        assert stub["required_model_outputs"]
        assert stub["required_evidence_inputs"]


def test_ss17_source_refs_resolve_and_quotes_match_exact_line_windows() -> None:
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
                assert end - start + 1 <= 24, ref
                extracted = " ".join(line.strip() for line in lines[start - 1 : end])
                assert ref["quote"] == extracted, (ref, extracted)
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


def test_validator_accepts_live_ss17_matrix() -> None:
    assert validate_matrix(MATRIX_PATH, ROOT) == []


def test_validator_flags_acceptance_promotion_and_accepted_rows(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["acceptance_boundary"]["accepted_runtime_claim"] = True
        matrix["channels"][0]["status"] = "accepted"

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "acceptance_flag_not_false" in rules
    assert "accepted_row_forbidden_in_ss17" in rules


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


def test_validator_rejects_scalar_only_comparator_shortcuts(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        stub = next(row for row in matrix["channels"] if row["channel"] == "comparator_stubs")["comparators"][0]
        stub["accepts_scalar_only_input"] = True
        stub["promotes_acceptance"] = True

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "comparator_accepts_scalar_only_input" in rules
    assert "comparator_promotes_acceptance" in rules


def test_validator_requires_uncertainty_and_review_for_comparators(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        stub = next(row for row in matrix["channels"] if row["channel"] == "comparator_stubs")["comparators"][0]
        stub["requires_uncertainty"] = False
        stub["requires_review_certificate"] = False

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "comparator_missing_uncertainty_gate" in rules
    assert "comparator_missing_review_gate" in rules
