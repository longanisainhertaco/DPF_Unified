from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/SS16_STARTUP_BVP_EVIDENCE_PACKET_MATRIX_2026_05_23.json"
_SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_validator = importlib.import_module("validate_ss16_startup_bvp_evidence_packet")
validate_matrix = _validator.validate_matrix

REQUIRED_CHANNELS = [
    "d2_breakdown",
    "preionization_state",
    "insulator_flashover",
    "sheath_liftoff",
    "early_circuit_handoff",
    "same_scope_material_geometry",
    "startup_bvp_payload",
    "uncertainty_budget",
    "review_certificate",
]


def test_ss16_startup_packet_covers_required_channels_fail_closed() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    assert matrix["packet_id"] == "ss16_startup_bvp_evidence_closure"
    assert matrix["validation_scope"] == "pf1000_full_energy_27_to_40_kv_startup_bvp"
    assert matrix["authority_rule"] == "local_knowledge_reference_line_cited_sources_only"
    assert matrix["acceptance_boundary"] == {
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }

    channels = matrix["channels"]
    assert [row["channel"] for row in channels] == REQUIRED_CHANNELS
    assert not any(row["status"] == "accepted" for row in channels)
    assert all(row["blocked_reason"] for row in channels if row["status"] != "accepted")

    by_channel = {row["channel"]: row for row in channels}
    assert by_channel["d2_breakdown"]["status"] in {"candidate", "blocked"}
    assert by_channel["preionization_state"]["status"] == "blocked"
    assert by_channel["insulator_flashover"]["status"] in {"candidate", "cross_scope_candidate"}
    assert by_channel["sheath_liftoff"]["status"] in {"candidate", "cross_scope_candidate"}
    assert by_channel["early_circuit_handoff"]["status"] == "candidate"
    assert by_channel["same_scope_material_geometry"]["status"] == "candidate"
    assert by_channel["startup_bvp_payload"]["status"] == "blocked"
    assert by_channel["review_certificate"]["status"] == "candidate"
    assert by_channel["review_certificate"]["observables"] == [
        {
            "name": "independent_review_pass_non_promoting_packet",
            "value": "PASS: SS16 startup BVP evidence closure is correctly fail-closed and source-grounded.",
            "unit": "review_verdict",
            "uncertainty": None,
            "review_artifact": "/tmp/dpf_claude_bridge_t_5e6556b8_2026-05-23T054142.418751Z0000.txt",
            "note": "Review accepts the fail-closed packet posture only; it does not close missing startup payload or uncertainty evidence.",
        }
    ]


def test_ss16_source_refs_resolve_and_quotes_match_exact_line_windows() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    checked_refs = 0
    for channel in matrix["channels"]:
        for observable in channel.get("observables", []):
            for ref in observable.get("source_refs", []):
                source_path = ROOT / ref["source_path"]
                assert source_path.exists(), ref
                lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                start = ref["line_start"]
                end = ref["line_end"]
                assert 1 <= start <= end <= len(lines), ref
                assert end - start + 1 <= 24, ref
                extracted = " ".join(line.strip() for line in lines[start - 1 : end])
                assert ref["quote"] == extracted, (ref, extracted)
                checked_refs += 1

    assert checked_refs >= 8


def test_ss16_startup_packet_records_non_promoting_runtime_bridge() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    runtime_bridge = matrix["runtime_bridge"]

    assert runtime_bridge["module"] == "dpf.first_principles.startup_bvp"
    assert runtime_bridge["status"] == "blocked_runtime_startup_bvp_not_closed"
    assert runtime_bridge["accepted_runtime_claim"] is False
    assert runtime_bridge["can_support_first_principles_acceptance"] is False
    assert runtime_bridge["promotes_acceptance"] is False
    assert "startup_bvp_payload" in runtime_bridge["blocking_channels"]


def _write_mutated_matrix(tmp_path: Path, mutator: object) -> Path:
    matrix = json.loads(MATRIX_PATH.read_text())
    mutator(matrix)  # type: ignore[operator]
    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(matrix, indent=2))
    return target


def _rules(path: Path) -> set[str]:
    return {issue["rule"] for issue in validate_matrix(path, ROOT)}


def test_validator_accepts_live_ss16_matrix() -> None:
    assert validate_matrix(MATRIX_PATH, ROOT) == []


def test_validator_flags_acceptance_promotion_and_accepted_rows(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["acceptance_boundary"]["accepted_runtime_claim"] = True
        matrix["channels"][0]["status"] = "accepted"

    rules = _rules(_write_mutated_matrix(tmp_path, mutate))
    assert "acceptance_flag_not_false" in rules
    assert "accepted_row_forbidden_in_ss16" in rules


def test_validator_flags_fabricated_quote(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["channels"][0]["observables"][0]["source_refs"][0]["quote"] = "fabricated"

    assert "source_ref_quote_mismatch" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_rejects_non_knowledge_reference_refs(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        ref = matrix["channels"][0]["observables"][0]["source_refs"][0]
        ref["source_path"] = "docs/DPF_UNIFIED_FULL_PROJECT_KANBAN_PLAN_2026_05_22.md"
        ref["line_start"] = 1
        ref["line_end"] = 1
        ref["quote"] = "# DPF Unified Full Project Kanban Plan"

    assert "source_ref_not_knowledge_reference" in _rules(_write_mutated_matrix(tmp_path, mutate))


def test_validator_flags_runtime_bridge_promotion(tmp_path: Path) -> None:
    def mutate(matrix: dict[str, Any]) -> None:
        matrix["runtime_bridge"]["promotes_acceptance"] = True

    assert "runtime_bridge_promotes_acceptance" in _rules(_write_mutated_matrix(tmp_path, mutate))
