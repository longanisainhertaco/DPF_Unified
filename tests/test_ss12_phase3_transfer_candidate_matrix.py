from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/SS12_P1_PHASE3_TRANSFER_CANDIDATE_MATRIX_2026_05_22.json"


def _normalize(text: str) -> str:
    return " ".join(text.replace("×", "x").replace("- ", "").split())


def test_phase3_transfer_matrix_is_non_promoting() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    assert matrix["acceptance_boundary"] == {
        "promotes_acceptance": False,
        "can_fill_same_scope_channel": False,
        "requires_transfer_rule_review": True,
    }
    assert matrix["transfer_candidates"]
    assert all(row["status"] != "accepted" for row in matrix["transfer_candidates"])
    assert any(row["status"] == "same_source_absence_proof" for row in matrix["transfer_candidates"])
    assert any("cannot promote acceptance" in blocker for blocker in matrix["global_blockers"])


def test_phase3_transfer_matrix_source_refs_resolve_and_quotes_are_exact() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())

    for row in matrix["transfer_candidates"]:
        source_path = ROOT / row["source_path"]
        assert source_path.exists(), row
        lines = source_path.read_text(errors="ignore").splitlines()
        start = row["line_start"]
        end = row["line_end"]
        assert 1 <= start <= end <= len(lines), row
        extracted = " ".join(line.strip() for line in lines[start - 1 : end])
        assert _normalize(row["quote"]) == _normalize(extracted), row
