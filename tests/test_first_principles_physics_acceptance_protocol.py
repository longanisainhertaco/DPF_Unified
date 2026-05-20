from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

PROTOCOL = (
    DOCS / "FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_PROMOTION_PROTOCOL_2026_05_20.md"
)
GATE_LEDGER = (
    DOCS / "FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv"
)
NEXT_PLAN = (
    DOCS / "CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md"
)

GATE_HEADER = [
    "item_id",
    "domain",
    "required_blocker_ids",
    "required_source_packets",
    "required_code_surfaces",
    "required_test_gates",
    "required_numerical_gates",
    "required_comparator_gate",
    "other_team_verification_required",
    "codex_independent_verification_required",
    "automated_repro_verification_required",
    "current_state",
    "accepted_physics_allowed",
    "next_promotion_action",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows, f"{path} is empty"
    header = rows[0]
    bad_rows = [
        (line_no, len(row))
        for line_no, row in enumerate(rows[1:], start=2)
        if len(row) != len(header)
    ]
    assert not bad_rows, (
        f"{path}: rows with wrong field counts for {len(header)}-column header: "
        f"{bad_rows}"
    )
    return header, [dict(zip(header, row, strict=True)) for row in rows[1:]]


def test_physics_acceptance_gate_ledger_requires_triple_verification() -> None:
    header, records = _read_csv(GATE_LEDGER)

    assert header == GATE_HEADER
    assert len(records) >= 13
    assert len({row["item_id"] for row in records}) == len(records)
    records_by_id = {row["item_id"]: row for row in records}
    assert "package_native_3d_acceptance_contract" in records_by_id
    assert records_by_id["package_native_3d_acceptance_contract"][
        "accepted_physics_allowed"
    ] == "false"
    assert "FIRST-PRINCIPLES-MHD-GATE-PARITY" in records_by_id[
        "package_native_3d_acceptance_contract"
    ]["required_blocker_ids"]

    for row in records:
        assert row["other_team_verification_required"] == "true"
        assert row["codex_independent_verification_required"] == "true"
        assert row["automated_repro_verification_required"] == "true"
        assert row["accepted_physics_allowed"] == "false"
        assert row["current_state"].endswith("_not_accepted")
        assert row["required_blocker_ids"]
        assert row["required_source_packets"]
        assert row["required_code_surfaces"]
        assert row["required_test_gates"]


def test_protocol_defines_acceptance_without_promoting_current_physics() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")

    required_phrases = [
        "Triple Verification Rule",
        "Lane 1 - Other-Team Evidence And Implementation Packet",
        "Lane 2 - Codex Independent Audit",
        "Lane 3 - Executable Reproducibility Gate",
        "accepted_physics_allowed=true",
        "accepted_physics_module",
        "validated_scope_certificate",
        "KnowledgeReference/",
        "every physics item remains unaccepted",
        "Package-Native 3-D Acceptance Contract",
        "observable_excluded_not_validated",
        "caveat_accepted",
    ]
    for phrase in required_phrases:
        assert phrase in text


def test_next_plan_references_physics_acceptance_protocol_and_ledger() -> None:
    text = NEXT_PLAN.read_text(encoding="utf-8")

    assert (
        "docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_PROMOTION_PROTOCOL_2026_05_20.md"
        in text
    )
    assert (
        "docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv"
        in text
    )
    assert "package_native_3d_acceptance_contract" in text
    assert "observable_excluded_not_validated" in text
    assert "caveat_accepted" in text
    assert "other-team pass, Codex pass, and automated" in text
