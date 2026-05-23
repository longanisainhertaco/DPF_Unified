from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_ss12_phase6b_uq_propagation_scaffold.py"
SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json"


def _run_validator(path: Path = SCAFFOLD) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(SCRIPT), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_scaffold() -> dict:
    return json.loads(SCAFFOLD.read_text())


def test_phase6b_uq_propagation_scaffold_validates_blocked_state() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["passed"] is True


def test_phase6b_propagation_rows_are_blocked_and_non_promoting() -> None:
    scaffold = _load_scaffold()
    phase6a = json.loads((ROOT / "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json").read_text())

    assert scaffold["phase6a_uq_budget_scaffold"] == "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"
    assert len(scaffold["propagation_packets"]) == len(phase6a["uq_budget_rows"]) == 4
    assert {row["uq_budget_row_id"] for row in scaffold["propagation_packets"]} == {row["id"] for row in phase6a["uq_budget_rows"]}
    for packet in scaffold["propagation_packets"]:
        assert packet["propagation_status"] == "blocked_uq_budget_incomplete"
        assert packet["propagated_observable"] is None
        assert packet["propagated_uncertainty"] is None
        assert packet["review_certificate_status"] == "missing"
        assert packet["accepted_propagation_claim"] is False
        assert packet["accepted_runtime_claim"] is False
        assert packet["promotes_acceptance"] is False
        assert packet["can_support_first_principles_acceptance"] is False


def test_phase6b_validator_rejects_noncanonical_phase6a_link(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase6a_uq_budget_scaffold"] = "docs/../docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"
    mutated = tmp_path / "bad_phase6a_link.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase6a_uq_budget_scaffold_not_canonical" in completed.stdout


def test_phase6b_validator_rejects_accepted_propagation_claim(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["propagation_packets"][0]["accepted_propagation_claim"] = True
    mutated = tmp_path / "accepted_propagation.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "propagation_acceptance_flag_not_false" in completed.stdout


def test_phase6b_validator_rejects_complete_propagation_without_review(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["propagation_packets"][0]
    packet["propagation_status"] = "complete_not_accepted"
    packet["propagated_observable"] = 1.0
    packet["propagated_uncertainty"] = 0.1
    mutated = tmp_path / "complete_without_review.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_propagation_requires_review_certificate" in completed.stdout


def test_phase6b_validator_rejects_nonfinite_propagated_values(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["propagation_packets"][0]
    packet["propagation_status"] = "complete_not_accepted"
    packet["propagated_observable"] = float("inf")
    packet["propagated_uncertainty"] = 0.1
    packet["review_certificate_status"] = "complete"
    mutated = tmp_path / "nonfinite_propagation.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_json_nonfinite_or_malformed" in completed.stdout


def test_phase6b_validator_rejects_arbitrary_propagation_status(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["propagation_packets"][0]["propagation_status"] = "accepted_by_typo"
    mutated = tmp_path / "bad_status.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_propagation_status" in completed.stdout


def test_phase6b_validator_rejects_blocked_packet_with_propagated_values(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["propagation_packets"][0]["propagated_observable"] = 1.0
    mutated = tmp_path / "blocked_with_values.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "blocked_propagation_values_forbidden" in completed.stdout


def test_phase6b_validator_rejects_uq_linkage_mismatch(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["propagation_packets"][0]["figure_source_id"] = "wrong_figure_source"
    mutated = tmp_path / "bad_linkage.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "propagation_packet_uq_linkage_mismatch" in completed.stdout


def test_phase6b_validator_rejects_negative_propagated_uncertainty(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["propagation_packets"][0]
    packet["propagation_status"] = "complete_not_accepted"
    packet["propagated_observable"] = 1.0
    packet["propagated_uncertainty"] = -0.1
    packet["review_certificate_status"] = "complete"
    mutated = tmp_path / "negative_uncertainty.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_propagation_requires_finite_values" in completed.stdout


def test_phase6b_validator_rejects_top_level_acceptance_flag(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["acceptance_boundary"]["promotes_acceptance"] = True
    mutated = tmp_path / "top_level_acceptance.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "top_level_acceptance_flag_not_false" in completed.stdout


def test_phase6b_validator_rejects_non_object_json(tmp_path) -> None:
    mutated = tmp_path / "non_object.json"
    mutated.write_text(json.dumps([]))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "scaffold_not_object" in completed.stdout
