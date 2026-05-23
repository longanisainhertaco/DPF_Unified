from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_ss12_phase6c_power_port_certification.py"
SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6C_POWER_PORT_CERTIFICATION_SCAFFOLD_2026_05_22.json"
EXPECTED_PHASE4B_REFS = [
    "src/dpf/first_principles/circuit_power_port.py",
    "tests/test_first_principles_circuit_power_port_phase4b.py",
    "docs/SS12_P1_PHASE4B_EVALUATE_LEARN_CONTINUE_2026_05_22.md",
]
EXPECTED_PHASE6B_REF = "docs/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json"
EXPECTED_CERTIFICATION_ROWS = {
    "cert_crowbar_timing",
    "cert_current_sheath_acceleration",
    "cert_pinch_focus_dynamics",
}


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


def test_phase6c_power_port_certification_scaffold_validates_blocked_state() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["passed"] is True


def test_phase6c_certification_rows_cover_required_dynamics_and_fail_closed() -> None:
    scaffold = _load_scaffold()

    assert scaffold["phase4b_circuit_power_port_artifacts"] == EXPECTED_PHASE4B_REFS
    assert scaffold["phase6b_uq_propagation_scaffold"] == EXPECTED_PHASE6B_REF
    rows = scaffold["power_port_certification_rows"]
    assert {row["id"] for row in rows} == EXPECTED_CERTIFICATION_ROWS
    for row in rows:
        assert row["power_port_evidence_status"] == "blocked_power_port_evidence_incomplete"
        assert row["uq_propagation_status"] == "blocked_uq_budget_incomplete"
        assert row["certification_status"] == "blocked_certification_incomplete"
        assert row["review_certificate_status"] == "missing"
        assert row["accepted_power_port_claim"] is False
        assert row["accepted_runtime_claim"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False
        assert row["certified_observable"] is None
        assert row["certified_uncertainty"] is None


def test_phase6c_validator_rejects_noncanonical_phase4b_ref(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase4b_circuit_power_port_artifacts"][0] = "src/dpf/first_principles/../first_principles/circuit_power_port.py"
    mutated = tmp_path / "bad_phase4b_ref.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase4b_artifacts_not_canonical" in completed.stdout


def test_phase6c_validator_rejects_external_phase6b_ref(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase6b_uq_propagation_scaffold"] = "https://example.test/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json"
    mutated = tmp_path / "external_phase6b_ref.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase6b_uq_propagation_scaffold_not_canonical" in completed.stdout


def test_phase6c_validator_rejects_absolute_canonical_looking_phase6b_ref(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase6b_uq_propagation_scaffold"] = str(ROOT / EXPECTED_PHASE6B_REF)
    mutated = tmp_path / "absolute_phase6b_ref.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase6b_uq_propagation_scaffold_not_canonical" in completed.stdout


def test_phase6c_validator_rejects_accepted_certification_claim(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["power_port_certification_rows"][0]["promotes_acceptance"] = True
    mutated = tmp_path / "accepted_certification.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "certification_acceptance_flag_not_false" in completed.stdout


def test_phase6c_validator_rejects_arbitrary_certification_status(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["power_port_certification_rows"][0]["certification_status"] = "accepted_by_typo"
    mutated = tmp_path / "bad_status.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_certification_status" in completed.stdout


def test_phase6c_validator_rejects_blocked_row_with_certified_values(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["power_port_certification_rows"][0]["certified_observable"] = 1.0
    mutated = tmp_path / "blocked_with_value.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "blocked_certification_values_forbidden" in completed.stdout


def test_phase6c_validator_rejects_nonfinite_certified_values(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["power_port_certification_rows"][0]
    row["certification_status"] = "complete_not_accepted"
    row["power_port_evidence_status"] = "complete_not_accepted"
    row["uq_propagation_status"] = "complete_not_accepted"
    row["review_certificate_status"] = "complete"
    row["certified_observable"] = float("nan")
    row["certified_uncertainty"] = 0.1
    mutated = tmp_path / "nonfinite_certified_value.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_json_nonfinite_or_malformed" in completed.stdout


def test_phase6c_validator_rejects_complete_certification_without_review(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["power_port_certification_rows"][0]
    row["certification_status"] = "complete_not_accepted"
    row["power_port_evidence_status"] = "complete_not_accepted"
    row["uq_propagation_status"] = "complete_not_accepted"
    row["certified_observable"] = 1.0
    row["certified_uncertainty"] = 0.1
    mutated = tmp_path / "complete_without_review.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_certification_requires_review_certificate" in completed.stdout


def test_phase6c_validator_rejects_duplicate_certification_row_ids(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["power_port_certification_rows"][1]["id"] = scaffold["power_port_certification_rows"][0]["id"]
    mutated = tmp_path / "duplicate_ids.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "duplicate_certification_row_id" in completed.stdout


def test_phase6c_validator_rejects_top_level_acceptance_flag(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["acceptance_boundary"]["accepted_runtime_claim"] = True
    mutated = tmp_path / "top_level_acceptance.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "top_level_acceptance_flag_not_false" in completed.stdout


def test_phase6c_validator_rejects_non_object_json(tmp_path) -> None:
    mutated = tmp_path / "non_object.json"
    mutated.write_text(json.dumps([]))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "scaffold_not_object" in completed.stdout
