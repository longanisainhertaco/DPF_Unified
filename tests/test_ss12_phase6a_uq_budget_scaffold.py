from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_ss12_phase6a_uq_budget_scaffold.py"
SCAFFOLD = ROOT / "docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json"


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


def test_phase6a_uq_budget_scaffold_validates_blocked_state() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True


def test_phase6a_uq_rows_map_one_to_one_to_phase5e_digitization_packets() -> None:
    scaffold = _load_scaffold()
    phase5e = json.loads((ROOT / "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json").read_text())

    assert len(scaffold["uq_budget_rows"]) == len(phase5e["digitization_packets"]) == 4
    assert {row["digitization_packet_id"] for row in scaffold["uq_budget_rows"]} == {
        packet["id"] for packet in phase5e["digitization_packets"]
    }
    for row in scaffold["uq_budget_rows"]:
        assert row["uq_status"] == "blocked_digitization_not_reviewed"
        assert row["source_uncertainty"] is None
        assert row["digitization_uncertainty"] is None
        assert row["calibration_uncertainty"] is None
        assert row["numerical_uncertainty"] is None
        assert row["model_inadequacy_uncertainty"] is None
        assert row["combined_uncertainty"] is None
        assert row["review_certificate_status"] == "missing"
        assert row["accepted_uq_claim"] is False
        assert row["accepted_runtime_claim"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False


def test_phase6a_validator_rejects_accepted_uq_claim(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["uq_budget_rows"][0]["accepted_uq_claim"] = True
    mutated = tmp_path / "accepted_uq.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "uq_acceptance_flag_not_false" in completed.stdout


def test_phase6a_validator_rejects_complete_uq_without_all_terms(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["uq_budget_rows"][0]
    row["uq_status"] = "complete_not_accepted"
    row["source_uncertainty"] = 0.1
    row["digitization_uncertainty"] = 0.1
    row["calibration_uncertainty"] = 0.1
    row["combined_uncertainty"] = 0.3
    row["review_certificate_status"] = "complete"
    mutated = tmp_path / "missing_terms.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_uq_requires_all_uncertainty_terms" in completed.stdout


def test_phase6a_validator_rejects_nonfinite_uncertainty(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["uq_budget_rows"][0]
    row["uq_status"] = "complete_not_accepted"
    row["source_uncertainty"] = float("nan")
    row["digitization_uncertainty"] = 0.1
    row["calibration_uncertainty"] = 0.1
    row["numerical_uncertainty"] = 0.1
    row["model_inadequacy_uncertainty"] = 0.1
    row["combined_uncertainty"] = 0.2
    row["review_certificate_status"] = "complete"
    mutated = tmp_path / "nonfinite_uq.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_json_nonfinite_or_malformed" in completed.stdout


def test_phase6a_validator_rejects_complete_uq_without_review(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["uq_budget_rows"][0]
    row["uq_status"] = "complete_not_accepted"
    row["source_uncertainty"] = 0.1
    row["digitization_uncertainty"] = 0.1
    row["calibration_uncertainty"] = 0.1
    row["numerical_uncertainty"] = 0.1
    row["model_inadequacy_uncertainty"] = 0.1
    row["combined_uncertainty"] = 0.22360679775
    mutated = tmp_path / "complete_without_review.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_uq_requires_review_certificate" in completed.stdout


def test_phase6a_validator_rejects_digitization_packet_mapping_mismatch(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["uq_budget_rows"][0]["digitization_packet_id"] = "missing_digitization_packet"
    mutated = tmp_path / "bad_mapping.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "uq_rows_do_not_match_phase5e_digitization_packets" in completed.stdout


def test_phase6a_validator_rejects_figure_or_crop_mismatch_for_digitization_packet(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["uq_budget_rows"][0]["figure_source_id"] = "wrong_figure_source"
    mutated = tmp_path / "bad_row_linkage.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "uq_row_digitization_linkage_mismatch" in completed.stdout


def test_phase6a_validator_rejects_top_level_acceptance_flag(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["acceptance_boundary"]["promotes_acceptance"] = True
    mutated = tmp_path / "top_level_acceptance.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "top_level_acceptance_flag_not_false" in completed.stdout


def test_phase6a_validator_rejects_blocked_row_with_uncertainty_value(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["uq_budget_rows"][0]["source_uncertainty"] = 0.1
    mutated = tmp_path / "blocked_with_uncertainty.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "blocked_uq_row_uncertainty_forbidden" in completed.stdout


def test_phase6a_validator_rejects_arbitrary_uq_status(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["uq_budget_rows"][0]["uq_status"] = "accepted_by_typo"
    mutated = tmp_path / "bad_status.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_uq_status" in completed.stdout


def test_phase6a_validator_rejects_external_phase5e_linkage_path(tmp_path) -> None:
    scaffold = _load_scaffold()
    spoof = tmp_path / "spoofed_phase5e.json"
    spoof.write_text(json.dumps({"digitization_packets": []}))
    scaffold["phase5e_digitization_scaffold"] = str(spoof)
    mutated = tmp_path / "external_phase5e_path.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase5e_digitization_scaffold_not_canonical" in completed.stdout


def test_phase6a_validator_rejects_traversal_phase5e_linkage_path(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase5e_digitization_scaffold"] = "../outside_phase5e.json"
    mutated = tmp_path / "traversal_phase5e_path.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase5e_digitization_scaffold_not_canonical" in completed.stdout


def test_phase6a_validator_rejects_traversal_to_canonical_phase5e_path(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase5e_digitization_scaffold"] = "docs/../docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json"
    mutated = tmp_path / "traversal_to_canonical_phase5e_path.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase5e_digitization_scaffold_not_canonical" in completed.stdout


def test_phase6a_validator_rejects_absolute_canonical_phase5e_path(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase5e_digitization_scaffold"] = str(ROOT / "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json")
    mutated = tmp_path / "absolute_canonical_phase5e_path.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "phase5e_digitization_scaffold_not_canonical" in completed.stdout
