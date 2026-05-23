from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_ss12_phase5e_digitization_schema.py"
SCAFFOLD = ROOT / "docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json"


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


def test_phase5e_digitization_scaffold_validates() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True


def test_phase5e_scaffold_has_one_non_promoting_packet_per_crop() -> None:
    scaffold = _load_scaffold()

    assert scaffold["scaffold_id"] == "ss12_p1_phase5e_digitization_scaffold"
    assert len(scaffold["digitization_packets"]) == 4
    assert scaffold["acceptance_boundary"]["accepted_digitization_claim"] is False
    assert scaffold["acceptance_boundary"]["promotes_acceptance"] is False
    for packet in scaffold["digitization_packets"]:
        assert packet["digitization_status"] == "blocked_calibration_missing"
        assert packet["axis_calibration_status"] == "missing"
        assert packet["digitized_series"] == []
        assert packet["digitization_hash"] is None
        assert packet["uncertainty_budget_status"] == "missing"
        assert packet["review_certificate_status"] == "missing"
        assert packet["accepted_digitization_claim"] is False
        assert packet["accepted_runtime_claim"] is False
        assert packet["promotes_acceptance"] is False
        assert packet["can_support_first_principles_acceptance"] is False


def test_phase5e_validator_rejects_accepted_digitization_claim(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["digitization_packets"][0]["accepted_digitization_claim"] = True
    mutated = tmp_path / "accepted_digitization.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "digitization_acceptance_flag_not_false" in completed.stdout


def test_phase5e_validator_rejects_digitized_series_without_calibration(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["digitization_packets"][0]["digitized_series"] = [{"x": 1.0, "y": 2.0}]
    mutated = tmp_path / "series_without_calibration.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "digitized_series_requires_axis_calibration" in completed.stdout


def test_phase5e_validator_rejects_complete_status_without_uncertainty_and_review(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["digitization_status"] = "digitized_not_reviewed"
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    packet["digitized_series"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    packet["digitization_hash"] = "a" * 64
    mutated = tmp_path / "digitized_without_review.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "digitized_packet_requires_uncertainty_and_review" in completed.stdout


def test_phase5e_validator_rejects_blocked_packet_with_digitization_hash(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["digitization_packets"][0]["digitization_hash"] = "a" * 64
    mutated = tmp_path / "blocked_with_hash.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "blocked_packet_digitization_hash_forbidden" in completed.stdout


def test_phase5e_validator_rejects_non_hex_digitization_hash(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["digitization_status"] = "digitized_not_reviewed"
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    packet["digitized_series"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    packet["digitization_hash"] = "z" * 64
    mutated = tmp_path / "non_hex_hash.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_digitization_hash" in completed.stdout


def test_phase5e_validator_rejects_bool_calibration_points(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[True, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    mutated = tmp_path / "bool_calibration.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_axis_calibration" in completed.stdout


def test_phase5e_validator_rejects_malformed_digitized_series(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["digitization_status"] = "digitized_not_reviewed"
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    packet["digitized_series"] = [{"x": True, "y": 0.0}]
    packet["digitization_hash"] = "a" * 64
    mutated = tmp_path / "bad_series.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_digitized_series_point" in completed.stdout


def test_phase5e_validator_rejects_nonfinite_digitized_series(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["digitization_status"] = "digitized_not_reviewed"
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    packet["digitized_series"] = [{"x": float("nan"), "y": 0.0}]
    packet["digitization_hash"] = "a" * 64
    packet["uncertainty_budget_status"] = "complete"
    packet["review_certificate_status"] = "complete"
    mutated = tmp_path / "nonfinite_series.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_json_nonfinite_or_malformed" in completed.stdout


def test_phase5e_validator_rejects_nonfinite_calibration_points(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[float("inf"), 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    mutated = tmp_path / "nonfinite_calibration.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_json_nonfinite_or_malformed" in completed.stdout


def test_phase5e_validator_rejects_arbitrary_status_strings(tmp_path) -> None:
    scaffold = _load_scaffold()
    packet = scaffold["digitization_packets"][0]
    packet["digitization_status"] = "accepted_by_typo"
    packet["axis_calibration_status"] = "calibrated"
    packet["axis_calibration"] = {
        "x_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "ns"},
        "y_axis": {"points": [[0, 0.0], [100, 1.0]], "unit": "a.u."},
    }
    packet["digitized_series"] = [{"x": 0.0, "y": 0.0}]
    packet["digitization_hash"] = "a" * 64
    packet["uncertainty_budget_status"] = "complete"
    packet["review_certificate_status"] = "complete"
    mutated = tmp_path / "bad_status.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "invalid_digitization_status" in completed.stdout
