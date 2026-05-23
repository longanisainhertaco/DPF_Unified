from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_ss12_phase7a_review_certificate.py"
SCAFFOLD = ROOT / "docs/SS12_P1_PHASE7A_REVIEW_CERTIFICATE_SKELETON_2026_05_22.json"
PHASE6C = "docs/SS12_P1_PHASE6C_POWER_PORT_CERTIFICATION_SCAFFOLD_2026_05_22.json"
EXPECTED_OBSERVABLES = {
    "crowbar_timing",
    "current_sheath_acceleration",
    "pinch_focus_dynamics",
    "magnetic_field_history",
    "temperature_distribution_history",
    "neutron_yield_timing_spectrum_anisotropy_detector_response",
}
REQUIRED_UNCERTAINTY_TERMS = {"measurement", "model", "numerical"}
REQUIRED_REVIEW_PLACEHOLDERS = {
    "reviewer_id",
    "reviewer_affiliation",
    "reviewed_at",
    "review_packet_hash",
    "review_status",
    "blocking_findings",
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


def test_phase7a_review_certificate_skeleton_validates_blocked_state() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["accepted_certificate_emitted"] is False


def test_phase7a_schema_maps_every_required_observable_and_placeholders() -> None:
    scaffold = _load_scaffold()

    assert scaffold["phase6c_power_port_certification_scaffold"] == PHASE6C
    assert scaffold["acceptance_boundary"] == {
        "accepted_review_certificate": False,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
        "emits_accepted_certificate": False,
    }
    rows = scaffold["review_certificate_rows"]
    assert {row["observable_id"] for row in rows} == EXPECTED_OBSERVABLES

    for row in rows:
        assert row["output_field_mapping"]["observable_id"] == row["observable_id"]
        assert row["output_field_mapping"]["runtime_output_field"]
        assert row["output_field_mapping"]["source_evidence_field"]
        assert row["output_field_mapping"]["comparison_field"]
        assert set(row["uncertainty_placeholders"]) == REQUIRED_UNCERTAINTY_TERMS
        for term_name, term in row["uncertainty_placeholders"].items():
            assert term["term"] == term_name
            assert term["status"] == "placeholder_incomplete"
            assert term["value"] is None
            assert term["unit"] is None
            assert term["evidence_hash"] is None
        metrics = row["pass_fail_metrics"]
        assert metrics["metric_id"]
        assert metrics["status"] == "placeholder_incomplete"
        assert metrics["tolerance"]["value"] is None
        assert metrics["tolerance"]["unit"] is None
        assert metrics["result"] is None
        assert row["negative_controls"][0]["status"] == "placeholder_incomplete"
        assert row["run_evidence_hashes"]["runtime_run_hash"] is None
        assert row["run_evidence_hashes"]["source_evidence_hash"] is None
        assert row["run_evidence_hashes"]["uq_packet_hash"] is None
        assert set(row["independent_review_placeholders"]) == REQUIRED_REVIEW_PLACEHOLDERS
        assert row["certificate_status"] == "blocked_review_certificate_incomplete"
        assert row["accepted_review_certificate"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False


def test_phase7a_validator_rejects_any_accepted_or_promoted_state(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["acceptance_boundary"]["promotes_acceptance"] = True
    scaffold["review_certificate_rows"][0]["accepted_review_certificate"] = True
    mutated = tmp_path / "forged_accepted_certificate.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "top_level_acceptance_flag_not_false" in completed.stdout
    assert "row_acceptance_flag_not_false" in completed.stdout


def test_phase7a_validator_rejects_complete_certificate_while_review_placeholders_incomplete(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["review_certificate_rows"][0]
    row["certificate_status"] = "complete_not_accepted"
    mutated = tmp_path / "complete_with_placeholders.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "complete_certificate_requires_no_placeholders" in completed.stdout
    assert "complete_certificate_blocked_by_phase7a" in completed.stdout


def test_phase7a_validator_rejects_missing_mapping_uncertainty_controls_hashes_or_review(tmp_path) -> None:
    scaffold = _load_scaffold()
    row = scaffold["review_certificate_rows"][0]
    del row["output_field_mapping"]["comparison_field"]
    del row["uncertainty_placeholders"]["measurement"]
    row["negative_controls"] = []
    del row["run_evidence_hashes"]["source_evidence_hash"]
    del row["independent_review_placeholders"]["review_status"]
    mutated = tmp_path / "missing_required_schema_slots.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "output_field_mapping_incomplete" in completed.stdout
    assert "uncertainty_placeholders_incomplete" in completed.stdout
    assert "negative_controls_missing" in completed.stdout
    assert "run_evidence_hashes_incomplete" in completed.stdout
    assert "independent_review_placeholders_incomplete" in completed.stdout


def test_phase7a_validator_rejects_noncanonical_phase6c_link_and_bad_json_shape(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["phase6c_power_port_certification_scaffold"] = str(ROOT / PHASE6C)
    bad_link = tmp_path / "absolute_phase6c_link.json"
    bad_link.write_text(json.dumps(scaffold))
    bad_shape = tmp_path / "bad_shape.json"
    bad_shape.write_text(json.dumps([]))

    bad_link_completed = _run_validator(bad_link)
    bad_shape_completed = _run_validator(bad_shape)

    assert bad_link_completed.returncode == 1
    assert "phase6c_scaffold_not_canonical" in bad_link_completed.stdout
    assert bad_shape_completed.returncode == 1
    assert "scaffold_not_object" in bad_shape_completed.stdout


def test_phase7a_validator_rejects_duplicate_observable_ids(tmp_path) -> None:
    scaffold = _load_scaffold()
    scaffold["review_certificate_rows"][1]["observable_id"] = scaffold["review_certificate_rows"][0]["observable_id"]
    mutated = tmp_path / "duplicate_observable_ids.json"
    mutated.write_text(json.dumps(scaffold))

    completed = _run_validator(mutated)

    assert completed.returncode == 1
    assert "duplicate_observable_id" in completed.stdout
