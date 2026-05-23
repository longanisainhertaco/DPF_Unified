from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json"
VALIDATOR = ROOT / "scripts/validate_ss12_phase5_figure_source_manifest.py"

REQUIRED_BOUNDARY_FLAGS = (
    "accepted_figure_claim",
    "accepted_observable_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
REQUIRED_ROW_FIELDS = (
    "id",
    "channel",
    "source_path",
    "line_start",
    "line_end",
    "figure_id",
    "scope_classification",
    "extraction_priority",
    "review_state",
    "status",
)
FORBIDDEN_ACCEPTED_STATES = {"accepted", "reviewed_as_accepted", "same_source_accepted"}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_phase5_manifest_schema_and_acceptance_boundary() -> None:
    manifest = _load_manifest()

    assert manifest["manifest_id"] == "ss12_p1_phase5_figure_source_manifest"
    assert manifest["validation_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert isinstance(manifest["figure_sources"], list)
    assert len(manifest["figure_sources"]) >= 4

    boundary = manifest["acceptance_boundary"]
    for flag in REQUIRED_BOUNDARY_FLAGS:
        assert boundary[flag] is False

    for row in manifest["figure_sources"]:
        for field in REQUIRED_ROW_FIELDS:
            assert field in row, f"{field} missing from {row.get('id')}"
        assert row["review_state"] not in FORBIDDEN_ACCEPTED_STATES
        assert row["status"] not in FORBIDDEN_ACCEPTED_STATES
        assert row.get("accepted_observable_claim") is False
        assert row.get("promotes_acceptance") is False
        assert row.get("can_support_first_principles_acceptance") is False


def test_phase5_manifest_source_paths_are_local_and_line_cited() -> None:
    manifest = _load_manifest()

    for row in manifest["figure_sources"]:
        source_path = row["source_path"]
        if row["status"] == "blocked_missing_line_citable_source":
            assert source_path is None
            assert row["line_start"] is None
            assert row["line_end"] is None
            assert row.get("blocked_reason")
            continue

        resolved = (ROOT / source_path).resolve()
        assert resolved.is_relative_to(ROOT.resolve())
        assert resolved.exists(), source_path
        lines = resolved.read_text(errors="ignore").splitlines()
        assert 1 <= row["line_start"] <= row["line_end"] <= len(lines)
        excerpt = " ".join(lines[row["line_start"] - 1 : row["line_end"]]).lower()
        assert any(token.lower() in excerpt for token in row["evidence_tokens"])


def test_phase5_manifest_has_required_channels_or_explicit_blockers() -> None:
    manifest = _load_manifest()
    rows_by_channel = {row["channel"]: row for row in manifest["figure_sources"]}

    for channel in (
        "current_waveform",
        "density_history",
        "em_field_history",
        "neutron_timing_or_spectrum",
    ):
        assert channel in rows_by_channel
        row = rows_by_channel[channel]
        assert row["status"] in {
            "figure_source_candidate",
            "transfer_figure_source_candidate",
            "blocked_missing_line_citable_source",
        }


def test_phase5_validator_accepts_manifest() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(VALIDATOR),
            "--repo-root",
            str(ROOT),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["passed"] is True
    assert summary["issue_count"] == 0


def test_phase5_validator_rejects_accepted_row(tmp_path) -> None:
    manifest = _load_manifest()
    manifest["figure_sources"][0]["status"] = " Accepted "
    mutated = tmp_path / "mutated_manifest.json"
    mutated.write_text(json.dumps(manifest))

    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(VALIDATOR),
            "--repo-root",
            str(ROOT),
            "--manifest",
            str(mutated),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert "figure_source_accepted_status_forbidden" in {
        issue["rule"] for issue in summary["issues"]
    }


def test_phase5_validator_rejects_row_level_acceptance_flags(tmp_path) -> None:
    manifest = _load_manifest()
    manifest["figure_sources"][0]["accepted_figure_claim"] = True
    manifest["figure_sources"][0]["accepted_runtime_claim"] = True
    mutated = tmp_path / "mutated_manifest.json"
    mutated.write_text(json.dumps(manifest))

    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(VALIDATOR),
            "--repo-root",
            str(ROOT),
            "--manifest",
            str(mutated),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    flagged = {issue["flag"] for issue in summary["issues"] if issue["rule"] == "figure_source_acceptance_flag_not_false"}
    assert {"accepted_figure_claim", "accepted_runtime_claim"} <= flagged


def test_phase5_validator_rejects_source_path_escape(tmp_path) -> None:
    manifest = _load_manifest()
    manifest["figure_sources"][0]["source_path"] = "../outside.md"
    mutated = tmp_path / "mutated_manifest.json"
    mutated.write_text(json.dumps(manifest))

    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(VALIDATOR),
            "--repo-root",
            str(ROOT),
            "--manifest",
            str(mutated),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert "figure_source_outside_repo" in {issue["rule"] for issue in summary["issues"]}
