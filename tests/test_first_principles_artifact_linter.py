from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_first_principles_artifacts.py"


def _load_linter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("artifact_linter", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _valid_first_principles_artifact() -> dict[str, object]:
    return {
        "tool": "dpf first-principles-3d",
        "artifact_generation_commit": "0123456789abcdef0123456789abcdef01234567",
        "command_argv": ["dpf", "first-principles-3d", "--steps", "2"],
        "conservation_telemetry": {
            "finite_state": True,
            "energy_conservation_assessed": (
                "not_assessed_no_accepted_tolerance"
            ),
        },
        "telemetry_packets": {
            "power_port": {
                "stage0_packet_scaffolds": {
                    "status": "candidate_stage0_packet_scaffolds_not_validation",
                },
            },
        },
        "manifest": {
            "candidate_evidence": {
                "deck_diff_packet": {
                    "status": "candidate_deck_diff_packet_not_validation",
                },
            },
        },
        "deck": {"preset": "pf1000_akel_16kv"},
        "can_support_first_principles_acceptance": False,
    }


def test_artifact_linter_accepts_current_schema_first_principles_artifact(
    tmp_path: Path,
) -> None:
    linter = _load_linter()
    artifact = _write_json(tmp_path / "current.json", _valid_first_principles_artifact())

    result = linter.lint_artifact(artifact)

    assert result.status == "PASS"
    assert result.failed_checks == []


def test_artifact_linter_rejects_stale_conservation_and_missing_provenance(
    tmp_path: Path,
) -> None:
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    payload.pop("artifact_generation_commit")
    payload.pop("command_argv")
    assert isinstance(payload["conservation_telemetry"], dict)
    payload["conservation_telemetry"]["passed"] = True
    artifact = _write_json(tmp_path / "stale.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "FAIL"
    assert set(result.failed_checks) >= {"C1", "C2", "C3"}


def test_artifact_linter_rejects_any_acceptance_true(tmp_path: Path) -> None:
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    payload["nested"] = {"can_support_first_principles_acceptance": True}
    artifact = _write_json(tmp_path / "promoting.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "FAIL"
    assert "C6" in result.failed_checks


def test_artifact_linter_skips_non_first_principles_artifacts(tmp_path: Path) -> None:
    linter = _load_linter()
    artifact = _write_json(
        tmp_path / "calibration.json",
        {"tool": "dpf inverse-calibration", "status": "candidate"},
    )

    result = linter.lint_artifact(artifact)

    assert result.status == "SKIP"
    assert result.counts_against_exit is False
