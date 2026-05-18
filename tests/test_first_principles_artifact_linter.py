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
            "provenance_complete": True,
            "missing_provenance_fields": [],
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


# ---------------------------------------------------------------------------
# Codex A-1: C7 -- manifest.provenance_complete
# ---------------------------------------------------------------------------


def test_artifact_linter_passes_full_provenance_artifact(tmp_path: Path) -> None:
    """A current artifact whose manifest reports complete provenance and
    satisfies every other check passes all seven checks."""
    linter = _load_linter()
    artifact = _write_json(tmp_path / "full.json", _valid_first_principles_artifact())

    result = linter.lint_artifact(artifact)

    assert result.status == "PASS"
    assert result.failed_checks == []
    assert result.counts_against_exit is False


def test_artifact_linter_fails_provenance_complete_false(tmp_path: Path) -> None:
    """A-1: a first-principles artifact whose manifest reports
    ``provenance_complete: false`` fails check C7."""
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["provenance_complete"] = False
    payload["manifest"]["missing_provenance_fields"] = ["source_truth_index_sha256"]
    artifact = _write_json(tmp_path / "no_provenance.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "FAIL"
    assert "C7" in result.failed_checks
    assert result.counts_against_exit is True


def test_artifact_linter_fails_missing_provenance_complete_key(tmp_path: Path) -> None:
    """A-1: a manifest with no ``provenance_complete`` key at all fails C7 --
    absence is treated as incomplete provenance, not as a pass."""
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    payload["manifest"].pop("provenance_complete", None)
    artifact = _write_json(tmp_path / "no_key.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "FAIL"
    assert "C7" in result.failed_checks


# ---------------------------------------------------------------------------
# Codex A-2: archive / non-authority scope policy
# ---------------------------------------------------------------------------


def test_artifact_linter_exempts_archived_artifact_with_reason(
    tmp_path: Path,
) -> None:
    """A-2: an artifact under results/archive_stale_pre_ssr* is reported
    EXEMPT with an explicit reason, never silently skipped, and never fails
    the run -- even if it would otherwise fail C7."""
    linter = _load_linter()
    archive_dir = tmp_path / "results" / "archive_stale_pre_ssr_2026_05_18"
    archive_dir.mkdir(parents=True)
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["provenance_complete"] = False  # would fail C7 if checked
    artifact = _write_json(archive_dir / "stale_probe.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "EXEMPT"
    assert result.exempt_reason is not None
    assert "archive_stale_pre_ssr" in result.exempt_reason
    assert "cannot support first-principles acceptance" in result.exempt_reason
    assert result.failed_checks == []
    assert result.counts_against_exit is False


def test_artifact_linter_exempts_non_authority_evidence_with_reason(
    tmp_path: Path,
) -> None:
    """A-2: non-authority evidence surfaces (checkpoint/restart,
    reproducibility, split-continuation, numerical-family) are EXEMPT with a
    status reason proving they cannot support first-principles acceptance."""
    linter = _load_linter()
    non_authority_tools = (
        "dpf experimental-checkpoint-restart",
        "dpf experimental-reproducibility",
        "dpf experimental-split-continuation",
        "dpf experimental-numerical-family",
    )
    for index, tool in enumerate(non_authority_tools):
        artifact = _write_json(
            tmp_path / f"probe_{index}.json",
            {
                "tool": tool,
                "scientific_status": "engineering_candidate_not_validation",
                "can_support_first_principles_acceptance": False,
            },
        )

        result = linter.lint_artifact(artifact)

        assert result.status == "EXEMPT", tool
        assert result.exempt_reason is not None
        assert "no candidate physics ledger" in result.exempt_reason
        assert result.counts_against_exit is False


def test_artifact_linter_exit_code_ignores_exempt_artifacts(tmp_path: Path) -> None:
    """A-2: a directory mixing a passing artifact, an exempt non-authority
    probe, and an exempt archived artifact exits 0 -- exempt artifacts never
    fail the run, while the passing artifact still passes."""
    linter = _load_linter()
    _write_json(tmp_path / "good.json", _valid_first_principles_artifact())
    _write_json(
        tmp_path / "checkpoint_probe.json",
        {"tool": "dpf experimental-checkpoint-restart", "scientific_status": "x"},
    )
    archive_dir = tmp_path / "archive_stale_pre_ssr_2026_05_18"
    archive_dir.mkdir()
    _write_json(archive_dir / "stale.json", _valid_first_principles_artifact())

    exit_code = linter.main([str(tmp_path / "*.json"), str(archive_dir / "*.json")])

    assert exit_code == 0


def test_artifact_linter_exit_code_fails_on_provenance_gap(tmp_path: Path) -> None:
    """A-1: an active (non-exempt) artifact with incomplete provenance fails
    the run with a nonzero exit code."""
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["provenance_complete"] = False
    _write_json(tmp_path / "active_gap.json", payload)

    exit_code = linter.main([str(tmp_path / "*.json")])

    assert exit_code == 1
