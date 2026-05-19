from __future__ import annotations

import importlib.util
import json
import subprocess
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
            "command_argv": ["dpf", "first-principles-3d", "--steps", "2"],
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "source_truth_index_sha256": "a" * 64,
            "source_packet_hashes": {"hybrid_pic_3d_source": "b" * 64},
            "input_deck_sha256": "c" * 64,
            "artifact_schema_version": "first_principles_artifact_v1",
            "artifact_generation_commit": "0123456789abcdef0123456789abcdef01234567",
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
    head = _live_head()
    # Stamp the real HEAD so C8 passes for the active artifact.
    good_payload = _valid_current_head_artifact(head)
    _write_json(tmp_path / "good.json", good_payload)
    _write_json(
        tmp_path / "checkpoint_probe.json",
        {"tool": "dpf experimental-checkpoint-restart", "scientific_status": "x"},
    )
    archive_dir = tmp_path / "archive_stale_pre_ssr_2026_05_18"
    archive_dir.mkdir()
    _write_json(archive_dir / "stale.json", _valid_first_principles_artifact())

    exit_code = linter.main([str(tmp_path / "*.json"), str(archive_dir / "*.json")])

    assert exit_code == 0


def test_artifact_linter_fails_c7_on_stale_lying_manifest(tmp_path: Path) -> None:
    """A-1: C7 must NOT trust a self-reported ``provenance_complete: true``.

    A stale/lying manifest can set ``provenance_complete: true`` while
    carrying ``source_packet_hashes: {}`` (empty dict).  C7 must re-derive
    completeness from the actual manifest fields and fail in this case.
    """
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    # Self-reports complete but the hash map is empty -- a stale lying manifest.
    payload["manifest"]["provenance_complete"] = True
    payload["manifest"]["source_packet_hashes"] = {}
    # Populate all other fields so C7 failure is specifically due to empty hashes.
    payload["manifest"]["command_argv"] = ["dpf", "first-principles-3d"]
    payload["manifest"]["git_commit"] = "0123456789abcdef0123456789abcdef01234567"
    payload["manifest"]["source_truth_index_sha256"] = "a" * 64
    payload["manifest"]["input_deck_sha256"] = "b" * 64
    payload["manifest"]["artifact_schema_version"] = "first_principles_artifact_v1"
    payload["manifest"]["artifact_generation_commit"] = "0123456789abcdef0123456789abcdef01234567"
    artifact = _write_json(tmp_path / "lying_manifest.json", payload)

    result = linter.lint_artifact(artifact)

    assert result.status == "FAIL"
    assert "C7" in result.failed_checks
    assert result.counts_against_exit is True


def test_artifact_linter_passes_c7_on_genuinely_complete_manifest(tmp_path: Path) -> None:
    """A-1: a manifest that genuinely satisfies every required provenance
    field, including a non-empty ``source_packet_hashes``, does not fail C7.
    """
    linter = _load_linter()
    payload = _valid_first_principles_artifact()
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["provenance_complete"] = True
    payload["manifest"]["source_packet_hashes"] = {
        "hybrid_pic_3d_source": "a" * 64,
    }
    payload["manifest"]["command_argv"] = ["dpf", "first-principles-3d"]
    payload["manifest"]["git_commit"] = "0123456789abcdef0123456789abcdef01234567"
    payload["manifest"]["source_truth_index_sha256"] = "a" * 64
    payload["manifest"]["input_deck_sha256"] = "b" * 64
    payload["manifest"]["artifact_schema_version"] = "first_principles_artifact_v1"
    payload["manifest"]["artifact_generation_commit"] = "0123456789abcdef0123456789abcdef01234567"
    artifact = _write_json(tmp_path / "complete_manifest.json", payload)

    result = linter.lint_artifact(artifact)

    assert "C7" not in result.failed_checks


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


# ---------------------------------------------------------------------------
# RC-7: C7 required-field tuple drift test
# ---------------------------------------------------------------------------


def test_c7_required_provenance_fields_matches_manifest_module() -> None:
    """RC-7: the linter's module-level ``C7_REQUIRED_PROVENANCE_FIELDS``
    constant must equal ``dpf.first_principles.manifest.REQUIRED_PROVENANCE_FIELDS``
    exactly, so a drift in the manifest module is detected immediately."""
    from dpf.first_principles.manifest import REQUIRED_PROVENANCE_FIELDS

    linter = _load_linter()

    assert linter.C7_REQUIRED_PROVENANCE_FIELDS == REQUIRED_PROVENANCE_FIELDS


# ---------------------------------------------------------------------------
# RC-5: C8 -- active artifact commit-match gate
# ---------------------------------------------------------------------------


def _live_head() -> str:
    """Return the current git HEAD SHA for C8 fixture stamping."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _valid_current_head_artifact(head: str) -> dict[str, object]:
    """Build a minimal first-principles artifact whose commit fields equal ``head``
    and whose worktree is clean -- passes C1-C8 on the live tree."""
    return {
        "tool": "dpf first-principles-3d",
        "artifact_generation_commit": head,
        "dirty_worktree": False,
        "command_argv": ["dpf", "first-principles-3d", "--steps", "2"],
        "conservation_telemetry": {
            "finite_state": True,
            "energy_conservation_assessed": "not_assessed_no_accepted_tolerance",
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
            "command_argv": ["dpf", "first-principles-3d", "--steps", "2"],
            "git_commit": head,
            "source_truth_index_sha256": "a" * 64,
            "source_packet_hashes": {"hybrid_pic_3d_source": "b" * 64},
            "input_deck_sha256": "c" * 64,
            "artifact_schema_version": "first_principles_artifact_v1",
            "artifact_generation_commit": head,
            "candidate_evidence": {
                "deck_diff_packet": {
                    "status": "candidate_deck_diff_packet_not_validation",
                },
            },
        },
        "deck": {"preset": "pf1000_akel_16kv"},
        "can_support_first_principles_acceptance": False,
    }


# RC-6: positive current-schema fixture that passes C1-C8 (a real PASS)

def test_artifact_linter_current_head_artifact_passes_all_checks(
    tmp_path: Path,
) -> None:
    """RC-6: a dynamically stamped first-principles artifact whose commit fields
    equal the live HEAD and whose worktree flag is False passes every linter
    check C1-C8 (genuine PASS, not SKIP or EXEMPT)."""
    linter = _load_linter()
    head = _live_head()
    artifact = _write_json(
        tmp_path / "current_head.json", _valid_current_head_artifact(head)
    )

    result = linter.lint_artifact(artifact, head_commit=head)

    assert result.status == "PASS", f"failed checks: {result.failed_checks}"
    assert result.failed_checks == []


# C8 negative controls


def test_artifact_linter_c8_fails_stale_top_level_commit(tmp_path: Path) -> None:
    """C8: top-level ``artifact_generation_commit`` != HEAD fails C8."""
    linter = _load_linter()
    head = _live_head()
    payload = _valid_current_head_artifact(head)
    payload["artifact_generation_commit"] = "0" * 40  # stale
    artifact = _write_json(tmp_path / "stale_top.json", payload)

    result = linter.lint_artifact(artifact, head_commit=head)

    assert "C8" in result.failed_checks


def test_artifact_linter_c8_fails_stale_manifest_git_commit(tmp_path: Path) -> None:
    """C8: ``manifest.git_commit`` != HEAD fails C8."""
    linter = _load_linter()
    head = _live_head()
    payload = _valid_current_head_artifact(head)
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["git_commit"] = "0" * 40  # stale
    artifact = _write_json(tmp_path / "stale_git.json", payload)

    result = linter.lint_artifact(artifact, head_commit=head)

    assert "C8" in result.failed_checks


def test_artifact_linter_c8_fails_stale_manifest_artifact_commit(
    tmp_path: Path,
) -> None:
    """C8: ``manifest.artifact_generation_commit`` != HEAD fails C8."""
    linter = _load_linter()
    head = _live_head()
    payload = _valid_current_head_artifact(head)
    assert isinstance(payload["manifest"], dict)
    payload["manifest"]["artifact_generation_commit"] = "0" * 40  # stale
    artifact = _write_json(tmp_path / "stale_manifest_commit.json", payload)

    result = linter.lint_artifact(artifact, head_commit=head)

    assert "C8" in result.failed_checks


def test_artifact_linter_c8_fails_dirty_worktree_true(tmp_path: Path) -> None:
    """C8: ``dirty_worktree: true`` fails C8."""
    linter = _load_linter()
    head = _live_head()
    payload = _valid_current_head_artifact(head)
    payload["dirty_worktree"] = True
    artifact = _write_json(tmp_path / "dirty.json", payload)

    result = linter.lint_artifact(artifact, head_commit=head)

    assert "C8" in result.failed_checks


def test_artifact_linter_c8_fails_missing_dirty_worktree(tmp_path: Path) -> None:
    """C8: absent ``dirty_worktree`` key fails C8 (missing is not False)."""
    linter = _load_linter()
    head = _live_head()
    payload = _valid_current_head_artifact(head)
    payload.pop("dirty_worktree", None)
    artifact = _write_json(tmp_path / "no_dirty.json", payload)

    result = linter.lint_artifact(artifact, head_commit=head)

    assert "C8" in result.failed_checks


def test_artifact_linter_c8_skipped_when_head_is_none(tmp_path: Path) -> None:
    """C8: when head_commit is None (git unavailable), C8 must NOT be appended
    to failed_checks -- degrade gracefully, do not crash."""
    linter = _load_linter()
    head = _live_head()
    # Use an otherwise-valid artifact so only C8 could fire.
    payload = _valid_current_head_artifact(head)
    artifact = _write_json(tmp_path / "no_git.json", payload)

    result = linter.lint_artifact(artifact, head_commit=None)

    assert "C8" not in result.failed_checks
