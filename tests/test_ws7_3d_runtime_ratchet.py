"""Sprint 8 WS7: 3-D runtime ratchet tests.

Covers:
- CLI parity: experimental-segmented-whole-shot accepts --dt-policy and
  --auto-step-budget (same surface as experimental-whole-shot).
- Duration-request satisfaction is explicit in every run manifest.
- The WS3 24-rod deck preset is selectable from the CLI.
- combine-whole-run CLI route merges contiguous run directories.
- hybrid_pic_3d_readiness continues to list missing capabilities and
  stays blocked; acceptance is never unlocked by runtime success.

These tests are engineering-candidate probes.  None of them set
accepted_runtime_claim or can_support_first_principles_acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from dpf.cli.main import cli

# ---------------------------------------------------------------------------
# CLI parity: --dt-policy on experimental-segmented-whole-shot
# ---------------------------------------------------------------------------


def test_segmented_whole_shot_accepts_dt_policy_deck(tmp_path: Path) -> None:
    """--dt-policy deck is the default and must not alter the run."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "2",
            "--explicit-total-steps", "4",
            "--run-dir", str(run_dir),
            "--dt-policy", "deck",
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == (
        "experimental_segmented_whole_shot_engineering_candidate_not_validation"
    )
    assert payload["can_support_first_principles_acceptance"] is False
    assert payload["horizon_complete"] is True
    assert payload["total_steps_completed"] == 4


def test_segmented_whole_shot_dt_policy_vacuum_cfl_is_accepted(tmp_path: Path) -> None:
    """--dt-policy vacuum-cfl must not error and must produce a valid manifest."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "2",
            "--explicit-total-steps", "4",
            "--run-dir", str(run_dir),
            "--dt-policy", "vacuum-cfl",
            "--vacuum-cfl", "0.95",
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == (
        "experimental_segmented_whole_shot_engineering_candidate_not_validation"
    )
    assert payload["can_support_first_principles_acceptance"] is False
    assert payload["horizon_complete"] is True


def test_segmented_whole_shot_auto_step_budget_resolves_explicit_steps(
    tmp_path: Path,
) -> None:
    """--auto-step-budget + --max-auto-steps must set explicit_total_steps from
    ceil(target_time_s / dt_s) before the segment planner runs."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    # dt=1e-13, target=1e-13 -> ceil(1e-13/1e-13)=1 step; segment-steps=1
    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "1",
            "--target-time-s", "1e-13",
            "--auto-step-budget",
            "--max-auto-steps", "10",
            "--run-dir", str(run_dir),
            "--dt-policy", "deck",
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["horizon_complete"] is True
    assert payload["can_support_first_principles_acceptance"] is False
    # Duration request was satisfied: ran 1 step for 1e-13 s target.
    assert payload["total_steps_completed"] >= 1


def test_segmented_whole_shot_auto_step_budget_guard_blocks_large_run() -> None:
    """--auto-step-budget exceeding --max-auto-steps must fail."""
    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "2",
            "--target-time-s", "1e-6",
            "--auto-step-budget",
            "--max-auto-steps", "2",
            "--run-dir", "/tmp/should_not_exist_ws7_guard",
            "--dt-policy", "deck",
        ],
    )

    assert result.exit_code != 0
    assert "auto step budget would require" in result.output


# ---------------------------------------------------------------------------
# Duration-request satisfaction is explicit
# ---------------------------------------------------------------------------


def test_segmented_whole_shot_duration_satisfaction_explicit_in_manifest(
    tmp_path: Path,
) -> None:
    """horizon_complete and total_steps_completed must be explicit in manifests."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "3",
            "--explicit-total-steps", "6",
            "--run-dir", str(run_dir),
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    # Duration status is explicit, never absent.
    assert "horizon_complete" in payload
    assert "total_steps_completed" in payload
    # WholeShotPlan serialises as "total_steps" (not "planned_total_steps").
    assert "total_steps" in payload["plan"]
    # Blocker verdicts include the compute-wall and wall-time-cap IDs.
    verdict_ids = {v["id"] for v in payload["blocker_verdicts"]["verdicts"]}
    assert "B-WPN4-12US-COMPUTE-WALL" in verdict_ids
    assert "B-WPN4-WALL-TIME-CAP" in verdict_ids
    assert payload["can_support_first_principles_acceptance"] is False


def test_segmented_whole_shot_wall_time_truncated_duration_unsatisfied(
    tmp_path: Path,
) -> None:
    """A wall-time-capped partial run must report horizon_complete=False."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--segment-steps", "2",
            "--explicit-total-steps", "40",
            "--run-dir", str(run_dir),
            "--wall-time-cap-s", "1e-9",
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["wall_time_cap_reached"] is True
    assert payload["horizon_complete"] is False
    assert payload["total_steps_completed"] < 40
    verdicts = {v["id"]: v for v in payload["blocker_verdicts"]["verdicts"]}
    assert verdicts["B-WPN4-WALL-TIME-CAP"]["status"] == "triggered"
    assert payload["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# WS3 24-rod deck preset is selectable via CLI
# ---------------------------------------------------------------------------


def test_segmented_whole_shot_24rod_deck_preset_accepted(tmp_path: Path) -> None:
    """pf1000_scholz_2001_24rod_full_energy preset must be selectable and
    produce a manifest tagged with the selected scope."""
    run_dir = tmp_path / "run"
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        cli,
        [
            "experimental-segmented-whole-shot",
            "--deck-preset", "pf1000_scholz_2001_24rod_full_energy",
            "--segment-steps", "2",
            "--explicit-total-steps", "4",
            "--run-dir", str(run_dir),
            "--no-verify-restart-equivalence",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == (
        "experimental_segmented_whole_shot_engineering_candidate_not_validation"
    )
    assert payload["can_support_first_principles_acceptance"] is False
    # The deck must be tagged with the selected scope.
    deck_name = payload["deck_name"]
    assert "pf1000" in deck_name or "pf_1000" in deck_name.lower() or (
        "scholz" in deck_name.lower()
    ), f"unexpected deck_name: {deck_name}"


def test_first_principles_3d_24rod_deck_preset_is_non_promoting() -> None:
    """The 24-rod preset on first-principles-3d must not set acceptance=True."""
    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck-preset",
            "pf1000_scholz_2001_24rod_full_energy",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["can_support_first_principles_acceptance"] is False
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    # The deck source must be tagged as the 24-rod built-in.
    assert "pf1000_scholz_2001_24rod_full_energy" in payload["deck"]["source"]


# ---------------------------------------------------------------------------
# combine-whole-run CLI route
# ---------------------------------------------------------------------------


def test_combine_whole_run_merges_two_contiguous_run_dirs(tmp_path: Path) -> None:
    """combine-whole-run must produce a combined manifest from two restart dirs."""
    from dpf.first_principles.runner import FirstPrinciples3DDeck
    from dpf.first_principles.segmented_whole_shot import run_segmented_whole_shot

    deck = FirstPrinciples3DDeck.from_deck({"n_steps": 6, "seed": 0})

    # First run: steps 0..4 (two segments of 2, wall-time cap after segment 1).
    run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "first",
        segment_steps=2,
        explicit_total_steps=6,
        checkpoint_every_segments=1,
        wall_time_cap_s=1e-9,
        verify_restart_equivalence=False,
    )
    # This may have stopped after the first segment (wall-time cap is very short).
    # Resume from the last checkpoint to finish the horizon.
    first_checkpoint = tmp_path / "first" / "segments" / "segment_0000.npz"
    if not first_checkpoint.is_file():
        pytest.skip("wall-time cap did not produce segment_0000.npz checkpoint")

    run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "second",
        segment_steps=2,
        explicit_total_steps=6,
        resume_from_checkpoint=first_checkpoint,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )

    output = tmp_path / "combined.json"
    result = CliRunner().invoke(
        cli,
        [
            "combine-whole-run",
            str(tmp_path / "first"),
            str(tmp_path / "second"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == (
        "experimental_whole_run_combined_manifest_not_validation"
    )
    assert payload["restart_count"] == 2
    assert payload["can_support_first_principles_acceptance"] is False
    assert "Combined whole-run engineering-candidate manifest" in result.output


def test_combine_whole_run_rejects_noncontiguous_dirs(tmp_path: Path) -> None:
    """combine-whole-run must fail closed on a step gap between run dirs."""
    from dpf.first_principles.runner import FirstPrinciples3DDeck
    from dpf.first_principles.segmented_whole_shot import run_segmented_whole_shot

    deck = FirstPrinciples3DDeck.from_deck({"n_steps": 6, "seed": 0})

    # Two independent (non-contiguous) runs of the same horizon.
    run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "run_a",
        segment_steps=2,
        explicit_total_steps=4,
        verify_restart_equivalence=False,
    )
    run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "run_b",
        segment_steps=2,
        explicit_total_steps=4,
        verify_restart_equivalence=False,
    )
    # Both start at step 0: not contiguous.
    result = CliRunner().invoke(
        cli,
        [
            "combine-whole-run",
            str(tmp_path / "run_a"),
            str(tmp_path / "run_b"),
        ],
    )

    assert result.exit_code != 0 or "Error" in result.output or (
        json.loads(result.output or "{}").get("status", "").startswith("Error")
    ), (
        "combine-whole-run should fail on non-contiguous runs"
    )


# ---------------------------------------------------------------------------
# hybrid_pic_3d_readiness stays blocked — acceptance never unlocked by runtime
# ---------------------------------------------------------------------------


def test_hybrid_pic_3d_readiness_blocked_in_first_principles_3d() -> None:
    """hybrid_pic_3d_readiness must stay blocked after a successful run."""
    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    readiness = payload["telemetry_packets"]["hybrid_pic_3d_readiness"]
    assert readiness["status"] == "blocked", (
        "hybrid_pic_3d_readiness was unlocked by a runtime run; "
        "acceptance is not valid"
    )
    assert readiness["can_support_first_principles_acceptance"] is False
    assert readiness["missing_capabilities"], (
        "hybrid_pic_3d_readiness reports no missing capabilities; "
        "acceptance gate should not pass"
    )
    # Overall payload acceptance must also stay false.
    assert payload["can_support_first_principles_acceptance"] is False


def test_hybrid_pic_3d_readiness_blocked_with_24rod_preset() -> None:
    """hybrid_pic_3d_readiness must stay blocked with the 24-rod preset."""
    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck-preset",
            "pf1000_scholz_2001_24rod_full_energy",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    readiness = payload["telemetry_packets"]["hybrid_pic_3d_readiness"]
    assert readiness["status"] == "blocked"
    assert readiness["can_support_first_principles_acceptance"] is False
    assert payload["can_support_first_principles_acceptance"] is False
