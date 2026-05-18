"""WP-N4 segmented whole-shot runner tests.

Covers the planner, the segmented whole-shot executor, the run-directory
emission, cumulative-ledger carry-across, the wall-time cap, resume, and the
staged restart-equivalence evidence.

These tests prove that a segmented run is bit-identical to one uninterrupted
run at staged small horizons.  They do NOT run a 12 us shot -- that is a known
compute-wall blocker, asserted explicitly in
``test_blocker_verdicts_report_12us_compute_wall``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpf.first_principles.runner import FirstPrinciples3DDeck
from dpf.first_principles.segmented_whole_shot import (
    WholeShotWallTimeError,
    build_staged_restart_equivalence_evidence,
    plan_segmented_whole_shot,
    run_segmented_whole_shot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smoke_deck(n_steps: int = 6) -> FirstPrinciples3DDeck:
    """Smallest deterministic fixed-step deck: default 5x5x5, seed 0."""
    return FirstPrinciples3DDeck.from_deck({"n_steps": n_steps, "seed": 0})


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_planner_partitions_horizon_into_segments() -> None:
    """ceil(target/dt) steps split into ceil(steps/segment) segments."""
    plan = plan_segmented_whole_shot(
        target_time_s=7e-13,
        dt_s=1e-13,
        segment_steps=2,
        checkpoint_every_segments=1,
    )
    assert plan.total_steps == 7
    assert plan.segment_count == 4  # ceil(7 / 2)
    assert plan.last_segment_steps == 1  # 7 - 2 * 3
    # A checkpoint at every segment plus the final one.
    assert plan.checkpoint_segment_indices == (0, 1, 2, 3)


def test_planner_explicit_total_steps_bypasses_float_roundtrip() -> None:
    """6 * 1e-13 rounds up in float; explicit_total_steps must stay exact."""
    # The lossy path: 6e-13 / 1e-13 == 6.000000000000001 -> ceil -> 7.
    lossy = plan_segmented_whole_shot(
        target_time_s=6.0 * 1e-13,
        dt_s=1e-13,
        segment_steps=2,
    )
    assert lossy.total_steps == 7  # documents the float round-trip

    exact = plan_segmented_whole_shot(
        target_time_s=6.0 * 1e-13,
        dt_s=1e-13,
        segment_steps=2,
        explicit_total_steps=6,
    )
    assert exact.total_steps == 6
    assert exact.segment_count == 3


def test_planner_checkpoint_cadence_skips_non_cadence_segments() -> None:
    """checkpoint_every_segments=3 checkpoints segment 2 and the final one."""
    plan = plan_segmented_whole_shot(
        target_time_s=1.0e-12,
        dt_s=1e-13,
        segment_steps=1,
        checkpoint_every_segments=3,
    )
    assert plan.total_steps == 10
    # Indices 2, 5, 8 are on cadence; index 9 is the forced final checkpoint.
    assert plan.checkpoint_segment_indices == (2, 5, 8, 9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_time_s": 0.0, "dt_s": 1e-13, "segment_steps": 2},
        {"target_time_s": 1e-12, "dt_s": 0.0, "segment_steps": 2},
        {"target_time_s": 1e-12, "dt_s": 1e-13, "segment_steps": 0},
        {
            "target_time_s": 1e-12,
            "dt_s": 1e-13,
            "segment_steps": 2,
            "checkpoint_every_segments": 0,
        },
        {
            "target_time_s": 1e-12,
            "dt_s": 1e-13,
            "segment_steps": 2,
            "wall_time_cap_s": -1.0,
        },
    ],
)
def test_planner_rejects_invalid_arguments(kwargs: dict) -> None:
    """Non-positive horizon / dt / segment / cadence / cap must raise."""
    with pytest.raises(ValueError):
        plan_segmented_whole_shot(**kwargs)


# ---------------------------------------------------------------------------
# Segmented whole-shot run: equivalence + run directory
# ---------------------------------------------------------------------------


def test_segmented_whole_shot_matches_uninterrupted_run(tmp_path: Path) -> None:
    """The segmented run must be bit-identical to one uninterrupted run.

    A dropped state channel, a double-counted ledger, or divergence across a
    checkpoint boundary would break the fingerprint or observable match.
    """
    deck = _smoke_deck(n_steps=6)
    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "run",
        segment_steps=2,
        explicit_total_steps=6,
        checkpoint_every_segments=1,
        verify_restart_equivalence=True,
    )

    assert manifest["horizon_complete"] is True
    assert manifest["total_steps_completed"] == 6
    assert manifest["plan"]["segment_count"] == 3

    equivalence = manifest["restart_equivalence"]
    assert equivalence["verified"] is True
    assert equivalence["state_fingerprints_match"] is True, (
        "segmented run diverged from the uninterrupted run -- non-equivalent "
        "segmented integration or an unsaved state channel"
    )
    assert equivalence["tracked_observables_match_exactly"] is True

    # The probe must never claim acceptance.
    assert manifest["can_support_first_principles_acceptance"] is False


def test_run_directory_contains_all_required_artifacts(tmp_path: Path) -> None:
    """The run directory must carry deck, command, plan, manifests, checkpoints.

    Audit A-4 requires a run directory with deck, command, commit, dirty flag,
    source hashes, per-segment manifests, checkpoint hashes, and blockers.
    """
    run_dir = tmp_path / "run"
    manifest = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=6),
        run_dir=run_dir,
        segment_steps=2,
        explicit_total_steps=6,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )

    # Top-level run-directory files.
    for name in ("deck.json", "command.json", "plan.json", "run_manifest.json"):
        assert (run_dir / name).is_file(), f"missing run-directory file: {name}"

    # Per-segment manifests + checkpoint payloads (3 segments).
    for index in range(3):
        assert (run_dir / "segments" / f"segment_{index:04d}.npz").is_file()
        assert (
            run_dir / "segments" / f"segment_{index:04d}.manifest.json"
        ).is_file()

    # The command record carries argv, git commit, dirty flag, source hashes.
    command = json.loads((run_dir / "command.json").read_text())
    assert "command_argv" in command
    assert "git_commit" in command
    assert "dirty_worktree" in command
    assert command["source_module_sha256"], "source module hashes are empty"
    assert (
        "segmented_whole_shot.py"
        in " ".join(command["source_module_sha256"].keys())
    )

    # Every per-segment manifest with a checkpoint records its content hashes.
    for segment in manifest["segments"]:
        checkpoint = segment["checkpoint"]
        if checkpoint is not None:
            assert checkpoint["write_content_sha256"] is not None
            assert checkpoint["read_content_sha256"] is not None
            assert checkpoint["write_read_hashes_match"] is True

    # Blocker verdicts are emitted.
    verdict_ids = {v["id"] for v in manifest["blocker_verdicts"]["verdicts"]}
    assert {
        "B-WPN4-12US-COMPUTE-WALL",
        "B-WPN4-WALL-TIME-CAP",
        "B-WPN4-CHECKPOINT-INTEGRITY",
        "B-WPN4-RESTART-EQUIVALENCE",
    } == verdict_ids


def test_cumulative_ledgers_carry_across_segments(tmp_path: Path) -> None:
    """Cumulative ledgers must cover the full executed horizon, not a segment.

    Per-run telemetry resets cumulative counters at every segment start, so a
    runner that forgot to accumulate would report a single segment's count.
    """
    n_steps = 8
    manifest = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=n_steps),
        run_dir=tmp_path / "run",
        segment_steps=2,
        explicit_total_steps=n_steps,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )
    ledgers = manifest["cumulative_ledgers"]

    # The limiter ledger is observed every step; it must span the full horizon.
    assert ledgers["limiter_steps_observed"] == n_steps, (
        "cumulative limiter ledger did not cover the full segmented horizon"
    )
    assert ledgers["covers_executed_horizon"] is True
    assert ledgers["executed_steps"] == n_steps

    # If volume J.E was integrated at all, its step count spans the horizon.
    if ledgers["cumulative_j_dot_e_step_count"] > 0:
        assert ledgers["cumulative_j_dot_e_step_count"] == n_steps, (
            "cumulative J.E step count was not accumulated across segments"
        )


def test_cumulative_ledger_counts_executed_not_planned_steps(
    tmp_path: Path,
) -> None:
    """A wall-time-truncated run must not inflate the ledger to the horizon.

    History capping caps retained samples only; the ledger must report the
    steps actually executed, never the larger planned horizon.
    """
    manifest = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=40),
        run_dir=tmp_path / "run",
        segment_steps=2,
        explicit_total_steps=40,
        checkpoint_every_segments=1,
        wall_time_cap_s=1e-9,  # forces truncation after the first segment
        verify_restart_equivalence=False,
    )
    assert manifest["wall_time_cap_reached"] is True
    assert manifest["horizon_complete"] is False

    ledgers = manifest["cumulative_ledgers"]
    executed = manifest["total_steps_completed"]
    # The ledger spans exactly the executed steps, which is < the planned 40.
    assert executed < 40
    assert ledgers["executed_steps"] == executed
    assert ledgers["limiter_steps_observed"] == executed
    assert ledgers["planned_horizon_steps"] == 40
    assert ledgers["covers_executed_horizon"] is True


# ---------------------------------------------------------------------------
# Wall-time cap
# ---------------------------------------------------------------------------


def test_wall_time_cap_emits_honest_partial_run(tmp_path: Path) -> None:
    """A wall-time cap stops the run and records a partial-run blocker.

    The run must be honestly labelled incomplete, never relabelled finished,
    and restart equivalence must NOT be asserted on a partial run.
    """
    run_dir = tmp_path / "run"
    manifest = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=40),
        run_dir=run_dir,
        segment_steps=2,
        explicit_total_steps=40,
        checkpoint_every_segments=1,
        wall_time_cap_s=1e-9,
        verify_restart_equivalence=True,
    )

    assert manifest["wall_time_cap_reached"] is True
    assert manifest["horizon_complete"] is False
    assert manifest["total_steps_completed"] < 40

    # Equivalence is intentionally not asserted on a partial run.
    equivalence = manifest["restart_equivalence"]
    assert equivalence["verified"] is False
    assert "partial" in equivalence["reason"] or "cap" in equivalence["reason"]

    # The wall-time-cap blocker verdict is triggered.
    verdicts = {v["id"]: v for v in manifest["blocker_verdicts"]["verdicts"]}
    assert verdicts["B-WPN4-WALL-TIME-CAP"]["status"] == "triggered"
    assert manifest["blocker_verdicts"]["summary"] == (
        "engineering_candidate_partial_or_blocked"
    )

    # A partial run directory was still emitted.
    assert (run_dir / "run_manifest.json").is_file()


def test_wall_time_cap_can_raise_when_requested(tmp_path: Path) -> None:
    """raise_on_wall_time_cap converts truncation into a hard error."""
    with pytest.raises(WholeShotWallTimeError, match="wall-time cap"):
        run_segmented_whole_shot(
            deck=_smoke_deck(n_steps=40),
            run_dir=tmp_path / "run",
            segment_steps=2,
            explicit_total_steps=40,
            wall_time_cap_s=1e-9,
            verify_restart_equivalence=False,
            raise_on_wall_time_cap=True,
        )
    # The partial run directory is written before the error is raised.
    assert (tmp_path / "run" / "run_manifest.json").is_file()


# ---------------------------------------------------------------------------
# Blocker verdicts
# ---------------------------------------------------------------------------


def test_blocker_verdicts_report_12us_compute_wall(tmp_path: Path) -> None:
    """A small run must honestly report the 12 us shot as compute-wall blocked.

    The runner must never imply a 12 us run was produced.  At dt=1e-13 a 12 us
    run is ~1.2e8 steps; the 12 us blocker verdict must say so.
    """
    manifest = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=6),
        run_dir=tmp_path / "run",
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )
    verdicts = {v["id"]: v for v in manifest["blocker_verdicts"]["verdicts"]}
    twelve_us = verdicts["B-WPN4-12US-COMPUTE-WALL"]
    assert twelve_us["status"] == "blocked"
    assert "12 us" in twelve_us["detail"]
    # 12 us / 1e-13 == 1.2e8 steps.
    assert "120000000" in twelve_us["detail"]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_from_checkpoint_completes_the_horizon(tmp_path: Path) -> None:
    """A run resumed from a mid-horizon checkpoint must finish equivalently.

    Run a 4-step horizon, then resume from its segment-0 checkpoint (step 2)
    and confirm the resumed run completes all 4 steps and stays bit-identical
    to the uninterrupted run.
    """
    deck = _smoke_deck(n_steps=4)

    first = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "first",
        segment_steps=2,
        explicit_total_steps=4,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )
    assert first["horizon_complete"] is True
    segment0_checkpoint = (
        tmp_path / "first" / "segments" / "segment_0000.npz"
    )
    assert segment0_checkpoint.is_file()

    resumed = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "resumed",
        segment_steps=2,
        explicit_total_steps=4,
        checkpoint_every_segments=1,
        resume_from_checkpoint=segment0_checkpoint,
        verify_restart_equivalence=True,
    )

    # The resume started at step 2 and finished the 4-step horizon.
    assert resumed["resume_started_at_step"] == 2
    assert resumed["total_steps_completed"] == 4
    assert resumed["horizon_complete"] is True
    # Resumed run is bit-identical to the uninterrupted 4-step run.
    assert resumed["restart_equivalence"]["state_fingerprints_match"] is True
    assert (
        resumed["restart_equivalence"]["tracked_observables_match_exactly"]
        is True
    )


# ---------------------------------------------------------------------------
# Checkpoint mismatch fails before state is written
# ---------------------------------------------------------------------------


def test_resume_from_mismatched_deck_fails_closed(tmp_path: Path) -> None:
    """Resuming a checkpoint under a different grid must fail attributably.

    The fail-closed loader must reject a grid/closure mismatch BEFORE any
    state array is written into the resumed session.
    """
    first = run_segmented_whole_shot(
        deck=_smoke_deck(n_steps=4),
        run_dir=tmp_path / "first",
        segment_steps=2,
        explicit_total_steps=4,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )
    assert first["horizon_complete"] is True
    checkpoint = tmp_path / "first" / "segments" / "segment_0000.npz"

    mismatched_deck = FirstPrinciples3DDeck.from_deck(
        {"n_steps": 4, "seed": 0, "grid_shape": (6, 6, 6)}
    )
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        run_segmented_whole_shot(
            deck=mismatched_deck,
            run_dir=tmp_path / "resumed",
            segment_steps=2,
            explicit_total_steps=4,
            resume_from_checkpoint=checkpoint,
            verify_restart_equivalence=False,
        )
    msg = str(exc_info.value).lower()
    assert "grid" in msg or "shape" in msg or "checkpoint" in msg


# ---------------------------------------------------------------------------
# Staged restart-equivalence evidence
# ---------------------------------------------------------------------------


def test_staged_restart_equivalence_evidence_proves_all_stages(
    tmp_path: Path,
) -> None:
    """Several staged small horizons must each be bit-identical when segmented.

    This is the WP-N4 "first restart-equivalence evidence" deliverable: it is
    explicitly NOT a 12 us run.
    """
    evidence = build_staged_restart_equivalence_evidence(
        deck={"seed": 0},
        staged_segment_plans=((4, 2), (6, 2), (9, 3)),
        run_root=tmp_path / "staged",
    )

    assert evidence["stage_count"] == 3
    assert evidence["all_stages_equivalence_proven"] is True

    for stage in evidence["stages"]:
        # The horizon is exactly the requested step count (no float inflation).
        expected_segments = -(-stage["total_steps"] // stage["segment_steps"])
        assert stage["segment_count"] == expected_segments
        assert stage["horizon_complete"] is True
        assert stage["state_fingerprints_match"] is True
        assert stage["tracked_observables_match_exactly"] is True
        assert stage["equivalence_proven"] is True

    # The 12 us run is explicitly NOT attempted and is labelled blocked.
    twelve_us = evidence["twelve_microsecond_run"]
    assert twelve_us["attempted"] is False
    assert twelve_us["status"] == "compute_wall_blocked"

    # The evidence must never claim acceptance.
    assert evidence["can_support_first_principles_acceptance"] is False


def test_staged_evidence_rejects_single_segment_horizon(tmp_path: Path) -> None:
    """A staged plan whose segment >= total does not exercise equivalence."""
    with pytest.raises(ValueError, match="single-segment"):
        build_staged_restart_equivalence_evidence(
            deck={"seed": 0},
            staged_segment_plans=((4, 4),),
            run_root=tmp_path / "staged",
        )


def test_staged_evidence_run_directories_are_emitted(tmp_path: Path) -> None:
    """Each staged run must emit its own complete run directory."""
    evidence = build_staged_restart_equivalence_evidence(
        deck={"seed": 0},
        staged_segment_plans=((4, 2), (6, 3)),
        run_root=tmp_path / "staged",
    )
    for stage in evidence["stages"]:
        run_dir = Path(stage["run_dir"])
        assert (run_dir / "run_manifest.json").is_file()
        assert (run_dir / "deck.json").is_file()
        assert (run_dir / "command.json").is_file()
        assert (run_dir / "plan.json").is_file()
