"""WP-N4B cross-restart ledger merge and whole-run artifact combiner tests.

Positive test (P-1): a 4-step horizon executed as two separate
``run_segmented_whole_shot`` invocations (2 steps then resume 2 more) is
merged; the merged ledger equals an uninterrupted 4-step run's
``cumulative_ledgers`` for every counter field, and
``covers_executed_horizon`` is True.

Negative tests:
  N-1: a non-contiguous restart pair with a step GAP fails closed.
  N-2: a non-contiguous restart pair with a step OVERLAP fails closed.
  N-3: a missing ``run_manifest.json`` fails closed.

Fail-closed labels are preserved throughout; nothing promotes to accepted or
validated status.

Audit source: ``docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md``
Finding F1 / Next Instruction 4; spec §7.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpf.first_principles.runner import FirstPrinciples3DDeck
from dpf.first_principles.segmented_whole_shot import run_segmented_whole_shot
from dpf.first_principles.segmented_whole_shot_combine import (
    LedgerMergeError,
    combine_whole_run_artifacts,
    merge_cumulative_ledgers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADDITIVE_COUNTER_FIELDS = (
    "cumulative_j_dot_e_work_J",
    "cumulative_j_dot_e_step_count",
    "cumulative_active_port_work_J",
    "cumulative_active_port_step_count",
    "limiter_steps_observed",
    "limiter_total_activations",
)
_FINAL_SCALAR_FIELDS = (
    "final_cumulative_neutrons",
    "final_circuit_current_A",
    "final_circuit_charge_C",
    "final_electron_energy_J",
    "final_ion_temperature_K",
    "final_ionization_electron_density_m3",
    "final_particle_count",
)


def _smoke_deck(n_steps: int = 4) -> FirstPrinciples3DDeck:
    return FirstPrinciples3DDeck.from_deck({"n_steps": n_steps, "seed": 0})


def _run_two_restart_sequence(
    tmp_path: Path,
    *,
    total_steps: int = 4,
    steps_per_restart: int = 2,
) -> tuple[dict, dict, Path, Path]:
    """Run a horizon as two separate invocations.

    First run: steps 0..steps_per_restart.
    Second run: resume from checkpoint, steps steps_per_restart..total_steps.

    Returns (manifest_1, manifest_2, run_dir_1, run_dir_2).
    """
    deck = _smoke_deck(n_steps=total_steps)

    # First invocation: steps 0 to steps_per_restart.
    run_dir_1 = tmp_path / "restart_0"
    manifest_1 = run_segmented_whole_shot(
        deck=deck,
        run_dir=run_dir_1,
        segment_steps=steps_per_restart,
        explicit_total_steps=total_steps,
        checkpoint_every_segments=1,
        wall_time_cap_s=1e-9,  # stop after first segment group
        verify_restart_equivalence=False,
    )
    # wall_time_cap stops after the first segment (steps_per_restart steps).
    assert manifest_1["wall_time_cap_reached"] is True or (
        manifest_1["total_steps_completed"] == steps_per_restart
    )
    completed_1 = manifest_1["total_steps_completed"]
    assert completed_1 > 0, "first invocation must complete at least one step"

    # Locate the latest checkpoint from the first run.
    checkpoint_path: Path | None = None
    for seg in manifest_1.get("segments", []):
        chk = seg.get("checkpoint")
        if isinstance(chk, dict) and chk.get("checkpoint_path"):
            candidate = Path(chk["checkpoint_path"])
            if not candidate.is_absolute():
                candidate = run_dir_1 / "segments" / candidate.name
            if candidate.is_file():
                checkpoint_path = candidate

    assert checkpoint_path is not None, (
        "first invocation did not write a checkpoint; cannot build restart pair"
    )

    # Second invocation: resume from where the first stopped.
    run_dir_2 = tmp_path / "restart_1"
    manifest_2 = run_segmented_whole_shot(
        deck=deck,
        run_dir=run_dir_2,
        segment_steps=steps_per_restart,
        explicit_total_steps=total_steps,
        checkpoint_every_segments=1,
        resume_from_checkpoint=checkpoint_path,
        verify_restart_equivalence=False,
    )
    assert manifest_2["horizon_complete"] is True, (
        "second invocation did not complete the full horizon"
    )
    return manifest_1, manifest_2, run_dir_1, run_dir_2


# ---------------------------------------------------------------------------
# P-1: positive -- merged ledger equals uninterrupted 4-step run
# ---------------------------------------------------------------------------


def test_merged_ledger_equals_uninterrupted_run(tmp_path: Path) -> None:
    """P-1: two-restart merge produces ledgers equal to a fresh 4-step run.

    This is the ledger-equivalence proof: the merge is valid only when the
    restarts tile the horizon contiguously (spec §4), and the merged additive
    counters must match those of an uninterrupted run of the same horizon.
    """
    total_steps = 4
    deck = _smoke_deck(n_steps=total_steps)

    # Reference: one uninterrupted run.
    ref_manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "reference",
        segment_steps=total_steps,  # single segment -- no restarts
        explicit_total_steps=total_steps,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )
    assert ref_manifest["horizon_complete"] is True
    ref_ledgers = ref_manifest["cumulative_ledgers"]

    # Two-restart sequence.
    manifest_1, manifest_2, run_dir_1, run_dir_2 = _run_two_restart_sequence(
        tmp_path / "two_restart",
        total_steps=total_steps,
        steps_per_restart=2,
    )

    manifest_paths = [
        run_dir_1 / "run_manifest.json",
        run_dir_2 / "run_manifest.json",
    ]
    merged = merge_cumulative_ledgers(manifest_paths)

    # covers_executed_horizon must be True.
    assert merged["covers_executed_horizon"] is True, (
        "merged ledger does not cover the executed horizon"
    )
    assert merged["executed_steps"] == total_steps

    # Additive counters must equal the reference uninterrupted run.
    for field in _ADDITIVE_COUNTER_FIELDS:
        assert merged[field] == ref_ledgers[field], (
            f"merged ledger field '{field}' "
            f"({merged[field]}) != reference ({ref_ledgers[field]})"
        )

    # Fail-closed label must be present.
    assert "not_validation" in merged["ledger_status"]
    assert merged.get("restart_count") == 2


def test_combine_whole_run_artifacts_positive(tmp_path: Path) -> None:
    """P-1 (combiner): combine_whole_run_artifacts wraps the merged ledger.

    horizon_complete must be True; ledger_status must carry the fail-closed
    label; acceptance must be False.
    """
    total_steps = 4
    manifest_1, manifest_2, run_dir_1, run_dir_2 = _run_two_restart_sequence(
        tmp_path / "two_restart",
        total_steps=total_steps,
        steps_per_restart=2,
    )

    combined = combine_whole_run_artifacts([run_dir_1, run_dir_2])

    assert combined["horizon_complete"] is True
    assert combined["total_steps_combined"] == total_steps
    assert combined["restart_count"] == 2
    assert "not_validation" in combined["ledger_status"]
    assert combined["can_support_first_principles_acceptance"] is False
    assert (
        combined["acceptance_state"]["can_support_first_principles_acceptance"]
        is False
    )

    # Segment inventory must be non-empty and monotonically re-indexed.
    inventory = combined["segment_inventory"]
    assert len(inventory) >= 1
    for expected_idx, entry in enumerate(inventory):
        assert entry["global_segment_index"] == expected_idx

    # Checkpoint inventory carries at least one entry.
    assert len(combined["checkpoint_inventory"]) >= 1


# ---------------------------------------------------------------------------
# N-1: step GAP between restarts fails closed
# ---------------------------------------------------------------------------


def test_merge_fails_closed_on_step_gap(tmp_path: Path) -> None:
    """N-1: a step gap between restart manifests raises LedgerMergeError."""
    deck = _smoke_deck(n_steps=6)

    # First run: 2 steps (0..2).
    run_dir_1 = tmp_path / "r0"
    run_segmented_whole_shot(
        deck=deck,
        run_dir=run_dir_1,
        segment_steps=2,
        explicit_total_steps=6,
        wall_time_cap_s=1e-9,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )

    # Synthetic second manifest with resume_started_at_step=4 (gap at step 2-4).
    run_dir_2 = tmp_path / "r1_gap"
    run_dir_2.mkdir(parents=True, exist_ok=True)
    gapped_manifest = {
        "resume_started_at_step": 4,  # gap: step 2..4 missing
        "total_steps_completed": 2,
        "planned_total_steps": 6,
        "cumulative_ledgers": {
            "cumulative_j_dot_e_work_J": 0.0,
            "cumulative_j_dot_e_step_count": 0,
            "cumulative_active_port_work_J": 0.0,
            "cumulative_active_port_step_count": 0,
            "limiter_steps_observed": 2,
            "limiter_total_activations": 0,
        },
        "plan": {"total_steps": 6},
        "segments": [],
    }
    (run_dir_2 / "run_manifest.json").write_text(
        json.dumps(gapped_manifest), encoding="utf-8"
    )

    with pytest.raises(LedgerMergeError, match="gap"):
        merge_cumulative_ledgers(
            [run_dir_1 / "run_manifest.json", run_dir_2 / "run_manifest.json"]
        )

    with pytest.raises(LedgerMergeError, match="gap"):
        combine_whole_run_artifacts([run_dir_1, run_dir_2])


# ---------------------------------------------------------------------------
# N-2: step OVERLAP between restarts fails closed
# ---------------------------------------------------------------------------


def test_merge_fails_closed_on_step_overlap(tmp_path: Path) -> None:
    """N-2: an overlapping restart pair raises LedgerMergeError."""
    deck = _smoke_deck(n_steps=6)

    # First run: 2 steps (0..2).
    run_dir_1 = tmp_path / "r0"
    run_segmented_whole_shot(
        deck=deck,
        run_dir=run_dir_1,
        segment_steps=2,
        explicit_total_steps=6,
        wall_time_cap_s=1e-9,
        checkpoint_every_segments=1,
        verify_restart_equivalence=False,
    )

    # Synthetic second manifest with resume_started_at_step=1 (overlaps step 1-2).
    run_dir_2 = tmp_path / "r1_overlap"
    run_dir_2.mkdir(parents=True, exist_ok=True)
    overlapping_manifest = {
        "resume_started_at_step": 1,  # overlap: step 1 covered twice
        "total_steps_completed": 2,
        "planned_total_steps": 6,
        "cumulative_ledgers": {
            "cumulative_j_dot_e_work_J": 0.0,
            "cumulative_j_dot_e_step_count": 0,
            "cumulative_active_port_work_J": 0.0,
            "cumulative_active_port_step_count": 0,
            "limiter_steps_observed": 2,
            "limiter_total_activations": 0,
        },
        "plan": {"total_steps": 6},
        "segments": [],
    }
    (run_dir_2 / "run_manifest.json").write_text(
        json.dumps(overlapping_manifest), encoding="utf-8"
    )

    with pytest.raises(LedgerMergeError, match="overlap"):
        merge_cumulative_ledgers(
            [run_dir_1 / "run_manifest.json", run_dir_2 / "run_manifest.json"]
        )

    with pytest.raises(LedgerMergeError, match="overlap"):
        combine_whole_run_artifacts([run_dir_1, run_dir_2])


# ---------------------------------------------------------------------------
# N-3: missing run_manifest.json fails closed
# ---------------------------------------------------------------------------


def test_merge_fails_closed_on_missing_manifest(tmp_path: Path) -> None:
    """N-3: a missing run_manifest.json raises LedgerMergeError."""
    missing_path = tmp_path / "does_not_exist" / "run_manifest.json"

    with pytest.raises(LedgerMergeError, match="not found"):
        merge_cumulative_ledgers([missing_path])

    with pytest.raises(LedgerMergeError, match="missing"):
        combine_whole_run_artifacts([tmp_path / "does_not_exist"])


def test_merge_fails_closed_on_empty_input() -> None:
    """Calling with an empty list must raise immediately."""
    with pytest.raises(LedgerMergeError, match="empty"):
        merge_cumulative_ledgers([])

    with pytest.raises(LedgerMergeError, match="empty"):
        combine_whole_run_artifacts([])
