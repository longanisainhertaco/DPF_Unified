"""WP-N4B cross-process restart ledger merge and whole-run artifact combiner.

This module is a pure engineering orchestration layer.  It does NOT modify
the checkpoint ``.npz`` schema, the fail-closed loader in ``state_checkpoint``,
or any physics code.  It reads ``run_manifest.json`` files produced by
``run_segmented_whole_shot`` across N separate OS-level process invocations
and combines them into one auditable whole-run artifact.

Audit source: ``docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md``
Finding F1 / Next Instruction 4: implement WP-N4B as runtime code with
contiguous-tiling proof and merged-ledger equality to an uninterrupted run.

Spec: ``docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/
sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`` §5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dpf.first_principles.segmented_whole_shot import _CumulativeLedgers

_COMBINED_STATUS = (
    "experimental_whole_run_combined_manifest_not_validation"
)

# Counter fields whose values in a rehydrated manifest must be non-decreasing
# across restarts ordered by resume_started_at_step.  The A-6 sidecar ensures
# each restart's cumulative_ledgers already covers all steps executed before
# that restart, so a later restart's counters must be >= any earlier restart's.
_MONOTONE_COUNTER_FIELDS: tuple[str, ...] = (
    "cumulative_j_dot_e_step_count",
    "cumulative_active_port_step_count",
    "limiter_steps_observed",
    "limiter_total_activations",
)


class LedgerMergeError(ValueError):
    """Raised when restarts are non-contiguous or a manifest is missing.

    Always attributable: the message names the offending step indices.
    Fail-closed -- no combined ledger is emitted when this is raised.
    """


def _validate_manifest_step_bounds(
    manifest: dict[str, Any],
    *,
    restart_index: int,
) -> tuple[int, int]:
    start = int(manifest.get("resume_started_at_step", 0) or 0)
    completed = int(manifest.get("total_steps_completed", 0) or 0)
    if completed < start:
        raise LedgerMergeError(
            f"malformed manifest at restart {restart_index}: "
            f"total_steps_completed={completed} is less than "
            f"resume_started_at_step={start}"
        )
    return start, completed


def merge_cumulative_ledgers(
    manifest_paths: list[Path | str],
) -> dict[str, Any]:
    """Merge the ``cumulative_ledgers`` blocks from N restart manifests.

    Each path in ``manifest_paths`` must point to a ``run_manifest.json``
    produced by ``run_segmented_whole_shot``.  The restarts must tile the
    planned horizon contiguously with no step gap and no overlap; any
    violation raises :class:`LedgerMergeError` before any merge is attempted
    (fail-closed).

    Additive counters (``cumulative_j_dot_e_work_J``, etc.) are summed across
    all restarts in step order.  Final-state scalars (``final_circuit_current_A``,
    etc.) are taken from the highest-``total_steps_completed`` restart, which
    is the terminal state of the combined run.

    Returns a merged-ledger dict in the same shape as the per-restart
    ``cumulative_ledgers`` block, extended with ``covers_executed_horizon``
    and provenance bookkeeping.  The dict is labelled with a fail-closed
    status; it never claims acceptance or validation.

    Raises :class:`LedgerMergeError` on: a missing manifest, an unparseable
    manifest, a step gap between restarts, a step overlap between restarts,
    or zero manifests supplied.
    """
    if not manifest_paths:
        raise LedgerMergeError("manifest_paths is empty; nothing to merge")

    # --- load all manifests, fail closed on any missing/unreadable file ------
    manifests: list[dict[str, Any]] = []
    for raw in manifest_paths:
        path = Path(raw)
        if not path.is_file():
            raise LedgerMergeError(
                f"run_manifest.json not found: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise LedgerMergeError(
                f"failed to parse run_manifest.json at {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LedgerMergeError(
                f"run_manifest.json at {path} is not a JSON object"
            )
        manifests.append(payload)

    # --- sort by resume_started_at_step so order does not depend on caller ---
    def _start_step(m: dict[str, Any]) -> int:
        v = m.get("resume_started_at_step")
        if v is None:
            return 0
        return int(v)

    manifests.sort(key=_start_step)

    for idx, manifest in enumerate(manifests):
        _validate_manifest_step_bounds(manifest, restart_index=idx)

    # --- whole-run invariant: first restart must start at step 0 -------------
    # If the first restart begins after step 0 this is a suffix run, not a
    # whole run.  Fail closed rather than silently merging an incomplete prefix.
    first_start = _start_step(manifests[0])
    if first_start != 0:
        raise LedgerMergeError(
            f"first restart begins at step {first_start}, not step 0; "
            "this is a suffix run, not a whole run -- whole-run merge refused"
        )

    # --- contiguity check: verify restarts tile the horizon with no gap/overlap
    # Run before the ledger invariant check so structural errors (gap, overlap)
    # get their own attributable message rather than tripping on the counter
    # comparison of an otherwise-corrupted manifest pair.
    expected_next_step = _start_step(manifests[0])
    for idx, manifest in enumerate(manifests):
        actual_start = _start_step(manifest)
        if actual_start != expected_next_step:
            if actual_start > expected_next_step:
                raise LedgerMergeError(
                    f"step gap detected before restart {idx}: expected start "
                    f"at step {expected_next_step}, but restart begins at "
                    f"step {actual_start} (gap of "
                    f"{actual_start - expected_next_step} steps)"
                )
            raise LedgerMergeError(
                f"step overlap detected before restart {idx}: expected start "
                f"at step {expected_next_step}, but restart begins at "
                f"step {actual_start} (overlap of "
                f"{expected_next_step - actual_start} steps)"
            )
        steps_completed = int(manifest.get("total_steps_completed", 0))
        expected_next_step = steps_completed

    # --- input-invariant check: every manifest's cumulative_ledgers must be ---
    # rehydrated (cumulative from step 0), not per-restart-only.  Evidence:
    # for manifests sorted by resume_started_at_step, each restart's counter
    # values must be >= those of the immediately preceding restart, because the
    # A-6 sidecar carries the full prefix into every resume.  A counter that
    # decreases relative to an earlier restart proves the ledger was NOT
    # rehydrated from the sidecar (or was corrupted), which invalidates the
    # terminal-ledger approach.  Fail closed immediately.
    for idx in range(1, len(manifests)):
        prev_ledger = manifests[idx - 1].get("cumulative_ledgers", {})
        curr_ledger = manifests[idx].get("cumulative_ledgers", {})
        if not isinstance(prev_ledger, dict):
            prev_ledger = {}
        if not isinstance(curr_ledger, dict):
            curr_ledger = {}
        curr_start = _start_step(manifests[idx])
        prev_start = _start_step(manifests[idx - 1])
        for field in _MONOTONE_COUNTER_FIELDS:
            prev_val = int(prev_ledger.get(field, 0) or 0)
            curr_val = int(curr_ledger.get(field, 0) or 0)
            if curr_val < prev_val:
                raise LedgerMergeError(
                    f"non-cumulative ledger detected at restart {idx} "
                    f"(resume_started_at_step={curr_start}): "
                    f"field '{field}' dropped from {prev_val} "
                    f"(restart {idx - 1}, step {prev_start}) to {curr_val} -- "
                    "manifest cumulative_ledgers were not rehydrated from the "
                    "A-6 sidecar; terminal-ledger merge is invalid"
                )

    # --- identify the terminal manifest (highest total_steps_completed) ------
    terminal = max(
        manifests,
        key=lambda m: int(m.get("total_steps_completed", 0)),
    )
    terminal_ledger_raw = terminal.get("cumulative_ledgers", {})
    if not isinstance(terminal_ledger_raw, dict):
        terminal_ledger_raw = {}

    # --- derive the whole-run additive counters --------------------------------
    # After the A-6 sidecar fix each restart's cumulative_ledgers is rehydrated
    # from the per-checkpoint sidecar so it already covers all steps executed
    # before that restart, not only the post-resume segments.  Therefore the
    # terminal manifest's ledger (highest total_steps_completed) already holds
    # the correct whole-run totals for every additive counter.
    #
    # For the first restart (resume_started_at_step == 0) the ledger covers only
    # its own steps; for later restarts the sidecar carries the prefix.  In both
    # cases the terminal restart's ledger is the accumulation of all restarts,
    # so we take it directly rather than summing per-restart blocks (which would
    # double-count the prefix carried by the sidecar).
    terminal_state = {
        field: terminal_ledger_raw.get(field)
        for field in _CumulativeLedgers._STATE_FIELDS
    }
    whole_run = _CumulativeLedgers.from_state_dict(terminal_state)

    # planned_total_steps: maximum declared horizon across restarts.
    planned_total_steps: int | None = None
    for manifest in manifests:
        plan_block = manifest.get("plan", {})
        if isinstance(plan_block, dict):
            pts = plan_block.get("total_steps")
            if pts is not None:
                pts = int(pts)
                if planned_total_steps is None or pts > planned_total_steps:
                    planned_total_steps = pts

    # total_steps_combined: steps executed in this combined run.  Each restart's
    # contribution is (total_steps_completed - resume_started_at_step) to avoid
    # double-counting the cumulative baseline from the sidecar.
    total_steps_combined = sum(
        int(m.get("total_steps_completed", 0))
        - int(m.get("resume_started_at_step", 0))
        for m in manifests
    )

    if planned_total_steps is None:
        planned_total_steps = total_steps_combined

    covers = total_steps_combined >= planned_total_steps

    merged: dict[str, Any] = {
        "ledger_status": "candidate_merged_whole_run_ledger_not_validation",
        "cumulative_j_dot_e_work_J": whole_run.cumulative_j_dot_e_work_J,
        "cumulative_j_dot_e_step_count": whole_run.cumulative_j_dot_e_step_count,
        "cumulative_active_port_work_J": whole_run.cumulative_active_port_work_J,
        "cumulative_active_port_step_count": whole_run.cumulative_active_port_step_count,
        "limiter_steps_observed": whole_run.limiter_steps_observed,
        "limiter_total_activations": whole_run.limiter_total_activations,
        "final_cumulative_neutrons": whole_run.final_cumulative_neutrons,
        "final_circuit_current_A": whole_run.final_circuit_current_A,
        "final_circuit_charge_C": whole_run.final_circuit_charge_C,
        "final_electron_energy_J": whole_run.final_electron_energy_J,
        "final_ion_temperature_K": whole_run.final_ion_temperature_K,
        "final_ionization_electron_density_m3": whole_run.final_ionization_electron_density_m3,
        "final_particle_count": whole_run.final_particle_count,
        "covers_executed_horizon": covers,
        "executed_steps": total_steps_combined,
        "planned_horizon_steps": planned_total_steps,
        "restart_count": len(manifests),
    }
    return merged


def combine_whole_run_artifacts(
    run_dirs: list[Path | str],
) -> dict[str, Any]:
    """Combine the run directories from N restarts into one whole-run manifest.

    ``run_dirs`` is an ordered list of ``Path`` objects.  Each directory must
    contain a ``run_manifest.json`` produced by ``run_segmented_whole_shot``.
    The restarts must be contiguous: for every consecutive pair ``(k-1, k)``
    the condition
    ``run_dirs[k].resume_started_at_step == run_dirs[k-1].total_steps_completed``
    must hold; the function fails closed with an attributable
    :class:`LedgerMergeError` otherwise.

    The output dict carries:
    * ``horizon_complete`` -- true when combined steps >= planned horizon.
    * ``merged_cumulative_ledgers`` -- via ``merge_cumulative_ledgers``.
    * ``segment_inventory`` -- all per-segment manifests re-indexed
      monotonically across restarts.
    * ``checkpoint_inventory`` -- all ``.npz`` checkpoint paths with their
      global step offsets.
    * Fail-closed labels throughout; ``can_support_first_principles_acceptance``
      is always False.

    Raises :class:`LedgerMergeError` on: non-contiguous restarts, missing
    manifest, or zero dirs supplied.
    """
    if not run_dirs:
        raise LedgerMergeError("run_dirs is empty; nothing to combine")

    resolved: list[Path] = [Path(d) for d in run_dirs]

    # --- load all manifests, fail closed on any missing file -----------------
    manifests: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    for run_path in resolved:
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.is_file():
            raise LedgerMergeError(
                f"run_manifest.json missing in run_dir: {run_path}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise LedgerMergeError(
                f"failed to parse run_manifest.json at {manifest_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LedgerMergeError(
                f"run_manifest.json at {manifest_path} is not a JSON object"
            )
        manifests.append(payload)
        manifest_paths.append(manifest_path)

    # --- contiguity check across run_dirs order (caller-supplied order) ------
    for k, manifest in enumerate(manifests):
        _validate_manifest_step_bounds(manifest, restart_index=k)

    for k in range(1, len(manifests)):
        prev_completed = int(manifests[k - 1].get("total_steps_completed", -1))
        curr_start = int(manifests[k].get("resume_started_at_step", -1))
        if curr_start != prev_completed:
            if curr_start > prev_completed:
                raise LedgerMergeError(
                    f"step gap between run_dirs[{k - 1}] and run_dirs[{k}]: "
                    f"run_dirs[{k - 1}] completed {prev_completed} steps but "
                    f"run_dirs[{k}] started at step {curr_start} "
                    f"(gap of {curr_start - prev_completed} steps)"
                )
            raise LedgerMergeError(
                f"step overlap between run_dirs[{k - 1}] and run_dirs[{k}]: "
                f"run_dirs[{k - 1}] completed {prev_completed} steps but "
                f"run_dirs[{k}] started at step {curr_start} "
                f"(overlap of {prev_completed - curr_start} steps)"
            )

    # --- merged ledger via merge_cumulative_ledgers --------------------------
    merged_ledger = merge_cumulative_ledgers(manifest_paths)

    # --- segment inventory: re-index monotonically across restarts -----------
    global_segment_index = 0
    segment_inventory: list[dict[str, Any]] = []
    for restart_idx, (manifest, run_path) in enumerate(
        zip(manifests, resolved, strict=True)
    ):
        restart_start_step = int(manifest.get("resume_started_at_step", 0))
        for seg in manifest.get("segments", []):
            if not isinstance(seg, dict):
                continue
            entry: dict[str, Any] = {
                "global_segment_index": global_segment_index,
                "restart_index": restart_idx,
                "local_segment_index": seg.get("segment_index"),
                "segment_steps_requested": seg.get("segment_steps_requested"),
                "segment_steps_completed": seg.get("segment_steps_completed"),
                "global_step_start": restart_start_step + (
                    seg.get("total_steps_completed_after_segment", 0)
                    - seg.get("segment_steps_completed", 0)
                ),
                "global_step_end": restart_start_step + int(
                    seg.get("total_steps_completed_after_segment", 0)
                ),
                "state_fingerprint_sha256": seg.get("state_fingerprint_sha256"),
                "checkpoint": seg.get("checkpoint"),
                "run_dir": str(run_path),
            }
            segment_inventory.append(entry)
            global_segment_index += 1

    # --- checkpoint inventory: all .npz files with global step offsets -------
    checkpoint_inventory: list[dict[str, Any]] = []
    for restart_idx, (manifest, run_path) in enumerate(
        zip(manifests, resolved, strict=True)
    ):
        restart_start_step = int(manifest.get("resume_started_at_step", 0))
        segments_dir = run_path / "segments"
        for seg in manifest.get("segments", []):
            if not isinstance(seg, dict):
                continue
            checkpoint_block = seg.get("checkpoint")
            if not isinstance(checkpoint_block, dict):
                continue
            cp_path_str = checkpoint_block.get("checkpoint_path")
            if cp_path_str is None:
                continue
            cp_path = Path(cp_path_str)
            if not cp_path.is_absolute():
                cp_path = segments_dir / cp_path.name
            global_step_end = restart_start_step + int(
                seg.get("total_steps_completed_after_segment", 0)
            )
            checkpoint_inventory.append({
                "restart_index": restart_idx,
                "global_step_at_checkpoint": global_step_end,
                "checkpoint_path": str(cp_path),
                "write_read_hashes_match": checkpoint_block.get(
                    "write_read_hashes_match"
                ),
                "terminal_state_fingerprint_sha256": checkpoint_block.get(
                    "terminal_state_fingerprint_sha256"
                ),
            })

    # Latest checkpoint is the resume point for any further restart.
    latest_checkpoint: str | None = None
    if checkpoint_inventory:
        latest_checkpoint = max(
            checkpoint_inventory,
            key=lambda c: int(c.get("global_step_at_checkpoint", 0)),
        )["checkpoint_path"]

    total_steps_combined = int(merged_ledger["executed_steps"])
    planned_total_steps = int(merged_ledger["planned_horizon_steps"])
    horizon_complete = total_steps_combined >= planned_total_steps

    combined: dict[str, Any] = {
        "status": _COMBINED_STATUS,
        "run_intent": "whole_run_combined_artifact_engineering_candidate",
        "restart_count": len(manifests),
        "run_dirs": [str(p) for p in resolved],
        "total_steps_combined": total_steps_combined,
        "planned_total_steps": planned_total_steps,
        "horizon_complete": bool(horizon_complete),
        "merged_cumulative_ledgers": merged_ledger,
        "ledger_status": merged_ledger["ledger_status"],
        "segment_inventory": segment_inventory,
        "checkpoint_inventory": checkpoint_inventory,
        "latest_checkpoint_for_resume": latest_checkpoint,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "combined_artifact_is_engineering_only": True,
            "validation_promotion_allowed": False,
        },
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "whole_run_combined_engineering_candidate_only",
        },
        "can_support_first_principles_acceptance": False,
    }
    return combined
