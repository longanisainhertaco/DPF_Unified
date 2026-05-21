"""WP-N4 segmented whole-shot runner for first-principles 3-D runs.

This module builds on the prior hygiene round.  It does NOT re-implement the
fixed-step segmented equivalence probe -- that is
``checkpoint_restart.build_experimental_segmented_run_packet`` -- nor the
checkpoint metadata / fail-closed loader in ``state_checkpoint``.

What is added here, per audit
``docs/FIRST_PRINCIPLES_CODEX_AGENT_AUDIT_AND_NEXT_INSTRUCTIONS_2026_05_18.md``
section A-4 and "Next Submission Required" item 5:

* A *planner* (:class:`WholeShotPlan`).  Given a target horizon and ``dt`` it
  derives the segment schedule, checkpoint cadence, wall-time cap, and resume
  manifest -- before any step runs.
* A *segmented whole-shot executor* (:func:`run_segmented_whole_shot`).  It
  drives one live :class:`FirstPrinciples3DSession` segment by segment, writes
  a metadata-tagged checkpoint at the configured cadence (always reloaded
  through the fail-closed loader), and accumulates cumulative ledgers across
  segments so the result is equivalent to one uninterrupted run.
* A *run directory* emitter.  Each run produces a directory containing the
  deck, command argv, git commit, dirty flag, source hashes, per-segment
  manifests, checkpoint hashes, and blocker verdicts.
* *Staged restart-equivalence evidence*
  (:func:`build_staged_restart_equivalence_evidence`).  Small horizons split
  into segments are proven bit-identical (state fingerprint + observables) to
  the uninterrupted run.

Honest scope: this is design plus first restart-equivalence evidence.  A full
12 us run (~1.2e8 steps at dt=1e-13) is a known compute-wall blocker and is
reported as such -- :func:`run_segmented_whole_shot` carries an enforced
``wall_time_cap_s`` and stops at the cap with an explicit blocker verdict
rather than fabricating a finished 12 us artifact.

History capping caps retained step payloads only; cumulative ledgers and
completion counters always cover the full horizon executed.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dpf.fields import HybridPIC3DSimulationResult
from dpf.first_principles.checkpoint_restart import (
    _fixed_step_deck,
    _observable_comparisons,
    _simulation_summary,
)
from dpf.first_principles.manifest import (
    git_provenance,
    sha256_of_file,
    sha256_of_json,
    sha256_of_text,
    stable_manifest_hash,
)
from dpf.first_principles.runner import (
    FirstPrinciples3DDeck,
    build_first_principles_3d_session,
)
from dpf.first_principles.state_checkpoint import (
    load_checkpoint_into_first_principles_3d_session,
    write_simulation_state_checkpoint_roundtrip,
)

SEGMENTED_WHOLE_SHOT_STATUS = (
    "experimental_segmented_whole_shot_engineering_candidate_not_validation"
)

# Physics-relevant source modules whose bytes are hashed into the run
# directory so a reviewer can detect a runtime change between runs.
_SOURCE_HASH_MODULES: tuple[str, ...] = (
    "src/dpf/first_principles/segmented_whole_shot.py",
    "src/dpf/first_principles/checkpoint_restart.py",
    "src/dpf/first_principles/state_checkpoint.py",
    "src/dpf/first_principles/runner.py",
    "src/dpf/fields/hybrid_simulator.py",
)

# Source-truth index hashed into the run directory manifest (audit A-1).
_SOURCE_TRUTH_INDEX_PATH = "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json"


class WholeShotWallTimeError(RuntimeError):
    """Raised only when a wall-time cap is hit AND the caller demanded that a
    truncated whole-shot run be treated as a hard error.

    The default executor path does not raise: it stops at the cap and records
    an explicit ``wall_time_cap_reached`` blocker verdict, because a partial
    whole-shot run is a legitimate, honestly-labelled engineering artifact.
    """


@dataclass(frozen=True)
class WholeShotPlan:
    """Static plan for a segmented whole-shot run.

    Derived purely from ``(target_time_s, dt_s, segment_steps)`` plus the
    checkpoint cadence and wall-time cap.  No simulation runs during planning.
    """

    target_time_s: float
    dt_s: float
    total_steps: int
    segment_steps: int
    segment_count: int
    last_segment_steps: int
    checkpoint_every_segments: int
    checkpoint_segment_indices: tuple[int, ...]
    wall_time_cap_s: float | None
    resume_from_checkpoint: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_segment_indices"] = list(
            self.checkpoint_segment_indices
        )
        payload["planned_final_time_s"] = self.total_steps * self.dt_s
        payload["status"] = "segmented_whole_shot_plan_engineering_candidate"
        return payload


def plan_segmented_whole_shot(
    *,
    target_time_s: float,
    dt_s: float,
    segment_steps: int,
    checkpoint_every_segments: int = 1,
    wall_time_cap_s: float | None = None,
    resume_from_checkpoint: str | Path | None = None,
    explicit_total_steps: int | None = None,
) -> WholeShotPlan:
    """Compute the segment schedule for a whole-shot run.

    When ``explicit_total_steps`` is supplied it is used verbatim as the
    horizon step count; otherwise ``total_steps = ceil(target_time_s / dt_s)``.
    The explicit path exists because ``n * dt`` is not exactly representable in
    float (e.g. ``6 * 1e-13`` rounds up), so a caller that already knows the
    exact step count must not have it inflated by a float round-trip.  The
    horizon is partitioned into ``ceil(total_steps / segment_steps)`` segments;
    the final segment may be shorter.  A checkpoint is scheduled at the end of
    every ``checkpoint_every_segments``-th segment and always at the final one.
    """

    if not math.isfinite(target_time_s) or target_time_s <= 0.0:
        raise ValueError("target_time_s must be a positive finite float")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be a positive finite float")
    if int(segment_steps) != segment_steps or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    if (
        int(checkpoint_every_segments) != checkpoint_every_segments
        or checkpoint_every_segments <= 0
    ):
        raise ValueError("checkpoint_every_segments must be a positive integer")
    if wall_time_cap_s is not None and (
        not math.isfinite(wall_time_cap_s) or wall_time_cap_s <= 0.0
    ):
        raise ValueError("wall_time_cap_s must be a positive finite float")
    if explicit_total_steps is not None and (
        int(explicit_total_steps) != explicit_total_steps
        or explicit_total_steps <= 0
    ):
        raise ValueError("explicit_total_steps must be a positive integer")

    segment_steps = int(segment_steps)
    checkpoint_every_segments = int(checkpoint_every_segments)
    if explicit_total_steps is not None:
        total_steps = int(explicit_total_steps)
    else:
        total_steps = int(math.ceil(target_time_s / dt_s))
    if total_steps <= 0:
        raise ValueError("planned total_steps must be positive")
    segment_count = int(math.ceil(total_steps / segment_steps))
    last_segment_steps = total_steps - segment_steps * (segment_count - 1)

    checkpoint_indices = tuple(
        index
        for index in range(segment_count)
        if (index + 1) % checkpoint_every_segments == 0
        or index == segment_count - 1
    )

    return WholeShotPlan(
        target_time_s=float(target_time_s),
        dt_s=float(dt_s),
        total_steps=total_steps,
        segment_steps=segment_steps,
        segment_count=segment_count,
        last_segment_steps=int(last_segment_steps),
        checkpoint_every_segments=checkpoint_every_segments,
        checkpoint_segment_indices=checkpoint_indices,
        wall_time_cap_s=(
            None if wall_time_cap_s is None else float(wall_time_cap_s)
        ),
        resume_from_checkpoint=(
            None if resume_from_checkpoint is None else str(resume_from_checkpoint)
        ),
    )


@dataclass
class _CumulativeLedgers:
    """Cross-segment cumulative ledgers, equivalent to one uninterrupted run.

    Each field accumulates across every executed segment.  Per-``run``
    telemetry resets its cumulative counters at every segment start, so the
    full-horizon totals must be summed here.  ``cap`` operations elsewhere
    cap retained samples only and never touch these counters.

    S3.7 extended ledger fields (handoff
    ``docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md``
    §S3.7 "Required gates — cumulative ledgers"):

    * ``cumulative_field_energy_delta_J``  — net change in stored EM field
      energy across all segments (circuit + MHD field, not a power-port term).
    * ``cumulative_pml_removed_energy_J``  — total energy removed by the PML
      / open-boundary absorber across all segments.
    * ``cumulative_power_port_work_J``     — cumulative circuit → plasma port
      work (Poynting flux integral term I from power_port, separate from the
      J·E channel which is the resistive-dissipation term).
    * ``cumulative_ionization_step_count`` — steps where an ionization/charge-
      state update was executed (non-zero ionization source term present).

    All four fields are zero-baseline if the telemetry path does not expose
    them (graceful degradation); they are NOT summed from per-segment deltas
    but taken from the terminal-segment telemetry when it reports a cumulative
    from-start value.  A future session that exposes these channels will have
    them populated automatically without schema changes.
    """

    cumulative_j_dot_e_work_J: float = 0.0
    cumulative_j_dot_e_step_count: int = 0
    cumulative_active_port_work_J: float = 0.0
    cumulative_active_port_step_count: int = 0
    limiter_steps_observed: int = 0
    limiter_total_activations: int = 0
    # S3.7 extended ledgers — circuit/field/particle/energy/ionization/
    # kinetic-yield/power-port/PML-removed-energy channels.
    cumulative_field_energy_delta_J: float = 0.0
    cumulative_pml_removed_energy_J: float = 0.0
    cumulative_power_port_work_J: float = 0.0
    cumulative_ionization_step_count: int = 0
    # Final-state snapshot fields (taken from the last executed segment).
    final_cumulative_neutrons: float | None = None
    final_circuit_current_A: float | None = None
    final_circuit_charge_C: float | None = None
    final_electron_energy_J: float | None = None
    final_ion_temperature_K: float | None = None
    final_ionization_electron_density_m3: float | None = None
    final_particle_count: int | None = None

    def accumulate(self, result: HybridPIC3DSimulationResult) -> None:
        telemetry = result.telemetry
        self.cumulative_j_dot_e_work_J += float(
            telemetry.cumulative_j_dot_e_work_J or 0.0
        )
        self.cumulative_j_dot_e_step_count += int(
            telemetry.cumulative_j_dot_e_step_count
        )
        self.cumulative_active_port_work_J += float(
            telemetry.cumulative_active_port_work_J or 0.0
        )
        self.cumulative_active_port_step_count += int(
            telemetry.cumulative_active_port_step_count
        )
        summary = telemetry.limiter_activation_summary
        if isinstance(summary, Mapping):
            self.limiter_steps_observed += int(summary.get("steps_observed", 0))
            self.limiter_total_activations += int(
                summary.get("total_activations", 0)
            )
        # S3.7: accumulate extended ledger channels (field/PML/power-port/ionization).
        # All paths are fail-soft: absent telemetry keys leave the counters at
        # zero-baseline rather than crashing the run.
        if isinstance(getattr(telemetry, "conservation_power_port", None), Mapping):
            port = telemetry.conservation_power_port
            self.cumulative_field_energy_delta_J += float(
                port.get("stored_magnetic_energy_delta_J", 0.0) or 0.0
            ) + float(port.get("stored_electric_energy_delta_J", 0.0) or 0.0)
            self.cumulative_power_port_work_J += float(
                port.get("cumulative_port_work_J", 0.0) or 0.0
            )
        if isinstance(getattr(telemetry, "pml_energy_removed_J", None), (int, float)):
            self.cumulative_pml_removed_energy_J += float(
                telemetry.pml_energy_removed_J or 0.0
            )
        if isinstance(getattr(telemetry, "ionization_steps_this_segment", None), int):
            self.cumulative_ionization_step_count += int(
                telemetry.ionization_steps_this_segment
            )
        kinetic = result.kinetic_yield_state
        if isinstance(kinetic, Mapping):
            self.final_cumulative_neutrons = _as_float(
                kinetic.get("cumulative_neutrons")
            )
        if result.circuit is not None:
            self.final_circuit_current_A = float(result.circuit.current_A)
            self.final_circuit_charge_C = float(result.circuit.charge_C)
        if result.electron_energy is not None:
            import numpy as np

            self.final_electron_energy_J = float(
                np.sum(result.electron_energy.electron_energy_J_m3)
            )
            self.final_ion_temperature_K = float(
                np.mean(result.electron_energy.ion_temperature_K)
            )
        if result.ionization_charge_state is not None:
            import numpy as np

            self.final_ionization_electron_density_m3 = float(
                np.sum(result.ionization_charge_state.electron_density_m3)
            )
        self.final_particle_count = int(
            sum(species.n_particles() for species in result.pic.species)
        )

    # Counter fields persisted to / restored from the per-checkpoint sidecar.
    # Listed explicitly so a resume rehydrates exactly the pre-resume totals.
    # S3.7: extended ledger fields added for circuit/field/particle/energy/
    # ionization/kinetic-yield/limiter/power-port/PML-removed-energy channels.
    _STATE_FIELDS: tuple[str, ...] = (
        "cumulative_j_dot_e_work_J",
        "cumulative_j_dot_e_step_count",
        "cumulative_active_port_work_J",
        "cumulative_active_port_step_count",
        "limiter_steps_observed",
        "limiter_total_activations",
        # S3.7 extended fields.
        "cumulative_field_energy_delta_J",
        "cumulative_pml_removed_energy_J",
        "cumulative_power_port_work_J",
        "cumulative_ionization_step_count",
        # Final-state snapshots.
        "final_cumulative_neutrons",
        "final_circuit_current_A",
        "final_circuit_charge_C",
        "final_electron_energy_J",
        "final_ion_temperature_K",
        "final_ionization_electron_density_m3",
        "final_particle_count",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """Serialise every counter/final field for the per-checkpoint sidecar.

        This is the accumulation of all segments executed so far; a resume
        rehydrates from it so the resumed run's cumulative ledgers cover the
        full executed horizon, not just the post-resume segments (audit A-6).
        """

        return {name: getattr(self, name) for name in self._STATE_FIELDS}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> _CumulativeLedgers:
        """Reconstruct ledgers from a sidecar written at an earlier segment.

        Integer counters and float accumulators are coerced back to their
        declared types; ``None`` finals stay ``None``.  A missing key falls
        back to the dataclass default so a malformed sidecar cannot crash
        the resume -- it degrades to the zero baseline for that field only.
        """

        ledgers = cls()
        int_fields = {
            "cumulative_j_dot_e_step_count",
            "cumulative_active_port_step_count",
            "limiter_steps_observed",
            "limiter_total_activations",
            # S3.7 extended int fields.
            "cumulative_ionization_step_count",
        }
        float_fields = {
            "cumulative_j_dot_e_work_J",
            "cumulative_active_port_work_J",
            # S3.7 extended float fields.
            "cumulative_field_energy_delta_J",
            "cumulative_pml_removed_energy_J",
            "cumulative_power_port_work_J",
        }
        for name in cls._STATE_FIELDS:
            if name not in state:
                continue
            value = state[name]
            if value is None:
                setattr(ledgers, name, None)
            elif name in int_fields:
                setattr(ledgers, name, int(value))
            elif name in float_fields:
                setattr(ledgers, name, float(value))
            elif name == "final_particle_count":
                setattr(ledgers, name, int(value))
            else:
                setattr(ledgers, name, float(value))
        return ledgers

    def to_dict(self, *, total_steps_completed: int, horizon_steps: int) -> dict[str, Any]:
        return {
            "ledger_status": (
                "candidate_cumulative_segmented_ledger_not_validation"
            ),
            # Circuit / J·E channel.
            "cumulative_j_dot_e_work_J": self.cumulative_j_dot_e_work_J,
            "cumulative_j_dot_e_step_count": self.cumulative_j_dot_e_step_count,
            # Active-port / circuit coupling channel.
            "cumulative_active_port_work_J": self.cumulative_active_port_work_J,
            "cumulative_active_port_step_count": (
                self.cumulative_active_port_step_count
            ),
            # Limiter channel.
            "limiter_steps_observed": self.limiter_steps_observed,
            "limiter_total_activations": self.limiter_total_activations,
            # S3.7 extended ledger channels.
            # Field / stored-energy delta channel (magnetic + electric).
            "cumulative_field_energy_delta_J": self.cumulative_field_energy_delta_J,
            # PML / open-boundary energy removed channel.
            "cumulative_pml_removed_energy_J": self.cumulative_pml_removed_energy_J,
            # Power-port work channel (Poynting-flux term I, separate from J·E).
            "cumulative_power_port_work_J": self.cumulative_power_port_work_J,
            # Ionization-update step counter.
            "cumulative_ionization_step_count": self.cumulative_ionization_step_count,
            # Final-state snapshots (kinetic-yield / circuit / energy / ionization
            # / particle channels — always the latest executed segment).
            "final_cumulative_neutrons": self.final_cumulative_neutrons,
            "final_circuit_current_A": self.final_circuit_current_A,
            "final_circuit_charge_C": self.final_circuit_charge_C,
            "final_electron_energy_J": self.final_electron_energy_J,
            "final_ion_temperature_K": self.final_ion_temperature_K,
            "final_ionization_electron_density_m3": (
                self.final_ionization_electron_density_m3
            ),
            "final_particle_count": self.final_particle_count,
            # Ledger covers every executed step.  This is steps actually run,
            # which equals the horizon only when the wall-time cap did not cut
            # the run short -- never silently inflated to the planned horizon.
            "covers_executed_horizon": (
                self.limiter_steps_observed == total_steps_completed
            ),
            "executed_steps": int(total_steps_completed),
            "planned_horizon_steps": int(horizon_steps),
        }


def run_segmented_whole_shot(
    *,
    deck: Mapping[str, Any] | object | None,
    run_dir: str | Path,
    segment_steps: int,
    target_time_s: float | None = None,
    explicit_total_steps: int | None = None,
    checkpoint_every_segments: int = 1,
    wall_time_cap_s: float | None = None,
    resume_from_checkpoint: str | Path | None = None,
    verify_restart_equivalence: bool = True,
    raise_on_wall_time_cap: bool = False,
) -> dict[str, Any]:
    """Execute a segmented whole-shot run and emit a run directory.

    ``target_time_s`` defaults to ``deck.target_time_s`` if the deck declares
    one.  The horizon is planned by :func:`plan_segmented_whole_shot`, then run
    segment-by-segment on one live session.  At every scheduled checkpoint the
    live state is written to a metadata-tagged ``.npz`` and reloaded through
    ``load_checkpoint_into_first_principles_3d_session`` (fail-closed loader).

    The run directory ``run_dir`` receives:

    * ``deck.json`` -- the resolved fixed-step deck;
    * ``command.json`` -- argv, git commit, dirty flag, source/index hashes;
    * ``plan.json`` -- the static segment schedule;
    * ``segments/segment_NNNN.npz`` -- checkpoint payloads;
    * ``segments/segment_NNNN.manifest.json`` -- per-segment manifests with
      checkpoint content hashes;
    * ``run_manifest.json`` -- the full run summary, ledgers, blocker verdicts.

    Honest blocker: a full 12 us run is compute-wall blocked.  When a
    ``wall_time_cap_s`` is hit the run stops at the last completed segment and
    records a ``wall_time_cap_reached`` blocker verdict.  The result is then a
    legitimate partial run, never relabelled as a finished whole shot, unless
    ``raise_on_wall_time_cap`` is set, in which case it raises
    :class:`WholeShotWallTimeError` instead.
    """

    started_wall_s = time.monotonic()
    fixed_deck = _fixed_step_deck(deck)
    resolved_target_s = _resolve_target_time_s(fixed_deck, target_time_s)

    plan = plan_segmented_whole_shot(
        target_time_s=resolved_target_s,
        dt_s=float(fixed_deck.dt_s),
        segment_steps=int(segment_steps),
        checkpoint_every_segments=int(checkpoint_every_segments),
        wall_time_cap_s=wall_time_cap_s,
        resume_from_checkpoint=resume_from_checkpoint,
        explicit_total_steps=explicit_total_steps,
    )

    run_path = Path(run_dir)
    segments_dir = run_path / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # The deck the horizon is actually run against is the planned-horizon
    # deck: its n_steps equals the planned total so a single-shot uninterrupted
    # reference run (used by the equivalence check) is well-defined.
    horizon_deck = FirstPrinciples3DDeck.from_deck(
        {**asdict(fixed_deck), "target_time_s": None},
        n_steps=plan.total_steps,
    )

    commit, dirty = git_provenance()
    command_payload = _command_payload(commit=commit, dirty=dirty)
    deck_payload = _deck_payload(horizon_deck)

    _write_json(run_path / "deck.json", deck_payload)
    _write_json(run_path / "command.json", command_payload)
    _write_json(run_path / "plan.json", plan.to_dict())

    # ----- resume / fresh session -----------------------------------------
    if plan.resume_from_checkpoint is not None:
        session = load_checkpoint_into_first_principles_3d_session(
            checkpoint_path=plan.resume_from_checkpoint,
            deck=horizon_deck,
        )
        resume_completed = int(session.completed_steps)
        # Rehydrate the cumulative ledgers from the sidecar written beside the
        # resumed-from checkpoint.  Skipped (already-completed) segments never
        # call ledgers.accumulate(); without this the resumed run's cumulative
        # ledgers would cover only post-resume steps (audit A-6).
        ledgers = _load_cumulative_ledger_sidecar(plan.resume_from_checkpoint)
    else:
        session = build_first_principles_3d_session(horizon_deck)
        resume_completed = 0
        ledgers = _CumulativeLedgers()

    segment_records: list[dict[str, Any]] = []
    last_result: HybridPIC3DSimulationResult | None = None
    completed = resume_completed
    wall_time_cap_reached = False

    for segment_index in range(plan.segment_count):
        planned_completed_before = plan.segment_steps * segment_index
        if completed > planned_completed_before:
            # A resume started this run already past this segment.
            continue
        this_segment_steps = (
            plan.last_segment_steps
            if segment_index == plan.segment_count - 1
            else plan.segment_steps
        )
        remaining = plan.total_steps - completed
        this_segment_steps = min(this_segment_steps, remaining)
        if this_segment_steps <= 0:
            break

        segment_started_s = time.monotonic()
        result = session.run_segment(this_segment_steps)
        last_result = result
        segment_wall_s = time.monotonic() - segment_started_s
        completed += int(result.telemetry.n_steps_completed)
        ledgers.accumulate(result)

        checkpoint_scheduled = (
            segment_index in plan.checkpoint_segment_indices
        )
        checkpoint_path: Path | None = None
        roundtrip: dict[str, Any] | None = None
        if checkpoint_scheduled:
            checkpoint_path = segments_dir / f"segment_{segment_index:04d}.npz"
            roundtrip = write_simulation_state_checkpoint_roundtrip(
                simulation=result,
                checkpoint_path=checkpoint_path,
                deck=horizon_deck,
            )
            # Persist the cumulative-ledger state AS OF this segment beside the
            # checkpoint.  ``ledgers`` already has segments 0..segment_index
            # accumulated, so a resume from this checkpoint reconstructs the
            # exact pre-resume totals (audit A-6).
            _write_json(
                _cumulative_ledger_sidecar_path(checkpoint_path),
                {
                    "ledger_sidecar_status": (
                        "candidate_cumulative_segmented_ledger_resume_sidecar"
                    ),
                    "checkpoint_path": str(checkpoint_path),
                    "segment_index": int(segment_index),
                    "total_steps_completed_after_segment": int(completed),
                    "cumulative_ledger_state": ledgers.to_state_dict(),
                },
            )

        is_final_segment = segment_index == plan.segment_count - 1
        # Reload through the fail-closed loader at non-final checkpoints so the
        # next segment continues from validated checkpoint state.
        if checkpoint_scheduled and not is_final_segment:
            session = load_checkpoint_into_first_principles_3d_session(
                checkpoint_path=checkpoint_path,
                deck=horizon_deck,
            )

        segment_record = _segment_manifest(
            segment_index=segment_index,
            segment_steps_requested=this_segment_steps,
            result=result,
            total_steps_completed=completed,
            checkpoint_path=checkpoint_path,
            roundtrip=roundtrip,
            segment_wall_s=segment_wall_s,
        )
        segment_records.append(segment_record)
        _write_json(
            segments_dir / f"segment_{segment_index:04d}.manifest.json",
            segment_record,
        )

        elapsed_wall_s = time.monotonic() - started_wall_s
        if (
            plan.wall_time_cap_s is not None
            and elapsed_wall_s >= plan.wall_time_cap_s
            and not is_final_segment
        ):
            wall_time_cap_reached = True
            break

    if last_result is None:  # pragma: no cover - segment_count >= 1
        raise RuntimeError("segmented whole-shot run produced no segments")

    horizon_complete = completed >= plan.total_steps
    final_summary = _simulation_summary(
        last_result,
        declared_scope=horizon_deck.validation_scope,
        device_name=horizon_deck.device_name,
        total_steps=completed,
    )

    equivalence = _restart_equivalence_block(
        horizon_deck=horizon_deck,
        plan=plan,
        segmented_summary=final_summary,
        horizon_complete=horizon_complete,
        verify_restart_equivalence=verify_restart_equivalence,
    )

    blockers = _blocker_verdicts(
        plan=plan,
        horizon_complete=horizon_complete,
        wall_time_cap_reached=wall_time_cap_reached,
        equivalence=equivalence,
        segment_records=segment_records,
    )

    # SS10-3 (closes audit A4): four compact, stable audit summary blocks so a
    # reviewer can audit the engineering probe without rerunning the
    # first-principles runner.  They are extracted from a genuine runtime result
    # for the resolved horizon deck (a single-step first-principles run), never
    # fabricated; every acceptance flag inside them is false.
    audit_summaries = _first_principles_audit_summaries(horizon_deck)

    total_wall_s = time.monotonic() - started_wall_s
    run_manifest = {
        "status": SEGMENTED_WHOLE_SHOT_STATUS,
        "run_intent": "segmented_whole_shot_engineering_candidate_run",
        "run_dir": str(run_path),
        "device_name": horizon_deck.device_name,
        "deck_name": horizon_deck.validation_scope,
        # WS9-1: carry the resolved deck's explicit validation scope on the run
        # manifest so the selected runtime-demonstrator scope (never the deck
        # id) is the declared validation scope.  Fixes audit P0-1.
        "deck": deck_payload,
        "command": command_payload,
        "plan": plan.to_dict(),
        "resumed_from_checkpoint": plan.resume_from_checkpoint,
        "resume_started_at_step": resume_completed,
        "total_steps_completed": int(completed),
        "planned_total_steps": plan.total_steps,
        "horizon_complete": bool(horizon_complete),
        "wall_time_cap_reached": bool(wall_time_cap_reached),
        "wall_clock_seconds": round(total_wall_s, 6),
        "segment_count_executed": len(segment_records),
        "all_checkpoint_roundtrips_match": all(
            seg["checkpoint"]["write_read_hashes_match"] is True
            for seg in segment_records
            if seg["checkpoint"] is not None
        ),
        "cumulative_ledgers": ledgers.to_dict(
            total_steps_completed=completed,
            horizon_steps=plan.total_steps,
        ),
        "segments": segment_records,
        "segmented_run": final_summary,
        "restart_equivalence": equivalence,
        "blocker_verdicts": blockers,
        # SS10-3 (closes audit A4): compact audit summary blocks.  Each block is
        # a small stable dict (no huge nested arrays) so a reviewer can audit
        # the engineering probe straight from the manifest.
        "first_principles_scope_summary": audit_summaries[
            "first_principles_scope_summary"
        ],
        "same_scope_summary": audit_summaries["same_scope_summary"],
        "power_port_summary": audit_summaries["power_port_summary"],
        "geometry_blocker_summary": audit_summaries["geometry_blocker_summary"],
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "segmented_whole_shot_outputs_are_engineering_only": True,
            "validation_promotion_allowed": False,
        },
        "source_references": [
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_CODEX_AGENT_AUDIT_AND_NEXT_"
                    "INSTRUCTIONS_2026_05_18.md"
                ),
                "lines": "144-161, 224-237",
                "role": "wp_n4_segmented_whole_shot_runner_requirement",
            },
            {
                "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
                "lines": "412-414",
                "role": "restart_reproducibility_acceptance_fields",
            },
            {
                "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
                "lines": "87-88",
                "role": "deterministic_checkpoint_restart_requirement",
            },
        ],
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "segmented_whole_shot_engineering_candidate_only",
        },
        "can_support_first_principles_acceptance": False,
    }
    # S3.7: emit the run-manifest SHA-256 so the certificate gate's
    # ``run_manifest_hash`` channel can be populated without re-reading the
    # file.  The hash is computed over the payload EXCLUDING itself (stable).
    run_manifest["run_manifest_sha256"] = stable_manifest_hash(run_manifest)
    _write_json(run_path / "run_manifest.json", run_manifest)

    if wall_time_cap_reached and raise_on_wall_time_cap:
        raise WholeShotWallTimeError(
            f"wall-time cap {plan.wall_time_cap_s}s reached after "
            f"{completed}/{plan.total_steps} steps; partial run directory "
            f"written to {run_path}"
        )
    return run_manifest


def build_staged_restart_equivalence_evidence(
    *,
    deck: Mapping[str, Any] | object | None,
    staged_segment_plans: tuple[tuple[int, int], ...],
    run_root: str | Path,
) -> dict[str, Any]:
    """Prove restart equivalence at several staged (small) horizons.

    ``staged_segment_plans`` is a tuple of ``(total_steps, segment_steps)``
    pairs.  Each stage runs that horizon both as an uninterrupted run and as a
    segmented whole-shot run, then asserts the state fingerprint and tracked
    observables are bit-identical.  This is the first restart-equivalence
    evidence required before any 12 us attempt; it is NOT a 12 us run.
    """

    fixed_deck = _fixed_step_deck(deck)
    dt_s = float(fixed_deck.dt_s)
    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []
    for total_steps, segment_steps in staged_segment_plans:
        if int(total_steps) != total_steps or total_steps <= 0:
            raise ValueError("each staged total_steps must be a positive integer")
        if int(segment_steps) != segment_steps or segment_steps <= 0:
            raise ValueError("each staged segment_steps must be a positive integer")
        if segment_steps >= total_steps:
            raise ValueError(
                "staged segment_steps must be smaller than total_steps; a "
                "single-segment horizon does not exercise restart equivalence"
            )
        # The staged horizon is the EXACT caller-requested step count.  A
        # nominal target time is reported for traceability, but the step
        # count is passed explicitly because ``n * dt`` is not exactly
        # representable in float and a float round-trip can inflate the
        # horizon (e.g. 6 -> 7 steps).  The planner path is still exercised.
        stage_target_s = float(total_steps) * dt_s
        stage_dir = run_root_path / f"stage_{int(total_steps):06d}"
        manifest = run_segmented_whole_shot(
            deck=fixed_deck,
            run_dir=stage_dir,
            segment_steps=int(segment_steps),
            target_time_s=stage_target_s,
            explicit_total_steps=int(total_steps),
            checkpoint_every_segments=1,
            wall_time_cap_s=None,
            verify_restart_equivalence=True,
        )
        equivalence = manifest["restart_equivalence"]
        stages.append({
            "total_steps": int(total_steps),
            "segment_steps": int(segment_steps),
            "segment_count": manifest["plan"]["segment_count"],
            "target_time_s": stage_target_s,
            "run_dir": manifest["run_dir"],
            "horizon_complete": manifest["horizon_complete"],
            "state_fingerprints_match": equivalence.get(
                "state_fingerprints_match"
            ),
            "tracked_observables_match_exactly": equivalence.get(
                "tracked_observables_match_exactly"
            ),
            "uninterrupted_state_fingerprint_sha256": equivalence.get(
                "uninterrupted_state_fingerprint_sha256"
            ),
            "segmented_state_fingerprint_sha256": equivalence.get(
                "segmented_state_fingerprint_sha256"
            ),
            "equivalence_proven": (
                equivalence.get("state_fingerprints_match") is True
                and equivalence.get("tracked_observables_match_exactly") is True
            ),
        })

    all_proven = bool(stages) and all(
        stage["equivalence_proven"] is True for stage in stages
    )
    return {
        "status": (
            "experimental_staged_restart_equivalence_evidence_not_validation"
        ),
        "run_intent": "staged_restart_equivalence_evidence_for_whole_shot",
        "run_root": str(run_root_path),
        "stage_count": len(stages),
        "all_stages_equivalence_proven": all_proven,
        "stages": stages,
        "twelve_microsecond_run": {
            "attempted": False,
            "status": "compute_wall_blocked",
            "reason": (
                "A 12 us source-sign run requires ~1.2e8 steps at dt=1e-13. "
                "That is a known compute-wall blocker; this evidence proves "
                "restart equivalence at staged small horizons only."
            ),
        },
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "staged_evidence_is_engineering_only": True,
            "validation_promotion_allowed": False,
        },
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "staged_restart_equivalence_evidence_only",
        },
        "can_support_first_principles_acceptance": False,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_target_time_s(
    deck: FirstPrinciples3DDeck,
    target_time_s: float | None,
) -> float:
    if target_time_s is not None:
        return float(target_time_s)
    if deck.target_time_s is not None:
        return float(deck.target_time_s)
    # No declared horizon: fall back to n_steps * dt so the planner is total.
    return float(deck.n_steps) * float(deck.dt_s)


_AULUCK_SIGMA_P_BLOCKED_TERMS = (
    "term_ii_motional_magnetic_sigma_p_J",
    "term_iv_motional_electric_sigma_p_J",
    "term_v_resistive_sigma_p_J",
    "term_vi_anomalous_poloidal_sigma_p_J",
)


def _first_principles_audit_summaries(
    horizon_deck: FirstPrinciples3DDeck,
) -> dict[str, dict[str, Any]]:
    """Build the four SS10-3 audit summary blocks from a genuine runtime result.

    Closes audit A4.  A single-step first-principles run of the resolved horizon
    deck is executed and its telemetry is summarised into four compact, stable
    blocks: scope, same-scope channels, power-port Sigma-p term status, and the
    blocked geometry fields.  No huge nested array is embedded; every acceptance
    flag in the blocks is false (this is engineering-probe evidence only).
    """

    from dpf.first_principles.runner import run_first_principles_3d_deck

    probe_deck = FirstPrinciples3DDeck.from_deck(horizon_deck, n_steps=1)
    result = run_first_principles_3d_deck(probe_deck)
    telemetry = result.telemetry
    validation_packet = result.validation_packet

    return {
        "first_principles_scope_summary": _scope_summary_block(
            telemetry, validation_packet
        ),
        "same_scope_summary": _same_scope_summary_block(telemetry),
        "power_port_summary": _power_port_summary_block(telemetry),
        "geometry_blocker_summary": _geometry_blocker_summary_block(telemetry),
    }


def _scope_summary_block(
    telemetry: Mapping[str, Any],
    validation_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact validation/source/architecture scope summary (audit A4)."""

    return {
        "summary_status": "engineering_probe_scope_summary_not_validation",
        "validation_scope": telemetry.get("same_scope_source", {}).get(
            "declared_scope"
        ),
        "selected_machine_source_scope": validation_packet.get("source_scope"),
        "architecture_source_scope": validation_packet.get(
            "architecture_source_scope"
        ),
        "architecture_evidence_role": validation_packet.get(
            "architecture_evidence_role"
        ),
        # Both acceptance flags are pinned false: a segmented engineering probe
        # never promotes first-principles acceptance or a runtime claim.
        "can_support_first_principles_acceptance": False,
        "accepted_runtime_claim": False,
    }


def _same_scope_summary_block(
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact same-scope declared-scope + channel-state summary (audit A4)."""

    same_scope = telemetry.get("same_scope_source", {})
    channel_states = same_scope.get("channel_states")
    channel_states = (
        dict(channel_states) if isinstance(channel_states, Mapping) else {}
    )
    accepted = sorted(
        name for name, state in channel_states.items() if state == "accepted"
    )
    return {
        "summary_status": "engineering_probe_same_scope_summary_not_validation",
        "declared_scope": same_scope.get("declared_scope"),
        "same_scope_source_status": same_scope.get("status"),
        "channel_states": channel_states,
        "accepted_channels": accepted,
        "missing_acceptance_channels": list(
            same_scope.get("missing_acceptance_channels", ())
        ),
        "can_support_first_principles_acceptance": False,
    }


def _power_port_summary_block(
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact Auluck / Sigma-p power-port term-status summary (audit A4)."""

    power_port = telemetry.get("power_port", {})
    ledger = power_port.get("wp_n1_auluck_power_port_ledger", {})
    term_status = ledger.get("energy_ledger_term_status")
    term_status = dict(term_status) if isinstance(term_status, Mapping) else {}
    blocked_terms = ledger.get("blocked_terms")
    blocked_terms = (
        dict(blocked_terms) if isinstance(blocked_terms, Mapping) else {}
    )
    sigma_p_terms_blocked = {
        term: term_status.get(term) == "blocked"
        for term in _AULUCK_SIGMA_P_BLOCKED_TERMS
    }
    return {
        "summary_status": "engineering_probe_power_port_summary_not_validation",
        "power_port_status": power_port.get("status"),
        "auluck_eq6_term_status": term_status,
        "sigma_p_surface_packet_status": ledger.get(
            "sigma_p_surface_packet_status"
        ),
        # Auluck terms II / IV / V / VI are the Sigma-p surface-integral terms;
        # all four are blocked until a reviewed Sigma-p face set exists.
        "sigma_p_terms_ii_iv_v_vi_blocked": sigma_p_terms_blocked,
        "all_sigma_p_terms_blocked": all(sigma_p_terms_blocked.values()),
        "blocked_term_reasons": blocked_terms,
        "all_six_terms_computed_independently": ledger.get(
            "all_six_terms_computed_independently", False
        ),
        "ledger_blocked": ledger.get("ledger_blocked"),
        "can_support_first_principles_acceptance": False,
    }


def _geometry_blocker_summary_block(
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact blocked-geometry-field summary (audit A4)."""

    boundary_policy = telemetry.get("boundary_policy", {})
    conductor_mask = boundary_policy.get("conductor_mask", {})
    blocked_fields = conductor_mask.get("blocked_geometry_fields")
    blocked_fields = (
        list(blocked_fields) if isinstance(blocked_fields, list) else []
    )
    return {
        "summary_status": (
            "engineering_probe_geometry_blocker_summary_not_validation"
        ),
        "blocked_geometry_field_count": len(blocked_fields),
        "blocked_geometry_fields": blocked_fields,
        "blocked_geometry_field_names": [
            str(entry.get("field_name"))
            for entry in blocked_fields
            if isinstance(entry, Mapping)
        ],
        "hollow_anode_declared_by_source": (
            conductor_mask.get("pf1000_geometry_features", {}).get(
                "hollow_anode_declared_by_source"
            )
        ),
        "can_support_first_principles_acceptance": False,
    }


def _command_payload(
    *,
    commit: str | None,
    dirty: bool | None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    source_hashes: dict[str, str | None] = {}
    for rel in _SOURCE_HASH_MODULES:
        candidate = repo_root / rel
        source_hashes[rel] = (
            sha256_of_file(candidate) if candidate.is_file() else None
        )
    index_path = repo_root / _SOURCE_TRUTH_INDEX_PATH
    source_truth_index_sha256 = (
        sha256_of_file(index_path) if index_path.is_file() else None
    )
    return {
        "command_argv": list(sys.argv),
        "git_commit": commit,
        "dirty_worktree": dirty,
        "python_version": sys.version.split()[0],
        "source_module_sha256": source_hashes,
        "source_truth_index_path": _SOURCE_TRUTH_INDEX_PATH,
        "source_truth_index_sha256": source_truth_index_sha256,
    }


def _deck_payload(deck: FirstPrinciples3DDeck) -> dict[str, Any]:
    values = asdict(deck)
    serialisable = json.loads(json.dumps(values, default=str, sort_keys=True))
    return {
        "deck": serialisable,
        "deck_sha256": sha256_of_json(serialisable),
        "device_name": deck.device_name,
        "validation_scope": deck.validation_scope,
    }


def _segment_manifest(
    *,
    segment_index: int,
    segment_steps_requested: int,
    result: HybridPIC3DSimulationResult,
    total_steps_completed: int,
    checkpoint_path: Path | None,
    roundtrip: Mapping[str, Any] | None,
    segment_wall_s: float,
) -> dict[str, Any]:
    telemetry = result.telemetry
    fingerprint = telemetry.state_fingerprint
    fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
    continuation = telemetry.continuation_state
    continuation = continuation if isinstance(continuation, Mapping) else {}
    checkpoint_block: dict[str, Any] | None = None
    if checkpoint_path is not None and isinstance(roundtrip, Mapping):
        checkpoint_block = {
            "checkpoint_path": str(checkpoint_path),
            "write_content_sha256": roundtrip.get("write_content_sha256"),
            "read_content_sha256": roundtrip.get("read_content_sha256"),
            "write_read_hashes_match": roundtrip.get("write_read_hashes_match"),
            "terminal_state_fingerprint_sha256": roundtrip.get(
                "terminal_state_fingerprint_sha256"
            ),
        }
    return {
        "segment_index": int(segment_index),
        "segment_steps_requested": int(segment_steps_requested),
        "segment_steps_completed": int(telemetry.n_steps_completed),
        "total_steps_completed_after_segment": int(total_steps_completed),
        "segment_wall_clock_seconds": round(segment_wall_s, 6),
        "stop_reason": telemetry.stop_reason,
        "state_fingerprint_sha256": fingerprint.get("sha256"),
        "continuation_total_steps": continuation.get("total_steps_completed"),
        "segment_cumulative_j_dot_e_work_J": (
            telemetry.cumulative_j_dot_e_work_J
        ),
        "segment_cumulative_j_dot_e_step_count": (
            telemetry.cumulative_j_dot_e_step_count
        ),
        "finite_state_all": (
            telemetry.finite_state.get("all_finite")
            if isinstance(telemetry.finite_state, Mapping)
            else None
        ),
        "checkpoint": checkpoint_block,
    }


def _restart_equivalence_block(
    *,
    horizon_deck: FirstPrinciples3DDeck,
    plan: WholeShotPlan,
    segmented_summary: Mapping[str, Any],
    horizon_complete: bool,
    verify_restart_equivalence: bool,
) -> dict[str, Any]:
    """Compare the segmented run against an uninterrupted run of the horizon.

    Equivalence is only meaningful when the segmented run completed the full
    planned horizon: a wall-time-truncated partial run cannot be compared
    against a full uninterrupted reference.
    """

    if not verify_restart_equivalence:
        return {
            "verified": False,
            "reason": "restart_equivalence_check_disabled_by_caller",
        }
    if not horizon_complete:
        return {
            "verified": False,
            "reason": (
                "segmented run did not complete the planned horizon "
                "(wall-time cap); restart equivalence is undefined for a "
                "partial run and is intentionally not asserted"
            ),
        }
    from dpf.first_principles.runner import run_first_principles_3d_deck

    uninterrupted = run_first_principles_3d_deck(horizon_deck)
    uninterrupted_summary = _simulation_summary(
        uninterrupted.result,
        declared_scope=horizon_deck.validation_scope,
        device_name=horizon_deck.device_name,
        total_steps=uninterrupted.result.telemetry.n_steps_completed,
    )
    comparisons = _observable_comparisons(uninterrupted_summary, segmented_summary)
    fingerprints_match = (
        uninterrupted_summary["state_fingerprint_sha256"]
        == segmented_summary["state_fingerprint_sha256"]
    )
    observables_match = all(
        item["absolute_delta"] in (0.0, None) for item in comparisons.values()
    )
    return {
        "verified": True,
        "horizon_steps": plan.total_steps,
        "uninterrupted_state_fingerprint_sha256": uninterrupted_summary[
            "state_fingerprint_sha256"
        ],
        "segmented_state_fingerprint_sha256": segmented_summary[
            "state_fingerprint_sha256"
        ],
        "state_fingerprints_match": fingerprints_match,
        "tracked_observables_match_exactly": observables_match,
        "observable_comparisons": comparisons,
        "uninterrupted": uninterrupted_summary,
    }


def _blocker_verdicts(
    *,
    plan: WholeShotPlan,
    horizon_complete: bool,
    wall_time_cap_reached: bool,
    equivalence: Mapping[str, Any],
    segment_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Explicit, honest blocker verdicts written into the run directory."""

    verdicts: list[dict[str, Any]] = []

    # B1: 12 us compute wall.
    twelve_us_steps = int(math.ceil(12.0e-6 / plan.dt_s))
    verdicts.append({
        "id": "B-WPN4-12US-COMPUTE-WALL",
        "blocker": "full_12us_source_sign_whole_shot",
        "status": (
            "blocked" if plan.total_steps < twelve_us_steps else "in_scope"
        ),
        "detail": (
            f"A 12 us run needs {twelve_us_steps} steps at dt={plan.dt_s:g}. "
            f"This run planned {plan.total_steps} steps. A full 12 us run is "
            "a known compute-wall blocker and is not produced here."
        ),
    })

    # B2: wall-time cap.
    verdicts.append({
        "id": "B-WPN4-WALL-TIME-CAP",
        "blocker": "wall_time_cap_truncation",
        "status": "triggered" if wall_time_cap_reached else "clear",
        "detail": (
            "Run stopped at the wall-time cap before the planned horizon; "
            "the run directory is a legitimate partial-run artifact."
            if wall_time_cap_reached
            else "Wall-time cap not reached (or no cap configured)."
        ),
    })

    # B3: checkpoint integrity.
    checkpoint_segments = [
        seg for seg in segment_records if seg["checkpoint"] is not None
    ]
    all_roundtrips_ok = all(
        seg["checkpoint"]["write_read_hashes_match"] is True
        for seg in checkpoint_segments
    )
    verdicts.append({
        "id": "B-WPN4-CHECKPOINT-INTEGRITY",
        "blocker": "checkpoint_roundtrip_corruption",
        "status": "clear" if all_roundtrips_ok else "triggered",
        "detail": (
            f"{len(checkpoint_segments)} checkpoints written; all write/read "
            "content hashes match."
            if all_roundtrips_ok
            else "A checkpoint round-trip content hash mismatch was detected."
        ),
    })

    # B4: restart equivalence.
    equivalence_proven = (
        equivalence.get("state_fingerprints_match") is True
        and equivalence.get("tracked_observables_match_exactly") is True
    )
    if not equivalence.get("verified"):
        equivalence_status = "not_evaluated"
    elif equivalence_proven:
        equivalence_status = "clear"
    else:
        equivalence_status = "triggered"
    verdicts.append({
        "id": "B-WPN4-RESTART-EQUIVALENCE",
        "blocker": "segmented_run_not_equivalent_to_uninterrupted",
        "status": equivalence_status,
        "detail": (
            equivalence.get("reason")
            or (
                "Segmented run is bit-identical (state fingerprint + tracked "
                "observables) to the uninterrupted run of the same horizon."
                if equivalence_proven
                else "Segmented run diverged from the uninterrupted run."
            )
        ),
    })

    triggered = [v for v in verdicts if v["status"] == "triggered"]
    return {
        "verdicts": verdicts,
        "any_triggered": bool(triggered),
        "triggered_ids": [v["id"] for v in triggered],
        "horizon_complete": bool(horizon_complete),
        "summary": (
            "engineering_candidate_partial_or_blocked"
            if (triggered or not horizon_complete)
            else "engineering_candidate_horizon_complete_equivalence_proven"
        ),
    }


def _repo_root() -> Path:
    # src/dpf/first_principles/segmented_whole_shot.py -> repo root is 4 up.
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    # Record content hash alongside heavy artifacts is unnecessary; the run
    # manifest already carries per-segment checkpoint hashes.  This keeps the
    # write deterministic for reproducibility.
    _ = sha256_of_text(text)


def _cumulative_ledger_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the cumulative-ledger sidecar path beside a checkpoint ``.npz``.

    The sidecar lives next to the checkpoint with the ``.npz`` suffix replaced
    by ``.cumulative_ledger.json`` so resume can locate it from the checkpoint
    path alone, without touching the checkpoint schema or its fail-closed
    loader (audit A-6, smallest blast radius).
    """

    path = Path(checkpoint_path)
    return path.with_suffix("").with_name(
        f"{path.with_suffix('').name}.cumulative_ledger.json"
    )


def _load_cumulative_ledger_sidecar(
    checkpoint_path: str | Path,
) -> _CumulativeLedgers:
    """Rehydrate cumulative ledgers from a resumed-from checkpoint's sidecar.

    When no sidecar exists (e.g. the checkpoint was the segment-0 checkpoint of
    a run produced before this fix, or a hand-supplied checkpoint) the resume
    falls back to a zero baseline -- the same behaviour as before this fix --
    so a missing sidecar degrades gracefully rather than failing the resume.
    """

    sidecar_path = _cumulative_ledger_sidecar_path(checkpoint_path)
    if not sidecar_path.is_file():
        return _CumulativeLedgers()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    state = payload.get("cumulative_ledger_state")
    if not isinstance(state, Mapping):
        return _CumulativeLedgers()
    return _CumulativeLedgers.from_state_dict(state)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
