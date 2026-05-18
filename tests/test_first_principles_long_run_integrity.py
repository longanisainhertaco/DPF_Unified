"""WP-4 / SSR-007 long-run runtime integrity tests.

Negative tests: prove that hidden state repairs, history truncations, silent
conservation passes, and untelemetered density floors would be CAUGHT.

Tests 1-3: close the "equivalence machinery exists but is never exercised" gap.
Tests 4-6: regression guards for the integrity fixes already applied to
           runner.py (_conservation_telemetry) and hybrid_loop.py
           (_source_workflow_telemetry electron_density_floor).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dpf.first_principles.runner import (
    FirstPrinciples3DDeck,
    build_first_principles_3d_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smoke_deck(n_steps: int = 4) -> FirstPrinciples3DDeck:
    """Smallest deterministic fixed-step deck: default 5x5x5, seed 0."""
    return FirstPrinciples3DDeck.from_deck({"n_steps": n_steps, "seed": 0})


# ---------------------------------------------------------------------------
# Test 1: checkpoint / restart equivalence is actually exercised
# ---------------------------------------------------------------------------


def test_checkpoint_restart_reproduces_uninterrupted_run(tmp_path: Path) -> None:
    """Correct checkpoint/restart must match the uninterrupted fingerprint.

    Without this test, a hidden state repair on the uninterrupted path -- or a
    checkpoint that drops electron-energy / ionization / lagged-field-work --
    would go unnoticed.  Closes B-WP4-5.
    """
    from dpf.first_principles.checkpoint_restart import (
        build_experimental_checkpoint_restart_packet,
    )

    deck = _smoke_deck(n_steps=4)
    packet = build_experimental_checkpoint_restart_packet(
        deck=deck,
        split_after_steps=2,
        checkpoint_path=tmp_path / "ckpt.npz",
    )

    # Round-trip write/read must be bit-identical.
    assert packet["checkpoint_roundtrip"]["write_read_hashes_match"] is True

    # The restarted run must cover the full step budget.
    assert packet["restart_total_steps_completed"] == deck.n_steps

    # The loaded session must have restored lagged state and circuit state.
    # loaded_completed_steps reflects the session total after the second segment
    # ran (split_after_steps + remaining = n_steps).
    rs = packet["restart_state"]
    assert rs["loaded_completed_steps"] == deck.n_steps
    assert rs["previous_total_current_loaded"] is True

    # Equivalence: fingerprint + observables must match the uninterrupted run.
    assert packet["state_fingerprints_match"] is True, (
        "checkpoint/restart diverged from uninterrupted run -- hidden state "
        "repair or unsaved state channel detected"
    )
    assert packet["tracked_observables_match_exactly"] is True, (
        "tracked observables after checkpoint restart do not match uninterrupted"
    )


# ---------------------------------------------------------------------------
# Test 2: split-continuation equivalence is actually exercised
# ---------------------------------------------------------------------------


def test_split_continuation_reproduces_uninterrupted_run() -> None:
    """A+B live continuation must match the uninterrupted run.

    Catches mutable module-level buffers or per-run-only repairs in the stepper.
    Closes B-WP4-5.
    """
    from dpf.first_principles.split_continuation import (
        build_experimental_split_continuation_packet,
    )

    deck = _smoke_deck(n_steps=4)
    packet = build_experimental_split_continuation_packet(
        deck=deck,
        split_after_steps=2,
    )

    assert packet["split_total_steps_completed"] == deck.n_steps

    # Lagged field work must be preserved into the second segment.
    assert (
        packet["continuation_state"]["lagged_field_work_preserved_into_second_segment"]
        is True
    ), "lagged field work was not preserved across the live split"

    assert packet["state_fingerprints_match"] is True, (
        "split-continuation diverged -- non-equivalent segmented integration"
    )
    assert packet["tracked_observables_match_exactly"] is True


# ---------------------------------------------------------------------------
# Test 3: cumulative ledgers survive aggressive payload capping
# ---------------------------------------------------------------------------


def test_cumulative_ledgers_survive_payload_capping() -> None:
    """Capping retained step payloads must NOT truncate cumulative counters.

    WP-4 deliverable: cumulative histories independent of retained payload
    count.  Regression guard: max_step_results must never bound the cumulative
    ledgers.  Closes B-WP4-5.
    """
    n_steps = 6
    # Force aggressive payload capping: only 2 retained summaries, stride 3.
    deck = FirstPrinciples3DDeck.from_deck(
        {
            "n_steps": n_steps,
            "max_step_results": 2,
            "history_stride": 3,
            "seed": 0,
        }
    )
    session = build_first_principles_3d_session(deck)
    result = session.run_segment(n_steps)
    tel = result.telemetry

    # Retained payloads are aggressively capped ...
    assert tel.retained_step_result_count <= 2
    assert len(tel.history_summary) <= 2

    # ... but every cumulative / completion counter covers the FULL horizon.
    assert tel.n_steps_completed == n_steps, (
        "n_steps_completed was truncated by payload capping"
    )
    assert tel.limiter_activation_summary is not None
    assert tel.limiter_activation_summary["steps_observed"] == n_steps, (
        "limiter_activation_summary.steps_observed was truncated by payload capping"
    )
    if tel.cumulative_j_dot_e_step_count > 0:
        assert tel.cumulative_j_dot_e_step_count == n_steps, (
            "cumulative_j_dot_e_step_count was truncated by payload capping"
        )
    assert tel.state_fingerprint is not None
    assert tel.continuation_state is not None
    assert tel.continuation_state["total_steps_completed"] == n_steps


# ---------------------------------------------------------------------------
# Test 4: conservation telemetry is honest (no lying "passed" flag)
# ---------------------------------------------------------------------------


def test_conservation_telemetry_has_no_passed_key_and_is_honest() -> None:
    """REGRESSION GUARD: runner._conservation_telemetry must not emit 'passed'.

    The fix replaced 'passed: finite' (finiteness-only, dishonest) with
    'finite_state' + 'energy_conservation_assessed: not_assessed_no_accepted_tolerance'.
    A consumer keying on 'passed' would silently inherit a false green on a
    run that lost 58% of its energy.  This test asserts the FIXED contract.
    """
    from dpf.first_principles import runner as fp_runner

    deck = _smoke_deck(n_steps=2)
    session = build_first_principles_3d_session(deck)
    session.run_segment(2)

    # Verify the packet structure through the runner's public
    # _conservation_telemetry directly, independent of the session wrapper.
    grid = deck.grid()
    initial = {"tracked_total_energy_J": 1e5}
    final_large_drift = {"tracked_total_energy_J": 4.16e4}  # -58.4% like artifact
    final_finite = {"tracked_total_energy_J": 9.9e4}  # small drift, still finite

    for final in (final_large_drift, final_finite):
        packet = fp_runner._conservation_telemetry(
            grid=grid,
            n_steps=10,
            dt_s=1e-13,
            initial=initial,
            final=final,
            final_diagnostics={"max_abs_div_B_T_per_m": 0.0},
        )

        # Must NOT contain the dishonest "passed" key.
        assert "passed" not in packet, (
            "conservation telemetry still emits the dishonest 'passed' key; "
            "the fix was not applied or was reverted"
        )

        # Must contain the honest replacement keys.
        assert "finite_state" in packet, (
            "conservation telemetry missing 'finite_state' after fix"
        )
        assert "energy_conservation_assessed" in packet, (
            "conservation telemetry missing 'energy_conservation_assessed' after fix"
        )
        assert packet["energy_conservation_assessed"] == "not_assessed_no_accepted_tolerance"

        # The raw drift number must be present so a reviewer can see the loss.
        assert "relative_tracked_total_energy_change" in packet
        assert isinstance(packet["relative_tracked_total_energy_change"], float)

    # Specifically: a -58% run must carry a finite_state that is True (the run
    # stayed finite) but the packet must NOT claim the energy loss passed any
    # tolerance -- because no tolerance is evaluated.
    pkt_drift = fp_runner._conservation_telemetry(
        grid=grid,
        n_steps=55580,
        dt_s=2.159e-10,
        initial={"tracked_total_energy_J": 170534.0},
        final={"tracked_total_energy_J": 71024.7},
        final_diagnostics={"max_abs_div_B_T_per_m": 14.03},
    )
    assert pkt_drift["finite_state"] is True   # the state was finite
    assert "passed" not in pkt_drift           # but there is no pass/fail claim
    rel = pkt_drift["relative_tracked_total_energy_change"]
    assert abs(rel - (-0.5835)) < 1e-2         # the drift is recorded honestly


# ---------------------------------------------------------------------------
# Test 5: electron-density floor is telemetered (not silent)
# ---------------------------------------------------------------------------


def test_electron_density_floor_is_telemetered_in_source_workflow() -> None:
    """REGRESSION GUARD for floors F1/F2 (hybrid_loop.py:190-193, :203).

    The fix replaced bare np.maximum(..., 1.0) with a named density_floor_m3
    parameter and added an electron_density_floor packet to
    _source_workflow_telemetry.  This test asserts the FIXED contract:
    every step's loop telemetry must expose floor_active_cells (count of
    floored cells) and density_floor_m3 (the floor value in m^-3).
    """
    deck = _smoke_deck(n_steps=2)
    session = build_first_principles_3d_session(deck)
    result = session.run_segment(2)
    last = result.telemetry.last_step
    assert last is not None, "last_step telemetry is None after 2 steps"

    # The electron_density_floor packet lives under source_workflow.
    source_workflow = last.get("source_workflow")
    assert source_workflow is not None, (
        "last_step.source_workflow is absent -- loop telemetry structure changed"
    )

    floor_tel = source_workflow.get("electron_density_floor")
    assert floor_tel is not None, (
        "electron_density_floor packet is absent from source_workflow; "
        "the floor F1/F2 fix was not applied or was reverted"
    )

    # Required keys after the fix.
    assert "floor_active_cells" in floor_tel, (
        "floor_active_cells count missing from electron_density_floor telemetry"
    )
    assert "density_floor_m3" in floor_tel, (
        "density_floor_m3 value missing from electron_density_floor telemetry"
    )
    assert "total_cells" in floor_tel
    assert "floor_source" in floor_tel
    assert "can_support_first_principles_acceptance" in floor_tel
    assert floor_tel["can_support_first_principles_acceptance"] is False

    # floor_active_cells must be a non-negative integer.
    assert isinstance(floor_tel["floor_active_cells"], int)
    assert floor_tel["floor_active_cells"] >= 0

    # density_floor_m3 must be positive.
    assert float(floor_tel["density_floor_m3"]) > 0.0

    # total_cells must be >= floor_active_cells.
    assert int(floor_tel["total_cells"]) >= floor_tel["floor_active_cells"]


# ---------------------------------------------------------------------------
# Test 6: checkpoint loaded into mismatched grid must fail attributably
# ---------------------------------------------------------------------------


def test_checkpoint_load_into_mismatched_grid_fails_attributably(
    tmp_path: Path,
) -> None:
    """REGRESSION GUARD for B-WP4-6 (closed by WP-N4 loader validation).

    load_checkpoint_into_first_principles_3d_session now validates checkpoint
    metadata (grid shape/spacing, circuit mode, closure policy, particle
    species, state-channel hashes) against the restart deck BEFORE writing any
    state array.  A deck/grid mismatch must raise an attributable error whose
    message contains 'grid', 'shape', or 'checkpoint' -- not a generic
    ValueError deep in the stepper.
    """
    from dpf.first_principles.state_checkpoint import (
        load_checkpoint_into_first_principles_3d_session,
        write_simulation_state_checkpoint_roundtrip,
    )

    writer_deck = _smoke_deck(n_steps=2)
    writer_session = build_first_principles_3d_session(writer_deck)
    first_seg = writer_session.run_segment(2)

    ckpt = tmp_path / "mismatch.npz"
    write_simulation_state_checkpoint_roundtrip(
        simulation=first_seg,
        checkpoint_path=ckpt,
    )

    # Build a deck with a deliberately different grid shape.
    original_shape = writer_deck.grid_shape  # (5, 5, 5) default
    bumped_shape = tuple(n + 1 for n in original_shape)
    mismatched_deck = FirstPrinciples3DDeck.from_deck(
        {"n_steps": 2, "seed": 0, "grid_shape": bumped_shape}
    )

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_checkpoint_into_first_principles_3d_session(
            checkpoint_path=ckpt,
            deck=mismatched_deck,
        )
    msg = str(exc_info.value).lower()
    assert "grid" in msg or "shape" in msg or "checkpoint" in msg, (
        f"mismatch error message '{exc_info.value}' does not mention "
        "'grid', 'shape', or 'checkpoint' -- the failure is not attributable"
    )


# ---------------------------------------------------------------------------
# Test 7: checkpoint loaded under a mismatched circuit/closure deck fails
# ---------------------------------------------------------------------------


def test_checkpoint_load_into_mismatched_closure_fails_attributably(
    tmp_path: Path,
) -> None:
    """A same-grid but different-closure restart deck must also fail closed.

    Grid shape alone is not enough: restarting a checkpoint under a different
    circuit mode or closure policy is not equivalent to one uninterrupted run.
    """
    from dpf.first_principles.state_checkpoint import (
        CheckpointDeckMismatchError,
        load_checkpoint_into_first_principles_3d_session,
        write_simulation_state_checkpoint_roundtrip,
    )

    writer_deck = _smoke_deck(n_steps=2)
    writer_session = build_first_principles_3d_session(writer_deck)
    first_seg = writer_session.run_segment(2)

    ckpt = tmp_path / "closure_mismatch.npz"
    write_simulation_state_checkpoint_roundtrip(
        simulation=first_seg,
        checkpoint_path=ckpt,
        deck=writer_deck,
    )

    # Same grid, different closure policy (Hall term toggled on).
    mismatched_deck = FirstPrinciples3DDeck.from_deck(
        {"n_steps": 2, "seed": 0, "include_hall": True}
    )
    with pytest.raises(CheckpointDeckMismatchError) as exc_info:
        load_checkpoint_into_first_principles_3d_session(
            checkpoint_path=ckpt,
            deck=mismatched_deck,
        )
    assert "closure" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 8: segmented checkpointed run equals one uninterrupted run
# ---------------------------------------------------------------------------


def test_segmented_run_equals_uninterrupted_and_preserves_ledgers(
    tmp_path: Path,
) -> None:
    """WP-N4: the segmented checkpointed run must equal one uninterrupted run.

    A segmented run that drops a state channel, double-counts a cumulative
    ledger, or diverges across a checkpoint boundary would be caught here.
    """
    from dpf.first_principles.checkpoint_restart import (
        build_experimental_segmented_run_packet,
    )

    n_steps = 6
    deck = _smoke_deck(n_steps=n_steps)
    packet = build_experimental_segmented_run_packet(
        deck=deck,
        segment_steps=2,
        checkpoint_dir=tmp_path / "segments",
        verify_against_uninterrupted=True,
    )

    # The horizon ran in 3 segments and completed the full step budget.
    assert packet["segment_count"] == 3
    assert packet["total_steps_completed"] == n_steps

    # Every segment boundary checkpoint round-tripped bit-identically.
    assert packet["all_segment_checkpoint_roundtrips_match"] is True

    # Cumulative ledgers cover the FULL horizon, not a single segment.
    ledgers = packet["cumulative_ledgers"]
    assert ledgers["limiter_steps_observed"] == n_steps, (
        "cumulative limiter ledger did not cover the full segmented horizon"
    )
    assert ledgers["covers_full_horizon"] is True
    if ledgers["cumulative_j_dot_e_step_count"] > 0:
        assert ledgers["cumulative_j_dot_e_step_count"] == n_steps, (
            "cumulative J.E step count was not accumulated across segments"
        )

    # Equivalence: the segmented run matches the uninterrupted run exactly.
    equivalence = packet["equivalence"]
    assert equivalence["state_fingerprints_match"] is True, (
        "segmented run diverged from the uninterrupted run -- non-equivalent "
        "segmented integration or an unsaved state channel"
    )
    assert equivalence["tracked_observables_match_exactly"] is True

    # The probe must never claim acceptance.
    assert packet["can_support_first_principles_acceptance"] is False
