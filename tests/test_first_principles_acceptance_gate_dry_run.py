"""Tests for the report-only acceptance-gate dry run (SS10-7).

These tests run the dry run against the six-step PF-1000 full-energy
engineering probe and assert that the ledger is fail-closed: every gate is
reported, no acceptance flag is promoted, and every blocked gate names a
non-empty list of missing inputs.
"""

from __future__ import annotations

import json

import pytest

from dpf.first_principles.acceptance_gate_dry_run import (
    AcceptanceGateDryRunLedger,
    GateDryRunResult,
    run_acceptance_gate_dry_run,
    write_ledger_json,
)

EXPECTED_GATES = (
    "numerical_fidelity",
    "same_scope_comparator",
    "uq",
    "certificate",
    "geometry",
    "startup",
    "power_port",
    "neutron",
)


@pytest.fixture(scope="module")
def ledger() -> AcceptanceGateDryRunLedger:
    """Run the PF-1000 full-energy dry run once for the whole module."""

    return run_acceptance_gate_dry_run()


def test_all_eight_gates_are_reported(ledger: AcceptanceGateDryRunLedger) -> None:
    assert ledger.all_gates_reported
    reported = tuple(gate.gate for gate in ledger.gates)
    assert reported == EXPECTED_GATES
    assert len(ledger.gates) == 8


def test_ledger_is_fail_closed(ledger: AcceptanceGateDryRunLedger) -> None:
    assert ledger.is_fail_closed
    for gate in ledger.gates:
        assert gate.status in {"pass", "blocked"}


def test_no_acceptance_flag_is_promoted(ledger: AcceptanceGateDryRunLedger) -> None:
    assert ledger.report_only is True
    assert ledger.promotes_acceptance is False
    assert ledger.accepted_runtime_claim is False
    assert ledger.can_support_first_principles_acceptance is False
    for gate in ledger.gates:
        assert gate.promotes_acceptance is False
        assert gate.can_support_first_principles_acceptance is False


def test_every_blocked_gate_names_missing_inputs(
    ledger: AcceptanceGateDryRunLedger,
) -> None:
    blocked = [gate for gate in ledger.gates if gate.status == "blocked"]
    # The PF-1000 full-energy probe has no accepted gate today.
    assert len(blocked) == 8
    for gate in blocked:
        assert gate.missing, f"{gate.gate} is blocked with no named missing input"
        assert all(isinstance(item, str) and item for item in gate.missing)
        assert gate.next_action, f"{gate.gate} is blocked with no next_action"


def test_geometry_gate_names_the_five_blocked_geometry_fields(
    ledger: AcceptanceGateDryRunLedger,
) -> None:
    geometry = next(gate for gate in ledger.gates if gate.gate == "geometry")
    assert geometry.status == "blocked"
    # SS10-2: the conductor-mask packet exposes five blocked geometry fields.
    assert len(geometry.missing) == 5
    assert "anode_hollow_bore_length_m" in geometry.missing
    assert "same_scope_reviewed_geometry_mask" in geometry.missing


def test_ledger_summary_reports_blocked_and_pass_counts(
    ledger: AcceptanceGateDryRunLedger,
) -> None:
    summary = ledger.summary
    assert summary["gate_count"] == 8
    assert summary["blocked_count"] == 8
    assert summary["pass_count"] == 0
    assert sorted(summary["blocked_gates"]) == sorted(EXPECTED_GATES)
    assert summary["runtime_can_support_first_principles_acceptance"] is False


def test_ledger_as_dict_is_json_serializable(
    ledger: AcceptanceGateDryRunLedger,
) -> None:
    payload = ledger.as_dict()
    text = json.dumps(payload, sort_keys=True)
    restored = json.loads(text)
    assert restored["report_only"] is True
    assert restored["promotes_acceptance"] is False
    assert restored["accepted_runtime_claim"] is False
    assert restored["can_support_first_principles_acceptance"] is False
    assert len(restored["gates"]) == 8


def test_write_ledger_json_writes_to_caller_path(
    ledger: AcceptanceGateDryRunLedger,
    tmp_path,
) -> None:
    destination = tmp_path / "nested" / "dry_run_ledger.json"
    written = write_ledger_json(ledger, destination)
    assert written == destination
    assert written.exists()
    payload = json.loads(written.read_text())
    assert payload["deck_preset"].endswith("pf1000_scholz_2001_24rod_full_energy")
    assert len(payload["gates"]) == 8


def test_dry_run_accepts_a_prebuilt_runtime_payload() -> None:
    """A blocked synthetic runtime still yields a fail-closed ledger."""

    synthetic = {
        "scientific_status": "engineering_candidate_not_validation",
        "deck": {"source": "built_in:pf1000_scholz_2001_24rod_full_energy"},
        "can_support_first_principles_acceptance": False,
        "telemetry_packets": {},
    }
    result = run_acceptance_gate_dry_run(synthetic)
    assert result.all_gates_reported
    assert result.is_fail_closed
    # With no telemetry packets at all, every gate is blocked and still names
    # the absent-packet blocker.
    for gate in result.gates:
        assert gate.status == "blocked"
        assert gate.missing


def test_blocked_gate_result_requires_named_missing() -> None:
    """A GateDryRunResult marked blocked with an empty missing list is not
    fail-closed -- the ledger contract must reject it."""

    bad = GateDryRunResult(
        gate="numerical_fidelity",
        status="blocked",
        packet_status="blocked_numerical_fidelity_packet_not_available",
        missing=(),
        next_action="x",
    )
    bad_ledger = AcceptanceGateDryRunLedger(
        deck_preset="pf1000_scholz_2001_24rod_full_energy",
        runtime_status="engineering_candidate_not_validation",
        gates=(bad,),
    )
    assert bad_ledger.is_fail_closed is False


def _synthetic_runtime(numerical_fidelity_packet: dict) -> dict:
    """A PF-1000-shaped synthetic runtime carrying one crafted gate packet.

    Only the ``numerical_fidelity`` gate is backed; every other gate sees an
    absent packet and stays blocked.  This isolates the acceptance-flag logic
    of ``_build_gate_result`` for the numerical_fidelity gate.
    """

    return {
        "scientific_status": "engineering_candidate_not_validation",
        "deck": {"source": "built_in:pf1000_scholz_2001_24rod_full_energy"},
        "can_support_first_principles_acceptance": False,
        "telemetry_packets": {"numerical_fidelity": numerical_fidelity_packet},
    }


def _numerical_fidelity_gate(
    ledger: AcceptanceGateDryRunLedger,
) -> GateDryRunResult:
    return next(
        gate for gate in ledger.gates if gate.gate == "numerical_fidelity"
    )


def test_accepted_status_missing_acceptance_flag_stays_blocked() -> None:
    """S10-A5: accepted status + empty missing, but the acceptance flag is
    ABSENT -- the gate must stay blocked, never pass."""

    # Packet has an accepted status and no missing inputs, but never sets
    # ``can_support_first_principles_acceptance``.  ``.get()`` returns None.
    packet = {"status": "accepted"}
    ledger = run_acceptance_gate_dry_run(_synthetic_runtime(packet))
    gate = _numerical_fidelity_gate(ledger)
    assert gate.status == "blocked"
    assert gate.missing
    assert ledger.is_fail_closed


def test_accepted_status_none_acceptance_flag_stays_blocked() -> None:
    """S10-A5: accepted status + empty missing, acceptance flag explicitly
    ``None`` -- the gate must stay blocked.  ``None is True`` is False."""

    packet = {
        "status": "accepted",
        "can_support_first_principles_acceptance": None,
    }
    ledger = run_acceptance_gate_dry_run(_synthetic_runtime(packet))
    gate = _numerical_fidelity_gate(ledger)
    assert gate.status == "blocked"
    assert gate.missing
    assert ledger.is_fail_closed


def test_accepted_status_false_acceptance_flag_stays_blocked() -> None:
    """S10-A5: accepted status + empty missing, acceptance flag ``False`` --
    the gate must stay blocked."""

    packet = {
        "status": "accepted",
        "can_support_first_principles_acceptance": False,
    }
    ledger = run_acceptance_gate_dry_run(_synthetic_runtime(packet))
    gate = _numerical_fidelity_gate(ledger)
    assert gate.status == "blocked"
    assert gate.missing
    assert ledger.is_fail_closed


def test_accepted_status_with_true_acceptance_flag_passes() -> None:
    """Positive control: a gate CAN pass when its backing packet is legitimately
    authorized -- accepted status, empty missing, and the acceptance flag is
    the literal ``True``.  This proves the predicate is not unconditionally
    blocking.  A passing gate does not break ``is_fail_closed`` because the
    ledger-level flags remain false."""

    packet = {
        "status": "accepted",
        "can_support_first_principles_acceptance": True,
    }
    ledger = run_acceptance_gate_dry_run(_synthetic_runtime(packet))
    gate = _numerical_fidelity_gate(ledger)
    assert gate.status == "pass"
    assert gate.missing == ()
    # The other seven gates have absent packets and stay blocked.
    assert ledger.summary["pass_count"] == 1
    assert ledger.summary["blocked_count"] == 7
    # is_fail_closed is report-only + no promoted flags, NOT "no gate passes".
    assert ledger.is_fail_closed
    assert ledger.report_only is True
    assert ledger.promotes_acceptance is False
    assert ledger.accepted_runtime_claim is False
    assert ledger.can_support_first_principles_acceptance is False
