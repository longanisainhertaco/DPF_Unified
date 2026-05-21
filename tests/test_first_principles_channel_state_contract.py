"""Sprint 8 WS1 contract tests: the shared 7-state acceptance channel vocabulary.

These tests pin down Codex audit findings S7-A7 (acceptance channel internals
are contradictory) and S7-A8 (manual same-scope channel injection can mislead):

- a claimed channel can never be BOTH ``accepted`` and missing;
- ``excluded_not_validated`` channels never count as comparator evidence;
- candidate / manual evidence never unlocks acceptance;
- runner, CLI, manifest, and certificate packets agree on the channel-state
  vocabulary.

They assert only fail-closed contract fields and invent no tolerance and no
acceptance threshold.  If any test starts failing, a packet has gained an
accepting path and must be re-audited.
"""

from __future__ import annotations

from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.channel_state import (
    CHANNEL_STATE_VALUES,
    ChannelState,
    all_states_canonical,
    channel_state_summary,
    counts_as_comparator_evidence,
    is_accepted,
)
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.runner import run_first_principles_3d_deck
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.validation.first_principles_mhd import (
    PACKAGE_NATIVE_3D_RUN_MODE,
    is_package_native_3d_result,
)

PF1000_SCOPE = "pf1000_akel_16kv_1p2torr_shot_12581"
PF1000_DEVICE = "PF-1000/Akel"

_SEVEN_STATES = (
    "accepted",
    "blocked_missing_source",
    "blocked_wrong_scope",
    "blocked_missing_review",
    "blocked_missing_uncertainty",
    "excluded_not_validated",
    "not_claimed",
)


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------


def test_exactly_seven_canonical_channel_states_defined() -> None:
    """The canonical enum defines exactly the seven required states."""
    assert tuple(CHANNEL_STATE_VALUES) == _SEVEN_STATES
    assert {s.value for s in ChannelState} == set(_SEVEN_STATES)
    assert len(ChannelState) == 7


def test_only_accepted_state_counts_as_acceptance_or_comparator_evidence() -> None:
    """Only ``accepted`` may count toward acceptance / comparator evidence."""
    for state in ChannelState:
        accepted = state is ChannelState.ACCEPTED
        assert is_accepted(state) is accepted
        assert counts_as_comparator_evidence(state) is accepted


def test_excluded_not_validated_never_counts_as_comparator_evidence() -> None:
    """Exit criterion: excluded channels never count as comparator evidence."""
    assert counts_as_comparator_evidence(ChannelState.EXCLUDED_NOT_VALIDATED) is False
    assert counts_as_comparator_evidence(ChannelState.NOT_CLAIMED) is False


# ---------------------------------------------------------------------------
# S7-A7: a channel can never be both accepted and missing
# ---------------------------------------------------------------------------


def test_channel_state_summary_has_no_accepted_and_missing_overlap() -> None:
    """A channel in any state appears in exactly one of accepted / missing."""
    states = {
        "a": ChannelState.ACCEPTED,
        "b": ChannelState.BLOCKED_MISSING_SOURCE,
        "c": ChannelState.EXCLUDED_NOT_VALIDATED,
        "d": ChannelState.NOT_CLAIMED,
    }
    summary = channel_state_summary(states)
    accepted = set(summary["accepted_channels"])
    missing = set(summary["missing_acceptance_channels"])
    assert accepted.isdisjoint(missing), "accepted and missing channels overlap"
    assert accepted == {"a"}
    assert missing == {"b", "c", "d"}
    assert summary["contradictions"] == []


def test_same_scope_packet_no_accepted_missing_contradiction() -> None:
    """S7-A7: the same-scope packet never lists a channel as accepted+missing."""
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    accepted = set(packet["accepted_same_scope_channels"])
    missing = set(packet["missing_acceptance_channels"])
    assert accepted.isdisjoint(missing)
    assert packet["channel_state_summary"]["contradictions"] == []
    # Every published per-channel state is canonical.
    assert all_states_canonical(packet["channel_states"].values())


def test_numerical_fidelity_packet_no_accepted_missing_contradiction() -> None:
    """S7-A7: numerical-fidelity packet has no accepted+missing channel."""
    all_required = list(
        build_numerical_fidelity_packet(
            declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE
        )["required_channels"]
    )
    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        accepted_channels=all_required,
    )
    accepted = set(packet["accepted_channels"])
    missing = set(packet["missing_acceptance_channels"])
    assert accepted.isdisjoint(missing)
    # With every channel declared accepted, missing must be empty (the old
    # code unconditionally re-added all channels to missing -- the S7-A7 bug).
    assert missing == set()
    assert accepted == set(all_required)
    assert all_states_canonical(packet["channel_states"].values())
    # Top-level acceptance still hard-blocked regardless.
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_certificate_packet_no_accepted_missing_contradiction() -> None:
    """S7-A7: certificate packet has no accepted+missing channel."""
    all_required = list(
        build_first_principles_certificate_gate_packet(
            declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE
        )["required_channels"]
    )
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        accepted_channels=all_required,
    )
    accepted = set(packet["accepted_channels"])
    missing = set(packet["missing_acceptance_channels"])
    assert accepted.isdisjoint(missing)
    assert missing == set()
    assert all_states_canonical(packet["channel_states"].values())
    # The certificate still cannot be written -- candidate/fixture content
    # never unlocks acceptance.
    assert packet["can_write_accepted_certificate"] is False
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# S7-A8: manual same-scope channel injection is a request, not evidence
# ---------------------------------------------------------------------------


def test_manual_non_te_ti_channel_is_requested_not_accepted() -> None:
    """S7-A8: a manual non-Te/Ti channel must NOT become accepted evidence.

    Before the fix a manual entry in ``accepted_same_scope_channels`` showed
    up as ``accepted_same_scope``.  It must now be a *requested* channel,
    carry the ``excluded_not_validated`` state, and stay out of
    ``accepted_same_scope_channels``.
    """
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(),
        accepted_same_scope_channels=("neutron_scalar_yield", "em_field_history"),
    )
    accepted = set(packet["accepted_same_scope_channels"])
    assert "neutron_scalar_yield" not in accepted
    assert "em_field_history" not in accepted
    assert set(packet["requested_manual_channels"]) == {
        "neutron_scalar_yield",
        "em_field_history",
    }
    assert set(packet["requested_manual_channels_not_evidence"]) == {
        "neutron_scalar_yield",
        "em_field_history",
    }
    for channel in ("neutron_scalar_yield", "em_field_history"):
        assert packet["channel_states"][channel] == "excluded_not_validated"
    # Manual decisions record the demotion explicitly.
    manual = [
        d
        for d in packet["validation_target_scope_decisions"]
        if d.get("status") == "manual_requested_same_scope_channel"
    ]
    assert len(manual) == 2
    for decision in manual:
        assert decision["decision"] == "requested_manual_channel_not_evidence"
        assert decision["backed_by_reviewed_target"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_manual_channel_cannot_unlock_acceptance_without_reviewed_target() -> None:
    """S7-A8: a manual channel list cannot flip acceptance.

    Injecting every required same-scope channel manually must leave the packet
    blocked -- a manual list is not a reviewed target with uncertainty.
    """
    from dpf.first_principles.same_scope import REQUIRED_SAME_SCOPE_CHANNELS

    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(),
        accepted_same_scope_channels=tuple(REQUIRED_SAME_SCOPE_CHANNELS),
    )
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert packet["can_support_first_principles_acceptance"] is False
    # No required channel becomes accepted purely by manual injection.
    accepted = set(packet["accepted_same_scope_channels"])
    for channel in REQUIRED_SAME_SCOPE_CHANNELS:
        if channel == "declared_validation_scope":
            continue  # scope flag, not an evidence channel
        assert channel not in accepted, (
            f"manual injection unlocked {channel} without a reviewed target"
        )


def test_synthetic_fixture_flag_does_not_promote_manual_channels() -> None:
    """A synthetic_fixture marker never promotes a manual channel to accepted."""
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(),
        accepted_same_scope_channels=("neutron_scalar_yield",),
        synthetic_fixture=True,
    )
    assert packet["synthetic_fixture"] is True
    assert "neutron_scalar_yield" not in packet["accepted_same_scope_channels"]
    assert packet["channel_states"]["neutron_scalar_yield"] == "excluded_not_validated"
    assert packet["can_support_first_principles_acceptance"] is False


def test_production_runner_deck_is_not_a_synthetic_fixture() -> None:
    """A production runner deck must never produce a synthetic-fixture packet."""
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "device_name": PF1000_DEVICE,
            "validation_scope": PF1000_SCOPE,
            "apply_circuit_boundary": False,
        }
    )
    packet = result.telemetry["same_scope_source"]
    assert packet["synthetic_fixture"] is False


# ---------------------------------------------------------------------------
# 3-D gate separation: package-native 3-D never judged by cylindrical gate
# ---------------------------------------------------------------------------


def test_is_package_native_3d_result_detects_3d_runs() -> None:
    """A package-native 3-D run is recognized by run_mode or dimensionality."""
    assert is_package_native_3d_result({"run_mode": PACKAGE_NATIVE_3D_RUN_MODE})
    assert is_package_native_3d_result({"geometry_dimensionality": "cartesian_3d"})
    assert is_package_native_3d_result({"dimensionality": "3d"})
    # A cylindrical MHD run is not a package-native 3-D run.
    assert not is_package_native_3d_result(
        {"run_mode": "first_principles_mhd", "geometry_dimensionality": "cylindrical"}
    )
    assert not is_package_native_3d_result({})


def test_cylindrical_gate_defers_package_native_3d_run() -> None:
    """The legacy cylindrical gate must defer a 3-D run, not score it.

    A package-native 3-D result must NOT be silently accepted or rejected by
    cylindrical key expectations.  The legacy gate returns ``ready=False`` with
    an explicit deferral blocker pointing at the hybrid_pic_3d gate.
    """
    from dpf.validation.first_principles_mhd import (
        first_principles_mhd_readiness_report,
    )

    three_d_result = {
        "run_mode": PACKAGE_NATIVE_3D_RUN_MODE,
        "geometry_dimensionality": "cartesian_3d",
        "status": "engineering_candidate_not_validation",
        # Deliberately no cylindrical keys (I_MA, rho, sheath_position).
    }
    readiness = first_principles_mhd_readiness_report(three_d_result)
    assert readiness.ready is False
    assert readiness.status == "blocked_package_native_3d_run_uses_hybrid_pic_3d_gate"
    # The single blocker is the gate-separation deferral, not a cylindrical
    # "missing output" complaint.
    assert len(readiness.blockers) == 1
    assert "package_native_3d_run_detected" in readiness.blockers[0]
    assert "hybrid_pic_3d" in readiness.blockers[0]
    # No cylindrical output was scored as satisfied or missing.
    assert readiness.output_status == {}
    assert readiness.satisfied_evidence == []
    # The deferral surfaces the authoritative 3-D readiness packet.
    assert readiness.hybrid_pic_3d_status.get("status") in {"blocked", "accepted"}


def test_cylindrical_gate_still_judges_cylindrical_runs() -> None:
    """A cylindrical run is still judged by the cylindrical gate (no regression)."""
    from dpf.validation.first_principles_mhd import (
        first_principles_mhd_readiness_report,
    )

    cylindrical_result = {
        "run_mode": "first_principles_mhd",
        "geometry_dimensionality": "cylindrical",
    }
    readiness = first_principles_mhd_readiness_report(cylindrical_result)
    # It is blocked (missing everything) but NOT via the 3-D deferral path.
    assert readiness.status == "blocked"
    assert readiness.output_status != {}


# ---------------------------------------------------------------------------
# Contract: runner / CLI / manifest / certificate agree on the vocabulary
# ---------------------------------------------------------------------------


def test_runner_packets_use_canonical_channel_state_vocabulary() -> None:
    """Runner same-scope / numerical / certificate packets use the 7-state set."""
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "device_name": PF1000_DEVICE,
            "validation_scope": PF1000_SCOPE,
            "apply_circuit_boundary": False,
        }
    )
    for packet_name in ("same_scope_source", "numerical_fidelity", "certificate_gate"):
        packet = result.telemetry[packet_name]
        assert "channel_states" in packet, f"{packet_name} missing channel_states"
        states = packet["channel_states"]
        assert states, f"{packet_name} channel_states is empty"
        assert all_states_canonical(states.values()), (
            f"{packet_name} emitted a non-canonical channel state"
        )
        # The summary's missing list and the packet's missing list agree.
        summary = packet["channel_state_summary"]
        assert set(summary["missing_acceptance_channels"]) == set(
            packet["missing_acceptance_channels"]
        )
        # accepted vs missing are disjoint everywhere.
        assert set(summary["accepted_channels"]).isdisjoint(
            set(summary["missing_acceptance_channels"])
        )


def test_runner_3d_gate_is_hybrid_pic_not_cylindrical() -> None:
    """The package-native runner's 3-D gate is hybrid_pic_3d_readiness."""
    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "device_name": PF1000_DEVICE,
            "validation_scope": PF1000_SCOPE,
            "apply_circuit_boundary": False,
        }
    )
    readiness = result.telemetry["hybrid_pic_3d_readiness"]
    assert readiness["source"].endswith(".md")
    assert "hybrid-pic-fluid" in readiness["source"]
    assert readiness["can_support_first_principles_acceptance"] is False
    # The package-native run is not annotated by the legacy cylindrical gate.
    assert "first_principles_mhd_readiness" not in result.telemetry


def test_candidate_evidence_never_unlocks_acceptance() -> None:
    """Exit criterion: candidate runtime evidence never unlocks acceptance."""
    result = run_first_principles_3d_deck(
        {
            "n_steps": 2,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "device_name": PF1000_DEVICE,
            "validation_scope": PF1000_SCOPE,
            "apply_circuit_boundary": False,
        }
    )
    # Top-level run flag stays false.
    assert result.telemetry["can_support_first_principles_acceptance"] is False
    # Every channel-state-bearing packet stays non-accepting.
    for packet_name in ("same_scope_source", "numerical_fidelity", "certificate_gate"):
        packet = result.telemetry[packet_name]
        assert packet["can_support_first_principles_acceptance"] is False
        summary = packet["channel_state_summary"]
        # No claimed channel was promoted by candidate telemetry.
        assert summary["all_claimed_channels_accepted"] is False
