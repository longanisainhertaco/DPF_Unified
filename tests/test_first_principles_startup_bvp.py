"""Negative tests for the first-principles startup BVP (SSR-004 / WP-2).

These tests prove that seeded, uniform/profile, text-only, and self-declared
startup states cannot pass a first-principles acceptance gate.  They are
intentionally adversarial: Group 2 tests assert that the packet must NOT grant
acceptance on caller-declared channels alone.  These tests pass against the
FIXED startup_bvp.py (acceptance gate now requires
startup_payload_review["channel_acceptance_eligible"]).

Source basis (KnowledgeReference only):
- gribkov-2007-pf1000-jphysd-part2.md:55-80 -- DPF phase structure: insulator
  gas breakdown, kinetic surface discharge, MHD inverse pinch, microsecond
  axial acceleration.  Startup is a multi-stage breakdown problem, not a
  seeded layer.
- effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-
  particle-accelera-b2e95b88.md:616-642 -- Paschen-style pressure regimes are
  variable guidelines only; the Paschen<->DPF breakdown link is fragile.
  A CIV/Paschen scaffold cannot be promoted to an accepted startup BVP without
  local KR closure.
"""

from __future__ import annotations

import pytest

from dpf.first_principles.startup_bvp import (
    ACCEPTED_STARTUP_MODES,
    ENGINEERING_ONLY_STARTUP_MODES,
    REJECTED_STARTUP_MODES,
    REQUIRED_STARTUP_CHANNELS,
    build_startup_bvp_packet,
)

# ---------------------------------------------------------------------------
# Group 1: seeded_layer and legacy modes must fail closed (SSR-004 hard rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(REJECTED_STARTUP_MODES))
def test_rejected_startup_modes_cannot_support_acceptance(mode: str) -> None:
    """seeded_layer, uniform, and profile startup must never be accepted."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["status"] == "rejected_startup_mode_for_first_principles"
    assert packet["startup_mode_class"] == "rejected_for_accepted_claims"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_whole_shot_acceptance"] is False
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["startup_mode_status"][mode]["decision"] == "must_fail_acceptance_gate"


def test_seeded_layer_rejection_is_immune_to_declared_channels() -> None:
    """Declaring every required channel must not rescue a seeded layer."""
    packet = build_startup_bvp_packet(
        {
            "mode": "seeded_layer",
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "missing_channels": (),
        }
    )
    assert packet["status"] == "rejected_startup_mode_for_first_principles"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["negative_test_policy"]["seeded_layer_rejection_required"] is True


# ---------------------------------------------------------------------------
# Group 2: accepted modes must NOT accept on self-declaration alone.
# These catch the packet-honesty defect documented in WP-2 section (c).
# The fix (payload_acceptance_eligible in can_support) makes these pass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_without_payload_cannot_support_acceptance(
    mode: str,
) -> None:
    """Accepted mode + no startup_payload must stay blocked.

    Declaring all required channels but supplying no payload is a text-only
    acceptance attempt.  The headline status must agree with the packet's own
    startup_payload_review: if the payload is not supplied, acceptance must be
    False.
    """
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    review = packet["startup_payload_review"]
    assert review["status"] == "startup_payload_not_supplied"
    assert packet["can_support_first_principles_acceptance"] is False, (
        f"accepted-mode '{mode}' startup granted acceptance with no payload "
        "supplied; headline status contradicts startup_payload_review"
    )
    assert packet["status"] != "accepted_startup_bvp_packet"
    assert packet["whole_shot_startup_blocked"] is True


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_headline_status_matches_payload_review(
    mode: str,
) -> None:
    """Headline acceptance must never exceed payload-review eligibility."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    review = packet["startup_payload_review"]
    if not review["channel_acceptance_eligible"]:
        assert packet["can_support_first_principles_acceptance"] is False
        assert packet["status"] != "accepted_startup_bvp_packet"


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_with_incomplete_payload_stays_blocked(
    mode: str,
) -> None:
    """A partial payload (one channel) must not pass the acceptance gate."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "source_scope": "pf1000_akel_16kv_shot_12581",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_payload": {
                "mode": mode,
                "evidence_status": "reviewed",
                "source_scope": "pf1000_akel_16kv_shot_12581",
                "can_support_whole_shot_acceptance": True,
                # Only one payload channel present; the rest are missing.
                "magnetic_field": {"value": "placeholder"},
                "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            },
        }
    )
    review = packet["startup_payload_review"]
    assert review["status"] in {
        "startup_payload_incomplete",
        "startup_payload_blocked",
    }
    assert review["missing_payload_fields"], "incomplete payload not detected"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["status"] != "accepted_startup_bvp_packet"


# ---------------------------------------------------------------------------
# Group 3: unknown / undeclared modes fail closed
# ---------------------------------------------------------------------------


def test_undeclared_startup_mode_is_blocked() -> None:
    """A startup packet with no declared mode must block, not accept."""
    packet = build_startup_bvp_packet({})
    assert packet["status"] == "blocked_startup_bvp_packet_not_available"
    assert packet["startup_mode_class"] == "unknown"
    assert packet["can_support_first_principles_acceptance"] is False


def test_unknown_startup_mode_blocks_acceptance() -> None:
    """An invented mode name must be classed unknown and fail the gate."""
    packet = build_startup_bvp_packet(
        {
            "mode": "definitely_not_a_real_startup_mode",
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["startup_mode_class"] == "unknown"
    assert packet["can_support_first_principles_acceptance"] is False
    assert (
        packet["startup_mode_status"]["definitely_not_a_real_startup_mode"]["status"]
        == "unknown_startup_mode_blocks_acceptance"
    )


# ---------------------------------------------------------------------------
# Group 4: engineering-only modes cannot reach whole-shot acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(ENGINEERING_ONLY_STARTUP_MODES))
def test_engineering_only_modes_cannot_support_whole_shot(mode: str) -> None:
    """Engineering-only startup modes must never reach whole-shot acceptance."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["startup_mode_class"] == "engineering_only"
    assert packet["can_support_first_principles_acceptance"] is False
    assert (
        packet["startup_mode_status"][mode]["status"]
        == "engineering_candidate_not_whole_shot"
    )


# ---------------------------------------------------------------------------
# Group 5: CIV/Paschen breakdown audit cannot be promoted
# ---------------------------------------------------------------------------


def test_candidate_breakdown_audit_cannot_promote_startup() -> None:
    """A CIV/Paschen breakdown audit is engineering-only and must never lift
    the startup packet to acceptance, even if the caller forges the flag."""
    doctored_audit = {
        "status": "candidate_civ_paschen_breakdown_audit_engineering_only",
        "can_support_first_principles_acceptance": True,  # adversarial
        "breakdown": {"initial_ionization_fraction": 0.1},
        "liftoff": {"candidate_liftoff_delay_s": 1.0e-8},
    }
    packet = build_startup_bvp_packet(
        {
            "mode": "seeded_layer",
            "evidence_status": "reviewed",
        },
        candidate_breakdown_audit=doctored_audit,
    )
    audit = packet["candidate_breakdown_audit"]
    assert audit["can_support_first_principles_acceptance"] is False
    assert audit["can_support_whole_shot_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# Group 6: runner + certificate-gate integration
# ---------------------------------------------------------------------------


def test_runner_seeded_startup_blocks_certificate_gate() -> None:
    """A seeded-layer startup must propagate a blocking status into the
    certificate gate so no accepted certificate can be written."""
    from dpf.first_principles.runner import run_first_principles_3d_deck

    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "seeded_layer",
            "startup_evidence_status": "reviewed",
            "startup_can_support_whole_shot_acceptance": False,
            "startup_missing_channels": (),
        }
    )
    startup = result.telemetry["startup"]
    assert startup["status"] == "rejected_startup_mode_for_first_principles"
    assert startup["can_support_first_principles_acceptance"] is False
    gate = result.telemetry["certificate_gate"]
    assert (
        gate["upstream_packet_statuses"]["startup_bvp"]
        == "rejected_startup_mode_for_first_principles"
    )
    assert gate["can_support_first_principles_acceptance"] is False


def test_runner_text_declared_accepted_startup_does_not_pass_certificate() -> None:
    """Even if a deck declares an accepted startup mode with all channels but
    no payload, the certificate gate must not be accepted.  This is the
    runner-level guard for the WP-2 section (c) fail-open defect."""
    from dpf.first_principles.runner import run_first_principles_3d_deck

    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "surface_breakdown_bvp",
            "startup_evidence_status": "reviewed",
            "startup_can_support_whole_shot_acceptance": True,
            "startup_accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_missing_channels": (),
        }
    )
    startup = result.telemetry["startup"]
    assert startup["startup_payload_review"]["status"] == "startup_payload_not_supplied"
    assert startup["can_support_first_principles_acceptance"] is False, (
        "text-declared accepted startup with no payload reached acceptance"
    )
    gate = result.telemetry["certificate_gate"]
    assert gate["can_support_first_principles_acceptance"] is False
