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


# ---------------------------------------------------------------------------
# Group 7: S3.4 typed startup BVP channel packet
#
# Handoff: docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md
#          section "S3.4 Startup BVP Packet".
# Research basis: WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md -- every channel is
# candidate or blocked; 0 supported; 0 computed.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from dpf.first_principles.startup_bvp import (  # noqa: E402
    FORBIDDEN_STARTUP_INPUTS,
    STARTUP_BVP_CHANNELS,
    STARTUP_CHANNEL_STATUSES,
    StartupChannel,
    StartupPacket,
    StartupSourceRef,
    build_startup_packet,
)

# The 13 startup channels the S3.4 handoff lists as required.
S34_REQUIRED_STARTUP_CHANNELS = (
    "gas_and_fill_conditions",
    "breakdown_paschen_or_alternative",
    "preionization",
    "flashover",
    "secondary_emission",
    "photoemission",
    "surface_plasma",
    "initial_e_b_j",
    "species_and_charge_state",
    "ionization_recombination_status",
    "electron_and_ion_temperature",
    "sheath_surface_liftoff",
    "handoff_interval_into_3d_solver",
)

# The S3.4 handoff "Required packet fields" list.
S34_REQUIRED_PACKET_FIELDS = (
    "channel_id",
    "status",
    "source_refs",
    "units",
    "symbol_map",
    "input_dependencies",
    "output_fields",
    "blocker_reason",
    "first_principles_claim_effect",
)

# Repository root: tests/ -> repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_startup_packet_has_every_required_channel() -> None:
    """The typed startup packet must enumerate all 13 S3.4 channels."""
    packet = build_startup_packet()
    channel_ids = {channel.channel_id for channel in packet.channels}
    for required in S34_REQUIRED_STARTUP_CHANNELS:
        assert required in channel_ids, f"S3.4 channel '{required}' missing"
    assert len(packet.channels) == len(S34_REQUIRED_STARTUP_CHANNELS)


def test_startup_packet_channels_carry_every_required_field() -> None:
    """Every channel record carries the S3.4 required packet fields."""
    packet = build_startup_packet()
    for channel in packet.channels:
        record = channel.as_dict()
        for field_name in S34_REQUIRED_PACKET_FIELDS:
            assert field_name in record, (
                f"channel '{channel.channel_id}' missing field "
                f"'{field_name}'"
            )
        # Units and symbol map must be non-empty mappings.
        assert record["units"], f"channel '{channel.channel_id}' has no units"
        assert record["symbol_map"], (
            f"channel '{channel.channel_id}' has no symbol map"
        )
        assert record["output_fields"], (
            f"channel '{channel.channel_id}' has no output fields"
        )
        assert record["blocker_id"], (
            f"channel '{channel.channel_id}' has no blocker id"
        )


def test_no_startup_channel_is_computed_or_supported() -> None:
    """WP-N2: no channel reaches computed/supported for a DPF startup BVP.

    Every channel must be candidate or blocked. Promoting any channel to
    'computed' without a cited DPF-specific source is forbidden.
    """
    packet = build_startup_packet()
    for channel in packet.channels:
        assert channel.status in {"candidate", "blocked"}, (
            f"channel '{channel.channel_id}' has status '{channel.status}'; "
            "no channel may be computed/supported without a DPF-specific "
            "source per WP-N2"
        )
        assert channel.supports_first_principles is False
    counts = packet.status_counts()
    assert counts["computed"] == 0
    assert counts["candidate"] + counts["blocked"] == len(packet.channels)
    assert "supported" not in STARTUP_CHANNEL_STATUSES or counts.get(
        "supported", 0
    ) == 0


def test_startup_packet_blocks_first_principles_acceptance() -> None:
    """The typed startup packet must block first-principles authority."""
    packet = build_startup_packet()
    assert packet.can_support_first_principles_acceptance is False
    assert packet.status == "blocked_startup_channel_packet_no_computed_channel"
    # Every non-computed channel contributes a blocker ID.
    assert len(packet.blocker_ids) == len(packet.channels)
    assert len(set(packet.blocker_ids)) == len(packet.blocker_ids), (
        "blocker IDs must be unique"
    )
    for blocker_id in packet.blocker_ids:
        assert blocker_id.startswith("STARTUP-BVP-CH")


def test_startup_packet_dict_reports_blocked_authority() -> None:
    """The serialized packet must report blocked startup authority exactly."""
    record = build_startup_packet().as_dict()
    assert record["packet_type"] == (
        "first_principles_startup_bvp_channel_packet"
    )
    assert record["can_support_first_principles_acceptance"] is False
    assert sorted(record["channels_blocking_startup_authority"]) == sorted(
        S34_REQUIRED_STARTUP_CHANNELS
    )
    assert record["channel_status_counts"]["computed"] == 0
    assert record["requirement_ids"] == [
        "DPF-PHYS-010",
        "DPF-PHYS-017",
        "DPF-PHYS-021",
    ]


def test_blocked_channel_is_photoemission_without_local_source() -> None:
    """Photoemission is blocked: handoff says 'if sourced, otherwise blocked'."""
    packet = build_startup_packet()
    blocked = [c for c in packet.channels if c.status == "blocked"]
    assert [c.channel_id for c in blocked] == ["photoemission"]
    photoemission = packet.channels_by_id["photoemission"]
    assert photoemission.source_refs == ()
    assert photoemission.blocker_id == "STARTUP-BVP-CH06-PHOTOEMISSION-NO-LOCAL-SOURCE"


def test_startup_channel_source_refs_resolve_to_local_files() -> None:
    """Every non-blocked channel cites a real local KnowledgeReference file."""
    packet = build_startup_packet()
    for channel in packet.channels:
        if channel.status == "blocked":
            assert channel.source_refs == ()
            continue
        assert channel.source_refs, (
            f"non-blocked channel '{channel.channel_id}' has no source ref"
        )
        for ref in channel.source_refs:
            assert ref.path.startswith("KnowledgeReference/")
            assert (_REPO_ROOT / ref.path).is_file(), (
                f"channel '{channel.channel_id}' cites missing source "
                f"'{ref.path}'"
            )
            assert ref.lines, "source ref must carry a line range"
            assert ref.equation_or_figure, (
                "source ref must carry an equation/figure identifier"
            )


def test_startup_channel_rejects_invalid_status() -> None:
    """A StartupChannel with an unknown status must fail closed."""
    with pytest.raises(ValueError, match="invalid status"):
        StartupChannel(
            channel_id="bad",
            status="accepted",
            source_refs=(StartupSourceRef("KnowledgeReference/x.md", "1-2", "p"),),
            units={"x": "m"},
            symbol_map={"x": "x"},
            input_dependencies=(),
            output_fields=("x",),
            blocker_reason="",
            blocker_id="",
            first_principles_claim_effect="",
        )


def test_blocked_startup_channel_requires_blocker_id() -> None:
    """A blocked channel without a blocker ID must fail closed."""
    with pytest.raises(ValueError, match="blocker_id"):
        StartupChannel(
            channel_id="bad",
            status="blocked",
            source_refs=(),
            units={"x": "m"},
            symbol_map={"x": "x"},
            input_dependencies=(),
            output_fields=("x",),
            blocker_reason="no source",
            blocker_id="",
            first_principles_claim_effect="blocked",
        )


def test_non_blocked_startup_channel_requires_source_ref() -> None:
    """A candidate/computed channel with no source reference must fail closed."""
    with pytest.raises(ValueError, match="source reference"):
        StartupChannel(
            channel_id="bad",
            status="candidate",
            source_refs=(),
            units={"x": "m"},
            symbol_map={"x": "x"},
            input_dependencies=(),
            output_fields=("x",),
            blocker_reason="",
            blocker_id="",
            first_principles_claim_effect="candidate",
        )


def test_startup_packet_records_forbidden_inputs() -> None:
    """The packet must record the S3.4 forbidden startup inputs.

    Arbitrary seed density, back-solving an initial condition from published
    end-state results, and silent fallback to engineering defaults are all
    forbidden as accepted startup.
    """
    record = build_startup_packet().as_dict()
    forbidden = set(record["forbidden_startup_inputs"])
    assert "arbitrary_seed_density_as_accepted_startup" in forbidden
    assert (
        "back_solve_initial_condition_from_published_end_state_results"
        in forbidden
    )
    assert (
        "silent_fallback_to_engineering_defaults_in_first_principles_mode"
        in forbidden
    )
    assert forbidden == set(FORBIDDEN_STARTUP_INPUTS)


def test_startup_channel_packet_embedded_in_bvp_packet() -> None:
    """build_startup_bvp_packet must embed the typed startup channel packet."""
    bvp = build_startup_bvp_packet({"mode": "seeded_layer"})
    assert "startup_channel_packet" in bvp
    channel_packet = bvp["startup_channel_packet"]
    assert channel_packet["packet_type"] == (
        "first_principles_startup_bvp_channel_packet"
    )
    assert channel_packet["can_support_first_principles_acceptance"] is False
    assert len(channel_packet["channels"]) == len(S34_REQUIRED_STARTUP_CHANNELS)


def test_startup_channel_packet_immune_to_accepted_mode_declaration() -> None:
    """An accepted-mode + all-channels declaration must not promote the
    typed startup channel packet: it stays blocked regardless of the gate."""
    bvp = build_startup_bvp_packet(
        {
            "mode": "surface_breakdown_bvp",
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    channel_packet = bvp["startup_channel_packet"]
    assert channel_packet["status"] == (
        "blocked_startup_channel_packet_no_computed_channel"
    )
    assert channel_packet["can_support_first_principles_acceptance"] is False
    assert channel_packet["channel_status_counts"]["computed"] == 0


def test_startup_packet_is_typed_dataclass() -> None:
    """The startup packet must be a typed StartupPacket of StartupChannels."""
    packet = build_startup_packet()
    assert isinstance(packet, StartupPacket)
    assert all(isinstance(c, StartupChannel) for c in packet.channels)
    assert all(
        isinstance(ref, StartupSourceRef)
        for c in packet.channels
        for ref in c.source_refs
    )
    assert packet.channels == STARTUP_BVP_CHANNELS


# ---------------------------------------------------------------------------
# Group 8: A1 negative tests — typed packet is the single acceptance source
#
# These tests verify that the A1 finding (legacy acceptance path not bound to
# the typed StartupPacket) is closed.  The typed packet always reports
# can_support_first_principles_acceptance=False (WP-N2: all 13 channels are
# candidate or blocked).  No caller-supplied payload, declared channels, or
# review metadata may override this.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_spoof_all_channel_names_stays_blocked(mode: str) -> None:
    """Spoofing ALL REQUIRED_STARTUP_CHANNELS in an accepted-mode payload must
    not promote acceptance: the typed StartupPacket remains blocked because no
    channel has computed status (WP-N2), so status != accepted_startup_bvp_packet.
    """
    bvp = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_payload": {
                "mode": mode,
                "evidence_status": "accepted_same_scope_source",
                "source_scope": "fake_full_payload_scope",
                "can_support_whole_shot_acceptance": True,
                "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
                # Inject every mode-required payload field as a non-None value
                # so _startup_payload_review cannot block on missing fields.
                **{
                    field: {"spoofed": True}
                    for field in (
                        "mesh_mapping",
                        "particles",
                        "electron_density",
                        "ion_density",
                        "electron_temperature",
                        "ion_temperature",
                        "velocity",
                        "electric_field",
                        "magnetic_field",
                        "current_density",
                        "charge_consistency",
                        "boundary_labels",
                        "source_references",
                        "hashes",
                        "units",
                        "conservation_checks",
                        "surface_flashover_equations",
                        "secondary_emission_or_material_model",
                        "avalanche_streamer_closure",
                        "preionization_model",
                        "pressure_regime_classifier",
                        "electrode_insulator_boundary_data",
                        "verification_tests",
                    )
                },
            },
        }
    )
    # The typed packet blocks acceptance regardless of caller payload.
    assert bvp["status"] != "accepted_startup_bvp_packet", (
        f"accepted-mode '{mode}' spoof-all-channels payload promoted acceptance; "
        "typed StartupPacket must be the single acceptance authority (A1)"
    )
    assert bvp["can_support_first_principles_acceptance"] is False
    assert bvp["whole_shot_startup_blocked"] is True
    # The embedded typed packet must also report blocked.
    channel_packet = bvp["startup_channel_packet"]
    assert channel_packet["can_support_first_principles_acceptance"] is False
    assert channel_packet["status"] == (
        "blocked_startup_channel_packet_no_computed_channel"
    )


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_reviewed_evidence_without_source_hashes_stays_blocked(mode: str) -> None:
    """A payload that declares reviewed evidence_status but omits source hashes
    (i.e. no 'hashes' field) must stay blocked: reviewed status without
    source-hash verification cannot promote acceptance (A1).
    """
    bvp = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_payload": {
                "mode": mode,
                "evidence_status": "reviewed",
                "source_scope": "no_hash_scope",
                "can_support_whole_shot_acceptance": True,
                "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
                # Deliberately omit 'hashes' so payload_field_status has a gap.
                "mesh_mapping": {"present": True},
                "particles": {"present": True},
                "electron_density": {"present": True},
                "ion_density": {"present": True},
                "electron_temperature": {"present": True},
                "ion_temperature": {"present": True},
                "velocity": {"present": True},
                "electric_field": {"present": True},
                "magnetic_field": {"present": True},
                "current_density": {"present": True},
                "charge_consistency": {"present": True},
                "boundary_labels": {"present": True},
                "source_references": {"present": True},
                # hashes: intentionally absent
                "units": {"present": True},
                "conservation_checks": {"present": True},
                "surface_flashover_equations": {"present": True},
                "secondary_emission_or_material_model": {"present": True},
                "avalanche_streamer_closure": {"present": True},
                "preionization_model": {"present": True},
                "pressure_regime_classifier": {"present": True},
                "electrode_insulator_boundary_data": {"present": True},
                "verification_tests": {"present": True},
            },
        }
    )
    assert bvp["can_support_first_principles_acceptance"] is False, (
        f"mode '{mode}': reviewed evidence without source hashes promoted acceptance"
    )
    assert bvp["status"] != "accepted_startup_bvp_packet"
    assert bvp["whole_shot_startup_blocked"] is True
    # Typed packet remains blocked regardless.
    assert (
        bvp["startup_channel_packet"]["can_support_first_principles_acceptance"]
        is False
    )


def test_candidate_seeded_layer_stays_blocked_for_whole_shot_startup() -> None:
    """A seeded-layer startup with candidate-channel flags must remain blocked
    for whole-shot startup: rejected mode cannot be rescued by any channel
    declaration, and the typed packet provides an independent block (A1).
    """
    bvp = build_startup_bvp_packet(
        {
            "mode": "seeded_layer",
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "missing_channels": (),
        }
    )
    # Mode-level block (rejected class)
    assert bvp["status"] == "rejected_startup_mode_for_first_principles"
    assert bvp["startup_mode_class"] == "rejected_for_accepted_claims"
    assert bvp["whole_shot_startup_blocked"] is True
    assert bvp["can_support_whole_shot_acceptance"] is False
    assert bvp["can_support_first_principles_acceptance"] is False
    # Typed packet independent block
    assert (
        bvp["startup_channel_packet"]["can_support_first_principles_acceptance"]
        is False
    )


def test_cli_reports_typed_startup_packet_blocker() -> None:
    """The CLI first-principles-3d run must report the typed startup packet
    blocker in the output, confirming the typed packet's block is surfaced to
    the user (A1 requirement: typed packet is the visible authority).
    """
    from dpf.first_principles.runner import run_first_principles_3d_deck

    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "surface_breakdown_bvp",
            "startup_evidence_status": "accepted_same_scope_source",
            "startup_can_support_whole_shot_acceptance": True,
            "startup_accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_missing_channels": (),
        }
    )
    startup = result.telemetry["startup"]
    channel_packet = startup["startup_channel_packet"]
    # Typed packet must be present and blocked.
    assert channel_packet["packet_type"] == (
        "first_principles_startup_bvp_channel_packet"
    )
    assert channel_packet["can_support_first_principles_acceptance"] is False
    assert channel_packet["status"] == (
        "blocked_startup_channel_packet_no_computed_channel"
    )
    # The blocker IDs must be populated (one per non-computed channel).
    assert len(channel_packet["blocker_ids"]) > 0
    # The outer startup packet must be blocked too.
    assert startup["can_support_first_principles_acceptance"] is False
    assert startup["whole_shot_startup_blocked"] is True
    # Certificate gate must propagate the block.
    gate = result.telemetry["certificate_gate"]
    assert gate["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# Group 9: Sprint 4 — per-channel source-supported/blocked verdicts
#
# Source authority: KnowledgeReference/ + tracked verified extracts only.
# WP-N2 and the source-availability recon established that D2 Townsend α(E/p)
# tables, D2-specific Paschen A/B constants, DPF-material secondary emission
# yields, and start-of-shot species/Te/Ti fields are ABSENT from the local
# corpus.  Every channel must carry a definite status: candidate (corpus has
# qualitative basis; no DPF-specific closure) or blocked (fully source-empty).
# No channel is source_supported/computed.  can_support_first_principles_
# acceptance must remain False (0/13 source_supported channels).
# ---------------------------------------------------------------------------


# Expected (channel_id, blocker_id, missing_parameter_ids, status) for all 13
# S3.4 channels.
_SPRINT4_CHANNEL_VERDICTS = [
    (
        "gas_and_fill_conditions",
        "STARTUP-BVP-CH01-FILL-NO-DPF-CLOSURE",
        ("M7",),
        "candidate",
    ),
    (
        "breakdown_paschen_or_alternative",
        "STARTUP-BVP-CH02-BREAKDOWN-PASCHEN-CONTRADICTED-FOR-DPF",
        # M2 = "alpha, gamma, sigma_i0, eta, beta_ep, R_ph numerical values for
        #        D2/H2/Ne/Ar at DPF voltages" — D2 Townsend α(E/p) absent from KR
        # M1 = DPF surface-flashover BVP closure absent
        ("M1", "M2"),
        "candidate",
    ),
    (
        "preionization",
        "STARTUP-BVP-CH03-PREIONIZATION-NO-QUANTITATIVE-MODEL",
        ("M4",),
        "candidate",
    ),
    (
        "flashover",
        "STARTUP-BVP-CH04-FLASHOVER-NO-CLOSED-DELAY-MODEL",
        ("M1", "M5"),
        "candidate",
    ),
    (
        "secondary_emission",
        "STARTUP-BVP-CH05-SECONDARY-EMISSION-NO-DPF-MATERIAL-GAMMA",
        # M3 = "secondary-emission coefficients for DPF materials (Cu anode,
        #        pyrex/alumina insulator)" — absent from KR
        # M2 = alpha/gamma numerical values absent for D2 at DPF voltages
        ("M2", "M3"),
        "candidate",
    ),
    (
        "photoemission",
        "STARTUP-BVP-CH06-PHOTOEMISSION-NO-LOCAL-SOURCE",
        (),
        "blocked",
    ),
    (
        "surface_plasma",
        "STARTUP-BVP-CH07-SURFACE-PLASMA-NO-CLOSED-FIELD-SET",
        ("M1",),
        "candidate",
    ),
    (
        "initial_e_b_j",
        "STARTUP-BVP-CH08-INITIAL-FIELDS-NO-BREAKDOWN-PHASE-SET",
        ("M6",),
        "candidate",
    ),
    (
        "species_and_charge_state",
        "STARTUP-BVP-CH09-SPECIES-NO-START-OF-SHOT-FIELDS",
        # M7 = "start-of-shot density/species and Te/Ti fields" — absent from KR
        ("M7",),
        "candidate",
    ),
    (
        "ionization_recombination_status",
        "STARTUP-BVP-CH10-IONIZATION-NO-DPF-COEFFICIENTS",
        # M2 = D2/H2 alpha(E/p), beta_ep, R_ph absent from KR
        ("M2",),
        "candidate",
    ),
    (
        "electron_and_ion_temperature",
        "STARTUP-BVP-CH11-TEMPERATURE-NO-DPF-VALID-RELATION",
        # M7 = start-of-shot Te/Ti fields absent; M9 = homogeneous-field
        # relation invalid for DPF coaxial gap
        ("M7", "M9"),
        "candidate",
    ),
    (
        "sheath_surface_liftoff",
        "STARTUP-BVP-CH12-SHEATH-NO-BREAKDOWN-BVP-STATE",
        # M1 = DPF surface-flashover BVP closure absent
        # M7 = start-of-shot density fields absent
        # Note: UCSD/Beg (KR:160-205,458-500,616-670) provides
        # METHOD CONTEXT for sheath liftoff sequence and pressure regimes —
        # candidate_method_context_not_acceptance (UCSD 4.6 kJ DPF,
        # not PF-1000/Akel; no closed BVP initial/boundary data).
        ("M1", "M7"),
        "candidate",
    ),
    (
        "handoff_interval_into_3d_solver",
        "STARTUP-BVP-CH13-HANDOFF-NO-NUMERICAL-DEFINITION",
        ("M8",),
        "candidate",
    ),
]


@pytest.mark.parametrize(
    "channel_id,expected_blocker_id,expected_missing_ids,expected_status",
    _SPRINT4_CHANNEL_VERDICTS,
    ids=[v[0] for v in _SPRINT4_CHANNEL_VERDICTS],
)
def test_sprint4_channel_has_correct_blocker_id_and_missing_params(
    channel_id: str,
    expected_blocker_id: str,
    expected_missing_ids: tuple[str, ...],
    expected_status: str,
) -> None:
    """Sprint 4: every channel carries its named blocker_id and missing-param IDs.

    Source basis: KnowledgeReference/ absence verified in Sprint 4 KR recon.
    D2 Townsend α(E/p) and Paschen A/B → M2 (absent).
    DPF-material secondary emission yields → M3 (absent).
    Start-of-shot species/Te/Ti fields → M7 (absent).
    """
    packet = build_startup_packet()
    channel = packet.channels_by_id[channel_id]
    assert channel.blocker_id == expected_blocker_id, (
        f"channel '{channel_id}' has blocker_id '{channel.blocker_id}'; "
        f"expected '{expected_blocker_id}'"
    )
    assert channel.status == expected_status, (
        f"channel '{channel_id}' has status '{channel.status}'; "
        f"expected '{expected_status}'"
    )
    assert set(channel.missing_parameter_ids) == set(expected_missing_ids), (
        f"channel '{channel_id}' missing_parameter_ids "
        f"{channel.missing_parameter_ids} != expected {expected_missing_ids}"
    )
    assert channel.supports_first_principles is False


def test_sprint4_d2_townsend_absent_from_kr_corpus() -> None:
    """Sprint 4: D2 Townsend α(E/p) absent — breakdown channel carries M2 blocker.

    The KR recon confirmed no file in KnowledgeReference/ supplies a numerical
    D2-specific Townsend ionization coefficient table (α vs E/p) or Paschen
    A/B constants for deuterium.  The breakdown_paschen_or_alternative channel
    explicitly notes the corpus contradicts canonical Paschen for DPFs and
    supplies no D2 closure.
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["breakdown_paschen_or_alternative"]
    assert "M2" in ch.missing_parameter_ids, (
        "D2 Townsend α(E/p) absent: breakdown channel must list M2 "
        "(alpha/gamma numerical values for D2/H2)"
    )
    assert ch.status == "candidate"
    assert "PASCHEN" in ch.blocker_id.upper() or "BREAKDOWN" in ch.blocker_id.upper()


def test_sprint4_d2_paschen_ab_constants_absent_from_kr_corpus() -> None:
    """Sprint 4: D2 Paschen A/B constants absent — ionization channel carries M2.

    No KnowledgeReference file supplies D2-specific Paschen A and B constants.
    The ionization_recombination_status channel is blocked by M2 (alpha/gamma/
    beta_ep numerical values absent for D2).
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["ionization_recombination_status"]
    assert "M2" in ch.missing_parameter_ids, (
        "D2 Paschen A/B absent: ionization channel must list M2"
    )
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH10-IONIZATION-NO-DPF-COEFFICIENTS"


def test_sprint4_see_gamma_absent_for_dpf_materials() -> None:
    """Sprint 4: secondary emission yield γ absent for Cu/pyrex/alumina DPF materials.

    The KR recon found no file supplying ion-induced secondary electron emission
    yields for DPF electrode materials (copper anode, pyrex/alumina insulator).
    The secondary_emission channel is blocked by M3 (material-specific gamma
    absent) and M2 (generic gas-discharge values only, not DPF-specific).
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["secondary_emission"]
    assert "M3" in ch.missing_parameter_ids, (
        "SEE γ for Cu/pyrex/alumina absent: secondary_emission channel must "
        "list M3"
    )
    assert "M2" in ch.missing_parameter_ids, (
        "generic alpha/gamma values absent for D2 at DPF voltages: must list M2"
    )
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH05-SECONDARY-EMISSION-NO-DPF-MATERIAL-GAMMA"


def test_sprint4_ucsd_beg_liftoff_is_method_context_not_acceptance() -> None:
    """Sprint 4: UCSD/Beg KR provides method context for sheath liftoff — not acceptance.

    Source: KnowledgeReference/effect-of-current-sheath-initiation-on-the-
    radial-collapse-and-energetic-particle-accelera-b2e95b88.md:160-205,
    458-500,616-670 (UCSD 4.6 kJ / 20 kV / 230-260 kA DPF, not PF-1000/Akel).

    This source gives method/phenomenology targets (pressure regime boundaries,
    Liz/Li=2.4 scaling, startup sequence) and is labeled
    candidate_method_context_not_acceptance in source_targets.py.  It does NOT
    supply a closed BVP initial/boundary data set for sheath_surface_liftoff,
    so that channel remains candidate (not source_supported).
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["sheath_surface_liftoff"]
    assert ch.status == "candidate"
    assert ch.supports_first_principles is False
    assert ch.blocker_id == "STARTUP-BVP-CH12-SHEATH-NO-BREAKDOWN-BVP-STATE"
    assert "M7" in ch.missing_parameter_ids
    assert len(ch.source_refs) > 0


def test_sprint4_preionization_seed_state_absent() -> None:
    """Sprint 4: quantitative preionization seed-density model absent from KR.

    No KR file supplies a preionization seed-density / ionization-fraction
    model for the DPF start-of-discharge state (M4).  The preionization
    channel documents experimental yield deltas but has no numerical closure.
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["preionization"]
    assert "M4" in ch.missing_parameter_ids
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH03-PREIONIZATION-NO-QUANTITATIVE-MODEL"


def test_sprint4_initial_density_charge_state_absent() -> None:
    """Sprint 4: start-of-shot density/species/charge-state fields absent from KR.

    The species_and_charge_state channel cites only end-of-rundown prefill
    densities (hybrid-PIC KR Table 1); no start-of-discharge density/species/
    charge-state initial conditions are present (M7 absent).
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["species_and_charge_state"]
    assert "M7" in ch.missing_parameter_ids
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH09-SPECIES-NO-START-OF-SHOT-FIELDS"


def test_sprint4_initial_te_ti_relation_invalid_for_dpf() -> None:
    """Sprint 4: initial Te/Ti relation invalid for DPF coaxial gap (M7, M9).

    The electron_and_ion_temperature channel cites Eq. (4) Te = xi*lambda*e*U/d
    (noble gas breakdown KR) — valid only for homogeneous fields.  The DPF
    coaxial gap is inhomogeneous; the relation is not a valid DPF closure.
    Te ~ 4 eV is an analysis assumption from the UCSD/Beg source, not a
    computed DPF-specific result.
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["electron_and_ion_temperature"]
    assert "M7" in ch.missing_parameter_ids
    assert "M9" in ch.missing_parameter_ids
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH11-TEMPERATURE-NO-DPF-VALID-RELATION"


def test_sprint4_initial_ebj_no_breakdown_phase_set() -> None:
    """Sprint 4: no breakdown-phase initial E/J field distributions in KR (M6).

    The initial_e_b_j channel cites implosion-phase circuit relations (Eq. 34/35
    in hybrid-PIC KR) and a qualitative radial E-field description.  No closed
    source-derived initial E/J distribution for the breakdown phase exists.
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["initial_e_b_j"]
    assert "M6" in ch.missing_parameter_ids
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH08-INITIAL-FIELDS-NO-BREAKDOWN-PHASE-SET"


def test_sprint4_handoff_interval_no_numerical_definition() -> None:
    """Sprint 4: no numerical handoff-interval definition or PIC import payload (M8).

    The handoff_interval_into_3d_solver channel cites ALEGRA's arbitrary
    '1 eV thin layer' seed (called arbitrary by the source itself) and ALEGRA's
    PIC import capability.  No numerical handoff definition or same-device
    reviewed PIC import payload exists locally.
    """
    packet = build_startup_packet()
    ch = packet.channels_by_id["handoff_interval_into_3d_solver"]
    assert "M8" in ch.missing_parameter_ids
    assert ch.status == "candidate"
    assert ch.blocker_id == "STARTUP-BVP-CH13-HANDOFF-NO-NUMERICAL-DEFINITION"


def test_sprint4_all_channels_have_definite_status() -> None:
    """Sprint 4: every channel has a definite status (candidate or blocked).

    No channel may be in an ambiguous state.  The source-availability recon
    confirmed all 13 channels are either 'candidate' (corpus has qualitative
    basis; no DPF-specific closure) or 'blocked' (fully source-empty).
    """
    packet = build_startup_packet()
    for channel in packet.channels:
        assert channel.status in {"candidate", "blocked"}, (
            f"channel '{channel.channel_id}' has ambiguous status "
            f"'{channel.status}'"
        )
        assert channel.blocker_id, (
            f"channel '{channel.channel_id}' has no blocker_id"
        )


def test_sprint4_zero_source_supported_channels() -> None:
    """Sprint 4: acceptance status = 0/13 source_supported channels.

    The source-availability recon confirmed D2 Townsend α(E/p), D2 Paschen A/B,
    DPF-material SEE γ, preionization seed density, start-of-shot density/
    species/Te/Ti, breakdown-phase E/J fields, and numerical handoff-interval
    are all ABSENT from the local KnowledgeReference corpus.  No channel reaches
    computed/source_supported status; can_support_first_principles_acceptance
    must be False.
    """
    packet = build_startup_packet()
    source_supported = [c for c in packet.channels if c.supports_first_principles]
    assert source_supported == [], (
        f"channels unexpectedly promoted to source_supported: "
        f"{[c.channel_id for c in source_supported]}"
    )
    assert packet.can_support_first_principles_acceptance is False
    counts = packet.status_counts()
    assert counts["computed"] == 0
    assert len(packet.channels) == 13


def test_sprint4_can_support_first_principles_acceptance_is_false() -> None:
    """Sprint 4: non-negotiable — can_support_first_principles_acceptance is False.

    Rule 7 from the controlling /goal: can_support_first_principles_acceptance=
    false everywhere.  Typed packet and BVP packet must both report False.
    """
    typed = build_startup_packet()
    assert typed.can_support_first_principles_acceptance is False
    bvp = build_startup_bvp_packet({"mode": "surface_breakdown_bvp"})
    assert bvp["can_support_first_principles_acceptance"] is False
    assert bvp["startup_channel_packet"][
        "can_support_first_principles_acceptance"
    ] is False
