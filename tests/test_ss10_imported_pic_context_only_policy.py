"""SS10-4 (A5) regression tests: imported-PIC startup is context-only policy.

These tests encode the policy correction required by Super-Sprint 10 finding A5:

* ``imported_pic_sheath_state`` is NOT in ``ACCEPTED_STARTUP_MODES``; it lives
  in ``CONTEXT_ONLY_STARTUP_MODES`` and must never satisfy ``mode_is_accepted``.
* A context-only mode cannot reach ``channel_acceptance_eligible=True`` at the
  payload-review layer even when the payload is fully reviewed and complete.
* A forced accepting typed startup packet path still cannot promote an
  imported-PIC payload.

All acceptance flags must remain False (project non-negotiable).
"""

from __future__ import annotations

from dpf.first_principles.startup_bvp import (
    ACCEPTED_STARTUP_MODES,
    CONTEXT_ONLY_STARTUP_MODES,
    REQUIRED_STARTUP_CHANNELS,
    build_startup_bvp_packet,
    build_startup_packet,
)

# ---------------------------------------------------------------------------
# Test 1: imported_pic_sheath_state is context-only at the mode-policy level
# ---------------------------------------------------------------------------


def test_imported_pic_sheath_state_is_in_context_only_modes_not_accepted() -> None:
    """imported_pic_sheath_state must live in CONTEXT_ONLY_STARTUP_MODES and
    must NOT appear in ACCEPTED_STARTUP_MODES (SS10-4 A5 mode-taxonomy fix)."""
    assert "imported_pic_sheath_state" in CONTEXT_ONLY_STARTUP_MODES, (
        "imported_pic_sheath_state must be in CONTEXT_ONLY_STARTUP_MODES"
    )
    assert "imported_pic_sheath_state" not in ACCEPTED_STARTUP_MODES, (
        "imported_pic_sheath_state must NOT be in ACCEPTED_STARTUP_MODES; "
        "it is context-only (SS10-4 A5)"
    )


def test_imported_pic_mode_class_is_context_only() -> None:
    """The mode-class label for imported_pic_sheath_state must be 'context_only'."""
    packet = build_startup_bvp_packet({"mode": "imported_pic_sheath_state"})
    assert packet["startup_mode_class"] == "context_only", (
        f"expected 'context_only', got '{packet['startup_mode_class']}'"
    )


def test_imported_pic_mode_status_is_context_only_not_an_acceptance_path() -> None:
    """The mode-status entry for imported_pic_sheath_state must be
    'context_only_not_an_acceptance_path' (not an accepted-mode label)."""
    packet = build_startup_bvp_packet({"mode": "imported_pic_sheath_state"})
    mode_status = packet["startup_mode_status"]["imported_pic_sheath_state"]
    assert mode_status["status"] == "context_only_not_an_acceptance_path", (
        f"got '{mode_status['status']}'; imported_pic_sheath_state must be "
        "unambiguously context-only at mode-policy level (SS10-4 A5)"
    )
    assert mode_status["mode_class"] == "context_only"
    assert mode_status["can_support_acceptance_without_complete_payload"] is False


def test_context_only_modes_tuple_exposes_imported_pic() -> None:
    """CONTEXT_ONLY_STARTUP_MODES is a non-empty tuple and contains the mode."""
    assert isinstance(CONTEXT_ONLY_STARTUP_MODES, tuple)
    assert len(CONTEXT_ONLY_STARTUP_MODES) >= 1
    assert "imported_pic_sheath_state" in CONTEXT_ONLY_STARTUP_MODES


def test_context_only_modes_listed_in_bvp_packet_output() -> None:
    """The BVP packet output dict exposes 'context_only_modes' so callers can
    inspect the taxonomy without importing the module constant."""
    packet = build_startup_bvp_packet({"mode": "imported_pic_sheath_state"})
    assert "context_only_modes" in packet
    assert "imported_pic_sheath_state" in packet["context_only_modes"]
    # accepted_modes must NOT contain imported_pic_sheath_state.
    assert "imported_pic_sheath_state" not in packet["accepted_modes"]


# ---------------------------------------------------------------------------
# Test 2: forced accepting typed-packet path cannot promote imported-PIC
# ---------------------------------------------------------------------------


def test_forced_accepting_typed_packet_path_cannot_promote_imported_pic() -> None:
    """Regression: even when the typed startup packet were to flip to accepting
    (future sprint), an imported-PIC payload must still not promote because
    ``mode_is_accepted`` is False for a context-only mode.

    This test simulates the 'forced accepting typed packet' path by passing a
    fully-populated, reviewed, all-channels-declared imported-PIC payload — the
    most permissive possible caller input.  The acceptance gate must still reject
    because mode_is_accepted=False (imported_pic_sheath_state not in
    ACCEPTED_STARTUP_MODES) and channel_acceptance_eligible=False
    (context-only mode, never eligible at payload-review layer).
    """
    full_pic_payload = {
        "mode": "imported_pic_sheath_state",
        "evidence_status": "reviewed",
        "source_scope": "same_scope_pic_import_fixture",
        "can_support_whole_shot_acceptance": True,
        "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        # All mode-required payload fields present.
        "mesh_mapping": {"status": "reviewed"},
        "particles": {"status": "reviewed"},
        "electron_density": {"units": "m^-3"},
        "ion_density": {"units": "m^-3"},
        "electron_temperature": {"units": "K"},
        "ion_temperature": {"units": "K"},
        "velocity": {"units": "m/s"},
        "electric_field": {"units": "V/m"},
        "magnetic_field": {"units": "T"},
        "current_density": {"units": "A/m^2"},
        "charge_consistency": {"max_residual": 0.0},
        "boundary_labels": {"status": "reviewed"},
        "source_references": [{"path": "KnowledgeReference/pic-import.md"}],
        "hashes": {"payload": "sha256:test"},
        "units": {"system": "SI"},
        "conservation_checks": {"status": "reviewed"},
    }

    packet = build_startup_bvp_packet(
        {
            "mode": "imported_pic_sheath_state",
            "evidence_status": "reviewed",
            "source_scope": "same_scope_pic_import_fixture",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_payload": full_pic_payload,
        }
    )

    # --- Headline packet ---
    assert packet["status"] == "blocked_startup_bvp_packet_not_available", (
        f"imported-PIC packet must be blocked, got '{packet['status']}'"
    )
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["can_support_first_principles_acceptance"] is False, (
        "can_support_first_principles_acceptance must be False for context-only mode"
    )
    assert packet["can_support_whole_shot_acceptance"] is False

    # --- Payload review: context-only mode blocks channel_acceptance_eligible ---
    review = packet["startup_payload_review"]
    assert review["channel_acceptance_eligible"] is False, (
        "context-only mode must never be channel_acceptance_eligible at the "
        "payload-review layer (SS10-4 A5)"
    )
    assert review["status"] == "startup_payload_for_context_only_mode_not_promoting"
    assert review["can_support_first_principles_acceptance"] is False
    assert review["can_support_whole_shot_acceptance"] is False

    # --- Typed StartupPacket: independent block (WP-N2 / A1) ---
    channel_packet = packet["startup_channel_packet"]
    assert channel_packet["can_support_first_principles_acceptance"] is False
    assert channel_packet["status"] == (
        "blocked_startup_channel_packet_no_computed_channel"
    )

    # --- Mode taxonomy: context-only, not accepted ---
    assert packet["startup_mode_class"] == "context_only"
    mode_status = packet["startup_mode_status"]["imported_pic_sheath_state"]
    assert mode_status["status"] == "context_only_not_an_acceptance_path"
    assert mode_status["can_support_acceptance_without_complete_payload"] is False

    # --- Typed packet itself is still blocked (WP-N2) ---
    typed = build_startup_packet()
    assert typed.can_support_first_principles_acceptance is False
