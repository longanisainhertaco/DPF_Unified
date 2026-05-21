"""SS10-4 (A5) / SS11-1 (S10-A1) regression tests: imported-PIC startup is
context-only policy.

These tests encode the policy correction required by Super-Sprint 10 finding A5
and Super-Sprint 11 finding S10-A1:

* ``imported_pic_sheath_state`` is NOT in ``ACCEPTED_STARTUP_MODES``; it lives
  in ``CONTEXT_ONLY_STARTUP_MODES`` and must never satisfy ``mode_is_accepted``.
* A context-only mode cannot reach ``channel_acceptance_eligible=True`` at the
  payload-review layer even when the payload is fully reviewed and complete.
* A forced accepting typed startup packet path still cannot promote an
  imported-PIC payload.
* SS11-1: ``deck.py`` carries a deck-level ``CONTEXT_ONLY_STARTUP_MODES`` set
  and ``StartupPolicy.__post_init__`` unconditionally forces
  ``can_support_whole_shot_acceptance=False`` for imported PIC; a COMPLETE,
  REVIEWED imported-PIC ``FirstPrinciplesInputDeck`` converted through
  ``FirstPrinciples3DDeck.from_deck`` can never carry
  ``startup_can_support_whole_shot_acceptance=True``.

All acceptance flags must remain False (project non-negotiable).
"""

from __future__ import annotations

from dpf.first_principles.deck import (
    CONTEXT_ONLY_STARTUP_MODES as DECK_CONTEXT_ONLY_STARTUP_MODES,
)
from dpf.first_principles.deck import (
    FirstPrinciplesInputDeck,
    StartupPolicy,
)
from dpf.first_principles.runner import FirstPrinciples3DDeck
from dpf.first_principles.startup_bvp import (
    ACCEPTED_STARTUP_MODES,
    CONTEXT_ONLY_STARTUP_MODES,
    MODE_REQUIRED_PAYLOADS,
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


# ---------------------------------------------------------------------------
# Test 3 (SS11-1 / S10-A1): deck.py StartupPolicy + FirstPrinciples3DDeck
# conversion clamp imported PIC to non-promoting, unconditionally.
# ---------------------------------------------------------------------------


def _complete_reviewed_imported_pic_startup_payload() -> dict[str, object]:
    """A fully-populated, reviewed imported-PIC startup_payload.

    Every mode-required payload field for ``imported_pic_sheath_state`` is
    present as a reviewed non-None value — the most permissive caller input.
    """
    payload: dict[str, object] = {
        field: {"status": "reviewed"}
        for field in MODE_REQUIRED_PAYLOADS["imported_pic_sheath_state"]
    }
    payload["mode"] = "imported_pic_sheath_state"
    payload["evidence_status"] = "reviewed"
    return payload


def test_deck_context_only_startup_modes_mirror_startup_bvp_taxonomy() -> None:
    """deck.py must carry a deck-level CONTEXT_ONLY_STARTUP_MODES set that
    contains imported_pic_sheath_state and mirrors the startup_bvp taxonomy
    (SS11-1 / S10-A1)."""
    assert "imported_pic_sheath_state" in DECK_CONTEXT_ONLY_STARTUP_MODES
    # deck-level set and startup_bvp tuple must agree on membership.
    assert set(DECK_CONTEXT_ONLY_STARTUP_MODES) == set(CONTEXT_ONLY_STARTUP_MODES)


def test_startup_policy_reviewed_complete_imported_pic_payload_forced_nonpromoting() -> (
    None
):
    """A StartupPolicy for imported PIC with REVIEWED evidence and a COMPLETE
    payload must have can_support_whole_shot_acceptance forced to False — the
    SS11-1 fix closes the reviewed-payload->True gap (S10-A1)."""
    policy = StartupPolicy(
        mode="imported_pic_sheath_state",
        evidence_status="reviewed",
        can_support_whole_shot_acceptance=True,
        startup_payload=_complete_reviewed_imported_pic_startup_payload(),
    )
    assert policy.can_support_whole_shot_acceptance is False, (
        "imported-PIC StartupPolicy with reviewed complete payload must be "
        "forced non-promoting (SS11-1 / S10-A1)"
    )
    assert policy.whole_shot_startup_blocked is True


def test_startup_policy_accepted_same_scope_imported_pic_payload_forced_nonpromoting() -> (
    None
):
    """Even accepted_same_scope_source evidence_status cannot lift an
    imported-PIC StartupPolicy to whole-shot acceptance (SS11-1 / S10-A1)."""
    policy = StartupPolicy(
        mode="imported_pic_sheath_state",
        evidence_status="accepted_same_scope_source",
        can_support_whole_shot_acceptance=True,
        startup_payload=_complete_reviewed_imported_pic_startup_payload(),
    )
    assert policy.can_support_whole_shot_acceptance is False
    assert policy.whole_shot_startup_blocked is True


def test_imported_pic_input_deck_converts_to_nonpromoting_runtime_deck() -> None:
    """SS11-1 / S10-A1 required test: a COMPLETE, REVIEWED imported-PIC
    FirstPrinciplesInputDeck converted through FirstPrinciples3DDeck.from_deck
    yields a runtime startup policy that is context-only and non-promoting.

    A reviewed complete imported-PIC payload was the S10-A1 gap: it could be
    converted into a runtime deck with startup_can_support_whole_shot_acceptance
    =True.  The deck-level CONTEXT_ONLY_STARTUP_MODES force closes that gap.
    """
    source_sha = "a" * 64
    deck_payload: dict[str, object] = {
        "deck_id": "ss11-1-imported-pic-context-only",
        "description": "SS11-1 imported-PIC context-only conversion regression",
        "device_geometry": {
            "coordinate_system": "cartesian_3d",
            "anode_radius_m": 0.01,
            "cathode_radius_m": 0.03,
            "anode_length_m": 0.05,
            "cathode_length_m": 0.10,
            "source_reference_ids": ["pf1000_geometry"],
        },
        "circuit": {
            "capacitance_F": 1.332e-3,
            "initial_voltage_V": 16_000.0,
            "static_inductance_H": 25.0e-9,
            "static_resistance_ohm": 2.3e-3,
            "source_reference_ids": ["pf1000_geometry"],
        },
        "gas": {
            "fill_pressure_Pa": 466.6,
            "fill_temperature_K": 300.0,
            "species": [
                {
                    "name": "D2",
                    "atomic_mass_amu": 2.014,
                    "charge_state": 0.0,
                    "number_fraction": 1.0,
                    "source_reference_ids": ["pf1000_geometry"],
                }
            ],
        },
        "grid": {
            "dimensionality": "3d",
            "coordinate_system": "cartesian",
            "shape": [4, 4, 8],
            "spacing_m": [0.001, 0.001, 0.001],
            "field_layout": "staggered_yee",
        },
        # COMPLETE, REVIEWED imported-PIC startup policy with a caller-declared
        # accepting flag — the most permissive S10-A1 input.
        "startup_policy": {
            "mode": "imported_pic_sheath_state",
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "startup_payload": _complete_reviewed_imported_pic_startup_payload(),
            "source_reference_ids": ["pf1000_geometry"],
        },
        "source_references": [
            {
                "source_id": "pf1000_geometry",
                "path": "KnowledgeReference/pf1000/geometry.md",
                "sha256": source_sha,
                "title": "PF-1000 geometry source packet",
                "source_scope": "pf1000_16kv_2021_akel",
            }
        ],
    }

    input_deck = FirstPrinciplesInputDeck.from_dict(deck_payload)

    # The package input deck's StartupPolicy is already clamped.
    assert input_deck.startup.mode == "imported_pic_sheath_state"
    assert input_deck.startup.can_support_whole_shot_acceptance is False
    assert input_deck.startup.whole_shot_startup_blocked is True

    # The FirstPrinciplesInputDeck -> FirstPrinciples3DDeck conversion must
    # never carry startup_can_support_whole_shot_acceptance=True for imported
    # PIC, even from a reviewed complete payload.
    runtime_deck = FirstPrinciples3DDeck.from_deck(input_deck)
    assert runtime_deck.startup_mode == "imported_pic_sheath_state"
    assert runtime_deck.startup_can_support_whole_shot_acceptance is False, (
        "FirstPrinciples3DDeck.from_deck carried an accepting whole-shot flag "
        "for an imported-PIC deck (S10-A1 regression)"
    )

    # The runtime startup packet must be context-only and non-promoting.
    runtime_startup = runtime_deck.startup_packet()
    assert runtime_startup["whole_shot_startup_blocked"] is True
    assert runtime_startup["can_support_whole_shot_acceptance"] is False
    assert runtime_startup["can_support_first_principles_acceptance"] is False
    assert (
        runtime_startup["startup_mode_status"]["imported_pic_sheath_state"]["status"]
        == "context_only_not_an_acceptance_path"
    )
