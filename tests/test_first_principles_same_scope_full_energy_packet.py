"""Super-Sprint 9 WS9-3: same-scope full-energy packet repair (Codex P1-1).

Codex P1-1: ``_looks_like_pf1000_akel_scope()`` matched any ``pf1000`` /
``pf-1000`` / ``akel`` string, so the full-energy scope
``pf1000_full_energy_27_to_40_kv`` wrongly received Akel text-supported
channels and an Akel-named acceptance-gate label.  Akel is the 16 kV /
shot-12581 PF-1000 revision -- a separate scope.

The WS9-3 follow-up found the same buggy helper duplicated as independent
copies in five packet builders.  They are now replaced by one shared
exact-match classifier ``looks_like_pf1000_akel_16kv_scope`` in
``dpf.first_principles.channel_state`` (the already-vetted non-promoting
packet-vocabulary module).

These tests pin the repaired behaviour:

- the full-energy scope receives NO Akel reference channels and NO
  Akel-named gate label (full-energy text-supported set is empty /
  selected-scope-only until KR target-extraction supplies records) -- for
  the same-scope, waveform-phase, spatial-field-temperature, comparator-UQ,
  and neutron-authority packets;
- the Akel 16 kV scope still receives its Akel text-supported channels as
  non-acceptance (``excluded_not_validated``) evidence in every packet;
- ``check_scope_consistency()`` still rejects mixed full-energy + Akel
  source sets;
- the shared classifier is the single source of truth -- the five builders
  no longer define their own ``_looks_like_pf1000_akel*`` helper.

They assert only fail-closed contract fields and invent no tolerance and no
acceptance threshold.  Every packet must keep
``can_support_first_principles_acceptance=False``.
"""

from __future__ import annotations

from dpf.first_principles.runtime_demonstrator_scope import (
    SELECTED_SCOPE_LABEL,
    check_scope_consistency,
)
from dpf.first_principles.same_scope import (
    PF1000_AKEL_TEXT_SUPPORTED_CHANNELS,
    build_same_scope_source_packet,
)

FULL_ENERGY_SCOPE = "pf1000_full_energy_27_to_40_kv"
FULL_ENERGY_DEVICE = "PF-1000 full energy"

AKEL_16KV_SCOPE = "pf1000_akel_16kv_1p2torr_shot_12581"
AKEL_16KV_DEVICE = "PF-1000/Akel"

# Akel-named reference channels that must never appear in a full-energy packet.
_AKEL_ONLY_CHANNELS = ("pinch_geometry_lee_output", "timing_uncertainty_text")


# ---------------------------------------------------------------------------
# Full-energy scope receives no Akel evidence
# ---------------------------------------------------------------------------


def test_full_energy_scope_label_is_the_selected_scope() -> None:
    """The repaired test scope string is exactly the selected runtime scope."""
    assert FULL_ENERGY_SCOPE == SELECTED_SCOPE_LABEL
    assert SELECTED_SCOPE_LABEL == "pf1000_full_energy_27_to_40_kv"


def test_full_energy_packet_receives_no_akel_reference_channels() -> None:
    """P1-1: the full-energy scope must carry no Akel text-supported channels.

    The full-energy text-supported set is empty / selected-scope-only until KR
    target-extraction supplies selected-scope records -- so no Akel-named
    reference channel may appear.
    """
    packet = build_same_scope_source_packet(
        declared_scope=FULL_ENERGY_SCOPE,
        device_name=FULL_ENERGY_DEVICE,
    )
    text_supported = packet["text_supported_reference_channels"]
    # Full-energy text-supported set is currently empty (fail-closed).
    assert text_supported == [], (
        "full-energy scope received text-supported reference channels; the "
        "set must stay empty until KR supplies selected-scope records"
    )
    # No individual Akel channel leaked through.
    for channel in PF1000_AKEL_TEXT_SUPPORTED_CHANNELS:
        assert channel not in text_supported, (
            f"Akel channel {channel} leaked into the full-energy packet"
        )
    for channel in _AKEL_ONLY_CHANNELS:
        assert channel not in text_supported
        assert channel not in packet["text_supported_not_acceptance_channels"]
    # The packet is still blocked and non-promoting.
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert packet["can_support_first_principles_acceptance"] is False


def test_full_energy_packet_has_no_akel_named_gate_label() -> None:
    """P1-1: the full-energy acceptance gate must not be Akel-named.

    The Akel-named gate label begins with
    ``text_supported_pf1000_akel_scalars_...``.  A full-energy packet must not
    carry an Akel-scoped gate label.
    """
    packet = build_same_scope_source_packet(
        declared_scope=FULL_ENERGY_SCOPE,
        device_name=FULL_ENERGY_DEVICE,
    )
    gate = packet["acceptance_gate"]
    assert "akel" not in gate.lower(), (
        f"full-energy acceptance gate carries an Akel-named label: {gate!r}"
    )


def test_full_energy_packet_channel_states_carry_no_excluded_text_reference() -> None:
    """Without Akel channels, no full-energy channel is excluded as text-only.

    A text-supported reference channel maps to ``excluded_not_validated``.  With
    the full-energy text-supported set empty, the only ``excluded_not_validated``
    channels may come from manual requests -- and none are passed here.
    """
    packet = build_same_scope_source_packet(
        declared_scope=FULL_ENERGY_SCOPE,
        device_name=FULL_ENERGY_DEVICE,
    )
    states = packet["channel_states"]
    excluded = {ch for ch, st in states.items() if st == "excluded_not_validated"}
    assert excluded == set(), (
        f"full-energy packet excluded channels as text-only references: {excluded}"
    )
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# Akel 16 kV scope still receives Akel evidence (non-acceptance)
# ---------------------------------------------------------------------------


def test_akel_16kv_scope_still_receives_akel_text_supported_channels() -> None:
    """The Akel 16 kV / shot-12581 scope keeps its Akel text-supported channels.

    The repair narrowed the helper to the Akel revision only; it must not
    remove Akel evidence from the Akel scope itself.  The channels are
    non-acceptance: text-supported and ``excluded_not_validated``.
    """
    packet = build_same_scope_source_packet(
        declared_scope=AKEL_16KV_SCOPE,
        device_name=AKEL_16KV_DEVICE,
    )
    text_supported = set(packet["text_supported_reference_channels"])
    assert text_supported == set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
    for channel in _AKEL_ONLY_CHANNELS:
        assert channel in text_supported
    # Text-supported channels are non-acceptance evidence.
    assert set(packet["text_supported_not_acceptance_channels"]) == set(
        PF1000_AKEL_TEXT_SUPPORTED_CHANNELS
    )
    for channel in PF1000_AKEL_TEXT_SUPPORTED_CHANNELS:
        assert packet["channel_states"].get(channel) in (
            "excluded_not_validated",
            None,
        )
        if channel in packet["channel_states"]:
            assert packet["channel_states"][channel] == "excluded_not_validated"
    # Still blocked and non-promoting.
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert packet["can_support_first_principles_acceptance"] is False


def test_akel_16kv_scope_keeps_akel_named_gate_label() -> None:
    """The Akel 16 kV scope keeps its Akel-named acceptance gate label."""
    packet = build_same_scope_source_packet(
        declared_scope=AKEL_16KV_SCOPE,
        device_name=AKEL_16KV_DEVICE,
    )
    assert packet["acceptance_gate"].startswith(
        "text_supported_pf1000_akel_scalars_and_cross_scope_diagnostics"
    )


def test_akel_helper_does_not_match_full_energy_or_bare_pf1000() -> None:
    """P1-1 direct: the exact helper matches only the Akel 16 kV revision.

    A bare ``pf1000`` string (no Akel / 16 kV / shot-12581 marker) must not be
    classified as the Akel scope.
    """
    from dpf.first_principles.channel_state import looks_like_pf1000_akel_16kv_scope

    # Akel revision -- matches.
    assert looks_like_pf1000_akel_16kv_scope(AKEL_16KV_SCOPE, AKEL_16KV_DEVICE)
    assert looks_like_pf1000_akel_16kv_scope(
        "pf1000_akel_16kv_1p2torr_deuterium_shot_12581", None
    )
    assert looks_like_pf1000_akel_16kv_scope(
        "PF1000 Akel shot 12581 reference candidate", None
    )
    # Full-energy scope -- must NOT match.
    assert not looks_like_pf1000_akel_16kv_scope(FULL_ENERGY_SCOPE, FULL_ENERGY_DEVICE)
    assert not looks_like_pf1000_akel_16kv_scope(SELECTED_SCOPE_LABEL, None)
    # Bare PF-1000 strings with no Akel marker -- must NOT match.
    assert not looks_like_pf1000_akel_16kv_scope("pf1000", None)
    assert not looks_like_pf1000_akel_16kv_scope("pf-1000", "PF-1000")
    # "akel" alone, with no 16 kV / shot marker -- must NOT match (fail-closed).
    assert not looks_like_pf1000_akel_16kv_scope("pf1000_akel_generic", None)


# ---------------------------------------------------------------------------
# Mixed full-energy + Akel source sets are still rejected
# ---------------------------------------------------------------------------


def test_check_scope_consistency_rejects_mixed_full_energy_and_akel() -> None:
    """check_scope_consistency must reject an in-scope + Akel wrong-scope mix.

    ``akel_2021_pf1000_neutron_yield_16kv`` is registered as a wrong-scope
    source for the full-energy demonstrator; mixing it with an in-scope
    full-energy source must fail the scope-consistency check.
    """
    result = check_scope_consistency(
        [
            "scholz_gribkov_2007_partii",  # in-scope full-energy source
            "akel_2021_pf1000_neutron_yield_16kv",  # Akel 16 kV wrong-scope
        ]
    )
    assert result["consistent"] is False
    assert result["failure_reason"] == (
        "mixed_in_scope_and_wrong_scope_without_transfer_rule"
    )
    assert "akel_2021_pf1000_neutron_yield_16kv" in result["wrong_scope_found"]
    assert "scholz_gribkov_2007_partii" in result["in_scope_found"]


def test_check_scope_consistency_accepts_pure_full_energy_set() -> None:
    """A pure full-energy in-scope set stays consistent (control)."""
    result = check_scope_consistency(
        ["scholz_gribkov_2007_partii", "malir_2024_interferometry_dpf"]
    )
    assert result["consistent"] is True
    assert result["wrong_scope_found"] == []
    assert result["selected_scope_label"] == SELECTED_SCOPE_LABEL


# ---------------------------------------------------------------------------
# WS9-3 follow-up: the four sibling packet builders share the exact helper
# ---------------------------------------------------------------------------
#
# The same P1-1 helper bug was duplicated in waveform_phase, comparator_uq,
# spatial_field_temperature, and neutron_authority.  Each builder must now give
# the full-energy scope NO Akel text-supported channels, while the Akel 16 kV
# scope keeps its Akel channels as non-acceptance evidence.


def _sibling_full_energy_packet(build_fn: object) -> dict:
    return build_fn(  # type: ignore[operator]
        declared_scope=FULL_ENERGY_SCOPE,
        device_name=FULL_ENERGY_DEVICE,
    )


def _sibling_akel_packet(build_fn: object) -> dict:
    return build_fn(  # type: ignore[operator]
        declared_scope=AKEL_16KV_SCOPE,
        device_name=AKEL_16KV_DEVICE,
    )


def test_waveform_phase_full_energy_packet_receives_no_akel_channels() -> None:
    """waveform_phase: full-energy scope gets no Akel text-supported channels."""
    from dpf.first_principles.waveform_phase import build_waveform_phase_packet

    packet = _sibling_full_energy_packet(build_waveform_phase_packet)
    assert packet["text_supported_reference_channels"] == []
    assert packet["text_supported_not_acceptance_channels"] == []
    assert "akel" not in packet["acceptance_gate"].lower()
    assert packet["can_support_first_principles_acceptance"] is False


def test_waveform_phase_akel_16kv_packet_keeps_akel_channels() -> None:
    """waveform_phase: Akel 16 kV scope keeps Akel channels as non-acceptance."""
    from dpf.first_principles.waveform_phase import (
        PF1000_AKEL_TEXT_SUPPORTED_WAVEFORM_PHASE_CHANNELS,
        build_waveform_phase_packet,
    )

    packet = _sibling_akel_packet(build_waveform_phase_packet)
    assert set(packet["text_supported_reference_channels"]) == set(
        PF1000_AKEL_TEXT_SUPPORTED_WAVEFORM_PHASE_CHANNELS
    )
    # Text-supported channels are non-acceptance evidence.
    assert set(packet["text_supported_not_acceptance_channels"]) == set(
        PF1000_AKEL_TEXT_SUPPORTED_WAVEFORM_PHASE_CHANNELS
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_spatial_field_temperature_full_energy_packet_no_akel_channels() -> None:
    """spatial_field_temperature: full-energy scope gets no Akel channels."""
    from dpf.first_principles.spatial_field_temperature import (
        build_spatial_field_temperature_packet,
    )

    packet = _sibling_full_energy_packet(build_spatial_field_temperature_packet)
    assert packet["text_supported_reference_channels"] == []
    assert packet["text_supported_not_acceptance_channels"] == []
    assert "akel" not in packet["acceptance_gate"].lower()
    assert packet["can_support_first_principles_acceptance"] is False


def test_spatial_field_temperature_akel_16kv_packet_keeps_akel_channels() -> None:
    """spatial_field_temperature: Akel 16 kV scope keeps its Akel channels."""
    from dpf.first_principles.spatial_field_temperature import (
        PF1000_AKEL_TEXT_SUPPORTED_CHANNELS as SFT_AKEL_CHANNELS,
    )
    from dpf.first_principles.spatial_field_temperature import (
        build_spatial_field_temperature_packet,
    )

    packet = _sibling_akel_packet(build_spatial_field_temperature_packet)
    assert set(packet["text_supported_reference_channels"]) == set(SFT_AKEL_CHANNELS)
    assert set(packet["text_supported_not_acceptance_channels"]) == set(
        SFT_AKEL_CHANNELS
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_comparator_uq_full_energy_packet_receives_no_akel_channels() -> None:
    """comparator_uq: full-energy scope gets no Akel text-supported channels."""
    from dpf.first_principles.comparator_uq import build_comparator_uq_packet

    packet = _sibling_full_energy_packet(build_comparator_uq_packet)
    assert packet["text_supported_reference_channels"] == []
    assert packet["text_supported_not_acceptance_channels"] == []
    assert "akel" not in packet["acceptance_gate"].lower()
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_comparator_acceptance"] is False


def test_comparator_uq_akel_16kv_packet_keeps_akel_channels() -> None:
    """comparator_uq: Akel 16 kV scope keeps its Akel channels."""
    from dpf.first_principles.comparator_uq import (
        PF1000_AKEL_TEXT_SUPPORTED_CHANNELS as CUQ_AKEL_CHANNELS,
    )
    from dpf.first_principles.comparator_uq import build_comparator_uq_packet

    packet = _sibling_akel_packet(build_comparator_uq_packet)
    assert set(packet["text_supported_reference_channels"]) == set(CUQ_AKEL_CHANNELS)
    assert set(packet["text_supported_not_acceptance_channels"]) == set(
        CUQ_AKEL_CHANNELS
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_neutron_authority_full_energy_packet_receives_no_akel_channels() -> None:
    """neutron_authority: full-energy scope gets no Akel text-supported channels."""
    from dpf.first_principles.neutron_authority import (
        build_mechanism_separated_neutron_packet,
    )

    packet = _sibling_full_energy_packet(build_mechanism_separated_neutron_packet)
    assert packet["text_supported_reference_channels"] == []
    assert packet["text_supported_not_acceptance_channels"] == []
    assert "akel" not in packet["acceptance_gate"].lower()
    assert packet["can_support_first_principles_acceptance"] is False


def test_neutron_authority_akel_16kv_packet_keeps_akel_channels() -> None:
    """neutron_authority: Akel 16 kV scope keeps its Akel channels."""
    from dpf.first_principles.neutron_authority import (
        PF1000_AKEL_TEXT_SUPPORTED_CHANNELS as NA_AKEL_CHANNELS,
    )
    from dpf.first_principles.neutron_authority import (
        build_mechanism_separated_neutron_packet,
    )

    packet = _sibling_akel_packet(build_mechanism_separated_neutron_packet)
    assert set(packet["text_supported_reference_channels"]) == set(NA_AKEL_CHANNELS)
    assert set(packet["text_supported_not_acceptance_channels"]) == set(
        NA_AKEL_CHANNELS
    )
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# The shared classifier is the single source of truth
# ---------------------------------------------------------------------------


def test_akel_helper_is_single_source_of_truth() -> None:
    """The five builders no longer define their own ``_looks_like_pf1000_akel*``.

    P1-1 was caused by five independent copies of the buggy helper.  After the
    WS9-3 follow-up the only definition is the public
    ``looks_like_pf1000_akel_16kv_scope`` in
    ``dpf.first_principles.channel_state``.  A private
    ``_looks_like_pf1000_akel*`` symbol in any of the five builder modules
    would re-introduce the duplication this repair removed.
    """
    from dpf.first_principles import (
        comparator_uq,
        neutron_authority,
        same_scope,
        spatial_field_temperature,
        waveform_phase,
    )

    for module in (
        same_scope,
        waveform_phase,
        spatial_field_temperature,
        comparator_uq,
        neutron_authority,
    ):
        local_akel_helpers = [
            name
            for name in vars(module)
            if name.startswith("_looks_like_pf1000_akel")
        ]
        assert local_akel_helpers == [], (
            f"{module.__name__} still defines a local Akel helper "
            f"{local_akel_helpers}; it must import the shared classifier"
        )

    # The shared classifier exists and is the exact-match implementation.
    from dpf.first_principles.channel_state import looks_like_pf1000_akel_16kv_scope

    assert looks_like_pf1000_akel_16kv_scope(AKEL_16KV_SCOPE, AKEL_16KV_DEVICE) is True
    assert (
        looks_like_pf1000_akel_16kv_scope(FULL_ENERGY_SCOPE, FULL_ENERGY_DEVICE)
        is False
    )


def test_all_five_builders_use_the_shared_classifier_consistently() -> None:
    """Every builder classifies full-energy vs Akel 16 kV the same way.

    A regression in any single builder's scope routing would show up as a
    full-energy packet that still carries Akel text-supported channels.
    """
    from dpf.first_principles.comparator_uq import build_comparator_uq_packet
    from dpf.first_principles.neutron_authority import (
        build_mechanism_separated_neutron_packet,
    )
    from dpf.first_principles.spatial_field_temperature import (
        build_spatial_field_temperature_packet,
    )
    from dpf.first_principles.waveform_phase import build_waveform_phase_packet

    builders = (
        build_same_scope_source_packet,
        build_waveform_phase_packet,
        build_spatial_field_temperature_packet,
        build_comparator_uq_packet,
        build_mechanism_separated_neutron_packet,
    )
    for build_fn in builders:
        full_energy = _sibling_full_energy_packet(build_fn)
        akel = _sibling_akel_packet(build_fn)
        assert full_energy["text_supported_reference_channels"] == [], (
            f"{build_fn.__name__} leaked Akel channels into the full-energy scope"
        )
        # The Akel 16 kV scope always receives at least one text-supported
        # channel -- proof the exact helper still matches the Akel revision.
        assert len(akel["text_supported_reference_channels"]) > 0, (
            f"{build_fn.__name__} dropped Akel channels from the Akel 16 kV scope"
        )
        assert full_energy["can_support_first_principles_acceptance"] is False
        assert akel["can_support_first_principles_acceptance"] is False
