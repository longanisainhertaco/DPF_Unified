"""SSR-009 / WP-6 negative controls: neutron authority is mechanism-separated and
cannot accept on scalar total yield or reduced-model outputs.

Verified against KnowledgeReference:
- radiation-physics-and-chemistry-188-2021-109633.md:282-288 (scalar yield 6.14e9)
- radiation-physics-and-chemistry-188-2021-109633.md:862-889 (Lee is a fitted model)
- sand2009-6373-b93aec67.md:511-512 (MHD cannot model non-thermonuclear production)
- fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:34-43 (hybrid under-
  predicts ~100x; only fully kinetic reaches experimental yield)
- anisotropy-...-527cc533.md:269-288 (separate scattered vs direct before TOF inversion)
"""

from __future__ import annotations

from dpf.fields.kinetic_yield import (
    kinetic_neutron_yield_authority_status,
)
from dpf.first_principles.neutron_authority import (
    BLOCKING_NEUTRON_AUTHORITY_CHANNELS,
    REQUIRED_NEUTRON_AUTHORITY_CHANNELS,
    build_mechanism_separated_neutron_packet,
)

_AKEL = "pf1000_akel_16kv_1p2torr_shot_12581"


def test_scalar_total_yield_only_cannot_accept_neutron_authority() -> None:
    """A same-scope, accepted scalar yield target must NOT grant authority.

    SSR-009 Rejection Criterion: "Codex will reject total-yield-only claims."
    KR: radiation-physics-and-chemistry-188-2021-109633.md:282-288 (scalar yield
    6.14e9 is a comparator, not a mechanism-separated authority).
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        validation_targets=[
            {
                "name": "pf1000_akel_total_yield",
                "observable": "neutron_scalar_yield",
                "status": "accepted_same_scope_source",
                "declared_scope": _AKEL,
            }
        ],
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_total_yield_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    # Scalar channel may be the ONLY accepted channel, and it is comparator-only.
    assert packet["accepted_channels"] == ["same_scope_scalar_yield"]
    assert packet["mechanism_separation_policy"][
        "scalar_yield_agreement_usable_for"
    ] == "baseline_comparison_only"
    # Every blocking mechanism channel is still missing.
    missing = set(packet["missing_acceptance_channels"])
    assert set(BLOCKING_NEUTRON_AUTHORITY_CHANNELS).issubset(missing)


def test_packet_fails_closed_even_if_every_channel_declared_accepted() -> None:
    """Declaring all required channels accepted must still not accept.

    The blocking channels are unconditionally re-asserted into missing and
    acceptance flags are hardcoded False — fail-closed by construction.
    neutron_authority.py:199-200 (missing.update(BLOCKING_...)) and
    neutron_authority.py:258-259 (hardcoded False literals).
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        accepted_channels=tuple(REQUIRED_NEUTRON_AUTHORITY_CHANNELS),
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_total_yield_acceptance"] is False
    # Blocking channels are re-injected into "missing" regardless of input.
    missing = set(packet["missing_acceptance_channels"])
    assert set(BLOCKING_NEUTRON_AUTHORITY_CHANNELS).issubset(missing)


def test_lee_reduced_beam_target_model_stays_text_reference_not_acceptance() -> None:
    """Lee thermonuclear+beam-target model text is comparator/reference only and must
    never appear as an accepted authority channel.

    KR: radiation-physics-and-chemistry-188-2021-109633.md:862-889 — Lee model is a
    fitted reduced model (current-waveform fitting; fc held constant 0.7).
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
    )
    text_refs = set(packet["text_supported_reference_channels"])
    assert "lee_thermonuclear_and_beam_target_model_text" in text_refs
    assert "lee_beam_target_formula_context" in text_refs
    # Lee channels are reference-only, never accepted.
    assert set(packet["accepted_channels"]).isdisjoint(text_refs)
    statuses = packet["neutron_authority_channel_status"]
    for ch in (
        "accepted_thermonuclear_yield_history",
        "accepted_beam_target_yield_history",
    ):
        assert statuses[ch] == "missing_or_blocked"
    # Measured scalar yield text is also reference-only, not an authority channel.
    assert "measured_scalar_yield_shot_12581" in text_refs


def test_candidate_pic_yield_is_runtime_diagnostic_not_authority() -> None:
    """A candidate PIC kinetic-yield history must not promote the packet.

    KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1259-1266
    — "order-of-magnitude validation rather than precise prediction";
    kinetic_yield.py:116-117 self-declares mechanism_separation_status=
    "not_mechanism_separated".
    """
    kinetic_yield = {
        "status": "candidate_engineering_kinetic_yield_history",
        "mechanism_separation_status": "not_mechanism_separated",
        "mechanism_channels": ["dd_particle_distribution_total"],
        "cumulative_neutrons": 4.2e9,
    }
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        kinetic_yield=kinetic_yield,
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["mechanism_separation_policy"][
        "candidate_pic_yield_usable_for"
    ] == "runtime_diagnostic_only"
    assert packet["kinetic_yield_mechanism_separation_status"] == (
        "not_mechanism_separated"
    )
    # PIC channels must surface only as candidate runtime channels.
    for ch in packet["candidate_runtime_channels"]:
        assert ch.startswith("candidate_")


def test_cross_scope_target_cannot_accept_pf1000_akel_neutron_authority() -> None:
    """An accepted target from a different scope must be rejected for Akel authority.

    SSR-003 / Rule 9: PF-1000 full-energy anisotropy paper (450-500 kJ, 3.5 Torr) is
    explicitly in OTHER_SCOPE_SOURCE_GROUPS as pf1000_full_energy_anisotropy with
    scope_mismatch "PF-1000 operated at 450-500 kJ and 3.5 Torr, not Akel 16 kV".
    cross_scope_policy.can_use_other_scope_for_acceptance is hardcoded False.
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        validation_targets=[
            {
                "name": "pf1000_full_energy_anisotropy",
                "observable": "neutron_anisotropy",
                "status": "accepted_same_scope_source",
                "declared_scope": "pf1000_full_energy_450kj_3p5torr",
            }
        ],
    )
    decisions = packet["validation_target_scope_decisions"]
    assert any(
        d["decision"] == "rejected_missing_or_mismatched_scope_metadata"
        for d in decisions
    )
    assert "neutron_anisotropy_angular_yield" not in packet["accepted_channels"]
    assert packet["cross_scope_policy"]["can_use_other_scope_for_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_detector_response_required_before_kinetic_yield_authority() -> None:
    """kinetic_neutron_yield_authority_status must block when detector-response
    evidence is missing, even if kinetic yield and mechanism evidence are accepted.

    KR: tomographic-reconstruction-...-b78f1154.md:390-427 — shadow-bar detector
    system requires foreground-vs-shadowed separation before accepting yield claims.
    """
    accepted_kinetic: dict = {"passed": True, "status": "accepted"}
    accepted_mech: dict = {"passed": True, "status": "accepted"}

    result = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence=accepted_kinetic,
        mechanism_evidence=accepted_mech,
        detector_response_evidence=None,
        uncertainty_evidence={"passed": True, "status": "accepted"},
    )
    assert result["status"] == "blocked"
    assert "same_scope_detector_response" in result["missing_evidence"]
    assert result["can_support_first_principles_acceptance"] is False


def test_uq_required_before_kinetic_yield_authority() -> None:
    """kinetic_neutron_yield_authority_status must block when UQ evidence is missing.

    KR: radiation-physics-and-chemistry-188-2021-109633.md:130-131 — +-0.2e9 yield
    uncertainty from silver-activation counters must be part of authority gate.
    """
    accepted_kinetic: dict = {"passed": True, "status": "accepted"}
    accepted_mech: dict = {"passed": True, "status": "accepted"}

    result = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence=accepted_kinetic,
        mechanism_evidence=accepted_mech,
        detector_response_evidence={"passed": True, "status": "accepted"},
        uncertainty_evidence=None,
    )
    assert result["status"] == "blocked"
    assert "yield_uncertainty_budget" in result["missing_evidence"]
    assert result["can_support_first_principles_acceptance"] is False


def test_electron_temperature_authority_gates_kinetic_yield() -> None:
    """A Te closure that cannot support quantitative claims must block yield authority.

    KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1226-1240
    — "factor of a few uncertainty in the absolute neutron yield" from Te sensitivity.
    kinetic_yield.py:170-178 appends "electron_temperature_authority" to missing when
    can_support_pressure_hall_quantitative_claims is False.
    """
    status = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence={"passed": True, "status": "accepted"},
        mechanism_evidence={"passed": True, "status": "accepted"},
        detector_response_evidence={"passed": True, "status": "accepted"},
        uncertainty_evidence={"passed": True, "status": "accepted"},
        temperature_authority={
            "status": "blocked",
            "can_support_pressure_hall_quantitative_claims": False,
        },
    )
    assert status["status"] == "blocked"
    assert "electron_temperature_authority" in status["missing_evidence"]
    assert status["can_support_first_principles_acceptance"] is False
