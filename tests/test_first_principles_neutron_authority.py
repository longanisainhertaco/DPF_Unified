"""SSR-009 / WP-6 negative controls: neutron authority is mechanism-separated and
cannot accept on scalar total yield or reduced-model outputs.

S3.6 (handoff docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md
lines 438-483; research basis WP-N6
docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/WP_N6_NEUTRON_AUTHORITY_PACKET.md).

Verified against KnowledgeReference:
- radiation-physics-and-chemistry-188-2021-109633.md:195-215 (Lee eq. 1 fitted
  reduced model), :282-288 (scalar yield 6.14e9), :124-131 (silver activation
  +-0.2e9 uncertainty)
- bosch-hale-1992-fusion-reactivity.md:59-93,106-109 (DD reactivity fit, 6%
  D-D uncertainty) — the volumetric 1/4 prefactor is NOT in this source
- 2019nrlplasma-formulary-037290d4.md:3802-3814 (DD reaction channels, 2.45 MeV)
- sand2009-6373-b93aec67.md:345-355,511-512 (MHD cannot model non-thermonuclear
  production)
- fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:38-43,152-161
  (fluid predicts no neutrons; only fully kinetic reaches experimental yield)
- anisotropy-...-527cc533.md:199-209,280-283 (separate scattered vs direct
  before TOF inversion; >=54% scattered)
"""

from __future__ import annotations

from dpf.diagnostics.beam_target import BEAM_TARGET_ANISOTROPY_LAW_STATUS
from dpf.diagnostics.neutron_tof import THERMONUCLEAR_DOPPLER_WIDTH_STATUS
from dpf.diagnostics.neutron_yield import THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS
from dpf.fields.kinetic_yield import (
    kinetic_neutron_yield_authority_status,
)
from dpf.first_principles.neutron_authority import (
    BLOCKING_NEUTRON_AUTHORITY_CHANNELS,
    NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE,
    NEUTRON_MECHANISM_CHANNELS,
    REQUIRED_NEUTRON_AUTHORITY_CHANNELS,
    KRRef,
    MechanismYieldHistory,
    NeutronAuthorityRuntime,
    build_mechanism_separated_neutron_packet,
)

_AKEL = "pf1000_akel_16kv_1p2torr_shot_12581"

# A KR ref that points at a real local KnowledgeReference file.
_KR_REACTIVITY = KRRef(
    path="KnowledgeReference/bosch-hale-1992-fusion-reactivity.md",
    lines="59-93",
    role="dd_reactivity_fit",
)


def _time_series() -> dict[str, tuple[float, ...]]:
    """A minimal valid mechanism yield history time series (length > 1)."""
    return {
        "times_s": (0.0, 1.0e-9, 2.0e-9),
        "rate_per_s": (0.0, 1.0e15, 2.0e15),
        "cumulative": (0.0, 1.0e6, 3.0e6),
    }


# ---------------------------------------------------------------------------
# Pre-existing negative controls (unchanged behaviour)
# ---------------------------------------------------------------------------


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

    KR: radiation-physics-and-chemistry-188-2021-109633.md:195-215 — Lee model is
    a fitted reduced model (current-waveform fitting; fc held constant 0.7).
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
    kinetic_yield.py self-declares mechanism_separation_status=
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

    SSR-003 / Rule 9: PF-1000 full-energy anisotropy paper (450-500 kJ, 3.5 Torr)
    is explicitly in OTHER_SCOPE_SOURCE_GROUPS.
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

    KR: tomographic-reconstruction-...-b78f1154.md:122-133 — modified ToF detector
    pairs reduce neutron scatter background; foreground/shadow separation required.
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

    KR: radiation-physics-and-chemistry-188-2021-109633.md:124-131 — +-0.2e9 yield
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


# ---------------------------------------------------------------------------
# S3.6 mechanism-separated runtime interface — structural controls
# ---------------------------------------------------------------------------


def test_empty_packet_blocks_every_mechanism_channel() -> None:
    """With no runtime record, all ten mechanism channels are missing/blocked.

    Handoff lines 451-462: thermonuclear, beam-target, ion distribution,
    stopping/transport, spectrum, anisotropy, detector, activation, scatter, UQ.
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
    )
    status = packet["mechanism_channel_status"]
    assert set(status.keys()) == set(NEUTRON_MECHANISM_CHANNELS)
    # None is accepted; the five no-KR-source channels are blocked_by_missing.
    assert all(v != "accepted_neutron_authority" for v in status.values())
    for ch in NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE:
        assert status[ch] == "blocked_by_missing_local_source"
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    assert set(packet["missing_mechanism_channels"]) == set(NEUTRON_MECHANISM_CHANNELS)
    assert packet["can_support_first_principles_acceptance"] is False


def test_mechanism_channels_match_handoff_required_channels() -> None:
    """The ten mechanism channels are exactly the S3.6 'Required channels' list."""
    assert NEUTRON_MECHANISM_CHANNELS == (
        "thermonuclear_history",
        "beam_target_history",
        "ion_energy_distribution",
        "stopping_transport",
        "neutron_spectrum",
        "anisotropy",
        "detector_response",
        "activation_response",
        "scatter_background",
        "uq",
    )


def test_every_blocking_channel_has_a_kr_ref_slot() -> None:
    """Each WP-N6 §5.2 control: blocking channels map to KRRef-typed fields.

    A NeutronAuthorityRuntime carries a typed *_ref slot for every channel that
    can block; absent slot -> channel stays missing_or_blocked.
    """
    ref_fields = (
        "ion_energy_distribution_ref",
        "beam_transport_stopping_ref",
        "neutron_spectrum_ref",
        "doppler_width_law_ref",
        "neutron_anisotropy_ref",
        "intrinsic_anisotropy_law_ref",
        "vessel_scatter_anisotropy_ref",
        "tof_detector_response_ref",
        "activation_counter_response_ref",
        "direct_scattered_transport_ref",
        "same_scope_scalar_yield_ref",
        "yield_uncertainty_budget_ref",
        "electron_temperature_yield_sensitivity_ref",
    )
    runtime = NeutronAuthorityRuntime(declared_scope=_AKEL)
    for fname in ref_fields:
        assert hasattr(runtime, fname)
        # All default to None — fail-closed.
        assert getattr(runtime, fname) is None


def test_mechanism_histories_are_time_series_not_scalars() -> None:
    """A mechanism yield history must be a time series, not a scalar.

    WP-N6 §6.1 rule 3: thermonuclear and beam-target channels carry
    times_s/rate_per_s/cumulative arrays of equal length > 1.
    """
    # A "scalar" history (single sample) is not a valid time series.
    scalar_like = MechanismYieldHistory(
        times_s=(0.0,), rate_per_s=(1.0e15,), cumulative=(1.0e6,)
    )
    assert scalar_like.is_time_series() is False
    # Mismatched-length arrays are not a valid time series.
    ragged = MechanismYieldHistory(
        times_s=(0.0, 1.0e-9), rate_per_s=(1.0e15,), cumulative=(1.0e6, 2.0e6)
    )
    assert ragged.is_time_series() is False
    # A proper >1-length series is valid.
    proper = MechanismYieldHistory(**_time_series())
    assert proper.is_time_series() is True


def test_runtime_record_round_trips_into_packet_still_blocked() -> None:
    """A fully populated synthetic runtime record is accepted by the builder
    without raising, but the packet still cannot support first-principles
    acceptance because blocked_by_missing_local_source channels remain.

    WP-N6 §5.2 control 12 + §7: the interface shape can round-trip; physics
    authority stays blocked.
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=_KR_REACTIVITY,
    )
    bt = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        thermonuclear_yield_history=tn,
        beam_target_yield_history=bt,
        mechanism_separation_status="mechanism_separated",
        neutron_spectrum_ref=_KR_REACTIVITY,
        doppler_width_law_ref=_KR_REACTIVITY,
        neutron_anisotropy_ref=_KR_REACTIVITY,
        intrinsic_anisotropy_law_ref=_KR_REACTIVITY,
        vessel_scatter_anisotropy_ref=_KR_REACTIVITY,
        yield_uncertainty_budget_ref=_KR_REACTIVITY,
        electron_temperature_yield_sensitivity_ref=_KR_REACTIVITY,
        same_scope_scalar_yield=6.14e9,
        same_scope_scalar_yield_uncertainty=0.2e9,
        same_scope_scalar_yield_ref=_KR_REACTIVITY,
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    # No exception; packet still blocked.
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_first_principles_acceptance"] is False
    # The five no-KR-source channels stay blocked even with refs supplied.
    status = packet["mechanism_channel_status"]
    for ch in NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE:
        assert status[ch] == "blocked_by_missing_local_source"
    assert packet["missing_mechanism_channels"]


# ---------------------------------------------------------------------------
# S3.6 mechanism-separated runtime interface — negative controls
# ---------------------------------------------------------------------------


def test_scalar_yield_only_runtime_does_not_separate_mechanisms() -> None:
    """A runtime record with only a scalar yield must not separate mechanisms.

    WP-N6 §5.1 control 1 / §7.1: a record that agrees with the measured 6.14e9
    scalar yield but carries no separated histories stays not_mechanism_separated
    and the scalar is comparator-only.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        same_scope_scalar_yield=6.14e9,
        same_scope_scalar_yield_uncertainty=0.2e9,
        same_scope_scalar_yield_ref=_KR_REACTIVITY,
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    comparator = packet["scalar_yield_comparator"]
    assert comparator["status"] == "candidate_comparator_only"
    assert comparator["present"] is True
    assert comparator["value"] == 6.14e9
    assert comparator["is_mechanism_authority"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    # The thermonuclear and beam-target channels are still missing.
    status = packet["mechanism_channel_status"]
    assert status["thermonuclear_history"] == "missing_or_blocked"
    assert status["beam_target_history"] == "missing_or_blocked"


def test_thermonuclear_only_does_not_grant_total_authority() -> None:
    """A populated thermonuclear history with no beam-target history stays blocked.

    WP-N6 §5.1 control 2: per sand2009-6373-b93aec67.md:345-355,511-512 an
    MHD/thermonuclear-only result cannot represent total DPF yield.
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=_KR_REACTIVITY,
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        thermonuclear_yield_history=tn,
        mechanism_separation_status="mechanism_separated",
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    # No beam-target history => not mechanism-separated.
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    assert packet["mechanism_channel_status"]["beam_target_history"] == (
        "missing_or_blocked"
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_lee_reduced_model_basis_is_comparator_not_authority() -> None:
    """A beam-target history with the Lee/Saw reduced-model basis is comparator-only.

    WP-N6 §5.1 control 3 / §7.2: Lee eq. (1) is a fitted model
    (radiation-physics-and-chemistry-188-2021-109633.md:195-215; fc held 0.7).
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=_KR_REACTIVITY,
    )
    bt_lee = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="lee_reduced_model",
        source_ref=_KR_REACTIVITY,
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        thermonuclear_yield_history=tn,
        beam_target_yield_history=bt_lee,
        mechanism_separation_status="mechanism_separated",
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    # Lee basis -> comparator_only, never an authority channel.
    assert packet["mechanism_channel_status"]["beam_target_history"] == (
        "candidate_comparator_only"
    )
    # A reduced-model beam-target basis cannot grant mechanism separation.
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    assert packet["can_support_first_principles_acceptance"] is False


def test_missing_prefactor_citation_flags_thermonuclear_inferred_candidate() -> None:
    """A thermonuclear history with no prefactor citation is inferred_candidate.

    WP-N6 §4 / §6.1 rule 5: the 1/4 volumetric prefactor is uncited. The
    channel must be inferred_candidate (isolated from authority), not accepted.
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=None,  # uncited 1/4 prefactor
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        thermonuclear_yield_history=tn,
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["thermonuclear_history"] == (
        "inferred_candidate"
    )
    assert "thermonuclear_history" in packet["inferred_candidate_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_blocked_stopping_model_blocks_beam_target_transport() -> None:
    """The stopping/transport channel has no KR source and stays blocked.

    WP-N6 §5.1 control 5 / §1.6: no KR deuteron stopping-power formula exists.
    Even a populated transport status cannot promote the channel.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        beam_transport_stopping_status="blocked_no_kr_source",
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["stopping_transport"] == (
        "blocked_by_missing_local_source"
    )
    assert "stopping_transport" in packet["blocked_by_missing_local_source_channels"]


def test_beam_target_authority_requires_ion_distribution_and_stopping() -> None:
    """A kinetic-ion beam-target history still blocks without ion dist + stopping.

    Handoff line 467 (forbidden: "beam-target authority without ion distribution
    and stopping"); WP-N6 §1.4. Both are blocked_by_missing_local_source for the
    Akel scope, so the kinetic-ion beam-target channel can never be accepted.
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=_KR_REACTIVITY,
    )
    bt = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        thermonuclear_yield_history=tn,
        beam_target_yield_history=bt,
        mechanism_separation_status="mechanism_separated",
        # ion_energy_distribution_ref absent, stopping not computed
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["beam_target_history"] == (
        "missing_or_blocked"
    )
    assert packet["mechanism_channel_status"]["ion_energy_distribution"] == (
        "blocked_by_missing_local_source"
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_anisotropy_without_intrinsic_law_is_inferred_candidate() -> None:
    """A populated anisotropy channel with an uncited A_bt law is inferred_candidate.

    WP-N6 §5.1 control 11 / §4: the 1+0.3*sqrt(E/100) A_bt law is uncited.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        neutron_anisotropy_ref=_KR_REACTIVITY,
        intrinsic_anisotropy_law_ref=None,  # uncited A_bt law
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["anisotropy"] == "inferred_candidate"
    assert "anisotropy" in packet["inferred_candidate_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_spectrum_without_doppler_law_is_inferred_candidate() -> None:
    """A spectrum channel with an uncited Doppler-width law is inferred_candidate.

    WP-N6 §4: the 82.5*sqrt(Ti) "Brysk 1973" Doppler width has no KR source.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        neutron_spectrum_ref=_KR_REACTIVITY,
        doppler_width_law_ref=None,  # uncited 82.5*sqrt(Ti) law
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["neutron_spectrum"] == (
        "inferred_candidate"
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_detector_and_activation_channels_blocked_by_missing_local_source() -> None:
    """Detector and activation channels have no runtime response model.

    Handoff line 468 (forbidden: "detector or activation authority without
    response/scatter packet"); WP-N6 §1.9, §1.10. They stay
    blocked_by_missing_local_source even with refs supplied.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        tof_detector_response_ref=_KR_REACTIVITY,
        activation_counter_response_ref=_KR_REACTIVITY,
        activation_calibration_constant=1.0e9,
        direct_scattered_transport_ref=_KR_REACTIVITY,
        scatter_fraction=0.54,
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    status = packet["mechanism_channel_status"]
    assert status["detector_response"] == "blocked_by_missing_local_source"
    assert status["activation_response"] == "blocked_by_missing_local_source"
    assert status["scatter_background"] == "blocked_by_missing_local_source"
    assert packet["can_support_first_principles_acceptance"] is False


def test_missing_uq_budget_blocks_uq_channel() -> None:
    """The UQ channel blocks when the uncertainty budget ref is missing.

    WP-N6 §5.1 control 8 / §1.12: no runtime UQ budget is computed.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        yield_uncertainty_budget_ref=None,
        electron_temperature_yield_sensitivity_ref=None,
        source_review_status="passed_same_scope_review",
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_channel_status"]["uq"] == "missing_or_blocked"
    assert "uq" in packet["mechanism_channel_blockers"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_cross_scope_runtime_record_rejected_wholesale() -> None:
    """A runtime record with a mismatched declared_scope is rejected wholesale.

    WP-N6 §5.1 control 10 / §3.2 interface contract clause a: a record whose
    declared_scope does not match the target scope cannot accept any channel.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope="llnl_fully_kinetic_dpf_180ka",
        same_scope_scalar_yield=8.6e6,
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["runtime_scope_decision"].startswith(
        "rejected_cross_scope_runtime_record"
    )
    status = packet["mechanism_channel_status"]
    assert all(v != "accepted_neutron_authority" for v in status.values())
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    assert packet["can_support_first_principles_acceptance"] is False


def test_source_review_required_to_accept_any_mechanism_channel() -> None:
    """Without a passed source review, a source-backed channel stays candidate.

    WP-N6 §3.2 interface contract clause d / §6.1 rule 2: a channel only reaches
    accepted_neutron_authority when source_review_status is
    passed_same_scope_review.
    """
    tn = MechanismYieldHistory(
        **_time_series(),
        mechanism_basis="kinetic_ion_distribution",
        source_ref=_KR_REACTIVITY,
        prefactor_citation=_KR_REACTIVITY,
    )
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        thermonuclear_yield_history=tn,
        source_review_status="absent",  # no passed review
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    # Source-backed but not reviewed -> source_backed_candidate, not accepted.
    assert packet["mechanism_channel_status"]["thermonuclear_history"] == (
        "source_backed_candidate"
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_mechanism_separation_by_naming_only_is_rejected() -> None:
    """Declaring mechanism_separation_status alone does not separate mechanisms.

    Handoff line 469 (forbidden: "mechanism separation by naming only"). A record
    that sets mechanism_separation_status="mechanism_separated" but supplies no
    real histories stays not_mechanism_separated.
    """
    runtime = NeutronAuthorityRuntime(
        declared_scope=_AKEL,
        mechanism_separation_status="mechanism_separated",  # naming only
    )
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        runtime=runtime,
    )
    assert packet["mechanism_separation_status"] == "not_mechanism_separated"
    assert packet["can_support_first_principles_acceptance"] is False


def test_uncited_coefficients_isolated_and_reported() -> None:
    """The packet reports the four WP-N6 §4 uncited coefficients as isolated.

    The 1/4 thermonuclear prefactor, the 82.5*sqrt(Ti) Doppler width, the
    1+0.3*sqrt(E/100) anisotropy law, and the missing deuteron stopping model
    must be flagged inferred_candidate / blocked_by_missing_local_source.
    """
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
    )
    isolation = {
        item["coefficient"]: item
        for item in packet["uncited_coefficient_isolation"]
    }
    assert len(isolation) == 4
    valid = {"inferred_candidate", "blocked_by_missing_local_source"}
    for item in isolation.values():
        assert item["isolation"] in valid
        assert item["effect_on_authority"]


def test_diagnostic_modules_flag_uncited_coefficients() -> None:
    """The diagnostic modules expose typed isolation flags, not silent defaults.

    WP-N6 §4: neutron_yield.py (1/4 prefactor), neutron_tof.py (82.5*sqrt(Ti)
    Doppler width), beam_target.py (1+0.3*sqrt(E/100) A_bt) each flag the
    coefficient as inferred_candidate and isolate it from authority.
    """
    assert THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS["status"] == "inferred_candidate"
    assert THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS["kr_source"] == (
        "none_for_full_reaction_rate_equation"
    )
    assert (
        THERMONUCLEAR_VOLUMETRIC_PREFACTOR_STATUS[
            "can_support_first_principles_acceptance"
        ]
        is False
    )

    assert THERMONUCLEAR_DOPPLER_WIDTH_STATUS["status"] == "inferred_candidate"
    assert THERMONUCLEAR_DOPPLER_WIDTH_STATUS["kr_source"] == "none"
    assert THERMONUCLEAR_DOPPLER_WIDTH_STATUS["legacy_attribution"] == (
        "brysk_1973_not_in_knowledgereference"
    )

    assert BEAM_TARGET_ANISOTROPY_LAW_STATUS["status"] == "inferred_candidate"
    assert BEAM_TARGET_ANISOTROPY_LAW_STATUS["kr_source"] == (
        "none_coefficient_0p3_uncited_empirical"
    )
