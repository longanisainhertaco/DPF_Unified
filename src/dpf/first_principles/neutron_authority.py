"""Fail-closed neutron mechanism-authority packets.

S3.6 neutron authority packet (handoff
``docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`` lines
438-483; research basis WP-N6
``docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/WP_N6_NEUTRON_AUTHORITY_PACKET.md``).

The neutron-authority interface is mechanism-separated and fail-closed:

- Ten typed channels (thermonuclear history, beam-target history, ion energy
  distribution, stopping/transport, neutron spectrum, anisotropy, detector
  response, activation response, scatter/background, UQ). Each carries a
  status and source references; an absent or uncited channel stays blocked.
- Scalar total yield is never mechanism authority. It is recorded only as
  ``same_scope_scalar_yield`` and tagged ``candidate_comparator_only``.
- Channel mechanism map: see ``REQUIRED_NEUTRON_CHANNELS`` and
  ``NEUTRON_BLOCKER_IDS`` for the authoritative list. The Sprint 4 Priority 4
  update added three scope-mismatched method-context labels (Talebitaher 2012
  NX2, Krasa 2008 PF-1000 full-energy, Klir 2011 ToF; each
  ``can_promote_authority = False``) and attached Bosch-Hale 1992 D-D
  cross-section to the thermonuclear channel, upgrading it from
  ``missing_or_blocked`` to ``inferred_candidate`` (still non-accepting; the
  1/4 volumetric prefactor is uncited per WP-N6 §4 and is verified NOT FOUND
  in Bernard 1977 per the Sprint 5 extraction packet). The V2 audit-handoff
  ledger
  ``docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`` is the
  authoritative per-blocker source-availability source: NEUTRON-BLK-001 has
  been reclassified V1→V2 from ``external_acquisition_required`` to
  ``existing_kr_target_extraction_pending`` (Scholz/Gribkov Part II is
  already in KR at ``KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md``
  per the Codex audit). Mechanism authority stays blocked until every
  mechanism and detector packet exists with a reviewed same-scope source and
  a passed review certificate.
- Uncited coefficients found by WP-N6 §4 (the ``1/4`` thermonuclear volumetric
  prefactor, the ``82.5*sqrt(Ti)`` "Brysk 1973" Doppler width in
  ``neutron_tof.py``, the ``1+0.3*sqrt(E/100)`` beam-target anisotropy law in
  ``beam_target.py``, and the missing deuteron stopping model) are
  ``inferred_candidate`` / ``blocked_by_missing_local_source`` — isolated from
  authority, never silently kept as defaults.

``can_support_first_principles_acceptance`` is ``False`` by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

NEUTRON_AUTHORITY_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "120-131,190-215,282-288,862-889",
        "role": "pf1000_akel_neutron_detector_yield_and_lee_baseline_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "952-970,1037-1040,1083-1089,1214-1266",
        "role": "hybrid_pic_fluid_yield_history_and_limitations",
    },
    {
        "path": "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md",
        "lines": "34-43,68-78,126-161",
        "role": "fully_kinetic_mev_ion_and_beam_target_requirement",
    },
    {
        "path": "KnowledgeReference/sand2009-6373-b93aec67.md",
        "lines": "346-352,394-397,511-557,671-679",
        "role": "user_validated_alegra_mhd_thermonuclear_limit_and_nonthermal_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "lines": "39-44,409-418,433-445,551-613",
        "role": "mechanism_separation_timing_spectrum_anisotropy_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md"
        ),
        "lines": "32-53,337-351,390-427,518-526",
        "role": "tof_tomography_detector_response_and_scatter_subtraction_schema",
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "121-137,175-204,269-288",
        "role": "pf1000_anisotropy_detector_and_scattering_schema_other_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "open-access-proceedings-journal-of-physics-conference-series-ed196711.md"
        ),
        "lines": "93-141,152-190,680-697,782-805",
        "role": "user_validated_current_abruption_plasma_diode_mitl_and_neutron_application_context",
    },
    {
        "path": "KnowledgeReference/original-research-f7894f85.md",
        "lines": "269-288,300-323",
        "role": "user_validated_ir_mpf100_neutron_activation_and_double_pinch_context",
    },
    {
        "path": "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md",
        "lines": "180-200,210-237,291-295",
        "role": "user_validated_compact_dpf_tof_fwhm_pressure_yield_context",
    },
)

REQUIRED_NEUTRON_AUTHORITY_CHANNELS = (
    "accepted_thermonuclear_yield_history",
    "accepted_beam_target_yield_history",
    "mechanism_separated_yield_channels",
    "ion_energy_distribution_history",
    "beam_angular_distribution_history",
    "beam_transport_stopping_model",
    "target_density_path_length_history",
    "dd_cross_section_source_and_units",
    "neutron_timing_history",
    "neutron_energy_spectrum",
    "neutron_anisotropy_angular_yield",
    "detector_response_model",
    "activation_counter_response_model",
    "direct_scattered_neutron_transport",
    "same_scope_scalar_yield",
    "yield_uncertainty_budget",
    "electron_temperature_yield_sensitivity_uq",
    "output_mapping_and_comparator",
    "source_review_certificate",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "scintillator_detector_layout_0_90_180_degrees",
    "time_of_flight_mean_neutron_deuteron_energy_method",
    "silver_activation_total_yield_measurement",
    "am_be_activation_calibration_text",
    "yield_uncertainty_scalar",
    "measured_scalar_yield_shot_12581",
    "lee_thermonuclear_and_beam_target_model_text",
    "lee_beam_target_formula_context",
    "current_derivative_t0_reference_for_neutron_timing",
    "average_yield_series_fit_context",
)

BLOCKING_NEUTRON_AUTHORITY_CHANNELS = (
    "accepted_thermonuclear_yield_history",
    "accepted_beam_target_yield_history",
    "mechanism_separated_yield_channels",
    "ion_energy_distribution_history",
    "beam_angular_distribution_history",
    "beam_transport_stopping_model",
    "target_density_path_length_history",
    "neutron_energy_spectrum",
    "neutron_anisotropy_angular_yield",
    "detector_response_model",
    "activation_counter_response_model",
    "direct_scattered_neutron_transport",
    "yield_uncertainty_budget",
    "electron_temperature_yield_sensitivity_uq",
    "output_mapping_and_comparator",
    "source_review_certificate",
)

TRANSFER_RULE_REQUIRED_CHANNELS = (
    "source_scope_identity",
    "target_scope_identity",
    "changed_device_or_shot_parameters",
    "mechanism_transfer_equations_or_bounds",
    "detector_response_transfer_bounds",
    "spectrum_anisotropy_transfer_bounds",
    "uncertainty_inflation_rule",
    "review_certificate",
    "negative_test_cross_scope_promotion",
)

OTHER_SCOPE_SOURCE_GROUPS = (
    {
        "name": "new_2026_axisymmetric_hybrid_pic_fluid",
        "scope_mismatch": (
            "2D axisymmetric compact/LLNL-like hybrid simulation, not "
            "PF-1000/Akel shot 12581."
        ),
        "usable_for": "resolved ion-distribution yield-history requirements",
    },
    {
        "name": "llnl_fully_kinetic_dpf",
        "scope_mismatch": "LLNL low-current DPF, not PF-1000/Akel shot 12581.",
        "usable_for": "requirement for kinetic MeV ions, beam formation, and beam-target yield",
    },
    {
        "name": "mjolnir_ma_class_mechanism_separation",
        "scope_mismatch": "MA/MJ-class MJOLNIR source, not PF-1000/Akel shot 12581.",
        "usable_for": "thermonuclear vs beam-target timing, spectrum, and anisotropy schema",
    },
    {
        "name": "tof_tomography_detector_response",
        "scope_mismatch": "NNSS deuterium DPF detector setup, not PF-1000/Akel shot 12581.",
        "usable_for": "time-energy spectrum inversion and detector/scatter-response schema",
    },
    {
        "name": "pf1000_full_energy_anisotropy",
        "scope_mismatch": "PF-1000 operated at 450-500 kJ and 3.5 Torr, not Akel 16 kV.",
        "usable_for": "anisotropy, direct/scattered neutron, and detector-response schema",
    },
)

# ---------------------------------------------------------------------------
# S3.6 typed mechanism-separated neutron-authority runtime interface
# ---------------------------------------------------------------------------
#
# WP-N6 §3.2 defines one structured ``NeutronAuthorityRuntime`` record per
# accepted same-scope shot, fed into ``build_mechanism_separated_neutron_packet``.
# Every physics channel below carries a ``KRRef`` slot; a ``None`` ref keeps the
# channel ``missing_or_blocked``. No channel is satisfied "by naming only" —
# acceptance requires a reviewed same-scope source AND a passed review
# certificate (handoff lines 464-482, WP-N6 §6.1).

# Mechanism channels in handoff order (S3.6 "Required channels", lines 451-462).
NEUTRON_MECHANISM_CHANNELS = (
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

# Per-channel status vocabulary. ``inferred_candidate`` and
# ``blocked_by_missing_local_source`` are the WP-N6 §4 / §5 labels for the
# uncited-coefficient and no-source cases; they must never become a default.
_MECHANISM_CHANNEL_STATUSES = (
    "missing_or_blocked",
    "blocked_by_missing_local_source",
    "inferred_candidate",
    "candidate_comparator_only",
    "source_backed_candidate",
    "accepted_neutron_authority",
)

# Mechanisms whose first-principles source is absent from KnowledgeReference
# (WP-N6 §2 ``blocked`` rows 3, 4, 7, 8 + the §4 missing-parameter table).
# These cannot be promoted past ``blocked_by_missing_local_source`` no matter
# what a runtime record carries.
NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE = (
    "ion_energy_distribution",
    "stopping_transport",
    "detector_response",
    "activation_response",
    "scatter_background",
)

# ---------------------------------------------------------------------------
# Sprint 4 Priority 4 — method-context labels for scope-mismatched sources
# ---------------------------------------------------------------------------
#
# Three extraction packets are available but NONE matches the Akel 16 kV scope.
# They supply METHOD DESIGN context only and must never promote authority for
# the Akel target scope.  ``can_promote_authority`` is False for all three.
#
# Blocker IDs cross-referenced in per-channel verdicts:
#   NEUTRON-BLK-001  ion_energy_distribution  (no same-scope PF-1000 KR source)
#   NEUTRON-BLK-002  stopping_transport        (no tabulated dE/dx in KR)
#   NEUTRON-BLK-003  beam_target_yield_history (missing distribution + stopping)
#   NEUTRON-BLK-004  neutron_spectrum          (Brysk 1973 Doppler not in KR)
#   NEUTRON-BLK-005  anisotropy                (no same-scope anisotropy in KR)
SPRINT4_METHOD_CONTEXT_LABELS = (
    {
        "source_id": "talebitaher_2012_nx2_detector_anisotropy",
        "label": "candidate_method_context_wrong_scope_nx2_not_pf1000",
        "scope": "NX2 1.6 kJ device, ~1-3e8 n/shot, NOT PF-1000 (~1 MA, ~1e10-1e11 n/shot)",
        "kr_path": (
            "KnowledgeReference/coded-aperture-imaging-of-nuclear-fusion-"
            "in-the-plasma-focus-device-9b79429f.md"
        ),
        "what_transfers": (
            "directed-deuteron-cone model (30 deg forward cone) as beam-target "
            "METHOD design context; BC-408 detector geometry as METHOD design context"
        ),
        "what_does_not_transfer": (
            "fast-ion distribution authority for PF-1000/Akel 16 kV; "
            "anisotropy authority for PF-1000/Akel 16 kV"
        ),
        "can_promote_authority": False,
        "blocker_ids_addressed": ["NEUTRON-BLK-001", "NEUTRON-BLK-005"],
        "channels_affected": ["ion_energy_distribution", "anisotropy"],
    },
    {
        "source_id": "krasa_2008_pf1000_vessel_scatter_anisotropy",
        "label": "candidate_method_context_pf1000_full_energy_not_akel_16kv",
        "scope": (
            "PF-1000 full-energy 450-500 kJ, 3.5 Torr, ~3.5e11 n/shot, "
            "NOT PF-1000/Akel 16 kV 1.2 Torr, ~6.14e9 n/shot"
        ),
        "kr_path": (
            "KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-"
            "neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "kr_lines": "121-137,175-204,269-288",
        "what_transfers": (
            "MCNP vessel-scatter schema (material, thickness, TOF kernel L^2/t^5); "
            "direct/scattered neutron separation requirement; "
            "detector geometry design context"
        ),
        "what_does_not_transfer": (
            "scatter fraction or anisotropy ratio authority for Akel 16 kV; "
            "same-scope detector-response acceptance without a reviewed transfer rule"
        ),
        "can_promote_authority": False,
        "blocker_ids_addressed": ["NEUTRON-BLK-005"],
        "channels_affected": ["scatter_background", "detector_response", "anisotropy"],
        "note": (
            "A reviewed transfer rule with uncertainty inflation is required before "
            "any Krasa 2008 number can be applied to the Akel 16 kV scope."
        ),
    },
    {
        "source_id": "klir_2011_tof_detector_response",
        "label": "candidate_method_context_pf1000_full_energy_not_akel_16kv",
        "scope": (
            "PF-1000 / z-pinch BC-408 detector calibration (Klir 2011); "
            "used at PF-1000 but NOT same-shot-condition as Akel 16 kV"
        ),
        "kr_path": (
            "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-"
            "measurements-in-z-pinch-and-plasma-focus-214fbdae.md"
        ),
        "kr_lines": "78-102,118-138,154-170,171-198,199-207",
        "what_transfers": (
            "BC-408 scintillator timing parameters (FWHM 5.7 ns, rise 2.9 ns, "
            "fall 8.0 ns); PMT Hamamatsu H1949-51 design context; "
            "2.45 MeV calibration point"
        ),
        "what_does_not_transfer": (
            "detector-response authority for Akel 16 kV acceptance; "
            "absolute sensitivity without same-detector geometry mapping"
        ),
        "can_promote_authority": False,
        "blocker_ids_addressed": [],
        "channels_affected": ["detector_response"],
        "note": (
            "Same detector hardware was used at PF-1000; however acceptance "
            "requires a same-scope detector-response model (digitized sensitivity "
            "curve + geometry mapping), not available in this KR extract."
        ),
    },
)

@dataclass(frozen=True)
class KRRef:
    """A local KnowledgeReference citation slot (WP-N6 §3.2 ``KRRef``).

    ``path`` is a repo-relative ``KnowledgeReference/`` file, ``lines`` an exact
    line range, ``role`` the equation/figure/table the citation supports. A
    channel whose ``KRRef`` is ``None`` stays ``missing_or_blocked``.
    """

    path: str
    lines: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "lines": self.lines, "role": self.role}

    def is_local_knowledge_reference(self) -> bool:
        return self.path.startswith("KnowledgeReference/")


# Bosch-Hale KR source for the DD cross-section σ(E) and reactivity <σv>(T)
# path.  This is the ONLY thermonuclear channel ingredient that is
# KR-source-backed; the 1/4 volumetric prefactor remains inferred_candidate
# (WP-N6 §4) and keeps the thermonuclear channel at inferred_candidate until
# prefactor_citation is also a reviewed KR source.
BOSCH_HALE_DD_CROSS_SECTION_KR_REF = KRRef(
    path="KnowledgeReference/bosch-hale-1992-fusion-reactivity.md",
    lines="59-93,106-109",
    role=(
        "DD fusion cross-section sigma(E) fit (Table IV, D(d,n)He-3 branch) "
        "and reactivity <sigma v>(Ti) fit (Table VII, D(d,n)3He branch); "
        "Bosch & Hale 1992 Nucl. Fusion 32(4):611-631"
    ),
)


@dataclass(frozen=True)
class MechanismYieldHistory:
    """A mechanism-separated yield history — a time series, NOT a scalar.

    WP-N6 §6.1 rule 3: thermonuclear and beam-target channels must each carry
    ``times_s`` / ``rate_per_s`` / ``cumulative`` arrays of equal length > 1. A
    scalar total is rejected here and recorded only in
    ``NeutronAuthorityRuntime.same_scope_scalar_yield``.
    """

    times_s: tuple[float, ...] = ()
    rate_per_s: tuple[float, ...] = ()
    cumulative: tuple[float, ...] = ()
    # ``kinetic_ion_distribution`` or ``lee_reduced_model`` (WP-N6 §6.1 rule 4:
    # a ``lee_reduced_model`` basis is comparator-only, never authority).
    mechanism_basis: str | None = None
    source_ref: KRRef | None = None
    # The thermonuclear ``1/4`` volumetric prefactor is uncited (WP-N6 §4); a
    # ``None`` here keeps the thermonuclear channel ``missing_or_blocked`` even
    # when the history arrays are populated (WP-N6 §6.1 rule 5).
    prefactor_citation: KRRef | None = None

    def is_time_series(self) -> bool:
        n = len(self.times_s)
        return (
            n > 1
            and len(self.rate_per_s) == n
            and len(self.cumulative) == n
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": len(self.times_s),
            "is_time_series": self.is_time_series(),
            "mechanism_basis": self.mechanism_basis,
            "source_ref": None if self.source_ref is None else self.source_ref.to_dict(),
            "prefactor_citation": (
                None
                if self.prefactor_citation is None
                else self.prefactor_citation.to_dict()
            ),
        }


@dataclass(frozen=True)
class NeutronAuthorityRuntime:
    """One mechanism-separated neutron-authority runtime record (WP-N6 §3.2).

    Every field is fail-closed: absent or uncited keeps the corresponding
    channel ``missing_or_blocked``. ``same_scope_scalar_yield`` alone never
    satisfies mechanism separation — it is comparator-only by construction.
    """

    declared_scope: str
    device_name: str | None = None

    # --- mechanism-separated yield histories (REQUIRED for authority) ---
    thermonuclear_yield_history: MechanismYieldHistory | None = None
    beam_target_yield_history: MechanismYieldHistory | None = None
    # "mechanism_separated" only when both histories are real time series with
    # a non-reduced-model basis (WP-N6 §3.2 interface contract clause b).
    mechanism_separation_status: str = "not_mechanism_separated"

    # --- ion energy distribution (WP-N6 §1.5; blocked for Akel authority) ---
    ion_energy_distribution_ref: KRRef | None = None
    ion_distribution_is_core_tail_separated: bool = False

    # --- DD cross-section / reactivity (Sprint 4: Bosch-Hale KR-source-backed) ---
    # σ(E) and <σv>(T) are KR-source-backed via bosch-hale-1992-fusion-reactivity.md.
    # The thermonuclear channel still stays inferred_candidate until
    # prefactor_citation is also reviewed (WP-N6 §4: 1/4 prefactor uncited).
    bosch_hale_dd_reactivity_ref: KRRef | None = None

    # --- transport / stopping (WP-N6 §1.6 / §4: no KR source) ---
    beam_transport_stopping_status: str = "blocked_no_kr_source"
    beam_transport_stopping_ref: KRRef | None = None

    # --- neutron spectrum (WP-N6 §1.7) ---
    neutron_spectrum_ref: KRRef | None = None
    # The thermonuclear Doppler-broadening law (the ``82.5*sqrt(Ti)`` /
    # "Brysk 1973" coefficient in ``neutron_tof.py``) has no KR source
    # (WP-N6 §4) — a ``None`` here keeps the spectrum channel blocked.
    doppler_width_law_ref: KRRef | None = None

    # --- anisotropy (WP-N6 §1.8) ---
    neutron_anisotropy_ref: KRRef | None = None
    # The ``1+0.3*sqrt(E/100)`` beam-target anisotropy law in
    # ``beam_target.py`` is uncited/empirical (WP-N6 §4).
    intrinsic_anisotropy_law_ref: KRRef | None = None
    vessel_scatter_anisotropy_ref: KRRef | None = None

    # --- detector response (WP-N6 §1.9 / §1.10; blocked) ---
    tof_detector_response_ref: KRRef | None = None
    activation_counter_response_ref: KRRef | None = None
    activation_calibration_constant: float | None = None

    # --- scatter / background (WP-N6 §1.11; blocked) ---
    direct_scattered_transport_ref: KRRef | None = None
    scatter_fraction: float | None = None

    # --- comparator + UQ ---
    # Comparator ONLY. Recording a scalar yield never advances mechanism
    # separation (handoff lines 466,479; WP-N6 §7.1).
    same_scope_scalar_yield: float | None = None
    same_scope_scalar_yield_uncertainty: float | None = None
    same_scope_scalar_yield_ref: KRRef | None = None
    yield_uncertainty_budget_ref: KRRef | None = None
    electron_temperature_yield_sensitivity_ref: KRRef | None = None

    # --- review certificate ---
    # ``passed_same_scope_review`` is the only review status that can clear a
    # blocking channel (WP-N6 §3.2 interface contract clause d).
    source_review_status: str = "absent"
    source_review_reviewer: str | None = None

    def has_scalar_yield(self) -> bool:
        return self.same_scope_scalar_yield is not None

    def review_passed(self) -> bool:
        return self.source_review_status == "passed_same_scope_review"


def build_mechanism_separated_neutron_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_channels: tuple[str, ...] | list[str] = (),
    kinetic_yield: Mapping[str, Any] | None = None,
    same_scope_source: Mapping[str, Any] | None = None,
    physics_closure: Mapping[str, Any] | None = None,
    runtime: NeutronAuthorityRuntime | None = None,
) -> dict[str, Any]:
    """Return a non-promoting neutron-yield authority packet.

    When ``runtime`` is supplied (a :class:`NeutronAuthorityRuntime` record),
    the builder evaluates the ten mechanism-separated channels (WP-N6 §3.2):
    each gets a typed status from :data:`_MECHANISM_CHANNEL_STATUSES`. A
    cross-scope record is rejected wholesale; ``same_scope_scalar_yield`` is
    reported only as a ``candidate_comparator_only`` comparator. The packet
    stays ``blocked`` and ``can_support_first_principles_acceptance`` stays
    ``False`` until every mechanism and detector channel is
    ``accepted_neutron_authority``.
    """

    accepted = {str(channel) for channel in accepted_channels}
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted.update(target_channels)
    text_supported = (
        set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
        if _looks_like_pf1000_akel_scope(declared_scope, device_name)
        else set()
    )

    missing = set(REQUIRED_NEUTRON_AUTHORITY_CHANNELS) - accepted
    missing.update(BLOCKING_NEUTRON_AUTHORITY_CHANNELS)

    mechanism_report = _evaluate_mechanism_runtime(
        runtime,
        declared_scope=declared_scope,
        device_name=device_name,
    )

    return {
        "status": "blocked_mechanism_separated_neutron_authority_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_total_neutron_yield_authority",
        "acceptance_gate": (
            "scalar_yield_reduced_model_text_and_other_scope_neutron_diagnostics_"
            "cannot_support_total_yield_authority_until_same_scope_mechanism_"
            "separated_histories_detector_transport_comparator_uq_and_review_pass"
        ),
        "required_channels": list(REQUIRED_NEUTRON_AUTHORITY_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "candidate_runtime_channels": _candidate_runtime_channels(kinetic_yield),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "neutron_authority_channel_status": _channel_statuses(
            required_channels=REQUIRED_NEUTRON_AUTHORITY_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "cross_scope_policy": {
            "status": "blocked_without_reviewed_transfer_rule",
            "required_transfer_rule_channels": list(TRANSFER_RULE_REQUIRED_CHANNELS),
            "other_scope_sources_usable_for": "requirements_or_schema_only",
            "can_use_other_scope_for_acceptance": False,
        },
        "mechanism_separation_policy": {
            "total_yield_is_not_authoritative_without_separate_mechanisms": True,
            "required_mechanisms": [
                "thermonuclear_yield_history",
                "beam_target_yield_history",
            ],
            "scalar_yield_agreement_usable_for": "baseline_comparison_only",
            "candidate_pic_yield_usable_for": "runtime_diagnostic_only",
        },
        "validation_target_scope_decisions": target_decisions,
        "mechanism_channels": list(NEUTRON_MECHANISM_CHANNELS),
        "mechanism_channel_status": mechanism_report["channel_status"],
        "mechanism_channel_blockers": mechanism_report["channel_blockers"],
        "mechanism_separation_status": mechanism_report["mechanism_separation_status"],
        "missing_mechanism_channels": mechanism_report["missing_mechanism_channels"],
        "blocked_by_missing_local_source_channels": (
            mechanism_report["blocked_by_missing_local_source_channels"]
        ),
        "inferred_candidate_channels": mechanism_report["inferred_candidate_channels"],
        "scalar_yield_comparator": mechanism_report["scalar_yield_comparator"],
        "runtime_scope_decision": mechanism_report["runtime_scope_decision"],
        "uncited_coefficient_isolation": mechanism_report["uncited_coefficient_isolation"],
        "source_references": list(NEUTRON_AUTHORITY_SOURCE_REFS),
        # Sprint 4 Priority 4: scope-mismatched method-context packets.
        # These three sources are candidate_method_context only — none can
        # promote accepted_neutron_authority for the Akel 16 kV scope.
        "sprint4_method_context_labels": list(SPRINT4_METHOD_CONTEXT_LABELS),
        "sprint4_bosch_hale_thermonuclear_status": {
            "channel": "thermonuclear_history",
            "cross_section_path": "source_supported",
            "cross_section_kr_ref": BOSCH_HALE_DD_CROSS_SECTION_KR_REF.to_dict(),
            "prefactor_1_4_status": "inferred_candidate_no_kr_source",
            "channel_verdict": (
                "thermonuclear channel stays inferred_candidate: σ(E)/<σv>(T) "
                "is KR-source-backed via Bosch-Hale 1992, but the 1/4 volumetric "
                "prefactor has no verbatim KR formula (WP-N6 §4); "
                "beam-target channel stays blocked: ion-distribution (BLK-001) "
                "and stopping (BLK-002) are absent"
            ),
            "beam_target_blocker_ids": [
                "NEUTRON-BLK-001-ion-distribution-no-same-scope-kr-source",
                "NEUTRON-BLK-002-deuteron-stopping-power-no-tabulated-kr-source",
                "NEUTRON-BLK-003-beam-target-yield-no-distribution-no-stopping",
            ],
            "spectrum_blocker_id": (
                "NEUTRON-BLK-004-doppler-broadening-brysk-1973-not-in-kr"
            ),
            "anisotropy_blocker_id": (
                "NEUTRON-BLK-005-anisotropy-same-scope-not-in-kr"
            ),
        },
        "same_scope_source_status": (
            None if same_scope_source is None else same_scope_source.get("status")
        ),
        "physics_closure_status": (
            None if physics_closure is None else physics_closure.get("status")
        ),
        "beam_target_closure_status": _beam_target_closure_status(physics_closure),
        "kinetic_yield_status": (
            None if kinetic_yield is None else kinetic_yield.get("status")
        ),
        "kinetic_yield_mechanism_separation_status": (
            None
            if kinetic_yield is None
            else kinetic_yield.get("mechanism_separation_status")
        ),
        "validation_target_count": len(validation_targets),
        "can_support_total_yield_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _candidate_runtime_channels(kinetic_yield: Mapping[str, Any] | None) -> list[str]:
    if kinetic_yield is None:
        return []
    channels = ["candidate_pic_ion_neutron_yield_history"]
    for channel in kinetic_yield.get("mechanism_channels", ()) or ():
        channels.append(f"candidate_{channel}")
    return sorted(set(str(channel) for channel in channels))


# ---------------------------------------------------------------------------
# S3.6 mechanism-runtime evaluation
# ---------------------------------------------------------------------------

# The empty-record report: every mechanism channel ``missing_or_blocked`` (the
# five no-KR-source channels are ``blocked_by_missing_local_source``). Used
# when no runtime record is supplied and as the base for cross-scope rejection.
def _blocked_mechanism_report(reason: str) -> dict[str, Any]:
    status: dict[str, str] = {}
    blockers: dict[str, str] = {}
    for channel in NEUTRON_MECHANISM_CHANNELS:
        if channel in NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE:
            status[channel] = "blocked_by_missing_local_source"
        else:
            status[channel] = "missing_or_blocked"
        blockers[channel] = reason
    return {
        "channel_status": status,
        "channel_blockers": blockers,
        "mechanism_separation_status": "not_mechanism_separated",
        "missing_mechanism_channels": list(NEUTRON_MECHANISM_CHANNELS),
        "blocked_by_missing_local_source_channels": list(
            NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE
        ),
        "inferred_candidate_channels": [],
        "scalar_yield_comparator": {
            "status": "candidate_comparator_only",
            "present": False,
            "value": None,
            "uncertainty": None,
            "is_mechanism_authority": False,
            "note": (
                "Scalar total neutron yield is a baseline comparator only; it "
                "never satisfies mechanism separation (WP-N6 §7.1)."
            ),
        },
        "runtime_scope_decision": reason,
        "uncited_coefficient_isolation": _UNCITED_COEFFICIENT_ISOLATION,
    }


# WP-N6 §4 uncited coefficients: each is isolated from authority. The diagnostic
# modules tag the live coefficient ``inferred_candidate`` /
# ``blocked_by_missing_local_source``; the authority packet records why.
_UNCITED_COEFFICIENT_ISOLATION = (
    {
        "coefficient": "thermonuclear_volumetric_prefactor_one_quarter",
        "where": "src/dpf/diagnostics/neutron_yield.py dY/dt = (1/4) n_D^2 <sigma v> V",
        "kr_status": "no_verbatim_kr_formula_for_full_reaction_rate_equation",
        "isolation": "inferred_candidate",
        "effect_on_authority": (
            "thermonuclear channel stays missing_or_blocked until "
            "prefactor_citation is a reviewed KR source"
        ),
    },
    {
        "coefficient": "thermonuclear_doppler_width_82p5_sqrt_Ti_brysk_1973",
        "where": "src/dpf/diagnostics/neutron_tof.py sigma = 82.5*sqrt(Ti_keV) keV",
        "kr_status": "no_kr_source_brysk_1973_coefficient_not_in_knowledgereference",
        "isolation": "inferred_candidate",
        "effect_on_authority": (
            "neutron spectrum channel stays missing_or_blocked until "
            "doppler_width_law_ref is a reviewed KR source"
        ),
    },
    {
        "coefficient": "beam_target_anisotropy_one_plus_0p3_sqrt_E_over_100",
        "where": "src/dpf/diagnostics/beam_target.py A_bt = 1 + 0.3*sqrt(E_beam/100 keV)",
        "kr_status": "no_kr_source_coefficient_0p3_uncited_empirical",
        "isolation": "inferred_candidate",
        "effect_on_authority": (
            "anisotropy channel stays missing_or_blocked until "
            "intrinsic_anisotropy_law_ref is a reviewed KR source"
        ),
    },
    {
        "coefficient": "deuteron_beam_stopping_power_in_deuterium_plasma",
        "where": "beam-target transport (no KR formula; single fixed sigma(3*Vmax))",
        "kr_status": "no_kr_source_no_bethe_or_plasma_stopping_model_in_corpus",
        "isolation": "blocked_by_missing_local_source",
        "effect_on_authority": "stopping/transport channel blocked_by_missing_local_source",
    },
)


def _evaluate_mechanism_runtime(
    runtime: NeutronAuthorityRuntime | None,
    *,
    declared_scope: str,
    device_name: str | None,
) -> dict[str, Any]:
    """Evaluate the ten mechanism-separated channels for a runtime record.

    Fail-closed: an absent record blocks every channel; a cross-scope record
    is rejected wholesale (WP-N6 §3.2 interface contract clause a); the five
    no-KR-source channels can never exceed ``blocked_by_missing_local_source``;
    scalar yield is always reported as ``candidate_comparator_only``.
    """
    if runtime is None:
        return _blocked_mechanism_report("no_runtime_record_supplied")

    if not _runtime_scope_matches(runtime, declared_scope, device_name):
        report = _blocked_mechanism_report(
            "rejected_cross_scope_runtime_record_declared_scope_mismatch"
        )
        report["scalar_yield_comparator"]["present"] = runtime.has_scalar_yield()
        return report

    review_ok = runtime.review_passed()
    status: dict[str, str] = {}
    blockers: dict[str, str] = {}

    for channel in NEUTRON_MECHANISM_CHANNELS:
        channel_status, blocker = _evaluate_mechanism_channel(
            channel, runtime, review_ok=review_ok
        )
        status[channel] = channel_status
        if blocker:
            blockers[channel] = blocker

    missing = [
        channel
        for channel in NEUTRON_MECHANISM_CHANNELS
        if status[channel] != "accepted_neutron_authority"
    ]
    blocked_local = sorted(
        channel
        for channel in NEUTRON_MECHANISM_CHANNELS
        if status[channel] == "blocked_by_missing_local_source"
    )
    inferred = sorted(
        channel
        for channel in NEUTRON_MECHANISM_CHANNELS
        if status[channel] == "inferred_candidate"
    )

    return {
        "channel_status": status,
        "channel_blockers": dict(sorted(blockers.items())),
        "mechanism_separation_status": _runtime_mechanism_separation_status(runtime),
        "missing_mechanism_channels": missing,
        "blocked_by_missing_local_source_channels": blocked_local,
        "inferred_candidate_channels": inferred,
        "scalar_yield_comparator": {
            "status": "candidate_comparator_only",
            "present": runtime.has_scalar_yield(),
            "value": runtime.same_scope_scalar_yield,
            "uncertainty": runtime.same_scope_scalar_yield_uncertainty,
            "is_mechanism_authority": False,
            "note": (
                "Scalar total neutron yield is a baseline comparator only; a "
                "scalar match can occur with both mechanisms wrong in "
                "compensating directions (WP-N6 §7.1)."
            ),
        },
        "runtime_scope_decision": "runtime_record_scope_matches_declared_scope",
        "uncited_coefficient_isolation": _UNCITED_COEFFICIENT_ISOLATION,
    }


def _evaluate_mechanism_channel(
    channel: str,
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    """Return ``(status, blocker)`` for one mechanism channel (fail-closed).

    A channel only reaches ``accepted_neutron_authority`` when its evidence is
    source-backed AND ``source_review_status == "passed_same_scope_review"``.
    The five :data:`NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE` channels can never
    exceed ``blocked_by_missing_local_source`` (WP-N6 §2 / §4).
    """
    if channel in NEUTRON_BLOCKED_BY_MISSING_LOCAL_SOURCE:
        return _blocked_local_channel(channel, runtime)
    if channel == "thermonuclear_history":
        return _evaluate_thermonuclear_channel(runtime, review_ok=review_ok)
    if channel == "beam_target_history":
        return _evaluate_beam_target_channel(runtime, review_ok=review_ok)
    if channel == "neutron_spectrum":
        return _evaluate_spectrum_channel(runtime, review_ok=review_ok)
    if channel == "anisotropy":
        return _evaluate_anisotropy_channel(runtime, review_ok=review_ok)
    if channel == "uq":
        return _evaluate_uq_channel(runtime, review_ok=review_ok)
    return "missing_or_blocked", "unknown_mechanism_channel"


def _blocked_local_channel(
    channel: str,
    runtime: NeutronAuthorityRuntime,
) -> tuple[str, str]:
    """Channels with no first-principles KR source (WP-N6 §2 blocked rows).

    Even when a runtime record carries a ref, these stay
    ``blocked_by_missing_local_source`` — the WP-N6 §6.2 same-scope evidence
    map found no Akel-scope source for any of them.
    """
    refs = {
        "ion_energy_distribution": runtime.ion_energy_distribution_ref,
        "stopping_transport": runtime.beam_transport_stopping_ref,
        "detector_response": runtime.tof_detector_response_ref,
        "activation_response": runtime.activation_counter_response_ref,
        "scatter_background": runtime.direct_scattered_transport_ref,
    }
    reasons = {
        "ion_energy_distribution": (
            "NEUTRON-BLK-001: no same-scope PF-1000/Akel ion f(E) KR source; "
            "Talebitaher 2012 NX2 30-deg cone model is "
            "candidate_method_context_wrong_scope_nx2_not_pf1000 only "
            "(Sprint 4 P4)"
        ),
        "stopping_transport": (
            "NEUTRON-BLK-002: no tabulated deuteron dE/dx (Andersen-Ziegler / "
            "ICRU / Bethe) KR source; plasma stopping model also absent "
            "(WP-N6 §1.6, §4)"
        ),
        "detector_response": (
            "NEUTRON-BLK: no runtime TOF/activation detector-response model; "
            "Klir 2011 (PF-1000 hardware, not Akel 16 kV) is "
            "candidate_method_context_pf1000_full_energy_not_akel_16kv; "
            "Talebitaher 2012 BC-408 NX2 is "
            "candidate_method_context_wrong_device_nx2_not_pf1000 (Sprint 4 P4)"
        ),
        "activation_response": (
            "no runtime activation-counter response model; calibration "
            "constant absent from KR (WP-N6 §1.10)"
        ),
        "scatter_background": (
            "no runtime scatter-transport model; Krasa 2008 MCNP vessel-scatter "
            "is candidate_method_context_pf1000_full_energy_not_akel_16kv — "
            "KR requires direct-vs-scattered separation before TOF inversion "
            "(WP-N6 §1.11, Sprint 4 P4)"
        ),
    }
    ref = refs[channel]
    blocker = reasons[channel]
    if ref is not None:
        blocker = (
            f"{blocker}; supplied ref is other-scope schema only and cannot "
            "promote an Akel-scope authority channel"
        )
    return "blocked_by_missing_local_source", blocker


def _evaluate_thermonuclear_channel(
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    history = runtime.thermonuclear_yield_history
    if history is None:
        # Sprint 4: Bosch-Hale σ(E)/<σv>(T) is KR-source-backed, but a yield
        # history is still required to evaluate the thermonuclear rate channel.
        if runtime.bosch_hale_dd_reactivity_ref is not None:
            return (
                "inferred_candidate",
                "thermonuclear cross-section path (Bosch-Hale 1992) is "
                "KR-source-backed via bosch_hale_dd_reactivity_ref; "
                "thermonuclear_yield_history still absent and the 1/4 "
                "volumetric prefactor has no KR citation (WP-N6 §4) — "
                "channel stays inferred_candidate not accepted",
            )
        return "missing_or_blocked", "thermonuclear_yield_history not supplied"
    if not history.is_time_series():
        return (
            "missing_or_blocked",
            "thermonuclear_yield_history is not a time series "
            "(times_s/rate_per_s/cumulative arrays of equal length > 1)",
        )
    if history.prefactor_citation is None:
        # WP-N6 §6.1 rule 5: the 1/4 volumetric prefactor is uncited.
        # Sprint 4: note whether the cross-section path is KR-source-backed.
        bh_note = (
            "; Bosch-Hale σ(E)/<σv>(T) is KR-source-backed via "
            "BOSCH_HALE_DD_CROSS_SECTION_KR_REF (Sprint 4 P4)"
            if runtime.bosch_hale_dd_reactivity_ref is not None
            else ""
        )
        return (
            "inferred_candidate",
            "thermonuclear (1/4) volumetric prefactor has no KR citation "
            f"(prefactor_citation is None) — WP-N6 §4{bh_note}",
        )
    if history.source_ref is None or not history.source_ref.is_local_knowledge_reference():
        return (
            "missing_or_blocked",
            "thermonuclear source_ref absent or not a local KnowledgeReference path",
        )
    if not review_ok:
        return (
            "source_backed_candidate",
            "thermonuclear history source-backed but source_review_status "
            "is not passed_same_scope_review",
        )
    return "accepted_neutron_authority", ""


def _evaluate_beam_target_channel(
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    history = runtime.beam_target_yield_history
    if history is None:
        return "missing_or_blocked", "beam_target_yield_history not supplied"
    if not history.is_time_series():
        return (
            "missing_or_blocked",
            "beam_target_yield_history is not a time series",
        )
    if history.mechanism_basis == "lee_reduced_model":
        # WP-N6 §6.1 rule 4 / §7.2: Lee/Saw eq.(1) is a fitted reduced model.
        return (
            "candidate_comparator_only",
            "beam-target basis is the Lee/Saw reduced model (fitted, fc held "
            "constant 0.7) — comparator only, never authority (WP-N6 §7.2)",
        )
    if history.mechanism_basis != "kinetic_ion_distribution":
        return (
            "missing_or_blocked",
            "beam_target_yield_history mechanism_basis must be "
            "kinetic_ion_distribution for an authority channel",
        )
    # A kinetic-ion beam-target history still requires the ion distribution and
    # the stopping model — both blocked_by_missing_local_source (WP-N6 §1.4).
    if runtime.ion_energy_distribution_ref is None:
        return (
            "missing_or_blocked",
            "beam-target authority requires an ion energy distribution "
            "(blocked_by_missing_local_source) — WP-N6 §1.4",
        )
    if runtime.beam_transport_stopping_status != "computed":
        return (
            "missing_or_blocked",
            "beam-target authority requires a stopping/transport model "
            "(blocked_by_missing_local_source) — WP-N6 §1.4, §1.6",
        )
    if not review_ok:
        return (
            "source_backed_candidate",
            "beam-target history source-backed but not yet reviewed",
        )
    return "accepted_neutron_authority", ""


def _evaluate_spectrum_channel(
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    if runtime.neutron_spectrum_ref is None:
        return "missing_or_blocked", "neutron_spectrum_ref not supplied"
    if runtime.doppler_width_law_ref is None:
        # WP-N6 §4: the 82.5*sqrt(Ti) "Brysk 1973" Doppler width is uncited.
        return (
            "inferred_candidate",
            "thermonuclear Doppler-broadening law (82.5*sqrt(Ti), 'Brysk 1973') "
            "has no KR source (doppler_width_law_ref is None) — WP-N6 §4",
        )
    if not review_ok:
        return (
            "source_backed_candidate",
            "spectrum source-backed but not yet reviewed",
        )
    return "accepted_neutron_authority", ""


def _evaluate_anisotropy_channel(
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    if runtime.neutron_anisotropy_ref is None:
        return "missing_or_blocked", "neutron_anisotropy_ref not supplied"
    if runtime.intrinsic_anisotropy_law_ref is None:
        # WP-N6 §4: the 1+0.3*sqrt(E/100) A_bt law is uncited/empirical.
        return (
            "inferred_candidate",
            "intrinsic beam-target anisotropy law (1+0.3*sqrt(E/100 keV)) is "
            "uncited/empirical (intrinsic_anisotropy_law_ref is None) — WP-N6 §4",
        )
    if runtime.vessel_scatter_anisotropy_ref is None:
        return (
            "missing_or_blocked",
            "vessel-scattering anisotropy contribution not modelled "
            "(vessel_scatter_anisotropy_ref is None) — WP-N6 §1.8",
        )
    if not review_ok:
        return (
            "source_backed_candidate",
            "anisotropy source-backed but not yet reviewed",
        )
    return "accepted_neutron_authority", ""


def _evaluate_uq_channel(
    runtime: NeutronAuthorityRuntime,
    *,
    review_ok: bool,
) -> tuple[str, str]:
    if runtime.yield_uncertainty_budget_ref is None:
        return (
            "missing_or_blocked",
            "yield_uncertainty_budget_ref not supplied — no runtime UQ budget "
            "(WP-N6 §1.12)",
        )
    if runtime.electron_temperature_yield_sensitivity_ref is None:
        return (
            "missing_or_blocked",
            "electron-temperature yield-sensitivity UQ absent; the Te=alpha*Ti "
            "closure gives a factor-of-a-few yield spread (WP-N6 §1.12)",
        )
    if not review_ok:
        return (
            "source_backed_candidate",
            "UQ budget source-backed but not yet reviewed",
        )
    return "accepted_neutron_authority", ""


def _runtime_mechanism_separation_status(runtime: NeutronAuthorityRuntime) -> str:
    """Mechanism separation holds only with two real, non-reduced histories.

    WP-N6 §3.2 interface contract clause b: ``mechanism_separated`` requires
    both ``thermonuclear_yield_history`` and ``beam_target_yield_history`` to
    be real time series, and the beam-target basis must not be the Lee/Saw
    reduced model. ``mechanism_separation_status`` declared on the record is
    honoured only when the evidence actually supports it (no naming-only
    separation — handoff line 469).
    """
    tn = runtime.thermonuclear_yield_history
    bt = runtime.beam_target_yield_history
    if tn is None or bt is None:
        return "not_mechanism_separated"
    if not tn.is_time_series() or not bt.is_time_series():
        return "not_mechanism_separated"
    if bt.mechanism_basis == "lee_reduced_model":
        return "not_mechanism_separated"
    if runtime.mechanism_separation_status != "mechanism_separated":
        return "not_mechanism_separated"
    return "mechanism_separated"


def _runtime_scope_matches(
    runtime: NeutronAuthorityRuntime,
    declared_scope: str,
    device_name: str | None,
) -> bool:
    """A runtime record is acceptance-eligible only if its scope matches.

    WP-N6 §3.2 interface contract clause a. ``device_name`` is informational
    only — scope identity is decided on ``declared_scope``.
    """
    _ = device_name
    return _normalized_scope(runtime.declared_scope) == _normalized_scope(declared_scope)


def _accepted_channels_from_targets(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    declared_scope: str,
    device_name: str | None,
) -> tuple[set[str], list[dict[str, Any]]]:
    accepted: set[str] = set()
    decisions: list[dict[str, Any]] = []
    aliases = {
        "thermonuclear_yield_history": "accepted_thermonuclear_yield_history",
        "beam_target_yield_history": "accepted_beam_target_yield_history",
        "mechanism_separated_yield": "mechanism_separated_yield_channels",
        "ion_energy_distribution": "ion_energy_distribution_history",
        "ion_distribution_history": "ion_energy_distribution_history",
        "beam_angular_distribution": "beam_angular_distribution_history",
        "neutron_timing": "neutron_timing_history",
        "neutron_spectrum": "neutron_energy_spectrum",
        "neutron_anisotropy": "neutron_anisotropy_angular_yield",
        "detector_response": "detector_response_model",
        "activation_response": "activation_counter_response_model",
        "direct_scattered_neutron_transport": "direct_scattered_neutron_transport",
        "neutron_scalar_yield": "same_scope_scalar_yield",
        "yield_uncertainty": "yield_uncertainty_budget",
    }
    # Scalar yield is a comparator-only channel; accepted scalar targets are
    # NEVER added to the mechanism-authority accepted set (A2 fix: prevents
    # same_scope_scalar_yield from being stamped accepted_neutron_authority).
    _COMPARATOR_ONLY_OBSERVABLES = frozenset({"neutron_scalar_yield"})
    for target in validation_targets:
        status = str(target.get("status", ""))
        observable = str(target.get("observable", "")).strip()
        name = str(target.get("name", observable or "unnamed_target"))
        if status not in {
            "accepted_same_scope_source",
            "reviewed_same_scope_source",
            "accepted",
        }:
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "not_accepted_neutron_authority_status",
            })
            continue
        if not _target_scope_matches(target, declared_scope, device_name):
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "rejected_missing_or_mismatched_scope_metadata",
            })
            continue
        if observable in _COMPARATOR_ONLY_OBSERVABLES:
            # Scalar yield evidence is recorded only as candidate_comparator_only;
            # it must never enter the mechanism-authority accepted set (WP-N6 §7.1).
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "candidate_comparator_only_scalar_not_mechanism_authority",
            })
        elif observable in aliases:
            accepted.add(aliases[observable])
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_neutron_authority_target_channel",
            })
        else:
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "ignored_unmapped_neutron_authority_observable",
            })
    return accepted, decisions


def _channel_statuses(
    *,
    required_channels: tuple[str, ...],
    accepted: set[str],
    text_supported: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in required_channels:
        # same_scope_scalar_yield is always candidate_comparator_only regardless
        # of the accepted set — scalar total yield is never mechanism authority
        # (A2 fix: belt-and-suspenders guard; WP-N6 §7.1).
        if channel == "same_scope_scalar_yield":
            statuses[channel] = "candidate_comparator_only"
        elif channel in accepted:
            statuses[channel] = "accepted_neutron_authority"
        elif channel in text_supported:
            statuses[channel] = "text_supported_reference_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _target_scope_matches(
    target: Mapping[str, Any],
    declared_scope: str,
    device_name: str | None,
) -> bool:
    target_scope = str(
        target.get("declared_scope")
        or target.get("validation_scope")
        or target.get("scope")
        or ""
    ).strip()
    if target_scope:
        return _normalized_scope(target_scope) == _normalized_scope(declared_scope)

    source_reference = target.get("source_reference")
    if isinstance(source_reference, Mapping):
        haystack = " ".join(
            str(source_reference.get(key, ""))
            for key in ("record_id", "role", "path")
        ).lower()
        if _looks_like_pf1000_akel_scope(declared_scope, device_name):
            return (
                "akel" in haystack
                and ("12581" in haystack or "16kv" in haystack or "16_kv" in haystack)
            )
    return False


def _normalized_scope(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _beam_target_closure_status(physics_closure: Mapping[str, Any] | None) -> str | None:
    if physics_closure is None:
        return None
    effects = physics_closure.get("effects")
    if not isinstance(effects, Mapping):
        return None
    beam_target = effects.get("beam_target_coupling")
    if not isinstance(beam_target, Mapping):
        return None
    return None if beam_target.get("status") is None else str(beam_target["status"])


def _looks_like_pf1000_akel_scope(
    declared_scope: str,
    device_name: str | None,
) -> bool:
    haystack = f"{declared_scope} {device_name or ''}".lower()
    return "pf1000" in haystack or "pf-1000" in haystack or "akel" in haystack
