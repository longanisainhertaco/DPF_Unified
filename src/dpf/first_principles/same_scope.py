"""Fail-closed same-scope source packets for first-principles DPF claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.first_principles.channel_state import (
    ACCEPTED,
    AKEL_16KV_SHOT_MARKERS,
    BLOCKED_MISSING_SOURCE,
    EXCLUDED_NOT_VALIDATED,
    NOT_CLAIMED,
    ChannelState,
    channel_state_map,
    channel_state_summary,
    looks_like_pf1000_akel_16kv_scope,
)

# SS12-P0-3 (closes audit SS11-A3): ``SAME_SCOPE_SOURCE_REFS`` carries ONLY
# strict same-scope evidence -- reviewed source references whose scope exactly
# matches the declared runtime scope.  The two PF-1000 interferometry/anisotropy
# references previously held here carried roles ending in ``_other_scope``:
# they describe different PF-1000 campaigns (full-scale 450-500 kJ at 3.5 Torr
# and a separate interferometry shot set) than the Akel 16 kV scope, so they are
# genuinely cross-scope.  They are relocated -- not deleted -- to
# ``CROSS_SCOPE_CONTEXT_SOURCE_REFS`` and emitted under the sibling
# ``cross_scope_context_sources`` runtime field, never under ``same_scope``.
SAME_SCOPE_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "108-142,256-333,862-889",
        "role": "pf1000_akel_16kv_current_yield_reference_candidate",
    },
)

# SS12-P0-3 (closes audit SS11-A3): cross-scope context source references.
# These PF-1000 references are NOT same-scope evidence for any declared runtime
# scope -- they belong to different campaigns/operating points.  They are
# retained in full as diagnostic-requirement and schema context and emitted by
# the runner under the sibling, non-``same_scope``-named field
# ``cross_scope_context_sources``.  They promote nothing.
CROSS_SCOPE_CONTEXT_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md"
        ),
        "lines": "129-131,162-176",
        "role": "pf1000_spatial_density_cross_scope_context",
        "scope": "cross_scope_context_not_same_scope",
        "usable_for": "density_history_diagnostic_requirement_and_schema_only",
        "can_support_first_principles_acceptance": False,
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "121-137,175-177,269-275",
        "role": "pf1000_neutron_transport_detector_cross_scope_context",
        "scope": "cross_scope_context_not_same_scope",
        "usable_for": "detector_response_and_neutron_transport_requirement_schema_only",
        "can_support_first_principles_acceptance": False,
    },
)

# SS11-3 (audit S10-A3): the LLNL-like hybrid-PIC paper is ARCHITECTURE /
# equation-method / schema-context evidence -- it is NOT a same-scope PF-1000
# source.  It MUST NOT live under ``SAME_SCOPE_SOURCE_REFS`` nor be emitted
# under any ``same_scope``-named runtime field, because that namespace blurs
# selected-machine same-scope evidence with other-scope architecture context.
# The reference is retained here in full as architecture context and is emitted
# under the clearly non-same-scope field ``architecture_or_schema_context_sources``.
ARCHITECTURE_OR_SCHEMA_CONTEXT_SOURCES = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "1220-1266",
        "role": "hybrid_pic_architecture_order_of_magnitude_other_scope",
        "scope": "other_scope_architecture_or_schema_context_not_same_scope",
        "usable_for": "architecture_and_closure_gap_requirements_or_schema_only",
        "can_support_first_principles_acceptance": False,
    },
)

REQUIRED_SAME_SCOPE_CHANNELS = (
    "declared_validation_scope",
    "device_geometry_and_electrode_dimensions",
    "bank_circuit_drive",
    "gas_species_pressure_temperature",
    "accepted_digitized_current_waveform",
    "startup_breakdown_preionization",
    "density_spatial_history",
    "em_field_history",
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
    "neutron_scalar_yield",
    "neutron_timing_history",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response_and_calibration",
    "uncertainty_budget",
    "source_review_certificate",
    "cross_scope_transfer_rule_or_rejection_tests",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "device_geometry_and_electrode_dimensions",
    "bank_circuit_drive",
    "gas_species_pressure_temperature",
    "peak_current_scalar",
    "pinch_geometry_lee_output",
    "neutron_scalar_yield",
    "neutron_detector_layout_text",
    "timing_uncertainty_text",
)

BLOCKING_SAME_SCOPE_CHANNELS = (
    "accepted_digitized_current_waveform",
    "startup_breakdown_preionization",
    "density_spatial_history",
    "em_field_history",
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
    "neutron_timing_history",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response_and_calibration",
    "uncertainty_budget",
    "source_review_certificate",
    "cross_scope_transfer_rule_or_rejection_tests",
)

TEMPERATURE_HISTORY_CHANNELS = (
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
)

_DIRECT_TEMPERATURE_EVIDENCE_TYPES = {
    "direct_same_scope_temperature_diagnostic",
    "direct_same_scope_spectroscopic_temperature_history",
    "direct_same_scope_ion_distribution_measurement",
}

_ACCEPTED_REVIEW_STATES = {"accepted", "reviewed", "passed"}
_ACCEPTED_UNCERTAINTY_STATES = {"accepted", "bounded", "quantified", "passed"}

TRANSFER_RULE_REQUIRED_CHANNELS = (
    "source_scope_identity",
    "target_scope_identity",
    "changed_device_or_shot_parameters",
    "observable_transfer_equations_or_bounds",
    "uncertainty_inflation_rule",
    "review_certificate",
    "negative_test_cross_scope_promotion",
)

# Acceptance-gate labels.  The Akel 16 kV / shot-12581 revision carries an
# Akel-named gate label because its text-supported scalars are Akel scalars.
# Every other scope -- including the full-energy scope
# pf1000_full_energy_27_to_40_kv -- must NOT carry Akel-named wording
# (Codex S9 P1-1): a full-energy packet has no Akel evidence, so an Akel-named
# gate label would be a scope-classification defect.
#
# SS12-P0-3 (closes audit SS11-A3): these gate strings live INSIDE the
# ``same_scope_source`` packet.  They must not carry the literal ``other_scope``
# token, because no value under a ``same_scope``-named key may name other-scope
# material.  The wording uses ``cross_scope_diagnostics`` instead; the meaning
# is unchanged (cross-scope diagnostics cannot promote whole-shot acceptance).
_AKEL_16KV_ACCEPTANCE_GATE = (
    "text_supported_pf1000_akel_scalars_and_cross_scope_diagnostics_"
    "cannot_support_whole_shot_acceptance_until_all_same_scope_targets_"
    "current_startup_density_fields_temperatures_neutrons_detector_uq_"
    "review_and_cross_scope_rejection_tests_pass"
)
_SAME_SCOPE_ACCEPTANCE_GATE = (
    "text_supported_scalars_and_cross_scope_diagnostics_"
    "cannot_support_whole_shot_acceptance_until_all_same_scope_targets_"
    "current_startup_density_fields_temperatures_neutrons_detector_uq_"
    "review_and_cross_scope_rejection_tests_pass"
)

# SS12-P0-3 (closes audit SS11-A3): cross-scope source groups.  These describe
# scope-mismatched sources kept as diagnostic-requirement / schema context.
# They are NOT same-scope evidence and are emitted only under the sibling
# ``cross_scope_context_sources`` field, never inside ``same_scope_source``.
CROSS_SCOPE_SOURCE_GROUPS = (
    {
        "name": "pf1000_interferometry_density_other_campaign",
        "scope_mismatch": "PF-1000 interferometry shot is not the Akel 16 kV, 1.05-1.2 Torr shot set.",
        "usable_for": "diagnostic requirement and density-history schema",
    },
    {
        "name": "pf1000_anisotropy_detector_other_campaign",
        "scope_mismatch": "PF-1000 anisotropy work is full-scale 450-500 kJ at 3.5 Torr, not Akel 16 kV.",
        "usable_for": "detector-response and neutron-transport requirement schema",
    },
    {
        "name": "hybrid_pic_non_pf1000_order_of_magnitude",
        "scope_mismatch": "Hybrid PIC/fluid source is 2-D/non-hollow and order-of-magnitude, not PF-1000/Akel.",
        "usable_for": "architecture and closure-gap requirements",
    },
)


def build_cross_scope_context_sources() -> dict[str, Any]:
    """Return the non-same-scope cross-scope context block.

    SS12-P0-3 (closes audit SS11-A3): cross-scope source references, scope-
    mismatched source groups, the cross-scope transfer policy, and the
    transfer-rule required-channel list are all CONTEXT, not same-scope
    evidence.  They were previously emitted inside ``build_same_scope_source_packet``
    under ``other_scope_source_groups`` / ``cross_scope_policy``, which let
    other-scope material live under a ``same_scope``-named subtree.

    They are relocated here -- preserved in full, deleted from nowhere -- so the
    runner can emit them under the sibling, non-``same_scope``-named runtime
    field ``cross_scope_context_sources``.  This block promotes nothing.
    """
    return {
        "status": "cross_scope_context_only_not_same_scope_evidence",
        "role": "cross_scope_context_and_transfer_policy_sources",
        "usable_for": "diagnostic_requirements_or_schema_only",
        "is_same_scope_validation_evidence": False,
        "source_references": list(CROSS_SCOPE_CONTEXT_SOURCE_REFS),
        "cross_scope_source_groups": list(CROSS_SCOPE_SOURCE_GROUPS),
        "cross_scope_policy": {
            "status": "blocked_without_reviewed_transfer_rule",
            "required_transfer_rule_channels": list(TRANSFER_RULE_REQUIRED_CHANNELS),
            "cross_scope_sources_usable_for": "requirements_or_schema_only",
            "can_use_cross_scope_for_acceptance": False,
        },
        "can_support_first_principles_acceptance": False,
    }


def build_same_scope_source_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_same_scope_channels: tuple[str, ...] | list[str] = (),
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Return a non-promoting packet describing same-scope source availability.

    Sprint 8 WS1 (Codex S7-A8): the ``accepted_same_scope_channels`` argument
    is a list of *manually requested* channels.  A manual channel is NOT
    evidence -- it can only become ``accepted`` when a reviewed, scope-matched
    validation target with an uncertainty budget backs it.  Manual channels
    with no such target are reported as ``requested_manual_channel_not_evidence``
    and carry the ``excluded_not_validated`` per-channel state.

    ``synthetic_fixture`` only changes labelling for test/fixture transparency;
    it never weakens acceptance and never promotes a manual channel.
    """

    # Channels backed by a reviewed, scope-matched validation target.  This is
    # the ONLY path to ``accepted``.
    target_channels, target_decisions = _accepted_channels_from_targets(
        validation_targets,
        declared_scope=declared_scope,
        device_name=device_name,
    )
    accepted: set[str] = set(target_channels)

    # Manual same-scope channel list -> *requested* channels, never evidence
    # (Codex S7-A8).  A manual channel only counts when a target above already
    # accepted it; otherwise it is recorded as a request and excluded.
    requested_manual_channels: list[str] = []
    manual_decisions: list[dict[str, Any]] = []
    for channel in accepted_same_scope_channels:
        channel_name = str(channel)
        requested_manual_channels.append(channel_name)
        backed_by_target = channel_name in accepted
        if channel_name in TEMPERATURE_HISTORY_CHANNELS:
            manual_decisions.append({
                "target": f"manual_channel:{channel_name}",
                "observable": channel_name,
                "status": "manual_requested_same_scope_channel",
                "decision": (
                    "rejected_temperature_history_requires_direct_same_scope_"
                    "diagnostic_review_and_uncertainty"
                ),
                "backed_by_reviewed_target": False,
            })
            continue
        manual_decisions.append({
            "target": f"manual_channel:{channel_name}",
            "observable": channel_name,
            "status": "manual_requested_same_scope_channel",
            "decision": (
                "accepted_via_reviewed_validation_target"
                if backed_by_target
                else "requested_manual_channel_not_evidence"
            ),
            "backed_by_reviewed_target": backed_by_target,
        })
    target_decisions = manual_decisions + target_decisions
    declared = bool(str(declared_scope).strip()) and declared_scope != "not_declared"
    if declared:
        accepted.add("declared_validation_scope")

    is_akel_16kv_scope = looks_like_pf1000_akel_16kv_scope(declared_scope, device_name)
    if is_akel_16kv_scope:
        # Akel 16 kV / shot-12581 revision: text scalars are non-acceptance
        # engineering reference only (channels stay excluded_not_validated).
        text_supported = set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
    else:
        # Codex S9 P1-1: every other scope -- including the full-energy
        # scope pf1000_full_energy_27_to_40_kv -- gets no Akel reference
        # channels.  A full-energy text-supported set is added only once KR
        # target-extraction supplies selected-scope records; until then this
        # is empty (fail-closed, selected-scope-only).
        text_supported = set()

    # Canonical per-channel states (Codex S7-A7).  A channel is in exactly one
    # state, so ``accepted`` and ``missing`` are mutually exclusive.
    requested_not_accepted = {
        name for name in requested_manual_channels if name not in accepted
    }
    channel_states = _same_scope_channel_states(
        accepted=accepted,
        text_supported=text_supported,
        requested_not_accepted=requested_not_accepted,
    )
    state_summary = channel_state_summary(channel_states)
    missing = set(state_summary["missing_acceptance_channels"])
    # SS12-P0-3 (closes audit SS11-A3): channel_state_summary.counts_by_state
    # enumerates EVERY canonical channel-state value, including
    # ``blocked_wrong_scope``.  That zero-count vocabulary key would otherwise
    # publish the ``wrong_scope`` token inside this ``same_scope_source``
    # packet, even though no same-scope channel is in that state.  Publish only
    # the states that actually occur: a zero count carries no information, and
    # if a same-scope channel ever genuinely became ``blocked_wrong_scope`` the
    # nonzero count would (correctly) surface it as a real namespace defect.
    state_summary = {
        **state_summary,
        "counts_by_state": {
            state: count
            for state, count in state_summary["counts_by_state"].items()
            if count
        },
    }

    return {
        "status": "blocked_same_scope_source_packet_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "synthetic_fixture": bool(synthetic_fixture),
        "decision": "do_not_promote_whole_shot_first_principles_claim",
        "scope_policy": (
            "single_declared_scope_only_no_cross_device_shot_or_configuration_mix "
            "without_reviewed_transfer_rule"
        ),
        "required_channels": list(REQUIRED_SAME_SCOPE_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "text_supported_not_acceptance_channels": sorted(text_supported - accepted),
        "accepted_same_scope_channels": sorted(accepted),
        "requested_manual_channels": sorted(set(requested_manual_channels)),
        "requested_manual_channels_not_evidence": sorted(requested_not_accepted),
        "missing_acceptance_channels": sorted(missing),
        "channel_states": channel_state_map(channel_states),
        "channel_state_summary": state_summary,
        "same_scope_channel_status": _channel_statuses(
            required_channels=REQUIRED_SAME_SCOPE_CHANNELS,
            accepted=accepted,
            text_supported=text_supported,
            missing=missing,
        ),
        "same_scope_target_policy": {
            "text_supported_channels_can_seed_engineering_reference": True,
            "text_supported_channels_can_support_acceptance": False,
            "accepted_targets_must_match_declared_scope": True,
            "accepted_targets_must_include_review_certificate": True,
            "cross_scope_targets_require_reviewed_transfer_rule": True,
        },
        # SS12-P0-3 (closes audit SS11-A3): cross-scope source groups and the
        # cross-scope transfer policy are NOT same-scope evidence.  They no
        # longer live inside this ``same_scope_source`` packet -- the runner
        # emits them under the sibling field ``cross_scope_context_sources``
        # (see ``build_cross_scope_context_sources``).  This pointer names the
        # sibling so consumers can find the relocated context.
        "cross_scope_context_sources_emitted_under": "cross_scope_context_sources",
        "acceptance_gate": (
            _AKEL_16KV_ACCEPTANCE_GATE
            if is_akel_16kv_scope
            else _SAME_SCOPE_ACCEPTANCE_GATE
        ),
        "negative_test_policy": {
            "text_reference_promotion_rejection_required": True,
            "cross_scope_diagnostic_promotion_rejection_required": True,
            "mismatched_shot_or_pressure_rejection_required": True,
            "missing_review_certificate_rejection_required": True,
            "missing_uncertainty_budget_rejection_required": True,
            "cross_scope_without_transfer_rule_rejection_required": True,
        },
        "validation_target_scope_decisions": target_decisions,
        # SS11-3 (audit S10-A3): ``source_references`` carries only same-device
        # PF-1000 same-scope references.  The LLNL-like hybrid-PIC architecture
        # reference is NOT same-scope and is emitted by the runner under the
        # sibling, non-``same_scope``-named runtime field
        # ``architecture_or_schema_context_sources``
        # (see ARCHITECTURE_OR_SCHEMA_CONTEXT_SOURCES) -- never under this
        # ``same_scope``-named packet.
        "source_references": list(SAME_SCOPE_SOURCE_REFS),
        "validation_target_count": len(validation_targets),
        "can_support_first_principles_acceptance": False,
    }


def _accepted_channels_from_targets(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    declared_scope: str,
    device_name: str | None,
) -> tuple[set[str], list[dict[str, Any]]]:
    accepted: set[str] = set()
    decisions: list[dict[str, Any]] = []
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
                "decision": "not_accepted_same_scope_status",
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
        if observable:
            evidence_type = str(target.get("evidence_type", "")).strip()
            if observable in TEMPERATURE_HISTORY_CHANNELS and not (
                evidence_type in _DIRECT_TEMPERATURE_EVIDENCE_TYPES
                and _target_has_review_and_uncertainty(target)
            ):
                decisions.append({
                    "target": name,
                    "observable": observable,
                    "status": status,
                    "decision": (
                        "rejected_temperature_history_requires_direct_same_scope_"
                        "diagnostic_review_and_uncertainty"
                    ),
                    "evidence_type": evidence_type or "not_declared",
                })
                continue
            if evidence_type == "lee_model_output" and observable in BLOCKING_SAME_SCOPE_CHANNELS:
                # Lee model outputs are NOT independent measurements.
                # They cannot satisfy blocking same-scope channels.
                # WP-N7 §6.1, N7-NEG-10.
                decisions.append({
                    "target": name,
                    "observable": observable,
                    "status": status,
                    "decision": "rejected_lee_model_output_not_independent_measurement",
                    "evidence_type": evidence_type,
                })
                continue
            accepted.add(observable)
            decisions.append({
                "target": name,
                "observable": observable,
                "status": status,
                "decision": "accepted_same_scope_target_channel",
            })
    return accepted, decisions


def _target_has_review_and_uncertainty(target: Mapping[str, Any]) -> bool:
    review_state = str(
        target.get("review_certificate_status")
        or target.get("review_status")
        or ""
    ).strip().lower()
    uncertainty_state = str(
        target.get("uncertainty_status")
        or target.get("uncertainty_budget_status")
        or ""
    ).strip().lower()
    return (
        review_state in _ACCEPTED_REVIEW_STATES
        and uncertainty_state in _ACCEPTED_UNCERTAINTY_STATES
    )


def _channel_statuses(
    *,
    required_channels: tuple[str, ...],
    accepted: set[str],
    text_supported: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in required_channels:
        if channel in accepted:
            statuses[channel] = "accepted_same_scope"
        elif channel in text_supported:
            statuses[channel] = "text_supported_reference_only_not_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _same_scope_channel_states(
    *,
    accepted: set[str],
    text_supported: set[str],
    requested_not_accepted: set[str],
) -> dict[str, ChannelState]:
    """Map every required same-scope channel onto a canonical state.

    - ``accepted`` -- backed by a reviewed, scope-matched validation target.
    - ``blocked_missing_source`` -- a blocking same-scope channel with no
      independent source packet.  Includes
      ``cross_scope_transfer_rule_or_rejection_tests``: an absent transfer
      rule / rejection-test artifact is a missing same-scope source, so it
      blocks acceptance as a missing-source channel.
    - ``excluded_not_validated`` -- a manually requested channel that is not
      backed by a reviewed target (Codex S7-A8), or a text-supported reference
      channel which is explicitly not acceptance evidence.
    - ``not_claimed`` -- no evidence offered and not otherwise constrained.

    SS12-P0-3 (closes audit SS11-A3): the
    ``cross_scope_transfer_rule_or_rejection_tests`` channel previously used
    ``blocked_wrong_scope``.  That published the ``wrong_scope`` token inside
    the ``same_scope_source`` packet's ``channel_states`` map.  The transfer
    rule is an absent source artifact, not wrong-scope material present, so the
    accurate fail-closed state is ``blocked_missing_source`` -- it reaches that
    state through ``BLOCKING_SAME_SCOPE_CHANNELS`` like every other blocking
    channel.  Both states block acceptance; this only sharpens the label and
    keeps the same-scope namespace free of the ``wrong_scope`` token.
    """

    states: dict[str, ChannelState] = {}
    for channel in REQUIRED_SAME_SCOPE_CHANNELS:
        if channel in accepted:
            states[channel] = ACCEPTED
        elif channel in requested_not_accepted:
            # Manual request that no reviewed target backed.
            states[channel] = EXCLUDED_NOT_VALIDATED
        elif channel in text_supported:
            # Text scalars seed an engineering reference only.
            states[channel] = EXCLUDED_NOT_VALIDATED
        elif channel in BLOCKING_SAME_SCOPE_CHANNELS:
            states[channel] = BLOCKED_MISSING_SOURCE
        else:
            states[channel] = NOT_CLAIMED
    return states


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
        if looks_like_pf1000_akel_16kv_scope(declared_scope, device_name):
            return "akel" in haystack and any(
                marker in haystack for marker in AKEL_16KV_SHOT_MARKERS
            )
    return False


def _normalized_scope(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
