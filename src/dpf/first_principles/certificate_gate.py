"""Fail-closed first-principles certificate gate packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.first_principles.channel_state import (
    ACCEPTED,
    BLOCKED_MISSING_SOURCE,
    ChannelState,
    channel_state_map,
    channel_state_summary,
)

CERTIFICATE_GATE_SOURCE_REFS = (
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "lines": "82-148,170-171,646-673",
        "role": "first_principles_certificate_claim_states_and_payload_requirements",
    },
    {
        "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
        "lines": "82-84",
        "role": "first_principles_certificate_and_fail_closed_artifact_requirements",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "20",
        "role": "source_ingestion_does_not_make_validation_target",
    },
    {
        "path": "src/dpf/validation/artifacts.py",
        "lines": "398-445",
        "role": "existing_fail_closed_certificate_schema_reference",
    },
    {
        "path": "tests/test_validation_artifacts.py",
        "lines": "531-666",
        "role": "existing_negative_certificate_tests_reference",
    },
)

REQUIRED_CERTIFICATE_CHANNELS = (
    "run_manifest_hash",
    "evidence_packet_hashes",
    "validation_scope_and_source_scope",
    "package_native_execution_proof",
    "same_scope_source_packet_accepted",
    "waveform_phase_packet_accepted",
    "spatial_field_temperature_packet_accepted",
    "neutron_authority_packet_accepted",
    "comparator_uq_packet_accepted",
    "numerical_fidelity_packet_accepted",
    "physics_closure_packet_accepted",
    "limiter_zero_or_physical_bounds_packet",
    "power_port_packet_accepted",
    "startup_packet_accepted",
    "dimensionality_handoff_packet_accepted",
    "reduced_model_rejection_proof",
    "reviewer_metadata",
    "accepted_review_status",
    "comparator_metrics_and_uq_ids",
    "requirement_links",
    "commands_and_versions",
    "release_label",
    "release_decision",
    "negative_test_draft_evidence",
    "negative_test_blocked_evidence",
    "negative_test_cross_scope_evidence",
    "negative_test_missing_uq",
    "negative_test_missing_review",
    "negative_test_hidden_limiter",
    "negative_test_app_only_or_reduced_model_fallback",
    "certificate_artifact_hash",
)

BLOCKING_UPSTREAM_STATUSES = (
    "blocked",
    "blocked_by_review",
    "not_validation",
    "candidate",
    "candidate_engineering_power_port_not_validation",
    "candidate_engineering_dimensionality_handoff_not_validation",
    "candidate_engineering_closure_packet_not_validation",
    "rejected_startup_mode_for_first_principles",
    "blocked_same_scope_source_packet_not_available",
    "blocked_waveform_phase_packet_not_available",
    "blocked_spatial_field_temperature_packet_not_available",
    "blocked_mechanism_separated_neutron_authority_not_available",
    "blocked_comparator_uq_matrix_not_available",
)

REQUIRED_UPSTREAM_PACKET_CHANNELS = {
    "same_scope_source_packet_accepted": "same_scope_source",
    "waveform_phase_packet_accepted": "waveform_phase",
    "spatial_field_temperature_packet_accepted": "spatial_field_temperature",
    "neutron_authority_packet_accepted": "neutron_authority",
    "comparator_uq_packet_accepted": "comparator_uq",
    "numerical_fidelity_packet_accepted": "numerical_fidelity",
    "physics_closure_packet_accepted": "physics_closure",
    "limiter_zero_or_physical_bounds_packet": "limiter_readiness",
    "power_port_packet_accepted": "power_port",
    "startup_packet_accepted": "startup_bvp",
    "dimensionality_handoff_packet_accepted": "dimensionality_handoff",
}

REQUIRED_NEGATIVE_TEST_CHANNELS = (
    "negative_test_draft_evidence",
    "negative_test_blocked_evidence",
    "negative_test_cross_scope_evidence",
    "negative_test_missing_uq",
    "negative_test_missing_review",
    "negative_test_hidden_limiter",
    "negative_test_app_only_or_reduced_model_fallback",
)


def build_first_principles_certificate_gate_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting first-principles certificate gate packet."""

    accepted = {str(channel) for channel in accepted_channels}
    # Canonical per-channel states (Codex S7-A7): each certificate channel is
    # either ``accepted`` or ``blocked_missing_source``.  The old code
    # unconditionally re-added every channel to ``missing``, so an accepted
    # channel was also reported missing -- that contradiction is now gone.
    channel_states = _certificate_channel_states(accepted)
    state_summary = channel_state_summary(channel_states)
    missing = set(state_summary["missing_acceptance_channels"])
    upstream_statuses = _upstream_statuses(upstream_packets)
    upstream_blockers = {
        name: status
        for name, status in upstream_statuses.items()
        if _status_blocks_certificate(status)
    }

    return {
        "status": "blocked_first_principles_certificate_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_write_accepted_first_principles_certificate",
        "release_label": "engineering_candidate_not_releasable_for_first_principles_claim",
        "required_channels": list(REQUIRED_CERTIFICATE_CHANNELS),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "channel_states": channel_state_map(channel_states),
        "channel_state_summary": state_summary,
        "certificate_channel_status": _certificate_channel_statuses(
            accepted=accepted,
            missing=missing,
        ),
        "release_decision": "do_not_release_first_principles_claim",
        "acceptance_policy": {
            "all_certificate_channels_required": True,
            "all_upstream_packets_must_be_accepted": True,
            "draft_candidate_blocked_or_rejected_packets_block_release": True,
            "reduced_model_or_app_only_evidence_blocks_release": True,
            "cross_scope_evidence_blocks_release_without_reviewed_transfer_rule": True,
        },
        "upstream_packet_statuses": upstream_statuses,
        "upstream_certificate_blockers": upstream_blockers,
        "upstream_packet_acceptance_matrix": _upstream_packet_acceptance_matrix(
            upstream_statuses
        ),
        "negative_test_matrix": _negative_test_matrix(accepted),
        "source_references": list(CERTIFICATE_GATE_SOURCE_REFS),
        "can_write_accepted_certificate": False,
        "can_release_first_principles_claim": False,
        "can_support_first_principles_acceptance": False,
    }


def _upstream_statuses(
    upstream_packets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    for name, packet in (upstream_packets or {}).items():
        statuses[str(name)] = (
            None if packet.get("status") is None else str(packet["status"])
        )
    return statuses


def _certificate_channel_states(accepted: set[str]) -> dict[str, ChannelState]:
    """Map every required certificate channel onto a canonical state.

    A channel is ``accepted`` only when the deck declares it; otherwise it is
    ``blocked_missing_source``.  Top-level certificate acceptance is hard-coded
    False regardless -- this map only makes the per-channel accounting honest.
    """

    return {
        channel: (ACCEPTED if channel in accepted else BLOCKED_MISSING_SOURCE)
        for channel in REQUIRED_CERTIFICATE_CHANNELS
    }


def _certificate_channel_statuses(
    *,
    accepted: set[str],
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_CERTIFICATE_CHANNELS:
        if channel in accepted:
            statuses[channel] = "accepted_certificate_channel"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "not_available"
    return statuses


def _upstream_packet_acceptance_matrix(
    upstream_statuses: Mapping[str, str | None],
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for channel, packet_name in REQUIRED_UPSTREAM_PACKET_CHANNELS.items():
        status = upstream_statuses.get(packet_name)
        accepted = _status_is_accepted_for_certificate(status)
        matrix[channel] = {
            "packet": packet_name,
            "upstream_status": status,
            "accepted_for_certificate": accepted,
            "decision": "accepted" if accepted else "missing_or_blocking_upstream_packet",
        }
    return matrix


def _negative_test_matrix(accepted: set[str]) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for channel in REQUIRED_NEGATIVE_TEST_CHANNELS:
        present = channel in accepted
        matrix[channel] = {
            "present": present,
            "decision": "accepted" if present else "missing_required_negative_test",
        }
    return matrix


def _status_blocks_certificate(status: str | None) -> bool:
    if status is None:
        return True
    normalized = status.strip().lower()
    if normalized in BLOCKING_UPSTREAM_STATUSES:
        return True
    return (
        normalized.startswith("blocked")
        or normalized.startswith("candidate")
        or normalized.startswith("rejected")
    )


def _status_is_accepted_for_certificate(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().lower()
    return normalized.startswith("accepted") or normalized in {
        "ready",
        "passed",
        "reviewed_accepted",
    }
