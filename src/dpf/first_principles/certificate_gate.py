"""Fail-closed first-principles certificate gate packets."""

from __future__ import annotations

import hashlib
import json
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

SS19_REQUIRED_SOURCE_PACKET_HASH_IDS = (
    "ss14",
    "ss16",
    "ss17",
    "ss18",
)

SS19_COMPARATOR_MAPPING_FIELDS = (
    "output_path",
    "source_target_id",
    "metric",
    "unit_conversion",
    "time_alignment",
    "tolerance_id",
)

SS19_UNCERTAINTY_BUDGET_FIELDS = (
    "measurement_uncertainty",
    "model_uncertainty",
    "numerical_uncertainty",
    "propagation_method",
    "observable_uncertainties",
)

SS19_NEGATIVE_CONTROLS = (
    "draft_evidence",
    "blocked_evidence",
    "cross_scope_evidence",
    "missing_uq",
    "missing_review",
    "hidden_limiter",
    "app_only_or_reduced_model_fallback",
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


def build_ss19_certificate_pipeline(
    *,
    declared_scope: str,
    device_name: str | None = None,
    run_manifest_hash: str | None = None,
    source_packet_hashes: Mapping[str, str] | None = None,
    comparator_mapping: Mapping[str, Mapping[str, Any]] | None = None,
    uncertainty_budget: Mapping[str, Any] | None = None,
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
    negative_controls: Mapping[str, bool] | None = None,
    review_certificate: Mapping[str, Any] | None = None,
    synthetic_complete_fixture: bool = False,
) -> dict[str, Any]:
    """Evaluate the SS19 comparator/UQ/certificate pipeline fail-closed.

    The only positive emission path is an explicitly marked synthetic complete
    fixture. Production inputs, even when structurally complete, remain refused
    until a later review gate deliberately enables real acceptance.
    """

    source_hash_matrix = _ss19_source_hash_matrix(source_packet_hashes)
    comparator_matrix = _ss19_comparator_mapping_matrix(comparator_mapping)
    uncertainty_matrix = _ss19_uncertainty_budget_matrix(uncertainty_budget)
    negative_matrix = _ss19_negative_control_matrix(negative_controls)
    upstream_statuses = _upstream_statuses(upstream_packets)
    upstream_blockers = {
        name: status
        for name, status in upstream_statuses.items()
        if _status_blocks_certificate(status)
    }
    review_status = None if review_certificate is None else str(
        review_certificate.get("status", "")
    )
    review_accepted = bool(
        review_status
        and (
            review_status.startswith("accepted_synthetic_fixture")
            or review_status in {"reviewed_accepted", "accepted"}
        )
    )

    refusal_reasons: list[str] = []
    if not _ss19_hash_present(run_manifest_hash):
        refusal_reasons.append("missing_run_manifest_hash")
    if not all(item["present"] for item in source_hash_matrix.values()):
        refusal_reasons.append("incomplete_source_packet_hashes")
    if not comparator_matrix["complete"]:
        refusal_reasons.append("incomplete_comparator_mapping")
    if not uncertainty_matrix["complete"]:
        refusal_reasons.append("incomplete_uncertainty_budget")
    if not all(item["passed"] for item in negative_matrix.values()):
        refusal_reasons.append("incomplete_negative_controls")
    if upstream_blockers:
        refusal_reasons.append("blocked_upstream_packets")
    if not review_accepted:
        refusal_reasons.append("missing_or_unaccepted_review_certificate")

    stack_complete = not refusal_reasons
    certificate_kind = "synthetic_fixture" if synthetic_complete_fixture else "production"
    if stack_complete and synthetic_complete_fixture:
        status = "accepted_synthetic_complete_fixture"
        can_emit_certificate = True
    elif stack_complete:
        status = "refused_production_acceptance_disabled"
        can_emit_certificate = False
        refusal_reasons = ["production_acceptance_requires_real_review_gate"]
    else:
        status = "refused_incomplete_certificate_stack"
        can_emit_certificate = False

    artifact_payload = {
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "run_manifest_hash": run_manifest_hash,
        "source_packet_hashes": dict(source_packet_hashes or {}),
        "comparator_mapping": dict(comparator_mapping or {}),
        "uncertainty_budget": dict(uncertainty_budget or {}),
        "negative_controls": dict(negative_controls or {}),
        "review_certificate": dict(review_certificate or {}),
        "synthetic_complete_fixture": synthetic_complete_fixture,
    }
    certificate_artifact_hash = "sha256:" + hashlib.sha256(
        json.dumps(artifact_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return {
        "status": status,
        "certificate_kind": certificate_kind,
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "stack_complete": stack_complete,
        "can_emit_certificate": can_emit_certificate,
        "refusal_reasons": refusal_reasons,
        "run_manifest_hash": run_manifest_hash,
        "source_packet_hash_matrix": source_hash_matrix,
        "comparator_mapping_matrix": comparator_matrix,
        "uncertainty_budget_matrix": uncertainty_matrix,
        "negative_control_matrix": negative_matrix,
        "upstream_packet_statuses": upstream_statuses,
        "upstream_certificate_blockers": upstream_blockers,
        "review_certificate_status": review_status,
        "certificate_artifact_hash": certificate_artifact_hash,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }


def _ss19_hash_present(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip()
    if not normalized.startswith("sha256:"):
        return False
    digest = normalized.removeprefix("sha256:")
    return len(digest) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in digest)


def _ss19_source_hash_matrix(
    source_packet_hashes: Mapping[str, str] | None,
) -> dict[str, dict[str, Any]]:
    hashes = source_packet_hashes or {}
    return {
        source_id: {
            "hash": hashes.get(source_id),
            "present": _ss19_hash_present(hashes.get(source_id)),
            "decision": (
                "hash_present"
                if _ss19_hash_present(hashes.get(source_id))
                else "missing_or_invalid_source_packet_hash"
            ),
        }
        for source_id in SS19_REQUIRED_SOURCE_PACKET_HASH_IDS
    }


def _ss19_comparator_mapping_matrix(
    comparator_mapping: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    mapping = comparator_mapping or {}
    observable_rows: dict[str, dict[str, Any]] = {}
    for observable, row in mapping.items():
        missing = [
            field for field in SS19_COMPARATOR_MAPPING_FIELDS
            if not _ss19_nonempty(row.get(field))
        ]
        observable_rows[str(observable)] = {
            "missing_fields": missing,
            "complete": not missing,
            "decision": "mapped" if not missing else "incomplete_mapping",
        }
    complete = bool(observable_rows) and all(
        row["complete"] for row in observable_rows.values()
    )
    return {
        "complete": complete,
        "required_fields": list(SS19_COMPARATOR_MAPPING_FIELDS),
        "observable_rows": observable_rows,
    }


def _ss19_uncertainty_budget_matrix(
    uncertainty_budget: Mapping[str, Any] | None,
) -> dict[str, Any]:
    budget = uncertainty_budget or {}
    missing = [
        field for field in SS19_UNCERTAINTY_BUDGET_FIELDS
        if not _ss19_nonempty(budget.get(field))
    ]
    return {
        "complete": not missing,
        "required_fields": list(SS19_UNCERTAINTY_BUDGET_FIELDS),
        "missing_fields": missing,
        "decision": "complete" if not missing else "incomplete_uncertainty_budget",
    }


def _ss19_negative_control_matrix(
    negative_controls: Mapping[str, bool] | None,
) -> dict[str, dict[str, Any]]:
    controls = negative_controls or {}
    return {
        control: {
            "passed": controls.get(control) is True,
            "decision": (
                "negative_control_passed"
                if controls.get(control) is True
                else "missing_required_negative_control"
            ),
        }
        for control in SS19_NEGATIVE_CONTROLS
    }


def _ss19_nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (str, bytes, bytearray)):
        return bool(value.strip() if isinstance(value, str) else value)
    return True


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
