"""Fail-closed first-principles certificate gate packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def build_first_principles_certificate_gate_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    accepted_channels: tuple[str, ...] | list[str] = (),
    upstream_packets: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting first-principles certificate gate packet."""

    accepted = {str(channel) for channel in accepted_channels}
    missing = set(REQUIRED_CERTIFICATE_CHANNELS) - accepted
    missing.update(REQUIRED_CERTIFICATE_CHANNELS)
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
        "upstream_packet_statuses": upstream_statuses,
        "upstream_certificate_blockers": upstream_blockers,
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
