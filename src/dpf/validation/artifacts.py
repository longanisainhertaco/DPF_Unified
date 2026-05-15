"""Traceable validation artifacts for SRS-grade result governance."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ResultLabel(str, Enum):
    """User-facing result authority labels."""

    REFERENCE = "Reference"
    PREVIEW = "Preview"
    DERIVED_DIAGNOSTIC = "Derived Diagnostic"
    EXPLORATORY = "Exploratory"
    SUPERSEDED = "Superseded"
    INVALID = "Invalid"


class ValidationStatus(str, Enum):
    """Validation state carried by manifests and certificates."""

    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class BackendAuthority(str, Enum):
    """Backend authority class before evidence-specific result classification."""

    REFERENCE_CANDIDATE = "reference_candidate"
    PREVIEW = "preview"


REFERENCE_CANDIDATE_BACKENDS = frozenset({"python", "athena", "athenak"})
PREVIEW_BACKENDS = frozenset({"metal", "mlx", "hybrid", "auto"})

_VALIDATION_EVIDENCE_SUMMARY_KEYS = (
    "pf1000_16kv_current_waveform_comparison_candidate",
    "pf1000_16kv_current_waveform_digitization",
    "scientific_closure_digitization_status",
    "digitization_status",
    "scientific_closure_source_acquisition_queue",
    "scientific_closure_digitization_queue",
    "predictive_readiness",
    "high_fidelity_readiness",
    "readiness_scope",
    "source_blockers",
    "mhd_numerical_verification_packet_status",
    "mhd_numerical_fidelity",
    "spatial_validation_scope_closure",
    "neutron_validation_scope_closure",
    "validation_uncertainty_coverage",
    "uncertainty_validation",
    "first_principles_limiter_ledger",
    "first_principles_backend_scope",
)

_COMPACT_EVIDENCE_FIELDS = (
    "passed",
    "validation_status",
    "waveform_comparison_status",
    "waveform_digitization_status",
    "production_packet_status",
    "status",
    "ready",
    "validation_scope",
    "model_role",
    "validation_tier",
    "source",
    "backend",
    "requested_backend",
    "requested_run_mode",
    "blocked_backend",
    "required_limiter_telemetry",
    "reason",
    "schema",
    "event_count",
    "entry_count",
    "activation_count",
    "acceptance_blocking_activation_count",
    "activated_acceptance_blockers",
    "by_classification",
    "can_support_first_principles_acceptance",
    "source_lines",
    "missing_required_packets",
    "missing_or_unvalidated_evidence",
    "missing_or_unvalidated_components",
    "missing_uncertainty_observables",
    "validation_blockers",
    "blockers",
    "source_blockers",
)


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for artifact metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_hash(payload: Any) -> str:
    """Hash JSON-serializable payloads deterministically."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for_output(output_path: str | Path) -> Path:
    """Return the sidecar manifest path for a primary run output."""

    path = Path(output_path)
    return path.with_name(f"{path.name}.run_manifest.json")


def _config_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    return dict(config)


def _artifact_classification_from_config_payload(
    config_payload: Mapping[str, Any],
) -> ArtifactClassification:
    diagnostics = config_payload.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    return ArtifactClassification(
        owner=diagnostics.get("artifact_owner") or None,
        classification=(
            diagnostics.get("artifact_classification") or "owner_unspecified"
        ),
        distribution=(
            diagnostics.get("artifact_distribution") or "owner_unspecified"
        ),
        handling_notes=diagnostics.get("artifact_handling_notes") or "",
    )


def artifact_classification_from_config(config: Any) -> ArtifactClassification:
    """Extract owner-supplied artifact classification from run config."""

    return _artifact_classification_from_config_payload(_config_payload(config))


def embed_hdf5_run_metadata(
    output_path: str | Path,
    *,
    backend: str,
    solver_mode: str,
    validation_status: ValidationStatus | str,
    result_classification: ResultClassification,
    artifact_classification: ArtifactClassification | None = None,
    summary: Mapping[str, Any] | None = None,
) -> bool:
    """Embed run-governance metadata in a HDF5 diagnostics file."""

    path = Path(output_path)
    if str(output_path) == ":memory:" or not path.exists():
        return False

    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        return False

    status = ValidationStatus(validation_status)
    classification = artifact_classification or ArtifactClassification()
    with h5py.File(path, "a") as handle:
        handle.attrs["dpf_backend"] = backend
        handle.attrs["dpf_solver_mode"] = solver_mode
        handle.attrs["dpf_validation_status"] = status.value
        handle.attrs["dpf_result_label"] = result_classification.label.value
        handle.attrs["dpf_can_support_validation_claims"] = (
            result_classification.can_support_validation_claims
        )
        handle.attrs["dpf_result_classification_json"] = json.dumps(
            result_classification.model_dump(mode="json"),
            sort_keys=True,
        )
        handle.attrs["dpf_artifact_classification_json"] = json.dumps(
            classification.model_dump(mode="json"),
            sort_keys=True,
        )
        handle.attrs["dpf_source_authority"] = "local KnowledgeReference only for validation claims"
        validation_evidence = _validation_evidence_from_summary(
            dict(summary) if isinstance(summary, Mapping) else None
        )
        if validation_evidence:
            evidence_json = json.dumps(
                validation_evidence,
                sort_keys=True,
                default=str,
            )
            handle.attrs["dpf_validation_evidence_json"] = evidence_json
            handle.attrs["dpf_readiness_summary_json"] = evidence_json
    return True


def backend_authority_for(backend: str) -> BackendAuthority:
    """Return the default authority class for a backend name."""

    normalized = backend.lower().strip()
    if normalized in REFERENCE_CANDIDATE_BACKENDS:
        return BackendAuthority.REFERENCE_CANDIDATE
    return BackendAuthority.PREVIEW


class ResultClassification(BaseModel):
    """Fail-closed classification for a simulation result or derived artifact."""

    label: ResultLabel
    reason: str
    backend_authority: BackendAuthority = BackendAuthority.PREVIEW
    validation_status: ValidationStatus = ValidationStatus.NOT_EVALUATED
    can_support_validation_claims: bool = False
    requirement_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_reference_rules(self) -> ResultClassification:
        if self.can_support_validation_claims and self.label is not ResultLabel.REFERENCE:
            raise ValueError("only Reference results may support validation claims")
        if self.label is ResultLabel.REFERENCE:
            if self.backend_authority is not BackendAuthority.REFERENCE_CANDIDATE:
                raise ValueError("Reference label requires a reference-candidate backend")
            if self.validation_status is not ValidationStatus.ACCEPTED:
                raise ValueError("Reference label requires accepted validation status")
            self.can_support_validation_claims = True
        else:
            self.can_support_validation_claims = False
        return self


def classify_result(
    *,
    backend: str,
    validation_status: ValidationStatus | str = ValidationStatus.NOT_EVALUATED,
    reason: str | None = None,
) -> ResultClassification:
    """Classify a result from backend and validation state."""

    status = ValidationStatus(validation_status)
    authority = backend_authority_for(backend)
    if authority is BackendAuthority.REFERENCE_CANDIDATE and status is ValidationStatus.ACCEPTED:
        return ResultClassification(
            label=ResultLabel.REFERENCE,
            backend_authority=authority,
            validation_status=status,
            reason=reason or "reference-candidate backend with accepted evidence",
        )
    return ResultClassification(
        label=ResultLabel.PREVIEW,
        backend_authority=authority,
        validation_status=status,
        reason=reason or "not accepted as Reference evidence",
    )


class ManifestOutput(BaseModel):
    """One output artifact emitted by a run."""

    path: str
    kind: str
    sha256: str | None = None


class ArtifactClassification(BaseModel):
    """Owner-supplied classification and distribution metadata for artifacts."""

    owner: str | None = None
    classification: str = "owner_unspecified"
    distribution: str = "owner_unspecified"
    handling_notes: str = ""


class RunManifest(BaseModel):
    """Run provenance manifest schema."""

    manifest_version: Literal["1.0"] = "1.0"
    run_id: str
    created_utc: str = Field(default_factory=utc_now_iso)
    config_hash: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    backend: str
    solver_mode: str
    precision: str | None = None
    hardware_profile: dict[str, str] = Field(default_factory=dict)
    dependency_hashes: dict[str, str] = Field(default_factory=dict)
    summary_hash: str | None = None
    seed: int | None = None
    artifact_classification: ArtifactClassification = Field(default_factory=ArtifactClassification)
    outputs: list[ManifestOutput] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_EVALUATED
    result_classification: ResultClassification
    requirement_ids: list[str] = Field(default_factory=list)
    validation_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_manifest_consistency(self) -> RunManifest:
        if self.result_classification.validation_status is not self.validation_status:
            raise ValueError("manifest validation_status must match result classification")
        if (
            self.result_classification.label is ResultLabel.REFERENCE
            and self.validation_status is not ValidationStatus.ACCEPTED
        ):
            raise ValueError("Reference manifests require accepted validation status")
        return self


def _compact_validation_evidence(value: Any) -> Any:
    """Keep validation manifest evidence small and blocker-oriented."""

    if isinstance(value, Mapping):
        compact = {
            field: value[field]
            for field in _COMPACT_EVIDENCE_FIELDS
            if field in value
        }
        details = value.get("details")
        if isinstance(details, Mapping):
            detail_compact = {
                field: details[field]
                for field in _COMPACT_EVIDENCE_FIELDS
                if field in details
            }
            digitization = details.get("digitization_readiness")
            if isinstance(digitization, Mapping):
                detail_compact["digitization_readiness"] = {
                    field: digitization[field]
                    for field in _COMPACT_EVIDENCE_FIELDS
                    if field in digitization
                }
            if detail_compact:
                compact["details"] = detail_compact
        return compact or dict(value)
    if isinstance(value, list):
        compacted = []
        for item in value:
            if isinstance(item, Mapping):
                compacted.append(_compact_validation_evidence(item))
            elif isinstance(item, (str, int, float, bool)) or item is None:
                compacted.append(item)
        return compacted
    return value


def _validation_evidence_from_summary(
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    evidence: dict[str, Any] = {}
    existing = summary.get("validation_evidence")
    if isinstance(existing, Mapping):
        evidence.update({
            str(key): _compact_validation_evidence(value)
            for key, value in existing.items()
        })
    for key in _VALIDATION_EVIDENCE_SUMMARY_KEYS:
        if key in summary:
            evidence[key] = _compact_validation_evidence(summary[key])
    return evidence


class CertificateEvidenceLink(BaseModel):
    """Evidence item linked to a validation certificate."""

    requirement_id: str
    evidence_uri: str
    status: ValidationStatus
    validation_scope: str
    notes: str = ""


class ValidationCertificate(BaseModel):
    """Validation certificate schema with fail-closed acceptance rules."""

    certificate_version: Literal["1.0"] = "1.0"
    certificate_id: str
    created_utc: str = Field(default_factory=utc_now_iso)
    requirement_ids: list[str]
    validation_scope: str
    validation_status: ValidationStatus
    result_label: ResultLabel
    result_classification: dict[str, Any] = Field(default_factory=dict)
    artifact_classification: ArtifactClassification = Field(default_factory=ArtifactClassification)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    evidence_links: list[CertificateEvidenceLink]
    reviewers: list[str] = Field(default_factory=list)
    review_status: Literal["accepted", "blocked", "not_required", "rejected"] = "blocked"
    supersedes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_certificate_rules(self) -> ValidationCertificate:
        if self.result_classification:
            label = self.result_classification.get("label")
            if label is not None and label != self.result_label.value:
                raise ValueError("certificate result label must match result classification")
        if self.validation_status is ValidationStatus.ACCEPTED:
            if self.blockers:
                raise ValueError("accepted certificates cannot carry blockers")
            if (
                self.result_classification
                and self.result_classification.get("can_support_validation_claims") is False
            ):
                raise ValueError(
                    "accepted certificates require validation-supporting classification"
                )
            if self.result_label is not ResultLabel.REFERENCE:
                raise ValueError("accepted certificates require Reference result label")
            if self.review_status != "accepted":
                raise ValueError("accepted certificates require accepted review status")
            if not self.reviewers:
                raise ValueError("accepted certificates require at least one reviewer")
            if not self.evidence_links:
                raise ValueError("accepted certificates require evidence links")
            for evidence in self.evidence_links:
                if evidence.status is not ValidationStatus.ACCEPTED:
                    raise ValueError("accepted certificates cannot include blocked or failed evidence")
                if evidence.validation_scope != self.validation_scope:
                    raise ValueError("accepted certificates require same-scope evidence")
        return self


def build_run_manifest(
    *,
    config: Any,
    backend: str,
    summary: dict[str, Any] | None = None,
    artifact_classification: ArtifactClassification | dict[str, Any] | None = None,
    validation_status: ValidationStatus | str = ValidationStatus.NOT_EVALUATED,
    reason: str | None = None,
    outputs: list[ManifestOutput] | None = None,
    requirement_ids: list[str] | None = None,
) -> RunManifest:
    """Build a fail-closed run manifest from a config and optional summary."""

    config_payload = _config_payload(config)

    status = ValidationStatus(validation_status)
    classification = classify_result(
        backend=backend,
        validation_status=status,
        reason=reason,
    )
    req_ids = list(requirement_ids or ["DPF-DATA-001", "DPF-DATA-002"])
    output_records = list(outputs or [])

    diagnostics = config_payload.get("diagnostics", {})
    hdf5_filename = diagnostics.get("hdf5_filename")
    if hdf5_filename and hdf5_filename != ":memory:":
        hdf5_path = Path(str(hdf5_filename))
        if hdf5_path.exists():
            output_records.append(
                ManifestOutput(
                    path=str(hdf5_path),
                    kind="hdf5_diagnostics",
                    sha256=file_sha256(hdf5_path),
                )
            )

    fluid = config_payload.get("fluid", {})
    geometry = config_payload.get("geometry", {})
    config_hash = stable_json_hash(config_payload)
    return RunManifest(
        run_id=f"run-{uuid.uuid4().hex}",
        config_hash=config_hash,
        input_hashes={"config_sha256": config_hash},
        backend=backend,
        solver_mode=f"{geometry.get('type', 'cartesian')}_mhd",
        precision=fluid.get("precision"),
        hardware_profile={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        summary_hash=stable_json_hash(summary) if summary is not None else None,
        artifact_classification=(
            ArtifactClassification.model_validate(artifact_classification)
            if artifact_classification is not None
            else _artifact_classification_from_config_payload(config_payload)
        ),
        outputs=output_records,
        validation_status=status,
        result_classification=classification,
        requirement_ids=req_ids,
        validation_evidence=_validation_evidence_from_summary(summary),
    )


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write a run manifest as deterministic, reviewer-readable JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return out_path


def build_validation_certificate(
    *,
    certificate_id: str,
    requirement_ids: list[str],
    validation_scope: str,
    evidence_links: list[CertificateEvidenceLink],
    result_label: ResultLabel,
    validation_status: ValidationStatus | str,
    result_classification: dict[str, Any] | None = None,
    artifact_classification: ArtifactClassification | dict[str, Any] | None = None,
    readiness_summary: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    reviewers: list[str] | None = None,
    review_status: Literal["accepted", "blocked", "not_required", "rejected"] = "blocked",
    supersedes: list[str] | None = None,
) -> ValidationCertificate:
    """Build a certificate through the fail-closed certificate model."""

    return ValidationCertificate(
        certificate_id=certificate_id,
        requirement_ids=requirement_ids,
        validation_scope=validation_scope,
        validation_status=ValidationStatus(validation_status),
        result_label=result_label,
        result_classification=dict(result_classification or {}),
        artifact_classification=(
            ArtifactClassification.model_validate(artifact_classification)
            if artifact_classification is not None
            else ArtifactClassification()
        ),
        readiness_summary=dict(readiness_summary or {}),
        blockers=list(blockers or []),
        evidence_links=evidence_links,
        reviewers=list(reviewers or []),
        review_status=review_status,
        supersedes=list(supersedes or []),
    )


def write_validation_certificate(certificate: ValidationCertificate, path: str | Path) -> Path:
    """Write a validation certificate that has already passed model validation."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        certificate.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return out_path
