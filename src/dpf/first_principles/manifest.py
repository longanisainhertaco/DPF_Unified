"""Run manifest helpers for package-native first-principles candidates."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

from dpf.first_principles.conservation import (
    ARTIFACT_STATUS_ENGINEERING_CANDIDATE,
    RUN_STATUS_ENGINEERING_CANDIDATE,
    VALIDATION_STATUS_NOT_VALIDATION,
    FirstPrinciplesConservationLedger,
    SourceIndexReference,
    build_conservation_ledger_from_hybrid_telemetry,
    normalize_source_index_references,
)


@dataclass(frozen=True)
class ManifestArtifact:
    """Input or output artifact attached to a first-principles run."""

    path: str
    kind: str
    sha256: str | None = None
    role: str = "engineering_artifact_not_validation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstPrinciplesRunManifest:
    """Manifest metadata for a 3-D hybrid EM/PIC-fluid candidate run."""

    manifest_version: str = "1.0"
    run_id: str = field(default_factory=lambda: f"fp-run-{uuid.uuid4().hex}")
    created_utc: str = field(default_factory=lambda: _utc_now_iso())
    solver_family: str = "3d_hybrid_em_pic_fluid"
    backend: str = "package_native"
    run_status: str = RUN_STATUS_ENGINEERING_CANDIDATE
    validation_status: str = VALIDATION_STATUS_NOT_VALIDATION
    artifact_status: str = ARTIFACT_STATUS_ENGINEERING_CANDIDATE
    geometry_dimensionality: str = "3d"
    n_steps_requested: int | None = None
    n_steps_completed: int | None = None
    final_time_s: float | None = None
    grid_shape: tuple[int, int, int] | None = None
    grid_spacing_m: tuple[float, float, float] | None = None
    conservation: FirstPrinciplesConservationLedger | None = None
    source_index_references: tuple[SourceIndexReference, ...] = field(default_factory=tuple)
    inputs: tuple[ManifestArtifact, ...] = field(default_factory=tuple)
    outputs: tuple[ManifestArtifact, ...] = field(default_factory=tuple)
    runtime: dict[str, str] = field(default_factory=lambda: _runtime_profile())
    metadata: dict[str, Any] = field(default_factory=dict)
    can_support_first_principles_acceptance: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.run_status != RUN_STATUS_ENGINEERING_CANDIDATE:
            raise ValueError("first-principles manifests must be engineering_candidate")
        if self.validation_status != VALIDATION_STATUS_NOT_VALIDATION:
            raise ValueError("first-principles manifests must be not_validation")
        if self.artifact_status != ARTIFACT_STATUS_ENGINEERING_CANDIDATE:
            raise ValueError("first-principles manifests must remain non-promoting")
        if self.can_support_first_principles_acceptance:
            raise ValueError("candidate manifests cannot support first-principles acceptance")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_index_references"] = [
            ref.to_dict() for ref in self.source_index_references
        ]
        payload["inputs"] = [artifact.to_dict() for artifact in self.inputs]
        payload["outputs"] = [artifact.to_dict() for artifact in self.outputs]
        payload["manifest_sha256"] = stable_manifest_hash(payload)
        return payload


def build_first_principles_run_manifest(
    *,
    run_id: str | None = None,
    backend: str = "package_native",
    telemetry: Mapping[str, Any] | None = None,
    conservation: FirstPrinciplesConservationLedger | None = None,
    source_index_references: Sequence[SourceIndexReference | Mapping[str, Any] | str] | None = None,
    inputs: Sequence[ManifestArtifact | Mapping[str, Any]] | None = None,
    outputs: Sequence[ManifestArtifact | Mapping[str, Any]] | None = None,
    grid_shape: Sequence[int] | None = None,
    grid_spacing_m: Sequence[float] | None = None,
    n_steps_requested: int | None = None,
    n_steps_completed: int | None = None,
    final_time_s: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    notes: str = "",
) -> FirstPrinciplesRunManifest:
    """Build a fail-closed manifest for a package-native first-principles run."""

    telem = dict(telemetry or {})
    refs = normalize_source_index_references(source_index_references)
    ledger = conservation
    if ledger is None and telem:
        ledger = build_conservation_ledger_from_hybrid_telemetry(
            telem,
            source_index_references=refs,
        )
    return FirstPrinciplesRunManifest(
        run_id=run_id or f"fp-run-{uuid.uuid4().hex}",
        backend=str(backend),
        n_steps_requested=_optional_int(
            n_steps_requested,
            telem.get("n_steps_requested"),
        ),
        n_steps_completed=_optional_int(
            n_steps_completed,
            telem.get("n_steps_completed"),
        ),
        final_time_s=_optional_float(final_time_s, telem.get("final_time_s")),
        grid_shape=_optional_int_tuple(grid_shape, 3),
        grid_spacing_m=_optional_float_tuple(grid_spacing_m, 3),
        conservation=ledger,
        source_index_references=refs,
        inputs=tuple(_coerce_artifact(item) for item in inputs or ()),
        outputs=tuple(_coerce_artifact(item) for item in outputs or ()),
        metadata=dict(metadata or {}),
        notes=notes,
    )


def build_first_principles_manifest_from_hybrid_result(
    result: Any,
    *,
    run_id: str | None = None,
    backend: str = "package_native",
    source_index_references: Sequence[SourceIndexReference | Mapping[str, Any] | str] | None = None,
    grid: Any | None = None,
    inputs: Sequence[ManifestArtifact | Mapping[str, Any]] | None = None,
    outputs: Sequence[ManifestArtifact | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    notes: str = "",
) -> FirstPrinciplesRunManifest:
    """Build a manifest from a HybridPIC3DSimulationResult-like object."""

    telemetry = _to_mapping(getattr(result, "telemetry", result))
    shape = None
    spacing = None
    if grid is not None:
        shape = getattr(grid, "shape", None)
        spacing = getattr(grid, "spacing", None)
    return build_first_principles_run_manifest(
        run_id=run_id,
        backend=backend,
        telemetry=telemetry,
        source_index_references=source_index_references,
        inputs=inputs,
        outputs=outputs,
        grid_shape=shape,
        grid_spacing_m=spacing,
        metadata=metadata,
        notes=notes,
    )


def stable_manifest_hash(payload: FirstPrinciplesRunManifest | Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 over manifest JSON content."""

    if isinstance(payload, FirstPrinciplesRunManifest):
        content = payload.to_dict()
    else:
        content = dict(payload)
    content.pop("manifest_sha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _runtime_profile() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _coerce_artifact(value: ManifestArtifact | Mapping[str, Any]) -> ManifestArtifact:
    if isinstance(value, ManifestArtifact):
        return value
    if isinstance(value, Mapping):
        return ManifestArtifact(
            path=str(value["path"]),
            kind=str(value["kind"]),
            sha256=None if value.get("sha256") is None else str(value["sha256"]),
            role=str(value.get("role", "engineering_artifact_not_validation")),
        )
    raise TypeError("artifacts must be ManifestArtifact or mapping")


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        mapped = value.to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    raise TypeError("value cannot be converted to a manifest telemetry mapping")


def _optional_int(preferred: int | None, fallback: Any) -> int | None:
    value = preferred if preferred is not None else fallback
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("integer manifest fields must be non-negative")
    return parsed


def _optional_float(preferred: float | None, fallback: Any) -> float | None:
    value = preferred if preferred is not None else fallback
    if value is None:
        return None
    return float(value)


def _optional_int_tuple(value: Sequence[int] | None, length: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    parsed = tuple(int(item) for item in value)
    if len(parsed) != length:
        raise ValueError(f"expected tuple length {length}")
    return parsed


def _optional_float_tuple(
    value: Sequence[float] | None,
    length: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    parsed = tuple(float(item) for item in value)
    if len(parsed) != length:
        raise ValueError(f"expected tuple length {length}")
    return parsed
