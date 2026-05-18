"""Run manifest helpers for package-native first-principles candidates."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
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

# Provenance fields that an externally reproducible, certificate-eligible
# manifest must carry as non-empty values (Codex finding A-8 / WP-N7).
# A manifest missing any of these cannot back an accepted certificate; the
# certificate gate's ``commands_and_versions`` and ``run_manifest_hash``
# channels depend on them.  This list is descriptive only -- the manifest
# itself stays fail-closed via ``__post_init__`` regardless.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "command_argv",
    "git_commit",
    "source_truth_index_sha256",
    "source_packet_hashes",
    "input_deck_sha256",
    "artifact_schema_version",
    "artifact_generation_commit",
)

ARTIFACT_SCHEMA_VERSION = "first_principles_artifact_v1"


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
    # Provenance fields (Codex A-8 / WP-N7).  All optional/None-tolerant so
    # existing construction still works, but present in the schema so an
    # external engineer can reproduce exactly what code, deck, sources, and
    # command produced an artifact.  Empty/None provenance is honest -- it
    # signals the manifest cannot back an accepted certificate.
    command_argv: tuple[str, ...] | None = None
    git_commit: str | None = None
    dirty_worktree: bool | None = None
    source_truth_index_sha256: str | None = None
    source_packet_hashes: dict[str, str] = field(default_factory=dict)
    input_deck_sha256: str | None = None
    artifact_schema_version: str | None = None
    artifact_generation_commit: str | None = None
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

    def missing_provenance_fields(self) -> tuple[str, ...]:
        """Return required provenance fields that are absent or empty.

        A non-empty result means the manifest cannot back an accepted
        first-principles certificate (Codex A-8): the certificate gate's
        ``commands_and_versions`` and ``run_manifest_hash`` channels need
        complete command/commit provenance.
        """

        missing: list[str] = []
        for name in REQUIRED_PROVENANCE_FIELDS:
            value = getattr(self, name)
            if value is None or (
                isinstance(value, str | tuple | list | dict | Mapping)
                and len(value) == 0
            ):
                missing.append(name)
        return tuple(missing)

    def has_complete_provenance(self) -> bool:
        """True only when every required provenance field is populated."""

        return not self.missing_provenance_fields()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_index_references"] = [
            ref.to_dict() for ref in self.source_index_references
        ]
        payload["inputs"] = [artifact.to_dict() for artifact in self.inputs]
        payload["outputs"] = [artifact.to_dict() for artifact in self.outputs]
        payload["command_argv"] = (
            None if self.command_argv is None else list(self.command_argv)
        )
        payload["source_packet_hashes"] = dict(self.source_packet_hashes)
        payload["provenance_complete"] = self.has_complete_provenance()
        payload["missing_provenance_fields"] = list(self.missing_provenance_fields())
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
    command_argv: Sequence[str] | None = None,
    git_commit: str | None = None,
    dirty_worktree: bool | None = None,
    source_truth_index_sha256: str | None = None,
    source_packet_hashes: Mapping[str, str] | None = None,
    input_deck_sha256: str | None = None,
    artifact_schema_version: str | None = None,
    artifact_generation_commit: str | None = None,
    notes: str = "",
) -> FirstPrinciplesRunManifest:
    """Build a fail-closed manifest for a package-native first-principles run.

    Provenance arguments are optional.  When omitted the manifest still
    constructs, but ``has_complete_provenance()`` reports ``False`` and the
    manifest cannot back an accepted certificate (Codex A-8).
    """

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
        command_argv=_optional_str_tuple(command_argv),
        git_commit=_optional_str(git_commit),
        dirty_worktree=None if dirty_worktree is None else bool(dirty_worktree),
        source_truth_index_sha256=_optional_str(source_truth_index_sha256),
        source_packet_hashes=_coerce_hash_map(source_packet_hashes),
        input_deck_sha256=_optional_str(input_deck_sha256),
        artifact_schema_version=_optional_str(artifact_schema_version),
        artifact_generation_commit=_optional_str(artifact_generation_commit),
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
    command_argv: Sequence[str] | None = None,
    git_commit: str | None = None,
    dirty_worktree: bool | None = None,
    source_truth_index_sha256: str | None = None,
    source_packet_hashes: Mapping[str, str] | None = None,
    input_deck_sha256: str | None = None,
    artifact_schema_version: str | None = None,
    artifact_generation_commit: str | None = None,
    notes: str = "",
) -> FirstPrinciplesRunManifest:
    """Build a manifest from a HybridPIC3DSimulationResult-like object.

    Provenance arguments are passed through to
    :func:`build_first_principles_run_manifest` unchanged; omitting them
    leaves the manifest non-reproducible and certificate-ineligible.
    """

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
        command_argv=command_argv,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        source_truth_index_sha256=source_truth_index_sha256,
        source_packet_hashes=source_packet_hashes,
        input_deck_sha256=input_deck_sha256,
        artifact_schema_version=artifact_schema_version,
        artifact_generation_commit=artifact_generation_commit,
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
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _runtime_profile() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def git_provenance() -> tuple[str | None, bool | None]:
    """Return ``(HEAD commit, dirty-worktree flag)``; ``(None, None)`` if git is
    unavailable. Provenance collection must fail soft, never block a run."""

    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return (commit or None), dirty
    except (subprocess.SubprocessError, OSError):
        return None, None


def stamp_artifact_provenance(payload: Any) -> Any:
    """Return ``payload`` with top-level artifact provenance fields added.

    No-op for non-mapping payloads. Provenance keys are authoritative over any
    pre-existing keys of the same name."""

    if not isinstance(payload, Mapping):
        return payload
    commit, dirty = git_provenance()
    return {
        **dict(payload),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_generation_commit": commit,
        "command_argv": list(sys.argv),
        "dirty_worktree": dirty,
        "generated_at_utc": _utc_now_iso(),
    }


def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Raises if the file is missing -- provenance hashing must fail closed
    rather than silently substitute a hash of empty content.
    """

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_file_soft(path: str | Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` if unreadable.

    Fail-soft companion to :func:`sha256_of_file` for provenance collection
    inside a live run: a missing or unreadable cited source must never crash
    the run, but it must also never be silently fabricated. ``None`` is the
    honest result -- it propagates into ``source_packet_hashes`` as an absent
    hash, which keeps the manifest's provenance state truthful.
    """

    try:
        return sha256_of_file(path)
    except (OSError, ValueError):
        return None


def source_packet_hashes_from_references(
    references: Iterable[SourceIndexReference | Mapping[str, Any]],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    """Hash every cited source packet, keyed by its ``source_id``.

    For each reference this resolves ``path`` (relative paths are joined to
    ``repo_root`` when given) and records its SHA-256 under the reference's
    ``source_id``. Hashing is fail-soft: a cited source whose file cannot be
    read is omitted from the result rather than crashing the run -- an absent
    key is honest about a missing hash. A reference without a usable
    ``source_id`` or ``path`` is skipped.
    """

    root = None if repo_root is None else Path(repo_root)
    hashes: dict[str, str] = {}
    for reference in references:
        if isinstance(reference, SourceIndexReference):
            source_id = reference.source_id
            path = reference.path
        elif isinstance(reference, Mapping):
            source_id = reference.get("source_id") or reference.get("id")
            path = reference.get("path")
        else:  # pragma: no cover - defensive; refs are normalized upstream
            continue
        if not source_id or not path:
            continue
        candidate = Path(path)
        if root is not None and not candidate.is_absolute():
            candidate = root / candidate
        digest = sha256_of_file_soft(candidate)
        if digest is not None:
            hashes[str(source_id)] = digest
    return hashes


def sha256_of_text(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 over a JSON-serializable mapping."""

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _optional_str(value: Any) -> str | None:
    """Coerce to a stripped string, treating None and blanks as absent."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_tuple(value: Sequence[str] | None) -> tuple[str, ...] | None:
    """Coerce a sequence to a string tuple; None stays None.

    An empty input sequence is preserved as an empty tuple so the
    manifest can record "command argv was captured but empty" distinctly
    from "command argv was never captured" (None).
    """

    if value is None:
        return None
    return tuple(str(item) for item in value)


def _coerce_hash_map(value: Mapping[str, str] | None) -> dict[str, str]:
    """Coerce a source-packet hash mapping to a plain ``dict[str, str]``."""

    if value is None:
        return {}
    return {str(key): str(item) for key, item in value.items()}
