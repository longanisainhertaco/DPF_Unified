"""Local project lifecycle operations with provenance preservation."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dpf.config import SimulationConfig
from dpf.validation.artifacts import ArtifactClassification, stable_json_hash, utc_now_iso

PROJECT_MANIFEST_FILENAME = "project_manifest.json"
PROJECT_CONFIG_FILENAME = "config.json"


class ProjectManifest(BaseModel):
    """Project-level manifest for local create/load/duplicate/archive operations."""

    manifest_version: Literal["1.0"] = "1.0"
    project_id: str
    name: str
    status: Literal["active", "archived"] = "active"
    created_utc: str = Field(default_factory=utc_now_iso)
    updated_utc: str = Field(default_factory=utc_now_iso)
    archived_utc: str | None = None
    archive_reason: str | None = None
    source_project_id: str | None = None
    config_path: str = PROJECT_CONFIG_FILENAME
    config_hash: str
    outputs: list[str] = Field(default_factory=list)
    run_manifests: list[str] = Field(default_factory=list)
    validation_status: str = "not_evaluated"
    result_classification: dict[str, Any] = Field(default_factory=dict)
    artifact_classification: ArtifactClassification = Field(default_factory=ArtifactClassification)
    logs: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ProjectBundle:
    """Loaded project state."""

    root: Path
    manifest: ProjectManifest
    config: SimulationConfig


def project_manifest_path(project_root: str | Path) -> Path:
    """Return the manifest path for a project root."""

    return Path(project_root) / PROJECT_MANIFEST_FILENAME


def _config_payload(config: SimulationConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def _write_manifest(project_root: Path, manifest: ProjectManifest) -> Path:
    path = project_manifest_path(project_root)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def create_project(
    project_root: str | Path,
    *,
    name: str,
    config: SimulationConfig,
    outputs: list[str] | None = None,
    run_manifests: list[str] | None = None,
    validation_status: str = "not_evaluated",
    result_classification: dict[str, Any] | None = None,
    artifact_classification: ArtifactClassification | dict[str, Any] | None = None,
    logs: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> ProjectBundle:
    """Create a local project directory and write its config and manifest."""

    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / PROJECT_CONFIG_FILENAME
    config_path.write_text(config.to_json(), encoding="utf-8")

    config_hash = stable_json_hash(_config_payload(config))
    manifest = ProjectManifest(
        project_id=f"project-{uuid.uuid4().hex}",
        name=name,
        config_hash=config_hash,
        outputs=list(outputs or []),
        run_manifests=list(run_manifests or []),
        validation_status=validation_status,
        result_classification=dict(result_classification or {}),
        artifact_classification=(
            ArtifactClassification.model_validate(artifact_classification)
            if artifact_classification is not None
            else ArtifactClassification()
        ),
        logs=list(logs or []),
        provenance=dict(provenance or {}),
    )
    _write_manifest(root, manifest)
    return ProjectBundle(root=root, manifest=manifest, config=config)


def load_project(project_root: str | Path) -> ProjectBundle:
    """Load a project manifest and config from disk."""

    root = Path(project_root)
    manifest = ProjectManifest.model_validate_json(
        project_manifest_path(root).read_text(encoding="utf-8")
    )
    config = SimulationConfig.from_file(str(root / manifest.config_path))
    expected_hash = stable_json_hash(_config_payload(config))
    if expected_hash != manifest.config_hash:
        raise ValueError("project config hash does not match manifest")
    return ProjectBundle(root=root, manifest=manifest, config=config)


def duplicate_project(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    name: str | None = None,
) -> ProjectBundle:
    """Duplicate a project directory while preserving files and provenance."""

    source = Path(source_root)
    destination = Path(destination_root)
    source_bundle = load_project(source)
    shutil.copytree(source, destination)

    now = utc_now_iso()
    duplicated_manifest = source_bundle.manifest.model_copy(
        update={
            "project_id": f"project-{uuid.uuid4().hex}",
            "name": name or f"{source_bundle.manifest.name} copy",
            "status": "active",
            "created_utc": now,
            "updated_utc": now,
            "archived_utc": None,
            "archive_reason": None,
            "source_project_id": source_bundle.manifest.project_id,
        }
    )
    _write_manifest(destination, duplicated_manifest)
    return load_project(destination)


def archive_project(project_root: str | Path, *, reason: str = "") -> ProjectBundle:
    """Mark a project archived without mutating config or output files."""

    bundle = load_project(project_root)
    now = utc_now_iso()
    archived = bundle.manifest.model_copy(
        update={
            "status": "archived",
            "updated_utc": now,
            "archived_utc": now,
            "archive_reason": reason or "archived",
        }
    )
    _write_manifest(bundle.root, archived)
    return load_project(bundle.root)
