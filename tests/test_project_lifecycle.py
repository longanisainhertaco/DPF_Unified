from __future__ import annotations

import json

import pytest

from dpf.config import SimulationConfig
from dpf.project.lifecycle import (
    archive_project,
    create_project,
    duplicate_project,
    load_project,
    project_manifest_path,
)
from dpf.validation.artifacts import file_sha256


def _config() -> SimulationConfig:
    return SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": ":memory:"},
    )


def test_create_and_load_project_preserves_config_and_manifest(tmp_path) -> None:
    project_root = tmp_path / "case-a"

    created = create_project(
        project_root,
        name="Case A",
        config=_config(),
        validation_status="not_evaluated",
        result_classification={"label": "Preview"},
        artifact_classification={
            "owner": "owner-team",
            "classification": "internal",
            "distribution": "project-only",
        },
        logs=["logs/run.log"],
    )
    loaded = load_project(project_root)

    assert project_manifest_path(project_root).exists()
    assert loaded.manifest.project_id == created.manifest.project_id
    assert loaded.manifest.name == "Case A"
    assert loaded.manifest.result_classification["label"] == "Preview"
    assert loaded.manifest.artifact_classification.owner == "owner-team"
    assert loaded.manifest.artifact_classification.distribution == "project-only"
    assert loaded.manifest.logs == ["logs/run.log"]
    assert loaded.config.grid_shape == [4, 4, 4]


def test_load_project_rejects_silent_config_mutation(tmp_path) -> None:
    project_root = tmp_path / "case-a"
    create_project(project_root, name="Case A", config=_config())
    config_path = project_root / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["sim_time"] = 2e-9
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config hash"):
        load_project(project_root)


def test_duplicate_project_preserves_outputs_and_records_source(tmp_path) -> None:
    project_root = tmp_path / "case-a"
    created = create_project(
        project_root,
        name="Case A",
        config=_config(),
        outputs=["outputs/diag.h5"],
        run_manifests=["outputs/diag.h5.run_manifest.json"],
    )
    output_path = project_root / "outputs" / "diag.h5"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"diagnostics")
    original_hash = file_sha256(output_path)

    duplicate = duplicate_project(project_root, tmp_path / "case-b", name="Case B")

    assert duplicate.manifest.project_id != created.manifest.project_id
    assert duplicate.manifest.source_project_id == created.manifest.project_id
    assert duplicate.manifest.name == "Case B"
    assert duplicate.manifest.outputs == ["outputs/diag.h5"]
    assert file_sha256(duplicate.root / "outputs" / "diag.h5") == original_hash


def test_archive_project_marks_status_without_mutating_outputs(tmp_path) -> None:
    project_root = tmp_path / "case-a"
    create_project(project_root, name="Case A", config=_config(), outputs=["outputs/diag.h5"])
    output_path = project_root / "outputs" / "diag.h5"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"diagnostics")
    original_hash = file_sha256(output_path)

    archived = archive_project(project_root, reason="release snapshot")

    assert archived.manifest.status == "archived"
    assert archived.manifest.archive_reason == "release snapshot"
    assert archived.manifest.archived_utc is not None
    assert file_sha256(output_path) == original_hash
