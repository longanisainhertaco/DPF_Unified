from __future__ import annotations

import json

from fastapi.testclient import TestClient

from dpf.config import SimulationConfig
from dpf.server.app import app


def _config_payload() -> dict:
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
    ).model_dump(mode="json")


def test_project_api_create_load_duplicate_archive_within_projects_root(
    tmp_path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DPF_PROJECTS_ROOT", str(projects_root))
    client = TestClient(app)

    root_response = client.get("/api/projects/root")
    assert root_response.status_code == 200
    assert root_response.json()["root"] == str(projects_root.resolve())

    create_response = client.post(
        "/api/projects",
        json={
            "root": "case-a",
            "name": "Case A",
            "config": _config_payload(),
            "validation_status": "not_evaluated",
            "result_classification": {"label": "Preview"},
            "artifact_classification": {
                "owner": "qa",
                "classification": "internal",
                "distribution": "project-only",
            },
            "logs": ["logs/run.log"],
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["root"] == str((projects_root / "case-a").resolve())
    assert created["manifest"]["name"] == "Case A"
    assert created["manifest"]["result_classification"]["label"] == "Preview"
    assert created["manifest"]["artifact_classification"]["owner"] == "qa"
    assert created["config"]["grid_shape"] == [4, 4, 4]

    load_response = client.post("/api/projects/load", json={"root": "case-a"})
    assert load_response.status_code == 200
    assert load_response.json()["manifest"]["project_id"] == created["manifest"]["project_id"]

    duplicate_response = client.post(
        "/api/projects/duplicate",
        json={"source_root": "case-a", "destination_root": "case-b", "name": "Case B"},
    )
    assert duplicate_response.status_code == 200
    duplicated = duplicate_response.json()
    assert duplicated["manifest"]["name"] == "Case B"
    assert duplicated["manifest"]["source_project_id"] == created["manifest"]["project_id"]

    archive_response = client.post(
        "/api/projects/archive",
        json={"root": "case-a", "reason": "release snapshot"},
    )
    assert archive_response.status_code == 200
    archived = archive_response.json()
    assert archived["manifest"]["status"] == "archived"
    assert archived["manifest"]["archive_reason"] == "release snapshot"


def test_project_api_rejects_paths_outside_projects_root(tmp_path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DPF_PROJECTS_ROOT", str(projects_root))
    client = TestClient(app)

    response = client.post(
        "/api/projects",
        json={
            "root": str(tmp_path / "outside"),
            "name": "Outside",
            "config": _config_payload(),
        },
    )

    assert response.status_code == 403
    assert "Project paths must stay under" in response.text


def test_project_api_load_rejects_silent_config_mutation(tmp_path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DPF_PROJECTS_ROOT", str(projects_root))
    client = TestClient(app)

    create_response = client.post(
        "/api/projects",
        json={"root": "case-a", "name": "Case A", "config": _config_payload()},
    )
    assert create_response.status_code == 200

    config_path = projects_root / "case-a" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["sim_time"] = 2e-9
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    load_response = client.post("/api/projects/load", json={"root": "case-a"})
    assert load_response.status_code == 422
    assert "config hash" in load_response.text
