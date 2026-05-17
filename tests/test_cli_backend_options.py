"""Focused tests for CLI backend option consistency."""

from __future__ import annotations

import sys
import types
import json
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner


@pytest.fixture()
def cli_config_file(tmp_path, default_circuit_params) -> str:
    from dpf.config import SimulationConfig

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
    )
    path = tmp_path / "config.json"
    path.write_text(config.to_json())
    return str(path)


def test_simulate_backend_help_includes_mlx() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(cli, ["simulate", "--help"])

    assert result.exit_code == 0
    assert "--backend" in result.output
    assert "--run-mode" in result.output
    assert "mlx" in result.output
    assert "hybrid" in result.output
    assert "first_principles_mhd" in result.output


def test_simulate_backend_mlx_is_accepted(cli_config_file: str, monkeypatch) -> None:
    import dpf.engine as engine_module
    from dpf.cli.main import cli

    observed: dict[str, object] = {}

    class FakeSimulationEngine:
        def __init__(self, config) -> None:
            observed["backend"] = config.fluid.backend
            self.backend = config.fluid.backend

        def run(self, max_steps=None):
            observed["max_steps"] = max_steps
            return {"status": "ok"}

    monkeypatch.setattr(engine_module, "SimulationEngine", FakeSimulationEngine)

    result = CliRunner().invoke(
        cli,
        ["simulate", cli_config_file, "--backend=mlx", "--steps=1"],
    )

    assert result.exit_code == 0, result.output
    assert observed == {"backend": "mlx", "max_steps": 1}
    assert "Backend: mlx" in result.output


def test_simulate_first_principles_run_mode_metadata_is_forwarded(
    cli_config_file: str,
    monkeypatch,
) -> None:
    import dpf.engine as engine_module
    from dpf.cli.main import cli

    class ForbiddenSimulationEngine:
        def __init__(self, _config) -> None:
            raise AssertionError("legacy SimulationEngine first-principles path used")

    monkeypatch.setattr(engine_module, "SimulationEngine", ForbiddenSimulationEngine)

    result = CliRunner().invoke(
        cli,
        [
            "simulate",
            cli_config_file,
            "--run-mode=first_principles_mhd",
            "--validation-scope=pf1000_akel_16kv_1p2torr_shot_12581",
            "--source-scope=pf1000_akel_16kv_1p2torr_shot_12581",
            "--source-scope-status=same_scope_blocked_by_review",
            "--preset-name=pf1000_akel",
            "--steps=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Backend: package_native" in result.output
    assert "first_principles_3d_hybrid_em_pic_fluid" in result.output
    assert "blocked_same_scope_source_packet_not_available" in result.output


def test_first_principles_command_runs_field_coupled_candidate(
    tmp_path,
) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "first_principles.json"

    result = CliRunner().invoke(
        cli,
        [
            "first-principles",
            "--sim-time-us=0.2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["tool"] == "dpf first-principles"
    assert payload["package_native_tool"] == "dpf first-principles-3d"
    assert payload["first_principles_only_enforced"] is True
    assert payload["execution_backend"] == "package_native"
    assert payload["grid_preset"] == "coarse"
    assert payload["requested_sim_time_us"] == 0.2
    assert payload["simulated_time_us"] == pytest.approx(2.0e-7)
    assert payload["candidate_step_budget"] == 2
    assert payload["history_stride"] == 1
    assert payload["simulation"]["history_stride"] == 1
    assert payload["simulation"]["retained_step_result_count"] == 2
    assert payload["duration_request_satisfied"] is False
    assert payload["duration_gate_status"] == (
        "blocked_requested_duration_exceeds_candidate_step_budget"
    )
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    assert payload["validation_packet"]["status"] == "not_validation"
    assert payload["validation_packet"]["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )
    assert payload["manifest"]["candidate_evidence"]["certificate_gate_packet"][
        "release_decision"
    ] == "do_not_release_first_principles_claim"
    assert "Package-native first-principles PF-1000/Akel engineering candidate" in (
        result.output
    )
    assert "duration_gate: blocked_requested_duration_exceeds_candidate_step_budget" in (
        result.output
    )


def test_first_principles_command_has_no_legacy_app_runner_loader() -> None:
    from dpf.cli.main import cli
    import dpf.cli.main as cli_module

    assert not hasattr(cli_module, "_load_first_principles_runner")

    result = CliRunner().invoke(cli, ["first-principles"])

    assert result.exit_code == 0, result.output
    assert "Package-native first-principles PF-1000/Akel engineering candidate" in (
        result.output
    )


def test_hybrid_3d_smoke_command_writes_blocked_candidate(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "hybrid_3d_smoke.json"

    result = CliRunner().invoke(
        cli,
        [
            "hybrid-3d-smoke",
            "--steps=1",
            "--shape=4,4,4",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["tool"] == "dpf hybrid-3d-smoke"
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    assert payload["simulation"]["status"] == (
        "candidate_engineering_3d_hybrid_pic_simulation"
    )
    assert payload["simulation"]["last_step"]["source_workflow"]["status"] == (
        "candidate_engineering_source_ordered_loop"
    )
    assert payload["simulation"]["circuit"]["status"] == (
        "candidate_engineering_circuit_boundary_coupled"
    )
    assert payload["validation_packet"]["status"] == "blocked"
    assert "3D hybrid PIC-fluid engineering candidate" in result.output


def test_hybrid_3d_smoke_rejects_invalid_shape() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(cli, ["hybrid-3d-smoke", "--shape=2,4,4"])

    assert result.exit_code != 0
    assert "all shape entries must be >= 3" in result.output


def test_backends_command_lists_mlx() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(cli, ["backends"])

    assert result.exit_code == 0
    assert "mlx" in result.output


def test_export_well_backend_help_matches_config_backends() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(cli, ["export-well", "--help"])

    assert result.exit_code == 0
    for backend in ("python", "athena", "athenak", "metal", "mlx", "hybrid", "auto"):
        assert backend in result.output


def test_export_well_cli_forwards_artifact_classification(
    cli_config_file: str,
    monkeypatch,
) -> None:
    import dpf.ai.well_exporter as well_module
    import dpf.engine as engine_module
    from dpf.cli.main import cli

    observed: dict[str, object] = {}

    class FakeSimulationEngine:
        def __init__(self, config) -> None:
            self.backend = config.fluid.backend

        def step(self):
            return SimpleNamespace(
                time=1e-9,
                current=1.0,
                voltage=2.0,
                finished=True,
            )

        def get_field_snapshot(self):
            return {
                "rho": np.ones((2, 2, 2)),
                "B": np.ones((3, 2, 2, 2)),
            }

    class FakeWellExporter:
        def __init__(self, **kwargs) -> None:
            observed["kwargs"] = kwargs
            self.n_snapshots = 0

        def add_snapshot(self, state, time, circuit_scalars=None) -> None:
            self.n_snapshots += 1
            observed["circuit_scalars"] = circuit_scalars

        def finalize(self):
            return observed["kwargs"]["output_path"]

    monkeypatch.setattr(engine_module, "SimulationEngine", FakeSimulationEngine)
    monkeypatch.setattr(well_module, "WellExporter", FakeWellExporter)

    result = CliRunner().invoke(
        cli,
        [
            "export-well",
            cli_config_file,
            "--backend=python",
            "--steps=1",
            "--field-interval=1",
            "--output=classified.h5",
            "--artifact-owner=qa-team",
            "--artifact-classification=internal",
            "--artifact-distribution=project-only",
            "--artifact-handling-notes=review required",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = observed["kwargs"]
    assert kwargs["artifact_classification"] == {
        "owner": "qa-team",
        "classification": "internal",
        "distribution": "project-only",
        "handling_notes": "review required",
    }
    assert kwargs["sim_params"]["backend"] == "python"
    assert kwargs["sim_params"]["validation_status"] == "not_validation_evidence"
    assert observed["circuit_scalars"] == {"current": 1.0, "voltage": 2.0}


def test_server_health_reports_full_backend_contract() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi.testclient import TestClient

    from dpf.server.app import app

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    backends = response.json()["backends"]
    assert set(backends) == {"python", "athena", "athenak", "metal", "mlx", "hybrid"}
    assert backends["python"] is True
    assert all(isinstance(value, bool) for value in backends.values())


def test_validate_command_reports_source_authority(monkeypatch) -> None:
    import dpf.presets as presets_module
    from dpf.cli.main import cli

    fake_engine = types.ModuleType("app_engine")
    fake_engine.run_simulation_core = lambda **kwargs: {
        "I_peak": 1.0,
        "t_peak": 2.0,
        "n_steps": 10,
    }
    fake_validation = types.ModuleType("app_validation")
    fake_validation.validate_against_published = lambda data, preset_key: {
        "I_peak_dev_pct": 4.0,
        "I_peak_sim_MA": 1.0,
        "I_peak_ref_MA": 1.04,
        "t_peak_sim_us": 2.0,
    }

    monkeypatch.setitem(sys.modules, "app_engine", fake_engine)
    monkeypatch.setitem(sys.modules, "app_validation", fake_validation)
    monkeypatch.setattr(
        presets_module,
        "list_presets",
        lambda: [{"device": "PF-1000", "name": "pf1000"}],
    )

    result = CliRunner().invoke(cli, ["validate", "--device", "PF-1000"])

    assert result.exit_code == 0, result.output
    assert "Authority" in result.output
    assert "Blockers" in result.output
    assert "Preview/not_evaluated" in result.output
    assert "PASS" in result.output
    assert "Source authority: PASS/FAIR/POOR are peak-current engineering grades" in result.output
