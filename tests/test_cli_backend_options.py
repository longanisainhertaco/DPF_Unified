"""Focused tests for CLI backend option consistency."""

from __future__ import annotations

import sys
import types
import json
from pathlib import Path
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
    from dpf.validation import PF1000_AKEL_SOURCE_SCOPE, PF1000_AKEL_VALIDATION_SCOPE

    observed: dict[str, object] = {}

    class FakeSimulationEngine:
        def __init__(self, config) -> None:
            observed["run_mode"] = config.run_mode
            observed["validation_scope"] = config.validation_scope
            observed["source_scope"] = config.source_scope
            observed["source_scope_status"] = config.source_scope_status
            observed["preset_name"] = config.preset_name
            self.backend = config.fluid.backend

        def run(self, max_steps=None):
            observed["max_steps"] = max_steps
            return {
                "run_mode": observed["run_mode"],
                "first_principles_mhd_readiness": {"status": "blocked"},
            }

    monkeypatch.setattr(engine_module, "SimulationEngine", FakeSimulationEngine)

    result = CliRunner().invoke(
        cli,
        [
            "simulate",
            cli_config_file,
            "--run-mode=first_principles_mhd",
            f"--validation-scope={PF1000_AKEL_VALIDATION_SCOPE}",
            f"--source-scope={PF1000_AKEL_SOURCE_SCOPE}",
            "--source-scope-status=same_scope_blocked_by_review",
            "--preset-name=pf1000_akel",
            "--steps=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "run_mode": "first_principles_mhd",
        "validation_scope": PF1000_AKEL_VALIDATION_SCOPE,
        "source_scope": PF1000_AKEL_SOURCE_SCOPE,
        "source_scope_status": "same_scope_blocked_by_review",
        "preset_name": "pf1000_akel",
        "max_steps": 1,
    }
    assert "first_principles_mhd_readiness" in result.output


def test_first_principles_command_runs_field_coupled_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    import dpf.cli.main as cli_module
    from dpf.cli.main import cli

    output = tmp_path / "first_principles.json"
    observed: dict[str, object] = {}

    def fake_runner(**kwargs):
        observed.update(kwargs)
        return {
            "run_mode": "first_principles_mhd",
            "backend": "python",
            "source_scope": "pf1000_16kv_2021_akel_shot12581",
            "validation_scope": "pf1000_16kv_2021_akel",
            "field_coupled_candidate": True,
            "has_snowplow": False,
            "n_steps": 3,
            "nan_detected": False,
            "I_peak": 0.25,
            "t_peak": 0.1,
            "back_emf_V": np.array([0.0, 12.0, 15.0]),
            "Lp_field_nH": np.array([0.0, 1.1, 1.2]),
            "B_max": np.array([0.0, 0.2, 0.3]),
            "joule_energy_kJ": np.array([0.0, 0.01, 0.02]),
            "field_energy_residual_kJ": np.array([0.0, -0.01, -0.02]),
            "field_limiter_activation_count": np.array([0, 0, 0]),
            "first_principles_mhd_readiness": {"status": "blocked"},
            "first_principles_neutron_yield_authority": {"status": "not_produced"},
            "t_us": np.array([0.0, 0.1, 0.2]),
            "I_MA": np.array([0.0, 0.1, 0.25]),
            "V_kV": np.array([16.0, 15.9, 15.8]),
        }

    monkeypatch.setattr(cli_module, "_load_first_principles_runner", lambda: fake_runner)

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
    assert observed["grid_preset"] == "coarse"
    assert observed["sim_time_us"] == 0.2
    assert observed["gas_key"] == "D2"
    payload = json.loads(output.read_text())
    assert payload["first_principles_only_enforced"] is True
    assert payload["scientific_status"] == "engineering_probe_not_validation"
    assert payload["metrics"]["back_emf_abs_max_V"] == 15.0
    assert payload["readiness"]["status"] == "blocked"
    assert "First-principles PF-1000/Akel engineering candidate" in result.output


def test_first_principles_command_fails_on_reduced_fallback(monkeypatch) -> None:
    import dpf.cli.main as cli_module
    from dpf.cli.main import cli

    def fake_runner(**_kwargs):
        return {
            "run_mode": "first_principles_mhd",
            "backend": "python",
            "field_coupled_candidate": False,
            "has_snowplow": True,
            "n_steps": 1,
            "nan_detected": False,
            "back_emf_V": np.array([1.0]),
        }

    monkeypatch.setattr(cli_module, "_load_first_principles_runner", lambda: fake_runner)

    result = CliRunner().invoke(cli, ["first-principles"])

    assert result.exit_code != 0
    assert "first-principles-only enforcement failed" in result.output


def test_first_principles_runner_loader_adds_checkout_root(monkeypatch) -> None:
    import dpf.cli.main as cli_module

    repo_root = str(Path(cli_module.__file__).resolve().parents[3])
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != repo_root])

    runner = cli_module._load_first_principles_runner()

    assert sys.path[0] == repo_root
    assert runner.__name__ == "run_pf1000_akel_first_principles"


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
