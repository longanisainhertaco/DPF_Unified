"""Tests for Phase F.4: CLI and Server Backend Integration.

Validates:
1. CLI --backend flag on simulate command
2. CLI verify command shows backend info
3. CLI backends command shows availability
4. Server /api/health reports backend availability
5. Server SimulationInfo includes backend field
6. Presets work with backend field
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from dpf.cli.main import cli
from dpf.config import SimulationConfig

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def config_file(tmp_path, default_circuit_params):
    """Create a temporary config JSON file."""
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
    )
    path = tmp_path / "test_config.json"
    path.write_text(config.to_json())
    return str(path)


@pytest.fixture
def config_file_with_backend(tmp_path, default_circuit_params):
    """Create a config file with explicit backend field."""
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
        fluid={"backend": "python"},
    )
    path = tmp_path / "test_config_backend.json"
    path.write_text(config.to_json())
    return str(path)


# ============================================================
# Test: CLI --backend flag
# ============================================================


class TestCLIBackendFlag:
    """Tests for the --backend flag on the simulate command."""

    def test_simulate_help_shows_backend(self, runner):
        """--backend appears in simulate help text."""
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "python" in result.output
        assert "athena" in result.output
        assert "auto" in result.output

    def test_simulate_python_backend(self, runner, config_file):
        """--backend=python runs with Python backend."""
        result = runner.invoke(
            cli, ["simulate", config_file, "--backend=python", "--steps=1"]
        )
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_simulate_default_backend(self, runner, config_file):
        """Without --backend, default (python) is used."""
        result = runner.invoke(
            cli, ["simulate", config_file, "--steps=1"]
        )
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_backend_override_in_config(self, runner, config_file_with_backend):
        """--backend flag overrides config file backend setting."""
        result = runner.invoke(
            cli,
            ["simulate", config_file_with_backend, "--backend=python", "--steps=1"],
        )
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_invalid_backend_rejected(self, runner, config_file):
        """Invalid backend value is rejected by Click."""
        result = runner.invoke(
            cli, ["simulate", config_file, "--backend=cuda", "--steps=1"]
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


# ============================================================
# Test: CLI verify command
# ============================================================


class TestCLIVerifyCommand:
    """Tests for the verify command backend display."""

    def test_verify_shows_backend(self, runner, config_file):
        """verify command shows Backend field."""
        result = runner.invoke(cli, ["verify", config_file])
        assert result.exit_code == 0
        assert "Backend:" in result.output
        assert "python" in result.output

    def test_verify_shows_all_fields(self, runner, config_file):
        """verify shows Grid, dx, sim_time, Circuit, Fluid, Backend."""
        result = runner.invoke(cli, ["verify", config_file])
        assert result.exit_code == 0
        assert "Grid:" in result.output
        assert "dx:" in result.output
        assert "sim_time:" in result.output
        assert "Circuit:" in result.output
        assert "Fluid:" in result.output
        assert "Backend:" in result.output


# ============================================================
# Test: CLI backends command
# ============================================================


class TestCLIBackendsCommand:
    """Tests for the backends command."""

    def test_backends_command_exists(self, runner):
        """backends command is available."""
        result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0

    def test_backends_shows_python(self, runner):
        """backends command always shows python as available."""
        result = runner.invoke(cli, ["backends"])
        assert "python" in result.output
        assert "always available" in result.output

    def test_backends_shows_athena_status(self, runner):
        """backends command shows athena status (available or not)."""
        result = runner.invoke(cli, ["backends"])
        assert "athena" in result.output
        # Either "available" or "not compiled" should appear
        assert "available" in result.output or "not compiled" in result.output


# ============================================================
# Test: Server health endpoint backend reporting
# ============================================================


class TestServerBackendReporting:
    """Tests for server API backend reporting."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_health_reports_backends(self, client):
        """GET /api/health includes backends dict."""
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert "backends" in data
        assert data["backends"]["python"] is True
        assert isinstance(data["backends"]["athena"], bool)

    def test_simulation_info_includes_backend(self, client):
        """POST /api/simulations returns backend in SimulationInfo."""
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "backend" in data
        assert data["backend"] == "python"

    def test_simulation_info_backend_persists(self, client):
        """GET /api/simulations/{id} also returns backend."""
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial"},
        )
        sim_id = r.json()["sim_id"]

        r = client.get(f"/api/simulations/{sim_id}")
        assert r.status_code == 200
        assert r.json()["backend"] == "python"

    def test_create_with_backend_in_config(self, client):
        """Config with explicit backend field works in server."""
        config = {
            "grid_shape": [4, 4, 4],
            "dx": 1e-3,
            "sim_time": 1e-7,
            "circuit": {
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            "fluid": {"backend": "python"},
        }
        r = client.post("/api/simulations", json={"config": config})
        assert r.status_code == 200
        assert r.json()["backend"] == "python"


# ============================================================
# Test: Presets with backend
# ============================================================


class TestPresetsBackend:
    """Verify presets work with backend field in config."""

    def test_preset_default_backend(self):
        """Presets default to python backend."""
        from dpf.presets import get_preset

        data = get_preset("tutorial")
        config = SimulationConfig(**data)
        assert config.fluid.backend == "python"

    def test_preset_override_backend(self):
        """Can override backend in preset config."""
        from dpf.presets import get_preset

        data = get_preset("tutorial")
        data["fluid"] = {"backend": "auto"}
        config = SimulationConfig(**data)
        assert config.fluid.backend == "auto"

    def test_all_presets_valid_with_backend(self):
        """All presets produce valid configs with backend field accessible."""
        from dpf.presets import get_preset, get_preset_names

        for name in get_preset_names():
            data = get_preset(name)
            config = SimulationConfig(**data)
            assert config.fluid.backend in ("python", "athena", "auto")


# ============================================================
# Test: Config backend serialization
# ============================================================


class TestConfigBackendSerialization:
    """Verify backend field survives config serialization."""

    def test_backend_in_json(self, default_circuit_params):
        """Backend appears in JSON output."""
        config = SimulationConfig(
            grid_shape=[4, 4, 4],
            dx=1e-3,
            sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        data = json.loads(config.to_json())
        assert data["fluid"]["backend"] == "python"

    def test_backend_roundtrip(self, default_circuit_params):
        """Backend survives JSON round-trip."""
        config = SimulationConfig(
            grid_shape=[4, 4, 4],
            dx=1e-3,
            sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "auto"},
        )
        json_str = config.to_json()
        restored = SimulationConfig.model_validate_json(json_str)
        assert restored.fluid.backend == "auto"

    def test_backend_file_roundtrip(self, tmp_path, default_circuit_params):
        """Backend survives file write/read round-trip."""
        config = SimulationConfig(
            grid_shape=[4, 4, 4],
            dx=1e-3,
            sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        path = tmp_path / "roundtrip.json"
        path.write_text(config.to_json())
        restored = SimulationConfig.from_file(str(path))
        assert restored.fluid.backend == "python"
