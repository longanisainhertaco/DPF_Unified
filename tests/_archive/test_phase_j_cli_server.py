"""Phase J.1 — CLI and server integration tests for AthenaK backend.

Tests cover:
- CLI --backend choice includes "athenak"
- CLI `backends` command shows athenak status
- Server /api/health includes athenak availability
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

# ── CLI tests ───────────────────────────────────────────────────


class TestCLIBackendOption:
    """Test CLI --backend option includes athenak."""

    def test_simulate_help_shows_athenak(self):
        from dpf.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "athenak" in result.output

    def test_backends_command_shows_athenak(self):
        from dpf.cli.main import cli
        runner = CliRunner()
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "athenak" in result.output

    def test_backends_command_available(self):
        from dpf.cli.main import cli
        runner = CliRunner()
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=True),
        ):
            result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "athenak" in result.output
        assert "available" in result.output

    def test_export_well_help_shows_athenak(self):
        from dpf.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["export-well", "--help"])
        assert result.exit_code == 0
        assert "athenak" in result.output


# ── Server tests ────────────────────────────────────────────────


class TestServerHealth:
    """Test /api/health endpoint includes athenak."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from dpf.server.app import app
        return TestClient(app)

    def test_health_includes_athenak(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "athenak" in data["backends"]
        assert isinstance(data["backends"]["athenak"], bool)

    def test_health_athenak_available(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=True),
        ):
            resp = client.get("/api/health")
        data = resp.json()
        assert data["backends"]["athenak"] is True

    def test_health_still_has_python_and_athena(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            resp = client.get("/api/health")
        data = resp.json()
        assert "python" in data["backends"]
        assert "athena" in data["backends"]
        assert data["backends"]["python"] is True
