from __future__ import annotations

import pytest
from click.testing import CliRunner

from dpf.cli.main import cli
from dpf.security.local_first import (
    LocalFirstPolicy,
    local_first_security_audit,
    scan_hardware_control_imports,
    scan_renderer_external_assets,
    scan_runtime_ai_mutation_boundaries,
)
from dpf.server.app import local_cors_origins


def test_hardware_control_scan_detects_direct_driver_import(tmp_path) -> None:
    probe = tmp_path / "hardware_probe.py"
    probe.write_text("import serial\nfrom pyvisa import ResourceManager\n", encoding="utf-8")

    findings = scan_hardware_control_imports([probe])

    assert {finding.module for finding in findings} == {"serial", "pyvisa"}
    assert findings[0].line == 1


def test_local_first_audit_passes_current_source_defaults() -> None:
    audit = local_first_security_audit(".")

    assert audit["passed"] is True
    statuses = {control["id"]: control["status"] for control in audit["controls"]}
    assert statuses == {
        "DPF-SEC-001": "passed",
        "DPF-SEC-002": "passed",
        "DPF-SEC-003": "passed",
        "DPF-SEC-004": "passed",
        "DPF-SEC-005": "passed",
    }


def test_local_first_audit_fails_for_public_share_default() -> None:
    audit = local_first_security_audit(
        ".",
        policy=LocalFirstPolicy(default_bind_host="0.0.0.0", public_share_default=True),
    )

    assert audit["passed"] is False
    assert {control["id"]: control["status"] for control in audit["controls"]}[
        "DPF-SEC-002"
    ] == "failed"


def test_runtime_ai_boundary_scan_detects_active_simulation_mutation(tmp_path) -> None:
    router = tmp_path / "runtime_ai.py"
    router.write_text("from dpf.server.simulation import SimulationManager\nmgr.start()\n")

    findings = scan_runtime_ai_mutation_boundaries([router])

    assert {finding.pattern for finding in findings} == {"SimulationManager", ".start("}


def test_renderer_external_asset_scan_detects_remote_font(tmp_path) -> None:
    html = tmp_path / "index.html"
    html.write_text(
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<script src="http://localhost:5173/main.tsx"></script>\n',
        encoding="utf-8",
    )

    findings = scan_renderer_external_assets([html])

    assert len(findings) == 1
    assert findings[0].url == "https://fonts.googleapis.com"


def test_renderer_html_is_local_first() -> None:
    html = "gui/src/renderer/index.html"

    findings = scan_renderer_external_assets([html])

    assert findings == []


def test_cors_defaults_are_local_and_wildcard_requires_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("DPF_CORS_ORIGINS", raising=False)
    assert "*" not in local_cors_origins()
    assert "http://127.0.0.1:7860" in local_cors_origins()

    monkeypatch.setenv("DPF_CORS_ORIGINS", "*")
    monkeypatch.delenv("DPF_ALLOW_WILDCARD_CORS", raising=False)
    with pytest.raises(RuntimeError, match="Wildcard CORS requires"):
        local_cors_origins()

    monkeypatch.setenv("DPF_ALLOW_WILDCARD_CORS", "1")
    assert local_cors_origins() == ["*"]


def test_ui_command_exposes_local_host_default() -> None:
    result = CliRunner().invoke(cli, ["ui", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "127.0.0.1" in result.output
