"""Fail-closed air-gap release gate."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AirgapGateResult(BaseModel):
    """Air-gap readiness gate result."""

    passed: bool
    required_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    offline_commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


REQUIRED_AIRGAP_ARTIFACTS = (
    "pyproject.toml",
    "requirements.txt",
    "gui/package-lock.json",
    "dist/wheelhouse",
    "dist/wheelhouse/SHA256SUMS",
    "docs/airgap_logs/python-offline-smoke.log",
    "docs/airgap_logs/gui-offline-typecheck.log",
)

OFFLINE_COMMANDS = (
    "python3 -m pip install --no-index --find-links dist/wheelhouse '.[dev,server]'",
    "python3 -m pytest tests/test_validation_artifacts.py tests/test_export_scope.py tests/test_server_readiness.py -q",
    "npm --prefix gui ci --offline",
    "npm --prefix gui run typecheck",
)


def airgap_release_gate(project_root: str | Path) -> AirgapGateResult:
    """Inspect whether the repository has enough artifacts for an air-gap release."""

    root = Path(project_root)
    missing = [
        artifact
        for artifact in REQUIRED_AIRGAP_ARTIFACTS
        if not (root / artifact).exists()
    ]
    notes = [
        "Gate is fail-closed: release is blocked until wheelhouse hashes and offline logs exist.",
        "Network-created artifacts must be reviewed for licensing before vendoring.",
    ]
    return AirgapGateResult(
        passed=not missing,
        required_artifacts=list(REQUIRED_AIRGAP_ARTIFACTS),
        missing_artifacts=missing,
        offline_commands=list(OFFLINE_COMMANDS),
        notes=notes,
    )
