from __future__ import annotations

from dpf.release.airgap_gate import airgap_release_gate


def test_airgap_gate_blocks_current_repo_without_wheelhouse() -> None:
    result = airgap_release_gate(".")

    assert result.passed is False
    assert "dist/wheelhouse" in result.missing_artifacts
    assert "dist/wheelhouse/SHA256SUMS" in result.missing_artifacts
    assert any("--no-index" in command for command in result.offline_commands)


def test_airgap_gate_passes_when_required_artifacts_exist(tmp_path) -> None:
    for path in [
        "pyproject.toml",
        "requirements.txt",
        "gui/package-lock.json",
        "dist/wheelhouse/SHA256SUMS",
        "docs/airgap_logs/python-offline-smoke.log",
        "docs/airgap_logs/gui-offline-typecheck.log",
    ]:
        artifact = tmp_path / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("ok", encoding="utf-8")
    (tmp_path / "dist/wheelhouse").mkdir(exist_ok=True)

    result = airgap_release_gate(tmp_path)

    assert result.passed is True
    assert result.missing_artifacts == []
