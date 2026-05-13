"""Claim hygiene tests for the legacy Gradio UI."""

from __future__ import annotations

from pathlib import Path


def test_gradio_backend_copy_uses_preview_source_gated_language() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    banned = [
        "VALIDATED against 7+ published devices",
        "publication-grade accuracy",
        "Publication Quality",
        "97x demonstrated",
        "Lee-validated current waveforms",
        "validated accuracy",
        "validated 0D",
        '"WORKING"',
    ]
    for phrase in banned:
        assert phrase not in source

    assert "PREVIEW; source-gated validation only" in source
    assert "accepted local KnowledgeReference evidence" in source
    assert '"first_principles_mhd"' in source
    assert "FAIL-CLOSED PF-1000/AKEL READINESS" in source
    assert "baseline_reduced_model only" in source


def test_gradio_validation_markdown_declares_engineering_comparison() -> None:
    source = Path("app_validation.py").read_text(encoding="utf-8")

    assert "Engineering Comparison vs. Published Data" in source
    assert "Source authority: these grades are preview comparison metrics" in source
    assert "Validation vs. Published Data" not in source
