"""Tests for the Akel digitization source-integrity script."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.verify_akel_digitization_source_integrity import (
    build_akel_digitization_source_integrity_report,
)


def _copy_current_akel_files(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.json",
        "archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf",
        "KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png",
        "KnowledgeReference/digitization/akel-2021-page3.svg",
        (
            "KnowledgeReference/digitization/"
            "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
        ),
    ]
    for rel_path in files:
        source = repo_root / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path


def test_current_akel_draft_passes_non_review_integrity(tmp_path):
    base_path = _copy_current_akel_files(tmp_path)

    report = build_akel_digitization_source_integrity_report(
        base_path=base_path,
        check_pdf_text_parity=False,
    )

    assert report["passed"] is True
    assert report["accepted_for_validation"] is False
    assert report["validation_status"] == "blocked_by_review"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["digitization_non_review_integrity"]["passed"] is True
    assert checks["series_point_counts"]["details"]["actual_counts"] == {
        "computed_current": 34,
        "measured_current": 294,
    }


def test_corrupted_akel_figure_fails_integrity(tmp_path):
    base_path = _copy_current_akel_files(tmp_path)
    figure_path = (
        base_path
        / "KnowledgeReference/figures/"
        / "akel-2021-fig1-current-waveform-shot-12581.png"
    )
    figure_path.write_bytes(b"not the original figure bytes")

    report = build_akel_digitization_source_integrity_report(
        base_path=base_path,
        check_pdf_text_parity=False,
    )

    assert report["passed"] is False
    assert "figure_image_hash" in report["fatal_failed_checks"]
