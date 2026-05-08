"""Tests for radiation model provenance metadata."""

from __future__ import annotations

from dpf.radiation import line_radiation_model_metadata


def test_line_radiation_metadata_marks_empirical_high_z_limit():
    metadata = line_radiation_model_metadata()
    assert metadata["model_role"] == "empirical_cooling_estimate"
    assert metadata["validation_role"] == "not_high_z_predictive"
    assert metadata["predictive_high_z"] is False
    assert "Empirical" in metadata["components"]["line_radiation"]
    assert "KnowledgeReference" in metadata["components"]["line_radiation"]
    assert "multigroup radiation diffusion" in metadata["validity_notes"]["transport"]


def test_line_radiation_metadata_keeps_bremsstrahlung_separate():
    metadata = line_radiation_model_metadata()
    assert "NRL-backed" in metadata["components"]["bremsstrahlung"]
    assert "dopant performance" in metadata["validity_notes"]["coupling"]
