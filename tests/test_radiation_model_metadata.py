"""Tests for radiation model provenance metadata."""

from __future__ import annotations

import pytest

from dpf.radiation import (
    line_radiation_model_metadata,
    qmf_model_metadata,
    radiation_transport_model_metadata,
)


def test_line_radiation_metadata_marks_empirical_high_z_limit():
    metadata = line_radiation_model_metadata()
    assert metadata["model_role"] == "empirical_cooling_estimate"
    assert metadata["validation_role"] == "not_high_z_predictive"
    assert metadata["source_status"] == "unknown_provenance_empirical_fits"
    assert metadata["validation_status"] == "not_validation_evidence"
    assert metadata["claim_scope"] == "engineering_cooling_estimate"
    assert metadata["predictive_high_z"] is False
    assert "Empirical" in metadata["components"]["line_radiation"]
    assert "KnowledgeReference" in metadata["components"]["line_radiation"]
    assert "multigroup radiation diffusion" in metadata["validity_notes"]["transport"]


def test_line_radiation_metadata_keeps_bremsstrahlung_separate():
    metadata = line_radiation_model_metadata()
    assert "NRL-backed" in metadata["components"]["bremsstrahlung"]
    assert "dopant performance" in metadata["validity_notes"]["coupling"]


def test_qmf_metadata_marks_suppression_as_unverified():
    metadata = qmf_model_metadata()
    assert metadata["model_role"] == "heuristic_qmf_radiation_diagnostic"
    assert metadata["validation_role"] == "unverified_not_design_evidence"
    assert metadata["source_status"] == "free_free_suppression_source_missing"
    assert metadata["validation_status"] == "not_validation_evidence"
    assert metadata["predictive_qmf_suppression"] is False
    assert "verified local free-free" in metadata["components"]["suppression_factor"]
    assert "p-B11 feasibility" in metadata["validity_notes"]["claim_limit"]


def test_radiation_transport_metadata_keeps_fld_source_blocked():
    metadata = radiation_transport_model_metadata()
    assert metadata["model_role"] == "engineering_fld_transport_scaffold"
    assert metadata["source_status"] == "rosseland_kramers_fld_source_packet_missing"
    assert metadata["validation_status"] == "not_validation_evidence"
    assert metadata["can_support_validation_claims"] is False
    assert "not source-closed" in metadata["components"]["rosseland_opacity"]
    assert "NRL Eq. 31 and Eq. 32" in metadata["components"]["local_nrl_rows"]


def test_mlx_line_radiation_metadata_matches_cpu_source_status():
    pytest.importorskip("mlx.core")
    from dpf.metal.mlx_line_radiation import mlx_line_radiation_model_metadata

    cpu = line_radiation_model_metadata()
    mlx = mlx_line_radiation_model_metadata()

    assert mlx["model_role"] == cpu["model_role"]
    assert mlx["validation_role"] == cpu["validation_role"]
    assert mlx["source_status"] == cpu["source_status"]
    assert mlx["validation_status"] == cpu["validation_status"]
    assert mlx["predictive_high_z"] is False
    assert "not direct CHIANTI/ADAS/Post" in mlx["notes"]
