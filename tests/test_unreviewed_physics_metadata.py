"""Fail-closed metadata tests for newly audited physics helper modules."""

from __future__ import annotations

from dpf.atomic.ablation import ablation_model_metadata
from dpf.experimental.civ_breakdown import civ_breakdown_model_metadata
from dpf.fluid.nernst import nernst_model_metadata
from dpf.fluid.two_temperature import two_temperature_model_metadata
from dpf.fluid.viscosity import braginskii_viscosity_model_metadata
from dpf.sheath.bohm import sheath_model_metadata
from dpf.turbulence.anomalous import anomalous_resistivity_model_metadata


def _assert_not_validation_evidence(metadata: dict[str, object]) -> None:
    assert metadata["validation_status"] == "not_validation_evidence"
    assert metadata["can_support_validation_claims"] is False
    assert "source_status" in metadata
    assert metadata["components"]
    assert metadata["validity_notes"]


def test_ablation_metadata_fails_closed_for_missing_source_packet() -> None:
    metadata = ablation_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["model_role"] == "constant_efficiency_electrode_ablation_scaffold"
    assert metadata["source_status"] == "ablation_efficiency_source_packet_missing"


def test_two_temperature_metadata_fails_closed_for_equilibration_audit() -> None:
    metadata = two_temperature_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert (
        metadata["source_status"]
        == "equilibration_convention_source_audit_incomplete"
    )


def test_viscosity_metadata_fails_closed_for_collision_log_audit() -> None:
    metadata = braginskii_viscosity_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["source_status"] == (
        "partial_nrl_coefficients_ion_collision_log_audit_needed"
    )
    assert "eta_i0" in metadata["components"]["eta_coefficients"]


def test_nernst_metadata_fails_closed_for_missing_source_packet() -> None:
    metadata = nernst_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["source_status"] == "nernst_thermomagnetic_source_packet_missing"


def test_sheath_metadata_fails_closed_for_startup_claims() -> None:
    metadata = sheath_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["source_status"] == "partial_nrl_textbook_support_needs_scope_packet"


def test_anomalous_resistivity_metadata_fails_closed() -> None:
    metadata = anomalous_resistivity_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["source_status"] == "microinstability_source_packets_missing"
    assert "validated anomalous resistance" in metadata["validity_notes"]["claim_limit"]


def test_civ_breakdown_metadata_fails_closed() -> None:
    metadata = civ_breakdown_model_metadata()
    _assert_not_validation_evidence(metadata)
    assert metadata["source_status"] == (
        "civ_paschen_gas_coefficients_source_packets_missing"
    )
