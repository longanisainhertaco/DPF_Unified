"""Tests for KR-gated circuit/field coupling evidence."""

from __future__ import annotations

from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
    dynamic_inductance_power_balance_from_waveforms,
    field_coupling_component_evidence,
    field_coupling_evidence_from_result,
)
from dpf.validation.quality_assessment import scientific_accuracy_gap_report


def test_empty_result_blocks_field_coupling_validation():
    evidence = field_coupling_evidence_from_result({})

    assert evidence["passed"] is False
    required = evidence["required_evidence"]
    assert required["plasma_inductance_series"]["status"] == "absent"
    assert required["dLdt_or_back_emf"]["status"] == "absent"
    assert set(evidence["missing_or_unvalidated_evidence"]) == set(required)


def test_reduced_inductance_series_is_only_candidate_evidence():
    evidence = field_coupling_evidence_from_result({
        "t_us": [0.0, 0.1, 0.2],
        "L_p_nH": [1.0, 1.4, 2.1],
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["plasma_inductance_series"]["status"] == (
        "implemented_not_validated"
    )
    assert required["field_derived_inductance"]["status"] == "reduced_or_unknown_source"
    assert required["dLdt_or_back_emf"]["status"] == (
        "candidate_from_inductance_derivative"
    )
    assert "L_p_nH_from_t_us" in required["dLdt_or_back_emf"]["evidence_keys"]


def test_dynamic_inductance_power_balance_is_internal_diagnostic_only():
    balance = dynamic_inductance_power_balance_from_waveforms(
        times_s=[0.0, 1.0e-6, 2.0e-6, 3.0e-6],
        current_A=[0.0, 1.0e6, 1.5e6, 1.2e6],
        inductance_H=[1.0e-9, 1.4e-9, 2.0e-9, 2.4e-9],
    )

    assert balance["passed"] is True
    assert balance["model_role"] == "lee_dynamic_inductance_power_accounting"
    assert balance["source"].startswith("KnowledgeReference/")
    assert balance["details"]["max_relative_residual"] <= 1.0e-9


def test_dynamic_inductance_power_balance_marks_energy_channel_present():
    balance = dynamic_inductance_power_balance_from_waveforms(
        times_s=[0.0, 1.0e-6, 2.0e-6],
        current_A=[0.0, 1.0e6, 1.2e6],
        inductance_H=[1.0e-9, 1.4e-9, 1.8e-9],
    )
    evidence = field_coupling_evidence_from_result({
        "dynamic_inductance_power_balance": balance,
    })

    required = evidence["required_evidence"]
    assert required["circuit_energy_balance"]["status"] == "diagnostic_not_validated"
    assert "dynamic_inductance_power_balance" in (
        required["circuit_energy_balance"]["evidence_keys"]
    )
    assert evidence["passed"] is False


def test_circuit_coupled_energy_evidence_passes_power_and_energy_history():
    evidence = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
    )

    assert evidence["passed"] is True
    assert evidence["validation_tier"] == 3
    assert evidence["model_role"] == "code_verification_circuit_coupled_energy_balance"
    assert evidence["metrics"]["circuit_power_matches_poynting"] is True
    assert evidence["metrics"]["integrated_energy_accounted"] is True


def test_circuit_coupled_energy_evidence_rejects_power_mismatch():
    evidence = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[8.0, 8.0, 8.0],
        stored_energy_J=[0.0, 8.0, 16.0],
        relative_tolerance=0.01,
    )

    assert evidence["passed"] is False
    assert "circuit_power_matches_poynting" in (
        evidence["missing_or_failed_metrics"]
    )


def test_circuit_coupled_energy_evidence_supports_two_audit_channels():
    coupled = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
    )
    evidence = field_coupling_evidence_from_result({
        "circuit_coupled_energy_verification": coupled,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["poynting_power_balance"]["status"] == "supported"
    assert required["poynting_power_balance"]["validated"] is True
    assert required["circuit_energy_balance"]["status"] == "supported"
    assert required["circuit_energy_balance"]["validated"] is True
    assert "kr_experimental_comparison" in evidence["missing_or_unvalidated_evidence"]


def test_field_coupling_component_evidence_supports_one_component():
    component = field_coupling_component_evidence(
        "plasma_inductance_series",
        validation_scope="unit_test_scope",
        notes="inductance series comes from field energy with uncertainty",
    )
    evidence = field_coupling_evidence_from_result({
        "field_coupling_component_validation": {
            "plasma_inductance_series": component,
        },
    })
    required = evidence["required_evidence"]

    assert component["passed"] is True
    assert evidence["passed"] is False
    assert required["plasma_inductance_series"]["status"] == "supported"
    assert required["plasma_inductance_series"]["validated"] is True
    assert "field_coupling_component_validation" in (
        required["plasma_inductance_series"]["evidence_keys"]
    )
    assert "kr_experimental_comparison" in evidence["missing_or_unvalidated_evidence"]


def test_complete_field_coupling_component_packet_can_pass():
    components = {
        component: field_coupling_component_evidence(
            component,
            validation_scope="synthetic_complete_field_coupling_packet",
        )
        for component in (
            "plasma_inductance_series",
            "field_derived_inductance",
            "dLdt_or_back_emf",
            "poynting_power_balance",
            "circuit_energy_balance",
            "handoff_transition_metadata",
            "kr_experimental_comparison",
        )
    }
    evidence = field_coupling_evidence_from_result({
        "field_coupling_component_validations": components,
    })

    assert evidence["passed"] is True
    assert evidence["same_scope_passed"] is True
    assert evidence["missing_or_unvalidated_evidence"] == []
    assert all(
        item["validated"] is True
        for item in evidence["required_evidence"].values()
    )


def test_complete_field_coupling_components_must_share_validation_scope():
    component_names = (
        "plasma_inductance_series",
        "field_derived_inductance",
        "dLdt_or_back_emf",
        "poynting_power_balance",
        "circuit_energy_balance",
        "handoff_transition_metadata",
        "kr_experimental_comparison",
    )
    components = {
        component: field_coupling_component_evidence(
            component,
            validation_scope=f"synthetic_field_scope_{idx % 2}",
        )
        for idx, component in enumerate(component_names)
    }
    evidence = field_coupling_evidence_from_result({
        "field_coupling_component_validations": components,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({
            "field_coupling_validation": evidence,
        })
    }

    assert evidence["passed"] is False
    assert evidence["same_scope_passed"] is False
    assert "same_scope_field_coupling_packet" in (
        evidence["missing_or_unvalidated_evidence"]
    )
    assert gaps["circuit_field_coupling"].status == "blocked"


def test_mhd_poynting_and_handoff_channels_remain_unvalidated():
    evidence = field_coupling_evidence_from_result({
        "t_us": [0.0, 0.1, 0.2],
        "Lp_snowplow_nH": [2.0, 2.5, 3.0],
        "Lp_mhd_nH": [1.8, 2.7, 3.4],
        "back_emf_V": [0.0, 120.0, 180.0],
        "poynting_balance": {"interface_power_W": [0.0, 1.0e6, 1.5e6]},
        "E_cap_kJ": [100.0, 96.0, 91.0],
        "E_ind_kJ": [0.0, 3.0, 6.0],
        "E_res_kJ": [0.0, 0.8, 1.7],
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["field_derived_inductance"]["status"] == (
        "implemented_not_validated"
    )
    assert required["dLdt_or_back_emf"]["status"] == "implemented_not_validated"
    assert required["poynting_power_balance"]["status"] == "diagnostic_not_validated"
    assert required["circuit_energy_balance"]["status"] == "diagnostic_not_validated"
    assert required["handoff_transition_metadata"]["status"] == (
        "diagnostic_not_validated"
    )
    assert "kr_experimental_comparison" in evidence["missing_or_unvalidated_evidence"]


def test_scientific_gap_report_marks_coupling_audit_as_partial():
    result = {
        "field_coupling_validation": field_coupling_evidence_from_result({
            "L_p_nH": [1.0, 1.2],
        }),
    }
    gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(result)}

    assert gaps["circuit_field_coupling"].status == "partial"
