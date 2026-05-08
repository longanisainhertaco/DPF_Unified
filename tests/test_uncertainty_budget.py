"""Tests for KR-gated uncertainty-budget evidence."""

from __future__ import annotations

from dpf.validation.quality_assessment import scientific_accuracy_gap_report
from dpf.validation.uncertainty_budget import (
    uncertainty_component_evidence,
    uncertainty_evidence_from_result,
    validation_uncertainty_coverage_from_result,
)


def test_empty_result_blocks_uncertainty_validation():
    evidence = uncertainty_evidence_from_result({})

    assert evidence["passed"] is False
    components = evidence["required_components"]
    assert components["experimental_measurement_uncertainty"]["status"] == "absent"
    assert components["shot_to_shot_variability"]["status"] == "absent"
    assert set(evidence["missing_or_unvalidated_components"]) == set(components)


def test_circuit_uncertainty_is_partial_not_validated_budget():
    evidence = uncertainty_evidence_from_result({
        "circuit_validation": {
            "details": {
                "uncertainty": {"peak_current_exp_1sigma": 0.05},
                "peak_current_tolerance": 0.15,
                "timing_tolerance": 0.20,
                "waveform_tolerance": 0.20,
            },
        },
        "uncertainty": {"agreement_within_2sigma": True},
    })
    components = evidence["required_components"]

    assert evidence["passed"] is False
    assert components["experimental_measurement_uncertainty"]["status"] == (
        "diagnostic_not_validated"
    )
    assert components["uncertainty_propagation_to_observables"]["status"] == (
        "observable_coverage_present"
    )
    assert components["validation_acceptance_rule"]["status"] == (
        "rule_present_not_full_budget"
    )
    assert "kr_uncertainty_targets" in evidence["missing_or_unvalidated_components"]


def test_full_diagnostic_channels_still_require_kr_uncertainty_targets():
    evidence = uncertainty_evidence_from_result({
        "measurement_uncertainty": {"voltage": 0.10},
        "input_uncertainty": {"fill_pressure": 0.05},
        "grid_convergence": {"order": 2.0},
        "model_form_uncertainty": {"closure": "reduced_mhd"},
        "shot_to_shot_uncertainty": {"yield_sigma": 0.50},
        "uq_summary": {"Y_neutron": {"relative_sigma": 0.60}},
        "validation_tiers": [{"level": 1, "status": "supported"}],
    })
    components = evidence["required_components"]

    assert evidence["passed"] is False
    assert components["input_parameter_uncertainty"]["status"] == (
        "diagnostic_not_validated"
    )
    assert components["numerical_discretization_uncertainty"]["status"] == (
        "diagnostic_not_validated"
    )
    assert components["model_form_uncertainty"]["status"] == "blocker_reported"
    assert components["shot_to_shot_variability"]["status"] == (
        "diagnostic_not_validated"
    )
    assert components["kr_uncertainty_targets"]["status"] == "validation_absent"


def test_validation_uncertainty_coverage_lists_missing_observable_budgets():
    coverage = validation_uncertainty_coverage_from_result({
        "circuit_validation": {
            "passed": True,
            "details": {"uncertainty": {"peak_current_sigma": 0.05}},
        },
        "neutron_spectrum_validation": {
            "passed": True,
            "validated_features": {"spectrum": True},
        },
    })

    records = {item["observable"]: item for item in coverage["observables"]}
    assert coverage["passed"] is False
    assert records["circuit_validation"]["has_uncertainty"] is True
    assert records["neutron_spectrum_validation"]["has_uncertainty"] is False
    assert "neutron_spectrum_validation" in coverage["missing_uncertainty_observables"]


def test_validation_uncertainty_coverage_uses_result_level_observable_budget():
    coverage = validation_uncertainty_coverage_from_result({
        "neutron_spectrum_validation": {
            "passed": True,
            "validated_features": {"spectrum": True},
        },
        "uq_summary": {
            "observables": {
                "neutron_spectrum": {"relative_sigma": 0.20},
            },
        },
    })

    assert coverage["passed"] is True
    record = coverage["observables"][0]
    assert record["observable"] == "neutron_spectrum_validation"
    assert record["has_uncertainty"] is True
    assert "uq_summary.observables.neutron_spectrum" in record["uncertainty_paths"]


def test_uncertainty_evidence_reports_observable_coverage_without_validating_uq():
    evidence = uncertainty_evidence_from_result({
        "neutron_spectrum_validation": {
            "passed": True,
            "validated_features": {"spectrum": True},
        },
        "uq_summary": {
            "observables": {
                "neutron_spectrum": {"relative_sigma": 0.20},
            },
        },
    })

    component = evidence["required_components"][
        "uncertainty_propagation_to_observables"
    ]
    assert evidence["passed"] is False
    assert component["status"] == "observable_coverage_present"
    assert "validation_uncertainty_coverage" in component["evidence_keys"]


def test_uncertainty_component_evidence_supports_one_component():
    component = uncertainty_component_evidence(
        "input_parameter_uncertainty",
        validation_scope="unit_test_scope",
        notes="input covariance is propagated for this scope",
    )
    evidence = uncertainty_evidence_from_result({
        "uncertainty_component_validation": {
            "input_parameter_uncertainty": component,
        },
    })
    item = evidence["required_components"]["input_parameter_uncertainty"]

    assert component["passed"] is True
    assert evidence["passed"] is False
    assert item["status"] == "supported"
    assert item["validated"] is True
    assert "uncertainty_component_validation" in item["evidence_keys"]
    assert "kr_uncertainty_targets" in evidence["missing_or_unvalidated_components"]


def test_complete_uncertainty_component_packet_can_pass_gap_gate():
    components = {
        component: uncertainty_component_evidence(
            component,
            validation_scope="synthetic_complete_uq_packet",
        )
        for component in (
            "experimental_measurement_uncertainty",
            "input_parameter_uncertainty",
            "numerical_discretization_uncertainty",
            "model_form_uncertainty",
            "shot_to_shot_variability",
            "uncertainty_propagation_to_observables",
            "validation_acceptance_rule",
            "kr_uncertainty_targets",
        )
    }
    evidence = uncertainty_evidence_from_result({
        "uncertainty_component_validations": components,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({
            "uncertainty_validation": evidence,
        })
    }

    assert evidence["passed"] is True
    assert evidence["same_scope_passed"] is True
    assert evidence["missing_or_unvalidated_components"] == []
    assert gaps["uncertainty_quantification"].status == "supported"


def test_complete_uncertainty_components_must_share_validation_scope():
    components = {}
    for idx, component in enumerate((
        "experimental_measurement_uncertainty",
        "input_parameter_uncertainty",
        "numerical_discretization_uncertainty",
        "model_form_uncertainty",
        "shot_to_shot_variability",
        "uncertainty_propagation_to_observables",
        "validation_acceptance_rule",
        "kr_uncertainty_targets",
    )):
        components[component] = uncertainty_component_evidence(
            component,
            validation_scope=f"synthetic_uq_scope_{idx % 2}",
        )

    evidence = uncertainty_evidence_from_result({
        "uncertainty_component_validations": components,
    })

    assert evidence["passed"] is False
    assert evidence["same_scope_passed"] is False
    assert "same_scope_uncertainty_packet" in (
        evidence["missing_or_unvalidated_components"]
    )


def test_scientific_gap_report_marks_uncertainty_audit_as_partial():
    result = {"uncertainty_validation": uncertainty_evidence_from_result({})}
    gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(result)}

    assert gaps["uncertainty_quantification"].status == "partial"
