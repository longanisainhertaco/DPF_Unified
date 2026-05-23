"""Tests for KR-gated circuit/field coupling evidence."""

from __future__ import annotations

from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
    dynamic_inductance_power_balance_from_waveforms,
    field_coupling_component_evidence,
    field_coupling_evidence_from_result,
    field_power_diagnostics_from_cylindrical_state,
    implicit_midpoint_power_port_back_emf,
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
    assert evidence["coupling_interval_authority"]["labels"] == ["snowplow_loaded"]
    assert required["coupling_interval_authority"]["status"] == (
        "incomplete_interval_authority"
    )


def test_density_weighted_mhd_inductance_is_not_validated_field_coupling():
    evidence = field_coupling_evidence_from_result({
        "t_us": [0.0, 0.1, 0.2],
        "Lp_mhd_nH": [1.0, 1.4, 2.1],
        "coupling_interval_authority": "density_weighted_mhd",
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["field_derived_inductance"]["status"] == (
        "implemented_not_validated"
    )
    assert required["field_derived_inductance"]["validated"] is False
    assert "density-weighted" in required["field_derived_inductance"]["notes"]
    assert "validated_field_coupled" not in (
        evidence["coupling_interval_authority"]["labels"]
    )
    assert required["coupling_interval_authority"]["status"] == (
        "incomplete_interval_authority"
    )


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


def test_implicit_midpoint_power_port_enforces_power_without_current_floor():
    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState

    result = implicit_midpoint_power_port_back_emf(
        current_A=0.0,
        capacitor_voltage_V=16_000.0,
        L_total_H=33.0e-9,
        resistance_ohm=0.01,
        capacitance_F=1.332e-3,
        dL_dt_H_s=0.0,
        dt_s=1.0e-12,
        power_W=100.0,
    )

    assert bool(result["passed"]) is True
    assert result["method"] == "implicit_midpoint_power_port"
    assert result["current_mid_A"] != 0.0
    assert abs(
        result["current_mid_A"] * result["back_emf_V"] - result["power_W"]
    ) < 1.0e-8

    circuit = RLCSolver(C=1.332e-3, V0=16_000.0, L0=33.0e-9, R0=0.01)
    coupling = circuit.step(
        CouplingState(Lp=0.0, dL_dt=0.0),
        back_emf=result["back_emf_V"],
        dt=1.0e-12,
    )
    assert abs(coupling.current - result["current_new_A"]) < 1.0e-12


def test_implicit_midpoint_power_port_blocks_impossible_power_load():
    result = implicit_midpoint_power_port_back_emf(
        current_A=0.0,
        capacitor_voltage_V=0.0,
        L_total_H=33.0e-9,
        resistance_ohm=0.01,
        capacitance_F=1.332e-3,
        dL_dt_H_s=0.0,
        dt_s=1.0e-12,
        power_W=1.0e3,
    )

    assert result["passed"] is False
    assert result["reason"] in {
        "no_real_midpoint_power_port_root",
        "zero_midpoint_current_for_nonzero_power",
    }


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


def test_field_power_diagnostics_records_sign_time_centering_and_residual():
    import numpy as np

    B = np.zeros((3, 2, 2), dtype=float)
    B[1, :, :] = 2.0
    state = {"B": B, "velocity": np.zeros_like(B)}

    diagnostic = field_power_diagnostics_from_cylindrical_state(
        state,
        dr=0.01,
        dz=0.02,
        current_A=1.0e5,
        eta_ohm_m=2.5e-6,
    )

    assert diagnostic["validation_status"] == "not_validation_evidence"
    assert diagnostic["power_port_method"] == "axisymmetric_j_dot_e_volume_integral"
    assert diagnostic["time_centering"] == "instantaneous_cell_centered_fields"
    assert diagnostic["terminal_voltage_orientation"] == "load_positive_opposes_source_current"
    assert diagnostic["poynting_j_dot_e_residual_W"] == 0.0
    assert diagnostic["poynting_j_dot_e_relative_residual"] == 0.0


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


def test_circuit_coupled_energy_evidence_requires_valid_interval_labels():
    invalid = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        interval_labels=["snowplow_loaded", "unsupported_magic"],
    )
    valid = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        interval_labels=["snowplow_loaded", "field_derived_candidate"],
    )

    assert invalid["passed"] is False
    assert "recognized_interval_labels" in invalid["missing_or_failed_metrics"]
    assert invalid["details"]["interval_labels"] == [
        "snowplow_loaded",
        "unsupported_magic",
    ]
    assert valid["passed"] is True
    assert valid["metrics"]["recognized_interval_labels"] is True


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
            "coupling_interval_authority",
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
        "coupling_interval_authority",
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


def test_interval_authority_distinguishes_staged_coupling_modes():
    evidence = field_coupling_evidence_from_result({
        "t_us": [0.0, 0.1, 0.2, 0.3],
        "Lp_snowplow_nH": [2.0, 2.5, 3.0, 3.2],
        "Lp_mhd_nH": [1.8, 2.7, 3.4, 3.5],
        "coupling_alpha": [0.0, 0.25, 0.8, 1.0],
        "field_coupling_intervals": [
            {"authority": "snowplow_loaded", "t_start_us": 0.0, "t_end_us": 0.1},
            {"authority": "blended", "t_start_us": 0.1, "t_end_us": 0.2},
            {
                "authority": "field_derived_candidate",
                "t_start_us": 0.2,
                "t_end_us": 0.25,
            },
            {
                "authority": "validated_field_coupled",
                "t_start_us": 0.25,
                "t_end_us": 0.3,
            },
        ],
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert evidence["coupling_interval_authority"]["labels"] == [
        "blended",
        "field_derived_candidate",
        "snowplow_loaded",
        "validated_field_coupled",
    ]
    assert evidence["coupling_interval_authority"]["missing_labels"] == []
    assert required["coupling_interval_authority"]["status"] == (
        "staged_not_validated"
    )
    assert required["coupling_interval_authority"]["validated"] is False
    assert "coupling_interval_authority" in (
        evidence["missing_or_unvalidated_evidence"]
    )


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
