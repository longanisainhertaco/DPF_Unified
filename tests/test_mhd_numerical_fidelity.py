"""Tests for KR-gated MHD numerical-fidelity evidence."""

from __future__ import annotations

from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
)
from dpf.validation.mhd_numerical_fidelity import (
    backend_parity_evidence_from_results,
    cylindrical_convergence_evidence_from_results,
    mhd_numerical_fidelity_evidence_from_result,
    mhd_scope_limit_evidence_from_phases,
    resistive_diffusion_convergence_evidence_from_results,
)
from dpf.validation.quality_assessment import scientific_accuracy_gap_report


def test_empty_result_blocks_mhd_numerical_fidelity():
    evidence = mhd_numerical_fidelity_evidence_from_result({})

    assert evidence["passed"] is False
    required = evidence["required_evidence"]
    assert required["finite_volume_mhd_verification"]["status"] == "absent"
    assert required["circuit_coupled_energy_verification"]["status"] == "absent"
    assert set(evidence["missing_or_unvalidated_evidence"]) == set(required)


def test_generic_mhd_verification_is_not_full_dpf_numerical_fidelity():
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "backend": "python",
        "has_mhd": True,
        "mhd_verification": {
            "passed": True,
            "analytic_tests": {"sod": True, "brio_wu": True},
            "model_role": "code_verification_analytic_tests",
        },
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["finite_volume_mhd_verification"]["status"] == (
        "implemented_not_complete"
    )
    assert required["cylindrical_geometry_verification"]["status"] == "absent"
    assert required["backend_parity"]["status"] == "single_backend_only"


def test_mhd_method_metadata_is_not_verification_by_itself():
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "backend": "python",
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["finite_volume_mhd_verification"]["status"] == (
        "method_metadata_only"
    )
    assert "mhd_numerical_method" in (
        required["finite_volume_mhd_verification"]["evidence_keys"]
    )
    assert required["cylindrical_geometry_verification"]["status"] == (
        "diagnostic_not_validated"
    )


def test_finite_volume_method_and_named_tests_support_only_generic_mhd_channel():
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cartesian",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
        "mhd_verification": {
            "passed": True,
            "validation_tier": 3,
            "model_role": "code_verification_analytic_tests",
            "analytic_tests": {"sod": True, "brio_wu": True},
        },
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["finite_volume_mhd_verification"]["status"] == "supported"
    assert required["finite_volume_mhd_verification"]["validated"] is True
    assert required["cylindrical_geometry_verification"]["status"] == "absent"
    assert "backend_parity" in evidence["missing_or_unvalidated_evidence"]


def test_cylindrical_convergence_evidence_passes_kr_scoped_result():
    evidence = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
        "pressure_errors": [3.0e-3, 1.4e-3, 6.5e-4],
        "velocity_errors": [2.0e-3, 9.0e-4, 4.0e-4],
        "convergence_order": 1.15,
    })

    assert evidence["passed"] is True
    assert evidence["validation_tier"] == 3
    assert evidence["model_role"] == "code_verification_cylindrical_convergence"
    assert evidence["metrics"]["btheta_errors_decrease"] is True
    assert evidence["source_basis"]["cylindrical_mhd_convergence"].endswith(
        "beresnyak_2022_pulsed_power_ideal_mhd.md"
    )


def test_cylindrical_convergence_evidence_rejects_low_order_or_flat_errors():
    low_order = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
        "convergence_order": 0.7,
    })
    flat_errors = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 4.1e-3, 8.0e-4],
        "convergence_order": 1.2,
    })

    assert low_order["passed"] is False
    assert "convergence_order_passed" in low_order["missing_or_failed_metrics"]
    assert flat_errors["passed"] is False
    assert "btheta_errors_decrease" in flat_errors["missing_or_failed_metrics"]


def test_cylindrical_convergence_supports_only_two_mhd_audit_channels():
    convergence = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
        "convergence_order": 1.15,
    })
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "cylindrical_convergence_verification": convergence,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["cylindrical_geometry_verification"]["status"] == "supported"
    assert required["cylindrical_geometry_verification"]["validated"] is True
    assert required["convergence_study"]["status"] == "supported"
    assert required["convergence_study"]["validated"] is True
    assert "cylindrical_convergence_verification" in (
        required["convergence_study"]["evidence_keys"]
    )
    assert "finite_volume_mhd_verification" in (
        evidence["missing_or_unvalidated_evidence"]
    )


def test_resistive_diffusion_convergence_evidence_passes_kr_scoped_result():
    evidence = resistive_diffusion_convergence_evidence_from_results({
        "method": "implicit",
        "resolutions": [32, 64, 128, 256],
        "errors": [1.6e-2, 7.4e-3, 3.3e-3, 1.5e-3],
        "convergence_order": 1.12,
        "eta": 1.0e-7,
        "sigma0": 0.05,
        "t_end": 0.0125,
    })

    assert evidence["passed"] is True
    assert evidence["validation_tier"] == 3
    assert evidence["model_role"] == (
        "code_verification_resistive_diffusion_convergence"
    )
    assert evidence["metrics"]["errors_decrease"] is True
    assert evidence["source_basis"]["resistive_magnetic_diffusion_operator"].endswith(
        "modeling-and-simulation-in-science-engineering-and-technology-"
        "mathematical-models-and.md"
    )


def test_resistive_diffusion_convergence_rejects_bad_method_or_flat_errors():
    bad_method = resistive_diffusion_convergence_evidence_from_results({
        "method": "unknown",
        "resolutions": [32, 64, 128],
        "errors": [1.0e-2, 5.0e-3, 2.5e-3],
        "convergence_order": 1.0,
        "eta": 1.0e-7,
    })
    flat_errors = resistive_diffusion_convergence_evidence_from_results({
        "method": "sts",
        "resolutions": [32, 64, 128],
        "errors": [1.0e-2, 1.1e-2, 2.5e-3],
        "convergence_order": 1.0,
        "eta": 1.0e-7,
    })

    assert bad_method["passed"] is False
    assert "recognized_diffusion_method" in bad_method["missing_or_failed_metrics"]
    assert flat_errors["passed"] is False
    assert "errors_decrease" in flat_errors["missing_or_failed_metrics"]


def test_resistive_diffusion_supports_only_nonideal_audit_channel():
    diffusion = resistive_diffusion_convergence_evidence_from_results({
        "method": "implicit",
        "resolutions": [32, 64, 128],
        "errors": [1.0e-2, 4.5e-3, 2.0e-3],
        "convergence_order": 1.1,
        "eta": 1.0e-7,
    })
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "resistive_diffusion_verification": diffusion,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["resistive_or_nonideal_verification"]["status"] == "supported"
    assert required["resistive_or_nonideal_verification"]["validated"] is True
    assert "resistive_diffusion_verification" in (
        required["resistive_or_nonideal_verification"]["evidence_keys"]
    )
    assert required["convergence_study"]["status"] == "absent"
    assert "backend_parity" in evidence["missing_or_unvalidated_evidence"]


def test_circuit_coupled_energy_supports_mhd_numerical_channel_only():
    coupled = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
    )
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "circuit_coupled_energy_verification": coupled,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["circuit_coupled_energy_verification"]["status"] == "supported"
    assert required["circuit_coupled_energy_verification"]["validated"] is True
    assert "circuit_coupled_energy_verification" in (
        required["circuit_coupled_energy_verification"]["evidence_keys"]
    )
    assert required["finite_volume_mhd_verification"]["status"] == "absent"


def test_backend_parity_evidence_passes_supplied_observable_agreement():
    parity = backend_parity_evidence_from_results({
        "reference_backend": "python",
        "backends": {
            "python": {"I_peak_MA": 1.20, "pinch_time_ns": 95.0},
            "metal": {"I_peak_MA": 1.19, "pinch_time_ns": 96.0},
        },
        "relative_tolerances": {"I_peak_MA": 0.02, "pinch_time_ns": 0.02},
    })

    assert parity["passed"] is True
    assert parity["validation_tier"] == 3
    assert parity["model_role"] == "code_verification_backend_parity"
    assert parity["metrics"]["relative_errors_within_tolerance"] is True
    assert parity["details"]["max_relative_error"] < 0.02


def test_backend_parity_evidence_rejects_single_backend_or_mismatch():
    single = backend_parity_evidence_from_results({
        "backends": {"python": {"I_peak_MA": 1.20}},
    })
    mismatch = backend_parity_evidence_from_results({
        "backends": {
            "python": {"I_peak_MA": 1.20},
            "metal": {"I_peak_MA": 1.00},
        },
        "relative_tolerances": {"I_peak_MA": 0.02},
    })

    assert single["passed"] is False
    assert "two_or_more_backends" in single["missing_or_failed_metrics"]
    assert mismatch["passed"] is False
    assert "relative_errors_within_tolerance" in (
        mismatch["missing_or_failed_metrics"]
    )


def test_backend_parity_evidence_supports_backend_audit_channel_only():
    parity = backend_parity_evidence_from_results({
        "backends": {
            "python": {"I_peak_MA": 1.20, "pinch_time_ns": 95.0},
            "metal": {"I_peak_MA": 1.19, "pinch_time_ns": 96.0},
        },
        "relative_tolerances": {"I_peak_MA": 0.02, "pinch_time_ns": 0.02},
    })
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "backend_parity_verification": parity,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["backend_parity"]["status"] == "supported"
    assert required["backend_parity"]["validated"] is True
    assert "backend_parity_verification" in (
        required["backend_parity"]["evidence_keys"]
    )
    assert required["finite_volume_mhd_verification"]["status"] == "absent"


def test_mhd_scope_limit_evidence_passes_pre_disruption_boundary():
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["formation", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "non-ideal electric fields"],
    )

    assert scope["passed"] is True
    assert scope["validation_tier"] == 3
    assert scope["model_role"] == "mhd_phase_scope_limit"
    assert scope["metrics"]["pre_disruption_scope_declared"] is True
    assert scope["metrics"]["post_disruption_or_post_collapse_excluded"] is True


def test_mhd_scope_limit_evidence_rejects_unbounded_claim():
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["all_phases"],
        invalid_phases=[],
        limit_reasons=[],
    )

    assert scope["passed"] is False
    assert "post_disruption_or_post_collapse_excluded" in (
        scope["missing_or_failed_metrics"]
    )


def test_mhd_scope_limit_evidence_supports_scope_audit_channel_only():
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["before_first_collapse"],
        invalid_phases=["after_first_collapse"],
        limit_reasons=["beyond ideal MHD after disruption"],
    )
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "mhd_scope_limit": scope,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["dpf_scope_limit"]["status"] == "supported"
    assert required["dpf_scope_limit"]["validated"] is True
    assert "mhd_scope_limit" in required["dpf_scope_limit"]["evidence_keys"]
    assert required["finite_volume_mhd_verification"]["status"] == "absent"


def test_complete_mhd_numerical_evidence_packet_can_pass_gap_gate():
    scope_id = "synthetic_mhd_numerical_packet"
    cylindrical = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
        "convergence_order": 1.15,
    }, verification_scope=scope_id)
    circuit = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope=scope_id,
    )
    resistive = resistive_diffusion_convergence_evidence_from_results({
        "method": "implicit",
        "resolutions": [32, 64, 128],
        "errors": [1.0e-2, 4.5e-3, 2.0e-3],
        "convergence_order": 1.1,
        "eta": 1.0e-7,
    }, verification_scope=scope_id)
    parity = backend_parity_evidence_from_results({
        "backends": {
            "python": {"I_peak_MA": 1.20, "pinch_time_ns": 95.0},
            "metal": {"I_peak_MA": 1.19, "pinch_time_ns": 96.0},
        },
        "relative_tolerances": {"I_peak_MA": 0.02, "pinch_time_ns": 0.02},
    }, verification_scope=scope_id)
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["formation", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "non-ideal electric fields"],
        verification_scope=scope_id,
    )
    audit = mhd_numerical_fidelity_evidence_from_result({
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
            "verification_scope": scope_id,
        },
        "mhd_verification": {
            "passed": True,
            "validation_tier": 3,
            "model_role": "code_verification_analytic_tests",
            "verification_scope": scope_id,
            "analytic_tests": {"sod": True, "brio_wu": True},
        },
        "cylindrical_convergence_verification": cylindrical,
        "circuit_coupled_energy_verification": circuit,
        "resistive_diffusion_verification": resistive,
        "backend_parity_verification": parity,
        "mhd_scope_limit": scope,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({"mhd_numerical_fidelity": audit})
    }

    assert audit["passed"] is True
    assert audit["same_scope_passed"] is True
    assert audit["missing_or_unvalidated_evidence"] == []
    assert gaps["mhd_numerical_fidelity"].status == "supported"


def test_complete_mhd_numerical_packet_must_share_verification_scope():
    cylindrical = cylindrical_convergence_evidence_from_results({
        "resolutions": [32, 64, 128],
        "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
        "convergence_order": 1.15,
    }, verification_scope="scope_cylindrical")
    circuit = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope="scope_circuit",
    )
    resistive = resistive_diffusion_convergence_evidence_from_results({
        "method": "implicit",
        "resolutions": [32, 64, 128],
        "errors": [1.0e-2, 4.5e-3, 2.0e-3],
        "convergence_order": 1.1,
        "eta": 1.0e-7,
    }, verification_scope="scope_resistive")
    parity = backend_parity_evidence_from_results({
        "backends": {
            "python": {"I_peak_MA": 1.20, "pinch_time_ns": 95.0},
            "metal": {"I_peak_MA": 1.19, "pinch_time_ns": 96.0},
        },
        "relative_tolerances": {"I_peak_MA": 0.02, "pinch_time_ns": 0.02},
    }, verification_scope="scope_backend")
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["formation", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "non-ideal electric fields"],
        verification_scope="scope_limit",
    )
    audit = mhd_numerical_fidelity_evidence_from_result({
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
            "verification_scope": "scope_fv",
        },
        "mhd_verification": {
            "passed": True,
            "validation_tier": 3,
            "model_role": "code_verification_analytic_tests",
            "verification_scope": "scope_fv",
            "analytic_tests": {"sod": True, "brio_wu": True},
        },
        "cylindrical_convergence_verification": cylindrical,
        "circuit_coupled_energy_verification": circuit,
        "resistive_diffusion_verification": resistive,
        "backend_parity_verification": parity,
        "mhd_scope_limit": scope,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({"mhd_numerical_fidelity": audit})
    }

    assert audit["passed"] is False
    assert audit["same_scope_passed"] is False
    assert "same_scope_mhd_numerical_packet" in (
        audit["missing_or_unvalidated_evidence"]
    )
    assert gaps["mhd_numerical_fidelity"].status == "blocked"


def test_dpf_mhd_channels_remain_unvalidated_without_full_evidence():
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "backend": "metal_cylindrical",
        "cylindrical_verification": {"mag_noh": True},
        "field_coupling_validation": {"passed": False},
        "E_cap_kJ": [100.0, 95.0],
        "E_ind_kJ": [0.0, 4.0],
        "E_res_kJ": [0.0, 0.6],
        "resistivity": {"model": "spitzer"},
        "grid_convergence": {"density_peak": 0.12},
        "backend_parity": {"python_vs_metal": 0.08},
        "physics_fidelity_evidence": {"passed": False},
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["cylindrical_geometry_verification"]["status"] == (
        "diagnostic_not_validated"
    )
    assert required["circuit_coupled_energy_verification"]["status"] == (
        "diagnostic_not_validated"
    )
    assert required["resistive_or_nonideal_verification"]["status"] == (
        "implemented_not_validated"
    )
    assert required["convergence_study"]["status"] == "diagnostic_not_validated"
    assert required["backend_parity"]["status"] == "diagnostic_not_validated"
    assert required["dpf_scope_limit"]["status"] == "scope_limiter_reported"


def test_scientific_gap_report_marks_mhd_numerical_audit_as_partial():
    result = {
        "mhd_numerical_fidelity": mhd_numerical_fidelity_evidence_from_result({}),
    }
    gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(result)}

    assert gaps["mhd_numerical_fidelity"].status == "partial"
