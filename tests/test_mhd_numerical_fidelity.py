"""Tests for KR-gated MHD numerical-fidelity evidence."""

from __future__ import annotations

from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
)
from dpf.validation.mhd_numerical_fidelity import (
    backend_parity_evidence_from_results,
    build_mhd_numerical_verification_packet,
    cylindrical_convergence_evidence_from_results,
    limiter_zero_evidence_from_limiter_ledger,
    mhd_numerical_fidelity_evidence_from_result,
    mhd_numerical_verification_packet_status,
    mhd_scope_limit_evidence_from_phases,
    resistive_diffusion_convergence_evidence_from_results,
    restart_reproducibility_evidence_from_results,
)
from dpf.validation.quality_assessment import scientific_accuracy_gap_report
from scripts.build_mhd_restart_reproducibility_evidence import (
    build_restart_reproducibility_evidence,
)


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


def test_generic_shock_tests_cannot_claim_spatial_or_neutron_validation():
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
        "mhd_verification": {
            "passed": True,
            "validation_tier": 4,
            "model_role": "experimental_dpf_validation",
            "analytic_tests": {"sod": True, "brio_wu": True},
        },
    })
    finite_volume = evidence["required_evidence"]["finite_volume_mhd_verification"]

    assert evidence["passed"] is False
    assert finite_volume["status"] == "implemented_not_complete"
    assert finite_volume["evidence_class"] == "code_numerical_verification"
    assert finite_volume["experimental_dpf_validation"] is False
    assert finite_volume["supports_predictive_scientific_claims"] is False
    assert finite_volume["supports_high_fidelity_scientific_claims"] is False
    assert finite_volume["supports_validation_tiers"] == [3]
    assert finite_volume["cannot_substitute_for_validation_tiers"] == [4, 5]


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
    assert required["finite_volume_mhd_verification"]["source"].endswith(
        "finite-volume-methods-for-hyperbolic-problems.md"
    )
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
    assert parity["authority_label"] == "BackendParityVerification"
    assert parity["authority_label"] != "Reference"
    assert "reference_scientific_authority" in parity["cannot_substitute_for"]
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


def test_restart_reproducibility_evidence_passes_checkpoint_match():
    restart = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
        "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
        "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
        "restart_step": 512,
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    })

    assert restart["passed"] is True
    assert restart["validation_tier"] == 3
    assert restart["model_role"] == "code_verification_restart_reproducibility"
    assert restart["metrics"]["config_hashes_match"] is True
    assert restart["metrics"]["relative_errors_within_tolerance"] is True
    assert restart["experimental_dpf_validation"] is False


def test_restart_reproducibility_builder_runs_checkpoint_fixture():
    restart = build_restart_reproducibility_evidence(
        scope="test_restart_fixture_scope",
        restart_step=3,
        total_steps=6,
        relative_tolerance=1.0e-12,
    )

    assert restart["passed"] is True
    assert restart["model_role"] == "code_verification_restart_reproducibility"
    assert restart["verification_scope"] == "test_restart_fixture_scope"
    assert restart["missing_or_failed_metrics"] == []
    assert restart["details"]["max_relative_error"] == 0.0
    assert restart["metrics"]["config_hashes_match"] is True
    assert restart["run_metadata"]["restart_step"] == 3
    assert restart["run_metadata"]["target_step"] == 6


def test_restart_reproducibility_evidence_rejects_missing_marker_or_hash_mismatch():
    missing_marker = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2},
        "restarted": {"I_peak_MA": 1.2},
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    })
    hash_mismatch = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2},
        "restarted": {"I_peak_MA": 1.2},
        "restart_step": 512,
        "config_hash": "cfg-a",
        "restart_config_hash": "cfg-b",
    })

    assert missing_marker["passed"] is False
    assert "restart_marker_present" in missing_marker["missing_or_failed_metrics"]
    assert hash_mismatch["passed"] is False
    assert "config_hashes_match" in hash_mismatch["missing_or_failed_metrics"]


def test_restart_reproducibility_supports_restart_channel_only():
    restart = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
        "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
        "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
        "restart_step": 512,
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    })
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "restart_reproducibility_verification": restart,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["restart_reproducibility"]["status"] == "supported"
    assert required["restart_reproducibility"]["validated"] is True
    assert "restart_reproducibility_verification" in (
        required["restart_reproducibility"]["evidence_keys"]
    )
    assert required["backend_parity"]["status"] == "absent"


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


def test_limiter_zero_evidence_passes_no_acceptance_blocking_ledger():
    limiter_zero = limiter_zero_evidence_from_limiter_ledger({
        "status": "clear",
        "entry_count": 4,
        "acceptance_blocking_activation_count": 0,
        "verified_numerical_method_activation_count": 4,
        "acceptance_blocking_limiter_ids": [],
    })

    assert limiter_zero["passed"] is True
    assert limiter_zero["validation_tier"] == 3
    assert limiter_zero["model_role"] == "code_verification_limiter_zero_acceptance"
    assert limiter_zero["metrics"]["no_acceptance_blocking_limiter_activations"] is True
    assert limiter_zero["experimental_dpf_validation"] is False
    assert limiter_zero["supports_predictive_scientific_claims"] is False


def test_limiter_zero_evidence_rejects_hidden_acceptance_blocking_activity():
    missing = limiter_zero_evidence_from_limiter_ledger({})
    active = limiter_zero_evidence_from_limiter_ledger({
        "status": "blocked",
        "entry_count": 3,
        "acceptance_blocking_activation_count": 2,
        "acceptance_blocking_limiter_ids": ["density_floor", "temperature_floor"],
    })

    assert missing["passed"] is False
    assert "ledger_present" in missing["missing_or_failed_metrics"]
    assert active["passed"] is False
    assert "no_acceptance_blocking_limiter_activations" in (
        active["missing_or_failed_metrics"]
    )


def test_limiter_zero_supports_limiter_audit_channel_only():
    limiter_zero = limiter_zero_evidence_from_limiter_ledger({
        "status": "clear",
        "entry_count": 1,
        "acceptance_blocking_activation_count": 0,
    })
    evidence = mhd_numerical_fidelity_evidence_from_result({
        "limiter_zero_verification": limiter_zero,
    })
    required = evidence["required_evidence"]

    assert evidence["passed"] is False
    assert required["limiter_zero_acceptance"]["status"] == "supported"
    assert required["limiter_zero_acceptance"]["validated"] is True
    assert "limiter_zero_verification" in (
        required["limiter_zero_acceptance"]["evidence_keys"]
    )
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
    restart = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
        "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
        "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
        "restart_step": 512,
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    }, verification_scope=scope_id)
    limiter_zero = limiter_zero_evidence_from_limiter_ledger({
        "status": "clear",
        "entry_count": 4,
        "acceptance_blocking_activation_count": 0,
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
        "restart_reproducibility_verification": restart,
        "limiter_zero_verification": limiter_zero,
        "mhd_scope_limit": scope,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({"mhd_numerical_fidelity": audit})
    }

    assert audit["passed"] is True
    assert audit["same_scope_passed"] is True
    assert audit["missing_or_unvalidated_evidence"] == []
    assert audit["evidence_class"] == "code_numerical_verification"
    assert audit["experimental_dpf_validation"] is False
    assert audit["supports_predictive_scientific_claims"] is False
    assert audit["supports_high_fidelity_scientific_claims"] is False
    assert audit["supports_validation_tiers"] == [3]
    assert audit["cannot_substitute_for_validation_tiers"] == [4, 5]
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
    restart = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
        "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
        "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
        "restart_step": 512,
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    }, verification_scope="scope_restart")
    limiter_zero = limiter_zero_evidence_from_limiter_ledger({
        "status": "clear",
        "entry_count": 4,
        "acceptance_blocking_activation_count": 0,
    }, verification_scope="scope_limiter")
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
        "restart_reproducibility_verification": restart,
        "limiter_zero_verification": limiter_zero,
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


def test_mhd_numerical_packet_status_reports_missing_required_packets():
    status = mhd_numerical_verification_packet_status({
        "mhd_numerical_method": {
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
    })

    assert status["passed"] is False
    assert status["production_packet_status"] == "blocked"
    assert status["packet_status"]["finite_volume_mhd_verification"]["status"] == (
        "attached_non_validating"
    )
    assert status["packet_status"]["dpf_scope_limit"]["status"] == (
        "missing_required"
    )
    assert "restart_reproducibility" in status["missing_required_packets"]
    assert status["supports_validation_tiers"] == [3]
    assert status["supports_predictive_scientific_claims"] is False


def test_mhd_numerical_packet_status_passes_complete_same_scope_packet():
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
    restart = restart_reproducibility_evidence_from_results({
        "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
        "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
        "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
        "restart_step": 512,
        "config_hash": "cfg-sha",
        "restart_config_hash": "cfg-sha",
    }, verification_scope=scope_id)
    limiter_zero = limiter_zero_evidence_from_limiter_ledger({
        "status": "clear",
        "entry_count": 4,
        "acceptance_blocking_activation_count": 0,
    }, verification_scope=scope_id)
    scope = mhd_scope_limit_evidence_from_phases(
        applicable_phases=["formation", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "non-ideal electric fields"],
        verification_scope=scope_id,
    )

    status = mhd_numerical_verification_packet_status({
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
        "restart_reproducibility_verification": restart,
        "limiter_zero_verification": limiter_zero,
        "mhd_scope_limit": scope,
    })

    assert status["passed"] is True
    assert status["production_packet_status"] == "complete"
    assert status["missing_required_packets"] == []
    assert set(status["attached_validated_packets"]) == set(
        status["packet_status"]
    )


def test_packet_builder_attaches_scheduled_results_without_promotion():
    packet = build_mhd_numerical_verification_packet(
        verification_scope="scheduled_tier3_cpu_mhd_numerical",
        mhd_numerical_method={
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
        cylindrical_convergence={
            "resolutions": [16, 32, 64],
            "Btheta_errors": [
                1.4407098338781483e-3,
                5.092017281638689e-4,
                1.800153038522951e-4,
            ],
            "pressure_errors": [4.08e-2, 2.88e-2, 2.04e-2],
            "velocity_errors": [619.4, 446.4, 318.6],
            "convergence_order": 1.5002941662044993,
        },
        resistive_diffusion_convergence={
            "method": "implicit",
            "resolutions": [16, 32, 64],
            "errors": [
                4.222374566471515e-2,
                1.0642732372904487e-2,
                2.618808803093956e-3,
            ],
            "convergence_order": 2.005535963474413,
            "eta": 1.2566370612700002e-7,
        },
        applicable_phases=["formation", "rundown", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "beyond ideal MHD"],
    )
    status = packet["mhd_numerical_verification_packet_status"]

    assert packet["passed"] is False
    assert packet["production_packet_status"] == "blocked"
    assert packet["experimental_dpf_validation"] is False
    assert set(status["attached_validated_packets"]) == {
        "cylindrical_geometry_verification",
        "resistive_or_nonideal_verification",
        "convergence_study",
        "dpf_scope_limit",
    }
    assert set(status["missing_required_packets"]) == {
        "finite_volume_mhd_verification",
        "circuit_coupled_energy_verification",
        "backend_parity",
        "restart_reproducibility",
        "limiter_zero_acceptance",
    }


def test_packet_builder_can_close_complete_same_scope_tier3_packet():
    scope_id = "complete_scheduled_tier3_cpu_mhd_numerical"
    circuit = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope=scope_id,
    )

    packet = build_mhd_numerical_verification_packet(
        verification_scope=scope_id,
        mhd_numerical_method={
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
        mhd_verification={
            "passed": True,
            "validation_tier": 3,
            "model_role": "code_verification_analytic_tests",
            "analytic_tests": {"sod": True, "brio_wu": True},
        },
        cylindrical_convergence={
            "resolutions": [32, 64, 128],
            "Btheta_errors": [4.0e-3, 1.8e-3, 8.0e-4],
            "convergence_order": 1.15,
        },
        circuit_coupled_energy_verification=circuit,
        resistive_diffusion_convergence={
            "method": "implicit",
            "resolutions": [32, 64, 128],
            "errors": [1.0e-2, 4.5e-3, 2.0e-3],
            "convergence_order": 1.1,
            "eta": 1.0e-7,
        },
        backend_parity_results={
            "backends": {
                "python": {"I_peak_MA": 1.20, "pinch_time_ns": 95.0},
                "metal": {"I_peak_MA": 1.19, "pinch_time_ns": 96.0},
            },
            "relative_tolerances": {"I_peak_MA": 0.02, "pinch_time_ns": 0.02},
        },
        restart_reproducibility_results={
            "continuous": {"I_peak_MA": 1.2000000000, "pinch_time_ns": 95.0000000},
            "restarted": {"I_peak_MA": 1.2000000002, "pinch_time_ns": 95.0000001},
            "relative_tolerances": {"I_peak_MA": 1.0e-8, "pinch_time_ns": 1.0e-8},
            "restart_step": 512,
            "config_hash": "cfg-sha",
            "restart_config_hash": "cfg-sha",
        },
        limiter_zero_ledger={
            "status": "clear",
            "entry_count": 4,
            "acceptance_blocking_activation_count": 0,
        },
        applicable_phases=["formation", "first_collapse"],
        invalid_phases=["after_first_collapse", "post_disruption"],
        limit_reasons=["Rayleigh-Taylor instability", "non-ideal electric fields"],
    )

    assert packet["passed"] is True
    assert packet["production_packet_status"] == "complete"
    assert packet["mhd_numerical_fidelity"]["same_scope_passed"] is True
    assert (
        packet["mhd_numerical_verification_packet_status"][
            "missing_required_packets"
        ]
        == []
    )
