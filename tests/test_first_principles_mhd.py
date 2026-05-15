"""Fail-closed tests for the first-principles MHD run mode."""

from __future__ import annotations

from dpf.validation.first_principles_mhd import (
    FIRST_PRINCIPLES_MHD_EXECUTION_MODE,
    FIRST_PRINCIPLES_MHD_MODE,
    PF1000_AKEL_SOURCE_SCOPE,
    PF1000_AKEL_VALIDATION_SCOPE,
    annotate_first_principles_mhd_result,
    first_principles_backend_scope_status,
    first_principles_energy_accounting_status,
    first_principles_intervals_from_sources,
    first_principles_mhd_readiness_report,
    first_principles_neutron_yield_authority_status,
    first_principles_startup_initialization_status,
    normalize_first_principles_run_mode,
    reduced_model_baseline_authority,
)
from dpf.validation.first_principles_limiters import (
    limiter_event,
    summarize_limiter_ledger,
)
from dpf.validation.hybrid_pic_3d import (
    HYBRID_PIC_3D_CAPABILITY_IDS,
    HYBRID_PIC_3D_SOURCE,
    hybrid_pic_3d_readiness_status,
)


def test_first_principles_mode_normalizes_to_mhd_execution() -> None:
    execution_mode, requested = normalize_first_principles_run_mode(
        FIRST_PRINCIPLES_MHD_MODE
    )

    assert execution_mode == FIRST_PRINCIPLES_MHD_EXECUTION_MODE
    assert requested is True
    assert normalize_first_principles_run_mode("mhd") == ("mhd", False)


def test_first_principles_readiness_blocks_closure_factors_and_missing_fields() -> None:
    result = {
        "I_MA": [0.0, 0.5],
        "V_kV": [16.0, 15.5],
        "Lp_snowplow_nH": [0.0, 1.2],
        "phases": ["rundown", "rundown"],
        "fc": 0.70,
        "fm": 0.17,
    }

    readiness = first_principles_mhd_readiness_report(
        result,
        preset_name="pf1000_akel",
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_blocked_by_review",
    )

    assert readiness.ready is False
    assert readiness.status == "blocked"
    assert readiness.closure_factor_status["present"] is True
    assert "fc" in readiness.closure_factor_status["keys"]
    assert "reduced_model_active_closure_rejected" in readiness.missing_evidence
    assert any("Lee/RADPF closure factors" in blocker for blocker in readiness.blockers)
    assert (
        readiness.output_status["field_derived_plasma_inductance"]["status"]
        == "missing"
    )
    assert (
        readiness.reduced_model_baselines["lee_snowplow"][
            "can_support_first_principles_acceptance"
        ]
        is False
    )


def test_readiness_blocks_acceptance_blocking_limiter_activation() -> None:
    ledger = summarize_limiter_ledger([
        limiter_event(
            limiter_id="fp2.velocity_cap",
            code_path="unit.test",
            affected_field="velocity",
            classification="acceptance_blocker",
            activation_count=2,
            before=[3.0e6, 4.0e6],
            after=[2.0e6, 2.0e6],
            threshold={"cap_m_s": 2.0e6},
            acceptance_blocking=True,
            justification="Synthetic acceptance-blocking limiter activation.",
        )
    ])
    readiness = first_principles_mhd_readiness_report(
        {"first_principles_limiter_ledger": ledger},
        preset_name="pf1000_akel",
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_source_reviewed_not_certificate",
    )

    assert readiness.ready is False
    assert readiness.status == "blocked"
    assert "acceptance_blocking_limiter_activation" in readiness.missing_evidence
    assert readiness.limiter_ledger_status[
        "acceptance_blocking_activation_count"
    ] == 2
    assert (
        "fp2.velocity_cap"
        in readiness.limiter_ledger_status["activated_acceptance_blockers"]
    )
    assert any("acceptance-blocking limiter" in item for item in readiness.blockers)


def test_verified_numerical_method_events_do_not_block_limiter_zero_gate() -> None:
    ledger = summarize_limiter_ledger([
        limiter_event(
            limiter_id="dpf.fluid.cylindrical_mhd.plm_minmod_reconstruction",
            code_path="dpf.fluid.cylindrical_mhd.CylindricalMHDSolver._plm_reconstruct",
            affected_field="reconstructed_state",
            classification="verified_numerical_method",
            activation_count=0,
            threshold={"verification_tests": ["tests/test_cylindrical_godunov.py"]},
            acceptance_blocking=False,
            justification="Synthetic verified numerical-method registry entry.",
        )
    ])

    assert ledger["status"] == "clear"
    assert ledger["acceptance_blocking_activation_count"] == 0
    assert ledger["can_support_first_principles_acceptance"] is True
    assert ledger["entries"][0]["classification"] == "verified_numerical_method"
    assert ledger["entries"][0]["acceptance_blocking"] is False


def test_first_principles_backend_scope_rejects_uninstrumented_metal() -> None:
    status = first_principles_backend_scope_status({
        "backend": "metal_plm",
        "requested_backend": "metal_plm",
        "requested_run_mode": "first_principles_mhd",
    })

    assert status["status"] == "backend_scope_blocked"
    assert status["can_support_first_principles_acceptance"] is False
    assert status["blocked_backend"] == "metal"

    readiness = first_principles_mhd_readiness_report(
        {
            "backend": "metal_plm",
            "requested_backend": "metal_plm",
            "requested_run_mode": "first_principles_mhd",
            "first_principles_limiter_ledger": summarize_limiter_ledger([]),
        },
        preset_name="pf1000_akel",
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_source_reviewed_not_certificate",
    )

    assert "instrumented_backend_scope" in readiness.missing_evidence
    assert any("outside first-principles acceptance scope" in item for item in readiness.blockers)


def test_first_principles_backend_scope_rejects_each_uninstrumented_backend() -> None:
    cases = [
        ("metal_plm", "metal_plm", "metal"),
        ("metal_weno5", "metal_weno5", "metal"),
        ("mlx", "mlx", "mlx"),
        ("athena", "athena", "athena"),
        ("athenak", "athenak", "athenak"),
        ("hybrid", "hybrid", "hybrid"),
        ("metal_plm (fallback from athena)", "athena", "athena"),
    ]

    for backend, requested_backend, blocked_backend in cases:
        status = first_principles_backend_scope_status({
            "backend": backend,
            "requested_backend": requested_backend,
            "requested_run_mode": "first_principles_mhd",
        })
        assert status["status"] == "backend_scope_blocked", backend
        assert status["blocked_backend"] == blocked_backend
        assert status["can_support_first_principles_acceptance"] is False
        assert status["required_limiter_telemetry"] == (
            "backend_native_first_principles_limiter_ledger"
        )


def test_hybrid_pic_3d_gate_blocks_current_mhd_path_without_evidence() -> None:
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "2d_axisymmetric",
    })

    assert status["status"] == "blocked"
    assert status["can_support_first_principles_acceptance"] is False
    assert status["source"] == HYBRID_PIC_3D_SOURCE
    assert "explicit_3d_geometry" in status["missing_capabilities"]
    assert "kinetic_ion_pic_push_deposition" in status["missing_capabilities"]
    assert (
        status["capabilities"]["full_maxwell_vacuum_plasma_fields"]["source"]
        == HYBRID_PIC_3D_SOURCE
    )


def test_hybrid_pic_3d_gate_is_public_validation_api() -> None:
    import dpf.validation as validation

    assert "hybrid_pic_3d_readiness_status" in validation.__all__
    assert validation.HYBRID_PIC_3D_SOURCE == HYBRID_PIC_3D_SOURCE
    assert validation.HYBRID_PIC_3D_CAPABILITY_IDS == HYBRID_PIC_3D_CAPABILITY_IDS
    assert validation.hybrid_pic_3d_readiness_status({})["status"] == "blocked"


def test_hybrid_pic_3d_gate_requires_all_reviewed_capabilities() -> None:
    evidence = {
        capability_id: {
            "passed": True,
            "status": "accepted",
            "source": HYBRID_PIC_3D_SOURCE,
        }
        for capability_id in HYBRID_PIC_3D_CAPABILITY_IDS
    }

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": evidence,
    })

    assert status["status"] == "accepted"
    assert status["can_support_first_principles_acceptance"] is True
    assert status["missing_capabilities"] == []
    assert set(status["satisfied_capabilities"]) == set(HYBRID_PIC_3D_CAPABILITY_IDS)


def test_reduced_model_outputs_are_baseline_only() -> None:
    authority = reduced_model_baseline_authority({
        "Lp_snowplow_nH": [1.0, 1.2],
        "phase_model_authority": {"model": "reduced_mlx_snowplow"},
        "snowplow_cfg": {
            "current_fraction": 0.7,
            "mass_fraction": 0.17,
            "radial_current_fraction": 0.75,
        },
    })
    snowplow = authority["lee_snowplow"]

    assert snowplow["output_role"] == "baseline_reduced_model"
    assert snowplow["validation_status"] == "not_first_principles_evidence"
    assert snowplow["can_support_first_principles_acceptance"] is False
    assert "snowplow_cfg.current_fraction" in snowplow["closure_factor_keys"]


def test_energy_accounting_requires_field_poynting_and_residual() -> None:
    status = first_principles_energy_accounting_status({
        "Lp_mhd_nH": [1.0, 1.2, 1.4],
        "back_emf_V": [0.0, 10.0, 20.0],
        "E_cap_kJ": [170.0, 160.0, 150.0],
        "E_ind_kJ": [0.0, 5.0, 8.0],
        "E_res_kJ": [0.0, 1.0, 2.0],
    })

    assert status["status"] == "incomplete"
    assert status["can_support_first_principles_acceptance"] is False
    assert status["required_channels"]["circuit_energy_channels"]["present"] is True
    assert "field_poynting_power" in status["missing_channels"]
    assert "circuit_energy_residual" in status["missing_channels"]


def test_complete_energy_accounting_remains_candidate_without_validation() -> None:
    status = first_principles_energy_accounting_status({
        "Lp_mhd_nH": [1.0, 1.2, 1.4],
        "back_emf_V": [0.0, 10.0, 20.0],
        "poynting_power_W": [0.0, 1.0e6, 1.2e6],
        "E_cap_kJ": [170.0, 160.0, 150.0],
        "E_ind_kJ": [0.0, 5.0, 8.0],
        "E_res_kJ": [0.0, 1.0, 2.0],
        "circuit_energy_residual_kJ": [0.0, 4.0, 10.0],
    })

    assert status["status"] == "complete_candidate_not_validated"
    assert status["missing_channels"] == []
    assert status["validated"] is False
    assert status["can_support_first_principles_acceptance"] is False


def test_neutron_authority_blocks_reduced_beam_target_total_yield() -> None:
    status = first_principles_neutron_yield_authority_status({
        "neutron_yield_details": {
            "Y_thermonuclear": 6.0e7,
            "Y_beam_target": 4.0e7,
            "Y_neutron": 1.0e8,
        },
        "yield_time_resolved": {
            "t_s": [0.0, 1.0e-9],
            "dY_thermo": [0.0, 6.0e7],
            "dY_bt": [0.0, 4.0e7],
        },
        "neutron_yield_validation": {"passed": True},
        "neutron_mechanism_timing_validation": {"passed": True},
        "neutron_spectrum_validation": {"passed": True},
        "neutron_anisotropy_validation": {"passed": True},
        "neutron_detector_response_validation": {"passed": True},
        "neutron_uncertainty_validation": {"passed": True},
        "mhd_numerical_fidelity": {"passed": True},
        "physics_fidelity_evidence": {"passed": True},
    })

    assert status["passed"] is False
    assert status["can_support_first_principles_acceptance"] is False
    assert status["ten_percent_paper_yield_accuracy"]["status"] == "blocked"
    assert (
        status["mechanisms"]["thermonuclear"]["authority"]
        == "resolved_field_history_candidate"
    )
    assert status["mechanisms"]["beam_target"]["authority"] == "baseline_reduced_model"
    assert "kinetic_or_hybrid_beam_target_model_missing" in status["blockers"]
    assert "beam_target_yield_uses_reduced_or_calibrated_model" in status["blockers"]


def test_neutron_authority_blocks_final_state_duration_approximation() -> None:
    status = first_principles_neutron_yield_authority_status({
        "final_state": {"rho": object(), "pressure": object()},
        "neutron_yield": {
            "Y_thermonuclear": 1.0e6,
            "Y_beam_target": 0.0,
            "Y_neutron": 1.0e6,
        },
    })

    assert status["passed"] is False
    assert (
        status["mechanisms"]["thermonuclear"]["authority"]
        == "final_state_duration_approximation"
    )
    assert "thermonuclear_yield_not_integrated_from_field_history" in status["blockers"]
    assert "thermonuclear_yield_uses_final_state_duration_approximation" in status[
        "blockers"
    ]


def test_startup_initialization_blocks_seeded_sheath_scaffold() -> None:
    status = first_principles_startup_initialization_status({
        "startup_sheath_initialization": {
            "classification": "engineering_initialization_scaffold",
            "can_support_first_principles_startup": False,
        },
        "electrode_boundary_conditions": {
            "classification": "implemented_not_validated",
        },
        "z_sheath_cm": [0.0, 0.1, 0.2],
    })

    assert status["status"] == "incomplete"
    assert status["can_support_first_principles_acceptance"] is False
    assert "breakdown_model" in status["missing_channels"]
    assert "preionization_state" in status["missing_channels"]
    assert status["required_channels"]["resolved_sheath_position"]["present"] is True


def test_scope_mixing_blocks_standard_pf1000_in_first_principles_mode() -> None:
    readiness = first_principles_mhd_readiness_report(
        {"I_MA": [0.0, 1.0], "V_kV": [27.0, 25.0]},
        preset_name="pf1000",
        validation_scope="",
        source_scope="pf1000_standard_27kv_lee_malek",
        source_scope_status="same_scope_source_reviewed_not_certificate",
    )

    assert readiness.ready is False
    assert "pf1000_akel_same_scope" in readiness.missing_evidence
    assert any("initially scoped to PF-1000/Akel" in item for item in readiness.blockers)


def test_annotation_exports_fail_closed_intervals_and_readiness() -> None:
    result = {
        "t_us": [0.0, 0.1, 0.2],
        "I_MA": [0.0, 0.6, 0.9],
        "V_kV": [16.0, 15.0, 14.0],
        "Lp_snowplow_nH": [1.0, 1.2, 1.3],
        "Lp_mhd_nH": [0.8, 1.1, 1.4],
        "back_emf_V": [0.0, 10.0, 20.0],
        "phases": ["rundown", "rundown", "radial"],
        "coupling_source": ["snowplow", "mhd_blend", "mhd"],
        "fc": 0.70,
    }

    annotated = annotate_first_principles_mhd_result(
        result,
        preset_name="pf1000_akel",
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_blocked_by_review",
    )

    labels = {
        item["interval_label"]
        for item in first_principles_intervals_from_sources(
            annotated["t_us"],
            annotated["coupling_source"],
        )
    }
    readiness = annotated["first_principles_mhd_readiness"]

    assert labels == {"snowplow_loaded", "handoff", "field_coupled"}
    assert annotated["run_mode"] == FIRST_PRINCIPLES_MHD_MODE
    assert annotated["execution_mode"] == FIRST_PRINCIPLES_MHD_EXECUTION_MODE
    assert annotated["first_principles_energy_accounting"]["status"] == "incomplete"
    assert (
        annotated["first_principles_startup_initialization"]["status"]
        == "incomplete"
    )
    assert annotated["field_coupling_validation"]["passed"] is False
    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    assert "validated_field_coupling_packet" in readiness["missing_evidence"]
    assert "field_coupled_energy_accounting" in readiness["missing_evidence"]
    assert "first_principles_startup_initialization" in readiness["missing_evidence"]
    assert "first_principles_neutron_yield_authority" in readiness["missing_evidence"]
    assert "hybrid_pic_3d_first_principles_core" in readiness["missing_evidence"]
    assert annotated["hybrid_pic_3d_readiness"]["status"] == "blocked"
    assert (
        annotated["first_principles_neutron_yield_authority"]["status"]
        == "not_produced"
    )


def test_simulation_engine_summary_carries_first_principles_readiness() -> None:
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-10,
        run_mode=FIRST_PRINCIPLES_MHD_MODE,
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_blocked_by_review",
        preset_name="pf1000_akel",
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": ":memory:"},
    )

    summary = SimulationEngine(config).run(max_steps=1)

    readiness = summary["first_principles_mhd_readiness"]
    assert summary["run_mode"] == FIRST_PRINCIPLES_MHD_MODE
    assert readiness["status"] == "blocked"
    assert readiness["source_scope"] == PF1000_AKEL_SOURCE_SCOPE
    assert readiness["validation_scope"] == PF1000_AKEL_VALIDATION_SCOPE
    assert "accepted_same_scope_akel_digitization" in readiness["missing_evidence"]
    assert "field_coupled_energy_accounting" in readiness["missing_evidence"]
    assert "run_manifest" in summary
