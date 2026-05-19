"""Tests for simulation quality assessment."""

from __future__ import annotations

from types import SimpleNamespace

from dpf.validation.circuit_field_coupling import (
    field_coupling_component_evidence,
    field_coupling_evidence_from_result,
)
from dpf.validation.digitization import (
    akel_fig1_draft_digitization_packet,
    scientific_closure_digitization_status,
)
from dpf.validation.physics_fidelity import (
    physics_effect_validation_evidence,
    physics_fidelity_evidence_from_result,
)
from dpf.validation.quality_assessment import (
    assess_quality,
    circuit_validation_evidence_from_result,
    circuit_validation_evidence_from_waveform,
    combine_spatial_validation_evidence,
    high_fidelity_readiness_report,
    mhd_verification_evidence_from_shock_tube_results,
    mhd_verification_evidence_from_tests,
    neutron_timing_validation_evidence_from_errors,
    neutron_validation_scope_closure_report,
    predictive_readiness_report,
    scientific_accuracy_gap_report,
    snowplow_phase_observation_from_history,
    snowplow_validation_evidence_from_phase_errors,
    snowplow_validation_evidence_from_phase_history,
    source_authority_evidence,
    source_authority_evidence_from_result,
    spatial_validation_evidence_from_quantity_errors,
    spatial_validation_scope_closure_report,
    validation_tier_report,
)
from dpf.validation.suite import MetricResult, ValidationResult
from dpf.validation.uncertainty_budget import (
    uncertainty_component_evidence,
    uncertainty_evidence_from_result,
)


def _kr_neutron_yield_evidence(
    scope: str = "KnowledgeReference/neutron.md",
) -> dict[str, object]:
    return {
        "passed": True,
        "validated_features": {"yield": True},
        "validation_scope": scope,
        "source": (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-"
            "ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "source_lines": "548-616",
        "validation_tier": 5,
        "model_role": "simulation_to_kr_target_comparison",
    }


def _kr_neutron_detector_response_evidence(
    scope: str = "KnowledgeReference/neutron.md",
) -> dict[str, object]:
    return {
        "passed": True,
        "validated_features": {"detector_response": True},
        "diagnostics": {"activation_response": True},
        "validation_scope": scope,
        "source": (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-"
            "ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "source_lines": "132-168,449-509,595-607",
        "validation_tier": 5,
        "model_role": "simulation_to_kr_detector_response_audit",
    }


def _kr_neutron_uncertainty_evidence(
    scope: str = "KnowledgeReference/neutron.md",
) -> dict[str, object]:
    return {
        "passed": True,
        "validated_features": {"uncertainty": True},
        "validation_scope": scope,
        "source": (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-"
            "ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "source_lines": "565-604",
        "validation_tier": 5,
        "model_role": "simulation_to_kr_neutron_uncertainty_comparison",
        "source_uncertainty_values": {
            "yield_relative_sigma": 0.10,
            "arrival_time_sigma_ns": 1.0,
        },
    }


class TestQualityAssessment:
    def _good_result(self) -> dict:
        return {
            "I_peak": 1.733,
            "t_peak": 5.8,
            "dip_pct": 15.0,
            "n_steps": 500,
            "has_snowplow": True,
            "has_mhd": False,
            "bennett": {"T_bennett_keV": 0.48},
            "neutron_yield": {"Y_neutron": 1e8, "bt_fraction": 0.6},
            "breakdown": {"mechanism": "Paschen", "civ_ratio": 11020.0},
        }

    def test_good_result_high_grade(self):
        qa = assess_quality(self._good_result())
        assert qa.grade in ("A", "B")
        assert qa.score > 0.5
        assert qa.n_passed >= 4

    def test_empty_result_f_grade(self):
        qa = assess_quality({"I_peak": 0, "n_steps": 0})
        assert qa.grade in ("D", "F")
        assert qa.n_critical_failures > 0

    def test_no_dip_warning(self):
        r = self._good_result()
        r["dip_pct"] = 0
        qa = assess_quality(r)
        dip_check = [c for c in qa.checks if c.name == "Current dip"]
        assert len(dip_check) == 1
        assert not dip_check[0].passed

    def test_summary_not_empty(self):
        qa = assess_quality(self._good_result())
        assert len(qa.summary) > 50
        assert "Grade" in qa.summary
        assert "Validation tiers" in qa.summary
        assert "Predictive readiness" in qa.summary

    def test_mhd_low_compression(self):
        import numpy as np
        r = {
            "I_peak": 1.0, "n_steps": 200,
            "has_mhd": True, "has_snowplow": False,
            "rho_max": np.array([1.5e-4]), "rho0": 1e-4,
        }
        qa = assess_quality(r)
        comp_check = [c for c in qa.checks if c.name == "Density compression"]
        assert len(comp_check) == 1
        assert not comp_check[0].passed  # 1.5x < 2.0 threshold

    def test_with_regime(self):
        r = self._good_result()
        r["plasma_regime"] = {
            "knudsen": 4.5, "mhd_valid": False,
            "summary": "Kinetic regime", "kinetic_needed": True,
        }
        qa = assess_quality(r)
        regime_check = [c for c in qa.checks if c.name == "Regime validity"]
        assert len(regime_check) == 1

    def test_validation_tiers_separate_circuit_from_spatial_validation(self):
        qa = assess_quality(self._good_result())
        tiers = {t.level: t for t in qa.validation_tiers}
        assert tiers[1].status == "diagnostic_present"
        assert tiers[2].status == "partial"
        assert tiers[4].status == "not_validated"
        assert "spatial MHD" in tiers[1].limitation
        assert tiers[5].status == "decomposed_estimate"
        assert qa.predictive_readiness is not None
        assert qa.predictive_readiness.ready is False

    def test_mhd_result_is_verification_only_without_spatial_data(self):
        r = {
            "I_peak": 1.0,
            "n_steps": 200,
            "has_mhd": True,
            "has_snowplow": False,
            "rho_max": [5.0e-4],
            "rho0": 1.0e-4,
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[3].status == "verification_only"
        assert tiers[4].status == "not_validated"

    def test_neutron_details_are_marked_estimate_until_timing_validated(self):
        r = {"I_peak": 1.0, "n_steps": 20, "neutron_yield_details": {"Y_neutron": 1e8}}
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"
        assert "kinetic beam" in tiers[5].limitation

    def test_predictive_readiness_blocks_good_quality_without_required_evidence(self):
        readiness = predictive_readiness_report(self._good_result())
        assert readiness.ready is False
        assert readiness.status == "not_predictive_ready"
        assert "Spatial DPF experimental validation" in readiness.missing_evidence
        assert any("T4" in blocker for blocker in readiness.blockers)

    def test_scientific_accuracy_gap_report_lists_remaining_work(self):
        gaps = {
            gap.area: gap for gap in scientific_accuracy_gap_report(self._good_result())
        }

        assert gaps["source_authority_data"].status == "partial"
        assert "registered devices" in gaps["source_authority_data"].blocker
        assert gaps["kr_source_review"].status == "partial"
        assert "DPF-relevant KnowledgeReference markdown files still need" in (
            gaps["kr_source_review"].blocker
        )
        assert gaps["kr_target_coverage"].status == "partial"
        assert "phase_timing" in gaps["kr_target_coverage"].blocker
        assert "Widest same-scope closure path" in gaps["kr_target_coverage"].blocker
        assert gaps["figure_digitization"].status == "blocked"
        assert "0/6 local scientific-closure figure" in (
            gaps["figure_digitization"].blocker
        )
        assert gaps["same_scope_high_fidelity_claim"].status == "partial"
        assert gaps["spatial_dpf_validation"].status == "blocked"
        assert "density" in gaps["spatial_dpf_validation"].next_ratcheting_step
        assert gaps["neutron_validation"].status == "partial"
        assert "timing" in gaps["neutron_validation"].done_condition
        assert gaps["missing_physics_fidelity"].status == "blocked"
        assert gaps["export_claim_hygiene"].status == "partial"

    def test_gap_report_distinguishes_draft_digitization_from_missing_packet(self):
        digitization_status = scientific_closure_digitization_status([
            akel_fig1_draft_digitization_packet()
        ])

        gaps = {
            gap.area: gap
            for gap in scientific_accuracy_gap_report({
                **self._good_result(),
                "scientific_closure_digitization_status": digitization_status,
            })
        }

        assert gaps["figure_digitization"].status == "blocked"
        assert "1 draft/failed packet(s) need review" in (
            gaps["figure_digitization"].blocker
        )
        assert "5 remain open" in gaps["figure_digitization"].blocker

    def test_result_level_source_authority_evidence_can_support_gap(self):
        evidence = source_authority_evidence(
            validation_scope="pf1000_shot_12581",
            sources=[
                "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
            ],
            source_lines=["250-285"],
            provenance="published_table",
        )
        gaps = {
            gap.area: gap
            for gap in scientific_accuracy_gap_report({
                "source_authority_validation": evidence,
            })
        }

        assert evidence["passed"] is True
        assert gaps["source_authority_data"].status == "supported"

    def test_source_authority_evidence_from_result_collects_kr_lines(self):
        result = {
            "neutron_spectrum_validation": {
                "passed": True,
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
                "validation_scope": "mjolnir_neutron_timing_2025_goyon",
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "source_lines": {
                    "spectrum_anisotropy": "548-616",
                },
            },
            "mhd_numerical_fidelity": {
                "passed": True,
                "validation_tier": "mhd_numerical_fidelity",
                "model_role": "mhd_numerical_fidelity_audit",
                "source": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
                "required_evidence": {
                    "backend_parity": {
                        "status": "supported",
                        "validated": True,
                        "source": (
                            "KnowledgeReference/"
                            "beresnyak_2022_pulsed_power_ideal_mhd.md"
                        ),
                        "source_lines": "1900-1903, 1939-1955",
                    },
                },
            },
        }

        evidence = source_authority_evidence_from_result(result)

        assert evidence["passed"] is True
        assert evidence["source"].startswith("KnowledgeReference/")
        assert "548-616" in evidence["source_lines"]
        assert "1900-1903, 1939-1955" in evidence["source_lines"]
        assert "neutron_spectrum_validation" in evidence["details"]["claimed_evidence"]
        assert "mhd_numerical_fidelity" in evidence["details"]["claimed_evidence"]

    def test_source_authority_evidence_from_result_rejects_unlined_claim(self):
        result = {
            "neutron_spectrum_validation": {
                "passed": True,
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
                "validation_scope": "mjolnir_neutron_timing_2025_goyon",
                "source": "ExternalArchive/neutron.md",
            },
        }

        evidence = source_authority_evidence_from_result(result)

        assert evidence["passed"] is False
        assert evidence["source"] == ""
        missing = evidence["details"]["missing_source_authority"]
        assert missing[0]["evidence_key"] == "neutron_spectrum_validation"

    def test_source_authority_evidence_rejects_missing_kr_file(self):
        evidence = source_authority_evidence(
            validation_scope="missing_file_scope",
            sources=["KnowledgeReference/not-a-real-source.md"],
            source_lines=["1-2"],
        )

        assert evidence["passed"] is False

    def test_source_authority_evidence_rejects_out_of_bounds_line_range(self):
        evidence = source_authority_evidence(
            validation_scope="bad_line_range_scope",
            sources=[
                "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
            ],
            source_lines=["999999-1000000"],
        )

        assert evidence["passed"] is False

    def test_gap_report_uses_failed_result_source_authority_as_blocker(self):
        source_audit = source_authority_evidence_from_result({
            "neutron_spectrum_validation": {
                "passed": True,
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
                "validation_scope": "mjolnir_neutron_timing_2025_goyon",
                "source": "ExternalArchive/neutron.md",
            },
        })

        gaps = {
            gap.area: gap
            for gap in scientific_accuracy_gap_report({
                "source_authority_validation": source_audit,
            })
        }

        assert source_audit["passed"] is False
        assert gaps["source_authority_data"].status == "blocked"
        assert "neutron_spectrum_validation" in gaps["source_authority_data"].blocker

    def test_gap_report_cross_checks_manual_source_authority_packet(self):
        manual_source_audit = source_authority_evidence(
            validation_scope="manual_packet",
            sources=[
                "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md",
            ],
            source_lines=["548-616"],
        )
        gaps = {
            gap.area: gap
            for gap in scientific_accuracy_gap_report({
                "source_authority_validation": manual_source_audit,
                "neutron_spectrum_validation": {
                    "passed": True,
                    "validation_tier": 5,
                    "model_role": "simulation_to_kr_target_comparison",
                    "validation_scope": "mjolnir_neutron_timing_2025_goyon",
                    "source": (
                        "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                        "ma-class-dense-plasma-focus-z-pinch-5.md"
                    ),
                },
            })
        }

        assert manual_source_audit["passed"] is True
        assert gaps["source_authority_data"].status == "blocked"
        assert "neutron_spectrum_validation" in gaps["source_authority_data"].blocker

    def test_neutron_gap_requires_detector_response_beyond_tier5_birth_evidence(self):
        r = self._good_result()
        scope = "mjolnir_neutron_timing_2025_goyon"
        r.update({
            "neutron_yield_validation": _kr_neutron_yield_evidence(scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": {"thermonuclear": True, "beam_target": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": {"spectrum": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": {"anisotropy": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
        })

        gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(r)}
        assert gaps["neutron_validation"].status == "partial"
        assert "detector/activation response status is not_supported" in (
            gaps["neutron_validation"].blocker
        )

        r["neutron_detector_response_validation"] = (
            _kr_neutron_detector_response_evidence(scope)
        )
        gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(r)}
        assert gaps["neutron_validation"].status == "partial"

        r["neutron_uncertainty_validation"] = _kr_neutron_uncertainty_evidence(scope)
        gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(r)}
        assert gaps["neutron_validation"].status == "supported"

    def test_high_fidelity_readiness_requires_gap_closure(self):
        r = self._good_result()
        scope = "synthetic_predictive_scope"
        r.update({
            "circuit_validation": {
                "passed": True,
                "metrics": ["peak_current", "peak_time", "waveform_shape"],
                "validation_scope": scope,
                "details": {"source_authority": {"passed": True}},
            },
            "snowplow_validation": {
                "passed": True,
                "phases": ["axial", "radial", "pinch"],
                "validation_scope": scope,
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": ["Sod", "Brio-Wu"],
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "validation_scope": scope,
                "source": "KnowledgeReference/spatial.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence(scope)
            ),
            "neutron_uncertainty_validation": _kr_neutron_uncertainty_evidence(scope),
        })

        readiness = high_fidelity_readiness_report(r)

        assert predictive_readiness_report(r).ready is True
        assert readiness.ready is False
        assert readiness.status == "scientific_accuracy_gaps_open"
        assert "missing_physics_fidelity" in readiness.remaining_areas
        assert any("uncertainty_quantification" in blocker for blocker in readiness.blockers)

    def test_high_fidelity_readiness_can_pass_with_complete_evidence_packet(self):
        r = self._good_result()
        r.update({
            "source_authority_validation": source_authority_evidence(
                validation_scope="synthetic_complete_high_fidelity_scope",
                sources=[
                    "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
                ],
                source_lines=["250-285"],
                provenance="published_table",
            ),
            "circuit_validation": {
                "passed": True,
                "metrics": ["peak_current", "peak_time", "waveform_shape"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
                "source_lines": "250-285",
                "details": {"source_authority": {"passed": True}},
            },
            "snowplow_validation": {
                "passed": True,
                "phases": ["axial", "radial", "pinch"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
                "source_lines": "250-285",
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": ["Sod", "Brio-Wu"],
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
                "source": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
                "source_lines": "336-347",
            },
            "mhd_numerical_fidelity": {
                "passed": True,
                "source": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
                "source_lines": "336-347",
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": "KnowledgeReference/malir-2024-interferometry-dpf.md",
                "source_lines": "331-348",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(
                "synthetic_complete_high_fidelity_scope"
            ),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "source_lines": "405-448",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "source_lines": "548-616",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "validation_scope": "synthetic_complete_high_fidelity_scope",
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "source_lines": "595-607",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence(
                    "synthetic_complete_high_fidelity_scope"
                )
            ),
            "neutron_uncertainty_validation": (
                _kr_neutron_uncertainty_evidence(
                    "synthetic_complete_high_fidelity_scope"
                )
            ),
            "field_coupling_validation": field_coupling_evidence_from_result({
                "field_coupling_component_validations": {
                    component: field_coupling_component_evidence(
                        component,
                        validation_scope="synthetic_complete_high_fidelity_scope",
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
                },
            }),
            "physics_fidelity_evidence": physics_fidelity_evidence_from_result({
                "physics_effect_validations": {
                    effect: physics_effect_validation_evidence(
                        effect,
                        validation_scope="synthetic_complete_high_fidelity_scope",
                    )
                    for effect in (
                        "tabulated_eos_and_conductivity",
                        "ionization_and_charge_state_kinetics",
                        "two_temperature_energy_partition",
                        "radiation_transport_opacity",
                        "material_ablation_impurity_mixing",
                        "hall_flr_kinetic_or_pic_effects",
                        "three_dimensional_instabilities",
                        "flashover_sheath_initiation",
                        "restrike_and_anomalous_resistance",
                        "beam_generation_and_target_coupling",
                    )
                },
            }),
            "uncertainty_validation": uncertainty_evidence_from_result({
                "uncertainty_component_validations": {
                    component: uncertainty_component_evidence(
                        component,
                        validation_scope="synthetic_complete_high_fidelity_scope",
                        source_uncertainty_values={
                            f"{component}_relative_sigma": 0.05,
                        },
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
                },
            }),
            "kr_corpus_review_status": {
                "model_role": "kr_corpus_review_status",
                "reviewed_dpf_relevant_md_files": 1,
                "unreviewed_dpf_relevant_md_files": [],
                "corpus_counts": {
                    "dpf_relevant_md_files": 1,
                },
            },
            "kr_validation_target_coverage": {
                "passed": True,
                "model_role": "kr_validation_target_coverage_report",
                "missing_or_partial_groups": [],
                "groups": [],
            },
            "kr_validation_target_semantic_audit": {
                "passed": True,
                "model_role": "kr_validation_target_semantic_audit",
                "missing_or_failed_targets": [],
            },
            "kr_validation_same_scope_targets": {
                "passed": True,
                "model_role": "kr_validation_same_scope_target_report",
                "passed_scopes": ["synthetic_complete_high_fidelity_scope"],
                "best_available_scope": {
                    "validation_scope": "synthetic_complete_high_fidelity_scope",
                    "missing_groups": [],
                    "partial_groups": [],
                },
            },
            "scientific_closure_digitization_status": {
                "model_role": "scientific_closure_digitization_status",
                "queue_complete": True,
                "accepted_task_count": 6,
                "failed_task_count": 0,
                "open_task_count": 0,
                "task_count": 6,
            },
            "validation_tiers": [{"level": 1, "status": "supported"}],
            "predictive_readiness": {"status": "predictive_ready"},
        })

        readiness = high_fidelity_readiness_report(r)
        gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(r)}

        assert predictive_readiness_report(r).ready is True
        assert gaps["same_scope_high_fidelity_claim"].status == "supported", (
            gaps["same_scope_high_fidelity_claim"].blocker
        )
        assert readiness.ready is True
        assert readiness.status == "high_fidelity_ready"
        assert readiness.remaining_areas == []

    def test_high_fidelity_scope_alignment_blocks_cross_scope_packets(self):
        field_scope = "synthetic_field_scope"
        physics_scope = "synthetic_physics_scope"
        uq_scope = "synthetic_uq_scope"
        target_scope = "synthetic_target_scope"
        result = {
            "source_authority_validation": source_authority_evidence(
                validation_scope=target_scope,
                sources=[
                    "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
                ],
                source_lines=["250-285"],
                provenance="published_table",
            ),
            "kr_validation_same_scope_targets": {
                "passed": True,
                "model_role": "kr_validation_same_scope_target_report",
                "passed_scopes": [target_scope],
                "best_available_scope": {
                    "validation_scope": target_scope,
                    "missing_groups": [],
                    "partial_groups": [],
                },
            },
            "circuit_validation": {
                "passed": True,
                "validation_scope": target_scope,
            },
            "snowplow_validation": {
                "passed": True,
                "validation_scope": target_scope,
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "validation_scope": target_scope,
                "source": "KnowledgeReference/malir-2024-interferometry-dpf.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(target_scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "validation_scope": target_scope,
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "validation_scope": target_scope,
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "validation_scope": target_scope,
                "source": (
                    "KnowledgeReference/neutron-generation-dynamics-inside-a-"
                    "ma-class-dense-plasma-focus-z-pinch-5.md"
                ),
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence(target_scope)
            ),
            "neutron_uncertainty_validation": (
                _kr_neutron_uncertainty_evidence(target_scope)
            ),
            "field_coupling_validation": field_coupling_evidence_from_result({
                "field_coupling_component_validations": {
                    component: field_coupling_component_evidence(
                        component,
                        validation_scope=field_scope,
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
                },
            }),
            "physics_fidelity_evidence": physics_fidelity_evidence_from_result({
                "physics_effect_validations": {
                    effect: physics_effect_validation_evidence(
                        effect,
                        validation_scope=physics_scope,
                    )
                    for effect in (
                        "tabulated_eos_and_conductivity",
                        "ionization_and_charge_state_kinetics",
                        "two_temperature_energy_partition",
                        "radiation_transport_opacity",
                        "material_ablation_impurity_mixing",
                        "hall_flr_kinetic_or_pic_effects",
                        "three_dimensional_instabilities",
                        "flashover_sheath_initiation",
                        "restrike_and_anomalous_resistance",
                        "beam_generation_and_target_coupling",
                    )
                },
            }),
            "uncertainty_validation": uncertainty_evidence_from_result({
                "uncertainty_component_validations": {
                    component: uncertainty_component_evidence(
                        component,
                        validation_scope=uq_scope,
                        source_uncertainty_values={
                            f"{component}_relative_sigma": 0.05,
                        },
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
                },
            }),
        }

        gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(result)}

        assert result["field_coupling_validation"]["passed"] is True
        assert result["physics_fidelity_evidence"]["passed"] is True
        assert result["uncertainty_validation"]["passed"] is True
        assert gaps["same_scope_high_fidelity_claim"].status == "blocked"
        assert "do not share one validation_scope" in (
            gaps["same_scope_high_fidelity_claim"].blocker
        )

    def test_predictive_readiness_passes_only_with_all_required_tiers(self):
        r = self._good_result()
        scope = "synthetic_predictive_scope"
        r.update({
            "circuit_validation": {
                "passed": True,
                "metrics": ["peak_current", "peak_time", "waveform_shape"],
                "validation_scope": scope,
                "details": {"source_authority": {"passed": True}},
            },
            "snowplow_validation": {
                "passed": True,
                "phases": ["axial", "radial", "pinch"],
                "pinch_time_error": 0.1,
                "validation_scope": scope,
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": ["Sod", "Brio-Wu"],
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "validation_scope": scope,
                "source": "KnowledgeReference/spatial.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence(scope)
            ),
            "neutron_uncertainty_validation": _kr_neutron_uncertainty_evidence(scope),
        })
        readiness = predictive_readiness_report(r)
        assert readiness.ready is True
        assert readiness.status == "predictive_ready"
        assert readiness.missing_evidence == []
        assert "Spatial DPF experimental validation" in readiness.satisfied_evidence

    def test_predictive_readiness_requires_one_validation_scope(self):
        r = self._good_result()
        r.update({
            "circuit_validation": {
                "passed": True,
                "metrics": ["peak_current", "peak_time", "waveform_shape"],
                "validation_scope": "scope_circuit",
                "details": {"source_authority": {"passed": True}},
            },
            "snowplow_validation": {
                "passed": True,
                "phases": ["axial", "radial", "pinch"],
                "validation_scope": "scope_snowplow",
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": ["Sod", "Brio-Wu"],
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "validation_scope": "scope_spatial",
                "source": "KnowledgeReference/spatial.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence("scope_neutron"),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "validation_scope": "scope_neutron",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "validation_scope": "scope_neutron",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "validation_scope": "scope_neutron",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence("scope_neutron")
            ),
            "neutron_uncertainty_validation": (
                _kr_neutron_uncertainty_evidence("scope_neutron")
            ),
        })

        readiness = predictive_readiness_report(r)

        assert readiness.ready is False
        assert "Predictive validation scope alignment" in readiness.missing_evidence
        assert any("do not share one validation_scope" in b for b in readiness.blockers)

    def test_predictive_readiness_blocks_validation_pipeline_errors(self):
        r = self._good_result()
        r.update({
            "circuit_validation": {
                "passed": True,
                "metrics": ["peak_current", "peak_time", "waveform_shape"],
                "details": {"source_authority": {"passed": True}},
            },
            "snowplow_validation": {
                "passed": True,
                "phases": ["axial", "radial", "pinch"],
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": ["Sod", "Brio-Wu"],
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
            },
            "spatial_validation": {
                "passed": True,
                "validated_quantities": ["density", "magnetic_field", "temperature"],
                "source": "KnowledgeReference/spatial.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear", "beam_target"],
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": ["spectrum"],
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": ["anisotropy"],
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "validation_errors": [{
                "stage": "spatial_validation",
                "error_type": "RuntimeError",
                "message": "forced failure",
            }],
        })

        readiness = predictive_readiness_report(r)
        assert readiness.ready is False
        assert readiness.status == "validation_pipeline_error"
        assert "Validation pipeline health" in readiness.missing_evidence
        assert any("spatial_validation" in blocker for blocker in readiness.blockers)

    def test_validation_tier_report_distinguishes_supported_tiers(self):
        r = {
            "I_peak": 1.0,
            "n_steps": 20,
            "circuit_validation": {
                "passed": True,
                "metrics": {
                    "peak_current": True,
                    "peak_time": True,
                    "waveform_shape": True,
                },
                "details": {"source_authority": {"passed": True}},
            },
            "has_snowplow": True,
            "snowplow_validation": {
                "passed": True,
                "phases": {"axial": True, "radial": True, "pinch": True},
                "details": {"source_authority": {"passed": True}},
            },
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": {"sod": True, "brio_wu": True},
                "validation_tier": 3,
                "model_role": "code_verification_analytic_tests",
            },
            "spatial_validation": {
                "passed": True,
                "diagnostics": {
                    "density": True,
                    "magnetic_field": True,
                    "temperature": True,
                },
                "source": "KnowledgeReference/spatial.md",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_yield_validation": _kr_neutron_yield_evidence(),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "validated_mechanisms": {
                    "thermonuclear": True,
                    "beam_target": True,
                },
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": {"spectrum": True},
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": {"anisotropy": True},
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence()
            ),
            "neutron_uncertainty_validation": _kr_neutron_uncertainty_evidence(),
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[1].status == "supported"
        assert tiers[2].status == "supported"
        assert tiers[3].status == "supported"
        assert tiers[4].status == "supported"
        assert tiers[5].status == "supported"

    def test_predictive_readiness_rejects_placeholder_evidence(self):
        r = self._good_result()
        r.update({
            "circuit_validation": {"passed": True, "metrics": ["peak_current"]},
            "snowplow_validation": {"passed": True, "phases": ["axial"]},
            "has_mhd": True,
            "mhd_verification": {"passed": True},
            "spatial_validation": {"passed": True, "diagnostics": {"density": True}},
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": ["thermonuclear"],
            },
        })
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[1].status == "diagnostic_present"
        assert tiers[2].status == "partial"
        assert tiers[3].status == "verification_only"
        assert tiers[4].status == "not_validated"
        assert tiers[5].status == "decomposed_estimate"
        readiness = predictive_readiness_report(r)
        assert readiness.ready is False
        assert "Spatial DPF experimental validation" in readiness.missing_evidence

    def test_circuit_validation_evidence_requires_waveform_shape(self):
        validation = ValidationResult(
            device="PF-1000",
            metrics=[
                MetricResult("peak_current", 1.8e6, 1.87e6, 0.04, 0.10, True, "A"),
                MetricResult("peak_current_time", 5.7e-6, 5.8e-6, 0.02, 0.10, True, "s"),
            ],
            overall_score=0.97,
            passed=True,
        )
        evidence = circuit_validation_evidence_from_result(validation)
        assert evidence["passed"] is False
        assert evidence["metrics"]["peak_current"] is True
        assert evidence["metrics"]["peak_time"] is True
        assert "waveform_shape" not in evidence["metrics"]

    def test_circuit_validation_evidence_can_support_tier_one(self):
        validation = ValidationResult(
            device="PF-1000",
            metrics=[
                MetricResult("peak_current", 1.8e6, 1.87e6, 0.04, 0.10, True, "A"),
                MetricResult("peak_current_time", 5.7e-6, 5.8e-6, 0.02, 0.10, True, "s"),
            ],
            overall_score=0.97,
            passed=True,
        )
        evidence = circuit_validation_evidence_from_result(
            validation,
            waveform_nrmse=0.08,
            waveform_tolerance=0.10,
        )
        assert evidence["passed"] is True
        evidence["details"]["source_authority"] = {"passed": True}
        r = self._good_result()
        r["circuit_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[1].status == "supported"

    def test_circuit_validation_tier_requires_source_authority(self):
        r = self._good_result()
        r["circuit_validation"] = {
            "passed": True,
            "metrics": ["peak_current", "peak_time", "waveform_shape"],
        }

        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[1].status == "diagnostic_present"

    def test_circuit_waveform_evidence_can_support_tier_one(self):
        from dpf.validation.experimental import PF1000_DATA

        evidence = circuit_validation_evidence_from_waveform(
            PF1000_DATA.waveform_t,
            PF1000_DATA.waveform_I,
            "PF-1000",
        )

        assert evidence["passed"] is True
        assert evidence["metrics"]["peak_current"] is True
        assert evidence["metrics"]["peak_time"] is True
        assert evidence["metrics"]["waveform_shape"] is True
        r = self._good_result()
        r["circuit_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[1].status == "supported"

    def test_circuit_waveform_evidence_rejects_distorted_trace(self):
        from dpf.validation.experimental import PF1000_DATA

        evidence = circuit_validation_evidence_from_waveform(
            PF1000_DATA.waveform_t,
            0.5 * PF1000_DATA.waveform_I,
            "PF-1000",
        )

        assert evidence["passed"] is False
        assert evidence["metrics"]["peak_current"] is False

    def test_circuit_waveform_evidence_rejects_unverified_reconstructed_trace(self):
        from dpf.validation.experimental import PF1000_16KV_DATA

        evidence = circuit_validation_evidence_from_waveform(
            PF1000_16KV_DATA.waveform_t,
            PF1000_16KV_DATA.waveform_I,
            "PF-1000-16kV",
        )

        assert evidence["metrics"]["waveform_shape"] is True
        assert evidence["details"]["source_authority"]["kr_status"] == "unverified"
        assert evidence["details"]["source_authority"]["waveform_provenance"] == "reconstructed"
        assert evidence["details"]["source_authority"]["passed"] is False
        assert evidence["passed"] is False

    def test_circuit_waveform_evidence_rejects_external_archive_trace(self):
        from dpf.validation.experimental import POSEIDON_60KV_DATA

        evidence = circuit_validation_evidence_from_waveform(
            POSEIDON_60KV_DATA.waveform_t,
            POSEIDON_60KV_DATA.waveform_I,
            "POSEIDON-60kV",
        )

        assert evidence["metrics"]["waveform_shape"] is True
        authority = evidence["details"]["source_authority"]
        assert authority["kr_status"] == "verified"
        assert authority["waveform_provenance"] == "measured"
        assert authority["waveform_kr_status"] == "unverified"
        assert authority["passed"] is False
        assert evidence["passed"] is False

    def test_mhd_verification_evidence_requires_named_analytic_tests(self):
        evidence = mhd_verification_evidence_from_tests({"Sod": True})
        assert evidence["passed"] is False

        evidence = mhd_verification_evidence_from_tests({
            "Sod": True,
            "Brio-Wu": True,
        })
        assert evidence["passed"] is True
        r = {
            "I_peak": 1.0,
            "n_steps": 20,
            "has_mhd": True,
            "mhd_verification": evidence,
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[3].status == "supported"

    def test_mhd_verification_rejects_bare_named_tests_without_metadata(self):
        r = {
            "has_mhd": True,
            "mhd_verification": {
                "passed": True,
                "analytic_tests": {"sod": True, "brio_wu": True},
            },
        }

        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[3].status == "verification_only"

    def test_mhd_shock_tube_results_can_support_tier_three(self):
        sod_result = SimpleNamespace(
            errors={"rho": 0.03, "u": 0.04, "p": 0.03},
            checks={"no_nan": True, "rho_positive": True, "p_positive": True},
        )
        brio_result = SimpleNamespace(
            checks={
                "no_nan": True,
                "rho_positive": True,
                "p_positive": True,
                "Bx_preserved": True,
                "has_wave_structure": True,
                "By_sign_change": True,
            },
        )
        evidence = mhd_verification_evidence_from_shock_tube_results(
            sod_result,
            brio_result,
        )

        assert evidence["passed"] is True
        assert evidence["analytic_tests"]["sod"] is True
        assert evidence["analytic_tests"]["brio_wu"] is True
        r = {"has_mhd": True, "mhd_verification": evidence}
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[3].status == "supported"

    def test_mhd_shock_tube_results_reject_failed_brio_wu_check(self):
        sod_result = SimpleNamespace(
            errors={"rho": 0.03, "u": 0.04, "p": 0.03},
            checks={"no_nan": True, "rho_positive": True, "p_positive": True},
        )
        brio_result = SimpleNamespace(
            checks={
                "no_nan": True,
                "rho_positive": True,
                "p_positive": True,
                "Bx_preserved": True,
                "has_wave_structure": True,
                "By_sign_change": False,
            },
        )
        evidence = mhd_verification_evidence_from_shock_tube_results(
            sod_result,
            brio_result,
        )

        assert evidence["passed"] is False
        assert evidence["analytic_tests"]["brio_wu"] is False

    def test_snowplow_validation_evidence_requires_all_phases(self):
        evidence = snowplow_validation_evidence_from_phase_errors({
            "axial": 0.05,
            "radial": 0.08,
        })
        assert evidence["passed"] is False
        assert evidence["phases"]["pinch"] is False

        evidence = snowplow_validation_evidence_from_phase_errors({
            "axial": 0.05,
            "radial": 0.08,
            "pinch": 0.10,
        })
        assert evidence["passed"] is True
        r = self._good_result()
        r["snowplow_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[2].status == "partial"

    def test_snowplow_phase_history_can_support_tier_two_with_targets(self):
        times_s = [0.0, 1.0e-6, 2.0e-6, 3.0e-6]
        phases = ["rundown", "rundown", "radial", "pinch"]
        evidence = snowplow_validation_evidence_from_phase_history(
            times_s,
            phases,
            {"axial": 2.0e-6, "radial": 1.0e-6, "pinch": 3.0e-6},
            reference_source="KnowledgeReference/a-course-on-plasma-focus.md",
            reference_kr_status="verified",
        )

        assert evidence["passed"] is True
        assert evidence["phases"]["axial"] is True
        assert evidence["phases"]["radial"] is True
        assert evidence["phases"]["pinch"] is True
        r = self._good_result()
        r["snowplow_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[2].status == "supported"

    def test_snowplow_phase_history_rejects_unverified_targets(self):
        evidence = snowplow_validation_evidence_from_phase_history(
            [0.0, 1.0e-6, 2.0e-6],
            ["rundown", "radial", "pinch"],
            {"axial": 1.0e-6, "radial": 1.0e-6, "pinch": 2.0e-6},
        )

        assert evidence["phases"]["axial"] is True
        assert evidence["details"]["source_authority"]["passed"] is False
        assert evidence["passed"] is False

    def test_snowplow_phase_history_requires_pinch_observation(self):
        times_s = [0.0, 1.0e-6, 2.0e-6]
        phases = ["rundown", "radial", "radial"]
        evidence = snowplow_validation_evidence_from_phase_history(
            times_s,
            phases,
            {"axial": 1.0e-6, "radial": 1.0e-6, "pinch": 2.0e-6},
            reference_source="KnowledgeReference/a-course-on-plasma-focus.md",
            reference_kr_status="verified",
        )

        assert evidence["passed"] is False
        assert evidence["phases"]["pinch"] is False

    def test_snowplow_phase_observation_is_not_validation(self):
        evidence = snowplow_phase_observation_from_history(
            [0.0, 1.0e-6, 2.0e-6],
            ["rundown", "radial", "pinch"],
        )

        assert evidence["passed"] is False
        assert evidence["phases"]["axial"] is True
        assert evidence["phases"]["radial"] is True
        assert evidence["phases"]["pinch"] is True
        r = self._good_result()
        r["snowplow_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[2].status == "partial"

    def test_spatial_validation_evidence_requires_core_diagnostics(self):
        evidence = spatial_validation_evidence_from_quantity_errors({
            "density": 0.2,
            "B_field": 0.2,
        })
        assert evidence["passed"] is False
        assert evidence["diagnostics"]["magnetic_field"] is True
        assert evidence["diagnostics"]["temperature"] is False

        evidence = spatial_validation_evidence_from_quantity_errors({
            "rho": 0.2,
            "B_field": 0.2,
            "Te": 0.3,
        })
        assert evidence["passed"] is True
        r = self._good_result()
        r["spatial_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[4].status == "not_validated"

        evidence.update({
            "source": "KnowledgeReference/spatial.md",
            "validation_tier": 4,
            "model_role": "simulation_to_kr_target_comparison",
        })
        r["spatial_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[4].status == "supported"

    def test_neutron_timing_evidence_requires_both_mechanisms(self):
        evidence = neutron_timing_validation_evidence_from_errors({
            "thermonuclear": 0.2,
        })
        assert evidence["passed"] is False
        assert evidence["mechanisms"]["beam_target"] is False

        evidence = neutron_timing_validation_evidence_from_errors({
            "thermonuclear": 0.2,
            "beam-target": 0.3,
        })
        assert evidence["passed"] is True
        r = self._good_result()
        r["neutron_mechanism_timing_validation"] = evidence
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

        evidence.update({
            "source": "KnowledgeReference/neutron.md",
            "validation_tier": 5,
            "model_role": "simulation_to_kr_target_comparison",
        })
        r["neutron_mechanism_timing_validation"] = evidence

        r["neutron_spectrum_validation"] = {
            "passed": True,
            "validated_features": {"spectrum": True},
            "source": "KnowledgeReference/neutron.md",
            "validation_tier": 5,
            "model_role": "simulation_to_kr_target_comparison",
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

        r["neutron_anisotropy_validation"] = {
            "passed": True,
            "validated_features": {"anisotropy": True},
            "source": "KnowledgeReference/neutron.md",
            "validation_tier": 5,
            "model_role": "simulation_to_kr_target_comparison",
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

        r["neutron_yield_validation"] = _kr_neutron_yield_evidence()
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

        r["neutron_detector_response_validation"] = (
            _kr_neutron_detector_response_evidence()
        )
        r["neutron_uncertainty_validation"] = _kr_neutron_uncertainty_evidence()
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "supported"

    def test_neutron_validation_requires_same_scope_closure(self):
        r = {
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": {"thermonuclear": True, "beam_target": True},
                "validation_scope": "mjolnir_timing",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": {"spectrum": True},
                "validation_scope": "mjolnir_spectrum",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": {"anisotropy": True},
                "validation_scope": "mjolnir_anisotropy",
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
        }

        closure = neutron_validation_scope_closure_report(r)
        assert closure["passed"] is False
        assert closure["closed_scopes"] == []
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

    def test_neutron_scope_closure_report_identifies_closed_scope(self):
        scope = "mjolnir_neutron_timing_2025_goyon"
        r = {
            "neutron_yield_validation": _kr_neutron_yield_evidence(scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": {"thermonuclear": True, "beam_target": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": {"spectrum": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": {"anisotropy": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_detector_response_validation": (
                _kr_neutron_detector_response_evidence(scope)
            ),
            "neutron_uncertainty_validation": _kr_neutron_uncertainty_evidence(scope),
        }

        closure = neutron_validation_scope_closure_report(r)
        assert closure["passed"] is True
        assert closure["closed_scopes"] == [scope]
        assert closure["scopes"][0]["missing_features"] == []

    def test_neutron_scope_closure_requires_detector_response_and_uncertainty(self):
        scope = "mjolnir_neutron_timing_2025_goyon"
        r = {
            "neutron_yield_validation": _kr_neutron_yield_evidence(scope),
            "neutron_mechanism_timing_validation": {
                "passed": True,
                "mechanisms": {"thermonuclear": True, "beam_target": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_spectrum_validation": {
                "passed": True,
                "validated_features": {"spectrum": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
            "neutron_anisotropy_validation": {
                "passed": True,
                "validated_features": {"anisotropy": True},
                "validation_scope": scope,
                "source": "KnowledgeReference/neutron.md",
                "validation_tier": 5,
                "model_role": "simulation_to_kr_target_comparison",
            },
        }

        closure = neutron_validation_scope_closure_report(r)
        assert closure["passed"] is False
        assert closure["closed_scopes"] == []
        assert set(closure["scopes"][0]["missing_features"]) == {
            "detector_response",
            "uncertainty",
        }
        tiers = {t.level: t for t in validation_tier_report(r)}
        assert tiers[5].status == "decomposed_estimate"

        r["neutron_detector_response_validation"] = (
            _kr_neutron_detector_response_evidence(scope)
        )
        bad_uncertainty = _kr_neutron_uncertainty_evidence(scope)
        bad_uncertainty.pop("source_uncertainty_values")
        r["neutron_uncertainty_validation"] = bad_uncertainty
        closure = neutron_validation_scope_closure_report(r)
        assert closure["passed"] is False
        assert closure["scopes"][0]["missing_features"] == ["uncertainty"]
        uncertainty_component = [
            component for component in closure["scopes"][0]["components"]
            if component["feature"] == "uncertainty"
        ][0]
        assert uncertainty_component["source_uncertainty_values_passed"] is False

        r["neutron_uncertainty_validation"] = _kr_neutron_uncertainty_evidence(scope)
        assert neutron_validation_scope_closure_report(r)["passed"] is True

    def test_combined_spatial_evidence_requires_consistent_scope(self):
        density = {
            "passed": True,
            "diagnostics": {"density": True},
            "validation_scope": "pf1000_density",
            "validation_tier": 4,
            "model_role": "simulation_to_kr_target_comparison",
            "source": "KnowledgeReference/density.md",
        }
        magnetic = {
            "passed": True,
            "diagnostics": {"magnetic_field": True},
            "validation_scope": "llnl_em",
            "validation_tier": 4,
            "model_role": "simulation_to_kr_target_comparison",
            "source": "KnowledgeReference/magnetic.md",
        }
        temperature = {
            "passed": True,
            "diagnostics": {"temperature": True},
            "validation_scope": "generic_temperature",
            "validation_tier": 4,
            "model_role": "simulation_to_kr_target_comparison",
            "source": "KnowledgeReference/temperature.md",
        }

        evidence = combine_spatial_validation_evidence([
            density,
            magnetic,
            temperature,
        ])

        assert evidence["passed"] is False
        assert evidence["scope_consistent"] is False
        assert all(evidence["diagnostics"].values())
        tiers = {t.level: t for t in validation_tier_report({
            "spatial_validation": evidence,
        })}
        assert tiers[4].status == "not_validated"

    def test_spatial_scope_closure_report_groups_missing_quantities(self):
        report = spatial_validation_scope_closure_report([
            {
                "passed": True,
                "diagnostics": {"density": True},
                "validation_scope": "pf1000_interferometry_density_2024_malir",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/malir-2024-interferometry-dpf.md",
            },
            {
                "passed": True,
                "diagnostics": {"magnetic_field": True},
                "validation_scope": "llnl_em_fluctuation_2014_schmidt",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/llnl.md",
            },
            {
                "passed": True,
                "diagnostics": {"temperature": True},
                "validation_scope": "dpf_pinch_temperature_review_regime",
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/review.md",
            },
        ])

        assert report["passed"] is False
        assert report["closed_scopes"] == []
        scopes = {scope["validation_scope"]: scope for scope in report["scopes"]}
        pf1000 = scopes["pf1000_interferometry_density_2024_malir"]
        assert pf1000["covered_quantities"]["density"] is True
        assert pf1000["missing_quantities"] == ["magnetic_field", "temperature"]

    def test_combined_spatial_evidence_supports_tier_four_when_scope_matches(self):
        scope = "same_shot_spatial_diagnostics"
        evidence = combine_spatial_validation_evidence([
            {
                "passed": True,
                "diagnostics": {"density": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/density.md",
            },
            {
                "passed": True,
                "diagnostics": {"magnetic_field": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/magnetic.md",
            },
            {
                "passed": True,
                "diagnostics": {"temperature": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/temperature.md",
            },
        ])

        assert evidence["passed"] is True
        assert evidence["scope_consistent"] is True
        tiers = {t.level: t for t in validation_tier_report({
            "spatial_validation": evidence,
        })}
        assert tiers[4].status == "supported"

    def test_spatial_scope_closure_report_identifies_closed_scope(self):
        scope = "same_shot_spatial_diagnostics"
        report = spatial_validation_scope_closure_report([
            {
                "passed": True,
                "diagnostics": {"density": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/density.md",
            },
            {
                "passed": True,
                "diagnostics": {"magnetic_field": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/magnetic.md",
            },
            {
                "passed": True,
                "diagnostics": {"temperature": True},
                "validation_scope": scope,
                "validation_tier": 4,
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/temperature.md",
            },
        ])

        assert report["passed"] is True
        assert report["closed_scopes"] == [scope]
        assert report["scopes"][0]["missing_quantities"] == []

    def test_combined_spatial_evidence_rejects_unsourced_components(self):
        scope = "same_shot_spatial_diagnostics"
        evidence = combine_spatial_validation_evidence([
            {
                "passed": True,
                "diagnostics": {"density": True},
                "validation_scope": scope,
            },
            {
                "passed": True,
                "diagnostics": {"magnetic_field": True},
                "validation_scope": scope,
            },
            {
                "passed": True,
                "diagnostics": {"temperature": True},
                "validation_scope": scope,
            },
        ])

        assert evidence["scope_consistent"] is True
        assert evidence["passed"] is False
        assert not any(evidence["diagnostics"].values())

    def test_combined_spatial_evidence_handles_malformed_tier_metadata(self):
        scope = "same_shot_spatial_diagnostics"
        evidence = combine_spatial_validation_evidence([
            {
                "passed": True,
                "diagnostics": {"density": True},
                "validation_scope": scope,
                "validation_tier": "tier-four",
                "model_role": "simulation_to_kr_target_comparison",
                "source": "KnowledgeReference/density.md",
            },
        ])

        assert evidence["passed"] is False
        assert evidence["diagnostics"]["density"] is False
