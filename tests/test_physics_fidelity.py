"""Tests for high-fidelity physics blocker audits."""

from __future__ import annotations

from dpf.validation import (
    high_fidelity_readiness_report,
    physics_effect_validation_evidence,
    physics_fidelity_evidence_from_result,
    scientific_accuracy_gap_report,
)


def test_physics_fidelity_audit_blocks_reduced_default_result():
    evidence = physics_fidelity_evidence_from_result({
        "I_peak": 1.0,
        "n_steps": 20,
        "has_mhd": True,
    })

    assert evidence["passed"] is False
    assert evidence["source"].startswith("KnowledgeReference/")
    assert "tabulated_eos_and_conductivity" in evidence["required_effects"]
    assert "hall_flr_kinetic_or_pic_effects" in evidence["missing_or_unvalidated_effects"]
    assert evidence["engineering_run_blocked"] is False
    assert evidence["required_effects"]["tabulated_eos_and_conductivity"][
        "fidelity_status"
    ] == "absent"


def test_physics_fidelity_audit_marks_active_modules_unvalidated():
    evidence = physics_fidelity_evidence_from_result(
        {
            "plasma_regime": {"kinetic_needed": True},
            "radiation_regime": {"optically_thin": False},
            "beam_tracker": {"n_particles": 128},
            "breakdown": {"mechanism": "Paschen"},
            "post_pinch_empirical_resistance": True,
        },
        active_modules=[
            "FLD radiation transport",
            "Electrode ablation (Cu)",
            "CR ionization (non-LTE)",
            "Sheath BC (Bohm)",
            "Nernst B-advection",
        ],
    )

    effects = evidence["required_effects"]
    assert effects["ionization_and_charge_state_kinetics"]["status"] == (
        "implemented_not_validated"
    )
    assert effects["radiation_transport_opacity"]["status"] == (
        "implemented_not_validated"
    )
    assert effects["material_ablation_impurity_mixing"]["status"] == (
        "empirical_not_validated"
    )
    assert effects["material_ablation_impurity_mixing"]["fidelity_status"] == (
        "empirical"
    )
    assert effects["hall_flr_kinetic_or_pic_effects"]["status"] == (
        "required_unvalidated"
    )
    assert evidence["passed"] is False


def test_physics_effect_validation_evidence_supports_one_effect():
    effect = physics_effect_validation_evidence(
        "tabulated_eos_and_conductivity",
        validation_scope="unit_test_scope",
        notes="table range covers the claimed state",
    )
    evidence = physics_fidelity_evidence_from_result({
        "physics_effect_validation": {
            "tabulated_eos_and_conductivity": effect,
        },
    })
    effects = evidence["required_effects"]

    assert effect["passed"] is True
    assert evidence["passed"] is False
    assert effects["tabulated_eos_and_conductivity"]["status"] == "validated"
    assert effects["tabulated_eos_and_conductivity"]["fidelity_status"] == "validated"
    assert effects["tabulated_eos_and_conductivity"]["verified"] is True
    assert effects["tabulated_eos_and_conductivity"]["validated"] is True
    assert "physics_effect_validation" in (
        effects["tabulated_eos_and_conductivity"]["evidence_keys"]
    )
    assert "ionization_and_charge_state_kinetics" in (
        evidence["missing_or_unvalidated_effects"]
    )


def test_physics_effect_validation_evidence_can_bound_effect_out_of_scope():
    effect = physics_effect_validation_evidence(
        "beam_generation_and_target_coupling",
        validation_scope="pre_disruption_current_voltage_only",
        implemented=False,
        bounded_out=True,
        notes="scope excludes neutron and beam observables",
    )
    evidence = physics_fidelity_evidence_from_result({
        "beam_generation_and_target_coupling_validation": effect,
    })
    beam = evidence["required_effects"]["beam_generation_and_target_coupling"]

    assert effect["passed"] is True
    assert beam["status"] == "bounded_out"
    assert beam["fidelity_status"] == "bounded_out"
    assert beam["implemented"] is False
    assert beam["validated"] is True


def test_unverified_physics_effect_evidence_does_not_pass():
    effect = physics_effect_validation_evidence(
        "tabulated_eos_and_conductivity",
        validation_scope="unit_test_scope",
        verified=False,
        notes="implementation exists, but numerical/source verification is absent",
    )
    evidence = physics_fidelity_evidence_from_result({
        "physics_effect_validation": {
            "tabulated_eos_and_conductivity": effect,
        },
    })

    assert effect["passed"] is False
    assert evidence["required_effects"]["tabulated_eos_and_conductivity"][
        "fidelity_status"
    ] == "absent"
    assert "tabulated_eos_and_conductivity" in (
        evidence["missing_or_unvalidated_effects"]
    )


def test_physics_fidelity_claim_blockers_are_scope_specific():
    beam = physics_effect_validation_evidence(
        "beam_generation_and_target_coupling",
        validation_scope="current_waveform_only_scope",
        implemented=False,
        bounded_out=True,
        notes="scope excludes neutron and p-B11 observables",
    )
    evidence = physics_fidelity_evidence_from_result({
        "beam_generation_and_target_coupling_validation": beam,
    })
    claim_blockers = evidence["claim_blockers"]

    assert "beam_generation_and_target_coupling" not in claim_blockers[
        "circuit_waveform_prediction"
    ]["blocking_effects"]
    assert "beam_generation_and_target_coupling" not in claim_blockers[
        "neutron_prediction"
    ]["blocking_effects"]
    assert "hall_flr_kinetic_or_pic_effects" in claim_blockers[
        "neutron_prediction"
    ]["blocking_effects"]
    assert "p_b11_prediction" in evidence["blocked_claims"]
    assert evidence["engineering_run_blocked"] is False


def test_complete_physics_effect_packet_can_pass_gap_gate():
    validations = {
        effect: physics_effect_validation_evidence(
            effect,
            validation_scope="synthetic_complete_high_fidelity_packet",
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
    }
    evidence = physics_fidelity_evidence_from_result({
        "physics_effect_validations": validations,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({
            "physics_fidelity_evidence": evidence,
        })
    }

    assert evidence["passed"] is True
    assert evidence["same_scope_passed"] is True
    assert evidence["missing_or_unvalidated_effects"] == []
    assert evidence["blocked_claims"] == []
    assert all(
        blocker["blocked"] is False
        for blocker in evidence["claim_blockers"].values()
    )
    assert gaps["missing_physics_fidelity"].status == "supported"


def test_complete_physics_effects_must_share_validation_scope():
    effects = (
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
    validations = {
        effect: physics_effect_validation_evidence(
            effect,
            validation_scope=f"synthetic_scope_{idx % 2}",
        )
        for idx, effect in enumerate(effects)
    }
    evidence = physics_fidelity_evidence_from_result({
        "physics_effect_validations": validations,
    })
    gaps = {
        gap.area: gap
        for gap in scientific_accuracy_gap_report({
            "physics_fidelity_evidence": evidence,
        })
    }

    assert evidence["passed"] is False
    assert evidence["same_scope_passed"] is False
    assert "same_scope_physics_packet" in (
        evidence["missing_or_unvalidated_effects"]
    )
    assert gaps["missing_physics_fidelity"].status == "blocked"


def test_physics_fidelity_audit_feeds_gap_and_high_fidelity_gate():
    result = {
        "physics_fidelity_evidence": physics_fidelity_evidence_from_result({}),
    }
    gaps = {gap.area: gap for gap in scientific_accuracy_gap_report(result)}

    assert gaps["missing_physics_fidelity"].status == "blocked"

    readiness = high_fidelity_readiness_report(result)
    assert readiness.ready is False
    assert "missing_physics_fidelity" in readiness.remaining_areas
