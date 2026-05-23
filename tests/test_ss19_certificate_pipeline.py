"""SS19 certificate pipeline boundary tests.

These tests are deliberately source/claim conservative: production packets must
refuse unless the complete stack is present, and the only positive path is a
synthetic fixture that proves wiring without promoting first-principles runtime
acceptance.
"""

from __future__ import annotations

from typing import Any

from dpf.first_principles.certificate_gate import build_ss19_certificate_pipeline

PF1000_SCOPE = "pf1000_akel_16kv_1p2torr_shot_12581"
PF1000_DEVICE = "PF-1000/Akel"


def _complete_pipeline_inputs() -> dict[str, Any]:
    return {
        "declared_scope": PF1000_SCOPE,
        "device_name": PF1000_DEVICE,
        "run_manifest_hash": "sha256:" + "1" * 64,
        "source_packet_hashes": {
            "ss14": "sha256:" + "2" * 64,
            "ss16": "sha256:" + "3" * 64,
            "ss17": "sha256:" + "4" * 64,
            "ss18": "sha256:" + "5" * 64,
        },
        "comparator_mapping": {
            "current_waveform": {
                "output_path": "outputs/current_A",
                "source_target_id": "akel_fig1_current",
                "metric": "rmse_with_uq",
                "unit_conversion": "A_to_kA",
                "time_alignment": "absolute_time_no_phase_shift",
                "tolerance_id": "synthetic_current_tolerance",
            },
            "neutron_scalar_yield": {
                "output_path": "outputs/neutron_yield",
                "source_target_id": "akel_scalar_yield",
                "metric": "interval_overlap",
                "unit_conversion": "count_to_count",
                "time_alignment": "not_time_series",
                "tolerance_id": "synthetic_yield_tolerance",
            },
        },
        "uncertainty_budget": {
            "measurement_uncertainty": {"relative_sigma": 0.05},
            "model_uncertainty": {"relative_sigma": 0.10},
            "numerical_uncertainty": {"relative_sigma": 0.02},
            "propagation_method": "synthetic_linearized_covariance",
            "observable_uncertainties": {
                "current_waveform": {"relative_sigma": 0.06},
                "neutron_scalar_yield": {"relative_sigma": 0.12},
            },
        },
        "upstream_packets": {
            "same_scope_source": {"status": "accepted_synthetic_fixture"},
            "waveform_phase": {"status": "accepted_synthetic_fixture"},
            "spatial_field_temperature": {"status": "accepted_synthetic_fixture"},
            "neutron_authority": {"status": "accepted_synthetic_fixture"},
            "comparator_uq": {"status": "accepted_synthetic_fixture"},
            "numerical_fidelity": {"status": "accepted_synthetic_fixture"},
            "physics_closure": {"status": "accepted_synthetic_fixture"},
            "limiter_readiness": {"status": "accepted_synthetic_fixture"},
            "power_port": {"status": "accepted_synthetic_fixture"},
            "startup_bvp": {"status": "accepted_synthetic_fixture"},
            "dimensionality_handoff": {"status": "accepted_synthetic_fixture"},
        },
        "negative_controls": {
            "draft_evidence": True,
            "blocked_evidence": True,
            "cross_scope_evidence": True,
            "missing_uq": True,
            "missing_review": True,
            "hidden_limiter": True,
            "app_only_or_reduced_model_fallback": True,
        },
        "review_certificate": {
            "status": "accepted_synthetic_fixture_review",
            "reviewer": "synthetic-reviewer",
            "reviewed_artifact_hash": "sha256:" + "6" * 64,
        },
    }


def test_ss19_refuses_incomplete_production_stack_with_explicit_reasons() -> None:
    packet = build_ss19_certificate_pipeline(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets={
            "same_scope_source": {"status": "blocked_same_scope_source_packet_not_available"},
        },
        comparator_mapping={
            "current_waveform": {
                "output_path": "outputs/current_A",
                "source_target_id": "akel_fig1_current",
            },
        },
        uncertainty_budget={"measurement_uncertainty": {"relative_sigma": 0.05}},
        negative_controls={"draft_evidence": True},
    )

    assert packet["status"] == "refused_incomplete_certificate_stack"
    assert packet["certificate_kind"] == "production"
    assert packet["can_emit_certificate"] is False
    assert packet["accepted_runtime_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["promotes_acceptance"] is False
    assert "missing_run_manifest_hash" in packet["refusal_reasons"]
    assert "incomplete_source_packet_hashes" in packet["refusal_reasons"]
    assert "incomplete_comparator_mapping" in packet["refusal_reasons"]
    assert "incomplete_uncertainty_budget" in packet["refusal_reasons"]
    assert "incomplete_negative_controls" in packet["refusal_reasons"]
    assert "blocked_upstream_packets" in packet["refusal_reasons"]


def test_ss19_refuses_complete_production_stack_without_synthetic_fixture_flag() -> None:
    packet = build_ss19_certificate_pipeline(**_complete_pipeline_inputs())

    assert packet["status"] == "refused_production_acceptance_disabled"
    assert packet["certificate_kind"] == "production"
    assert packet["stack_complete"] is True
    assert packet["can_emit_certificate"] is False
    assert packet["accepted_runtime_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["promotes_acceptance"] is False
    assert packet["refusal_reasons"] == ["production_acceptance_requires_real_review_gate"]


def test_ss19_accepts_only_synthetic_complete_fixture_without_runtime_promotion() -> None:
    inputs = _complete_pipeline_inputs()
    packet = build_ss19_certificate_pipeline(
        **inputs,
        synthetic_complete_fixture=True,
    )

    assert packet["status"] == "accepted_synthetic_complete_fixture"
    assert packet["certificate_kind"] == "synthetic_fixture"
    assert packet["stack_complete"] is True
    assert packet["can_emit_certificate"] is True
    assert packet["accepted_runtime_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["promotes_acceptance"] is False
    assert packet["certificate_artifact_hash"].startswith("sha256:")
    assert packet["refusal_reasons"] == []


def test_ss19_synthetic_fixture_still_refuses_missing_negative_control() -> None:
    inputs = _complete_pipeline_inputs()
    negative_controls = dict(inputs["negative_controls"])
    negative_controls["hidden_limiter"] = False
    inputs["negative_controls"] = negative_controls

    packet = build_ss19_certificate_pipeline(
        **inputs,
        synthetic_complete_fixture=True,
    )

    assert packet["status"] == "refused_incomplete_certificate_stack"
    assert packet["can_emit_certificate"] is False
    assert packet["stack_complete"] is False
    assert "incomplete_negative_controls" in packet["refusal_reasons"]
    assert packet["negative_control_matrix"]["hidden_limiter"]["passed"] is False
