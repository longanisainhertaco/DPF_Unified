"""WP-7 negative controls: the first-principles certificate and its WP-7
upstream packets must fail closed for draft, blocked, cross-scope,
missing-UQ, missing-review, hidden-limiter, reduced-model-fallback, and
app-only evidence.

These tests assert ONLY existing fail-closed contract fields.  They invent
no tolerance and no acceptance threshold.  If any test starts failing, a
packet has gained an accepting path and must be re-audited against
SSR-010/011/012.
"""

from __future__ import annotations

import pytest

from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.comparator_uq import build_comparator_uq_packet
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.manifest import FirstPrinciplesRunManifest
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.first_principles.waveform_phase import build_waveform_phase_packet

PF1000_SCOPE = "pf1000_akel_16kv_1p2torr_shot_12581"
PF1000_DEVICE = "PF-1000/Akel"


def _accepted_upstream() -> dict[str, dict[str, str]]:
    """All-accepted upstream set used to prove the gate STILL fails closed
    even if every dependency were (hypothetically) accepted, because the
    certificate's own required channels remain unfilled."""
    names = (
        "startup_bvp",
        "limiter_readiness",
        "power_port",
        "dimensionality_handoff",
        "physics_closure",
        "same_scope_source",
        "waveform_phase",
        "spatial_field_temperature",
        "neutron_authority",
        "comparator_uq",
        "numerical_fidelity",
    )
    return {name: {"status": "accepted_engineering_review"} for name in names}


# ---------------------------------------------------------------------------
# NC-1  blocked upstream blocks the certificate
# ---------------------------------------------------------------------------

def test_certificate_blocked_when_any_upstream_blocked() -> None:
    """blocked control: one blocked upstream packet must block the certificate
    and must appear as a certificate blocker."""
    upstream = _accepted_upstream()
    upstream["power_port"] = {"status": "blocked_power_port_not_available"}
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    assert packet["status"] == "blocked_first_principles_certificate_not_available"
    assert packet["can_write_accepted_certificate"] is False
    assert packet["can_release_first_principles_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "power_port" in packet["upstream_certificate_blockers"]
    matrix = packet["upstream_packet_acceptance_matrix"]
    assert matrix["power_port_packet_accepted"]["accepted_for_certificate"] is False


# ---------------------------------------------------------------------------
# NC-2  draft / empty upstream blocks the certificate
# ---------------------------------------------------------------------------

def test_certificate_blocked_with_no_upstream_packets() -> None:
    """draft/empty control: with no upstream packets the certificate must not
    release and the release label must remain engineering-candidate."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    assert packet["release_decision"] == "do_not_release_first_principles_claim"
    assert packet["release_label"].startswith("engineering_candidate")
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# NC-3  structural: all-accepted upstream still cannot flip the certificate
# ---------------------------------------------------------------------------

def test_certificate_cannot_accept_even_with_all_upstream_accepted() -> None:
    """Structural control: even an all-accepted upstream set cannot flip the
    certificate, because the required certificate channels are unfilled."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    assert packet["status"] == "blocked_first_principles_certificate_not_available"
    assert packet["can_write_accepted_certificate"] is False
    assert packet["missing_acceptance_channels"], "channels must remain missing"


# ---------------------------------------------------------------------------
# NC-4  missing negative-test channels stay missing
# ---------------------------------------------------------------------------

def test_certificate_negative_test_channels_required_and_missing() -> None:
    """missing-negative-test control: all 7 required negative-test channels
    must report missing when the deck declares none."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    matrix = packet["negative_test_matrix"]
    for channel in (
        "negative_test_draft_evidence",
        "negative_test_blocked_evidence",
        "negative_test_cross_scope_evidence",
        "negative_test_missing_uq",
        "negative_test_missing_review",
        "negative_test_hidden_limiter",
        "negative_test_app_only_or_reduced_model_fallback",
    ):
        assert matrix[channel]["present"] is False
        assert matrix[channel]["decision"] == "missing_required_negative_test"


# ---------------------------------------------------------------------------
# NC-5  cross-scope evidence rejected by same-scope packet
# ---------------------------------------------------------------------------

def test_same_scope_rejects_cross_scope_target() -> None:
    """cross-scope control: an accepted-status target from a different scope
    must be rejected with mismatched-scope metadata and must not add a
    same-scope channel."""
    cross = {
        "name": "pf1000_full_energy_anisotropy",
        "observable": "neutron_anisotropy",
        "status": "accepted_same_scope_source",
        "declared_scope": "pf1000_full_energy_450kj_3p5torr",
    }
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(cross,),
    )
    decisions = packet["validation_target_scope_decisions"]
    assert any(
        d["decision"] == "rejected_missing_or_mismatched_scope_metadata"
        for d in decisions
    )
    assert "neutron_anisotropy" not in packet["accepted_same_scope_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# NC-6  missing UQ channels block comparator-UQ
# ---------------------------------------------------------------------------

def test_comparator_uq_blocked_when_missing_uq_channels() -> None:
    """missing-UQ control: with no accepted UQ channels every UQ channel must
    be missing and comparator acceptance must be blocked."""
    packet = build_comparator_uq_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    missing = set(packet["missing_acceptance_channels"])
    for channel in (
        "measurement_uncertainty_by_observable",
        "model_uncertainty_by_observable",
        "numerical_uncertainty_by_observable",
        "uq_propagation_method",
        "independent_review_certificate",
    ):
        assert channel in missing
    assert packet["can_support_comparator_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# NC-7  missing independent review blocks waveform-phase
# ---------------------------------------------------------------------------

def test_waveform_phase_blocked_when_missing_review() -> None:
    """missing-review control: the draft Akel Fig.1 packet must stay
    review_status=draft and the waveform-phase packet must not accept."""
    packet = build_waveform_phase_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    draft = packet["draft_digitization_packet_status"]
    assert draft["review_status"] == "draft"
    assert draft["independent_review_count"] == 0
    assert draft["accepted_for_validation"] is False
    assert "independent_review_accepted" in packet["missing_acceptance_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# NC-8  hidden-limiter: limiter zero probe with blockers does not promote
# ---------------------------------------------------------------------------

def test_numerical_fidelity_hidden_limiter_does_not_pass() -> None:
    """hidden-limiter control: a runtime limiter-zero probe that observed
    blockers must NOT promote the numerical limiter-zero surface to accepted."""
    upstream = {
        "experimental_limiter_zero_probe": {
            "status": "experimental_limiter_zero_probe_not_validation",
            "zero_acceptance_blockers_observed": True,
        },
    }
    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    surface = packet["test_surface_status"]["limiter_zero_acceptance"]
    assert surface["can_support_numerical_acceptance"] is False
    # Status must not be a bare "accepted" — it may contain "not_acceptance"
    status = surface["status"]
    assert "accepted" not in status or "not_acceptance" in status
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# NC-9  reduced-model-fallback / app-only: generalization stays blocked
# ---------------------------------------------------------------------------

def test_generalization_rejects_reduced_model_and_app_only_scope() -> None:
    """reduced-model-fallback / app-only control: a general-DPF claim must be
    blocked, and candidate second scopes must stay candidate-only."""
    packet = build_generalized_dpf_machine_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets={
            "certificate_gate": {
                "status": "blocked_first_principles_certificate_not_available",
            },
        },
    )
    assert packet["can_claim_generalized_dpf_machine"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    for decision in packet["candidate_second_scope_decisions"]:
        assert decision["decision"] == "candidate_requirement_material_not_acceptance"
        assert decision["must_write_independent_certificate"] is True


# ---------------------------------------------------------------------------
# NC-10  app-only / reduced-model certificate channels missing
# ---------------------------------------------------------------------------

def test_certificate_app_only_evidence_channel_missing() -> None:
    """app-only control: the certificate's package-native execution proof
    channel and reduced-model rejection proof must be missing when no
    package-native proof is declared."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    missing = packet["missing_acceptance_channels"]
    assert "package_native_execution_proof" in missing
    assert "reduced_model_rejection_proof" in missing


# ---------------------------------------------------------------------------
# NC-11  manifest raises ValueError on any non-candidate status
# ---------------------------------------------------------------------------

def test_manifest_rejects_accepted_run_status() -> None:
    """manifest structural control: FirstPrinciplesRunManifest.__post_init__
    must raise ValueError if run_status is not the engineering-candidate
    constant, enforcing that a manifest cannot be built in an accepted state."""
    with pytest.raises(ValueError, match="engineering_candidate"):
        FirstPrinciplesRunManifest(run_status="accepted_first_principles")


def test_manifest_rejects_accepted_validation_status() -> None:
    """manifest structural control: validation_status must remain not_validation."""
    with pytest.raises(ValueError, match="not_validation"):
        FirstPrinciplesRunManifest(validation_status="accepted_validation")


def test_manifest_rejects_true_acceptance_flag() -> None:
    """manifest structural control: can_support_first_principles_acceptance=True
    must be rejected at construction time."""
    with pytest.raises(ValueError, match="candidate"):
        FirstPrinciplesRunManifest(can_support_first_principles_acceptance=True)
