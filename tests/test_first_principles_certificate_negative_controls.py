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


# ---------------------------------------------------------------------------
# S3.8 / WP-N7 negative controls
# (WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md §6.1 N7-NEG-01 through N7-NEG-10)
# ---------------------------------------------------------------------------


def test_n7_neg01_blocked_same_scope_packet_blocks_certificate() -> None:
    """N7-NEG-01: certificate with a blocked same-scope packet must stay blocked.

    The gate must list ``same_scope_source`` in upstream_certificate_blockers
    and ``can_write_accepted_certificate`` must remain False.
    WP-N7 §6.1, N7-NEG-01.
    """
    upstream = _accepted_upstream()
    upstream["same_scope_source"] = {
        "status": "blocked_same_scope_source_packet_not_available",
    }
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    assert packet["status"] == "blocked_first_principles_certificate_not_available"
    assert packet["can_write_accepted_certificate"] is False
    assert "same_scope_source" in packet["upstream_certificate_blockers"]
    matrix = packet["upstream_packet_acceptance_matrix"]
    assert matrix["same_scope_source_packet_accepted"]["accepted_for_certificate"] is False


def test_n7_neg02_candidate_comparator_uq_blocks_certificate() -> None:
    """N7-NEG-02: a candidate comparator-UQ packet must block the certificate.

    ``comparator_uq_packet_accepted`` must appear in missing_acceptance_channels
    and ``can_write_accepted_certificate`` must be False.
    WP-N7 §6.1, N7-NEG-02.
    """
    upstream = _accepted_upstream()
    upstream["comparator_uq"] = {
        "status": "blocked_comparator_uq_matrix_not_available",
    }
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    assert packet["can_write_accepted_certificate"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    missing = set(packet["missing_acceptance_channels"])
    assert "comparator_uq_packet_accepted" in missing


def test_n7_neg03_missing_reviewer_metadata_blocks_certificate() -> None:
    """N7-NEG-03: no reviewer_metadata channel means the certificate is blocked.

    reviewer_metadata must appear in missing_acceptance_channels.
    WP-N7 §6.1, N7-NEG-03.
    """
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    assert packet["can_write_accepted_certificate"] is False
    missing = set(packet["missing_acceptance_channels"])
    assert "reviewer_metadata" in missing


def test_n7_neg04_cross_scope_evidence_blocks_certificate() -> None:
    """N7-NEG-04: cross-scope evidence without a transfer rule must block.

    validation_scope_and_source_scope must be in missing_acceptance_channels.
    WP-N7 §6.1, N7-NEG-04.
    """
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    missing = set(packet["missing_acceptance_channels"])
    assert "validation_scope_and_source_scope" in missing
    assert packet["can_write_accepted_certificate"] is False


def test_n7_neg05_missing_run_manifest_hash_blocks_certificate() -> None:
    """N7-NEG-05: run_manifest_hash must be in missing_acceptance_channels.

    WP-N7 §6.1, N7-NEG-05.
    """
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )
    missing = set(packet["missing_acceptance_channels"])
    assert "run_manifest_hash" in missing


def test_n7_neg06_can_support_first_principles_acceptance_always_false() -> None:
    """N7-NEG-06: can_support_first_principles_acceptance must always be False.

    This invariant is enforced by manifest.__post_init__ and the certificate
    gate independently.  With all upstream packets declared as accepted the
    certificate gate must still refuse to set this flag.
    WP-N7 §6.1, N7-NEG-06.
    """
    # Even if ALL channels were somehow accepted, the gate returns False.
    all_channels = list(build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )["required_channels"])
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        accepted_channels=all_channels,
        upstream_packets=_accepted_upstream(),
    )
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_write_accepted_certificate"] is False


def test_n7_neg07_text_only_scalar_yield_cannot_be_accepted_comparator_target() -> None:
    """N7-NEG-07: a text-only scalar yield target must not flip a same-scope channel.

    build_same_scope_source_packet with only text-supported scalars must still
    return blocked_same_scope_source_packet_not_available; no BLOCKING_SAME_SCOPE_CHANNEL
    can be satisfied by text scalars alone.
    WP-N7 §6.1, N7-NEG-07.
    """
    text_only_target = {
        "name": "akel_shot_12581_neutron_yield",
        "observable": "neutron_scalar_yield",
        "status": "accepted_same_scope_source",
        "declared_scope": PF1000_SCOPE,
        "evidence_type": "text_supported_scalar_only",
    }
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(text_only_target,),
    )
    # Status must remain blocked — text scalars cannot satisfy blocking channels.
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert packet["can_support_first_principles_acceptance"] is False


def test_n7_neg08_dirty_worktree_noted_in_provenance() -> None:
    """N7-NEG-08: a manifest with dirty_worktree=True must report it in provenance.

    The artifact linter check C8 catches dirty-worktree artifacts.  This test
    confirms that FirstPrinciplesRunManifest.to_dict() propagates dirty_worktree
    so the linter / reviewer can see it.
    WP-N7 §6.1, N7-NEG-08.
    """
    manifest = FirstPrinciplesRunManifest(dirty_worktree=True)
    payload = manifest.to_dict()
    assert payload.get("dirty_worktree") is True, (
        "dirty_worktree=True was not propagated to the manifest dict"
    )
    # provenance_complete must be False (dirty_worktree is set but required
    # fields like git_commit and source_packet_hashes are absent).
    assert payload["provenance_complete"] is False


def test_n7_neg09_text_supported_channels_cannot_flip_blocking_channels() -> None:
    """N7-NEG-09: PF1000_AKEL_TEXT_SUPPORTED_CHANNELS cannot satisfy blockers.

    Accepting all text-supported channels must not flip any BLOCKING_SAME_SCOPE_CHANNEL
    in the same-scope packet; the packet must stay at
    blocked_same_scope_source_packet_not_available.
    WP-N7 §6.1, N7-NEG-09.
    """
    from dpf.first_principles.same_scope import (
        PF1000_AKEL_TEXT_SUPPORTED_CHANNELS,
        build_same_scope_source_packet,
    )

    text_targets = [
        {
            "name": ch,
            "observable": ch,
            "status": "accepted_same_scope_source",
            "declared_scope": PF1000_SCOPE,
        }
        for ch in PF1000_AKEL_TEXT_SUPPORTED_CHANNELS
    ]
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=text_targets,
    )
    assert packet["status"] == "blocked_same_scope_source_packet_not_available", (
        "text-supported scalars promoted the same-scope packet beyond blocked"
    )
    assert packet["can_support_first_principles_acceptance"] is False


def test_n7_neg10_lee_model_outputs_not_density_spatial_history() -> None:
    """N7-NEG-10: Lee model outputs cannot satisfy density_spatial_history channel.

    Table 2 of Akel 2021 gives Lee model n_i and pinch dimensions — these are
    NOT independent measurements and cannot populate density_spatial_history in
    the same-scope packet.  A target declared as 'density_spatial_history'
    sourced from a Lee model output must be rejected.
    WP-N7 §6.1, N7-NEG-10.
    """
    lee_density_target = {
        "name": "akel_2021_table2_n_i_lee_model",
        "observable": "density_spatial_history",
        "status": "accepted_same_scope_source",
        "declared_scope": PF1000_SCOPE,
        "evidence_type": "lee_model_output",  # model output, not measurement
    }
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        validation_targets=(lee_density_target,),
    )
    # density_spatial_history is a BLOCKING_SAME_SCOPE_CHANNEL; Lee model outputs
    # cannot satisfy it, so the packet must remain blocked.
    assert "density_spatial_history" not in packet.get("accepted_same_scope_channels", [])
    assert packet["status"] == "blocked_same_scope_source_packet_not_available"
    assert packet["can_support_first_principles_acceptance"] is False


def test_n7_s38_pf1000_akel_source_packet_hashes_are_candidate_only() -> None:
    """S3.8: pf1000_akel_source_packet_hashes must label all channels as
    candidate_comparator_only and must not claim acceptance.

    Handoff §S3.8 requirement: comparator-only channels labeled
    ``candidate_comparator_only``.
    WP-N7 §8 'Do Not Promote' notes.
    """
    from dpf.first_principles.source_targets import pf1000_akel_source_packet_hashes

    packet = pf1000_akel_source_packet_hashes()

    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_validation_claims"] is False
    assert packet["accepted_any"] is False
    assert packet["all_comparator_channels_labeled_candidate_comparator_only"] is True
    assert packet["status"] == "candidate_comparator_only_not_accepted"

    entries = packet["source_entries"]
    assert len(entries) >= 1, "source_entries must be non-empty"
    for entry in entries:
        assert entry["candidate_comparator_only"] is True, (
            f"entry {entry['source_id']} does not have candidate_comparator_only=True"
        )
        assert "blocking_channels" in entry
        assert "candidate_channels" in entry


def test_n7_s38_source_packet_hash_keys_present_for_all_entries() -> None:
    """S3.8: every source entry must have source_id, path, sha256, and scope fields.

    sha256 may be None if the file is absent from disk; but the key must exist.
    WP-N7 §4.1 required provenance fields.
    """
    from dpf.first_principles.source_targets import pf1000_akel_source_packet_hashes

    packet = pf1000_akel_source_packet_hashes()
    for entry in packet["source_entries"]:
        for required_key in ("source_id", "path", "sha256", "scope"):
            assert required_key in entry, (
                f"source entry {entry.get('source_id', '?')} missing key '{required_key}'"
            )


def test_n7_s38_certificate_scaffold_wiring_with_synthetic_fixture() -> None:
    """S3.8: a synthetic all-accepted fixture proves certificate WIRING only.

    Per handoff §S3.8: 'a synthetic positive fixture may prove WIRING only'.
    Even when all channels are declared accepted, the gate returns
    can_write_accepted_certificate=False because the gate's own required
    channels (including those that depend on real production artifacts) are
    always missing in Sprint 3.

    This test does NOT claim validation.  It proves that the gate correctly
    wires accepted_channels into accepted_certificate_channel statuses, and
    that all remaining channels still block the certificate.
    """
    all_required = list(build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
    )["required_channels"])

    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        accepted_channels=all_required,
        upstream_packets=_accepted_upstream(),
    )

    # Wiring check: declared accepted channels appear in accepted_channels list.
    accepted_set = set(packet["accepted_channels"])
    for channel in all_required:
        assert channel in accepted_set, (
            f"wiring failure: channel '{channel}' was declared accepted but "
            "does not appear in accepted_channels"
        )

    # Even with all channels declared accepted, Sprint 3 gates stay blocked.
    # The gate enforces that no Sprint 3 production artifact has passed all
    # required gates (WP-N7 §7.2 contract).
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_release_first_principles_claim"] is False
    # status and release_decision are hard-coded in the gate.
    assert packet["release_decision"] == "do_not_release_first_principles_claim"
