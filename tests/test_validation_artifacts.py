from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.validation.artifacts import (
    ArtifactClassification,
    BackendAuthority,
    CertificateEvidenceLink,
    ResultClassification,
    ResultLabel,
    RunManifest,
    ValidationCertificate,
    ValidationStatus,
    artifact_classification_from_config,
    backend_authority_for,
    build_run_manifest,
    build_validation_certificate,
    embed_hdf5_run_metadata,
    file_sha256,
    manifest_path_for_output,
    classify_result,
    stable_json_hash,
    write_validation_certificate,
    write_run_manifest,
)


def test_backend_authority_defaults_fail_closed_for_mlx() -> None:
    assert backend_authority_for("python") is BackendAuthority.REFERENCE_CANDIDATE
    assert backend_authority_for("mlx") is BackendAuthority.PREVIEW

    classification = classify_result(backend="mlx", validation_status=ValidationStatus.ACCEPTED)
    assert classification.label is ResultLabel.PREVIEW
    assert classification.can_support_validation_claims is False


def test_reference_classification_requires_accepted_reference_candidate() -> None:
    classification = classify_result(
        backend="python",
        validation_status=ValidationStatus.ACCEPTED,
    )

    assert classification.label is ResultLabel.REFERENCE
    assert classification.can_support_validation_claims is True

    with pytest.raises(ValidationError, match="Reference label requires accepted"):
        ResultClassification(
            label=ResultLabel.REFERENCE,
            backend_authority=BackendAuthority.REFERENCE_CANDIDATE,
            validation_status=ValidationStatus.BLOCKED,
            reason="blocked evidence cannot be Reference",
        )


def test_preview_result_cannot_support_validation_claims() -> None:
    with pytest.raises(ValidationError, match="only Reference results"):
        ResultClassification(
            label=ResultLabel.PREVIEW,
            reason="preview output",
            can_support_validation_claims=True,
        )


def test_run_manifest_rejects_status_mismatch() -> None:
    config_hash = stable_json_hash({"backend": "python", "sim_time": 1e-6})
    classification = classify_result(
        backend="python",
        validation_status=ValidationStatus.ACCEPTED,
    )

    manifest = RunManifest(
        run_id="run-001",
        config_hash=config_hash,
        backend="python",
        solver_mode="mhd",
        precision="float64",
        validation_status=ValidationStatus.ACCEPTED,
        result_classification=classification,
        requirement_ids=["DPF-DATA-001"],
    )
    assert manifest.result_classification.label is ResultLabel.REFERENCE

    with pytest.raises(ValidationError, match="validation_status must match"):
        RunManifest(
            run_id="run-002",
            config_hash=config_hash,
            backend="python",
            solver_mode="mhd",
            validation_status=ValidationStatus.BLOCKED,
            result_classification=classification,
        )


def test_build_run_manifest_records_preview_runtime_metadata(tmp_path) -> None:
    output = tmp_path / "diag.h5"
    output.write_bytes(b"diagnostics")
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": str(output)},
    )

    manifest = build_run_manifest(
        config=config,
        backend="python",
        summary={"steps": 1},
        reason="unit-test manifest",
    )

    assert manifest.backend == "python"
    assert manifest.validation_status is ValidationStatus.NOT_EVALUATED
    assert manifest.result_classification.label is ResultLabel.PREVIEW
    assert manifest.outputs[0].sha256 == file_sha256(output)
    assert manifest.summary_hash == stable_json_hash({"steps": 1})
    assert manifest.requirement_ids == ["DPF-DATA-001", "DPF-DATA-002"]
    assert manifest.artifact_classification.classification == "owner_unspecified"


def test_run_manifest_carries_blocked_s1_s2_waveform_evidence(tmp_path) -> None:
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
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
    summary = {
        "pf1000_16kv_current_waveform_comparison_candidate": {
            "passed": False,
            "waveform_comparison_status": "blocked_by_review",
            "validation_scope": "pf1000_16kv_2021_akel",
            "validation_blockers": [
                "independent_review_missing",
                "review_status_not_accepted",
            ],
            "details": {
                "digitization_readiness": {
                    "waveform_digitization_status": "blocked_by_review",
                    "validation_blockers": [
                        "independent_review_missing",
                        "review_status_not_accepted",
                    ],
                },
            },
            "candidate_trace_points": list(range(294)),
        },
    }

    manifest = build_run_manifest(
        config=config,
        backend="python",
        summary=summary,
        validation_status=ValidationStatus.BLOCKED,
        reason="Akel S1/S2 evidence blocked by review",
    )

    evidence = manifest.validation_evidence[
        "pf1000_16kv_current_waveform_comparison_candidate"
    ]
    assert manifest.validation_status is ValidationStatus.BLOCKED
    assert evidence["waveform_comparison_status"] == "blocked_by_review"
    assert evidence["validation_scope"] == "pf1000_16kv_2021_akel"
    assert "candidate_trace_points" not in evidence
    assert evidence["details"]["digitization_readiness"][
        "waveform_digitization_status"
    ] == "blocked_by_review"


def test_run_manifest_carries_first_principles_limiter_ledger_summary(tmp_path) -> None:
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
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
    summary = {
        "first_principles_limiter_ledger": {
            "schema": "dpf.first_principles.limiter_ledger.v1",
            "status": "blocked",
            "validation_status": "blocked",
            "activation_count": 2,
            "acceptance_blocking_activation_count": 2,
            "activated_acceptance_blockers": ["fp2.velocity_cap"],
            "can_support_first_principles_acceptance": False,
            "entries": [
                {
                    "limiter_id": "fp2.velocity_cap",
                    "activation_count": 2,
                    "before": {"min": 3.0e6, "max": 4.0e6},
                }
            ],
        },
        "first_principles_backend_scope": {
            "status": "backend_scope_blocked",
            "backend": "metal_plm",
            "requested_backend": "metal_plm",
            "requested_run_mode": "first_principles_mhd",
            "blocked_backend": "metal",
            "can_support_first_principles_acceptance": False,
            "required_limiter_telemetry": (
                "backend_native_first_principles_limiter_ledger"
            ),
            "reason": "outside first-principles acceptance scope",
        },
    }

    manifest = build_run_manifest(
        config=config,
        backend="python",
        summary=summary,
        validation_status=ValidationStatus.BLOCKED,
        reason="first-principles limiter ledger blocked",
    )

    ledger = manifest.validation_evidence["first_principles_limiter_ledger"]
    assert ledger["status"] == "blocked"
    assert ledger["acceptance_blocking_activation_count"] == 2
    assert ledger["activated_acceptance_blockers"] == ["fp2.velocity_cap"]
    assert "entries" not in ledger
    backend_scope = manifest.validation_evidence["first_principles_backend_scope"]
    assert backend_scope["status"] == "backend_scope_blocked"
    assert backend_scope["blocked_backend"] == "metal"
    assert backend_scope["required_limiter_telemetry"] == (
        "backend_native_first_principles_limiter_ledger"
    )


def test_build_run_manifest_accepts_owner_classification_metadata(tmp_path) -> None:
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
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

    manifest = build_run_manifest(
        config=config,
        backend="python",
        artifact_classification=ArtifactClassification(
            owner="owner-team",
            classification="internal",
            distribution="project-only",
            handling_notes="example metadata",
        ),
    )

    assert manifest.artifact_classification.owner == "owner-team"
    assert manifest.artifact_classification.distribution == "project-only"


def test_build_run_manifest_reads_config_artifact_classification(tmp_path) -> None:
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={
            "hdf5_filename": ":memory:",
            "artifact_owner": "config-owner",
            "artifact_classification": "internal",
            "artifact_distribution": "project-only",
            "artifact_handling_notes": "configured metadata",
        },
    )

    classification = artifact_classification_from_config(config)
    manifest = build_run_manifest(config=config, backend="python")

    assert classification.owner == "config-owner"
    assert classification.classification == "internal"
    assert manifest.artifact_classification.owner == "config-owner"
    assert manifest.artifact_classification.distribution == "project-only"


def test_build_run_manifest_supports_blocked_status_without_reference_promotion(tmp_path) -> None:
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
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

    manifest = build_run_manifest(
        config=config,
        backend="python",
        validation_status=ValidationStatus.BLOCKED,
        reason="preflight blocked launch",
    )

    assert manifest.validation_status is ValidationStatus.BLOCKED
    assert manifest.result_classification.label is ResultLabel.PREVIEW
    assert manifest.result_classification.can_support_validation_claims is False


def test_write_run_manifest_round_trips_json(tmp_path) -> None:
    classification = classify_result(backend="python")
    manifest = RunManifest(
        run_id="run-round-trip",
        config_hash=stable_json_hash({"backend": "python"}),
        backend="python",
        solver_mode="cartesian_mhd",
        validation_status=ValidationStatus.NOT_EVALUATED,
        result_classification=classification,
    )
    path = write_run_manifest(manifest, tmp_path / "run_manifest.json")

    payload = json.loads(path.read_text())
    assert payload["run_id"] == "run-round-trip"
    assert payload["result_classification"]["label"] == "Preview"


def test_engine_run_emits_preview_classification_and_sidecar_manifest(tmp_path) -> None:
    import h5py

    hdf5_path = tmp_path / "diag.h5"
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": str(hdf5_path), "output_interval": 1},
    )

    summary = SimulationEngine(config).run(max_steps=1)

    assert summary["validation_status"] == "not_evaluated"
    assert summary["result_classification"]["label"] == "Preview"
    assert summary["result_classification"]["can_support_validation_claims"] is False
    manifest_path = manifest_path_for_output(hdf5_path)
    assert summary["run_manifest_path"] == str(manifest_path)
    payload = json.loads(manifest_path.read_text())
    assert payload["backend"] == "python"
    assert payload["outputs"][0]["kind"] == "hdf5_diagnostics"
    assert payload["outputs"][0]["sha256"] == file_sha256(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        assert handle.attrs["dpf_backend"] == "python"
        assert handle.attrs["dpf_validation_status"] == "not_evaluated"
        assert handle.attrs["dpf_result_label"] == "Preview"
        assert not bool(handle.attrs["dpf_can_support_validation_claims"])
        assert "local KnowledgeReference" in handle.attrs["dpf_source_authority"]


def test_hdf5_metadata_embeds_compact_readiness_evidence(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "readiness_diag.h5"
    with h5py.File(path, "w"):
        pass

    summary = {
        "predictive_readiness": {
            "ready": False,
            "validation_status": "blocked",
            "blockers": ["missing_same_scope_targets"],
            "large_payload": list(range(100)),
        },
        "digitization_status": {
            "status": "blocked",
            "validation_blockers": ["independent_review_missing"],
        },
        "source_blockers": [
            "independent_review_missing",
            "review_status_not_accepted",
        ],
    }
    embed_hdf5_run_metadata(
        path,
        backend="python",
        solver_mode="cartesian_mhd",
        validation_status=ValidationStatus.BLOCKED,
        result_classification=classify_result(
            backend="python",
            validation_status=ValidationStatus.BLOCKED,
        ),
        summary=summary,
    )

    with h5py.File(path, "r") as handle:
        evidence = json.loads(handle.attrs["dpf_readiness_summary_json"])
        assert evidence["predictive_readiness"]["ready"] is False
        assert evidence["predictive_readiness"]["blockers"] == [
            "missing_same_scope_targets"
        ]
        assert "large_payload" not in evidence["predictive_readiness"]
        assert evidence["source_blockers"] == [
            "independent_review_missing",
            "review_status_not_accepted",
        ]


def test_engine_run_propagates_config_artifact_classification(tmp_path) -> None:
    import h5py

    hdf5_path = tmp_path / "classified_diag.h5"
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={
            "hdf5_filename": str(hdf5_path),
            "output_interval": 1,
            "well_output_interval": 1,
            "well_filename_prefix": "classified_well",
            "artifact_owner": "config-owner",
            "artifact_classification": "internal",
            "artifact_distribution": "project-only",
            "artifact_handling_notes": "configured export metadata",
        },
    )

    summary = SimulationEngine(config).run(max_steps=1)

    manifest_payload = json.loads(manifest_path_for_output(hdf5_path).read_text())
    assert summary["run_manifest"]["artifact_classification"]["owner"] == "config-owner"
    assert manifest_payload["artifact_classification"]["classification"] == "internal"
    assert manifest_payload["artifact_classification"]["distribution"] == "project-only"

    with h5py.File(hdf5_path, "r") as handle:
        payload = json.loads(handle.attrs["dpf_artifact_classification_json"])
        assert payload["owner"] == "config-owner"
        assert payload["classification"] == "internal"

    well_files = sorted(tmp_path.glob("classified_well_*.h5"))
    assert well_files
    with h5py.File(well_files[0], "r") as handle:
        assert handle.attrs["artifact_owner"] == "config-owner"
        assert handle.attrs["artifact_classification"] == "internal"
        assert handle.attrs["artifact_distribution"] == "project-only"
        assert handle.attrs["validation_status"] == "not_validation_evidence"


def test_engine_run_emits_failed_manifest_before_reraising(tmp_path, monkeypatch) -> None:
    hdf5_path = tmp_path / "failed.h5"
    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": str(hdf5_path), "output_interval": 1},
    )
    engine = SimulationEngine(config)

    def fail_step(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(engine, "step", fail_step)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        engine.run(max_steps=1)

    payload = json.loads(manifest_path_for_output(hdf5_path).read_text())
    assert payload["validation_status"] == "failed"
    assert payload["result_classification"]["label"] == "Preview"
    assert payload["result_classification"]["can_support_validation_claims"] is False


def test_validation_certificate_rejects_blocked_draft_evidence() -> None:
    with pytest.raises(ValidationError, match="blocked or failed evidence"):
        ValidationCertificate(
            certificate_id="cert-akel-draft",
            requirement_ids=["DPF-PHYS-004"],
            validation_scope="akel_2021_shot_12581_16kv",
            validation_status=ValidationStatus.ACCEPTED,
            result_label=ResultLabel.REFERENCE,
            review_status="accepted",
            reviewers=["independent-reviewer"],
            evidence_links=[
                CertificateEvidenceLink(
                    requirement_id="DPF-PHYS-004",
                    evidence_uri=(
                        "KnowledgeReference/digitization/"
                        "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
                    ),
                    status=ValidationStatus.BLOCKED,
                    validation_scope="akel_2021_shot_12581_16kv",
                    notes="waveform_digitization_status=blocked_by_review",
                )
            ],
        )


def test_validation_certificate_rejects_cross_scope_evidence() -> None:
    with pytest.raises(ValidationError, match="same-scope evidence"):
        ValidationCertificate(
            certificate_id="cert-cross-scope",
            requirement_ids=["DPF-PHYS-004"],
            validation_scope="akel_2021_shot_12581_16kv",
            validation_status=ValidationStatus.ACCEPTED,
            result_label=ResultLabel.REFERENCE,
            review_status="accepted",
            reviewers=["independent-reviewer"],
            evidence_links=[
                CertificateEvidenceLink(
                    requirement_id="DPF-PHYS-004",
                    evidence_uri="KnowledgeReference/pf1000-full-energy-waveform.md",
                    status=ValidationStatus.ACCEPTED,
                    validation_scope="pf1000_27kv_full_energy",
                )
            ],
        )


def test_validation_certificate_accepts_reviewed_same_scope_evidence() -> None:
    certificate = build_validation_certificate(
        certificate_id="cert-akel-accepted",
        requirement_ids=["DPF-PHYS-004"],
        validation_scope="akel_2021_shot_12581_16kv",
        validation_status=ValidationStatus.ACCEPTED,
        result_label=ResultLabel.REFERENCE,
        review_status="accepted",
        reviewers=["independent-reviewer"],
        evidence_links=[
            CertificateEvidenceLink(
                requirement_id="DPF-PHYS-004",
                evidence_uri="KnowledgeReference/digitization/accepted-akel-waveform.json",
                status=ValidationStatus.ACCEPTED,
                validation_scope="akel_2021_shot_12581_16kv",
            )
        ],
    )

    assert certificate.validation_status is ValidationStatus.ACCEPTED


def test_validation_certificate_carries_blocked_readiness_context(tmp_path) -> None:
    certificate = build_validation_certificate(
        certificate_id="cert-blocked-context",
        requirement_ids=["DPF-PHYS-004"],
        validation_scope="pf1000_16kv_2021_akel",
        validation_status=ValidationStatus.BLOCKED,
        result_label=ResultLabel.PREVIEW,
        result_classification={
            "label": "Preview",
            "can_support_validation_claims": False,
            "validation_status": "blocked",
        },
        artifact_classification={
            "owner": "validation-team",
            "classification": "internal",
            "distribution": "review-only",
        },
        readiness_summary={
            "predictive_readiness": {"ready": False},
            "high_fidelity_readiness": {"ready": False},
        },
        blockers=["independent_review_missing"],
        review_status="blocked",
        evidence_links=[
            CertificateEvidenceLink(
                requirement_id="DPF-PHYS-004",
                evidence_uri="KnowledgeReference/digitization/akel-draft.json",
                status=ValidationStatus.BLOCKED,
                validation_scope="pf1000_16kv_2021_akel",
            )
        ],
    )

    path = write_validation_certificate(certificate, tmp_path / "blocked_cert.json")
    payload = json.loads(path.read_text())
    assert payload["result_classification"]["label"] == "Preview"
    assert payload["artifact_classification"]["classification"] == "internal"
    assert payload["readiness_summary"]["predictive_readiness"]["ready"] is False
    assert payload["blockers"] == ["independent_review_missing"]


def test_accepted_validation_certificate_rejects_blockers() -> None:
    with pytest.raises(ValidationError, match="cannot carry blockers"):
        build_validation_certificate(
            certificate_id="cert-blocked-accepted",
            requirement_ids=["DPF-PHYS-004"],
            validation_scope="akel_2021_shot_12581_16kv",
            validation_status=ValidationStatus.ACCEPTED,
            result_label=ResultLabel.REFERENCE,
            result_classification={
                "label": "Reference",
                "can_support_validation_claims": True,
            },
            blockers=["source_blocker"],
            review_status="accepted",
            reviewers=["independent-reviewer"],
            evidence_links=[
                CertificateEvidenceLink(
                    requirement_id="DPF-PHYS-004",
                    evidence_uri="KnowledgeReference/digitization/accepted-akel-waveform.json",
                    status=ValidationStatus.ACCEPTED,
                    validation_scope="akel_2021_shot_12581_16kv",
                )
            ],
        )


def test_validation_certificate_writer_persists_only_validated_model(tmp_path) -> None:
    certificate = build_validation_certificate(
        certificate_id="cert-write",
        requirement_ids=["DPF-PHYS-004"],
        validation_scope="akel_2021_shot_12581_16kv",
        validation_status=ValidationStatus.ACCEPTED,
        result_label=ResultLabel.REFERENCE,
        review_status="accepted",
        reviewers=["independent-reviewer"],
        evidence_links=[
            CertificateEvidenceLink(
                requirement_id="DPF-PHYS-004",
                evidence_uri="KnowledgeReference/digitization/accepted-akel-waveform.json",
                status=ValidationStatus.ACCEPTED,
                validation_scope="akel_2021_shot_12581_16kv",
            )
        ],
    )

    path = write_validation_certificate(certificate, tmp_path / "certificate.json")

    payload = json.loads(path.read_text())
    assert payload["certificate_id"] == "cert-write"
    assert payload["validation_status"] == "accepted"
