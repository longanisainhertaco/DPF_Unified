import numpy as np

from dpf.first_principles.certificate_gate import REQUIRED_CERTIFICATE_CHANNELS
from dpf.first_principles.conservation import (
    ARTIFACT_STATUS_ENGINEERING_CANDIDATE,
    VALIDATION_STATUS_NOT_VALIDATION,
    SourceIndexReference,
    build_conservation_ledger,
    build_conservation_ledger_from_hybrid_telemetry,
    build_residual_ledger,
    compute_circuit_energy,
    compute_electron_energy,
    compute_field_energy,
)
from dpf.first_principles.manifest import (
    REQUIRED_PROVENANCE_FIELDS,
    FirstPrinciplesRunManifest,
    ManifestArtifact,
    build_first_principles_run_manifest,
    sha256_of_file,
    sha256_of_file_soft,
    sha256_of_text,
    source_packet_hashes_from_references,
    stable_manifest_hash,
)


def test_conservation_ledger_records_energy_particles_and_residuals() -> None:
    electric0 = np.zeros((2, 2, 2, 3), dtype=float)
    magnetic0 = np.zeros((2, 2, 2, 3), dtype=float)
    electric1 = np.ones((2, 2, 2, 3), dtype=float) * 10.0
    magnetic1 = np.ones((2, 2, 2, 3), dtype=float) * 0.01
    field0 = compute_field_energy(electric0, magnetic0, cell_volume_m3=1.0e-9)
    field1 = compute_field_energy(electric1, magnetic1, cell_volume_m3=1.0e-9)
    circuit0 = compute_circuit_energy(
        current_A=100.0,
        voltage_V=10_000.0,
        capacitance_F=2.0e-6,
        inductance_H=1.0e-7,
    )
    circuit1 = compute_circuit_energy(
        current_A=90.0,
        charge_C=0.015,
        capacitance_F=2.0e-6,
        inductance_H=1.0e-7,
    )
    electron0 = compute_electron_energy(np.ones((2, 2, 2)), cell_volume_m3=1.0e-9)
    electron1 = compute_electron_energy(
        np.ones((2, 2, 2)) * 1.5,
        cell_volume_m3=1.0e-9,
    )

    ledger = build_conservation_ledger(
        field_energy_initial=field0,
        field_energy_final=field1,
        circuit_energy_initial=circuit0,
        circuit_energy_final=circuit1,
        particle_count_initial=12,
        particle_count_final=10,
        electron_energy_initial=electron0,
        electron_energy_final=electron1,
        residuals=build_residual_ledger(
            gauss_law_linf=3.0e-4,
            div_B_linf=2.0e-5,
            current_continuity_linf_A_m3=1.0e6,
            current_residual_linf_A_m2=7.0e-2,
        ),
        source_index_references=[
            SourceIndexReference(
                source_id="hybrid_pic_3d_source",
                path="KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md",
            )
        ],
    )

    assert ledger.validation_status == VALIDATION_STATUS_NOT_VALIDATION
    assert ledger.artifact_status == ARTIFACT_STATUS_ENGINEERING_CANDIDATE
    assert ledger.can_support_first_principles_acceptance is False
    assert ledger.field_energy is not None
    assert ledger.field_energy.total.delta_J == field1.total_J - field0.total_J
    assert ledger.circuit_energy is not None
    assert ledger.circuit_energy.total.status == "measured_accounting_not_validation"
    assert ledger.particle_count is not None
    assert ledger.particle_count.delta_count == -2
    assert ledger.particle_count.status == "changed_not_validation"
    assert ledger.electron_energy is not None
    assert ledger.electron_energy.delta_J == electron1.total_J - electron0.total_J
    assert ledger.residuals.status == "measured_residuals_not_validation"
    assert ledger.source_index_references[0].status == "source_reference_not_validation"


def test_hybrid_telemetry_conservation_extracts_measured_residual_channels() -> None:
    telemetry = {
        "n_steps_requested": 3,
        "n_steps_completed": 2,
        "final_time_s": 2.0e-13,
        "n_particles_initial": 5,
        "n_particles_final": 5,
        "initial_field_energy_J": 1.0e-6,
        "final_field_energy_J": 1.4e-6,
        "last_step": {
            "field_step": {
                "diagnostics_after": {"max_abs_div_B_T_per_m": 8.0e-9},
                "marder": {"residual_after_linf": 4.0e-3},
                "current_port": {"continuity_linf_A_per_m3": 9.0e4},
                "ohm_solver": {"max_algebraic_residual_A_m2": 2.0e-7},
            }
        },
    }

    ledger = build_conservation_ledger_from_hybrid_telemetry(telemetry)

    assert ledger.field_energy is not None
    assert ledger.field_energy.total.initial_J == 1.0e-6
    assert ledger.field_energy.total.final_J == 1.4e-6
    assert ledger.particle_count is not None
    assert ledger.particle_count.status == "conserved_not_validation"
    assert ledger.residuals.gauss_law_linf == 4.0e-3
    assert ledger.residuals.div_B_linf == 8.0e-9
    assert ledger.residuals.current_continuity_linf_A_m3 == 9.0e4
    assert ledger.residuals.current_residual_linf_A_m2 == 2.0e-7
    assert ledger.residuals.missing_channels == ()


def test_first_principles_manifest_is_engineering_candidate_not_validation() -> None:
    telemetry = {
        "n_steps_requested": 1,
        "n_steps_completed": 1,
        "final_time_s": 1.0e-13,
        "n_particles_initial": 2,
        "n_particles_final": 2,
        "initial_field_energy_J": 0.0,
        "final_field_energy_J": 1.0e-9,
        "last_step": {"field_step": {"diagnostics_after": {}}},
    }

    manifest = build_first_principles_run_manifest(
        run_id="fp-test-run",
        backend="package_native",
        telemetry=telemetry,
        grid_shape=(4, 4, 4),
        grid_spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
        source_index_references=[
            {
                "id": "source-truth-index:hybrid-pic-3d",
                "path": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json",
                "status": "candidate_source_index_not_validation",
            }
        ],
        outputs=[
            ManifestArtifact(
                path="results/fp-run.json",
                kind="run_summary",
                sha256="abc123",
            )
        ],
        metadata={"solver": "hybrid_pic_3d"},
    )
    payload = manifest.to_dict()

    assert payload["run_id"] == "fp-test-run"
    assert payload["run_status"] == "engineering_candidate"
    assert payload["validation_status"] == "not_validation"
    assert payload["artifact_status"] == "engineering_candidate_not_validation"
    assert payload["can_support_first_principles_acceptance"] is False
    assert payload["solver_family"] == "3d_hybrid_em_pic_fluid"
    assert payload["n_steps_completed"] == 1
    assert payload["grid_shape"] == (4, 4, 4)
    assert payload["conservation"]["validation_status"] == "not_validation"
    assert payload["conservation"]["field_energy"]["total"]["final_J"] == 1.0e-9
    assert payload["source_index_references"][0]["status"] == (
        "candidate_source_index_not_validation"
    )
    assert payload["outputs"][0]["role"] == "engineering_artifact_not_validation"
    assert len(payload["manifest_sha256"]) == 64
    assert stable_manifest_hash(payload) == payload["manifest_sha256"]


# ---------------------------------------------------------------------------
# Codex A-8 / WP-N7: manifest provenance fields
# ---------------------------------------------------------------------------

def _fully_provenanced_manifest() -> FirstPrinciplesRunManifest:
    """A manifest with every required provenance field populated."""
    return build_first_principles_run_manifest(
        run_id="fp-provenance-run",
        backend="package_native",
        command_argv=(
            "dpf",
            "first-principles-3d",
            "--deck-preset",
            "pf1000_akel_16kv",
            "--steps",
            "2",
        ),
        git_commit="0123456789abcdef0123456789abcdef01234567",
        dirty_worktree=False,
        source_truth_index_sha256=sha256_of_text("source-truth-index-content"),
        source_packet_hashes={
            "hybrid_pic_3d_source": sha256_of_text("hybrid-pic-3d-packet"),
        },
        input_deck_sha256=sha256_of_text("pf1000-akel-deck"),
        artifact_schema_version="first_principles_artifact_v1",
        artifact_generation_commit="0123456789abcdef0123456789abcdef01234567",
    )


def test_manifest_exposes_provenance_fields() -> None:
    """A-8: the eight provenance fields exist as first-class manifest fields
    and survive serialization."""
    manifest = _fully_provenanced_manifest()

    # Fields exist on the dataclass instance.
    assert manifest.command_argv == (
        "dpf",
        "first-principles-3d",
        "--deck-preset",
        "pf1000_akel_16kv",
        "--steps",
        "2",
    )
    assert manifest.git_commit == "0123456789abcdef0123456789abcdef01234567"
    assert manifest.dirty_worktree is False
    assert len(manifest.source_truth_index_sha256) == 64
    assert manifest.source_packet_hashes["hybrid_pic_3d_source"]
    assert len(manifest.input_deck_sha256) == 64
    assert manifest.artifact_schema_version == "first_principles_artifact_v1"
    # artifact_generation_commit is a git commit SHA (40 hex chars), not a
    # content hash.
    assert manifest.artifact_generation_commit == (
        "0123456789abcdef0123456789abcdef01234567"
    )

    # Fields are serialized into the manifest payload.
    payload = manifest.to_dict()
    for name in (
        "command_argv",
        "git_commit",
        "dirty_worktree",
        "source_truth_index_sha256",
        "source_packet_hashes",
        "input_deck_sha256",
        "artifact_schema_version",
        "artifact_generation_commit",
    ):
        assert name in payload, f"provenance field {name!r} missing from payload"

    # command_argv serializes as a JSON-friendly list, not a tuple.
    assert payload["command_argv"] == [
        "dpf",
        "first-principles-3d",
        "--deck-preset",
        "pf1000_akel_16kv",
        "--steps",
        "2",
    ]
    # The hash covers the new provenance fields.
    assert stable_manifest_hash(payload) == payload["manifest_sha256"]


def test_manifest_construction_still_works_without_provenance() -> None:
    """Provenance fields are optional/None-tolerant: a manifest with no
    provenance still constructs (existing call sites must not break)."""
    manifest = build_first_principles_run_manifest(run_id="fp-no-provenance")

    assert manifest.command_argv is None
    assert manifest.git_commit is None
    assert manifest.dirty_worktree is None
    assert manifest.source_truth_index_sha256 is None
    assert manifest.source_packet_hashes == {}
    assert manifest.input_deck_sha256 is None
    assert manifest.artifact_schema_version is None
    assert manifest.artifact_generation_commit is None
    # The fail-closed contract is unchanged.
    assert manifest.can_support_first_principles_acceptance is False


def test_missing_provenance_blocks_certificate_acceptance() -> None:
    """A-8: a manifest without command_argv / artifact_generation_commit
    cannot support an accepted certificate.

    The certificate gate requires the `commands_and_versions` and
    `run_manifest_hash` channels.  Those channels cannot be filled by a
    manifest whose provenance is incomplete, so an incomplete-provenance
    manifest fails the provenance contract that the certificate depends on.
    """
    incomplete = build_first_principles_run_manifest(run_id="fp-incomplete")

    missing = incomplete.missing_provenance_fields()
    # The provenance contract reports the gap explicitly.
    assert "command_argv" in missing
    assert "artifact_generation_commit" in missing
    assert set(missing) == set(REQUIRED_PROVENANCE_FIELDS)
    assert incomplete.has_complete_provenance() is False

    payload = incomplete.to_dict()
    assert payload["provenance_complete"] is False
    assert "command_argv" in payload["missing_provenance_fields"]

    # The certificate's command/manifest channels exist precisely so this
    # provenance must be present; an incomplete manifest cannot fill them.
    assert "commands_and_versions" in REQUIRED_CERTIFICATE_CHANNELS
    assert "run_manifest_hash" in REQUIRED_CERTIFICATE_CHANNELS

    # A fully provenanced manifest clears the contract.
    complete = _fully_provenanced_manifest()
    assert complete.missing_provenance_fields() == ()
    assert complete.has_complete_provenance() is True
    # Even with complete provenance the manifest still refuses to claim
    # acceptance -- provenance is necessary, never sufficient.
    assert complete.can_support_first_principles_acceptance is False


def test_empty_provenance_strings_are_treated_as_missing() -> None:
    """A-8: empty/blank provenance values do not count as provenance.

    An empty command_argv tuple or a whitespace-only commit must be
    reported as missing, so blank values cannot sneak past the gate.
    """
    blank = build_first_principles_run_manifest(
        run_id="fp-blank-provenance",
        command_argv=(),
        git_commit="   ",
        artifact_generation_commit="",
    )

    missing = blank.missing_provenance_fields()
    assert "command_argv" in missing
    assert "git_commit" in missing
    assert "artifact_generation_commit" in missing
    assert blank.has_complete_provenance() is False


# ---------------------------------------------------------------------------
# Codex A-1: source-truth-index and source-packet hashing helpers
# ---------------------------------------------------------------------------


def test_sha256_of_file_soft_returns_none_for_missing_file(tmp_path) -> None:
    """A-1: fail-soft hashing returns None for an unreadable file rather than
    raising -- provenance collection must never crash a run."""
    missing = tmp_path / "does_not_exist.json"

    assert sha256_of_file_soft(missing) is None


def test_sha256_of_file_soft_matches_hard_hash_for_real_file(tmp_path) -> None:
    """A-1: when the file exists, the fail-soft hash equals the strict hash --
    the soft variant must not fabricate or alter a present digest."""
    target = tmp_path / "packet.md"
    target.write_text("source-packet-body", encoding="utf-8")

    soft = sha256_of_file_soft(target)
    hard = sha256_of_file(target)

    assert soft == hard
    assert soft is not None and len(soft) == 64


def test_source_packet_hashes_from_references_hashes_cited_packets(
    tmp_path,
) -> None:
    """A-1: every cited source packet is hashed under its source_id; the
    digest matches a direct hash of the file."""
    packet = tmp_path / "hybrid_pic_3d.md"
    packet.write_text("hybrid-pic-3d-architecture", encoding="utf-8")
    references = [
        SourceIndexReference(
            source_id="hybrid_pic_3d_architecture_source",
            path=str(packet),
        )
    ]

    hashes = source_packet_hashes_from_references(references)

    assert set(hashes) == {"hybrid_pic_3d_architecture_source"}
    assert hashes["hybrid_pic_3d_architecture_source"] == sha256_of_file(packet)


def test_source_packet_hashes_from_references_omits_missing_source(
    tmp_path,
) -> None:
    """A-1: a cited source whose file is missing is omitted from the hash map
    (fail-soft) -- an absent key is honest, a fabricated hash is not."""
    present = tmp_path / "present.md"
    present.write_text("present-packet", encoding="utf-8")
    references = [
        SourceIndexReference(source_id="present_source", path=str(present)),
        SourceIndexReference(
            source_id="missing_source",
            path=str(tmp_path / "missing.md"),
        ),
    ]

    hashes = source_packet_hashes_from_references(references)

    assert "present_source" in hashes
    assert "missing_source" not in hashes


def test_source_packet_hashes_from_references_resolves_repo_root(
    tmp_path,
) -> None:
    """A-1: a relative reference path is resolved against repo_root so a run
    invoked from any working directory still hashes the cited packet."""
    sub = tmp_path / "KnowledgeReference"
    sub.mkdir()
    packet = sub / "packet.md"
    packet.write_text("relative-packet", encoding="utf-8")
    references = [
        {"source_id": "rel_source", "path": "KnowledgeReference/packet.md"}
    ]

    hashes = source_packet_hashes_from_references(references, repo_root=tmp_path)

    assert hashes["rel_source"] == sha256_of_file(packet)


def test_empty_source_packet_hashes_makes_provenance_incomplete() -> None:
    """A-1: a manifest with every required provenance field populated but
    ``source_packet_hashes={}`` must report incomplete provenance.

    An empty dict is not a populated hash map -- the manifest cannot certify
    which source packets were used, so provenance is incomplete (Codex A-1).
    """
    manifest = build_first_principles_run_manifest(
        run_id="fp-empty-hashes",
        command_argv=("dpf", "first-principles-3d", "--steps", "2"),
        git_commit="0123456789abcdef0123456789abcdef01234567",
        dirty_worktree=False,
        source_truth_index_sha256=sha256_of_text("source-truth-index-content"),
        source_packet_hashes={},  # deliberately empty
        input_deck_sha256=sha256_of_text("pf1000-akel-deck"),
        artifact_schema_version="first_principles_artifact_v1",
        artifact_generation_commit="0123456789abcdef0123456789abcdef01234567",
    )

    assert manifest.has_complete_provenance() is False
    assert "source_packet_hashes" in manifest.missing_provenance_fields()


def test_nonempty_source_packet_hashes_with_all_fields_gives_complete_provenance() -> None:
    """A-1 positive: all required provenance fields populated including a
    non-empty ``source_packet_hashes`` → ``has_complete_provenance() is True``.
    """
    manifest = build_first_principles_run_manifest(
        run_id="fp-complete-hashes",
        command_argv=("dpf", "first-principles-3d", "--steps", "2"),
        git_commit="0123456789abcdef0123456789abcdef01234567",
        dirty_worktree=False,
        source_truth_index_sha256=sha256_of_text("source-truth-index-content"),
        source_packet_hashes={"src": sha256_of_text("packet-body")},
        input_deck_sha256=sha256_of_text("pf1000-akel-deck"),
        artifact_schema_version="first_principles_artifact_v1",
        artifact_generation_commit="0123456789abcdef0123456789abcdef01234567",
    )

    assert manifest.has_complete_provenance() is True
    assert "source_packet_hashes" not in manifest.missing_provenance_fields()


def test_source_packet_hashes_from_references_skips_incomplete_reference(
    tmp_path,
) -> None:
    """A-1: a reference lacking a source_id or a path contributes nothing --
    it cannot be keyed or hashed, so it is skipped rather than failing."""
    packet = tmp_path / "packet.md"
    packet.write_text("body", encoding="utf-8")
    references = [
        {"path": str(packet)},  # no source_id
        {"source_id": "no_path_source"},  # no path
    ]

    hashes = source_packet_hashes_from_references(references)

    assert hashes == {}


def test_full_provenance_manifest_with_index_and_packet_hashes_is_complete(
    tmp_path,
) -> None:
    """A-1: a manifest carrying source_truth_index_sha256 and populated
    source_packet_hashes has complete provenance and a stable hash."""
    index = tmp_path / "FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json"
    index.write_text('{"sources": []}', encoding="utf-8")
    packet = tmp_path / "hybrid_pic_3d.md"
    packet.write_text("hybrid-pic-3d", encoding="utf-8")
    references = [
        SourceIndexReference(
            source_id="hybrid_pic_3d_architecture_source",
            path=str(packet),
        )
    ]

    manifest = build_first_principles_run_manifest(
        run_id="fp-index-provenance",
        backend="package_native",
        command_argv=("dpf", "first-principles-3d", "--steps", "2"),
        git_commit="0123456789abcdef0123456789abcdef01234567",
        dirty_worktree=False,
        source_index_references=references,
        source_truth_index_sha256=sha256_of_file(index),
        source_packet_hashes=source_packet_hashes_from_references(references),
        input_deck_sha256=sha256_of_text("deck"),
        artifact_schema_version="first_principles_artifact_v1",
        artifact_generation_commit="0123456789abcdef0123456789abcdef01234567",
    )

    assert manifest.source_truth_index_sha256 == sha256_of_file(index)
    assert manifest.source_packet_hashes == {
        "hybrid_pic_3d_architecture_source": sha256_of_file(packet),
    }
    assert manifest.has_complete_provenance() is True
    assert "source_truth_index_sha256" in REQUIRED_PROVENANCE_FIELDS

    payload = manifest.to_dict()
    assert payload["provenance_complete"] is True
    assert payload["missing_provenance_fields"] == []
    assert payload["source_packet_hashes"] == {
        "hybrid_pic_3d_architecture_source": sha256_of_file(packet),
    }
    assert stable_manifest_hash(payload) == payload["manifest_sha256"]
