import numpy as np

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
    ManifestArtifact,
    build_first_principles_run_manifest,
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


def test_hybrid_telemetry_conservation_extracts_residual_placeholders() -> None:
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
