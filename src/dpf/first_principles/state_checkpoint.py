"""Experimental first-principles terminal-state checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENTAL_STATE_CHECKPOINT_STATUS = (
    "experimental_state_checkpoint_roundtrip_not_restart_acceptance"
)


def write_terminal_state_checkpoint_roundtrip(
    *,
    run_result: Any,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Write/read a terminal state checkpoint and compare content hashes."""

    return write_simulation_state_checkpoint_roundtrip(
        simulation=run_result.result,
        checkpoint_path=checkpoint_path,
        manifest=_mapping(getattr(run_result, "manifest", None)),
    )


def write_simulation_state_checkpoint_roundtrip(
    *,
    simulation: Any,
    checkpoint_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write/read a simulation state checkpoint and compare content hashes."""

    path = Path(checkpoint_path)
    arrays, metadata = _checkpoint_arrays_and_metadata(
        simulation=simulation,
        manifest=_mapping(manifest),
    )
    write_hash = _checkpoint_content_hash(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    loaded_arrays, loaded_metadata = _read_checkpoint_payload(path)
    read_hash = _checkpoint_content_hash(loaded_arrays, loaded_metadata)
    hashes_match = write_hash == read_hash
    terminal_fingerprint = _mapping(
        getattr(simulation.telemetry, "state_fingerprint", None)
    )
    return {
        "status": EXPERIMENTAL_STATE_CHECKPOINT_STATUS,
        "checkpoint_path": str(path),
        "checkpoint_format": "npz",
        "write_content_sha256": write_hash,
        "read_content_sha256": read_hash,
        "write_read_hashes_match": hashes_match,
        "terminal_state_fingerprint_sha256": terminal_fingerprint.get("sha256"),
        "terminal_state_fingerprint_status": terminal_fingerprint.get("status"),
        "array_count": len(arrays),
        "metadata": metadata,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "checkpoint_artifact_is_restart_plumbing_only": True,
            "validation_promotion_allowed": False,
        },
        "restart_acceptance": {
            "can_restart_live_runner_from_checkpoint": False,
            "continued_run_equivalence_available": False,
            "status": "checkpoint_roundtrip_only_no_live_restart",
            "still_required": [
                "load_checkpoint_into_first_principles_runner_state",
                "restore_lagged_field_work_and_circuit_sequence_state",
                "continue_from_multiple_restart_offsets",
                "compare_against_uninterrupted_terminal_state_fingerprint",
            ],
        },
        "source_references": [
            {
                "path": "docs/DPF_REQUIREMENTS_BASELINE.md",
                "lines": "87-88",
                "role": "checkpoint_restart_deterministic_comparison_requirement",
            },
            {
                "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
                "lines": "412-414",
                "role": "restart_reproducibility_acceptance_fields",
            },
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_"
                    "2026_05_15.md"
                ),
                "lines": "60-85",
                "role": "restart_hash_and_artifact_requirement",
            },
        ],
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "validated": False,
            "review_decision": "terminal_checkpoint_roundtrip_only",
        },
        "can_support_first_principles_acceptance": False,
    }


def load_checkpoint_into_first_principles_3d_session(
    *,
    checkpoint_path: str | Path,
    deck: Mapping[str, Any] | object | None,
) -> Any:
    """Load a terminal checkpoint into a fresh first-principles 3-D session."""

    from dpf.fields import CircuitState, DeuteriumIonizationState, ElectronEnergyState
    from dpf.first_principles.runner import build_first_principles_3d_session

    arrays, metadata = _read_checkpoint_payload(Path(checkpoint_path))
    session = build_first_principles_3d_session(deck)
    state = session.simulator.state
    state.E.Ex_edge = np.array(arrays["E_Ex_edge"], copy=True)
    state.E.Ey_edge = np.array(arrays["E_Ey_edge"], copy=True)
    state.E.Ez_edge = np.array(arrays["E_Ez_edge"], copy=True)
    state.B.Bx_face = np.array(arrays["B_Bx_face"], copy=True)
    state.B.By_face = np.array(arrays["B_By_face"], copy=True)
    state.B.Bz_face = np.array(arrays["B_Bz_face"], copy=True)

    if "electron_energy_J_m3" in arrays:
        session.electron_state = ElectronEnergyState(
            electron_energy_J_m3=np.array(arrays["electron_energy_J_m3"], copy=True),
            electron_temperature_K=np.array(arrays["electron_temperature_K"], copy=True),
            ion_temperature_K=np.array(arrays["ion_temperature_K"], copy=True),
        )
    if "ionization_neutral_density_m3" in arrays:
        session.ionization_state = DeuteriumIonizationState(
            neutral_density_m3=np.array(
                arrays["ionization_neutral_density_m3"],
                copy=True,
            ),
            ion_density_m3=np.array(arrays["ionization_ion_density_m3"], copy=True),
            electron_density_m3=np.array(
                arrays["ionization_electron_density_m3"],
                copy=True,
            ),
            mean_charge_state=np.array(
                arrays["ionization_mean_charge_state"],
                copy=True,
            ),
        )
    if "circuit_state" in arrays:
        circuit = np.asarray(arrays["circuit_state"], dtype=float)
        session.circuit_state = CircuitState(
            current_A=float(circuit[0]),
            charge_C=float(circuit[1]),
        )
    if "previous_total_current_A_m2" in arrays:
        session.simulator.loop.field_stepper.previous_total_current_A_m2 = np.array(
            arrays["previous_total_current_A_m2"],
            copy=True,
        )
    for species_info in metadata.get("particle_species", []):
        if not isinstance(species_info, Mapping):
            continue
        index = int(species_info["index"])
        if index >= len(session.simulator.pic.species):
            raise ValueError("checkpoint species index exceeds session species count")
        prefix = f"species_{index}"
        species = session.simulator.pic.species[index]
        species.positions = np.array(arrays[f"{prefix}_positions"], copy=True)
        species.positions_old = np.array(arrays[f"{prefix}_positions_old"], copy=True)
        species.velocities = np.array(arrays[f"{prefix}_velocities"], copy=True)
        species.weights = np.array(arrays[f"{prefix}_weights"], copy=True)
    kinetic = metadata.get("kinetic_yield_state")
    history = session.simulator.loop.kinetic_yield_history
    if isinstance(kinetic, Mapping) and history is not None:
        history.cumulative_neutrons = float(kinetic.get("cumulative_neutrons", 0.0))
        history.time_s = float(kinetic.get("time_s", 0.0))
    continuation = metadata.get("continuation_state")
    if isinstance(continuation, Mapping):
        session.completed_steps = int(
            continuation.get(
                "total_steps_completed",
                metadata.get("n_steps_completed", 0),
            )
        )
        lagged = continuation.get("lagged_field_work")
        session.lagged_field_work = dict(lagged) if isinstance(lagged, Mapping) else None
    else:
        session.completed_steps = int(metadata.get("n_steps_completed", 0))
    return session


def _checkpoint_arrays_and_metadata(
    *,
    simulation: Any,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {
        "E_Ex_edge": np.asarray(simulation.state.E.Ex_edge),
        "E_Ey_edge": np.asarray(simulation.state.E.Ey_edge),
        "E_Ez_edge": np.asarray(simulation.state.E.Ez_edge),
        "B_Bx_face": np.asarray(simulation.state.B.Bx_face),
        "B_By_face": np.asarray(simulation.state.B.By_face),
        "B_Bz_face": np.asarray(simulation.state.B.Bz_face),
    }
    metadata: dict[str, Any] = {
        "status": EXPERIMENTAL_STATE_CHECKPOINT_STATUS,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "telemetry_status": simulation.telemetry.status,
        "n_steps_completed": simulation.telemetry.n_steps_completed,
        "final_time_s": simulation.telemetry.final_time_s,
        "continuation_state": simulation.telemetry.continuation_state,
        "kinetic_yield_state": simulation.kinetic_yield_state,
        "particle_species": [],
        "has_electron_energy": simulation.electron_energy is not None,
        "has_ionization_charge_state": (
            simulation.ionization_charge_state is not None
        ),
        "has_circuit_state": simulation.circuit is not None,
    }
    if simulation.electron_energy is not None:
        arrays["electron_energy_J_m3"] = np.asarray(
            simulation.electron_energy.electron_energy_J_m3
        )
        arrays["electron_temperature_K"] = np.asarray(
            simulation.electron_energy.electron_temperature_K
        )
        arrays["ion_temperature_K"] = np.asarray(
            simulation.electron_energy.ion_temperature_K
        )
    if simulation.ionization_charge_state is not None:
        arrays["ionization_neutral_density_m3"] = np.asarray(
            simulation.ionization_charge_state.neutral_density_m3
        )
        arrays["ionization_ion_density_m3"] = np.asarray(
            simulation.ionization_charge_state.ion_density_m3
        )
        arrays["ionization_electron_density_m3"] = np.asarray(
            simulation.ionization_charge_state.electron_density_m3
        )
        arrays["ionization_mean_charge_state"] = np.asarray(
            simulation.ionization_charge_state.mean_charge_state
        )
    if simulation.circuit is not None:
        arrays["circuit_state"] = np.asarray(
            [simulation.circuit.current_A, simulation.circuit.charge_C],
            dtype=np.float64,
        )
    if simulation.previous_total_current_A_m2 is not None:
        arrays["previous_total_current_A_m2"] = np.asarray(
            simulation.previous_total_current_A_m2
        )
    for index, species in enumerate(simulation.pic.species):
        prefix = f"species_{index}"
        metadata["particle_species"].append({
            "index": index,
            "name": species.name,
            "mass": float(species.mass),
            "charge": float(species.charge),
            "particle_count": species.n_particles(),
        })
        arrays[f"{prefix}_positions"] = np.asarray(species.positions)
        arrays[f"{prefix}_positions_old"] = np.asarray(species.positions_old)
        arrays[f"{prefix}_velocities"] = np.asarray(species.velocities)
        arrays[f"{prefix}_weights"] = np.asarray(species.weights)
    return arrays, metadata


def _read_checkpoint_payload(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as loaded:
        metadata_value = loaded["metadata_json"]
        metadata = json.loads(str(metadata_value.item()))
        arrays = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "metadata_json"
        }
    return arrays, metadata


def _checkpoint_content_hash(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        hasher.update(key.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
