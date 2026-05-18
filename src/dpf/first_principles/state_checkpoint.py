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

# Bumped whenever the checkpoint metadata schema changes in a way that makes a
# previously written checkpoint unsafe to load into the current loader.
CHECKPOINT_METADATA_SCHEMA_VERSION = "first_principles_3d_checkpoint_v2"


class CheckpointDeckMismatchError(RuntimeError):
    """Raised when a checkpoint does not match the target deck/grid.

    The message always names the mismatched dimension ('grid'/'shape'/
    'spacing'/'circuit'/'closure'/'species'/'checkpoint') so a reviewer can
    attribute the failure.  This error is raised BEFORE any checkpoint array is
    written into the target session, so a mismatch can never leave a session in
    a partially-overwritten, wrong-shape state.
    """


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
    deck: Any | None = None,
) -> dict[str, Any]:
    """Write/read a simulation state checkpoint and compare content hashes.

    ``deck`` is the resolved first-principles deck used to produce ``simulation``.
    When supplied, its grid shape/spacing, circuit mode, closure policy, and
    particle-species identity are embedded as checkpoint metadata so the loader
    can fail-closed on a mismatched restart target.
    """

    path = Path(checkpoint_path)
    arrays, metadata = _checkpoint_arrays_and_metadata(
        simulation=simulation,
        manifest=_mapping(manifest),
        deck=deck,
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
    # Fail-closed BEFORE any state array is written into the session: a
    # grid/spacing/circuit/closure/species mismatch must raise here, never
    # silently overwrite session state with wrong-shape checkpoint arrays.
    _validate_checkpoint_against_session(
        arrays=arrays,
        metadata=metadata,
        session=session,
    )
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
    deck: Any | None = None,
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
        "checkpoint_metadata_schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
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
        "deck_fingerprint": _deck_fingerprint(deck),
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
    # State-channel hashes: per-channel content fingerprints so the loader can
    # verify each array round-tripped intact before assigning it into a session.
    metadata["state_channel_shapes"] = {
        key: list(np.asarray(value).shape) for key, value in sorted(arrays.items())
    }
    metadata["state_channel_hashes"] = {
        key: _array_sha256(value) for key, value in sorted(arrays.items())
    }
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


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


# Deck fields that change the meaning of a checkpoint.  Restarting from a
# checkpoint written under a different value for any of these is not equivalent
# to one uninterrupted run, so they are fingerprinted and checked at load time.
_DECK_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "grid_shape",
    "grid_spacing_m",
    "dt_s",
    "sigma0_S_m",
    "background_density_m3",
    "density_floor_m3",
    "initial_ionization_fraction",
    "pressure_density_threshold_m3",
    "ion_species_name",
    "ion_mass_kg",
    "ion_charge_C",
    "include_hall",
    "use_predictor_corrector",
    "use_source_ordered_velocity_update",
    "marder_factor_scale",
    "marder_nondominance_threshold",
    "ohmic_cfl_safety",
    "apply_circuit_boundary",
    "circuit_capacitance_F",
    "circuit_voltage_V",
    "circuit_inductance_H",
    "circuit_resistance_ohm",
    "circuit_udpf_mode",
    "circuit_feedback_min_current_A",
    "circuit_z_index",
    "circuit_blend",
    "pml_cells",
    "pml_strength",
    "particle_absorption_enabled",
    "open_boundary",
)


def _deck_values(deck: Any | None) -> dict[str, Any]:
    """Resolve a deck (mapping/object/None) to a first-principles deck dict."""

    if deck is None:
        return {}
    from dpf.first_principles.runner import FirstPrinciples3DDeck

    resolved = FirstPrinciples3DDeck.from_deck(deck)
    values: dict[str, Any] = {}
    for name in _DECK_FINGERPRINT_FIELDS:
        value = getattr(resolved, name, None)
        if isinstance(value, (tuple, list)):
            values[name] = [
                float(item) if isinstance(item, float) else item for item in value
            ]
        else:
            values[name] = value
    return values


def _deck_fingerprint(deck: Any | None) -> dict[str, Any]:
    """Produce a checkpoint-embeddable fingerprint of the originating deck.

    ``deck_hash`` is the sha256 of the physics-relevant deck fields.  The raw
    field values are kept alongside the hash so a loader mismatch can report
    exactly which dimension diverged instead of an opaque hash difference.
    """

    values = _deck_values(deck)
    if not values:
        return {
            "available": False,
            "deck_hash": None,
            "circuit_mode": None,
            "closure_policy": None,
            "fields": {},
        }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    deck_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    closure_policy = {
        "include_hall": values.get("include_hall"),
        "use_predictor_corrector": values.get("use_predictor_corrector"),
        "use_source_ordered_velocity_update": values.get(
            "use_source_ordered_velocity_update"
        ),
        "marder_factor_scale": values.get("marder_factor_scale"),
        "density_floor_m3": values.get("density_floor_m3"),
        "pressure_density_threshold_m3": values.get("pressure_density_threshold_m3"),
    }
    return {
        "available": True,
        "deck_hash": deck_hash,
        "circuit_mode": values.get("circuit_udpf_mode"),
        "apply_circuit_boundary": values.get("apply_circuit_boundary"),
        "closure_policy": closure_policy,
        "particle_species": {
            "ion_species_name": values.get("ion_species_name"),
            "ion_mass_kg": values.get("ion_mass_kg"),
            "ion_charge_C": values.get("ion_charge_C"),
        },
        "fields": values,
    }


def _validate_checkpoint_against_session(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    session: Any,
) -> None:
    """Fail-closed checkpoint/session compatibility gate.

    Raises :class:`CheckpointDeckMismatchError` if the checkpoint's grid shape,
    grid spacing, circuit mode, closure policy, or particle species disagree
    with the freshly built target ``session``.  Called BEFORE any state array
    is assigned, so a mismatch can never partially overwrite session state.
    """

    schema = metadata.get("checkpoint_metadata_schema_version")
    if schema != CHECKPOINT_METADATA_SCHEMA_VERSION:
        raise CheckpointDeckMismatchError(
            "checkpoint metadata schema "
            f"'{schema}' does not match loader schema "
            f"'{CHECKPOINT_METADATA_SCHEMA_VERSION}'; the checkpoint predates "
            "the current loader and cannot be safely restored"
        )

    simulator = session.simulator
    grid = simulator.grid

    # 1. Grid array shapes: every checkpoint field/B array must match the
    #    freshly built session's array of the same name, exactly.
    target_arrays = {
        "E_Ex_edge": simulator.state.E.Ex_edge,
        "E_Ey_edge": simulator.state.E.Ey_edge,
        "E_Ez_edge": simulator.state.E.Ez_edge,
        "B_Bx_face": simulator.state.B.Bx_face,
        "B_By_face": simulator.state.B.By_face,
        "B_Bz_face": simulator.state.B.Bz_face,
    }
    for name, target in target_arrays.items():
        if name not in arrays:
            raise CheckpointDeckMismatchError(
                f"checkpoint is missing required state-channel array '{name}'"
            )
        checkpoint_shape = tuple(np.asarray(arrays[name]).shape)
        target_shape = tuple(np.asarray(target).shape)
        if checkpoint_shape != target_shape:
            raise CheckpointDeckMismatchError(
                f"checkpoint grid shape mismatch for channel '{name}': "
                f"checkpoint array shape {checkpoint_shape} != target session "
                f"grid array shape {target_shape}; the checkpoint deck grid "
                "does not match the restart deck grid"
            )

    # 2. Grid spacing must match (same shape, different spacing is still a
    #    physically different problem and not restart-equivalent).
    deck_fp = metadata.get("deck_fingerprint")
    if isinstance(deck_fp, Mapping) and deck_fp.get("available"):
        fields = deck_fp.get("fields")
        fields = fields if isinstance(fields, Mapping) else {}
        ckpt_shape = fields.get("grid_shape")
        if ckpt_shape is not None and tuple(ckpt_shape) != tuple(grid.shape):
            raise CheckpointDeckMismatchError(
                f"checkpoint grid shape {tuple(ckpt_shape)} != target deck "
                f"grid shape {tuple(grid.shape)}"
            )
        ckpt_spacing = fields.get("grid_spacing_m")
        if ckpt_spacing is not None:
            target_spacing = tuple(float(v) for v in grid.spacing)
            if not _spacing_close(tuple(ckpt_spacing), target_spacing):
                raise CheckpointDeckMismatchError(
                    f"checkpoint grid spacing {tuple(ckpt_spacing)} != target "
                    f"deck grid spacing {target_spacing}"
                )

        # 3. Circuit mode and 4. closure policy must match the restart deck.
        target_fp = _deck_fingerprint(session.deck)
        target_fields = target_fp.get("fields", {})
        if deck_fp.get("circuit_mode") != target_fp.get("circuit_mode"):
            raise CheckpointDeckMismatchError(
                f"checkpoint circuit mode '{deck_fp.get('circuit_mode')}' != "
                f"target deck circuit mode '{target_fp.get('circuit_mode')}'"
            )
        if deck_fp.get("apply_circuit_boundary") != target_fp.get(
            "apply_circuit_boundary"
        ):
            raise CheckpointDeckMismatchError(
                "checkpoint circuit boundary policy "
                f"'{deck_fp.get('apply_circuit_boundary')}' != target deck "
                f"circuit boundary policy '{target_fp.get('apply_circuit_boundary')}'"
            )
        if deck_fp.get("closure_policy") != target_fp.get("closure_policy"):
            raise CheckpointDeckMismatchError(
                "checkpoint closure policy does not match the restart deck "
                "closure policy (one of include_hall, predictor-corrector, "
                "source-ordered velocity update, marder scale, density floor, "
                "or pressure-density threshold differs)"
            )
        # 5. Particle species identity must match.
        if deck_fp.get("particle_species") != target_fp.get("particle_species"):
            raise CheckpointDeckMismatchError(
                "checkpoint particle species "
                f"{deck_fp.get('particle_species')} != target deck particle "
                f"species {target_fp.get('particle_species')}"
            )
        # 6. Full physics-deck hash: any remaining divergence is attributable.
        if deck_fp.get("deck_hash") != target_fp.get("deck_hash"):
            divergent = _first_divergent_field(
                deck_fp.get("fields", {}),
                target_fields,
            )
            raise CheckpointDeckMismatchError(
                "checkpoint deck hash does not match the restart deck; first "
                f"divergent physics field: {divergent}"
            )

    # 7. Particle-species count/identity check against the live session PIC.
    ckpt_species = metadata.get("particle_species")
    ckpt_species = ckpt_species if isinstance(ckpt_species, list) else []
    session_species = list(getattr(simulator.pic, "species", []))
    if len(ckpt_species) != len(session_species):
        raise CheckpointDeckMismatchError(
            f"checkpoint species count {len(ckpt_species)} != target session "
            f"species count {len(session_species)}"
        )
    for entry in ckpt_species:
        if not isinstance(entry, Mapping):
            continue
        index = int(entry["index"])
        if index >= len(session_species):
            raise CheckpointDeckMismatchError(
                "checkpoint species index "
                f"{index} exceeds target session species count "
                f"{len(session_species)}"
            )
        target_species = session_species[index]
        if str(entry.get("name")) != str(target_species.name):
            raise CheckpointDeckMismatchError(
                f"checkpoint species[{index}] name '{entry.get('name')}' != "
                f"target session species name '{target_species.name}'"
            )
        if not _scalar_close(entry.get("mass"), float(target_species.mass)):
            raise CheckpointDeckMismatchError(
                f"checkpoint species[{index}] mass {entry.get('mass')} != "
                f"target session species mass {float(target_species.mass)}"
            )
        if not _scalar_close(entry.get("charge"), float(target_species.charge)):
            raise CheckpointDeckMismatchError(
                f"checkpoint species[{index}] charge {entry.get('charge')} != "
                f"target session species charge {float(target_species.charge)}"
            )

    # 8. State-channel content hashes: confirm the npz arrays round-tripped.
    declared_hashes = metadata.get("state_channel_hashes")
    if isinstance(declared_hashes, Mapping):
        for name, expected in declared_hashes.items():
            if name not in arrays:
                raise CheckpointDeckMismatchError(
                    f"checkpoint declares state channel '{name}' but the array "
                    "is absent from the checkpoint payload"
                )
            actual = _array_sha256(arrays[name])
            if actual != expected:
                raise CheckpointDeckMismatchError(
                    f"checkpoint state-channel '{name}' content hash mismatch; "
                    "the checkpoint payload is corrupt or was modified"
                )


def _spacing_close(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(float(a) - float(b)) <= 1e-12 + 1e-9 * abs(float(b))
        for a, b in zip(left, right, strict=True)
    )


def _scalar_close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return left == right
    return abs(a - b) <= 1e-12 + 1e-9 * abs(b)


def _first_divergent_field(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> str:
    for name in _DECK_FINGERPRINT_FIELDS:
        lv = left.get(name)
        rv = right.get(name)
        if isinstance(lv, float) or isinstance(rv, float):
            if not _scalar_close(lv, rv):
                return f"{name} ({lv} != {rv})"
        elif lv != rv:
            return f"{name} ({lv} != {rv})"
    return "unknown_field"
