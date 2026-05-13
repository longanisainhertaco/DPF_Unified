"""Build Tier-3 checkpoint/restart reproducibility evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.validation.artifacts import stable_json_hash
from dpf.validation.mhd_numerical_fidelity import (
    restart_reproducibility_evidence_from_results,
)


_REQUIRED_OBSERVABLES = [
    "step_count",
    "time_s",
    "current_A",
    "voltage_V",
    "circuit_total_energy_J",
    "max_density_kg_m3",
    "max_pressure_Pa",
    "max_Te_eV",
    "rho_l2",
    "pressure_l2",
    "velocity_l2",
    "B_l2",
]


def _restart_fixture_config() -> SimulationConfig:
    return SimulationConfig(
        grid_shape=(4, 4, 4),
        dx=1.0e-3,
        sim_time=1.0e-7,
        dt_init=1.0e-10,
        circuit={
            "C": 1.0e-6,
            "V0": 1.0e3,
            "L0": 1.0e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": ":memory:"},
        radiation={
            "bremsstrahlung_enabled": False,
            "line_radiation_enabled": False,
            "fld_enabled": False,
        },
        collision={"enabled": False},
    )


def _run_until(engine: SimulationEngine, target_step: int) -> None:
    while engine.step_count < target_step:
        result = engine.step(_max_steps=target_step)
        if result.finished and engine.step_count < target_step:
            raise RuntimeError(
                "restart fixture finished before target step "
                f"{target_step}: step={engine.step_count}, t={engine.time:.6e}"
            )


def _observables(engine: SimulationEngine) -> dict[str, float]:
    state = engine.state
    return {
        "step_count": float(engine.step_count),
        "time_s": float(engine.time),
        "current_A": float(engine.circuit.current),
        "voltage_V": float(engine.circuit.voltage),
        "circuit_total_energy_J": float(engine.circuit.total_energy()),
        "max_density_kg_m3": float(np.max(state["rho"])),
        "max_pressure_Pa": float(np.max(state["pressure"])),
        "max_Te_eV": float(np.max(state["Te"])),
        "rho_l2": float(np.linalg.norm(state["rho"])),
        "pressure_l2": float(np.linalg.norm(state["pressure"])),
        "velocity_l2": float(np.linalg.norm(state["velocity"])),
        "B_l2": float(np.linalg.norm(state["B"])),
    }


def build_restart_reproducibility_evidence(
    *,
    scope: str,
    restart_step: int,
    total_steps: int,
    relative_tolerance: float,
) -> dict[str, object]:
    if restart_step <= 0:
        raise ValueError("restart_step must be positive")
    if total_steps <= restart_step:
        raise ValueError("total_steps must be greater than restart_step")

    config = _restart_fixture_config()
    config_payload = config.model_dump(mode="json")
    config_hash = stable_json_hash(config_payload)

    continuous_engine = SimulationEngine(config)
    _run_until(continuous_engine, total_steps)
    continuous = _observables(continuous_engine)

    checkpoint_engine = SimulationEngine(config)
    _run_until(checkpoint_engine, restart_step)

    checkpoint_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
            checkpoint_path = handle.name
        checkpoint_engine.save_checkpoint(checkpoint_path)

        restarted_engine = SimulationEngine(config)
        restarted_engine.load_from_checkpoint(checkpoint_path)
        _run_until(restarted_engine, total_steps)
        restarted = _observables(restarted_engine)
    finally:
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)

    results = {
        "verification_scope": scope,
        "continuous": continuous,
        "restarted": restarted,
        "restart_step": restart_step,
        "checkpoint_step": restart_step,
        "checkpoint_time_s": float(checkpoint_engine.time),
        "config_hash": config_hash,
        "restart_config_hash": config_hash,
        "relative_tolerances": {
            observable: relative_tolerance for observable in _REQUIRED_OBSERVABLES
        },
    }
    evidence = restart_reproducibility_evidence_from_results(
        results,
        verification_scope=scope,
        relative_tolerance=relative_tolerance,
        required_observables=_REQUIRED_OBSERVABLES,
    )
    evidence["run_metadata"] = {
        "artifact_role": "scheduled_tier3_restart_reproducibility",
        "generated_by": "scripts/build_mhd_restart_reproducibility_evidence.py",
        "fixture": "minimal_cpu_python_checkpoint_restart",
        "backend": "python",
        "grid_shape": list(config.grid_shape),
        "target_step": total_steps,
        "restart_step": restart_step,
        "config_sha256": config_hash,
        "input_result_sha256": stable_json_hash(results),
    }
    evidence["raw_results"] = {
        "continuous": continuous,
        "restarted": restarted,
    }
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an uninterrupted CPU fixture and a checkpoint/restarted fixture "
            "to produce Tier-3 restart reproducibility evidence."
        )
    )
    parser.add_argument(
        "--scope",
        default="scheduled_tier3_cpu_mhd_numerical_2026_05_09",
        help="same-scope identifier applied to generated evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mhd_restart_reproducibility_evidence.json"),
        help="restart evidence JSON output path",
    )
    parser.add_argument(
        "--restart-step",
        type=int,
        default=5,
        help="step at which the checkpoint is written and reloaded",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10,
        help="final step count for uninterrupted and restarted runs",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1.0e-12,
        help="relative tolerance applied to compared observables",
    )
    args = parser.parse_args(argv)

    evidence = build_restart_reproducibility_evidence(
        scope=args.scope,
        restart_step=args.restart_step,
        total_steps=args.total_steps,
        relative_tolerance=args.relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": evidence["passed"],
        "verification_scope": evidence["verification_scope"],
        "missing_or_failed_metrics": evidence["missing_or_failed_metrics"],
        "max_relative_error": evidence["details"]["max_relative_error"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
