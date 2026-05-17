"""Command-line interface for the DPF simulator.

Usage:
    dpf simulate config.json --steps=100
    dpf verify config.json
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import click

FIRST_PRINCIPLES_3D_DECK_PRESETS = (
    "pf1000_akel_16kv",
    "ir_mpf_100",
    "compact_chinese_dpf",
    "willenborg_hendricks",
    "gv_pf24_krakow_16092202",
)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """DPF Unified — Dense Plasma Focus simulator."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--steps", type=int, default=None, help="Max timesteps (default: run to sim_time).")
@click.option("--output", "-o", type=str, default=None, help="Override output HDF5 filename.")
@click.option("--restart", type=click.Path(exists=True), default=None, help="Restart from checkpoint.")
@click.option("--checkpoint-interval", type=int, default=0, help="Auto-checkpoint every N steps (0=off).")
@click.option(
    "--backend",
    type=click.Choice(
        ["python", "athena", "athenak", "metal", "mlx", "hybrid", "auto"],
        case_sensitive=False,
    ),
    default=None,
    help="MHD solver backend. Overrides config file setting. "
    "'python'=NumPy/Numba, 'athena'=Athena++ C++, 'athenak'=AthenaK Kokkos, "
    "'metal'=PyTorch MPS, 'mlx'=MLX Metal v2, 'hybrid'=Athena/WALRUS hybrid, "
    "'auto'=best available.",
)
@click.option(
    "--run-mode",
    type=click.Choice(["first_principles_mhd"], case_sensitive=False),
    default=None,
    help=(
        "Optional public run-mode authority label. first_principles_mhd "
        "keeps the selected backend but adds fail-closed PF-1000/Akel readiness."
    ),
)
@click.option(
    "--validation-scope",
    type=str,
    default=None,
    help="Optional same-scope validation target label for run-mode readiness.",
)
@click.option(
    "--source-scope",
    type=str,
    default=None,
    help="Optional local source-scope label for run-mode readiness.",
)
@click.option(
    "--source-scope-status",
    type=str,
    default=None,
    help="Optional source-scope status such as same_scope_blocked_by_review.",
)
@click.option(
    "--preset-name",
    type=str,
    default=None,
    help="Optional preset key recorded in run-mode authority metadata.",
)
def simulate(
    config_file: str,
    steps: int | None,
    output: str | None,
    restart: str | None,
    checkpoint_interval: int,
    backend: str | None,
    run_mode: str | None,
    validation_scope: str | None,
    source_scope: str | None,
    source_scope_status: str | None,
    preset_name: str | None,
) -> None:
    """Run a DPF simulation from a configuration file."""
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    click.echo(f"Loading config from {config_file}")
    config = SimulationConfig.from_file(config_file)

    if run_mode and run_mode.lower() == "first_principles_mhd":
        payload = _first_principles_package_native_payload(
            preset=preset_name or "pf1000_akel",
            grid_preset="coarse",
            sim_time_us=float(config.sim_time) * 1.0e6,
            history_stride=1,
            steps=steps or 2,
        )
        if output:
            import json

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = {
            "run_mode": payload["run_mode"],
            "execution_backend": payload["execution_backend"],
            "command_status": payload["command_status"],
            "scientific_status": payload["scientific_status"],
            "requested_sim_time_us": payload["requested_sim_time_us"],
            "simulated_time_us": payload["simulated_time_us"],
            "duration_gate_status": payload["duration_gate_status"],
            "validation_packet_status": payload["validation_packet"]["status"],
            "same_scope_source_status": payload["validation_packet"][
                "same_scope_source_status"
            ],
            "can_support_first_principles_acceptance": payload[
                "can_support_first_principles_acceptance"
            ],
        }
        click.echo("Backend: package_native")
        click.echo("\n--- Simulation Summary ---")
        for key, val in summary.items():
            click.echo(f"  {key}: {val}")
        return

    if backend:
        config.fluid.backend = backend
    if run_mode:
        config.run_mode = run_mode
    if validation_scope:
        config.validation_scope = validation_scope
    if source_scope:
        config.source_scope = source_scope
    if source_scope_status:
        config.source_scope_status = source_scope_status
    if preset_name:
        config.preset_name = preset_name

    if output:
        config.diagnostics.hdf5_filename = output

    engine = SimulationEngine(config)
    click.echo(f"Backend: {engine.backend}")

    if checkpoint_interval > 0:
        engine.checkpoint_interval = checkpoint_interval

    if restart:
        click.echo(f"Restarting from checkpoint: {restart}")
        engine.load_from_checkpoint(restart)

    summary = engine.run(max_steps=steps)

    click.echo("\n--- Simulation Summary ---")
    for key, val in summary.items():
        if isinstance(val, float):
            click.echo(f"  {key}: {val:.6e}")
        else:
            click.echo(f"  {key}: {val}")


def _parse_grid_shape(value: str) -> tuple[int, int, int]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise click.BadParameter("shape must be formatted as NX,NY,NZ")
    try:
        shape = tuple(int(item) for item in parts)
    except ValueError as exc:
        raise click.BadParameter("shape entries must be integers") from exc
    if min(shape) < 3:
        raise click.BadParameter("all shape entries must be >= 3")
    return shape  # type: ignore[return-value]


def _hybrid_3d_smoke_payload(
    *,
    steps: int,
    shape: tuple[int, int, int],
    dt_s: float,
    tool: str = "dpf hybrid-3d-smoke",
    deck_name: str = "built_in_hybrid_3d_smoke",
    initial_ex_V_m: float = 1.0e5,
    sigma0_S_m: float = 1.0e2,
    background_density_m3: float = 1.0e20,
    electron_density_m3: float = 1.0e20,
    electron_temperature_K: float = 1.0e5,
    ion_temperature_K: float = 1.0e5,
    particle_weight: float = 1.0e8,
    include_hall: bool = False,
    use_predictor_corrector: bool = True,
    apply_circuit_boundary: bool = True,
) -> dict[str, Any]:
    """Run a compact fail-closed 3-D hybrid PIC-fluid engineering smoke."""
    if steps <= 0:
        raise click.BadParameter("must be positive", param_hint="--steps")
    if dt_s <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--dt-s")
    if electron_density_m3 <= 0.0:
        raise click.BadParameter("must be positive", param_hint="electron_density_m3")
    if background_density_m3 <= 0.0:
        raise click.BadParameter("must be positive", param_hint="background_density_m3")
    if particle_weight <= 0.0:
        raise click.BadParameter("must be positive", param_hint="particle_weight")

    import numpy as np

    from dpf.constants import e as elementary_charge
    from dpf.experimental.pic.hybrid import HybridPIC
    from dpf.fields import (
        CircuitMagneticBoundaryDrive,
        CircuitState,
        ElectronEnergyClosure,
        HybridPIC3DLoop,
        HybridPIC3DSimulator,
        HybridPICSourceGeometry,
        KineticIonYieldHistory,
    )
    from dpf.validation import (
        candidate_packet_from_source_geometry,
        evaluate_hybrid_pic_3d_validation_packet,
    )

    deuteron_mass_kg = 3.344e-27
    geometry = HybridPICSourceGeometry()
    grid = geometry.smoke_grid(shape=shape)
    center = np.array(
        [
            0.5 * grid.nx * grid.dx,
            0.5 * grid.ny * grid.dy,
            0.5 * grid.nz * grid.dz,
        ],
        dtype=float,
    )
    offsets = np.array(
        [
            [-0.25 * grid.dx, 0.0, 0.0],
            [0.25 * grid.dx, 0.0, 0.0],
            [0.0, -0.25 * grid.dy, 0.0],
            [0.0, 0.25 * grid.dy, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [8.0e5, 0.0, 0.0],
            [-8.0e5, 0.0, 0.0],
            [0.0, 8.0e5, 0.0],
            [0.0, -8.0e5, 0.0],
        ],
        dtype=float,
    )
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=dt_s,
        use_esirkepov=True,
        use_binary_collisions=False,
    )
    pic.add_species(
        "d",
        deuteron_mass_kg,
        elementary_charge,
        positions=center[np.newaxis, :] + offsets,
        velocities=velocities,
        weights=np.full(4, float(particle_weight)),
    )

    electron_closure = ElectronEnergyClosure(grid)
    electron_density = np.full(grid.shape, float(electron_density_m3))
    electron_state = electron_closure.initialize(
        electron_temperature_K=electron_temperature_K,
        ion_temperature_K=ion_temperature_K,
        electron_density_m3=electron_density,
    )
    loop = HybridPIC3DLoop(
        grid,
        electron_energy_closure=electron_closure,
        kinetic_yield_history=KineticIonYieldHistory(grid),
    )
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(float(initial_ex_V_m))
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=state,
        pic=pic,
        circuit_boundary=(
            CircuitMagneticBoundaryDrive(grid) if apply_circuit_boundary else None
        ),
    )
    result = simulator.run(
        n_steps=steps,
        dt_s=dt_s,
        sigma0_S_m=sigma0_S_m,
        background_density_m3=background_density_m3,
        ohmic_cfl_safety=1.0,
        density_floor_m3=electron_density_m3,
        include_hall=include_hall,
        use_predictor_corrector=use_predictor_corrector,
        marder_factor_m2=1.0e-6 * min(grid.spacing) ** 2,
        marder_nondominance_threshold=0.5,
        electron_energy_state=electron_state,
        mass_density_kg_m3=electron_density * deuteron_mass_kg,
        plasma_velocity_m_s=np.zeros(grid.shape + (3,)),
        electron_temperature_floor_K=10.0,
        use_source_ordered_velocity_update=True,
        circuit_state=CircuitState() if apply_circuit_boundary else None,
        apply_circuit_boundary=apply_circuit_boundary,
    )
    validation_packet = candidate_packet_from_source_geometry(geometry)
    validation_status = evaluate_hybrid_pic_3d_validation_packet(validation_packet)
    telemetry = result.telemetry.to_dict()
    return {
        "tool": tool,
        "runner": "dpf.fields.HybridPIC3DSimulator",
        "deck": {
            "name": deck_name,
            "steps": int(steps),
            "grid_shape": list(grid.shape),
            "dt_s": float(dt_s),
            "initial_ex_V_m": float(initial_ex_V_m),
            "sigma0_S_m": float(sigma0_S_m),
            "background_density_m3": float(background_density_m3),
            "electron_density_m3": float(electron_density_m3),
            "electron_temperature_K": float(electron_temperature_K),
            "ion_temperature_K": float(ion_temperature_K),
            "particle_weight": float(particle_weight),
            "include_hall": bool(include_hall),
            "use_predictor_corrector": bool(use_predictor_corrector),
            "apply_circuit_boundary": bool(apply_circuit_boundary),
        },
        "scientific_status": "engineering_candidate_not_validation",
        "source": geometry.source,
        "source_scope": geometry.source_scope,
        "grid_shape": list(grid.shape),
        "dt_s": dt_s,
        "n_steps": steps,
        "simulation": telemetry,
        "validation_packet": validation_status,
        "final_particle_count": telemetry["n_particles_final"],
        "final_field_energy_J": telemetry["final_field_energy_J"],
    }


def _default_first_principles_3d_deck() -> dict[str, Any]:
    from dpf.first_principles import pf1000_akel_16kv_engineering_deck

    return pf1000_akel_16kv_engineering_deck(n_steps=2).to_dict()


def _first_principles_3d_deck_preset(deck_preset: str) -> dict[str, Any]:
    from dpf.first_principles import (
        compact_chinese_dpf_engineering_deck,
        gv_verified_engineering_deck,
        ir_mpf_100_engineering_deck,
        pf1000_akel_16kv_engineering_deck,
        willenborg_hendricks_engineering_deck,
    )

    normalized = deck_preset.lower()
    builders = {
        "pf1000_akel_16kv": pf1000_akel_16kv_engineering_deck,
        "ir_mpf_100": ir_mpf_100_engineering_deck,
        "compact_chinese_dpf": compact_chinese_dpf_engineering_deck,
        "willenborg_hendricks": willenborg_hendricks_engineering_deck,
        "gv_pf24_krakow_16092202": (
            lambda *, n_steps: gv_verified_engineering_deck(
                "pf24_krakow_16092202",
                n_steps=n_steps,
            )
        ),
    }
    try:
        deck = builders[normalized](n_steps=2).to_dict()
    except KeyError as exc:
        allowed = ", ".join(FIRST_PRINCIPLES_3D_DECK_PRESETS)
        raise click.BadParameter(
            f"deck preset must be one of {allowed}",
            param_hint="--deck-preset",
        ) from exc

    deck["source"] = (
        "built_in"
        if normalized == "pf1000_akel_16kv"
        else f"built_in:{normalized}"
    )
    return deck


def _default_compact_first_principles_3d_deck() -> dict[str, Any]:
    return {
        "name": "minimal_engineering_3d",
        "steps": 2,
        "grid_shape": [5, 5, 5],
        "dt_s": 1.0e-13,
        "initial_ex_V_m": 1.0e5,
        "sigma0_S_m": 1.0e2,
        "background_density_m3": 1.0e20,
        "electron_density_m3": 1.0e20,
        "electron_temperature_K": 1.0e5,
        "ion_temperature_K": 1.0e5,
        "particle_weight": 1.0e8,
        "include_hall": False,
        "use_predictor_corrector": True,
        "apply_circuit_boundary": True,
    }


def _first_principles_grid_shape(grid_preset: str) -> tuple[int, int, int]:
    shapes = {
        "coarse": (5, 5, 5),
        "medium": (7, 7, 7),
        "fine": (9, 9, 9),
    }
    try:
        return shapes[grid_preset.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(shapes))
        raise click.BadParameter(f"grid preset must be one of {allowed}") from exc


def _load_first_principles_3d_deck(
    deck_path: Path | None,
    *,
    deck_preset: str = "pf1000_akel_16kv",
) -> dict[str, Any]:
    if deck_path is None:
        return _first_principles_3d_deck_preset(deck_preset)

    import json

    try:
        raw = json.loads(deck_path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"invalid first-principles 3-D deck JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise click.ClickException("first-principles 3-D deck must be a JSON object")

    if {"device", "circuit", "grid"}.issubset(raw.keys()):
        deck = dict(raw)
        deck["source"] = str(deck_path)
        return deck

    deck = _default_compact_first_principles_3d_deck()
    deck.update(raw)
    deck["source"] = str(deck_path)
    return deck


def _override_first_principles_3d_deck_runtime(
    deck: dict[str, Any],
    *,
    steps: int | None = None,
    dt_s: float | None = None,
    history_stride: int | None = None,
    max_step_results: int | None = None,
    target_time_s: float | None = None,
) -> dict[str, Any]:
    """Apply explicit CLI runtime overrides without changing source metadata."""

    if steps is not None and steps <= 0:
        raise click.BadParameter("must be positive", param_hint="--steps")
    if dt_s is not None and dt_s <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--dt-s")
    if history_stride is not None and history_stride <= 0:
        raise click.BadParameter("must be positive", param_hint="--history-stride")
    if max_step_results is not None and max_step_results < 0:
        raise click.BadParameter(
            "must be non-negative",
            param_hint="--max-step-results",
        )
    if target_time_s is not None and target_time_s <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--target-time-s")

    overridden = dict(deck)
    if {"device", "circuit", "grid"}.issubset(overridden.keys()):
        diagnostics = dict(overridden.get("diagnostics", {}))
        if steps is not None:
            diagnostics["n_steps"] = int(steps)
        if dt_s is not None:
            diagnostics["dt_s"] = float(dt_s)
        if history_stride is not None:
            diagnostics["history_stride"] = int(history_stride)
        if max_step_results is not None:
            diagnostics["max_step_results"] = int(max_step_results)
        if target_time_s is not None:
            diagnostics["target_time_s"] = float(target_time_s)
        overridden["diagnostics"] = diagnostics
        return overridden

    if steps is not None:
        overridden["steps"] = int(steps)
    if dt_s is not None:
        overridden["dt_s"] = float(dt_s)
    if history_stride is not None:
        overridden["history_stride"] = int(history_stride)
    if max_step_results is not None:
        overridden["max_step_results"] = int(max_step_results)
    if target_time_s is not None:
        overridden["target_time_s"] = float(target_time_s)
    return overridden


def _apply_experimental_dt_policy(
    deck: dict[str, Any],
    *,
    dt_policy: str,
    vacuum_cfl: float,
) -> dict[str, Any]:
    if vacuum_cfl <= 0.0 or vacuum_cfl > 1.0:
        raise click.BadParameter(
            "must satisfy 0 < vacuum CFL <= 1",
            param_hint="--vacuum-cfl",
        )
    if dt_policy == "deck":
        return deck
    if dt_policy not in {"vacuum-cfl", "ohmic-cfl", "combined-cfl"}:
        raise click.BadParameter("unknown dt policy", param_hint="--dt-policy")

    from dpf.first_principles.experimental_shot import stable_vacuum_cfl_dt_s

    candidate_dt_values: list[float] = []
    if dt_policy in {"vacuum-cfl", "combined-cfl"}:
        candidate_dt_values.append(
            stable_vacuum_cfl_dt_s(
                _experimental_deck_grid_spacing(deck),
                cfl=vacuum_cfl,
            )
        )
    if dt_policy in {"ohmic-cfl", "combined-cfl"}:
        candidate_dt_values.append(_experimental_deck_initial_ohmic_cfl_dt_s(deck))
    dt_s = min(candidate_dt_values)
    return _set_first_principles_3d_deck_runtime_value(
        deck,
        section="diagnostics",
        key="dt_s",
        value=dt_s,
        compact_key="dt_s",
    )


def _apply_experimental_auto_step_budget(
    deck: dict[str, Any],
    *,
    enabled: bool,
    max_auto_steps: int,
) -> dict[str, Any]:
    if not enabled:
        return deck
    if max_auto_steps <= 0:
        raise click.BadParameter(
            "must be positive",
            param_hint="--max-auto-steps",
        )
    target_time_s = _experimental_deck_target_time_s(deck)
    dt_s = _experimental_deck_dt_s(deck)
    required_steps = _ceil_division_time_steps(
        target_time_s=target_time_s,
        dt_s=dt_s,
    )
    if required_steps > max_auto_steps:
        raise click.ClickException(
            "auto step budget would require "
            f"{required_steps} steps, above --max-auto-steps={max_auto_steps}. "
            "Increase the cap deliberately or reduce target time/dt."
        )
    return _set_first_principles_3d_deck_runtime_value(
        deck,
        section="diagnostics",
        key="n_steps",
        value=required_steps,
        compact_key="steps",
    )


def _set_first_principles_3d_deck_runtime_value(
    deck: dict[str, Any],
    *,
    section: str,
    key: str,
    value: float | int,
    compact_key: str,
) -> dict[str, Any]:
    updated = dict(deck)
    if {"device", "circuit", "grid"}.issubset(updated.keys()):
        section_values = dict(updated.get(section, {}))
        section_values[key] = value
        updated[section] = section_values
        return updated
    updated[compact_key] = value
    return updated


def _experimental_deck_grid_spacing(deck: dict[str, Any]) -> tuple[float, float, float]:
    if {"device", "circuit", "grid"}.issubset(deck.keys()):
        grid = deck.get("grid", {})
        if isinstance(grid, dict) and "spacing_m" in grid:
            spacing = grid["spacing_m"]
        else:
            spacing = None
    else:
        spacing = deck.get("grid_spacing_m", deck.get("spacing"))
    if spacing is None:
        spacing = (1.0e-3, 1.0e-3, 1.0e-3)
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise click.ClickException("experimental deck grid spacing must have 3 entries")
    values = tuple(float(item) for item in spacing)
    if any(value <= 0.0 for value in values):
        raise click.ClickException("experimental deck grid spacing must be positive")
    return values  # type: ignore[return-value]


def _experimental_deck_dt_s(deck: dict[str, Any]) -> float:
    if {"device", "circuit", "grid"}.issubset(deck.keys()):
        diagnostics = deck.get("diagnostics", {})
        value = diagnostics.get("dt_s") if isinstance(diagnostics, dict) else None
    else:
        value = deck.get("dt_s")
    if value is None:
        raise click.ClickException("experimental deck is missing dt_s")
    dt_s = float(value)
    if dt_s <= 0.0:
        raise click.ClickException("experimental deck dt_s must be positive")
    return dt_s


def _experimental_deck_target_time_s(deck: dict[str, Any]) -> float:
    if {"device", "circuit", "grid"}.issubset(deck.keys()):
        diagnostics = deck.get("diagnostics", {})
        value = (
            diagnostics.get("target_time_s")
            if isinstance(diagnostics, dict)
            else None
        )
    else:
        value = deck.get("target_time_s")
    if value is None:
        raise click.ClickException(
            "--auto-step-budget requires --target-time-s"
        )
    target_time_s = float(value)
    if target_time_s <= 0.0:
        raise click.ClickException("target_time_s must be positive")
    return target_time_s


def _experimental_deck_initial_ohmic_cfl_dt_s(deck: dict[str, Any]) -> float:
    import numpy as np

    from dpf.fields.conductivity import partial_ionized_conductivity
    from dpf.first_principles.experimental_shot import stable_ohmic_cfl_dt_s
    from dpf.first_principles.runner import FirstPrinciples3DDeck

    resolved = FirstPrinciples3DDeck.from_deck(deck)
    electron_density = max(
        float(resolved.background_density_m3)
        * float(resolved.initial_ionization_fraction),
        1.0,
    )
    neutral_density = max(
        float(resolved.background_density_m3) - electron_density,
        0.0,
    )
    sigma, _ = partial_ionized_conductivity(
        electron_density_m3=np.asarray([electron_density], dtype=float),
        neutral_density_m3=np.asarray([neutral_density], dtype=float),
        electron_temperature_K=np.asarray(
            [float(resolved.electron_temperature_K)],
            dtype=float,
        ),
    )
    sigma_max = max(float(np.max(sigma)), float(resolved.sigma0_S_m), 1.0e-300)
    return stable_ohmic_cfl_dt_s(
        sigma_max,
        ohmic_cfl_safety=float(resolved.ohmic_cfl_safety),
        cfl=0.95,
    )


def _ceil_division_time_steps(*, target_time_s: float, dt_s: float) -> int:
    import math

    return int(math.ceil(float(target_time_s) / float(dt_s)))


def _parse_positive_float_tuple(
    value: str,
    *,
    param_hint: str,
    min_count: int = 2,
) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise click.BadParameter(
            "must be a comma-separated list of positive numbers",
            param_hint=param_hint,
        ) from exc
    if len(values) < min_count or any(item <= 0.0 for item in values):
        raise click.BadParameter(
            f"must contain at least {min_count} positive number(s)",
            param_hint=param_hint,
        )
    return values


def _parse_calibration_parameter_tuple(value: str) -> tuple[str, ...]:
    allowed = {"inductance", "resistance", "voltage", "pressure"}
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not values:
        raise click.BadParameter(
            "must contain at least one parameter",
            param_hint="--parameters",
        )
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise click.BadParameter(
            "unknown calibration parameter(s): " + ", ".join(unknown),
            param_hint="--parameters",
        )
    return values


def _parse_calibration_parameter_scales(
    specs: tuple[str, ...],
    *,
    parameter_names: tuple[str, ...],
    default_scales: tuple[float, ...],
) -> dict[str, tuple[float, ...]] | None:
    if not specs:
        return None
    allowed = set(parameter_names)
    parameter_scales = {name: default_scales for name in parameter_names}
    for spec in specs:
        if "=" not in spec:
            raise click.BadParameter(
                "must use name=scale,scale form",
                param_hint="--parameter-scale",
            )
        name, scale_text = spec.split("=", 1)
        normalized = name.strip().lower()
        if normalized not in allowed:
            raise click.BadParameter(
                f"{normalized} is not listed in --parameters",
                param_hint="--parameter-scale",
            )
        parameter_scales[normalized] = _parse_positive_float_tuple(
            scale_text,
            param_hint="--parameter-scale",
            min_count=1,
        )
    return parameter_scales


def _parse_positive_int_tuple(value: str, *, param_hint: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise click.BadParameter(
            "must be a comma-separated list of positive integers",
            param_hint=param_hint,
        ) from exc
    if len(values) < 1 or any(item <= 0 for item in values):
        raise click.BadParameter(
            "must contain at least one positive integer",
            param_hint=param_hint,
        )
    return values


def _parse_shape_family(value: str, *, param_hint: str) -> tuple[tuple[int, int, int], ...]:
    try:
        shapes = tuple(_parse_grid_shape(item.strip()) for item in value.split(";") if item.strip())
    except click.BadParameter as exc:
        raise click.BadParameter(
            "must be semicolon-separated NX,NY,NZ shapes",
            param_hint=param_hint,
        ) from exc
    if len(shapes) < 2:
        raise click.BadParameter(
            "must contain at least two shapes",
            param_hint=param_hint,
        )
    return shapes


def _set_experimental_deck_grid_shape_keep_extent(
    deck: dict[str, Any],
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    updated = dict(deck)
    if {"device", "circuit", "grid"}.issubset(updated.keys()):
        grid = dict(updated.get("grid", {}))
        old_shape = tuple(int(item) for item in grid.get("shape", shape))
        old_spacing = tuple(float(item) for item in grid.get("spacing_m", (1.0e-3,) * 3))
        grid["shape"] = list(shape)
        grid["spacing_m"] = list(
            _rescale_spacing_keep_extent(
                old_shape=old_shape,
                old_spacing=old_spacing,
                new_shape=shape,
            )
        )
        updated["grid"] = grid
        return updated

    old_shape = _deck_grid_shape(updated)
    old_spacing = _experimental_deck_grid_spacing(updated)
    updated["grid_shape"] = list(shape)
    updated["grid_spacing_m"] = list(
        _rescale_spacing_keep_extent(
            old_shape=old_shape,
            old_spacing=old_spacing,
            new_shape=shape,
        )
    )
    return updated


def _rescale_spacing_keep_extent(
    *,
    old_shape: tuple[int, int, int],
    old_spacing: tuple[float, float, float],
    new_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    return tuple(
        float(old_spacing[axis]) * max(int(old_shape[axis]) - 1, 1)
        / max(int(new_shape[axis]) - 1, 1)
        for axis in range(3)
    )  # type: ignore[return-value]


def _positive_int_deck_value(deck: dict[str, Any], key: str) -> int:
    try:
        value = int(deck[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise click.BadParameter(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise click.BadParameter(f"{key} must be positive")
    return value


def _positive_float_deck_value(deck: dict[str, Any], key: str) -> float:
    try:
        value = float(deck[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise click.BadParameter(f"{key} must be a positive number") from exc
    if value <= 0.0:
        raise click.BadParameter(f"{key} must be positive")
    return value


def _bool_deck_value(deck: dict[str, Any], key: str) -> bool:
    value = deck[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise click.BadParameter(f"{key} must be a boolean")


def _nonnegative_int_deck_value(deck: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = int(deck.get(key, default))
    except (TypeError, ValueError) as exc:
        raise click.BadParameter(f"{key} must be a non-negative integer") from exc
    if value < 0:
        raise click.BadParameter(f"{key} must be non-negative")
    return value


def _nonnegative_float_deck_value(
    deck: dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    try:
        value = float(deck.get(key, default))
    except (TypeError, ValueError) as exc:
        raise click.BadParameter(f"{key} must be a non-negative number") from exc
    if value < 0.0:
        raise click.BadParameter(f"{key} must be non-negative")
    return value


def _deck_grid_shape(deck: dict[str, Any]) -> tuple[int, int, int]:
    value = deck.get("grid_shape", deck.get("shape"))
    if isinstance(value, str):
        return _parse_grid_shape(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise click.BadParameter("grid_shape must be [NX, NY, NZ] or NX,NY,NZ")
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise click.BadParameter("grid_shape entries must be integers") from exc
    if min(shape) < 3:
        raise click.BadParameter("all grid_shape entries must be >= 3")
    return shape  # type: ignore[return-value]


def _first_principles_3d_payload(deck: dict[str, Any]) -> dict[str, Any]:
    from dpf.first_principles import FirstPrinciplesInputDeck
    from dpf.first_principles.runner import run_first_principles_3d_deck

    if {"device", "circuit", "grid"}.issubset(deck.keys()):
        package_deck = FirstPrinciplesInputDeck.from_mapping(deck)
        steps = package_deck.diagnostics.n_steps
        shape = package_deck.grid.shape
        dt_s = package_deck.diagnostics.dt_s
        history_stride = package_deck.diagnostics.history_stride
        max_step_results = package_deck.diagnostics.max_step_results
        target_time_s = package_deck.diagnostics.target_time_s
        apply_circuit_boundary = package_deck.closures.apply_circuit_boundary
        boundary_policy = {
            "pml_cells": package_deck.boundaries.pml_cells,
            "pml_strength": package_deck.boundaries.pml_strength,
            "particle_absorption_enabled": (
                package_deck.boundaries.particle_absorption_enabled
            ),
            "open_boundary": package_deck.boundaries.open_boundary,
            "conductor_mask_status": package_deck.boundaries.conductor_mask_status,
            "conductor_mask_mode": package_deck.boundaries.conductor_mask_mode,
        }
        deck_summary = {
            "name": package_deck.deck_id,
            "source": deck.get("source", "built_in"),
            "steps": steps,
            "grid_shape": list(shape),
            "dt_s": dt_s,
            "history_stride": history_stride,
            "max_step_results": max_step_results,
            "target_time_s": target_time_s,
            "apply_circuit_boundary": apply_circuit_boundary,
            "circuit_udpf_mode": package_deck.closures.circuit_udpf_mode,
            "boundary_policy": boundary_policy,
            "device_name": package_deck.device.name,
            "scientific_status": package_deck.scientific_status,
        }
        run = run_first_principles_3d_deck(package_deck)
    else:
        shape = _deck_grid_shape(deck)
        steps = _positive_int_deck_value(deck, "steps")
        dt_s = _positive_float_deck_value(deck, "dt_s")
        history_stride = _positive_int_deck_value(
            deck,
            "history_stride",
        ) if "history_stride" in deck else 1
        max_step_results = (
            None
            if deck.get("max_step_results", 256) is None
            else _nonnegative_int_deck_value(deck, "max_step_results", 256)
        )
        target_time_s = (
            None
            if deck.get("target_time_s") is None
            else _positive_float_deck_value(deck, "target_time_s")
        )
        apply_circuit_boundary = _bool_deck_value(deck, "apply_circuit_boundary")
        circuit_udpf_mode = str(
            deck.get("circuit_udpf_mode", "lagged_volume_j_dot_e")
        )
        boundary_policy = {
            "pml_cells": _nonnegative_int_deck_value(deck, "pml_cells"),
            "pml_strength": _nonnegative_float_deck_value(deck, "pml_strength"),
            "particle_absorption_enabled": (
                _bool_deck_value(deck, "particle_absorption_enabled")
                if "particle_absorption_enabled" in deck
                else False
            ),
            "open_boundary": (
                _bool_deck_value(deck, "open_boundary")
                if "open_boundary" in deck
                else True
            ),
            "conductor_mask_status": "not_supplied",
        }
        deck_summary = {
            "name": str(deck.get("name", "minimal_engineering_3d")),
            "source": deck.get("source", "built_in"),
            "steps": steps,
            "grid_shape": list(shape),
            "dt_s": dt_s,
            "history_stride": history_stride,
            "max_step_results": max_step_results,
            "target_time_s": target_time_s,
            "apply_circuit_boundary": apply_circuit_boundary,
            "circuit_udpf_mode": circuit_udpf_mode,
            "boundary_policy": boundary_policy,
        }
        run = run_first_principles_3d_deck(
            {
                "n_steps": steps,
                "history_stride": history_stride,
                "max_step_results": max_step_results,
                "target_time_s": target_time_s,
                "grid_shape": shape,
                "dt_s": dt_s,
                "sigma0_S_m": _positive_float_deck_value(deck, "sigma0_S_m"),
                "background_density_m3": _positive_float_deck_value(
                    deck, "background_density_m3"
                ),
                "density_floor_m3": _positive_float_deck_value(
                    deck, "background_density_m3"
                ),
                "electron_temperature_K": _positive_float_deck_value(
                    deck, "electron_temperature_K"
                ),
                "ion_temperature_K": _positive_float_deck_value(
                    deck, "ion_temperature_K"
                ),
                "particle_weight": _positive_float_deck_value(deck, "particle_weight"),
                "initial_E_x_V_m": float(deck.get("initial_ex_V_m", 1.0e5)),
                "include_hall": _bool_deck_value(deck, "include_hall"),
                "use_predictor_corrector": _bool_deck_value(
                    deck, "use_predictor_corrector"
                ),
                "apply_circuit_boundary": apply_circuit_boundary,
                "circuit_udpf_mode": circuit_udpf_mode,
                "pml_cells": boundary_policy["pml_cells"],
                "pml_strength": boundary_policy["pml_strength"],
                "particle_absorption_enabled": boundary_policy[
                    "particle_absorption_enabled"
                ],
                "open_boundary": boundary_policy["open_boundary"],
            }
        )
    simulation = run.telemetry["simulation"]
    payload = {
        "tool": "dpf first-principles-3d",
        "runner": "dpf.fields.HybridPIC3DSimulator",
        "command_status": "package_native_first_principles_3d_engineering_run",
        "deck": deck_summary,
        "scientific_status": run.status,
        "source": run.telemetry["source"],
        "source_scope": run.telemetry["source_scope"],
        "grid_shape": run.telemetry["grid_shape"],
        "dt_s": dt_s,
        "n_steps": steps,
        "n_steps_completed": simulation["n_steps_completed"],
        "history_stride": simulation["history_stride"],
        "max_step_results": simulation["max_step_results"],
        "target_time_s": simulation["target_time_s"],
        "duration_request_satisfied": simulation["duration_request_satisfied"],
        "termination_reason": simulation["termination_reason"],
        "simulation": simulation,
        "boundary_policy": run.telemetry["boundary_policy"],
        "validation_packet": run.validation_packet,
        "engineering_current_waveform_comparison": run.telemetry[
            "engineering_current_waveform_comparison"
        ],
        "experimental_whole_shot": run.telemetry["experimental_whole_shot"],
        "experimental_numerics": run.telemetry["experimental_numerics"],
        "telemetry_packets": _first_principles_telemetry_packets(run.telemetry),
        "manifest": run.manifest,
        "conservation_telemetry": run.conservation_telemetry,
        "final_particle_count": simulation["n_particles_final"],
        "final_field_energy_J": simulation["final_field_energy_J"],
        "reduced_models_used": run.reduced_models_used,
        "can_support_first_principles_acceptance": (
            run.can_support_first_principles_acceptance
        ),
    }
    payload["command_status"] = "package_native_first_principles_3d_engineering_run"
    payload["engineering_firm_dossier"] = _engineering_firm_dossier(payload)
    return payload


def _experimental_whole_shot_payload(deck: dict[str, Any]) -> dict[str, Any]:
    payload = _first_principles_3d_payload(deck)
    packet = payload["telemetry_packets"]["experimental_whole_shot"]
    payload["tool"] = "dpf experimental-whole-shot"
    payload["command_status"] = "experimental_whole_shot_engineering_candidate_run"
    payload["experimental_whole_shot"] = packet
    payload["engineering_firm_dossier"]["experimental_whole_shot_status"] = (
        packet["status"]
    )
    payload["engineering_firm_dossier"]["experimental_candidate_module_count"] = (
        packet["candidate_module_count"]
    )
    return payload


def _experimental_machine_shot_family_payload(
    *,
    scope: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    include_gv_waveforms: bool,
) -> dict[str, Any]:
    from dpf.first_principles import build_experimental_inverse_parameter_packet

    decks = _experimental_machine_family_decks(scope)
    cases: list[dict[str, Any]] = []
    for case_label, deck_dict in decks:
        try:
            runtime_deck = _override_first_principles_3d_deck_runtime(
                deck_dict,
                steps=steps,
                dt_s=dt_s,
                history_stride=history_stride,
                max_step_results=max_step_results,
                target_time_s=target_time_s,
            )
            runtime_deck = _apply_experimental_dt_policy(
                runtime_deck,
                dt_policy=dt_policy,
                vacuum_cfl=vacuum_cfl,
            )
            runtime_deck = _apply_experimental_auto_step_budget(
                runtime_deck,
                enabled=auto_step_budget,
                max_auto_steps=max_auto_steps,
            )
            case_payload = _experimental_whole_shot_payload(runtime_deck)
            case_payload["case_label"] = case_label
            case_payload["case_status"] = "completed_engineering_candidate_run"
            cases.append(case_payload)
        except click.ClickException as exc:
            cases.append(
                {
                    "case_label": case_label,
                    "case_status": "blocked_before_run",
                    "blocking_reason": str(exc),
                    "target_time_s": target_time_s,
                    "dt_policy": dt_policy,
                    "auto_step_budget": auto_step_budget,
                    "max_auto_steps": max_auto_steps,
                    "can_support_first_principles_acceptance": False,
                }
            )

    completed_cases = [
        case for case in cases if case.get("case_status") == "completed_engineering_candidate_run"
    ]
    duration_satisfied_cases = [
        case for case in completed_cases if case.get("duration_request_satisfied") is True
    ]
    finite_cases = [
        case
        for case in completed_cases
        if case.get("simulation", {}).get("finite_state", {}).get("all_finite") is True
    ]
    inverse_packet = build_experimental_inverse_parameter_packet(
        scope=scope,
        include_gv_waveforms=include_gv_waveforms,
    )
    return {
        "tool": "dpf experimental-machine-shot-family",
        "command_status": "experimental_machine_shot_family_engineering_candidate_run",
        "status": "experimental_machine_shot_family_not_validation",
        "scope": scope,
        "case_count": len(cases),
        "completed_case_count": len(completed_cases),
        "blocked_case_count": len(cases) - len(completed_cases),
        "duration_satisfied_case_count": len(duration_satisfied_cases),
        "finite_case_count": len(finite_cases),
        "target_time_s": target_time_s,
        "dt_policy": dt_policy,
        "auto_step_budget": auto_step_budget,
        "max_auto_steps": max_auto_steps,
        "inverse_parameter_summary": {
            "status": inverse_packet["status"],
            "machine_count": inverse_packet["machine_count"],
            "status_counts": inverse_packet["status_counts"],
            "unresolved_parameter_count": inverse_packet[
                "unresolved_parameter_count"
            ],
            "contradiction_or_scope_mismatch_count": inverse_packet[
                "contradiction_or_scope_mismatch_count"
            ],
            "can_support_first_principles_acceptance": inverse_packet[
                "can_support_first_principles_acceptance"
            ],
        },
        "cases": cases,
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_machine_family_decks(scope: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    from dpf.first_principles import (
        gv_verified_engineering_decks,
        may15_second_scope_engineering_decks,
        pf1000_akel_16kv_engineering_deck,
    )

    normalized = scope.lower()
    if normalized not in {"all", "pf1000", "may15", "gv"}:
        raise click.BadParameter("scope must be one of all, pf1000, may15, gv")

    decks: list[tuple[str, dict[str, Any]]] = []
    if normalized in {"all", "pf1000"}:
        deck = pf1000_akel_16kv_engineering_deck(n_steps=2).to_dict()
        deck["source"] = "built_in:pf1000_akel_16kv"
        decks.append(("pf1000_akel_16kv_shot_12581", deck))
    if normalized in {"all", "may15"}:
        for deck_obj in may15_second_scope_engineering_decks(n_steps=2):
            deck = deck_obj.to_dict()
            deck["source"] = f"built_in:{deck_obj.deck_id}"
            decks.append((deck_obj.deck_id, deck))
    if normalized in {"all", "gv"}:
        for deck_obj in gv_verified_engineering_decks(n_steps=2):
            deck = deck_obj.to_dict()
            deck["source"] = f"built_in:{deck_obj.deck_id}"
            decks.append((deck_obj.deck_id, deck))
    return tuple(decks)


def _experimental_inverse_calibration_payload(
    *,
    deck_preset: str,
    parameter_names: tuple[str, ...],
    scale_values: tuple[float, ...],
    parameter_scale_values: dict[str, tuple[float, ...]] | None,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
) -> dict[str, Any]:
    from dpf.first_principles import (
        build_experimental_inverse_calibration_packet,
        build_source_bounded_candidate_grid,
        build_source_bounded_candidate_grid_from_parameter_scales,
        score_current_history_against_targets,
    )

    target_observables = _calibration_target_observables(deck_preset)
    base_deck = _first_principles_3d_deck_preset(deck_preset)
    baseline_parameters = _calibration_baseline_parameters(base_deck, parameter_names)
    candidate_grid = (
        build_source_bounded_candidate_grid(
            baseline_parameters=baseline_parameters,
            parameter_names=parameter_names,
            scale_values=scale_values,
        )
        if parameter_scale_values is None
        else build_source_bounded_candidate_grid_from_parameter_scales(
            baseline_parameters=baseline_parameters,
            parameter_names=parameter_names,
            parameter_scale_values=parameter_scale_values,
        )
    )

    candidate_results: list[dict[str, Any]] = []
    for candidate in candidate_grid:
        runtime_deck = _apply_calibration_candidate_parameters(
            base_deck,
            candidate["parameter_values"],
        )
        try:
            runtime_deck = _override_first_principles_3d_deck_runtime(
                runtime_deck,
                steps=steps,
                dt_s=dt_s,
                history_stride=history_stride,
                max_step_results=max_step_results,
                target_time_s=target_time_s,
            )
            runtime_deck = _apply_experimental_dt_policy(
                runtime_deck,
                dt_policy=dt_policy,
                vacuum_cfl=vacuum_cfl,
            )
            runtime_deck = _apply_experimental_auto_step_budget(
                runtime_deck,
                enabled=auto_step_budget,
                max_auto_steps=max_auto_steps,
            )
            payload = _experimental_whole_shot_payload(runtime_deck)
            current_history = payload["simulation"]["circuit"]["current_history"]
            scoring = score_current_history_against_targets(
                current_history=current_history,
                target_observables=target_observables,
            )
            plasma_loading_summary = _experimental_plasma_loading_summary(payload)
            candidate_with_baseline = dict(candidate)
            candidate_with_baseline["baseline_parameters"] = baseline_parameters
            candidate_results.append(
                {
                    "candidate": candidate_with_baseline,
                    "case_status": "completed_engineering_candidate_run",
                    "runtime_coupling": runtime_deck.get(
                        "_experimental_calibration_coupling",
                        {},
                    ),
                    "plasma_loading_summary": plasma_loading_summary,
                    "scoring": scoring,
                    "n_steps_completed": payload["n_steps_completed"],
                    "final_time_s": payload["simulation"]["final_time_s"],
                    "duration_request_satisfied": payload[
                        "duration_request_satisfied"
                    ],
                    "finite_state_all_finite": payload["simulation"][
                        "finite_state"
                    ]["all_finite"],
                    "termination_reason": payload["termination_reason"],
                    "final_circuit_current_A": payload["simulation"][
                        "circuit"
                    ]["final_current_A"],
                    "can_support_first_principles_acceptance": False,
                }
            )
        except (
            click.ClickException,
            FloatingPointError,
            OverflowError,
            ValueError,
        ) as exc:
            candidate_with_baseline = dict(candidate)
            candidate_with_baseline["baseline_parameters"] = baseline_parameters
            candidate_results.append(
                {
                    "candidate": candidate_with_baseline,
                    "case_status": "blocked_before_run",
                    "blocking_reason": str(exc),
                    "scoring": {"status": "not_run", "score": None, "usable": False},
                    "can_support_first_principles_acceptance": False,
                }
            )

    first_deck = _first_principles_3d_deck_preset(deck_preset)
    device_name = str(first_deck.get("device", {}).get("name", deck_preset))
    packet = build_experimental_inverse_calibration_packet(
        declared_scope=deck_preset,
        device_name=device_name,
        target_observables=target_observables,
        candidate_results=tuple(candidate_results),
        parameter_names=parameter_names,
    )
    packet["runtime_policy"] = {
        "target_time_s": target_time_s,
        "dt_policy": dt_policy,
        "vacuum_cfl": vacuum_cfl,
        "auto_step_budget": auto_step_budget,
        "max_auto_steps": max_auto_steps,
        "history_stride": history_stride,
        "max_step_results": max_step_results,
        "shared_candidate_scales": list(scale_values),
        "parameter_scale_values": (
            None
            if parameter_scale_values is None
            else {
                name: list(parameter_scale_values[name])
                for name in parameter_names
            }
        ),
    }
    return packet


def _experimental_plasma_loading_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize retained field-work telemetry for inverse-calibration candidates."""

    simulation = payload.get("simulation", {})
    history = simulation.get("history_summary", [])
    if not isinstance(history, list):
        history = []
    circuit = simulation.get("circuit", {})
    if not isinstance(circuit, dict):
        circuit = {}
    current_history = circuit.get("current_history", [])
    if not isinstance(current_history, list):
        current_history = []

    j_dot_e_values = _finite_history_values(history, "j_dot_e_power_W")
    field_energy_values = _finite_history_values(history, "field_energy_J")
    electric_energy_values = _finite_history_values(history, "electric_energy_J")
    magnetic_energy_values = _finite_history_values(history, "magnetic_energy_J")
    time_values = _finite_history_values(history, "time_s")
    current_values = _finite_history_values(current_history, "current_A")
    electron_density_min_values = _finite_history_values(
        history,
        "electron_density_min_m3",
    )
    electron_density_max_values = _finite_history_values(
        history,
        "electron_density_max_m3",
    )
    neutral_density_max_values = _finite_history_values(
        history,
        "source_backed_neutral_density_max_m3",
    )
    sigma_max_values = _finite_history_values(
        history,
        "source_backed_sigma_max_S_m",
    )
    resistivity_max_values = _finite_history_values(
        history,
        "source_backed_resistivity_max_ohm_m",
    )
    conductivity_effective_values = _finite_history_values(
        history,
        "conductivity_effective_max_S_m",
    )
    conductivity_cfl_fraction_values = _finite_history_values(
        history,
        "conductivity_cfl_limited_fraction",
    )
    conductivity_ohmic_limit_counts = _count_history_values(
        history,
        "conductivity_ohmic_cfl_limit_applied",
    )
    ionization_fraction_values = _finite_history_values(
        history,
        "ionization_fraction_max",
    )
    circuit_source_counts = _count_history_values(current_history, "udpf_source")
    last_circuit_record = circuit.get("last")
    if not isinstance(last_circuit_record, dict):
        last_circuit_record = {}
    last_circuit_step = last_circuit_record.get("circuit_step")
    if not isinstance(last_circuit_step, dict):
        last_circuit_step = {}
    final_terminal_current = _optional_finite_float(last_circuit_step.get("current_A"))
    final_terminal_voltage = _optional_finite_float(last_circuit_step.get("udpf_V"))
    final_active_power = (
        None
        if final_terminal_current is None or final_terminal_voltage is None
        else final_terminal_current * final_terminal_voltage
    )

    final_current = _optional_finite_float(circuit.get("final_current_A"))
    if final_current is None and current_values:
        final_current = current_values[-1]

    initial_field_energy = _optional_finite_float(
        simulation.get("initial_field_energy_J")
    )
    final_field_energy = _optional_finite_float(simulation.get("final_field_energy_J"))
    return {
        "status": "experimental_plasma_loading_observability_not_validation",
        "history_scope": "retained_history_summary_only",
        "history_point_count": len(history),
        "history_stride": simulation.get("history_stride"),
        "max_step_results": simulation.get("max_step_results"),
        "j_dot_e_power_W_min": _min_or_none(j_dot_e_values),
        "j_dot_e_power_W_max": _max_or_none(j_dot_e_values),
        "j_dot_e_power_W_final": _last_or_none(j_dot_e_values),
        "j_dot_e_power_W_max_abs": _max_abs_or_none(j_dot_e_values),
        "j_dot_e_energy_trapezoid_J": _trapezoid_integral(
            x_values=time_values,
            y_values=j_dot_e_values,
        ),
        "field_energy_J_initial": initial_field_energy,
        "field_energy_J_final": final_field_energy,
        "field_energy_delta_J": (
            None
            if initial_field_energy is None or final_field_energy is None
            else final_field_energy - initial_field_energy
        ),
        "retained_field_energy_J_min": _min_or_none(field_energy_values),
        "retained_field_energy_J_max": _max_or_none(field_energy_values),
        "retained_electric_energy_J_final": _last_or_none(electric_energy_values),
        "retained_magnetic_energy_J_final": _last_or_none(magnetic_energy_values),
        "electron_density_m3_min_retained": _min_or_none(
            electron_density_min_values
        ),
        "electron_density_m3_max_retained": _max_or_none(
            electron_density_max_values
        ),
        "neutral_density_m3_max_retained": _max_or_none(
            neutral_density_max_values
        ),
        "source_backed_sigma_S_m_max_retained": _max_or_none(sigma_max_values),
        "source_backed_resistivity_ohm_m_max_retained": _max_or_none(
            resistivity_max_values
        ),
        "conductivity_effective_S_m_max_retained": _max_or_none(
            conductivity_effective_values
        ),
        "conductivity_cfl_limited_fraction_max_retained": _max_or_none(
            conductivity_cfl_fraction_values
        ),
        "conductivity_ohmic_cfl_limit_applied_counts": (
            conductivity_ohmic_limit_counts
        ),
        "ionization_fraction_max_retained": _max_or_none(
            ionization_fraction_values
        ),
        "circuit_current_A_initial": _first_or_none(current_values),
        "circuit_current_A_final": final_current,
        "circuit_current_A_peak_abs": _max_abs_or_none(current_values),
        "circuit_udpf_source_counts": circuit_source_counts,
        "circuit_final_udpf_source": last_circuit_record.get("udpf_source"),
        "circuit_terminal_current_A_final_step": final_terminal_current,
        "circuit_terminal_voltage_V_final_step": final_terminal_voltage,
        "circuit_active_power_W_final_step": final_active_power,
        "active_to_j_dot_e_power_ratio_final": _safe_ratio(
            final_active_power,
            _last_or_none(j_dot_e_values),
        ),
        "can_support_first_principles_acceptance": False,
    }


def _finite_history_values(
    rows: list[Any],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _optional_finite_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _count_history_values(
    rows: list[Any],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get(key) is None:
            continue
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _first_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return values[0]


def _last_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return values[-1]


def _min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return min(values)


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)


def _max_abs_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return max(abs(value) for value in values)


def _trapezoid_integral(
    *,
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    total = 0.0
    for left_index, right_index in zip(
        range(len(x_values) - 1),
        range(1, len(x_values)),
        strict=True,
    ):
        dt = x_values[right_index] - x_values[left_index]
        total += 0.5 * (y_values[right_index] + y_values[left_index]) * dt
    return total


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _experimental_inverse_calibration_family_payload(
    *,
    deck_presets: tuple[str, ...],
    parameter_names: tuple[str, ...],
    scale_values: tuple[float, ...],
    parameter_scale_values: dict[str, tuple[float, ...]] | None,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
) -> dict[str, Any]:
    packets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for preset in deck_presets:
        try:
            packets.append(
                _experimental_inverse_calibration_payload(
                    deck_preset=preset,
                    parameter_names=parameter_names,
                    scale_values=scale_values,
                    parameter_scale_values=parameter_scale_values,
                    steps=steps,
                    dt_s=dt_s,
                    dt_policy=dt_policy,
                    vacuum_cfl=vacuum_cfl,
                    history_stride=history_stride,
                    max_step_results=max_step_results,
                    target_time_s=target_time_s,
                    auto_step_budget=auto_step_budget,
                    max_auto_steps=max_auto_steps,
                )
            )
        except click.ClickException as exc:
            skipped.append({"deck_preset": preset, "reason": str(exc)})

    return {
        "tool": "dpf experimental-inverse-calibration",
        "status": "experimental_inverse_calibration_family_not_validation",
        "requested_deck_presets": list(deck_presets),
        "completed_calibration_count": len(packets),
        "skipped_calibration_count": len(skipped),
        "skipped": skipped,
        "calibrations": packets,
        "can_support_first_principles_acceptance": False,
    }


def _calibration_target_observables(deck_preset: str) -> dict[str, Any]:
    from dpf.first_principles import (
        extract_gv_current_waveform_packet,
        may15_user_validated_source_targets,
    )

    normalized = deck_preset.lower()
    targets = may15_user_validated_source_targets()["device_deck_targets"]
    if normalized == "compact_chinese_dpf":
        target = targets["compact_chinese_dpf_2018"]
        return {
            "peak_current_A": target["circuit"]["delivered_current_A_approx"],
            "source": target["source"],
            "observable_status": "source_approximate_delivered_current",
            "accepted_for_validation": False,
        }
    if normalized == "ir_mpf_100":
        target = targets["ir_mpf_100_salehizadeh_2012"]
        return {
            "peak_current_A": target["circuit"]["theoretical_peak_current_A"],
            "source": target["source"],
            "observable_status": "source_theoretical_peak_current",
            "accepted_for_validation": False,
        }
    if normalized == "gv_pf24_krakow_16092202":
        packet = extract_gv_current_waveform_packet("pf24_krakow_16092202")
        summary = packet["summary"]
        series = packet["digitized_series"][0]
        currents = [abs(float(value)) for value in series["y"]]
        peak_index = max(range(len(currents)), key=lambda index: currents[index])
        peak_time_s = float(series["x"][peak_index]) * 1.0e-6
        return {
            "peak_current_A": currents[peak_index] * 1.0e3,
            "peak_time_s": peak_time_s if peak_time_s > 0.0 else None,
            "source": packet["source_paths"]["workbook"],
            "series_sha256": series["series_sha256"],
            "observable_status": "user_verified_workbook_waveform_candidate",
            "accepted_for_validation": False,
            "target_time_range_us": [
                summary["time_min_us"],
                summary["time_max_us"],
            ],
            "minimum_waveform_coverage_fraction": 0.95,
            "minimum_waveform_overlap_points": 3,
            "waveform": {
                "time_us": [float(value) for value in series["x"]],
                "current_kA": [float(value) for value in series["y"]],
                "series_sha256": series["series_sha256"],
                "source": packet["source_paths"]["workbook"],
                "status": "user_verified_workbook_waveform_candidate",
                "accepted_for_validation": False,
            },
        }
    raise click.ClickException(
        f"{deck_preset} does not yet have a typed calibration observable. "
        "Add a source-backed current waveform, peak-current target, or timing "
        "target before fitting parameters."
    )


def _calibration_baseline_parameters(
    deck: dict[str, Any],
    parameter_names: tuple[str, ...],
) -> dict[str, float]:
    circuit = deck.get("circuit", {})
    gas = deck.get("gas", {})
    if not isinstance(circuit, dict) or not isinstance(gas, dict):
        raise click.ClickException("calibration requires a full first-principles deck")
    available = {
        "inductance": float(circuit["inductance_H"]),
        "resistance": float(circuit["resistance_ohm"]),
        "voltage": float(circuit["voltage_V"]),
        "pressure": float(gas["pressure_Pa"]),
    }
    baselines: dict[str, float] = {}
    for name in parameter_names:
        if name not in available:
            raise click.BadParameter(
                f"unknown calibration parameter {name}",
                param_hint="--parameters",
            )
        if available[name] <= 0.0:
            raise click.ClickException(
                f"calibration parameter {name} has non-positive baseline "
                f"{available[name]}; supply a source-backed nonzero baseline "
                "before multiplicative fitting"
            )
        baselines[name] = available[name]
    return baselines


def _apply_calibration_candidate_parameters(
    deck: dict[str, Any],
    parameter_values: dict[str, float],
) -> dict[str, Any]:
    updated = dict(deck)
    circuit = dict(updated.get("circuit", {}))
    gas = dict(updated.get("gas", {}))
    startup = dict(updated.get("startup", {}))
    closures = dict(updated.get("closures", {}))
    coupling: dict[str, Any] = {}
    if "inductance" in parameter_values:
        circuit["inductance_H"] = float(parameter_values["inductance"])
    if "resistance" in parameter_values:
        circuit["resistance_ohm"] = float(parameter_values["resistance"])
    if "voltage" in parameter_values:
        circuit["voltage_V"] = float(parameter_values["voltage"])
    if "pressure" in parameter_values:
        gas["pressure_Pa"] = float(parameter_values["pressure"])
        total_density_m3 = _ideal_gas_number_density_m3(gas)
        startup["background_density_m3"] = total_density_m3
        closures["density_floor_m3"] = total_density_m3
        coupling["pressure_density_coupling"] = {
            "status": "experimental_ideal_gas_pressure_to_startup_density",
            "formula": "n = pressure_Pa / (k_B * gas_temperature_K)",
            "gas_pressure_Pa": float(gas["pressure_Pa"]),
            "gas_temperature_K": float(gas.get("temperature_K", 300.0)),
            "background_density_m3": total_density_m3,
            "density_floor_m3": total_density_m3,
            "accepted_for_validation": False,
        }
    updated["circuit"] = circuit
    updated["gas"] = gas
    updated["startup"] = startup
    updated["closures"] = closures
    if coupling:
        updated["_experimental_calibration_coupling"] = coupling
    return updated


def _ideal_gas_number_density_m3(gas: dict[str, Any]) -> float:
    from dpf.constants import k_B

    pressure = float(gas["pressure_Pa"])
    temperature = float(gas.get("temperature_K", 300.0))
    if pressure <= 0.0:
        raise click.ClickException("gas pressure must be positive for calibration")
    if temperature <= 0.0:
        raise click.ClickException("gas temperature must be positive for calibration")
    return pressure / (float(k_B) * temperature)


def _experimental_limiter_proof_payload(deck: dict[str, Any]) -> dict[str, Any]:
    from dpf.first_principles.limiter_proof import (
        build_experimental_limiter_zero_probe_packet,
    )

    payload = _first_principles_3d_payload(deck)
    deck_summary = payload["deck"]
    packet = build_experimental_limiter_zero_probe_packet(
        declared_scope=str(deck_summary["name"]),
        device_name=str(deck_summary.get("device_name", "not_declared")),
        simulation_telemetry=payload["simulation"],
    )
    payload["tool"] = "dpf experimental-limiter-proof"
    payload["command_status"] = "experimental_limiter_proof_engineering_candidate_run"
    payload["experimental_limiter_zero_probe"] = packet
    payload["limiter_zero_probe"] = packet
    payload["telemetry_packets"]["experimental_limiter_zero_probe"] = packet
    payload["scientific_status"] = "engineering_candidate_not_validation"
    payload["reduced_models_used"] = False
    payload["can_support_first_principles_acceptance"] = False
    return payload


def _experimental_numerical_family_payload(
    deck: dict[str, Any],
    *,
    family: str,
    timestep_scales: tuple[float, ...],
    mesh_shapes: tuple[tuple[int, int, int], ...],
    dt_policy: str,
    vacuum_cfl: float,
    auto_step_budget: bool,
    max_auto_steps: int,
) -> dict[str, Any]:
    from dpf.first_principles.experimental_numerics import (
        build_experimental_numerical_family_packet,
    )

    family_kind = family.lower()
    cases: list[dict[str, Any]] = []
    if family_kind in {"timestep", "both"}:
        base = _apply_experimental_dt_policy(
            deck,
            dt_policy=dt_policy,
            vacuum_cfl=vacuum_cfl,
        )
        base_dt_s = _experimental_deck_dt_s(base)
        for scale in timestep_scales:
            case_deck = _set_first_principles_3d_deck_runtime_value(
                base,
                section="diagnostics",
                key="dt_s",
                value=base_dt_s * float(scale),
                compact_key="dt_s",
            )
            case_deck = _apply_experimental_auto_step_budget(
                case_deck,
                enabled=auto_step_budget,
                max_auto_steps=max_auto_steps,
            )
            case_payload = _experimental_whole_shot_payload(case_deck)
            case_payload["case_label"] = f"timestep_scale_{scale:g}"
            case_payload["case_family_axis"] = {
                "kind": "timestep",
                "dt_scale": float(scale),
            }
            cases.append(case_payload)
    if family_kind in {"mesh", "both"}:
        for shape in mesh_shapes:
            case_deck = _set_experimental_deck_grid_shape_keep_extent(deck, shape)
            case_deck = _apply_experimental_dt_policy(
                case_deck,
                dt_policy=dt_policy,
                vacuum_cfl=vacuum_cfl,
            )
            case_deck = _apply_experimental_auto_step_budget(
                case_deck,
                enabled=auto_step_budget,
                max_auto_steps=max_auto_steps,
            )
            case_payload = _experimental_whole_shot_payload(case_deck)
            case_payload["case_label"] = "mesh_" + "x".join(str(item) for item in shape)
            case_payload["case_family_axis"] = {
                "kind": "mesh",
                "grid_shape": list(shape),
            }
            cases.append(case_payload)
    if len(cases) < 2:
        raise click.ClickException("experimental numerical family requires at least two cases")

    first = cases[0]
    packet = build_experimental_numerical_family_packet(
        family_kind=family_kind,
        case_payloads=cases,
        declared_scope=str(first["deck"].get("name", "not_declared")),
        device_name=str(first["deck"].get("device_name", "not_declared")),
    )
    return {
        "tool": "dpf experimental-numerical-family",
        "command_status": "experimental_numerical_family_engineering_candidate_run",
        "experimental_numerical_family": packet,
        "family_probe": packet,
        "cases": cases,
        "case_count": len(cases),
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_reproducibility_payload(
    deck: dict[str, Any],
    *,
    repeat_count: int,
) -> dict[str, Any]:
    from copy import deepcopy

    from dpf.first_principles.experimental_numerics import (
        build_experimental_reproducibility_packet,
    )

    if repeat_count < 2:
        raise click.ClickException(
            "experimental reproducibility requires at least two reruns"
        )

    runs: list[dict[str, Any]] = []
    for index in range(int(repeat_count)):
        case_payload = _experimental_whole_shot_payload(deepcopy(deck))
        case_payload["case_label"] = f"deterministic_rerun_{index}"
        case_payload["case_family_axis"] = {
            "kind": "deterministic_rerun",
            "repeat_index": index,
        }
        runs.append(case_payload)

    first = runs[0]
    packet = build_experimental_reproducibility_packet(
        run_payloads=runs,
        declared_scope=str(first["deck"].get("name", "not_declared")),
        device_name=str(first["deck"].get("device_name", "not_declared")),
    )
    return {
        "tool": "dpf experimental-reproducibility",
        "command_status": "experimental_reproducibility_engineering_candidate_run",
        "experimental_reproducibility": packet,
        "reproducibility_probe": packet,
        "runs": runs,
        "run_count": len(runs),
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_state_checkpoint_payload(
    deck: dict[str, Any],
    *,
    checkpoint_output: Path,
) -> dict[str, Any]:
    from dpf.first_principles import FirstPrinciplesInputDeck
    from dpf.first_principles.runner import run_first_principles_3d_deck
    from dpf.first_principles.state_checkpoint import (
        write_terminal_state_checkpoint_roundtrip,
    )

    package_deck: Any
    if {"device", "circuit", "grid"}.issubset(deck.keys()):
        package_deck = FirstPrinciplesInputDeck.from_mapping(deck)
        deck_name = package_deck.deck_id
        device_name = package_deck.device.name
    else:
        package_deck = deck
        deck_name = str(deck.get("name", "minimal_engineering_3d"))
        device_name = str(deck.get("device_name", "not_declared"))
    run = run_first_principles_3d_deck(package_deck)
    checkpoint_packet = write_terminal_state_checkpoint_roundtrip(
        run_result=run,
        checkpoint_path=checkpoint_output,
    )
    simulation = run.telemetry["simulation"]
    return {
        "tool": "dpf experimental-state-checkpoint",
        "command_status": "experimental_state_checkpoint_engineering_candidate_run",
        "deck_name": deck_name,
        "device_name": device_name,
        "simulation": simulation,
        "conservation_telemetry": run.conservation_telemetry,
        "manifest": run.manifest,
        "experimental_state_checkpoint": checkpoint_packet,
        "checkpoint_probe": checkpoint_packet,
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_split_continuation_payload(
    deck: dict[str, Any],
    *,
    split_after_steps: int,
) -> dict[str, Any]:
    from dpf.first_principles.split_continuation import (
        build_experimental_split_continuation_packet,
    )

    packet = build_experimental_split_continuation_packet(
        deck=deck,
        split_after_steps=split_after_steps,
    )
    return {
        "tool": "dpf experimental-split-continuation",
        "command_status": "experimental_split_continuation_engineering_candidate_run",
        "experimental_split_continuation": packet,
        "split_continuation_probe": packet,
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_checkpoint_restart_payload(
    deck: dict[str, Any],
    *,
    split_after_steps: int,
    checkpoint_output: Path,
) -> dict[str, Any]:
    from dpf.first_principles.checkpoint_restart import (
        build_experimental_checkpoint_restart_packet,
    )

    packet = build_experimental_checkpoint_restart_packet(
        deck=deck,
        split_after_steps=split_after_steps,
        checkpoint_path=checkpoint_output,
    )
    return {
        "tool": "dpf experimental-checkpoint-restart",
        "command_status": "experimental_checkpoint_restart_engineering_candidate_run",
        "experimental_checkpoint_restart": packet,
        "checkpoint_restart_probe": packet,
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _experimental_checkpoint_restart_family_payload(
    deck: dict[str, Any],
    *,
    split_offsets: tuple[int, ...],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    from dpf.first_principles.checkpoint_restart import (
        build_experimental_checkpoint_restart_family_packet,
    )

    packet = build_experimental_checkpoint_restart_family_packet(
        deck=deck,
        split_after_steps=split_offsets,
        checkpoint_dir=checkpoint_dir,
    )
    return {
        "tool": "dpf experimental-checkpoint-restart-family",
        "command_status": (
            "experimental_checkpoint_restart_family_engineering_candidate_run"
        ),
        "experimental_checkpoint_restart_family": packet,
        "checkpoint_restart_family_probe": packet,
        "scientific_status": "engineering_candidate_not_validation",
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
    }


def _first_principles_telemetry_packets(
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    packet_keys = (
        "startup",
        "pic_particle_loading",
        "boundary_policy",
        "limiter_readiness",
        "experimental_limiter_zero_probe",
        "power_port",
        "dimensionality_handoff",
        "physics_closure",
        "numerical_fidelity",
        "same_scope_source",
        "waveform_phase",
        "spatial_field_temperature",
        "neutron_authority",
        "comparator_uq",
        "certificate_gate",
        "generalization",
        "experimental_whole_shot",
        "experimental_numerics",
    )
    return {key: telemetry[key] for key in packet_keys if key in telemetry}


def _engineering_firm_dossier(payload: dict[str, Any]) -> dict[str, Any]:
    packets = payload.get("telemetry_packets", {})
    packet_statuses = {
        key: value.get("status", "unknown")
        for key, value in packets.items()
        if isinstance(value, dict)
    }
    active_blockers = [
        {
            "packet": key,
            "status": status,
            "review_focus": _engineering_review_focus(key),
        }
        for key, status in packet_statuses.items()
        if status.startswith("blocked") or "not_available" in status
    ]
    return {
        "status": "engineering_firm_experimental_test_dossier_not_validation",
        "intended_use": (
            "independent engineering review of the package-native 3-D "
            "first-principles experimental simulator path"
        ),
        "not_for": [
            "accepted validation certificate",
            "predictive neutron-yield authority",
            "generalized DPF-machine claim",
        ],
        "recommended_command": (
            "PYTHONPATH=src dpf first-principles-3d --deck-preset "
            f"{_deck_source_to_preset(payload['deck'].get('source', 'built_in'))} "
            "--steps <N> --history-stride <N> --max-step-results <N> "
            "--output results/engineering_firm_probe.json"
        ),
        "runtime_scope": {
            "runner": payload["runner"],
            "deck": payload["deck"]["name"],
            "device_name": payload["deck"].get("device_name", "not_declared"),
            "grid_shape": payload["grid_shape"],
            "n_steps_requested": payload["n_steps"],
            "n_steps_completed": payload["n_steps_completed"],
            "dt_s": payload["dt_s"],
            "simulated_time_s": float(payload["simulation"]["final_time_s"]),
            "termination_reason": payload["termination_reason"],
            "target_time_s": payload["target_time_s"],
            "duration_request_satisfied": payload["duration_request_satisfied"],
            "history_stride": payload["history_stride"],
            "max_step_results": payload["max_step_results"],
            "retained_step_result_count": payload["simulation"][
                "retained_step_result_count"
            ],
            "finite_state": payload["simulation"]["finite_state"],
            "reduced_models_used": payload["reduced_models_used"],
            "first_principles_only_enforced": payload["reduced_models_used"] is False,
        },
        "observable_surfaces": [
            "terminal current and voltage history",
            "field-energy and particle-energy conservation ledger",
            "power-port J.E and magnetic-energy accounting",
            "3-D field/PIC/electron-energy step telemetry",
            "ionization, transport, heat-flux, and neutron-authority packets",
            "PlasmaPy optional community-formulary audit packet when installed",
        ],
        "packet_statuses": packet_statuses,
        "active_blockers": active_blockers,
        "active_blocker_count": len(active_blockers),
        "review_instructions": [
            "Treat every candidate packet as engineering evidence only.",
            "Check whether runtime floors, clips, repairs, and timestep choices are physical or numerical blockers.",
            "Review power-port sign convention, time centering, and residual budget before trusting circuit coupling.",
            "Review startup BVP and same-scope observables before any whole-shot claim.",
            "Use PlasmaPy audit results only as formula/unit cross-checks, not source authority.",
        ],
        "can_support_first_principles_acceptance": False,
    }


def _engineering_review_focus(packet_key: str) -> str:
    focus = {
        "startup": "whole-shot breakdown, preionization, flashover, and sheath liftoff BVP",
        "limiter_readiness": "full-horizon zero hidden limiter/floor/repair proof",
        "power_port": "resolved terminal power, Poynting or J.E accounting, electrode work, residuals",
        "dimensionality_handoff": "3-D/hybrid/PIC scope and MHD-to-kinetic handoff",
        "physics_closure": "EOS, ionization, 2T, transport, radiation, ablation, anomalous resistance",
        "numerical_fidelity": "convergence, div-B/Gauss law, shocks, restart, backend/precision parity",
        "experimental_numerics": "runtime CFL, conservation, divergence, limiter, convergence, and restart troubleshooting",
        "same_scope_source": "accepted same-scope PF-1000/Akel targets and transfer rules",
        "waveform_phase": "digitized current waveform, derivative dip, timing UQ",
        "spatial_field_temperature": "same-shot density, field, and temperature diagnostics",
        "neutron_authority": "mechanism-separated thermonuclear and beam-target neutron evidence",
        "comparator_uq": "observable metrics, tolerances, uncertainty propagation, pass/fail rules",
        "certificate_gate": "manifest hashes, reviewer metadata, requirement links, release decision",
        "generalization": "second-device repeatability without hidden PF-1000/Akel assumptions",
    }
    return focus.get(packet_key, "engineering review required")


def _deck_source_to_preset(source: str) -> str:
    if source.startswith("built_in:"):
        return source.split(":", 1)[1]
    return "pf1000_akel_16kv"


def _first_principles_package_native_payload(
    *,
    preset: str,
    grid_preset: str,
    sim_time_us: float,
    history_stride: int,
    steps: int = 2,
) -> dict[str, Any]:
    from dpf.first_principles import pf1000_akel_16kv_engineering_deck

    step_count = max(int(steps), 1)
    deck = pf1000_akel_16kv_engineering_deck(
        n_steps=step_count,
        shape=_first_principles_grid_shape(grid_preset),
    ).to_dict()
    diagnostics = dict(deck.get("diagnostics", {}))
    diagnostics["history_stride"] = int(history_stride)
    diagnostics.setdefault("max_step_results", 256)
    deck["diagnostics"] = diagnostics
    deck["source"] = "built_in"
    payload = _first_principles_3d_payload(deck)
    simulated_time_us = float(payload["n_steps"]) * float(payload["dt_s"]) * 1.0e6
    requested_sim_time_us = float(sim_time_us)
    duration_satisfied = simulated_time_us >= requested_sim_time_us
    payload["tool"] = "dpf first-principles"
    payload["package_native_tool"] = "dpf first-principles-3d"
    payload["preset"] = preset
    payload["grid_preset"] = grid_preset
    payload["requested_sim_time_us"] = requested_sim_time_us
    payload["simulated_time_us"] = simulated_time_us
    payload["candidate_step_budget"] = step_count
    payload["duration_request_satisfied"] = duration_satisfied
    payload["duration_gate_status"] = (
        "candidate_duration_satisfies_request"
        if duration_satisfied
        else "blocked_requested_duration_exceeds_candidate_step_budget"
    )
    payload["duration_gate_reason"] = (
        "Package-native first-principles currently executes the explicit "
        "candidate step budget; accepted whole-shot duration remains blocked "
        "until startup, limiter, numerical-fidelity, physics-closure, UQ, and "
        "certificate gates pass."
    )
    payload["history_stride"] = history_stride
    payload["run_mode"] = "first_principles_3d_hybrid_em_pic_fluid"
    payload["execution_backend"] = "package_native"
    payload["first_principles_only_enforced"] = (
        payload.get("reduced_models_used") is False
    )
    return payload


@cli.command("first-principles")
@click.option(
    "--preset",
    type=click.Choice(["pf1000_akel"], case_sensitive=False),
    default="pf1000_akel",
    show_default=True,
    help="First-principles demonstrator scope. Locked to PF-1000/Akel for now.",
)
@click.option(
    "--grid-preset",
    type=click.Choice(["coarse", "medium", "fine"], case_sensitive=False),
    default="coarse",
    show_default=True,
    help="MHD grid preset.",
)
@click.option(
    "--sim-time-us",
    type=float,
    default=0.2,
    show_default=True,
    help="Requested candidate duration recorded in the artifact.",
)
@click.option(
    "--steps",
    type=int,
    default=2,
    show_default=True,
    help="Explicit package-native 3-D candidate step budget.",
)
@click.option("--gas", "gas_key", default="D2", show_default=True, help="Fill gas key.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write compact JSON artifact with selected time histories.",
)
@click.option(
    "--history-stride",
    type=int,
    default=1,
    show_default=True,
    help="Keep every Nth sample in the JSON history artifact.",
)
@click.option(
    "--require-field-feedback/--allow-zero-field-feedback",
    default=True,
    show_default=True,
    help="Fail when field-derived back-EMF stays zero.",
)
def first_principles(
    preset: str,
    grid_preset: str,
    sim_time_us: float,
    steps: int,
    gas_key: str,
    output: Path | None,
    history_stride: int,
    require_field_feedback: bool,
) -> None:
    """Run the PF-1000/Akel first-principles-only engineering candidate."""
    if sim_time_us <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--sim-time-us")
    if steps <= 0:
        raise click.BadParameter("must be positive", param_hint="--steps")
    if history_stride <= 0:
        raise click.BadParameter("must be positive", param_hint="--history-stride")

    payload = _first_principles_package_native_payload(
        preset=preset,
        grid_preset=grid_preset,
        sim_time_us=sim_time_us,
        history_stride=history_stride,
        steps=steps,
    )

    if output is not None:
        import json

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    simulation = payload["simulation"]
    validation_packet = payload["validation_packet"]
    click.echo("Package-native first-principles PF-1000/Akel engineering candidate")
    click.echo(f"  steps: {simulation['n_steps_completed']}")
    click.echo(f"  status: {simulation['status']}")
    click.echo(f"  simulated_time_us: {payload['simulated_time_us']:.6e}")
    click.echo(f"  duration_gate: {payload['duration_gate_status']}")
    click.echo(f"  final_particle_count: {payload['final_particle_count']}")
    click.echo(f"  final_field_energy_J: {payload['final_field_energy_J']:.6e}")
    click.echo(
        "  validation_packet: "
        f"{validation_packet.get('status', 'unknown')}"
    )
    if output is not None:
        click.echo(f"  artifact: {output}")

    if payload["first_principles_only_enforced"] is not True:
        raise click.ClickException(
            "first-principles-only enforcement failed: run did not stay on the "
            "package-native 3-D candidate path"
        )


@cli.command("hybrid-3d-smoke")
@click.option(
    "--steps",
    type=int,
    default=2,
    show_default=True,
    help="Number of candidate 3-D hybrid PIC-fluid steps.",
)
@click.option(
    "--shape",
    type=str,
    default="5,5,5",
    show_default=True,
    help="Cartesian smoke grid shape as NX,NY,NZ.",
)
@click.option(
    "--dt-s",
    type=float,
    default=1.0e-13,
    show_default=True,
    help="Candidate timestep in seconds.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write compact JSON artifact for the 3-D hybrid smoke.",
)
def hybrid_3d_smoke(
    steps: int,
    shape: str,
    dt_s: float,
    output: Path | None,
) -> None:
    """Run a fail-closed 3-D hybrid PIC-fluid engineering smoke."""
    payload = _hybrid_3d_smoke_payload(
        steps=steps,
        shape=_parse_grid_shape(shape),
        dt_s=dt_s,
    )
    if output is not None:
        import json

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    click.echo("3D hybrid PIC-fluid engineering candidate")
    click.echo(f"  steps: {payload['n_steps']}")
    click.echo(f"  grid_shape: {payload['grid_shape']}")
    click.echo(f"  final_particle_count: {payload['final_particle_count']}")
    click.echo(f"  final_field_energy_J: {payload['final_field_energy_J']:.6e}")
    click.echo(
        "  validation_packet: "
        f"{payload['validation_packet'].get('status', 'unknown')}"
    )
    click.echo(f"  scientific_status: {payload['scientific_status']}")
    if output is not None:
        click.echo(f"  artifact: {output}")


@cli.command("first-principles-3d")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help=(
        "Built-in source-scoped engineering deck preset. Ignored when --deck "
        "is supplied."
    ),
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Override the deck diagnostic step count for engineering sweeps.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep in seconds for engineering sweeps.",
)
@click.option(
    "--history-stride",
    type=int,
    default=None,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=None,
    help="Cap retained full step results; use 0 for summary-only long runs.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=None,
    help="Stop after reaching this simulated duration, within the step budget.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def first_principles_3d(
    deck: Path | None,
    deck_preset: str,
    steps: int | None,
    dt_s: float | None,
    history_stride: int | None,
    max_step_results: int | None,
    target_time_s: float | None,
    output: Path | None,
) -> None:
    """Run the package-native 3-D first-principles engineering candidate."""
    runtime_deck = _override_first_principles_3d_deck_runtime(
        _load_first_principles_3d_deck(deck, deck_preset=deck_preset),
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    payload = _first_principles_3d_payload(
        runtime_deck
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    click.echo("Package-native 3-D first-principles engineering candidate")
    click.echo(f"  deck: {payload['deck']['source']}")
    click.echo(f"  steps: {payload['n_steps']}")
    click.echo(f"  steps_completed: {payload['n_steps_completed']}")
    click.echo(f"  termination_reason: {payload['termination_reason']}")
    click.echo(f"  grid_shape: {payload['grid_shape']}")
    click.echo(f"  final_particle_count: {payload['final_particle_count']}")
    click.echo(f"  final_field_energy_J: {payload['final_field_energy_J']:.6e}")
    click.echo(f"  scientific_status: {payload['scientific_status']}")
    click.echo(
        "  current_waveform_comparison: "
        f"{payload['engineering_current_waveform_comparison']['status']}"
    )
    click.echo(
        "  blocker_count: "
        f"{len(payload['validation_packet'].get('blocking_reasons', ()))}"
    )
    click.echo(f"  artifact: {output}")


@cli.command("experimental-whole-shot")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help=(
        "Built-in source-scoped engineering deck preset. Ignored when --deck "
        "is supplied."
    ),
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Step budget for the experimental engineering attempt.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep in seconds for engineering sweeps.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="deck",
    show_default=True,
    help=(
        "Experimental timestep policy. 'deck' uses the deck/--dt-s value; "
        "'vacuum-cfl' uses the explicit 3-D Yee limit; 'ohmic-cfl' uses the "
        "source-grounded explicit Ohmic relaxation limit; 'combined-cfl' uses "
        "the stricter of both."
    ),
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=8,
    show_default=True,
    help="Cap retained full step results; use 0 for summary-only long runs.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-6,
    show_default=True,
    help="Requested whole-shot duration for the engineering attempt.",
)
@click.option(
    "--auto-step-budget",
    is_flag=True,
    help=(
        "Set steps to ceil(target_time_s / dt_s) after dt policy is applied. "
        "The cap must still permit the run."
    ),
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=100_000,
    show_default=True,
    help="Safety cap for --auto-step-budget.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_whole_shot(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Run the explicit whole-shot engineering candidate packet."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_whole_shot_payload(runtime_deck)

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["experimental_whole_shot"]
    click.echo("Experimental whole-shot engineering candidate")
    click.echo(f"  deck: {payload['deck']['source']}")
    click.echo(f"  steps: {payload['n_steps']}")
    click.echo(f"  steps_completed: {payload['n_steps_completed']}")
    click.echo(f"  target_time_s: {payload['target_time_s']}")
    click.echo(f"  dt_s: {payload['dt_s']}")
    click.echo(
        "  duration_request_satisfied: "
        f"{payload['duration_request_satisfied']}"
    )
    click.echo(f"  termination_reason: {payload['termination_reason']}")
    click.echo(
        "  required_steps_current_dt: "
        f"{packet['duration_plan']['steps_required_current_dt']}"
    )
    click.echo(
        "  required_steps_vacuum_cfl_dt: "
        f"{packet['duration_plan']['steps_required_vacuum_cfl_dt']}"
    )
    click.echo(f"  candidate_module_count: {packet['candidate_module_count']}")
    click.echo(f"  experimental_status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-machine-shot-family")
@click.option(
    "--scope",
    type=click.Choice(["all", "pf1000", "may15", "gv"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Machine registry subset to run.",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Step budget before optional auto-step-budget adjustment.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep in seconds before dt-policy.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="vacuum-cfl",
    show_default=True,
    help="Experimental timestep policy applied to every machine deck.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by vacuum/combined CFL policies.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=4,
    show_default=True,
    help="Cap retained full step results per machine.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-9,
    show_default=True,
    help="Requested target horizon for every machine case.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=True,
    show_default=True,
    help="Set each machine step budget to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=5000,
    show_default=True,
    help="Per-machine safety cap for auto-step-budget.",
)
@click.option(
    "--include-gv-waveforms/--no-include-gv-waveforms",
    default=True,
    show_default=True,
    help="Include GV waveform-derived entries in the inverse-parameter summary.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_machine_shot_family(
    scope: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    include_gv_waveforms: bool,
    output: Path | None,
) -> None:
    """Run a non-promoting source-backed whole-shot family across machine decks."""
    if steps <= 0:
        raise click.BadParameter("must be positive", param_hint="--steps")
    if target_time_s <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--target-time-s")
    if history_stride <= 0:
        raise click.BadParameter("must be positive", param_hint="--history-stride")
    if max_step_results < 0:
        raise click.BadParameter(
            "must be non-negative",
            param_hint="--max-step-results",
        )
    payload = _experimental_machine_shot_family_payload(
        scope=scope.lower(),
        steps=steps,
        dt_s=dt_s,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
        auto_step_budget=auto_step_budget,
        max_auto_steps=max_auto_steps,
        include_gv_waveforms=include_gv_waveforms,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    click.echo("Experimental machine-shot family engineering candidate")
    click.echo(f"  scope: {payload['scope']}")
    click.echo(f"  cases: {payload['case_count']}")
    click.echo(f"  completed_cases: {payload['completed_case_count']}")
    click.echo(f"  blocked_cases: {payload['blocked_case_count']}")
    click.echo(
        "  duration_satisfied_cases: "
        f"{payload['duration_satisfied_case_count']}"
    )
    click.echo(f"  status: {payload['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-limiter-proof")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Step budget for the limiter inventory run.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep in seconds for engineering sweeps.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="deck",
    show_default=True,
    help="Timestep policy for the limiter inventory run.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=8,
    show_default=True,
    help="Cap retained full step results; use 0 for summary-only long runs.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-6,
    show_default=True,
    help="Requested horizon for the limiter inventory run.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=False,
    show_default=True,
    help="Set steps to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=100_000,
    show_default=True,
    help="Safety cap for auto-step-budget.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_limiter_proof(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Run a non-promoting full-horizon limiter inventory probe."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_limiter_proof_payload(runtime_deck)

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["limiter_zero_probe"]
    click.echo("Experimental limiter-zero engineering candidate")
    click.echo(f"  deck: {payload['deck']['source']}")
    click.echo(f"  steps_completed: {payload['n_steps_completed']}")
    click.echo(f"  target_time_s: {payload['target_time_s']}")
    click.echo(
        "  inventory_complete: "
        f"{packet['runtime_horizon']['inventory_complete_for_completed_steps']}"
    )
    click.echo(
        "  zero_acceptance_blockers_observed: "
        f"{packet['zero_acceptance_blockers_observed']}"
    )
    click.echo(
        "  total_acceptance_blocking_activations: "
        f"{packet['total_acceptance_blocking_activations']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-numerical-family")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--family",
    type=click.Choice(["timestep", "mesh", "both"], case_sensitive=False),
    default="timestep",
    show_default=True,
    help="Experimental numerical family to run.",
)
@click.option(
    "--timestep-scales",
    type=str,
    default="1,0.5",
    show_default=True,
    help="Comma-separated dt multipliers for timestep family cases.",
)
@click.option(
    "--mesh-shapes",
    type=str,
    default="5,5,5;6,6,6",
    show_default=True,
    help="Semicolon-separated grid shapes for mesh family cases.",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Initial step budget before optional auto-step-budget adjustment.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep before family scaling.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="vacuum-cfl",
    show_default=True,
    help="Base timestep policy before timestep family scaling.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=4,
    show_default=True,
    help="Cap retained full step results per case.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-9,
    show_default=True,
    help="Requested duration for each engineering case.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=True,
    show_default=True,
    help="Set each case step budget to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for each auto-step-budget case.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_numerical_family(
    deck: Path | None,
    deck_preset: str,
    family: str,
    timestep_scales: str,
    mesh_shapes: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Run a non-promoting mesh/timestep troubleshooting family."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    payload = _experimental_numerical_family_payload(
        runtime_deck,
        family=family,
        timestep_scales=_parse_positive_float_tuple(
            timestep_scales,
            param_hint="--timestep-scales",
        ),
        mesh_shapes=_parse_shape_family(mesh_shapes, param_hint="--mesh-shapes"),
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
        auto_step_budget=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["family_probe"]
    click.echo("Experimental numerical family engineering candidate")
    click.echo(f"  family: {packet['family_kind']}")
    click.echo(f"  cases: {packet['case_count']}")
    click.echo(
        "  duration_satisfied_cases: "
        f"{packet['duration_satisfied_case_count']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-reproducibility")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--repeat-count",
    type=int,
    default=2,
    show_default=True,
    help="Number of identical package-native reruns to hash and compare.",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Initial step budget before optional auto-step-budget adjustment.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="vacuum-cfl",
    show_default=True,
    help="Timestep policy for every rerun.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=4,
    show_default=True,
    help="Cap retained full step results per rerun.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-9,
    show_default=True,
    help="Requested duration for each deterministic rerun.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=True,
    show_default=True,
    help="Set each rerun step budget to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for auto-step-budget reruns.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_reproducibility(
    deck: Path | None,
    deck_preset: str,
    repeat_count: int,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Run identical package-native reruns and hash their observables."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_reproducibility_payload(
        runtime_deck,
        repeat_count=repeat_count,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["reproducibility_probe"]
    click.echo("Experimental reproducibility engineering candidate")
    click.echo(f"  reruns: {packet['run_count']}")
    click.echo(
        "  all_state_observable_hashes_identical: "
        f"{packet['all_state_observable_hashes_identical']}"
    )
    click.echo(
        "  checkpoint_restart_available: "
        f"{packet['checkpoint_restart']['available']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-state-checkpoint")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Initial step budget before optional auto-step-budget adjustment.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="vacuum-cfl",
    show_default=True,
    help="Timestep policy for the checkpoint run.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=4,
    show_default=True,
    help="Cap retained full step results.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-9,
    show_default=True,
    help="Requested duration for the checkpoint run.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=True,
    show_default=True,
    help="Set step budget to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for auto-step-budget.",
)
@click.option(
    "--checkpoint-output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Write the terminal state checkpoint NPZ artifact.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_state_checkpoint(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    checkpoint_output: Path,
    output: Path | None,
) -> None:
    """Write/read a terminal state checkpoint and compare content hashes."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_state_checkpoint_payload(
        runtime_deck,
        checkpoint_output=checkpoint_output,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["checkpoint_probe"]
    click.echo("Experimental state-checkpoint engineering candidate")
    click.echo(f"  checkpoint: {packet['checkpoint_path']}")
    click.echo(f"  write_read_hashes_match: {packet['write_read_hashes_match']}")
    click.echo(
        "  live_restart_available: "
        f"{packet['restart_acceptance']['can_restart_live_runner_from_checkpoint']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-split-continuation")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--steps",
    type=int,
    default=6,
    show_default=True,
    help="Total uninterrupted and split-continuation step budget.",
)
@click.option(
    "--split-after-steps",
    type=int,
    default=3,
    show_default=True,
    help="Run the first split segment for this many steps.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="deck",
    show_default=True,
    help="Timestep policy for the continuation comparison.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=1,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=8,
    show_default=True,
    help="Cap retained full step results.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=None,
    help=(
        "Optional requested duration used only with --auto-step-budget to choose "
        "the total step count; the split comparison itself is fixed-step."
    ),
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=False,
    show_default=True,
    help="Set total steps to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for auto-step-budget.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_split_continuation(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    split_after_steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float | None,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Compare uninterrupted N-step run against A+B live continuation."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_split_continuation_payload(
        runtime_deck,
        split_after_steps=split_after_steps,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["split_continuation_probe"]
    click.echo("Experimental split-continuation engineering candidate")
    click.echo(f"  total_steps: {packet['total_steps']}")
    click.echo(f"  split_after_steps: {packet['split_after_steps']}")
    click.echo(f"  state_fingerprints_match: {packet['state_fingerprints_match']}")
    click.echo(
        "  tracked_observables_match_exactly: "
        f"{packet['tracked_observables_match_exactly']}"
    )
    click.echo(f"  checkpoint_restart_available: {packet['checkpoint_restart']['available']}")
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-checkpoint-restart")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--steps",
    type=int,
    default=6,
    show_default=True,
    help="Total uninterrupted and checkpoint-restarted step budget.",
)
@click.option(
    "--split-after-steps",
    type=int,
    default=3,
    show_default=True,
    help="Write the checkpoint after this many steps.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="deck",
    show_default=True,
    help="Timestep policy for the checkpoint-restart comparison.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=1,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=8,
    show_default=True,
    help="Cap retained full step results.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=None,
    help=(
        "Optional requested duration used only with --auto-step-budget to choose "
        "the total step count; the restart comparison itself is fixed-step."
    ),
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=False,
    show_default=True,
    help="Set total steps to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for auto-step-budget.",
)
@click.option(
    "--checkpoint-output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Write the intermediate checkpoint NPZ artifact.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_checkpoint_restart(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    split_after_steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float | None,
    auto_step_budget: bool,
    max_auto_steps: int,
    checkpoint_output: Path,
    output: Path | None,
) -> None:
    """Compare uninterrupted N-step run against checkpoint-loaded restart."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_checkpoint_restart_payload(
        runtime_deck,
        split_after_steps=split_after_steps,
        checkpoint_output=checkpoint_output,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["checkpoint_restart_probe"]
    click.echo("Experimental checkpoint-restart engineering candidate")
    click.echo(f"  total_steps: {packet['total_steps']}")
    click.echo(f"  split_after_steps: {packet['split_after_steps']}")
    click.echo(f"  checkpoint: {packet['checkpoint_path']}")
    click.echo(f"  state_fingerprints_match: {packet['state_fingerprints_match']}")
    click.echo(
        "  tracked_observables_match_exactly: "
        f"{packet['tracked_observables_match_exactly']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-checkpoint-restart-family")
@click.option(
    "--deck",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional JSON input deck. Defaults to the built-in PF-1000/Akel "
        "16 kV engineering deck."
    ),
)
@click.option(
    "--deck-preset",
    type=click.Choice(FIRST_PRINCIPLES_3D_DECK_PRESETS, case_sensitive=False),
    default="pf1000_akel_16kv",
    show_default=True,
    help="Built-in source-scoped engineering deck preset.",
)
@click.option(
    "--steps",
    type=int,
    default=6,
    show_default=True,
    help="Total uninterrupted and checkpoint-restarted step budget.",
)
@click.option(
    "--split-after-steps",
    type=str,
    default="2,3,4",
    show_default=True,
    help="Comma-separated checkpoint split offsets.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="deck",
    show_default=True,
    help="Timestep policy for the checkpoint-restart family.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by --dt-policy vacuum-cfl.",
)
@click.option(
    "--history-stride",
    type=int,
    default=1,
    show_default=True,
    help="Retain every Nth full step result while counting every completed step.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=8,
    show_default=True,
    help="Cap retained full step results.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=None,
    help=(
        "Optional requested duration used only with --auto-step-budget to choose "
        "the total step count; each restart comparison itself is fixed-step."
    ),
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=False,
    show_default=True,
    help="Set total steps to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Safety cap for auto-step-budget.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for intermediate checkpoint NPZ artifacts.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def experimental_checkpoint_restart_family(
    deck: Path | None,
    deck_preset: str,
    steps: int,
    split_after_steps: str,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float | None,
    auto_step_budget: bool,
    max_auto_steps: int,
    checkpoint_dir: Path,
    output: Path | None,
) -> None:
    """Run checkpoint-loaded restart probes for multiple split offsets."""
    runtime_deck = _load_first_principles_3d_deck(deck, deck_preset=deck_preset)
    runtime_deck = _override_first_principles_3d_deck_runtime(
        runtime_deck,
        steps=steps,
        dt_s=dt_s,
        history_stride=history_stride,
        max_step_results=max_step_results,
        target_time_s=target_time_s,
    )
    runtime_deck = _apply_experimental_dt_policy(
        runtime_deck,
        dt_policy=dt_policy.lower(),
        vacuum_cfl=vacuum_cfl,
    )
    runtime_deck = _apply_experimental_auto_step_budget(
        runtime_deck,
        enabled=auto_step_budget,
        max_auto_steps=max_auto_steps,
    )
    payload = _experimental_checkpoint_restart_family_payload(
        runtime_deck,
        split_offsets=_parse_positive_int_tuple(
            split_after_steps,
            param_hint="--split-after-steps",
        ),
        checkpoint_dir=checkpoint_dir,
    )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    packet = payload["checkpoint_restart_family_probe"]
    click.echo("Experimental checkpoint-restart family engineering candidate")
    click.echo(f"  total_steps: {packet['total_steps']}")
    click.echo(f"  cases: {packet['case_count']}")
    click.echo(f"  matching_cases: {packet['matching_case_count']}")
    click.echo(f"  all_cases_match: {packet['all_cases_match']}")
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("first-principles-gv-waveform")
@click.option(
    "--shot-id",
    default="pf24_krakow_16092202",
    show_default=True,
    help="Verified GV shot id to extract.",
)
@click.option(
    "--series",
    type=click.Choice(["preferred", "raw", "measured", "smoothed"], case_sensitive=False),
    default="preferred",
    show_default=True,
    help="Workbook waveform series to extract.",
)
@click.option(
    "--summary",
    is_flag=True,
    help="Emit a compact manifest without full waveform arrays.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON packet. Without this option, JSON is printed to stdout.",
)
def first_principles_gv_waveform(
    shot_id: str,
    series: str,
    summary: bool,
    output: Path | None,
) -> None:
    """Extract a non-promoting GV workbook current-waveform target packet."""

    from dpf.first_principles import (
        extract_gv_current_waveform_packet,
        gv_waveform_packet_summary,
    )

    packet = extract_gv_current_waveform_packet(shot_id, series=series.lower())
    payload = gv_waveform_packet_summary(packet) if summary else packet

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    click.echo("GV workbook current-waveform candidate packet")
    click.echo(f"  shot_id: {packet['shot_id']}")
    click.echo(f"  series: {packet['digitized_series'][0]['name']}")
    click.echo(f"  points: {packet['summary']['point_count']}")
    click.echo("  scientific_status: candidate_not_validation")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-inverse-parameters")
@click.option(
    "--scope",
    type=click.Choice(["all", "pf1000", "may15", "gv"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Machine-source group to algebraically complete.",
)
@click.option(
    "--include-gv-waveforms/--no-include-gv-waveforms",
    default=True,
    show_default=True,
    help=(
        "Extract verified GV workbook current waveforms as candidate algebraic "
        "observables when the local bundle is present."
    ),
)
@click.option(
    "--gv-series",
    type=click.Choice(["preferred", "raw", "measured", "smoothed"], case_sensitive=False),
    default="preferred",
    show_default=True,
    help="GV workbook waveform series for current-derived candidate fills.",
)
@click.option(
    "--allow-gv-hash-mismatch",
    is_flag=True,
    help="Record GV waveform extraction even when the workbook hash differs.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON packet. Without this option, JSON is printed to stdout.",
)
def experimental_inverse_parameters(
    scope: str,
    include_gv_waveforms: bool,
    gv_series: str,
    allow_gv_hash_mismatch: bool,
    output: Path | None,
) -> None:
    """Build a non-promoting algebraic missing-parameter completion packet."""

    from dpf.first_principles import build_experimental_inverse_parameter_packet

    packet = build_experimental_inverse_parameter_packet(
        scope=scope.lower(),
        include_gv_waveforms=include_gv_waveforms,
        gv_series=gv_series.lower(),
        require_gv_hash_match=not allow_gv_hash_mismatch,
    )

    import json

    text = json.dumps(packet, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    click.echo("Experimental inverse-parameter completion packet")
    click.echo(f"  scope: {packet['scope']}")
    click.echo(f"  machines: {packet['machine_count']}")
    click.echo(f"  unresolved_parameters: {packet['unresolved_parameter_count']}")
    click.echo(
        "  contradictions_or_scope_mismatches: "
        f"{packet['contradiction_or_scope_mismatch_count']}"
    )
    click.echo(f"  status: {packet['status']}")
    click.echo(f"  artifact: {output}")


@cli.command("experimental-inverse-calibration")
@click.option(
    "--deck-preset",
    type=click.Choice(
        ("all", "ir_mpf_100", "compact_chinese_dpf", "gv_pf24_krakow_16092202"),
        case_sensitive=False,
    ),
    default="compact_chinese_dpf",
    show_default=True,
    help=(
        "Source-backed machine to calibrate. 'all' runs every preset with a "
        "typed calibration observable and skips missing-observable presets."
    ),
)
@click.option(
    "--parameters",
    type=str,
    default="inductance",
    show_default=True,
    help="Comma-separated candidate parameters: inductance,resistance,voltage,pressure.",
)
@click.option(
    "--candidate-scales",
    type=str,
    default="0.75,1,1.25",
    show_default=True,
    help="Comma-separated multiplicative factors around source/deck values.",
)
@click.option(
    "--parameter-scale",
    "parameter_scale_specs",
    type=str,
    multiple=True,
    help=(
        "Parameter-specific scale list, repeatable: "
        "--parameter-scale inductance=0.75,1 --parameter-scale resistance=0.5,1,2. "
        "Unspecified parameters use --candidate-scales."
    ),
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Step budget before optional auto-step-budget adjustment.",
)
@click.option(
    "--dt-s",
    type=float,
    default=None,
    help="Override the deck timestep before dt-policy.",
)
@click.option(
    "--dt-policy",
    type=click.Choice(
        ["deck", "vacuum-cfl", "ohmic-cfl", "combined-cfl"],
        case_sensitive=False,
    ),
    default="vacuum-cfl",
    show_default=True,
    help="Experimental timestep policy for every candidate.",
)
@click.option(
    "--vacuum-cfl",
    type=float,
    default=0.95,
    show_default=True,
    help="CFL fraction used by vacuum/combined CFL policies.",
)
@click.option(
    "--history-stride",
    type=int,
    default=5,
    show_default=True,
    help="Retain every Nth circuit-history sample.",
)
@click.option(
    "--max-step-results",
    type=int,
    default=4,
    show_default=True,
    help="Cap retained full step results per candidate.",
)
@click.option(
    "--target-time-s",
    type=float,
    default=1.0e-9,
    show_default=True,
    help="Requested target horizon for every candidate.",
)
@click.option(
    "--auto-step-budget/--no-auto-step-budget",
    default=True,
    show_default=True,
    help="Set each candidate step budget to ceil(target_time_s / dt_s).",
)
@click.option(
    "--max-auto-steps",
    type=int,
    default=5000,
    show_default=True,
    help="Per-candidate safety cap for auto-step-budget.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON packet. Without this option, JSON is printed to stdout.",
)
def experimental_inverse_calibration(
    deck_preset: str,
    parameters: str,
    candidate_scales: str,
    parameter_scale_specs: tuple[str, ...],
    steps: int,
    dt_s: float | None,
    dt_policy: str,
    vacuum_cfl: float,
    history_stride: int,
    max_step_results: int,
    target_time_s: float,
    auto_step_budget: bool,
    max_auto_steps: int,
    output: Path | None,
) -> None:
    """Infer candidate parameters by fitting simulated observables to source targets."""
    if steps <= 0:
        raise click.BadParameter("must be positive", param_hint="--steps")
    if target_time_s <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--target-time-s")
    if history_stride <= 0:
        raise click.BadParameter("must be positive", param_hint="--history-stride")
    if max_step_results < 0:
        raise click.BadParameter(
            "must be non-negative",
            param_hint="--max-step-results",
        )
    if max_auto_steps <= 0:
        raise click.BadParameter(
            "must be positive",
            param_hint="--max-auto-steps",
        )
    parsed_parameters = _parse_calibration_parameter_tuple(parameters)
    parsed_scales = _parse_positive_float_tuple(
        candidate_scales,
        param_hint="--candidate-scales",
        min_count=1 if parameter_scale_specs else 2,
    )
    parsed_parameter_scales = _parse_calibration_parameter_scales(
        parameter_scale_specs,
        parameter_names=parsed_parameters,
        default_scales=parsed_scales,
    )
    if deck_preset.lower() == "all":
        payload = _experimental_inverse_calibration_family_payload(
            deck_presets=(
                "ir_mpf_100",
                "compact_chinese_dpf",
                "gv_pf24_krakow_16092202",
            ),
            parameter_names=parsed_parameters,
            scale_values=parsed_scales,
            parameter_scale_values=parsed_parameter_scales,
            steps=steps,
            dt_s=dt_s,
            dt_policy=dt_policy.lower(),
            vacuum_cfl=vacuum_cfl,
            history_stride=history_stride,
            max_step_results=max_step_results,
            target_time_s=target_time_s,
            auto_step_budget=auto_step_budget,
            max_auto_steps=max_auto_steps,
        )
    else:
        payload = _experimental_inverse_calibration_payload(
            deck_preset=deck_preset.lower(),
            parameter_names=parsed_parameters,
            scale_values=parsed_scales,
            parameter_scale_values=parsed_parameter_scales,
            steps=steps,
            dt_s=dt_s,
            dt_policy=dt_policy.lower(),
            vacuum_cfl=vacuum_cfl,
            history_stride=history_stride,
            max_step_results=max_step_results,
            target_time_s=target_time_s,
            auto_step_budget=auto_step_budget,
            max_auto_steps=max_auto_steps,
        )

    import json

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    click.echo("Experimental inverse calibration packet")
    click.echo(f"  deck_preset: {deck_preset}")
    click.echo(f"  status: {payload['status']}")
    if payload["status"] == "experimental_inverse_calibration_family_not_validation":
        click.echo(f"  completed_calibrations: {payload['completed_calibration_count']}")
        click.echo(f"  skipped_calibrations: {payload['skipped_calibration_count']}")
    else:
        click.echo(f"  candidates: {payload['candidate_count']}")
        click.echo(
            "  identifiability: "
            f"{payload['identifiability']['status']}"
        )
    click.echo(f"  artifact: {output}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def verify(config_file: str) -> None:
    """Verify a configuration file is valid."""
    from dpf.config import SimulationConfig

    try:
        config = SimulationConfig.from_file(config_file)
        click.echo("Configuration is valid:")
        click.echo(f"  Grid: {config.grid_shape}")
        click.echo(f"  dx: {config.dx:.2e} m")
        click.echo(f"  sim_time: {config.sim_time:.2e} s")
        click.echo(f"  Circuit: C={config.circuit.C:.2e} F, V0={config.circuit.V0:.1f} V")
        click.echo(f"  Fluid: {config.fluid.reconstruction}, CFL={config.fluid.cfl}")
        click.echo(f"  Backend: {config.fluid.backend}")
    except Exception as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)


@cli.command()
def backends() -> None:
    """Show available MHD solver backends."""
    from dpf.athena_wrapper import is_available as athena_available
    from dpf.athenak_wrapper import is_available as athenak_available

    click.echo("Available backends:")
    click.echo("  python  — NumPy/Numba MHD solver (always available)")

    if athena_available():
        click.echo("  athena  — Athena++ C++ MHD solver (available)")
    else:
        click.echo("  athena  — Athena++ C++ MHD solver (not compiled)")

    if athenak_available():
        click.echo("  athenak — AthenaK Kokkos MHD solver (available)")
    else:
        click.echo("  athenak — AthenaK Kokkos MHD solver (not built)")

    # Metal GPU backend
    try:
        from dpf.metal.metal_solver import MetalMHDSolver
        if MetalMHDSolver.is_available():
            click.echo("  metal   — Apple Metal GPU MHD solver (available)")
        else:
            click.echo("  metal   — Apple Metal GPU MHD solver (no MPS device)")
    except ImportError:
        click.echo("  metal   — Apple Metal GPU MHD solver (not installed)")

    # MLX Metal v2 backend
    try:
        from dpf.metal.mlx_solver import MLXMHDSolver
        if MLXMHDSolver.is_available():
            click.echo("  mlx     — MLX Metal v2 MHD solver (available)")
        else:
            click.echo("  mlx     — MLX Metal v2 MHD solver (not available)")
    except ImportError:
        click.echo("  mlx     — MLX Metal v2 MHD solver (not installed)")

    click.echo("\nDefault: python")
    if athenak_available():
        click.echo("Auto selection: athenak (preferred when available)")
    elif athena_available():
        click.echo("Auto selection: athena (preferred when available)")


@cli.command("metal-info")
def metal_info() -> None:
    """Show Apple Silicon Metal GPU capabilities."""
    try:
        from dpf.metal.device import get_device_manager
        dm = get_device_manager()
        click.echo(dm.summary())
    except ImportError:
        click.echo("Metal module not installed. Install with: pip install mlx torch")
    except Exception as e:
        click.echo(f"Error detecting Metal capabilities: {e}")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", type=int, default=8765, help="Port number.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev only).")
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="WALRUS checkpoint directory (contains walrus.pt + extended_config.yaml).",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "mps", "cuda"]),
    default="cpu",
    help="Device for AI inference.",
)
def serve(host: str, port: int, reload: bool, checkpoint: str | None, device: str) -> None:
    """Start the DPF simulation server (FastAPI + WebSocket)."""
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Server dependencies not installed. Run:\n"
            "  pip install dpf-unified[server]\n"
            "or:\n"
            "  pip install fastapi uvicorn websockets",
            err=True,
        )
        sys.exit(1)

    if checkpoint:
        click.echo(f"Loading WALRUS model from {checkpoint} on {device} ...")
        from dpf.ai.realtime_server import load_surrogate

        load_surrogate(checkpoint, device=device)
        click.echo("WALRUS model loaded successfully")

    click.echo(f"Starting DPF server on {host}:{port}")
    click.echo(f"  REST API: http://{host}:{port}/api/health")
    click.echo(f"  WebSocket: ws://{host}:{port}/ws/{{sim_id}}")
    click.echo(f"  Docs: http://{host}:{port}/docs")
    if checkpoint:
        click.echo(f"  AI Status: http://{host}:{port}/api/ai/status")

    uvicorn.run(
        "dpf.server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ── AI / ML commands ─────────────────────────────────────────────


@cli.command("export-well")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="well_output.h5", help="Output HDF5 file.")
@click.option("--field-interval", type=int, default=10, help="Steps between field snapshots.")
@click.option("--steps", type=int, default=None, help="Max timesteps.")
@click.option(
    "--backend",
    type=click.Choice(
        ["python", "athena", "athenak", "metal", "mlx", "hybrid", "auto"],
        case_sensitive=False,
    ),
    default=None,
)
@click.option("--artifact-owner", type=str, default=None, help="Owner recorded in Well artifact metadata.")
@click.option(
    "--artifact-classification",
    type=str,
    default="owner_unspecified",
    help="Owner-supplied artifact classification label for Well output.",
)
@click.option(
    "--artifact-distribution",
    type=str,
    default="owner_unspecified",
    help="Owner-supplied distribution scope for Well output.",
)
@click.option(
    "--artifact-handling-notes",
    type=str,
    default="",
    help="Owner-supplied handling notes for Well output.",
)
def export_well(
    config_file: str,
    output: str,
    field_interval: int,
    steps: int | None,
    backend: str | None,
    artifact_owner: str | None,
    artifact_classification: str,
    artifact_distribution: str,
    artifact_handling_notes: str,
) -> None:
    """Run a simulation and export to Well format for WALRUS training."""
    from dpf.ai.well_exporter import WellExporter
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    config = SimulationConfig.from_file(config_file)
    if backend:
        config.fluid.backend = backend

    engine = SimulationEngine(config)
    click.echo(f"Running simulation (backend={engine.backend}) ...")

    exporter = WellExporter(
        output_path=output,
        grid_shape=tuple(config.grid_shape),
        dx=config.dx,
        dz=config.geometry.dz,
        geometry=config.geometry.type,
        sim_params={
            "V0": config.circuit.V0,
            "C": config.circuit.C,
            "backend": engine.backend,
            "validation_status": "not_validation_evidence",
            "result_label": "Preview",
        },
        artifact_classification={
            "owner": artifact_owner,
            "classification": artifact_classification,
            "distribution": artifact_distribution,
            "handling_notes": artifact_handling_notes,
        },
    )

    step_count = 0
    while True:
        result = engine.step()
        step_count += 1
        if step_count % field_interval == 0:
            snapshot = engine.get_field_snapshot()
            exporter.add_snapshot(
                snapshot, result.time,
                {"current": result.current, "voltage": result.voltage},
            )
        if result.finished or (steps and step_count >= steps):
            break

    path = exporter.finalize()
    click.echo(f"Exported {exporter.n_snapshots} snapshots to {path}")


@cli.command()
@click.argument("sweep_config", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="sweep_output", help="Output directory.")
@click.option("--workers", "-w", type=int, default=4, help="Parallel workers.")
def sweep(sweep_config: str, output: str, workers: int) -> None:
    """Run a parameter sweep to generate WALRUS training data."""
    import json as json_mod

    from dpf.ai.batch_runner import BatchRunner, ParameterRange
    from dpf.config import SimulationConfig

    with open(sweep_config) as f:
        sweep_data = json_mod.load(f)

    base_config = SimulationConfig(**sweep_data["base_config"])
    ranges = [
        ParameterRange(
            name=r["name"], low=r["low"], high=r["high"],
            log_scale=r.get("log_scale", False),
        )
        for r in sweep_data.get("parameter_ranges", [])
    ]
    n_samples = sweep_data.get("n_samples", 100)

    runner = BatchRunner(
        base_config=base_config,
        parameter_ranges=ranges,
        n_samples=n_samples,
        output_dir=output,
        workers=workers,
    )
    click.echo(f"Running {n_samples} samples with {workers} workers ...")
    result = runner.run()
    click.echo(f"Done: {result.n_success}/{result.n_total} succeeded")
    if result.failed_configs:
        click.echo(f"  {result.n_failed} failures", err=True)


@cli.command("validate-dataset")
@click.argument("directory", type=click.Path(exists=True))
def validate_dataset(directory: str) -> None:
    """Validate a Well-format training dataset."""
    from dpf.ai.dataset_validator import DatasetValidator

    validator = DatasetValidator()
    results = validator.validate_directory(directory)
    report = validator.summary_report(results)
    click.echo(report)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="WALRUS checkpoint.")
@click.option("--steps", type=int, default=100, help="Rollout steps.")
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default="cpu")
def predict(config_file: str, checkpoint: str, steps: int, device: str) -> None:
    """Run WALRUS surrogate prediction for a configuration."""
    from dpf.ai.surrogate import DPFSurrogate
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    config = SimulationConfig.from_file(config_file)
    surrogate = DPFSurrogate(checkpoint, device=device)
    click.echo(f"Loaded surrogate on {device}")

    # Generate initial states from a short physics run
    engine = SimulationEngine(config)
    history = []
    for _ in range(surrogate.history_length):
        engine.step()
        history.append(engine.get_field_snapshot())

    # Run surrogate rollout
    trajectory = surrogate.rollout(history, n_steps=steps)
    click.echo(f"Rollout complete: {len(trajectory)} steps predicted")


@cli.command()
@click.argument("targets_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="WALRUS checkpoint.")
@click.option(
    "--method", type=click.Choice(["bayesian", "evolutionary"]), default="bayesian",
)
@click.option("--n-trials", type=int, default=100, help="Optimization trials.")
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default="cpu")
def inverse(
    targets_file: str, checkpoint: str, method: str, n_trials: int, device: str,
) -> None:
    """Run inverse design to find configurations matching targets."""
    import json as json_mod

    from dpf.ai.inverse_design import InverseDesigner
    from dpf.ai.surrogate import DPFSurrogate

    with open(targets_file) as f:
        data = json_mod.load(f)

    targets = data["targets"]
    constraints = data.get("constraints", {})
    param_ranges = {k: tuple(v) for k, v in data.get("parameter_ranges", {}).items()}

    surrogate = DPFSurrogate(checkpoint, device=device)
    designer = InverseDesigner(surrogate, parameter_ranges=param_ranges)

    click.echo(f"Running {method} optimization ({n_trials} trials) ...")
    result = designer.find_config(
        targets=targets, constraints=constraints, method=method, n_trials=n_trials,
    )

    click.echo("\n--- Inverse Design Result ---")
    click.echo(f"  Best score: {result.best_score:.6e}")
    for key, val in result.best_params.items():
        click.echo(f"  {key}: {val:.6e}")


@cli.command("serve-ai")
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="WALRUS checkpoint.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", type=int, default=8766, help="Port number.")
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default="cpu")
def serve_ai(checkpoint: str, host: str, port: int, device: str) -> None:
    """Start the AI inference server with a loaded WALRUS model."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Server deps missing. Run: pip install dpf-unified[server]", err=True)
        sys.exit(1)

    from dpf.ai.realtime_server import load_surrogate

    click.echo(f"Loading WALRUS model from {checkpoint} on {device} ...")
    load_surrogate(checkpoint, device=device)

    click.echo(f"Starting AI server on {host}:{port}")
    click.echo(f"  Status: http://{host}:{port}/api/ai/status")
    click.echo(f"  Docs: http://{host}:{port}/docs")

    uvicorn.run(
        "dpf.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


@cli.command()
@click.option("--port", type=int, default=7860, help="Port to bind the web UI (default: 7860).")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--share", is_flag=True, help="Create a public Gradio share link.")
def ui(port: int, host: str, share: bool) -> None:
    """Launch the Gradio web interface."""
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        click.echo(
            "Gradio is not installed. Install it with:\n"
            "  pip install gradio",
            err=True,
        )
        sys.exit(1)

    import importlib.util
    import pathlib

    # app.py lives at the repo root (two levels above this file's package)
    app_path = pathlib.Path(__file__).resolve().parents[3] / "app.py"
    if not app_path.exists():
        click.echo(f"Web UI not found at {app_path}", err=True)
        sys.exit(1)

    click.echo(f"Starting DPF web UI on http://{host}:{port}")
    spec = importlib.util.spec_from_file_location("dpf_app", app_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.app.queue(max_size=5)
    module.app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=module.gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=module.CSS,
    )


@cli.command()
@click.option("--all", "run_all", is_flag=True, help="Validate against all preset devices.")
@click.option(
    "--device",
    type=str,
    default=None,
    help="Single device name to validate (e.g. PF-1000, UNU-ICTP).",
)
@click.option("--sim-time-us", type=float, default=40.0, help="Simulation time in microseconds.")
def validate(run_all: bool, device: str | None, sim_time_us: float) -> None:
    """Validate the Lee model against published device data.

    Use --all to run validation for every preset device, or --device
    to target a single device by name.
    """
    from dpf.validation.experimental import DEVICES

    if not run_all and device is None:
        click.echo(
            "Specify --all to validate all devices, or --device NAME for one.",
            err=True,
        )
        sys.exit(1)

    if device is not None and device not in DEVICES:
        available = ", ".join(sorted(DEVICES.keys()))
        click.echo(f"Unknown device '{device}'. Available: {available}", err=True)
        sys.exit(1)

    targets: list[str] = list(DEVICES.keys()) if run_all else [device]  # type: ignore[list-item]

    # Import simulation runner — avoid heavy imports at module level
    try:
        from app_engine import run_simulation_core
    except ImportError:
        # Fall back to package-relative import when not running from repo root
        try:
            import pathlib
            import sys as _sys
            repo_root = pathlib.Path(__file__).resolve().parents[3]
            if str(repo_root) not in _sys.path:
                _sys.path.insert(0, str(repo_root))
            from app_engine import run_simulation_core
        except ImportError:
            click.echo(
                "Could not import app_engine. Run from the dpf-unified repo root.",
                err=True,
            )
            sys.exit(1)

    # Preset name map: device name -> preset key
    from dpf.presets import list_presets
    preset_info = list_presets()
    device_to_preset: dict[str, str] = {}
    for p in preset_info:
        meta_device = p.get("device", "")
        if meta_device:
            device_to_preset[meta_device] = p["name"]

    # Supplemental manual mappings for name mismatches
    _MANUAL_MAP: dict[str, str] = {
        "UNU-ICTP": "unu_ictp",
        "NX2": "nx2",
        "PF-1000": "pf1000",
    }
    for dev_name, preset_key in _MANUAL_MAP.items():
        device_to_preset.setdefault(dev_name, preset_key)

    col_w = (14, 12, 12, 12, 12, 8, 22, 8)
    header = (
        f"{'Device':<{col_w[0]}}  "
        f"{'I_peak sim':<{col_w[1]}}  "
        f"{'I_peak ref':<{col_w[2]}}  "
        f"{'Error':>{col_w[3]}}  "
        f"{'t_peak sim':<{col_w[4]}}  "
        f"{'Status':<{col_w[5]}}  "
        f"{'Authority':<{col_w[6]}}  "
        f"{'Blockers':>{col_w[7]}}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    def source_authority(result: dict[str, object] | None = None) -> tuple[str, str]:
        from dpf.server.readiness import api_readiness_payload

        readiness = api_readiness_payload(
            backend="python",
            result=result,
            validation_status="not_evaluated",
        )
        classification = readiness.get("result_classification", {})
        label = str(classification.get("label", "Preview"))
        status = str(readiness.get("validation_status", "not_evaluated"))
        blocker_count = len(readiness.get("source_blockers", []))
        return f"{label}/{status}", str(blocker_count)

    any_fail = False
    for dev_name in targets:
        preset_key = device_to_preset.get(dev_name)
        if preset_key is None:
            row = (
                f"{dev_name:<{col_w[0]}}  "
                f"{'n/a':<{col_w[1]}}  "
                f"{'n/a':<{col_w[2]}}  "
                f"{'n/a':>{col_w[3]}}  "
                f"{'n/a':<{col_w[4]}}  "
                f"{'SKIP':<{col_w[5]}}  "
                f"{'n/a':<{col_w[6]}}  "
                f"{'n/a':>{col_w[7]}}"
            )
            click.echo(row)
            continue

        from dpf.validation.experimental import DEVICES as _DEVS
        dev = _DEVS[dev_name]
        if getattr(dev, "reliability", "measured") == "reference_only":
            row = (
                f"{dev_name:<{col_w[0]}}  "
                f"{'n/a':<{col_w[1]}}  "
                f"{'n/a':<{col_w[2]}}  "
                f"{'n/a':>{col_w[3]}}  "
                f"{'n/a':<{col_w[4]}}  "
                f"{'EXCL':<{col_w[5]}}  "
                f"{'source-excluded':<{col_w[6]}}  "
                f"{'n/a':>{col_w[7]}}"
            )
            click.echo(row)
            continue

        try:
            data = run_simulation_core(
                preset_name=preset_key,
                sim_time_us=sim_time_us,
            )
        except Exception as exc:
            row = (
                f"{dev_name:<{col_w[0]}}  "
                f"{'ERROR':<{col_w[1]}}  "
                f"{'n/a':<{col_w[2]}}  "
                f"{'n/a':>{col_w[3]}}  "
                f"{'n/a':<{col_w[4]}}  "
                f"{'FAIL':<{col_w[5]}}  "
                f"{'n/a':<{col_w[6]}}  "
                f"{'n/a':>{col_w[7]}}"
            )
            click.echo(row)
            click.echo(f"  -> {exc}", err=True)
            any_fail = True
            continue

        from app_validation import validate_against_published

        val = validate_against_published(data, preset_key)
        if val is None:
            row = (
                f"{dev_name:<{col_w[0]}}  "
                f"{'n/a':<{col_w[1]}}  "
                f"{'n/a':<{col_w[2]}}  "
                f"{'n/a':>{col_w[3]}}  "
                f"{'n/a':<{col_w[4]}}  "
                f"{'SKIP':<{col_w[5]}}  "
                f"{'n/a':<{col_w[6]}}  "
                f"{'n/a':>{col_w[7]}}"
            )
            click.echo(row)
            continue

        dI = val["I_peak_dev_pct"]
        status = "PASS" if dI <= 5 else "FAIR" if dI <= 15 else "POOR" if dI <= 30 else "FAIL"
        if status == "FAIL":
            any_fail = True

        I_sim = val["I_peak_sim_MA"]
        I_ref = val["I_peak_ref_MA"]
        t_sim = val["t_peak_sim_us"]
        authority, blockers = source_authority(
            {
                "backend": "python",
                "step": data.get("n_steps", 0),
                "current": I_sim * 1e6,
                "validation_status": "not_evaluated",
            }
        )

        row = (
            f"{dev_name:<{col_w[0]}}  "
            f"{I_sim:.3f} MA   "
            f"{I_ref:.3f} MA   "
            f"{dI:>6.1f}%   "
            f"{t_sim:.1f} us     "
            f"{status:<{col_w[5]}}  "
            f"{authority:<{col_w[6]}}  "
            f"{blockers:>{col_w[7]}}"
        )
        click.echo(row)

    click.echo(
        "\nSource authority: PASS/FAIR/POOR are peak-current engineering grades. "
        "Reference validation requires accepted KnowledgeReference evidence and "
        "same-scope source gates."
    )

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    cli()
