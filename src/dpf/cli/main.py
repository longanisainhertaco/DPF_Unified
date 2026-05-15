"""Command-line interface for the DPF simulator.

Usage:
    dpf simulate config.json --steps=100
    dpf verify config.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import click


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


def _load_first_principles_runner():
    """Load the app-backed first-principles runner from a local checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import importlib.util

    app_path = repo_root / "app_mhd.py"
    if not app_path.exists():
        raise click.ClickException(
            "first-principles runner requires the local checkout app_mhd.py path"
        )
    spec = importlib.util.spec_from_file_location("app_mhd", app_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"unable to load first-principles runner at {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_pf1000_akel_first_principles


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


def _load_first_principles_3d_deck(deck_path: Path | None) -> dict[str, Any]:
    if deck_path is None:
        deck = _default_first_principles_3d_deck()
        deck["source"] = "built_in"
        return deck

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
        apply_circuit_boundary = package_deck.closures.apply_circuit_boundary
        deck_summary = {
            "name": package_deck.deck_id,
            "source": deck.get("source", "built_in"),
            "steps": steps,
            "grid_shape": list(shape),
            "dt_s": dt_s,
            "apply_circuit_boundary": apply_circuit_boundary,
            "device_name": package_deck.device.name,
            "scientific_status": package_deck.scientific_status,
        }
        run = run_first_principles_3d_deck(package_deck)
    else:
        shape = _deck_grid_shape(deck)
        steps = _positive_int_deck_value(deck, "steps")
        dt_s = _positive_float_deck_value(deck, "dt_s")
        apply_circuit_boundary = _bool_deck_value(deck, "apply_circuit_boundary")
        deck_summary = {
            "name": str(deck.get("name", "minimal_engineering_3d")),
            "source": deck.get("source", "built_in"),
            "steps": steps,
            "grid_shape": list(shape),
            "dt_s": dt_s,
            "apply_circuit_boundary": apply_circuit_boundary,
        }
        run = run_first_principles_3d_deck(
            {
                "n_steps": steps,
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
        "simulation": simulation,
        "validation_packet": run.validation_packet,
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
    return payload


def _json_series(value: object, *, stride: int) -> list[object]:
    """Convert a scalar/array-like history to a JSON-safe list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[assignment]
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if not isinstance(value, list):
        try:
            value = list(value)  # type: ignore[arg-type]
        except TypeError:
            return [value]
    return value[:: max(int(stride), 1)]


def _float_metric(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _max_abs_series(value: object) -> float:
    values = _json_series(value, stride=1)
    finite = [_float_metric(item) for item in values]
    return max((abs(item) for item in finite), default=0.0)


def _last_series_float(value: object) -> float:
    values = _json_series(value, stride=1)
    return _float_metric(values[-1]) if values else 0.0


def _first_principles_payload(
    result: dict[str, Any],
    *,
    preset: str,
    grid_preset: str,
    sim_time_us: float,
    history_stride: int,
) -> dict[str, Any]:
    """Build a compact first-principles engineering-probe artifact."""
    limiter_counts = _json_series(
        result.get("field_limiter_activation_count"),
        stride=1,
    )
    histories = {
        key: _json_series(result.get(key), stride=history_stride)
        for key in (
            "t_us",
            "I_MA",
            "V_kV",
            "back_emf_V",
            "Lp_field_nH",
            "magnetic_energy_kJ",
            "joule_power_W",
            "joule_energy_kJ",
            "field_energy_residual_kJ",
            "B_max",
            "rho_max",
            "T_max",
            "field_limiter_activation_count",
        )
    }
    readiness = result.get("first_principles_mhd_readiness")
    neutron_authority = result.get("first_principles_neutron_yield_authority")
    backend_scope = result.get("first_principles_backend_scope")
    if not isinstance(backend_scope, dict) and isinstance(readiness, dict):
        backend_scope = readiness.get("backend_scope_status")
    limiter_ledger = result.get("first_principles_limiter_ledger")
    limiter_ledger_summary: dict[str, Any] = {}
    if isinstance(limiter_ledger, dict):
        limiter_ledger_summary = {
            key: value
            for key, value in limiter_ledger.items()
            if key != "entries"
        }
    return {
        "tool": "dpf first-principles",
        "preset": preset,
        "grid_preset": grid_preset,
        "requested_sim_time_us": sim_time_us,
        "run_mode": result.get("run_mode"),
        "execution_backend": result.get("backend"),
        "source_scope": result.get("source_scope"),
        "validation_scope": result.get("validation_scope"),
        "scientific_status": "engineering_probe_not_validation",
        "first_principles_only_enforced": (
            result.get("field_coupled_candidate") is True
            and result.get("has_snowplow") is False
        ),
        "metrics": {
            "n_steps": int(_float_metric(result.get("n_steps"), 0.0)),
            "nan_detected": bool(result.get("nan_detected")),
            "I_peak_MA": _float_metric(result.get("I_peak")),
            "t_peak_us": _float_metric(result.get("t_peak")),
            "back_emf_abs_max_V": _max_abs_series(result.get("back_emf_V")),
            "B_max_T": _max_abs_series(result.get("B_max")),
            "L_field_max_nH": _max_abs_series(result.get("Lp_field_nH")),
            "joule_energy_final_kJ": _last_series_float(result.get("joule_energy_kJ")),
            "field_energy_residual_final_kJ": _last_series_float(
                result.get("field_energy_residual_kJ")
            ),
            "limiter_activation_max": max(
                (int(_float_metric(item)) for item in limiter_counts),
                default=0,
            ),
            "acceptance_blocking_limiter_activation_count": int(
                _float_metric(
                    limiter_ledger_summary.get("acceptance_blocking_activation_count")
                )
            ),
        },
        "readiness": readiness if isinstance(readiness, dict) else {},
        "neutron_yield_authority": (
            neutron_authority if isinstance(neutron_authority, dict) else {}
        ),
        "backend_scope": backend_scope if isinstance(backend_scope, dict) else {},
        "limiter_ledger_summary": limiter_ledger_summary,
        "histories": histories,
    }


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
    help="Candidate run duration in microseconds.",
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
    gas_key: str,
    output: Path | None,
    history_stride: int,
    require_field_feedback: bool,
) -> None:
    """Run the PF-1000/Akel first-principles-only engineering candidate."""
    if sim_time_us <= 0.0:
        raise click.BadParameter("must be positive", param_hint="--sim-time-us")
    if history_stride <= 0:
        raise click.BadParameter("must be positive", param_hint="--history-stride")

    runner = _load_first_principles_runner()
    result = runner(
        grid_preset=grid_preset,
        sim_time_us=sim_time_us,
        gas_key=gas_key,
    )
    payload = _first_principles_payload(
        result,
        preset=preset,
        grid_preset=grid_preset,
        sim_time_us=sim_time_us,
        history_stride=history_stride,
    )

    if output is not None:
        import json

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    metrics = payload["metrics"]
    click.echo("First-principles PF-1000/Akel engineering candidate")
    click.echo(f"  steps: {metrics['n_steps']}")
    click.echo(f"  nan_detected: {metrics['nan_detected']}")
    click.echo(f"  I_peak_MA: {metrics['I_peak_MA']:.6e}")
    click.echo(f"  back_emf_abs_max_V: {metrics['back_emf_abs_max_V']:.6e}")
    click.echo(f"  L_field_max_nH: {metrics['L_field_max_nH']:.6e}")
    click.echo(f"  joule_energy_final_kJ: {metrics['joule_energy_final_kJ']:.6e}")
    click.echo(
        "  readiness: "
        f"{payload.get('readiness', {}).get('status', 'unknown')}"
    )
    if output is not None:
        click.echo(f"  artifact: {output}")

    if payload["first_principles_only_enforced"] is not True:
        raise click.ClickException(
            "first-principles-only enforcement failed: run did not stay on the "
            "field-coupled candidate path"
        )
    if metrics["nan_detected"]:
        raise click.ClickException("first-principles candidate produced nonfinite state")
    if require_field_feedback and metrics["back_emf_abs_max_V"] <= 0.0:
        raise click.ClickException(
            "field-derived back-EMF stayed zero; increase --sim-time-us or inspect "
            "the field-current floor"
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
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON artifact. Without this option, JSON is printed to stdout.",
)
def first_principles_3d(deck: Path | None, output: Path | None) -> None:
    """Run the package-native 3-D first-principles engineering candidate."""
    payload = _first_principles_3d_payload(_load_first_principles_3d_deck(deck))

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
    click.echo(f"  grid_shape: {payload['grid_shape']}")
    click.echo(f"  final_particle_count: {payload['final_particle_count']}")
    click.echo(f"  final_field_energy_J: {payload['final_field_energy_J']:.6e}")
    click.echo(f"  scientific_status: {payload['scientific_status']}")
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
