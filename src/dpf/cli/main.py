"""Command-line interface for the DPF simulator.

Usage:
    dpf simulate config.json --steps=100
    dpf verify config.json
"""

from __future__ import annotations

import logging
import sys

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
    type=click.Choice(["python", "athena", "athenak", "metal", "auto"], case_sensitive=False),
    default=None,
    help="MHD solver backend. Overrides config file setting. "
    "'python'=NumPy/Numba, 'athena'=Athena++ C++, 'athenak'=AthenaK Kokkos, 'auto'=best available.",
)
def simulate(
    config_file: str,
    steps: int | None,
    output: str | None,
    restart: str | None,
    checkpoint_interval: int,
    backend: str | None,
) -> None:
    """Run a DPF simulation from a configuration file."""
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    click.echo(f"Loading config from {config_file}")
    config = SimulationConfig.from_file(config_file)

    if backend:
        config.fluid.backend = backend

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
@click.option("--host", default="127.0.0.1", help="Bind address.")
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
@click.option("--backend", type=click.Choice(["python", "athena", "athenak", "auto"]), default=None)
def export_well(
    config_file: str, output: str, field_interval: int, steps: int | None, backend: str | None,
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
        sim_params={"V0": config.circuit.V0, "C": config.circuit.C},
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
    click.echo(f"Exported {len(exporter._times)} snapshots to {path}")


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
@click.option("--host", default="127.0.0.1", help="Bind address.")
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


if __name__ == "__main__":
    cli()
