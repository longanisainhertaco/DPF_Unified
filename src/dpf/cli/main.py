"""Command-line interface for the DPF simulator.

Usage:
    dpf simulate config.json --steps=100
    dpf verify config.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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
def simulate(config_file: str, steps: int | None, output: str | None) -> None:
    """Run a DPF simulation from a configuration file."""
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    click.echo(f"Loading config from {config_file}")
    config = SimulationConfig.from_file(config_file)

    if output:
        config.diagnostics.hdf5_filename = output

    engine = SimulationEngine(config)
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
        click.echo(f"Configuration is valid:")
        click.echo(f"  Grid: {config.grid_shape}")
        click.echo(f"  dx: {config.dx:.2e} m")
        click.echo(f"  sim_time: {config.sim_time:.2e} s")
        click.echo(f"  Circuit: C={config.circuit.C:.2e} F, V0={config.circuit.V0:.1f} V")
        click.echo(f"  Fluid: {config.fluid.reconstruction}, CFL={config.fluid.cfl}")
    except Exception as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
