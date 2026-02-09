#!/usr/bin/env python3
"""Generate WALRUS training data from DPF simulations.

Runs batch DPF simulations across a diverse parameter space and exports
field trajectories in The Well HDF5 format for WALRUS fine-tuning.

Usage:
    python scripts/generate_training_data.py --n-samples 50 --output-dir training_data/dpf
    python scripts/generate_training_data.py --preset pf1000 --n-samples 10
    python scripts/generate_training_data.py --quick  # 10 fast trajectories for testing

The parameter space covers:
    - Voltage: 5-40 kV (typical DPF range)
    - Capacitance: 0.5-1500 μF (small lab to large facility)
    - Inductance: 5-200 nH (connection + collector plate)
    - Fill density: 1e-5 to 1e-2 kg/m³ (0.5-20 Torr deuterium)
    - Anode radius: 3-60 mm (Mather-type geometry)
    - Cathode radius: scaled from anode (ratio 1.3-2.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dpf.ai.batch_runner import BatchResult, BatchRunner, ParameterRange  # noqa: E402
from dpf.config import SimulationConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_training_data")


# ── Parameter space definitions ──────────────────────────────────────

# Small/fast: for testing the pipeline
QUICK_RANGES = [
    ParameterRange("circuit.V0", 5e3, 20e3, log_scale=False),
    ParameterRange("circuit.C", 1e-6, 50e-6, log_scale=True),
    ParameterRange("rho0", 1e-5, 1e-3, log_scale=True),
]

# Full DPF parameter space
FULL_RANGES = [
    ParameterRange("circuit.V0", 5e3, 40e3, log_scale=False),
    ParameterRange("circuit.C", 0.5e-6, 1.5e-3, log_scale=True),
    ParameterRange("circuit.L0", 5e-9, 200e-9, log_scale=True),
    ParameterRange("circuit.R0", 1e-3, 50e-3, log_scale=True),
    ParameterRange("rho0", 1e-5, 1e-2, log_scale=True),
    ParameterRange("circuit.anode_radius", 3e-3, 60e-3, log_scale=True),
]

# Device-specific parameter ranges (focused around known devices)
PF1000_RANGES = [
    ParameterRange("circuit.V0", 20e3, 35e3, log_scale=False),
    ParameterRange("circuit.C", 0.8e-3, 2e-3, log_scale=True),
    ParameterRange("rho0", 1e-4, 8e-4, log_scale=True),
    ParameterRange("circuit.L0", 10e-9, 30e-9, log_scale=True),
]

NX2_RANGES = [
    ParameterRange("circuit.V0", 8e3, 16e3, log_scale=False),
    ParameterRange("circuit.C", 0.5e-6, 2e-6, log_scale=True),
    ParameterRange("rho0", 3e-5, 2e-4, log_scale=True),
    ParameterRange("circuit.L0", 10e-9, 40e-9, log_scale=True),
]


def create_base_config(
    grid_shape: list[int],
    dx: float,
    sim_time: float,
    geometry: str = "cylindrical",
) -> SimulationConfig:
    """Create a base simulation config for training data generation.

    Parameters
    ----------
    grid_shape : list[int]
        Grid dimensions [nx, ny, nz]
    dx : float
        Grid spacing [m]
    sim_time : float
        Total simulation time [s]
    geometry : str
        Coordinate system ("cylindrical" or "cartesian")
    """
    config_dict = {
        "grid_shape": grid_shape,
        "dx": dx,
        "sim_time": sim_time,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "circuit": {
            "C": 1e-6,
            "V0": 10e3,
            "L0": 20e-9,
            "R0": 5e-3,
            "anode_radius": 0.01,
            "cathode_radius": 0.025,
        },
        "fluid": {
            "backend": "python",
            "reconstruction": "weno5",
            "riemann_solver": "hll",
            "cfl": 0.3,
            "enable_powell": True,
            "enable_viscosity": True,
            "full_braginskii_viscosity": True,
            "enable_resistive": True,
            "enable_energy_equation": True,
        },
        "radiation": {
            "bremsstrahlung_enabled": True,
            "fld_enabled": False,
        },
        "diagnostics": {
            "hdf5_filename": ":memory:",
            "output_interval": 10,
            "field_output_interval": 0,
        },
    }

    if geometry == "cylindrical":
        config_dict["geometry"] = {"type": "cylindrical"}
        # Cylindrical requires ny=1 (axisymmetric): [nr, 1, nz]
        config_dict["grid_shape"] = [grid_shape[0], 1, grid_shape[2]]

    return SimulationConfig(**config_dict)


def ensure_valid_geometry(params: dict[str, float]) -> dict[str, float]:
    """Ensure cathode_radius > anode_radius by a factor of 1.3-2.5.

    If cathode_radius is not in params, set it based on anode_radius.
    """
    params = dict(params)  # Don't modify original

    anode_r = params.get("circuit.anode_radius")
    if anode_r is not None and "circuit.cathode_radius" not in params:
        # Set cathode_radius = 2.0 * anode_radius (Mather-type geometry)
        params["circuit.cathode_radius"] = anode_r * 2.0

    return params


class TrainingDataGenerator:
    """Orchestrate training data generation for WALRUS fine-tuning."""

    def __init__(
        self,
        output_dir: str | Path,
        n_samples: int = 100,
        grid_shape: list[int] | None = None,
        dx: float = 5e-4,
        sim_time: float = 2e-6,
        geometry: str = "cylindrical",
        field_interval: int = 5,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_samples = n_samples
        self.grid_shape = grid_shape or [32, 1, 64]
        self.dx = dx
        self.sim_time = sim_time
        self.geometry = geometry
        self.field_interval = field_interval
        self.workers = workers
        self.seed = seed

    def generate(
        self,
        parameter_ranges: list[ParameterRange] | None = None,
        progress_callback=None,
    ) -> BatchResult:
        """Run the full training data generation pipeline.

        Parameters
        ----------
        parameter_ranges : list[ParameterRange] | None
            Parameter ranges to sample (defaults to FULL_RANGES)
        progress_callback : callable | None
            Optional callback(completed, total)

        Returns
        -------
        BatchResult
            Summary of batch run
        """
        ranges = parameter_ranges or FULL_RANGES

        base_config = create_base_config(
            grid_shape=self.grid_shape,
            dx=self.dx,
            sim_time=self.sim_time,
            geometry=self.geometry,
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Custom build_config that ensures valid geometry
        class GeometryAwareBatchRunner(BatchRunner):
            def build_config(self, params):
                params = ensure_valid_geometry(params)
                return super().build_config(params)

        runner = GeometryAwareBatchRunner(
            base_config=base_config,
            parameter_ranges=ranges,
            n_samples=self.n_samples,
            output_dir=self.output_dir,
            workers=self.workers,
            field_interval=self.field_interval,
        )

        logger.info(
            "Generating %d trajectories: grid=%s, dx=%.1e, sim_time=%.1e, geometry=%s",
            self.n_samples,
            self.grid_shape,
            self.dx,
            self.sim_time,
            self.geometry,
        )
        logger.info("Parameter ranges: %s", [f"{r.name}=[{r.low:.2e}, {r.high:.2e}]" for r in ranges])

        t_start = time.monotonic()
        result = runner.run(progress_callback=progress_callback)
        t_elapsed = time.monotonic() - t_start

        logger.info(
            "Batch complete: %d/%d succeeded in %.1f s (%.1f s/sim avg)",
            result.n_success,
            result.n_total,
            t_elapsed,
            t_elapsed / max(result.n_total, 1),
        )

        if result.n_failed > 0:
            logger.warning("Failed simulations:")
            for idx, err in result.failed_configs:
                logger.warning(f"  [{idx}] {err}")

        # Write metadata
        self._write_metadata(result, ranges, t_elapsed)

        return result

    def _write_metadata(
        self,
        result: BatchResult,
        ranges: list[ParameterRange],
        elapsed: float,
    ) -> None:
        """Write metadata file describing the training dataset."""
        meta = {
            "dataset_name": "dpf_training_data",
            "n_samples": result.n_total,
            "n_success": result.n_success,
            "n_failed": result.n_failed,
            "grid_shape": self.grid_shape,
            "dx": self.dx,
            "sim_time": self.sim_time,
            "geometry": self.geometry,
            "field_interval": self.field_interval,
            "generation_time_s": elapsed,
            "parameter_ranges": {
                r.name: {"low": r.low, "high": r.high, "log_scale": r.log_scale}
                for r in ranges
            },
            "failed_indices": [idx for idx, _ in result.failed_configs],
        }
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Wrote metadata to %s", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WALRUS training data from DPF simulations",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of trajectories to generate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="training_data/dpf",
        help="Output directory for Well HDF5 files",
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, default=[32, 1, 64],
        help="Grid shape [nx ny nz] (default: 32 1 64)",
    )
    parser.add_argument(
        "--dx", type=float, default=5e-4,
        help="Grid spacing [m] (default: 5e-4)",
    )
    parser.add_argument(
        "--sim-time", type=float, default=2e-6,
        help="Simulation time [s] (default: 2e-6)",
    )
    parser.add_argument(
        "--field-interval", type=int, default=5,
        help="Capture field snapshot every N steps (default: 5)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--preset", choices=["full", "pf1000", "nx2", "quick"],
        default="full",
        help="Parameter range preset (default: full)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 10 samples, small grid, short sim time",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS sampling (default: 42)",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.n_samples = 10
        args.grid = [16, 1, 32]
        args.sim_time = 5e-7
        args.preset = "quick"
        args.field_interval = 3

    # Select parameter ranges
    ranges_map = {
        "full": FULL_RANGES,
        "pf1000": PF1000_RANGES,
        "nx2": NX2_RANGES,
        "quick": QUICK_RANGES,
    }
    ranges = ranges_map[args.preset]

    # Progress callback
    def progress(completed: int, total: int) -> None:
        pct = 100.0 * completed / total
        logger.info(f"Progress: {completed}/{total} ({pct:.0f}%)")

    generator = TrainingDataGenerator(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        grid_shape=args.grid,
        dx=args.dx,
        sim_time=args.sim_time,
        geometry="cylindrical",
        field_interval=args.field_interval,
        workers=args.workers,
        seed=args.seed,
    )

    result = generator.generate(parameter_ranges=ranges, progress_callback=progress)

    # Print summary
    print(f"\n{'='*60}")
    print("Training Data Generation Complete")
    print(f"{'='*60}")
    print(f"  Total:     {result.n_total}")
    print(f"  Succeeded: {result.n_success}")
    print(f"  Failed:    {result.n_failed}")
    print(f"  Output:    {result.output_dir}")
    print(f"{'='*60}")

    sys.exit(0 if result.n_failed == 0 else 1)


if __name__ == "__main__":
    main()
