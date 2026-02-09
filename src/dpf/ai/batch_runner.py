from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dpf.ai.well_exporter import WellExporter
from dpf.config import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Parameter range specification for batch sampling.

    Attributes
    ----------
    name : str
        Parameter name (supports nested keys with dot notation, e.g., "circuit.V0")
    low : float
        Lower bound of parameter range
    high : float
        Upper bound of parameter range
    log_scale : bool
        If True, sample uniformly in log space
    """

    name: str
    low: float
    high: float
    log_scale: bool = False


@dataclass
class BatchResult:
    """Results from batch simulation run.

    Attributes
    ----------
    n_total : int
        Total number of simulations attempted
    n_success : int
        Number of successful simulations
    n_failed : int
        Number of failed simulations
    output_dir : str
        Directory where trajectories were saved
    failed_configs : list[tuple[int, str]]
        List of (index, error_message) for failed runs
    """

    n_total: int = 0
    n_success: int = 0
    n_failed: int = 0
    output_dir: str = ""
    failed_configs: list[tuple[int, str]] = field(default_factory=list)


class BatchRunner:
    """Run batch simulations with parameter sweeps for ML training data generation.

    Generates Latin Hypercube samples over parameter ranges, runs simulations,
    and exports field snapshots in WELL format for trajectory optimization.
    """

    def __init__(
        self,
        base_config: SimulationConfig,
        parameter_ranges: list[ParameterRange],
        n_samples: int = 100,
        output_dir: str | Path = "training_data",
        workers: int = 4,
        field_interval: int = 10,
    ) -> None:
        """Initialize batch runner.

        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration to modify for each sample
        parameter_ranges : list[ParameterRange]
            Parameter ranges to sample over
        n_samples : int
            Number of samples to generate
        output_dir : str | Path
            Directory to save trajectory files
        workers : int
            Number of parallel workers (1 for sequential)
        field_interval : int
            Save field snapshots every N timesteps
        """
        self.base_config = base_config
        self.parameter_ranges = parameter_ranges
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.field_interval = field_interval

    def generate_samples(self) -> list[dict[str, float]]:
        """Generate Latin Hypercube samples over parameter ranges.

        Returns
        -------
        list[dict[str, float]]
            List of parameter dictionaries, one per sample
        """
        logger.info(f"Generating {self.n_samples} LHS samples over {len(self.parameter_ranges)} parameters")

        # Generate normalized samples in [0, 1]
        lhs_samples = self._latin_hypercube(self.n_samples, len(self.parameter_ranges))

        # Map to parameter ranges
        samples = []
        for i in range(self.n_samples):
            param_dict = {}
            for j, param_range in enumerate(self.parameter_ranges):
                normalized_value = lhs_samples[i, j]

                if param_range.log_scale:
                    # Log-uniform sampling
                    value = param_range.low * (param_range.high / param_range.low) ** normalized_value
                else:
                    # Linear sampling
                    value = param_range.low + (param_range.high - param_range.low) * normalized_value

                param_dict[param_range.name] = value

            samples.append(param_dict)

        return samples

    def build_config(self, params: dict[str, float]) -> SimulationConfig:
        """Build simulation config by applying parameter overrides to base config.

        Parameters
        ----------
        params : dict[str, float]
            Parameter values to apply (supports nested keys with dot notation)

        Returns
        -------
        SimulationConfig
            Modified configuration
        """
        # Start with base config
        config_dict = self.base_config.model_dump()

        # Apply parameter overrides
        for key, value in params.items():
            if "." in key:
                # Handle nested keys (e.g., "circuit.V0")
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Top-level key
                config_dict[key] = value

        return SimulationConfig(**config_dict)

    def run_single(self, idx: int, params: dict[str, float]) -> tuple[int, str | None]:
        """Run single simulation with given parameters.

        Runs the simulation step-by-step, capturing field snapshots at regular
        intervals directly from the engine state. This avoids dependency on
        HDF5Writer's internal snapshot storage and ensures snapshots are always
        captured regardless of diagnostics config.

        Parameters
        ----------
        idx : int
            Sample index
        params : dict[str, float]
            Parameter values

        Returns
        -------
        tuple[int, str | None]
            (index, error_message) where error_message is None on success
        """
        try:
            # Lazy import to avoid issues with multiprocessing
            from dpf.engine import SimulationEngine

            # Build config and create engine
            config = self.build_config(params)
            engine = SimulationEngine(config)

            logger.info(f"Starting simulation {idx}/{self.n_samples}")

            # Set up Well exporter
            output_path = self.output_dir / f"trajectory_{idx:04d}.h5"
            dz = config.geometry.dz if config.geometry.dz is not None else config.dx
            exporter = WellExporter(
                output_path=output_path,
                grid_shape=tuple(config.grid_shape),
                dx=config.dx,
                dz=dz,
                geometry=config.geometry.type,
                sim_params=params,
            )

            # Capture initial state
            exporter.add_snapshot(
                state=engine.get_field_snapshot(),
                time=0.0,
                circuit_scalars={
                    "current": engine.circuit.current,
                    "voltage": engine.circuit.voltage,
                },
            )

            # Run simulation step-by-step, capturing snapshots at intervals
            step_count = 0
            while True:
                result = engine.step()
                step_count += 1

                # Capture snapshot at interval
                if step_count % self.field_interval == 0:
                    exporter.add_snapshot(
                        state=engine.get_field_snapshot(),
                        time=engine.time,
                        circuit_scalars={
                            "current": engine.circuit.current,
                            "voltage": engine.circuit.voltage,
                        },
                    )

                if result.finished:
                    break

            # Capture final state if not already captured
            if step_count % self.field_interval != 0:
                exporter.add_snapshot(
                    state=engine.get_field_snapshot(),
                    time=engine.time,
                    circuit_scalars={
                        "current": engine.circuit.current,
                        "voltage": engine.circuit.voltage,
                    },
                )

            # Finalize diagnostics and write Well HDF5
            engine.diagnostics.finalize()
            exporter.finalize()

            logger.info(
                f"Completed simulation {idx} -> {output_path} "
                f"({step_count} steps, {len(exporter._snapshots)} snapshots)"
            )
            return (idx, None)

        except Exception as e:
            error_msg = f"Failed: {type(e).__name__}: {str(e)}"
            logger.error(f"Simulation {idx} failed: {error_msg}")
            return (idx, error_msg)

    def run(self, progress_callback: Any | None = None) -> BatchResult:
        """Run batch simulations.

        Parameters
        ----------
        progress_callback : callable | None
            Optional callback(completed, total) called after each simulation

        Returns
        -------
        BatchResult
            Summary of batch run results
        """
        # Generate samples
        samples = self.generate_samples()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Run simulations
        results: list[tuple[int, str | None]] = []

        if self.workers <= 1:
            # Sequential execution
            logger.info("Running simulations sequentially")
            for idx, params in enumerate(samples):
                result = self.run_single(idx, params)
                results.append(result)

                if progress_callback is not None:
                    progress_callback(idx + 1, self.n_samples)
        else:
            # Parallel execution
            import multiprocessing

            logger.info(f"Running simulations with {self.workers} workers")

            # Create wrapper function for multiprocessing
            def _run_wrapper(args):
                idx, params = args
                return self.run_single(idx, params)

            with multiprocessing.Pool(self.workers) as pool:
                # Use imap for progress tracking
                for completed, result in enumerate(
                    pool.imap(_run_wrapper, enumerate(samples)), start=1
                ):
                    results.append(result)

                    if progress_callback is not None:
                        progress_callback(completed, self.n_samples)

        # Compile results
        n_success = sum(1 for _, error in results if error is None)
        n_failed = sum(1 for _, error in results if error is not None)
        failed_configs = [(idx, error) for idx, error in results if error is not None]

        batch_result = BatchResult(
            n_total=self.n_samples,
            n_success=n_success,
            n_failed=n_failed,
            output_dir=str(self.output_dir),
            failed_configs=failed_configs,
        )

        logger.info(
            f"Batch complete: {n_success}/{self.n_samples} succeeded, "
            f"{n_failed} failed"
        )

        return batch_result

    @staticmethod
    def _latin_hypercube(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
        """Generate Latin Hypercube samples.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_dims : int
            Number of dimensions
        seed : int
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_dims) with values in [0, 1]
        """
        try:
            # Try to use scipy's optimized LHS
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
            samples = sampler.random(n=n_samples)
            return samples

        except ImportError:
            # Fallback: manual LHS implementation
            logger.warning("scipy not available, using fallback LHS implementation")

            rng = np.random.default_rng(seed)
            samples = np.zeros((n_samples, n_dims))

            for dim in range(n_dims):
                # Divide [0, 1] into n_samples bins
                bins = np.arange(n_samples) / n_samples
                # Add random offset within each bin
                offsets = rng.random(n_samples) / n_samples
                values = bins + offsets
                # Randomly permute
                samples[:, dim] = rng.permutation(values)

            return samples
