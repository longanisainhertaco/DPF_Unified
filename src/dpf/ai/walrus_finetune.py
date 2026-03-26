"""WALRUS fine-tuning orchestrator for DPF training data.

End-to-end pipeline:
1. Generate DPF simulation trajectories via BatchRunner
2. Export to Well HDF5 format via WellExporter
3. Validate dataset integrity via DatasetValidator
4. Launch WALRUS fine-tuning (local or external)
5. Save fine-tuned checkpoint

Usage::

    from dpf.ai.walrus_finetune import fine_tune_pipeline

    result = fine_tune_pipeline(
        preset_name="pf1000",
        parameter_ranges={"circuit.V0": (15000, 25000), "init.pressure0": (0.5, 2.0)},
        n_samples=50,
        output_dir="training_data/pf1000_v1",
    )

References:
    WALRUS: github.com/PolymathicAI/walrus (IsotropicModel, 1.3B params)
    The Well: github.com/PolymathicAI/the_well (HDF5 dataset format)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FinetuneResult:
    """Result of the fine-tuning pipeline."""

    n_trajectories: int = 0
    n_valid: int = 0
    n_failed: int = 0
    output_dir: str = ""
    checkpoint_path: str | None = None
    training_launched: bool = False
    errors: list[str] = field(default_factory=list)


def generate_training_data(
    preset_name: str = "pf1000",
    parameter_ranges: dict[str, tuple[float, float]] | None = None,
    n_samples: int = 50,
    output_dir: str = "training_data/walrus_v1",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    sim_time: float = 8e-6,
    field_interval: int = 10,
    workers: int = 1,
) -> FinetuneResult:
    """Step 1-3: Generate, export, and validate DPF training trajectories.

    Args:
        preset_name: Base device preset.
        parameter_ranges: Dict of param path → (min, max) for Latin Hypercube.
            Example: {"circuit.V0": (15000, 25000)}
        n_samples: Number of simulation trajectories.
        output_dir: Directory for Well HDF5 output files.
        grid_shape: Simulation grid (nr, ny, nz).
        sim_time: Simulation duration [s].
        field_interval: Save field snapshot every N engine steps.
        workers: Number of parallel simulation workers.

    Returns:
        FinetuneResult with trajectory counts and validation status.
    """
    from dpf.config import SimulationConfig
    from dpf.presets import get_preset

    result = FinetuneResult(output_dir=output_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build base config
    preset = get_preset(preset_name)
    preset["grid_shape"] = list(grid_shape)
    preset["sim_time"] = sim_time
    preset["fluid"] = preset.get("fluid", {})
    preset["fluid"]["backend"] = "mlx"
    preset["fluid"]["riemann_solver"] = "hll"
    preset["fluid"]["reconstruction"] = "plm"
    if "diagnostics" not in preset:
        preset["diagnostics"] = {}
    preset["diagnostics"]["hdf5_enabled"] = False

    base_config = SimulationConfig(**preset)

    # Default parameter ranges if not specified
    if parameter_ranges is None:
        parameter_ranges = {
            "circuit.V0": (15000, 30000),
            "snowplow.current_fraction": (0.5, 0.85),
            "snowplow.mass_fraction": (0.03, 0.20),
        }

    logger.info(
        "Generating %d trajectories for %s (grid=%s, sim_time=%.1f us)",
        n_samples, preset_name, grid_shape, sim_time * 1e6,
    )

    # Generate trajectories
    try:
        from dpf.ai.batch_runner import BatchRunner

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=n_samples,
            output_dir=str(out_path),
            workers=workers,
            field_interval=field_interval,
        )
        batch_result = runner.run()
        result.n_trajectories = batch_result.n_success + batch_result.n_failed
        result.n_valid = batch_result.n_success
        result.n_failed = batch_result.n_failed
    except Exception as exc:
        result.errors.append(f"Batch generation failed: {exc}")
        logger.error("Batch generation failed: %s", exc)
        return result

    # Validate
    try:
        from dpf.ai.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        val_results = validator.validate_directory(str(out_path))
        n_invalid = sum(1 for v in val_results.values() if not v.valid)
        if n_invalid > 0:
            result.errors.append(f"{n_invalid} trajectories failed validation")
        logger.info("Validation: %d/%d valid", result.n_valid - n_invalid, result.n_valid)
    except Exception as exc:
        result.errors.append(f"Validation failed: {exc}")

    return result


def launch_walrus_training(
    data_dir: str,
    checkpoint_dir: str = "models/walrus-finetuned",
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    gradient_checkpointing: bool = True,
) -> bool:
    """Step 4: Launch WALRUS fine-tuning via subprocess.

    Invokes the WALRUS training script with Hydra config overrides
    for Apple Silicon (AMP disabled, gradient checkpointing).

    Args:
        data_dir: Path to Well HDF5 training data.
        checkpoint_dir: Output directory for fine-tuned checkpoint.
        epochs: Number of training epochs.
        batch_size: Batch size (1-2 for M3 Pro 36GB).
        learning_rate: Adam learning rate.
        gradient_checkpointing: Enable gradient checkpointing for memory.

    Returns:
        True if training launched successfully.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "walrus.train",
        "distribution=local",
        "model=isotropic_model",
        "finetune=True",
        f"data.root_path={data_dir}",
        f"trainer.max_epoch={epochs}",
        f"data.module_parameters.batch_size={batch_size}",
        f"optimizer.lr={learning_rate}",
        "trainer.enable_amp=False",
        "trainer.prediction_type=delta",
        "model.causal_in_time=True",
    ]
    if gradient_checkpointing:
        cmd.append("model.gradient_checkpointing_freq=1")

    logger.info("Launching WALRUS training: %s", " ".join(cmd[:5]))

    try:
        subprocess.run(cmd, check=True, timeout=3600 * 24)
        return True
    except FileNotFoundError:
        logger.error("walrus package not found. Install: pip install git+https://github.com/PolymathicAI/walrus.git")
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("WALRUS training failed: %s", exc)
        return False
    except subprocess.TimeoutExpired:
        logger.error("WALRUS training timed out (24h limit)")
        return False


def fine_tune_pipeline(
    preset_name: str = "pf1000",
    parameter_ranges: dict[str, tuple[float, float]] | None = None,
    n_samples: int = 50,
    output_dir: str = "training_data/walrus_v1",
    epochs: int = 50,
    skip_training: bool = False,
) -> FinetuneResult:
    """Run the full fine-tuning pipeline: generate data → validate → train.

    Args:
        preset_name: Base device preset for training data.
        parameter_ranges: Parameter sweep ranges.
        n_samples: Number of training trajectories.
        output_dir: Output directory for training data.
        epochs: Fine-tuning epochs.
        skip_training: If True, only generate data (don't launch training).

    Returns:
        FinetuneResult with full pipeline status.
    """
    result = generate_training_data(
        preset_name=preset_name,
        parameter_ranges=parameter_ranges,
        n_samples=n_samples,
        output_dir=output_dir,
    )

    if result.n_valid < 10:
        result.errors.append(f"Too few valid trajectories ({result.n_valid}). Need >= 10.")
        return result

    if not skip_training:
        result.training_launched = launch_walrus_training(
            data_dir=output_dir,
            epochs=epochs,
        )

    return result
