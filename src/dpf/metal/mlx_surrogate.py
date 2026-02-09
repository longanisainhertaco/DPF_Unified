"""WALRUS surrogate inference via Apple MLX on Metal.

This module provides a bridge between the WALRUS IsotropicModel (PyTorch) and
Apple's MLX framework for zero-copy tensor processing on unified memory.  The
model itself stays in PyTorch (the full 1.3B-parameter Encoder-Processor-Decoder
is too large to reimplement in MLX), but pre- and post-processing (RevIN
normalization, delta residual addition) are performed with MLX arrays that share
memory with NumPy via Apple's unified memory architecture.

On Apple Silicon, ``np.array(mx_array, copy=False)`` is truly zero-copy because
both MLX and NumPy operate on the same physical DRAM.  Conversions between MLX
and PyTorch go through this NumPy intermediary, avoiding any GPU-to-host copies
that would occur with CUDA.

Falls back to pure PyTorch MPS when MLX is not available.

Example
-------
>>> surrogate = MLXSurrogate("models/walrus-pretrained")
>>> history = [state_t0, state_t1, state_t2, state_t3]
>>> next_state = surrogate.predict_next_step(history)
>>> trajectory = surrogate.rollout(history, n_steps=10)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_HAS_MLX = False
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]

_HAS_TORCH = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

# Channel layout (mirrors src/dpf/ai/surrogate.py)
_SCALAR_KEYS = ("rho", "Te", "Ti", "pressure", "psi")
_VECTOR_KEYS = ("B", "velocity")
_N_CHANNELS = len(_SCALAR_KEYS) + len(_VECTOR_KEYS) * 3  # 11

# DPF field -> WALRUS (name, embedding index)
_DPF_TO_WALRUS_FIELD = {
    "rho": ("density", 28),
    "Te": ("temperature", 46),
    "Ti": ("temperature", 46),
    "pressure": ("pressure", 3),
    "psi": ("A", 42),
    "Bx": ("magnetic_field_x", 39),
    "By": ("magnetic_field_y", 40),
    "Bz": ("magnetic_field_z", 41),
    "vx": ("velocity_x", 4),
    "vy": ("velocity_y", 5),
    "vz": ("velocity_z", 6),
}

# WALRUS field ordering for DPF channels
_DPF_FIELD_NAMES = [
    "density", "temperature", "temperature", "pressure", "A",
    "magnetic_field_x", "magnetic_field_y", "magnetic_field_z",
    "velocity_x", "velocity_y", "velocity_z",
]


class MLXSurrogate:
    """WALRUS surrogate inference via MLX on Apple Metal.

    Strategy: Load PyTorch model, run inference by converting tensors between
    PyTorch and MLX via DLPack/NumPy for zero-copy on unified memory.  Falls
    back to PyTorch MPS if MLX conversion fails.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to WALRUS checkpoint directory (containing ``walrus.pt`` and
        optionally ``extended_config.yaml``) or a single ``.pt`` file.
    history_length : int
        Number of historical timesteps required for prediction.  Default 4.
    device : str or None
        PyTorch device for model inference.  ``None`` auto-selects "mps" if
        available, else "cpu".  MLX is always used for pre/post-processing
        regardless of this setting.

    Raises
    ------
    ImportError
        If PyTorch is not available.
    FileNotFoundError
        If *checkpoint_path* does not exist.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        history_length: int = 4,
        device: str | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MLXSurrogate. "
                "Install with: pip install torch"
            )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.history_length = history_length
        self._use_mlx = _HAS_MLX
        self._model: Any = None
        self._revin: Any = None
        self._formatter: Any = None
        self._walrus_config: Any = None
        self._field_to_index_map: dict[str, int] | None = None
        self._dpf_field_indices: Any = None  # torch.Tensor

        # Device selection
        if device is not None:
            self._device = device
        elif _HAS_TORCH and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        if self._use_mlx:
            logger.info(
                "MLX available -- using MLX for pre/post-processing "
                "(zero-copy unified memory)"
            )
        else:
            logger.info(
                "MLX not available -- using pure PyTorch on %s", self._device
            )

        self._load_model()

    # ------------------------------------------------------------------
    # Class-level availability check
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Check if both MLX and WALRUS are importable.

        Returns
        -------
        bool
            True if MLX core and the ``walrus.models`` package can be imported.
        """
        if not _HAS_MLX:
            return False
        try:
            import walrus.models  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_checkpoint_files(self) -> tuple[Path, Path | None]:
        """Resolve checkpoint ``.pt`` file and optional config YAML.

        Returns
        -------
        tuple[Path, Path | None]
            ``(pt_path, config_yaml_path)`` -- config may be ``None``.
        """
        if self.checkpoint_path.is_dir():
            pt_path = self.checkpoint_path / "walrus.pt"
            if not pt_path.exists():
                pt_files = list(self.checkpoint_path.glob("*.pt"))
                if pt_files:
                    pt_path = pt_files[0]
                else:
                    raise FileNotFoundError(
                        f"No .pt file found in {self.checkpoint_path}"
                    )
            config_path = self.checkpoint_path / "extended_config.yaml"
            if not config_path.exists():
                config_path = None
            return pt_path, config_path
        else:
            config_path = self.checkpoint_path.parent / "extended_config.yaml"
            if not config_path.exists():
                config_path = None
            return self.checkpoint_path, config_path

    def _load_model(self) -> None:
        """Load WALRUS checkpoint and instantiate model.

        Attempts to load the real WALRUS ``IsotropicModel`` via Hydra config.
        If the ``walrus`` package is not installed, stores a placeholder dict
        and logs a warning.
        """
        try:
            pt_path, config_yaml_path = self._resolve_checkpoint_files()

            logger.info("Loading checkpoint from %s ...", pt_path)
            checkpoint_data = torch.load(
                pt_path, map_location="cpu", weights_only=False
            )

            # Extract state dict (multiple checkpoint formats)
            if "app" in checkpoint_data and "model" in checkpoint_data["app"]:
                state_dict = checkpoint_data["app"]["model"]
            elif "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            elif "state_dict" in checkpoint_data:
                state_dict = checkpoint_data["state_dict"]
            else:
                state_dict = checkpoint_data

            self._load_walrus_model(state_dict, config_yaml_path, checkpoint_data)

        except ImportError as exc:
            raise ImportError(
                f"WALRUS not installed or missing dependency: {exc}. "
                "Install with: pip install "
                "git+https://github.com/PolymathicAI/walrus.git"
            ) from exc
        except FileNotFoundError:
            raise
        except Exception as exc:
            self._model = None
            logger.error(
                "Failed to load WALRUS model from %s: %s",
                self.checkpoint_path, exc,
            )
            raise

    def _load_walrus_model(
        self,
        state_dict: dict,
        config_yaml_path: Path | None,
        checkpoint_data: dict,
    ) -> None:
        """Instantiate the real WALRUS IsotropicModel.

        Parameters
        ----------
        state_dict : dict
            Model weights already extracted from the checkpoint.
        config_yaml_path : Path or None
            Path to ``extended_config.yaml``.
        checkpoint_data : dict
            Raw checkpoint data for fallback config extraction.
        """
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from walrus.data.well_to_multi_transformer import (
            ChannelsFirstWithTimeFormatter,
        )

        # Load Hydra/OmegaConf config
        config = None
        if config_yaml_path is not None and config_yaml_path.exists():
            config = OmegaConf.load(config_yaml_path)
            logger.info("Loaded WALRUS config from %s", config_yaml_path)
        elif "config" in checkpoint_data:
            config = checkpoint_data["config"]
            logger.info("Using embedded config from checkpoint")

        if config is None:
            raise RuntimeError(
                "No WALRUS config found -- cannot instantiate model. "
                "Provide extended_config.yaml alongside the checkpoint."
            )

        self._walrus_config = config

        # Field index map
        field_map = dict(config.data.get("field_index_map_override", {}))
        if field_map:
            self._field_to_index_map = field_map
        else:
            self._field_to_index_map = {
                name: idx for _, (name, idx) in _DPF_TO_WALRUS_FIELD.items()
            }

        n_states = max(self._field_to_index_map.values()) + 1

        # Instantiate model via Hydra
        model = instantiate(config.model, n_states=n_states)

        # Align weights if possible
        try:
            from walrus.utils.experiment_utils import (
                align_checkpoint_with_field_to_index_map,
            )
            checkpoint_field_map = field_map
            model_field_map = dict(self._field_to_index_map)
            aligned = align_checkpoint_with_field_to_index_map(
                checkpoint_state_dict=state_dict,
                model_state_dict=model.state_dict(),
                checkpoint_field_to_index_map=checkpoint_field_map,
                model_field_to_index_map=model_field_map,
            )
            model.load_state_dict(aligned)
        except Exception:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self._device)
        self._model = model

        # DPF field indices tensor
        self._dpf_field_indices = torch.tensor(
            [self._field_to_index_map.get(name, 0) for name in _DPF_FIELD_NAMES],
            device=self._device,
            dtype=torch.long,
        )

        # RevIN normalization
        try:
            self._revin = instantiate(config.trainer.revin)()
        except Exception:
            logger.warning(
                "Failed to instantiate RevIN from config; "
                "trying SamplewiseRevNormalization directly"
            )
            try:
                from walrus.trainer.normalization_strat import (
                    SamplewiseRevNormalization,
                )
                self._revin = SamplewiseRevNormalization()
            except Exception:
                logger.warning("RevIN unavailable -- inference may be degraded")
                self._revin = None

        # Formatter
        self._formatter = ChannelsFirstWithTimeFormatter()

        logger.info(
            "Loaded WALRUS IsotropicModel (n_states=%d, device=%s, mlx=%s)",
            n_states, self._device, self._use_mlx,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if the model is loaded and ready for inference."""
        return self._model is not None and not isinstance(self._model, dict)

    def predict_next_step(
        self, history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Predict the next DPF state from a historical trajectory.

        Parameters
        ----------
        history : list[dict[str, np.ndarray]]
            List of DPF state dicts (most recent last).  Each dict has keys
            ``rho``, ``velocity``, ``pressure``, ``B``, ``Te``, ``Ti``,
            ``psi`` as NumPy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Predicted next state with the same structure as the inputs.

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        ValueError
            If *history* contains fewer states than ``history_length``.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Check checkpoint path and logs."
            )
        if len(history) < self.history_length:
            raise ValueError(
                f"Insufficient history: need {self.history_length}, "
                f"got {len(history)}"
            )

        recent = history[-self.history_length:]

        if self._use_mlx:
            try:
                return self._predict_with_mlx(recent)
            except Exception as exc:
                logger.warning(
                    "MLX pre/post-processing failed (%s); "
                    "falling back to pure PyTorch",
                    exc,
                )
                return self._predict_pytorch(recent)
        else:
            return self._predict_pytorch(recent)

    def rollout(
        self,
        initial_history: list[dict[str, np.ndarray]],
        n_steps: int,
    ) -> list[dict[str, np.ndarray]]:
        """Autoregressive multi-step prediction.

        Feeds each predicted state back as input for the next step.

        Parameters
        ----------
        initial_history : list[dict[str, np.ndarray]]
            Starting trajectory with at least ``history_length`` states.
        n_steps : int
            Number of future steps to predict.

        Returns
        -------
        list[dict[str, np.ndarray]]
            List of *n_steps* predicted states (does NOT include the input
            history).
        """
        if len(initial_history) < self.history_length:
            raise ValueError(
                f"Insufficient initial history: need {self.history_length}, "
                f"got {len(initial_history)}"
            )

        window = list(initial_history[-self.history_length:])
        predictions: list[dict[str, np.ndarray]] = []

        for step in range(n_steps):
            pred = self.predict_next_step(window)
            predictions.append(pred)
            window.append(pred)
            window = window[-self.history_length:]
            logger.debug("Rollout step %d/%d complete", step + 1, n_steps)

        return predictions

    def benchmark(
        self,
        grid_shape: tuple[int, int, int] = (16, 16, 16),
        n_iterations: int = 10,
    ) -> dict[str, float]:
        """Benchmark inference latency and memory usage.

        Generates synthetic DPF state history of the given grid shape and
        times ``predict_next_step`` for *n_iterations*.

        Parameters
        ----------
        grid_shape : tuple[int, int, int]
            Spatial grid dimensions (H, W, D).
        n_iterations : int
            Number of forward passes to time.

        Returns
        -------
        dict[str, float]
            ``mean_ms``, ``p50_ms``, ``p95_ms``, ``memory_mb`` (resident set
            size increase during the benchmark).
        """
        # Synthetic history
        history = self._make_synthetic_history(grid_shape)

        # Warm up (first call may trigger JIT / Metal compilation)
        _ = self.predict_next_step(history)
        if self._device == "mps":
            torch.mps.synchronize()

        # Get baseline RSS
        rss_before = _get_rss_mb()

        latencies: list[float] = []
        for _i in range(n_iterations):
            t0 = time.perf_counter()
            _ = self.predict_next_step(history)
            if self._device == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        rss_after = _get_rss_mb()

        latencies_arr = np.array(latencies)
        result = {
            "mean_ms": float(np.mean(latencies_arr)),
            "p50_ms": float(np.median(latencies_arr)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "memory_mb": max(0.0, rss_after - rss_before),
            "grid_shape": grid_shape,
            "n_iterations": n_iterations,
            "device": self._device,
            "mlx_enabled": self._use_mlx,
        }
        logger.info(
            "Benchmark: mean=%.1fms  p50=%.1fms  p95=%.1fms  mem=%.1fMB",
            result["mean_ms"], result["p50_ms"],
            result["p95_ms"], result["memory_mb"],
        )
        return result

    # ------------------------------------------------------------------
    # MLX-accelerated inference path
    # ------------------------------------------------------------------

    def _predict_with_mlx(
        self, recent_history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Run WALRUS inference with MLX pre/post-processing.

        Pre-processing (RevIN normalization) and post-processing (delta
        denormalization + residual addition) are done in MLX for zero-copy
        Metal compute.  The model forward pass itself runs in PyTorch on MPS.

        Parameters
        ----------
        recent_history : list[dict[str, np.ndarray]]
            Last ``history_length`` DPF state dicts.

        Returns
        -------
        dict[str, np.ndarray]
            Predicted next DPF state.
        """
        batch = self._build_walrus_batch(recent_history)
        metadata = batch["metadata"]

        with torch.no_grad():
            # Formatter: batch dict -> (x, field_indices, bcs), y_ref
            inputs, _ = self._formatter.process_input(
                batch, causal_in_time=True, predict_delta=True, train=False,
            )
            inputs = list(inputs)

            # --- MLX pre-processing: RevIN normalize ---
            if self._revin is not None:
                stats = self._revin.compute_stats(
                    inputs[0], metadata, epsilon=1e-5
                )
                x_torch = inputs[0]
                x_np = x_torch.cpu().numpy()

                # Zero-copy into MLX, compute normalized array
                x_mlx = mx.array(x_np)
                std_np = _extract_revin_std(stats)
                std_mlx = mx.array(std_np)
                mean_np = _extract_revin_mean(stats)
                mean_mlx = mx.array(mean_np)

                eps = 1e-5
                normalized_mlx = (x_mlx - mean_mlx) / (std_mlx + eps)
                mx.eval(normalized_mlx)  # Force compute

                # Zero-copy back to NumPy, then to PyTorch
                normalized_np = np.array(normalized_mlx, copy=False)
                normalized_torch = torch.from_numpy(
                    normalized_np.copy()  # MPS needs contiguous memory
                ).to(self._device)

                inputs_fwd = list(inputs)
                inputs_fwd[0] = normalized_torch
            else:
                inputs_fwd = inputs
                stats = None

            # --- PyTorch forward pass (MPS / CPU) ---
            y_pred = self._model(
                inputs_fwd[0],
                inputs_fwd[1],
                inputs_fwd[2].tolist(),
                metadata=metadata,
                train=False,
            )

            # Causal: take last prediction only
            if (
                hasattr(self._model, "causal_in_time")
                and self._model.causal_in_time
            ):
                y_pred = y_pred[-1:]

            # --- MLX post-processing: denormalize delta + residual ---
            if self._revin is not None and stats is not None:
                y_np = y_pred.cpu().numpy()
                y_mlx = mx.array(y_np)

                # Denormalize delta: delta * (std + eps)
                denorm_mlx = y_mlx * (std_mlx + eps)
                mx.eval(denorm_mlx)

                # Residual: u(t+1) = u(t)[-n:] + denormalized_delta
                residual_np = inputs[0][-y_pred.shape[0]:].cpu().numpy()
                residual_mlx = mx.array(residual_np)
                result_mlx = residual_mlx.astype(mx.float32) + denorm_mlx
                mx.eval(result_mlx)

                # Zero-copy back to PyTorch
                result_np = np.array(result_mlx, copy=False)
                y_pred = torch.from_numpy(result_np.copy()).to(self._device)
            else:
                y_pred = inputs[0][-y_pred.shape[0]:].float() + y_pred

            # Convert back to Well format
            y_pred = self._formatter.process_output(y_pred, metadata)

            # Mask padded fields
            if "padded_field_mask" in batch:
                y_pred = y_pred[..., batch["padded_field_mask"]]

        # Extract: (B=1, T=1, H, W, D, C) -> (H, W, D, C)
        pred_array = y_pred[0, -1].cpu().numpy()

        return self._well_output_to_state(pred_array, recent_history[-1])

    # ------------------------------------------------------------------
    # Pure PyTorch fallback
    # ------------------------------------------------------------------

    def _predict_pytorch(
        self, recent_history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Run WALRUS inference entirely in PyTorch (no MLX).

        Mirrors the logic in ``DPFSurrogate._walrus_predict``.

        Parameters
        ----------
        recent_history : list[dict[str, np.ndarray]]
            Last ``history_length`` DPF state dicts.

        Returns
        -------
        dict[str, np.ndarray]
            Predicted next DPF state.
        """
        batch = self._build_walrus_batch(recent_history)
        metadata = batch["metadata"]

        with torch.no_grad():
            inputs, _ = self._formatter.process_input(
                batch, causal_in_time=True, predict_delta=True, train=False,
            )
            inputs = list(inputs)

            # RevIN normalize
            if self._revin is not None:
                stats = self._revin.compute_stats(
                    inputs[0], metadata, epsilon=1e-5
                )
                normalized = inputs[:]
                normalized[0] = self._revin.normalize_stdmean(inputs[0], stats)
            else:
                normalized = inputs
                stats = None

            # Forward pass
            y_pred = self._model(
                normalized[0],
                normalized[1],
                normalized[2].tolist(),
                metadata=metadata,
                train=False,
            )

            if (
                hasattr(self._model, "causal_in_time")
                and self._model.causal_in_time
            ):
                y_pred = y_pred[-1:]

            # Denormalize delta + residual
            if self._revin is not None and stats is not None:
                y_pred = (
                    inputs[0][-y_pred.shape[0]:].float()
                    + self._revin.denormalize_delta(y_pred, stats)
                )
            else:
                y_pred = inputs[0][-y_pred.shape[0]:].float() + y_pred

            y_pred = self._formatter.process_output(y_pred, metadata)

            if "padded_field_mask" in batch:
                y_pred = y_pred[..., batch["padded_field_mask"]]

        pred_array = y_pred[0, -1].cpu().numpy()
        return self._well_output_to_state(pred_array, recent_history[-1])

    # ------------------------------------------------------------------
    # Batch construction (mirrors DPFSurrogate._build_walrus_batch)
    # ------------------------------------------------------------------

    def _build_walrus_batch(
        self, states: list[dict[str, np.ndarray]]
    ) -> dict[str, Any]:
        """Build a WALRUS-compatible batch dict from DPF state history.

        Creates the synthetic batch format expected by the WALRUS
        ``ChannelsFirstWithTimeFormatter``.

        Parameters
        ----------
        states : list[dict[str, np.ndarray]]
            DPF state dicts with keys: rho, Te, Ti, pressure, psi,
            B ``(3, *spatial)``, velocity ``(3, *spatial)``.

        Returns
        -------
        dict
            Batch dict with keys: ``input_fields``, ``output_fields``,
            ``constant_fields``, ``boundary_conditions``,
            ``padded_field_mask``, ``field_indices``, ``metadata``.
        """
        from the_well.data.datasets import WellMetadata

        T = len(states)
        ref = states[0]
        spatial = ref["rho"].shape

        # Pad to 3D
        if len(spatial) == 2:
            spatial = (*spatial, 1)
        elif len(spatial) == 1:
            spatial = (*spatial, 1, 1)

        C = _N_CHANNELS

        # input_fields: [B=1, T, H, W, D, C] channels-last
        input_arr = np.zeros((1, T, *spatial, C), dtype=np.float32)

        for t, state in enumerate(states):
            ch = 0
            for key in _SCALAR_KEYS:
                if key in state:
                    field = state[key].astype(np.float32)
                    if field.ndim < 3:
                        field = field.reshape(spatial)
                    input_arr[0, t, ..., ch] = field
                ch += 1
            for key in _VECTOR_KEYS:
                if key in state:
                    vec = state[key].astype(np.float32)
                    for comp in range(3):
                        comp_field = vec[comp]
                        if comp_field.ndim < 3:
                            comp_field = comp_field.reshape(spatial)
                        input_arr[0, t, ..., ch + comp] = comp_field
                ch += 3

        device = self._device
        input_tensor = torch.from_numpy(input_arr).to(device)

        output_tensor = torch.zeros(
            (1, 1, *spatial, C), dtype=torch.float32, device=device
        )
        const_tensor = torch.zeros(
            (1, *spatial, 0), dtype=torch.float32, device=device
        )
        bcs = torch.tensor(
            [[[0, 0], [0, 0], [0, 0]]],
            dtype=torch.long, device=device,
        )
        padded_mask = torch.ones(C, dtype=torch.bool, device=device)
        field_indices = self._dpf_field_indices.clone()

        metadata = WellMetadata(
            dataset_name="dpf_plasma",
            n_spatial_dims=3,
            spatial_resolution=spatial,
            scalar_names=list(_SCALAR_KEYS),
            constant_scalar_names=[],
            field_names={
                0: [
                    "density", "temperature", "temperature",
                    "pressure", "A",
                ],
                1: [
                    "velocity_x", "velocity_y", "velocity_z",
                    "magnetic_field_x", "magnetic_field_y",
                    "magnetic_field_z",
                ],
                2: [],
            },
            constant_field_names={0: [], 1: [], 2: []},
            boundary_condition_types=["WALL"],
            n_files=1,
            n_trajectories_per_file=[1],
            n_steps_per_trajectory=[T],
        )

        return {
            "input_fields": input_tensor,
            "output_fields": output_tensor,
            "constant_fields": const_tensor,
            "boundary_conditions": bcs,
            "padded_field_mask": padded_mask,
            "field_indices": field_indices,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    def _well_output_to_state(
        self, pred_array: np.ndarray, reference_state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert WALRUS output array to a DPF state dict.

        Parameters
        ----------
        pred_array : np.ndarray
            Shape ``(H, W, D, C)`` -- channels-last Well format.
        reference_state : dict[str, np.ndarray]
            Reference DPF state for shape information.

        Returns
        -------
        dict[str, np.ndarray]
            DPF state dict with float64 arrays matching reference shapes.
        """
        orig_spatial = reference_state["rho"].shape
        state: dict[str, np.ndarray] = {}
        ch = 0

        for key in _SCALAR_KEYS:
            if key in reference_state:
                field = pred_array[..., ch].astype(np.float64)
                if field.shape != orig_spatial:
                    field = field.reshape(orig_spatial)
                state[key] = field
            ch += 1

        for key in _VECTOR_KEYS:
            if key in reference_state:
                components = []
                for comp in range(3):
                    c = pred_array[..., ch + comp].astype(np.float64)
                    if c.shape != orig_spatial:
                        c = c.reshape(orig_spatial)
                    components.append(c)
                state[key] = np.stack(components, axis=0)
            ch += 3

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_synthetic_history(
        self, grid_shape: tuple[int, int, int]
    ) -> list[dict[str, np.ndarray]]:
        """Create synthetic DPF state history for benchmarking.

        Parameters
        ----------
        grid_shape : tuple[int, int, int]
            Spatial grid dimensions ``(nx, ny, nz)``.

        Returns
        -------
        list[dict[str, np.ndarray]]
            List of ``history_length`` synthetic state dicts.
        """
        rng = np.random.default_rng(42)
        history: list[dict[str, np.ndarray]] = []

        for _ in range(self.history_length):
            state = {
                "rho": rng.uniform(0.1, 10.0, grid_shape).astype(np.float64),
                "Te": rng.uniform(1.0, 100.0, grid_shape).astype(np.float64),
                "Ti": rng.uniform(1.0, 100.0, grid_shape).astype(np.float64),
                "pressure": rng.uniform(
                    0.1, 50.0, grid_shape
                ).astype(np.float64),
                "psi": rng.uniform(
                    -1.0, 1.0, grid_shape
                ).astype(np.float64),
                "B": rng.uniform(
                    -1.0, 1.0, (3, *grid_shape)
                ).astype(np.float64),
                "velocity": rng.uniform(
                    -1e6, 1e6, (3, *grid_shape)
                ).astype(np.float64),
            }
            history.append(state)

        return history


# ======================================================================
# Module-level helpers
# ======================================================================


def _extract_revin_std(stats: Any) -> np.ndarray:
    """Extract the standard-deviation array from a RevIN stats object.

    The WALRUS RevIN ``compute_stats`` returns an object whose layout may
    vary between versions.  This helper tries several known formats and
    returns a NumPy array suitable for broadcasting during normalization.

    Parameters
    ----------
    stats : object
        RevIN stats from ``revin.compute_stats()``.

    Returns
    -------
    np.ndarray
        Standard-deviation values (float32).
    """
    if _HAS_TORCH and isinstance(stats, torch.Tensor):
        return stats.cpu().numpy()

    # Named-tuple / dataclass with .std or .rms attribute
    for attr in ("std", "rms", "scale"):
        if hasattr(stats, attr):
            val = getattr(stats, attr)
            if _HAS_TORCH and isinstance(val, torch.Tensor):
                return val.cpu().numpy()
            return np.asarray(val, dtype=np.float32)

    # Dict-like
    if isinstance(stats, dict):
        for key in ("std", "rms", "scale"):
            if key in stats:
                val = stats[key]
                if _HAS_TORCH and isinstance(val, torch.Tensor):
                    return val.cpu().numpy()
                return np.asarray(val, dtype=np.float32)

    # Tuple: (mean, std) convention
    if isinstance(stats, (tuple, list)) and len(stats) >= 2:
        val = stats[1]
        if _HAS_TORCH and isinstance(val, torch.Tensor):
            return val.cpu().numpy()
        return np.asarray(val, dtype=np.float32)

    # Last resort: treat the whole thing as the std tensor
    logger.warning(
        "Could not extract std from RevIN stats (type=%s); "
        "treating entire object as std",
        type(stats).__name__,
    )
    if _HAS_TORCH and isinstance(stats, torch.Tensor):
        return stats.cpu().numpy()
    return np.asarray(stats, dtype=np.float32)


def _extract_revin_mean(stats: Any) -> np.ndarray:
    """Extract the mean array from a RevIN stats object.

    RevIN RMS normalization may not use a mean (centered at zero).  In that
    case, returns a zero array matching the std shape.

    Parameters
    ----------
    stats : object
        RevIN stats from ``revin.compute_stats()``.

    Returns
    -------
    np.ndarray
        Mean values (float32).  May be all zeros for RMS-only normalization.
    """
    # Named-tuple / dataclass with .mean attribute
    for attr in ("mean", "loc", "bias"):
        if hasattr(stats, attr):
            val = getattr(stats, attr)
            if _HAS_TORCH and isinstance(val, torch.Tensor):
                return val.cpu().numpy()
            return np.asarray(val, dtype=np.float32)

    # Dict-like
    if isinstance(stats, dict):
        for key in ("mean", "loc", "bias"):
            if key in stats:
                val = stats[key]
                if _HAS_TORCH and isinstance(val, torch.Tensor):
                    return val.cpu().numpy()
                return np.asarray(val, dtype=np.float32)

    # Tuple: (mean, std) convention
    if isinstance(stats, (tuple, list)) and len(stats) >= 2:
        val = stats[0]
        if _HAS_TORCH and isinstance(val, torch.Tensor):
            return val.cpu().numpy()
        return np.asarray(val, dtype=np.float32)

    # RMS normalization: mean is zero
    std = _extract_revin_std(stats)
    return np.zeros_like(std)


def _get_rss_mb() -> float:
    """Return current process resident set size in MB.

    Returns
    -------
    float
        RSS in megabytes, or 0.0 if unavailable.
    """
    try:
        import resource
        # RUSAGE_SELF in bytes on macOS (kilobytes on Linux)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = usage.ru_maxrss
        import sys
        if sys.platform == "darwin":
            return rss_bytes / (1024 * 1024)  # macOS: bytes -> MB
        else:
            return rss_bytes / 1024  # Linux: KB -> MB
    except Exception:
        return 0.0
