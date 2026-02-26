from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dpf.ai import HAS_TORCH, HAS_WALRUS
from dpf.ai._walrus_base import (
    DPF_TO_WALRUS_FIELD,
    WALRUS_N_CHANNELS,
    WALRUS_SCALAR_KEYS,
    WALRUS_VECTOR_KEYS,
    WalrusInferenceMixin,
)
from dpf.ai.field_mapping import (
    SCALAR_FIELDS,  # noqa: F401 — re-exported for external use
    VECTOR_FIELDS,  # noqa: F401 — re-exported for external use
    dpf_scalar_to_well,  # noqa: F401 — re-exported for external use
    dpf_vector_to_well,  # noqa: F401 — re-exported for external use
    well_scalar_to_dpf,  # noqa: F401 — re-exported for external use
    well_vector_to_dpf,  # noqa: F401 — re-exported for external use
)

if HAS_TORCH:
    import torch

logger = logging.getLogger(__name__)

# Backward-compatible aliases — tests import these from dpf.ai.surrogate
_SCALAR_KEYS = WALRUS_SCALAR_KEYS
_VECTOR_KEYS = WALRUS_VECTOR_KEYS
_N_CHANNELS = WALRUS_N_CHANNELS
_DPF_TO_WALRUS_FIELD = DPF_TO_WALRUS_FIELD


class DPFSurrogate(WalrusInferenceMixin):
    """
    WALRUS-based surrogate model for DPF plasma dynamics predictions.

    Wraps a WALRUS checkpoint and provides prediction interfaces for DPF state
    evolution. Handles conversion between DPF field format and WALRUS tensor format.

    The checkpoint can be either:
    - A **directory** containing ``walrus.pt`` + ``extended_config.yaml``
      (official HuggingFace format)
    - A single ``.pt`` file with ``model_state_dict`` and ``config`` keys
      (fine-tuned format)

    When the ``walrus`` package is installed, loads a real ``IsotropicModel`` with
    RevIN normalization and runs delta-prediction inference. Otherwise falls back
    to a placeholder that returns the last input state unchanged.

    Args:
        checkpoint_path: Path to WALRUS checkpoint (directory or .pt file)
        device: PyTorch device ("cpu", "cuda", "mps")
        history_length: Number of historical timesteps required for prediction

    Raises:
        ImportError: If PyTorch is not available
        FileNotFoundError: If checkpoint_path does not exist
    """

    def __init__(
        self, checkpoint_path: str | Path | None = None, device: str = "cpu", history_length: int = 4
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DPFSurrogate. "
                "Install with: pip install dpf-unified[ai]"
            )

        if checkpoint_path is None:
            checkpoint_path = self._find_default_checkpoint()

        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
            if not self.checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {self.checkpoint_path}, falling back to placeholder")
                self.checkpoint_path = None
        else:
            self.checkpoint_path = None

        self.device = device
        self.history_length = history_length
        self._model = None
        self._revin = None
        self._formatter = None
        self._walrus_config = None
        self._field_to_index_map: dict[str, int] | None = None
        self._dpf_field_indices: Any = None  # torch.Tensor when loaded

        self._load_model()

    def _find_default_checkpoint(self) -> Path | None:
        """Attempt to locate a default checkpoint.

        Scans common locations in priority order:
        1. ``WALRUS_CHECKPOINT`` environment variable
        2. ``models/walrus-pretrained/walrus.pt`` (project-relative)
        3. ``~/.dpf/walrus.pt`` (user home)
        """
        import os

        # 1. Environment variable
        env_path = os.environ.get("WALRUS_CHECKPOINT")
        if env_path and Path(env_path).is_file():
            return Path(env_path)

        # 2. Project-relative (from cwd)
        project_path = Path("models/walrus-pretrained/walrus.pt")
        if project_path.is_file():
            return project_path

        # 3. User home
        home_path = Path.home() / ".dpf" / "walrus.pt"
        if home_path.is_file():
            return home_path

        return None

    def _load_model(self) -> None:
        """Load WALRUS checkpoint from disk.

        Supports two checkpoint formats:
        1. HuggingFace format: ``{"app": {"model": {state_dict}}}`` +
           separate ``extended_config.yaml``
        2. Fine-tuned format: ``{"model_state_dict": ..., "config": ...}``

        When ``walrus`` is installed (``HAS_WALRUS=True``), instantiates a real
        ``IsotropicModel``. Otherwise stores a placeholder dict.
        """
        if self.checkpoint_path is None:
            logger.warning("No checkpoint path provided. Using prediction placeholder.")
            self._model = {"placeholder": True}
            return

        try:
            pt_path, config_yaml_path = self._resolve_checkpoint_files()

            checkpoint_data = torch.load(
                pt_path, map_location=self.device, weights_only=False
            )

            # Extract state dict — handle both formats
            if "app" in checkpoint_data and "model" in checkpoint_data["app"]:
                # HuggingFace format: checkpoint["app"]["model"]
                state_dict = checkpoint_data["app"]["model"]
            elif "model_state_dict" in checkpoint_data:
                # Fine-tuned format
                state_dict = checkpoint_data["model_state_dict"]
            elif "state_dict" in checkpoint_data:
                state_dict = checkpoint_data["state_dict"]
            else:
                # Assume the checkpoint IS the state dict
                state_dict = checkpoint_data

            if HAS_WALRUS:
                self._load_walrus_model(state_dict, config_yaml_path, checkpoint_data)
            else:
                # Placeholder — store checkpoint info for later
                self._model = {
                    "checkpoint_path": self.checkpoint_path,
                    "device": self.device,
                    "data": checkpoint_data,
                }
                logger.info(
                    f"Loaded checkpoint placeholder from {self.checkpoint_path} "
                    f"(device: {self.device}, walrus not installed)"
                )

        except Exception as e:
            # Fallback to placeholder if load fails
            logger.warning(
                f"Failed to load checkpoint from {self.checkpoint_path}: {e}. "
                "Using prediction placeholder."
            )
            self._model = {"placeholder": True}

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    @property
    def _is_walrus_model(self) -> bool:
        """Return True if ``_model`` is a real WALRUS model (not a placeholder dict)."""
        return self._model is not None and not isinstance(self._model, dict)

    def predict_next_step(
        self, history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """
        Predict next DPF state from historical trajectory.

        Args:
            history: List of DPF states (most recent last). Each state is dict with keys:
                     rho, velocity, pressure, B, Te, Ti, psi (as NumPy arrays)

        Returns:
            Predicted next state with same structure as input states

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If history length is insufficient
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Check checkpoint path and _load_model() logs."
            )

        if len(history) < self.history_length:
            raise ValueError(
                f"Insufficient history: need {self.history_length}, got {len(history)}"
            )

        # Use last history_length states
        recent_history = history[-self.history_length :]

        if self._is_walrus_model:
            return self._walrus_predict(recent_history)

        # Placeholder: return copy of last state
        logger.debug("Running placeholder prediction (walrus not installed)")
        predicted_state = {k: v.copy() for k, v in recent_history[-1].items()}
        return predicted_state

    def _walrus_predict(
        self, recent_history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Run real WALRUS inference on recent history.

        Uses the official WALRUS inference pipeline:
        1. Build a synthetic batch dict from DPF states
        2. Process with ChannelsFirstWithTimeFormatter
        3. RevIN normalize
        4. Model forward pass with field_indices, bcs, metadata
        5. Denormalize delta and add residual
        6. Convert back to DPF state dict

        Parameters
        ----------
        recent_history : list[dict[str, np.ndarray]]
            Last ``history_length`` DPF state dicts.

        Returns
        -------
        dict[str, np.ndarray]
            Predicted next state.
        """
        batch = self._build_walrus_batch(recent_history)
        metadata = batch["metadata"]

        with torch.no_grad():
            # Process input through formatter: batch dict -> (x, field_indices, bcs), y
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

            # Forward pass with all required WALRUS args
            y_pred = self._model(
                normalized[0],      # x: (T, B, C, H, W, D)
                normalized[1],      # field_indices: int tensor
                normalized[2].tolist(),  # boundary_conditions: list
                metadata=metadata,
                train=False,
            )

            # For causal models, take only the last prediction
            if hasattr(self._model, "causal_in_time") and self._model.causal_in_time:
                y_pred = y_pred[-1:]

            # Denormalize delta and add residual
            if self._revin is not None and stats is not None:
                y_pred = (
                    inputs[0][-y_pred.shape[0]:].float()
                    + self._revin.denormalize_delta(y_pred, stats)
                )
            else:
                y_pred = inputs[0][-y_pred.shape[0]:].float() + y_pred

            # Convert back to Well format: (T, B, C, ...) -> (B, T, ..., C)
            y_pred = self._formatter.process_output(y_pred, metadata)

            # Mask padded fields
            if "padded_field_mask" in batch:
                y_pred = y_pred[..., batch["padded_field_mask"]]

        # Extract last predicted step: (B=1, T=1, H, W, D, C) -> (H, W, D, C)
        pred_array = y_pred[0, -1].cpu().numpy()  # (H, W, D, C)

        return self._well_output_to_state(pred_array, recent_history[-1])

    def rollout(
        self, initial_states: list[dict[str, np.ndarray]], n_steps: int
    ) -> list[dict[str, np.ndarray]]:
        """
        Auto-regressive rollout from initial states.

        Args:
            initial_states: Initial trajectory of length >= history_length
            n_steps: Number of steps to predict forward

        Returns:
            List of n_steps predicted states

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If initial_states length is insufficient
        """
        if len(initial_states) < self.history_length:
            raise ValueError(
                f"Need at least {self.history_length} initial states, "
                f"got {len(initial_states)}"
            )

        trajectory = list(initial_states)
        predictions = []

        for step in range(n_steps):
            next_state = self.predict_next_step(trajectory)
            predictions.append(next_state)
            trajectory.append(next_state)

            if (step + 1) % 10 == 0:
                logger.debug(f"Rollout progress: {step + 1}/{n_steps}")

        return predictions

    def parameter_sweep(
        self, configs: list[dict[str, Any]], n_steps: int = 100
    ) -> list[dict[str, Any]]:
        """
        Evaluate surrogate over parameter configurations.

        Args:
            configs: List of configuration dicts with DPF parameters
                     (V0, pressure0, etc.)
            n_steps: Number of timesteps to simulate per config

        Returns:
            List of summary dicts with ``config`` and ``metrics`` keys
        """
        results = []

        for i, config in enumerate(configs):
            logger.debug(f"Parameter sweep: config {i + 1}/{len(configs)}")

            # Create initial states (zeros with appropriate shape)
            # Use 16x16x16 default grid — WALRUS minimum kernel size requires >= 16
            shape = (16, 16, 16)
            initial_state = self._create_initial_state(config, shape)

            # Create history by repeating initial state
            initial_history = [initial_state] * self.history_length

            # Run rollout
            try:
                trajectory = self.rollout(initial_history, n_steps)

                # Extract scalar summaries
                summary = self._extract_summary(trajectory, config)
                results.append(summary)

            except Exception as e:
                logger.error(f"Parameter sweep failed for config {i}: {e}")
                results.append({"error": str(e), "config_idx": i})

        return results

    def validate_against_physics(
        self,
        trajectory: list[dict[str, np.ndarray]],
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Cross-validate surrogate predictions against a physics trajectory.

        Slides a window over the physics trajectory, predicts each next step
        with the surrogate, and computes per-field L2 errors. This quantifies
        how well the surrogate tracks the physics engine, which is useful for
        identifying where fine-tuning data would have the highest impact.

        Parameters
        ----------
        trajectory : list[dict[str, np.ndarray]]
            Full physics trajectory (at least ``history_length + 1`` states).
        fields : list[str] or None
            Fields to compare. If None, uses all scalar + vector fields.

        Returns
        -------
        dict[str, Any]
            Validation report with keys:
            - ``n_steps``: number of steps validated
            - ``per_field_l2``: dict mapping field name to list of L2 errors
            - ``mean_l2``: overall mean L2 error across all fields/steps
            - ``max_l2``: maximum L2 error seen
            - ``diverging_steps``: list of step indices where L2 > 0.3
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        hl = self.history_length
        if len(trajectory) < hl + 1:
            raise ValueError(
                f"Trajectory too short for validation: need {hl + 1}, "
                f"got {len(trajectory)}"
            )

        if fields is None:
            fields = list(_SCALAR_KEYS) + list(_VECTOR_KEYS)

        per_field_l2: dict[str, list[float]] = {f: [] for f in fields}
        diverging_steps: list[int] = []
        all_l2: list[float] = []

        for i in range(hl, len(trajectory)):
            history = trajectory[i - hl:i]
            actual = trajectory[i]
            predicted = self.predict_next_step(history)

            step_l2_values: list[float] = []
            for field in fields:
                if field not in actual or field not in predicted:
                    continue
                pred_arr = predicted[field]
                actual_arr = actual[field]
                if pred_arr.shape != actual_arr.shape:
                    continue
                diff_norm = float(np.linalg.norm(pred_arr - actual_arr))
                actual_norm = max(float(np.linalg.norm(actual_arr)), 1e-10)
                l2 = diff_norm / actual_norm
                per_field_l2[field].append(l2)
                step_l2_values.append(l2)

            if step_l2_values:
                step_mean = float(np.mean(step_l2_values))
                all_l2.append(step_mean)
                if step_mean > 0.3:
                    diverging_steps.append(i)

        return {
            "n_steps": len(all_l2),
            "per_field_l2": per_field_l2,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else 0.0,
            "max_l2": float(np.max(all_l2)) if all_l2 else 0.0,
            "diverging_steps": diverging_steps,
        }

    def _create_initial_state(
        self, config: dict[str, Any], shape: tuple[int, ...]
    ) -> dict[str, np.ndarray]:
        """
        Create initial DPF state from configuration.

        Args:
            config: Configuration dict with DPF parameters
            shape: Spatial grid shape

        Returns:
            Initial state dict with all required fields
        """
        # Extract parameters with defaults
        rho0 = config.get("rho0", 1e-6)
        pressure0 = config.get("pressure0", 100.0)
        Te0 = config.get("Te0", 1.0)
        Ti0 = config.get("Ti0", 1.0)

        state = {
            "rho": np.full(shape, rho0, dtype=np.float64),
            "velocity": np.zeros((3, *shape), dtype=np.float64),
            "pressure": np.full(shape, pressure0, dtype=np.float64),
            "B": np.zeros((3, *shape), dtype=np.float64),
            "Te": np.full(shape, Te0, dtype=np.float64),
            "Ti": np.full(shape, Ti0, dtype=np.float64),
            "psi": np.zeros(shape, dtype=np.float64),
        }

        return state

    def _extract_summary(
        self, trajectory: list[dict[str, np.ndarray]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract scalar summaries from trajectory.

        Returns a dict with separate ``config`` and ``metrics`` keys
        (matching GUI frontend expectations).

        Args:
            trajectory: List of DPF states
            config: Configuration used for this trajectory

        Returns:
            Summary dict with ``config`` and ``metrics`` sub-dicts
        """
        # Compute max values over trajectory
        max_rho = max(np.max(state["rho"]) for state in trajectory)
        max_Te = max(np.max(state["Te"]) for state in trajectory)
        max_Ti = max(np.max(state["Ti"]) for state in trajectory)
        max_pressure = max(np.max(state["pressure"]) for state in trajectory)

        # Compute mean magnetic field magnitude
        B_mags = [np.sqrt(np.sum(state["B"] ** 2, axis=0)) for state in trajectory]
        mean_B = np.mean([np.mean(B_mag) for B_mag in B_mags])
        max_B = max(np.max(B_mag) for B_mag in B_mags)

        # Final values
        final_state = trajectory[-1]
        final_rho = np.mean(final_state["rho"])
        final_pressure = np.mean(final_state["pressure"])

        return {
            "config": config,
            "metrics": {
                "max_rho": float(max_rho),
                "max_Te": float(max_Te),
                "max_Ti": float(max_Ti),
                "max_pressure": float(max_pressure),
                "mean_B": float(mean_B),
                "max_B": float(max_B),
                "final_rho": float(final_rho),
                "final_pressure": float(final_pressure),
                "n_steps": len(trajectory),
            },
        }
