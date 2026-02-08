from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dpf.ai import HAS_TORCH, HAS_WALRUS
from dpf.ai.field_mapping import (
    SCALAR_FIELDS,
    VECTOR_FIELDS,
    dpf_scalar_to_well,
    dpf_vector_to_well,
    well_scalar_to_dpf,  # noqa: F401 — used by _states_to_tensor, monkeypatched in tests
    well_vector_to_dpf,  # noqa: F401 — used by _states_to_tensor, monkeypatched in tests
)

if HAS_TORCH:
    import torch

logger = logging.getLogger(__name__)

# Channel order for WALRUS tensor: 5 scalars + 2×3 vector components = 11
_SCALAR_KEYS = ("rho", "Te", "Ti", "pressure", "psi")
_VECTOR_KEYS = ("B", "velocity")
_N_CHANNELS = len(_SCALAR_KEYS) + len(_VECTOR_KEYS) * 3  # 11


class DPFSurrogate:
    """
    WALRUS-based surrogate model for DPF plasma dynamics predictions.

    Wraps a fine-tuned WALRUS checkpoint and provides prediction interfaces for DPF state
    evolution. Handles conversion between DPF field format and WALRUS tensor format.

    When the ``walrus`` package is installed, loads a real ``IsotropicModel`` with RevIN
    normalization and runs delta-prediction inference. Otherwise falls back to a
    placeholder that returns the last input state unchanged.

    Args:
        checkpoint_path: Path to WALRUS checkpoint file (.pt or .pth)
        device: PyTorch device ("cpu", "cuda", "mps")
        history_length: Number of historical timesteps required for prediction

    Raises:
        ImportError: If PyTorch is not available
        FileNotFoundError: If checkpoint_path does not exist
    """

    def __init__(
        self, checkpoint_path: str | Path, device: str = "cpu", history_length: int = 4
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DPFSurrogate. "
                "Install with: pip install dpf-unified[ai]"
            )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = device
        self.history_length = history_length
        self._model = None
        self._revin = None
        self._formatter = None

        self._load_model()

    def _load_model(self) -> None:
        """Load WALRUS checkpoint from disk.

        When ``walrus`` is installed (``HAS_WALRUS=True``), instantiates a real
        ``IsotropicModel`` with RevIN normalization. Otherwise stores a placeholder
        dict indicating the checkpoint was loaded successfully.

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        try:
            checkpoint_data = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )

            if HAS_WALRUS:
                self._load_walrus_model(checkpoint_data)
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
            self._model = None
            logger.warning(
                f"Failed to load checkpoint from {self.checkpoint_path}: {e}. "
                "Model will not be available for predictions."
            )

    def _load_walrus_model(self, checkpoint_data: dict) -> None:
        """Instantiate real WALRUS IsotropicModel from checkpoint data.

        Parameters
        ----------
        checkpoint_data : dict
            Loaded checkpoint containing ``model_state_dict`` and ``config``.
        """
        from hydra.utils import instantiate
        from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter

        config = checkpoint_data.get("config")
        if config is None:
            logger.warning("Checkpoint missing 'config' key; falling back to placeholder")
            self._model = {
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "data": checkpoint_data,
            }
            return

        # Instantiate model with correct number of DPF channels
        model = instantiate(config.model, n_states=_N_CHANNELS)

        # Load trained weights
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict"))
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)
        self._model = model

        # Set up RevIN normalization
        try:
            self._revin = instantiate(config.trainer.revin)()
        except Exception:
            logger.warning("Failed to instantiate RevIN from config; inference may be degraded")
            self._revin = None

        # Set up input formatter
        self._formatter = ChannelsFirstWithTimeFormatter()

        logger.info(
            f"Loaded WALRUS IsotropicModel from {self.checkpoint_path} "
            f"(n_states={_N_CHANNELS}, device={self.device})"
        )

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

        Pipeline: states → tensor → RevIN normalize → model forward →
                  denormalize delta → add residual → state dict.

        Parameters
        ----------
        recent_history : list[dict[str, np.ndarray]]
            Last ``history_length`` DPF state dicts.

        Returns
        -------
        dict[str, np.ndarray]
            Predicted next state.
        """
        # Convert DPF states to WALRUS input tensor: (1, T, C, *spatial)
        input_tensor = self._states_to_walrus_tensor(recent_history)

        with torch.no_grad():
            if self._revin is not None:
                # RevIN normalize
                stats = self._revin.compute_stats(input_tensor, metadata=None, epsilon=1e-5)
                normalized_x = self._revin.normalize_stdmean(input_tensor, stats)
            else:
                normalized_x = input_tensor
                stats = None

            # Forward pass — simplified direct call (no formatter/metadata)
            y_pred = self._model(normalized_x)

            if self._revin is not None and stats is not None:
                # Denormalize delta and add residual
                y_pred = input_tensor[-y_pred.shape[0]:].float() + self._revin.denormalize_delta(
                    y_pred, stats
                )
            else:
                # Delta mode without RevIN: output + last input
                y_pred = input_tensor[:, -1:, ...] + y_pred

        # Extract last predicted step: (1, 1, C, *spatial) → (C, *spatial)
        predicted_tensor = y_pred[0, -1]  # (C, *spatial)

        return self._tensor_to_state(predicted_tensor, recent_history[-1])

    def _states_to_walrus_tensor(
        self, states: list[dict[str, np.ndarray]]
    ) -> Any:
        """Convert DPF states to WALRUS input tensor.

        Builds a tensor of shape ``(1, T, C, *spatial)`` where ``C=11`` channels are
        ordered: rho, Te, Ti, pressure, psi, Bx, By, Bz, vx, vy, vz.

        Parameters
        ----------
        states : list[dict[str, np.ndarray]]
            List of DPF state dicts.

        Returns
        -------
        torch.Tensor
            Shape ``(1, T, 11, *spatial)``, float32, on ``self.device``.
        """
        T = len(states)
        # Infer spatial shape from first scalar field in first state
        ref_state = states[0]
        for key in _SCALAR_KEYS:
            if key in ref_state:
                spatial_shape = ref_state[key].shape
                break
        else:
            raise ValueError("No scalar field found in state dict")

        # Pre-allocate: (T, C, *spatial)
        arr = np.zeros((T, _N_CHANNELS, *spatial_shape), dtype=np.float32)

        for t, state in enumerate(states):
            ch = 0
            # Scalars: rho, Te, Ti, pressure, psi
            for key in _SCALAR_KEYS:
                if key in state:
                    arr[t, ch] = state[key].astype(np.float32)
                ch += 1

            # Vectors: B (3 components), velocity (3 components)
            for key in _VECTOR_KEYS:
                if key in state:
                    vec = state[key].astype(np.float32)  # (3, *spatial)
                    for comp in range(3):
                        arr[t, ch + comp] = vec[comp]
                ch += 3

        # Add batch dim: (1, T, C, *spatial)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return tensor.to(self.device)

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
            List of summary dicts with scalar metrics (max_rho, max_Te, etc.)
        """
        results = []

        for i, config in enumerate(configs):
            logger.debug(f"Parameter sweep: config {i + 1}/{len(configs)}")

            # Create initial states (zeros with appropriate shape)
            # Use 8x8x8 default grid for parameter sweeps
            shape = (8, 8, 8)
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

    def _states_to_tensor(self, states: list[dict[str, np.ndarray]]) -> Any:
        """
        Convert DPF states to WALRUS input tensor format.

        Args:
            states: List of DPF state dicts

        Returns:
            PyTorch tensor of shape (1, history_length, n_fields, *spatial)
        """
        # Stack all fields from all timesteps
        field_arrays = []

        for state in states:
            # Process scalar fields
            for field_name in SCALAR_FIELDS:
                if field_name in state:
                    well_field = dpf_scalar_to_well(state[field_name], field_name)
                    field_arrays.append(well_field)

            # Process vector fields
            for field_name in VECTOR_FIELDS:
                if field_name in state:
                    well_field = dpf_vector_to_well(state[field_name], field_name)
                    # Split components
                    for comp in range(well_field.shape[0]):
                        field_arrays.append(well_field[comp])

        # Stack into single array: (history_length, n_fields, *spatial)
        n_fields = len(field_arrays) // len(states)
        spatial_dims = field_arrays[0].shape

        # Reshape into (history_length, n_fields, *spatial)
        stacked = np.stack(field_arrays, axis=0)
        stacked = stacked.reshape(len(states), n_fields, *spatial_dims)

        # Add batch dimension: (1, history_length, n_fields, *spatial)
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()

        return tensor.to(self.device)

    def _tensor_to_state(
        self, tensor: Any, reference_state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Convert WALRUS output tensor back to DPF state format.

        Uses the same channel ordering as ``_states_to_walrus_tensor``:
        rho, Te, Ti, pressure, psi, Bx, By, Bz, vx, vy, vz.

        Args:
            tensor: PyTorch tensor or numpy array — shape (C, *spatial) or (1, C, *spatial)
            reference_state: Reference DPF state for shape information

        Returns:
            DPF state dict with same structure as reference_state
        """
        # Convert to numpy
        if hasattr(tensor, "cpu"):
            array = tensor.cpu().numpy()
        else:
            array = np.asarray(tensor)

        # Remove batch dimension if present
        if array.ndim >= 4 and array.shape[0] == 1:
            array = array[0]

        state: dict[str, np.ndarray] = {}
        ch = 0

        # Extract scalar fields: rho, Te, Ti, pressure, psi
        for key in _SCALAR_KEYS:
            if key in reference_state:
                state[key] = array[ch].astype(np.float64)
            ch += 1

        # Extract vector fields: B (3 components), velocity (3 components)
        for key in _VECTOR_KEYS:
            if key in reference_state:
                components = [array[ch + i].astype(np.float64) for i in range(3)]
                state[key] = np.stack(components, axis=0)
            ch += 3

        return state

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

        Args:
            trajectory: List of DPF states
            config: Configuration used for this trajectory

        Returns:
            Summary dict with max/mean/final values for key quantities
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

        summary = {
            "max_rho": float(max_rho),
            "max_Te": float(max_Te),
            "max_Ti": float(max_Ti),
            "max_pressure": float(max_pressure),
            "mean_B": float(mean_B),
            "max_B": float(max_B),
            "final_rho": float(final_rho),
            "final_pressure": float(final_pressure),
            "n_steps": len(trajectory),
            **config,  # Include original config parameters
        }

        return summary
