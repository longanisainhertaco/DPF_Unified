from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dpf.ai import HAS_TORCH
from dpf.ai.field_mapping import (
    SCALAR_FIELDS,
    VECTOR_FIELDS,
    dpf_scalar_to_well,
    dpf_vector_to_well,
    well_scalar_to_dpf,
    well_vector_to_dpf,
)

if HAS_TORCH:
    import torch

logger = logging.getLogger(__name__)


class DPFSurrogate:
    """
    WALRUS-based surrogate model for DPF plasma dynamics predictions.

    Wraps a fine-tuned WALRUS checkpoint and provides prediction interfaces for DPF state
    evolution. Handles conversion between DPF field format and WALRUS tensor format.

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

        self._load_model()

    def _load_model(self) -> None:
        """
        Load WALRUS checkpoint from disk.

        For now, stores a placeholder indicating model location exists. Full WALRUS loading
        will be implemented when walrus package integration is complete.

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        try:
            # Attempt to load checkpoint to verify it's valid
            checkpoint_data = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )

            # For now, store placeholder indicating checkpoint is valid
            # Full WALRUS model instantiation will be added in Phase I
            self._model = {
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "data": checkpoint_data,
            }

            logger.info(
                f"Loaded checkpoint placeholder from {self.checkpoint_path} "
                f"(device: {self.device})"
            )

        except Exception as e:
            self._model = None
            logger.warning(
                f"Failed to load checkpoint from {self.checkpoint_path}: {e}. "
                "Model will not be available for predictions."
            )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

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

        # Convert to tensor format (used when real WALRUS model is loaded)
        _input_tensor = self._states_to_tensor(recent_history)

        # Run forward pass (placeholder implementation until WALRUS integration)
        # For now, predict zero delta (return last state unchanged)
        logger.debug("Running placeholder prediction (WALRUS integration pending)")
        predicted_state = {k: v.copy() for k, v in recent_history[-1].items()}

        return predicted_state

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

        Args:
            tensor: PyTorch tensor from model output
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
        if array.ndim == 4:  # (1, n_fields, *spatial)
            array = array[0]

        state = {}
        field_idx = 0

        # Extract scalar fields
        for field_name in SCALAR_FIELDS:
            if field_name in reference_state:
                well_field = array[field_idx]
                state[field_name] = well_scalar_to_dpf(well_field, field_name)
                field_idx += 1

        # Extract vector fields
        for field_name in VECTOR_FIELDS:
            if field_name in reference_state:
                # Vector fields have 3 components
                components = [array[field_idx + i] for i in range(3)]
                well_field = np.stack(components, axis=0)
                state[field_name] = well_vector_to_dpf(well_field, field_name)
                field_idx += 3

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
