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

# Mapping from DPF field names to WALRUS pretrained field_to_index_map names.
# These indices come from the extended_config.yaml field_index_map_override.
_DPF_TO_WALRUS_FIELD = {
    "rho": ("density", 28),
    "Te": ("temperature", 46),
    "Ti": ("temperature", 46),  # Reuse temperature embedding for Ti
    "pressure": ("pressure", 3),
    "psi": ("A", 42),  # Vector potential analog
    "Bx": ("magnetic_field_x", 39),
    "By": ("magnetic_field_y", 40),
    "Bz": ("magnetic_field_z", 41),
    "vx": ("velocity_x", 4),
    "vy": ("velocity_y", 5),
    "vz": ("velocity_z", 6),
}


class DPFSurrogate:
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
        self._walrus_config = None
        self._field_to_index_map: dict[str, int] | None = None
        self._dpf_field_indices: Any = None  # torch.Tensor when loaded

        self._load_model()

    def _resolve_checkpoint_files(self) -> tuple[Path, Path | None]:
        """Resolve checkpoint .pt file and optional config YAML.

        Returns
        -------
        tuple[Path, Path | None]
            (checkpoint_pt_path, config_yaml_path) — config may be None if
            the checkpoint is a single .pt file containing its own config.
        """
        if self.checkpoint_path.is_dir():
            # Directory format: walrus.pt + extended_config.yaml
            pt_path = self.checkpoint_path / "walrus.pt"
            if not pt_path.exists():
                # Try any .pt file in the directory
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
            # Single .pt file — config may be embedded or in same directory
            config_path = self.checkpoint_path.parent / "extended_config.yaml"
            if not config_path.exists():
                config_path = None
            return self.checkpoint_path, config_path

    def _load_model(self) -> None:
        """Load WALRUS checkpoint from disk.

        Supports two checkpoint formats:
        1. HuggingFace format: ``{"app": {"model": {state_dict}}}`` +
           separate ``extended_config.yaml``
        2. Fine-tuned format: ``{"model_state_dict": ..., "config": ...}``

        When ``walrus`` is installed (``HAS_WALRUS=True``), instantiates a real
        ``IsotropicModel``. Otherwise stores a placeholder dict.
        """
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
            self._model = None
            logger.warning(
                f"Failed to load checkpoint from {self.checkpoint_path}: {e}. "
                "Model will not be available for predictions."
            )

    def _load_walrus_model(
        self,
        state_dict: dict,
        config_yaml_path: Path | None,
        checkpoint_data: dict,
    ) -> None:
        """Instantiate real WALRUS IsotropicModel from checkpoint.

        Parameters
        ----------
        state_dict : dict
            Model weights (already extracted from checkpoint).
        config_yaml_path : Path or None
            Path to ``extended_config.yaml``. If None, tries ``checkpoint_data["config"]``.
        checkpoint_data : dict
            Raw checkpoint data (for fallback config extraction).
        """
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter

        # Load config — prefer YAML file, fall back to embedded config
        config = None
        if config_yaml_path is not None and config_yaml_path.exists():
            config = OmegaConf.load(config_yaml_path)
            logger.info(f"Loaded WALRUS config from {config_yaml_path}")
        elif "config" in checkpoint_data:
            config = checkpoint_data["config"]
            logger.info("Using embedded config from checkpoint")

        if config is None:
            logger.warning(
                "No config found (no extended_config.yaml and no embedded config). "
                "Falling back to placeholder."
            )
            self._model = {
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "data": checkpoint_data,
            }
            return

        self._walrus_config = config

        # Get field_to_index_map from config
        field_map = dict(config.data.get("field_index_map_override", {}))
        if field_map:
            self._field_to_index_map = field_map
        else:
            # Use our DPF defaults
            self._field_to_index_map = {
                name: idx for _, (name, idx) in _DPF_TO_WALRUS_FIELD.items()
            }

        # Compute n_states from the field map
        n_states = max(self._field_to_index_map.values()) + 1

        # Instantiate model
        model = instantiate(config.model, n_states=n_states)

        # If our field map differs from the checkpoint's, align weights
        try:
            from walrus.utils.experiment_utils import (
                align_checkpoint_with_field_to_index_map,
            )
            checkpoint_field_map = field_map  # Same map for pretrained
            model_field_map = dict(self._field_to_index_map)
            aligned_state_dict = align_checkpoint_with_field_to_index_map(
                checkpoint_state_dict=state_dict,
                model_state_dict=model.state_dict(),
                checkpoint_field_to_index_map=checkpoint_field_map,
                model_field_to_index_map=model_field_map,
            )
            model.load_state_dict(aligned_state_dict)
        except Exception:
            # Direct load without alignment
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)
        self._model = model

        # Build DPF field indices tensor
        # Order: rho, Te, Ti, pressure, psi, Bx, By, Bz, vx, vy, vz
        dpf_field_names = [
            "density", "temperature", "temperature", "pressure", "A",
            "magnetic_field_x", "magnetic_field_y", "magnetic_field_z",
            "velocity_x", "velocity_y", "velocity_z",
        ]
        self._dpf_field_indices = torch.tensor(
            [self._field_to_index_map.get(name, 0) for name in dpf_field_names],
            device=self.device, dtype=torch.long,
        )

        # Set up RevIN normalization
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
                logger.warning("RevIN unavailable; inference may be degraded")
                self._revin = None

        # Set up input formatter
        self._formatter = ChannelsFirstWithTimeFormatter()

        logger.info(
            f"Loaded WALRUS IsotropicModel from {self.checkpoint_path} "
            f"(n_states={n_states}, device={self.device})"
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
            # Process input through formatter: batch dict → (x, field_indices, bcs), y
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

            # Convert back to Well format: (T, B, C, ...) → (B, T, ..., C)
            y_pred = self._formatter.process_output(y_pred, metadata)

            # Mask padded fields
            if "padded_field_mask" in batch:
                y_pred = y_pred[..., batch["padded_field_mask"]]

        # Extract last predicted step: (B=1, T=1, H, W, D, C) → (H, W, D, C)
        pred_array = y_pred[0, -1].cpu().numpy()  # (H, W, D, C)

        return self._well_output_to_state(pred_array, recent_history[-1])

    def _build_walrus_batch(
        self, states: list[dict[str, np.ndarray]]
    ) -> dict[str, Any]:
        """Build a WALRUS-compatible batch dict from DPF state history.

        Creates the synthetic batch format shown in WALRUS demo notebook
        ``walrus_example_1_RunningWalrus.ipynb`` (Part 2: Non-Well data).

        Parameters
        ----------
        states : list[dict[str, np.ndarray]]
            List of DPF state dicts with keys: rho, Te, Ti, pressure, psi,
            B (3, *spatial), velocity (3, *spatial).

        Returns
        -------
        dict
            Batch dict with keys: input_fields, output_fields, constant_fields,
            boundary_conditions, padded_field_mask, field_indices, metadata.
        """
        from the_well.data.datasets import WellMetadata

        T = len(states)
        ref = states[0]

        # Infer spatial shape from rho
        spatial = ref["rho"].shape  # e.g., (nx, ny, nz)
        n_spatial = len(spatial)

        # Pad to 3D if needed (WALRUS expects 3 spatial dims)
        if n_spatial == 2:
            spatial = (*spatial, 1)
        elif n_spatial == 1:
            spatial = (*spatial, 1, 1)

        C = _N_CHANNELS  # 11

        # Build input_fields: [B=1, T, H, W, D, C] — channels LAST (Well format)
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
                    vec = state[key].astype(np.float32)  # (3, *orig_spatial)
                    for comp in range(3):
                        comp_field = vec[comp]
                        if comp_field.ndim < 3:
                            comp_field = comp_field.reshape(spatial)
                        input_arr[0, t, ..., ch + comp] = comp_field
                ch += 3

        device = self.device
        input_tensor = torch.from_numpy(input_arr).to(device)

        # Output fields: same shape, single step (placeholder — not used in inference)
        output_tensor = torch.zeros(
            (1, 1, *spatial, C), dtype=torch.float32, device=device
        )

        # Constant fields: none for DPF
        const_tensor = torch.zeros(
            (1, *spatial, 0), dtype=torch.float32, device=device
        )

        # Boundary conditions: DPF uses wall boundaries on all faces
        # Shape: [B, n_dims, 2] with values WALL=0, OPEN=1, PERIODIC=2
        bcs = torch.tensor(
            [[[0, 0], [0, 0], [0, 0]]],
            dtype=torch.long, device=device,
        )

        # Padded field mask: all True (no padding)
        padded_mask = torch.ones(C, dtype=torch.bool, device=device)

        # Field indices: map each DPF channel to its WALRUS embedding index
        field_indices = self._dpf_field_indices.clone()

        # Metadata
        metadata = WellMetadata(
            dataset_name="dpf_plasma",
            n_spatial_dims=3,
            spatial_resolution=spatial,
            scalar_names=list(_SCALAR_KEYS),
            constant_scalar_names=[],
            field_names={
                0: ["density", "temperature", "temperature", "pressure", "A"],
                1: ["velocity_x", "velocity_y", "velocity_z",
                    "magnetic_field_x", "magnetic_field_y", "magnetic_field_z"],
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

    def _well_output_to_state(
        self, pred_array: np.ndarray, reference_state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert WALRUS output array (Well channels-last) to DPF state dict.

        Parameters
        ----------
        pred_array : np.ndarray
            Shape (H, W, D, C) — channels-last Well format output.
        reference_state : dict[str, np.ndarray]
            Reference DPF state for shape and key information.

        Returns
        -------
        dict[str, np.ndarray]
            DPF state dict.
        """
        orig_spatial = reference_state["rho"].shape

        state: dict[str, np.ndarray] = {}
        ch = 0

        for key in _SCALAR_KEYS:
            if key in reference_state:
                field = pred_array[..., ch].astype(np.float64)
                # Reshape back to original spatial dims
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
            List of summary dicts with ``config`` and ``metrics`` keys
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
