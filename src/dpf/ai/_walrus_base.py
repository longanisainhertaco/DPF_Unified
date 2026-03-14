"""Shared WALRUS inference utilities for DPF surrogate classes.

Provides ``WalrusInferenceMixin`` — a mixin class that extracts the common
model-loading, batch-construction, and output-conversion logic used by
:class:`~dpf.ai.surrogate.DPFSurrogate`.

All ``torch`` / ``walrus`` imports are lazy (inside methods) so that importing
this module never triggers heavyweight dependency loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Shared constants — canonical channel layout for DPF ↔ WALRUS
# -----------------------------------------------------------------------

WALRUS_SCALAR_KEYS: tuple[str, ...] = ("rho", "Te", "Ti", "pressure", "psi")
WALRUS_VECTOR_KEYS: tuple[str, ...] = ("B", "velocity")
WALRUS_N_CHANNELS: int = len(WALRUS_SCALAR_KEYS) + len(WALRUS_VECTOR_KEYS) * 3  # 11

# DPF field name → (WALRUS embedding name, embedding index)
DPF_TO_WALRUS_FIELD: dict[str, tuple[str, int]] = {
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

# Ordered WALRUS embedding names matching the 11-channel DPF layout
DPF_WALRUS_FIELD_NAMES: list[str] = [
    "density", "temperature", "temperature", "pressure", "A",
    "magnetic_field_x", "magnetic_field_y", "magnetic_field_z",
    "velocity_x", "velocity_y", "velocity_z",
]


class WalrusInferenceMixin:
    """Mixin providing shared WALRUS model-loading and inference helpers.

    Subclasses must set the following attributes before calling mixin methods:

    - ``checkpoint_path`` (`Path`): Path to WALRUS checkpoint
    - ``_model``: Loaded model or placeholder dict
    - ``_revin``: RevIN normalization instance or ``None``
    - ``_formatter``: ``ChannelsFirstWithTimeFormatter`` or ``None``
    - ``_walrus_config``: OmegaConf config or ``None``
    - ``_field_to_index_map`` (`dict | None`): WALRUS field index map
    - ``_dpf_field_indices``: ``torch.Tensor`` of embedding indices

    The mixin uses :pyattr:`_torch_device` to discover the PyTorch device
    string, resolving the attribute-name difference between subclasses
    (``DPFSurrogate.device`` vs ``MLXSurrogate._device``).
    """

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @property
    def _torch_device(self) -> str:
        """Return the PyTorch device string for WALRUS inference.

        Checks ``_device`` first (MLXSurrogate), then ``device``
        (DPFSurrogate), falling back to ``"cpu"``.
        """
        dev = getattr(self, "_device", None)
        if dev is not None:
            return dev
        return getattr(self, "device", "cpu")

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_checkpoint_files(self) -> tuple[Path, Path | None]:
        """Resolve checkpoint ``.pt`` file and optional config YAML.

        Returns
        -------
        tuple[Path, Path | None]
            ``(pt_path, config_yaml_path)`` — config may be ``None`` if
            the checkpoint is a single ``.pt`` file containing its own config.
        """
        cp: Path = self.checkpoint_path  # type: ignore[attr-defined]

        if cp.is_dir():
            pt_path = cp / "walrus.pt"
            if not pt_path.exists():
                pt_files = list(cp.glob("*.pt"))
                if pt_files:
                    pt_path = pt_files[0]
                else:
                    raise FileNotFoundError(
                        f"No .pt file found in {cp}"
                    )
            config_path = cp / "extended_config.yaml"
            if not config_path.exists():
                config_path = None
            return pt_path, config_path
        else:
            config_path = cp.parent / "extended_config.yaml"
            if not config_path.exists():
                config_path = None
            return cp, config_path

    # ------------------------------------------------------------------
    # Model instantiation
    # ------------------------------------------------------------------

    def _load_walrus_model(
        self,
        state_dict: dict,
        config_yaml_path: Path | None,
        checkpoint_data: dict,
    ) -> None:
        """Instantiate the real WALRUS ``IsotropicModel`` from checkpoint.

        Parameters
        ----------
        state_dict : dict
            Model weights already extracted from the checkpoint.
        config_yaml_path : Path or None
            Path to ``extended_config.yaml``.  If ``None``, tries
            ``checkpoint_data["config"]``.
        checkpoint_data : dict
            Raw checkpoint data for fallback config extraction.

        Raises
        ------
        RuntimeError
            If no config is found (no YAML and no embedded config).
        """
        import torch
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
                "No WALRUS config found (no extended_config.yaml and no "
                "embedded config). Cannot instantiate model."
            )

        self._walrus_config = config  # type: ignore[attr-defined]

        # Field index map
        field_map = dict(config.data.get("field_index_map_override", {}))
        if field_map:
            self._field_to_index_map = field_map  # type: ignore[attr-defined]
        else:
            self._field_to_index_map = {  # type: ignore[attr-defined]
                name: idx
                for _, (name, idx) in DPF_TO_WALRUS_FIELD.items()
            }

        n_states = max(self._field_to_index_map.values()) + 1  # type: ignore[attr-defined]

        # Instantiate model via Hydra
        model = instantiate(config.model, n_states=n_states)

        # Align weights if possible
        try:
            from walrus.utils.experiment_utils import (
                align_checkpoint_with_field_to_index_map,
            )
            aligned = align_checkpoint_with_field_to_index_map(
                checkpoint_state_dict=state_dict,
                model_state_dict=model.state_dict(),
                checkpoint_field_to_index_map=field_map,
                model_field_to_index_map=dict(self._field_to_index_map),  # type: ignore[attr-defined]
            )
            model.load_state_dict(aligned)
        except Exception:
            model.load_state_dict(state_dict)

        model.eval()
        device = self._torch_device
        model.to(device)
        self._model = model  # type: ignore[attr-defined]

        # DPF field indices tensor
        self._dpf_field_indices = torch.tensor(  # type: ignore[attr-defined]
            [
                self._field_to_index_map.get(name, 0)  # type: ignore[attr-defined]
                for name in DPF_WALRUS_FIELD_NAMES
            ],
            device=device,
            dtype=torch.long,
        )

        # RevIN normalization
        try:
            self._revin = instantiate(config.trainer.revin)()  # type: ignore[attr-defined]
        except Exception:
            logger.warning(
                "Failed to instantiate RevIN from config; "
                "trying SamplewiseRevNormalization directly"
            )
            try:
                from walrus.trainer.normalization_strat import (
                    SamplewiseRevNormalization,
                )
                self._revin = SamplewiseRevNormalization()  # type: ignore[attr-defined]
            except Exception:
                logger.warning(
                    "RevIN unavailable; inference may be degraded"
                )
                self._revin = None  # type: ignore[attr-defined]

        # Formatter
        self._formatter = ChannelsFirstWithTimeFormatter()  # type: ignore[attr-defined]

        logger.info(
            "Loaded WALRUS IsotropicModel (n_states=%d, device=%s)",
            n_states, device,
        )

    # ------------------------------------------------------------------
    # Batch construction
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
        import torch
        from the_well.data.datasets import WellMetadata

        T = len(states)

        # Squeeze singleton leading dims from Athena++ (which returns 4D
        # arrays like (1, nx, ny, nz) instead of 3D (nx, ny, nz))
        squeezed_states: list[dict[str, Any]] = []
        for state in states:
            squeezed: dict[str, Any] = {}
            for key, val in state.items():
                if isinstance(val, np.ndarray):
                    while val.ndim > 3 and key not in ("B", "velocity"):
                        if val.shape[0] == 1:
                            val = val.squeeze(0)
                        else:
                            break
                    while val.ndim > 4 and key in ("B", "velocity"):
                        if val.shape[0] == 3 and val.shape[1] == 1:
                            val = val.squeeze(1)
                        elif val.shape[0] == 1:
                            val = val.squeeze(0)
                        else:
                            break
                squeezed[key] = val
            squeezed_states.append(squeezed)
        states = squeezed_states

        ref = states[0]

        # Infer spatial shape from rho
        spatial = ref["rho"].shape  # e.g. (nx, ny, nz)

        # Pad to 3D if needed (WALRUS expects 3 spatial dims)
        if len(spatial) == 2:
            spatial = (*spatial, 1)
        elif len(spatial) == 1:
            spatial = (*spatial, 1, 1)

        C = WALRUS_N_CHANNELS

        # input_fields: [B=1, T, H, W, D, C] — channels LAST (Well format)
        input_arr = np.zeros((1, T, *spatial, C), dtype=np.float32)

        for t, state in enumerate(states):
            ch = 0
            for key in WALRUS_SCALAR_KEYS:
                if key in state:
                    field = state[key].astype(np.float32)
                    if field.ndim < 3:
                        field = field.reshape(spatial)
                    input_arr[0, t, ..., ch] = field
                ch += 1
            for key in WALRUS_VECTOR_KEYS:
                if key in state:
                    vec = state[key].astype(np.float32)
                    for comp in range(3):
                        comp_field = vec[comp]
                        if comp_field.ndim < 3:
                            comp_field = comp_field.reshape(spatial)
                        input_arr[0, t, ..., ch + comp] = comp_field
                ch += 3

        device = self._torch_device
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
        field_indices = self._dpf_field_indices.clone()  # type: ignore[attr-defined]

        metadata = WellMetadata(
            dataset_name="dpf_plasma",
            n_spatial_dims=3,
            spatial_resolution=spatial,
            scalar_names=list(WALRUS_SCALAR_KEYS),
            constant_scalar_names=[],
            field_names={
                0: ["density", "temperature", "temperature", "pressure", "A"],
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

    @staticmethod
    def _well_output_to_state(
        pred_array: np.ndarray, reference_state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert WALRUS output array (Well channels-last) to DPF state dict.

        Parameters
        ----------
        pred_array : np.ndarray
            Shape ``(H, W, D, C)`` — channels-last Well format output.
        reference_state : dict[str, np.ndarray]
            Reference DPF state for shape and key information.

        Returns
        -------
        dict[str, np.ndarray]
            DPF state dict with float64 arrays matching reference shapes.
        """
        orig_spatial = reference_state["rho"].shape
        state: dict[str, np.ndarray] = {}
        ch = 0

        for key in WALRUS_SCALAR_KEYS:
            if key in reference_state:
                field = pred_array[..., ch].astype(np.float64)
                if field.shape != orig_spatial:
                    field = field.reshape(orig_spatial)
                state[key] = field
            ch += 1

        for key in WALRUS_VECTOR_KEYS:
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
