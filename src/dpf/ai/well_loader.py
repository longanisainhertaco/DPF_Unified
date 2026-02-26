"""
well_loader.py

PyTorch Dataset for loading 'The Well' HDF5 files (Polymathic AI).
https://github.com/PolymathicAI/the_well
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None

from dpf.ai.field_mapping import WELL_TO_DPF_NAMES

logger = logging.getLogger(__name__)

class WellDataset(Dataset):
    """PyTorch Dataset for streaming data from 'The Well' HDF5 files.

    Loads scalar and vector fields, mapping them to DPF conventions.
    Supports normalization on-the-fly.
    """

    def __init__(
        self,
        hdf5_paths: list[str] | str,
        fields: list[str] | None = None,
        sequence_length: int = 10,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: dict[str, dict[str, float]] | None = None
    ):
        if h5py is None:
            raise ImportError("h5py is required for WellDataset")

        if fields is None:
            fields = ["density", "velocity", "magnetic_field", "pressure"]

        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]

        self.paths = [Path(p) for p in hdf5_paths]
        self.fields = fields
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.stats = normalization_stats or {}

        # Index the dataset
        self.samples = [] # List of (file_idx, traj_idx, start_step)
        self._index_files()

    def _index_files(self):
        """Scan files to build sample index."""
        logger.info(f"Indexing {len(self.paths)} Well files...")

        for file_idx, path in enumerate(self.paths):
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue

            with h5py.File(path, 'r') as f:
                # Check structure
                # Assuming Polymathic/MHD structure:
                # Groups: t0_fields, t1_fields
                # Datasets: [n_traj, n_steps, ...]

                # Find length from density
                dset = None
                if "t0_fields" in f and "density" in f["t0_fields"]:
                    dset = f["t0_fields"]["density"]
                elif "density" in f:
                    dset = f["density"]

                if dset is None:
                    continue

                n_traj = dset.shape[0]
                n_steps = dset.shape[1]

                valid_starts = n_steps - self.sequence_length * self.stride
                if valid_starts <= 0:
                    continue

                for t in range(n_traj):
                    for s in range(valid_starts):
                        self.samples.append((file_idx, t, s))

        logger.info(f"Indexed {len(self.samples)} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, traj_idx, start_step = self.samples[idx]
        path = self.paths[file_idx]

        # Open file (freshly to be thread-safe)
        with h5py.File(path, 'r') as f:
            t_slice = slice(
                start_step,
                start_step + self.sequence_length * self.stride,
                self.stride
            )

            data = {}
            for field in self.fields:
                dset = None
                # Locate dataset
                if field in f:
                    dset = f[field]
                elif "t0_fields" in f and field in f["t0_fields"]:
                    dset = f["t0_fields"][field]
                elif "t1_fields" in f and field in f["t1_fields"]:
                    dset = f["t1_fields"][field]

                if dset is None:
                    # Should handle missing fields better?
                    continue

                # Read chunk: (n_steps, ...)
                # dset is (n_traj, n_steps, x, y, z, [3])
                raw_chunk = dset[traj_idx, t_slice]

                # Convert to Torch + Permute
                # Scalars: (T, X, Y, Z) -> (T, 1, X, Y, Z) ? Or kept as (T, X, Y, Z)
                # DPFSurrogate usually expects: Batch x T x C x X x Y x Z (if 3D)
                # This loader returns single sample: T x C x ...

                tensor = torch.from_numpy(raw_chunk).float()

                # Mapping Dimensions
                # Scalar: (T, X, Y, Z) -> (T, 1, X, Y, Z)
                # Vector: (T, X, Y, Z, 3) -> (T, 3, X, Y, Z)

                if tensor.ndim == 4: # Scalar
                    tensor = tensor.unsqueeze(1)
                elif tensor.ndim == 5 and tensor.shape[-1] == 3: # Vector
                    tensor = tensor.permute(0, 4, 1, 2, 3)

                # Normalize
                if self.normalize and field in self.stats:
                    mu = self.stats[field].get("mean", 0.0)
                    sigma = self.stats[field].get("std", 1.0)
                    if sigma < 1e-9:
                        sigma = 1.0
                    tensor = (tensor - mu) / sigma

                # Map name to DPF
                dpf_name = WELL_TO_DPF_NAMES.get(field, field)
                data[dpf_name] = tensor

        return data

    def compute_stats(self, max_samples: int = 100) -> dict:
        """Compute per-field mean/std for normalization from a subset.

        First attempts to read statistics from HDF5 dataset attributes
        (``mean``, ``std``).  If attributes are absent, computes online
        statistics by sampling up to ``max_samples`` raw sequences.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping ``field_name -> {"mean": float, "std": float}``.
        """
        logger.info("Computing normalization stats...")

        # --- Strategy 1: read from HDF5 attributes ---
        stats: dict[str, dict[str, float]] = {}
        if self.paths:
            with h5py.File(self.paths[0], "r") as f:
                for field in self.fields:
                    dset = None
                    if field in f:
                        dset = f[field]
                    elif "t0_fields" in f and field in f["t0_fields"]:
                        dset = f["t0_fields"][field]
                    elif "t1_fields" in f and field in f["t1_fields"]:
                        dset = f["t1_fields"][field]
                    if dset is not None and "mean" in dset.attrs:
                        stats[field] = {
                            "mean": float(dset.attrs["mean"]),
                            "std": float(dset.attrs.get("std", 1.0)),
                        }

        if len(stats) == len(self.fields):
            logger.info("Loaded stats from HDF5 attributes for all fields.")
            self.stats = stats
            return stats

        # --- Strategy 2: online computation from sampled data ---
        if len(self) == 0:
            logger.warning("No samples available for stats computation.")
            return stats

        # Temporarily disable normalization to read raw values
        orig_normalize = self.normalize
        self.normalize = False

        n_samples = min(len(self), max_samples)
        indices = np.random.default_rng(42).choice(
            len(self), n_samples, replace=False
        )

        sums: dict[str, float] = {f: 0.0 for f in self.fields}
        sq_sums: dict[str, float] = {f: 0.0 for f in self.fields}
        counts: dict[str, int] = {f: 0 for f in self.fields}

        for idx in indices:
            sample = self[int(idx)]
            for field in self.fields:
                dpf_name = WELL_TO_DPF_NAMES.get(field, field)
                if dpf_name not in sample:
                    continue
                arr = sample[dpf_name].float()
                n = arr.numel()
                sums[field] += float(arr.sum())
                sq_sums[field] += float((arr * arr).sum())
                counts[field] += n

        self.normalize = orig_normalize

        for field in self.fields:
            if field in stats:
                continue
            n = counts[field]
            if n > 0:
                mean = sums[field] / n
                var = sq_sums[field] / n - mean * mean
                std = float(np.sqrt(max(var, 0.0)))
                stats[field] = {"mean": mean, "std": max(std, 1e-10)}
            else:
                stats[field] = {"mean": 0.0, "std": 1.0}

        logger.info(
            "Computed stats from %d samples for %d fields.",
            n_samples, len(stats),
        )
        self.stats = stats
        return stats
