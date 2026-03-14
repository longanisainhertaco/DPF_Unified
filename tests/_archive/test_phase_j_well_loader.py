
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dpf.ai.train_surrogate import collate_well_to_dpf  # noqa: E402
from dpf.ai.well_loader import WellDataset  # noqa: E402


class TestWellLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.h5_path = Path(self.temp_dir) / "mock_well.h5"

        # Create mock file
        with h5py.File(self.h5_path, 'w') as f:
            # Create a simple dataset: 2 traj, 20 steps, 16x16x16
            n_traj = 2
            n_steps = 20
            spatial = (16, 16, 16)

            # Density
            f.create_dataset("density", data=np.random.rand(n_traj, n_steps, *spatial).astype(np.float32))

            # Velocity
            # Well format: (N, T, X, Y, Z, 3) usually?
            # DPF Mapping expects (N, T, X, Y, Z, 3) based on loader logic "tensor.ndim == 5".
            # well_loader line 134: `tensor = tensor.permute(0, 4, 1, 2, 3)` suggests it reads `(T, X, Y, Z, 3)`.
            # Let's verify loader logic:
            # Loader reads: `raw_chunk = dset[traj_idx, t_slice]` -> (T, X, Y, Z, [3])
            # If 3 dims at end, it permutes.
            f.create_dataset("velocity", data=np.random.rand(n_traj, n_steps, *spatial, 3).astype(np.float32))

            # Magnetic Field
            f.create_dataset("magnetic_field", data=np.random.rand(n_traj, n_steps, *spatial, 3).astype(np.float32))

            f.attrs["dt"] = 0.01

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_well_loader_shapes(self):
        ds = WellDataset(
            hdf5_paths=[str(self.h5_path)],
            fields=["density", "velocity", "magnetic_field"],
            sequence_length=5,
            stride=1,
            normalize=False
        )

        self.assertTrue(len(ds) > 0)

        sample = ds[0]
        # Check keys mapped — WellDataset applies WELL_TO_DPF_NAMES:
        # "density" -> "rho", "velocity" -> "velocity", "magnetic_field" -> "B"
        self.assertIn("rho", sample)
        self.assertIn("velocity", sample) # well_loader keeps "velocity"
        self.assertIn("B", sample) # "magnetic_field" -> "B"

        # Check shapes
        # Density: (T, 1, X, Y, Z)
        rho = sample["rho"]
        self.assertEqual(rho.shape, (5, 1, 16, 16, 16))

        # B: (T, 3, X, Y, Z) (Permuted by loader)
        B = sample["B"]
        self.assertEqual(B.shape, (5, 3, 16, 16, 16))

    def test_collate_logic(self):
        ds = WellDataset(
            hdf5_paths=[str(self.h5_path)],
            fields=["density", "velocity", "magnetic_field"],
            sequence_length=5,
            normalize=False
        )

        batch = [ds[0], ds[1]]

        # Run collation
        tensor, mask = collate_well_to_dpf(batch)

        # Tensor Shape: (Batch, Time, 11, X, Y, Z)
        self.assertEqual(tensor.shape, (2, 5, 11, 16, 16, 16))

        # Mask Shape: (11,)
        # We have Density (idx 0), B (idx 5,6,7), V (idx 8,9,10)
        # Pressure (3), Te(1), Ti(2), Psi(4) should be 0

        expected_mask = torch.tensor([1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
        # Pressure is missing in mock data, so mask[3] should be 0.

        self.assertTrue(torch.allclose(mask, expected_mask), f"Got mask {mask}")

        # Check values
        # Index 0 (Density/rho) should match sample['rho']
        # tensor[0, :, 0] vs sample['rho'] (T, 1, X, Y, Z) -> squeeze -> (T, X, Y, Z)
        self.assertTrue(torch.allclose(tensor[0, :, 0], batch[0]["rho"].squeeze(1)))

if __name__ == "__main__":
    unittest.main()
