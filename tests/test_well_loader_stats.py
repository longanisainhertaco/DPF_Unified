"""Tests for WellDataset.compute_stats() — Bug C3 fix.

Verifies that compute_stats() never returns empty dict, computes
correct mean/std/rms statistics, handles edge cases (NaN, empty data,
single sample), and auto-triggers from __init__ when normalize=True.
"""

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch  # noqa: F401 — needed by WellDataset

from dpf.ai.well_loader import WellDataset


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _create_mock_h5(
    path: str,
    n_traj: int = 1,
    n_steps: int = 20,
    spatial: tuple[int, ...] = (8, 8, 8),
    density_fill: float | None = None,
    add_attrs: bool = False,
    inject_nan: bool = False,
) -> str:
    """Create a mock Well HDF5 file with scalar/vector fields."""
    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(42)
        shape_scalar = (n_traj, n_steps, *spatial)
        shape_vector = (n_traj, n_steps, *spatial, 3)

        if density_fill is not None:
            density_data = np.full(shape_scalar, density_fill, dtype=np.float32)
        else:
            density_data = rng.standard_normal(shape_scalar).astype(np.float32) + 5.0

        if inject_nan:
            # Put NaN in ~10% of values
            nan_mask = rng.random(shape_scalar) < 0.1
            density_data[nan_mask] = np.nan

        ds = f.create_dataset("density", data=density_data)
        if add_attrs:
            ds.attrs["mean"] = 5.0
            ds.attrs["std"] = 1.0
            ds.attrs["rms"] = 5.1

        vel_data = rng.standard_normal(shape_vector).astype(np.float32)
        vs = f.create_dataset("velocity", data=vel_data)
        if add_attrs:
            vs.attrs["mean"] = 0.0
            vs.attrs["std"] = 1.0

    return path


class TestComputeStatsBasic:
    """Test compute_stats() produces valid non-empty stats."""

    def test_stats_not_empty(self, tmp_dir):
        """C3 core fix: compute_stats() must never return empty dict."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert len(stats) > 0, "compute_stats() returned empty dict (C3 bug)"
        assert "density" in stats
        assert "velocity" in stats

    def test_stats_has_mean_std_rms(self, tmp_dir):
        """Each field entry must have mean, std, and rms keys."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        for field_name in ["density", "velocity"]:
            assert "mean" in stats[field_name]
            assert "std" in stats[field_name]
            assert "rms" in stats[field_name]

    def test_stats_values_correct_for_constant(self, tmp_dir):
        """Constant field should have std near 0 and rms equal to |mean|."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "const.h5"), density_fill=3.0
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        assert abs(s["mean"] - 3.0) < 0.01, f"Expected mean ~3.0, got {s['mean']}"
        assert s["std"] < 0.01, f"Expected std ~0 for constant, got {s['std']}"
        assert abs(s["rms"] - 3.0) < 0.01, f"Expected rms ~3.0, got {s['rms']}"

    def test_rms_formula(self, tmp_dir):
        """RMS should satisfy: rms^2 = mean^2 + std^2 (approximately)."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "test.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        rms_expected = np.sqrt(s["mean"] ** 2 + s["std"] ** 2)
        assert abs(s["rms"] - rms_expected) < 0.01, (
            f"RMS identity violation: rms={s['rms']}, "
            f"sqrt(mean^2+std^2)={rms_expected}"
        )


class TestComputeStatsEdgeCases:
    """Edge cases that previously caused C3 (empty dict return)."""

    def test_empty_dataset_returns_defaults(self, tmp_dir):
        """No valid files → stats should have safe defaults, not {}."""
        ds = WellDataset.__new__(WellDataset)
        ds.paths = []
        ds.fields = ["density", "velocity"]
        ds.sequence_length = 5
        ds.stride = 1
        ds.normalize = True
        ds.stats = {}
        ds.samples = []

        stats = ds.compute_stats()
        assert len(stats) == 2, f"Expected 2 field entries, got {len(stats)}"
        for field in ["density", "velocity"]:
            assert stats[field]["mean"] == 0.0
            assert stats[field]["std"] == 1.0
            assert stats[field]["rms"] == 1.0

    def test_nan_values_filtered(self, tmp_dir):
        """Fields with NaN values should still produce valid stats."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "nan.h5"), inject_nan=True
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        s = stats["density"]
        assert np.isfinite(s["mean"]), f"mean is not finite: {s['mean']}"
        assert np.isfinite(s["std"]), f"std is not finite: {s['std']}"
        assert np.isfinite(s["rms"]), f"rms is not finite: {s['rms']}"

    def test_single_sample_dataset(self, tmp_dir):
        """Dataset with exactly 1 sample should still compute stats."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "single.h5"),
            n_traj=1,
            n_steps=6,
            spatial=(4, 4, 4),
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        assert len(ds) >= 1
        stats = ds.compute_stats(max_samples=1)
        assert "density" in stats
        assert stats["density"]["rms"] > 0

    def test_std_floor_prevents_zero(self, tmp_dir):
        """std should never be exactly 0 (floor at 1e-10)."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "const.h5"), density_fill=7.0
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert stats["density"]["std"] >= 1e-10


class TestComputeStatsHDF5Attrs:
    """Test Strategy 1: reading stats from HDF5 attributes."""

    def test_reads_attrs_when_present(self, tmp_dir):
        """Stats should be loaded from HDF5 attrs without sampling data."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "attrs.h5"), add_attrs=True
        )
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density", "velocity"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        # density had attrs: mean=5.0, std=1.0, rms=5.1
        assert abs(stats["density"]["mean"] - 5.0) < 0.01
        assert abs(stats["density"]["std"] - 1.0) < 0.01
        assert abs(stats["density"]["rms"] - 5.1) < 0.01

    def test_attrs_rms_derived_when_missing(self, tmp_dir):
        """When HDF5 attrs have mean/std but no rms, derive rms from them."""
        h5_path = str(Path(tmp_dir) / "partial_attrs.h5")
        with h5py.File(h5_path, "w") as f:
            data = np.ones((1, 20, 4, 4, 4), dtype=np.float32) * 3.0
            ds = f.create_dataset("density", data=data)
            ds.attrs["mean"] = 3.0
            ds.attrs["std"] = 4.0
            # No rms attr — should be derived as sqrt(3^2 + 4^2) = 5.0
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        stats = ds.compute_stats()
        assert abs(stats["density"]["rms"] - 5.0) < 0.01


class TestAutoComputeStats:
    """Test that __init__ auto-calls compute_stats() when needed."""

    def test_auto_compute_on_init_with_normalize(self, tmp_dir):
        """When normalize=True and no stats provided, stats populated in __init__."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "auto.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
        )
        assert len(ds.stats) > 0, "Stats not auto-computed in __init__"
        assert "density" in ds.stats

    def test_no_auto_compute_when_stats_provided(self, tmp_dir):
        """External stats should be used as-is without re-computation."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "ext.h5"))
        external_stats = {"density": {"mean": 99.0, "std": 1.0, "rms": 99.0}}
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
            normalization_stats=external_stats,
        )
        assert ds.stats["density"]["mean"] == 99.0, "External stats overwritten"

    def test_no_auto_compute_when_normalize_false(self, tmp_dir):
        """When normalize=False, stats should remain empty."""
        h5_path = _create_mock_h5(str(Path(tmp_dir) / "noauto.h5"))
        ds = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        assert len(ds.stats) == 0, "Stats computed even with normalize=False"


class TestFindFieldDataset:
    """Test the _find_field_dataset static helper."""

    def test_finds_root_level(self, tmp_dir):
        h5_path = str(Path(tmp_dir) / "root.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("density", data=np.zeros((1, 10, 4, 4, 4)))
            dset = WellDataset._find_field_dataset(f, "density")
            assert dset is not None

    def test_finds_t0_fields(self, tmp_dir):
        h5_path = str(Path(tmp_dir) / "t0.h5")
        with h5py.File(h5_path, "w") as f:
            g = f.create_group("t0_fields")
            g.create_dataset("pressure", data=np.zeros((1, 10, 4, 4, 4)))
            dset = WellDataset._find_field_dataset(f, "pressure")
            assert dset is not None

    def test_finds_t1_fields(self, tmp_dir):
        h5_path = str(Path(tmp_dir) / "t1.h5")
        with h5py.File(h5_path, "w") as f:
            g = f.create_group("t1_fields")
            g.create_dataset("velocity", data=np.zeros((1, 10, 4, 4, 4, 3)))
            dset = WellDataset._find_field_dataset(f, "velocity")
            assert dset is not None

    def test_returns_none_for_missing(self, tmp_dir):
        h5_path = str(Path(tmp_dir) / "empty.h5")
        with h5py.File(h5_path, "w") as f:
            dset = WellDataset._find_field_dataset(f, "nonexistent")
            assert dset is None


class TestStatsUsedInNormalization:
    """Verify that computed stats actually affect __getitem__ output."""

    def test_normalization_changes_values(self, tmp_dir):
        """Data should be different with and without normalization."""
        h5_path = _create_mock_h5(
            str(Path(tmp_dir) / "norm.h5"), density_fill=5.0
        )
        ds_raw = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=False,
        )
        ds_norm = WellDataset(
            hdf5_paths=[h5_path],
            fields=["density"],
            sequence_length=5,
            normalize=True,
        )
        raw_val = ds_raw[0]["rho"].mean().item()
        norm_val = ds_norm[0]["rho"].mean().item()
        # Constant 5.0 data, mean ~5.0, std ~0 → normalized ≈ 0
        assert abs(raw_val - 5.0) < 0.01
        assert abs(norm_val) < abs(raw_val), (
            f"Normalization should reduce magnitude: raw={raw_val}, norm={norm_val}"
        )
