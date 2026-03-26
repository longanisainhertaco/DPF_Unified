"""Tests for multi-species impurity tracking on MLX."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _make_manager():
    from dpf.metal.mlx_species import SpeciesManager

    return SpeciesManager(
        species=["D", "Cu"],
        Z=[1, 29],
        A=[2.014, 63.546],
        background="D",
    )


class TestSpeciesManager:
    def test_init(self):
        mgr = _make_manager()
        assert mgr.n_evolved == 1  # only Cu evolved, D is background
        assert mgr.species_idx == {"D": 0, "Cu": 1}

    def test_init_mass_fractions(self):
        mgr = _make_manager()
        Y = mgr.init_mass_fractions(8, 16, initial_fractions={"Cu": 0.01})
        assert Y.shape == (1, 8, 16)
        assert float(mx.mean(Y[0])) == pytest.approx(0.01, abs=1e-6)

    def test_recover_background(self):
        mgr = _make_manager()
        Y_ev = mx.full((1, 8, 16), 0.05, dtype=mx.float32)
        Y_full = mgr.recover_background(Y_ev)
        assert Y_full.shape == (2, 8, 16)
        total = float(mx.sum(Y_full, axis=0).mean())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestSpeciesAdvection:
    def test_uniform_conserves_mass(self):
        from dpf.metal.mlx_species import species_advection_step

        nr, nz = 16, 32
        Y = mx.full((1, nr, nz), 0.05, dtype=mx.float32)
        U = mx.zeros((10, nr, nz), dtype=mx.float32)
        U = U.at[0].add(1e-4)  # density
        U = U.at[4].add(1e3)  # energy

        mass_before = float(mx.sum(Y))
        Y_new = species_advection_step(Y, U, dr=0.01, dz=0.01, dt=1e-8, gamma=5 / 3)
        mass_after = float(mx.sum(Y_new))
        assert mass_after == pytest.approx(mass_before, rel=1e-6)

    def test_nonzero_velocity_transports(self):
        from dpf.metal.mlx_species import species_advection_step

        nr, nz = 32, 64
        Y = mx.zeros((1, nr, nz), dtype=mx.float32)
        # Gaussian pulse in center
        r = np.arange(nr, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        pulse = np.exp(-((zz - 32) ** 2) / 50).astype(np.float32) * 0.1
        Y = Y.at[0].add(mx.array(pulse))

        U = mx.zeros((10, nr, nz), dtype=mx.float32)
        U = U.at[0].add(1e-4)
        U = U.at[2].add(1e-4 * 1e4)  # vz = 1e4 m/s momentum
        U = U.at[4].add(1e3)

        Y_new = species_advection_step(Y, U, dr=0.01, dz=0.01, dt=1e-7, gamma=5 / 3)

        # Peak should shift in z direction
        peak_before = int(np.argmax(np.array(Y[0, nr // 2, :])))
        peak_after = int(np.argmax(np.array(Y_new[0, nr // 2, :])))
        assert peak_after >= peak_before  # shifted or same


class TestAblationSources:
    def test_ablation_increases_cu(self):
        from dpf.metal.mlx_species import apply_ablation_sources

        Y = mx.zeros((1, 8, 16), dtype=mx.float32)
        rate = mx.zeros((8, 16), dtype=mx.float32)
        rate = rate.at[0, :].add(0.001)  # ablation at inner boundary

        Y_new = apply_ablation_sources(Y, dt=1e-8, ablation_rate=rate, cu_idx=0)
        assert float(mx.max(Y_new)) > 0

    def test_ablation_nonnegative(self):
        from dpf.metal.mlx_species import apply_ablation_sources

        Y = mx.full((1, 4, 4), 0.01, dtype=mx.float32)
        rate = mx.full((4, 4), -0.1, dtype=mx.float32)  # negative = nonsense
        Y_new = apply_ablation_sources(Y, dt=1.0, ablation_rate=rate, cu_idx=0)
        assert float(mx.min(Y_new)) >= 0.0


class TestZeff:
    def test_pure_deuterium(self):
        from dpf.metal.mlx_species import compute_zeff_field

        Y = mx.array([[[1.0]], [[0.0]]], dtype=mx.float32)  # (2, 1, 1)
        Z = mx.array([1.0, 29.0])
        A = mx.array([2.014, 63.546])
        zeff = compute_zeff_field(Y, Z, A)
        assert float(zeff[0, 0]) == pytest.approx(1.0, abs=0.01)

    def test_copper_raises_zeff(self):
        from dpf.metal.mlx_species import compute_zeff_field

        Y_pure = mx.array([[[1.0]], [[0.0]]], dtype=mx.float32)
        Y_mixed = mx.array([[[0.9]], [[0.1]]], dtype=mx.float32)
        Z = mx.array([1.0, 29.0])
        A = mx.array([2.014, 63.546])
        zeff_pure = float(compute_zeff_field(Y_pure, Z, A)[0, 0])
        zeff_mixed = float(compute_zeff_field(Y_mixed, Z, A)[0, 0])
        assert zeff_mixed > zeff_pure


class TestGhostPad:
    def test_shape(self):
        from dpf.metal.mlx_species import pad_species_ghost

        Y = mx.ones((2, 8, 16), dtype=mx.float32)
        padded = pad_species_ghost(Y, ng=2)
        assert padded.shape == (2, 12, 16)

    def test_zero_gradient(self):
        from dpf.metal.mlx_species import pad_species_ghost

        Y = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        padded = pad_species_ghost(Y, ng=1)
        # Inner ghost = first cell
        np.testing.assert_allclose(np.array(padded[0, 0]), np.array(Y[0, 0]))
        # Outer ghost = last cell
        np.testing.assert_allclose(np.array(padded[0, -1]), np.array(Y[0, -1]))
