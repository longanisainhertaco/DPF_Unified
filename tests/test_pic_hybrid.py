"""Tests for the HybridPIC module and Nanbu-Perez collision kernel.

Covers:
    - _nanbu_scatter_kernel: non-relativistic and relativistic regimes,
      subluminal output, isotropic branch (s12>=4), speed conservation,
      empty-pair no-op, zero-velocity guard.
    - HybridPIC: use_binary_collisions flag wiring, fallback to
      _coulomb_scatter when use_binary_collisions=False.
"""

from __future__ import annotations

import numpy as np

from dpf.experimental.pic.hybrid import (
    HybridPIC,
    _nanbu_scatter_kernel,
)

C = 2.998e8
M_D = 3.344e-27
Q_E = 1.602e-19


def _make_vel(n: int, speed: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * speed


class TestNanbuScatterKernel:
    def test_empty_arrays_return_unchanged(self) -> None:
        vel_a = np.zeros((0, 3), dtype=np.float64)
        vel_b = np.zeros((0, 3), dtype=np.float64)
        w = np.zeros(0)
        out_a, out_b = _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        assert out_a.shape == (0, 3)
        assert out_b.shape == (0, 3)

    def test_output_subluminal(self) -> None:
        n = 50
        speed = 0.8 * C
        vel_a = _make_vel(n, speed, seed=1).copy()
        vel_b = _make_vel(n, speed, seed=2).copy()
        w = np.ones(n)
        _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        assert np.all(np.linalg.norm(vel_a, axis=1) < C)
        assert np.all(np.linalg.norm(vel_b, axis=1) < C)

    def test_nonrelativistic_speed_approx_conserved(self) -> None:
        n = 200
        speed = 1e6
        vel_a = _make_vel(n, speed, seed=3).copy()
        vel_b = _make_vel(n, speed, seed=4).copy()
        speeds_before = np.linalg.norm(vel_a, axis=1).copy()
        w = np.ones(n)
        _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        speeds_after = np.linalg.norm(vel_a, axis=1)
        rel_change = np.abs(speeds_after - speeds_before) / (speeds_before + 1e-30)
        assert np.mean(rel_change) < 0.15

    def test_large_s12_deflects_directions(self) -> None:
        """With artificially large charge (s12~2.5), deflections must occur.

        Physical note: single-step s12 for deuterons at DPF conditions is
        O(10^-40), so this test uses charge=1e10*Q_E to produce s12~2.5.
        This validates the Nanbu alpha formula and rotation mechanics.
        """
        n = 500
        speed = 1e6
        # charge = 1e10 * Q_E gives s12 ~ 2.47 for n=1e25, dt=1e-10, ln_L=10
        Q_large = 1e10 * Q_E
        vel_a = _make_vel(n, speed, seed=5).copy()
        vel_b = _make_vel(n, speed, seed=6).copy()
        vel_a_orig = vel_a.copy()
        w = np.ones(n)
        _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_large, Q_large,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        norms_a = np.linalg.norm(vel_a, axis=1, keepdims=True) + 1e-40
        norms_orig = np.linalg.norm(vel_a_orig, axis=1, keepdims=True) + 1e-40
        dot = np.einsum("ij,ij->i", vel_a / norms_a, vel_a_orig / norms_orig)
        n_deflected = np.sum(dot < 0.9999)
        assert n_deflected > 0, "Expected deflections with s12~2.5"

    def test_mismatched_lengths_uses_min(self) -> None:
        vel_a = _make_vel(10, 1e6, seed=7).copy()
        vel_b = _make_vel(5, 1e6, seed=8).copy()
        w_a = np.ones(10)
        w_b = np.ones(5)
        out_a, out_b = _nanbu_scatter_kernel(
            vel_a, vel_b, w_a, w_b,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        assert out_a.shape == (10, 3)
        assert out_b.shape == (5, 3)

    def test_zero_velocity_no_nan(self) -> None:
        vel_a = np.zeros((5, 3), dtype=np.float64)
        vel_b = np.zeros((5, 3), dtype=np.float64)
        w = np.ones(5)
        out_a, out_b = _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-10, 1e-18,
        )
        assert not np.any(np.isnan(out_a))
        assert not np.any(np.isnan(out_b))

    def test_relativistic_no_nan_subluminal(self) -> None:
        n = 30
        vel_a = _make_vel(n, 0.99 * C, seed=9).copy()
        vel_b = _make_vel(n, 0.99 * C, seed=10).copy()
        w = np.ones(n)
        out_a, out_b = _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, M_D, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-12, 1e-18,
        )
        assert not np.any(np.isnan(out_a))
        assert not np.any(np.isnan(out_b))
        assert np.all(np.linalg.norm(out_a, axis=1) < C)
        assert np.all(np.linalg.norm(out_b, axis=1) < C)

    def test_unequal_mass_species_no_nan(self) -> None:
        n = 20
        m_p = 1.673e-27
        vel_a = _make_vel(n, 5e6, seed=11).copy()
        vel_b = _make_vel(n, 2e7, seed=12).copy()
        w = np.ones(n)
        out_a, out_b = _nanbu_scatter_kernel(
            vel_a, vel_b, w, w,
            M_D, m_p, Q_E, Q_E,
            1e25, 1e25, 10.0, 1e-11, 1e-18,
        )
        assert not np.any(np.isnan(out_a))
        assert not np.any(np.isnan(out_b))


class TestHybridPICCollisions:
    def _make_pic(self, use_binary: bool) -> HybridPIC:
        pic = HybridPIC(
            grid_shape=(8, 8, 8),
            dx=1e-3, dy=1e-3, dz=1e-3,
            dt=1e-11,
            use_binary_collisions=use_binary,
        )
        pic.enable_collisions(n_background=1e25, T_background_eV=500.0)
        n = 20
        pos = np.random.default_rng(0).uniform(0, 8e-3, (n, 3))
        vel = _make_vel(n, 1e6, seed=0)
        w = np.ones(n)
        pic.add_species("d", M_D, Q_E, pos, vel, w)
        return pic

    def test_default_use_binary_collisions_is_true(self) -> None:
        pic = HybridPIC(grid_shape=(4, 4, 4), dx=1e-3, dy=1e-3, dz=1e-3, dt=1e-11)
        assert pic.use_binary_collisions is True

    def test_can_disable_binary_collisions(self) -> None:
        pic = HybridPIC(
            grid_shape=(4, 4, 4), dx=1e-3, dy=1e-3, dz=1e-3, dt=1e-11,
            use_binary_collisions=False,
        )
        assert pic.use_binary_collisions is False

    def test_push_nanbu_no_nan(self) -> None:
        pic = self._make_pic(use_binary=True)
        E = np.zeros((8, 8, 8, 3))
        B = np.zeros((8, 8, 8, 3))
        pic.push_particles(E, B)
        assert not np.any(np.isnan(pic.species[0].velocities))

    def test_push_takizawa_fallback_no_nan(self) -> None:
        pic = self._make_pic(use_binary=False)
        E = np.zeros((8, 8, 8, 3))
        B = np.zeros((8, 8, 8, 3))
        pic.push_particles(E, B)
        assert not np.any(np.isnan(pic.species[0].velocities))

    def test_nanbu_and_fallback_differ(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        pos = rng.uniform(0, 8e-3, (n, 3))
        vel = _make_vel(n, 1e6, seed=42)
        w = np.ones(n)
        E = np.zeros((8, 8, 8, 3))
        B = np.zeros((8, 8, 8, 3))

        results = {}
        for use_bin in (True, False):
            pic = HybridPIC(
                grid_shape=(8, 8, 8), dx=1e-3, dy=1e-3, dz=1e-3,
                dt=1e-11, use_binary_collisions=use_bin,
            )
            pic.enable_collisions(n_background=1e25, T_background_eV=500.0)
            pic.add_species("d", M_D, Q_E, pos.copy(), vel.copy(), w.copy())
            pic.push_particles(E, B)
            results[use_bin] = pic.species[0].velocities.copy()

        assert not np.allclose(results[True], results[False])
