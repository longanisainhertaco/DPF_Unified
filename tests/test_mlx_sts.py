"""Tests for MLX RKL2 Super Time-Stepping module."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TestRKL2Coefficients:
    """Verify RKL2 coefficient properties."""

    def test_coefficients_basic(self):
        from dpf.fluid.super_time_step import rkl2_coefficients
        mu, nu, mu_t, gamma_t = rkl2_coefficients(4)
        assert len(mu) == 5
        assert mu_t[1] > 0  # first stage forward Euler coefficient

    def test_coefficients_s2_minimal(self):
        from dpf.fluid.super_time_step import rkl2_coefficients
        mu, nu, mu_t, gamma_t = rkl2_coefficients(2)
        assert len(mu) == 3


class TestRKL2StepMLX:
    """Verify RKL2 integrator on MLX arrays."""

    def test_diffusion_decay(self):
        """Sine wave decays under diffusion: u(x,t) = sin(kx) * exp(-D*k^2*t)."""
        from dpf.metal.mlx_sts import rkl2_step_mlx

        n = 64
        dx = 1.0 / n
        D = 0.01
        k = 2.0 * np.pi

        x = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
        u0 = np.sin(k * x).astype(np.float32)
        u_mx = mx.array(u0)

        def diffusion_rhs(u):
            """1D Laplacian with periodic BC."""
            return D * (mx.roll(u, -1) + mx.roll(u, 1) - 2.0 * u) / (dx * dx)

        dt_explicit = dx * dx / (2.0 * D)
        dt_super = 10.0 * dt_explicit  # 10x explicit CFL
        s = 8  # 0.25 * 64 * dt_explicit = 16x, so 10x is safe

        u_new = rkl2_step_mlx(u_mx, diffusion_rhs, dt_super, s_stages=s)
        u_np = np.array(u_new)

        # Analytical: amplitude decays by exp(-D * k^2 * dt)
        decay = np.exp(-D * k * k * dt_super)
        u_exact = u0 * decay

        error = np.max(np.abs(u_np - u_exact))
        assert error < 0.05, f"RKL2 diffusion error {error:.4f} > 0.05"

    def test_zero_rhs_preserves_state(self):
        """Zero RHS should preserve the state exactly."""
        from dpf.metal.mlx_sts import rkl2_step_mlx

        u = mx.array([1.0, 2.0, 3.0, 4.0])
        result = rkl2_step_mlx(u, lambda x: mx.zeros_like(x), dt=1.0, s_stages=4)
        np.testing.assert_allclose(np.array(result), np.array(u), atol=1e-5)

    def test_2d_array(self):
        """RKL2 works on 2D arrays."""
        from dpf.metal.mlx_sts import rkl2_step_mlx

        u = mx.ones((8, 16), dtype=mx.float32)

        def simple_rhs(u):
            return -0.01 * u  # exponential decay

        result = rkl2_step_mlx(u, simple_rhs, dt=0.1, s_stages=4)
        assert result.shape == (8, 16)
        assert float(mx.max(mx.abs(result))) < 1.0  # decayed


class TestComputeSTSStages:
    """Verify stage count computation."""

    def test_no_subcycling_needed(self):
        from dpf.metal.mlx_sts import compute_sts_stages
        s = compute_sts_stages(dt_mhd=1e-9, dt_parabolic=1e-8)
        assert s == 2  # dt_mhd < dt_parabolic, no acceleration needed

    def test_moderate_stiffness(self):
        from dpf.metal.mlx_sts import compute_sts_stages
        # dt_mhd = 1e-9, dt_parabolic = 1e-11 => ratio = 400 => s = ceil(sqrt(400)) = 20
        s = compute_sts_stages(dt_mhd=1e-9, dt_parabolic=1e-11)
        assert s == 20

    def test_max_stages_clamped(self):
        from dpf.metal.mlx_sts import compute_sts_stages
        s = compute_sts_stages(dt_mhd=1.0, dt_parabolic=1e-15, max_stages=15)
        assert s == 15
