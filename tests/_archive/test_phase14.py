"""Tests for Phase 14: Implicit Diffusion + Super Time-Stepping.

Covers:
    14.1 Semi-implicit resistive diffusion (Crank-Nicolson ADI)
    14.2 Super time-stepping (RKL2)
    14.3 FluidConfig diffusion options
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import FluidConfig
from dpf.constants import mu_0
from dpf.fluid.implicit_diffusion import (
    _thomas_solve,
    diffuse_field_1d,
    diffusion_cfl_dt,
    implicit_resistive_diffusion,
    implicit_thermal_diffusion,
)
from dpf.fluid.super_time_step import (
    _diffusion_operator_1d,
    _diffusion_operator_1d_scalar,
    rkl2_coefficients,
    rkl2_diffusion_3d,
    rkl2_diffusion_step,
    rkl2_stability_limit,
    rkl2_thermal_step,
)

# ===================================================================
# 14.1 — Semi-implicit diffusion (Crank-Nicolson)
# ===================================================================


class TestThomasSolve:
    """Tests for the tridiagonal Thomas algorithm."""

    def test_identity_system(self):
        """I * x = b gives x = b."""
        n = 10
        lower = np.zeros(n)
        diag = np.ones(n)
        upper = np.zeros(n)
        rhs = np.arange(n, dtype=np.float64)
        x = _thomas_solve(lower, diag, upper, rhs)
        np.testing.assert_allclose(x, rhs, atol=1e-14)

    def test_simple_tridiag(self):
        """Solve a known 3x3 tridiagonal system."""
        lower = np.array([0.0, -1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0, 0.0])
        rhs = np.array([1.0, 0.0, 1.0])
        x = _thomas_solve(lower, diag, upper, rhs)
        # Manual verification: this is -Laplacian with Dirichlet-like
        assert np.all(np.isfinite(x))
        # Verify A*x = rhs
        Ax = np.array([
            diag[0] * x[0] + upper[0] * x[1],
            lower[1] * x[0] + diag[1] * x[1] + upper[1] * x[2],
            lower[2] * x[1] + diag[2] * x[2],
        ])
        np.testing.assert_allclose(Ax, rhs, atol=1e-12)


class TestDiffuseField1D:
    """Tests for Crank-Nicolson 1D diffusion."""

    def test_uniform_field_unchanged(self):
        """Constant field is unchanged by diffusion."""
        n = 50
        field = np.full(n, 3.0)
        coeff = np.full(n, 1.0)
        result = diffuse_field_1d(field, coeff, dt=0.1, dx=0.01)
        np.testing.assert_allclose(result, 3.0, atol=1e-12)

    def test_gaussian_smoothing(self):
        """Gaussian peak is smoothed by diffusion."""
        n = 100
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        coeff = np.full(n, 0.01)
        result = diffuse_field_1d(field, coeff, dt=0.001, dx=1.0 / n)
        # Peak should decrease
        assert result[n // 2] < field[n // 2]
        # Total integral should be approximately conserved (Neumann BCs)
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-8)

    def test_conservation(self):
        """Total integral is conserved under Neumann BCs."""
        n = 64
        field = np.random.default_rng(42).standard_normal(n)
        field -= field.mean()  # zero mean
        field += 5.0  # shift to positive
        coeff = np.full(n, 0.5)
        result = diffuse_field_1d(field, coeff, dt=0.01, dx=0.1)
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-10)

    def test_short_array_passthrough(self):
        """Arrays shorter than 3 are returned unchanged."""
        field = np.array([1.0, 2.0])
        result = diffuse_field_1d(field, np.ones(2), dt=0.1, dx=0.1)
        np.testing.assert_allclose(result, field)


class TestImplicitResistiveDiffusion:
    """Tests for 3D ADI resistive diffusion."""

    def test_uniform_B_unchanged(self):
        """Uniform B-field is unchanged by diffusion."""
        shape = (8, 8, 8)
        Bx = np.full(shape, 1.0)
        By = np.full(shape, 2.0)
        Bz = np.full(shape, 3.0)
        eta = np.full(shape, 1e-3)
        Bx_n, By_n, Bz_n = implicit_resistive_diffusion(
            Bx, By, Bz, eta, dt=1e-6, dx=0.01, dy=0.01, dz=0.01,
        )
        np.testing.assert_allclose(Bx_n, 1.0, atol=1e-10)
        np.testing.assert_allclose(By_n, 2.0, atol=1e-10)
        np.testing.assert_allclose(Bz_n, 3.0, atol=1e-10)

    def test_diffusion_smooths_field(self):
        """Non-uniform B-field is smoothed by diffusion."""
        shape = (16, 8, 8)
        Bx = np.zeros(shape)
        Bx[7:9, :, :] = 1.0  # spike in the middle
        By = np.zeros(shape)
        Bz = np.zeros(shape)
        eta = np.full(shape, 1e-2)
        Bx_n, _, _ = implicit_resistive_diffusion(
            Bx, By, Bz, eta, dt=1e-4, dx=0.01, dy=0.01, dz=0.01,
        )
        # Peak should decrease
        assert Bx_n[8, 4, 4] < Bx[8, 4, 4]


class TestImplicitThermalDiffusion:
    """Tests for 3D ADI thermal diffusion."""

    def test_uniform_Te_unchanged(self):
        """Uniform temperature is unchanged."""
        shape = (8, 8, 8)
        Te = np.full(shape, 1e6)
        kappa = np.full(shape, 100.0)
        ne = np.full(shape, 1e23)
        Te_new = implicit_thermal_diffusion(
            Te, kappa, ne, dt=1e-9, dx=0.001, dy=0.001, dz=0.001,
        )
        np.testing.assert_allclose(Te_new, 1e6, atol=1.0)

    def test_hot_spot_cools(self):
        """Hot spot temperature decreases by conduction."""
        shape = (16, 8, 8)
        Te = np.full(shape, 1e5)
        Te[7:9, 3:5, 3:5] = 1e7  # hot spot
        kappa = np.full(shape, 1000.0)
        ne = np.full(shape, 1e23)
        Te_new = implicit_thermal_diffusion(
            Te, kappa, ne, dt=1e-7, dx=0.001, dy=0.001, dz=0.001,
        )
        assert Te_new[8, 4, 4] < Te[8, 4, 4]


class TestDiffusionCFL:
    """Tests for explicit diffusion CFL estimate."""

    def test_cfl_positive(self):
        """CFL timestep is positive."""
        dt = diffusion_cfl_dt(1e-3, 100.0, 1e22, 0.001)
        assert dt > 0.0

    def test_cfl_smaller_for_larger_eta(self):
        """Larger eta gives smaller resistive CFL dt."""
        dt1 = diffusion_cfl_dt(1e-3, 0.0, 0.0, 0.001)
        dt2 = diffusion_cfl_dt(1e-1, 0.0, 0.0, 0.001)
        assert dt2 < dt1

    def test_cfl_formula_resistive(self):
        """Resistive CFL matches analytical formula."""
        eta = 1e-3
        dx = 0.001
        dt = diffusion_cfl_dt(eta, 0.0, 0.0, dx)
        expected = dx * dx * mu_0 / (2 * eta)
        np.testing.assert_allclose(dt, expected, rtol=1e-12)


# ===================================================================
# 14.2 — Super time-stepping (RKL2)
# ===================================================================


class TestRKL2Coefficients:
    """Tests for RKL2 coefficient computation."""

    def test_s2_valid(self):
        """s=2 produces valid (finite) coefficients."""
        mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(2)
        assert len(mu) == 3
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(nu))
        assert np.all(np.isfinite(mu_tilde))
        assert np.all(np.isfinite(gamma_tilde))

    def test_s_less_than_2_raises(self):
        """s < 2 raises ValueError."""
        with pytest.raises(ValueError, match="s >= 2"):
            rkl2_coefficients(1)

    def test_s10_valid(self):
        """s=10 produces valid coefficients."""
        mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(10)
        assert len(mu) == 11
        assert mu_tilde[1] > 0  # first stage should be positive

    def test_mu_tilde_1_positive(self):
        """First-stage coefficient mu_tilde[1] is positive for all s."""
        for s in range(2, 17):
            _, _, mt, _ = rkl2_coefficients(s)
            assert mt[1] > 0, f"mu_tilde[1] <= 0 for s={s}"


class TestDiffusionOperator:
    """Tests for the 1D diffusion operator L(u) = d/dx(D du/dx)."""

    def test_constant_field_zero(self):
        """Laplacian of a constant is zero."""
        u = np.full(20, 5.0)
        D = np.full(20, 1.0)
        Lu = _diffusion_operator_1d(u, D, dx=0.1)
        np.testing.assert_allclose(Lu, 0.0, atol=1e-12)

    def test_quadratic_exact(self):
        """Laplacian of x^2 is 2*D (constant D)."""
        n = 50
        dx = 0.01
        x = np.arange(n) * dx
        u = x * x
        D = 3.0
        Lu = _diffusion_operator_1d_scalar(u, D, dx)
        # Interior points should be close to 2*D = 6.0
        np.testing.assert_allclose(Lu[5:-5], 2 * D, rtol=1e-6)


class TestRKL2DiffusionStep:
    """Tests for RKL2 1D diffusion step."""

    def test_uniform_unchanged(self):
        """Constant field unchanged by RKL2 diffusion."""
        field = np.full(32, 2.0)
        result = rkl2_diffusion_step(field, 0.1, dt_super=1e-4, dx=0.01, s_stages=4)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_gaussian_smoothing(self):
        """RKL2 smooths a Gaussian pulse."""
        n = 128
        x = np.linspace(0, 1, n)
        field = np.exp(-200 * (x - 0.5) ** 2)
        result = rkl2_diffusion_step(field, 0.01, dt_super=1e-4, dx=1.0 / n, s_stages=8)
        assert result[n // 2] < field[n // 2]

    def test_sts_larger_dt_than_explicit(self):
        """STS allows larger timestep than explicit CFL without blowup."""
        n = 64
        dx = 0.01
        D = 1.0
        dt_explicit = dx * dx / (2 * D)
        # RKL2 with s=10 should handle 10x explicit CFL
        dt_super = 10 * dt_explicit
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        result = rkl2_diffusion_step(field, D, dt_super=dt_super, dx=dx, s_stages=10)
        assert np.all(np.isfinite(result))
        assert result[n // 2] < field[n // 2]

    def test_conservation(self):
        """RKL2 conserves total integral under Neumann BCs."""
        n = 64
        field = np.random.default_rng(123).standard_normal(n)
        field -= field.mean()
        field += 10.0
        result = rkl2_diffusion_step(field, 0.5, dt_super=1e-5, dx=0.01, s_stages=6)
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-6)

    def test_spatially_varying_D(self):
        """RKL2 works with spatially-varying diffusion coefficient."""
        n = 64
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        D = 0.01 + 0.05 * x  # varying from 0.01 to 0.06
        result = rkl2_diffusion_step(field, D, dt_super=1e-5, dx=1.0 / n, s_stages=6)
        assert np.all(np.isfinite(result))


class TestRkl2ThreeD:
    """Tests for 3D RKL2 resistive diffusion."""

    def test_uniform_B_unchanged(self):
        """Uniform B-field unchanged."""
        shape = (8, 8, 8)
        Bx = np.full(shape, 1.0)
        By = np.full(shape, 2.0)
        Bz = np.full(shape, 3.0)
        eta = 1e-3
        Bx_n, By_n, Bz_n = rkl2_diffusion_3d(
            Bx, By, Bz, eta, dt=1e-6, dx=0.01, dy=0.01, dz=0.01, s_stages=4,
        )
        np.testing.assert_allclose(Bx_n, 1.0, atol=1e-8)
        np.testing.assert_allclose(By_n, 2.0, atol=1e-8)
        np.testing.assert_allclose(Bz_n, 3.0, atol=1e-8)


class TestRKL2Thermal:
    """Tests for RKL2 thermal diffusion."""

    def test_1d_uniform_unchanged(self):
        """Uniform 1D temperature unchanged."""
        n = 32
        Te = np.full(n, 1e6)
        ne = np.full(n, 1e23)
        Te_new = rkl2_thermal_step(Te, kappa=100.0, ne=ne, dt=1e-9, dx=0.001, s_stages=4)
        np.testing.assert_allclose(Te_new, 1e6, atol=1.0)

    def test_1d_hot_spot_smooths(self):
        """Hot spot in 1D smooths out."""
        n = 64
        Te = np.full(n, 1e5)
        Te[30:34] = 1e7
        ne = np.full(n, 1e23)
        Te_new = rkl2_thermal_step(Te, kappa=1000.0, ne=ne, dt=1e-8, dx=0.001, s_stages=8)
        assert Te_new[32] < Te[32]


class TestStabilityLimit:
    """Tests for RKL2 stability limit."""

    def test_stability_s10(self):
        """s=10 gives ~25x acceleration."""
        dt_exp = 1e-6
        dt_rkl2 = rkl2_stability_limit(10, dt_exp)
        assert dt_rkl2 == pytest.approx(0.25 * 100 * dt_exp)

    def test_stability_increases_with_stages(self):
        """More stages give larger stable timestep."""
        dt_exp = 1e-6
        dt4 = rkl2_stability_limit(4, dt_exp)
        dt8 = rkl2_stability_limit(8, dt_exp)
        dt16 = rkl2_stability_limit(16, dt_exp)
        assert dt16 > dt8 > dt4


# ===================================================================
# 14.3 — FluidConfig diffusion options
# ===================================================================


class TestFluidConfigDiffusion:
    """Tests for FluidConfig diffusion settings."""

    def test_default_explicit(self):
        """Default diffusion method is 'explicit'."""
        cfg = FluidConfig()
        assert cfg.diffusion_method == "explicit"

    def test_sts_config(self):
        """STS config is accepted."""
        cfg = FluidConfig(diffusion_method="sts", sts_stages=12)
        assert cfg.diffusion_method == "sts"
        assert cfg.sts_stages == 12

    def test_implicit_config(self):
        """Implicit config is accepted."""
        cfg = FluidConfig(diffusion_method="implicit", implicit_tol=1e-10)
        assert cfg.diffusion_method == "implicit"
        assert cfg.implicit_tol == 1e-10

    def test_default_sts_stages(self):
        """Default STS stages is 8."""
        cfg = FluidConfig()
        assert cfg.sts_stages == 8
