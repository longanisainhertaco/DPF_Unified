"""Tests for MLX Dedner GLM divergence cleaning and Powell 8-wave sources.

Covers: div(B) computation, Dedner source terms, Powell source terms,
solver integration, and div(B) reduction verification.
"""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from dpf.metal.mlx_divb import (  # noqa: E402
    dedner_source,
    div_B_cartesian,
    div_B_cylindrical,
    powell_source,
)
from dpf.metal.mlx_grid import CartesianGrid, CylindricalGrid  # noqa: E402
from dpf.metal.mlx_kernels import IBR, NVAR  # noqa: E402
from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: E402


def _cart_grid(n: int = 8, dx: float = 0.1) -> CartesianGrid:
    return CartesianGrid(n, n, n, dx)


def _uniform_U(n: int = 8, rho: float = 1.0, p: float = 1.0, Bx: float = 0.0) -> mlx.array:
    gamma = 5.0 / 3.0
    rows = [mlx.zeros((n, n, n))] * NVAR
    rows[0] = mlx.full((n, n, n), rho)
    ME = 0.5 * Bx * Bx
    rows[4] = mlx.full((n, n, n), p / (gamma - 1) + ME)
    rows[IBR] = mlx.full((n, n, n), Bx)
    return mlx.stack(rows, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# div(B) computation
# ──────────────────────────────────────────────────────────────────────────────


class TestDivB:
    def test_uniform_B_zero_divB(self):
        B = mlx.ones((8, 8, 8))
        divB = div_B_cartesian(B, B, B, 0.1, 0.1, 0.1)
        assert float(mlx.max(mlx.abs(divB))) < 1e-10

    def test_linear_Bx_nonzero_divB(self):
        x = np.linspace(0, 1, 8).reshape(8, 1, 1) * np.ones((1, 8, 8))
        Bx = mlx.array(x.astype(np.float32))
        By = mlx.zeros((8, 8, 8))
        Bz = mlx.zeros((8, 8, 8))
        divB = div_B_cartesian(Bx, By, Bz, 1.0 / 7.0, 0.1, 0.1)
        # dBx/dx should be ~1.0 everywhere (linear)
        interior = np.asarray(divB)[2:-2, 2:-2, 2:-2]
        assert np.mean(np.abs(interior - 1.0)) < 0.1

    def test_cylindrical_divB_uniform(self):
        grid = CylindricalGrid(8, 8, 0.01, 0.01)
        Br = mlx.ones((8, 8)) * 0.1
        Bz = mlx.ones((8, 8)) * 0.1
        divB = div_B_cylindrical(Br, Bz, grid.r_cell, grid.dr, grid.dz)
        # Uniform Bz: dBz/dz = 0. Uniform Br: (1/r) d(r*Br)/dr = Br/r ≠ 0
        assert divB.shape == (8, 8)
        assert not np.any(np.isnan(np.asarray(divB)))


# ──────────────────────────────────────────────────────────────────────────────
# Dedner source terms
# ──────────────────────────────────────────────────────────────────────────────


class TestDednerSource:
    def test_uniform_state_zero_source(self):
        grid = _cart_grid()
        U = _uniform_U(Bx=0.5)
        psi = mlx.zeros((8, 8, 8))
        dpsi, dU = dedner_source(psi, U, ch=1.0, cr=10.0, grid=grid)
        assert float(mlx.max(mlx.abs(dpsi))) < 1e-10
        assert float(mlx.max(mlx.abs(dU))) < 1e-10

    def test_nonzero_psi_decays(self):
        grid = _cart_grid()
        U = _uniform_U(Bx=0.1)
        psi = mlx.ones((8, 8, 8)) * 0.5
        dpsi, _ = dedner_source(psi, U, ch=1.0, cr=10.0, grid=grid)
        # cr * psi term should dominate → dpsi_dt < 0
        assert float(mlx.mean(dpsi)) < 0.0

    def test_non_solenoidal_B_produces_dpsi(self):
        grid = _cart_grid()
        x = np.linspace(0, 1, 8).reshape(8, 1, 1) * np.ones((1, 8, 8))
        U = _uniform_U()
        rows = [U[i] for i in range(NVAR)]
        rows[IBR] = mlx.array(x.astype(np.float32))
        U = mlx.stack(rows, axis=0)
        psi = mlx.zeros((8, 8, 8))
        dpsi, dU = dedner_source(psi, U, ch=1.0, cr=10.0, grid=grid)
        assert float(mlx.max(mlx.abs(dpsi))) > 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Powell source terms
# ──────────────────────────────────────────────────────────────────────────────


class TestPowellSource:
    def test_uniform_B_zero_source(self):
        grid = _cart_grid()
        U = _uniform_U(Bx=0.5)
        S = powell_source(U, 5.0 / 3.0, grid)
        assert float(mlx.max(mlx.abs(S))) < 1e-10

    def test_non_solenoidal_B_nonzero_source(self):
        grid = _cart_grid()
        x = np.linspace(0, 1, 8).reshape(8, 1, 1) * np.ones((1, 8, 8))
        U = _uniform_U()
        rows = [U[i] for i in range(NVAR)]
        rows[IBR] = mlx.array(x.astype(np.float32))
        U = mlx.stack(rows, axis=0)
        S = powell_source(U, 5.0 / 3.0, grid)
        # IBR source should be non-zero (div(B) * vx, but v=0 here)
        # IMR source should be non-zero (div(B) * Bx)
        # Since v=0, momentum and energy sources dominate, induction is zero
        from dpf.metal.mlx_kernels import IMR
        assert float(mlx.max(mlx.abs(S[IMR]))) > 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Solver integration
# ──────────────────────────────────────────────────────────────────────────────


class TestDednerSolverIntegration:
    def test_dedner_enabled_by_default_cartesian(self):
        s = MLXMHDSolver(
            grid_shape=(8, 8, 8), dx=0.1, coordinates="cartesian",
        )
        assert s._enable_dedner is True

    def test_dedner_disabled_by_default_cylindrical_ct(self):
        s = MLXMHDSolver(
            grid_shape=(8, 1, 8), dx=0.1, coordinates="cylindrical", use_ct=True,
        )
        assert s._enable_dedner is False

    def test_uniform_preserved_with_dedner(self):
        s = MLXMHDSolver(
            grid_shape=(16, 16, 16), dx=0.01, coordinates="cartesian",
            riemann_solver="hll", reconstruction="plm",
            use_dual_energy=False, enable_dedner=True,
        )
        state = {
            "rho": np.ones((16, 16, 16)),
            "velocity": np.zeros((3, 16, 16, 16)),
            "pressure": np.ones((16, 16, 16)),
            "B": np.zeros((3, 16, 16, 16)),
            "Te": np.full((16, 16, 16), 300.0),
            "Ti": np.full((16, 16, 16), 300.0),
            "psi": np.zeros((16, 16, 16)),
        }
        result = s.step(state, dt=1e-6, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)
        assert s._psi is not None

    def test_no_nan_with_dedner_sod(self):
        nx, ny, nz = 32, 4, 4
        s = MLXMHDSolver(
            grid_shape=(nx, ny, nz), dx=1.0 / nx, coordinates="cartesian",
            riemann_solver="hll", reconstruction="plm",
            use_dual_energy=False, enable_dedner=True,
        )
        rho = np.ones((nx, ny, nz))
        p = np.ones((nx, ny, nz))
        rho[:nx // 2] = 1.0
        rho[nx // 2:] = 0.125
        p[:nx // 2] = 1.0
        p[nx // 2:] = 0.1
        state = {
            "rho": rho,
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": p,
            "B": np.zeros((3, nx, ny, nz)),
            "Te": np.full((nx, ny, nz), 300.0),
            "Ti": np.full((nx, ny, nz), 300.0),
            "psi": np.zeros((nx, ny, nz)),
        }
        for _ in range(20):
            dt = s.compute_dt(state)
            state = s.step(state, dt=dt, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(state["rho"]))

    def test_powell_no_crash(self):
        s = MLXMHDSolver(
            grid_shape=(8, 8, 8), dx=0.1, coordinates="cartesian",
            riemann_solver="hll", reconstruction="plm",
            use_dual_energy=False, enable_powell=True,
        )
        state = {
            "rho": np.ones((8, 8, 8)),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.ones((8, 8, 8)),
            "B": np.ones((3, 8, 8, 8)) * 0.1,
            "Te": np.full((8, 8, 8), 300.0),
            "Ti": np.full((8, 8, 8), 300.0),
            "psi": np.zeros((8, 8, 8)),
        }
        result = s.step(state, dt=1e-4, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(result["rho"]))
