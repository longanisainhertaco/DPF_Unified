"""Tests for MLX solver Cartesian 3D support (Sprint 8).

Covers: instantiation, uniform state preservation, Sod shock tube,
conservation, and cross-backend parity with MetalMHDSolver.
"""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from dpf.metal.mlx_grid import CartesianGrid, CylindricalGrid  # noqa: E402
from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: E402

_NX, _NY, _NZ = 16, 16, 16
_DX = 0.01


def _uniform_state(
    nx: int = _NX,
    ny: int = _NY,
    nz: int = _NZ,
    rho: float = 1.0,
    p: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build a uniform Cartesian state dict."""
    return {
        "rho": np.full((nx, ny, nz), rho),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.full((nx, ny, nz), p),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), 300.0),
        "Ti": np.full((nx, ny, nz), 300.0),
        "psi": np.zeros((nx, ny, nz)),
    }


def _sod_state(nx: int = 64, ny: int = 4, nz: int = 4) -> dict[str, np.ndarray]:
    """Sod shock tube along x-axis."""
    rho = np.ones((nx, ny, nz))
    p = np.ones((nx, ny, nz))
    rho[:nx // 2] = 1.0
    rho[nx // 2:] = 0.125
    p[:nx // 2] = 1.0
    p[nx // 2:] = 0.1
    return {
        "rho": rho,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": p,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), 300.0),
        "Ti": np.full((nx, ny, nz), 300.0),
        "psi": np.zeros((nx, ny, nz)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Grid tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianGrid:
    def test_instantiation(self):
        g = CartesianGrid(nx=8, ny=8, nz=8, dx=0.1)
        assert g.nx == 8
        assert g.ny == 8
        assert g.nz == 8
        assert g.dx == pytest.approx(0.1)
        assert g.dy == pytest.approx(0.1)
        assert g.dz == pytest.approx(0.1)
        assert g.r_cell is None
        assert g.r_face is None
        assert g.inv_r is None

    def test_custom_spacing(self):
        g = CartesianGrid(nx=4, ny=8, nz=16, dx=0.1, dy=0.2, dz=0.05)
        assert g.dy == pytest.approx(0.2)
        assert g.dz == pytest.approx(0.05)
        assert g.cell_volume == pytest.approx(0.1 * 0.2 * 0.05)

    def test_total_volume(self):
        g = CartesianGrid(nx=10, ny=10, nz=10, dx=0.1)
        assert g.total_volume() == pytest.approx(1.0)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            CartesianGrid(nx=0, ny=1, nz=1, dx=0.1)

    def test_invalid_dx(self):
        with pytest.raises(ValueError):
            CartesianGrid(nx=4, ny=4, nz=4, dx=-0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Solver instantiation
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianInstantiation:
    def test_basic_cartesian(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
        )
        assert s.coordinates == "cartesian"
        assert isinstance(s._grid, CartesianGrid)
        assert s.ny == _NY

    def test_cylindrical_still_works(self):
        s = MLXMHDSolver(
            grid_shape=(16, 1, 16),
            dx=_DX,
            coordinates="cylindrical",
        )
        assert isinstance(s._grid, CylindricalGrid)

    def test_cylindrical_ny_guard(self):
        with pytest.raises(ValueError, match="ny=1"):
            MLXMHDSolver(
                grid_shape=(16, 4, 16),
                dx=_DX,
                coordinates="cylindrical",
            )

    def test_cartesian_allows_any_ny(self):
        s = MLXMHDSolver(
            grid_shape=(8, 12, 8),
            dx=_DX,
            coordinates="cartesian",
        )
        assert s.ny == 12


# ──────────────────────────────────────────────────────────────────────────────
# State dict round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianStateDictRoundTrip:
    def test_pack_unpack_shapes(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
        )
        state = _uniform_state()
        U = s._state_mgr.from_state_dict(state)
        assert U.shape == (10, _NX, _NY, _NZ)

        out = s._state_mgr.to_state_dict(U)
        assert out["rho"].shape == (_NX, _NY, _NZ)
        assert out["velocity"].shape == (3, _NX, _NY, _NZ)
        assert out["B"].shape == (3, _NX, _NY, _NZ)

    def test_pack_unpack_values(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
        )
        state = _uniform_state(rho=2.5, p=3.0)
        U = s._state_mgr.from_state_dict(state)
        out = s._state_mgr.to_state_dict(U)
        np.testing.assert_allclose(out["rho"], 2.5, atol=1e-6)
        np.testing.assert_allclose(out["pressure"], 3.0, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Uniform state preservation
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianUniformPreservation:
    def test_uniform_preserved_plm_hll(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
            riemann_solver="hll",
            reconstruction="plm",
            use_dual_energy=False,
        )
        state = _uniform_state()
        result = s.step(state, dt=1e-6, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["pressure"], 1.0, atol=1e-5)

    def test_uniform_preserved_weno5z_hll(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
            riemann_solver="hll",
            reconstruction="weno5z",
            use_dual_energy=False,
        )
        state = _uniform_state()
        result = s.step(state, dt=1e-6, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["pressure"], 1.0, atol=1e-5)

    def test_uniform_with_B_field(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
            riemann_solver="hll",
            reconstruction="plm",
            use_dual_energy=False,
        )
        state = _uniform_state()
        state["B"][0] = 0.1  # uniform Bx
        result = s.step(state, dt=1e-8, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-5)
        np.testing.assert_allclose(result["B"][0], 0.1, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Sod shock tube
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianSodShock:
    @pytest.mark.slow
    def test_sod_no_nan(self):
        nx, ny, nz = 64, 4, 4
        s = MLXMHDSolver(
            grid_shape=(nx, ny, nz),
            dx=1.0 / nx,
            coordinates="cartesian",
            riemann_solver="hll",
            reconstruction="plm",
            use_dual_energy=False,
        )
        state = _sod_state(nx, ny, nz)
        for _ in range(50):
            dt = s.compute_dt(state)
            state = s.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))
        # Shock should modify density profile
        rho_x = state["rho"][:, ny // 2, nz // 2]
        assert rho_x.min() < 0.5
        assert rho_x.max() > 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Conservation
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianConservation:
    def test_mass_conservation_uniform(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
            riemann_solver="hll",
            reconstruction="plm",
            use_dual_energy=False,
        )
        state = _uniform_state(rho=2.0)
        mass_initial = state["rho"].sum()
        for _ in range(10):
            dt = s.compute_dt(state)
            state = s.step(state, dt=dt, current=0.0, voltage=0.0)

        mass_final = state["rho"].sum()
        np.testing.assert_allclose(mass_final, mass_initial, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# CFL
# ──────────────────────────────────────────────────────────────────────────────


class TestCartesianCFL:
    def test_cfl_positive(self):
        s = MLXMHDSolver(
            grid_shape=(_NX, _NY, _NZ),
            dx=_DX,
            coordinates="cartesian",
        )
        state = _uniform_state()
        dt = s.compute_dt(state)
        assert dt > 0.0
        assert np.isfinite(dt)

    def test_cfl_respects_dy(self):
        s1 = MLXMHDSolver(
            grid_shape=(8, 8, 8),
            dx=0.1,
            coordinates="cartesian",
            dy=0.1,
        )
        s2 = MLXMHDSolver(
            grid_shape=(8, 8, 8),
            dx=0.1,
            coordinates="cartesian",
            dy=0.01,
        )
        state = _uniform_state(nx=8, ny=8, nz=8)
        dt1 = s1.compute_dt(state)
        dt2 = s2.compute_dt(state)
        # Smaller dy → smaller dt
        assert dt2 < dt1


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction dim=2
# ──────────────────────────────────────────────────────────────────────────────


class TestReconstructionDim2:
    def test_plm_dim2(self):
        from dpf.metal.mlx_reconstruction import plm_reconstruct
        Q = mlx.ones((10, 8, 8, 8))
        QL, QR = plm_reconstruct(Q, dim=2)
        assert QL.shape == (10, 8, 8, 7)
        assert QR.shape == (10, 8, 8, 7)

    def test_weno5z_dim2(self):
        from dpf.metal.mlx_reconstruction import weno5z_reconstruct
        Q = mlx.ones((10, 8, 8, 12))
        QL, QR = weno5z_reconstruct(Q, dim=2)
        assert QL.shape == (10, 8, 8, 7)
        assert QR.shape == (10, 8, 8, 7)
