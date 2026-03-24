"""Tests for the MLX cylindrical grid geometry module."""
from __future__ import annotations

import math

import pytest

mlx = pytest.importorskip("mlx", reason="MLX not available")
mx = pytest.importorskip("mlx.core", reason="MLX not available")

from dpf.metal.mlx_grid import CylindricalGrid  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(arr: mx.array) -> list[float]:
    """Convert mx.array to a Python list for assertions."""
    return arr.tolist()


# ---------------------------------------------------------------------------
# Construction / shape tests
# ---------------------------------------------------------------------------

class TestGridShape:
    def test_r_cell_length(self) -> None:
        g = CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=2e-3)
        assert len(_np(g.r_cell)) == 16

    def test_r_face_length_is_nr_plus_one(self) -> None:
        g = CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=2e-3)
        assert len(_np(g.r_face)) == 17  # nr+1

    def test_z_cell_length(self) -> None:
        g = CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=2e-3)
        assert len(_np(g.z_cell)) == 32

    def test_cell_volume_length(self) -> None:
        g = CylindricalGrid(nr=64, nz=128, dr=5e-4, dz=1e-3)
        assert len(_np(g.cell_volume)) == 64

    def test_face_area_r_length(self) -> None:
        g = CylindricalGrid(nr=64, nz=128, dr=5e-4, dz=1e-3)
        assert len(_np(g.face_area_r)) == 65  # nr+1

    def test_face_area_z_length(self) -> None:
        g = CylindricalGrid(nr=64, nz=128, dr=5e-4, dz=1e-3)
        assert len(_np(g.face_area_z)) == 64

    def test_inv_r_length(self) -> None:
        g = CylindricalGrid(nr=128, nz=512, dr=2e-4, dz=5e-4)
        assert len(_np(g.inv_r)) == 128


# ---------------------------------------------------------------------------
# Cell-center positions
# ---------------------------------------------------------------------------

class TestCellPositions:
    def test_r_cell_first_value_on_axis(self) -> None:
        dr = 1e-3
        g = CylindricalGrid(nr=8, nz=4, dr=dr, dz=1e-3, r_inner=0.0)
        assert math.isclose(_np(g.r_cell)[0], 0.5 * dr, rel_tol=1e-5)

    def test_r_cell_last_value(self) -> None:
        nr = 8
        dr = 1e-3
        g = CylindricalGrid(nr=nr, nz=4, dr=dr, dz=1e-3, r_inner=0.0)
        expected = (nr - 0.5) * dr
        assert math.isclose(_np(g.r_cell)[-1], expected, rel_tol=1e-5)

    def test_r_cell_with_nonzero_r_inner(self) -> None:
        r_inner = 0.02
        dr = 1e-3
        g = CylindricalGrid(nr=10, nz=5, dr=dr, dz=1e-3, r_inner=r_inner)
        assert math.isclose(_np(g.r_cell)[0], r_inner + 0.5 * dr, rel_tol=1e-5)

    def test_r_face_first_is_r_inner(self) -> None:
        g = CylindricalGrid(nr=8, nz=4, dr=1e-3, dz=1e-3, r_inner=0.005)
        assert math.isclose(_np(g.r_face)[0], 0.005, rel_tol=1e-5)

    def test_r_face_last_is_r_outer(self) -> None:
        nr = 8
        dr = 1e-3
        r_inner = 0.0
        g = CylindricalGrid(nr=nr, nz=4, dr=dr, dz=1e-3, r_inner=r_inner)
        assert math.isclose(_np(g.r_face)[-1], r_inner + nr * dr, rel_tol=1e-5)

    def test_z_cell_first_value(self) -> None:
        dz = 2e-3
        g = CylindricalGrid(nr=4, nz=10, dr=1e-3, dz=dz)
        assert math.isclose(_np(g.z_cell)[0], 0.5 * dz, rel_tol=1e-5)

    def test_z_cell_last_value(self) -> None:
        nz = 10
        dz = 2e-3
        g = CylindricalGrid(nr=4, nz=nz, dr=1e-3, dz=dz)
        assert math.isclose(_np(g.z_cell)[-1], (nz - 0.5) * dz, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Volume conservation
# ---------------------------------------------------------------------------

class TestVolumes:
    def _expected_total(self, nr: int, nz: int, dr: float, dz: float, r_inner: float) -> float:
        r_outer = r_inner + nr * dr
        length = nz * dz
        return math.pi * (r_outer**2 - r_inner**2) * length

    def test_volume_sum_full_cylinder_small(self) -> None:
        nr, nz = 16, 32
        dr, dz = 1e-2, 5e-3
        g = CylindricalGrid(nr=nr, nz=nz, dr=dr, dz=dz, r_inner=0.0)
        got = g.total_volume()
        expected = self._expected_total(nr, nz, dr, dz, 0.0)
        assert math.isclose(got, expected, rel_tol=1e-4), f"{got} != {expected}"

    def test_volume_sum_full_cylinder_medium(self) -> None:
        nr, nz = 64, 128
        dr, dz = 5e-4, 1e-3
        g = CylindricalGrid(nr=nr, nz=nz, dr=dr, dz=dz, r_inner=0.0)
        got = g.total_volume()
        expected = self._expected_total(nr, nz, dr, dz, 0.0)
        assert math.isclose(got, expected, rel_tol=1e-4)

    def test_volume_sum_full_cylinder_large(self) -> None:
        nr, nz = 128, 512
        dr, dz = 2e-4, 5e-4
        g = CylindricalGrid(nr=nr, nz=nz, dr=dr, dz=dz, r_inner=0.0)
        got = g.total_volume()
        expected = self._expected_total(nr, nz, dr, dz, 0.0)
        assert math.isclose(got, expected, rel_tol=1e-4)

    def test_volume_annular_cylinder(self) -> None:
        nr, nz = 32, 64
        dr, dz = 1e-3, 2e-3
        r_inner = 0.01
        g = CylindricalGrid(nr=nr, nz=nz, dr=dr, dz=dz, r_inner=r_inner)
        got = g.total_volume()
        expected = self._expected_total(nr, nz, dr, dz, r_inner)
        assert math.isclose(got, expected, rel_tol=1e-4)

    def test_cell_volume_all_positive(self) -> None:
        g = CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=2e-3)
        vols = _np(g.cell_volume)
        assert all(v > 0.0 for v in vols)


# ---------------------------------------------------------------------------
# Face areas
# ---------------------------------------------------------------------------

class TestFaceAreas:
    def test_face_area_r_at_axis_is_zero(self) -> None:
        g = CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=2e-3, r_inner=0.0)
        assert math.isclose(_np(g.face_area_r)[0], 0.0, abs_tol=1e-15)

    def test_face_area_r_outer_face(self) -> None:
        nr = 8
        dr = 1e-3
        dz = 2e-3
        r_outer = nr * dr
        g = CylindricalGrid(nr=nr, nz=4, dr=dr, dz=dz, r_inner=0.0)
        expected = 2.0 * math.pi * r_outer * dz
        assert math.isclose(_np(g.face_area_r)[-1], expected, rel_tol=1e-5)

    def test_face_area_z_equals_annular_area(self) -> None:
        dr = 1e-3
        g = CylindricalGrid(nr=4, nz=8, dr=dr, dz=2e-3, r_inner=0.0)
        areas = _np(g.face_area_z)
        for i, a in enumerate(areas):
            r_in = i * dr
            r_out = (i + 1) * dr
            expected = math.pi * (r_out**2 - r_in**2)
            assert math.isclose(a, expected, rel_tol=1e-5), f"cell {i}: {a} != {expected}"

    def test_face_area_r_nonzero_r_inner_inner_face(self) -> None:
        r_inner = 0.01
        dz = 2e-3
        g = CylindricalGrid(nr=8, nz=4, dr=1e-3, dz=dz, r_inner=r_inner)
        expected = 2.0 * math.pi * r_inner * dz
        assert math.isclose(_np(g.face_area_r)[0], expected, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# inv_r / L'Hopital
# ---------------------------------------------------------------------------

class TestInvR:
    def test_inv_r_at_axis_uses_lhopital(self) -> None:
        dr = 1e-3
        g = CylindricalGrid(nr=8, nz=4, dr=dr, dz=1e-3, r_inner=0.0)
        # First cell center is at dr/2; r_inner=0 so the formula is 2/dr
        assert math.isclose(_np(g.inv_r)[0], 2.0 / dr, rel_tol=1e-5)

    def test_inv_r_interior_is_reciprocal(self) -> None:
        dr = 1e-3
        g = CylindricalGrid(nr=8, nz=4, dr=dr, dz=1e-3, r_inner=0.0)
        inv_r = _np(g.inv_r)
        r_cell = _np(g.r_cell)
        for i in range(1, 8):
            assert math.isclose(inv_r[i], 1.0 / r_cell[i], rel_tol=1e-4)

    def test_inv_r_no_inf_or_nan(self) -> None:
        g = CylindricalGrid(nr=128, nz=512, dr=2e-4, dz=5e-4, r_inner=0.0)
        inv_r = _np(g.inv_r)
        assert all(math.isfinite(v) for v in inv_r)

    def test_inv_r_with_nonzero_r_inner_all_reciprocals(self) -> None:
        r_inner = 0.005
        dr = 1e-3
        g = CylindricalGrid(nr=8, nz=4, dr=dr, dz=1e-3, r_inner=r_inner)
        inv_r = _np(g.inv_r)
        r_cell = _np(g.r_cell)
        for i, (ir, rc) in enumerate(zip(inv_r, r_cell, strict=True)):
            assert math.isclose(ir, 1.0 / rc, rel_tol=1e-4), f"cell {i}: {ir} != 1/{rc}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_nr_raises(self) -> None:
        with pytest.raises(ValueError):
            CylindricalGrid(nr=0, nz=32, dr=1e-3, dz=1e-3)

    def test_negative_dr_raises(self) -> None:
        with pytest.raises(ValueError):
            CylindricalGrid(nr=16, nz=32, dr=-1e-3, dz=1e-3)

    def test_negative_r_inner_raises(self) -> None:
        with pytest.raises(ValueError):
            CylindricalGrid(nr=16, nz=32, dr=1e-3, dz=1e-3, r_inner=-0.01)
