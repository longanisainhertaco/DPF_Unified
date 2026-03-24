"""Cylindrical grid geometry for the MLX MHD solver.

Pre-computes cell-center radii, face radii, cell volumes, and face areas
as cached mx.array tensors. All geometry is axisymmetric (r, z) with
ny=1 implicitly.

Grid layout:
  - r: nr cells from r_inner + dr/2 to r_inner + (nr-0.5)*dr
  - z: nz cells from dz/2 to (nz-0.5)*dz
  - Faces at cell boundaries: r_face[i] = r_inner + i*dr (nr+1 values)
"""
from __future__ import annotations

import math

import mlx.core as mx


class CylindricalGrid:
    """Axisymmetric cylindrical grid geometry with cached MLX arrays.

    Parameters
    ----------
    nr : int
        Number of radial cells.
    nz : int
        Number of axial cells.
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    r_inner : float
        Inner radial boundary [m] (default 0.0 = axis).

    Attributes
    ----------
    r_cell : mx.array
        Cell-center radii, shape (nr,).
    r_face : mx.array
        Face radii at r_{i+1/2}, shape (nr+1,).
    z_cell : mx.array
        Cell-center z positions, shape (nz,).
    cell_volume : mx.array
        Cell volumes pi*(r_{i+1/2}^2 - r_{i-1/2}^2)*dz, shape (nr,).
    face_area_r : mx.array
        Radial face areas 2*pi*r_face*dz, shape (nr+1,).
    face_area_z : mx.array
        Axial face areas pi*(r_{i+1/2}^2 - r_{i-1/2}^2), shape (nr,).
    inv_r : mx.array
        1/r_cell with L'Hopital at axis (r=0), shape (nr,).
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        dr: float,
        dz: float,
        r_inner: float = 0.0,
    ) -> None:
        if nr < 1:
            raise ValueError(f"nr must be >= 1, got {nr}")
        if nz < 1:
            raise ValueError(f"nz must be >= 1, got {nz}")
        if dr <= 0.0:
            raise ValueError(f"dr must be > 0, got {dr}")
        if dz <= 0.0:
            raise ValueError(f"dz must be > 0, got {dz}")
        if r_inner < 0.0:
            raise ValueError(f"r_inner must be >= 0, got {r_inner}")

        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        self.r_inner = r_inner

        # Face radii: r_inner + i*dr for i in 0..nr inclusive
        r_face_np = [r_inner + i * dr for i in range(nr + 1)]
        self.r_face: mx.array = mx.array(r_face_np, dtype=mx.float32)

        # Cell-center radii: midpoint between consecutive faces
        r_cell_np = [r_inner + (i + 0.5) * dr for i in range(nr)]
        self.r_cell: mx.array = mx.array(r_cell_np, dtype=mx.float32)

        # Axial cell centers
        z_cell_np = [(j + 0.5) * dz for j in range(nz)]
        self.z_cell: mx.array = mx.array(z_cell_np, dtype=mx.float32)

        # Cell volumes: pi * (r_face[i+1]^2 - r_face[i]^2) * dz
        r_out = self.r_face[1:]   # shape (nr,)
        r_in = self.r_face[:-1]   # shape (nr,)
        self.cell_volume: mx.array = mx.array(math.pi, dtype=mx.float32) * (
            r_out * r_out - r_in * r_in
        ) * dz

        # Radial face areas: 2 * pi * r_face * dz, shape (nr+1,)
        self.face_area_r: mx.array = (
            mx.array(2.0 * math.pi * dz, dtype=mx.float32) * self.r_face
        )

        # Axial face areas: pi * (r_face[i+1]^2 - r_face[i]^2), shape (nr,)
        self.face_area_z: mx.array = mx.array(math.pi, dtype=mx.float32) * (
            r_out * r_out - r_in * r_in
        )

        # 1/r_cell with L'Hopital limit at axis: lim_{r->0} 1/r diverges,
        # but in cylindrical finite-volume the geometrically consistent
        # replacement for the first cell (r_inner=0) is 2/dr.
        inv_r_list: list[float] = []
        for i in range(nr):
            rc = r_inner + (i + 0.5) * dr
            if rc == 0.0:
                inv_r_list.append(2.0 / dr)
            else:
                inv_r_list.append(1.0 / rc)
        self.inv_r: mx.array = mx.array(inv_r_list, dtype=mx.float32)

        # Force evaluation so arrays are resident
        mx.eval(
            self.r_face,
            self.r_cell,
            self.z_cell,
            self.cell_volume,
            self.face_area_r,
            self.face_area_z,
            self.inv_r,
        )

    def total_volume(self) -> float:
        """Total grid volume [m^3]."""
        return float(mx.sum(self.cell_volume).item()) * self.nz
