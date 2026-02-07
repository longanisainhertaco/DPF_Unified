"""2D axisymmetric cylindrical (r, z) geometry operators.

For a Dense Plasma Focus, the natural coordinate system is (r, z) with
azimuthal symmetry (d/d_theta = 0). The plasma state is stored on a
2D grid: axis 0 = radial (r), axis 1 = axial (z).

The 3D arrays from the MHD solver become 2D slices:
    - Scalar fields: shape (nr, nz)
    - Vector fields: (v_r, v_theta, v_z) stored as shape (3, nr, nz)
      where component 0=r, 1=theta, 2=z

Key formulae for axisymmetric cylindrical coordinates:

Divergence of a vector F:
    div(F) = (1/r) * d(r * F_r)/dr + (1/r) * dF_theta/dtheta + dF_z/dz
    With d/dtheta = 0:
    div(F) = (1/r) * d(r * F_r)/dr + dF_z/dz
           = dF_r/dr + F_r/r + dF_z/dz

Gradient of a scalar p:
    grad(p) = (dp/dr, 0, dp/dz)

Curl of a vector B (axisymmetric):
    (curl B)_r     = -dB_theta/dz
    (curl B)_theta = dB_r/dz - dB_z/dr
    (curl B)_z     = (1/r) * d(r * B_theta)/dr = dB_theta/dr + B_theta/r

Laplacian of a scalar T:
    lap(T) = (1/r) * d(r * dT/dr)/dr + d^2T/dz^2
           = d^2T/dr^2 + (1/r) * dT/dr + d^2T/dz^2

Geometric source terms for momentum (hoop stress):
    S_r = p_theta / r  (centrifugal + magnetic pressure)
    For MHD: p_total / r contribution to radial momentum

Reference:
    Lieberman & Lichtenberg, "Principles of Plasma Discharges" Ch. 5
    Goedbloed & Poedts, "Principles of MHD" (2004)
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _safe_inv_r(r: np.ndarray) -> np.ndarray:
    """Compute 1/r with protection at r=0.

    At r=0 (axis), uses L'Hopital's rule:
        lim_{r->0} f(r)/r = f'(0) for smooth f vanishing at origin.
    We approximate 1/r -> 1/r_min for the first cell.

    Args:
        r: Radial coordinate array [m], shape (nr,).

    Returns:
        1/r array with axis protection, shape (nr,).
    """
    inv_r = np.empty_like(r)
    for i in range(len(r)):
        if r[i] < 1e-30:
            # At r=0, use 1/(dr/2) from the next cell's spacing
            if i + 1 < len(r) and r[i + 1] > 0:
                inv_r[i] = 1.0 / r[i + 1]
            else:
                inv_r[i] = 0.0
        else:
            inv_r[i] = 1.0 / r[i]
    return inv_r


class CylindricalGeometry:
    """2D axisymmetric cylindrical geometry operator provider.

    Manages the radial coordinate array and provides geometry-aware
    differential operators (div, grad, curl, laplacian) for the MHD solver.

    The grid is cell-centered with:
        r[i] = (i + 0.5) * dr  for i = 0, ..., nr-1
        z[j] = (j + 0.5) * dz  for j = 0, ..., nz-1

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.
        dr: Radial grid spacing [m].
        dz: Axial grid spacing [m].
    """

    def __init__(self, nr: int, nz: int, dr: float, dz: float) -> None:
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz

        # Cell-centered radial coordinates
        self.r = np.array([(i + 0.5) * dr for i in range(nr)])

        # Precompute 1/r (with axis protection)
        self.inv_r = _safe_inv_r(self.r)

        # 2D broadcast arrays for element-wise operations
        # Shape (nr, 1) for broadcasting with (nr, nz) fields
        self.r_2d = self.r[:, np.newaxis]
        self.inv_r_2d = self.inv_r[:, np.newaxis]

        # Face-centered radial coordinates for flux computation
        # r_face[i] = i * dr (left face of cell i; r_face[0] = 0 = axis)
        self.r_face = np.array([i * dr for i in range(nr + 1)])

    def divergence(self, F: np.ndarray) -> np.ndarray:
        """Compute divergence of a vector field in cylindrical coords.

        div(F) = (1/r) * d(r * F_r)/dr + dF_z/dz

        For axisymmetric fields, the theta component is ignored.

        Args:
            F: Vector field, shape (3, nr, nz) with components (F_r, F_theta, F_z).

        Returns:
            Scalar divergence field, shape (nr, nz).
        """
        F_r = F[0]
        F_z = F[2]

        # d(r * F_r)/dr using finite differences
        rFr = self.r_2d * F_r
        d_rFr_dr = np.gradient(rFr, self.dr, axis=0)

        # (1/r) * d(r * F_r)/dr
        div_r = self.inv_r_2d * d_rFr_dr

        # dF_z / dz
        dFz_dz = np.gradient(F_z, self.dz, axis=1)

        return div_r + dFz_dz

    def gradient(self, p: np.ndarray) -> np.ndarray:
        """Compute gradient of a scalar field in cylindrical coords.

        grad(p) = (dp/dr, 0, dp/dz)

        Args:
            p: Scalar field, shape (nr, nz).

        Returns:
            Vector gradient, shape (3, nr, nz).
        """
        grad = np.zeros((3, self.nr, self.nz))
        grad[0] = np.gradient(p, self.dr, axis=0)
        grad[2] = np.gradient(p, self.dz, axis=1)
        return grad

    def curl(self, B: np.ndarray) -> np.ndarray:
        """Compute curl of a vector field in axisymmetric cylindrical coords.

        (curl B)_r     = -dB_theta/dz
        (curl B)_theta = dB_r/dz - dB_z/dr
        (curl B)_z     = (1/r) * d(r * B_theta)/dr

        Args:
            B: Vector field, shape (3, nr, nz) with components (B_r, B_theta, B_z).

        Returns:
            Curl vector field, shape (3, nr, nz).
        """
        B_r = B[0]
        B_theta = B[1]
        B_z = B[2]

        curl = np.zeros((3, self.nr, self.nz))

        # (curl B)_r = -dB_theta/dz
        curl[0] = -np.gradient(B_theta, self.dz, axis=1)

        # (curl B)_theta = dB_r/dz - dB_z/dr
        curl[1] = np.gradient(B_r, self.dz, axis=1) - np.gradient(B_z, self.dr, axis=0)

        # (curl B)_z = (1/r) * d(r * B_theta)/dr
        rBtheta = self.r_2d * B_theta
        curl[2] = self.inv_r_2d * np.gradient(rBtheta, self.dr, axis=0)

        return curl

    def laplacian(self, T: np.ndarray) -> np.ndarray:
        """Compute Laplacian of a scalar field in cylindrical coords.

        lap(T) = (1/r) * d(r * dT/dr)/dr + d^2T/dz^2

        Uses second-order centered differences.

        Args:
            T: Scalar field, shape (nr, nz).

        Returns:
            Laplacian field, shape (nr, nz).
        """
        dT_dr = np.gradient(T, self.dr, axis=0)
        r_dTdr = self.r_2d * dT_dr
        lap_r = self.inv_r_2d * np.gradient(r_dTdr, self.dr, axis=0)

        # d^2T/dz^2 via second-order centered difference
        d2T_dz2 = np.gradient(np.gradient(T, self.dz, axis=1), self.dz, axis=1)

        return lap_r + d2T_dz2

    def geometric_source_momentum(
        self,
        rho: np.ndarray,
        velocity: np.ndarray,
        pressure: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Compute geometric source terms for momentum equation.

        In cylindrical coords, the momentum equation has additional source
        terms from the curvature:

        Radial momentum source:
            S_r = (rho * v_theta^2 + p + B_theta^2/(2*mu_0)) / r
                - (B_r * B_theta) / (mu_0 * r)

        For axisymmetric DPF (v_theta = 0, B_r = 0 in initial state):
            S_r = p / r + B_theta^2 / (2 * mu_0 * r)  (hoop stress)

        Theta momentum source:
            S_theta = -(rho * v_r * v_theta + B_r * B_theta / mu_0) / r

        Args:
            rho: Density, shape (nr, nz).
            velocity: Velocity (v_r, v_theta, v_z), shape (3, nr, nz).
            pressure: Thermal pressure, shape (nr, nz).
            B: Magnetic field (B_r, B_theta, B_z), shape (3, nr, nz).

        Returns:
            Source term for momentum, shape (3, nr, nz).
        """
        from dpf.constants import mu_0

        v_r = velocity[0]
        v_theta = velocity[1]
        B_r = B[0]
        B_theta = B[1]

        source = np.zeros((3, self.nr, self.nz))

        # Radial: hoop stress + centrifugal + magnetic tension
        # S_r = (rho * v_theta^2 + p + B_theta^2/(2*mu_0)) / r
        #     - B_r * B_theta / (mu_0 * r)
        p_mag_theta = B_theta**2 / (2.0 * mu_0)
        source[0] = self.inv_r_2d * (
            rho * v_theta**2 + pressure + p_mag_theta
            - B_r * B_theta / mu_0
        )

        # Theta: Coriolis + magnetic tension
        # S_theta = -(rho * v_r * v_theta + B_r * B_theta / mu_0) / r
        source[1] = -self.inv_r_2d * (
            rho * v_r * v_theta + B_r * B_theta / mu_0
        )

        # Axial: no geometric source
        # S_z = 0

        return source

    def cell_volumes(self) -> np.ndarray:
        """Compute cell volumes for a 2D axisymmetric grid.

        V[i,j] = pi * (r_face[i+1]^2 - r_face[i]^2) * dz

        For cell-centered r[i] = (i+0.5)*dr:
            r_face[i] = i*dr, r_face[i+1] = (i+1)*dr
            V = pi * ((i+1)^2 - i^2) * dr^2 * dz = pi * (2*i+1) * dr^2 * dz

        Returns:
            Cell volumes, shape (nr, nz).
        """
        from dpf.constants import pi

        volumes = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_out = self.r_face[i + 1]
            r_in = self.r_face[i]
            vol = pi * (r_out**2 - r_in**2) * self.dz
            volumes[i, :] = vol
        return volumes

    def face_areas_radial(self) -> np.ndarray:
        """Compute radial face areas (for flux through r-faces).

        A_r[i] = 2 * pi * r_face[i] * dz

        Returns:
            Radial face areas, shape (nr+1, nz).
        """
        from dpf.constants import pi

        areas = np.zeros((self.nr + 1, self.nz))
        for i in range(self.nr + 1):
            areas[i, :] = 2.0 * pi * self.r_face[i] * self.dz
        return areas

    def face_areas_axial(self) -> np.ndarray:
        """Compute axial face areas (for flux through z-faces).

        A_z[i] = pi * (r_face[i+1]^2 - r_face[i]^2)

        Returns:
            Axial face areas, shape (nr, nz+1).
        """
        from dpf.constants import pi

        areas = np.zeros((self.nr, self.nz + 1))
        for i in range(self.nr):
            r_out = self.r_face[i + 1]
            r_in = self.r_face[i]
            areas[i, :] = pi * (r_out**2 - r_in**2)
        return areas

    def integrate_volume(self, field: np.ndarray) -> float:
        """Integrate a scalar field over the cylindrical volume.

        integral = sum_ij(field[i,j] * V[i,j])

        Args:
            field: Scalar field, shape (nr, nz).

        Returns:
            Volume integral of the field.
        """
        return float(np.sum(field * self.cell_volumes()))

    def div_B_cylindrical(self, B: np.ndarray) -> np.ndarray:
        """Compute div(B) in cylindrical coordinates.

        div(B) = (1/r) * d(r * B_r)/dr + dB_z/dz

        (B_theta doesn't contribute to div(B) for axisymmetric fields.)

        Args:
            B: Magnetic field, shape (3, nr, nz).

        Returns:
            div(B) field, shape (nr, nz).
        """
        B_r = B[0]
        B_z = B[2]

        rBr = self.r_2d * B_r
        div_r = self.inv_r_2d * np.gradient(rBr, self.dr, axis=0)
        dBz_dz = np.gradient(B_z, self.dz, axis=1)

        return div_r + dBz_dz
