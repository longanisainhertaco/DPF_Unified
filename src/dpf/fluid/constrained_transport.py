"""Constrained transport (CT) for divergence-free magnetic field evolution.

Constrained transport evolves B on a staggered (Yee) grid where each
B-component lives on the corresponding cell face.  The induction
equation dB/dt = -curl(E) is discretised using edge-centred electric
fields (EMFs).  Because the discrete curl of an edge field onto faces
satisfies an exact discrete identity,

    div(curl(E)) = 0    (exactly, to machine precision)

the face-centred divergence div(B) is preserved to round-off if the
initial condition is divergence-free.  This is far superior to
divergence cleaning (Dedner) for long-time accuracy.

The implementation follows the "simple CT" approach of Gardiner & Stone
(2005, JCP 205, 509) where edge EMFs are arithmetic averages of
face-centred fluxes.

Data layout:
    Bx lives on x-faces: shape (nx+1, ny, nz)
    By lives on y-faces: shape (nx, ny+1, nz)
    Bz lives on z-faces: shape (nx, ny, nz+1)

    Ex lives on x-edges (y-z edges): shape (nx, ny+1, nz+1)
    Ey lives on y-edges (x-z edges): shape (nx+1, ny, nz+1)
    Ez lives on z-edges (x-y edges): shape (nx+1, ny+1, nz)

References:
    Evans C.R. & Hawley J.F., ApJ 332, 659 (1988).
    Gardiner T.A. & Stone J.M., JCP 205, 509 (2005).
    Toth G., JCP 161, 605 (2000).

Classes:
    StaggeredBField: Dataclass for face-centred magnetic field.

Functions:
    cell_centered_to_face: Average cell B to face B.
    face_to_cell_centered: Average face B to cell B.
    ct_update: Advance face B using edge EMFs (divergence-free).
    compute_div_B: Face-centred divergence (should be ~machine epsilon).
    emf_from_fluxes: Simple CT averaging of face fluxes to edge EMFs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# ============================================================
# Staggered B-field data structure
# ============================================================

@dataclass
class StaggeredBField:
    """Face-centred magnetic field on a staggered (Yee) grid.

    Attributes:
        Bx_face: x-component on x-faces, shape (nx+1, ny, nz).
        By_face: y-component on y-faces, shape (nx, ny+1, nz).
        Bz_face: z-component on z-faces, shape (nx, ny, nz+1).
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
    """

    Bx_face: np.ndarray
    By_face: np.ndarray
    Bz_face: np.ndarray
    dx: float
    dy: float
    dz: float


# ============================================================
# Cell-centred <-> face-centred conversions
# ============================================================

def cell_centered_to_face(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> StaggeredBField:
    """Average cell-centred B to face-centred (staggered) representation.

    B_face[i+1/2] = 0.5 * (B[i] + B[i+1]) along the component direction.

    For a cell-centred field of shape (nx, ny, nz), the face-centred
    field has shape (nx+1, ny, nz) for Bx, etc.  The outermost faces
    are extrapolated by copying the nearest interior face.

    Args:
        Bx, By, Bz: Cell-centred B components, each shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].

    Returns:
        StaggeredBField with face-centred components.
    """
    nx, ny, nz = Bx.shape

    # Bx on x-faces: (nx+1, ny, nz)
    Bx_f = np.zeros((nx + 1, ny, nz))
    Bx_f[1:nx, :, :] = 0.5 * (Bx[:nx - 1, :, :] + Bx[1:nx, :, :])
    Bx_f[0, :, :] = Bx[0, :, :]        # extrapolate left
    Bx_f[nx, :, :] = Bx[nx - 1, :, :]  # extrapolate right

    # By on y-faces: (nx, ny+1, nz)
    By_f = np.zeros((nx, ny + 1, nz))
    By_f[:, 1:ny, :] = 0.5 * (By[:, :ny - 1, :] + By[:, 1:ny, :])
    By_f[:, 0, :] = By[:, 0, :]
    By_f[:, ny, :] = By[:, ny - 1, :]

    # Bz on z-faces: (nx, ny, nz+1)
    Bz_f = np.zeros((nx, ny, nz + 1))
    Bz_f[:, :, 1:nz] = 0.5 * (Bz[:, :, :nz - 1] + Bz[:, :, 1:nz])
    Bz_f[:, :, 0] = Bz[:, :, 0]
    Bz_f[:, :, nz] = Bz[:, :, nz - 1]

    return StaggeredBField(
        Bx_face=Bx_f,
        By_face=By_f,
        Bz_face=Bz_f,
        dx=dx,
        dy=dy,
        dz=dz,
    )


def face_to_cell_centered(
    staggered: StaggeredBField,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average face-centred B back to cell centres.

    B_cell[i] = 0.5 * (B_face[i-1/2] + B_face[i+1/2])
              = 0.5 * (B_face[i] + B_face[i+1])

    Args:
        staggered: StaggeredBField with face-centred components.

    Returns:
        Tuple (Bx, By, Bz) cell-centred, each shape (nx, ny, nz).
    """
    Bx_f = staggered.Bx_face  # (nx+1, ny, nz)
    By_f = staggered.By_face  # (nx, ny+1, nz)
    Bz_f = staggered.Bz_face  # (nx, ny, nz+1)

    Bx = 0.5 * (Bx_f[:-1, :, :] + Bx_f[1:, :, :])
    By = 0.5 * (By_f[:, :-1, :] + By_f[:, 1:, :])
    Bz = 0.5 * (Bz_f[:, :, :-1] + Bz_f[:, :, 1:])

    return Bx, By, Bz


# ============================================================
# Constrained transport update
# ============================================================

@njit(cache=True)
def _ct_update_kernel(
    Bx_face: np.ndarray,
    By_face: np.ndarray,
    Bz_face: np.ndarray,
    Ex_edge: np.ndarray,
    Ey_edge: np.ndarray,
    Ez_edge: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated CT update kernel.

    Faraday's law on face-centred B using edge-centred E:

        dBx/dt = -(dEz/dy - dEy/dz)   at x-faces (i+1/2, j, k)
        dBy/dt = -(dEx/dz - dEz/dx)   at y-faces (i, j+1/2, k)
        dBz/dt = -(dEy/dx - dEx/dy)   at z-faces (i, j, k+1/2)

    The finite differences are taken between adjacent edges, which
    produces the exact discrete curl and preserves div(B) = 0.

    Args:
        Bx_face: shape (nx+1, ny, nz).
        By_face: shape (nx, ny+1, nz).
        Bz_face: shape (nx, ny, nz+1).
        Ex_edge: shape (nx, ny+1, nz+1).
        Ey_edge: shape (nx+1, ny, nz+1).
        Ez_edge: shape (nx+1, ny+1, nz).
        dx, dy, dz: Grid spacings.
        dt: Timestep.

    Returns:
        Updated (Bx_face, By_face, Bz_face).
    """
    # Shapes
    nxp1 = Bx_face.shape[0]  # nx + 1
    ny_ = Bx_face.shape[1]   # ny
    nz_ = Bx_face.shape[2]   # nz

    nx_ = By_face.shape[0]   # nx
    nyp1 = By_face.shape[1]  # ny + 1

    nzp1 = Bz_face.shape[2]  # nz + 1

    Bx_new = Bx_face.copy()
    By_new = By_face.copy()
    Bz_new = Bz_face.copy()

    # Update Bx at x-faces (i+1/2, j, k)
    # dBx/dt = -(dEz/dy - dEy/dz)
    # Ez lives on z-edges: (nx+1, ny+1, nz) -> dEz/dy at (i+1/2, j, k)
    # Ey lives on y-edges: (nx+1, ny, nz+1) -> dEy/dz at (i+1/2, j, k)
    for i in range(nxp1):
        for j in range(ny_):
            for k in range(nz_):
                dEz_dy = (Ez_edge[i, j + 1, k] - Ez_edge[i, j, k]) / dy
                dEy_dz = (Ey_edge[i, j, k + 1] - Ey_edge[i, j, k]) / dz
                Bx_new[i, j, k] -= dt * (dEz_dy - dEy_dz)

    # Update By at y-faces (i, j+1/2, k)
    # dBy/dt = -(dEx/dz - dEz/dx)
    # Ex lives on x-edges: (nx, ny+1, nz+1) -> dEx/dz at (i, j+1/2, k)
    # Ez lives on z-edges: (nx+1, ny+1, nz) -> dEz/dx at (i, j+1/2, k)
    for i in range(nx_):
        for j in range(nyp1):
            for k in range(nz_):
                dEx_dz = (Ex_edge[i, j, k + 1] - Ex_edge[i, j, k]) / dz
                dEz_dx = (Ez_edge[i + 1, j, k] - Ez_edge[i, j, k]) / dx
                By_new[i, j, k] -= dt * (dEx_dz - dEz_dx)

    # Update Bz at z-faces (i, j, k+1/2)
    # dBz/dt = -(dEy/dx - dEx/dy)
    # Ey lives on y-edges: (nx+1, ny, nz+1) -> dEy/dx at (i, j, k+1/2)
    # Ex lives on x-edges: (nx, ny+1, nz+1) -> dEx/dy at (i, j, k+1/2)
    for i in range(nx_):
        for j in range(ny_):
            for k in range(nzp1):
                dEy_dx = (Ey_edge[i + 1, j, k] - Ey_edge[i, j, k]) / dx
                dEx_dy = (Ex_edge[i, j + 1, k] - Ex_edge[i, j, k]) / dy
                Bz_new[i, j, k] -= dt * (dEy_dx - dEx_dy)

    return Bx_new, By_new, Bz_new


def ct_update(
    staggered: StaggeredBField,
    Ex_edge: np.ndarray,
    Ey_edge: np.ndarray,
    Ez_edge: np.ndarray,
    dt: float,
) -> StaggeredBField:
    """Advance face-centred B by one timestep using constrained transport.

    Applies Faraday's law in discrete form:
        B^{n+1} = B^n - dt * curl(E)

    where curl is the exact discrete curl from edges to faces.  This
    preserves div(B) = 0 to machine precision.

    Args:
        staggered: Current face-centred B-field.
        Ex_edge: x-edge EMF, shape (nx, ny+1, nz+1).
        Ey_edge: y-edge EMF, shape (nx+1, ny, nz+1).
        Ez_edge: z-edge EMF, shape (nx+1, ny+1, nz).
        dt: Timestep [s].

    Returns:
        Updated StaggeredBField.
    """
    Bx_new, By_new, Bz_new = _ct_update_kernel(
        staggered.Bx_face,
        staggered.By_face,
        staggered.Bz_face,
        Ex_edge,
        Ey_edge,
        Ez_edge,
        staggered.dx,
        staggered.dy,
        staggered.dz,
        dt,
    )

    return StaggeredBField(
        Bx_face=Bx_new,
        By_face=By_new,
        Bz_face=Bz_new,
        dx=staggered.dx,
        dy=staggered.dy,
        dz=staggered.dz,
    )


# ============================================================
# Divergence diagnostic
# ============================================================

@njit(cache=True)
def _compute_div_B_kernel(
    Bx_face: np.ndarray,
    By_face: np.ndarray,
    Bz_face: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Compute cell-centred divergence of face-centred B.

    div(B)[i,j,k] = (Bx[i+1/2] - Bx[i-1/2])/dx
                   + (By[j+1/2] - By[j-1/2])/dy
                   + (Bz[k+1/2] - Bz[k-1/2])/dz

    Args:
        Bx_face: shape (nx+1, ny, nz).
        By_face: shape (nx, ny+1, nz).
        Bz_face: shape (nx, ny, nz+1).
        dx, dy, dz: Grid spacings.

    Returns:
        div(B), shape (nx, ny, nz).
    """
    nx = Bx_face.shape[0] - 1
    ny = By_face.shape[1] - 1
    nz = Bz_face.shape[2] - 1

    div_B = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                div_B[i, j, k] = (
                    (Bx_face[i + 1, j, k] - Bx_face[i, j, k]) / dx
                    + (By_face[i, j + 1, k] - By_face[i, j, k]) / dy
                    + (Bz_face[i, j, k + 1] - Bz_face[i, j, k]) / dz
                )

    return div_B


def compute_div_B(staggered: StaggeredBField) -> np.ndarray:
    """Compute the cell-centred divergence of the face-centred B-field.

    After a CT update, this should be zero to machine precision (~ 1e-15)
    if the initial B was divergence-free.

    Args:
        staggered: Face-centred B-field.

    Returns:
        div(B) array, shape (nx, ny, nz).
    """
    return _compute_div_B_kernel(
        staggered.Bx_face,
        staggered.By_face,
        staggered.Bz_face,
        staggered.dx,
        staggered.dy,
        staggered.dz,
    )


# ============================================================
# EMF from face-centred fluxes (Gardiner & Stone 2005 "simple CT")
# ============================================================

@njit(cache=True)
def _emf_from_fluxes_kernel(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge-centred EMFs from face-centred fluxes via arithmetic averaging.

    For the "simple CT" method (Gardiner & Stone 2005), the edge EMF is
    computed by averaging the two face fluxes that share that edge.

    Face fluxes are defined as:
        Fx[i+1/2, j, k] = -(v x B)_x at x-face -> contributes to Ey, Ez edges
        Fy[i, j+1/2, k] = -(v x B)_y at y-face -> contributes to Ex, Ez edges
        Fz[i, j, k+1/2] = -(v x B)_z at z-face -> contributes to Ex, Ey edges

    In practice, the face flux for B_y at an x-face gives -Ez, etc.
    We take the convention that the input arrays Fx, Fy, Fz are the
    induction-equation fluxes for each B component at the corresponding
    faces.  The EMF is recovered by the averaging relationships.

    For a grid of nx cells, the face arrays have shape:
        Fx: (nx+1, ny, nz)  -- flux of B at x-faces
        Fy: (nx, ny+1, nz)  -- flux of B at y-faces
        Fz: (nx, ny, nz+1)  -- flux of B at z-faces

    Edge EMF shapes:
        Ex_edge: (nx, ny+1, nz+1)
        Ey_edge: (nx+1, ny, nz+1)
        Ez_edge: (nx+1, ny+1, nz)

    The averaging for Ez at edge (i+1/2, j+1/2, k):
        Ez = 0.25 * (Fx_By[i+1/2,j,k] + Fx_By[i+1/2,j+1,k]
                    + Fy_Bx[i,j+1/2,k] + Fy_Bx[i+1,j+1/2,k])

    Here we use the simplified approach: edge EMF = average of adjacent
    face contributions.  The input Fx, Fy, Fz represent the EMF
    contributions at each face and are averaged to edges.

    Args:
        Fx: EMF contribution at x-faces, shape (nx+1, ny, nz).
        Fy: EMF contribution at y-faces, shape (nx, ny+1, nz).
        Fz: EMF contribution at z-faces, shape (nx, ny, nz+1).

    Returns:
        (Ex_edge, Ey_edge, Ez_edge) at cell edges.
    """
    nxp1 = Fx.shape[0]  # nx + 1
    ny_ = Fx.shape[1]   # ny
    nz_ = Fx.shape[2]   # nz

    nx_ = Fy.shape[0]   # nx
    nyp1 = Fy.shape[1]  # ny + 1
    nzp1 = Fz.shape[2]  # nz + 1

    # Ex at x-edges: (nx, ny+1, nz+1)
    Ex_edge = np.zeros((nx_, nyp1, nzp1))
    for i in range(nx_):
        for j in range(nyp1):
            for k in range(nzp1):
                # Simple average of the 4 adjacent face contributions
                if j > 0 and j < nyp1 - 1 and k > 0 and k < nzp1 - 1:
                    Ex_edge[i, j, k] = 0.25 * (
                        Fz[i, j - 1, k] + Fz[i, j, k]
                        + Fy[i, j, k - 1] + Fy[i, j, k]
                    )
                else:
                    # Boundary: use available data
                    jm = max(j - 1, 0)
                    jm = min(jm, ny_ - 1)
                    jp = min(j, ny_ - 1)
                    km = max(k - 1, 0)
                    km = min(km, nz_ - 1)
                    kp = min(k, nz_ - 1)
                    Ex_edge[i, j, k] = 0.25 * (
                        Fz[i, jm, k] + Fz[i, jp, k]
                        + Fy[i, j, km] + Fy[i, j, kp]
                    )

    # Ey at y-edges: (nx+1, ny, nz+1)
    Ey_edge = np.zeros((nxp1, ny_, nzp1))
    for i in range(nxp1):
        for j in range(ny_):
            for k in range(nzp1):
                im = max(i - 1, 0)
                im = min(im, nx_ - 1)
                ip = min(i, nx_ - 1)
                km = max(k - 1, 0)
                km = min(km, nz_ - 1)
                kp = min(k, nz_ - 1)

                if i > 0 and i < nxp1 - 1 and k > 0 and k < nzp1 - 1:
                    Ey_edge[i, j, k] = 0.25 * (
                        Fx[i, j, k - 1] + Fx[i, j, k]
                        + Fz[i - 1, j, k] + Fz[i, j, k]
                    )
                else:
                    Ey_edge[i, j, k] = 0.25 * (
                        Fx[i, j, km] + Fx[i, j, kp]
                        + Fz[im, j, k] + Fz[ip, j, k]
                    )

    # Ez at z-edges: (nx+1, ny+1, nz)
    Ez_edge = np.zeros((nxp1, nyp1, nz_))
    for i in range(nxp1):
        for j in range(nyp1):
            for k in range(nz_):
                im = max(i - 1, 0)
                im = min(im, nx_ - 1)
                ip = min(i, nx_ - 1)
                jm = max(j - 1, 0)
                jm = min(jm, ny_ - 1)
                jp = min(j, ny_ - 1)

                if i > 0 and i < nxp1 - 1 and j > 0 and j < nyp1 - 1:
                    Ez_edge[i, j, k] = 0.25 * (
                        Fy[i - 1, j, k] + Fy[i, j, k]
                        + Fx[i, j - 1, k] + Fx[i, j, k]
                    )
                else:
                    Ez_edge[i, j, k] = 0.25 * (
                        Fy[im, j, k] + Fy[ip, j, k]
                        + Fx[i, jm, k] + Fx[i, jp, k]
                    )

    return Ex_edge, Ey_edge, Ez_edge


def emf_from_fluxes(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge-centred EMFs from face-centred fluxes.

    Uses the "simple CT" arithmetic averaging approach from Gardiner &
    Stone (2005).  The face flux arrays represent the EMF contributions
    (e.g. from -v x B + eta J) evaluated at each cell face.

    Args:
        Fx: EMF contribution at x-faces, shape (nx+1, ny, nz).
        Fy: EMF contribution at y-faces, shape (nx, ny+1, nz).
        Fz: EMF contribution at z-faces, shape (nx, ny, nz+1).
        dx, dy, dz: Grid spacings [m] (not used in averaging, kept for
            interface consistency and potential future upwind weighting).

    Returns:
        (Ex_edge, Ey_edge, Ez_edge) at cell edges with shapes:
            Ex_edge: (nx, ny+1, nz+1)
            Ey_edge: (nx+1, ny, nz+1)
            Ez_edge: (nx+1, ny+1, nz)
    """
    return _emf_from_fluxes_kernel(Fx, Fy, Fz)
