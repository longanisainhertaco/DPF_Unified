"""MLX constrained transport for the Phase B cylindrical MHD solver.

Maintains div(B) = 0 to machine precision for the axisymmetric (r, z) case.
Only Br (r-face) and Bz (z-face) need CT; Btheta is cell-centred and carries
no divergence contribution in axisymmetry.

Grid conventions (2D, ny=1 implicit):
    Cell-centred scalars:  shape (nr, nz)
    Br on r-faces:         shape (nr+1, nz)   -- Br[i, k] lives at r_{i-1/2}
    Bz on z-faces:         shape (nr, nz+1)   -- Bz[i, k] lives at z_{k-1/2}
    E_theta (EMF corners): shape (nr+1, nz+1) -- corner (i+1/2, k+1/2)

The CT update equations (Gardiner & Stone 2005, §2.3) in cylindrical geometry:

    dBr/dt = -dE_theta/dz
    dBz/dt = (1/r) * d(r * E_theta)/dr

where E_theta = -(vr * Bz - vz * Br) is the azimuthal EMF.

References:
    Gardiner T.A. & Stone J.M., JCP 205, 509 (2005).
    Evans C.R. & Hawley J.F., ApJ 332, 659 (1988).
"""

from __future__ import annotations

from dpf.metal.mlx_device import require_mlx

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_emf(
    vr: object,
    vz: object,
    Br_face: object,
    Bz_face: object,
    dr: float,
    dz: float,
) -> object:
    """Compute corner-centred EMF E_theta = -(vr*Bz - vz*Br).

    Velocities and B-fields are cell-centred; this function averages them to
    the four corners (i+1/2, k+1/2) of the staggered grid using arithmetic
    means with boundary replication at edges.

    Parameters
    ----------
    vr : mx.array
        Radial velocity, shape (nr, nz), float32.
    vz : mx.array
        Axial velocity, shape (nr, nz), float32.
    Br_face : mx.array
        Radial B-field on r-faces, shape (nr+1, nz), float32.
    Bz_face : mx.array
        Axial B-field on z-faces, shape (nr, nz+1), float32.
    dr : float
        Radial cell spacing [m] (unused here; reserved for upwind extension).
    dz : float
        Axial cell spacing [m] (unused here; reserved for upwind extension).

    Returns
    -------
    mx.array
        Corner EMF, shape (nr+1, nz+1), float32.
    """
    mx = require_mlx()
    nr, nz = vr.shape

    # --- Average vr to r-faces: shape (nr+1, nz) ---
    # Pad vr along dim=0 by replicating boundary rows, then average neighbours.
    vr_pad = mx.concatenate([vr[:1, :], vr, vr[-1:, :]], axis=0)     # (nr+2, nz)
    vr_rface = 0.5 * (vr_pad[:-1, :] + vr_pad[1:, :])                # (nr+1, nz)

    # --- Average vr to corners: shape (nr+1, nz+1) ---
    vr_pad_z = mx.concatenate([vr_rface[:, :1], vr_rface, vr_rface[:, -1:]], axis=1)  # (nr+1, nz+2)
    vr_corner = 0.5 * (vr_pad_z[:, :-1] + vr_pad_z[:, 1:])           # (nr+1, nz+1)

    # --- Average vz to z-faces: shape (nr, nz+1) ---
    vz_pad = mx.concatenate([vz[:, :1], vz, vz[:, -1:]], axis=1)      # (nr, nz+2)
    vz_zface = 0.5 * (vz_pad[:, :-1] + vz_pad[:, 1:])                 # (nr, nz+1)

    # --- Average vz to corners: shape (nr+1, nz+1) ---
    vz_pad_r = mx.concatenate([vz_zface[:1, :], vz_zface, vz_zface[-1:, :]], axis=0)  # (nr+2, nz+1)
    vz_corner = 0.5 * (vz_pad_r[:-1, :] + vz_pad_r[1:, :])           # (nr+1, nz+1)

    # --- Average Bz_face (nr, nz+1) to corners: shape (nr+1, nz+1) ---
    Bz_pad_r = mx.concatenate([Bz_face[:1, :], Bz_face, Bz_face[-1:, :]], axis=0)    # (nr+2, nz+1)
    Bz_corner = 0.5 * (Bz_pad_r[:-1, :] + Bz_pad_r[1:, :])           # (nr+1, nz+1)

    # --- Average Br_face (nr+1, nz) to corners: shape (nr+1, nz+1) ---
    Br_pad_z = mx.concatenate([Br_face[:, :1], Br_face, Br_face[:, -1:]], axis=1)    # (nr+1, nz+2)
    Br_corner = 0.5 * (Br_pad_z[:, :-1] + Br_pad_z[:, 1:])           # (nr+1, nz+1)

    # --- EMF: E_theta = -(vr * Bz - vz * Br) ---
    return -(vr_corner * Bz_corner - vz_corner * Br_corner)


def apply_ct(
    Br_face: object,
    Bz_face: object,
    emf: object,
    dt: float,
    dr: float,
    dz: float,
    r_cell: object,
    r_face: object,
) -> tuple[object, object]:
    """Update face-centred B fields via constrained transport.

    Applies Faraday's law in finite-volume form on the staggered grid:

        Br_new[i, k] = Br_old[i, k] - (dt/dz) * (emf[i, k+1] - emf[i, k])
        Bz_new[i, k] = Bz_old[i, k]
                       + (dt / (r_cell[i] * dr))
                         * (r_face[i+1] * emf[i+1, k] - r_face[i] * emf[i, k])

    The r-weighted Bz update preserves the cylindrical divergence constraint:
        (1/r) * d(r*Br)/dr + dBz/dz = 0

    Parameters
    ----------
    Br_face : mx.array
        Radial B on r-faces, shape (nr+1, nz), float32.
    Bz_face : mx.array
        Axial B on z-faces, shape (nr, nz+1), float32.
    emf : mx.array
        Corner EMF E_theta, shape (nr+1, nz+1), float32.
    dt : float
        Timestep [s].
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    r_cell : mx.array
        Cell-centre radii, shape (nr,) or (nr, 1), float32.
    r_face : mx.array
        Face radii r_{i+1/2}, shape (nr+1,) or (nr+1, 1), float32.

    Returns
    -------
    tuple[mx.array, mx.array]
        (Br_new, Bz_new) updated face-centred B fields.
    """
    mx = require_mlx()

    # Ensure 1D geometry arrays broadcast correctly over (nr, nz) fields.
    # r_cell: (nr,) -> (nr, 1) for safe broadcasting against (nr, nz+1)
    if r_cell.ndim == 1:
        r_cell_col = r_cell[:, None]                    # (nr, 1)
    else:
        r_cell_col = r_cell

    if r_face.ndim == 1:
        r_face_col = r_face[:, None]                    # (nr+1, 1)
    else:
        r_face_col = r_face

    # --- Br update: dBr/dt = -dE_theta/dz ---
    # emf shape (nr+1, nz+1); difference along axis=1 -> (nr+1, nz)
    dE_dz = (emf[:, 1:] - emf[:, :-1]) / dz            # (nr+1, nz)
    Br_new = Br_face - dt * dE_dz

    # --- Bz update: dBz/dt = (1/r) * d(r * E_theta)/dr ---
    # emf shape (nr+1, nz+1); r_face_col shape (nr+1, 1) -> broadcast to (nr+1, nz+1)
    r_emf = r_face_col * emf                            # (nr+1, nz+1)
    d_r_emf_dr = (r_emf[1:, :] - r_emf[:-1, :]) / dr  # (nr, nz+1)
    inv_r = 1.0 / mx.maximum(r_cell_col, mx.array(1e-30, dtype=mx.float32))
    Bz_new = Bz_face + dt * inv_r * d_r_emf_dr

    return Br_new, Bz_new


def div_B_cylindrical(
    Br_face: object,
    Bz_face: object,
    dr: float,
    dz: float,
    r_cell: object,
    r_face: object,
) -> object:
    """Compute cell-centred cylindrical divergence of face-centred B.

    div(B) = (1/r) * d(r*Br)/dr + dBz/dz

    After a CT update this should be zero to float32 machine precision.

    Parameters
    ----------
    Br_face : mx.array
        Radial B on r-faces, shape (nr+1, nz), float32.
    Bz_face : mx.array
        Axial B on z-faces, shape (nr, nz+1), float32.
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    r_cell : mx.array
        Cell-centre radii, shape (nr,) or (nr, 1), float32.
    r_face : mx.array
        Face radii, shape (nr+1,) or (nr+1, 1), float32.

    Returns
    -------
    mx.array
        Cell-centred divergence, shape (nr, nz), float32.
    """
    mx = require_mlx()

    if r_cell.ndim == 1:
        r_cell_col = r_cell[:, None]    # (nr, 1)
    else:
        r_cell_col = r_cell

    if r_face.ndim == 1:
        r_face_col = r_face[:, None]    # (nr+1, 1)
    else:
        r_face_col = r_face

    # Radial term: (1/r) * d(r*Br)/dr
    r_Br = r_face_col * Br_face                         # (nr+1, nz)
    d_rBr_dr = (r_Br[1:, :] - r_Br[:-1, :]) / dr      # (nr, nz)
    inv_r = 1.0 / mx.maximum(r_cell_col, mx.array(1e-30, dtype=mx.float32))
    radial_term = inv_r * d_rBr_dr

    # Axial term: dBz/dz
    dBz_dz = (Bz_face[:, 1:] - Bz_face[:, :-1]) / dz  # (nr, nz)

    return radial_term + dBz_dz
