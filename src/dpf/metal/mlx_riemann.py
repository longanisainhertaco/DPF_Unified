"""MHD right-hand side for the MLX solver: -div(F) + S_geom.

Pipeline per dimension:
  1. Reconstruct left/right states at interfaces (WENO5-Z or PLM)
  2. Solve Riemann problem (HLLD Metal kernel)
  3. Compute flux divergence with r-weighted differencing (cylindrical)
  4. Add geometric source terms

WENO5-Z produces n-5 interfaces from n cells (2 ghost cells consumed per side).
PLM produces n-1 interfaces from n cells (1 ghost cell consumed per side).
The interior update region is sized to match — for WENO5-Z the domain shrinks
by 2 on each side; for PLM it shrinks by 1.

References:
    Miyoshi & Kusano (2005), JCP 208:315 — HLLD solver
    Borges et al. (2008), JCP 227:3191 — WENO-Z weights
    Stone et al. (2008), ApJS 178:137 — cylindrical MHD finite volume
"""

from __future__ import annotations

import numpy as np

from dpf.metal.mlx_grid import CylindricalGrid
from dpf.metal.mlx_kernels import (
    IBR,
    IBT,
    IBZ,
    IDN,
    IEE,
    IEN,
    IMR,
    IMT,
    IMZ,
    ISR,
    NVAR,
    cylindrical_source_mlx,
    cylindrical_source_numpy,
    hlld_flux_mlx,
)
from dpf.metal.mlx_primitives import P_FLOOR, RHO_FLOOR, cons_to_prim
from dpf.metal.mlx_reconstruction import reconstruct

try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:  # pragma: no cover
    _HAS_MLX = False

__all__ = ["compute_fluxes", "mhd_rhs"]

# Ghost cells consumed per side by each reconstruction method.
_GHOST = {"weno5z": 2, "plm": 1}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _clamp_reconstructed(UL: mx.array, UR: mx.array) -> tuple[mx.array, mx.array]:
    """Floor density and energy in reconstructed states to prevent negatives."""
    # Density floor
    UL_rho = mx.maximum(UL[IDN : IDN + 1], RHO_FLOOR)
    UR_rho = mx.maximum(UR[IDN : IDN + 1], RHO_FLOOR)

    # Rebuild with floored density
    UL_parts = [
        UL_rho if i == IDN else UL[i : i + 1] for i in range(NVAR)
    ]
    UR_parts = [
        UR_rho if i == IDN else UR[i : i + 1] for i in range(NVAR)
    ]

    # Energy floor (lesson #55 from CLAUDE.md)
    UL_parts[IEN] = mx.maximum(UL_parts[IEN], P_FLOOR)
    UR_parts[IEN] = mx.maximum(UR_parts[IEN], P_FLOOR)

    return mx.concatenate(UL_parts, axis=0), mx.concatenate(UR_parts, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Public: per-dimension flux computation
# ──────────────────────────────────────────────────────────────────────────────


def _hll_flux(
    QL: object,
    QR: object,
    gamma: float,
    dim: int,
) -> object:
    """HLL two-wave Riemann flux via NumPy bridge (float64 for numerical safety).

    Args:
        QL: Left state at interfaces, shape (NVAR, n_ifaces, n_transverse).
        QR: Right state at interfaces, shape (NVAR, n_ifaces, n_transverse).
        gamma: Adiabatic index.
        dim: Normal direction (0=radial, 1=axial).

    Returns:
        Numerical flux, same shape as QL/QR.
    """
    TINY = 1e-20
    QL_np = np.asarray(QL).astype(np.float64)
    QR_np = np.asarray(QR).astype(np.float64)

    if dim == 0:
        im_n, im_t1, im_t2 = IMR, IMZ, IMT
        ib_n, ib_t1, ib_t2 = IBR, IBZ, IBT
    else:
        im_n, im_t1, im_t2 = IMZ, IMR, IMT
        ib_n, ib_t1, ib_t2 = IBZ, IBR, IBT

    rho_L = np.maximum(QL_np[IDN], RHO_FLOOR)
    rho_R = np.maximum(QR_np[IDN], RHO_FLOOR)
    inv_rL = 1.0 / rho_L
    inv_rR = 1.0 / rho_R

    vn_L = QL_np[im_n] * inv_rL
    vn_R = QR_np[im_n] * inv_rR
    Bn_L = QL_np[ib_n]
    Bn_R = QR_np[ib_n]
    Bt1_L = QL_np[ib_t1]
    Bt1_R = QR_np[ib_t1]
    Bt2_L = QL_np[ib_t2]
    Bt2_R = QR_np[ib_t2]

    gm1 = gamma - 1.0
    KE_L = 0.5 * rho_L * ((QL_np[IMR]*inv_rL)**2 + (QL_np[IMZ]*inv_rL)**2 + (QL_np[IMT]*inv_rL)**2)
    KE_R = 0.5 * rho_R * ((QR_np[IMR]*inv_rR)**2 + (QR_np[IMZ]*inv_rR)**2 + (QR_np[IMT]*inv_rR)**2)
    B2_L = QL_np[IBR]**2 + QL_np[IBZ]**2 + QL_np[IBT]**2
    B2_R = QR_np[IBR]**2 + QR_np[IBZ]**2 + QR_np[IBT]**2
    p_L = np.maximum(gm1 * (QL_np[IEN] - KE_L - 0.5*B2_L), P_FLOOR)
    p_R = np.maximum(gm1 * (QR_np[IEN] - KE_R - 0.5*B2_R), P_FLOOR)

    Bt_sq_L = np.maximum(B2_L - Bn_L**2, 0.0)
    Bt_sq_R = np.maximum(B2_R - Bn_R**2, 0.0)
    a_sq_L = np.minimum(gamma*p_L/rho_L, (3e8)**2)
    a_sq_R = np.minimum(gamma*p_R/rho_R, (3e8)**2)
    va_sq_L = np.minimum(B2_L/rho_L, (3e8)**2)
    va_sq_R = np.minimum(B2_R/rho_R, (3e8)**2)
    vat_sq_L = np.minimum(Bt_sq_L/rho_L, (3e8)**2)
    vat_sq_R = np.minimum(Bt_sq_R/rho_R, (3e8)**2)

    cf_L = np.sqrt(np.maximum(0.5*(a_sq_L + va_sq_L + np.sqrt(np.maximum((a_sq_L-va_sq_L)**2 + 4*a_sq_L*vat_sq_L, 0.0))), 0.0))
    cf_R = np.sqrt(np.maximum(0.5*(a_sq_R + va_sq_R + np.sqrt(np.maximum((a_sq_R-va_sq_R)**2 + 4*a_sq_R*vat_sq_R, 0.0))), 0.0))
    SL = np.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = np.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = np.maximum(SR, SL + TINY)

    def _pflux(U_arr, rho, inv_r, vn, vt1, vt2, p, Bn, Bt1, Bt2):
        B2 = Bn**2 + Bt1**2 + Bt2**2
        pt = p + 0.5 * B2
        E = U_arr[IEN]
        vB = vn*Bn + vt1*Bt1 + vt2*Bt2
        F = np.zeros_like(U_arr)
        F[IDN]  = rho * vn
        F[im_n] = rho * vn * vn + pt - Bn * Bn
        F[im_t1]= rho * vn * vt1 - Bn * Bt1
        F[im_t2]= rho * vn * vt2 - Bn * Bt2
        F[IEN]  = (E + pt) * vn - Bn * vB
        F[ISR]  = U_arr[ISR] * vn
        F[ib_n] = 0.0
        F[ib_t1]= vn * Bt1 - vt1 * Bn
        F[ib_t2]= vn * Bt2 - vt2 * Bn
        if U_arr.shape[0] > IEE:
            F[IEE] = U_arr[IEE] * vn
        return F

    vt1_L = QL_np[im_t1] * inv_rL
    vt2_L = QL_np[im_t2] * inv_rL
    vt1_R = QR_np[im_t1] * inv_rR
    vt2_R = QR_np[im_t2] * inv_rR

    FL = _pflux(QL_np, rho_L, inv_rL, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L)
    FR = _pflux(QR_np, rho_R, inv_rR, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R)

    inv_dS = 1.0 / np.maximum(SR - SL, TINY)
    F_hll = (SR*FL - SL*FR + SL*SR*(QR_np - QL_np)) * inv_dS
    F_out = np.where(SL >= 0.0, FL, np.where(SR <= 0.0, FR, F_hll))
    F_out[ib_n] = 0.0

    nans = np.isnan(F_out) | np.isinf(F_out)
    if np.any(nans):
        S_max = np.maximum(np.abs(SL), np.abs(SR))
        F_LF = 0.5*(FL + FR) - 0.5*S_max*(QR_np - QL_np)
        F_out = np.where(nans, F_LF, F_out)

    return mx.array(F_out.astype(np.float32))


def compute_fluxes(
    U: object,
    gamma: float,
    dim: int,
    method: str = "weno5z",
    riemann: str = "hlld",
) -> object:
    """Reconstruct + Riemann solve for one dimension.

    Args:
        U: Conserved state (10, nr, nz).
        gamma: Adiabatic index.
        dim: 0=radial, 1=axial.
        method: "weno5z" or "plm".
        riemann: "hlld" or "hll".

    Returns:
        Numerical flux at interfaces.
        Shape (10, n_ifaces, nz) for dim=0, (10, nr, n_ifaces) for dim=1.
    """
    if riemann not in ("hlld", "hll"):
        raise ValueError(f"Unknown Riemann solver: {riemann!r}. Use 'hlld' or 'hll'.")

    QL, QR = reconstruct(U, dim=dim, method=method)
    QL, QR = _clamp_reconstructed(QL, QR)

    if riemann == "hll":
        # HLL operates on (NVAR, n_ifaces, n_transverse) — same layout as HLLD.
        # For dim=1: reconstruct returns (NVAR, nr, n_ifaces_z); transpose for uniformity.
        if dim == 1:
            QL_t = mx.transpose(QL, axes=[0, 2, 1])
            QR_t = mx.transpose(QR, axes=[0, 2, 1])
            F_t = _hll_flux(QL_t, QR_t, gamma=gamma, dim=1)
            return mx.transpose(F_t, axes=[0, 2, 1])
        return _hll_flux(QL, QR, gamma=gamma, dim=dim)

    # HLLD expects (NVAR, n_ifaces, n_transverse)
    # For dim=0: QL shape (10, nr-5, nz) — already correct
    # For dim=1: QL shape (10, nr, nz-5) — need to reorder axes so iface is axis-1
    if dim == 1:
        # Transpose to (10, nz-5, nr) for the HLLD call, then transpose back
        QL_t = mx.transpose(QL, axes=[0, 2, 1])
        QR_t = mx.transpose(QR, axes=[0, 2, 1])
        F_t = hlld_flux_mlx(QL_t, QR_t, gamma=gamma, dim=1)
        return mx.transpose(F_t, axes=[0, 2, 1])

    return hlld_flux_mlx(QL, QR, gamma=gamma, dim=dim)


# ──────────────────────────────────────────────────────────────────────────────
# Public: full MHD right-hand side
# ──────────────────────────────────────────────────────────────────────────────


def mhd_rhs(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float = 5.0 / 3.0,
    dr: float = 1.0,
    dz: float = 1.0,
    method: str = "weno5z",
    riemann: str = "hlld",
) -> mx.array:
    """Full MHD right-hand side: dU/dt = -div(F) + S_geom.

    Dimension-split flux divergence in cylindrical coordinates::

        dU/dt = -(1/(r*dr)) * (r_{i+1/2}*F_r_{i+1/2} - r_{i-1/2}*F_r_{i-1/2})
              - (F_z_{k+1/2} - F_z_{k-1/2}) / dz
              + S_geom

    The interior update domain is determined by the ghost-cell consumption of
    the chosen reconstruction scheme:
      - WENO5-Z: 2 cells consumed per side → interior is [2:nr-2, 2:nz-2]
      - PLM: 1 cell consumed per side → interior is [1:nr-1, 1:nz-1]

    Args:
        U: Conserved state (10, nr, nz).
        grid: CylindricalGrid with r_cell, r_face, inv_r.
        gamma: Adiabatic index.
        dr: Radial cell spacing [m]. If 1.0, grid.dr is used when available.
        dz: Axial cell spacing [m]. If 1.0, grid.dz is used when available.
        method: Reconstruction method ("weno5z" or "plm").
        riemann: Riemann solver ("hlld").

    Returns:
        dU_dt: Time derivative (10, nr, nz). Boundary cells are zero.
    """
    nr = U.shape[1]
    nz = U.shape[2]

    # Prefer grid spacing attributes over defaults
    dr_eff = float(grid.dr) if hasattr(grid, "dr") and dr == 1.0 else dr
    dz_eff = float(grid.dz) if hasattr(grid, "dz") and dz == 1.0 else dz

    ng = _GHOST.get(method, 2)

    dU_dt = mx.zeros_like(U)

    # ── Radial flux divergence ────────────────────────────────────────────────
    # compute_fluxes returns shape (10, n_ifaces_r, nz)
    # n_ifaces_r = nr - 2*ng for WENO5-Z, nr - 2*(ng-0) for plm actually nr-1
    # For WENO5-Z: n_ifaces = nr - 5, interfaces at positions [ng, ng+1, ..., nr-ng-1]
    # The interface i+1/2 separates cell i and i+1.
    # Interfaces produced cover: i = ng-1 .. nr-ng-1 (i.e. ng interfaces on left side)
    # Interior cells updated: i = ng .. nr-ng-1

    F_r = compute_fluxes(U, gamma=gamma, dim=0, method=method, riemann=riemann)
    n_ifaces_r = F_r.shape[1]  # nr-5 for weno5z, nr-1 for plm

    # For WENO5-Z: the ng interfaces on each side are missing.
    # First interface is between cells ng-1 and ng, so:
    #   F_r[:, 0] = flux at r_{ng-1/2}   (left face of first updated cell ng)
    #   F_r[:, j] = flux at r_{ng+j-1/2}
    #   F_r[:, n_ifaces_r-1] = flux at r_{nr-ng-1/2}  (right face of last updated cell)
    # Updated cells: ir = ng .. nr-ng-1  (count = n_ifaces_r - 1)
    # Number of updated cells = n_ifaces_r - 1

    # r_face has shape (nr+1,): r_face[i] = radius of left face of cell i
    # Face at left boundary of updated cell ng: r_face[ng]
    # Face at right boundary of last updated cell: r_face[nr-ng]
    # We need r_face[ng] .. r_face[nr-ng], total = n_ifaces_r + 1 values (= nr - 2*ng + 1)
    # But for WENO5-Z ng=2: nr-4+1 = nr-3, and n_ifaces_r+1 = nr-5+1 = nr-4 — need ng+1 on each side.
    # Actually for WENO5-Z n_ifaces_r = nr-5, so updated cells count = nr-5-1 = nr-6?
    # No: flux divergence uses consecutive flux differences: dU/dt[j] = -(r[j+1]*F[j+1] - r[j]*F[j]) / (r_c[j]*dr)
    # F has n_ifaces_r faces, covering cells ng-1 to nr-ng in the stencil.
    # F[0] is between cells ng-1 and ng. F[n_ifaces_r-1] is between cells nr-ng-1 and nr-ng.
    # For cell ir (interior), we need F[ir-ng] (left face) and F[ir-ng+1] (right face).
    # Updated cells: ir = ng to nr-ng-1 (inclusive), count = nr - 2*ng.
    # Face indices needed: r_face[ng] to r_face[nr-ng], count = nr-2*ng+1.

    n_updated_r = n_ifaces_r - 1  # number of cells with both left and right flux

    if n_updated_r > 0:
        # Left face radii for each updated cell: r_face[ng], r_face[ng+1], ..., r_face[ng + n_updated_r - 1]
        # Right face radii: r_face[ng+1], ..., r_face[ng + n_updated_r]
        # Use mx.take with integer indices
        left_idx = mx.array(list(range(ng, ng + n_updated_r)), dtype=mx.int32)
        right_idx = mx.array(list(range(ng + 1, ng + n_updated_r + 1)), dtype=mx.int32)
        r_left = mx.take(grid.r_face, left_idx, axis=0)   # shape (n_updated_r,)
        r_right = mx.take(grid.r_face, right_idx, axis=0)

        # r-weighted flux differencing: -(r_R*F_R - r_L*F_L) / (r_c * dr)
        F_L = F_r[:, :n_updated_r, :]   # shape (10, n_updated_r, nz)
        F_R = F_r[:, 1:, :]             # shape (10, n_updated_r, nz)

        # Broadcast r to (1, n_updated_r, 1) for multiply
        r_left_bc = r_left[None, :, None]    # (1, n_updated_r, 1)
        r_right_bc = r_right[None, :, None]

        # Cell-centre radii for updated cells
        r_cell_idx = mx.array(list(range(ng, ng + n_updated_r)), dtype=mx.int32)
        r_cell_upd = mx.take(grid.r_cell, r_cell_idx, axis=0)  # (n_updated_r,)
        r_cell_bc = r_cell_upd[None, :, None]                   # (1, n_updated_r, 1)

        div_Fr = -(r_right_bc * F_R - r_left_bc * F_L) / (r_cell_bc * dr_eff)

        # Write into the interior region of dU_dt
        # Updated cell slice: ir = ng : ng + n_updated_r
        # Build update by scatter (concatenate zero pads)
        pad_shape_l = (NVAR, ng, nz)
        pad_shape_r = (NVAR, nr - ng - n_updated_r, nz)
        pad_l = mx.zeros(pad_shape_l, dtype=U.dtype)
        pad_r = mx.zeros(pad_shape_r, dtype=U.dtype)
        dU_r = mx.concatenate([pad_l, div_Fr, pad_r], axis=1)
        dU_dt = dU_dt + dU_r

    # ── Axial flux divergence ─────────────────────────────────────────────────
    F_z = compute_fluxes(U, gamma=gamma, dim=1, method=method, riemann=riemann)
    n_ifaces_z = F_z.shape[2]
    n_updated_z = n_ifaces_z - 1

    if n_updated_z > 0:
        F_L_z = F_z[:, :, :n_updated_z]   # (10, nr, n_updated_z)
        F_R_z = F_z[:, :, 1:]             # (10, nr, n_updated_z)

        div_Fz = -(F_R_z - F_L_z) / dz_eff

        pad_z_l = mx.zeros((NVAR, nr, ng), dtype=U.dtype)
        pad_z_r = mx.zeros((NVAR, nr, nz - ng - n_updated_z), dtype=U.dtype)
        dU_z = mx.concatenate([pad_z_l, div_Fz, pad_z_r], axis=2)
        dU_dt = dU_dt + dU_z

    # ── Geometric source terms ────────────────────────────────────────────────
    # cylindrical_source_mlx expects primitive state, not conserved.
    # Convert U -> Q (primitive) for the source kernel.
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma=gamma)
    s_specific = U[ISR] / mx.maximum(rho, RHO_FLOOR)
    Q_prim = mx.stack([rho, vr, vz, vt, p, s_specific, Br, Bz, Bt, U[IEE]], axis=0)

    # Use NumPy reference for geometric source when grid has negative r_cell
    # values (ghost-padded grids). The Metal kernel produces incorrect results
    # with negative r because it doesn't use abs(r) consistently.
    r_cell_np = np.asarray(grid.r_cell)
    if np.any(r_cell_np < 0):
        inv_r_np = np.asarray(grid.inv_r) if grid.inv_r is not None else None
        src_np = cylindrical_source_numpy(np.asarray(Q_prim), r_cell_np, inv_r_np, gamma)
        src = mx.array(src_np)
    else:
        src = cylindrical_source_mlx(Q_prim, grid.r_cell, grid.inv_r, gamma)

    # src layout matches primitive (vr/vz/vt accelerations at indices 1,2,3; Bt at IBT).
    # Convert velocity-space sources to conserved (momentum) sources.
    dmr = rho * src[IMR]   # src[IMR] = S_vr (acceleration)
    dmt = rho * src[IMT]
    dBt = src[IBT]

    # Energy source from geometric work: v . F_geom
    dE = vr * dmr + vt * dmt

    src_cons = mx.zeros_like(U)
    # Scatter individual components
    src_list = [
        mx.zeros_like(rho) if i == IDN else
        dmr if i == IMR else
        mx.zeros_like(rho) if i == IMZ else
        dmt if i == IMT else
        dE if i == IEN else
        mx.zeros_like(rho) if i == ISR else
        mx.zeros_like(rho) if i == IBR else
        mx.zeros_like(rho) if i == IBZ else
        dBt if i == IBT else
        mx.zeros_like(rho)
        for i in range(NVAR)
    ]
    src_cons = mx.stack(src_list, axis=0)

    dU_dt = dU_dt + src_cons

    mx.eval(dU_dt)
    return dU_dt
