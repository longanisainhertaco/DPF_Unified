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
    elif dim == 1:
        im_n, im_t1, im_t2 = IMZ, IMR, IMT
        ib_n, ib_t1, ib_t2 = IBZ, IBR, IBT
    else:  # dim == 2 (y-direction in Cartesian)
        im_n, im_t1, im_t2 = IMT, IMR, IMZ
        ib_n, ib_t1, ib_t2 = IBT, IBR, IBZ

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

    # Boris correction: cap Alfven speed at c_boris instead of c_light.
    # v_A'^2 = v_A^2 * c^2 / (v_A^2 + c^2) bounds vacuum wavespeeds
    # without the E-KE-ME cancellation that causes NaN in float32.
    # Gombosi 2002, validated for z-pinch by PERSEUS (Gourdain 2025).
    _C_BORIS_SQ = 5e5**2  # (500 km/s)^2
    a_sq_L = np.minimum(gamma*p_L/rho_L, _C_BORIS_SQ)
    a_sq_R = np.minimum(gamma*p_R/rho_R, _C_BORIS_SQ)
    va_sq_L = B2_L/rho_L
    va_sq_R = B2_R/rho_R
    # Boris: v_A'^2 = v_A^2 * c^2 / (v_A^2 + c^2)
    va_sq_L = va_sq_L * _C_BORIS_SQ / (va_sq_L + _C_BORIS_SQ)
    va_sq_R = va_sq_R * _C_BORIS_SQ / (va_sq_R + _C_BORIS_SQ)
    vat_sq_L = Bt_sq_L/rho_L
    vat_sq_R = Bt_sq_R/rho_R
    vat_sq_L = vat_sq_L * _C_BORIS_SQ / (vat_sq_L + _C_BORIS_SQ)
    vat_sq_R = vat_sq_R * _C_BORIS_SQ / (vat_sq_R + _C_BORIS_SQ)

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

    # Clamp to float32 range before cast to prevent overflow warnings
    F32_MAX = np.float64(np.finfo(np.float32).max)
    F_out = np.clip(F_out, -F32_MAX, F32_MAX)

    return mx.array(F_out.astype(np.float32))


def _hlld_flux_cpu64(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """HLLD Riemann flux in float64 on CPU via NumPy reference.

    Avoids float32 cancellation in HLLD star-state D_L/D_R denominators
    and pressure recovery at extreme electrode B_theta. ~2.5x slower
    than Metal GPU but numerically exact.
    """
    from dpf.metal.mlx_kernels import hlld_flux_numpy

    QL_np = np.asarray(QL)
    QR_np = np.asarray(QR)

    if dim == 1:
        QL_np = np.transpose(QL_np, axes=[0, 2, 1])
        QR_np = np.transpose(QR_np, axes=[0, 2, 1])
        F_np = hlld_flux_numpy(QL_np, QR_np, gamma, dim=1)
        F_np = np.transpose(F_np, axes=[0, 2, 1])
    else:
        F_np = hlld_flux_numpy(QL_np, QR_np, gamma, dim=dim)

    return mx.array(np.clip(F_np, -3.4e38, 3.4e38).astype(np.float32))


def compute_fluxes(
    U: object,
    gamma: float,
    dim: int,
    method: str = "weno5z",
    riemann: str = "hlld",
    precision: str = "float32",
) -> object:
    """Reconstruct + Riemann solve for one dimension.

    Args:
        U: Conserved state. 3D: (10, nr, nz). 4D: (10, nx, ny, nz).
        gamma: Adiabatic index.
        dim: 0=x/r, 1=z (cyl) or y (cart), 2=z (cart only).
        method: "weno5z" or "plm".
        riemann: "hlld" or "hll".
        precision: "float32" (GPU Metal) or "float64" (CPU NumPy).
            float64 avoids cancellation in HLLD star-states at extreme
            electrode B_theta. ~2.5x slower but numerically exact.

    Returns:
        Numerical flux at interfaces, matching U's dimensionality.
    """
    if riemann not in ("hlld", "hll"):
        raise ValueError(f"Unknown Riemann solver: {riemann!r}. Use 'hlld' or 'hll'.")

    QL, QR = reconstruct(U, dim=dim, method=method)
    QL, QR = _clamp_reconstructed(QL, QR)

    is_4d = U.ndim == 4

    if is_4d:
        return _compute_fluxes_4d(QL, QR, gamma, dim, riemann)

    if riemann == "hll":
        if dim == 1:
            QL_t = mx.transpose(QL, axes=[0, 2, 1])
            QR_t = mx.transpose(QR, axes=[0, 2, 1])
            F_t = _hll_flux(QL_t, QR_t, gamma=gamma, dim=1)
            return mx.transpose(F_t, axes=[0, 2, 1])
        return _hll_flux(QL, QR, gamma=gamma, dim=dim)

    # HLLD solver — float64 option uses CPU NumPy for numerical stability
    if precision == "float64":
        return _hlld_flux_cpu64(QL, QR, gamma, dim)

    # HLLD float32 on Metal GPU
    if dim == 1:
        QL_t = mx.transpose(QL, axes=[0, 2, 1])
        QR_t = mx.transpose(QR, axes=[0, 2, 1])
        F_t = hlld_flux_mlx(QL_t, QR_t, gamma=gamma, dim=1)
        return mx.transpose(F_t, axes=[0, 2, 1])

    return hlld_flux_mlx(QL, QR, gamma=gamma, dim=dim)


def _compute_fluxes_4d(
    QL: mx.array,
    QR: mx.array,
    gamma: float,
    dim: int,
    riemann: str,
) -> mx.array:
    """Riemann solve for 4D Cartesian states (NVAR, nx, ny, nz).

    Flattens transverse dimensions into a single axis for the 3D-only
    HLLD/HLL kernels, then reshapes back to 4D.
    """
    # After reconstruction, QL/QR have one axis shrunk by interfaces.
    # dim=0: (NVAR, n_ifaces, ny, nz) → flatten to (NVAR, n_ifaces, ny*nz)
    # dim=1: (NVAR, nx, n_ifaces, nz) → transpose to (NVAR, n_ifaces, nx*nz)
    # dim=2: (NVAR, nx, ny, n_ifaces) → transpose to (NVAR, n_ifaces, nx*ny)
    shape = QL.shape  # (NVAR, d1, d2, d3)

    if dim == 0:
        n_iface = shape[1]
        n_trans = shape[2] * shape[3]
        QL_3d = mx.reshape(QL, (NVAR, n_iface, n_trans))
        QR_3d = mx.reshape(QR, (NVAR, n_iface, n_trans))
        out_shape_4d = (NVAR, n_iface, shape[2], shape[3])
    elif dim == 1:
        n_iface = shape[2]
        n_trans = shape[1] * shape[3]
        QL_3d = mx.reshape(mx.transpose(QL, axes=[0, 2, 1, 3]), (NVAR, n_iface, n_trans))
        QR_3d = mx.reshape(mx.transpose(QR, axes=[0, 2, 1, 3]), (NVAR, n_iface, n_trans))
        out_shape_4d = None  # handled below
    else:  # dim == 2
        n_iface = shape[3]
        n_trans = shape[1] * shape[2]
        QL_3d = mx.reshape(mx.transpose(QL, axes=[0, 3, 1, 2]), (NVAR, n_iface, n_trans))
        QR_3d = mx.reshape(mx.transpose(QR, axes=[0, 3, 1, 2]), (NVAR, n_iface, n_trans))
        out_shape_4d = None

    if riemann == "hll":
        F_3d = _hll_flux(QL_3d, QR_3d, gamma=gamma, dim=dim)
    else:
        F_3d = hlld_flux_mlx(QL_3d, QR_3d, gamma=gamma, dim=dim)

    # Reshape back to 4D
    if dim == 0:
        return mx.reshape(F_3d, out_shape_4d)
    elif dim == 1:
        # F_3d is (NVAR, n_iface, nx*nz) → (NVAR, n_iface, nx, nz) → transpose to (NVAR, nx, n_iface, nz)
        F_4d = mx.reshape(F_3d, (NVAR, n_iface, shape[1], shape[3]))
        return mx.transpose(F_4d, axes=[0, 2, 1, 3])
    else:
        # F_3d is (NVAR, n_iface, nx*ny) → (NVAR, n_iface, nx, ny) → transpose to (NVAR, nx, ny, n_iface)
        F_4d = mx.reshape(F_3d, (NVAR, n_iface, shape[1], shape[2]))
        return mx.transpose(F_4d, axes=[0, 2, 3, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Public: full MHD right-hand side
# ──────────────────────────────────────────────────────────────────────────────


def mhd_rhs(
    U: mx.array,
    grid: object,
    gamma: float = 5.0 / 3.0,
    dr: float = 1.0,
    dz: float = 1.0,
    method: str = "weno5z",
    riemann: str = "hlld",
    precision: str = "float32",
) -> mx.array:
    """Full MHD right-hand side: dU/dt = -div(F) [+ S_geom for cylindrical].

    Supports both cylindrical (3D state, r-weighted flux) and Cartesian
    (3D or 4D state, standard flux divergence in up to 3 dimensions).

    Args:
        U: Conserved state. Cylindrical: (10, nr, nz). Cartesian: (10, nx, ny, nz).
        grid: CylindricalGrid or CartesianGrid.
        gamma: Adiabatic index.
        dr: Cell spacing [m]. If 1.0, grid.dr is used when available.
        dz: Cell spacing [m]. If 1.0, grid.dz is used when available.
        method: Reconstruction method ("weno5z" or "plm").
        riemann: Riemann solver ("hlld" or "hll").
        precision: "float32" or "float64" for Riemann solver.

    Returns:
        dU_dt: Time derivative, same shape as U. Boundary cells are zero.
    """
    is_cartesian = grid.r_cell is None

    if is_cartesian:
        return _mhd_rhs_cartesian(U, grid, gamma, method, riemann, precision)
    return _mhd_rhs_cylindrical(U, grid, gamma, dr, dz, method, riemann, precision)


def _mhd_rhs_cartesian(
    U: mx.array,
    grid: object,
    gamma: float,
    method: str,
    riemann: str,
    precision: str = "float32",
) -> mx.array:
    """Cartesian MHD RHS: dU/dt = -div(F) in up to 3 dimensions.

    No geometric source terms. Standard flux divergence -(F_R - F_L)/dx.
    """
    ng = _GHOST.get(method, 2)
    dU_dt = mx.zeros_like(U)

    is_4d = U.ndim == 4
    # For 4D: shape is (NVAR, nx, ny, nz)
    # For 3D: shape is (NVAR, nx, nz) — degenerate 2D Cartesian

    dims_to_sweep: list[tuple[int, float]] = [(0, grid.dx)]
    if is_4d:
        dims_to_sweep.append((1, grid.dy))
        dims_to_sweep.append((2, grid.dz))
    else:
        dims_to_sweep.append((1, grid.dz))

    for dim, ds in dims_to_sweep:
        axis = dim + 1
        n_along = U.shape[axis]
        if n_along < 2 * ng + 1:
            continue

        F = compute_fluxes(U, gamma=gamma, dim=dim, method=method, riemann=riemann, precision=precision)
        n_ifaces = F.shape[axis]
        n_updated = n_ifaces - 1
        if n_updated <= 0:
            continue

        # Standard flux divergence: -(F_R - F_L) / ds
        F_L = _slice_axis(F, axis, 0, n_updated)
        F_R = _slice_axis(F, axis, 1, n_updated)
        div_F = -(F_R - F_L) / ds

        # Pad to full domain size
        dU_dt = dU_dt + _pad_to_full(div_F, U.shape, axis, ng)

    return dU_dt


def _slice_axis(arr: mx.array, axis: int, start: int, length: int) -> mx.array:
    """Slice `length` elements starting at `start` along `axis`."""
    end = start + length
    if axis == 1:
        return arr[:, start:end]
    if axis == 2:
        return arr[:, :, start:end]
    if axis == 3:
        return arr[:, :, :, start:end]
    return arr[start:end]


def _pad_to_full(
    interior: mx.array,
    full_shape: tuple,
    axis: int,
    ng: int,
) -> mx.array:
    """Zero-pad interior array to match full_shape along axis."""
    n_full = full_shape[axis]
    n_interior = interior.shape[axis]
    n_right = n_full - ng - n_interior

    pad_l_shape = list(full_shape)
    pad_l_shape[axis] = ng
    pad_r_shape = list(full_shape)
    pad_r_shape[axis] = n_right

    pad_l = mx.zeros(pad_l_shape, dtype=interior.dtype)
    pad_r = mx.zeros(pad_r_shape, dtype=interior.dtype)
    return mx.concatenate([pad_l, interior, pad_r], axis=axis)


def _mhd_rhs_cylindrical(
    U: mx.array,
    grid: object,
    gamma: float,
    dr: float,
    dz: float,
    method: str,
    riemann: str,
    precision: str = "float32",
) -> mx.array:
    """Cylindrical MHD RHS with r-weighted flux divergence + geometric sources."""
    nr = U.shape[1]
    nz = U.shape[2]

    dr_eff = float(grid.dr) if hasattr(grid, "dr") and dr == 1.0 else dr
    dz_eff = float(grid.dz) if hasattr(grid, "dz") and dz == 1.0 else dz

    ng = _GHOST.get(method, 2)

    dU_dt = mx.zeros_like(U)

    # ── Radial flux divergence (r-weighted) ───────────────────────────────────
    F_r = compute_fluxes(U, gamma=gamma, dim=0, method=method, riemann=riemann, precision=precision)
    n_ifaces_r = F_r.shape[1]
    n_updated_r = n_ifaces_r - 1

    if n_updated_r > 0:
        left_idx = mx.array(list(range(ng, ng + n_updated_r)), dtype=mx.int32)
        right_idx = mx.array(list(range(ng + 1, ng + n_updated_r + 1)), dtype=mx.int32)
        r_left = mx.take(grid.r_face, left_idx, axis=0)
        r_right = mx.take(grid.r_face, right_idx, axis=0)

        F_L = F_r[:, :n_updated_r, :]
        F_R = F_r[:, 1:, :]

        r_left_bc = r_left[None, :, None]
        r_right_bc = r_right[None, :, None]

        r_cell_idx = mx.array(list(range(ng, ng + n_updated_r)), dtype=mx.int32)
        r_cell_upd = mx.take(grid.r_cell, r_cell_idx, axis=0)
        r_cell_bc = r_cell_upd[None, :, None]

        div_Fr = -(r_right_bc * F_R - r_left_bc * F_L) / (r_cell_bc * dr_eff)

        pad_shape_l = (NVAR, ng, nz)
        pad_shape_r = (NVAR, nr - ng - n_updated_r, nz)
        pad_l = mx.zeros(pad_shape_l, dtype=U.dtype)
        pad_r = mx.zeros(pad_shape_r, dtype=U.dtype)
        dU_r = mx.concatenate([pad_l, div_Fr, pad_r], axis=1)
        dU_dt = dU_dt + dU_r

    # ── Axial flux divergence ─────────────────────────────────────────────────
    F_z = compute_fluxes(U, gamma=gamma, dim=1, method=method, riemann=riemann, precision=precision)
    n_ifaces_z = F_z.shape[2]
    n_updated_z = n_ifaces_z - 1

    if n_updated_z > 0:
        F_L_z = F_z[:, :, :n_updated_z]
        F_R_z = F_z[:, :, 1:]

        div_Fz = -(F_R_z - F_L_z) / dz_eff

        pad_z_l = mx.zeros((NVAR, nr, ng), dtype=U.dtype)
        pad_z_r = mx.zeros((NVAR, nr, nz - ng - n_updated_z), dtype=U.dtype)
        dU_z = mx.concatenate([pad_z_l, div_Fz, pad_z_r], axis=2)
        dU_dt = dU_dt + dU_z

    # ── Geometric source terms ────────────────────────────────────────────────
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma=gamma)
    s_specific = U[ISR] / mx.maximum(rho, RHO_FLOOR)
    Q_prim = mx.stack([rho, vr, vz, vt, p, s_specific, Br, Bz, Bt, U[IEE]], axis=0)

    r_cell_for_src = grid.r_cell
    inv_r_for_src = grid.inv_r
    r_cell_np = np.asarray(grid.r_cell)
    if np.any(r_cell_np < 0):
        r_cell_for_src = mx.abs(grid.r_cell)
        inv_r_for_src = mx.where(
            mx.abs(grid.r_cell) < 0.5 * grid.dr,
            2.0 / grid.dr,
            1.0 / mx.maximum(mx.abs(grid.r_cell), 1e-30),
        )
    src = cylindrical_source_mlx(Q_prim, r_cell_for_src, inv_r_for_src, gamma)

    dmr = rho * src[IMR]
    dmt = rho * src[IMT]
    dBt = src[IBT]
    dE = vr * dmr + vt * dmt

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

    return dU_dt
