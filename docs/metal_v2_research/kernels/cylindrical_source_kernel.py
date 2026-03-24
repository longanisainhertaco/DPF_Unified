"""Kernel 3: Cylindrical Geometric Source Terms.

MLX Metal kernel computing geometric source terms for cylindrical MHD:

  S_r-mom   = (p + B^2/2 - Btheta^2) / r + rho*vtheta^2 / r
  S_theta-mom = -(rho*vr*vtheta - Br*Btheta) / r
  S_Btheta  = -(vr*Btheta - Br*vtheta) / r

All other source components are zero.

At r → 0 (first cell), uses L'Hopital rule: replace p/r with dp/dr.

Input: primitive state (10, nr, nz) + radii r (nr,)
Output: source term array (10, nr, nz)
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

NVAR = 10

# Variable indices (PRIMITIVE for this kernel)
IDN = 0   # rho
IVR = 1   # vr
IVZ = 2   # vz
IVT = 3   # vtheta
IPR = 4   # pressure
ISR = 5   # entropy tracer S
IBR = 6   # Br
IBZ = 7   # Bz
IBT = 8   # Btheta
IEE = 9   # e_electron

# ============================================================
# MSL Kernel Source
# ============================================================

_CYL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

constant float TINY = 1.0e-30f;
constant int NVAR = 10;

// Primitive variable indices
constant int IDN = 0;
constant int IVR = 1;
constant int IVZ = 2;
constant int IVT = 3;
constant int IPR = 4;
constant int ISR = 5;
constant int IBR = 6;
constant int IBZ = 7;
constant int IBT = 8;
constant int IEE = 9;
"""

_CYL_SOURCE = r"""
    uint ir = thread_position_in_grid.x;
    uint iz = thread_position_in_grid.y;

    uint nr = prim_shape[1];
    uint nz = prim_shape[2];

    if (ir >= nr || iz >= nz) return;

    uint stride = nr * nz;
    uint idx = ir * nz + iz;

    // Read primitive state at (ir, iz)
    float rho     = prim[IDN * stride + idx];
    float vr      = prim[IVR * stride + idx];
    float vz      = prim[IVZ * stride + idx];
    float vtheta  = prim[IVT * stride + idx];
    float p       = prim[IPR * stride + idx];
    float Br      = prim[IBR * stride + idx];
    float Bz      = prim[IBZ * stride + idx];
    float Btheta  = prim[IBT * stride + idx];

    float r = r_cell[ir];
    float dr = grid_params[0];

    float B2 = Br*Br + Bz*Bz + Btheta*Btheta;
    float p_tot = p + 0.5f * B2;

    // Default: source / r
    float inv_r = 1.0f / max(r, TINY);

    // r-momentum source: (p + B^2/2 - Btheta^2)/r + rho*vtheta^2/r
    float S_mr = (p_tot - Btheta * Btheta) * inv_r + rho * vtheta * vtheta * inv_r;

    // theta-momentum source: -(rho*vr*vtheta - Br*Btheta) / r
    float S_mt = -(rho * vr * vtheta - Br * Btheta) * inv_r;

    // Btheta source: -(vr*Btheta - Br*vtheta) / r
    float S_Bt = -(vr * Btheta - Br * vtheta) * inv_r;

    // L'Hopital rule at r → 0 (first cell): replace p_tot/r with dp_tot/dr
    if (ir == 0 && nr > 1) {
        // Forward difference for dp/dr
        float p_next = prim[IPR * stride + (ir + 1) * nz + iz];
        float Br_next = prim[IBR * stride + (ir + 1) * nz + iz];
        float Bz_next = prim[IBZ * stride + (ir + 1) * nz + iz];
        float Bt_next = prim[IBT * stride + (ir + 1) * nz + iz];
        float B2_next = Br_next*Br_next + Bz_next*Bz_next + Bt_next*Bt_next;
        float pt_next = p_next + 0.5f * B2_next;

        float dpt_dr = (pt_next - p_tot) / dr;

        // For r-momentum: replace (p_tot - Btheta^2)/r with dpt_dr - dBt2_dr/r → dpt_dr at r=0
        // Also rho*vtheta^2/r → 0 at r=0 (L'Hopital gives d(rho*vtheta^2)/dr at r=0 ≈ 0 for smooth data)
        S_mr = dpt_dr;

        // theta-momentum and Btheta: at r=0, vr=0 and Br=0 (axis symmetry), so S_mt=0, S_Bt=0
        S_mt = 0.0f;
        S_Bt = 0.0f;
    }

    // Write source terms — zero for all except r-mom, theta-mom, Btheta
    src[IDN * stride + idx] = 0.0f;
    src[IVR * stride + idx] = S_mr;    // Will be mapped to conservative momentum index
    src[IVZ * stride + idx] = 0.0f;
    src[IVT * stride + idx] = S_mt;
    src[IPR * stride + idx] = 0.0f;    // Energy: 0 (geometric sources cancel in energy eq)
    src[ISR * stride + idx] = 0.0f;
    src[IBR * stride + idx] = 0.0f;
    src[IBZ * stride + idx] = 0.0f;
    src[IBT * stride + idx] = S_Bt;
    src[IEE * stride + idx] = 0.0f;
"""


def _build_cyl_source_kernel():
    """Build and cache the cylindrical source term kernel."""
    return mx.fast.metal_kernel(
        name="cyl_source",
        input_names=["prim", "r_cell", "grid_params"],
        output_names=["src"],
        source=_CYL_SOURCE,
        header=_CYL_HEADER,
        ensure_row_contiguous=True,
    )


_cyl_kernel = None


def cylindrical_source_mlx(
    prim: mx.array,
    r_cell: mx.array,
    dr: float,
) -> mx.array:
    """Compute cylindrical geometric source terms using Metal kernel.

    Args:
        prim: Primitive state array, shape (10, nr, nz), float32.
            Layout: [rho, vr, vz, vtheta, p, S, Br, Bz, Btheta, e_e]
        r_cell: Cell-center radii, shape (nr,), float32.
        dr: Radial grid spacing [m].

    Returns:
        Source terms, shape (10, nr, nz), float32.
    """
    global _cyl_kernel
    if _cyl_kernel is None:
        _cyl_kernel = _build_cyl_source_kernel()

    nvar, nr, nz = prim.shape
    grid_params = mx.array([dr], dtype=mx.float32)

    tg_r = min(32, nr)
    tg_z = min(8, nz)
    grid_r = ((nr + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    outputs = _cyl_kernel(
        inputs=[prim, r_cell, grid_params],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, nr, nz)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ============================================================
# NumPy Reference Implementation
# ============================================================


def cylindrical_source_numpy(
    prim: np.ndarray,
    r_cell: np.ndarray,
    dr: float,
) -> np.ndarray:
    """Reference NumPy implementation of cylindrical geometric source terms.

    Args:
        prim: Primitive state, shape (10, nr, nz), float32.
        r_cell: Cell-center radii, shape (nr,), float32.
        dr: Radial spacing [m].

    Returns:
        Source terms, shape (10, nr, nz), float32.
    """
    nvar, nr, nz = prim.shape
    src = np.zeros_like(prim)

    rho = prim[IDN]
    vr = prim[IVR]
    vtheta = prim[IVT]
    p = prim[IPR]
    Br = prim[IBR]
    Bz = prim[IBZ]
    Btheta = prim[IBT]

    B2 = Br**2 + Bz**2 + Btheta**2
    p_tot = p + 0.5 * B2

    # r_cell is (nr,), broadcast to (nr, nz)
    r = r_cell[:, np.newaxis]  # (nr, 1)
    inv_r = 1.0 / np.maximum(r, 1e-30)

    # r-momentum: (p + B^2/2 - Btheta^2)/r + rho*vtheta^2/r
    S_mr = (p_tot - Btheta**2) * inv_r + rho * vtheta**2 * inv_r

    # theta-momentum: -(rho*vr*vtheta - Br*Btheta)/r
    S_mt = -(rho * vr * vtheta - Br * Btheta) * inv_r

    # Btheta: -(vr*Btheta - Br*vtheta)/r
    S_Bt = -(vr * Btheta - Br * vtheta) * inv_r

    # L'Hopital at ir=0
    if nr > 1:
        pt_0 = p_tot[0, :]
        pt_1 = p_tot[1, :]
        dpt_dr = (pt_1 - pt_0) / dr
        S_mr[0, :] = dpt_dr
        S_mt[0, :] = 0.0
        S_Bt[0, :] = 0.0

    src[IVR] = S_mr
    src[IVT] = S_mt
    src[IBT] = S_Bt

    return src.astype(np.float32)
