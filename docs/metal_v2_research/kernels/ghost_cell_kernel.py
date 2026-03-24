"""Kernel 1: Ghost Cell Padding with Electrode Boundary Conditions.

MLX Metal kernel that pads a (10, nr, nz) state array with ng=3 ghost cells
on the radial dimension, applying electrode BCs:
  - Inner ghosts (axis, r→0): reflecting (vr=0, Br=0, Btheta=0, sign flip)
  - Outer ghosts (cathode): Btheta = mu0*I/(2*pi*r), zero-gradient, vr=0, Br=0
  - Axial: reflecting at z=0, outflow at z=L (handled in a second pass)

State layout (10 conserved variables):
  0: rho          (density)
  1: rho*vr       (radial momentum)
  2: rho*vz       (axial momentum)
  3: rho*vtheta   (azimuthal momentum)
  4: E            (total energy)
  5: S*rho        (entropy tracer * density)
  6: Br           (radial B)
  7: Bz           (axial B)
  8: Btheta       (azimuthal B)
  9: e_electron   (electron energy density)
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

NVAR = 10
NG = 3
MU0 = 4.0 * math.pi * 1e-7

# Variable indices
IDN = 0
IMR = 1   # radial momentum
IMZ = 2   # axial momentum
IMT = 3   # theta momentum
IEN = 4
ISR = 5   # entropy tracer
IBR = 6
IBZ = 7
IBT = 8
IEE = 9

# ============================================================
# MSL Kernel Source
# ============================================================

_GHOST_HEADER = """
#include <metal_stdlib>
using namespace metal;

constant float MU0 = 1.2566370614359173e-6f;  // 4*pi*1e-7
constant int NG = 3;
constant int NVAR = 10;

// Variable indices
constant int IDN = 0;
constant int IMR = 1;
constant int IMZ = 2;
constant int IMT = 3;
constant int IEN = 4;
constant int ISR = 5;
constant int IBR = 6;
constant int IBZ = 7;
constant int IBT = 8;
constant int IEE = 9;
"""

_GHOST_SOURCE = """
    // Grid: thread (r_out, z_out) over the full padded array
    uint r_out = thread_position_in_grid.x;
    uint z_out = thread_position_in_grid.y;

    // Input dimensions
    uint nr = state_shape[1];
    uint nz = state_shape[2];

    // Output dimensions
    uint nr_g = nr + 2 * NG;
    uint nz_out = nz;  // no axial padding in this kernel

    if (r_out >= nr_g || z_out >= nz_out) return;

    // Strides for (NVAR, nr, nz) layout
    uint in_stride_var = nr * nz;
    uint out_stride_var = nr_g * nz_out;

    // Determine which region this thread handles
    int r_interior = (int)r_out - NG;  // maps to interior index

    for (int v = 0; v < NVAR; v++) {
        float val = 0.0f;

        if (r_interior >= 0 && r_interior < (int)nr) {
            // Interior cell: direct copy
            val = state[v * in_stride_var + r_interior * nz + z_out];
        } else if (r_interior < 0) {
            // Inner ghost (axis side): reflecting BC
            // Mirror index: ghost at r_interior = -1 maps to interior 0,
            //                       r_interior = -2 maps to interior 1, etc.
            int mirror = -r_interior - 1;
            if (mirror >= (int)nr) mirror = (int)nr - 1;  // clamp

            val = state[v * in_stride_var + mirror * nz + z_out];

            // Sign flip for radial quantities at axis
            if (v == IMR || v == IBR || v == IBT || v == IMT) {
                val = -val;
            }
        } else {
            // Outer ghost (cathode side): zero-gradient
            int src = (int)nr - 1;
            val = state[v * in_stride_var + src * nz + z_out];

            // Conducting wall: vr = 0, Br = 0
            if (v == IMR || v == IBR) {
                val = 0.0f;
            }

            // Btheta from circuit current: mu0 * I / (2 * pi * r)
            if (v == IBT) {
                float current = params[0];
                if (metal::abs(current) > 1.0e-10f) {
                    // r coordinate of this ghost cell
                    float r_inner = params[1];
                    float dr = params[2];
                    float r_pos = r_inner + ((float)r_out - (float)NG + 0.5f) * dr;
                    r_pos = metal::max(r_pos, 1.0e-10f);
                    val = MU0 * current / (2.0f * M_PI_F * r_pos);
                }
            }
        }

        padded[v * out_stride_var + r_out * nz_out + z_out] = val;
    }
"""


def _build_ghost_kernel():
    """Build and cache the ghost cell padding kernel."""
    return mx.fast.metal_kernel(
        name="ghost_cell_pad",
        input_names=["state", "params"],
        output_names=["padded"],
        source=_GHOST_SOURCE,
        header=_GHOST_HEADER,
        ensure_row_contiguous=True,
    )


_ghost_kernel = None


def ghost_cell_pad_mlx(
    state: mx.array,
    current: float,
    r_inner: float,
    dr: float,
    ng: int = NG,
) -> mx.array:
    """Pad state with ghost cells and electrode BCs using Metal kernel.

    Args:
        state: Conserved state array, shape (10, nr, nz), float32.
        current: Circuit current [A].
        r_inner: Inner boundary radius [m].
        dr: Radial cell spacing [m].
        ng: Number of ghost cells (default 3).

    Returns:
        Padded array, shape (10, nr + 2*ng, nz), float32.
    """
    global _ghost_kernel
    if _ghost_kernel is None:
        _ghost_kernel = _build_ghost_kernel()

    nvar, nr, nz = state.shape
    nr_g = nr + 2 * ng

    params = mx.array([current, r_inner, dr], dtype=mx.float32)

    # Thread group sizing for M3 Pro: 32x8 = 256 threads/group
    tg_r = min(32, nr_g)
    tg_z = min(8, nz)
    grid_r = ((nr_g + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    outputs = _ghost_kernel(
        inputs=[state, params],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, nr_g, nz)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ============================================================
# NumPy Reference Implementation
# ============================================================


def ghost_cell_pad_numpy(
    state: np.ndarray,
    current: float,
    r_inner: float,
    dr: float,
    ng: int = NG,
) -> np.ndarray:
    """Reference NumPy implementation of ghost cell padding.

    Args:
        state: Conserved state array, shape (10, nr, nz), float32.
        current: Circuit current [A].
        r_inner: Inner boundary radius [m].
        dr: Radial cell spacing [m].
        ng: Number of ghost cells (default 3).

    Returns:
        Padded array, shape (10, nr + 2*ng, nz), float32.
    """
    nvar, nr, nz = state.shape
    nr_g = nr + 2 * ng
    padded = np.zeros((nvar, nr_g, nz), dtype=np.float32)

    # Copy interior
    padded[:, ng:ng + nr, :] = state

    # Inner ghosts (axis side): reflecting BC
    for ig in range(ng):
        mirror = ng - 1 - ig  # ghost at ig maps to interior mirror
        src_idx = min(mirror, nr - 1)
        padded[:, ig, :] = state[:, src_idx, :]
        # Sign flip for radial and theta quantities
        for v in [IMR, IBR, IBT, IMT]:
            padded[v, ig, :] = -padded[v, ig, :]

    # Outer ghosts (cathode side): zero-gradient + electrode BC
    for ig in range(ng):
        out_idx = ng + nr + ig
        padded[:, out_idx, :] = state[:, nr - 1, :]
        # Conducting wall
        padded[IMR, out_idx, :] = 0.0
        padded[IBR, out_idx, :] = 0.0

        # Btheta from circuit current
        if abs(current) > 1e-10:
            r_pos = r_inner + (out_idx - ng + 0.5) * dr
            r_pos = max(r_pos, 1e-10)
            padded[IBT, out_idx, :] = MU0 * current / (2.0 * math.pi * r_pos)

    return padded
