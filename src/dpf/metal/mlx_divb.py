"""Dedner GLM divergence cleaning and Powell 8-wave sources for the MLX solver.

Implements two complementary div(B) control methods:

1. **Dedner GLM** (Mignone & Tzeferacos 2010): Evolves a scalar cleaning
   field psi that propagates and damps divergence errors. Operator-split
   as source terms applied after each RK stage.

2. **Powell 8-wave** (Powell et al. 1999): Adds source terms proportional
   to div(B) to momentum, energy, and induction equations. Not conservative
   but prevents div(B) growth.

References:
    Dedner et al., JCP 175:645-673 (2002) — GLM formulation.
    Mignone & Tzeferacos, JCP 229:5896-5920 (2010) — Optimal damping.
    Powell et al., JCP 154:284-309 (1999) — 8-wave source terms.
"""

from __future__ import annotations

import mlx.core as mx

from dpf.metal.mlx_kernels import (
    IBR,
    IBT,
    IBZ,
    IDN,
    IEN,
    IMR,
    IMT,
    IMZ,
    NVAR,
)
from dpf.metal.mlx_primitives import RHO_FLOOR


def _gradient_1d(f: mx.array, ds: float, axis: int) -> mx.array:
    """Central-difference gradient along one axis.

    Interior cells use 2nd-order central differences.
    Boundary cells use one-sided 1st-order differences.

    Args:
        f: Input array of any shape.
        ds: Grid spacing along axis.
        axis: Axis to differentiate along.

    Returns:
        Gradient array, same shape as f.
    """
    ndim = f.ndim
    n = f.shape[axis]
    if n < 2:
        return mx.zeros_like(f)

    # Build slicers for general axis
    def _sl(start: int, end: int | None) -> tuple:
        s = [slice(None)] * ndim
        s[axis] = slice(start, end)
        return tuple(s)

    inv_2ds = 1.0 / (2.0 * ds)
    inv_ds = 1.0 / ds

    # Central differences for interior
    central = (f[_sl(2, None)] - f[_sl(None, -2)]) * inv_2ds

    # One-sided for boundaries
    left = (f[_sl(1, 2)] - f[_sl(0, 1)]) * inv_ds
    right = (f[_sl(-1, None)] - f[_sl(-2, -1)]) * inv_ds

    return mx.concatenate([left, central, right], axis=axis)


def div_B_cartesian(
    Bx: mx.array,
    By: mx.array,
    Bz: mx.array,
    dx: float,
    dy: float,
    dz: float,
) -> mx.array:
    """Compute div(B) = dBx/dx + dBy/dy + dBz/dz in Cartesian coordinates.

    Args:
        Bx, By, Bz: B-field components, shape (nx, ny, nz).
        dx, dy, dz: Grid spacings.

    Returns:
        div(B) array, shape (nx, ny, nz).
    """
    return _gradient_1d(Bx, dx, axis=0) + _gradient_1d(By, dy, axis=1) + _gradient_1d(Bz, dz, axis=2)


def div_B_cylindrical(
    Br: mx.array,
    Bz: mx.array,
    r_cell: mx.array,
    dr: float,
    dz: float,
) -> mx.array:
    """Compute div(B) = (1/r) d(r Br)/dr + dBz/dz in cylindrical coordinates.

    Args:
        Br, Bz: B-field components, shape (nr, nz).
        r_cell: Cell-center radii, shape (nr,).
        dr, dz: Grid spacings.

    Returns:
        div(B) array, shape (nr, nz).
    """
    r = r_cell[:, None]  # (nr, 1)
    rBr = r * Br
    d_rBr_dr = _gradient_1d(rBr, dr, axis=0)
    inv_r = 1.0 / mx.maximum(mx.abs(r), 1e-30)
    dBz_dz = _gradient_1d(Bz, dz, axis=1)
    return inv_r * d_rBr_dr + dBz_dz


def dedner_source(
    psi: mx.array,
    U: mx.array,
    ch: float,
    cr: float,
    grid: object,
    coordinates: str = "cartesian",
) -> tuple[mx.array, mx.array]:
    """Dedner GLM divergence cleaning (Mignone & Tzeferacos 2010).

    Computes operator-split corrections:
        dpsi/dt = -ch^2 * div(B) - cr * psi
        dB/dt  += -grad(psi)

    Args:
        psi: Cleaning scalar, shape matching spatial dims of U.
        U: Conserved state array.
        ch: Hyperbolic cleaning speed [m/s].
        cr: Damping rate [1/s]. M&T2010 optimal: ch / dx.
        grid: Grid object (CartesianGrid or CylindricalGrid).
        coordinates: "cartesian" or "cylindrical".

    Returns:
        (dpsi_dt, dU_correction): psi source and conserved state correction.
    """
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    if coordinates == "cartesian":
        divB = div_B_cartesian(Br, Bz, Bt, grid.dx, grid.dy, grid.dz)
        dpsi_dx = _gradient_1d(psi, grid.dx, axis=0)
        dpsi_dy = _gradient_1d(psi, grid.dy, axis=1)
        dpsi_dz = _gradient_1d(psi, grid.dz, axis=2)
        dBr = -dpsi_dx
        dBz = -dpsi_dy
        dBt = -dpsi_dz
    else:
        divB = div_B_cylindrical(Br, Bz, grid.r_cell, grid.dr, grid.dz)
        dpsi_dr = _gradient_1d(psi, grid.dr, axis=0)
        dpsi_dz = _gradient_1d(psi, grid.dz, axis=1)
        dBr = -dpsi_dr
        dBz = -dpsi_dz
        dBt = mx.zeros_like(Bt)

    dpsi_dt = -(ch * ch) * divB - cr * psi

    # Build conserved state correction (only B components affected)
    dU = mx.zeros_like(U)
    rows = [mx.zeros_like(Br)] * NVAR
    rows[IBR] = dBr
    rows[IBZ] = dBz
    rows[IBT] = dBt
    dU = mx.stack(rows, axis=0)

    return dpsi_dt, dU


def powell_source(
    U: mx.array,
    gamma: float,
    grid: object,
    coordinates: str = "cartesian",
) -> mx.array:
    """Powell 8-wave div(B) source terms.

    S_Powell = -div(B) * [0, B, v.B, v, 0, 0]^T

    Not conservative — prevents div(B) growth via restoring force.

    Args:
        U: Conserved state array.
        gamma: Adiabatic index.
        grid: Grid object.
        coordinates: "cartesian" or "cylindrical".

    Returns:
        Source array, same shape as U.
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    if coordinates == "cartesian":
        divB = div_B_cartesian(Br, Bz, Bt, grid.dx, grid.dy, grid.dz)
    else:
        divB = div_B_cylindrical(Br, Bz, grid.r_cell, grid.dr, grid.dz)

    v_dot_B = vr * Br + vz * Bz + vt * Bt

    rows = [mx.zeros_like(rho)] * NVAR
    rows[IMR] = -divB * Br
    rows[IMZ] = -divB * Bz
    rows[IMT] = -divB * Bt
    rows[IEN] = -divB * v_dot_B
    rows[IBR] = -divB * vr
    rows[IBZ] = -divB * vz
    rows[IBT] = -divB * vt

    return mx.stack(rows, axis=0)
