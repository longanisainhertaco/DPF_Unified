"""Source terms for the MLX cylindrical MHD solver.

Wraps the Metal geometric source kernel and adds:
1. Cylindrical geometric sources (centrifugal, hoop stress, Coriolis)
2. Ohmic heating: Q_ohm = eta * J^2 (adds to energy and entropy)
3. Bremsstrahlung radiation: Q_rad = 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te)
4. Entropy tracer source: dSrho/dt from irreversible heating
"""

from __future__ import annotations

import mlx.core as mx
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
    cylindrical_source_mlx,
    cylindrical_source_numpy,
)

# Physical constants
_MU0 = 4.0 * 3.141592653589793 * 1e-7  # permeability of free space [H/m]
_KBOLTZ = 1.380649e-23                  # Boltzmann constant [J/K]
_BREM_COEFF = 1.42e-40                  # bremsstrahlung prefactor [W m^3 / sqrt(K)]

# Numerical floors
_RHO_FLOOR = 1.0e-12
_P_FLOOR = 1.0e-12


def _conserved_to_primitive(U: mx.array, gamma: float) -> mx.array:
    """Convert conserved state U to primitive state Q for cylindrical source call.

    Conserved layout: [rho, mr, mz, mt, E, Srho, Br, Bz, Bt, e_electron]
    Primitive layout: [rho, vr, vz, vtheta, p, Srho/rho, Br, Bz, Bt, e_electron]

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        gamma: Adiabatic index.

    Returns:
        Primitive state Q, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    v2 = vr * vr + vz * vz + vt * vt
    B2 = U[IBR] * U[IBR] + U[IBZ] * U[IBZ] + U[IBT] * U[IBT]

    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)
    s_specific = U[ISR] * inv_rho

    Q = mx.stack([rho, vr, vz, vt, p, s_specific, U[IBR], U[IBZ], U[IBT], U[IEE]], axis=0)
    return Q.astype(mx.float32)


def apply_geometric_sources(
    U: mx.array,
    r_cell: mx.array,
    inv_r: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
    use_metal_kernel: bool = True,
) -> mx.array:
    """Apply cylindrical geometric source terms to conserved state.

    Converts to primitive state, calls the Metal geometric source kernel,
    then maps source increments back to conserved variables.

    Source terms (applied to momentum only):
      S_mr = (rho*vtheta^2 - Btheta^2) / r + dp_tot/dr    [centrifugal + hoop]
      S_mt = -2*(rho*vr*vtheta - Br*Btheta) / r            [Coriolis + tension]

    L'Hopital at axis (r=0): uses dp/dr instead of p/r.

    Args:
        U: Conserved state array, shape (NVAR, nr, nz), float32.
        r_cell: Cell-center radii, shape (nr,), float32.
        inv_r: Pre-computed 1/r with L'Hopital at axis, shape (nr,), float32.
        dt: Time step [s].
        gamma: Adiabatic index.
        use_metal_kernel: Use Metal GPU kernel if available (default True).

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    Q = _conserved_to_primitive(U, gamma)

    if use_metal_kernel:
        src = cylindrical_source_mlx(Q, r_cell, inv_r, gamma)
    else:
        src_np = cylindrical_source_numpy(
            np.asarray(Q), np.asarray(r_cell), np.asarray(inv_r), gamma
        )
        src = mx.array(src_np)

    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    # src[1]=S_vr, src[2]=S_vz, src[3]=S_vt are accelerations (per unit mass not momentum)
    # cylindrical_source_numpy returns primitive increments: S_mr in velocity units
    dmr = rho * src[1] * dt
    dmz = rho * src[2] * dt
    dmt = rho * src[3] * dt

    # Work done by geometric forces contributes to total energy
    dE = vr * dmr + vz * dmz + vt * dmt

    updated_vars = [
        U[IDN],
        U[IMR] + dmr,
        U[IMZ] + dmz,
        U[IMT] + dmt,
        U[IEN] + dE,
        U[ISR],
        U[IBR] + src[IBR] * dt,
        U[IBZ],
        U[IBT] + src[IBT] * dt,
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)


def compute_current_density(
    U: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
) -> mx.array:
    """Compute J = curl(B) in cylindrical coordinates and return |J|^2.

    Uses central finite differences on interior cells; forward/backward
    differences at boundaries.

    J_r = -dBt/dz
    J_z = (1/r) * d(r*Bt)/dr
    J_theta = dBr/dz - dBz/dr

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        dr: Radial cell spacing [m].
        dz: Axial cell spacing [m].
        r_cell: Cell-center radii, shape (nr,), float32.

    Returns:
        J_sq = Jr^2 + Jz^2 + Jt^2, shape (nr, nz), float32.
    """
    Br = U[IBR]   # (nr, nz)
    Bz = U[IBZ]   # (nr, nz)
    Bt = U[IBT]   # (nr, nz)

    # dBt/dz — central diff in z, one-sided at boundaries
    dBt_dz = (mx.roll(Bt, -1, axis=1) - mx.roll(Bt, 1, axis=1)) / (2.0 * dz)
    # Fix boundary z=0 (forward) and z=nz-1 (backward) — still order 1 there
    dBt_dz = mx.where(
        mx.arange(Bt.shape[1]) == 0,
        (mx.roll(Bt, -1, axis=1) - Bt) / dz,
        dBt_dz,
    )
    dBt_dz = mx.where(
        mx.arange(Bt.shape[1]) == Bt.shape[1] - 1,
        (Bt - mx.roll(Bt, 1, axis=1)) / dz,
        dBt_dz,
    )

    Jr = -dBt_dz

    # d(r*Bt)/dr — radial derivative of r*Bt
    r = r_cell[:, None]          # (nr, 1) broadcast
    rBt = r * Bt                 # (nr, nz)
    drBt_dr = (mx.roll(rBt, -1, axis=0) - mx.roll(rBt, 1, axis=0)) / (2.0 * dr)
    drBt_dr = mx.where(
        mx.arange(rBt.shape[0])[:, None] == 0,
        (mx.roll(rBt, -1, axis=0) - rBt) / dr,
        drBt_dr,
    )
    drBt_dr = mx.where(
        mx.arange(rBt.shape[0])[:, None] == rBt.shape[0] - 1,
        (rBt - mx.roll(rBt, 1, axis=0)) / dr,
        drBt_dr,
    )
    inv_r = 1.0 / mx.maximum(r, 1e-30)
    Jz = inv_r * drBt_dr

    # dBr/dz
    dBr_dz = (mx.roll(Br, -1, axis=1) - mx.roll(Br, 1, axis=1)) / (2.0 * dz)
    dBr_dz = mx.where(
        mx.arange(Br.shape[1]) == 0,
        (mx.roll(Br, -1, axis=1) - Br) / dz,
        dBr_dz,
    )
    dBr_dz = mx.where(
        mx.arange(Br.shape[1]) == Br.shape[1] - 1,
        (Br - mx.roll(Br, 1, axis=1)) / dz,
        dBr_dz,
    )

    # dBz/dr
    dBz_dr = (mx.roll(Bz, -1, axis=0) - mx.roll(Bz, 1, axis=0)) / (2.0 * dr)
    dBz_dr = mx.where(
        mx.arange(Bz.shape[0])[:, None] == 0,
        (mx.roll(Bz, -1, axis=0) - Bz) / dr,
        dBz_dr,
    )
    dBz_dr = mx.where(
        mx.arange(Bz.shape[0])[:, None] == Bz.shape[0] - 1,
        (Bz - mx.roll(Bz, 1, axis=0)) / dr,
        dBz_dr,
    )

    Jt = dBr_dz - dBz_dr

    J_sq = Jr * Jr + Jz * Jz + Jt * Jt
    return J_sq.astype(mx.float32)


def apply_ohmic_heating(
    U: mx.array,
    eta: mx.array | float,
    J_sq: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
) -> mx.array:
    """Add ohmic heating eta*J^2 to total energy and entropy tracer.

    Q_ohm = eta * J^2 [W/m^3]

    Updates both U[IEN] and U[ISR] consistently. The entropy tracer
    tracks cumulative irreversible heating: dSrho = Q_ohm * dt / T,
    where T is estimated from current pressure and density.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        eta: Resistivity [Ohm·m], scalar or shape (nr, nz).
        J_sq: |J|^2 [A^2/m^4], shape (nr, nz), float32.
        dt: Time step [s].
        gamma: Adiabatic index.

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    v2 = (U[IMR] ** 2 + U[IMZ] ** 2 + U[IMT] ** 2) * inv_rho * inv_rho
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)

    if not isinstance(eta, mx.array):
        eta = mx.array(float(eta), dtype=mx.float32)

    Q_ohm = eta * J_sq          # (nr, nz)
    dE = Q_ohm * dt             # total energy increment

    # Entropy tracer: dSrho = Q_ohm * dt * (gamma-1) / p  (dimensionless tracer)
    inv_p = 1.0 / mx.maximum(p, _P_FLOOR)
    dSrho = Q_ohm * dt * (gamma - 1.0) * inv_p * rho

    updated_vars = [
        U[IDN],
        U[IMR],
        U[IMZ],
        U[IMT],
        U[IEN] + dE,
        U[ISR] + dSrho,
        U[IBR],
        U[IBZ],
        U[IBT],
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)


def apply_bremsstrahlung(
    U: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
    Z_eff: float = 1.0,
    gaunt_factor: float = 1.2,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Remove bremsstrahlung radiation from total energy.

    Q_rad = 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te) [W/m^3]

    Assumes fully ionized hydrogen-like plasma: ne = rho / ion_mass.
    Te derived from electron pressure component via p = ne * kB * Te.

    Applied as energy sink: U[IEN] -= Q_rad * dt.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        dt: Time step [s].
        gamma: Adiabatic index.
        Z_eff: Effective ion charge (default 1.0 for deuterium).
        gaunt_factor: Free-free Gaunt factor (default 1.2).
        ion_mass: Ion mass [kg] (default: deuterium 3.34e-27).

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    v2 = (U[IMR] ** 2 + U[IMZ] ** 2 + U[IMT] ** 2) * inv_rho * inv_rho
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)

    # Compute Q_rad in float64 via NumPy: 1.42e-40 is subnormal in float32 and
    # would flush to zero if left in the MLX float32 graph.
    rho_np = np.asarray(rho).astype(np.float64)
    p_np = np.asarray(p).astype(np.float64)
    ne_np = rho_np / ion_mass
    Te_np = np.maximum(p_np * ion_mass / (rho_np * _KBOLTZ), 1.0)
    Q_rad_np = (_BREM_COEFF * gaunt_factor * Z_eff * ne_np * ne_np * np.sqrt(Te_np)).astype(
        np.float32
    )
    Q_rad = mx.array(Q_rad_np)
    dE = Q_rad * dt

    # Clamp: cannot remove more energy than available above the kinetic+magnetic floor
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_thermal_floor = _P_FLOOR / (gamma - 1.0)
    e_available = mx.maximum(U[IEN] - e_kin - e_mag - e_thermal_floor, 0.0)
    dE = mx.minimum(dE, e_available)
    dE = mx.maximum(dE, 0.0)

    updated_vars = [
        U[IDN],
        U[IMR],
        U[IMZ],
        U[IMT],
        U[IEN] - dE,
        U[ISR],
        U[IBR],
        U[IBZ],
        U[IBT],
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)
