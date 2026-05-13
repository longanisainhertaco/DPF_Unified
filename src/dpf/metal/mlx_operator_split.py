"""Operator-split transport operators for the MLX MHD solver.

Extracted from MLXMHDSolver (God Class decomposition). Contains:
  - Resistive diffusion (implicit Thomas + RKL2 super-timestepping)
  - Braginskii parallel thermal conduction (implicit + RKL2)
  - Braginskii parallel viscosity

All functions are standalone (no class dependency) and operate on
MLX arrays with explicit grid/gamma parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dpf.metal.constants import K_B, MU_0
from dpf.metal.mlx_device import require_mlx


def do_resistive_diffusion(
    U: Any, dt: float, eta: Any, grid: Any, gamma: float,
) -> Any:
    """Implicit resistive diffusion of the B-field with Ohmic heating.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    dt : float
        Timestep [s].
    eta : float or mx.array
        Resistivity [Ohm*m].
    grid : object
        Grid with .dr, .dz, .r_cell attributes.
    gamma : float
        Adiabatic index.

    Returns
    -------
    mx.array
        Updated U with diffused B and Ohmic pressure increment.
    """
    mx = require_mlx()
    from dpf.metal.mlx_kernels import IBR, IBT, IBZ, IEN, ISR, NVAR
    from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
    from dpf.metal.mlx_transport import apply_resistive_diffusion

    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)

    Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
        Br=Br, Bz=Bz, Bt=Bt,
        rho=rho, p=p,
        eta=eta, dt=dt,
        dr=grid.dr, dz=grid.dz,
        r_cell=grid.r_cell,
        gamma=gamma,
    )

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME_new = 0.5 * (Br_new * Br_new + Bz_new * Bz_new + Bt_new * Bt_new)
    E_new = mx.maximum(p_new, P_FLOOR) / gm1 + KE + ME_new
    Srho_new = mx.maximum(p_new, P_FLOOR) * mx.power(
        mx.maximum(rho, 1e-30), 1.0 - gamma
    )

    rows = [U[i] for i in range(NVAR)]
    rows[IBR] = Br_new
    rows[IBZ] = Bz_new
    rows[IBT] = Bt_new
    rows[IEN] = E_new
    rows[ISR] = Srho_new
    return mx.stack(rows, axis=0)


def do_resistive_diffusion_rkl2(
    U: Any, dt: float, eta: Any, grid: Any, gamma: float, coordinates: str,
) -> Any:
    """RKL2 super-timestepped resistive diffusion -- fully on GPU.

    Replaces the Thomas CPU solver with explicit RKL2 stages on Metal.
    ~7x faster on 32x64 grids, ~15x faster on 64x128.
    """
    mx = require_mlx()
    from dpf.metal.mlx_kernels import IBR, IBT, IBZ, IEN, ISR, NVAR
    from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
    from dpf.metal.mlx_sts import compute_sts_stages, rkl2_step_mlx
    from dpf.metal.mlx_sts_operators import (
        compute_parabolic_dt,
        resistive_diffusion_rhs,
    )

    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)

    if isinstance(eta, (int, float)):
        alpha = mx.full(Br.shape, float(eta) / MU_0, dtype=Br.dtype)
    else:
        alpha = eta / MU_0

    dt_para = compute_parabolic_dt(alpha, grid.dr, grid.dz)
    s = compute_sts_stages(dt, dt_para)
    is_cyl = coordinates == "cylindrical"

    def _rhs_B(B_comp):
        return resistive_diffusion_rhs(
            B_comp, alpha, grid.dr, grid.dz,
            grid.r_cell, cylindrical=is_cyl,
        )

    Br_new = rkl2_step_mlx(Br, _rhs_B, dt, s_stages=s)
    Bz_new = rkl2_step_mlx(Bz, _rhs_B, dt, s_stages=s)
    Bt_new = rkl2_step_mlx(Bt, _rhs_B, dt, s_stages=s)

    B2_old = Br * Br + Bz * Bz + Bt * Bt
    B2_new = Br_new * Br_new + Bz_new * Bz_new + Bt_new * Bt_new
    Q_ohmic = mx.maximum(0.5 * (B2_old - B2_new), 0.0)
    p_new = mx.maximum(p + (gamma - 1.0) * Q_ohmic, P_FLOOR)

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME_new = 0.5 * B2_new
    E_new = mx.maximum(p_new, P_FLOOR) / gm1 + KE + ME_new
    Srho_new = mx.maximum(p_new, P_FLOOR) * mx.power(
        mx.maximum(rho, 1e-30), 1.0 - gamma
    )

    rows = [U[i] for i in range(NVAR)]
    rows[IBR] = Br_new
    rows[IBZ] = Bz_new
    rows[IBT] = Bt_new
    rows[IEN] = E_new
    rows[ISR] = Srho_new
    return mx.stack(rows, axis=0)


def do_braginskii_viscosity(
    U: Any, dt: float, grid: Any, gamma: float, ion_mass: float,
    coordinates: str,
) -> Any:
    """Operator-split Braginskii parallel viscosity."""
    from dpf.metal.mlx_viscosity import apply_braginskii_viscosity

    return apply_braginskii_viscosity(
        U, dt, grid, gamma, ion_mass,
        coordinates=coordinates,
    )


def do_thermal_conduction(
    U: Any, dt: float, kappa: float | Any, grid: Any, gamma: float,
    ion_mass: float, coordinates: str,
) -> Any:
    """Implicit Braginskii parallel conduction along z.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    dt : float
        Timestep [s].
    kappa : float or mx.array
        Parallel conductivity [W/(m*K)].
    grid : object
        Grid with .dr, .dz, .r_cell attributes.
    gamma : float
        Adiabatic index.
    ion_mass : float
        Ion mass [kg].
    coordinates : str
        "cylindrical" or "cartesian".

    Returns
    -------
    mx.array
        Updated U with thermally diffused pressure and energy.
    """
    mx = require_mlx()
    from dpf.metal.mlx_kernels import IEN, ISR, NVAR
    from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
    from dpf.metal.mlx_transport import apply_thermal_conduction, flux_limit_kappa

    nr = U.shape[1]
    nz = U.shape[2]
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)

    T = p * ion_mass / (2.0 * mx.maximum(rho, 1e-30) * K_B)

    kappa_np = np.asarray(kappa, dtype=np.float64) if not isinstance(
        kappa, (int, float)
    ) else np.full((nr, nz), float(kappa), dtype=np.float64)
    Te_np = np.asarray(T, dtype=np.float64)
    rho_np = np.asarray(rho, dtype=np.float64)
    kappa_limited = flux_limit_kappa(
        kappa_np, Te_np, rho_np, grid.dz,
        f_limit=0.1, ion_mass=ion_mass,
    )

    Te_new, Ti_new = apply_thermal_conduction(
        Te=T, Ti=T, rho=rho, B=Bz,
        kappa_parallel=mx.array(kappa_limited.astype(np.float32)),
        dt=dt, dz=grid.dz,
        dr=grid.dr,
        Br=Br, Bz=Bz, Bt=Bt,
        anisotropic=True,
        ion_mass=ion_mass,
    )

    T_avg = 0.5 * (Te_new + Ti_new)
    p_new = mx.maximum(2.0 * rho * K_B * T_avg / ion_mass, P_FLOOR)

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    E_new = p_new / gm1 + KE + ME
    Srho_new = p_new * mx.power(mx.maximum(rho, 1e-30), 1.0 - gamma)

    rows = [U[i] for i in range(NVAR)]
    rows[IEN] = E_new
    rows[ISR] = Srho_new
    return mx.stack(rows, axis=0)


def do_thermal_conduction_rkl2(
    U: Any, dt: float, kappa: float | Any, grid: Any, gamma: float,
    ion_mass: float, coordinates: str,
) -> Any:
    """RKL2 super-timestepped thermal conduction -- fully on GPU."""
    mx = require_mlx()
    from dpf.metal.mlx_kernels import IEN, ISR, NVAR
    from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
    from dpf.metal.mlx_sts import compute_sts_stages, rkl2_step_mlx
    from dpf.metal.mlx_sts_operators import compute_parabolic_dt, thermal_conduction_rhs
    from dpf.metal.mlx_transport import flux_limit_kappa

    nr = U.shape[1]
    nz = U.shape[2]
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)
    T = p * ion_mass / (2.0 * mx.maximum(rho, 1e-30) * K_B)

    kappa_np = np.asarray(kappa, dtype=np.float64) if not isinstance(
        kappa, (int, float)
    ) else np.full((nr, nz), float(kappa), dtype=np.float64)
    kappa_limited = flux_limit_kappa(
        kappa_np, np.asarray(T, dtype=np.float64),
        np.asarray(rho, dtype=np.float64), grid.dz,
        f_limit=0.1, ion_mass=ion_mass,
    )
    kappa_mx = mx.array(kappa_limited.astype(np.float32))

    n_e = mx.maximum(rho / ion_mass, 1e-10)
    chi = kappa_mx / (n_e * K_B)
    dt_para = compute_parabolic_dt(chi, grid.dr, grid.dz)
    s = compute_sts_stages(dt, dt_para)
    is_cyl = coordinates == "cylindrical"

    def _rhs_T(T_field):
        return thermal_conduction_rhs(
            T_field, kappa_mx, rho, grid.dr, grid.dz,
            grid.r_cell, ion_mass=ion_mass, cylindrical=is_cyl,
        )

    T_new = mx.maximum(rkl2_step_mlx(T, _rhs_T, dt, s_stages=s), 1.0)

    p_new = mx.maximum(2.0 * rho * K_B * T_new / ion_mass, P_FLOOR)
    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    E_new = p_new / gm1 + KE + ME
    Srho_new = p_new * mx.power(mx.maximum(rho, 1e-30), 1.0 - gamma)

    rows = [U[i] for i in range(NVAR)]
    rows[IEN] = E_new
    rows[ISR] = Srho_new
    return mx.stack(rows, axis=0)
