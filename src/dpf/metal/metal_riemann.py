"""Fully vectorized HLL/HLLD Riemann solver and PLM/WENO5 reconstruction on PyTorch MPS.

Production physics code for ideal MHD.  The HLL/HLLD solvers conserve mass,
momentum (3 components), total energy, and magnetic flux (3 components) --
8 conservative variables per cell.

Conservative variable ordering::

    U = [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]

where E_total = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2.

All tensors are float32 on the MPS device.  There are NO Python loops over
grid cells -- every operation is a vectorized tensor op suitable for GPU
execution on Apple Metal.

Reconstruction uses Piecewise Linear Method (PLM, 2nd-order) or WENO5
(5th-order, Borges WENO-Z variant).  Riemann solver is HLL (2-wave) or
HLLD (4-wave, Miyoshi & Kusano 2005).

References:
    Harten A., Lax P.D., van Leer B., SIAM Rev. 25, 35 (1983).
    Davis S.F., SIAM J. Sci. Stat. Comp. 9, 445 (1988).
    Miyoshi T. & Kusano K., JCP 208, 315 (2005)  -- HLLD, wave speed estimates.
    Borges R. et al., JCP 227, 3191-3211 (2008)  -- WENO-Z.
    Stone J.M. et al., ApJS 249, 4 (2020)  -- Athena++ methods paper.

Functions:
    plm_reconstruct_mps           -- PLM reconstruction with slope limiters.
    weno5_reconstruct_mps         -- WENO5-Z reconstruction.
    hll_flux_mps                  -- HLL approximate Riemann solver (8-component).
    hlld_flux_mps                 -- HLLD 4-wave Riemann solver (8-component).
    compute_fluxes_mps            -- Full reconstruction + Riemann solve for one dim.
    mhd_rhs_mps                   -- Full MHD right-hand side: -div(F) in 3D.
    mhd_rhs_cylindrical_mps       -- Cylindrical MHD RHS with r-weighted differencing.
"""

from __future__ import annotations

import torch

# ============================================================
# Re-export all public API from submodules (backward compat)
# ============================================================
from dpf.metal._riemann_constants import (  # noqa: F401
    IB1,
    IB2,
    IB3,
    IDN,
    IEE,
    IEN,
    IM1,
    IM2,
    IM3,
    ISR,
    NVAR,
    P_FLOOR,
    RHO_FLOOR,
)
from dpf.metal._riemann_nan_safety import (  # noqa: F401
    _should_check_nan,
    advance_nan_step_count,
    get_repair_stats,
    reset_repair_stats,
    set_nan_safety_level,
)
from dpf.metal._riemann_primitives import (  # noqa: F401
    _cons_to_prim_mps,
    _fast_magnetosonic_mps,
    _physical_flux_mps,
    _prim_to_cons_mps,
)
from dpf.metal._riemann_reconstruction import (  # noqa: F401
    _mc_limiter,
    _minmod,
    _positivity_fallback,
    _weno5_left_biased,
    plm_reconstruct_mps,
    weno5_reconstruct_mps,
)
from dpf.metal._riemann_solvers import hll_flux_mps, hlld_flux_mps  # noqa: F401
from dpf.metal._utils import _check_no_nan, _ensure_mps  # noqa: F401

# ============================================================
# Flux computation: PLM/WENO5 + HLL/HLLD for one dimension
# ============================================================


def compute_fluxes_mps(
    state: torch.Tensor,
    gamma: float,
    dx: float,
    dy: float,
    dz: float,
    dim: int,
    limiter: str = "minmod",
    riemann_solver: str = "hll",
    reconstruction: str = "plm",
    reconstruction_precision: str = "float32",
) -> torch.Tensor:
    """Compute numerical flux along one dimension using reconstruction + Riemann solver.

    Pipeline:
        1. Reconstruction of conservative variables at cell interfaces (PLM or WENO5).
        2. Positivity-preserving fallback at troubled interfaces.
        3. Riemann solve (HLL or HLLD) at each interface.

    Args:
        state: Conservative state, shape (8, nx, ny, nz), float32/64.
        gamma: Adiabatic index.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
        dim: Dimension to compute fluxes along (0=x, 1=y, 2=z).
        limiter: Slope limiter for PLM ("minmod" or "mc").
        riemann_solver: Riemann solver to use: "hll" or "hlld".
        reconstruction: Reconstruction method: "plm" (2nd order) or "weno5" (5th order).
        reconstruction_precision: Floating-point precision for PLM slope computation.
            "float16" enables opt-in speedup; "float32" (default) is unchanged.

    Returns:
        Numerical flux at interfaces, shape (8, ...) where the axis
        corresponding to *dim* has reduced entries.
    """
    _ensure_mps(state, "state")

    if reconstruction == "weno5" and state.shape[dim + 1] >= 5:
        UL, UR = weno5_reconstruct_mps(state, dim=dim)
    else:
        UL, UR = plm_reconstruct_mps(
            state, dim=dim, limiter=limiter,
            reconstruction_precision=reconstruction_precision,
        )

    UL, UR = _positivity_fallback(UL, UR, state, gamma, dim)

    if riemann_solver == "hlld":
        flux = hlld_flux_mps(UL, UR, gamma, dim)
    else:
        flux = hll_flux_mps(UL, UR, gamma, dim)

    return flux


# ============================================================
# Full MHD right-hand side: -div(F)
# ============================================================


def mhd_rhs_mps(
    state: dict[str, torch.Tensor],
    gamma: float,
    dx: float,
    dy: float,
    dz: float,
    limiter: str = "minmod",
    riemann_solver: str = "hll",
    reconstruction: str = "plm",
    bc: tuple[str, str, str] = ("outflow", "outflow", "outflow"),
    reconstruction_precision: str = "float32",
) -> dict[str, torch.Tensor]:
    """Compute the full ideal MHD right-hand side dU/dt = -div(F).

    Applies dimension-split flux differencing in all three directions using
    the chosen reconstruction (PLM or WENO5) and Riemann solver (HLL or HLLD).

    Args:
        state: Dictionary of MPS tensors with keys:
            'rho':      Density, shape (nx, ny, nz).
            'velocity': Velocity, shape (3, nx, ny, nz).
            'pressure': Thermal pressure, shape (nx, ny, nz).
            'B':        Magnetic field, shape (3, nx, ny, nz).
        gamma: Adiabatic index.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
        limiter: Slope limiter for PLM ("minmod" or "mc").
        riemann_solver: Riemann solver: "hll" or "hlld".
        reconstruction: Reconstruction method: "plm" (2nd order) or
            "weno5" (5th order).
        reconstruction_precision: Floating-point precision for PLM slope
            computation.  "float16" enables opt-in speedup; "float32"
            (default) is unchanged.

    Returns:
        Dictionary with time derivatives of the state:
            'rho':      d(rho)/dt, shape (nx, ny, nz).
            'velocity': d(velocity)/dt, shape (3, nx, ny, nz).
            'pressure': d(pressure)/dt, shape (nx, ny, nz).
            'B':        d(B)/dt, shape (3, nx, ny, nz).
    """
    rho = state["rho"]
    vel = state["velocity"]
    p = state["pressure"]
    B = state["B"]
    e_electron = state.get("e_electron")
    s_rho = state.get("s_rho")

    _ensure_mps(rho, "rho")
    _ensure_mps(vel, "velocity")
    _ensure_mps(p, "pressure")
    _ensure_mps(B, "B")

    U = _prim_to_cons_mps(rho, vel, p, B, gamma, e_electron=e_electron)

    if s_rho is not None:
        _ensure_mps(s_rho, "s_rho")
        s_rho_row = s_rho.unsqueeze(0)
        U = torch.cat([U, s_rho_row], dim=0)

    dU_dt = torch.zeros_like(U)

    dh = [dx, dy, dz]

    for dim_idx in range(3):
        n_dim = U.shape[dim_idx + 1]

        if n_dim < 2:
            continue

        axis = dim_idx + 1
        dim_bc = bc[dim_idx] if dim_idx < len(bc) else "outflow"

        if dim_bc == "periodic":
            gh = 3 if (reconstruction == "weno5" and n_dim >= 5) else 2
            pad_spec_p = [0, 0, 0, 0, 0, 0]
            pad_idx_p = 2 * (3 - axis)
            pad_spec_p[pad_idx_p] = gh
            pad_spec_p[pad_idx_p + 1] = gh
            U_padded = torch.nn.functional.pad(U, pad_spec_p, mode="circular")

            flux = compute_fluxes_mps(
                U_padded, gamma, dx, dy, dz, dim_idx,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            F_right = torch.narrow(flux, axis, gh, n_dim)
            F_left = torch.narrow(flux, axis, gh - 1, n_dim)

        else:
            flux = compute_fluxes_mps(
                U, gamma, dx, dy, dz, dim_idx,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            pad_spec = [0, 0, 0, 0, 0, 0]
            pad_idx = 2 * (3 - axis)
            pad_spec[pad_idx] = 1
            pad_spec[pad_idx + 1] = 1

            flux_padded = torch.nn.functional.pad(
                flux, pad_spec, mode="replicate",
            )
            F_right = torch.narrow(flux_padded, axis, 1, n_dim)
            F_left = torch.narrow(flux_padded, axis, 0, n_dim)

        dU_dt = dU_dt - (F_right - F_left) / dh[dim_idx]

    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    inv_rho = 1.0 / rho_safe

    drho_dt = dU_dt[IDN]

    dvx_dt = (dU_dt[IM1] - vel[0] * drho_dt) * inv_rho
    dvy_dt = (dU_dt[IM2] - vel[1] * drho_dt) * inv_rho
    dvz_dt = (dU_dt[IM3] - vel[2] * drho_dt) * inv_rho
    dvel_dt = torch.stack([dvx_dt, dvy_dt, dvz_dt], dim=0)

    dBx_dt = dU_dt[IB1]
    dBy_dt = dU_dt[IB2]
    dBz_dt = dU_dt[IB3]
    dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], dim=0)

    v_dot_dmom = (vel[0] * dU_dt[IM1] + vel[1] * dU_dt[IM2] + vel[2] * dU_dt[IM3])
    v_sq = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
    B_dot_dB = B[0] * dBx_dt + B[1] * dBy_dt + B[2] * dBz_dt

    dp_dt = (gamma - 1.0) * (
        dU_dt[IEN] - v_dot_dmom + 0.5 * v_sq * drho_dt - B_dot_dB
    )

    result: dict[str, torch.Tensor] = {
        "rho": drho_dt,
        "velocity": dvel_dt,
        "pressure": dp_dt,
        "B": dB_dt,
    }

    if e_electron is not None:
        result["e_electron"] = dU_dt[IEE]

    if s_rho is not None:
        result["s_rho"] = dU_dt[ISR]

    return result


# ============================================================
# Cylindrical MHD right-hand side: r-weighted -div(F) + geometric sources
# ============================================================


def mhd_rhs_cylindrical_mps(
    state: dict[str, torch.Tensor],
    gamma: float,
    dx: float,
    dy: float,
    dz: float,
    r_cell: torch.Tensor,
    r_face: torch.Tensor,
    limiter: str = "minmod",
    riemann_solver: str = "hll",
    reconstruction: str = "plm",
    bc: tuple[str, str, str] = ("outflow", "outflow", "outflow"),
    reconstruction_precision: str = "float32",
) -> dict[str, torch.Tensor]:
    """Cylindrical MHD right-hand side using r-weighted finite-volume differencing.

    Implements the conservative cylindrical MHD update:

        dU/dt = -(1/(r*dr)) * (r_{i+1/2}*F_{i+1/2} - r_{i-1/2}*F_{i-1/2})
              - (G_{j+1/2} - G_{j-1/2}) / dz
              + S_geom

    where axis=0 is radial (r), axis=1 is azimuthal (theta, axisymmetric
    so ny=1), and axis=2 is axial (z).

    Geometric source terms added after flux divergence (Heaviside-Lorentz units):

        r-momentum:  +(rho*vtheta^2 - Btheta^2) / r   (centrifugal + hoop stress)
        theta-momentum: -2*(rho*vr*vtheta - Br*Btheta) / r   (Coriolis + magnetic tension)

    References:
        Stone & Norman, ApJS 80:753 (1992) -- ZEUS-2D cylindrical.
        Mignone et al., ApJS 170:228 (2007) -- PLUTO code.
        Stone et al., ApJS 249:4 (2020) -- Athena++ methods.

    Args:
        state: Dictionary of MPS tensors with keys:
            'rho':      Density, shape (nx, ny, nz).
            'velocity': Velocity, shape (3, nx, ny, nz).
            'pressure': Thermal pressure, shape (nx, ny, nz).
            'B':        Magnetic field, shape (3, nx, ny, nz).
        gamma: Adiabatic index.
        dx: Radial grid spacing [m].
        dy: Azimuthal grid spacing [m] (unused for axisymmetric; kept for API compat).
        dz: Axial grid spacing [m].
        r_cell: Cell-centre radii, shape (nx, 1, 1), float32, MPS.
        r_face: Face radii at r_{i+1/2}, shape (nx+1, 1, 1), float32, MPS.
        limiter: Slope limiter for PLM ("minmod" or "mc").
        riemann_solver: Riemann solver: "hll" or "hlld".
        reconstruction: Reconstruction method: "plm" or "weno5".
        bc: Boundary condition per dimension.
        reconstruction_precision: Floating-point precision for PLM slope
            computation.

    Returns:
        Dictionary with time derivatives of the state:
            'rho':      d(rho)/dt, shape (nx, ny, nz).
            'velocity': d(velocity)/dt, shape (3, nx, ny, nz).
            'pressure': d(pressure)/dt, shape (nx, ny, nz).
            'B':        d(B)/dt, shape (3, nx, ny, nz).
    """
    rho = state["rho"]
    vel = state["velocity"]
    p = state["pressure"]
    B = state["B"]
    e_electron = state.get("e_electron")
    s_rho = state.get("s_rho")

    _ensure_mps(rho, "rho")
    _ensure_mps(vel, "velocity")
    _ensure_mps(p, "pressure")
    _ensure_mps(B, "B")
    _ensure_mps(r_cell, "r_cell")
    _ensure_mps(r_face, "r_face")

    nx, ny, nz = rho.shape

    U = _prim_to_cons_mps(rho, vel, p, B, gamma, e_electron=e_electron)

    if s_rho is not None:
        _ensure_mps(s_rho, "s_rho")
        s_rho_row = s_rho.unsqueeze(0)
        U = torch.cat([U, s_rho_row], dim=0)

    dU_dt = torch.zeros_like(U)

    # ---- Radial dimension (dim=0): r-weighted flux differencing ----
    if nx >= 2:
        axis = 1
        dim_bc = bc[0] if len(bc) > 0 else "outflow"

        if dim_bc == "periodic":
            gh = 3 if (reconstruction == "weno5" and nx >= 5) else 2
            pad_spec_p = [0, 0, 0, 0, gh, gh]
            U_padded = torch.nn.functional.pad(U, pad_spec_p, mode="circular")
            flux_r = compute_fluxes_mps(
                U_padded, gamma, dx, dy, dz, 0,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            F_right = torch.narrow(flux_r, axis, gh, nx)
            F_left = torch.narrow(flux_r, axis, gh - 1, nx)
            r_right = r_face[gh:gh + nx]
            r_left = r_face[gh - 1:gh - 1 + nx]
        elif dim_bc == "axis":
            nvar = U.shape[0]
            sign_mask = torch.ones(nvar, 1, 1, 1, dtype=U.dtype, device=U.device)
            sign_mask[IM1] = -1.0
            sign_mask[IM2] = -1.0
            sign_mask[IB1] = -1.0
            sign_mask[IB2] = -1.0
            ghost_left = sign_mask * U[:, :1, :, :]
            ghost_right = U[:, -1:, :, :]
            U_padded = torch.cat([ghost_left, U, ghost_right], dim=axis)
            flux_r = compute_fluxes_mps(
                U_padded, gamma, dx, dy, dz, 0,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            F_right = torch.narrow(flux_r, axis, 1, nx)
            F_left = torch.narrow(flux_r, axis, 0, nx)
            r_right = r_face[1:]
            r_left = r_face[:-1]
        else:
            flux_r = compute_fluxes_mps(
                U, gamma, dx, dy, dz, 0,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            pad_spec = [0, 0, 0, 0, 1, 1]
            flux_r_padded = torch.nn.functional.pad(flux_r, pad_spec, mode="replicate")
            F_right = torch.narrow(flux_r_padded, axis, 1, nx)
            F_left = torch.narrow(flux_r_padded, axis, 0, nx)
            r_right = r_face[1:]
            r_left = r_face[:-1]

        inv_r_cell = 1.0 / torch.clamp(r_cell, min=1e-30)
        dU_dt = dU_dt - (r_right * F_right - r_left * F_left) * inv_r_cell / dx

    # ---- Azimuthal dimension (dim=1): skip for axisymmetric (ny=1) ----

    # ---- Axial dimension (dim=2): standard Cartesian differencing ----
    if nz >= 2:
        axis = 3
        dim_bc = bc[2] if len(bc) > 2 else "outflow"

        if dim_bc == "periodic":
            gh = 3 if (reconstruction == "weno5" and nz >= 5) else 2
            pad_spec_p = [gh, gh, 0, 0, 0, 0]
            U_padded = torch.nn.functional.pad(U, pad_spec_p, mode="circular")
            flux_z = compute_fluxes_mps(
                U_padded, gamma, dx, dy, dz, 2,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            F_right = torch.narrow(flux_z, axis, gh, nz)
            F_left = torch.narrow(flux_z, axis, gh - 1, nz)
        else:
            flux_z = compute_fluxes_mps(
                U, gamma, dx, dy, dz, 2,
                limiter, riemann_solver, reconstruction,
                reconstruction_precision,
            )
            pad_spec = [1, 1, 0, 0, 0, 0]
            flux_z_padded = torch.nn.functional.pad(flux_z, pad_spec, mode="replicate")
            F_right = torch.narrow(flux_z_padded, axis, 1, nz)
            F_left = torch.narrow(flux_z_padded, axis, 0, nz)

        dU_dt = dU_dt - (F_right - F_left) / dz

    # ---- Convert conservative RHS to primitive RHS ----
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    inv_rho = 1.0 / rho_safe

    drho_dt = dU_dt[IDN]

    dvx_dt = (dU_dt[IM1] - vel[0] * drho_dt) * inv_rho
    dvy_dt = (dU_dt[IM2] - vel[1] * drho_dt) * inv_rho
    dvz_dt = (dU_dt[IM3] - vel[2] * drho_dt) * inv_rho

    dBx_dt = dU_dt[IB1]
    dBy_dt = dU_dt[IB2]
    dBz_dt = dU_dt[IB3]
    dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], dim=0)

    v_dot_dmom = (vel[0] * dU_dt[IM1] + vel[1] * dU_dt[IM2] + vel[2] * dU_dt[IM3])
    v_sq = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
    B_dot_dB = B[0] * dBx_dt + B[1] * dBy_dt + B[2] * dBz_dt
    dp_dt = (gamma - 1.0) * (
        dU_dt[IEN] - v_dot_dmom + 0.5 * v_sq * drho_dt - B_dot_dB
    )

    # ---- Geometric source terms ----
    inv_r = 1.0 / torch.clamp(r_cell, min=1e-30)

    v_r = vel[0]
    v_theta = vel[1]
    B_r = B[0]
    B_theta = B[1]

    S_mr_geom = (rho * v_theta ** 2 - B_theta ** 2) * inv_r
    dvx_dt = dvx_dt + S_mr_geom * inv_rho

    S_mtheta = -2.0 * (rho * v_r * v_theta - B_r * B_theta) * inv_r
    dvy_dt = dvy_dt + S_mtheta * inv_rho

    dvel_dt = torch.stack([dvx_dt, dvy_dt, dvz_dt], dim=0)

    result: dict[str, torch.Tensor] = {
        "rho": drho_dt,
        "velocity": dvel_dt,
        "pressure": dp_dt,
        "B": dB_dt,
    }

    if e_electron is not None:
        result["e_electron"] = dU_dt[IEE]

    if s_rho is not None:
        result["s_rho"] = dU_dt[ISR]

    return result
