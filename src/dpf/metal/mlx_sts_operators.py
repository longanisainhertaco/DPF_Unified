"""Explicit parabolic RHS operators for RKL2 super-timestepping on MLX.

Provides the explicit Laplacian operators that RKL2 (mlx_sts.py) needs to
replace the implicit Thomas solver for resistive diffusion and thermal
conduction. All operations are pure MLX — no Python loops, no CPU sync.

Cylindrical diffusion operator:
    L(f) = (1/r) d/dr (r * alpha * df/dr) + d/dz (alpha * df/dz)

where alpha = eta/mu_0 (resistive) or kappa/(n*k_B) (conduction).

References:
    Meyer, Balsara & Aslam, JCP 231:2963 (2012) -- RKL2 method.
    Stone & Norman, ApJS 80:753 (1992) -- operator-split diffusion.
"""

from __future__ import annotations

import mlx.core as mx

from dpf.metal.constants import K_B  # noqa: E402
from dpf.metal.constants import M_DEUTERIUM as M_D


def resistive_diffusion_rhs(
    B_component: mx.array,
    alpha: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
    cylindrical: bool = True,
) -> mx.array:
    """Explicit RHS for resistive diffusion of a single B-field component.

    Computes L(B) = div(alpha * grad(B)) in cylindrical or Cartesian geometry.
    Neumann (zero-flux) boundary conditions applied via one-sided stencils.

    Args:
        B_component: Field to diffuse, shape (nr, nz).
        alpha: Diffusivity eta/mu_0, shape (nr, nz).
        dr: Radial cell spacing [m].
        dz: Axial cell spacing [m].
        r_cell: Cell-center radii, shape (nr,).
        cylindrical: If True, use (1/r) d/dr(r * alpha * dB/dr).

    Returns:
        dB/dt from diffusion, shape (nr, nz).
    """
    nr, nz = B_component.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # --- z-direction diffusion (uniform grid) ---
    # Interior: alpha_{i,j+1/2} * (B_{i,j+1} - B_{i,j}) - alpha_{i,j-1/2} * (B_{i,j} - B_{i,j-1})
    if nz > 2:
        B_zp = mx.concatenate([B_component[:, 1:], B_component[:, -1:]], axis=1)
        B_zm = mx.concatenate([B_component[:, :1], B_component[:, :-1]], axis=1)
        alpha_zp = 0.5 * (alpha + mx.concatenate([alpha[:, 1:], alpha[:, -1:]], axis=1))
        alpha_zm = 0.5 * (alpha + mx.concatenate([alpha[:, :1], alpha[:, :-1]], axis=1))
        Lz = (alpha_zp * (B_zp - B_component) - alpha_zm * (B_component - B_zm)) * inv_dz2
    else:
        Lz = mx.zeros_like(B_component)

    # --- r-direction diffusion ---
    if nr > 2:
        B_rp = mx.concatenate([B_component[1:, :], B_component[-1:, :]], axis=0)
        B_rm = mx.concatenate([B_component[:1, :], B_component[:-1, :]], axis=0)
        alpha_rp = 0.5 * (alpha + mx.concatenate([alpha[1:, :], alpha[-1:, :]], axis=0))
        alpha_rm = 0.5 * (alpha + mx.concatenate([alpha[:1, :], alpha[:-1, :]], axis=0))

        if cylindrical:
            # (1/r) d/dr(r * alpha * dB/dr)
            # r-face values: r_{i+1/2} = r_i + dr/2, r_{i-1/2} = r_i - dr/2
            r = r_cell[:, None]  # (nr, 1) broadcast
            r_safe = mx.maximum(r, 0.5 * dr)
            r_face_p = r_safe + 0.5 * dr
            r_face_m = mx.maximum(r_safe - 0.5 * dr, 0.0)

            inv_dr = 1.0 / dr
            flux_rp = r_face_p * alpha_rp * (B_rp - B_component) * inv_dr
            flux_rm = r_face_m * alpha_rm * (B_component - B_rm) * inv_dr
            Lr = (flux_rp - flux_rm) / (r_safe * dr)

            # L'Hopital at axis (ir=0):
            # lim_{r->0} (1/r) d/dr(r*alpha*dB/dr) = 2*alpha*d2B/dr2
            # d2B/dr2 ≈ (B[2] - 2*B[1] + B[0]) / dr^2 (one-sided 2nd derivative)
            if nr > 2:
                d2B = (B_component[2:3, :] - 2.0 * B_component[1:2, :] + B_component[:1, :])
                axis_Lr = 2.0 * alpha[:1, :] * d2B * inv_dr2
                Lr = mx.concatenate([axis_Lr, Lr[1:, :]], axis=0)
            elif nr > 1:
                # Only 2 cells: use forward difference as approximation
                axis_Lr = 2.0 * alpha[:1, :] * (B_component[1:2, :] - B_component[:1, :]) / (dr * dr)
                Lr = mx.concatenate([axis_Lr, Lr[1:, :]], axis=0)
        else:
            # Cartesian: d/dx(alpha * dB/dx)
            Lr = (alpha_rp * (B_rp - B_component) - alpha_rm * (B_component - B_rm)) * inv_dr2
    else:
        Lr = mx.zeros_like(B_component)

    return Lr + Lz


def thermal_conduction_rhs(
    T: mx.array,
    kappa: mx.array,
    rho: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
    ion_mass: float = M_D,
    cylindrical: bool = True,
) -> mx.array:
    """Explicit RHS for thermal conduction: dT/dt = div(chi * grad(T)).

    chi = kappa / (n * k_B) is the thermal diffusivity.

    Args:
        T: Temperature [K], shape (nr, nz).
        kappa: Thermal conductivity [W/(m*K)], shape (nr, nz).
        rho: Density [kg/m3], shape (nr, nz).
        dr, dz: Cell spacings [m].
        r_cell: Cell-center radii, shape (nr,).
        ion_mass: Ion mass [kg].
        cylindrical: Cylindrical geometry flag.

    Returns:
        dT/dt from conduction, shape (nr, nz).
    """
    # Thermal diffusivity: e_th = (3/2)*n*kB*T, so dT/dt = kappa/(1.5*n*kB) * lap(T)
    n = mx.maximum(rho / ion_mass, 1e-10)
    chi = kappa / (1.5 * n * K_B)
    return resistive_diffusion_rhs(T, chi, dr, dz, r_cell, cylindrical=cylindrical)


def compute_parabolic_dt(
    alpha: mx.array,
    dr: float,
    dz: float,
) -> float:
    """Compute explicit CFL limit for the parabolic operator.

    dt_parabolic = 0.5 * min(dr^2, dz^2) / max(alpha)

    Args:
        alpha: Diffusivity field, shape (nr, nz).
        dr, dz: Cell spacings [m].

    Returns:
        Maximum stable explicit timestep [s].
    """
    alpha_max = float(mx.max(alpha))
    if alpha_max <= 0:
        return 1e10  # no diffusion
    dx_min_sq = min(dr * dr, dz * dz)
    return 0.5 * dx_min_sq / alpha_max
