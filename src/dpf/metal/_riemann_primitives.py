"""Primitive <-> conservative variable conversions and physical flux for Metal MHD.

Functions:
    _prim_to_cons_mps      -- Convert primitive to conservative variables.
    _cons_to_prim_mps      -- Convert conservative to primitive variables.
    _physical_flux_mps     -- Compute ideal MHD flux F(U) along one dimension.
    _fast_magnetosonic_mps -- Compute fast magnetosonic speed.
"""

from __future__ import annotations

import torch

from dpf.metal._riemann_constants import (
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
from dpf.metal._utils import _ensure_mps  # noqa: F401

# ============================================================
# Primitive <-> Conservative conversion
# ============================================================


def _prim_to_cons_mps(
    rho: torch.Tensor,
    vel: torch.Tensor,
    p: torch.Tensor,
    B: torch.Tensor,
    gamma: float,
    e_electron: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert primitive variables to conservative state vector.

    Args:
        rho: Density, shape (...).
        vel: Velocity, shape (3, ...).
        p: Thermal pressure, shape (...).
        B: Magnetic field, shape (3, ...).
        gamma: Adiabatic index.
        e_electron: Electron energy density, shape (...).  Optional;
            when provided the returned tensor has shape (9, ...).

    Returns:
        Conservative state U, shape (8, ...) or (9, ...) if e_electron given.
    """
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    p_safe = torch.clamp(p, min=P_FLOOR)

    KE = 0.5 * rho_safe * (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
    ME = 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    E_total = p_safe / (gamma - 1.0) + KE + ME

    components = [
        rho_safe,
        rho_safe * vel[0],
        rho_safe * vel[1],
        rho_safe * vel[2],
        E_total,
        B[0],
        B[1],
        B[2],
    ]

    if e_electron is not None:
        components.append(torch.clamp(e_electron, min=0.0))

    return torch.stack(components, dim=0)


def _cons_to_prim_mps(
    U: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert conservative state vector to primitive variables.

    Args:
        U: Conservative state, shape (8, ...) or (9, ...).
        gamma: Adiabatic index.

    Returns:
        Tuple (rho, vel, p, B) where:
            rho: shape (...), clamped above RHO_FLOOR.
            vel: shape (3, ...).
            p:   shape (...), clamped above P_FLOOR.
            B:   shape (3, ...).

    Note: e_electron (index 8) is not returned here -- it passes
    through the Riemann solver as a passive scalar and is extracted
    separately from the conservative vector after the RHS computation.
    """
    rho = torch.clamp(U[IDN], min=RHO_FLOOR)
    inv_rho = 1.0 / rho

    vx = U[IM1] * inv_rho
    vy = U[IM2] * inv_rho
    vz = U[IM3] * inv_rho
    vel = torch.stack([vx, vy, vz], dim=0)

    Bx = U[IB1]
    By = U[IB2]
    Bz = U[IB3]
    B = torch.stack([Bx, By, Bz], dim=0)

    E = U[IEN]
    KE = 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
    ME = 0.5 * (Bx ** 2 + By ** 2 + Bz ** 2)
    p = (gamma - 1.0) * (E - KE - ME)
    p = torch.clamp(p, min=P_FLOOR)

    return rho, vel, p, B


# ============================================================
# Physical flux along a given dimension
# ============================================================


def _physical_flux_mps(
    U: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """Compute the ideal MHD flux F(U) along dimension *dim*.

    The MHD flux along the n-direction (n = x, y, or z) is:

        F_rho    = rho * vn
        F_mom_i  = rho * vi * vn - Bi * Bn + delta_{in} * p_total
        F_energy = (E + p_total) * vn - Bn * (v . B)
        F_Bi     = Bi * vn - vi * Bn

    where p_total = p + 0.5*|B|^2 is the total (thermal + magnetic) pressure.

    Args:
        U: Conservative state, shape (8, ...).
        gamma: Adiabatic index.
        dim: Normal direction index (0=x, 1=y, 2=z).

    Returns:
        Flux tensor, shape (8, ...).
    """
    rho, vel, p, B = _cons_to_prim_mps(U, gamma)

    _V_MAX = 1e6
    vel = torch.clamp(vel, min=-_V_MAX, max=_V_MAX)

    vn = vel[dim]
    Bn = B[dim]
    v_dot_B = vel[0] * B[0] + vel[1] * B[1] + vel[2] * B[2]
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    p_total = p + 0.5 * B_sq

    E = torch.clamp(U[IEN], min=P_FLOOR)

    F_rho = rho * vn

    F_m1 = rho * vel[0] * vn - B[0] * Bn
    F_m2 = rho * vel[1] * vn - B[1] * Bn
    F_m3 = rho * vel[2] * vn - B[2] * Bn

    if dim == 0:
        F_m1 = F_m1 + p_total
    elif dim == 1:
        F_m2 = F_m2 + p_total
    else:
        F_m3 = F_m3 + p_total

    F_E = (E + p_total) * vn - Bn * v_dot_B

    F_B1 = B[0] * vn - vel[0] * Bn
    F_B2 = B[1] * vn - vel[1] * Bn
    F_B3 = B[2] * vn - vel[2] * Bn

    components = [F_rho, F_m1, F_m2, F_m3, F_E, F_B1, F_B2, F_B3]

    if U.shape[0] > NVAR:
        F_ee = U[IEE] * vn
        components.append(F_ee)

    if U.shape[0] > IEE + 1:
        F_sr = U[ISR] * vn
        components.append(F_sr)

    return torch.stack(components, dim=0)


# ============================================================
# Fast magnetosonic speed
# ============================================================


def _fast_magnetosonic_mps(
    rho: torch.Tensor,
    p: torch.Tensor,
    B: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """Compute the fast magnetosonic speed for MHD.

    The fast magnetosonic speed is the largest characteristic speed of the
    ideal MHD system:

        cf^2 = 0.5 * [ (a^2 + va^2) + sqrt( (a^2 + va^2)^2 - 4*a^2*van^2 ) ]

    where:
        a^2  = gamma * p / rho          (sound speed squared)
        va^2 = |B|^2 / rho              (Alfven speed squared, total B)
        van^2 = Bn^2 / rho              (Alfven speed squared, normal B)

    We use a numerically stable form of the discriminant to avoid
    catastrophic cancellation in float32:

        (a^2 + va^2)^2 - 4*a^2*van^2
            = (a^2 - va^2)^2 + 4*a^2*(va^2 - van^2)
            = (a^2 - va^2)^2 + 4*a^2*Bt^2/rho

    where Bt^2 = |B|^2 - Bn^2 is the transverse magnetic field squared.

    Note: We use natural units where mu_0 is absorbed into B (Heaviside-Lorentz
    convention, standard for Athena++/AthenaK).

    References:
        Stone J.M. et al., ApJS 249, 4 (2020), Appendix C -- wave speed.
        Miyoshi T. & Kusano K., JCP 208, 315 (2005), Section 2.

    Args:
        rho: Density, shape (...).
        p: Thermal pressure, shape (...).
        B: Magnetic field, shape (3, ...).
        gamma: Adiabatic index.
        dim: Normal direction (0, 1, 2).

    Returns:
        Fast magnetosonic speed cf, shape (...).
    """
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    p_safe = torch.clamp(p, min=P_FLOOR)
    inv_rho = 1.0 / rho_safe

    a_sq = gamma * p_safe * inv_rho
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    Bn_sq = B[dim] ** 2
    Bt_sq = torch.clamp(B_sq - Bn_sq, min=0.0)
    va_sq = B_sq * inv_rho

    # Clamp squared speeds to prevent float32 overflow in discriminant.
    # |diff|^2 overflows float32 (~3.4e38) when va_sq > ~1e19, which occurs
    # at electrode boundaries (B_HL ~ 2e4, rho ~ 1e-12).  Cap at c^2.
    _CF_SQ_MAX = torch.tensor(9.0e16, dtype=rho.dtype, device=rho.device)
    a_sq = torch.clamp(a_sq, max=_CF_SQ_MAX)
    va_sq = torch.clamp(va_sq, max=_CF_SQ_MAX)

    diff = a_sq - va_sq
    vat_sq = torch.clamp(Bt_sq * inv_rho, max=_CF_SQ_MAX)
    discriminant = diff * diff + 4.0 * a_sq * vat_sq
    discriminant = torch.clamp(discriminant, min=0.0)

    sum_sq = a_sq + va_sq
    cf_sq = 0.5 * (sum_sq + torch.sqrt(discriminant))
    cf_sq = torch.clamp(cf_sq, min=0.0, max=_CF_SQ_MAX)

    return torch.sqrt(cf_sq)
