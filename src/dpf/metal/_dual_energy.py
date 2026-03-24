"""Dual-energy pressure recovery with entropy-based switching for Metal MHD.

Eliminates catastrophic cancellation when kinetic + magnetic energy dominates
thermal energy (high Mach / low beta). Uses an entropy tracer Srho advected
alongside the conservative variables; when total-energy subtraction is
unreliable, pressure is recovered from entropy instead.

Dimensional analysis (gamma = 5/3 throughout):
    Srho = p * rho^(1-gamma) = p * rho^(-2/3)
    [Srho] = Pa * (kg/m^3)^(-2/3)

    Recovery:  p = Srho * rho^(gamma-1) = Srho * rho^(2/3)
    [p] = Pa * rho^(-2/3) * rho^(2/3) = Pa.  Correct.

    Switching criterion: eta = p_S / |E_total|.
    Both [Pa] = [J/m^3].  Dimensionless.  Correct.

    Ohmic source: dSrho = (gamma-1) * eta_field * J^2 * dt / rho^(gamma-1)
    [(1) * (Ohm m) * (A/m^2)^2 * (s) / (kg/m^3)^(2/3)]
    = [W/m^3 * s / rho^(2/3)]  = [Pa / rho^(2/3)] = [Srho].  Correct.

References:
    Bryan et al. (2014), ApJS 211:19 -- dual-energy formalism in Enzo
    Ryu et al. (1993), ApJ 414:1 -- entropy switching for MHD
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from dpf.metal._riemann_constants import P_FLOOR as _DEFAULT_P_FLOOR
from dpf.metal._riemann_constants import RHO_FLOOR


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite smoothstep: 0 for x<=0, 1 for x>=1, smooth in between."""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def recover_pressure_dual_energy(
    rho: torch.Tensor,
    vel: torch.Tensor,
    B: torch.Tensor,
    E: torch.Tensor,
    Srho: torch.Tensor | None,
    gamma: float = 5.0 / 3.0,
    eta1: float = 1e-5,
    eta2: float = 1e-2,
    p_floor: float = _DEFAULT_P_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover thermal pressure using dual-energy switching.

    When kinetic + magnetic energy is comparable to total energy, the
    subtraction ``E - KE - ME`` suffers catastrophic cancellation.  The
    entropy tracer Srho provides a cancellation-free pressure estimate.
    A smooth blend transitions between the two based on how much of the
    total energy is thermal.

    Args:
        rho: Mass density, shape (...).  [kg/m^3]
        vel: Velocity components, shape (3, ...).  [m/s]
        B: Magnetic field components, shape (3, ...).  [T]  (Heaviside-Lorentz
            units where mu0 = 1, so B^2/2 gives magnetic energy density.)
        E: Total energy density, shape (...).  [J/m^3]
        Srho: Entropy tracer = p * rho^(1-gamma), shape (...).  None to
            disable entropy recovery (returns total-energy pressure only).
        gamma: Adiabatic index (default 5/3).
        eta1: Lower switching threshold -- below this, use entropy only.
        eta2: Upper switching threshold -- above this, use total energy only.
        p_floor: Minimum pressure (default from _riemann_constants).

    Returns:
        p: Recovered thermal pressure, shape (...).  [Pa]
        w: Blend weight (0 = pure entropy, 1 = pure total-energy), shape (...).
    """
    gm1 = gamma - 1.0

    KE = 0.5 * rho * torch.sum(vel * vel, dim=0)
    ME = 0.5 * torch.sum(B * B, dim=0)

    p_E = gm1 * (E - KE - ME)

    if Srho is None:
        p_out = torch.clamp(p_E, min=p_floor)
        w = torch.ones_like(p_out)
        return p_out, w

    p_S = Srho * torch.pow(torch.clamp(rho, min=RHO_FLOOR), gm1)

    E_abs = torch.clamp(torch.abs(E), min=1e-30)
    eta = torch.abs(p_S) / E_abs

    blend_arg = (eta - eta1) / max(eta2 - eta1, 1e-30)
    w = _smoothstep(blend_arg)

    p_out = w * p_E + (1.0 - w) * p_S
    p_out = torch.clamp(p_out, min=p_floor)

    return p_out, w


def shock_entropy_sync(
    rho: torch.Tensor,
    vel: torch.Tensor,
    B: torch.Tensor,
    E: torch.Tensor,
    Srho: torch.Tensor,
    p: torch.Tensor,
    div_v: torch.Tensor,
    gamma: float = 5.0 / 3.0,
    cs: torch.Tensor | None = None,
    dx: float = 1.0,
    sync_threshold: float = 1e-2,
) -> torch.Tensor:
    """Re-synchronize entropy tracer to total energy in shocked cells.

    At shocks the entropy tracer (which is advected as a passive scalar)
    does not satisfy Rankine-Hugoniot jump conditions and drifts.  Where
    the total-energy subtraction is still reliable, we reset Srho from
    the energy-derived pressure.

    Args:
        rho: Density (...).  [kg/m^3]
        vel: Velocity (3, ...).  [m/s]
        B: Magnetic field (3, ...).  [T]
        E: Total energy density (...).  [J/m^3]
        Srho: Entropy tracer (...).
        p: Current pressure (...).  [Pa]  (used for gradient-based shock
            detection; can be either p_E or blended p).
        div_v: Divergence of velocity (...).  [1/s]
        gamma: Adiabatic index.
        cs: Sound speed (...).  Computed from p, rho if None.
        dx: Grid spacing [m].
        sync_threshold: Minimum p_S / E to trust total-energy subtraction.

    Returns:
        Srho_synced: Updated entropy tracer with shocked cells reset.
    """
    gm1 = gamma - 1.0
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)

    if cs is None:
        cs = torch.sqrt(torch.clamp(gamma * p / rho_safe, min=0.0))

    compression = div_v < -0.33 * cs / dx

    p_abs = torch.clamp(torch.abs(p), min=1e-30)
    grad_p_mag = torch.zeros_like(p)
    for dim in range(p.ndim):
        grad_comp = torch.zeros_like(p)
        slc_fwd = [slice(None)] * p.ndim
        slc_bwd = [slice(None)] * p.ndim
        slc_ctr = [slice(None)] * p.ndim
        slc_fwd[dim] = slice(2, None)
        slc_bwd[dim] = slice(None, -2)
        slc_ctr[dim] = slice(1, -1)
        grad_comp[tuple(slc_ctr)] = (
            torch.abs(p[tuple(slc_fwd)] - p[tuple(slc_bwd)]) / (2.0 * dx)
        )
        grad_p_mag = grad_p_mag + grad_comp * grad_comp
    grad_p_mag = torch.sqrt(grad_p_mag)

    steep_gradient = grad_p_mag / p_abs > 0.33

    is_shock = compression & steep_gradient

    KE = 0.5 * rho * torch.sum(vel * vel, dim=0)
    ME = 0.5 * torch.sum(B * B, dim=0)
    p_E = gm1 * (E - KE - ME)

    p_S_current = Srho * torch.pow(rho_safe, gm1)
    E_abs = torch.clamp(torch.abs(E), min=1e-30)
    subtraction_reliable = torch.abs(p_S_current) / E_abs > sync_threshold

    sync_mask = is_shock & subtraction_reliable

    p_E_safe = torch.clamp(p_E, min=1e-30)
    Srho_from_E = p_E_safe * torch.pow(rho_safe, 1.0 - gamma)

    Srho_synced = torch.where(sync_mask, Srho_from_E, Srho)
    return Srho_synced


def entropy_ohmic_source(
    Srho: torch.Tensor,
    rho: torch.Tensor,
    eta_field: torch.Tensor,
    J_sq: torch.Tensor,
    dt: float,
    gamma: float = 5.0 / 3.0,
) -> torch.Tensor:
    """Add ohmic heating source term to entropy tracer.

    Ohmic dissipation (eta * J^2) heats the plasma.  The total-energy
    equation picks this up automatically via resistive diffusion of B,
    but the entropy tracer must be updated explicitly so the two stay
    consistent.

    Source derivation:
        ds/dt = (gamma-1) * Q / rho^(gamma-1)   where Q = eta * J^2
        Srho = rho * s   =>   dSrho/dt includes advection (handled by
        Riemann solver) + source.  The source on Srho per unit volume is:
        dSrho_source = (gamma-1) * eta * J^2 * dt / rho^(gamma-1)

    Dimensional check (gamma=5/3):
        [(Ohm m)(A/m^2)^2 (s)] / [(kg/m^3)^(2/3)]
        = [W/m^3 * s] / [rho^(2/3)]
        = [Pa] / [rho^(2/3)]  =  [Srho].  Correct.

    Args:
        Srho: Entropy tracer (...).
        rho: Density (...).  [kg/m^3]
        eta_field: Resistivity (...).  [Ohm m]
        J_sq: Current density squared |J|^2 (...).  [A^2/m^4]
        dt: Timestep [s].
        gamma: Adiabatic index.

    Returns:
        Srho_updated: Entropy tracer with ohmic source applied.
    """
    gm1 = gamma - 1.0
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    rho_gm1 = torch.pow(rho_safe, gm1)

    dSrho = gm1 * eta_field * J_sq * dt / rho_gm1
    return Srho + dSrho


def _coerce_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert to tensor, using float64 on CPU or native dtype on MPS."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float64)


def srho_from_state(
    rho: torch.Tensor | np.ndarray,
    p: torch.Tensor | np.ndarray,
    gamma: float = 5.0 / 3.0,
) -> torch.Tensor:
    """Compute entropy tracer from density and pressure (convenience alias)."""
    return initialize_entropy_tracer(_coerce_tensor(rho), _coerce_tensor(p), gamma=gamma)


def p_from_srho(
    rho: torch.Tensor | np.ndarray,
    Srho: torch.Tensor | np.ndarray,
    gamma: float = 5.0 / 3.0,
) -> torch.Tensor:
    """Recover pressure from entropy tracer: p = Srho * rho^(gamma-1)."""
    rho_t = _coerce_tensor(rho)
    Srho_t = _coerce_tensor(Srho)
    rho_safe = torch.clamp(rho_t, min=RHO_FLOOR)
    return Srho_t * torch.pow(rho_safe, gamma - 1.0)


def blend_weight(
    rho: torch.Tensor | np.ndarray,
    p_S: torch.Tensor | np.ndarray,
    E: torch.Tensor | np.ndarray,
    eta_1: float = 1e-5,
    eta_2: float = 1e-2,
) -> torch.Tensor:
    """Compute dual-energy blend weight (0=entropy, 1=total-energy)."""
    p_S_t = _coerce_tensor(p_S)
    E_t = _coerce_tensor(E)
    E_abs = torch.clamp(torch.abs(E_t), min=1e-30)
    eta = torch.abs(p_S_t) / E_abs
    blend_arg = (eta - eta_1) / max(eta_2 - eta_1, 1e-30)
    return _smoothstep(blend_arg)


def ohmic_source_srho(
    rho: torch.Tensor | np.ndarray,
    Srho: torch.Tensor | np.ndarray,
    Q_ohm: torch.Tensor | np.ndarray,
    dt: float,
    gamma: float = 5.0 / 3.0,
) -> torch.Tensor:
    """Add ohmic heating Q_ohm = eta*J^2 to entropy tracer (convenience wrapper)."""
    rho_t = _coerce_tensor(rho)
    Srho_t = _coerce_tensor(Srho)
    Q_t = _coerce_tensor(Q_ohm)
    gm1 = gamma - 1.0
    rho_safe = torch.clamp(rho_t, min=RHO_FLOOR)
    rho_gm1 = torch.pow(rho_safe, gm1)
    dSrho = gm1 * Q_t * dt / rho_gm1
    return Srho_t + dSrho


def initialize_entropy_tracer(
    rho: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 5.0 / 3.0,
) -> torch.Tensor:
    """Compute initial entropy tracer from known density and pressure.

    Srho = p * rho^(1-gamma)

    For gamma = 5/3:  Srho = p * rho^(-2/3).
    Verify:  p = Srho * rho^(gamma-1) = p * rho^(-2/3) * rho^(2/3) = p.  QED.

    Args:
        rho: Mass density (...).  [kg/m^3]
        p: Thermal pressure (...).  [Pa]
        gamma: Adiabatic index.

    Returns:
        Srho: Entropy tracer, same shape as rho.
    """
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    return p * torch.pow(rho_safe, 1.0 - gamma)
