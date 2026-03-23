"""PLM and WENO5 reconstruction plus positivity-preserving fallback.

Functions:
    _minmod                -- Minmod slope limiter.
    _mc_limiter            -- Monotonized Central slope limiter.
    _weno5_left_biased     -- Left-biased WENO5-Z reconstruction kernel.
    weno5_reconstruct_mps  -- WENO5 reconstruction at cell interfaces.
    plm_reconstruct_mps    -- PLM reconstruction at cell interfaces.
    _positivity_fallback   -- Replace unphysical interfaces with donor cell.
"""

from __future__ import annotations

import torch

from dpf.metal._riemann_constants import IDN, IEN, P_FLOOR, RHO_FLOOR
from dpf.metal._riemann_nan_safety import _repair_stats, _should_check_nan
from dpf.metal._utils import _ensure_mps

# ============================================================
# Slope limiters
# ============================================================


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minmod slope limiter, fully vectorized.

    minmod(a, b) = sign(a) * min(|a|, |b|)   if sign(a) == sign(b)
                 = 0                           otherwise

    Args:
        a: First slope, shape (...).
        b: Second slope, shape (...).

    Returns:
        Limited slope, shape (...).
    """
    return torch.where(
        a * b > 0.0,
        torch.sign(a) * torch.minimum(torch.abs(a), torch.abs(b)),
        torch.zeros_like(a),
    )


def _mc_limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Monotonized Central (MC, van Leer) slope limiter, fully vectorized.

    The MC limiter is the median of (2a, (a+b)/2, 2b), subject to the
    constraint that the result vanishes when a and b have different signs.

    Args:
        a: Left slope, shape (...).
        b: Right slope, shape (...).

    Returns:
        Limited slope, shape (...).
    """
    c1 = 2.0 * a
    c2 = 0.5 * (a + b)
    c3 = 2.0 * b

    max_val = torch.maximum(torch.maximum(c1, c2), c3)
    min_val = torch.minimum(torch.minimum(c1, c2), c3)
    med = c1 + c2 + c3 - max_val - min_val

    return torch.where(a * b > 0.0, med, torch.zeros_like(a))


# ============================================================
# WENO5-Z reconstruction
# ============================================================


def _weno5_left_biased(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    v4: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Left-biased WENO5-Z reconstruction at interface i+1/2.

    Uses 5-point stencil of POINT VALUES ``{f[i-2], f[i-1], f[i],
    f[i+1], f[i+2]}`` to reconstruct the value at ``x_{i+1/2}``
    (right face of cell i).

    The polynomial coefficients are for **point-value** interpolation
    (finite difference), NOT cell-average reconstruction (finite
    volume).  Ideal weights are ``d0=1/16, d1=10/16, d2=5/16``.

    Uses **WENO-Z** weights (Borges et al. 2008) instead of the
    classical WENO-JS (Jiang & Shu 1996).

    References:
        Shu C.-W., SIAM Rev. 51, 82-126 (2009), Sec. 2.2.
        Jiang G.-S. & Shu C.-W., JCP 126, 202-228 (1996).
        Borges R. et al., JCP 227, 3191-3211 (2008) -- WENO-Z.

    Returns the reconstructed value (same shape as inputs).
    """
    # Point-value candidate polynomials (Lagrange interpolation at u=+0.5)
    # S0 = {i-2, i-1, i}: coefficients (3/8, -10/8, 15/8)
    p0 = (3.0 * v0 - 10.0 * v1 + 15.0 * v2) / 8.0
    # S1 = {i-1, i, i+1}: coefficients (-1/8, 6/8, 3/8)
    p1 = (-v1 + 6.0 * v2 + 3.0 * v3) / 8.0
    # S2 = {i, i+1, i+2}: coefficients (3/8, 6/8, -1/8)
    p2 = (3.0 * v2 + 6.0 * v3 - v4) / 8.0

    d0 = 1.0 / 16.0
    d1 = 10.0 / 16.0
    d2 = 5.0 / 16.0

    beta0 = ((13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2
             + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2)
    beta1 = ((13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2
             + 0.25 * (v1 - v3) ** 2)
    beta2 = ((13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2
             + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2)

    # WENO-Z global smoothness indicator (Borges et al. 2008, Eq. 25)
    tau5 = torch.abs(beta0 - beta2)

    # WENO-Z nonlinear weights: alpha_k = d_k * (1 + (tau5/(eps+beta_k))^2)
    a0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)
    a1 = d1 * (1.0 + (tau5 / (eps + beta1)) ** 2)
    a2 = d2 * (1.0 + (tau5 / (eps + beta2)) ** 2)
    a_sum = torch.clamp(a0 + a1 + a2, min=1e-30)

    return (a0 / a_sum) * p0 + (a1 / a_sum) * p1 + (a2 / a_sum) * p2


def weno5_reconstruct_mps(
    U: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WENO5 (5th-order) reconstruction at cell interfaces.

    Weighted Essentially Non-Oscillatory reconstruction using three
    candidate stencils with nonlinear weights based on smoothness
    indicators (Jiang & Shu 1996).  Achieves 5th-order accuracy in
    smooth regions and reduces to ENO-like near discontinuities.

    Both left and right use the same ``_weno5_left_biased`` function;
    the right state simply shifts the stencil by +1.  This requires
    ``n >= 6`` cells (n-5 interior interfaces).  Boundary interfaces
    are filled with PLM for a total of ``n-1`` interfaces.

    References:
        Jiang G.-S. & Shu C.-W., JCP 126, 202-228 (1996).
        Shu C.-W., SIAM Rev. 51, 82-126 (2009).

    Args:
        U: Conservative state vector, shape (8, nx, ny, nz).
        dim: Spatial dimension to reconstruct along (0, 1, 2).

    Returns:
        Tuple (UL, UR) of left and right interface states.
        Each has ``n-1`` entries along the reconstruction axis,
        matching the PLM interface convention.
    """
    _ensure_mps(U, "U")

    axis = dim + 1
    n = U.shape[axis]

    if n < 6:
        return plm_reconstruct_mps(U, dim=dim, limiter="mc")

    UL_full, UR_full = plm_reconstruct_mps(U, dim=dim, limiter="mc")

    n_w = n - 5

    vL0 = torch.narrow(U, axis, 0, n_w)
    vL1 = torch.narrow(U, axis, 1, n_w)
    vL2 = torch.narrow(U, axis, 2, n_w)
    vL3 = torch.narrow(U, axis, 3, n_w)
    vL4 = torch.narrow(U, axis, 4, n_w)

    UL_weno = _weno5_left_biased(vL0, vL1, vL2, vL3, vL4)

    vR0 = torch.narrow(U, axis, 5, n_w)
    vR1 = torch.narrow(U, axis, 4, n_w)
    vR2 = torch.narrow(U, axis, 3, n_w)
    vR3 = torch.narrow(U, axis, 2, n_w)
    vR4 = torch.narrow(U, axis, 1, n_w)

    UR_weno = _weno5_left_biased(vR0, vR1, vR2, vR3, vR4)

    UL_out = UL_full.clone()
    UR_out = UR_full.clone()

    s_w = [slice(None)] * UL_out.ndim
    s_w[axis] = slice(2, 2 + n_w)
    UL_out[tuple(s_w)] = UL_weno
    UR_out[tuple(s_w)] = UR_weno

    UL_out[IDN] = torch.clamp(UL_out[IDN], min=RHO_FLOOR)
    UR_out[IDN] = torch.clamp(UR_out[IDN], min=RHO_FLOOR)
    if UL_out.shape[0] > IEN:
        UL_out[IEN] = torch.clamp(UL_out[IEN], min=P_FLOOR)
        UR_out[IEN] = torch.clamp(UR_out[IEN], min=P_FLOOR)

    return UL_out, UR_out


# ============================================================
# PLM Reconstruction
# ============================================================


def plm_reconstruct_mps(
    U: torch.Tensor,
    dim: int,
    limiter: str = "minmod",
    reconstruction_precision: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise Linear Method (PLM) reconstruction at cell interfaces.

    For each cell i, compute a limited slope and extrapolate to the left
    and right faces of the cell::

        U_L[i+1/2] = U[i]   + 0.5 * slope[i]     (right face of cell i)
        U_R[i+1/2] = U[i+1] - 0.5 * slope[i+1]   (left face of cell i+1)

    The output arrays UL, UR correspond to interfaces between cells
    [0..n-2] and [1..n-1] along the reconstruction axis.  There are
    (n-1) interfaces for n cells.

    Boundary treatment: zero-slope (constant) extrapolation at the first
    and last cells.

    Args:
        U: Conservative state vector, shape (8, nx, ny, nz), float32, MPS.
        dim: Spatial dimension to reconstruct along.
            0 -> x (tensor axis 1), 1 -> y (tensor axis 2), 2 -> z (tensor axis 3).
        limiter: Slope limiter, one of "minmod" or "mc".
        reconstruction_precision: Floating-point precision for the slope
            computation.  "float16" casts to half for the slope and
            linear extrapolation steps (1.9x speedup on MPS), then
            recasts to float32 before returning.  "float32" (default)
            leaves the tensor dtype unchanged.

    Returns:
        Tuple (UL, UR) of left and right interface states:
            UL: shape (8, ...) with (n-1) entries along the reconstruction axis.
            UR: shape (8, ...) with (n-1) entries along the reconstruction axis.
    """
    _ensure_mps(U, "U")

    orig_dtype = U.dtype
    if reconstruction_precision == "float16":
        U = U.half()

    axis = dim + 1
    n = U.shape[axis]

    if n < 2:
        raise ValueError(
            f"PLM reconstruction requires at least 2 cells along dim={dim}, got {n}"
        )

    if limiter == "mc":
        limiter_fn = _mc_limiter
    else:
        limiter_fn = _minmod

    fwd = torch.narrow(U, axis, 1, n - 1) - torch.narrow(U, axis, 0, n - 1)

    slope = torch.zeros_like(U)

    if n >= 3:
        left_slope = torch.narrow(fwd, axis, 0, n - 2)
        right_slope = torch.narrow(fwd, axis, 1, n - 2)
        limited = limiter_fn(left_slope, right_slope)

        slices = [slice(None)] * U.ndim
        slices[axis] = slice(1, 1 + (n - 2))
        slope[tuple(slices)] = limited

    UL = torch.narrow(U, axis, 0, n - 1) + 0.5 * torch.narrow(slope, axis, 0, n - 1)
    UR = torch.narrow(U, axis, 1, n - 1) - 0.5 * torch.narrow(slope, axis, 1, n - 1)

    rho_L = UL[IDN]
    rho_R = UR[IDN]

    UL = UL.clone()
    UR = UR.clone()
    UL[IDN] = torch.clamp(rho_L, min=RHO_FLOOR)
    UR[IDN] = torch.clamp(rho_R, min=RHO_FLOOR)

    if reconstruction_precision == "float16":
        UL = UL.to(orig_dtype)
        UR = UR.to(orig_dtype)

    return UL, UR


# ============================================================
# Positivity-preserving fallback
# ============================================================


@torch.compiler.disable
def _positivity_fallback(
    UL: torch.Tensor,
    UR: torch.Tensor,
    U: torch.Tensor,
    gamma: float,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace unphysical reconstructed states with first-order donor cell.

    After PLM/WENO5 reconstruction, some interface states may have
    negative pressure (E < KE + ME) or extreme velocities from
    overshooting at strong discontinuities.  These are replaced with
    the safe donor cell (piecewise constant) values::

        UL[i+1/2] = U[i],   UR[i+1/2] = U[i+1]

    Cell-centre values are guaranteed positive (floors enforced every
    step), so donor cell is always a safe fallback.

    References:
        Balsara D.S. & Spicer D.S., JCP 149, 270 (1999).
        Stone J.M. et al., ApJS 249, 4 (2020), Sec. 4.7.

    Args:
        UL: Left reconstructed state, shape (8, ...).
        UR: Right reconstructed state, shape (8, ...).
        U: Cell-centre conservative state, shape (8, nx, ny, nz).
        gamma: Adiabatic index.
        dim: Spatial dimension (0, 1, 2).

    Returns:
        Corrected (UL, UR) with unphysical interfaces replaced.
    """
    from dpf.metal._riemann_constants import IB1, IB2, IB3, IDN, IEN, IM1, IM2, IM3

    axis = dim + 1
    n = U.shape[axis]

    rho_L = torch.clamp(UL[IDN], min=RHO_FLOOR)
    inv_rho_L = 1.0 / rho_L
    KE_L = 0.5 * (UL[IM1] ** 2 + UL[IM2] ** 2 + UL[IM3] ** 2) * inv_rho_L
    ME_L = 0.5 * (UL[IB1] ** 2 + UL[IB2] ** 2 + UL[IB3] ** 2)
    p_L = (gamma - 1.0) * (UL[IEN] - KE_L - ME_L)

    rho_R = torch.clamp(UR[IDN], min=RHO_FLOOR)
    inv_rho_R = 1.0 / rho_R
    KE_R = 0.5 * (UR[IM1] ** 2 + UR[IM2] ** 2 + UR[IM3] ** 2) * inv_rho_R
    ME_R = 0.5 * (UR[IB1] ** 2 + UR[IB2] ** 2 + UR[IB3] ** 2)
    p_R = (gamma - 1.0) * (UR[IEN] - KE_R - ME_R)

    v_sq_L = (UL[IM1] ** 2 + UL[IM2] ** 2 + UL[IM3] ** 2) * inv_rho_L ** 2
    v_sq_R = (UR[IM1] ** 2 + UR[IM2] ** 2 + UR[IM3] ** 2) * inv_rho_R ** 2

    bad = (p_L < P_FLOOR) | (p_R < P_FLOOR)
    bad = bad | (v_sq_L > 2.5e11) | (v_sq_R > 2.5e11)
    bad = bad | torch.isnan(p_L) | torch.isnan(p_R)
    bad = bad | torch.isnan(UL[IDN]) | torch.isnan(UR[IDN])

    if not _should_check_nan():
        return UL, UR

    _repair_stats["calls"] += 1
    _repair_stats["total_checked"] += int(bad.numel())

    if not bad.any():
        return UL, UR

    import logging
    logger = logging.getLogger(__name__)

    n_bad = int(bad.sum().item())
    _repair_stats["total_repaired"] += n_bad
    if n_bad > 100:
        logger.debug(
            "Positivity fallback dim=%d: %d/%d interfaces to donor cell",
            dim, n_bad, bad.numel(),
        )

    UL_donor = torch.narrow(U, axis, 0, n - 1)
    UR_donor = torch.narrow(U, axis, 1, n - 1)

    bad_8 = bad.unsqueeze(0).expand_as(UL)

    UL_out = torch.where(bad_8, UL_donor, UL)
    UR_out = torch.where(bad_8, UR_donor, UR)

    return UL_out, UR_out
