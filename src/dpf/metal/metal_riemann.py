"""Fully vectorized HLL Riemann solver and PLM reconstruction on PyTorch MPS.

Production physics code for ideal MHD.  The HLL solver conserves mass,
momentum (3 components), total energy, and magnetic flux (3 components) --
8 conservative variables per cell.

Conservative variable ordering::

    U = [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]

where E_total = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2.

All tensors are float32 on the MPS device.  There are NO Python loops over
grid cells -- every operation is a vectorized tensor op suitable for GPU
execution on Apple Metal.

Reconstruction uses Piecewise Linear Method (PLM, 2nd-order) with
selectable slope limiters (minmod, MC).  The Riemann solver uses
HLL (Harten-Lax-van Leer) wave speed estimates with Davis bounds and
the full fast magnetosonic speed for MHD.

References:
    Harten A., Lax P.D., van Leer B., SIAM Rev. 25, 35 (1983).
    Davis S.F., SIAM J. Sci. Stat. Comp. 9, 445 (1988).
    Miyoshi T. & Kusano K., JCP 208, 315 (2005)  -- wave speed estimates.
    Toro E.F., Riemann Solvers and Numerical Methods for Fluid Dynamics
        (3rd ed., Springer, 2009).
    Stone J.M. et al., ApJS 249, 4 (2020)  -- Athena++ methods paper.

Functions:
    plm_reconstruct_mps    -- PLM reconstruction with slope limiters.
    hll_flux_mps           -- HLL approximate Riemann solver (8-component).
    compute_fluxes_mps     -- Full reconstruction + Riemann solve for one dim.
    mhd_rhs_mps            -- Full MHD right-hand side: -div(F) in 3D.
"""

from __future__ import annotations

import logging

import torch

from dpf.metal._utils import _check_no_nan, _ensure_mps  # noqa: F401

logger = logging.getLogger(__name__)

# Number of conservative MHD variables:
#   [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]
NVAR: int = 8

# Index aliases for the conservative state vector U[8, ...]
IDN: int = 0   # density
IM1: int = 1   # x-momentum  (rho * vx)
IM2: int = 2   # y-momentum  (rho * vy)
IM3: int = 3   # z-momentum  (rho * vz)
IEN: int = 4   # total energy
IB1: int = 5   # Bx
IB2: int = 6   # By
IB3: int = 7   # Bz

# Density and pressure floors
RHO_FLOOR: float = 1e-12
P_FLOOR: float = 1e-12


# ============================================================
# Primitive <-> Conservative conversion
# ============================================================


def _prim_to_cons_mps(
    rho: torch.Tensor,
    vel: torch.Tensor,
    p: torch.Tensor,
    B: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Convert primitive variables to conservative state vector.

    Args:
        rho: Density, shape (...).
        vel: Velocity, shape (3, ...).
        p: Thermal pressure, shape (...).
        B: Magnetic field, shape (3, ...).
        gamma: Adiabatic index.

    Returns:
        Conservative state U, shape (8, ...).
    """
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    p_safe = torch.clamp(p, min=P_FLOOR)

    # Kinetic energy density: 0.5 * rho * |v|^2
    KE = 0.5 * rho_safe * (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

    # Magnetic energy density: 0.5 * |B|^2
    ME = 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

    # Total energy: internal + kinetic + magnetic
    E_total = p_safe / (gamma - 1.0) + KE + ME

    U = torch.stack([
        rho_safe,
        rho_safe * vel[0],
        rho_safe * vel[1],
        rho_safe * vel[2],
        E_total,
        B[0],
        B[1],
        B[2],
    ], dim=0)  # (8, ...)

    return U


def _cons_to_prim_mps(
    U: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert conservative state vector to primitive variables.

    Args:
        U: Conservative state, shape (8, ...).
        gamma: Adiabatic index.

    Returns:
        Tuple (rho, vel, p, B) where:
            rho: shape (...), clamped above RHO_FLOOR.
            vel: shape (3, ...).
            p:   shape (...), clamped above P_FLOOR.
            B:   shape (3, ...).
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

    # Clamp velocities to prevent float32 overflow in products
    _V_MAX = 1e6
    vel = torch.clamp(vel, min=-_V_MAX, max=_V_MAX)

    vn = vel[dim]              # normal velocity
    Bn = B[dim]                # normal B component
    v_dot_B = vel[0] * B[0] + vel[1] * B[1] + vel[2] * B[2]
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    p_total = p + 0.5 * B_sq  # total pressure

    E = torch.clamp(U[IEN], min=P_FLOOR)  # ensure positive total energy

    # Mass flux
    F_rho = rho * vn

    # Momentum flux: F_mom_i = rho*vi*vn - Bi*Bn + delta_{in}*ptot
    F_m1 = rho * vel[0] * vn - B[0] * Bn
    F_m2 = rho * vel[1] * vn - B[1] * Bn
    F_m3 = rho * vel[2] * vn - B[2] * Bn

    # Add total pressure to the normal momentum component
    if dim == 0:
        F_m1 = F_m1 + p_total
    elif dim == 1:
        F_m2 = F_m2 + p_total
    else:
        F_m3 = F_m3 + p_total

    # Energy flux: (E + ptot)*vn - Bn*(v.B)
    F_E = (E + p_total) * vn - Bn * v_dot_B

    # Induction flux: F_Bi = Bi*vn - vi*Bn
    F_B1 = B[0] * vn - vel[0] * Bn
    F_B2 = B[1] * vn - vel[1] * Bn
    F_B3 = B[2] * vn - vel[2] * Bn

    flux = torch.stack([F_rho, F_m1, F_m2, F_m3, F_E, F_B1, F_B2, F_B3], dim=0)
    return flux


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
    This avoids subtracting two large nearly-equal quantities.

    Note: We use natural units where mu_0 is absorbed into B (Heaviside-Lorentz
    convention, standard for Athena++/AthenaK).  The Python engine includes
    mu_0 explicitly; the Metal solver works in code units without it, matching
    the Athena++ convention used by the stencil module.

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

    a_sq = gamma * p_safe * inv_rho           # sound speed squared
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    Bn_sq = B[dim] ** 2
    Bt_sq = torch.clamp(B_sq - Bn_sq, min=0.0)  # transverse B squared
    va_sq = B_sq * inv_rho                    # total Alfven speed squared

    # Numerically stable discriminant:
    #   (a^2 - va^2)^2 + 4*a^2*Bt^2/rho
    # This avoids catastrophic cancellation when a^2 ≈ va^2 (float32 issue).
    diff = a_sq - va_sq
    discriminant = diff * diff + 4.0 * a_sq * Bt_sq * inv_rho

    # Guard against negative discriminant from round-off (should not happen
    # with the stable form, but belt-and-suspenders for float32).
    discriminant = torch.clamp(discriminant, min=0.0)

    sum_sq = a_sq + va_sq
    cf_sq = 0.5 * (sum_sq + torch.sqrt(discriminant))

    # Guard against negative cf^2 from round-off
    cf_sq = torch.clamp(cf_sq, min=0.0)

    return torch.sqrt(cf_sq)


# ============================================================
# PLM Reconstruction
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

    This is more aggressive (less diffusive) than minmod while still TVD.

    Args:
        a: Left slope, shape (...).
        b: Right slope, shape (...).

    Returns:
        Limited slope, shape (...).
    """
    # All three candidates
    c1 = 2.0 * a
    c2 = 0.5 * (a + b)
    c3 = 2.0 * b

    # Median = max(min pairs) or equivalently sort and pick middle
    # median(x, y, z) = x + y + z - max(x, y, z) - min(x, y, z)
    max_val = torch.maximum(torch.maximum(c1, c2), c3)
    min_val = torch.minimum(torch.minimum(c1, c2), c3)
    med = c1 + c2 + c3 - max_val - min_val

    # Zero out where a and b have different signs (not monotone)
    return torch.where(a * b > 0.0, med, torch.zeros_like(a))


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
    classical WENO-JS (Jiang & Shu 1996).  WENO-Z introduces a
    global smoothness indicator ``tau_5 = |beta_0 - beta_2|`` that
    achieves the optimal convergence order at critical points
    (where f and f' vanish simultaneously), unlike WENO-JS which
    loses accuracy there.  The weight formula is:

        alpha_k = d_k * (1 + (tau_5 / (eps + beta_k))^p)

    with p=2 for 5th-order accuracy.

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

    # Ideal weights for point-value WENO5: d0=1/16, d1=10/16, d2=5/16
    d0 = 1.0 / 16.0
    d1 = 10.0 / 16.0
    d2 = 5.0 / 16.0

    # Smoothness indicators (Jiang-Shu — same for FD and FV)
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

    **Left state** at interface ``i+1/2``: left-biased WENO5 from
    stencil ``{i-2, i-1, i, i+1, i+2}``.

    **Right state** at interface ``i+1/2``: left-biased WENO5 applied
    from cell ``i+1``'s perspective using stencil
    ``{i-1, i, i+1, i+2, i+3}``.

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

    axis = dim + 1  # tensor axis (0 is the 8-component axis)
    n = U.shape[axis]

    if n < 6:
        # Need at least 6 cells for proper left + right WENO5 stencils
        return plm_reconstruct_mps(U, dim=dim, limiter="mc")

    # ------------------------------------------------------------------
    # Step 1: PLM reconstruction for ALL n-1 interfaces (baseline)
    # ------------------------------------------------------------------
    UL_full, UR_full = plm_reconstruct_mps(U, dim=dim, limiter="mc")

    # ------------------------------------------------------------------
    # Step 2: WENO5 for interior interfaces (overwrite interior of PLM)
    # ------------------------------------------------------------------
    # Left state at interface i+1/2 uses cells {i-2, i-1, i, i+1, i+2}.
    # Right state at interface i+1/2 uses cells {i-1, i, i+1, i+2, i+3}.
    # Both are valid for i ∈ [2, n-4], giving n_w = n-5 interfaces.
    # (Left alone could go to n-3, but the right requires i+3 ≤ n-1.)
    n_w = n - 5  # number of WENO5 interfaces (both left & right)

    # Left-biased stencil for i = 2..n-4
    vL0 = torch.narrow(U, axis, 0, n_w)  # i-2
    vL1 = torch.narrow(U, axis, 1, n_w)  # i-1
    vL2 = torch.narrow(U, axis, 2, n_w)  # i
    vL3 = torch.narrow(U, axis, 3, n_w)  # i+1
    vL4 = torch.narrow(U, axis, 4, n_w)  # i+2

    UL_weno = _weno5_left_biased(vL0, vL1, vL2, vL3, vL4)

    # Right state at i+1/2: left face of cell i+1.
    # This is the MIRROR of the left-biased reconstruction.
    # We reverse the stencil order so _weno5_left_biased computes at the
    # left face rather than the right face:
    #   v0=cell (i+1)+2 = i+3,  v1=cell (i+1)+1 = i+2,  v2=cell i+1,
    #   v3=cell i,  v4=cell i-1.
    vR0 = torch.narrow(U, axis, 5, n_w)  # i+3
    vR1 = torch.narrow(U, axis, 4, n_w)  # i+2
    vR2 = torch.narrow(U, axis, 3, n_w)  # i+1  (center cell)
    vR3 = torch.narrow(U, axis, 2, n_w)  # i
    vR4 = torch.narrow(U, axis, 1, n_w)  # i-1

    UR_weno = _weno5_left_biased(vR0, vR1, vR2, vR3, vR4)

    # ------------------------------------------------------------------
    # Step 3: Splice WENO5 into the PLM-initialized full arrays.
    # ------------------------------------------------------------------
    UL_out = UL_full.clone()
    UR_out = UR_full.clone()

    # WENO5 covers PLM interface indices 2..n-4 (n_w = n-5 interfaces)
    s_w = [slice(None)] * UL_out.ndim
    s_w[axis] = slice(2, 2 + n_w)
    UL_out[tuple(s_w)] = UL_weno
    UR_out[tuple(s_w)] = UR_weno

    # Enforce positivity on density and total energy
    UL_out[IDN] = torch.clamp(UL_out[IDN], min=RHO_FLOOR)
    UR_out[IDN] = torch.clamp(UR_out[IDN], min=RHO_FLOOR)
    if UL_out.shape[0] > IEN:
        UL_out[IEN] = torch.clamp(UL_out[IEN], min=P_FLOOR)
        UR_out[IEN] = torch.clamp(UR_out[IEN], min=P_FLOOR)

    return UL_out, UR_out


def plm_reconstruct_mps(
    U: torch.Tensor,
    dim: int,
    limiter: str = "minmod",
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

    Returns:
        Tuple (UL, UR) of left and right interface states:
            UL: shape (8, ...) with (n-1) entries along the reconstruction axis.
            UR: shape (8, ...) with (n-1) entries along the reconstruction axis.
            UL[..., i, ...] is the left state at interface i+1/2.
            UR[..., i, ...] is the right state at interface i+1/2.
    """
    _ensure_mps(U, "U")

    # Map spatial dim to tensor axis: dim 0 -> axis 1, dim 1 -> axis 2, etc.
    axis = dim + 1
    n = U.shape[axis]

    if n < 2:
        raise ValueError(
            f"PLM reconstruction requires at least 2 cells along dim={dim}, got {n}"
        )

    # Select limiter function
    if limiter == "mc":
        limiter_fn = _mc_limiter
    else:
        limiter_fn = _minmod

    # Build slope:
    #   left_slope[i]  = U[i] - U[i-1]   for i = 1..n-1
    #   right_slope[i] = U[i+1] - U[i]   for i = 0..n-2
    # These differ in length.  For the interior cells (1..n-2), both exist.
    # For cell 0: only right_slope exists -> slope = 0 (zero extrapolation).
    # For cell n-1: only left_slope exists -> slope = 0.

    # Use narrow/slice along the correct axis.
    # forward diff: U[1:] - U[:-1]  along axis => shape has (n-1) along axis
    fwd = torch.narrow(U, axis, 1, n - 1) - torch.narrow(U, axis, 0, n - 1)

    # Full slope array with zero boundaries
    slope = torch.zeros_like(U)

    if n >= 3:
        # Interior cells (indices 1..n-2):
        #   left_slope  = fwd[0..n-3]  (these are U[1]-U[0], ..., U[n-2]-U[n-3])
        #   right_slope = fwd[1..n-2]  (these are U[2]-U[1], ..., U[n-1]-U[n-2])
        left_slope = torch.narrow(fwd, axis, 0, n - 2)
        right_slope = torch.narrow(fwd, axis, 1, n - 2)
        limited = limiter_fn(left_slope, right_slope)

        # Place into slope[..., 1:n-1, ...]
        # We need to index into the slope tensor at positions 1..n-2 along axis
        idx_start = 1
        idx_len = n - 2
        # Create a view of slope at the interior positions
        # Use slice_scatter or manual indexing
        slices = [slice(None)] * U.ndim
        slices[axis] = slice(idx_start, idx_start + idx_len)
        slope[tuple(slices)] = limited

    # Reconstruct at interfaces:
    #   UL[i+1/2] = U[i] + 0.5 * slope[i]       -> right face of cell i
    #   UR[i+1/2] = U[i+1] - 0.5 * slope[i+1]   -> left face of cell i+1
    #
    # Interface i+1/2 exists for i = 0..n-2, so there are (n-1) interfaces.
    UL = torch.narrow(U, axis, 0, n - 1) + 0.5 * torch.narrow(slope, axis, 0, n - 1)
    UR = torch.narrow(U, axis, 1, n - 1) - 0.5 * torch.narrow(slope, axis, 1, n - 1)

    # Enforce positivity of density (component 0) and energy (component 4)
    # Use index_select on dim=0 to pick out component 0
    rho_L = UL[IDN]
    rho_R = UR[IDN]
    rho_L_floor = torch.clamp(rho_L, min=RHO_FLOOR)
    rho_R_floor = torch.clamp(rho_R, min=RHO_FLOOR)

    # Build corrected UL/UR with floored density
    # UL and UR are contiguous views; we must construct new tensors
    UL = UL.clone()
    UR = UR.clone()
    UL[IDN] = rho_L_floor
    UR[IDN] = rho_R_floor

    return UL, UR


# ============================================================
# HLL Riemann solver (8-component, fully vectorized)
# ============================================================


def hll_flux_mps(
    UL: torch.Tensor,
    UR: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """HLL (Harten-Lax-van Leer) approximate Riemann solver for ideal MHD.

    Computes the numerical flux at cell interfaces given left and right
    reconstructed states.  The HLL flux is:

        F_HLL = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

    where SL and SR are the left and right wave speed estimates (Davis bounds):

        SL = min(vn_L - cf_L,  vn_R - cf_R)
        SR = max(vn_L + cf_L,  vn_R + cf_R)

    and cf is the fast magnetosonic speed.

    The HLL solver conserves all 8 conservative quantities (mass, 3 momenta,
    energy, 3 B-field components) by construction because it uses the exact
    Rankine-Hugoniot jump condition across the two bounding waves.

    Note on the normal B-field flux: for ideal MHD with constrained transport
    (CT), the normal B component is updated separately via the CT algorithm
    (Faraday's law).  The HLL flux for Bn is still computed here for
    completeness, but in a CT scheme it is typically ignored for the normal
    component.

    Args:
        UL: Left state at interfaces, shape (8, ...), float32, MPS.
        UR: Right state at interfaces, shape (8, ...), float32, MPS.
        gamma: Adiabatic index.
        dim: Normal direction (0=x, 1=y, 2=z).

    Returns:
        HLL numerical flux, shape (8, ...), float32, MPS.
    """
    _ensure_mps(UL, "UL")
    _ensure_mps(UR, "UR")

    # ---- Sanitize inputs: replace any NaN with floor values ----
    # This prevents NaN from upstream (e.g., bad reconstruction at
    # strong discontinuities) from propagating into the flux.
    UL = torch.where(torch.isnan(UL), torch.zeros_like(UL) + RHO_FLOOR, UL)
    UR = torch.where(torch.isnan(UR), torch.zeros_like(UR) + RHO_FLOOR, UR)

    # Enforce positivity on density and energy in the interface states
    UL_clean = UL.clone()
    UR_clean = UR.clone()
    UL_clean[IDN] = torch.clamp(UL[IDN], min=RHO_FLOOR)
    UR_clean[IDN] = torch.clamp(UR[IDN], min=RHO_FLOOR)
    UL_clean[IEN] = torch.clamp(UL[IEN], min=P_FLOOR)
    UR_clean[IEN] = torch.clamp(UR[IEN], min=P_FLOOR)

    # Extract primitives
    rho_L, vel_L, p_L, B_L = _cons_to_prim_mps(UL_clean, gamma)
    rho_R, vel_R, p_R, B_R = _cons_to_prim_mps(UR_clean, gamma)

    # Clamp velocities to prevent extreme wave speeds in float32
    _V_MAX = 1e6  # reasonable upper bound for MHD velocities in code units
    vel_L = torch.clamp(vel_L, min=-_V_MAX, max=_V_MAX)
    vel_R = torch.clamp(vel_R, min=-_V_MAX, max=_V_MAX)

    # Normal velocities
    vn_L = vel_L[dim]
    vn_R = vel_R[dim]

    # Fast magnetosonic speeds (using numerically stable formula)
    cf_L = _fast_magnetosonic_mps(rho_L, p_L, B_L, gamma, dim)
    cf_R = _fast_magnetosonic_mps(rho_R, p_R, B_R, gamma, dim)

    # Davis wave speed estimates
    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)

    # Ensure SR > SL (physical requirement: right wave faster than left)
    SR = torch.maximum(SR, SL + 1e-10)

    # Physical fluxes
    FL = _physical_flux_mps(UL_clean, gamma, dim)
    FR = _physical_flux_mps(UR_clean, gamma, dim)

    # Replace any NaN in physical fluxes with zero (Lax-Friedrichs fallback)
    FL = torch.where(torch.isnan(FL), torch.zeros_like(FL), FL)
    FR = torch.where(torch.isnan(FR), torch.zeros_like(FR), FR)

    # Denominator with floor to avoid division by zero
    denom = SR - SL
    denom = torch.clamp(denom, min=1e-20)

    # HLL flux: (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)
    # Broadcast SL, SR from shape (...) to (8, ...) for the 8 components
    SL_8 = SL.unsqueeze(0)     # (1, ...)
    SR_8 = SR.unsqueeze(0)     # (1, ...)
    denom_8 = denom.unsqueeze(0)

    F_HLL = (SR_8 * FL - SL_8 * FR + SL_8 * SR_8 * (UR_clean - UL_clean)) / denom_8

    # Upwind selection: if SL >= 0, use FL; if SR <= 0, use FR
    # This improves accuracy when waves are purely left- or right-going.
    all_right = SL.unsqueeze(0) >= 0.0   # all waves going right -> use FL
    all_left = SR.unsqueeze(0) <= 0.0    # all waves going left  -> use FR

    F_HLL = torch.where(all_right, FL, F_HLL)
    F_HLL = torch.where(all_left, FR, F_HLL)

    # Final NaN guard: replace any remaining NaN with Lax-Friedrichs flux
    # F_LF = 0.5*(FL + FR) is dissipative but always well-defined.
    has_nan = torch.isnan(F_HLL)
    if has_nan.any():
        F_LF = 0.5 * (FL + FR)
        F_HLL = torch.where(has_nan, F_LF, F_HLL)
        logger.warning("HLL flux dim=%d: %d NaN values replaced with LF flux",
                        dim, int(has_nan.sum().item()))

    return F_HLL


# ============================================================
# HLLD Riemann solver (8-component, fully vectorized)
# ============================================================


def hlld_flux_mps(
    UL: torch.Tensor,
    UR: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """HLLD (Harten-Lax-van Leer-Discontinuities) Riemann solver for MHD.

    Fully vectorized 8-component HLLD solver that resolves four intermediate
    states: two outer fast magnetosonic shocks and two inner Alfven/rotational
    discontinuities separated by a contact surface.  This captures contact
    discontinuities and Alfven waves much more accurately than HLL.

    The HLLD flux is selected from five regions::

        SL    SL*    SM    SR*    SR
        |------|------|------|------|
        F_L   F*_L  F**_L  F**_R  F*_R   F_R

    Following Miyoshi & Kusano, JCP 208, 315 (2005).

    All operations are fully vectorized PyTorch tensor ops on MPS/CPU.
    Float32 safe with floor guards at every division.

    Args:
        UL: Left state at interfaces, shape (8, ...), float32.
        UR: Right state at interfaces, shape (8, ...), float32.
        gamma: Adiabatic index.
        dim: Normal direction (0=x, 1=y, 2=z).

    Returns:
        HLLD numerical flux, shape (8, ...), float32.
    """
    _ensure_mps(UL, "UL")
    _ensure_mps(UR, "UR")

    # Map dim to component indices: normal and two transverse
    # dim=0 (x-normal): n=0, t1=1, t2=2  →  momentum IM1, IM2, IM3; B IB1, IB2, IB3
    # dim=1 (y-normal): n=1, t1=2, t2=0  →  momentum IM2, IM3, IM1; B IB2, IB3, IB1
    # dim=2 (z-normal): n=2, t1=0, t2=1  →  momentum IM3, IM1, IM2; B IB3, IB1, IB2
    im_n = IM1 + dim           # normal momentum index
    im_t1 = IM1 + (dim + 1) % 3  # transverse 1 momentum
    im_t2 = IM1 + (dim + 2) % 3  # transverse 2 momentum
    ib_n = IB1 + dim           # normal B index
    ib_t1 = IB1 + (dim + 1) % 3  # transverse 1 B
    ib_t2 = IB1 + (dim + 2) % 3  # transverse 2 B

    # ---- Sanitize inputs ----
    UL = torch.where(torch.isnan(UL), torch.zeros_like(UL) + RHO_FLOOR, UL)
    UR = torch.where(torch.isnan(UR), torch.zeros_like(UR) + RHO_FLOOR, UR)
    UL = UL.clone()
    UR = UR.clone()
    UL[IDN] = torch.clamp(UL[IDN], min=RHO_FLOOR)
    UR[IDN] = torch.clamp(UR[IDN], min=RHO_FLOOR)
    UL[IEN] = torch.clamp(UL[IEN], min=P_FLOOR)
    UR[IEN] = torch.clamp(UR[IEN], min=P_FLOOR)

    # ---- Extract primitives ----
    rho_L, vel_L, p_L, B_L = _cons_to_prim_mps(UL, gamma)
    rho_R, vel_R, p_R, B_R = _cons_to_prim_mps(UR, gamma)

    vn_L = vel_L[dim]
    vn_R = vel_R[dim]

    Bn_L = B_L[dim]
    Bn_R = B_R[dim]
    # Use arithmetic mean of normal B (should be continuous across interface)
    Bn = 0.5 * (Bn_L + Bn_R)

    # ---- Fast magnetosonic speeds ----
    cf_L = _fast_magnetosonic_mps(rho_L, p_L, B_L, gamma, dim)
    cf_R = _fast_magnetosonic_mps(rho_R, p_R, B_R, gamma, dim)

    # Davis wave speed estimates for outer shocks
    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = torch.maximum(SR, SL + 1e-10)

    # ---- Total pressures ----
    B_sq_L = B_L[0] ** 2 + B_L[1] ** 2 + B_L[2] ** 2
    B_sq_R = B_R[0] ** 2 + B_R[1] ** 2 + B_R[2] ** 2
    pt_L = p_L + 0.5 * B_sq_L  # total pressure (thermal + magnetic)
    pt_R = p_R + 0.5 * B_sq_R

    # ---- Contact wave speed SM (Eq. 38, Miyoshi & Kusano) ----
    denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L)
    denom_SM = torch.where(
        torch.abs(denom_SM) < 1e-20,
        torch.full_like(denom_SM, 1e-20) * torch.sign(denom_SM + 1e-30),
        denom_SM,
    )
    SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM

    # ---- Total pressure in star region (Eq. 41) ----
    pt_star = pt_L + rho_L * (SL - vn_L) * (SM - vn_L)
    pt_star = torch.clamp(pt_star, min=P_FLOOR)

    # ---- Star-state densities (Eq. 43) ----
    denom_L = torch.clamp(torch.abs(SL - SM), min=1e-20) * torch.sign(SL - SM + 1e-30)
    denom_R = torch.clamp(torch.abs(SR - SM), min=1e-20) * torch.sign(SR - SM + 1e-30)
    rho_sL = torch.clamp(rho_L * (SL - vn_L) / denom_L, min=RHO_FLOOR)
    rho_sR = torch.clamp(rho_R * (SR - vn_R) / denom_R, min=RHO_FLOOR)

    # ---- Star-state transverse velocities and B (Eqs. 44-46) ----
    # Sign-preserving clamp (Miyoshi & Kusano 2005, Eq. 43-44):
    # D_L can legitimately be negative when Bn^2 > rho*(S-v)*(S-SM).
    # Using torch.abs() would flip the sign of transverse star states.
    D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn ** 2
    safe_D_L = torch.where(torch.abs(D_L) < 1e-20, torch.full_like(D_L, 1e-20), D_L)
    inv_rhoL_dSL = 1.0 / safe_D_L

    D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn ** 2
    safe_D_R = torch.where(torch.abs(D_R) < 1e-20, torch.full_like(D_R, 1e-20), D_R)
    inv_rhoR_dSR = 1.0 / safe_D_R

    # Flag for when Bn ≈ 0 (degenerates to hydro HLLC)
    Bn_small = torch.abs(Bn) < 1e-10

    # Left star transverse velocity (Eq. 44)
    vt1_sL = vel_L[(dim + 1) % 3] - Bn * B_L[(dim + 1) % 3] * (SM - vn_L) * inv_rhoL_dSL
    vt2_sL = vel_L[(dim + 2) % 3] - Bn * B_L[(dim + 2) % 3] * (SM - vn_L) * inv_rhoL_dSL
    vt1_sL = torch.where(Bn_small, vel_L[(dim + 1) % 3], vt1_sL)
    vt2_sL = torch.where(Bn_small, vel_L[(dim + 2) % 3], vt2_sL)

    # Right star transverse velocity
    vt1_sR = vel_R[(dim + 1) % 3] - Bn * B_R[(dim + 1) % 3] * (SM - vn_R) * inv_rhoR_dSR
    vt2_sR = vel_R[(dim + 2) % 3] - Bn * B_R[(dim + 2) % 3] * (SM - vn_R) * inv_rhoR_dSR
    vt1_sR = torch.where(Bn_small, vel_R[(dim + 1) % 3], vt1_sR)
    vt2_sR = torch.where(Bn_small, vel_R[(dim + 2) % 3], vt2_sR)

    # Left star transverse B (Eq. 45)
    Bt1_sL = B_L[(dim + 1) % 3] * (rho_L * (SL - vn_L) ** 2 - Bn ** 2) * inv_rhoL_dSL
    Bt2_sL = B_L[(dim + 2) % 3] * (rho_L * (SL - vn_L) ** 2 - Bn ** 2) * inv_rhoL_dSL
    Bt1_sL = torch.where(Bn_small, B_L[(dim + 1) % 3], Bt1_sL)
    Bt2_sL = torch.where(Bn_small, B_L[(dim + 2) % 3], Bt2_sL)

    # Right star transverse B
    Bt1_sR = B_R[(dim + 1) % 3] * (rho_R * (SR - vn_R) ** 2 - Bn ** 2) * inv_rhoR_dSR
    Bt2_sR = B_R[(dim + 2) % 3] * (rho_R * (SR - vn_R) ** 2 - Bn ** 2) * inv_rhoR_dSR
    Bt1_sR = torch.where(Bn_small, B_R[(dim + 1) % 3], Bt1_sR)
    Bt2_sR = torch.where(Bn_small, B_R[(dim + 2) % 3], Bt2_sR)

    # ---- Star-state energies (Eq. 48) ----
    vB_sL = SM * Bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL
    vB_L = vn_L * Bn_L + vel_L[(dim + 1) % 3] * B_L[(dim + 1) % 3] + vel_L[(dim + 2) % 3] * B_L[(dim + 2) % 3]
    e_sL = ((SL - vn_L) * UL[IEN] - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L

    vB_sR = SM * Bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR
    vB_R = vn_R * Bn_R + vel_R[(dim + 1) % 3] * B_R[(dim + 1) % 3] + vel_R[(dim + 2) % 3] * B_R[(dim + 2) % 3]
    e_sR = ((SR - vn_R) * UR[IEN] - pt_R * vn_R + pt_star * SM + Bn * (vB_R - vB_sR)) / denom_R

    # ---- Build star conservative states (8 components) ----
    U_sL = torch.zeros_like(UL)
    U_sL[IDN] = rho_sL
    U_sL[im_n] = rho_sL * SM
    U_sL[im_t1] = rho_sL * vt1_sL
    U_sL[im_t2] = rho_sL * vt2_sL
    U_sL[IEN] = e_sL
    U_sL[ib_n] = Bn
    U_sL[ib_t1] = Bt1_sL
    U_sL[ib_t2] = Bt2_sL

    U_sR = torch.zeros_like(UR)
    U_sR[IDN] = rho_sR
    U_sR[im_n] = rho_sR * SM
    U_sR[im_t1] = rho_sR * vt1_sR
    U_sR[im_t2] = rho_sR * vt2_sR
    U_sR[IEN] = e_sR
    U_sR[ib_n] = Bn
    U_sR[ib_t1] = Bt1_sR
    U_sR[ib_t2] = Bt2_sR

    # ---- Physical fluxes (used in Rankine-Hugoniot formula) ----
    FL = _physical_flux_mps(UL, gamma, dim)
    FR = _physical_flux_mps(UR, gamma, dim)
    FL = torch.where(torch.isnan(FL), torch.zeros_like(FL), FL)
    FR = torch.where(torch.isnan(FR), torch.zeros_like(FR), FR)

    # ---- Star-region fluxes via Rankine-Hugoniot ----
    # F*_L = F_L + SL * (U*_L - U_L)
    # F*_R = F_R + SR * (U*_R - U_R)
    SL_8 = SL.unsqueeze(0)
    SR_8 = SR.unsqueeze(0)
    F_sL = FL + SL_8 * (U_sL - UL)
    F_sR = FR + SR_8 * (U_sR - UR)

    # ---- Alfven wave speeds in star region (Miyoshi & Kusano Eq. 51) ----
    sqrt_rho_sL = torch.sqrt(torch.clamp(rho_sL, min=RHO_FLOOR))
    sqrt_rho_sR = torch.sqrt(torch.clamp(rho_sR, min=RHO_FLOOR))
    SL_star = SM - torch.abs(Bn) / sqrt_rho_sL
    SR_star = SM + torch.abs(Bn) / sqrt_rho_sR

    # ---- Double-star transverse quantities (Eqs. 59-62) ----
    sign_Bn = torch.sign(Bn + 1e-30)  # avoid sign(0)
    denom_ds = torch.clamp(sqrt_rho_sL + sqrt_rho_sR, min=1e-20)

    # Double-star transverse velocities (Eq. 59) — same on both sides
    vt1_ds = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR
              + (Bt1_sR - Bt1_sL) * sign_Bn) / denom_ds
    vt2_ds = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR
              + (Bt2_sR - Bt2_sL) * sign_Bn) / denom_ds

    # Double-star transverse B (Eq. 60) — same on both sides
    Bt1_ds = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL
              + sqrt_rho_sL * sqrt_rho_sR * (vt1_sR - vt1_sL) * sign_Bn) / denom_ds
    Bt2_ds = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL
              + sqrt_rho_sL * sqrt_rho_sR * (vt2_sR - vt2_sL) * sign_Bn) / denom_ds

    # When Bn ≈ 0, Alfven waves collapse → double-star = single-star
    vt1_dsL = torch.where(Bn_small, vt1_sL, vt1_ds)
    vt2_dsL = torch.where(Bn_small, vt2_sL, vt2_ds)
    vt1_dsR = torch.where(Bn_small, vt1_sR, vt1_ds)
    vt2_dsR = torch.where(Bn_small, vt2_sR, vt2_ds)
    Bt1_ds_L = torch.where(Bn_small, Bt1_sL, Bt1_ds)
    Bt2_ds_L = torch.where(Bn_small, Bt2_sL, Bt2_ds)
    Bt1_ds_R = torch.where(Bn_small, Bt1_sR, Bt1_ds)
    Bt2_ds_R = torch.where(Bn_small, Bt2_sR, Bt2_ds)

    # Double-star energy (Eq. 62)
    vB_dsL = SM * Bn + vt1_dsL * Bt1_ds_L + vt2_dsL * Bt2_ds_L
    e_dsL = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn

    vB_dsR = SM * Bn + vt1_dsR * Bt1_ds_R + vt2_dsR * Bt2_ds_R
    e_dsR = e_sR + sqrt_rho_sR * (vB_sR - vB_dsR) * sign_Bn

    # ---- Build double-star conservative states (8 components) ----
    U_dsL = torch.zeros_like(UL)
    U_dsL[IDN] = rho_sL             # density continuous across Alfven wave
    U_dsL[im_n] = rho_sL * SM
    U_dsL[im_t1] = rho_sL * vt1_dsL
    U_dsL[im_t2] = rho_sL * vt2_dsL
    U_dsL[IEN] = e_dsL
    U_dsL[ib_n] = Bn
    U_dsL[ib_t1] = Bt1_ds_L
    U_dsL[ib_t2] = Bt2_ds_L

    U_dsR = torch.zeros_like(UR)
    U_dsR[IDN] = rho_sR
    U_dsR[im_n] = rho_sR * SM
    U_dsR[im_t1] = rho_sR * vt1_dsR
    U_dsR[im_t2] = rho_sR * vt2_dsR
    U_dsR[IEN] = e_dsR
    U_dsR[ib_n] = Bn
    U_dsR[ib_t1] = Bt1_ds_R
    U_dsR[ib_t2] = Bt2_ds_R

    # ---- Double-star fluxes via Rankine-Hugoniot ----
    # F**_L = F*_L + SL* * (U**_L - U*_L)
    # F**_R = F*_R + SR* * (U**_R - U*_R)
    SL_star_8 = SL_star.unsqueeze(0)
    SR_star_8 = SR_star.unsqueeze(0)
    F_dsL = F_sL + SL_star_8 * (U_dsL - U_sL)
    F_dsR = F_sR + SR_star_8 * (U_dsR - U_sR)

    # ---- Select flux based on 6 wave regions ----
    #   SL      SL*     SM     SR*     SR
    #    | F*_L  | F**_L  | F**_R | F*_R  |
    # F_L                                    F_R
    SM_8 = SM.unsqueeze(0)

    # Start with F_R (rightmost region: SR <= 0)
    F_HLLD = FR.clone()

    # SR* to SR: use F*_R
    mask_sR = (SR_star_8 <= 0.0) & (SR_8 > 0.0)
    F_HLLD = torch.where(mask_sR, F_sR, F_HLLD)

    # SM to SR*: use F**_R
    mask_dsR = (SM_8 <= 0.0) & (SR_star_8 > 0.0)
    F_HLLD = torch.where(mask_dsR, F_dsR, F_HLLD)

    # SL* to SM: use F**_L
    mask_dsL = (SL_star_8 <= 0.0) & (SM_8 > 0.0)
    F_HLLD = torch.where(mask_dsL, F_dsL, F_HLLD)

    # SL to SL*: use F*_L
    mask_sL = (SL_8 <= 0.0) & (SL_star_8 > 0.0)
    F_HLLD = torch.where(mask_sL, F_sL, F_HLLD)

    # Left region: use F_L where SL > 0
    mask_L = SL_8 > 0.0
    F_HLLD = torch.where(mask_L, FL, F_HLLD)

    # ---- Final NaN guard ----
    has_nan = torch.isnan(F_HLLD)
    if has_nan.any():
        # Fall back to HLL (more robust) where HLLD produced NaN
        denom_hll = torch.clamp(SR - SL, min=1e-20).unsqueeze(0)
        F_HLL_fallback = (SR_8 * FL - SL_8 * FR + SL_8 * SR_8 * (UR - UL)) / denom_hll
        F_HLL_fallback = torch.where(torch.isnan(F_HLL_fallback), 0.5 * (FL + FR), F_HLL_fallback)
        F_HLLD = torch.where(has_nan, F_HLL_fallback, F_HLLD)
        logger.warning("HLLD flux dim=%d: %d NaN values replaced with HLL fallback",
                        dim, int(has_nan.sum().item()))

    return F_HLLD


# ============================================================
# Flux computation: PLM + HLL/HLLD for one dimension
# ============================================================


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

    This is the standard positivity-preserving approach used in
    production MHD codes (Athena++, FLASH, PLUTO).

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
    axis = dim + 1
    n = U.shape[axis]

    # --- Compute pressure from reconstructed left states ---
    rho_L = torch.clamp(UL[IDN], min=RHO_FLOOR)
    inv_rho_L = 1.0 / rho_L
    KE_L = 0.5 * (UL[IM1] ** 2 + UL[IM2] ** 2 + UL[IM3] ** 2) * inv_rho_L
    ME_L = 0.5 * (UL[IB1] ** 2 + UL[IB2] ** 2 + UL[IB3] ** 2)
    p_L = (gamma - 1.0) * (UL[IEN] - KE_L - ME_L)

    # --- Compute pressure from reconstructed right states ---
    rho_R = torch.clamp(UR[IDN], min=RHO_FLOOR)
    inv_rho_R = 1.0 / rho_R
    KE_R = 0.5 * (UR[IM1] ** 2 + UR[IM2] ** 2 + UR[IM3] ** 2) * inv_rho_R
    ME_R = 0.5 * (UR[IB1] ** 2 + UR[IB2] ** 2 + UR[IB3] ** 2)
    p_R = (gamma - 1.0) * (UR[IEN] - KE_R - ME_R)

    # --- Velocity magnitudes ---
    v_sq_L = (UL[IM1] ** 2 + UL[IM2] ** 2 + UL[IM3] ** 2) * inv_rho_L ** 2
    v_sq_R = (UR[IM1] ** 2 + UR[IM2] ** 2 + UR[IM3] ** 2) * inv_rho_R ** 2

    # --- Flag bad interfaces ---
    # Negative pressure on either side
    bad = (p_L < P_FLOOR) | (p_R < P_FLOOR)
    # Extreme velocity (> 500 km/s typical DPF upper bound)
    bad = bad | (v_sq_L > 2.5e11) | (v_sq_R > 2.5e11)
    # NaN in any quantity
    bad = bad | torch.isnan(p_L) | torch.isnan(p_R)
    bad = bad | torch.isnan(UL[IDN]) | torch.isnan(UR[IDN])

    if not bad.any():
        return UL, UR

    n_bad = int(bad.sum().item())
    if n_bad > 100:
        logger.debug(
            "Positivity fallback dim=%d: %d/%d interfaces to donor cell",
            dim, n_bad, bad.numel(),
        )

    # Donor cell values: UL[i+1/2] = U[i], UR[i+1/2] = U[i+1]
    UL_donor = torch.narrow(U, axis, 0, n - 1)
    UR_donor = torch.narrow(U, axis, 1, n - 1)

    # Expand bad mask to all 8 components
    bad_8 = bad.unsqueeze(0).expand_as(UL)

    UL_out = torch.where(bad_8, UL_donor, UL)
    UR_out = torch.where(bad_8, UR_donor, UR)

    return UL_out, UR_out


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

    Returns:
        Numerical flux at interfaces, shape (8, ...) where the axis
        corresponding to *dim* has reduced entries.
    """
    _ensure_mps(state, "state")

    # Step 1: Reconstruction
    if reconstruction == "weno5" and state.shape[dim + 1] >= 5:
        UL, UR = weno5_reconstruct_mps(state, dim=dim)
    else:
        UL, UR = plm_reconstruct_mps(state, dim=dim, limiter=limiter)

    # Step 1.5: Positivity-preserving fallback — replace interfaces with
    # negative pressure or extreme velocity with safe donor cell values.
    UL, UR = _positivity_fallback(UL, UR, state, gamma, dim)

    # Step 2: Riemann solve
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
) -> dict[str, torch.Tensor]:
    """Compute the full ideal MHD right-hand side dU/dt = -div(F).

    Applies dimension-split flux differencing in all three directions using
    the chosen reconstruction (PLM or WENO5) and Riemann solver (HLL or
    HLLD).  The update is:

        dU/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
              - (G_{j+1/2} - G_{j-1/2}) / dy
              - (H_{k+1/2} - H_{k-1/2}) / dz

    The function accepts and returns state as a dictionary of primitive
    variables for API compatibility with the Python engine, but internally
    operates on conservative variables.

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

    _ensure_mps(rho, "rho")
    _ensure_mps(vel, "velocity")
    _ensure_mps(p, "pressure")
    _ensure_mps(B, "B")

    nx, ny, nz = rho.shape

    # Convert to conservative variables: (8, nx, ny, nz)
    U = _prim_to_cons_mps(rho, vel, p, B, gamma)

    # Accumulate conservative RHS: dU/dt = -div(F)
    dU_dt = torch.zeros_like(U)

    # Grid spacings indexed by dim
    dh = [dx, dy, dz]

    for dim_idx in range(3):
        n_dim = U.shape[dim_idx + 1]  # size along this spatial axis

        if n_dim < 2:
            # Cannot compute fluxes with fewer than 2 cells
            continue

        axis = dim_idx + 1  # tensor axis (0 is the 8-component axis)
        dim_bc = bc[dim_idx] if dim_idx < len(bc) else "outflow"

        if dim_bc == "periodic":
            # ---- Periodic boundary conditions ----
            # Pad state circularly so reconstruction sees wrapped neighbors.
            # Ghost width: 2 for PLM, 3 for WENO5.
            gh = 3 if (reconstruction == "weno5" and n_dim >= 5) else 2
            pad_spec_p = [0, 0, 0, 0, 0, 0]
            pad_idx_p = 2 * (3 - axis)
            pad_spec_p[pad_idx_p] = gh
            pad_spec_p[pad_idx_p + 1] = gh
            U_padded = torch.nn.functional.pad(U, pad_spec_p, mode="circular")

            flux = compute_fluxes_mps(
                U_padded, gamma, dx, dy, dz, dim_idx,
                limiter, riemann_solver, reconstruction,
            )
            # flux has (n_dim + 2*gh - 1) interfaces.
            # We need n interfaces for periodic flux differencing:
            #   F_right[i] = flux at interface (gh+i), i.e. between padded
            #                cell (gh+i) and (gh+i+1)
            #   F_left[i]  = flux at interface (gh+i-1)
            F_right = torch.narrow(flux, axis, gh, n_dim)
            F_left = torch.narrow(flux, axis, gh - 1, n_dim)

        else:
            # ---- Outflow (zero-gradient) boundary conditions ----
            flux = compute_fluxes_mps(
                U, gamma, dx, dy, dz, dim_idx,
                limiter, riemann_solver, reconstruction,
            )
            # Pad flux by 1 on each side with replicate (outflow).
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

    # Convert conservative RHS back to primitive variable RHS
    # dU/dt has components [d(rho)/dt, d(rho*v)/dt, d(E)/dt, d(B)/dt]
    #
    # We need to convert to primitive updates:
    #   d(rho)/dt -> direct
    #   d(v)/dt   -> (d(rho*v)/dt - v * d(rho)/dt) / rho
    #   d(p)/dt   -> (gamma-1) * (d(E)/dt - v . d(rho*v)/dt + 0.5*|v|^2 * d(rho)/dt
    #                             - B . d(B)/dt)
    #   d(B)/dt   -> direct

    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    inv_rho = 1.0 / rho_safe

    drho_dt = dU_dt[IDN]

    # Velocity: d(v_i)/dt = (d(rho*v_i)/dt - v_i * drho/dt) / rho
    dvx_dt = (dU_dt[IM1] - vel[0] * drho_dt) * inv_rho
    dvy_dt = (dU_dt[IM2] - vel[1] * drho_dt) * inv_rho
    dvz_dt = (dU_dt[IM3] - vel[2] * drho_dt) * inv_rho
    dvel_dt = torch.stack([dvx_dt, dvy_dt, dvz_dt], dim=0)

    # B-field: direct from conservative
    dBx_dt = dU_dt[IB1]
    dBy_dt = dU_dt[IB2]
    dBz_dt = dU_dt[IB3]
    dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], dim=0)

    # Pressure: from total energy equation
    #   E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2
    #   dE/dt = dp/(gamma-1) + rho*(v . dv/dt) + 0.5*|v|^2*drho/dt + B . dB/dt
    #   dp/dt = (gamma-1) * (dE/dt - rho*(v . dv/dt) - 0.5*|v|^2*drho/dt - B . dB/dt)
    #
    # Alternatively use the momentum RHS directly:
    #   v . d(rho*v)/dt = v . (rho*dv/dt + v*drho/dt) = rho*(v.dv/dt) + |v|^2*drho/dt
    # So: rho*(v.dv/dt) = v . d(rho*v)/dt - |v|^2 * drho/dt
    #   dp/dt = (gamma-1) * (dE/dt - v . d(rho*v)/dt + 0.5*|v|^2*drho/dt - B . dB/dt)

    v_dot_dmom = (vel[0] * dU_dt[IM1] + vel[1] * dU_dt[IM2] + vel[2] * dU_dt[IM3])
    v_sq = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
    B_dot_dB = B[0] * dBx_dt + B[1] * dBy_dt + B[2] * dBz_dt

    dp_dt = (gamma - 1.0) * (
        dU_dt[IEN] - v_dot_dmom + 0.5 * v_sq * drho_dt - B_dot_dB
    )

    return {
        "rho": drho_dt,
        "velocity": dvel_dt,
        "pressure": dp_dt,
        "B": dB_dt,
    }
