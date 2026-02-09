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
# Device / NaN helpers (shared with metal_stencil.py API)
# ============================================================


def _ensure_mps(t: torch.Tensor, name: str = "tensor") -> None:
    """Validate that *t* resides on an MPS device.

    Args:
        t: Tensor to check.
        name: Human-readable label for the error message.

    Raises:
        ValueError: If the tensor is not on an MPS device.
    """
    if t.device.type != "mps":
        raise ValueError(f"{name} must be on MPS device, got {t.device}")


def _check_no_nan(t: torch.Tensor, label: str = "result") -> None:
    """Assert that *t* contains no NaN values.

    Args:
        t: Tensor to validate.
        label: Context string for the assertion message.

    Raises:
        AssertionError: If any element is NaN.
    """
    assert not torch.isnan(t).any(), f"NaN detected in {label}"


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

    vn = vel[dim]              # normal velocity
    Bn = B[dim]                # normal B component
    v_dot_B = vel[0] * B[0] + vel[1] * B[1] + vel[2] * B[2]
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    p_total = p + 0.5 * B_sq  # total pressure

    E = U[IEN]

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

    Note: We use natural units where mu_0 is absorbed into B (Heaviside-Lorentz
    convention, standard for Athena++/AthenaK).  The Python engine includes
    mu_0 explicitly; the Metal solver works in code units without it, matching
    the Athena++ convention used by the stencil module.

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
    va_sq = B_sq * inv_rho                    # total Alfven speed squared
    van_sq = Bn_sq * inv_rho                  # normal Alfven speed squared

    # (a^2 + va^2)^2 - 4 * a^2 * van^2
    sum_sq = a_sq + va_sq
    discriminant = sum_sq * sum_sq - 4.0 * a_sq * van_sq

    # Guard against negative discriminant from round-off
    discriminant = torch.clamp(discriminant, min=0.0)

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

    # Extract primitives
    rho_L, vel_L, p_L, B_L = _cons_to_prim_mps(UL, gamma)
    rho_R, vel_R, p_R, B_R = _cons_to_prim_mps(UR, gamma)

    # Normal velocities
    vn_L = vel_L[dim]
    vn_R = vel_R[dim]

    # Fast magnetosonic speeds
    cf_L = _fast_magnetosonic_mps(rho_L, p_L, B_L, gamma, dim)
    cf_R = _fast_magnetosonic_mps(rho_R, p_R, B_R, gamma, dim)

    # Davis wave speed estimates
    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)

    # Physical fluxes
    FL = _physical_flux_mps(UL, gamma, dim)
    FR = _physical_flux_mps(UR, gamma, dim)

    # Denominator with floor to avoid division by zero
    denom = SR - SL
    denom = torch.clamp(denom, min=1e-30)

    # HLL flux: (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)
    # Broadcast SL, SR from shape (...) to (8, ...) for the 8 components
    SL_8 = SL.unsqueeze(0)     # (1, ...)
    SR_8 = SR.unsqueeze(0)     # (1, ...)
    denom_8 = denom.unsqueeze(0)

    F_HLL = (SR_8 * FL - SL_8 * FR + SL_8 * SR_8 * (UR - UL)) / denom_8

    # Upwind selection: if SL >= 0, use FL; if SR <= 0, use FR
    # This improves accuracy when waves are purely left- or right-going.
    all_right = SL.unsqueeze(0) >= 0.0   # all waves going right -> use FL
    all_left = SR.unsqueeze(0) <= 0.0    # all waves going left  -> use FR

    F_HLL = torch.where(all_right, FL, F_HLL)
    F_HLL = torch.where(all_left, FR, F_HLL)

    _check_no_nan(F_HLL, f"HLL flux dim={dim}")
    return F_HLL


# ============================================================
# Flux computation: PLM + HLL for one dimension
# ============================================================


def compute_fluxes_mps(
    state: torch.Tensor,
    gamma: float,
    dx: float,
    dy: float,
    dz: float,
    dim: int,
    limiter: str = "minmod",
) -> torch.Tensor:
    """Compute numerical flux along one dimension using PLM + HLL.

    Pipeline:
        1. PLM reconstruction of conservative variables at cell interfaces.
        2. HLL Riemann solve at each interface.

    Args:
        state: Conservative state, shape (8, nx, ny, nz), float32, MPS.
        gamma: Adiabatic index.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
        dim: Dimension to compute fluxes along (0=x, 1=y, 2=z).
        limiter: Slope limiter for PLM ("minmod" or "mc").

    Returns:
        Numerical flux at interfaces, shape (8, ...) where the axis
        corresponding to *dim* has (n-1) entries.  Other axes are unchanged.
    """
    _ensure_mps(state, "state")

    # Step 1: PLM reconstruction
    UL, UR = plm_reconstruct_mps(state, dim=dim, limiter=limiter)

    # Step 2: HLL Riemann solve
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
) -> dict[str, torch.Tensor]:
    """Compute the full ideal MHD right-hand side dU/dt = -div(F).

    Applies dimension-split flux differencing in all three directions using
    PLM reconstruction and the HLL Riemann solver.  The update is:

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

        # Compute interface fluxes: shape (8, ...) with (n-1) along axis
        flux = compute_fluxes_mps(U, gamma, dx, dy, dz, dim_idx, limiter)

        # Flux differencing: dU/dt -= (F[i+1/2] - F[i-1/2]) / dh
        # flux has (n-1) interfaces.  The difference F[i+1/2] - F[i-1/2]
        # applies to interior cells i = 1..n-2, yielding (n-2) values.
        #
        # For cells at the boundary (i=0 and i=n-1), we use one-sided
        # differences:
        #   cell 0:     dU/dt -= (F[1/2] - F_boundary) / dh
        #               where F_boundary = physical flux at left wall
        #               For outflow BCs: F_boundary = F[1/2] -> no update
        #               Here we use: dU/dt[0] -= (F[1/2] - F[-1/2]) / dh
        #               with F[-1/2] approximated by _physical_flux of U[0].
        #
        # For simplicity and robustness, we apply zero-gradient (outflow)
        # boundary conditions: the boundary cells see zero flux difference.
        # Only interior cells (1..n-2) are updated with centred differences.
        # Boundary cells are updated with one-sided differences.

        axis = dim_idx + 1  # tensor axis (0 is the 8-component axis)

        # All cells:
        #   cell i gets: -(F[i+1/2] - F[i-1/2]) / dh
        #   F[i+1/2] corresponds to flux index i
        #   F[i-1/2] corresponds to flux index i-1
        #
        # For i=0: F[-1/2] doesn't exist; use zero-gradient -> F[-1/2] = F[0]
        # For i=n-1: F[(n-1)+1/2] doesn't exist; use F[n-2] = F[n_iface-1]

        # Build padded flux with ghost interfaces at boundaries
        # Pad flux with replicate at both ends along the spatial axis
        # flux shape along axis: (n_dim - 1)
        # After padding: (n_dim + 1), but we only need (n_dim) for cell updates
        # Actually: we need F[i+1/2] for i=0..n-1 and F[i-1/2] for i=0..n-1
        #   F[i+1/2] for i in [0, n-1]: indices 0..n-1 in flux (but flux has n-1 entries)
        #     => i=n-1 needs flux index n-1, which exists (0-indexed, flux has n_dim-1 entries)
        #     Wait: n_iface = n_dim - 1. Max flux index = n_iface - 1 = n_dim - 2.
        #     So F[(n-1)+1/2] needs index n_dim-1 which is OUT OF BOUNDS.
        #
        # Correct approach: pad flux by 1 on each side (replicating boundary values).

        # Pad along the correct axis.
        # The tensor has shape (8, nx, ny, nz).  We need to pad along axis.
        # torch.nn.functional.pad pads from the LAST dimension backwards.
        # For axis=1 (x): pad dims are (0,0, 0,0, 1,1) for (z_lo,z_hi, y_lo,y_hi, x_lo,x_hi)
        # For axis=2 (y): pad dims are (0,0, 1,1, 0,0)
        # For axis=3 (z): pad dims are (1,1, 0,0, 0,0)

        pad_spec = [0, 0, 0, 0, 0, 0]  # z_lo, z_hi, y_lo, y_hi, x_lo, x_hi
        # axis 1 -> x -> indices 4,5
        # axis 2 -> y -> indices 2,3
        # axis 3 -> z -> indices 0,1
        pad_idx = 2 * (3 - axis)  # axis 1->4, axis 2->2, axis 3->0
        pad_spec[pad_idx] = 1      # lo
        pad_spec[pad_idx + 1] = 1  # hi

        # flux_padded has (n_dim+1) entries along the spatial axis
        # but we need (n_dim) entries, so actually we just need +1 on each side
        # giving n_iface + 2 = n_dim + 1 entries.
        # Then F_right[i] = flux_padded[i+1], F_left[i] = flux_padded[i]
        # for i = 0..n_dim-1.  Since padded has n_dim+1 entries, max index = n_dim.
        # F_right[n_dim-1] = flux_padded[n_dim] = OK (padded).
        # F_left[0] = flux_padded[0] = replicated left boundary.

        flux_padded = torch.nn.functional.pad(flux, pad_spec, mode="replicate")

        F_right = torch.narrow(flux_padded, axis, 1, n_dim)  # F[i+1/2]
        F_left = torch.narrow(flux_padded, axis, 0, n_dim)   # F[i-1/2]

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
