"""SSP-RK3 time integrator for the MLX MHD solver.

Implements the 3-stage Strong Stability Preserving Runge-Kutta method
(Shu & Osher 1988, Gottlieb et al. 2001):

  U^(1) = U^n + dt * L(U^n)
  U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
  U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

Key feature: dual-energy pressure recovery after EVERY stage to prevent
chain-rule cancellation from corrupting intermediate fluxes.

References:
    Shu C.-W. & Osher S., JCP 77:439 (1988) -- SSP-RK schemes.
    Gottlieb S. et al., SIAM Rev. 43:89 (2001) -- SSP review.
    Bryan et al., ApJS 211:19 (2014) -- dual-energy formalism.
    Popovas et al., arXiv:2211.02438 (2025) -- DISPATCH HLLS entropy switch.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from dpf.metal.mlx_grid import CartesianGrid, CylindricalGrid
from dpf.metal.mlx_kernels import (
    IBR,
    IBT,
    IBZ,
    IDN,
    IEN,
    IMR,
    IMT,
    IMZ,
    ISR,
    NVAR,
)
from dpf.metal.mlx_primitives import (
    P_FLOOR,
    RHO_FLOOR,
    cons_to_prim,
    fast_magnetosonic,
    recover_pressure_dual_energy,
)
from dpf.metal.mlx_riemann import mhd_rhs as _riemann_mhd_rhs

# Velocity clamping: cap at V_CLAMP_FACTOR * fast magnetosonic speed
_V_CLAMP_FACTOR: float = 10.0

# ---------------------------------------------------------------------------
# Compiled stage post-processing (fuses floors + dual-energy + velocity clamp)
# ---------------------------------------------------------------------------

def _stage_post_impl(U: mx.array, gamma: float) -> mx.array:
    """Fused post-stage processing: floors + dual-energy resync + velocity clamp.

    Combines _apply_floors, _resync_energy, and _clamp_velocity into a single
    compiled function. This eliminates 3 redundant cons_to_prim decompositions
    per RK stage (9 per SSP-RK3 step) by sharing the primitive variables.
    """
    gm1 = gamma - 1.0

    # --- Floor enforcement (Boris-aware, no fake mass injection) ---
    # Previous approach injected fake mass (rho = max(rho, B²/va_max²)) which
    # corrupted species mass fractions. Boris correction in Riemann solver
    # wavespeeds + geometric source terms bounds all forces without fake mass.
    # Only enforce minimal RHO_FLOOR for numerical stability.
    rho_raw = mx.maximum(U[IDN], RHO_FLOOR)
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    B_sq = Br * Br + Bz * Bz + Bt * Bt
    rho = rho_raw
    drho = mx.zeros_like(rho)
    isr = mx.maximum(U[ISR], 0.0)

    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * B_sq

    # Energy with mass injection correction
    E_floored = mx.maximum(
        U[IEN] + P_FLOOR / gm1 * drho / mx.maximum(rho_raw, RHO_FLOOR),
        P_FLOOR,
    )

    # --- Dual-energy pressure recovery (from _resync_energy) ---
    # Entropy-based pressure: p_S = rho^gamma * exp(S/rho)
    S = mx.maximum(isr, 0.0)
    p_S = S * mx.power(rho, gm1)
    # Conservative pressure
    p_E = mx.maximum(gm1 * (E_floored - KE - ME), P_FLOOR)
    # Switching criterion: use entropy when E is contaminated
    E_ratio = p_S / mx.maximum(E_floored, P_FLOOR)
    use_entropy = E_ratio < 1e-3
    p = mx.where(use_entropy, p_S, p_E)
    p = mx.maximum(p, P_FLOOR)

    # --- Velocity clamping (Boris-corrected fast magnetosonic) ---
    a_sq = gamma * p * inv_rho
    va_sq = B_sq * inv_rho
    # Boris correction: v_A'^2 = v_A^2 * c^2 / (v_A^2 + c^2)
    from dpf.metal.mlx_primitives import _C_BORIS_DEFAULT
    _C_BORIS_SQ = _C_BORIS_DEFAULT * _C_BORIS_DEFAULT
    va_sq_boris = va_sq * _C_BORIS_SQ / (va_sq + _C_BORIS_SQ)
    cf = mx.sqrt(a_sq + va_sq_boris)

    v_max = _V_CLAMP_FACTOR * cf
    v_mag = mx.sqrt(mx.maximum(vr * vr + vz * vz + vt * vt, 0.0))
    scale = mx.where(v_mag > v_max, v_max / mx.maximum(v_mag, 1e-30), mx.ones_like(v_mag))

    vr_c = vr * scale
    vz_c = vz * scale
    vt_c = vt * scale

    KE_c = 0.5 * rho * (vr_c * vr_c + vz_c * vz_c + vt_c * vt_c)
    E_new = p / gm1 + KE_c + ME

    # Rebuild full state
    return mx.stack([
        rho,              # IDN=0
        rho * vr_c,       # IMR=1
        rho * vz_c,       # IMZ=2
        rho * vt_c,       # IMT=3
        E_new,            # IEN=4
        isr,              # ISR=5
        Br,               # IBR=6
        Bz,               # IBZ=7
        Bt,               # IBT=8
        U[9] if U.shape[0] > 9 else mx.zeros_like(rho),  # IEE=9
    ], axis=0)


try:
    _compiled_stage_post = mx.compile(_stage_post_impl)
except Exception:
    _compiled_stage_post = _stage_post_impl


# ---------------------------------------------------------------------------
# Spatial operator L(U)
# ---------------------------------------------------------------------------


def mhd_rhs(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float = 5.0 / 3.0,
    dr: float | None = None,
    dz: float | None = None,
    method: str = "weno5z",
    riemann: str = "hlld",
) -> mx.array:
    """Compute the MHD right-hand side dU/dt = L(U).

    Delegates to mlx_riemann.mhd_rhs which applies _clamp_reconstructed
    guards after WENO5-Z reconstruction (prevents negative-energy states from
    reaching the Riemann solver) and correctly handles dim=1 axis transposition
    for the HLLD Metal kernel.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid instance with geometry arrays.
        gamma: Adiabatic index (default 5/3).
        dr: Radial cell spacing [m]. Defaults to grid.dr.
        dz: Axial cell spacing [m]. Defaults to grid.dz.
        method: Reconstruction method: "weno5z" or "plm".
        riemann: Riemann solver: "hlld" or "hll".

    Returns:
        dU/dt array, shape (NVAR, nr, nz), float32.
    """
    return _riemann_mhd_rhs(
        U,
        grid,
        gamma=gamma,
        dr=grid.dr if dr is None else dr,
        dz=grid.dz if dz is None else dz,
        method=method,
        riemann=riemann,
    )


def _geometric_sources(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float,
) -> mx.array:
    """Compute cylindrical geometric source terms.

    S_mr = (rho*vt^2 - Bt^2) / r     [centrifugal + hoop stress]
    S_mt = -2*(rho*vr*vt - Br*Bt) / r [Coriolis + tension]

    These are velocity-space sources; energy source is v dot S.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        grid: CylindricalGrid with inv_r array.
        gamma: Adiabatic index.

    Returns:
        Source array dU/dt, shape (NVAR, nr, nz).
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bt = U[IBT]

    inv_r = grid.inv_r[:, None]  # (nr, 1) broadcast over z

    # Momentum sources (in momentum units, i.e. rho * a)
    S_mr = (rho * vt * vt - Bt * Bt) * inv_r
    S_mt = -2.0 * (rho * vr * vt - Br * Bt) * inv_r
    S_E = vr * S_mr + vt * S_mt

    rows = [mx.zeros_like(rho)] * NVAR
    rows[IMR] = S_mr
    rows[IMT] = S_mt
    rows[IEN] = S_E

    return mx.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# CFL timestep
# ---------------------------------------------------------------------------


def compute_dt_cfl(
    U: mx.array,
    grid: CylindricalGrid,
    gamma: float = 5.0 / 3.0,
    cfl: float = 0.3,
    rho_cfl_fraction: float = 1e-4,
    use_boris: bool = False,
    c_boris: float = 5e5,
) -> float:
    """Compute CFL-limited timestep, ignoring vacuum cells.

    dt = cfl * min(dr, dz) / max(|v| + cf)
    where cf is the fast magnetosonic speed.

    When use_boris=True, uses Boris-corrected wave speeds (Gombosi 2002)
    that bound vacuum Alfven speed at c_boris without density injection.
    This eliminates the need for vacuum cell masking, but the mask is
    kept as a safety net.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        grid: CylindricalGrid with dr, dz.
        gamma: Adiabatic index (default 5/3).
        cfl: Courant number (default 0.3).
        rho_cfl_fraction: Fraction of max density below which cells are
            excluded from the CFL computation (default 1e-4).
        use_boris: Use Boris-corrected wave speeds (default False).
        c_boris: Boris reduced speed of light [m/s] (default 5e5).

    Returns:
        dt [s], float.
    """
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, gamma)

    if use_boris:
        from dpf.metal.mlx_primitives import fast_magnetosonic_boris
        cf_r = fast_magnetosonic_boris(rho, p, Br, Bz, Bt, gamma, dim=0, c_boris=c_boris)
        cf_z = fast_magnetosonic_boris(rho, p, Br, Bz, Bt, gamma, dim=1, c_boris=c_boris)
    else:
        cf_r = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=0)
        cf_z = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=1)

    speed_r = mx.abs(vr) + cf_r
    speed_z = mx.abs(vz) + cf_z

    # Mask out vacuum cells — stays on GPU
    rho_max = mx.max(rho)
    rho_threshold = mx.maximum(rho_cfl_fraction * rho_max, 10.0 * RHO_FLOOR)
    active = rho >= rho_threshold

    speed_r = mx.where(active, speed_r, 0.0)
    speed_z = mx.where(active, speed_z, 0.0)

    # y-direction speed for Cartesian 3D
    is_cartesian_3d = isinstance(grid, CartesianGrid) and U.ndim == 4
    if is_cartesian_3d:
        cf_y = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=1)
        speed_y = mx.abs(U[IMT] / mx.maximum(rho, RHO_FLOOR)) + cf_y
        speed_y = mx.where(active, speed_y, 0.0)
    else:
        speed_y = mx.array(0.0)

    # Single GPU→CPU transfer
    max_r = float(mx.max(speed_r))
    max_z = float(mx.max(speed_z))
    max_y = float(mx.max(speed_y)) if is_cartesian_3d else 0.0

    if not math.isfinite(max_r) or max_r == 0.0:
        max_r = 1.0
    if not math.isfinite(max_z) or max_z == 0.0:
        max_z = 1.0
    if is_cartesian_3d and (not math.isfinite(max_y) or max_y == 0.0):
        max_y = 1.0

    if is_cartesian_3d:
        dx_min = min(grid.dx, grid.dy, grid.dz)
        max_speed = max(max_r, max_z, max_y)
    else:
        dx_min = min(grid.dr, grid.dz)
        max_speed = max(max_r, max_z)

    dt = cfl * dx_min / max_speed

    # Hall whistler CFL: dt_hall = dx^2 * e * min(ne) / (B^2 / (mu_0 * m_i))
    # Whistler dispersion: omega = k^2 * B / (mu_0 * n_e * e), so
    # v_whistler = k * B / (mu_0 * n_e * e) grows with k → CFL ~ dx^2.
    # Only applied when Hall MHD is active (checked by caller).
    _E_CHARGE = 1.602176634e-19
    _MU0 = 4.0 * 3.141592653589793 * 1e-7
    B2 = Br**2 + Bz**2 + Bt**2
    B2_active = mx.where(active, B2, 0.0)
    max_B2 = float(mx.max(B2_active))
    min_ne = float(mx.min(mx.where(active, rho, 1e30)))
    if max_B2 > 0 and min_ne < 1e29:
        # In HL units: B_SI = B_HL * sqrt(mu_0)
        B2_si = max_B2 * _MU0
        ion_mass = 3.3435e-27  # deuterium
        ne_min = min_ne / ion_mass
        dt_hall = 0.5 * dx_min**2 * _E_CHARGE * _MU0 * ne_min / max(B2_si, 1e-30)
        dt = min(dt, cfl * dt_hall)

    return float(dt)


# ---------------------------------------------------------------------------
# Floor and velocity clamping
# ---------------------------------------------------------------------------


def _apply_floors(
    U: mx.array,
    rho_vac_fraction: float = 1e-4,
    va_max: float = 1e6,
) -> mx.array:
    """Enforce density, pressure, and vacuum B-field floors.

    Clamps rho >= RHO_FLOOR. In vacuum cells (rho < rho_vac_fraction *
    rho_max), scale B so that the Alfven speed v_A = |B|/sqrt(rho) stays
    below va_max. Without this, vacuum cells behind the compression sheath
    accumulate extreme B_theta from electrode BCs and geometric source
    amplification, causing B to grow by 10+ orders in a single step.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        rho_vac_fraction: Fraction of max density below which a cell is
            considered vacuum (default 1e-4).
        va_max: Maximum allowed Alfven speed in vacuum cells [m/s].

    Returns:
        U with floors applied, same shape.
    """
    # Enforce scalar floors on density and entropy tracer.
    rho_floored = mx.maximum(U[IDN : IDN + 1], RHO_FLOOR)
    isr_floored = mx.maximum(U[ISR : ISR + 1], 0.0)

    # Alfven-speed limited density floor: prevent runaway wavespeeds in
    # vacuum cells behind the compression sheath. Instead of clamping B
    # (which loses physics information), inject mass so that v_A stays
    # below va_max. This is equivalent to the "density injection" approach
    # used by Athena++ (athena4.2/src/hydro/srcterms/gravitational_acceleration.cpp)
    # and FLASH (Grid_markRefineDerefine).
    #
    # rho_min = B^2 / va_max^2 ensures v_A = |B|/sqrt(rho) <= va_max.
    # Applied at EVERY floor call (pre-RHS and post-stage) so that flux
    # computations never see extreme wavespeeds.
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    B_sq = Br * Br + Bz * Bz + Bt * Bt
    rho_B_floor = B_sq / (va_max * va_max)
    rho_old = rho_floored[0]
    rho_new = mx.maximum(rho_old, rho_B_floor)
    # Only inject mass where rho actually increased (energy bookkeeping)
    drho = rho_new - rho_old
    # Injected mass gets thermal energy at local temperature
    # (or just the floor — we pick floor to avoid computing T)
    E_new = mx.maximum(
        U[IEN] + P_FLOOR / (5.0 / 3.0 - 1.0) * drho / mx.maximum(rho_old, RHO_FLOOR),
        P_FLOOR,
    )

    # Rebuild conserved state. Modified slots: IDN=0, IEN=4, ISR=5.
    # Actual slot order: IDN=0, IMR=1, IMZ=2, IMT=3, IEN=4, ISR=5, IBR=6, IBZ=7, IBT=8, IEE=9
    # Use contiguous slices for unchanged ranges to preserve all NVAR slots.
    return mx.concatenate(
        [
            rho_new[None],  # IDN=0 — density with both floors applied
            U[IMR:IEN],     # IMR=1, IMZ=2, IMT=3 — unchanged
            E_new[None],    # IEN=4 — updated for mass injection
            isr_floored,    # ISR=5 — entropy tracer floor
            U[IBR:],        # IBR=6, IBZ=7, IBT=8, IEE=9 — unchanged
        ],
        axis=0,
    )


def _clamp_velocity(U: mx.array, gamma: float) -> mx.array:
    """Clamp velocity to _V_CLAMP_FACTOR * local fast magnetosonic speed.

    Prevents extreme velocities at low-density vacuum cells from blowing
    up intermediate RK stages.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        gamma: Adiabatic index.

    Returns:
        U with velocity clamped, same shape.
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    p = mx.maximum(gm1 * (U[IEN] - KE - ME), P_FLOOR)

    cf = fast_magnetosonic(rho, p, Br, Bz, Bt, gamma, dim=0)
    v_max = _V_CLAMP_FACTOR * cf  # (nr, nz)

    v_mag = mx.sqrt(mx.maximum(vr * vr + vz * vz + vt * vt, 0.0))
    scale = mx.where(v_mag > v_max, v_max / mx.maximum(v_mag, 1e-30), mx.ones_like(v_mag))

    vr_c = vr * scale
    vz_c = vz * scale
    vt_c = vt * scale

    KE_c = 0.5 * rho * (vr_c * vr_c + vz_c * vz_c + vt_c * vt_c)
    E_c = p / gm1 + KE_c + ME

    # Rebuild: modified slots IMR=1, IMZ=2, IMT=3, IEN=4; rest unchanged.
    # Actual slot order: IDN=0, IMR=1, IMZ=2, IMT=3, IEN=4, ISR=5, IBR=6, IBZ=7, IBT=8, IEE=9
    return mx.concatenate(
        [
            U[IDN : IMR],       # IDN=0 — unchanged
            (rho * vr_c)[None], # IMR=1 — clamped radial momentum
            (rho * vz_c)[None], # IMZ=2 — clamped axial momentum
            (rho * vt_c)[None], # IMT=3 — clamped toroidal momentum
            E_c[None],          # IEN=4 — energy consistent with clamped KE
            U[ISR:],            # ISR=5, IBR=6, IBZ=7, IBT=8, IEE=9 — unchanged
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Dual-energy pressure resync at intermediate stages
# ---------------------------------------------------------------------------


def _resync_energy(U: mx.array, gamma: float) -> mx.array:
    """Recover pressure from dual-energy and rewrite U[IEN] for consistency.

    Prevents chain-rule cancellation in float32 from leaking into the next
    RK stage. After recovery, E is reconstructed from p_recovered + KE + ME
    so the next call to cons_to_prim gets a non-corrupted energy.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        gamma: Adiabatic index.

    Returns:
        U with IEN rewritten to match recovered pressure.
    """
    p, _ = recover_pressure_dual_energy(U, gamma)
    rho, vr, vz, vt, _, Br, Bz, Bt = cons_to_prim(U, gamma)

    gm1 = gamma - 1.0
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    E_new = p / gm1 + KE + ME

    # Only IEN=4 changes — concatenate prefix/suffix slices to preserve all NVAR slots.
    # Actual slot order: IDN=0, IMR=1, IMZ=2, IMT=3, IEN=4, ISR=5, IBR=6, IBZ=7, IBT=8, IEE=9
    return mx.concatenate(
        [
            U[:IEN],        # IDN, IMR, IMZ, IMT — unchanged
            E_new[None],    # IEN — recovered from dual-energy
            U[IEN + 1 :],   # IBR, IBZ, IBT, ISR — unchanged
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# SSP-RK3 integrator
# ---------------------------------------------------------------------------


def _mask_ghost_rhs(L: mx.array, ng: int) -> mx.array:
    """Zero RHS in ghost cell regions so ghosts retain BC values.

    Without this, ghost cells injected by electrode BCs are evolved by
    the RK integrator, destroying the 1/r B_theta profile and creating
    unbounded pressure jumps that cause HLLD NaN in float32.

    The mask is applied to the radial (axis 1) boundaries — ng cells
    on each side. This matches the PyTorch Metal solver's approach of
    stripping ghosts from the RHS before applying updates.

    Args:
        L: RHS array, shape (NVAR, nr_padded, nz).
        ng: Number of ghost cells on each side.

    Returns:
        L with ghost regions zeroed.
    """
    if ng <= 0:
        return L
    nr = L.shape[1]
    mask_np = np.ones((1, nr, 1), dtype=np.float32)
    mask_np[:, :ng, :] = 0.0
    mask_np[:, -ng:, :] = 0.0
    return L * mx.array(mask_np)


def ssp_rk3_step(
    U: mx.array,
    grid: CylindricalGrid,
    dt: float,
    gamma: float = 5.0 / 3.0,
    method: str = "weno5z",
    riemann: str = "hlld",
    use_dual_energy: bool = True,
    ghost_ng: int = 0,
) -> mx.array:
    """Advance U by one SSP-RK3 timestep.

    At each intermediate stage:
    1. Compute L(U) via mhd_rhs
    2. Zero RHS in ghost cell regions (if ghost_ng > 0)
    3. SSP combination
    4. Enforce density/pressure floors
    5. If use_dual_energy: recover pressure from conservative E + entropy tracer
    6. Clamp velocity to _V_CLAMP_FACTOR * fast magnetosonic speed

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid or CartesianGrid instance.
        dt: Timestep [s].
        gamma: Adiabatic index (default 5/3).
        method: Reconstruction method: "weno5z" or "plm".
        riemann: Riemann solver: "hlld" or "hll".
        use_dual_energy: Apply dual-energy pressure recovery at each stage.
        ghost_ng: Number of ghost cells per side to hold fixed (0 = none).

    Returns:
        U_new, shape (NVAR, nr, nz), float32.
    """
    dr, dz = grid.dr, grid.dz

    # Use compiled fused post-processing when dual-energy is enabled (default).
    # Falls back to sequential calls when dual-energy is off (test/debug).
    if use_dual_energy:
        def _stage_post(Uk: mx.array) -> mx.array:
            return _compiled_stage_post(Uk, gamma)
    else:
        def _stage_post(Uk: mx.array) -> mx.array:
            Uk = _apply_floors(Uk)
            Uk = _clamp_velocity(Uk, gamma)
            return Uk

    # Apply floors to input state BEFORE computing RHS
    U = _apply_floors(U)

    # Stage 1: U1 = Un + dt * L(Un)
    L1 = mhd_rhs(U, grid, gamma, dr, dz, method, riemann)
    L1 = _mask_ghost_rhs(L1, ghost_ng)
    U1 = U + dt * L1
    U1 = _stage_post(U1)

    # Stage 2: U2 = 3/4 * Un + 1/4 * (U1 + dt * L(U1))
    L2 = mhd_rhs(U1, grid, gamma, dr, dz, method, riemann)
    L2 = _mask_ghost_rhs(L2, ghost_ng)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = _stage_post(U2)

    # Stage 3: Un+1 = 1/3 * Un + 2/3 * (U2 + dt * L(U2))
    L3 = mhd_rhs(U2, grid, gamma, dr, dz, method, riemann)
    L3 = _mask_ghost_rhs(L3, ghost_ng)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    U_new = _stage_post(U_new)

    mx.eval(U_new)
    return U_new


# ---------------------------------------------------------------------------
# SSP-RK2 integrator
# ---------------------------------------------------------------------------


def ssp_rk2_step(
    U: mx.array,
    grid: CylindricalGrid,
    dt: float,
    gamma: float = 5.0 / 3.0,
    method: str = "plm",
    riemann: str = "hlld",
    use_dual_energy: bool = True,
    ghost_ng: int = 0,
) -> mx.array:
    """Advance U by one SSP-RK2 timestep (simpler, for testing).

    Scheme (Shu & Osher 1988, 2-stage):
        U^(1) = U^n + dt * L(U^n)
        U^(n+1) = 1/2 * U^n + 1/2 * (U^(1) + dt * L(U^(1)))

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        grid: CylindricalGrid instance.
        dt: Timestep [s].
        gamma: Adiabatic index (default 5/3).
        method: Reconstruction method: "plm".
        riemann: Riemann solver: "hlld" or "hll".
        use_dual_energy: Apply dual-energy pressure recovery at each stage.

    Returns:
        U_new, shape (NVAR, nr, nz), float32.
    """
    dr, dz = grid.dr, grid.dz

    if use_dual_energy:
        def _stage_post(Uk: mx.array) -> mx.array:
            return _compiled_stage_post(Uk, gamma)
    else:
        def _stage_post(Uk: mx.array) -> mx.array:
            Uk = _apply_floors(Uk)
            Uk = _clamp_velocity(Uk, gamma)
            return Uk

    U = _apply_floors(U)

    # Stage 1
    L1 = mhd_rhs(U, grid, gamma, dr, dz, method, riemann)
    L1 = _mask_ghost_rhs(L1, ghost_ng)
    U1 = U + dt * L1
    U1 = _stage_post(U1)

    # Stage 2
    L2 = mhd_rhs(U1, grid, gamma, dr, dz, method, riemann)
    L2 = _mask_ghost_rhs(L2, ghost_ng)
    U_new = 0.5 * U + 0.5 * (U1 + dt * L2)
    U_new = _stage_post(U_new)

    mx.eval(U_new)
    return U_new
