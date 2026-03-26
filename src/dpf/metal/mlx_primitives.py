"""Conservative <-> primitive variable conversions and dual-energy pressure recovery in MLX.

All operations are vectorized mx.array operations — no Python loops.

Variable layout (10-component cylindrical conserved state):
    U[IDN] = rho           (mass density)
    U[IMR] = rho * vr      (radial momentum)
    U[IMZ] = rho * vz      (axial momentum)
    U[IMT] = rho * vtheta  (azimuthal momentum)
    U[IEN] = E             (total energy)
    U[ISR] = S * rho       (entropy tracer)
    U[IBR] = Br            (radial B-field)
    U[IBZ] = Bz            (axial B-field)
    U[IBT] = Btheta        (azimuthal B-field)
    U[IEE] = e_electron    (electron energy density)

References:
    Bryan et al. (2014), ApJS 211:19 -- dual-energy formalism
    Ryu et al. (1993), ApJ 414:1 -- entropy switching
    Stone et al. (2020), ApJS 249:4 -- fast magnetosonic speed
    Miyoshi & Kusano (2005), JCP 208:315 -- HLLD wave speeds
"""

from __future__ import annotations

import mlx.core as mx

from dpf.metal.mlx_kernels import (  # noqa: F401
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
    NVAR,
)

RHO_FLOOR: float = 1e-12
P_FLOOR: float = 1e-12

# Speed of light for cf clamping in fast_magnetosonic
_C_LIGHT: float = 3e8
_CF_SQ_MAX: float = _C_LIGHT * _C_LIGHT

# Boris correction: reduced speed of light for wave speed limiting.
# v_A' = v_A * c_boris / sqrt(v_A^2 + c_boris^2) bounds Alfven speed
# at c_boris without injecting fake mass. Set to ~5-10x max flow velocity.
# Gombosi et al. 2002, JCP 177:176; Minoshima et al. 2019, ApJ 874:37.
_C_BORIS_DEFAULT: float = 5e5  # 500 km/s — bounds dt at ~dx/5e5

# ---------------------------------------------------------------------------
# Compile cache — populated lazily on first call.
# ---------------------------------------------------------------------------

_COMPILED: dict[str, object] = {}


def _compile_if_available(fn: object) -> object:
    """Wrap *fn* with mx.compile if MLX supports it, else return it unchanged."""
    try:
        return mx.compile(fn)  # type: ignore[attr-defined]
    except Exception:
        return fn


# ---------------------------------------------------------------------------
# Pure elementwise implementations (no Python control flow — compile-safe).
# ---------------------------------------------------------------------------


def _cons_to_prim_impl(
    U: mx.array,
    gamma: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    gm1 = gamma - 1.0
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = mx.reciprocal(rho)
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    p = mx.maximum(gm1 * (U[IEN] - KE - ME), P_FLOOR)
    return rho, vr, vz, vt, p, Br, Bz, Bt


def _recover_pressure_impl(
    U: mx.array,
    gamma: float,
    eta1: float,
    eta2: float,
) -> tuple[mx.array, mx.array]:
    gm1 = gamma - 1.0
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = mx.reciprocal(rho)
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho
    E = U[IEN]
    Srho = U[ISR]
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    p_E = gm1 * (E - KE - ME)
    p_S = Srho * mx.power(rho, gm1)
    E_abs = mx.maximum(mx.abs(E), 1e-30)
    eta = mx.abs(p_S) / E_abs
    denom = max(eta2 - eta1, 1e-30)
    t = mx.clip((eta - eta1) / denom, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    p = mx.maximum(w * p_E + (1.0 - w) * p_S, P_FLOOR)
    return p, w


def cons_to_prim(
    U: mx.array,
    gamma: float = 5.0 / 3.0,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Convert conserved (10, nr, nz) to primitive variables.

    Pressure is recovered via total-energy subtraction only (no dual-energy blend).
    Use recover_pressure_dual_energy for cancellation-safe pressure.

    Args:
        U: Conserved state, shape (10, nr, nz), float32.
        gamma: Adiabatic index (default 5/3).

    Returns:
        Tuple (rho, vr, vz, vt, p, Br, Bz, Bt), each shape (nr, nz).
        rho clamped above RHO_FLOOR, p clamped above P_FLOOR.
    """
    if "cons_to_prim" not in _COMPILED:
        _COMPILED["cons_to_prim"] = _compile_if_available(_cons_to_prim_impl)
    return _COMPILED["cons_to_prim"](U, gamma)  # type: ignore[operator]


def prim_to_cons(
    rho: mx.array,
    vr: mx.array,
    vz: mx.array,
    vt: mx.array,
    p: mx.array,
    Br: mx.array,
    Bz: mx.array,
    Bt: mx.array,
    gamma: float = 5.0 / 3.0,
    e_electron: mx.array | None = None,
) -> mx.array:
    """Convert primitive variables to conserved (10, nr, nz) array.

    Entropy tracer U[ISR] = p * rho^(1-gamma) is computed automatically.
    U[IEE] is set to e_electron if provided, else zeros.

    Args:
        rho: Density, shape (nr, nz).
        vr: Radial velocity, shape (nr, nz).
        vz: Axial velocity, shape (nr, nz).
        vt: Azimuthal velocity, shape (nr, nz).
        p: Thermal pressure, shape (nr, nz).
        Br: Radial B-field, shape (nr, nz).
        Bz: Axial B-field, shape (nr, nz).
        Bt: Azimuthal B-field, shape (nr, nz).
        gamma: Adiabatic index (default 5/3).
        e_electron: Electron energy density, shape (nr, nz).  Optional.

    Returns:
        Conserved state U, shape (10, nr, nz), float32.
    """
    gm1 = gamma - 1.0

    rho_safe = mx.maximum(rho, RHO_FLOOR)
    p_safe = mx.maximum(p, P_FLOOR)

    KE = 0.5 * rho_safe * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    E = p_safe / gm1 + KE + ME

    Srho = p_safe * mx.power(rho_safe, 1.0 - gamma)

    if e_electron is None:
        e_elec = mx.zeros_like(rho_safe)
    else:
        e_elec = mx.maximum(e_electron, 0.0)

    return mx.stack(
        [rho_safe, rho_safe * vr, rho_safe * vz, rho_safe * vt, E, Srho, Br, Bz, Bt, e_elec],
        axis=0,
    )


def recover_pressure_dual_energy(
    U: mx.array,
    gamma: float = 5.0 / 3.0,
    eta1: float = 1e-5,
    eta2: float = 1e-2,
) -> tuple[mx.array, mx.array]:
    """Recover pressure using dual-energy entropy-based switching.

    When kinetic + magnetic energy dominates total energy, the subtraction
    E - KE - ME suffers catastrophic cancellation in float32.  The entropy
    tracer U[ISR] = p * rho^(1-gamma) provides a cancellation-free estimate.
    A cubic smoothstep blends the two based on eta = |p_S| / |E|.

    Switching criterion follows Popovas (2025) / DISPATCH HLLS: eta = p_S / |E|
    avoids the circular dependency in Enzo (e_int/E) and FLASH (e_int/e_kin)
    where the numerator IS the corrupted float32 subtraction.

    Args:
        U: Conserved state, shape (10, nr, nz).
        gamma: Adiabatic index (default 5/3).
        eta1: Lower threshold — below this use entropy only (w=0).
        eta2: Upper threshold — above this use total-energy only (w=1).

    Returns:
        Tuple (pressure, blend_weight), each shape (nr, nz).
        pressure clamped above P_FLOOR.
        blend_weight in [0, 1]: 0 = pure entropy, 1 = pure total-energy.
    """
    if "recover_pressure" not in _COMPILED:
        _COMPILED["recover_pressure"] = _compile_if_available(_recover_pressure_impl)
    return _COMPILED["recover_pressure"](U, gamma, eta1, eta2)  # type: ignore[operator]


def fast_magnetosonic(
    rho: mx.array,
    p: mx.array,
    Br: mx.array,
    Bz: mx.array,
    Bt: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """Compute fast magnetosonic speed for CFL computation.

    cf^2 = 0.5 * [ (a^2 + va^2) + sqrt( (a^2 - va^2)^2 + 4*a^2*Bt^2/rho ) ]

    Uses numerically stable discriminant to avoid float32 cancellation.
    Clamps intermediate squared speeds at c^2 = (3e8)^2 to prevent overflow.

    Args:
        rho: Density, shape (nr, nz).
        p: Thermal pressure, shape (nr, nz).
        Br: Radial B-field, shape (nr, nz).
        Bz: Axial B-field, shape (nr, nz).
        Bt: Azimuthal B-field, shape (nr, nz).
        gamma: Adiabatic index.
        dim: Normal direction: 0 = radial (Bn=Br), 1 = axial (Bn=Bz).

    Returns:
        Fast magnetosonic speed cf, shape (nr, nz).
    """
    rho_safe = mx.maximum(rho, RHO_FLOOR)
    p_safe = mx.maximum(p, P_FLOOR)
    inv_rho = mx.reciprocal(rho_safe)

    a_sq = gamma * p_safe * inv_rho
    B_sq = Br * Br + Bz * Bz + Bt * Bt

    if dim == 0:
        Bn = Br
    else:
        Bn = Bz

    Bn_sq = Bn * Bn
    Bt_sq = mx.maximum(B_sq - Bn_sq, 0.0)

    va_sq = B_sq * inv_rho
    vat_sq = Bt_sq * inv_rho

    # Clamp to prevent float32 overflow at vacuum boundaries (B_HL ~ 2e4, rho ~ 1e-12)
    cf_sq_max = mx.array(_CF_SQ_MAX, dtype=rho.dtype)
    a_sq = mx.minimum(a_sq, cf_sq_max)
    va_sq = mx.minimum(va_sq, cf_sq_max)
    vat_sq = mx.minimum(vat_sq, cf_sq_max)

    diff = a_sq - va_sq
    discriminant = mx.maximum(diff * diff + 4.0 * a_sq * vat_sq, 0.0)

    cf_sq = mx.minimum(0.5 * (a_sq + va_sq + mx.sqrt(discriminant)), cf_sq_max)
    cf_sq = mx.maximum(cf_sq, 0.0)

    return mx.sqrt(cf_sq)


def fast_magnetosonic_boris(
    rho: mx.array,
    p: mx.array,
    Br: mx.array,
    Bz: mx.array,
    Bt: mx.array,
    gamma: float,
    dim: int,
    c_boris: float = _C_BORIS_DEFAULT,
) -> mx.array:
    """Boris-corrected fast magnetosonic speed (Gombosi 2002, Minoshima 2019).

    The Boris correction reduces the effective Alfven speed in low-density
    regions without injecting artificial mass:

        v_A_boris^2 = v_A^2 * c_boris^2 / (v_A^2 + c_boris^2)

    This bounds all wave speeds at c_boris, eliminating the need for
    va_max density floors that corrupt species mass fractions.

    In the limit v_A << c_boris (physical cells): v_A_boris ≈ v_A (unchanged).
    In the limit v_A >> c_boris (vacuum cells): v_A_boris ≈ c_boris (bounded).

    Args:
        rho, p, Br, Bz, Bt, gamma, dim: Same as fast_magnetosonic.
        c_boris: Reduced speed of light [m/s]. Default 5e5 (500 km/s).

    Returns:
        Boris-corrected fast magnetosonic speed cf', shape (nr, nz).
    """
    rho_safe = mx.maximum(rho, RHO_FLOOR)
    p_safe = mx.maximum(p, P_FLOOR)
    inv_rho = mx.reciprocal(rho_safe)

    a_sq = gamma * p_safe * inv_rho
    B_sq = Br * Br + Bz * Bz + Bt * Bt

    if dim == 0:
        Bn = Br
    else:
        Bn = Bz

    Bn_sq = Bn * Bn
    Bt_sq = mx.maximum(B_sq - Bn_sq, 0.0)

    va_sq = B_sq * inv_rho
    vat_sq = Bt_sq * inv_rho

    # Boris correction: v_A'^2 = v_A^2 * c^2 / (v_A^2 + c^2)
    c_sq = c_boris * c_boris
    va_sq_boris = va_sq * c_sq / (va_sq + c_sq)
    vat_sq_boris = vat_sq * c_sq / (vat_sq + c_sq)
    # Sound speed is physical — no Boris correction needed
    a_sq_safe = mx.minimum(a_sq, c_sq)

    diff = a_sq_safe - va_sq_boris
    discriminant = mx.maximum(diff * diff + 4.0 * a_sq_safe * vat_sq_boris, 0.0)

    cf_sq = mx.minimum(0.5 * (a_sq_safe + va_sq_boris + mx.sqrt(discriminant)), c_sq)
    cf_sq = mx.maximum(cf_sq, 0.0)

    return mx.sqrt(cf_sq)


def entropy_resync(
    U: mx.array,
    p: mx.array,
    div_v: mx.array,
    gamma: float = 5.0 / 3.0,
    dx: float = 1.0,
    sync_threshold: float = 1e-2,
) -> mx.array:
    """Resynchronize entropy tracer at shocks.

    In shocked cells where div_v < -0.33*cs/dx AND |dp|/p > 0.33,
    reset U[ISR] from the total-energy-derived pressure.  Only where
    the subtraction is reliable (eta = |p_S|/|E| > sync_threshold).

    Args:
        U: Conserved state, shape (10, nr, nz).
        p: Blended pressure, shape (nr, nz).  Used for gradient shock detection.
        div_v: Divergence of velocity, shape (nr, nz).  [1/s]
        gamma: Adiabatic index.
        dx: Grid spacing [m].
        sync_threshold: Minimum |p_S|/|E| to trust total-energy subtraction.

    Returns:
        Srho_synced: Updated U[ISR] with shocked cells reset, shape (nr, nz).
        Caller is responsible for writing this back into U[ISR].
    """
    gm1 = gamma - 1.0

    rho = mx.maximum(U[IDN], RHO_FLOOR)
    Srho = U[ISR]
    E = U[IEN]
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    inv_rho = mx.reciprocal(rho)
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
    p_E = gm1 * (E - KE - ME)

    p_safe = mx.maximum(p, P_FLOOR)
    cs = mx.sqrt(mx.maximum(gamma * p_safe * inv_rho, 0.0))

    # Compression criterion: div_v < -0.33 * cs / dx
    compression = div_v < (-0.33 * cs / dx)

    # Steep pressure gradient: |dp|/p > 0.33 (central difference, 2D)
    p_abs = mx.maximum(mx.abs(p), 1e-30)

    # Radial gradient (axis 0), axial gradient (axis 1) via central differences
    grad_r = mx.zeros_like(p)
    grad_z = mx.zeros_like(p)

    p_r_fwd = mx.concatenate([p[1:, :], p[-1:, :]], axis=0)
    p_r_bwd = mx.concatenate([p[:1, :], p[:-1, :]], axis=0)
    grad_r = mx.abs(p_r_fwd - p_r_bwd) / (2.0 * dx)

    p_z_fwd = mx.concatenate([p[:, 1:], p[:, -1:]], axis=1)
    p_z_bwd = mx.concatenate([p[:, :1], p[:, :-1]], axis=1)
    grad_z = mx.abs(p_z_fwd - p_z_bwd) / (2.0 * dx)

    grad_p_mag = mx.sqrt(grad_r * grad_r + grad_z * grad_z)
    steep = (grad_p_mag / p_abs) > 0.33

    is_shock = compression & steep

    # Reliability: |p_S|/|E| > sync_threshold
    p_S_current = Srho * mx.power(rho, gm1)
    E_abs = mx.maximum(mx.abs(E), 1e-30)
    reliable = (mx.abs(p_S_current) / E_abs) > sync_threshold

    sync_mask = is_shock & reliable

    p_E_safe = mx.maximum(p_E, P_FLOOR)
    Srho_from_E = p_E_safe * mx.power(rho, 1.0 - gamma)

    return mx.where(sync_mask, Srho_from_E, Srho)
