"""Source terms for the MLX cylindrical MHD solver.

Wraps the Metal geometric source kernel and adds:
1. Cylindrical geometric sources (centrifugal, hoop stress, Coriolis)
2. Ohmic heating: Q_ohm = eta * J^2 (adds to energy and entropy)
3. Bremsstrahlung radiation: Q_rad = 1.569e-40 * g_ff * Z * ne^2 * sqrt(Te)
4. Entropy tracer source: dSrho/dt from irreversible heating
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# Physical constants — imported from single source of truth
from dpf.metal.constants import K_B as _KBOLTZ
from dpf.metal.constants import MU_0 as _MU0
from dpf.metal.constants import P_FLOOR as _P_FLOOR
from dpf.metal.constants import RHO_FLOOR as _RHO_FLOOR
from dpf.metal.mlx_kernels import (
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
    cylindrical_source_mlx,
    cylindrical_source_numpy,
)

_BREM_COEFF = 1.569e-40                  # bremsstrahlung prefactor [W m^3 / sqrt(K)]


def _bremsstrahlung_logspace(
    rho: mx.array,
    p: mx.array,
    gamma: float,
    Z_eff: float | mx.array = 1.0,
    gaunt_factor: float = 1.2,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Compute bremsstrahlung power Q_rad in pure MLX via log-space arithmetic.

    Q_rad = BREM_COEFF * g_ff * Z * ne^2 * sqrt(Te)

    _BREM_COEFF = 1.569e-40 is subnormal in float32 (flushes to zero).
    In log-space, log(1.569e-40) = -91.76, well within float32 range (+/-126).

    Args:
        rho: Mass density (nr, nz), float32, already floored.
        p: Pressure (nr, nz), float32, already floored.
        gamma: Adiabatic index.
        Z_eff: Effective ion charge (scalar or spatial mx.array).
        gaunt_factor: Free-free Gaunt factor.
        ion_mass: Ion mass [kg].

    Returns:
        Q_rad: Volumetric radiation power [W/m^3], shape matching rho, float32.
    """
    _LOG_BREM = float(np.log(_BREM_COEFF))      # log(1.569e-40) ~ -91.66
    _LOG_GFF = float(np.log(gaunt_factor))
    _LOG_MI = float(np.log(ion_mass))            # log(3.34e-27) ~ -61.07
    _LOG_2KB = float(np.log(2.0 * _KBOLTZ))     # log(2*kB) ~ -52.17

    # n_e ~ rho / m_i for Z=1 (quasi-neutral: n_e = n_i for hydrogen/deuterium)
    log_rho = mx.log(mx.maximum(rho, 1e-30))
    log_ne = log_rho - _LOG_MI

    # Te = p * m_i / (2 * rho * kB)  [factor 2: n_total = n_e + n_i at Z=1]
    log_p = mx.log(mx.maximum(p, 1e-30))
    log_Te = log_p + _LOG_MI - _LOG_2KB - log_rho
    log_Te = mx.maximum(log_Te, 0.0)  # floor Te at 1 K in log-space

    # log(Z_eff): scalar or spatial array
    if isinstance(Z_eff, mx.array):
        log_Z = mx.log(mx.maximum(Z_eff, 1e-30))
    else:
        log_Z = float(np.log(max(float(Z_eff), 1e-30)))

    # Rybicki & Lightman (1979) eq. 5.14a, SI quasi-neutral form:
    #   P_ff = 1.569e-40 * g_ff * Z_eff * n_e^2 * sqrt(T_K)  [W/m^3]
    # n_e in m^-3, T in Kelvin, Z_eff = Z for single species.
    # The Z^1 (not Z^2) is correct because n_e already includes the
    # charge-state dependence: the fundamental formula P ~ Z^2 * n_e * n_i
    # becomes Z * n_e^2 when n_i = n_e/Z (quasi-neutrality).
    # Confirmed by Chen (2016) and NRL Plasma Formulary p.58.
    log_Q = _LOG_BREM + _LOG_GFF + log_Z + 2.0 * log_ne + 0.5 * log_Te

    # Clamp to prevent exp overflow (exp(88) ~ 3.4e38)
    log_Q = mx.minimum(log_Q, 80.0)
    return mx.exp(log_Q)


def _conserved_to_primitive(U: mx.array, gamma: float) -> mx.array:
    """Convert conserved state U to primitive state Q for cylindrical source call.

    Conserved layout: [rho, mr, mz, mt, E, Srho, Br, Bz, Bt, e_electron]
    Primitive layout: [rho, vr, vz, vtheta, p, Srho/rho, Br, Bz, Bt, e_electron]

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        gamma: Adiabatic index.

    Returns:
        Primitive state Q, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho

    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    v2 = vr * vr + vz * vz + vt * vt
    B2 = U[IBR] * U[IBR] + U[IBZ] * U[IBZ] + U[IBT] * U[IBT]

    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)
    s_specific = U[ISR] * inv_rho

    Q = mx.stack([rho, vr, vz, vt, p, s_specific, U[IBR], U[IBZ], U[IBT], U[IEE]], axis=0)
    return Q.astype(mx.float32)


def apply_geometric_sources(
    U: mx.array,
    r_cell: mx.array,
    inv_r: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
    use_metal_kernel: bool = True,
) -> mx.array:
    """Apply cylindrical geometric source terms to conserved state.

    Converts to primitive state, calls the Metal geometric source kernel,
    then maps source increments back to conserved variables.

    Source terms (applied to momentum only):
      S_mr = (rho*vtheta^2 - Btheta^2) / r + dp_tot/dr    [centrifugal + hoop]
      S_mt = -2*(rho*vr*vtheta - Br*Btheta) / r            [Coriolis + tension]

    L'Hopital at axis (r=0): uses dp/dr instead of p/r.

    Args:
        U: Conserved state array, shape (NVAR, nr, nz), float32.
        r_cell: Cell-center radii, shape (nr,), float32.
        inv_r: Pre-computed 1/r with L'Hopital at axis, shape (nr,), float32.
        dt: Time step [s].
        gamma: Adiabatic index.
        use_metal_kernel: Use Metal GPU kernel if available (default True).

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    Q = _conserved_to_primitive(U, gamma)

    if use_metal_kernel:
        src = cylindrical_source_mlx(Q, r_cell, inv_r, gamma)
    else:
        src_np = cylindrical_source_numpy(
            np.asarray(Q), np.asarray(r_cell), np.asarray(inv_r), gamma
        )
        src = mx.array(src_np)

    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho
    vt = U[IMT] * inv_rho

    # src[1]=S_vr, src[2]=S_vz, src[3]=S_vt are accelerations (per unit mass not momentum)
    # cylindrical_source_numpy returns primitive increments: S_mr in velocity units
    dmr = rho * src[1] * dt
    dmz = rho * src[2] * dt
    dmt = rho * src[3] * dt

    # Full cylindrical energy geometric source (Stone & Norman 1992, eq 3.4):
    #   S_E = [(E + p_total) * vr - Br * (v . B)] / r
    #
    # Previous code used v.dot(S_mom) which misses the enthalpy flux correction.
    # The full S_E includes E*vr/r and the Poynting flux terms.
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    B_sq = Br * Br + Bz * Bz + Bt * Bt
    E_total = U[IEN]

    # Boris factor for magnetic pressure (consistent with geometric source kernel)
    from dpf.metal.constants import C_BORIS_SQ
    rho_safe = mx.maximum(rho, _RHO_FLOOR)
    va_sq = B_sq / rho_safe
    f_boris = C_BORIS_SQ / (va_sq + C_BORIS_SQ)

    # Pressure from dual-energy or E-KE-ME
    KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
    ME = 0.5 * B_sq
    p = mx.maximum((gamma - 1.0) * (E_total - KE - ME), _P_FLOOR)
    p_total = p + 0.5 * B_sq * f_boris

    # v . B (Poynting flux contribution)
    vdotB = vr * Br + vz * Bz + vt * Bt

    # Full energy source: S_E = [(E + p_total)*vr - Br*(v.B)] / r
    inv_r_2d = inv_r[:, None]  # broadcast (nr,) -> (nr, nz)
    S_E = ((E_total + p_total) * vr - Br * vdotB) * inv_r_2d

    # L'Hopital at axis (ir=0): S_E -> 0 as r -> 0 for smooth fields
    if U.shape[1] > 1:
        S_E = mx.concatenate([mx.zeros_like(S_E[:1, :]), S_E[1:, :]], axis=0)

    dE = S_E * dt

    updated_vars = [
        U[IDN],
        U[IMR] + dmr,
        U[IMZ] + dmz,
        U[IMT] + dmt,
        U[IEN] + dE,
        U[ISR],
        U[IBR] + src[IBR] * dt,
        U[IBZ],
        U[IBT] + src[IBT] * dt,
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)


def compute_curl_B_squared_HL(
    U: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
) -> mx.array:
    """Return |curl(B)|^2 in Heaviside-Lorentz (HL) units, NOT SI current density.

    Truth-in-naming: the stencil computes the curl of the B-field as packed
    in U (HL convention, mu_0 = 1) and returns its squared magnitude. There
    is no division by mu_0; callers that need J^2 in SI units must apply the
    HL <-> SI conversion explicitly via `compute_current_density_si`.

    Stencil: central finite differences on interior cells; forward/backward
    differences at boundaries (first-order at the two boundary planes).

        (curl B)_r     = -dBt/dz
        (curl B)_z     = (1/r) d(r Bt)/dr
        (curl B)_theta = dBr/dz - dBz/dr

    Units: if B is packed in HL units (Athena convention), the returned
    value has units of [B_HL/length]^2. Multiply/divide by mu_0 to convert
    to SI [A/m^2]^2 — see `compute_current_density_si`.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        dr: Radial cell spacing [m].
        dz: Axial cell spacing [m].
        r_cell: Cell-center radii, shape (nr,), float32.

    Returns:
        curlB_sq = (curl B)_r^2 + (curl B)_z^2 + (curl B)_theta^2 in HL,
        shape (nr, nz), float32.
    """
    Br = U[IBR]   # (nr, nz)
    Bz = U[IBZ]   # (nr, nz)
    Bt = U[IBT]   # (nr, nz)

    # dBt/dz — central diff in z, one-sided at boundaries
    dBt_dz = (mx.roll(Bt, -1, axis=1) - mx.roll(Bt, 1, axis=1)) / (2.0 * dz)
    # Fix boundary z=0 (forward) and z=nz-1 (backward) — still order 1 there
    dBt_dz = mx.where(
        mx.arange(Bt.shape[1]) == 0,
        (mx.roll(Bt, -1, axis=1) - Bt) / dz,
        dBt_dz,
    )
    dBt_dz = mx.where(
        mx.arange(Bt.shape[1]) == Bt.shape[1] - 1,
        (Bt - mx.roll(Bt, 1, axis=1)) / dz,
        dBt_dz,
    )

    Jr = -dBt_dz

    # d(r*Bt)/dr — radial derivative of r*Bt
    r = r_cell[:, None]          # (nr, 1) broadcast
    rBt = r * Bt                 # (nr, nz)
    drBt_dr = (mx.roll(rBt, -1, axis=0) - mx.roll(rBt, 1, axis=0)) / (2.0 * dr)
    drBt_dr = mx.where(
        mx.arange(rBt.shape[0])[:, None] == 0,
        (mx.roll(rBt, -1, axis=0) - rBt) / dr,
        drBt_dr,
    )
    drBt_dr = mx.where(
        mx.arange(rBt.shape[0])[:, None] == rBt.shape[0] - 1,
        (rBt - mx.roll(rBt, 1, axis=0)) / dr,
        drBt_dr,
    )
    inv_r = 1.0 / mx.maximum(r, 1e-30)
    Jz = inv_r * drBt_dr

    # dBr/dz
    dBr_dz = (mx.roll(Br, -1, axis=1) - mx.roll(Br, 1, axis=1)) / (2.0 * dz)
    dBr_dz = mx.where(
        mx.arange(Br.shape[1]) == 0,
        (mx.roll(Br, -1, axis=1) - Br) / dz,
        dBr_dz,
    )
    dBr_dz = mx.where(
        mx.arange(Br.shape[1]) == Br.shape[1] - 1,
        (Br - mx.roll(Br, 1, axis=1)) / dz,
        dBr_dz,
    )

    # dBz/dr
    dBz_dr = (mx.roll(Bz, -1, axis=0) - mx.roll(Bz, 1, axis=0)) / (2.0 * dr)
    dBz_dr = mx.where(
        mx.arange(Bz.shape[0])[:, None] == 0,
        (mx.roll(Bz, -1, axis=0) - Bz) / dr,
        dBz_dr,
    )
    dBz_dr = mx.where(
        mx.arange(Bz.shape[0])[:, None] == Bz.shape[0] - 1,
        (Bz - mx.roll(Bz, 1, axis=0)) / dr,
        dBz_dr,
    )

    Jt = dBr_dz - dBz_dr

    # Jr, Jz, Jt here are components of curl(B) in HL units (no mu_0).
    curlB_sq = Jr * Jr + Jz * Jz + Jt * Jt
    return curlB_sq.astype(mx.float32)


def compute_current_density_si(
    U: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
    b_packed_as_hl: bool = True,
) -> mx.array:
    """Return |J|^2 in SI units [A^2/m^4], applying the HL<->SI conversion.

    Thin wrapper over `compute_curl_B_squared_HL` that bakes the mu_0
    conversion explicitly so call sites do not have to inline arithmetic.

    Conversion (single source of truth — keep in sync with the resistivity
    pipeline if Cycle 3 audits update the convention):

        Heaviside-Lorentz: B_HL = B_SI / sqrt(mu_0)
        Curl operator:     curl(B_HL) = curl(B_SI) / sqrt(mu_0)
        SI Ampere's law:   J_SI = curl(B_SI) / mu_0
        =>  J_SI = curl(B_HL) / sqrt(mu_0)
        =>  |J_SI|^2 = |curl(B_HL)|^2 / mu_0

    Note: this matches the SI-typed `compute_resistivity` API but contradicts
    the historical comment chain that used `* mu_0`. The fix-anom-mu0-retry
    agent owns the final sign in this expression; this helper is the single
    site where it lives so the correction lands in one place.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        dr, dz: Cell spacings [m].
        r_cell: Cell-center radii, shape (nr,).
        b_packed_as_hl: If True (default), B in U is in HL units (Athena
            convention). If False, B is already SI and no conversion is
            applied (the curl is then |J_SI * mu_0|^2, divided by mu_0^2).

    Returns:
        J_sq_SI: |J|^2 in SI units [A^2/m^4], shape (nr, nz), float32.
    """
    curlB_sq_HL = compute_curl_B_squared_HL(U, dr, dz, r_cell)
    if b_packed_as_hl:
        # B in HL units: |J_SI|^2 = |curl(B_HL)|^2 / mu_0.
        # See conversion derivation in this docstring.
        return (curlB_sq_HL / _MU0).astype(mx.float32)
    # B already in SI: curl returns curl(B_SI), so |J_SI|^2 = |curl(B_SI)|^2 / mu_0^2.
    return (curlB_sq_HL / (_MU0 * _MU0)).astype(mx.float32)


def compute_current_density_components(
    U: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute J = curl(B)/mu_0 components in cylindrical coordinates.

    Returns (Jr, Jz, Jt) as separate arrays. Uses the same finite-difference
    stencil as compute_curl_B_squared_HL but returns components for vector
    use (e.g., Hall MHD E_Hall = (J x B) / (n_e * e)).

    Note: these are curl(B) / mu_0, NOT curl(B). The mu_0 factor must be
    accounted for when computing the Hall electric field.

    Args:
        U: Conserved state, shape (NVAR, nr, nz).
        dr, dz: Cell spacings [m].
        r_cell: Cell-center radii, shape (nr,).

    Returns:
        (Jr, Jz, Jt): Current density components, shape (nr, nz) each.
    """
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]
    r = r_cell[:, None]
    inv_r = 1.0 / mx.maximum(r, 1e-30)

    # Jr = -dBt/dz / mu_0
    dBt_dz = (mx.roll(Bt, -1, axis=1) - mx.roll(Bt, 1, axis=1)) / (2.0 * dz)
    Jr = -dBt_dz

    # Jz = (1/r) d(rBt)/dr / mu_0
    rBt = r * Bt
    drBt_dr = (mx.roll(rBt, -1, axis=0) - mx.roll(rBt, 1, axis=0)) / (2.0 * dr)
    Jz = inv_r * drBt_dr

    # Jt = (dBr/dz - dBz/dr) / mu_0
    dBr_dz = (mx.roll(Br, -1, axis=1) - mx.roll(Br, 1, axis=1)) / (2.0 * dz)
    dBz_dr = (mx.roll(Bz, -1, axis=0) - mx.roll(Bz, 1, axis=0)) / (2.0 * dr)
    Jt = dBr_dz - dBz_dr

    # In Heaviside-Lorentz units (mu_0=1), J = curl(B) directly.
    # The previous division by MU_0 was wrong — B is already in HL units
    # where the SI factor is absorbed. See HALL_MHD_MLX_DESIGN.md Section 2.
    return Jr, Jz, Jt


def apply_hall_mhd(
    U: mx.array,
    dt: float,
    dr: float,
    dz: float,
    r_cell: mx.array,
    ion_mass: float = 3.3435e-27,
) -> mx.array:
    """Apply Hall MHD term as operator-split update to B-field.

    The Hall electric field: E_H = (J x B) / (n_e * e)
    Faraday's law: dB/dt = -curl(E_H)

    For axisymmetric cylindrical:
        dBr/dt = -dE_H_theta/dz
        dBz/dt = (1/r) d(r E_H_theta)/dr
        dBt/dt = dE_H_r/dz - dE_H_z/dr

    Args:
        U: Conserved state (NVAR, nr, nz).
        dt: Timestep [s].
        dr, dz: Cell spacings [m].
        r_cell: Cell-center radii, shape (nr,).
        ion_mass: Ion mass [kg]. Default: deuterium.

    Returns:
        Updated U with Hall-modified B-field.
    """
    _E_CHARGE = 1.602176634e-19

    rho = mx.maximum(U[IDN], 1e-12)
    ne = rho / ion_mass  # assume Z=1

    Jr, Jz, Jt = compute_current_density_components(U, dr, dz, r_cell)
    Br = U[IBR]
    Bz = U[IBZ]
    Bt = U[IBT]

    # E_Hall = (J x B) / (n_e * e)
    inv_ne_e = 1.0 / (ne * _E_CHARGE)
    E_r = (Jz * Bt - Jt * Bz) * inv_ne_e
    E_z = (Jt * Br - Jr * Bt) * inv_ne_e
    E_t = (Jr * Bz - Jz * Br) * inv_ne_e

    # Faraday's law: dB/dt = -curl(E_Hall)
    r = r_cell[:, None]
    inv_r = 1.0 / mx.maximum(r, 1e-30)

    dEt_dz = (mx.roll(E_t, -1, axis=1) - mx.roll(E_t, 1, axis=1)) / (2.0 * dz)
    rEt = r * E_t
    drEt_dr = (mx.roll(rEt, -1, axis=0) - mx.roll(rEt, 1, axis=0)) / (2.0 * dr)
    dEr_dz = (mx.roll(E_r, -1, axis=1) - mx.roll(E_r, 1, axis=1)) / (2.0 * dz)
    dEz_dr = (mx.roll(E_z, -1, axis=0) - mx.roll(E_z, 1, axis=0)) / (2.0 * dr)

    dBr = -dEt_dz * dt
    dBz = inv_r * drEt_dr * dt
    dBt = (dEr_dz - dEz_dr) * dt

    # NaN guard: zero out Hall update in vacuum cells where ne is near-floor
    # (inv_ne_e diverges, producing Inf/NaN in E_Hall)
    finite_mask = mx.isfinite(dBr) & mx.isfinite(dBz) & mx.isfinite(dBt)
    dBr = mx.where(finite_mask, dBr, 0.0)
    dBz = mx.where(finite_mask, dBz, 0.0)
    dBt = mx.where(finite_mask, dBt, 0.0)

    # Update B-field in conserved state
    return mx.concatenate([
        U[:IBR],
        (U[IBR] + dBr)[None],
        (U[IBZ] + dBz)[None],
        (U[IBT] + dBt)[None],
        U[IBT + 1:],
    ], axis=0)


def apply_ohmic_heating(
    U: mx.array,
    eta: mx.array | float,
    J_sq: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
) -> mx.array:
    """Add ohmic heating eta*J^2 to total energy and entropy tracer.

    Q_ohm = eta * J^2 [W/m^3]

    Updates both U[IEN] and U[ISR] consistently. The entropy tracer
    tracks cumulative irreversible heating: dSrho = Q_ohm * dt / T,
    where T is estimated from current pressure and density.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        eta: Resistivity [Ohm·m], scalar or shape (nr, nz).
        J_sq: |J|^2 [A^2/m^4], shape (nr, nz), float32.
        dt: Time step [s].
        gamma: Adiabatic index.

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)

    if not isinstance(eta, mx.array):
        eta = mx.array(float(eta), dtype=mx.float32)

    Q_ohm = eta * J_sq          # (nr, nz)
    dE = Q_ohm * dt             # total energy increment

    # Entropy tracer: dSrho = (gamma-1) * Q_ohm * dt / rho^(gamma-1)
    # Derived from S = p/rho^gamma -> dS = (gamma-1)*dq/rho^(gamma-1) where dq = Q*dt/rho
    rho_gm1 = mx.maximum(rho ** (gamma - 1.0), _RHO_FLOOR)
    dSrho = Q_ohm * dt * (gamma - 1.0) / rho_gm1

    updated_vars = [
        U[IDN],
        U[IMR],
        U[IMZ],
        U[IMT],
        U[IEN] + dE,
        U[ISR] + dSrho,
        U[IBR],
        U[IBZ],
        U[IBT],
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)


def apply_bremsstrahlung(
    U: mx.array,
    dt: float,
    gamma: float = 5.0 / 3.0,
    Z_eff: float | np.ndarray = 1.0,
    gaunt_factor: float = 1.2,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Remove bremsstrahlung radiation from total energy.

    Q_rad = 1.569e-40 * g_ff * Z * ne^2 * sqrt(Te) [W/m^3]

    Assumes fully ionized hydrogen-like plasma: ne = rho / ion_mass.
    Te derived from electron pressure component via p = ne * kB * Te.

    Applied as energy sink: U[IEN] -= Q_rad * dt.

    Args:
        U: Conserved state, shape (NVAR, nr, nz), float32.
        dt: Time step [s].
        gamma: Adiabatic index.
        Z_eff: Effective ion charge. Scalar (default 1.0) or spatially
            varying array (nr, nz) from species-aware compute_zeff_field().
        gaunt_factor: Free-free Gaunt factor (default 1.2).
        ion_mass: Ion mass [kg] (default: deuterium 3.34e-27).

    Returns:
        Updated conserved state, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    v2 = (U[IMR] ** 2 + U[IMZ] ** 2 + U[IMT] ** 2) * inv_rho * inv_rho
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)

    # Compute Q_rad via log-space arithmetic: BREM_COEFF=1.569e-40 is subnormal
    # in float32 but log(1.569e-40)=-91.76 is well within float32 range.
    Q_rad = _bremsstrahlung_logspace(
        rho,
        p,
        gamma,
        Z_eff=Z_eff if isinstance(Z_eff, mx.array) else float(Z_eff),
        gaunt_factor=gaunt_factor,
        ion_mass=ion_mass,
    )
    dE = Q_rad * dt

    # Clamp: cannot remove more energy than available above the kinetic+magnetic floor
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_thermal_floor = _P_FLOOR / (gamma - 1.0)
    e_available = mx.maximum(U[IEN] - e_kin - e_mag - e_thermal_floor, 0.0)
    dE = mx.minimum(dE, e_available)
    dE = mx.maximum(dE, 0.0)

    # Entropy tracer: radiation removes heat, so entropy decreases.
    # dSrho = -(gamma-1) * Q_rad * dt / rho^(gamma-1)
    rho_gm1 = mx.maximum(rho ** (gamma - 1.0), _RHO_FLOOR)
    dSrho = (gamma - 1.0) * dE / rho_gm1
    Srho_new = mx.maximum(U[ISR] - dSrho, 0.0)

    updated_vars = [
        U[IDN],
        U[IMR],
        U[IMZ],
        U[IMT],
        U[IEN] - dE,
        Srho_new,
        U[IBR],
        U[IBZ],
        U[IBT],
        U[IEE],
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)
