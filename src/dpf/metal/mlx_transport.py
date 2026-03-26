"""Implicit resistive diffusion and thermal conduction for the MLX solver.

Operator-split transport physics applied after the explicit MHD step.
Uses tridiagonal (Thomas) solver on CPU in float64 for stability.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    Stone J.M., Norman M.L., ApJS 80, 753 (1992) — operator-split resistivity.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

# ── Physical Constants ──────────────────────────────────────────

MU_0: float = 4.0 * math.pi * 1e-7
K_B: float = 1.380649e-23
M_D: float = 3.34358377e-27
P_FLOOR: float = 1e-12


# ── Thomas Tridiagonal Solver ───────────────────────────────────


def thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal system [a, b, c] * x = d.

    Standard Thomas algorithm. CPU float64.

    Parameters
    ----------
    a : np.ndarray
        Lower diagonal, shape (n-1,).
    b : np.ndarray
        Main diagonal, shape (n,).
    c : np.ndarray
        Upper diagonal, shape (n-1,).
    d : np.ndarray
        Right-hand side, shape (n,).

    Returns
    -------
    x : np.ndarray
        Solution, shape (n,).
    """
    n = len(b)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if n == 1:
        return np.array([d[0] / b[0]])

    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    x = np.zeros(n, dtype=np.float64)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ── Internal helpers ────────────────────────────────────────────


def _build_diffusion_system(
    field_col: np.ndarray,
    alpha: np.ndarray,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build implicit tridiagonal system for 1-D diffusion along uniform grid.

    Crank-Nicolson not used here; fully implicit (unconditionally stable):
        (I - dt * D) * x_new = x_old

    where D is the finite-difference Laplacian with spatially varying alpha.

    Parameters
    ----------
    field_col : np.ndarray
        1-D array of field values, shape (n,).
    alpha : np.ndarray
        Diffusivity at each cell, shape (n,).
    dt : float
        Timestep [s].
    dx : float
        Cell spacing [m].

    Returns
    -------
    a, b, c, d : np.ndarray
        Tridiagonal system components.
    """
    r = dt / (dx * dx)

    # Interface diffusivities at right faces: alpha_{i+1/2} = (alpha_i + alpha_{i+1})/2
    # shape (n-1,) for faces 0..n-2
    alpha_face = 0.5 * (alpha[:-1] + alpha[1:])

    # Lower diagonal a[k] = coefficient coupling x[k+1] <- x[k], shape (n-1,)
    a = -r * alpha_face
    a = np.where(np.isfinite(a), a, 0.0)

    # Upper diagonal c[k] = coefficient coupling x[k] <- x[k+1], shape (n-1,)
    c = -r * alpha_face
    c = np.where(np.isfinite(c), c, 0.0)

    # Main diagonal: 1 + r*(alpha_{i-1/2} + alpha_{i+1/2})
    # Neumann BCs: zero flux at boundaries -> zero diffusivity at boundary faces
    alpha_left = np.concatenate([[0.0], alpha_face])   # (n,)
    alpha_right = np.concatenate([alpha_face, [0.0]])  # (n,)
    b = 1.0 + r * (alpha_left + alpha_right)
    b = np.where(np.isfinite(b), b, 1.0)
    b = np.maximum(b, 1.0)

    d = field_col.copy()
    return a, b, c, d


def _build_cylindrical_diffusion_system(
    field_col: np.ndarray,
    alpha: np.ndarray,
    r_cell: np.ndarray,
    dt: float,
    dr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implicit 1-D diffusion in cylindrical r: (1/r) d/dr (r * alpha * dB/dr).

    Parameters
    ----------
    field_col : np.ndarray
        Field values along r, shape (nr,).
    alpha : np.ndarray
        Diffusivity, shape (nr,).
    r_cell : np.ndarray
        Cell-centre radii, shape (nr,).
    dt : float
        Timestep [s].
    dr : float
        Radial spacing [m].

    Returns
    -------
    a, b, c, d : np.ndarray
        Tridiagonal system components.
    """
    r_safe = np.maximum(r_cell, 0.5 * dr)  # avoid axis singularity

    # Face radii
    r_face_p = r_safe + 0.5 * dr
    r_face_m = np.maximum(r_safe - 0.5 * dr, 0.0)

    # Interface diffusivities
    alpha_p = np.concatenate([0.5 * (alpha[:-1] + alpha[1:]), [alpha[-1]]])
    alpha_m = np.concatenate([[alpha[0]], 0.5 * (alpha[:-1] + alpha[1:])])

    coeff_p = dt * r_face_p * alpha_p / (r_safe * dr * dr)
    coeff_m = dt * r_face_m * alpha_m / (r_safe * dr * dr)

    a = -coeff_m[1:]
    a = np.where(np.isfinite(a), a, 0.0)

    c = -coeff_p[:-1]
    c = np.where(np.isfinite(c), c, 0.0)

    b = 1.0 + coeff_p + coeff_m
    b = np.where(np.isfinite(b), b, 1.0)
    b = np.maximum(b, 1.0)

    d = field_col.copy()
    return a, b, c, d


# ── Resistive Diffusion ─────────────────────────────────────────


def apply_resistive_diffusion(
    Br: mx.array,
    Bz: mx.array,
    Bt: mx.array,
    rho: mx.array,
    p: mx.array,
    eta: mx.array | float,
    dt: float,
    dr: float,
    dz: float,
    r_cell: mx.array,
    gamma: float = 5.0 / 3.0,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Apply implicit resistive diffusion to B-field.

    Diffusion equation (cylindrical):
        dBr/dt = eta/mu_0 * (d^2 Br/dz^2 + (1/r) d/dr (r dBr/dr) - Br/r^2)
        dBz/dt = eta/mu_0 * (1/r) d/dr (r dBz/dr)
        dBt/dt = eta/mu_0 * (1/r) d/dr (r dBt/dr) + ...

    Ohmic heating Q = eta * J^2 is accumulated into pressure.
    Implicit solve column-by-column (Thomas algorithm, CPU float64).

    Parameters
    ----------
    Br, Bz, Bt : mx.array
        Magnetic field components, shape (nr, nz). HL code units (B_HL = B_SI/sqrt(mu0)).
    rho : mx.array
        Mass density [kg/m^3], shape (nr, nz).
    p : mx.array
        Gas pressure [Pa], shape (nr, nz).
    eta : mx.array or float
        Resistivity [Ohm*m]. Scalar or shape (nr, nz).
    dt : float
        Timestep [s].
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    r_cell : mx.array
        Cell-centre radii, shape (nr,).
    gamma : float
        Adiabatic index.

    Returns
    -------
    Br_new, Bz_new, Bt_new, p_new : mx.array
        Updated fields.
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required for apply_resistive_diffusion")

    # Move to NumPy for Thomas solver
    Br_np = np.asarray(Br, dtype=np.float64)
    Bz_np = np.asarray(Bz, dtype=np.float64)
    Bt_np = np.asarray(Bt, dtype=np.float64)
    p_np = np.asarray(p, dtype=np.float64)
    r_np = np.asarray(r_cell, dtype=np.float64)
    nr, nz = Br_np.shape

    if isinstance(eta, (int, float)):
        eta_np = np.full((nr, nz), float(eta), dtype=np.float64)
    else:
        eta_np = np.asarray(eta, dtype=np.float64)

    # Diffusivity field: alpha = eta / mu_0  [m^2/s]  (HL: mu_0=1 in code, but we
    # convert B to SI for Ohmic heating, so keep physical alpha here)
    alpha_np = eta_np / MU_0

    Br_new = Br_np.copy()
    Bz_new = Bz_np.copy()
    Bt_new = Bt_np.copy()
    p_new = p_np.copy()

    # ── Diffuse along z (uniform) for each r-index ──────────────
    if nz > 1:
        for ir in range(nr):
            alpha_col = alpha_np[ir, :]
            for field, field_new in [(Br_np, Br_new), (Bz_np, Bz_new), (Bt_np, Bt_new)]:
                a, b, c, d = _build_diffusion_system(field[ir, :], alpha_col, dt, dz)
                field_new[ir, :] = thomas_solve(a, b, c, d)

    # ── Diffuse along r (cylindrical) for each z-index ──────────
    # Feed the z-diffused output (Br_new, Bz_new, Bt_new) into the r-pass so
    # both sweeps are chained (operator-split dimensional factorisation).
    if nr > 1:
        for iz in range(nz):
            alpha_row = alpha_np[:, iz]
            for field_new in [Br_new, Bz_new, Bt_new]:
                a, b, c, d = _build_cylindrical_diffusion_system(
                    field_new[:, iz], alpha_row, r_np, dt, dr
                )
                field_new[:, iz] = thomas_solve(a, b, c, d)

    # ── Ohmic heating: Q = eta * J^2  ───────────────────────────
    # Approximate J^2 from field change: Q ≈ (delta_B)^2 / (mu_0^2 * eta * dt)
    # More precisely: Q = sigma * E^2 = E^2 / eta, and dB/dt = -curl E => E ~ dB*dx/dt
    # Use energy conservation: delta(B^2/2mu_0) is removed from B, add to thermal.
    # B is in HL units (B_HL = B_SI / sqrt(mu_0)), so B^2_HL = B^2_SI / mu_0.
    # Energy in B field (per volume) = 0.5 * B_HL^2  (in HL units where mu_0=1).
    # Ohmic heating = loss in magnetic energy density.
    dB_sq = (
        (Br_new - Br_np) ** 2
        + (Bz_new - Bz_np) ** 2
        + (Bt_new - Bt_np) ** 2
    )
    # Convert to SI pressure units: 0.5 * dB_HL^2 * mu_0  [J/m^3]
    Q_ohmic = 0.5 * dB_sq * MU_0
    # Ohmic heating raises pressure: dp = (gamma-1) * Q
    p_new = np.maximum(p_new + (gamma - 1.0) * Q_ohmic, P_FLOOR)

    return (
        mx.array(Br_new),
        mx.array(Bz_new),
        mx.array(Bt_new),
        mx.array(p_new),
    )


# ── Thermal Conduction ──────────────────────────────────────────


def apply_thermal_conduction(
    Te: mx.array,
    Ti: mx.array,
    rho: mx.array,
    B: mx.array,
    kappa_parallel: float | mx.array,
    dt: float,
    dz: float,
    dr: float | None = None,
    Br: mx.array | None = None,
    Bz: mx.array | None = None,
    Bt: mx.array | None = None,
    anisotropic: bool = True,
) -> tuple[mx.array, mx.array]:
    """Apply Braginskii anisotropic thermal conduction.

    Computes the effective conduction along the r-z plane accounting for
    the magnetic field direction. In axisymmetric DPF, B_theta dominates,
    so the parallel (along-B) direction is primarily toroidal. Conduction
    in the r-z plane comes from:
      - kappa_parallel * (b_r^2 or b_z^2) components (small when B_theta >> B_r,B_z)
      - kappa_perp * (1 - b_r^2 or 1 - b_z^2) components (tiny: ~10^-7 * kappa_par)

    When anisotropic=True, the effective diffusivity in each direction is:
      chi_r = kappa_par * b_r^2 + kappa_perp * (1 - b_r^2)
      chi_z = kappa_par * b_z^2 + kappa_perp * (1 - b_z^2)

    This correctly suppresses cross-field conduction by orders of magnitude
    compared to the isotropic approximation.

    Parameters
    ----------
    Te, Ti : mx.array, shape (nr, nz)
        Electron and ion temperatures [K].
    rho : mx.array, shape (nr, nz)
        Mass density [kg/m^3].
    B : mx.array
        Scalar B magnitude or unused (for backward compatibility).
    kappa_parallel : float or mx.array
        Parallel thermal conductivity [W/(m*K)].
    dt : float
        Timestep [s].
    dz : float
        Axial cell spacing [m].
    dr : float or None
        Radial cell spacing [m]. Required for r-direction conduction.
    Br, Bz, Bt : mx.array or None
        B-field components, shape (nr, nz). If None, falls back to isotropic.
    anisotropic : bool
        If True (default), compute direction-weighted conduction.
        If False, use isotropic kappa_parallel in z only (legacy behavior).

    Returns
    -------
    Te_new, Ti_new : mx.array
        Updated temperatures, shape (nr, nz).
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required for apply_thermal_conduction")

    Te_np = np.asarray(Te, dtype=np.float64)
    Ti_np = np.asarray(Ti, dtype=np.float64)
    rho_np = np.asarray(rho, dtype=np.float64)
    nr, nz = Te_np.shape

    if isinstance(kappa_parallel, (int, float)):
        kappa_np = np.full((nr, nz), float(kappa_parallel), dtype=np.float64)
    else:
        kappa_np = np.asarray(kappa_parallel, dtype=np.float64)

    n_np = np.maximum(rho_np / M_D, 1e-10)

    # Compute anisotropy weighting from B-field direction
    if anisotropic and Br is not None and Bz is not None and Bt is not None:
        Br_np = np.asarray(Br, dtype=np.float64)
        Bz_np = np.asarray(Bz, dtype=np.float64)
        Bt_np = np.asarray(Bt, dtype=np.float64)
        B_mag = np.sqrt(Br_np**2 + Bz_np**2 + Bt_np**2 + 1e-30)
        br = Br_np / B_mag
        bz = Bz_np / B_mag

        # Perpendicular suppression: kappa_perp/kappa_par ~ (omega_ce * tau_e)^{-2}
        # For DPF at pinch: ratio ~ 10^{-7}. Use a floor of 1e-6 for stability.
        kappa_perp_ratio = 1e-6

        # Effective kappa in each direction
        kappa_z = kappa_np * (bz**2 + kappa_perp_ratio * (1.0 - bz**2))
        kappa_r = kappa_np * (br**2 + kappa_perp_ratio * (1.0 - br**2))
    else:
        # Isotropic fallback
        kappa_z = kappa_np
        kappa_r = kappa_np if dr is not None else None

    chi_z = kappa_z / (n_np * K_B)

    Te_new = Te_np.copy()
    Ti_new = Ti_np.copy()

    # z-direction conduction (implicit Thomas per r-column)
    if nz > 1:
        for ir in range(nr):
            a, b, c, d = _build_diffusion_system(Te_np[ir, :], chi_z[ir, :], dt, dz)
            Te_new[ir, :] = np.maximum(thomas_solve(a, b, c, d), 1.0)

            a, b, c, d = _build_diffusion_system(Ti_np[ir, :], chi_z[ir, :], dt, dz)
            Ti_new[ir, :] = np.maximum(thomas_solve(a, b, c, d), 1.0)

    # r-direction conduction (implicit Thomas per z-column)
    if dr is not None and kappa_r is not None and nr > 1:
        chi_r = kappa_r / (n_np * K_B)
        for iz in range(nz):
            a, b, c, d = _build_diffusion_system(Te_new[:, iz], chi_r[:, iz], dt, dr)
            Te_new[:, iz] = np.maximum(thomas_solve(a, b, c, d), 1.0)

            a, b, c, d = _build_diffusion_system(Ti_new[:, iz], chi_r[:, iz], dt, dr)
            Ti_new[:, iz] = np.maximum(thomas_solve(a, b, c, d), 1.0)

    return mx.array(Te_new.astype(np.float32)), mx.array(Ti_new.astype(np.float32))
