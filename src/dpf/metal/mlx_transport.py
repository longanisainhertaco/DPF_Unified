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
    if nr > 1:
        for iz in range(nz):
            alpha_row = alpha_np[:, iz]
            for field, field_new in [(Br_np, Br_new), (Bz_np, Bz_new), (Bt_np, Bt_new)]:
                a, b, c, d = _build_cylindrical_diffusion_system(
                    field[:, iz], alpha_row, r_np, dt, dr
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
) -> tuple[mx.array, mx.array]:
    """Apply Braginskii parallel thermal conduction along B.

    For axisymmetric DPF, B_theta dominates, making field lines primarily
    in the theta direction.  The B_z component drives parallel conduction
    along z.  We solve the z-direction implicitly per r-column.

    Energy equation: d(nkT)/dt = d/dz (kappa_par * dT/dz)
    Re-written per temperature: dT/dt = kappa / (n*k_B) * d^2T/dz^2

    Parameters
    ----------
    Te : mx.array
        Electron temperature [K], shape (nr, nz).
    Ti : mx.array
        Ion temperature [K], shape (nr, nz).
    rho : mx.array
        Mass density [kg/m^3], shape (nr, nz).
    B : mx.array
        Not used directly; reserved for anisotropy weighting. Shape (nr, nz).
    kappa_parallel : float or mx.array
        Parallel thermal conductivity [W/(m*K)]. Scalar or shape (nr, nz).
    dt : float
        Timestep [s].
    dz : float
        Axial cell spacing [m].

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

    # Number density: n = rho / m_D
    n_np = np.maximum(rho_np / M_D, 1e-10)

    # Thermal diffusivity: chi = kappa / (n * k_B)  [m^2/s]
    chi_e = kappa_np / (n_np * K_B)
    chi_i = kappa_np / (n_np * K_B)

    Te_new = Te_np.copy()
    Ti_new = Ti_np.copy()

    if nz <= 1:
        return mx.array(Te_new), mx.array(Ti_new)

    for ir in range(nr):
        # Electron conduction along z
        chi_col = chi_e[ir, :]
        a, b, c, d = _build_diffusion_system(Te_np[ir, :], chi_col, dt, dz)
        Te_new[ir, :] = np.maximum(thomas_solve(a, b, c, d), 1.0)

        # Ion conduction along z
        chi_col_i = chi_i[ir, :]
        a, b, c, d = _build_diffusion_system(Ti_np[ir, :], chi_col_i, dt, dz)
        Ti_new[ir, :] = np.maximum(thomas_solve(a, b, c, d), 1.0)

    return mx.array(Te_new), mx.array(Ti_new)
