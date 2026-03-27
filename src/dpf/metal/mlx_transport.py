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
M_E: float = 9.10938e-31
E_CHARGE: float = 1.602176634e-19
EPS_0: float = 8.854187817e-12
P_FLOOR: float = 1e-12
COULOMB_LOG_DEFAULT: float = 10.0


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


# ── Resistivity Models ─────────────────────────────────────────


def spitzer_resistivity(
    Te: np.ndarray,
    Z_eff: float = 1.0,
    coulomb_log: float = COULOMB_LOG_DEFAULT,
) -> np.ndarray:
    """Spitzer resistivity: eta = 0.51 * m_e * nu_ei / (n_e * e^2).

    Simplified Spitzer formula (NRL Formulary):
        eta = 1.03e-4 * Z * ln(Lambda) / Te^{3/2}  [Ohm*m]

    Valid for T_e > ~10 eV where Coulomb collisions dominate.
    Diverges as T_e -> 0 (partially ionized gas).

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [eV], any shape.
    Z_eff : float
        Effective charge number.
    coulomb_log : float
        Coulomb logarithm (default 10).

    Returns
    -------
    eta : np.ndarray
        Resistivity [Ohm*m], same shape as Te.
    """
    Te_safe = np.maximum(Te, 0.1)  # floor at 0.1 eV to avoid divergence
    return 1.03e-4 * Z_eff * coulomb_log / Te_safe**1.5


def lee_more_resistivity(
    Te: np.ndarray,
    rho: np.ndarray,
    Z_eff: float = 1.0,
    ion_mass: float = M_D,
) -> np.ndarray:
    """Lee-More resistivity model for warm dense plasma.

    Bridges the gap between Spitzer (hot, fully ionized) and cold plasma.
    At T_e >> T_Fermi: recovers Spitzer scaling eta ~ T^{-3/2}.
    At T_e << T_Fermi: saturates at a finite value (no divergence).

    Based on Lee & More (1984), Phys. Fluids 27:1273.
    Simplified implementation following Epperlein (1991) and HYDRA.

    The key insight: electron-ion collision frequency is bounded by the
    inverse of the electron transit time across one ion sphere:
        nu_ei = max(nu_spitzer, v_th / r_ion)

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [eV], shape (nr, nz).
    rho : np.ndarray
        Mass density [kg/m^3], shape (nr, nz).
    Z_eff : float
        Effective charge number.
    ion_mass : float
        Ion mass [kg].

    Returns
    -------
    eta : np.ndarray
        Resistivity [Ohm*m], shape (nr, nz).
    """
    Te_safe = np.maximum(Te, 0.01)  # floor at 0.01 eV
    rho_safe = np.maximum(rho, 1e-20)

    # Number density
    n_i = rho_safe / ion_mass
    n_e = Z_eff * n_i

    # Coulomb logarithm (NRL, capped)
    # ln(Lambda) = 23.5 - ln(n_e^{1/2} * Te^{-5/4}) - sqrt(1e-5 + (ln(Te)-2)^2/16)
    # Simplified: clamp to [2, 20]
    ln_ne = np.log(np.maximum(n_e, 1.0))
    ln_Te = np.log(np.maximum(Te_safe, 0.01))
    coulomb_log = np.clip(23.5 - 0.5 * ln_ne / np.log(10) + 1.25 * ln_Te, 2.0, 20.0)

    # Spitzer collision frequency: nu_ei_spitzer
    # nu_ei = 4 * sqrt(2*pi) * n_e * Z^2 * e^4 * ln(Lambda) /
    #         (3 * (4*pi*eps_0)^2 * m_e^{1/2} * (k_B * Te)^{3/2})
    Te_J = Te_safe * E_CHARGE  # eV -> Joules
    prefactor = 4.0 * math.sqrt(2.0 * math.pi) / 3.0
    denom_const = (4.0 * math.pi * EPS_0) ** 2 * math.sqrt(M_E)
    nu_spitzer = (
        prefactor * n_e * Z_eff**2 * E_CHARGE**4 * coulomb_log
        / (denom_const * (K_B * Te_safe * E_CHARGE / E_CHARGE) ** 1.5)
    )
    # Simplify: Te is in eV, so k_B * Te_K = Te_eV * e_charge
    nu_spitzer = (
        prefactor * n_e * Z_eff**2 * E_CHARGE**4 * coulomb_log
        / (denom_const * Te_J**1.5)
    )

    # Lee-More saturation: electron-ion collision frequency bounded by
    # transit time across Wigner-Seitz ion sphere radius
    # r_WS = (3 / (4*pi*n_i))^{1/3}
    r_ws = (3.0 / (4.0 * math.pi * np.maximum(n_i, 1.0))) ** (1.0 / 3.0)
    # Thermal velocity: v_th = sqrt(k_B * Te / m_e)
    v_th = np.sqrt(Te_J / M_E)
    # Maximum collision frequency: nu_max = v_th / r_ws
    nu_max = v_th / np.maximum(r_ws, 1e-30)

    # Effective collision frequency: harmonic mean blending
    # 1/nu_eff = 1/nu_spitzer + 1/nu_max (smooth transition)
    nu_eff = nu_spitzer * nu_max / (nu_spitzer + nu_max + 1e-30)

    # Resistivity: eta = m_e * nu_eff / (n_e * e^2)
    eta = M_E * nu_eff / (np.maximum(n_e, 1.0) * E_CHARGE**2)

    # Clamp to physical range [1e-10, 1e-2] Ohm*m
    return np.clip(eta, 1e-10, 1e-2)


def compute_resistivity(
    Te: np.ndarray,
    rho: np.ndarray,
    model: str = "lee_more",
    Z_eff: float = 1.0,
    ion_mass: float = M_D,
    eta_floor: float = 1e-10,
    eta_cap: float = 1e-2,
    J_sq: np.ndarray | None = None,
    p: np.ndarray | None = None,
    anomalous_model: str | None = None,
) -> np.ndarray:
    """Compute spatially-varying resistivity from plasma conditions.

    Optionally adds anomalous resistivity (drift-velocity, sagdeev, or lhdi)
    when J_sq and p are provided. Total eta = eta_classical + eta_anomalous.

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [eV], shape (nr, nz).
    rho : np.ndarray
        Mass density [kg/m^3], shape (nr, nz).
    model : str
        Classical model: "lee_more" (default), "spitzer", or "constant".
    J_sq : np.ndarray or None
        |J|^2 [A^2/m^4] for anomalous contribution. None = no anomalous.
    p : np.ndarray or None
        Gas pressure [Pa] for anomalous contribution.
    anomalous_model : str or None
        "drift_velocity", "sagdeev", "lhdi", or None (no anomalous).

    Returns
    -------
    eta : np.ndarray
        Total resistivity [Ohm*m], shape (nr, nz).
    """
    if model == "lee_more":
        eta = lee_more_resistivity(Te, rho, Z_eff=Z_eff, ion_mass=ion_mass)
    elif model == "spitzer":
        eta = spitzer_resistivity(Te, Z_eff=Z_eff)
    elif model == "constant":
        eta = np.full_like(Te, eta_floor)
    else:
        raise ValueError(f"Unknown resistivity model: {model!r}")

    # Add anomalous contribution (additive — independent scattering mechanisms)
    if anomalous_model is not None and J_sq is not None and p is not None:
        eta_anom = anomalous_resistivity(
            J_sq, rho, p, Z_eff=Z_eff, ion_mass=ion_mass, model=anomalous_model,
        )
        eta = eta + eta_anom

    return np.clip(eta, eta_floor, eta_cap)


# ── Anomalous Resistivity ──────────────────────────────────────


def anomalous_resistivity(
    J_sq: np.ndarray,
    rho: np.ndarray,
    p: np.ndarray,
    gamma: float = 5.0 / 3.0,
    Z_eff: float = 1.0,
    ion_mass: float = M_D,
    model: str = "drift_velocity",
    alpha: float = 0.05,
) -> np.ndarray:
    """Compute anomalous resistivity from current-driven micro-instabilities.

    At DPF pinch, drift velocity v_d = J/(n_e*e) exceeds ion thermal speed,
    triggering micro-instabilities that scatter electrons far more effectively
    than Coulomb collisions. Dominates Spitzer/Lee-More by 3-30x at pinch.

    Parameters
    ----------
    J_sq : np.ndarray
        |J|^2 from curl(B), [A^2/m^4], shape (nr, nz).
    rho, p : np.ndarray
        Density [kg/m^3] and pressure [Pa], shape (nr, nz).
    model : str
        "drift_velocity" (Faerder 2024), "sagdeev", or "lhdi".

    Returns
    -------
    eta_anom : np.ndarray
        Anomalous resistivity [Ohm*m], shape (nr, nz). Zero below threshold.
    """
    rho_safe = np.maximum(rho, 1e-20)
    p_safe = np.maximum(p, 1e-12)
    n_i = rho_safe / ion_mass
    n_e = Z_eff * n_i
    J_mag = np.sqrt(np.maximum(J_sq, 0.0))
    v_d = J_mag / np.maximum(n_e * E_CHARGE, 1e-30)
    T_i = p_safe * ion_mass / (2.0 * rho_safe * K_B)
    v_ti = np.sqrt(K_B * np.maximum(T_i, 1.0) / ion_mass)

    if model == "drift_velocity":
        omega_pi = np.sqrt(n_i * E_CHARGE**2 / (EPS_0 * ion_mass))
        ratio_sq = np.minimum((v_d / np.maximum(v_ti, 1.0))**2, 1.0)
        eta_anom = M_E * omega_pi * ratio_sq / np.maximum(n_e * E_CHARGE**2, 1e-60)
        eta_anom = np.where(v_d > v_ti, eta_anom, 0.0)
    elif model == "sagdeev":
        c_s = np.sqrt(K_B * np.maximum(T_i, 1.0) / ion_mass)
        omega_pe = np.sqrt(n_e * E_CHARGE**2 / (EPS_0 * M_E))
        eta_anom = alpha * M_E * omega_pe / np.maximum(n_e * E_CHARGE**2, 1e-60)
        eta_anom = np.where(v_d > c_s, eta_anom, 0.0)
    elif model == "lhdi":
        v_crit = (M_E / ion_mass) ** 0.25 * v_ti
        omega_pe = np.sqrt(n_e * E_CHARGE**2 / (EPS_0 * M_E))
        eta_anom = alpha * M_E * omega_pe / np.maximum(n_e * E_CHARGE**2, 1e-60)
        eta_anom = np.where(v_d > v_crit, eta_anom, 0.0)
    else:
        raise ValueError(f"Unknown anomalous resistivity model: {model!r}")
    return np.clip(eta_anom, 0.0, 1e-2)


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
    # Use cylindrical diffusion operator (1/r)d/dr(r*chi*dT/dr) for correct geometry
    if dr is not None and kappa_r is not None and nr > 1:
        chi_r = kappa_r / (n_np * K_B)
        r_cell = np.array([(ir + 0.5) * dr for ir in range(nr)])
        for iz in range(nz):
            a, b, c, d = _build_cylindrical_diffusion_system(
                Te_new[:, iz], chi_r[:, iz], r_cell, dt, dr,
            )
            Te_new[:, iz] = np.maximum(thomas_solve(a, b, c, d), 1.0)

            a, b, c, d = _build_cylindrical_diffusion_system(
                Ti_new[:, iz], chi_r[:, iz], r_cell, dt, dr,
            )
            Ti_new[:, iz] = np.maximum(thomas_solve(a, b, c, d), 1.0)

    return mx.array(Te_new.astype(np.float32)), mx.array(Ti_new.astype(np.float32))


# ── Flux-Limited Thermal Conduction ────────────────────────────


def flux_limit_kappa(
    kappa: np.ndarray,
    Te: np.ndarray,
    rho: np.ndarray,
    dz: float,
    f_limit: float = 0.1,
    ion_mass: float = M_D,
    Z_eff: float = 1.0,
) -> np.ndarray:
    """Apply free-streaming flux limiter to thermal conductivity.

    When the electron mean free path exceeds the temperature gradient scale
    length, Braginskii conduction overestimates heat flux. The flux limiter
    caps the heat flux at a fraction of the free-streaming value:

        q_limited = min(q_braginskii, f * n_e * k_B * T_e * v_th_e)

    Implemented as an effective kappa reduction:
        kappa_eff = kappa * q_fs / (q_braginskii + q_fs)

    where q_fs = f * n_e * k_B * Te * v_th and q_braginskii ~ kappa * dT/dx.

    Parameters
    ----------
    kappa : np.ndarray
        Unrestricted thermal conductivity [W/(m*K)], shape (nr, nz).
    Te : np.ndarray
        Electron temperature [K], shape (nr, nz).
    rho : np.ndarray
        Mass density [kg/m^3], shape (nr, nz).
    dz : float
        Grid spacing [m] (used to estimate |grad T|).
    f_limit : float
        Flux limiter fraction (default 0.1). Typical range 0.03-0.15.
        0.1 is standard for laser-plasma (Malone et al. 1975).
        0.06 often used for z-pinch (Giuliani & Commisso 2015).
    ion_mass : float
        Ion mass [kg].
    Z_eff : float
        Effective charge number.

    Returns
    -------
    kappa_limited : np.ndarray
        Flux-limited conductivity [W/(m*K)], shape (nr, nz).

    References
    ----------
    Malone, McCrory & Morse (1975), PRL 34:721 — original flux limiter.
    Giuliani & Commisso (2015), PoP 22:032116 — z-pinch flux limiting.
    """
    Te_safe = np.maximum(Te, 1.0)  # floor at 1 K
    rho_safe = np.maximum(rho, 1e-20)

    # Electron number density
    n_e = Z_eff * rho_safe / ion_mass

    # Electron thermal velocity: v_th = sqrt(k_B * Te / m_e)
    v_th = np.sqrt(K_B * Te_safe / M_E)

    # Free-streaming heat flux: q_fs = f * n_e * k_B * Te * v_th
    q_fs = f_limit * n_e * K_B * Te_safe * v_th

    # Estimate Braginskii flux: q_brag ~ kappa * |grad T| ~ kappa * Te / L
    # Use dz as characteristic scale length (conservative estimate)
    grad_T_est = Te_safe / np.maximum(dz, 1e-10)
    q_brag = np.abs(kappa) * grad_T_est

    # Harmonic mean limiting: kappa_eff = kappa * q_fs / (q_brag + q_fs)
    # When q_brag << q_fs: kappa_eff ~ kappa (no limiting)
    # When q_brag >> q_fs: kappa_eff ~ kappa * q_fs / q_brag (flux-limited)
    kappa_limited = kappa * q_fs / (q_brag + q_fs + 1e-30)

    return np.maximum(kappa_limited, 0.0)
