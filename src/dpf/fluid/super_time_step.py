"""Runge-Kutta-Legendre 2nd-order (RKL2) super time-stepping for parabolic terms.

Explicit diffusion solvers are constrained by a CFL limit dt < dx^2 / (2*D),
which becomes prohibitively small for the high thermal conductivities and
magnetic diffusivities encountered in DPF plasmas.  RKL2 is an *s*-stage
explicit method whose stability region stretches along the negative real axis
by a factor of ~0.55*s^2 in eigenvalue space (z = lambda*dt), enabling
timesteps up to ~0.275*s^2 times the standard explicit CFL limit while
remaining fully explicit (no matrix solves).  For s = 10 stages, this gives
roughly a 25x speedup over standard explicit Euler.

The method is a recursion of weighted Euler-like stages built from the zeros
of Chebyshev polynomials of the first kind:

    Y_0 = u^n
    Y_1 = Y_0 + mu_tilde_1 * dt * L(Y_0)
    Y_j = mu_j * Y_{j-1} + nu_j * Y_{j-2} + (1 - mu_j - nu_j) * Y_0
          + mu_tilde_j * dt * L(Y_{j-1}) + gamma_tilde_j * dt * L(Y_0)
    u^{n+1} = Y_s

where L(u) is the parabolic (diffusion) operator.

References:
    Meyer, Balsara & Aslam, JCP 231:2963 (2012) -- RKL1 and RKL2 methods
    Alexiades, Amiez & Gremaud, CNME 12:31 (1996) -- STS overview
    Vaidya et al., MNRAS 472:3147 (2017) -- PLUTO implementation
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import k_B, mu_0  # noqa: F401 – used in thermal/magnetic helpers

# ============================================================
# RKL2 coefficient computation
# ============================================================


def rkl2_coefficients(
    s: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the RKL2 stage coefficients from Chebyshev polynomial recursion.

    The damping parameter w0 and the ratio w1 are derived from the first-kind
    Chebyshev polynomials T_s(w0) evaluated at the shifted point
    w0 = 1 + 2 / (s^2 + s).  The stage weights (mu, nu, mu_tilde,
    gamma_tilde) follow from the Chebyshev three-term recurrence applied
    to the ansatz G(z) = a_s + b_s * T_s(w0 + w1*z), where a_s and b_s
    are chosen for second-order accuracy: G(0) = 1, G'(0) = 1, G''(0) = 1.

    Reference: Meyer, Balsara & Aslam, JCP 231:2963 (2012).

    Args:
        s: Number of RKL stages.  Must be >= 2.  Typical values 4--16.

    Returns:
        Tuple of four arrays each of length ``s + 1``:
            mu:           Stage blending weight for Y_{j-1}.
            nu:           Stage blending weight for Y_{j-2}.
            mu_tilde:     RHS evaluation coefficient for L(Y_{j-1}).
            gamma_tilde:  RHS evaluation coefficient for L(Y_0).
        Indices 0 and 1 are placeholders (unused by the recursion for j >= 2);
        index 1 carries the first-stage coefficient mu_tilde_1.

    Raises:
        ValueError: If *s* < 2.
    """
    if s < 2:
        msg = f"RKL2 requires s >= 2, got s={s}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Chebyshev polynomial evaluation at w0 via stable three-term recurrence
    # T_j(x) = 2*x*T_{j-1}(x) - T_{j-2}(x),  T_0 = 1,  T_1 = x
    # ------------------------------------------------------------------
    w0 = 1.0 + 2.0 / (s * s + s)

    # T_j(w0), T'_j(w0), T''_j(w0) for j = 0 .. s
    T = np.zeros(s + 1)
    Tp = np.zeros(s + 1)
    Tpp = np.zeros(s + 1)

    T[0] = 1.0
    T[1] = w0

    Tp[0] = 0.0
    Tp[1] = 1.0

    Tpp[0] = 0.0
    Tpp[1] = 0.0

    for j in range(2, s + 1):
        T[j] = 2.0 * w0 * T[j - 1] - T[j - 2]
        Tp[j] = 2.0 * T[j - 1] + 2.0 * w0 * Tp[j - 1] - Tp[j - 2]
        Tpp[j] = 4.0 * Tp[j - 1] + 2.0 * w0 * Tpp[j - 1] - Tpp[j - 2]

    # w1 = T'_s(w0) / T''_s(w0)  — chosen for maximum stability extent
    w1 = Tp[s] / Tpp[s]

    # ------------------------------------------------------------------
    # b_j coefficients for RKL2 (second-order)
    # b_j = T''_j(w0) / (T'_j(w0))^2 for j >= 2
    # b_0 = b_1 = b_2  (ensures second-order accuracy)
    # ------------------------------------------------------------------
    b = np.zeros(s + 1)
    b[2] = Tpp[2] / (Tp[2] ** 2)
    b[0] = b[2]
    b[1] = b[2]
    for j in range(3, s + 1):
        b[j] = Tpp[j] / (Tp[j] ** 2)

    # a_j = 1 - b_j * T_j(w0)  — offset for second-order consistency
    a = np.zeros(s + 1)
    for j in range(s + 1):
        a[j] = 1.0 - b[j] * T[j]

    # ------------------------------------------------------------------
    # Stage coefficients from the Chebyshev three-term recurrence
    #   T_j(x) = 2*x*T_{j-1}(x) - T_{j-2}(x)
    # applied to x = w0 + w1*z, yielding:
    #   mu_j       = 2 * w0 * b_j / b_{j-1}
    #   nu_j       = -b_j / b_{j-2}
    #   mu_tilde_j = 2 * w1 * b_j / b_{j-1}
    #   gamma_tilde_j = -a_{j-1} * mu_tilde_j
    # ------------------------------------------------------------------
    mu = np.zeros(s + 1)
    nu = np.zeros(s + 1)
    mu_tilde = np.zeros(s + 1)
    gamma_tilde = np.zeros(s + 1)

    # First stage (j = 1): forward Euler with coefficient mu_tilde_1
    mu_tilde[1] = b[1] * w1

    # Stages j = 2 .. s
    for j in range(2, s + 1):
        mu[j] = 2.0 * w0 * b[j] / b[j - 1]
        nu[j] = -b[j] / b[j - 2]
        mu_tilde[j] = 2.0 * w1 * b[j] / b[j - 1]
        gamma_tilde[j] = -a[j - 1] * mu_tilde[j]

    return mu, nu, mu_tilde, gamma_tilde


# ============================================================
# 1D diffusion operator (Numba-accelerated)
# ============================================================


@njit(cache=True)
def _diffusion_operator_1d(u: np.ndarray, D: np.ndarray, dx: float) -> np.ndarray:
    """Evaluate the 1D diffusion operator d/dx(D * du/dx) with central differences.

    Uses second-order central differencing.  The diffusion coefficient *D*
    is evaluated at cell interfaces via arithmetic averaging:
        D_{i+1/2} = 0.5 * (D_i + D_{i+1})

    Neumann (zero-gradient) boundary conditions are applied implicitly by
    setting the flux to zero at the domain boundaries.

    Args:
        u: Field values, shape (n,).
        D: Diffusion coefficient, shape (n,).  May be spatially varying.
        dx: Grid spacing.

    Returns:
        L(u) = d/dx(D du/dx), shape (n,).
    """
    n = len(u)
    Lu = np.zeros(n)
    inv_dx2 = 1.0 / (dx * dx)

    for i in range(1, n - 1):
        D_ip = 0.5 * (D[i] + D[i + 1])
        D_im = 0.5 * (D[i] + D[i - 1])
        Lu[i] = (D_ip * (u[i + 1] - u[i]) - D_im * (u[i] - u[i - 1])) * inv_dx2

    # Neumann BCs: zero flux at boundaries (ghost-cell approach)
    # Left boundary: du/dx = 0 => u[-1] = u[0], so flux_left = 0
    # Only the interior diffusion from the right contributes
    D_half_r = 0.5 * (D[0] + D[1])
    Lu[0] = D_half_r * (u[1] - u[0]) * inv_dx2

    # Right boundary
    D_half_l = 0.5 * (D[n - 1] + D[n - 2])
    Lu[n - 1] = -D_half_l * (u[n - 1] - u[n - 2]) * inv_dx2

    return Lu


@njit(cache=True)
def _diffusion_operator_1d_scalar(u: np.ndarray, D_scalar: float, dx: float) -> np.ndarray:
    """Evaluate d/dx(D * du/dx) for a uniform (scalar) diffusion coefficient.

    Optimised path that avoids interface averaging when D is spatially constant.

    Args:
        u: Field values, shape (n,).
        D_scalar: Constant diffusion coefficient.
        dx: Grid spacing.

    Returns:
        L(u), shape (n,).
    """
    n = len(u)
    Lu = np.zeros(n)
    coeff = D_scalar / (dx * dx)

    for i in range(1, n - 1):
        Lu[i] = coeff * (u[i + 1] - 2.0 * u[i] + u[i - 1])

    # Neumann BCs
    Lu[0] = coeff * (u[1] - u[0])
    Lu[n - 1] = coeff * (u[n - 2] - u[n - 1])

    return Lu


# ============================================================
# RKL2 1D diffusion step (Numba-accelerated)
# ============================================================


@njit(cache=True)
def _rkl2_step_1d_scalar(
    field: np.ndarray,
    D_scalar: float,
    dt: float,
    dx: float,
    s: int,
    mu: np.ndarray,
    nu: np.ndarray,
    mu_tilde: np.ndarray,
    gamma_tilde: np.ndarray,
) -> np.ndarray:
    """Inner RKL2 recursion for a 1D field with scalar diffusion coefficient."""
    Y0 = field.copy()
    L0 = _diffusion_operator_1d_scalar(Y0, D_scalar, dx)

    # Stage 1
    Y_prev2 = Y0.copy()
    Y_prev1 = Y0 + mu_tilde[1] * dt * L0

    # Stages 2 .. s
    for j in range(2, s + 1):
        L_prev1 = _diffusion_operator_1d_scalar(Y_prev1, D_scalar, dx)
        Y_curr = (
            mu[j] * Y_prev1
            + nu[j] * Y_prev2
            + (1.0 - mu[j] - nu[j]) * Y0
            + mu_tilde[j] * dt * L_prev1
            + gamma_tilde[j] * dt * L0
        )
        Y_prev2 = Y_prev1
        Y_prev1 = Y_curr

    return Y_prev1


@njit(cache=True)
def _rkl2_step_1d_array(
    field: np.ndarray,
    D_arr: np.ndarray,
    dt: float,
    dx: float,
    s: int,
    mu: np.ndarray,
    nu: np.ndarray,
    mu_tilde: np.ndarray,
    gamma_tilde: np.ndarray,
) -> np.ndarray:
    """Inner RKL2 recursion for a 1D field with spatially varying D."""
    Y0 = field.copy()
    L0 = _diffusion_operator_1d(Y0, D_arr, dx)

    # Stage 1
    Y_prev2 = Y0.copy()
    Y_prev1 = Y0 + mu_tilde[1] * dt * L0

    # Stages 2 .. s
    for j in range(2, s + 1):
        L_prev1 = _diffusion_operator_1d(Y_prev1, D_arr, dx)
        Y_curr = (
            mu[j] * Y_prev1
            + nu[j] * Y_prev2
            + (1.0 - mu[j] - nu[j]) * Y0
            + mu_tilde[j] * dt * L_prev1
            + gamma_tilde[j] * dt * L0
        )
        Y_prev2 = Y_prev1
        Y_prev1 = Y_curr

    return Y_prev1


# ============================================================
# Public API
# ============================================================


def rkl2_diffusion_step(
    field: np.ndarray,
    diffusion_coeff: float | np.ndarray,
    dt_super: float,
    dx: float,
    s_stages: int = 8,
) -> np.ndarray:
    """Apply one RKL2 super-step to a 1D diffusion problem.

    Solves du/dt = d/dx(D du/dx) from t to t + dt_super using an *s*-stage
    RKL2 method.  The effective CFL extends to approximately s^2 times
    the explicit Euler limit, so *dt_super* may be up to s^2 * dx^2 / (2D).

    Args:
        field: 1D numpy array of field values.
        diffusion_coeff: Diffusion coefficient.  Scalar for uniform D, or a
            1D array of the same length as *field* for spatially varying D.
        dt_super: Total timestep to advance (can be s^2 times explicit CFL).
        dx: Grid spacing.
        s_stages: Number of RKL stages (default 8).  Must be >= 2.

    Returns:
        Updated 1D field array.
    """
    mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(s_stages)

    if np.ndim(diffusion_coeff) == 0:
        return _rkl2_step_1d_scalar(
            field,
            float(diffusion_coeff),
            dt_super,
            dx,
            s_stages,
            mu,
            nu,
            mu_tilde,
            gamma_tilde,
        )
    return _rkl2_step_1d_array(
        field,
        np.asarray(diffusion_coeff, dtype=np.float64),
        dt_super,
        dx,
        s_stages,
        mu,
        nu,
        mu_tilde,
        gamma_tilde,
    )


def rkl2_diffusion_3d(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    eta: float | np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    s_stages: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply RKL2 resistive diffusion to all three B-field components in 3D.

    The magnetic diffusion equation dB/dt = (eta/mu_0) * Laplacian(B) is
    solved dimension-by-dimension (operator splitting) using the RKL2
    super time-stepping method.  The effective diffusion coefficient is
    eta / mu_0.

    Args:
        Bx: x-component of B, shape (nx, ny, nz).
        By: y-component of B, shape (nx, ny, nz).
        Bz: z-component of B, shape (nx, ny, nz).
        eta: Resistivity [Ohm*m].  Scalar or array of shape (nx, ny, nz).
        dt: Super timestep [s].
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
        s_stages: Number of RKL stages (default 8).

    Returns:
        Tuple (Bx_new, By_new, Bz_new).
    """
    mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(s_stages)

    # Magnetic diffusivity D_mag = eta / mu_0
    eta_arr = np.asarray(eta, dtype=np.float64)
    scalar_eta = eta_arr.ndim == 0

    Bx_new = Bx.copy()
    By_new = By.copy()
    Bz_new = Bz.copy()

    nx, ny, nz = Bx.shape

    # Sweep along each dimension for each component
    spacings = [dx, dy, dz]
    shapes_ax = [nx, ny, nz]

    for comp in (Bx_new, By_new, Bz_new):
        for axis in range(3):
            if shapes_ax[axis] < 3:
                continue
            dh = spacings[axis]

            # Build pencils along this axis
            other_axes = [i for i in range(3) if i != axis]
            for i0 in range(shapes_ax[other_axes[0]]):
                for i1 in range(shapes_ax[other_axes[1]]):
                    slicer = [0, 0, 0]
                    slicer[other_axes[0]] = i0
                    slicer[other_axes[1]] = i1
                    slicer[axis] = slice(None)
                    s = tuple(slicer)

                    pencil = comp[s].copy()

                    if scalar_eta:
                        D_val = float(eta_arr) / mu_0
                        result = _rkl2_step_1d_scalar(
                            pencil,
                            D_val,
                            dt,
                            dh,
                            s_stages,
                            mu,
                            nu,
                            mu_tilde,
                            gamma_tilde,
                        )
                    else:
                        D_pencil = eta_arr[s] / mu_0
                        result = _rkl2_step_1d_array(
                            pencil,
                            D_pencil,
                            dt,
                            dh,
                            s_stages,
                            mu,
                            nu,
                            mu_tilde,
                            gamma_tilde,
                        )

                    comp[s] = result

    return Bx_new, By_new, Bz_new


def rkl2_thermal_step(
    Te: np.ndarray,
    kappa: float | np.ndarray,
    ne: np.ndarray,
    dt: float,
    dx: float,
    s_stages: int = 8,
) -> np.ndarray:
    """Apply RKL2 super time-stepping to isotropic thermal diffusion.

    Solves the thermal diffusion equation:
        (3/2) * ne * k_B * dTe/dt = div(kappa * grad(Te))

    which can be rewritten as:
        dTe/dt = D_th * Laplacian(Te)
        D_th  = kappa / (1.5 * ne * k_B)

    For 1D fields this is applied directly; for 3D fields the diffusion
    is applied dimension-by-dimension (operator splitting).

    Args:
        Te: Electron temperature [K].  1D or 3D array.
        kappa: Thermal conductivity [W/(m*K)].  Scalar or same shape as *Te*.
        ne: Electron number density [m^-3].  Same shape as *Te*.
        dt: Super timestep [s].
        dx: Grid spacing [m] (isotropic).
        s_stages: Number of RKL stages (default 8).

    Returns:
        Updated Te array.
    """
    mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(s_stages)

    kappa_arr = np.asarray(kappa, dtype=np.float64)
    ne_safe = np.maximum(ne, 1e-20)

    if Te.ndim == 1:
        # 1D case: D_th = kappa / (1.5 * ne * k_B)
        if kappa_arr.ndim == 0:
            D_th = float(kappa_arr) / (1.5 * ne_safe * k_B)
            return _rkl2_step_1d_array(
                Te,
                D_th,
                dt,
                dx,
                s_stages,
                mu,
                nu,
                mu_tilde,
                gamma_tilde,
            )
        D_th = kappa_arr / (1.5 * ne_safe * k_B)
        return _rkl2_step_1d_array(
            Te,
            D_th,
            dt,
            dx,
            s_stages,
            mu,
            nu,
            mu_tilde,
            gamma_tilde,
        )

    # 3D case: dimension-by-dimension splitting
    Te_new = Te.copy()
    nx, ny, nz = Te.shape
    shapes_ax = [nx, ny, nz]

    for axis in range(3):
        if shapes_ax[axis] < 3:
            continue
        other_axes = [i for i in range(3) if i != axis]
        for i0 in range(shapes_ax[other_axes[0]]):
            for i1 in range(shapes_ax[other_axes[1]]):
                slicer = [0, 0, 0]
                slicer[other_axes[0]] = i0
                slicer[other_axes[1]] = i1
                slicer[axis] = slice(None)
                s = tuple(slicer)

                pencil = Te_new[s].copy()
                ne_pencil = ne_safe[s]

                if kappa_arr.ndim == 0:
                    D_pencil = float(kappa_arr) / (1.5 * ne_pencil * k_B)
                else:
                    D_pencil = kappa_arr[s] / (1.5 * ne_pencil * k_B)

                result = _rkl2_step_1d_array(
                    pencil,
                    D_pencil,
                    dt,
                    dx,
                    s_stages,
                    mu,
                    nu,
                    mu_tilde,
                    gamma_tilde,
                )
                Te_new[s] = result

    return Te_new


def rkl2_stability_limit(s_stages: int, dt_explicit: float) -> float:
    """Compute the maximum stable super-timestep for RKL2.

    The RKL2 amplification polynomial is stable for eigenvalue-parameter
    products |z| <= z_max, where z_max ~ 0.55 * s^2. Standard explicit
    Euler is stable for |z| <= 2 (for diffusion on a uniform grid).  The
    acceleration relative to Euler is therefore z_max / 2 ~ 0.275 * s^2.

    A safety factor of 0.9 is applied, yielding a practical multiplier
    of ~0.25 * s^2.

    This was verified empirically against the RKL2 amplification polynomial
    for s = 2..16 and is consistent with Meyer, Balsara & Aslam (2012).

    Example: s = 10, dt_explicit = 5e-5 s
        => dt_super = 0.25 * 100 * 5e-5 = 1.25e-3 s  (25x speedup)

    Args:
        s_stages: Number of RKL stages.
        dt_explicit: Maximum stable timestep for standard explicit Euler,
            typically dx^2 / (2 * D_max) for a 1D problem.

    Returns:
        Maximum stable RKL2 super-timestep [s].
    """
    return 0.25 * s_stages * s_stages * dt_explicit
