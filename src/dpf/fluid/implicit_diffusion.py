"""Semi-implicit operator-split diffusion solver for resistive and thermal diffusion.

Provides Crank-Nicolson implicit diffusion along each spatial dimension
(ADI splitting) for:

1. **Resistive magnetic diffusion**: dB/dt = (eta / mu_0) * nabla^2(B)
2. **Thermal conduction**: dTe/dt = kappa / (1.5 * ne * k_B) * nabla^2(Te)

The 1D Crank-Nicolson solve produces a tridiagonal system solved via the
Thomas algorithm, giving unconditional stability for arbitrarily large
diffusion coefficients while retaining second-order temporal accuracy.

All performance-critical kernels use ``@njit(cache=True)`` for Numba
JIT compilation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import k_B, mu_0

# ============================================================
# Thomas algorithm for tridiagonal systems
# ============================================================


@njit(cache=True)
def _thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system using the Thomas algorithm.

    Solves  A x = rhs  where A is tridiagonal with bands:
        lower[i] = A[i, i-1]   for i = 1..n-1   (sub-diagonal)
        diag[i]  = A[i, i]     for i = 0..n-1   (main diagonal)
        upper[i] = A[i, i+1]   for i = 0..n-2   (super-diagonal)

    Arrays are modified in place during the forward sweep.

    Args:
        lower: Sub-diagonal, length n. lower[0] is unused.
        diag:  Main diagonal, length n.
        upper: Super-diagonal, length n. upper[n-1] is unused.
        rhs:   Right-hand side vector, length n.

    Returns:
        Solution vector x, length n.
    """
    n = len(diag)
    # Work on copies to avoid mutating caller data
    c = upper.copy()
    d = rhs.copy()
    b = diag.copy()

    # Forward elimination
    for i in range(1, n):
        if b[i - 1] == 0.0:
            continue
        m = lower[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Back substitution
    x = np.empty(n)
    if b[n - 1] == 0.0:
        x[n - 1] = 0.0
    else:
        x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        if b[i] == 0.0:
            x[i] = 0.0
        else:
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# ============================================================
# Crank-Nicolson 1D diffusion solve
# ============================================================


@njit(cache=True)
def diffuse_field_1d(
    field: np.ndarray,
    coeff: np.ndarray,
    dt: float,
    dx: float,
) -> np.ndarray:
    """Crank-Nicolson implicit diffusion solve on a 1D grid.

    Solves the diffusion equation:

        du/dt = d/dx [ D(x) * du/dx ]

    using the Crank-Nicolson (theta = 0.5) discretisation:

        (I - dt/2 * D_h) u^{n+1} = (I + dt/2 * D_h) u^n

    where D_h is the second-order finite-difference diffusion operator
    with spatially-varying coefficient, and Neumann (zero-gradient)
    boundary conditions are applied at both ends.

    Args:
        field: 1D array of the quantity to diffuse (e.g. a B component
               or temperature along one grid line).
        coeff: Diffusion coefficient [m^2/s], 1D array of the same
               length as *field*. For resistive diffusion this is
               ``eta / mu_0``; for thermal diffusion it is
               ``kappa / (1.5 * ne * k_B)``.
        dt:    Timestep [s].
        dx:    Grid spacing [m].

    Returns:
        Updated 1D field array after one implicit diffusion step.
    """
    n = len(field)
    if n < 3:
        return field.copy()

    dx2 = dx * dx

    # Build the interface-centred diffusion coefficient:
    #   D_{i+1/2} = 0.5 * (coeff[i] + coeff[i+1])
    # Used for the standard conservative finite-difference stencil:
    #   (D * d^2u/dx^2)_i ~ [D_{i+1/2}*(u_{i+1}-u_i) - D_{i-1/2}*(u_i-u_{i-1})] / dx^2

    # Pre-compute half-timestep ratio at each interface
    # sigma_{i+1/2} = dt / (2 * dx^2) * D_{i+1/2}
    sigma = np.empty(n - 1)
    for i in range(n - 1):
        sigma[i] = 0.5 * dt * 0.5 * (coeff[i] + coeff[i + 1]) / dx2

    # --- Build RHS: (I + dt/2 * D_h) u^n ---
    rhs = np.empty(n)
    # Interior points
    for i in range(1, n - 1):
        rhs[i] = (
            field[i]
            + sigma[i] * (field[i + 1] - field[i])
            - sigma[i - 1] * (field[i] - field[i - 1])
        )
    # Neumann BC: ghost value equals boundary value => the boundary
    # diffusion flux is zero. This is equivalent to sigma * 0 at the wall.
    rhs[0] = field[0] + sigma[0] * (field[1] - field[0])
    rhs[n - 1] = field[n - 1] - sigma[n - 2] * (field[n - 1] - field[n - 2])

    # --- Build LHS tridiagonal: (I - dt/2 * D_h) ---
    lower = np.zeros(n)
    diag = np.ones(n)
    upper = np.zeros(n)

    for i in range(1, n - 1):
        lower[i] = -sigma[i - 1]
        upper[i] = -sigma[i]
        diag[i] = 1.0 + sigma[i - 1] + sigma[i]

    # Neumann BC rows
    diag[0] = 1.0 + sigma[0]
    upper[0] = -sigma[0]
    lower[n - 1] = -sigma[n - 2]
    diag[n - 1] = 1.0 + sigma[n - 2]

    return _thomas_solve(lower, diag, upper, rhs)


# ============================================================
# 3D ADI resistive diffusion
# ============================================================


def implicit_resistive_diffusion(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    eta: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply implicit resistive magnetic diffusion to all three B components.

    Uses dimension-by-dimension Alternating Direction Implicit (ADI)
    splitting.  Each B component is diffused sequentially along x, then y,
    then z, with the diffusion coefficient ``eta / mu_0``.

    The induction equation's diffusive part is:

        dB/dt = (eta / mu_0) * nabla^2(B)

    Args:
        Bx: x-component of magnetic field [T], shape (nx, ny, nz).
        By: y-component of magnetic field [T], shape (nx, ny, nz).
        Bz: z-component of magnetic field [T], shape (nx, ny, nz).
        eta: Resistivity field [Ohm*m], shape (nx, ny, nz).
        dt: Timestep [s].
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Tuple ``(Bx_new, By_new, Bz_new)`` of updated magnetic field
        components, each with the same shape as the inputs.
    """
    # Diffusion coefficient: eta / mu_0  [m^2/s]
    diff_coeff = eta / mu_0

    spacings = (dx, dy, dz)
    nx, ny, nz = Bx.shape

    components = [Bx.copy(), By.copy(), Bz.copy()]

    for comp_idx in range(3):
        field = components[comp_idx]

        # --- Sweep along x (axis 0) ---
        for j in range(ny):
            for k in range(nz):
                pencil = field[:, j, k]
                coeff_pencil = diff_coeff[:, j, k]
                field[:, j, k] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[0])

        # --- Sweep along y (axis 1) ---
        for i in range(nx):
            for k in range(nz):
                pencil = field[i, :, k]
                coeff_pencil = diff_coeff[i, :, k]
                field[i, :, k] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[1])

        # --- Sweep along z (axis 2) ---
        for i in range(nx):
            for j in range(ny):
                pencil = field[i, j, :]
                coeff_pencil = diff_coeff[i, j, :]
                field[i, j, :] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[2])

        components[comp_idx] = field

    return components[0], components[1], components[2]


# ============================================================
# 3D ADI thermal diffusion
# ============================================================


def implicit_thermal_diffusion(
    Te: np.ndarray,
    kappa: np.ndarray,
    ne: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Apply implicit thermal conduction diffusion to electron temperature.

    Uses dimension-by-dimension ADI splitting with the isotropic thermal
    diffusion coefficient ``kappa / (1.5 * ne * k_B)``.

    The energy equation's conductive part is:

        dTe/dt = (1 / (1.5 * ne * k_B)) * div(kappa * grad(Te))

    which for spatially-varying kappa and ne yields a diffusion equation
    with local diffusivity ``D = kappa / (1.5 * ne * k_B)``.

    Args:
        Te:    Electron temperature [K], shape (nx, ny, nz).
        kappa: Thermal conductivity [W/(m*K)], shape (nx, ny, nz).
        ne:    Electron number density [m^-3], shape (nx, ny, nz).
        dt:    Timestep [s].
        dx:    Grid spacing in x [m].
        dy:    Grid spacing in y [m].
        dz:    Grid spacing in z [m].

    Returns:
        Updated electron temperature array [K], same shape as *Te*.
    """
    # Diffusion coefficient: kappa / (1.5 * ne * k_B)  [m^2/s]
    ne_safe = np.maximum(ne, 1e-20)
    diff_coeff = kappa / (1.5 * ne_safe * k_B)

    spacings = (dx, dy, dz)
    nx, ny, nz = Te.shape

    field = Te.copy()

    # --- Sweep along x (axis 0) ---
    for j in range(ny):
        for k in range(nz):
            pencil = field[:, j, k]
            coeff_pencil = diff_coeff[:, j, k]
            field[:, j, k] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[0])

    # --- Sweep along y (axis 1) ---
    for i in range(nx):
        for k in range(nz):
            pencil = field[i, :, k]
            coeff_pencil = diff_coeff[i, :, k]
            field[i, :, k] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[1])

    # --- Sweep along z (axis 2) ---
    for i in range(nx):
        for j in range(ny):
            pencil = field[i, j, :]
            coeff_pencil = diff_coeff[i, j, :]
            field[i, j, :] = diffuse_field_1d(pencil, coeff_pencil, dt, spacings[2])

    return field


# ============================================================
# Explicit diffusion CFL timestep estimate
# ============================================================


@njit(cache=True)
def diffusion_cfl_dt(
    eta_max: float,
    kappa_max: float,
    ne_min: float,
    dx: float,
) -> float:
    """Compute the explicit-diffusion CFL-limited timestep.

    Returns the minimum of the resistive and thermal diffusion stability
    limits.  This is useful for deciding whether to sub-cycle an explicit
    scheme or to switch to the implicit solver provided by this module.

    Stability limits (forward-Euler, 1D):

        dt_resist  = dx^2 * mu_0 / (2 * eta_max)
        dt_thermal = dx^2 * 1.5 * ne_min * k_B / (2 * kappa_max)

    Args:
        eta_max:   Maximum resistivity in the domain [Ohm*m].
        kappa_max: Maximum thermal conductivity in the domain [W/(m*K)].
        ne_min:    Minimum electron number density in the domain [m^-3].
        dx:        Grid spacing [m].

    Returns:
        CFL-limited timestep [s] (the more restrictive of the two).
    """
    dx2 = dx * dx

    # Resistive CFL
    if eta_max > 0.0:
        dt_resist = dx2 * mu_0 / (2.0 * eta_max)
    else:
        dt_resist = 1e30  # No resistive constraint

    # Thermal CFL
    if kappa_max > 0.0 and ne_min > 0.0:
        dt_thermal = dx2 * 1.5 * ne_min * k_B / (2.0 * kappa_max)
    else:
        dt_thermal = 1e30  # No thermal constraint

    return min(dt_resist, dt_thermal)
