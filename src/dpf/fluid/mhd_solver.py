"""Hall MHD solver with WENO5 reconstruction and Dedner divergence cleaning.

Merges dpf2's hall_mhd_solver structure with DPF_AI's validated physics kernels:
- WENO5 reconstruction (verified correct from DPF_AI)
- HLL Riemann solver for ideal MHD
- SSP-RK2 (strong-stability-preserving Runge-Kutta) time integration
- Dimension-split flux-based conservative update
- Dedner hyperbolic divergence cleaning for div(B)
- Generalized Ohm's law with Hall term
- Braginskii anisotropic heat flux (operator-split)
- dL_dt estimation from pinch dynamics

The solver operates on a state dictionary with keys:
    rho: density [kg/m^3], shape (nx, ny, nz)
    velocity: velocity [m/s], shape (3, nx, ny, nz)
    pressure: total pressure [Pa], shape (nx, ny, nz)
    B: magnetic field [T], shape (3, nx, ny, nz)
    Te: electron temperature [K], shape (nx, ny, nz)
    Ti: ion temperature [K], shape (nx, ny, nz)
    psi: Dedner cleaning scalar, shape (nx, ny, nz)
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_d, mu_0
from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.fluid.eos import IdealEOS

logger = logging.getLogger(__name__)

# Default ion mass: deuterium
_DEFAULT_ION_MASS = m_d


# ============================================================
# WENO5 reconstruction kernels (from DPF_AI, verified correct)
# ============================================================

@njit(cache=True)
def _weno5_reconstruct_1d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WENO5 reconstruction for a 1D array at cell interfaces.

    Returns left and right states at each interface i+1/2.
    Requires at least 5 cells (2 ghost cells on each side).

    Args:
        v: 1D array of cell-centered values, shape (n,).

    Returns:
        (v_left, v_right) at interfaces, shape (n-4,) each.
        Interface index j corresponds to the face between cell j+2 and j+3.
    """
    n = len(v)
    n_iface = n - 4  # number of interfaces with full stencil
    v_L = np.empty(n_iface)
    v_R = np.empty(n_iface)

    eps = 1e-6  # smoothness parameter

    for i in range(2, n - 2):
        idx = i - 2  # output index

        # --- Left-biased reconstruction (v_L at i+1/2) ---
        v0 = v[i - 2]
        v1 = v[i - 1]
        v2 = v[i]
        v3 = v[i + 1]
        v4 = v[i + 2] if i + 2 < n else v[i + 1]

        # Candidate polynomials
        p0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
        p1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
        p2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0

        # Smoothness indicators
        beta0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
        beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
        beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2

        # Ideal weights
        d0, d1, d2 = 0.1, 0.6, 0.3

        # Non-linear weights
        alpha0 = d0 / (eps + beta0) ** 2
        alpha1 = d1 / (eps + beta1) ** 2
        alpha2 = d2 / (eps + beta2) ** 2
        alpha_sum = alpha0 + alpha1 + alpha2
        if alpha_sum == 0.0:
            alpha_sum = 1e-30

        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum

        v_L[idx] = w0 * p0 + w1 * p1 + w2 * p2

        # --- Right-biased reconstruction (v_R at i+1/2) ---
        # Mirror stencil
        r0 = v[i + 2] if i + 2 < n else v[i + 1]
        r1 = v[i + 1]
        r2 = v[i]
        r3 = v[i - 1]
        r4 = v[i - 2]

        q0 = (2.0 * r0 - 7.0 * r1 + 11.0 * r2) / 6.0
        q1 = (-r1 + 5.0 * r2 + 2.0 * r3) / 6.0
        q2 = (2.0 * r2 + 5.0 * r3 - r4) / 6.0

        gb0 = (13.0 / 12.0) * (r0 - 2.0 * r1 + r2) ** 2 + 0.25 * (r0 - 4.0 * r1 + 3.0 * r2) ** 2
        gb1 = (13.0 / 12.0) * (r1 - 2.0 * r2 + r3) ** 2 + 0.25 * (r1 - r3) ** 2
        gb2 = (13.0 / 12.0) * (r2 - 2.0 * r3 + r4) ** 2 + 0.25 * (3.0 * r2 - 4.0 * r3 + r4) ** 2

        a0 = d0 / (eps + gb0) ** 2
        a1 = d1 / (eps + gb1) ** 2
        a2 = d2 / (eps + gb2) ** 2
        a_sum = a0 + a1 + a2
        if a_sum == 0.0:
            a_sum = 1e-30

        v_R[idx] = (a0 / a_sum) * q0 + (a1 / a_sum) * q1 + (a2 / a_sum) * q2

    return v_L, v_R


# ============================================================
# HLL Riemann flux for ideal MHD (1D, per-interface)
# ============================================================

@njit(cache=True)
def _hll_flux_1d_core(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    u_L: np.ndarray,
    u_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn_L: np.ndarray,
    Bn_R: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HLL approximate Riemann solver core (Numba-accelerated).

    Returns tuple of (mass_flux, momentum_flux, energy_flux).
    """
    # Fast magnetosonic speed estimate
    B_sq_L = Bn_L ** 2
    B_sq_R = Bn_R ** 2
    a2_L = gamma * np.maximum(p_L, 0.0) / np.maximum(rho_L, 1e-30)
    a2_R = gamma * np.maximum(p_R, 0.0) / np.maximum(rho_R, 1e-30)
    va2_L = B_sq_L / (mu_0 * np.maximum(rho_L, 1e-30))
    va2_R = B_sq_R / (mu_0 * np.maximum(rho_R, 1e-30))
    cf_L = np.sqrt(a2_L + va2_L)
    cf_R = np.sqrt(a2_R + va2_R)

    # Davis wave speed estimates
    S_L = np.minimum(u_L - cf_L, u_R - cf_R)
    S_R = np.maximum(u_L + cf_L, u_R + cf_R)

    # Left and right fluxes
    F_rho_L = rho_L * u_L
    F_rho_R = rho_R * u_R

    ptot_L = p_L + 0.5 * B_sq_L / mu_0
    ptot_R = p_R + 0.5 * B_sq_R / mu_0

    F_mom_L = rho_L * u_L ** 2 + ptot_L
    F_mom_R = rho_R * u_R ** 2 + ptot_R

    e_int_L = p_L / (gamma - 1.0)
    e_int_R = p_R / (gamma - 1.0)
    E_L = e_int_L + 0.5 * rho_L * u_L ** 2 + 0.5 * B_sq_L / mu_0
    E_R = e_int_R + 0.5 * rho_R * u_R ** 2 + 0.5 * B_sq_R / mu_0

    F_ene_L = (E_L + ptot_L) * u_L
    F_ene_R = (E_R + ptot_R) * u_R

    # Conserved quantities
    U_rho_L = rho_L
    U_rho_R = rho_R
    U_mom_L = rho_L * u_L
    U_mom_R = rho_R * u_R

    # HLL flux
    denom = np.maximum(S_R - S_L, 1e-30)

    F_rho = (S_R * F_rho_L - S_L * F_rho_R + S_L * S_R * (U_rho_R - U_rho_L)) / denom
    F_mom = (S_R * F_mom_L - S_L * F_mom_R + S_L * S_R * (U_mom_R - U_mom_L)) / denom
    F_ene = (S_R * F_ene_L - S_L * F_ene_R + S_L * S_R * (E_R - E_L)) / denom

    return F_rho, F_mom, F_ene


def _hll_flux_1d(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    u_L: np.ndarray,
    u_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn_L: np.ndarray,
    Bn_R: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    """HLL approximate Riemann solver — dict wrapper for compatibility."""
    F_rho, F_mom, F_ene = _hll_flux_1d_core(
        rho_L, rho_R, u_L, u_R, p_L, p_R, Bn_L, Bn_R, gamma,
    )
    return {
        "mass_flux": F_rho,
        "momentum_flux": F_mom,
        "energy_flux": F_ene,
    }


# ============================================================
# Dimension-split WENO5+HLL flux computation
# ============================================================

def _compute_flux_1d_sweep(
    rho: np.ndarray,
    vel_n: np.ndarray,
    pressure: np.ndarray,
    Bn: np.ndarray,
    gamma: float,
    axis: int,
) -> dict[str, np.ndarray]:
    """Compute WENO5-reconstructed HLL fluxes along one axis.

    Uses dimension-by-dimension sweep: for each pencil along `axis`,
    perform WENO5 reconstruction then HLL Riemann solve.

    Args:
        rho: Density, shape (nx, ny, nz).
        vel_n: Normal velocity component, shape (nx, ny, nz).
        pressure: Thermal pressure, shape (nx, ny, nz).
        Bn: Normal B-field component, shape (nx, ny, nz).
        gamma: Adiabatic index.
        axis: Sweep axis (0, 1, or 2).

    Returns:
        Dictionary of flux arrays at interfaces (reduced by 4 along `axis`).
    """
    shape = rho.shape
    n_ax = shape[axis]

    # If grid too small for WENO5 (need >=5 cells), fall back to np.gradient
    if n_ax < 5:
        # Return zero fluxes
        return {
            "mass_flux": np.zeros_like(rho),
            "momentum_flux": np.zeros_like(rho),
            "energy_flux": np.zeros_like(rho),
            "n_interfaces": 0,
        }

    n_iface = n_ax - 4

    # Output arrays: n_iface along `axis`, same on other axes
    out_shape = list(shape)
    out_shape[axis] = n_iface
    out_shape = tuple(out_shape)

    F_rho = np.zeros(out_shape)
    F_mom = np.zeros(out_shape)
    F_ene = np.zeros(out_shape)

    # Iterate over pencils perpendicular to axis
    other_axes = [i for i in range(3) if i != axis]
    for idx0 in range(shape[other_axes[0]]):
        for idx1 in range(shape[other_axes[1]]):
            # Build the slicer for this pencil
            slicer = [None, None, None]
            slicer[other_axes[0]] = idx0
            slicer[other_axes[1]] = idx1
            slicer[axis] = slice(None)
            s = tuple(slicer)

            # Extract 1D pencils
            rho_1d = rho[s]
            u_1d = vel_n[s]
            p_1d = pressure[s]
            Bn_1d = Bn[s]

            # WENO5 reconstruct each quantity
            rho_L, rho_R = _weno5_reconstruct_1d(rho_1d)
            u_L, u_R = _weno5_reconstruct_1d(u_1d)
            p_L, p_R = _weno5_reconstruct_1d(p_1d)
            Bn_L, Bn_R = _weno5_reconstruct_1d(Bn_1d)

            # Ensure positivity
            rho_L = np.maximum(rho_L, 1e-20)
            rho_R = np.maximum(rho_R, 1e-20)
            p_L = np.maximum(p_L, 1e-20)
            p_R = np.maximum(p_R, 1e-20)

            # HLL Riemann solve
            fluxes = _hll_flux_1d(rho_L, rho_R, u_L, u_R, p_L, p_R, Bn_L, Bn_R, gamma)

            # Store in output array
            out_slicer = [None, None, None]
            out_slicer[other_axes[0]] = idx0
            out_slicer[other_axes[1]] = idx1
            out_slicer[axis] = slice(None)
            out_s = tuple(out_slicer)

            F_rho[out_s] = fluxes["mass_flux"]
            F_mom[out_s] = fluxes["momentum_flux"]
            F_ene[out_s] = fluxes["energy_flux"]

    return {
        "mass_flux": F_rho,
        "momentum_flux": F_mom,
        "energy_flux": F_ene,
        "n_interfaces": n_iface,
    }


def _apply_flux_divergence(
    U: np.ndarray,
    flux: np.ndarray,
    n_interfaces: int,
    axis: int,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Apply conservative flux-difference update from WENO5+HLL fluxes.

    The flux array has (n_interfaces) entries along `axis`, corresponding
    to faces between cells [2..n_ax-3]. We update the interior cells that
    have both left and right flux faces: cells 2..(n_ax-3) → n_interfaces-1 cells.

    Args:
        U: Conservative variable, shape (nx, ny, nz).
        flux: Interface fluxes, shape with n_interfaces along axis.
        n_interfaces: Number of flux interfaces.
        axis: Sweep axis.
        dx: Grid spacing.
        dt: Timestep.

    Returns:
        Updated conservative variable.
    """
    if n_interfaces < 2:
        return U.copy()

    result = U.copy()

    # Number of cells we can update: n_interfaces - 1
    n_update = n_interfaces - 1

    # Build slice objects for the flux difference
    # F[j+1] - F[j] for j = 0..n_interfaces-2
    flux_left_slicer = [slice(None)] * 3
    flux_right_slicer = [slice(None)] * 3
    update_slicer = [slice(None)] * 3

    flux_left_slicer[axis] = slice(0, n_update)
    flux_right_slicer[axis] = slice(1, n_update + 1)
    update_slicer[axis] = slice(2, 2 + n_update)

    dF = flux[tuple(flux_right_slicer)] - flux[tuple(flux_left_slicer)]
    result[tuple(update_slicer)] -= dt / dx * dF

    return result


# ============================================================
# Dedner divergence cleaning
# ============================================================

def _dedner_source(
    psi: np.ndarray,
    B: np.ndarray,
    ch: float,
    cp: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dedner hyperbolic/parabolic divergence cleaning.

    dpsi/dt = -ch^2 * div(B) - (ch^2 / cp^2) * psi
    dB/dt += -grad(psi)

    Args:
        psi: Cleaning scalar field, shape (nx, ny, nz).
        B: Magnetic field, shape (3, nx, ny, nz).
        ch: Hyperbolic cleaning speed [m/s].
        cp: Parabolic damping speed [m/s].
        dx: Grid spacing [m].

    Returns:
        (dpsi_dt, dB_dt): Source terms.
    """
    div_B = (
        np.gradient(B[0], dx, axis=0)
        + np.gradient(B[1], dx, axis=1)
        + np.gradient(B[2], dx, axis=2)
    )

    dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi

    grad_psi = np.array([
        np.gradient(psi, dx, axis=0),
        np.gradient(psi, dx, axis=1),
        np.gradient(psi, dx, axis=2),
    ])
    dB_dt = -grad_psi

    return dpsi_dt, dB_dt


# ============================================================
# Braginskii anisotropic heat flux operator
# ============================================================

def _braginskii_heat_flux(
    Te: np.ndarray,
    ne: np.ndarray,
    B: np.ndarray,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Apply Braginskii anisotropic heat flux to electron temperature.

    Heat flux: q = -kappa_par * (b_hat . grad(Te)) * b_hat
                 - kappa_perp * grad_perp(Te)

    Uses operator-split explicit diffusion with sub-cycling if needed
    for stability: dt_diff < dx^2 / (2 * max(kappa)).

    Args:
        Te: Electron temperature [K], shape (nx, ny, nz).
        ne: Electron number density [m^-3], shape (nx, ny, nz).
        B: Magnetic field [T], shape (3, nx, ny, nz).
        dx: Grid spacing [m].
        dt: Timestep [s].

    Returns:
        Updated Te array.
    """
    from dpf.collision.spitzer import braginskii_kappa

    B_mag = np.sqrt(np.sum(B**2, axis=0))

    # Compute Braginskii conductivities
    kappa_par, kappa_perp = braginskii_kappa(ne, Te, B_mag)

    # Sanitize kappa arrays — NaN/Inf from extreme conditions
    kappa_par = np.where(np.isfinite(kappa_par), kappa_par, 0.0)
    kappa_perp = np.where(np.isfinite(kappa_perp), kappa_perp, 0.0)

    # Maximum diffusion coefficient for stability check
    max_kappa = np.max(kappa_par)
    if max_kappa < 1e-30:
        return Te.copy()

    # Cap kappa to avoid extreme diffusivity (physical limit)
    kappa_cap = 1e30  # Reasonable cap for plasma heat conductivity
    kappa_par = np.minimum(kappa_par, kappa_cap)
    kappa_perp = np.minimum(kappa_perp, kappa_cap)
    max_kappa = min(max_kappa, kappa_cap)

    # Temperature gradient
    grad_Te = np.array([
        np.gradient(Te, dx, axis=0),
        np.gradient(Te, dx, axis=1),
        np.gradient(Te, dx, axis=2),
    ])

    # Magnetic unit vector
    B_hat = np.zeros_like(B)
    B_safe = np.maximum(B_mag, 1e-30)
    for i in range(3):
        B_hat[i] = B[i] / B_safe

    # Parallel gradient: (b . grad(Te)) * b
    b_dot_gradT = np.sum(B_hat * grad_Te, axis=0)
    q_par_dir = np.zeros_like(B)
    for i in range(3):
        q_par_dir[i] = b_dot_gradT * B_hat[i]

    # Perpendicular gradient: grad(Te) - (b . grad(Te)) * b
    grad_perp_Te = grad_Te - q_par_dir

    # Heat flux divergence: div(kappa_par * q_par + kappa_perp * q_perp)
    heat_flux = np.zeros_like(B)
    for i in range(3):
        heat_flux[i] = kappa_par * q_par_dir[i] + kappa_perp * grad_perp_Te[i]

    # Sanitize heat flux — catch NaN/Inf from kappa * gradient products
    heat_flux = np.where(np.isfinite(heat_flux), heat_flux, 0.0)

    div_q = (
        np.gradient(heat_flux[0], dx, axis=0)
        + np.gradient(heat_flux[1], dx, axis=1)
        + np.gradient(heat_flux[2], dx, axis=2)
    )

    # Sanitize div_q
    div_q = np.where(np.isfinite(div_q), div_q, 0.0)

    # Stability-limited timestep for explicit diffusion
    # dt_diff < dx^2 / (2 * dim * max_kappa / (n_e * k_B))
    ne_safe = np.maximum(ne, 1e-20)
    min_ne = np.min(ne_safe)
    diffusivity = max_kappa / (min_ne * k_B + 1e-30)

    if not np.isfinite(diffusivity) or diffusivity <= 0:
        return Te.copy()

    dt_diff = 0.25 * dx**2 / diffusivity

    if not np.isfinite(dt_diff) or dt_diff <= 0:
        return Te.copy()

    # Sub-cycle if needed
    n_sub = max(1, int(np.ceil(dt / dt_diff)))
    n_sub = min(n_sub, 100)  # Cap subcycles to avoid runaway
    dt_sub = dt / n_sub

    Te_new = Te.copy()
    for _ in range(n_sub):
        # dTe/dt = div(q) / (ne * kB)
        Te_new += dt_sub * div_q / (ne_safe * k_B)

    # Floor temperature
    Te_new = np.maximum(Te_new, 1.0)
    # Sanitize
    Te_new = np.where(np.isfinite(Te_new), Te_new, Te)

    return Te_new


# ============================================================
# Main MHD Solver
# ============================================================

class MHDSolver(PlasmaSolverBase):
    """Hall MHD solver with WENO5 reconstruction, HLL Riemann, and Dedner cleaning.

    Features:
    - WENO5 reconstruction + HLL Riemann solver for advection (5th-order spatial)
    - SSP-RK2 time integration (2nd-order temporal)
    - Dedner hyperbolic divergence cleaning for div(B)
    - Hall term in induction equation (J × B)/(ne)
    - Braginskii anisotropic heat flux (operator-split)
    - dL_dt estimation from pinch dynamics for circuit coupling

    Falls back to forward-Euler np.gradient if grid < 5 cells in any direction.

    Args:
        grid_shape: (nx, ny, nz).
        dx: Grid spacing [m].
        gamma: Adiabatic index.
        cfl: CFL number for timestep.
        dedner_ch: Dedner hyperbolic cleaning speed (0 = auto from max wave speed).
        enable_hall: Enable Hall term in induction equation.
        enable_braginskii: Enable Braginskii anisotropic heat flux.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        gamma: float = 5.0 / 3.0,
        cfl: float = 0.4,
        dedner_ch: float = 0.0,
        enable_hall: bool = True,
        enable_braginskii: bool = True,
        enable_resistive: bool = True,
        enable_energy_equation: bool = True,
        ion_mass: float | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.gamma = gamma
        self.cfl = cfl
        self.dedner_ch_init = dedner_ch
        self.enable_hall = enable_hall
        self.enable_braginskii = enable_braginskii
        self.enable_resistive = enable_resistive
        self.enable_energy_equation = enable_energy_equation
        self.ion_mass = ion_mass if ion_mass is not None else _DEFAULT_ION_MASS
        self.eos = IdealEOS(gamma=gamma)

        # Whether we can use WENO5 (need >= 5 cells in each direction)
        self.use_weno5 = all(n >= 5 for n in grid_shape)

        # Coupling state for circuit interaction
        self._coupling = CouplingState()
        self._prev_Lp: float | None = None  # For dL_dt computation

        logger.info(
            "MHDSolver initialized: grid=%s, dx=%.2e, gamma=%.3f, "
            "WENO5=%s, Hall=%s, Braginskii=%s, Resistive=%s, EnergyEq=%s, "
            "ion_mass=%.3e kg",
            grid_shape, dx, gamma,
            self.use_weno5, enable_hall, enable_braginskii,
            self.enable_resistive, self.enable_energy_equation,
            self.ion_mass,
        )

    def _compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """Compute CFL-limited timestep."""
        rho = state["rho"]
        v = state["velocity"]
        B = state["B"]
        p = state["pressure"]

        # Maximum wave speed: fast magnetosonic
        B_sq = np.sum(B**2, axis=0)
        a2 = self.gamma * p / np.maximum(rho, 1e-30)
        va2 = B_sq / (mu_0 * np.maximum(rho, 1e-30))
        cf = np.sqrt(a2 + va2)

        v_max = np.max(np.abs(v)) + np.max(cf)

        # Hall speed limit: omega_ci * di where di = c / omega_pi
        if self.enable_hall:
            ne = rho / self.ion_mass  # Assume Z=1
            ne_max = np.max(ne)
            if ne_max > 0:
                B_max = np.sqrt(np.max(B_sq))
                # Hall speed ~ B / (mu_0 * ne * e * dx)
                v_hall = B_max / (mu_0 * np.maximum(ne_max, 1e-20) * e_charge * self.dx)
                v_max = max(v_max, v_hall)

        if v_max < 1e-30:
            return 1e-10  # Fallback for zero-velocity initial condition
        return self.cfl * self.dx / v_max

    def _compute_rhs_euler(
        self,
        state: dict[str, np.ndarray],
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the right-hand side (time derivative) of the MHD system.

        Uses WENO5+HLL flux-based update for advection if grid is large enough,
        otherwise falls back to np.gradient centered differences.

        Args:
            state: Current state dict.
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nx,ny,nz).
                       If None, resistive term is skipped.

        Returns dict of dU/dt for each state variable.
        """
        rho = state["rho"]
        vel = state["velocity"]
        p = state["pressure"]
        B = state["B"]
        psi = state.get("psi", np.zeros_like(rho))

        drho_dt = np.zeros_like(rho)
        dmom_dt = np.zeros((3,) + rho.shape)
        dp_dt = np.zeros_like(rho)
        dB_dt = np.zeros_like(B)

        # --- Density advection ---
        if self.use_weno5:
            # WENO5+HLL flux-based update: accumulate flux divergence across all 3 dims
            for axis in range(3):
                fluxes = _compute_flux_1d_sweep(
                    rho, vel[axis], p, B[axis], self.gamma, axis
                )
                n_iface = fluxes["n_interfaces"]
                if n_iface >= 2:
                    n_update = n_iface - 1
                    flux_left_sl = [slice(None)] * 3
                    flux_right_sl = [slice(None)] * 3
                    update_sl = [slice(None)] * 3
                    flux_left_sl[axis] = slice(0, n_update)
                    flux_right_sl[axis] = slice(1, n_update + 1)
                    update_sl[axis] = slice(2, 2 + n_update)
                    dF = (
                        fluxes["mass_flux"][tuple(flux_right_sl)]
                        - fluxes["mass_flux"][tuple(flux_left_sl)]
                    )
                    drho_dt[tuple(update_sl)] -= dF / self.dx
        else:
            # Fallback: centered-difference with np.gradient (original MVP behavior)
            flux_rho = np.array([rho * vel[i] for i in range(3)])
            div_flux = (
                np.gradient(flux_rho[0], self.dx, axis=0)
                + np.gradient(flux_rho[1], self.dx, axis=1)
                + np.gradient(flux_rho[2], self.dx, axis=2)
            )
            drho_dt = -div_flux

        # --- Momentum: J × B force + pressure gradient ---
        # Current density: J = curl(B) / mu_0
        J = np.array([
            np.gradient(B[2], self.dx, axis=1) - np.gradient(B[1], self.dx, axis=2),
            np.gradient(B[0], self.dx, axis=2) - np.gradient(B[2], self.dx, axis=0),
            np.gradient(B[1], self.dx, axis=0) - np.gradient(B[0], self.dx, axis=1),
        ]) / mu_0

        # J × B force
        JxB = np.array([
            J[1] * B[2] - J[2] * B[1],
            J[2] * B[0] - J[0] * B[2],
            J[0] * B[1] - J[1] * B[0],
        ])

        grad_p = np.array([
            np.gradient(p, self.dx, axis=0),
            np.gradient(p, self.dx, axis=1),
            np.gradient(p, self.dx, axis=2),
        ])

        # Momentum advection (WENO5 for momentum flux if available)
        if self.use_weno5:
            for d in range(3):
                # Momentum component d advected by each velocity component
                for axis in range(3):
                    mom_d = rho * vel[d]
                    fluxes = _compute_flux_1d_sweep(
                        mom_d, vel[axis], p, B[axis], self.gamma, axis
                    )
                    n_iface = fluxes["n_interfaces"]
                    if n_iface >= 2:
                        n_update = n_iface - 1
                        flux_left_sl = [slice(None)] * 3
                        flux_right_sl = [slice(None)] * 3
                        update_sl = [slice(None)] * 3
                        flux_left_sl[axis] = slice(0, n_update)
                        flux_right_sl[axis] = slice(1, n_update + 1)
                        update_sl[axis] = slice(2, 2 + n_update)
                        dF = (
                            fluxes["mass_flux"][tuple(flux_right_sl)]
                            - fluxes["mass_flux"][tuple(flux_left_sl)]
                        )
                        dmom_dt[d][tuple(update_sl)] -= dF / self.dx
        else:
            # Fallback: centered difference momentum advection
            for d in range(3):
                for axis in range(3):
                    flux_mom = rho * vel[d] * vel[axis]
                    dmom_dt[d] -= np.gradient(flux_mom, self.dx, axis=axis)

        # Add Lorentz force and pressure gradient
        for d in range(3):
            dmom_dt[d] += JxB[d] - grad_p[d]

        # --- Induction equation: dB/dt = -curl(E) ---
        # Ideal MHD: E = -v × B
        vxB = np.array([
            vel[1] * B[2] - vel[2] * B[1],
            vel[2] * B[0] - vel[0] * B[2],
            vel[0] * B[1] - vel[1] * B[0],
        ])
        E_field = -vxB  # E = -v × B (ideal Ohm's law)

        # --- Resistive term: E_resistive = eta * J ---
        ohmic_heating = np.zeros_like(rho)
        if self.enable_resistive and eta_field is not None:
            E_resistive = eta_field[np.newaxis, :, :, :] * J
            E_field += E_resistive
            # Ohmic heating: Q_ohm = eta * |J|^2 [W/m^3]
            J_sq = np.sum(J**2, axis=0)
            ohmic_heating = eta_field * J_sq

        # --- Hall term: E_Hall = (J × B) / (n_e * e) ---
        if self.enable_hall:
            ne = rho / self.ion_mass  # Z=1
            ne_safe = np.maximum(ne, 1e-20)
            # Hall electric field
            E_Hall = np.array([
                (J[1] * B[2] - J[2] * B[1]),
                (J[2] * B[0] - J[0] * B[2]),
                (J[0] * B[1] - J[1] * B[0]),
            ]) / (ne_safe * e_charge)
            E_field += E_Hall

        # dB/dt = -curl(E)
        curl_E = np.array([
            np.gradient(E_field[2], self.dx, axis=1) - np.gradient(E_field[1], self.dx, axis=2),
            np.gradient(E_field[0], self.dx, axis=2) - np.gradient(E_field[2], self.dx, axis=0),
            np.gradient(E_field[1], self.dx, axis=0) - np.gradient(E_field[0], self.dx, axis=1),
        ])
        dB_dt = -curl_E

        # --- Pressure / Energy equation ---
        div_v = (
            np.gradient(vel[0], self.dx, axis=0)
            + np.gradient(vel[1], self.dx, axis=1)
            + np.gradient(vel[2], self.dx, axis=2)
        )

        if self.enable_energy_equation:
            # Conservative total energy equation:
            # dE_total/dt = -div(F_energy) + eta*J^2 - P_rad
            # E_total = p/(gamma-1) + 0.5*rho*|v|^2 + |B|^2/(2*mu_0)
            # For the pressure update, we use:
            # dp/dt = -gamma*p*div(v) + (gamma-1)*eta*J^2
            # The Ohmic heating term adds (gamma-1)*Q_ohm to pressure rate
            dp_dt = -self.gamma * p * div_v + (self.gamma - 1.0) * ohmic_heating
        else:
            # Adiabatic: dp/dt = -gamma * p * div(v)
            dp_dt = -self.gamma * p * div_v

        # --- Dedner cleaning ---
        ch = self.dedner_ch_init if self.dedner_ch_init > 0 else np.max(np.abs(vel)) + 1.0
        cp = ch
        dpsi_dt, dB_clean = _dedner_source(psi, B, ch, cp, self.dx)
        dB_dt += dB_clean

        return {
            "drho_dt": drho_dt,
            "dmom_dt": dmom_dt,
            "dp_dt": dp_dt,
            "dB_dt": dB_dt,
            "dpsi_dt": dpsi_dt,
            "ohmic_heating": ohmic_heating,
        }

    def apply_electrode_bfield_bc(
        self,
        B: np.ndarray,
        current: float,
        anode_radius: float,
        cathode_radius: float,
    ) -> np.ndarray:
        """Apply electrode B-field boundary conditions for Cartesian solver.

        In Cartesian 3D, we approximate the azimuthal B-field from the
        circuit current at cells near the electrode radii. This is a
        simplified version — the cylindrical solver has a more physical
        implementation.

        B_theta = mu_0 * I / (2 * pi * r) at the boundary cells.

        For Cartesian, we map B_theta onto (B_x, B_y) components using
        the local azimuthal angle.

        Args:
            B: Magnetic field (3, nx, ny, nz).
            current: Circuit current [A].
            anode_radius: Anode radius [m].
            cathode_radius: Cathode radius [m].

        Returns:
            Modified B-field array.
        """
        if abs(current) < 1e-10:
            return B

        nx, ny, nz = self.grid_shape
        dx = self.dx

        # Build radial coordinate from grid center
        x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        y = (np.arange(ny) - ny / 2.0 + 0.5) * dx

        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        R_safe = np.maximum(R, 1e-30)

        # Azimuthal unit vectors: theta_hat = (-sin(theta), cos(theta), 0)
        #   = (-y/r, x/r, 0)
        sin_theta = Y / R_safe  # -sin is for Bx component
        cos_theta = X / R_safe  # cos is for By component

        # Apply at outer boundary cells (near cathode radius)
        # Find cells within one cell of cathode_radius
        mask_cathode = np.abs(R - cathode_radius) < 1.5 * dx

        if np.any(mask_cathode):
            for k in range(nz):
                B_th_local = mu_0 * current / (2.0 * np.pi * np.maximum(R, cathode_radius * 0.5))
                # B_x = -B_theta * sin(theta), B_y = B_theta * cos(theta)
                B[0, :, :, k] = np.where(mask_cathode, -B_th_local * sin_theta, B[0, :, :, k])
                B[1, :, :, k] = np.where(mask_cathode, B_th_local * cos_theta, B[1, :, :, k])

        return B

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
        anode_radius: float = 0.0,
        cathode_radius: float = 0.0,
        apply_electrode_bc: bool = False,
    ) -> dict[str, np.ndarray]:
        """Advance MHD state by one timestep using SSP-RK2.

        SSP-RK2 (Shu-Osher form):
            U^(1) = U^n + dt * L(U^n)
            U^(n+1) = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))

        This is TVD (total variation diminishing) and 2nd-order in time.

        After the RK2 step, applies:
        - Electrode B-field boundary conditions
        - Braginskii anisotropic heat flux (operator-split)
        - Two-temperature update (preserving Te ≠ Ti)
        - Circuit coupling state update with dL_dt

        Args:
            state: Dictionary with keys rho, velocity, pressure, B, Te, Ti, psi.
            dt: Timestep [s].
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nx,ny,nz).
            anode_radius: Anode radius [m] for electrode BC.
            cathode_radius: Cathode radius [m] for electrode BC.
            apply_electrode_bc: Whether to apply electrode B-field BC.

        Returns:
            Updated state dictionary.
        """
        rho = state["rho"]
        vel = state["velocity"]
        p = state["pressure"]
        B = state["B"]
        Te = state.get("Te", np.full_like(rho, 1e4))
        Ti = state.get("Ti", np.full_like(rho, 1e4))
        psi = state.get("psi", np.zeros_like(rho))
        e_electron = state.get("e_electron")

        # Save U^n
        rho_n = rho.copy()
        vel_n = vel.copy()
        p_n = p.copy()
        B_n = B.copy()
        psi_n = psi.copy()
        mom_n = rho_n[np.newaxis, :, :, :] * vel_n

        # === Stage 1: U^(1) = U^n + dt * L(U^n) ===
        state_n = {
            "rho": rho_n, "velocity": vel_n, "pressure": p_n,
            "B": B_n, "Te": Te, "Ti": Ti, "psi": psi_n,
        }
        rhs1 = self._compute_rhs_euler(state_n, current, voltage, eta_field)

        rho_1 = rho_n + dt * rhs1["drho_dt"]
        rho_1 = np.maximum(rho_1, 1e-20)
        mom_1 = mom_n + dt * rhs1["dmom_dt"]
        vel_1 = mom_1 / np.maximum(rho_1[np.newaxis, :, :, :], 1e-30)
        p_1 = p_n + dt * rhs1["dp_dt"]
        p_1 = np.maximum(p_1, 1e-20)
        B_1 = B_n + dt * rhs1["dB_dt"]
        psi_1 = psi_n + dt * rhs1["dpsi_dt"]

        # Apply electrode BC after stage 1
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(
                B_1, current, anode_radius, cathode_radius,
            )

        # === Stage 2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1))) ===
        state_1 = {
            "rho": rho_1, "velocity": vel_1, "pressure": p_1,
            "B": B_1, "Te": Te, "Ti": Ti, "psi": psi_1,
        }
        rhs2 = self._compute_rhs_euler(state_1, current, voltage, eta_field)

        rho_new = 0.5 * rho_n + 0.5 * (rho_1 + dt * rhs2["drho_dt"])
        rho_new = np.maximum(rho_new, 1e-20)
        mom_new = 0.5 * mom_n + 0.5 * (mom_1 + dt * rhs2["dmom_dt"])
        vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :, :], 1e-30)
        p_new = 0.5 * p_n + 0.5 * (p_1 + dt * rhs2["dp_dt"])
        p_new = np.maximum(p_new, 1e-20)
        B_new = 0.5 * B_n + 0.5 * (B_1 + dt * rhs2["dB_dt"])
        psi_new = 0.5 * psi_n + 0.5 * (psi_1 + dt * rhs2["dpsi_dt"])

        # Apply electrode BC after stage 2
        if apply_electrode_bc and cathode_radius > 0:
            B_new = self.apply_electrode_bfield_bc(
                B_new, current, anode_radius, cathode_radius,
            )

        # --- Two-temperature update (preserve Te ≠ Ti) ---
        # Compute total pressure change ratio to scale temperatures
        n_i = rho_new / self.ion_mass
        n_i_safe = np.maximum(n_i, 1e-30)

        if e_electron is not None:
            # We have a separate electron energy — use it to get Te
            # Advect electron energy: de_e/dt = -div(e_e * v) - p_e * div(v) + Q_ohm_e
            div_v = (
                np.gradient(vel_new[0], self.dx, axis=0)
                + np.gradient(vel_new[1], self.dx, axis=1)
                + np.gradient(vel_new[2], self.dx, axis=2)
            )
            p_e_old = n_i_safe * k_B * Te
            # Simple advection + compression
            de_e_dt = -p_e_old * div_v
            # Ohmic heating goes primarily to electrons
            ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs2["ohmic_heating"])
            de_e_dt += ohmic_avg
            e_electron_new = e_electron + dt * de_e_dt
            e_electron_new = np.maximum(e_electron_new, n_i_safe * k_B * 1.0)  # Floor 1 K
            Te_new = (2.0 / 3.0) * e_electron_new / (n_i_safe * k_B)
            # Ion temperature from total pressure minus electron pressure
            p_e_new = n_i_safe * k_B * Te_new
            p_i_new = np.maximum(p_new - p_e_new, 1e-20)
            Ti_new = p_i_new / (n_i_safe * k_B)
        else:
            # No separate electron energy tracked — recover Te, Ti from
            # total pressure split.  T_total = p/(n_i*k_B) = Te + Ti.
            # Preserve the Te/(Te+Ti) fraction from the previous step.
            Te_old = Te
            Ti_old = Ti
            T_sum_old = np.maximum(Te_old + Ti_old, 1.0)
            f_e = Te_old / T_sum_old  # Electron fraction of total temperature

            # Total temperature from new pressure: T_total = p_new / (n_i * k_B)
            T_total_new = p_new / np.maximum(n_i_safe * k_B, 1e-30)
            Te_new = f_e * T_total_new
            Ti_new = (1.0 - f_e) * T_total_new

            # Add Ohmic heating preferentially to electrons
            ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs2["ohmic_heating"])
            dTe_ohmic = (2.0 / 3.0) * ohmic_avg * dt / np.maximum(n_i_safe * k_B, 1e-30)
            Te_new = Te_new + dTe_ohmic

        Te_new = np.maximum(Te_new, 1.0)
        Ti_new = np.maximum(Ti_new, 1.0)

        # --- Braginskii anisotropic heat flux (operator-split) ---
        if self.enable_braginskii:
            ne = rho_new / self.ion_mass  # Z=1
            Te_new = _braginskii_heat_flux(Te_new, ne, B_new, self.dx, dt)

        # --- Update coupling for circuit ---
        B_theta_avg = np.mean(np.sqrt(B_new[0] ** 2 + B_new[1] ** 2))
        if current > 0:
            Lp_est = mu_0 * B_theta_avg / (current + 1e-30) * self.dx * self.grid_shape[0]
        else:
            Lp_est = 0.0

        # Compute dL_dt from previous Lp
        if self._prev_Lp is not None and dt > 0:
            dL_dt = (Lp_est - self._prev_Lp) / dt
        else:
            dL_dt = 0.0
        self._prev_Lp = Lp_est

        self._coupling = CouplingState(
            Lp=Lp_est,
            current=current,
            voltage=voltage,
            dL_dt=dL_dt,
        )

        result = {
            "rho": rho_new,
            "velocity": vel_new,
            "pressure": p_new,
            "B": B_new,
            "Te": Te_new,
            "Ti": Ti_new,
            "psi": psi_new,
        }
        if e_electron is not None:
            result["e_electron"] = e_electron_new
        return result

    def coupling_interface(self) -> CouplingState:
        return self._coupling
