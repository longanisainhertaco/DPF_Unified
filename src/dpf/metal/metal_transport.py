"""Metal GPU transport physics: Hall MHD, Braginskii conduction/viscosity, Nernst advection.

Operator-split implementations in PyTorch, compatible with both MPS (float32)
and CPU (float64) backends.  Follows the same pattern as ``_apply_resistive_diffusion``
in ``metal_solver.py``: each operator takes primitive-variable tensors and a timestep,
returns updated tensors.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    Sharma P., Hammett G.W., JCP 227, 123 (2007).
    Epperlein E.M., Haines M.G., Phys. Fluids 29, 1029 (1986).
    Nishiguchi A. et al., Phys. Rev. Lett. 53, 262 (1984).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# ── Physical Constants ──────────────────────────────────────────

E_CHARGE = 1.602176634e-19       # Elementary charge [C]
M_E = 9.1093837015e-31           # Electron mass [kg]
M_D = 3.34358377e-27             # Deuterium mass [kg]
K_B = 1.380649e-23               # Boltzmann constant [J/K]
EPSILON_0 = 8.8541878128e-12     # Vacuum permittivity [F/m]
MU_0 = 4.0e-7 * math.pi         # Vacuum permeability [H/m]
P_FLOOR = 1e-20                  # Minimum pressure floor


# ── Shared Utilities ────────────────────────────────────────────

def curl_B_mps(
    B: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    mu_0: float = MU_0,
) -> torch.Tensor:
    """Compute current density J = curl(B) / mu_0.

    Parameters
    ----------
    B : torch.Tensor
        Magnetic field, shape (3, nx, ny, nz).
    dx, dy, dz : float
        Grid spacings [m].
    mu_0 : float
        Vacuum permeability [H/m].

    Returns
    -------
    torch.Tensor
        Current density J, shape (3, nx, ny, nz).
    """
    dBz_dy = torch.gradient(B[2], dim=1, spacing=dy)[0]
    dBy_dz = torch.gradient(B[1], dim=2, spacing=dz)[0]
    dBx_dz = torch.gradient(B[0], dim=2, spacing=dz)[0]
    dBz_dx = torch.gradient(B[2], dim=0, spacing=dx)[0]
    dBy_dx = torch.gradient(B[1], dim=0, spacing=dx)[0]
    dBx_dy = torch.gradient(B[0], dim=1, spacing=dy)[0]

    J = torch.zeros_like(B)
    J[0] = (dBz_dy - dBy_dz) / mu_0
    J[1] = (dBx_dz - dBz_dx) / mu_0
    J[2] = (dBy_dx - dBx_dy) / mu_0
    return J


def _cross_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product of two (3, ...) tensors along dim 0."""
    c = torch.zeros_like(a)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


def _curl_field(
    F: torch.Tensor, dx: float, dy: float, dz: float,
) -> torch.Tensor:
    """Compute curl(F) for a (3, nx, ny, nz) vector field."""
    dFz_dy = torch.gradient(F[2], dim=1, spacing=dy)[0]
    dFy_dz = torch.gradient(F[1], dim=2, spacing=dz)[0]
    dFx_dz = torch.gradient(F[0], dim=2, spacing=dz)[0]
    dFz_dx = torch.gradient(F[2], dim=0, spacing=dx)[0]
    dFy_dx = torch.gradient(F[1], dim=0, spacing=dx)[0]
    dFx_dy = torch.gradient(F[0], dim=1, spacing=dy)[0]

    curl = torch.zeros_like(F)
    curl[0] = dFz_dy - dFy_dz
    curl[1] = dFx_dz - dFz_dx
    curl[2] = dFy_dx - dFx_dy
    return curl


# ── Hall MHD ────────────────────────────────────────────────────

def hall_electric_field_mps(
    J: torch.Tensor,
    B: torch.Tensor,
    rho: torch.Tensor,
    ion_mass: float = M_D,
    e_charge: float = E_CHARGE,
) -> torch.Tensor:
    """Compute the Hall electric field E_Hall = (J x B) / (n_e * e).

    Parameters
    ----------
    J : torch.Tensor
        Current density, shape (3, nx, ny, nz).
    B : torch.Tensor
        Magnetic field, shape (3, nx, ny, nz).
    rho : torch.Tensor
        Mass density, shape (nx, ny, nz).
    ion_mass : float
        Ion mass [kg].
    e_charge : float
        Elementary charge [C].

    Returns
    -------
    torch.Tensor
        Hall electric field, shape (3, nx, ny, nz).
    """
    ne = rho / ion_mass
    ne_safe = torch.clamp(ne, min=1e-20)
    JxB = _cross_product(J, B)
    return JxB / (ne_safe.unsqueeze(0) * e_charge)


def apply_hall_mhd_mps(
    B: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    ion_mass: float = M_D,
) -> torch.Tensor:
    """Operator-split Hall MHD update: dB/dt = -curl(E_Hall).

    The Hall term modifies the induction equation only (not momentum/energy).
    E_Hall = (J x B) / (n_e * e), where J = curl(B) / mu_0.

    Parameters
    ----------
    B : torch.Tensor
        Magnetic field, shape (3, nx, ny, nz).
    rho : torch.Tensor
        Mass density, shape (nx, ny, nz).
    dt : float
        Timestep [s].
    dx, dy, dz : float
        Grid spacings [m].
    ion_mass : float
        Ion mass [kg].

    Returns
    -------
    torch.Tensor
        Updated magnetic field B_new.
    """
    J = curl_B_mps(B, dx, dy, dz)
    E_Hall = hall_electric_field_mps(J, B, rho, ion_mass)
    curl_E = _curl_field(E_Hall, dx, dy, dz)
    B_new = B - dt * curl_E
    # Sanitize
    B_new = torch.where(torch.isfinite(B_new), B_new, B)
    return B_new


# ── Braginskii Thermal Conduction ──────────────────────────────

def braginskii_kappa_mps(
    ne: torch.Tensor,
    Te: torch.Tensor,
    B_mag: torch.Tensor,
    Z_eff: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Braginskii parallel and perpendicular thermal conductivities.

    Parameters
    ----------
    ne : torch.Tensor
        Electron number density [m^-3], shape (nx, ny, nz).
    Te : torch.Tensor
        Electron temperature [K], shape (nx, ny, nz).
    B_mag : torch.Tensor
        Magnetic field magnitude [T], shape (nx, ny, nz).
    Z_eff : float
        Effective ion charge.

    Returns
    -------
    kappa_par : torch.Tensor
        Parallel conductivity [W/(m*K)].
    kappa_perp : torch.Tensor
        Perpendicular conductivity [W/(m*K)].
    """
    Te_safe = torch.clamp(Te, min=1.0)
    ne_safe = torch.clamp(ne, min=1e-10)
    B_safe = torch.clamp(B_mag, min=1e-30)

    # Coulomb logarithm (NRL)
    Te_eV = Te_safe * K_B / E_CHARGE
    ne_cm3 = ne_safe * 1e-6
    arg = torch.sqrt(torch.clamp(ne_cm3, min=1.0)) * Z_eff / torch.clamp(Te_eV, min=1e-3) ** 1.5
    lnL = torch.clamp(23.0 - torch.log(torch.clamp(arg, min=1e-30)), min=2.0)

    # Electron collision time: tau_e ~ 3.44e5 * Te^{3/2} / (ne * lnL)
    tau_e = 3.44e5 * Te_safe ** 1.5 / torch.clamp(ne_safe * lnL, min=1e-30)

    # kappa_par = 3.16 * ne * kB^2 * Te * tau_e / m_e
    kappa_par = 3.16 * ne_safe * K_B**2 * Te_safe * tau_e / M_E
    kappa_par = torch.where(torch.isfinite(kappa_par), kappa_par, torch.zeros_like(kappa_par))

    # kappa_perp = 4.66 * ne * kB^2 * Te / (m_e * omega_ce^2 * tau_e)
    omega_ce = E_CHARGE * B_safe / M_E
    kappa_perp = 4.66 * ne_safe * K_B**2 * Te_safe / (
        M_E * torch.clamp(omega_ce**2 * tau_e, min=1e-300)
    )
    kappa_perp = torch.minimum(kappa_perp, kappa_par)
    kappa_perp = torch.where(torch.isfinite(kappa_perp), kappa_perp, torch.zeros_like(kappa_perp))

    # Cap for stability
    kappa_cap = 1e30
    kappa_par = torch.clamp(kappa_par, max=kappa_cap)
    kappa_perp = torch.clamp(kappa_perp, max=kappa_cap)

    return kappa_par, kappa_perp


def apply_braginskii_conduction_mps(
    Te: torch.Tensor,
    B: torch.Tensor,
    ne: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    Z_eff: float = 1.0,
    flux_limiter: float = 0.1,
) -> torch.Tensor:
    """Operator-split Braginskii anisotropic thermal conduction.

    Implements Sharma-Hammett slope-limited explicit diffusion with sub-cycling.

    Parameters
    ----------
    Te : torch.Tensor
        Electron temperature [K], shape (nx, ny, nz).
    B : torch.Tensor
        Magnetic field [T], shape (3, nx, ny, nz).
    ne : torch.Tensor
        Electron number density [m^-3], shape (nx, ny, nz).
    dt : float
        Timestep [s].
    dx, dy, dz : float
        Grid spacings [m].
    Z_eff : float
        Effective ion charge.
    flux_limiter : float
        Sharma-Hammett free-streaming limiter parameter (default 0.1).

    Returns
    -------
    torch.Tensor
        Updated Te after anisotropic conduction.
    """
    # B magnitude and unit vector
    B_mag = torch.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    B_safe = torch.clamp(B_mag, min=1e-30)
    b_hat = B / B_safe.unsqueeze(0)

    # Conductivities
    kappa_par, kappa_perp = braginskii_kappa_mps(ne, Te, B_mag, Z_eff)

    max_kappa = float(kappa_par.max())
    if max_kappa < 1e-30:
        return Te.clone()

    # Temperature gradient
    grad_Tx = torch.gradient(Te, dim=0, spacing=dx)[0]
    grad_Ty = torch.gradient(Te, dim=1, spacing=dy)[0]
    grad_Tz = torch.gradient(Te, dim=2, spacing=dz)[0]
    grad_T = torch.stack([grad_Tx, grad_Ty, grad_Tz], dim=0)  # (3, nx, ny, nz)

    # Parallel gradient: (b . grad_T) * b
    b_dot_gradT = (b_hat * grad_T).sum(dim=0)  # (nx, ny, nz)
    grad_T_par = b_hat * b_dot_gradT.unsqueeze(0)

    # Perpendicular gradient
    grad_T_perp = grad_T - grad_T_par

    # Sharma-Hammett flux limiter
    Te_safe = torch.clamp(Te, min=1.0)
    ne_safe = torch.clamp(ne, min=1e-10)
    v_th_e = torch.sqrt(K_B * Te_safe / M_E)
    q_free = ne_safe * K_B * Te_safe * v_th_e

    q_par_mag = kappa_par * torch.abs(b_dot_gradT)
    q_denom = q_par_mag + flux_limiter * q_free
    limiter_factor = torch.where(
        q_denom > 1e-30,
        flux_limiter * q_free / q_denom,
        torch.ones_like(q_denom),
    )
    kappa_par_limited = kappa_par * limiter_factor

    # Heat flux
    heat_flux = (
        kappa_par_limited.unsqueeze(0) * grad_T_par
        + kappa_perp.unsqueeze(0) * grad_T_perp
    )
    heat_flux = torch.where(torch.isfinite(heat_flux), heat_flux, torch.zeros_like(heat_flux))

    # Divergence of heat flux
    div_q = (
        torch.gradient(heat_flux[0], dim=0, spacing=dx)[0]
        + torch.gradient(heat_flux[1], dim=1, spacing=dy)[0]
        + torch.gradient(heat_flux[2], dim=2, spacing=dz)[0]
    )
    div_q = torch.where(torch.isfinite(div_q), div_q, torch.zeros_like(div_q))

    # Stability sub-cycling
    dx_min = min(dx, dy, dz)
    min_ne = float(ne_safe.min())
    diffusivity = max_kappa / max(min_ne * K_B, 1e-30)

    if not math.isfinite(diffusivity) or diffusivity <= 0:
        return Te.clone()

    dt_diff = 0.25 * dx_min**2 / diffusivity
    if not math.isfinite(dt_diff) or dt_diff <= 0:
        return Te.clone()

    n_sub = max(1, int(math.ceil(dt / dt_diff)))
    n_sub = min(n_sub, 100)
    dt_sub = dt / n_sub

    Te_new = Te.clone()
    for _ in range(n_sub):
        Te_new = Te_new + dt_sub * div_q / (1.5 * ne_safe * K_B)

    Te_new = torch.clamp(Te_new, min=1.0)
    Te_new = torch.where(torch.isfinite(Te_new), Te_new, Te)
    return Te_new


# ── Braginskii Viscosity ───────────────────────────────────────

def apply_braginskii_viscosity_mps(
    velocity: torch.Tensor,
    rho: torch.Tensor,
    pressure: torch.Tensor,
    B: torch.Tensor,
    Ti: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    ion_mass: float = M_D,
    full_braginskii: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Operator-split Braginskii viscosity: updates velocity and pressure.

    Computes the viscous stress from the strain rate tensor and applies
    the resulting acceleration to velocity.  Viscous heating is added
    to the pressure.

    Parameters
    ----------
    velocity : torch.Tensor
        Velocity field, shape (3, nx, ny, nz).
    rho : torch.Tensor
        Mass density, shape (nx, ny, nz).
    pressure : torch.Tensor
        Gas pressure, shape (nx, ny, nz).
    B : torch.Tensor
        Magnetic field, shape (3, nx, ny, nz).
    Ti : torch.Tensor
        Ion temperature [K], shape (nx, ny, nz).
    dt : float
        Timestep [s].
    dx, dy, dz : float
        Grid spacings [m].
    ion_mass : float
        Ion mass [kg].
    full_braginskii : bool
        If True, decompose stress into parallel/perp parts using B.

    Returns
    -------
    velocity_new : torch.Tensor
        Updated velocity.
    pressure_new : torch.Tensor
        Updated pressure (viscous heating added).
    """
    vx, vy, vz = velocity[0], velocity[1], velocity[2]

    # Strain rate via central differences
    dvx_dx = torch.gradient(vx, dim=0, spacing=dx)[0]
    dvy_dy = torch.gradient(vy, dim=1, spacing=dy)[0]
    dvz_dz = torch.gradient(vz, dim=2, spacing=dz)[0]
    dvx_dy = torch.gradient(vx, dim=1, spacing=dy)[0]
    dvy_dx = torch.gradient(vy, dim=0, spacing=dx)[0]
    dvx_dz = torch.gradient(vx, dim=2, spacing=dz)[0]
    dvz_dx = torch.gradient(vz, dim=0, spacing=dx)[0]
    dvy_dz = torch.gradient(vy, dim=2, spacing=dz)[0]
    dvz_dy = torch.gradient(vz, dim=1, spacing=dy)[0]

    Sxx = dvx_dx
    Syy = dvy_dy
    Szz = dvz_dz
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Sxz = 0.5 * (dvx_dz + dvz_dx)
    Syz = 0.5 * (dvy_dz + dvz_dy)
    S_trace = Sxx + Syy + Szz
    S_tr_third = S_trace / 3.0

    # Ion viscosity coefficients
    ni = rho / ion_mass
    ni_safe = torch.clamp(ni, min=1e-10)
    Ti_safe = torch.clamp(Ti, min=1.0)

    # Coulomb log (ion-ion)
    Ti_eV = Ti_safe * K_B / E_CHARGE
    ni_cm3 = ni_safe * 1e-6
    arg = torch.sqrt(torch.clamp(ni_cm3, min=1.0)) / torch.clamp(Ti_eV, min=1e-3) ** 1.5
    lnL = torch.clamp(23.0 - torch.log(torch.clamp(arg, min=1e-30)), min=2.0)

    # Ion collision time
    numerator = 3.0 * math.sqrt(2.0 * math.pi) * EPSILON_0**2 * math.sqrt(ion_mass) * (K_B * Ti_safe) ** 1.5
    denominator = ni_safe * E_CHARGE**4 * lnL
    tau_i = numerator / torch.clamp(denominator, min=1e-300)

    # eta_0 = 0.96 * ni * kB * Ti * tau_i
    eta0 = 0.96 * ni_safe * K_B * Ti_safe * tau_i

    if full_braginskii:
        # Compute eta_1 = 0.3 * ni * kB * Ti / (omega_ci^2 * tau_i)
        B_mag = torch.sqrt((B**2).sum(dim=0))
        B_safe = torch.clamp(B_mag, min=1e-30)
        omega_ci = E_CHARGE * B_safe / ion_mass
        eta1 = 0.3 * ni_safe * K_B * Ti_safe / torch.clamp(omega_ci**2 * tau_i, min=1e-300)
        eta1 = torch.minimum(eta1, eta0)

        # B unit vector
        bx = B[0] / B_safe
        by = B[1] / B_safe
        bz = B[2] / B_safe

        # Traceless strain
        Txx = Sxx - S_tr_third
        Tyy = Syy - S_tr_third
        Tzz = Szz - S_tr_third

        # Parallel projection: bb_k bb_l S_kl
        bbS = (bx*bx*Sxx + by*by*Syy + bz*bz*Szz
               + 2.0*bx*by*Sxy + 2.0*bx*bz*Sxz + 2.0*by*bz*Syz)

        W0_xx = (bx*bx - 1.0/3.0) * bbS
        W0_yy = (by*by - 1.0/3.0) * bbS
        W0_zz = (bz*bz - 1.0/3.0) * bbS
        W0_xy = bx * by * bbS
        W0_xz = bx * bz * bbS
        W0_yz = by * bz * bbS

        sigma_xx = eta0 * W0_xx + eta1 * (Txx - W0_xx)
        sigma_yy = eta0 * W0_yy + eta1 * (Tyy - W0_yy)
        sigma_zz = eta0 * W0_zz + eta1 * (Tzz - W0_zz)
        sigma_xy = eta0 * W0_xy + eta1 * (Sxy - W0_xy)
        sigma_xz = eta0 * W0_xz + eta1 * (Sxz - W0_xz)
        sigma_yz = eta0 * W0_yz + eta1 * (Syz - W0_yz)
    else:
        # Isotropic traceless
        sigma_xx = eta0 * (Sxx - S_tr_third)
        sigma_yy = eta0 * (Syy - S_tr_third)
        sigma_zz = eta0 * (Szz - S_tr_third)
        sigma_xy = eta0 * Sxy
        sigma_xz = eta0 * Sxz
        sigma_yz = eta0 * Syz

    # Divergence of stress tensor
    div_sigma_x = (
        torch.gradient(sigma_xx, dim=0, spacing=dx)[0]
        + torch.gradient(sigma_xy, dim=1, spacing=dy)[0]
        + torch.gradient(sigma_xz, dim=2, spacing=dz)[0]
    )
    div_sigma_y = (
        torch.gradient(sigma_xy, dim=0, spacing=dx)[0]
        + torch.gradient(sigma_yy, dim=1, spacing=dy)[0]
        + torch.gradient(sigma_yz, dim=2, spacing=dz)[0]
    )
    div_sigma_z = (
        torch.gradient(sigma_xz, dim=0, spacing=dx)[0]
        + torch.gradient(sigma_yz, dim=1, spacing=dy)[0]
        + torch.gradient(sigma_zz, dim=2, spacing=dz)[0]
    )

    rho_safe = torch.clamp(rho, min=1e-30)

    # Viscous acceleration: a = div(sigma) / rho
    vel_new = velocity.clone()
    vel_new[0] = velocity[0] + dt * div_sigma_x / rho_safe
    vel_new[1] = velocity[1] + dt * div_sigma_y / rho_safe
    vel_new[2] = velocity[2] + dt * div_sigma_z / rho_safe

    # Viscous heating: Q = eta_0 * |S_traceless|^2
    Q_visc = eta0 * (
        (Sxx - S_tr_third)**2 + (Syy - S_tr_third)**2 + (Szz - S_tr_third)**2
        + 2.0 * Sxy**2 + 2.0 * Sxz**2 + 2.0 * Syz**2
    )
    gamma = 5.0 / 3.0  # Adiabatic index
    p_new = pressure + dt * (gamma - 1.0) * Q_visc
    p_new = torch.clamp(p_new, min=P_FLOOR)

    # Sanitize
    vel_new = torch.where(torch.isfinite(vel_new), vel_new, velocity)
    p_new = torch.where(torch.isfinite(p_new), p_new, pressure)

    return vel_new, p_new


# ── Nernst Advection ───────────────────────────────────────────

def nernst_coefficient_mps(
    ne: torch.Tensor,
    Te: torch.Tensor,
    B_mag: torch.Tensor,
    Z_eff: float = 1.0,
) -> torch.Tensor:
    """Epperlein-Haines beta_wedge Nernst coefficient.

    beta_wedge = x_e * (2.5*x_e^2 + 4.664*x_e + 11.92)
                 / (x_e^4 + 14.79*x_e^3 + 16.86*x_e^2 + 7.095*x_e + 3.774)

    where x_e = omega_ce * tau_e (electron Hall parameter).

    Parameters
    ----------
    ne : torch.Tensor
        Electron number density [m^-3].
    Te : torch.Tensor
        Electron temperature [K].
    B_mag : torch.Tensor
        Magnetic field magnitude [T].
    Z_eff : float
        Effective ion charge.

    Returns
    -------
    torch.Tensor
        Dimensionless beta_wedge.
    """
    ne_safe = torch.clamp(ne, min=1e-10)
    Te_safe = torch.clamp(Te, min=1.0)

    # Coulomb logarithm
    Te_eV = Te_safe * K_B / E_CHARGE
    ne_cm3 = ne_safe * 1e-6
    arg = torch.sqrt(torch.clamp(ne_cm3, min=1.0)) * Z_eff / torch.clamp(Te_eV, min=1e-3) ** 1.5
    lnL = torch.clamp(23.0 - torch.log(torch.clamp(arg, min=1e-30)), min=2.0)

    # Electron collision time
    numerator = (
        3.0 * math.sqrt(2.0 * math.pi) * EPSILON_0**2
        * math.sqrt(M_E) * (K_B * Te_safe) ** 1.5
    )
    denominator = ne_safe * Z_eff * E_CHARGE**4 * lnL
    tau_e = numerator / torch.clamp(denominator, min=1e-300)

    # Hall parameter x_e = omega_ce * tau_e
    omega_ce = E_CHARGE * B_mag / M_E
    x_e = omega_ce * tau_e

    # Epperlein-Haines rational polynomial
    num = x_e * (2.5 * x_e**2 + 4.664 * x_e + 11.92)
    den = x_e**4 + 14.79 * x_e**3 + 16.86 * x_e**2 + 7.095 * x_e + 3.774
    beta_wedge = num / torch.clamp(den, min=1e-300)

    return beta_wedge


def apply_nernst_advection_mps(
    B: torch.Tensor,
    ne: torch.Tensor,
    Te: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    Z_eff: float = 1.0,
) -> torch.Tensor:
    """Operator-split Nernst B-field advection.

    B is advected by the Nernst velocity:
        v_N = -beta_wedge / (e * ne) * (b_hat x grad(Te))

    Uses first-order upwind differencing for stability.

    Parameters
    ----------
    B : torch.Tensor
        Magnetic field, shape (3, nx, ny, nz).
    ne : torch.Tensor
        Electron number density [m^-3], shape (nx, ny, nz).
    Te : torch.Tensor
        Electron temperature [K], shape (nx, ny, nz).
    dt : float
        Timestep [s].
    dx, dy, dz : float
        Grid spacings [m].
    Z_eff : float
        Effective ion charge.

    Returns
    -------
    torch.Tensor
        Updated B after Nernst advection.
    """
    # B magnitude and unit vector
    B_mag = torch.sqrt((B**2).sum(dim=0))
    B_safe = torch.clamp(B_mag, min=1e-30)
    b_hat = B / B_safe.unsqueeze(0)

    # Temperature gradient
    grad_Tx = torch.gradient(Te, dim=0, spacing=dx)[0]
    grad_Ty = torch.gradient(Te, dim=1, spacing=dy)[0]
    grad_Tz = torch.gradient(Te, dim=2, spacing=dz)[0]
    grad_Te = torch.stack([grad_Tx, grad_Ty, grad_Tz], dim=0)

    # b_hat x grad(Te)
    cross = _cross_product(b_hat, grad_Te)

    # Nernst coefficient
    beta_w = nernst_coefficient_mps(ne, Te, B_mag, Z_eff)
    ne_safe = torch.clamp(ne, min=1e-20)

    # v_N = -beta_wedge / (e * ne) * (b x grad_Te)
    prefactor = -beta_w / (E_CHARGE * ne_safe)
    v_nernst = prefactor.unsqueeze(0) * cross  # (3, nx, ny, nz)

    # Upwind advection: B^{n+1} = B^n - dt * (v . grad(B))
    B_new = B.clone()
    spacings = [dx, dy, dz]
    for comp in range(3):
        B_comp = B[comp]
        for dim_idx in range(3):
            v_comp = v_nernst[dim_idx]
            sp = spacings[dim_idx]

            # Forward and backward differences
            fwd = torch.zeros_like(B_comp)
            bwd = torch.zeros_like(B_comp)

            # Forward: (B[i+1] - B[i]) / dx
            if dim_idx == 0:
                fwd[:-1] = (B_comp[1:] - B_comp[:-1]) / sp
                fwd[-1] = fwd[-2]
                bwd[1:] = (B_comp[1:] - B_comp[:-1]) / sp
                bwd[0] = bwd[1]
            elif dim_idx == 1:
                fwd[:, :-1] = (B_comp[:, 1:] - B_comp[:, :-1]) / sp
                fwd[:, -1] = fwd[:, -2]
                bwd[:, 1:] = (B_comp[:, 1:] - B_comp[:, :-1]) / sp
                bwd[:, 0] = bwd[:, 1]
            else:
                fwd[:, :, :-1] = (B_comp[:, :, 1:] - B_comp[:, :, :-1]) / sp
                fwd[:, :, -1] = fwd[:, :, -2]
                bwd[:, :, 1:] = (B_comp[:, :, 1:] - B_comp[:, :, :-1]) / sp
                bwd[:, :, 0] = bwd[:, :, 1]

            # Upwind: use backward diff where v > 0, forward where v < 0
            grad_upwind = torch.where(v_comp > 0, bwd, fwd)
            B_new[comp] = B_new[comp] - dt * v_comp * grad_upwind

    # Sanitize
    B_new = torch.where(torch.isfinite(B_new), B_new, B)
    return B_new
