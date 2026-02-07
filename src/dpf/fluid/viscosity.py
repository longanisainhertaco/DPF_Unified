"""Braginskii ion viscosity for magnetised plasmas.

Implements the leading-order Braginskii viscosity coefficients and the
resulting viscous stress tensor for compressible MHD.  The three
viscosity coefficients are:

    eta_0:  Parallel (unmagnetised) viscosity -- dominates when ions are
            weakly magnetised (omega_ci * tau_i << 1).
    eta_3:  Gyroviscosity -- the non-dissipative off-diagonal stress that
            arises from Larmor gyration (finite Larmor radius effect).
    eta_1, eta_2:  Perpendicular viscosities (not implemented here; they
            are suppressed by (omega_ci * tau_i)^{-2} and are rarely
            significant in DPF conditions).

The viscous stress acceleration and heating rate are computed on a
uniform 3-D Cartesian grid using second-order centred finite differences.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    Haines M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).

Functions:
    ion_collision_time: Spitzer ion self-collision time.
    braginskii_eta0: Parallel viscosity coefficient.
    braginskii_eta3: Gyroviscosity coefficient.
    viscous_stress_rate: Viscous acceleration dv/dt.
    viscous_heating_rate: Viscous dissipation Q_visc.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e as e_charge
from dpf.constants import epsilon_0, k_B, m_d, pi

# ============================================================
# Ion collision time
# ============================================================

@njit(cache=True)
def ion_collision_time(
    ni: np.ndarray,
    Ti: np.ndarray,
    Z_eff: float = 1.0,
    m_ion: float = m_d,
) -> np.ndarray:
    r"""Ion-ion (Spitzer) collision time.

    tau_i = 3 sqrt(2 pi) epsilon_0^2 sqrt(m_ion) (k_B Ti)^{3/2}
            / (ni * Z^4 * e^4 * ln Lambda)

    The Coulomb logarithm uses the NRL expression for ion-ion collisions:
        ln Lambda = max(2, 23 - ln(sqrt(ni [cm^-3]) * Z / Ti^{3/2} [eV]))

    Args:
        ni: Ion number density [m^-3].
        Ti: Ion temperature [K].
        Z_eff: Effective ion charge state.
        m_ion: Ion mass [kg] (default: deuterium).

    Returns:
        Ion collision time [s].
    """
    # Coulomb logarithm
    Ti_eV = Ti * k_B / e_charge
    Ti_eV_safe = np.maximum(Ti_eV, 1e-3)
    ni_cm3 = ni * 1e-6
    arg = np.sqrt(np.maximum(ni_cm3, 1.0)) * Z_eff / np.maximum(Ti_eV_safe, 1e-3) ** 1.5
    lnL = 23.0 - np.log(np.maximum(arg, 1e-30))
    lnL = np.maximum(lnL, 2.0)

    numerator = 3.0 * np.sqrt(2.0 * pi) * epsilon_0**2 * np.sqrt(m_ion) * (k_B * Ti) ** 1.5
    denominator = ni * Z_eff**4 * e_charge**4 * lnL
    return numerator / np.maximum(denominator, 1e-300)


# ============================================================
# Braginskii eta_0 (parallel viscosity)
# ============================================================

@njit(cache=True)
def braginskii_eta0(
    ni: np.ndarray,
    Ti: np.ndarray,
    tau_i: np.ndarray,
) -> np.ndarray:
    r"""Braginskii parallel (unmagnetised) ion viscosity coefficient.

    eta_0 = 0.96 * ni * k_B * Ti * tau_i

    This is the dominant viscosity when ions are weakly magnetised.
    For strongly magnetised plasmas it still governs stress along B.

    Args:
        ni: Ion number density [m^-3].
        Ti: Ion temperature [K].
        tau_i: Ion collision time [s].

    Returns:
        eta_0 [Pa * s].
    """
    return 0.96 * ni * k_B * Ti * tau_i


# ============================================================
# Braginskii eta_3 (gyroviscosity)
# ============================================================

@njit(cache=True)
def braginskii_eta3(
    ni: np.ndarray,
    Ti: np.ndarray,
    B_mag: np.ndarray,
    m_ion: float = m_d,
) -> np.ndarray:
    r"""Braginskii gyroviscosity coefficient.

    eta_3 = ni * k_B * Ti / (2 * omega_ci)

    where omega_ci = Z * e * B / m_ion is the ion cyclotron frequency.
    The gyroviscosity is a non-dissipative finite-Larmor-radius effect
    that produces off-diagonal stress.

    Args:
        ni: Ion number density [m^-3].
        Ti: Ion temperature [K].
        B_mag: Magnetic field magnitude [T].
        m_ion: Ion mass [kg] (default: deuterium).

    Returns:
        eta_3 [Pa * s].
    """
    omega_ci = e_charge * B_mag / m_ion
    omega_ci_safe = np.maximum(omega_ci, 1e-30)
    return ni * k_B * Ti / (2.0 * omega_ci_safe)


# ============================================================
# Viscous stress rate (acceleration dv/dt)
# ============================================================

@njit(cache=True)
def _compute_strain_rate(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray,
]:
    """Compute the symmetric strain rate tensor and its trace.

    S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)

    Uses second-order centred differences on the interior and
    one-sided differences at the boundaries.

    Args:
        vx, vy, vz: Velocity components, each shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].

    Returns:
        (Sxx, Syy, Szz, Sxy, Sxz, Syz, S_trace) each shape (nx, ny, nz).
    """
    nx, ny, nz = vx.shape

    Sxx = np.zeros((nx, ny, nz))
    Syy = np.zeros((nx, ny, nz))
    Szz = np.zeros((nx, ny, nz))
    Sxy = np.zeros((nx, ny, nz))
    Sxz = np.zeros((nx, ny, nz))
    Syz = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # dvx/dx
                if i == 0:
                    dvx_dx = (vx[1, j, k] - vx[0, j, k]) / dx
                elif i == nx - 1:
                    dvx_dx = (vx[nx - 1, j, k] - vx[nx - 2, j, k]) / dx
                else:
                    dvx_dx = (vx[i + 1, j, k] - vx[i - 1, j, k]) / (2.0 * dx)

                # dvy/dy
                if j == 0:
                    dvy_dy = (vy[i, 1, k] - vy[i, 0, k]) / dy
                elif j == ny - 1:
                    dvy_dy = (vy[i, ny - 1, k] - vy[i, ny - 2, k]) / dy
                else:
                    dvy_dy = (vy[i, j + 1, k] - vy[i, j - 1, k]) / (2.0 * dy)

                # dvz/dz
                if k == 0:
                    dvz_dz = (vz[i, j, 1] - vz[i, j, 0]) / dz
                elif k == nz - 1:
                    dvz_dz = (vz[i, j, nz - 1] - vz[i, j, nz - 2]) / dz
                else:
                    dvz_dz = (vz[i, j, k + 1] - vz[i, j, k - 1]) / (2.0 * dz)

                # dvx/dy
                if j == 0:
                    dvx_dy = (vx[i, 1, k] - vx[i, 0, k]) / dy
                elif j == ny - 1:
                    dvx_dy = (vx[i, ny - 1, k] - vx[i, ny - 2, k]) / dy
                else:
                    dvx_dy = (vx[i, j + 1, k] - vx[i, j - 1, k]) / (2.0 * dy)

                # dvy/dx
                if i == 0:
                    dvy_dx = (vy[1, j, k] - vy[0, j, k]) / dx
                elif i == nx - 1:
                    dvy_dx = (vy[nx - 1, j, k] - vy[nx - 2, j, k]) / dx
                else:
                    dvy_dx = (vy[i + 1, j, k] - vy[i - 1, j, k]) / (2.0 * dx)

                # dvx/dz
                if k == 0:
                    dvx_dz = (vx[i, j, 1] - vx[i, j, 0]) / dz
                elif k == nz - 1:
                    dvx_dz = (vx[i, j, nz - 1] - vx[i, j, nz - 2]) / dz
                else:
                    dvx_dz = (vx[i, j, k + 1] - vx[i, j, k - 1]) / (2.0 * dz)

                # dvz/dx
                if i == 0:
                    dvz_dx = (vz[1, j, k] - vz[0, j, k]) / dx
                elif i == nx - 1:
                    dvz_dx = (vz[nx - 1, j, k] - vz[nx - 2, j, k]) / dx
                else:
                    dvz_dx = (vz[i + 1, j, k] - vz[i - 1, j, k]) / (2.0 * dx)

                # dvy/dz
                if k == 0:
                    dvy_dz = (vy[i, j, 1] - vy[i, j, 0]) / dz
                elif k == nz - 1:
                    dvy_dz = (vy[i, j, nz - 1] - vy[i, j, nz - 2]) / dz
                else:
                    dvy_dz = (vy[i, j, k + 1] - vy[i, j, k - 1]) / (2.0 * dz)

                # dvz/dy
                if j == 0:
                    dvz_dy = (vz[i, 1, k] - vz[i, 0, k]) / dy
                elif j == ny - 1:
                    dvz_dy = (vz[i, ny - 1, k] - vz[i, ny - 2, k]) / dy
                else:
                    dvz_dy = (vz[i, j + 1, k] - vz[i, j - 1, k]) / (2.0 * dy)

                Sxx[i, j, k] = dvx_dx
                Syy[i, j, k] = dvy_dy
                Szz[i, j, k] = dvz_dz
                Sxy[i, j, k] = 0.5 * (dvx_dy + dvy_dx)
                Sxz[i, j, k] = 0.5 * (dvx_dz + dvz_dx)
                Syz[i, j, k] = 0.5 * (dvy_dz + dvz_dy)

    S_trace = Sxx + Syy + Szz

    return Sxx, Syy, Szz, Sxy, Sxz, Syz, S_trace


@njit(cache=True)
def _div_stress(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_zz: np.ndarray,
    sigma_xy: np.ndarray,
    sigma_xz: np.ndarray,
    sigma_yz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the divergence of the stress tensor.

    (div sigma)_i = d(sigma_ix)/dx + d(sigma_iy)/dy + d(sigma_iz)/dz

    Uses centred differences on the interior, one-sided at boundaries.

    Args:
        sigma_xx ... sigma_yz: Stress tensor components, shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].

    Returns:
        (div_x, div_y, div_z) each shape (nx, ny, nz).
    """
    nx, ny, nz = sigma_xx.shape
    div_x = np.zeros((nx, ny, nz))
    div_y = np.zeros((nx, ny, nz))
    div_z = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # --- x-component: d(sigma_xx)/dx + d(sigma_xy)/dy + d(sigma_xz)/dz ---
                if i == 0:
                    dsxx_dx = (sigma_xx[1, j, k] - sigma_xx[0, j, k]) / dx
                elif i == nx - 1:
                    dsxx_dx = (sigma_xx[nx - 1, j, k] - sigma_xx[nx - 2, j, k]) / dx
                else:
                    dsxx_dx = (sigma_xx[i + 1, j, k] - sigma_xx[i - 1, j, k]) / (2.0 * dx)

                if j == 0:
                    dsxy_dy = (sigma_xy[i, 1, k] - sigma_xy[i, 0, k]) / dy
                elif j == ny - 1:
                    dsxy_dy = (sigma_xy[i, ny - 1, k] - sigma_xy[i, ny - 2, k]) / dy
                else:
                    dsxy_dy = (sigma_xy[i, j + 1, k] - sigma_xy[i, j - 1, k]) / (2.0 * dy)

                if k == 0:
                    dsxz_dz = (sigma_xz[i, j, 1] - sigma_xz[i, j, 0]) / dz
                elif k == nz - 1:
                    dsxz_dz = (sigma_xz[i, j, nz - 1] - sigma_xz[i, j, nz - 2]) / dz
                else:
                    dsxz_dz = (sigma_xz[i, j, k + 1] - sigma_xz[i, j, k - 1]) / (2.0 * dz)

                div_x[i, j, k] = dsxx_dx + dsxy_dy + dsxz_dz

                # --- y-component: d(sigma_xy)/dx + d(sigma_yy)/dy + d(sigma_yz)/dz ---
                if i == 0:
                    dsxy_dx = (sigma_xy[1, j, k] - sigma_xy[0, j, k]) / dx
                elif i == nx - 1:
                    dsxy_dx = (sigma_xy[nx - 1, j, k] - sigma_xy[nx - 2, j, k]) / dx
                else:
                    dsxy_dx = (sigma_xy[i + 1, j, k] - sigma_xy[i - 1, j, k]) / (2.0 * dx)

                if j == 0:
                    dsyy_dy = (sigma_yy[i, 1, k] - sigma_yy[i, 0, k]) / dy
                elif j == ny - 1:
                    dsyy_dy = (sigma_yy[i, ny - 1, k] - sigma_yy[i, ny - 2, k]) / dy
                else:
                    dsyy_dy = (sigma_yy[i, j + 1, k] - sigma_yy[i, j - 1, k]) / (2.0 * dy)

                if k == 0:
                    dsyz_dz = (sigma_yz[i, j, 1] - sigma_yz[i, j, 0]) / dz
                elif k == nz - 1:
                    dsyz_dz = (sigma_yz[i, j, nz - 1] - sigma_yz[i, j, nz - 2]) / dz
                else:
                    dsyz_dz = (sigma_yz[i, j, k + 1] - sigma_yz[i, j, k - 1]) / (2.0 * dz)

                div_y[i, j, k] = dsxy_dx + dsyy_dy + dsyz_dz

                # --- z-component: d(sigma_xz)/dx + d(sigma_yz)/dy + d(sigma_zz)/dz ---
                if i == 0:
                    dsxz_dx = (sigma_xz[1, j, k] - sigma_xz[0, j, k]) / dx
                elif i == nx - 1:
                    dsxz_dx = (sigma_xz[nx - 1, j, k] - sigma_xz[nx - 2, j, k]) / dx
                else:
                    dsxz_dx = (sigma_xz[i + 1, j, k] - sigma_xz[i - 1, j, k]) / (2.0 * dx)

                if j == 0:
                    dsyz_dy = (sigma_yz[i, 1, k] - sigma_yz[i, 0, k]) / dy
                elif j == ny - 1:
                    dsyz_dy = (sigma_yz[i, ny - 1, k] - sigma_yz[i, ny - 2, k]) / dy
                else:
                    dsyz_dy = (sigma_yz[i, j + 1, k] - sigma_yz[i, j - 1, k]) / (2.0 * dy)

                if k == 0:
                    dszz_dz = (sigma_zz[i, j, 1] - sigma_zz[i, j, 0]) / dz
                elif k == nz - 1:
                    dszz_dz = (sigma_zz[i, j, nz - 1] - sigma_zz[i, j, nz - 2]) / dz
                else:
                    dszz_dz = (sigma_zz[i, j, k + 1] - sigma_zz[i, j, k - 1]) / (2.0 * dz)

                div_z[i, j, k] = dsxz_dx + dsyz_dy + dszz_dz

    return div_x, div_y, div_z


def viscous_stress_rate(
    velocity: np.ndarray,
    rho: np.ndarray,
    eta0: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Viscous acceleration from the traceless symmetric stress tensor.

    Computes dv/dt = -div(sigma) / rho where:
        sigma_ij = eta_0 * (S_ij - S_trace/3 * delta_ij)

    and S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i) is the strain rate.

    Accepts velocity in either layout:
        (3, nx, ny, nz) -- component-first (internal convention)
        (nx, ny, nz, 3) -- component-last (convenience)

    The output matches the input layout.

    Args:
        velocity: Velocity field [m/s], shape (3, nx, ny, nz) or (nx, ny, nz, 3).
        rho: Mass density [kg/m^3], shape (nx, ny, nz).
        eta0: Parallel viscosity [Pa*s], shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].

    Returns:
        Viscous acceleration [m/s^2], same shape as velocity.
    """
    # Detect layout: if last dim is 3, assume (nx, ny, nz, 3)
    transposed = velocity.ndim == 4 and velocity.shape[-1] == 3
    if transposed:
        vx = velocity[:, :, :, 0]
        vy = velocity[:, :, :, 1]
        vz = velocity[:, :, :, 2]
    else:
        vx = velocity[0]
        vy = velocity[1]
        vz = velocity[2]

    Sxx, Syy, Szz, Sxy, Sxz, Syz, S_trace = _compute_strain_rate(vx, vy, vz, dx, dy, dz)

    # Traceless part
    S_tr_third = S_trace / 3.0
    sigma_xx = eta0 * (Sxx - S_tr_third)
    sigma_yy = eta0 * (Syy - S_tr_third)
    sigma_zz = eta0 * (Szz - S_tr_third)
    sigma_xy = eta0 * Sxy
    sigma_xz = eta0 * Sxz
    sigma_yz = eta0 * Syz

    div_x, div_y, div_z = _div_stress(
        sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, dx, dy, dz,
    )

    rho_safe = np.maximum(rho, 1e-30)

    if transposed:
        accel = np.zeros_like(velocity)
        accel[:, :, :, 0] = div_x / rho_safe
        accel[:, :, :, 1] = div_y / rho_safe
        accel[:, :, :, 2] = div_z / rho_safe
    else:
        accel = np.zeros_like(velocity)
        accel[0] = div_x / rho_safe
        accel[1] = div_y / rho_safe
        accel[2] = div_z / rho_safe

    return accel


def viscous_heating_rate(
    velocity: np.ndarray,
    eta0: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    r"""Viscous heating rate from the traceless symmetric stress tensor.

    Q_visc = eta_0 * |S_traceless|^2
           = eta_0 * sum_ij (S_ij - S_trace/3 * delta_ij)^2

    This is the local rate of kinetic energy dissipation to heat per
    unit volume [W/m^3].

    Accepts velocity in either layout:
        (3, nx, ny, nz) -- component-first (internal convention)
        (nx, ny, nz, 3) -- component-last (convenience)

    Args:
        velocity: Velocity field [m/s], shape (3, nx, ny, nz) or (nx, ny, nz, 3).
        eta0: Parallel viscosity [Pa*s], shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].

    Returns:
        Viscous heating rate [W/m^3], shape (nx, ny, nz).
    """
    # Detect layout: if last dim is 3, assume (nx, ny, nz, 3)
    if velocity.ndim == 4 and velocity.shape[-1] == 3:
        vx = velocity[:, :, :, 0]
        vy = velocity[:, :, :, 1]
        vz = velocity[:, :, :, 2]
    else:
        vx = velocity[0]
        vy = velocity[1]
        vz = velocity[2]

    Sxx, Syy, Szz, Sxy, Sxz, Syz, S_trace = _compute_strain_rate(vx, vy, vz, dx, dy, dz)

    S_tr_third = S_trace / 3.0

    # |S_traceless|^2 = sum of squares of traceless components
    # Diagonal: (Sxx - S_tr/3)^2 + (Syy - S_tr/3)^2 + (Szz - S_tr/3)^2
    # Off-diagonal: 2*(Sxy^2 + Sxz^2 + Syz^2)  (factor 2 because S is symmetric)
    Q = eta0 * (
        (Sxx - S_tr_third) ** 2
        + (Syy - S_tr_third) ** 2
        + (Szz - S_tr_third) ** 2
        + 2.0 * Sxy**2
        + 2.0 * Sxz**2
        + 2.0 * Syz**2
    )

    return Q
