"""Nernst effect: magnetic field advection by electron temperature gradients.

The Nernst effect arises from the Braginskii transport theory and describes
how magnetic field lines are swept along electron temperature gradients.  In
laser-plasma and Z-pinch experiments the Nernst term can dominate over
resistive diffusion, compressing or expelling B-field from hot regions.

The Nernst electric field is:

    E_Nernst = -beta_wedge * (b_hat x grad(Te)) / (|e| * ne)

where beta_wedge is the dimensionless Braginskii thermo-magnetic coefficient,
b_hat = B / |B|, and ne is the electron number density.

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    Nishiguchi et al., Phys. Rev. Lett. 53, 262 (1984).
    Ridgers et al., Phys. Plasmas 15, 092311 (2008).

Functions:
    nernst_coefficient: Braginskii beta_wedge coefficient.
    nernst_velocity: Velocity at which B-field is swept by grad(Te).
    nernst_electric_field: Nernst contribution to the electric field.
    apply_nernst_advection: Update B via upwind advection with Nernst velocity.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e as e_charge
from dpf.constants import epsilon_0, k_B, m_e, pi

# ============================================================
# Coulomb logarithm (local helper, consistent with spitzer.py)
# ============================================================

@njit(cache=True)
def _coulomb_log(ne: np.ndarray, Te: np.ndarray) -> np.ndarray:
    """Coulomb logarithm for electron-ion collisions.

    Uses the NRL Plasma Formulary expression, floored at 2.0.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].

    Returns:
        Coulomb logarithm array (floored at 2.0).
    """
    # NRL expression: ln Lambda = 23 - ln(sqrt(ne [cm^-3]) * Z / Te^(3/2) [eV])
    # Convert ne from m^-3 to cm^-3: ne * 1e-6
    # Convert Te from K to eV: Te * k_B / e_charge
    Te_eV = Te * k_B / e_charge
    Te_eV_safe = np.maximum(Te_eV, 1e-3)
    ne_cm3 = ne * 1e-6
    arg = np.sqrt(np.maximum(ne_cm3, 1.0)) / np.maximum(Te_eV_safe, 1e-3) ** 1.5
    lnL = 23.0 - np.log(np.maximum(arg, 1e-30))
    return np.maximum(lnL, 2.0)


# ============================================================
# Electron collision time
# ============================================================

@njit(cache=True)
def _electron_collision_time(
    ne: np.ndarray, Te: np.ndarray, Z: float, lnL: np.ndarray
) -> np.ndarray:
    r"""Electron-ion collision time (Spitzer).

    tau_e = 3 * sqrt(2 pi) * epsilon_0^2 * sqrt(m_e) * (k_B Te)^{3/2}
            / (ne * Z * e^4 * ln Lambda)

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Effective ion charge state.
        lnL: Coulomb logarithm.

    Returns:
        Electron collision time [s].
    """
    numerator = 3.0 * np.sqrt(2.0 * pi) * epsilon_0**2 * np.sqrt(m_e) * (k_B * Te) ** 1.5
    denominator = ne * Z * e_charge**4 * lnL
    return numerator / np.maximum(denominator, 1e-300)


# ============================================================
# Braginskii beta_wedge coefficient
# ============================================================

@njit(cache=True)
def nernst_coefficient(
    ne: np.ndarray,
    Te: np.ndarray,
    B_mag: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    r"""Braginskii thermo-magnetic (Nernst) coefficient beta_wedge.

    beta_wedge is computed from the electron Hall parameter
    x_e = omega_ce * tau_e using the Braginskii polynomial fits for Z=1:

        beta_wedge = x_e * (2.5 * x_e^2 + 4.664 * x_e + 11.92)
                     / (x_e^4 + 14.79 * x_e^3 + 16.86 * x_e^2
                        + 7.095 * x_e + 3.774)

    This expression is the Epperlein-Haines fit (1986) for Z=1 and
    interpolates correctly between weakly magnetised (beta_wedge -> 0)
    and strongly magnetised (beta_wedge -> 3/2 / x_e) limits.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        B_mag: Magnetic field magnitude [T].
        Z_eff: Effective ion charge state.

    Returns:
        Dimensionless beta_wedge coefficient.
    """
    lnL = _coulomb_log(ne, Te)
    tau_e = _electron_collision_time(ne, Te, Z_eff, lnL)
    omega_ce = e_charge * B_mag / m_e
    x_e = omega_ce * tau_e

    # Epperlein-Haines rational polynomial fit for Z=1
    numerator = x_e * (2.5 * x_e**2 + 4.664 * x_e + 11.92)
    denominator = x_e**4 + 14.79 * x_e**3 + 16.86 * x_e**2 + 7.095 * x_e + 3.774
    beta_wedge = numerator / np.maximum(denominator, 1e-300)

    return beta_wedge


# ============================================================
# Nernst velocity
# ============================================================

def nernst_velocity(
    ne: np.ndarray,
    Te: np.ndarray,
    B_field: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    r"""Nernst advection velocity for the magnetic field.

    v_N = -beta_wedge / (|e| * ne) * (b_hat x grad(Te))

    This is the velocity at which magnetic field lines are swept by
    electron temperature gradients in the Braginskii framework.

    Args:
        ne: Electron number density [m^-3], shape (nx, ny, nz).
        Te: Electron temperature [K], shape (nx, ny, nz).
        B_field: Magnetic field [T], shape (3, nx, ny, nz).
        Z_eff: Effective ion charge state.

    Returns:
        Nernst velocity [m/s], shape (3, nx, ny, nz).
    """
    B_mag = np.sqrt(np.sum(B_field**2, axis=0))
    B_mag_safe = np.maximum(B_mag, 1e-30)

    # Unit vector along B
    b_hat = B_field / B_mag_safe[np.newaxis, :, :, :]

    # Temperature gradient
    grad_Te = np.array([
        np.gradient(Te, axis=0),
        np.gradient(Te, axis=1),
        np.gradient(Te, axis=2),
    ])

    # b_hat x grad(Te)
    cross = np.array([
        b_hat[1] * grad_Te[2] - b_hat[2] * grad_Te[1],
        b_hat[2] * grad_Te[0] - b_hat[0] * grad_Te[2],
        b_hat[0] * grad_Te[1] - b_hat[1] * grad_Te[0],
    ])

    beta_w = nernst_coefficient(ne, Te, B_mag, Z_eff)
    ne_safe = np.maximum(ne, 1e-20)

    prefactor = -beta_w / (e_charge * ne_safe)

    v_nernst = prefactor[np.newaxis, :, :, :] * cross
    return v_nernst


# ============================================================
# Nernst electric field
# ============================================================

def nernst_electric_field(
    ne: np.ndarray | float,
    Te: np.ndarray | float,
    B_field: np.ndarray,
    grad_Te: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    r"""Nernst contribution to the electric field.

    E_N = -beta_wedge * (b x grad(Te)) / (e * ne)

    where b = B / |B| is the magnetic unit vector.

    Accepts both 1-D vector inputs (shape (3,)) for single-point
    evaluation and full 3-D field inputs (shape (3, nx, ny, nz)).

    Args:
        ne: Electron number density [m^-3], scalar or shape (nx, ny, nz).
        Te: Electron temperature [K], scalar or shape (nx, ny, nz).
        B_field: Magnetic field [T], shape (3,) or (3, nx, ny, nz).
        grad_Te: Electron temperature gradient [K/m], shape (3,) or (3, nx, ny, nz).
        Z_eff: Effective ion charge state.

    Returns:
        Nernst electric field [V/m], same shape as B_field.
    """
    B_field = np.asarray(B_field, dtype=np.float64)
    grad_Te = np.asarray(grad_Te, dtype=np.float64)
    ne_arr = np.atleast_1d(np.asarray(ne, dtype=np.float64))
    Te_arr = np.atleast_1d(np.asarray(Te, dtype=np.float64))

    # Handle 1-D vector case: B_field shape (3,)
    is_vector = B_field.ndim == 1

    if is_vector:
        B_mag = np.sqrt(np.sum(B_field**2))
        B_mag_safe = max(B_mag, 1e-30)
        b_hat = B_field / B_mag_safe

        # b x grad(Te)
        cross = np.array([
            b_hat[1] * grad_Te[2] - b_hat[2] * grad_Te[1],
            b_hat[2] * grad_Te[0] - b_hat[0] * grad_Te[2],
            b_hat[0] * grad_Te[1] - b_hat[1] * grad_Te[0],
        ])

        beta_w = nernst_coefficient(ne_arr, Te_arr, np.atleast_1d(B_mag), Z_eff)
        ne_safe = np.maximum(ne_arr, 1e-20)
        prefactor = float(-beta_w[0] / (e_charge * ne_safe[0]))

        return prefactor * cross

    # Full 3-D field case: B_field shape (3, nx, ny, nz)
    B_mag = np.sqrt(np.sum(B_field**2, axis=0))
    B_mag_safe = np.maximum(B_mag, 1e-30)

    b_hat = B_field / B_mag_safe[np.newaxis, :, :, :]

    # b x grad(Te)
    cross = np.array([
        b_hat[1] * grad_Te[2] - b_hat[2] * grad_Te[1],
        b_hat[2] * grad_Te[0] - b_hat[0] * grad_Te[2],
        b_hat[0] * grad_Te[1] - b_hat[1] * grad_Te[0],
    ])

    beta_w = nernst_coefficient(ne_arr, Te_arr, B_mag, Z_eff)
    ne_safe = np.maximum(ne_arr, 1e-20)

    prefactor = -beta_w / (e_charge * ne_safe)

    E_nernst = prefactor[np.newaxis, :, :, :] * cross
    return E_nernst


# ============================================================
# Nernst advection update of B-field
# ============================================================

@njit(cache=True)
def _upwind_advect_component(
    B_comp: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> np.ndarray:
    """Upwind advection of one B-field component by Nernst velocity.

    Uses first-order upwind differencing for stability:
        B^{n+1} = B^n - dt * (v . grad(B))
    where the gradient is taken with upwind bias.

    Args:
        B_comp: One component of B, shape (nx, ny, nz).
        vx, vy, vz: Nernst velocity components, shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].
        dt: Timestep [s].

    Returns:
        Updated B component.
    """
    nx, ny, nz = B_comp.shape
    B_new = B_comp.copy()

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                # Upwind x-derivative
                if vx[i, j, k] > 0.0:
                    dBdx = (B_comp[i, j, k] - B_comp[i - 1, j, k]) / dx
                else:
                    dBdx = (B_comp[i + 1, j, k] - B_comp[i, j, k]) / dx

                # Upwind y-derivative
                if vy[i, j, k] > 0.0:
                    dBdy = (B_comp[i, j, k] - B_comp[i, j - 1, k]) / dy
                else:
                    dBdy = (B_comp[i, j + 1, k] - B_comp[i, j, k]) / dy

                # Upwind z-derivative
                if vz[i, j, k] > 0.0:
                    dBdz = (B_comp[i, j, k] - B_comp[i, j, k - 1]) / dz
                else:
                    dBdz = (B_comp[i, j, k + 1] - B_comp[i, j, k]) / dz

                B_new[i, j, k] = B_comp[i, j, k] - dt * (
                    vx[i, j, k] * dBdx + vy[i, j, k] * dBdy + vz[i, j, k] * dBdz
                )

    return B_new


def apply_nernst_advection(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    Z_eff: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advect the magnetic field by the Nernst velocity.

    Computes the Nernst velocity from ne, Te, and B, then performs a
    first-order upwind advection step on each B component.

    The Nernst term moves B along -grad(Te) in the plane perpendicular
    to B.  This can compress field into cold regions or expel it from
    hot spots, which is critical in ICF and Z-pinch implosion physics.

    Args:
        Bx, By, Bz: Magnetic field components [T], each shape (nx, ny, nz).
        ne: Electron number density [m^-3], shape (nx, ny, nz).
        Te: Electron temperature [K], shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].
        dt: Timestep [s].
        Z_eff: Effective ion charge state.

    Returns:
        Tuple (Bx_new, By_new, Bz_new) after Nernst advection.
    """
    B_field = np.array([Bx, By, Bz])

    # Compute Nernst velocity
    v_N = nernst_velocity(ne, Te, B_field, Z_eff)

    # Upwind advection of each B component
    Bx_new = _upwind_advect_component(Bx, v_N[0], v_N[1], v_N[2], dx, dy, dz, dt)
    By_new = _upwind_advect_component(By, v_N[0], v_N[1], v_N[2], dx, dy, dz, dt)
    Bz_new = _upwind_advect_component(Bz, v_N[0], v_N[1], v_N[2], dx, dy, dz, dt)

    # Replace non-finite values from numerical artifacts
    Bx_new = np.where(np.isfinite(Bx_new), Bx_new, Bx)
    By_new = np.where(np.isfinite(By_new), By_new, By)
    Bz_new = np.where(np.isfinite(Bz_new), Bz_new, Bz)

    return Bx_new, By_new, Bz_new
