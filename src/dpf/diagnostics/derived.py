"""Derived diagnostic quantities for visualization.

Computes physically meaningful quantities from raw simulation state
arrays for use by the server/GUI layer.

Functions are pure numpy — no Numba needed since these are called
infrequently (per snapshot, not per timestep).
"""

from __future__ import annotations

import numpy as np

from dpf.constants import mu_0


def current_density_magnitude(
    B: np.ndarray,
    dx: float,
    geometry: str = "cartesian",
) -> np.ndarray:
    """Compute current density magnitude |J| = |curl(B)| / mu_0.

    Args:
        B: Magnetic field array, shape (3, nx, ny, nz).
        dx: Grid spacing [m] (uniform in all directions).
        geometry: 'cartesian' or 'cylindrical'.

    Returns:
        |J| array with shape (nx, ny, nz), units [A/m^2].
    """
    if geometry == "cylindrical":
        # Cylindrical (r, theta, z) with B = (Br, Btheta, Bz)
        # B[0]=Br, B[1]=Btheta, B[2]=Bz; axis 0 = r, axis 1 = theta, axis 2 = z
        Br, Btheta, Bz = B[0], B[1], B[2]
        nr = B.shape[1]
        # Build radial coordinate array, broadcast to match grid shape
        r = np.arange(nr, dtype=B.dtype) * dx + 0.5 * dx  # cell-centered r
        r_shape = [1] * (B.ndim - 1)
        r_shape[0] = nr
        r = r.reshape(r_shape)
        r_safe = np.maximum(r, 1e-30)

        # curl(B)_r = (1/r)*dBz/dtheta - dBtheta/dz
        curl_r = (
            np.gradient(Bz, dx, axis=1) / r_safe
            - np.gradient(Btheta, dx, axis=2)
        )
        # curl(B)_theta = dBr/dz - dBz/dr
        curl_theta = (
            np.gradient(Br, dx, axis=2) - np.gradient(Bz, dx, axis=0)
        )
        # curl(B)_z = (1/r)*d(r*Btheta)/dr - (1/r)*dBr/dtheta
        curl_z = (
            np.gradient(r * Btheta, dx, axis=0) / r_safe
            - np.gradient(Br, dx, axis=1) / r_safe
        )

        J_mag = np.sqrt(curl_r**2 + curl_theta**2 + curl_z**2) / mu_0
        return J_mag

    # Cartesian geometry (default)
    Bx, By, Bz = B[0], B[1], B[2]

    # curl(B)_x = dBz/dy - dBy/dz
    # curl(B)_y = dBx/dz - dBz/dx
    # curl(B)_z = dBy/dx - dBx/dy
    curl_x = np.gradient(Bz, dx, axis=1) - np.gradient(By, dx, axis=2)
    curl_y = np.gradient(Bx, dx, axis=2) - np.gradient(Bz, dx, axis=0)
    curl_z = np.gradient(By, dx, axis=0) - np.gradient(Bx, dx, axis=1)

    J_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2) / mu_0
    return J_mag


def plasma_beta(
    pressure: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Compute plasma beta = 2 * mu_0 * p / B^2.

    Args:
        pressure: Thermal pressure [Pa], shape (nx, ny, nz).
        B: Magnetic field array, shape (3, nx, ny, nz).

    Returns:
        Beta array, same shape as pressure (dimensionless).
    """
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    # Avoid division by zero where B = 0
    B_sq_safe = np.maximum(B_sq, 1e-30)
    return 2.0 * mu_0 * np.maximum(pressure, 0.0) / B_sq_safe


def mach_number(
    velocity: np.ndarray,
    pressure: np.ndarray,
    rho: np.ndarray,
    gamma: float = 5.0 / 3.0,
) -> np.ndarray:
    """Compute sonic Mach number M = |v| / c_s.

    Args:
        velocity: Velocity array, shape (3, nx, ny, nz) [m/s].
        pressure: Pressure [Pa], shape (nx, ny, nz).
        rho: Mass density [kg/m^3], shape (nx, ny, nz).
        gamma: Adiabatic index.

    Returns:
        Mach number array, same shape as rho (dimensionless).
    """
    v_mag = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    c_s = np.sqrt(gamma * np.maximum(pressure, 0.0) / np.maximum(rho, 1e-30))
    return v_mag / np.maximum(c_s, 1e-30)


def alfven_speed(
    B: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Compute Alfven speed v_A = |B| / sqrt(mu_0 * rho).

    Args:
        B: Magnetic field array, shape (3, nx, ny, nz) [T].
        rho: Mass density [kg/m^3], shape (nx, ny, nz).

    Returns:
        Alfven speed array [m/s], same shape as rho.
    """
    B_mag = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    return B_mag / np.sqrt(mu_0 * np.maximum(rho, 1e-30))


def fast_magnetosonic_speed(
    B: np.ndarray,
    pressure: np.ndarray,
    rho: np.ndarray,
    gamma: float = 5.0 / 3.0,
) -> np.ndarray:
    """Compute fast magnetosonic speed c_f = sqrt(c_s^2 + v_A^2).

    Args:
        B: Magnetic field array, shape (3, nx, ny, nz) [T].
        pressure: Thermal pressure [Pa], shape (nx, ny, nz).
        rho: Mass density [kg/m^3], shape (nx, ny, nz).
        gamma: Adiabatic index.

    Returns:
        Fast magnetosonic speed [m/s], same shape as rho.
    """
    rho_safe = np.maximum(rho, 1e-30)
    c_s2 = gamma * np.maximum(pressure, 0.0) / rho_safe
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    v_A2 = B_sq / (mu_0 * rho_safe)
    return np.sqrt(c_s2 + v_A2)


def bennett_radius(
    current: float,
    Te: np.ndarray | float,
    ne: float,
) -> float:
    """Compute the Bennett equilibrium pinch radius.

    The Bennett relation balances magnetic pinch pressure against
    kinetic pressure:  mu_0 * I^2 / (8*pi) = N_L * k_B * (Te + Ti)

    For a uniform-density pinch with N_L = ne * pi * a^2:

        a_B = I * sqrt(mu_0 / (8 * pi^2 * ne * k_B * (Te + Ti)))

    where we assume Te = Ti (thermal equilibrium).

    Args:
        current: Pinch current [A].
        Te: Electron temperature [K] (scalar or array — mean used).
        ne: Electron number density [m^-3].

    Returns:
        Bennett radius [m].

    References:
        Bennett, W.H., Phys. Rev. 45, 890 (1934).
        Haines, M.G., Plasma Phys. Control. Fusion 53, 093001 (2011), Sec. 3.1.
    """
    k_B = 1.380649e-23
    Te_val = float(np.mean(Te)) if hasattr(Te, "__len__") else float(Te)
    T_total = 2.0 * max(Te_val, 1.0)  # Te + Ti
    ne_val = max(float(ne), 1e10)
    I_abs = abs(float(current))

    import math

    # a_B = I * sqrt(mu_0 / (8 * pi^2 * ne * kB * 2*Te))
    a_B = I_abs * math.sqrt(mu_0 / (8.0 * math.pi**2 * ne_val * k_B * T_total))
    return a_B
