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
