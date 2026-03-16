"""Plasmoid detection and magnetic topology analysis (Challenge 14).

Detects magnetic islands (plasmoids) in 2D MHD output by:
1. Computing the magnetic flux function psi from B_r, B_z
2. Finding O-points (local extrema of psi) and X-points (saddle points)
3. Classifying topology as laminar, tearing, or plasmoid chain

For DPF: plasmoids form during the post-pinch disruption phase when
the current sheet exceeds the critical Lundquist number S > 10^4
(Loureiro et al. 2007).

References:
    Turner, Phys. Fluids B 5:2631 (1993) — relaxed states
    Loureiro et al., Phys. Plasmas 14:100703 (2007) — plasmoid instability
    Auluck & Upadhyay, Phys. Plasmas 30:092506 (2023) — DPF PMFE signals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpf.constants import mu_0


@dataclass
class PlasmoidResult:
    """Results of plasmoid detection."""

    n_plasmoids: int
    n_o_points: int
    n_x_points: int
    o_points: list[tuple[int, int]]  # (i, j) grid indices
    x_points: list[tuple[int, int]]
    psi_field: np.ndarray  # Magnetic flux function
    magnetic_energy_J: float
    topology: str  # "laminar", "tearing", "plasmoid_chain"


@dataclass
class ForceFreeDiag:
    """Force-free equilibrium diagnostic."""

    alpha_ff: float  # Force-free parameter: curl(B) = alpha * B
    j_parallel_frac: float  # Fraction of current parallel to B
    force_free_error: float  # |J x B| / (|J| * |B|) — 0 = perfect force-free
    is_relaxed: bool  # True if force_free_error < threshold


def compute_flux_function(
    B_field: np.ndarray,
    dr: float,
    dz: float,
) -> np.ndarray:
    """Compute magnetic flux function psi(r,z) from poloidal B-field.

    In cylindrical (r,z): B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr
    => psi = integral(r * B_z dr) at fixed z (cumulative trapezoid)

    In Cartesian 2D (midplane slice): use vector potential A_y
    where B_x = -dA_y/dz, B_z = dA_y/dx

    Args:
        B_field: Magnetic field (3, nx, ny, nz) [T].
        dr: Radial/x grid spacing [m].
        dz: Axial/z grid spacing [m].

    Returns:
        Flux function psi (nx, nz) from midplane slice [T*m^2].
    """
    ny_mid = B_field.shape[2] // 2
    # Take midplane slice
    Bx = B_field[0, :, ny_mid, :]  # shape (nx, nz)
    Bz = B_field[2, :, ny_mid, :]

    nx, nz = Bx.shape
    psi = np.zeros((nx, nz))

    # Integrate Bz along x to get psi: psi(x,z) = integral(Bz dx)
    for j in range(nz):
        psi[:, j] = np.cumsum(Bz[:, j]) * dr

    return psi


def find_critical_points(
    psi: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Find O-points (extrema) and X-points (saddle points) of psi.

    Uses discrete Hessian analysis:
    - O-point: det(H) > 0 (local min or max)
    - X-point: det(H) < 0 (saddle point)

    Args:
        psi: Flux function (nx, nz).

    Returns:
        (o_points, x_points) as lists of (i, j) grid indices.
    """
    nx, nz = psi.shape
    o_points = []
    x_points = []

    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            # Gradient (should be near zero at critical point)
            dpsi_dx = (psi[i + 1, j] - psi[i - 1, j]) / 2.0
            dpsi_dz = (psi[i, j + 1] - psi[i, j - 1]) / 2.0
            grad_mag = np.sqrt(dpsi_dx**2 + dpsi_dz**2)

            # Only consider points where gradient is small
            psi_scale = max(np.std(psi), 1e-30)
            if grad_mag > 0.1 * psi_scale:
                continue

            # Hessian
            d2_xx = psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j]
            d2_zz = psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1]
            d2_xz = (psi[i + 1, j + 1] - psi[i + 1, j - 1]
                      - psi[i - 1, j + 1] + psi[i - 1, j - 1]) / 4.0

            det_H = d2_xx * d2_zz - d2_xz**2

            if det_H > 0:
                o_points.append((i, j))
            elif det_H < 0:
                x_points.append((i, j))

    return o_points, x_points


def detect_plasmoids(
    B_field: np.ndarray,
    rho: np.ndarray,
    dr: float,
    dz: float,
) -> dict:
    """Detect plasmoids in MHD output.

    Args:
        B_field: Magnetic field (3, nx, ny, nz) [T].
        rho: Density (nx, ny, nz) [kg/m^3].
        dr: Grid spacing [m].
        dz: Grid spacing [m].

    Returns:
        Dict with plasmoid detection results.
    """
    psi = compute_flux_function(B_field, dr, dz)
    o_points, x_points = find_critical_points(psi)

    # Magnetic energy
    B_mag_sq = np.sum(B_field**2, axis=0)
    cell_vol = dr * dr * dz  # approximate
    E_mag = float(np.sum(B_mag_sq) / (2.0 * mu_0) * cell_vol)

    # Classify topology
    n_o = len(o_points)
    n_x = len(x_points)
    if n_o >= 3 and n_x >= 2:
        topology = "plasmoid_chain"
    elif n_o >= 1 and n_x >= 1:
        topology = "tearing"
    else:
        topology = "laminar"

    # Plasmoid count: each plasmoid = one O-point between two X-points
    n_plasmoids = max(0, min(n_o, n_x + 1)) if n_x > 0 else 0

    return {
        "n_plasmoids": n_plasmoids,
        "n_o_points": n_o,
        "n_x_points": n_x,
        "magnetic_energy_J": E_mag,
        "topology": topology,
        "psi_field": psi,
    }


def force_free_diagnostic(
    B_field: np.ndarray,
    dx: float,
    dz: float,
) -> ForceFreeDiag:
    """Compute force-free equilibrium diagnostic.

    A force-free state satisfies: J x B = 0, or equivalently
    curl(B) = alpha * B (linear force-free) or J || B.

    The force-free parameter alpha = mu_0 * (J . B) / B^2.

    Args:
        B_field: Magnetic field (3, nx, ny, nz) [T].
        dx: Grid spacing [m].
        dz: Grid spacing [m].

    Returns:
        ForceFreeDiag with relaxation metrics.
    """
    ny_mid = B_field.shape[2] // 2

    # Current density J = curl(B) / mu_0
    Bx = B_field[0, :, ny_mid, :]
    By = B_field[1, :, ny_mid, :]
    Bz = B_field[2, :, ny_mid, :]

    # curl(B) components (2D slice, finite differences)
    dBz_dx = np.gradient(Bz, dx, axis=0)
    dBx_dz = np.gradient(Bx, dz, axis=1)
    dBy_dx = np.gradient(By, dx, axis=0)
    dBy_dz = np.gradient(By, dz, axis=1)

    Jx = -dBy_dz / mu_0
    Jy = (dBz_dx - dBx_dz) / mu_0  # only non-trivial component in 2D
    Jz = dBy_dx / mu_0

    # J . B and |J x B|
    J_dot_B = Jx * Bx + Jy * By + Jz * Bz
    B_sq = Bx**2 + By**2 + Bz**2
    J_sq = Jx**2 + Jy**2 + Jz**2

    # J x B components
    JxB_x = Jy * Bz - Jz * By
    JxB_y = Jz * Bx - Jx * Bz
    JxB_z = Jx * By - Jy * Bx
    JxB_mag = np.sqrt(JxB_x**2 + JxB_y**2 + JxB_z**2)

    # Force-free error: |J x B| / (|J| * |B|)
    J_mag = np.sqrt(J_sq)
    B_mag = np.sqrt(B_sq)
    denom = np.maximum(J_mag * B_mag, 1e-30)
    ff_error_field = JxB_mag / denom

    # Mask regions with significant B and J
    active = (B_mag > 0.01 * np.max(B_mag)) & (J_mag > 0.01 * np.max(J_mag))
    if np.any(active):
        ff_error = float(np.mean(ff_error_field[active]))
        j_par_frac = float(np.mean(np.abs(J_dot_B[active]) / denom[active]))
    else:
        ff_error = 1.0
        j_par_frac = 0.0

    # Alpha parameter: alpha = mu_0 * <J.B> / <B^2>
    if np.any(active) and np.sum(B_sq[active]) > 0:
        alpha_ff = float(mu_0 * np.sum(J_dot_B[active]) / np.sum(B_sq[active]))
    else:
        alpha_ff = 0.0

    return ForceFreeDiag(
        alpha_ff=alpha_ff,
        j_parallel_frac=j_par_frac,
        force_free_error=ff_error,
        is_relaxed=ff_error < 0.3,  # EMPIRICAL: < 30% force-free error
    )
