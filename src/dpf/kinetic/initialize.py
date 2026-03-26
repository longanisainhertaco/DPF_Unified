"""Initialize PIC particles from MHD fluid state.

Samples macro-particle positions from the density distribution and
velocities from local Maxwellian at the ion temperature. Used when
activating the PIC module mid-discharge from a running MHD simulation.

References:
    Birdsall & Langdon, Plasma Physics via Computer Simulation (1985).
"""

from __future__ import annotations

import numpy as np

_K_B = 1.380649e-23


def initialize_particles_from_mhd(
    rho: np.ndarray,
    Te: np.ndarray,
    ion_mass: float = 3.3436e-27,
    n_particles: int = 10000,
    dx: float = 1e-3,
    dy: float | None = None,
    dz: float | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sample thermal particles from MHD density/temperature fields.

    Positions: density-weighted random sampling via CDF inversion.
    Velocities: Maxwellian at local Ti: v_th = sqrt(k_B * T / m_i).
    Weights: total_ions / n_particles.

    Args:
        rho: Mass density [kg/m^3], shape (nx, ny, nz) or (nx, nz).
        Te: Ion temperature [K], shape matching rho.
        ion_mass: Ion mass [kg].
        n_particles: Number of macro-particles to create.
        dx: Cell spacing x/r [m].
        dy: Cell spacing y [m]. If None, uses dx.
        dz: Cell spacing z [m]. If None, uses dx.
        seed: Random seed.

    Returns:
        Dict with 'positions' (N,3), 'velocities' (N,3), 'weights' (N,).
    """
    rng = np.random.default_rng(seed)
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # Handle 2D (nr, nz) by inserting ny=1
    if rho.ndim == 2:
        rho = rho[:, np.newaxis, :]
        Te = Te[:, np.newaxis, :]

    nx, ny, nz = rho.shape
    cell_vol = dx * dy * dz

    # Number density
    n_ion = np.maximum(rho / ion_mass, 0.0)
    total_ions = np.sum(n_ion) * cell_vol

    # CDF for density-weighted sampling
    prob = (n_ion / np.maximum(np.sum(n_ion), 1e-30)).ravel()
    indices = rng.choice(len(prob), size=n_particles, p=prob)

    # Convert flat indices to (ix, iy, iz)
    iz = indices % nz
    iy = (indices // nz) % ny
    ix = indices // (ny * nz)

    # Positions: cell center + uniform jitter within cell
    positions = np.zeros((n_particles, 3))
    positions[:, 0] = (ix + rng.uniform(0, 1, n_particles)) * dx
    positions[:, 1] = (iy + rng.uniform(0, 1, n_particles)) * dy
    positions[:, 2] = (iz + rng.uniform(0, 1, n_particles)) * dz

    # Velocities: Maxwellian at local temperature
    Te_flat = Te.ravel()
    T_local = np.maximum(Te_flat[indices], 1.0)
    v_th = np.sqrt(_K_B * T_local / ion_mass)

    velocities = np.zeros((n_particles, 3))
    for dim in range(3):
        velocities[:, dim] = rng.normal(0, v_th)

    # Weights: each macro-particle represents total_ions / n_particles real ions
    weights = np.full(n_particles, total_ions / max(n_particles, 1))

    return {
        "positions": positions.astype(np.float64),
        "velocities": velocities.astype(np.float64),
        "weights": weights.astype(np.float64),
    }
