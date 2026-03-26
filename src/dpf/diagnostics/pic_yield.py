"""PIC-based neutron yield calculation from kinetic ion distributions.

Replaces the analytical beam_target_yield_rate(I, V_pinch, n, L) with
a first-principles calculation using individual macro-particle velocities
and the Bosch-Hale D-D cross section.

For each macro-particle, the contribution to the yield rate is:
    dY/dt_i = w_i * n_target(x_i) * sigma_DD(E_cm_i) * v_rel_i

where w_i is the macro-particle weight, n_target is the local background
density, E_cm = m_D * v_rel^2 / 4 (equal-mass CM), and sigma_DD is the
Bosch-Hale parametrization.

References:
    Bosch & Hale, Nucl. Fusion 32:611 (1992) — DD cross section.
    Schmidt et al., PRL 109:205003 (2012) — kinetic DPF neutrons.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.diagnostics.beam_target import dd_cross_section

_EV = 1.602176634e-19
_M_D = 3.3436e-27


@njit(cache=True)
def _interpolate_scalar_cic(
    field: np.ndarray,
    x: float, y: float, z: float,
    dx: float, dy: float, dz: float,
) -> float:
    """CIC interpolation of a scalar field to a particle position."""
    nx, ny, nz = field.shape
    ix = x / dx
    iy = y / dy
    iz = z / dz

    i0 = int(ix)
    j0 = int(iy)
    k0 = int(iz)

    if i0 < 0 or i0 >= nx - 1 or j0 < 0 or j0 >= ny - 1 or k0 < 0 or k0 >= nz - 1:
        return 0.0

    fx = ix - i0
    fy = iy - j0
    fz = iz - k0

    val = (
        field[i0, j0, k0] * (1 - fx) * (1 - fy) * (1 - fz)
        + field[i0 + 1, j0, k0] * fx * (1 - fy) * (1 - fz)
        + field[i0, j0 + 1, k0] * (1 - fx) * fy * (1 - fz)
        + field[i0, j0, k0 + 1] * (1 - fx) * (1 - fy) * fz
        + field[i0 + 1, j0 + 1, k0] * fx * fy * (1 - fz)
        + field[i0 + 1, j0, k0 + 1] * fx * (1 - fy) * fz
        + field[i0, j0 + 1, k0 + 1] * (1 - fx) * fy * fz
        + field[i0 + 1, j0 + 1, k0 + 1] * fx * fy * fz
    )
    return val


@njit(cache=True)
def pic_neutron_yield_rate(
    positions: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray,
    n_target: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    ion_mass: float = 3.3436e-27,
) -> float:
    """Compute D-D neutron yield rate from PIC particle distribution.

    For each macro-particle, computes the beam-target reaction rate
    using the local background density and the Bosch-Hale cross section.

    Args:
        positions: Particle positions [m], shape (N, 3).
        velocities: Particle velocities [m/s], shape (N, 3).
        weights: Macro-particle weights, shape (N,).
        n_target: Background ion number density [m^-3], shape (nx, ny, nz).
        dx, dy, dz: Grid spacings [m].
        ion_mass: Ion mass [kg]. Default: deuterium.

    Returns:
        dY/dt [neutrons/s].
    """
    n_particles = positions.shape[0]
    dY_dt = 0.0
    eV_val = 1.602176634e-19

    for i in range(n_particles):
        vx = velocities[i, 0]
        vy = velocities[i, 1]
        vz = velocities[i, 2]
        v_rel = (vx * vx + vy * vy + vz * vz) ** 0.5

        if v_rel < 1e3:
            continue

        n_local = _interpolate_scalar_cic(
            n_target,
            positions[i, 0], positions[i, 1], positions[i, 2],
            dx, dy, dz,
        )

        if n_local <= 0:
            continue

        E_cm_J = 0.25 * ion_mass * v_rel * v_rel
        E_cm_keV = E_cm_J / (1e3 * eV_val)

        sigma = dd_cross_section(E_cm_keV)

        dY_dt += weights[i] * n_local * sigma * v_rel

    return dY_dt
