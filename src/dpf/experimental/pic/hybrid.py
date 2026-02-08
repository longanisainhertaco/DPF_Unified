"""Hybrid Fluid-PIC solver for Dense Plasma Focus simulations.

Implements a hybrid approach where electrons are treated as a massless
neutralising fluid and ions are tracked as macro-particles (the approach
used by Chicago/LSP codes for DPF modelling).

Key components:
    - ``ParticleSpecies``: container for macro-particle data (positions,
      velocities, weights).
    - ``boris_push``: Numba-accelerated Boris algorithm for charged-particle
      motion in combined E and B fields.
    - ``deposit_density`` / ``deposit_current``: cloud-in-cell (CIC)
      deposition of particles onto the grid.
    - ``interpolate_field_to_particles``: inverse CIC interpolation of grid
      fields to particle positions.
    - ``HybridPIC``: top-level driver that manages species, pushing,
      deposition, and beam injection.
    - ``detect_instability``: heuristic m=0 sausage instability detector.

Units: SI throughout (m, s, kg, C, V, T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from dpf.constants import e as e_charge

# =====================================================================
# Numba-accelerated kernels
# =====================================================================

@njit(cache=True)
def _boris_push_kernel(
    positions: np.ndarray,
    velocities: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
    charge: float,
    mass: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boris algorithm inner loop for N particles.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions [m].
    velocities : ndarray, shape (N, 3)
        Particle velocities [m/s].
    E_field : ndarray, shape (N, 3)
        Electric field interpolated to each particle position [V/m].
    B_field : ndarray, shape (N, 3)
        Magnetic field interpolated to each particle position [T].
    charge : float
        Particle charge [C].
    mass : float
        Particle mass [kg].
    dt : float
        Timestep [s].

    Returns
    -------
    new_pos : ndarray, shape (N, 3)
        Updated positions.
    new_vel : ndarray, shape (N, 3)
        Updated velocities.
    """
    n = positions.shape[0]
    new_pos = np.empty_like(positions)
    new_vel = np.empty_like(velocities)

    qdt_over_2m = charge * dt / (2.0 * mass)

    for i in range(n):
        # Half-acceleration from E
        vx_minus = velocities[i, 0] + qdt_over_2m * E_field[i, 0]
        vy_minus = velocities[i, 1] + qdt_over_2m * E_field[i, 1]
        vz_minus = velocities[i, 2] + qdt_over_2m * E_field[i, 2]

        # Rotation vector t = (q*B*dt)/(2*m)
        tx = qdt_over_2m * B_field[i, 0]
        ty = qdt_over_2m * B_field[i, 1]
        tz = qdt_over_2m * B_field[i, 2]

        t_mag2 = tx * tx + ty * ty + tz * tz

        # s = 2*t / (1 + |t|^2)
        s_factor = 2.0 / (1.0 + t_mag2)
        sx = s_factor * tx
        sy = s_factor * ty
        sz = s_factor * tz

        # v' = v_minus + v_minus x t
        vpx = vx_minus + (vy_minus * tz - vz_minus * ty)
        vpy = vy_minus + (vz_minus * tx - vx_minus * tz)
        vpz = vz_minus + (vx_minus * ty - vy_minus * tx)

        # v_plus = v_minus + v' x s
        vx_plus = vx_minus + (vpy * sz - vpz * sy)
        vy_plus = vy_minus + (vpz * sx - vpx * sz)
        vz_plus = vz_minus + (vpx * sy - vpy * sx)

        # Second half-acceleration from E
        new_vel[i, 0] = vx_plus + qdt_over_2m * E_field[i, 0]
        new_vel[i, 1] = vy_plus + qdt_over_2m * E_field[i, 1]
        new_vel[i, 2] = vz_plus + qdt_over_2m * E_field[i, 2]

        # Position update
        new_pos[i, 0] = positions[i, 0] + new_vel[i, 0] * dt
        new_pos[i, 1] = positions[i, 1] + new_vel[i, 1] * dt
        new_pos[i, 2] = positions[i, 2] + new_vel[i, 2] * dt

    return new_pos, new_vel


@njit(cache=True)
def _deposit_density_kernel(
    positions: np.ndarray,
    weights: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Cloud-in-cell (CIC) density deposition onto a 3-D grid.

    Each macro-particle is distributed to the 8 surrounding grid nodes
    using trilinear (volume-fraction) weighting.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    weights : ndarray, shape (N,)
        Macro-particle weights (physical particles per macro-particle).
    nx, ny, nz : int
        Grid dimensions.
    dx, dy, dz : float
        Grid spacings [m].

    Returns
    -------
    density : ndarray, shape (nx, ny, nz)
        Number density [m^-3].
    """
    density = np.zeros((nx, ny, nz), dtype=np.float64)
    cell_volume = dx * dy * dz

    for p in range(positions.shape[0]):
        # Normalised cell coordinates
        xn = positions[p, 0] / dx
        yn = positions[p, 1] / dy
        zn = positions[p, 2] / dz

        # Lower-left cell indices (clamped to valid range)
        ix = int(np.floor(xn))
        iy = int(np.floor(yn))
        iz = int(np.floor(zn))

        # Fractional offsets
        fx = xn - ix
        fy = yn - iy
        fz = zn - iz

        # Clamp indices to grid boundaries
        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iz = max(0, min(iz, nz - 2))

        # Ensure fractional parts are in [0, 1]
        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        fz = max(0.0, min(fz, 1.0))

        w = weights[p]

        # Trilinear deposition to 8 surrounding cells
        density[ix, iy, iz] += w * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        density[ix + 1, iy, iz] += w * fx * (1.0 - fy) * (1.0 - fz)
        density[ix, iy + 1, iz] += w * (1.0 - fx) * fy * (1.0 - fz)
        density[ix, iy, iz + 1] += w * (1.0 - fx) * (1.0 - fy) * fz
        density[ix + 1, iy + 1, iz] += w * fx * fy * (1.0 - fz)
        density[ix + 1, iy, iz + 1] += w * fx * (1.0 - fy) * fz
        density[ix, iy + 1, iz + 1] += w * (1.0 - fx) * fy * fz
        density[ix + 1, iy + 1, iz + 1] += w * fx * fy * fz

    # Convert from particle count to number density
    if cell_volume > 0.0:
        density /= cell_volume

    return density


@njit(cache=True)
def _deposit_current_kernel(
    positions: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray,
    charge: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIC deposition of current density J = q * n * v onto a 3-D grid.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    velocities : ndarray, shape (N, 3)
    weights : ndarray, shape (N,)
    charge : float
        Particle charge [C].
    nx, ny, nz : int
    dx, dy, dz : float

    Returns
    -------
    Jx, Jy, Jz : ndarray, each shape (nx, ny, nz)
        Current density components [A/m^2].
    """
    Jx = np.zeros((nx, ny, nz), dtype=np.float64)
    Jy = np.zeros((nx, ny, nz), dtype=np.float64)
    Jz = np.zeros((nx, ny, nz), dtype=np.float64)
    cell_volume = dx * dy * dz

    for p in range(positions.shape[0]):
        xn = positions[p, 0] / dx
        yn = positions[p, 1] / dy
        zn = positions[p, 2] / dz

        ix = int(np.floor(xn))
        iy = int(np.floor(yn))
        iz = int(np.floor(zn))

        fx = xn - ix
        fy = yn - iy
        fz = zn - iz

        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iz = max(0, min(iz, nz - 2))

        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        fz = max(0.0, min(fz, 1.0))

        # Current contribution: q * w * v
        qw = charge * weights[p]
        jx_p = qw * velocities[p, 0]
        jy_p = qw * velocities[p, 1]
        jz_p = qw * velocities[p, 2]

        # Trilinear weights
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w110 = fx * fy * (1.0 - fz)
        w101 = fx * (1.0 - fy) * fz
        w011 = (1.0 - fx) * fy * fz
        w111 = fx * fy * fz

        Jx[ix, iy, iz] += jx_p * w000
        Jx[ix + 1, iy, iz] += jx_p * w100
        Jx[ix, iy + 1, iz] += jx_p * w010
        Jx[ix, iy, iz + 1] += jx_p * w001
        Jx[ix + 1, iy + 1, iz] += jx_p * w110
        Jx[ix + 1, iy, iz + 1] += jx_p * w101
        Jx[ix, iy + 1, iz + 1] += jx_p * w011
        Jx[ix + 1, iy + 1, iz + 1] += jx_p * w111

        Jy[ix, iy, iz] += jy_p * w000
        Jy[ix + 1, iy, iz] += jy_p * w100
        Jy[ix, iy + 1, iz] += jy_p * w010
        Jy[ix, iy, iz + 1] += jy_p * w001
        Jy[ix + 1, iy + 1, iz] += jy_p * w110
        Jy[ix + 1, iy, iz + 1] += jy_p * w101
        Jy[ix, iy + 1, iz + 1] += jy_p * w011
        Jy[ix + 1, iy + 1, iz + 1] += jy_p * w111

        Jz[ix, iy, iz] += jz_p * w000
        Jz[ix + 1, iy, iz] += jz_p * w100
        Jz[ix, iy + 1, iz] += jz_p * w010
        Jz[ix, iy, iz + 1] += jz_p * w001
        Jz[ix + 1, iy + 1, iz] += jz_p * w110
        Jz[ix + 1, iy, iz + 1] += jz_p * w101
        Jz[ix, iy + 1, iz + 1] += jz_p * w011
        Jz[ix + 1, iy + 1, iz + 1] += jz_p * w111

    # Convert to current density [A/m^2]
    if cell_volume > 0.0:
        Jx /= cell_volume
        Jy /= cell_volume
        Jz /= cell_volume

    return Jx, Jy, Jz


@njit(cache=True)
def _interpolate_scalar_kernel(
    field: np.ndarray,
    positions: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Inverse CIC: interpolate a scalar grid field to particle positions.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
    positions : ndarray, shape (N, 3)
    dx, dy, dz : float

    Returns
    -------
    values : ndarray, shape (N,)
    """
    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]
    n = positions.shape[0]
    values = np.empty(n, dtype=np.float64)

    for p in range(n):
        xn = positions[p, 0] / dx
        yn = positions[p, 1] / dy
        zn = positions[p, 2] / dz

        ix = int(np.floor(xn))
        iy = int(np.floor(yn))
        iz = int(np.floor(zn))

        fx = xn - ix
        fy = yn - iy
        fz = zn - iz

        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iz = max(0, min(iz, nz - 2))

        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        fz = max(0.0, min(fz, 1.0))

        val = (
            field[ix, iy, iz] * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
            + field[ix + 1, iy, iz] * fx * (1.0 - fy) * (1.0 - fz)
            + field[ix, iy + 1, iz] * (1.0 - fx) * fy * (1.0 - fz)
            + field[ix, iy, iz + 1] * (1.0 - fx) * (1.0 - fy) * fz
            + field[ix + 1, iy + 1, iz] * fx * fy * (1.0 - fz)
            + field[ix + 1, iy, iz + 1] * fx * (1.0 - fy) * fz
            + field[ix, iy + 1, iz + 1] * (1.0 - fx) * fy * fz
            + field[ix + 1, iy + 1, iz + 1] * fx * fy * fz
        )
        values[p] = val

    return values


@njit(cache=True)
def _interpolate_vector_kernel(
    field: np.ndarray,
    positions: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Inverse CIC: interpolate a vector grid field to particle positions.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz, 3)
    positions : ndarray, shape (N, 3)
    dx, dy, dz : float

    Returns
    -------
    values : ndarray, shape (N, 3)
    """
    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]
    n = positions.shape[0]
    values = np.empty((n, 3), dtype=np.float64)

    for p in range(n):
        xn = positions[p, 0] / dx
        yn = positions[p, 1] / dy
        zn = positions[p, 2] / dz

        ix = int(np.floor(xn))
        iy = int(np.floor(yn))
        iz = int(np.floor(zn))

        fx = xn - ix
        fy = yn - iy
        fz = zn - iz

        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))
        iz = max(0, min(iz, nz - 2))

        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        fz = max(0.0, min(fz, 1.0))

        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w110 = fx * fy * (1.0 - fz)
        w101 = fx * (1.0 - fy) * fz
        w011 = (1.0 - fx) * fy * fz
        w111 = fx * fy * fz

        for c in range(3):
            values[p, c] = (
                field[ix, iy, iz, c] * w000
                + field[ix + 1, iy, iz, c] * w100
                + field[ix, iy + 1, iz, c] * w010
                + field[ix, iy, iz + 1, c] * w001
                + field[ix + 1, iy + 1, iz, c] * w110
                + field[ix + 1, iy, iz + 1, c] * w101
                + field[ix, iy + 1, iz + 1, c] * w011
                + field[ix + 1, iy + 1, iz + 1, c] * w111
            )

    return values


# =====================================================================
# Public API — thin wrappers around the Numba kernels
# =====================================================================

def boris_push(
    positions: np.ndarray,
    velocities: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
    charge: float,
    mass: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boris integrator for charged-particle motion in E and B fields.

    Applies the standard Boris split: half electric-field acceleration,
    magnetic-field rotation, then another half electric-field acceleration.
    The resulting velocity is time-centred (leap-frog) and preserves phase
    space volume.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Current particle positions [m].
    velocities : ndarray, shape (N, 3)
        Current particle velocities [m/s].
    E_field : ndarray, shape (N, 3)
        Electric field at each particle [V/m].
    B_field : ndarray, shape (N, 3)
        Magnetic field at each particle [T].
    charge : float
        Particle charge [C].
    mass : float
        Particle mass [kg].
    dt : float
        Timestep [s].

    Returns
    -------
    new_positions : ndarray, shape (N, 3)
    new_velocities : ndarray, shape (N, 3)
    """
    return _boris_push_kernel(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        np.ascontiguousarray(E_field, dtype=np.float64),
        np.ascontiguousarray(B_field, dtype=np.float64),
        float(charge),
        float(mass),
        float(dt),
    )


def deposit_density(
    positions: np.ndarray,
    weights: np.ndarray,
    grid_shape: tuple[int, int, int],
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Deposit particle number density onto a 3-D grid using CIC.

    Each macro-particle is spread across its 8 nearest grid nodes
    with trilinear (volume-fraction) weights.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions [m].
    weights : ndarray, shape (N,)
        Macro-particle weights (physical particles per macro-particle).
    grid_shape : tuple of 3 ints
        (nx, ny, nz) grid dimensions.
    dx, dy, dz : float
        Grid spacings [m].

    Returns
    -------
    density : ndarray, shape grid_shape
        Number density [m^-3].
    """
    return _deposit_density_kernel(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        float(dx),
        float(dy),
        float(dz),
    )


def deposit_current(
    positions: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray,
    charge: float,
    grid_shape: tuple[int, int, int],
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deposit current density J = q*n*v onto a 3-D grid using CIC.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    velocities : ndarray, shape (N, 3)
    weights : ndarray, shape (N,)
    charge : float
        Particle charge [C].
    grid_shape : tuple of 3 ints
    dx, dy, dz : float

    Returns
    -------
    Jx, Jy, Jz : ndarray, each shape grid_shape
        Current density components [A/m^2].
    """
    return _deposit_current_kernel(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        float(charge),
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        float(dx),
        float(dy),
        float(dz),
    )


def interpolate_field_to_particles(
    field: np.ndarray,
    positions: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Interpolate a grid field to particle positions using inverse CIC.

    Supports both scalar fields (shape ``(nx, ny, nz)``) and vector fields
    (shape ``(nx, ny, nz, 3)``).

    Parameters
    ----------
    field : ndarray
        Grid field, shape ``(nx, ny, nz)`` or ``(nx, ny, nz, 3)``.
    positions : ndarray, shape (N, 3)
        Particle positions [m].
    dx, dy, dz : float
        Grid spacings [m].

    Returns
    -------
    values : ndarray
        Shape ``(N,)`` for scalar field, ``(N, 3)`` for vector field.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    fld = np.ascontiguousarray(field, dtype=np.float64)

    if fld.ndim == 3:
        return _interpolate_scalar_kernel(fld, pos, float(dx), float(dy), float(dz))
    elif fld.ndim == 4 and fld.shape[3] == 3:
        return _interpolate_vector_kernel(fld, pos, float(dx), float(dy), float(dz))
    else:
        raise ValueError(
            f"field must have shape (nx,ny,nz) or (nx,ny,nz,3), got {fld.shape}"
        )


# =====================================================================
# ParticleSpecies dataclass
# =====================================================================

@dataclass
class ParticleSpecies:
    """Container for a single particle species in the hybrid PIC solver.

    Attributes
    ----------
    name : str
        Species identifier (e.g. ``"deuterium"``, ``"beam_D"``).
    mass : float
        Particle mass [kg].
    charge : float
        Particle charge [C].
    positions : ndarray, shape (N, 3)
        Macro-particle positions [m].
    velocities : ndarray, shape (N, 3)
        Macro-particle velocities [m/s].
    weights : ndarray, shape (N,)
        Macro-particle weights (physical particles per macro-particle).
    """

    name: str
    mass: float
    charge: float
    positions: np.ndarray
    velocities: np.ndarray
    weights: np.ndarray

    def n_particles(self) -> int:
        """Return the number of active macro-particles."""
        return int(self.positions.shape[0])


# =====================================================================
# HybridPIC driver
# =====================================================================

class HybridPIC:
    """Hybrid fluid-PIC driver for ion kinetics in a DPF.

    Electrons are treated as a massless charge-neutralising fluid; ions
    are tracked as macro-particles pushed with the Boris algorithm.

    Parameters
    ----------
    grid_shape : tuple of 3 ints
        (nx, ny, nz) grid dimensions.
    dx, dy, dz : float
        Grid spacings [m].
    dt : float
        Default timestep [s].
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        dy: float,
        dz: float,
        dt: float,
    ) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.species: list[ParticleSpecies] = []

        # Domain extents (particle positions live in [0, L])
        self._Lx = grid_shape[0] * dx
        self._Ly = grid_shape[1] * dy
        self._Lz = grid_shape[2] * dz

    # -----------------------------------------------------------------
    # Species management
    # -----------------------------------------------------------------

    def add_species(
        self,
        name: str,
        mass: float,
        charge: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        weights: np.ndarray,
    ) -> ParticleSpecies:
        """Add an ion species to the simulation.

        Parameters
        ----------
        name : str
            Species identifier.
        mass : float
            Ion mass [kg].
        charge : float
            Ion charge [C].
        positions : ndarray, shape (N, 3)
            Initial positions [m].
        velocities : ndarray, shape (N, 3)
            Initial velocities [m/s].
        weights : ndarray, shape (N,)
            Macro-particle weights.

        Returns
        -------
        ParticleSpecies
            The newly created species (also appended to ``self.species``).
        """
        sp = ParticleSpecies(
            name=name,
            mass=float(mass),
            charge=float(charge),
            positions=np.asarray(positions, dtype=np.float64),
            velocities=np.asarray(velocities, dtype=np.float64),
            weights=np.asarray(weights, dtype=np.float64),
        )
        self.species.append(sp)
        return sp

    # -----------------------------------------------------------------
    # Particle push
    # -----------------------------------------------------------------

    def push_particles(
        self,
        E: np.ndarray,
        B: np.ndarray,
        dt: float | None = None,
    ) -> None:
        """Push all species using the Boris algorithm and apply BCs.

        Parameters
        ----------
        E : ndarray, shape (nx, ny, nz, 3)
            Electric field on grid [V/m].
        B : ndarray, shape (nx, ny, nz, 3)
            Magnetic field on grid [T].
        dt : float or None
            Timestep [s].  If None, use ``self.dt``.
        """
        if dt is None:
            dt = self.dt

        for sp in self.species:
            if sp.n_particles() == 0:
                continue

            # Interpolate fields to particle positions
            E_at_p = interpolate_field_to_particles(E, sp.positions, self.dx, self.dy, self.dz)
            B_at_p = interpolate_field_to_particles(B, sp.positions, self.dx, self.dy, self.dz)

            # Boris push
            new_pos, new_vel = boris_push(
                sp.positions, sp.velocities, E_at_p, B_at_p,
                sp.charge, sp.mass, dt,
            )

            # Reflecting boundary conditions at domain edges
            new_pos, new_vel = self._apply_reflecting_bc(new_pos, new_vel)

            sp.positions = new_pos
            sp.velocities = new_vel

    def _apply_reflecting_bc(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply reflecting boundary conditions at domain edges.

        Particles that exit the domain are reflected back in, with
        the corresponding velocity component reversed.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
        velocities : ndarray, shape (N, 3)

        Returns
        -------
        positions, velocities : reflected arrays
        """
        limits = [self._Lx, self._Ly, self._Lz]
        for d in range(3):
            # Reflect off lower boundary (x < 0)
            mask_lo = positions[:, d] < 0.0
            positions[mask_lo, d] = -positions[mask_lo, d]
            velocities[mask_lo, d] = -velocities[mask_lo, d]

            # Reflect off upper boundary (x > L)
            mask_hi = positions[:, d] > limits[d]
            positions[mask_hi, d] = 2.0 * limits[d] - positions[mask_hi, d]
            velocities[mask_hi, d] = -velocities[mask_hi, d]

        return positions, velocities

    # -----------------------------------------------------------------
    # Deposition
    # -----------------------------------------------------------------

    def deposit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deposit all species onto the grid.

        Returns
        -------
        rho_grid : ndarray, shape grid_shape
            Total charge density [C/m^3].
        Jx, Jy, Jz : ndarray, each shape grid_shape
            Total current density components [A/m^2].
        """
        rho_grid = np.zeros(self.grid_shape, dtype=np.float64)
        Jx = np.zeros(self.grid_shape, dtype=np.float64)
        Jy = np.zeros(self.grid_shape, dtype=np.float64)
        Jz = np.zeros(self.grid_shape, dtype=np.float64)

        for sp in self.species:
            if sp.n_particles() == 0:
                continue

            # Number density -> charge density
            n = deposit_density(
                sp.positions, sp.weights, self.grid_shape,
                self.dx, self.dy, self.dz,
            )
            rho_grid += sp.charge * n

            # Current density
            jx, jy, jz = deposit_current(
                sp.positions, sp.velocities, sp.weights, sp.charge,
                self.grid_shape, self.dx, self.dy, self.dz,
            )
            Jx += jx
            Jy += jy
            Jz += jz

        return rho_grid, Jx, Jy, Jz

    # -----------------------------------------------------------------
    # Beam injection
    # -----------------------------------------------------------------

    def inject_beam(
        self,
        species_idx: int,
        n_beam: int,
        energy_eV: float,
        direction: np.ndarray | list[float] | tuple[float, float, float],
        position: np.ndarray | list[float] | tuple[float, float, float],
        spread: float = 0.0,
    ) -> None:
        """Inject a beam of particles into an existing species.

        Used for modelling beam-target neutron generation in the DPF
        pinch phase.

        Parameters
        ----------
        species_idx : int
            Index into ``self.species`` for the target species.
        n_beam : int
            Number of macro-particles to inject.
        energy_eV : float
            Beam kinetic energy per particle [eV].
        direction : array-like, shape (3,)
            Unit vector for beam direction.
        position : array-like, shape (3,)
            Starting position (x, y, z) [m].
        spread : float
            Angular spread in radians (half-angle of cone).
        """
        sp = self.species[species_idx]
        dir_vec = np.asarray(direction, dtype=np.float64)
        dir_vec = dir_vec / np.maximum(np.linalg.norm(dir_vec), 1e-300)

        pos0 = np.asarray(position, dtype=np.float64)

        # Kinetic energy -> speed: E = 0.5 * m * v^2
        speed = np.sqrt(2.0 * energy_eV * e_charge / sp.mass)

        # Generate beam velocities
        beam_vel = np.zeros((n_beam, 3), dtype=np.float64)

        if spread > 0.0:
            # Random perturbations within a cone of half-angle `spread`
            for i in range(n_beam):
                # Random rotation angles
                theta = spread * np.sqrt(np.random.random())  # uniform in solid angle
                phi = 2.0 * np.pi * np.random.random()

                # Perpendicular basis vectors
                if abs(dir_vec[0]) < 0.9:
                    perp1 = np.cross(dir_vec, np.array([1.0, 0.0, 0.0]))
                else:
                    perp1 = np.cross(dir_vec, np.array([0.0, 1.0, 0.0]))
                perp1 = perp1 / np.maximum(np.linalg.norm(perp1), 1e-300)
                perp2 = np.cross(dir_vec, perp1)

                # Perturbed direction
                v_dir = (
                    dir_vec * np.cos(theta)
                    + perp1 * np.sin(theta) * np.cos(phi)
                    + perp2 * np.sin(theta) * np.sin(phi)
                )
                beam_vel[i] = speed * v_dir
        else:
            for i in range(n_beam):
                beam_vel[i] = speed * dir_vec

        # All beam particles start at the same position
        beam_pos = np.tile(pos0, (n_beam, 1))

        # Default weight = 1.0 for each beam macro-particle
        beam_weights = np.ones(n_beam, dtype=np.float64)

        # Append to existing species arrays
        sp.positions = np.concatenate([sp.positions, beam_pos], axis=0)
        sp.velocities = np.concatenate([sp.velocities, beam_vel], axis=0)
        sp.weights = np.concatenate([sp.weights, beam_weights], axis=0)


# =====================================================================
# Instability detection
# =====================================================================

def detect_instability(
    rho: np.ndarray,
    B_field: np.ndarray,
    threshold: float = 5.0,
) -> bool:
    """Detect m=0 (sausage) instability from density and magnetic field.

    Heuristic criteria:
        1. Density compression: max(rho) / mean(rho) > threshold.
        2. B_z sign change along the axis (indicative of current
           filamentation / necking).

    Parameters
    ----------
    rho : ndarray, shape (nx, ny, nz)
        Mass or number density.
    B_field : ndarray, shape (nx, ny, nz, 3)
        Magnetic field (Bx, By, Bz).
    threshold : float
        Density compression ratio threshold (default 5.0).

    Returns
    -------
    bool
        True if instability is detected.
    """
    mean_rho = np.mean(rho)
    if mean_rho <= 0.0:
        return False

    # Criterion 1: density compression
    compression_ratio = np.max(rho) / mean_rho
    density_compressed = compression_ratio > threshold

    # Criterion 2: B_z sign change along axis
    # Take the central axis line (mid-x, mid-y, all z)
    nx, ny, nz = rho.shape
    mid_x = nx // 2
    mid_y = ny // 2

    Bz_axis = B_field[mid_x, mid_y, :, 2]
    has_sign_change = bool(np.any(Bz_axis[:-1] * Bz_axis[1:] < 0.0))

    return density_compressed and has_sign_change
