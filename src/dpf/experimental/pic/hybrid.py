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
def _nanbu_scatter_kernel(
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    mass_a: float,
    mass_b: float,
    charge_a: float,
    charge_b: float,
    n_a: float,
    n_b: float,
    ln_Lambda: float,
    dt: float,
    cell_volume: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nanbu-Perez relativistic binary Coulomb collision kernel.

    Implements the relativistic binary collision algorithm of Perez et al.
    (2012), Phys. Plasmas 19, 083104, which generalises Nanbu (1997) to
    arbitrary masses and relativistic velocities.

    Each pair of macro-particles (a, b) is scattered by a stochastic
    deflection angle chi drawn from the Nanbu distribution parameterised
    by s12.  The scatter preserves total relativistic 4-momentum by
    working in the centre-of-momentum (COM) frame and then Lorentz-boosting
    back to the lab frame.

    Parameters
    ----------
    vel_a, vel_b : ndarray, shape (N, 3)
        Lab-frame velocities [m/s] for species a and b. Modified in-place.
    weights_a, weights_b : ndarray, shape (N,)
        Macro-particle weights (physical particles per macro-particle).
    mass_a, mass_b : float
        Rest masses [kg].
    charge_a, charge_b : float
        Charges [C].
    n_a, n_b : float
        Number densities [m^-3].
    ln_Lambda : float
        Coulomb logarithm.
    dt : float
        Timestep [s].
    cell_volume : float
        Volume of the collision cell [m^3].

    Returns
    -------
    vel_a, vel_b : ndarray, shape (N, 3)
        Updated velocities (also modified in-place).

    References
    ----------
    Perez et al. (2012) Phys. Plasmas 19, 083104.
    Nanbu (1997) Phys. Rev. E 55, 4642.
    """
    c = 2.998e8
    eps0 = 8.854e-12
    pi = np.pi

    N = min(vel_a.shape[0], vel_b.shape[0])
    if N == 0:
        return vel_a, vel_b

    # Shuffle indices for random pairing (Fisher-Yates inside njit)
    idx_a = np.arange(N)
    idx_b = np.arange(N)
    for k in range(N - 1, 0, -1):
        j = int(np.random.random() * (k + 1))
        idx_a[k], idx_a[j] = idx_a[j], idx_a[k]
        j = int(np.random.random() * (k + 1))
        idx_b[k], idx_b[j] = idx_b[j], idx_b[k]

    c2 = c * c
    # Pre-factor for s12: q1^2*q2^2 / (4*pi*eps0^2*c^4)
    q2q2 = charge_a**2 * charge_b**2
    s12_prefactor = q2q2 / (4.0 * pi * eps0**2 * c2 * c2)

    for pair in range(N):
        ia = idx_a[pair]
        ib = idx_b[pair]

        vax = vel_a[ia, 0]
        vay = vel_a[ia, 1]
        vaz = vel_a[ia, 2]
        vbx = vel_b[ib, 0]
        vby = vel_b[ib, 1]
        vbz = vel_b[ib, 2]

        va2 = vax**2 + vay**2 + vaz**2
        vb2 = vbx**2 + vby**2 + vbz**2

        # Guard against superluminal velocities (numerical noise)
        if va2 >= c2:
            scale = 0.9999 * c / np.sqrt(va2)
            vax, vay, vaz = vax * scale, vay * scale, vaz * scale
            va2 = vax**2 + vay**2 + vaz**2
        if vb2 >= c2:
            scale = 0.9999 * c / np.sqrt(vb2)
            vbx, vby, vbz = vbx * scale, vby * scale, vbz * scale
            vb2 = vbx**2 + vby**2 + vbz**2

        gamma_a = 1.0 / np.sqrt(1.0 - va2 / c2)
        gamma_b = 1.0 / np.sqrt(1.0 - vb2 / c2)

        # 1. COM 3-velocity
        pa_fac = mass_a * gamma_a
        pb_fac = mass_b * gamma_b
        denom = pa_fac + pb_fac
        vcx = (pa_fac * vax + pb_fac * vbx) / denom
        vcy = (pa_fac * vay + pb_fac * vby) / denom
        vcz = (pa_fac * vaz + pb_fac * vbz) / denom

        vc2 = vcx**2 + vcy**2 + vcz**2
        if vc2 >= c2:
            vc2 = 0.9999 * c2
        gamma_com = 1.0 / np.sqrt(1.0 - vc2 / c2)

        # 2. Boost particle a to COM frame
        if vc2 < 1e-30:
            u_ax = vax
            u_ay = vay
            u_az = vaz
        else:
            dot_a = vax * vcx + vay * vcy + vaz * vcz
            factor = (gamma_com - 1.0) * dot_a / vc2 - gamma_com
            u_ax = vax + factor * vcx
            u_ay = vay + factor * vcy
            u_az = vaz + factor * vcz

        u_a2 = u_ax**2 + u_ay**2 + u_az**2
        if u_a2 >= c2:
            u_a2 = 0.9999 * c2
        gamma_a_com = 1.0 / np.sqrt(1.0 - u_a2 / c2)

        # COM-frame momentum magnitude |p*| = m_a * gamma_a_com * |u_a|
        p_star = mass_a * gamma_a_com * np.sqrt(u_a2)
        if p_star < 1e-40:
            continue

        # 3. s12 parameter (Perez 2012, eq. 14)
        s12 = (
            s12_prefactor
            * gamma_com * p_star
            * n_b * ln_Lambda * dt
            / (mass_a * gamma_a * mass_b * gamma_b)
        )

        # 4. Sample cos(chi) from Nanbu distribution
        u_rand = np.random.random()
        if s12 < 4.0:
            s12_2 = s12 * s12
            s12_3 = s12_2 * s12
            alpha = 0.37 * s12 - 0.005 * s12_2 - 0.0064 * s12_3
            alpha = max(alpha, 1e-10)
            # sin^2(chi/2) = alpha*u / sqrt((1-u) + alpha^2*u)
            sin2_half = alpha * u_rand / np.sqrt(
                (1.0 - u_rand) + alpha * alpha * u_rand
            )
            cos_chi = 1.0 - 2.0 * sin2_half
        else:
            # Isotropic scattering for large s12
            cos_chi = 2.0 * u_rand - 1.0

        cos_chi = max(-1.0, min(1.0, cos_chi))
        sin_chi = np.sqrt(max(0.0, 1.0 - cos_chi**2))
        phi = 2.0 * pi * np.random.random()
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # 5. Rotate momentum direction in COM frame
        u_a_mag = np.sqrt(u_a2)
        ex = u_ax / u_a_mag
        ey = u_ay / u_a_mag
        ez = u_az / u_a_mag

        # Gram-Schmidt perpendicular basis
        if abs(ex) < 0.9:
            p1x = 0.0
            p1y = -ez
            p1z = ey
        else:
            p1x = ez
            p1y = 0.0
            p1z = -ex
        p1norm = np.sqrt(p1x**2 + p1y**2 + p1z**2)
        if p1norm < 1e-30:
            continue
        p1x /= p1norm
        p1y /= p1norm
        p1z /= p1norm

        p2x = ey * p1z - ez * p1y
        p2y = ez * p1x - ex * p1z
        p2z = ex * p1y - ey * p1x

        nx_ = cos_chi * ex + sin_chi * (cos_phi * p1x + sin_phi * p2x)
        ny_ = cos_chi * ey + sin_chi * (cos_phi * p1y + sin_phi * p2y)
        nz_ = cos_chi * ez + sin_chi * (cos_phi * p1z + sin_phi * p2z)

        u_a_new_x = u_a_mag * nx_
        u_a_new_y = u_a_mag * ny_
        u_a_new_z = u_a_mag * nz_

        # 6. Inverse Lorentz boost back to lab frame
        if vc2 < 1e-30:
            new_vax = u_a_new_x
            new_vay = u_a_new_y
            new_vaz = u_a_new_z
            mass_ratio = mass_a / mass_b
            new_vbx = -mass_ratio * u_a_new_x
            new_vby = -mass_ratio * u_a_new_y
            new_vbz = -mass_ratio * u_a_new_z
        else:
            dot_new = u_a_new_x * vcx + u_a_new_y * vcy + u_a_new_z * vcz
            denom_boost = 1.0 + dot_new / c2
            fac_a = (gamma_com - 1.0) * dot_new / vc2 + gamma_com
            new_vax = (u_a_new_x + fac_a * vcx) / (gamma_com * denom_boost)
            new_vay = (u_a_new_y + fac_a * vcy) / (gamma_com * denom_boost)
            new_vaz = (u_a_new_z + fac_a * vcz) / (gamma_com * denom_boost)
            # p_b* = -p_a* in COM, so u_b_new ~ -(m_a/m_b)*u_a_new
            mass_ratio = mass_a / mass_b
            u_b_new_x = -mass_ratio * u_a_new_x
            u_b_new_y = -mass_ratio * u_a_new_y
            u_b_new_z = -mass_ratio * u_a_new_z
            dot_b = u_b_new_x * vcx + u_b_new_y * vcy + u_b_new_z * vcz
            denom_b = 1.0 + dot_b / c2
            fac_b = (gamma_com - 1.0) * dot_b / vc2 + gamma_com
            new_vbx = (u_b_new_x + fac_b * vcx) / (gamma_com * denom_b)
            new_vby = (u_b_new_y + fac_b * vcy) / (gamma_com * denom_b)
            new_vbz = (u_b_new_z + fac_b * vcz) / (gamma_com * denom_b)

        # Clamp to subluminal
        va_new2 = new_vax**2 + new_vay**2 + new_vaz**2
        if va_new2 >= c2:
            s = 0.9999 * c / np.sqrt(va_new2)
            new_vax, new_vay, new_vaz = new_vax * s, new_vay * s, new_vaz * s
        vb_new2 = new_vbx**2 + new_vby**2 + new_vbz**2
        if vb_new2 >= c2:
            s = 0.9999 * c / np.sqrt(vb_new2)
            new_vbx, new_vby, new_vbz = new_vbx * s, new_vby * s, new_vbz * s

        vel_a[ia, 0] = new_vax
        vel_a[ia, 1] = new_vay
        vel_a[ia, 2] = new_vaz
        vel_b[ib, 0] = new_vbx
        vel_b[ib, 1] = new_vby
        vel_b[ib, 2] = new_vbz

    return vel_a, vel_b


@njit(cache=True)
def _coulomb_scatter(
    velocities: np.ndarray,
    charge: float,
    mass: float,
    n_bg: float,
    T_bg_eV: float,
    dt: float,
) -> np.ndarray:
    """Coulomb collision operator (Takizuka-Abe 1977 simplified).

    Applies pitch-angle scattering to each particle based on its speed
    relative to a Maxwellian background plasma. The deflection angle
    per timestep is drawn from a Gaussian with variance proportional
    to the collision frequency nu_ei.

    This models beam-ion slowing down in the target plasma.

    Args:
        velocities: Particle velocities (N, 3) [m/s].
        charge: Particle charge [C].
        mass: Particle mass [kg].
        n_bg: Background plasma density [m^-3].
        T_bg_eV: Background temperature [eV].
        dt: Timestep [s].

    Returns:
        Scattered velocities (N, 3) [m/s].
    """
    N = velocities.shape[0]
    if N == 0:
        return velocities

    result = velocities.copy()
    eps0 = 8.854187817e-12
    eV = 1.602176634e-19

    for i in range(N):
        vx, vy, vz = result[i, 0], result[i, 1], result[i, 2]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v < 1e-10:
            continue

        # Coulomb logarithm (simplified)
        E_kin_eV = 0.5 * mass * v**2 / eV
        if E_kin_eV < 1.0:
            continue
        ln_Lambda = max(5.0, 23.0 - 0.5 * np.log(n_bg / 1e20) + 1.5 * np.log(T_bg_eV))

        # Collision frequency: nu = n_bg * Z^2 * e^4 * ln_Lambda / (4*pi*eps0^2 * m^2 * v^3)
        # EMPIRICAL: capped to prevent unphysical scattering in single timestep
        nu = n_bg * charge**4 * ln_Lambda / (
            4.0 * np.pi * eps0**2 * mass**2 * v**3
        )
        nu = min(nu, 0.5 / dt)  # cap: max 0.5 rad deflection per step

        # RMS scattering angle per timestep
        theta_rms = np.sqrt(nu * dt)

        # Random pitch-angle scatter (2D rotation in velocity space)
        # Generate two uniform random numbers
        u1 = np.random.random()
        u2 = np.random.random()
        # Box-Muller for Gaussian scattering angle
        theta = theta_rms * np.sqrt(-2.0 * np.log(max(u1, 1e-30))) * np.cos(2.0 * np.pi * u2)
        phi = 2.0 * np.pi * np.random.random()

        # Rotate velocity vector by (theta, phi) around a random axis
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)

        # Unit vector along velocity
        vhat_x, vhat_y, vhat_z = vx / v, vy / v, vz / v

        # Perpendicular unit vectors (Gram-Schmidt)
        if abs(vhat_x) < 0.9:
            perp1_x = 0.0
            perp1_y = -vhat_z
            perp1_z = vhat_y
        else:
            perp1_x = vhat_z
            perp1_y = 0.0
            perp1_z = -vhat_x
        pnorm = np.sqrt(perp1_x**2 + perp1_y**2 + perp1_z**2)
        if pnorm < 1e-30:
            continue
        perp1_x /= pnorm
        perp1_y /= pnorm
        perp1_z /= pnorm

        perp2_x = vhat_y * perp1_z - vhat_z * perp1_y
        perp2_y = vhat_z * perp1_x - vhat_x * perp1_z
        perp2_z = vhat_x * perp1_y - vhat_y * perp1_x

        # New direction after scattering
        new_vhat_x = ct * vhat_x + st * (cp * perp1_x + sp * perp2_x)
        new_vhat_y = ct * vhat_y + st * (cp * perp1_y + sp * perp2_y)
        new_vhat_z = ct * vhat_z + st * (cp * perp1_z + sp * perp2_z)

        # Preserve speed (elastic scattering in CM frame)
        result[i, 0] = v * new_vhat_x
        result[i, 1] = v * new_vhat_y
        result[i, 2] = v * new_vhat_z

    return result


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
def _deposit_current_esirkepov_kernel(
    positions_old: np.ndarray,
    positions_new: np.ndarray,
    weights: np.ndarray,
    charge: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Esirkepov (2001) charge-conserving current deposition for CIC shape functions.

    Computes J such that the discrete continuity equation
    ``d(rho)/dt + div(J) = 0`` is satisfied exactly on the grid.

    For CIC (linear shape), each particle spans 2 nodes per direction.
    The algorithm accumulates W-factors along each axis using the running-
    sum technique described in Esirkepov, Comput. Phys. Commun. 135 (2001).

    Parameters
    ----------
    positions_old : ndarray, shape (N, 3)
        Particle positions at time n [m].
    positions_new : ndarray, shape (N, 3)
        Particle positions at time n+1 [m].
    weights : ndarray, shape (N,)
        Macro-particle weights (physical particles per macro-particle).
    charge : float
        Particle charge [C].
    nx, ny, nz : int
        Grid dimensions.
    dx, dy, dz : float
        Grid spacings [m].
    dt : float
        Timestep [s].

    Returns
    -------
    Jx, Jy, Jz : ndarray, each shape (nx, ny, nz)
        Current density components [A/m^2].
    """
    Jx = np.zeros((nx, ny, nz), dtype=np.float64)
    Jy = np.zeros((nx, ny, nz), dtype=np.float64)
    Jz = np.zeros((nx, ny, nz), dtype=np.float64)

    cell_volume = dx * dy * dz
    prefactor = charge / (cell_volume * dt) if (cell_volume > 0.0 and dt > 0.0) else 0.0

    for p in range(len(weights)):
        w = weights[p]
        if w == 0.0:
            continue

        # Normalised coordinates (old and new)
        xo = positions_old[p, 0] / dx
        yo = positions_old[p, 1] / dy
        zo = positions_old[p, 2] / dz
        xn = positions_new[p, 0] / dx
        yn = positions_new[p, 1] / dy
        zn = positions_new[p, 2] / dz

        # Cell index of old position (left node)
        ix0 = int(np.floor(xo))
        iy0 = int(np.floor(yo))
        iz0 = int(np.floor(zo))

        # Clamp to valid range (ghost-safe: allow one cell from each edge)
        ix0 = max(0, min(ix0, nx - 2))
        iy0 = max(0, min(iy0, ny - 2))
        iz0 = max(0, min(iz0, nz - 2))

        # Fractional offsets within old cell
        fx0 = xo - ix0
        fy0 = yo - iy0
        fz0 = zo - iz0
        fx0 = max(0.0, min(fx0, 1.0))
        fy0 = max(0.0, min(fy0, 1.0))
        fz0 = max(0.0, min(fz0, 1.0))

        # Cell index of new position (left node)
        ix1 = int(np.floor(xn))
        iy1 = int(np.floor(yn))
        iz1 = int(np.floor(zn))
        ix1 = max(0, min(ix1, nx - 2))
        iy1 = max(0, min(iy1, ny - 2))
        iz1 = max(0, min(iz1, nz - 2))

        # Fractional offsets within new cell
        fx1 = xn - ix1
        fy1 = yn - iy1
        fz1 = zn - iz1
        fx1 = max(0.0, min(fx1, 1.0))
        fy1 = max(0.0, min(fy1, 1.0))
        fz1 = max(0.0, min(fz1, 1.0))

        # CIC shape functions on a 2-node stencil relative to ix0/iy0/iz0
        # S0[0] = 1-fx0, S0[1] = fx0 (old); S1 shifted to same reference
        # dS = S1 - S0 for each node

        # x-direction (2 nodes: ix0, ix0+1)
        S0x_0 = 1.0 - fx0
        S0x_1 = fx0
        # Map new cell to same 2-node stencil: S1 offset by (ix1 - ix0)
        offset_x = ix1 - ix0
        # New shape at global nodes ix0 and ix0+1
        # S1 at node ix0  : if offset_x==0 -> 1-fx1; if offset_x==-1 -> 0; if offset_x==1 -> fx1 mapped
        if offset_x == 0:
            S1x_0 = 1.0 - fx1
            S1x_1 = fx1
        elif offset_x == 1:
            S1x_0 = 0.0
            S1x_1 = 1.0 - fx1
        elif offset_x == -1:
            S1x_0 = fx1
            S1x_1 = 0.0
        else:
            S1x_0 = 0.0
            S1x_1 = 0.0
        dSx_0 = S1x_0 - S0x_0
        dSx_1 = S1x_1 - S0x_1

        # y-direction
        S0y_0 = 1.0 - fy0
        S0y_1 = fy0
        offset_y = iy1 - iy0
        if offset_y == 0:
            S1y_0 = 1.0 - fy1
            S1y_1 = fy1
        elif offset_y == 1:
            S1y_0 = 0.0
            S1y_1 = 1.0 - fy1
        elif offset_y == -1:
            S1y_0 = fy1
            S1y_1 = 0.0
        else:
            S1y_0 = 0.0
            S1y_1 = 0.0
        dSy_0 = S1y_0 - S0y_0
        dSy_1 = S1y_1 - S0y_1

        # z-direction
        S0z_0 = 1.0 - fz0
        S0z_1 = fz0
        offset_z = iz1 - iz0
        if offset_z == 0:
            S1z_0 = 1.0 - fz1
            S1z_1 = fz1
        elif offset_z == 1:
            S1z_0 = 0.0
            S1z_1 = 1.0 - fz1
        elif offset_z == -1:
            S1z_0 = fz1
            S1z_1 = 0.0
        else:
            S1z_0 = 0.0
            S1z_1 = 0.0
        dSz_0 = S1z_0 - S0z_0
        dSz_1 = S1z_1 - S0z_1

        # Esirkepov W-factors (2x2x2 stencil, local indices 0/1 per axis)
        # Wx(i,j,k) = dSx[i] * (S0y[j] + 0.5*dSy[j]) * (S0z[k] + 0.5*dSz[k])
        dSx = (dSx_0, dSx_1)
        S0y = (S0y_0, S0y_1)
        dSy = (dSy_0, dSy_1)
        S0z = (S0z_0, S0z_1)
        dSz = (dSz_0, dSz_1)
        dSy_arr = (dSy_0, dSy_1)
        dSz_arr = (dSz_0, dSz_1)

        # Jx: running accumulation along x for each (j,k)
        for lj in range(2):
            for lk in range(2):
                jj = iy0 + lj
                kk = iz0 + lk
                if jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                    continue
                wy_jk = S0y[lj] + 0.5 * dSy_arr[lj]
                wz_jk = S0z[lk] + 0.5 * dSz_arr[lk]
                Wx_running = 0.0
                for li in range(2):
                    ii = ix0 + li
                    if ii < 0 or ii >= nx:
                        continue
                    Wx_running += dSx[li] * wy_jk * wz_jk
                    Jx[ii, jj, kk] += w * prefactor * Wx_running

        # Jy: running accumulation along y for each (i,k)
        dSx_arr = (dSx_0, dSx_1)
        S0x = (S0x_0, S0x_1)
        for li in range(2):
            for lk in range(2):
                ii = ix0 + li
                kk = iz0 + lk
                if ii < 0 or ii >= nx or kk < 0 or kk >= nz:
                    continue
                wx_ik = S0x[li] + 0.5 * dSx_arr[li]
                wz_ik = S0z[lk] + 0.5 * dSz_arr[lk]
                Wy_running = 0.0
                for lj in range(2):
                    jj = iy0 + lj
                    if jj < 0 or jj >= ny:
                        continue
                    Wy_running += dSy[lj] * wx_ik * wz_ik
                    Jy[ii, jj, kk] += w * prefactor * Wy_running

        # Jz: running accumulation along z for each (i,j)
        for li in range(2):
            for lj in range(2):
                ii = ix0 + li
                jj = iy0 + lj
                if ii < 0 or ii >= nx or jj < 0 or jj >= ny:
                    continue
                wx_ij = S0x[li] + 0.5 * dSx_arr[li]
                wy_ij = S0y[lj] + 0.5 * dSy_arr[lj]
                Wz_running = 0.0
                for lk in range(2):
                    kk = iz0 + lk
                    if kk < 0 or kk >= nz:
                        continue
                    Wz_running += dSz[lk] * wx_ij * wy_ij
                    Jz[ii, jj, kk] += w * prefactor * Wz_running

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
# Public API -- thin wrappers around the Numba kernels
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


def deposit_current_esirkepov(
    positions_old: np.ndarray,
    positions_new: np.ndarray,
    weights: np.ndarray,
    charge: float,
    grid_shape: tuple[int, int, int],
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge-conserving current deposition via Esirkepov (2001) algorithm.

    Parameters
    ----------
    positions_old : ndarray, shape (N, 3)
        Particle positions at previous timestep [m].
    positions_new : ndarray, shape (N, 3)
        Particle positions at current timestep [m].
    weights : ndarray, shape (N,)
    charge : float
    grid_shape : tuple of 3 ints
    dx, dy, dz : float
    dt : float

    Returns
    -------
    Jx, Jy, Jz : ndarray, each shape grid_shape
    """
    return _deposit_current_esirkepov_kernel(
        np.ascontiguousarray(positions_old, dtype=np.float64),
        np.ascontiguousarray(positions_new, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        float(charge),
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        float(dx),
        float(dy),
        float(dz),
        float(dt),
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
    positions_old: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.positions_old is None:
            object.__setattr__(self, "positions_old", self.positions.copy())

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
        use_esirkepov: bool = True,
    ) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.use_esirkepov = use_esirkepov
        self.species: list[ParticleSpecies] = []

        # Domain extents (particle positions live in [0, L])
        self._Lx = grid_shape[0] * dx
        self._Ly = grid_shape[1] * dy
        self._Lz = grid_shape[2] * dz

        # Coulomb collision parameters (set via enable_collisions())
        self._collision_enabled = False
        self._n_background = 1e25  # background density [m^-3]
        self._T_background_eV = 100.0  # background temperature [eV]

    def enable_collisions(
        self, n_background: float = 1e25, T_background_eV: float = 100.0,
    ) -> None:
        """Enable Coulomb collisions with a background plasma.

        Uses the Takizuka-Abe (1977) binary scattering model.

        Args:
            n_background: Background plasma density [m^-3].
            T_background_eV: Background electron temperature [eV].
        """
        self._collision_enabled = True
        self._n_background = n_background
        self._T_background_eV = T_background_eV

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

        .. note::
            **Known limitation — no Coulomb collision operator.**
            The Boris pusher is collisionless.  For DPF beam-target scenarios,
            beam slowing-down in the dense target plasma requires a collision
            operator (Fokker-Planck or binary Monte Carlo Coulomb scattering).
            Without it, beam ions pass through the target without energy loss,
            underestimating thermalisation and over-estimating beam-target
            neutron yields.  This is planned for a future phase.
            The PIC module is currently gated by ``config.kinetic.enabled``
            (default ``False``) and is not called from ``engine.py``, so
            this has no impact on production simulations.

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
            E_at_p = interpolate_field_to_particles(
                E, sp.positions, self.dx, self.dy, self.dz
            )
            B_at_p = interpolate_field_to_particles(
                B, sp.positions, self.dx, self.dy, self.dz
            )

            # Snapshot positions before push (needed for Esirkepov deposition)
            sp.positions_old = sp.positions.copy()

            # Boris push
            new_pos, new_vel = boris_push(
                sp.positions, sp.velocities, E_at_p, B_at_p,
                sp.charge, sp.mass, dt,
            )

            # Reflecting boundary conditions at domain edges
            new_pos, new_vel = self._apply_reflecting_bc(new_pos, new_vel)

            # Coulomb collision operator (Takizuka-Abe 1977 binary scattering)
            if self._collision_enabled and sp.n_particles() > 1:
                new_vel = _coulomb_scatter(
                    new_vel, sp.charge, sp.mass,
                    self._n_background, self._T_background_eV,
                    dt,
                )

            sp.positions = new_pos
            sp.velocities = new_vel

    def _apply_reflecting_bc(
        self, positions: np.ndarray, velocities: np.ndarray,
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
            # Esirkepov requires a prior push; fall back to CIC when particles
            # haven't moved (positions_old == positions, i.e. first deposit).
            _esirkepov_usable = (
                self.use_esirkepov
                and not np.array_equal(sp.positions_old, sp.positions)
            )
            if _esirkepov_usable:
                jx, jy, jz = deposit_current_esirkepov(
                    sp.positions_old, sp.positions, sp.weights, sp.charge,
                    self.grid_shape, self.dx, self.dy, self.dz, self.dt,
                )
            else:
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
        weight_total: float = 1e16,
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
        weight_total : float
            Total number of physical ions represented by all macro-particles.
            Macro-particle weight = weight_total / n_beam.
            Default 1e16 ~ 1 mC at 100 keV over 10 ns.
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
            rng = np.random.default_rng()
            for i in range(n_beam):
                # Random rotation angles
                theta = spread * np.sqrt(rng.random())
                phi = 2.0 * np.pi * rng.random()

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

        # Physical weight: each macro-particle represents weight_total/n_beam physical ions.
        # Using weight=1.0 (bare count) would underestimate current by ~17 orders of magnitude.
        macro_weight = weight_total / max(n_beam, 1)
        beam_weights = np.full(n_beam, macro_weight, dtype=np.float64)

        # Append to existing species arrays
        sp.positions = np.concatenate([sp.positions, beam_pos], axis=0)
        sp.velocities = np.concatenate([sp.velocities, beam_vel], axis=0)
        sp.weights = np.concatenate([sp.weights, beam_weights], axis=0)
        # Keep positions_old in sync with positions so that a deposit() call
        # immediately after injection (before any push) falls back to CIC.
        sp.positions_old = sp.positions.copy()


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
