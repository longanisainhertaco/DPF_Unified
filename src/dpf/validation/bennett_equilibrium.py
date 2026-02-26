"""Bennett Z-pinch equilibrium profiles and verification.

Provides analytical profiles for the Bennett pinch — the foundational
quasi-steady-state equilibrium for a Z-pinch carrying axial current I.
Used as a first-principles verification benchmark for MHD solvers.

The Bennett equilibrium satisfies radial force balance::

    dp/dr + (B_theta / (mu_0 * r)) * d(r * B_theta)/dr = 0

which is equivalent to dp/dr = -J_z * B_theta (J x B pinch force).

Profiles:
    n(r)       = n_0 / (1 + r^2/a^2)^2         (number density)
    B_theta(r) = (mu_0 * I) / (2*pi) * r / (r^2 + a^2)  (azimuthal field)
    p(r)       = n(r) * k_B * (T_e + T_i)       (pressure)

Bennett relation (global constraint):
    mu_0 * I^2 / (8*pi) = N * k_B * (T_e + T_i)
    where N = pi * n_0 * a^2  is the line density [m^-1].

References:
    Bennett W.H., Phys. Rev. 45, 890 (1934).
    Haines M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
    Velikovich A.L. et al., Phys. Plasmas 19, 012707 (2012).
"""

from __future__ import annotations

import numpy as np

from dpf.constants import k_B, mu_0, pi


def bennett_density(r: np.ndarray, n_0: float, a: float) -> np.ndarray:
    """Bennett number density profile.

    Args:
        r: Radial positions [m], shape arbitrary.
        n_0: On-axis peak number density [m^-3].
        a: Bennett radius (characteristic scale length) [m].

    Returns:
        Number density n(r) [m^-3], same shape as *r*.
    """
    return n_0 / (1.0 + (r / a) ** 2) ** 2


def bennett_btheta(r: np.ndarray, I_total: float, a: float) -> np.ndarray:
    """Bennett azimuthal magnetic field B_theta(r).

    Args:
        r: Radial positions [m].
        I_total: Total axial current [A].
        a: Bennett radius [m].

    Returns:
        B_theta(r) [T], same shape as *r*.
    """
    return (mu_0 * I_total / (2.0 * pi)) * r / (r**2 + a**2)


def bennett_pressure(
    r: np.ndarray, n_0: float, a: float, Te: float, Ti: float
) -> np.ndarray:
    """Bennett pressure profile from ideal gas law.

    Args:
        r: Radial positions [m].
        n_0: On-axis number density [m^-3].
        a: Bennett radius [m].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].

    Returns:
        Pressure p(r) [Pa], same shape as *r*.
    """
    return bennett_density(r, n_0, a) * k_B * (Te + Ti)


def bennett_current_density(
    r: np.ndarray, I_total: float, a: float
) -> np.ndarray:
    """Axial current density J_z(r) from Ampere's law.

    J_z = (1/mu_0) * (1/r) * d(r * B_theta)/dr
        = I * a^2 / (pi * (r^2 + a^2)^2)

    Args:
        r: Radial positions [m].
        I_total: Total axial current [A].
        a: Bennett radius [m].

    Returns:
        J_z(r) [A/m^2], same shape as *r*.
    """
    return I_total * a**2 / (pi * (r**2 + a**2) ** 2)


def bennett_line_density(n_0: float, a: float) -> float:
    """Line density N = integral(n * 2*pi*r dr, 0, inf) = pi * n_0 * a^2.

    Args:
        n_0: On-axis number density [m^-3].
        a: Bennett radius [m].

    Returns:
        Line density N [m^-1].
    """
    return pi * n_0 * a**2


def bennett_current_from_temperature(
    n_0: float, a: float, Te: float, Ti: float
) -> float:
    """Compute Bennett current from the Bennett relation.

    mu_0 * I^2 / (8*pi) = N * k_B * (T_e + T_i)
    => I = sqrt(8*pi * N * k_B * (T_e + T_i) / mu_0)

    Args:
        n_0: On-axis number density [m^-3].
        a: Bennett radius [m].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].

    Returns:
        Total current I [A].
    """
    N = bennett_line_density(n_0, a)
    return np.sqrt(8.0 * pi * N * k_B * (Te + Ti) / mu_0)


def verify_force_balance(
    r: np.ndarray,
    n_0: float,
    a: float,
    I_total: float,
    Te: float,
    Ti: float,
) -> tuple[np.ndarray, float]:
    """Check dp/dr + J_z * B_theta = 0 (radial force balance).

    Computes the residual of the radial momentum equation at each
    radial position.  For the exact Bennett profile, this should be
    zero to machine precision.

    Args:
        r: Radial positions [m] (must be > 0; avoid r=0 singularity).
        n_0: On-axis number density [m^-3].
        a: Bennett radius [m].
        I_total: Total axial current [A].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].

    Returns:
        (residual, max_relative_error):
            residual: dp/dr + J_z * B_theta at each r [Pa/m].
            max_relative_error: max(|residual|) / max(|dp/dr|).
    """
    Jz = bennett_current_density(r, I_total, a)
    Bt = bennett_btheta(r, I_total, a)

    # Analytical dp/dr
    dp_dr = -4.0 * n_0 * k_B * (Te + Ti) * a**4 * r / (r**2 + a**2) ** 3

    # Force balance residual: should be zero
    residual = dp_dr + Jz * Bt

    max_dp = np.max(np.abs(dp_dr))
    if max_dp > 0:
        max_rel_err = np.max(np.abs(residual)) / max_dp
    else:
        max_rel_err = 0.0

    return residual, max_rel_err


def create_bennett_state(
    nr: int,
    nz: int,
    r_max: float,
    dz: float,
    n_0: float,
    a: float,
    Te: float,
    Ti: float,
    m_ion: float,
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    """Create a 2D cylindrical MHD state dict initialized with Bennett profiles.

    Generates state arrays compatible with CylindricalMHDSolver (nr, 1, nz).
    The Bennett profiles are uniform along z (infinite cylinder approximation).

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.
        r_max: Outer radial boundary [m].
        dz: Axial cell size [m].
        n_0: On-axis number density [m^-3].
        a: Bennett radius [m].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].
        m_ion: Ion mass [kg].

    Returns:
        (state, I_total, r_centers):
            state: Dict with rho, velocity, pressure, B, Te, Ti, psi arrays.
            I_total: Bennett current [A].
            r_centers: Radial cell centers [m], shape (nr,).
    """
    dr = r_max / nr
    r_centers = np.linspace(0.5 * dr, r_max - 0.5 * dr, nr)

    I_total = bennett_current_from_temperature(n_0, a, Te, Ti)
    n = bennett_density(r_centers, n_0, a)
    Bt = bennett_btheta(r_centers, I_total, a)
    p = bennett_pressure(r_centers, n_0, a, Te, Ti)
    rho = n * m_ion

    # Reshape for cylindrical solver: (nr, 1, nz)
    rho_3d = np.broadcast_to(rho[:, None, None], (nr, 1, nz)).copy()
    p_3d = np.broadcast_to(p[:, None, None], (nr, 1, nz)).copy()

    velocity = np.zeros((3, nr, 1, nz))
    B = np.zeros((3, nr, 1, nz))
    B[1, :, 0, :] = Bt[:, None]  # B_theta

    Te_3d = np.full((nr, 1, nz), Te)
    Ti_3d = np.full((nr, 1, nz), Ti)
    psi_3d = np.zeros((nr, 1, nz))

    state = {
        "rho": rho_3d,
        "velocity": velocity,
        "pressure": p_3d,
        "B": B,
        "Te": Te_3d,
        "Ti": Ti_3d,
        "psi": psi_3d,
    }
    return state, I_total, r_centers
