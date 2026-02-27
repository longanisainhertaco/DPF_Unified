"""Plasma regime diagnostics: collisionality, magnetic Reynolds number, Debye length.

Provides dimensionless parameters that determine the regime of validity of the
MHD fluid approximation used by DPF-Unified.  When these parameters exit the
MHD-valid range, the simulation results become physically unreliable and
kinetic effects (PIC) dominate.

Key diagnostics:
    plasma_parameter_ND: Ion plasma parameter (ND < 1 → collisional → MHD valid).
    magnetic_reynolds_number: Rm >> 1 → frozen-in flux → ideal MHD valid.
    debye_length: Spatial scale below which charge separation matters.
    ion_skin_depth: Scale below which Hall physics matters.

References:
    Kindi et al., Phys. Plasmas 33, 022112 (2026) — ND collisionality criterion.
    Auluck, Phys. Plasmas 31, 010704 (2024) — Rm and dynamo in DPF.
    NRL Plasma Formulary (2019).
"""

from __future__ import annotations

import numpy as np

from dpf.constants import e, epsilon_0, k_B, m_d, m_e, mu_0, pi


def plasma_parameter_ND(
    ne: np.ndarray,
    Ti: np.ndarray,
) -> np.ndarray:
    """Ion plasma parameter ND = (4pi/3) * lambda_Di^3 * n_i.

    ND determines the collisionality regime:
        ND < 1  → collisional (strongly coupled), MHD fluid valid
        ND > 1  → collisionless (weakly coupled), kinetic effects dominate
        ND ~ 1  → transition regime

    For DPF fill gas (D2, 300 K, 10^23 m^-3): ND ~ 0.01 (collisional).
    For DPF pinch (D2, 1 keV, 10^25 m^-3): ND ~ 10^4 (collisionless).

    Args:
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].

    Returns:
        Plasma parameter ND (dimensionless), same shape as ne.

    References:
        Kindi et al., Phys. Plasmas 33, 022112 (2026), Eq. 1.
    """
    # Ion Debye length: lambda_Di = sqrt(epsilon_0 * k_B * Ti / (n_i * e^2))
    # For quasi-neutral deuterium: n_i = ne
    lambda_Di = np.sqrt(epsilon_0 * k_B * Ti / (ne * e**2 + 1e-300))
    return (4.0 * pi / 3.0) * lambda_Di**3 * ne


def magnetic_reynolds_number(
    ne: np.ndarray,
    Te: np.ndarray,
    velocity: np.ndarray,
    L_scale: float,
    lnL: float = 10.0,
) -> np.ndarray:
    """Magnetic Reynolds number Rm = mu_0 * v * L / eta.

    Rm >> 1: magnetic field is frozen into the plasma (ideal MHD valid).
    Rm ~ 1:  resistive diffusion competes with advection.
    Rm << 1: field diffuses freely through plasma.

    For DPF radial implosion (v ~ 10^5 m/s, L ~ 1 cm, Te ~ 10 eV):
    Rm ~ 10-100 (frozen-in, but resistive effects non-negligible).

    The current fraction parameter f_c in the Lee model may absorb
    dynamo effects associated with Rm-dependent poloidal field generation
    (Auluck, Phys. Plasmas 31, 010704, 2024).

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        velocity: Flow velocity magnitude [m/s].
        L_scale: Characteristic length scale [m].
        lnL: Coulomb logarithm (default 10).

    Returns:
        Magnetic Reynolds number Rm (dimensionless), same shape as ne.

    References:
        Auluck, Phys. Plasmas 31, 010704 (2024).
        NRL Plasma Formulary (2019), p. 34.
    """
    # Spitzer resistivity: eta = m_e * nu_ei / (ne * e^2)
    # nu_ei = (4*sqrt(2*pi) * ne * e^4 * lnL) / (3 * (4*pi*eps0)^2 * sqrt(m_e) * (kT)^1.5)
    nu_ei = (
        4.0 * np.sqrt(2.0 * pi) * ne * e**4 * lnL
        / (3.0 * (4.0 * pi * epsilon_0) ** 2 * np.sqrt(m_e) * (k_B * Te + 1e-300) ** 1.5)
    )
    eta = m_e * nu_ei / (ne * e**2 + 1e-300)
    return mu_0 * velocity * L_scale / (eta + 1e-300)


def debye_length(
    ne: np.ndarray,
    Te: np.ndarray,
) -> np.ndarray:
    """Electron Debye length [m].

    lambda_De = sqrt(epsilon_0 * k_B * Te / (ne * e^2))

    Spatial scale below which charge separation is significant.
    MHD requires grid spacing >> lambda_De.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].

    Returns:
        Debye length [m], same shape as ne.
    """
    return np.sqrt(epsilon_0 * k_B * Te / (ne * e**2 + 1e-300))


def ion_skin_depth(
    ne: np.ndarray,
) -> np.ndarray:
    """Ion skin depth (ion inertial length) [m].

    d_i = c / omega_pi = sqrt(m_i / (mu_0 * ne * e^2))

    Scale below which Hall-MHD effects become important.
    For DPF pinch at ne ~ 10^25 m^-3: d_i ~ 0.1 mm.

    Args:
        ne: Electron number density [m^-3] (quasi-neutral: n_i = ne).

    Returns:
        Ion skin depth [m], same shape as ne.

    References:
        Vasconez et al., Phys. Plasmas 33, 012304 (2026) — Hall-MHD scales.
    """
    return np.sqrt(m_d / (mu_0 * ne * e**2 + 1e-300))


def regime_validity(
    ne: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    velocity: np.ndarray,
    dx: float,
) -> dict[str, np.ndarray | float]:
    """Comprehensive regime validity assessment for the MHD approximation.

    Computes all dimensionless parameters and flags cells where the MHD
    fluid model is outside its regime of validity.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].
        velocity: Flow velocity magnitude [m/s].
        dx: Grid spacing [m].

    Returns:
        Dictionary with:
            ND: Ion plasma parameter (ND < 1 → MHD valid).
            Rm: Magnetic Reynolds number (at grid scale).
            lambda_De: Electron Debye length [m].
            d_i: Ion skin depth [m].
            mhd_valid: Boolean array (True where MHD approximation holds).
            fraction_valid: Fraction of cells where MHD is valid.
    """
    ND = plasma_parameter_ND(ne, Ti)
    Rm = magnetic_reynolds_number(ne, Te, velocity, dx)
    lam_De = debye_length(ne, Te)
    d_i = ion_skin_depth(ne)

    # MHD validity: ND < 1 (collisional) AND dx >> lambda_De
    mhd_valid = (ND < 1.0) & (dx > 10.0 * lam_De)

    return {
        "ND": ND,
        "Rm": Rm,
        "lambda_De": lam_De,
        "d_i": d_i,
        "mhd_valid": mhd_valid,
        "fraction_valid": float(np.mean(mhd_valid)),
    }
