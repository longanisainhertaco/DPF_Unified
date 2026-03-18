"""Sub-grid turbulence model and reconnection diagnostics.

Provides:
1. Smagorinsky-type sub-grid stress for MHD (anomalous viscosity/conductivity)
2. Anomalous thermal conductivity from turbulent scattering
3. Sweet-Parker reconnection rate diagnostic
4. Plasmoid instability criterion (Loureiro et al. 2007)
5. Turbulent energy spectrum from MHD output (FFT-based)

References:
    Smagorinsky, Mon. Weather Rev. 91:99 (1963) — SGS model
    Sweet, IAU Symp. 6:123 (1958) — reconnection rate
    Parker, J. Geophys. Res. 62:509 (1957) — reconnection rate
    Loureiro et al., Phys. Plasmas 14:100703 (2007) — plasmoid instability
    Biskamp, Nonlinear Magnetohydrodynamics (1993) — MHD turbulence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from dpf.constants import k_B, m_e, mu_0


@dataclass
class ReconnectionDiag:
    """Sweet-Parker reconnection diagnostic results."""

    S_lundquist: float  # Lundquist number
    reconnection_rate: float  # v_in / v_A (dimensionless)
    delta_sp: float  # Sweet-Parker current sheet thickness [m]
    L_sheet: float  # Sheet length [m]
    plasmoid_unstable: bool  # True if S > S_crit (plasmoid regime)
    S_crit: float  # Critical S for plasmoid instability
    n_plasmoids_est: int  # Estimated number of plasmoids
    regime: str  # "sweet_parker", "plasmoid", or "collisionless"


@dataclass
class SpectrumResult:
    """Turbulent energy spectrum result."""

    k: np.ndarray  # Wavenumber array [1/m]
    E_k: np.ndarray  # Energy spectrum E(k) [J*m]
    E_mag_k: np.ndarray  # Magnetic energy spectrum
    E_kin_k: np.ndarray  # Kinetic energy spectrum
    spectral_index: float  # Best-fit power law index
    inertial_range: tuple[float, float]  # (k_min, k_max) of inertial range


@njit(cache=True)
def smagorinsky_viscosity(
    velocity: np.ndarray,
    dx: float,
    C_s: float = 0.1,
) -> np.ndarray:
    """Compute Smagorinsky sub-grid scale viscosity.

    nu_sgs = (C_s * dx)^2 * |S|

    where |S| = sqrt(2 * S_ij * S_ij) is the magnitude of the
    strain rate tensor.

    Args:
        velocity: Velocity field (3, nx, ny, nz) [m/s].
        dx: Grid spacing [m].
        C_s: Smagorinsky constant (0.1-0.2, default 0.1).

    Returns:
        SGS viscosity field (nx, ny, nz) [m^2/s].
    """
    nx, ny, nz = velocity.shape[1], velocity.shape[2], velocity.shape[3]
    nu_sgs = np.zeros((nx, ny, nz))
    delta = C_s * dx

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                # Strain rate components (central differences)
                dudx = (velocity[0, i + 1, j, k] - velocity[0, i - 1, j, k]) / (2 * dx)
                dvdy = (velocity[1, i, j + 1, k] - velocity[1, i, j - 1, k]) / (2 * dx)
                dwdz = (velocity[2, i, j, k + 1] - velocity[2, i, j, k - 1]) / (2 * dx)
                dudy = (velocity[0, i, j + 1, k] - velocity[0, i, j - 1, k]) / (2 * dx)
                dudz = (velocity[0, i, j, k + 1] - velocity[0, i, j, k - 1]) / (2 * dx)
                dvdx = (velocity[1, i + 1, j, k] - velocity[1, i - 1, j, k]) / (2 * dx)
                dvdz = (velocity[1, i, j, k + 1] - velocity[1, i, j, k - 1]) / (2 * dx)
                dwdx = (velocity[2, i + 1, j, k] - velocity[2, i - 1, j, k]) / (2 * dx)
                dwdy = (velocity[2, i, j + 1, k] - velocity[2, i, j - 1, k]) / (2 * dx)

                # |S|^2 = 2*(Sxx^2 + Syy^2 + Szz^2) + (Sxy+Syx)^2 + ...
                S2 = (dudx**2 + dvdy**2 + dwdz**2
                      + 0.5 * (dudy + dvdx)**2
                      + 0.5 * (dudz + dwdx)**2
                      + 0.5 * (dvdz + dwdy)**2)
                nu_sgs[i, j, k] = delta**2 * np.sqrt(2.0 * S2)

    return nu_sgs


def anomalous_thermal_conductivity(
    ne: np.ndarray,
    Te: np.ndarray,
    eta_anom: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    """Compute anomalous thermal conductivity from turbulent scattering.

    When anomalous resistivity is active, the effective mean free path
    is reduced, enhancing cross-field transport. The anomalous thermal
    conductivity is:

        kappa_anom = n_e * k_B * v_te * lambda_eff

    where lambda_eff ~ v_te / nu_eff and nu_eff = e^2 * n_e * eta_anom / m_e.

    Simplifies to: kappa_anom = k_B * Te / (m_e * nu_eff)
                              = k_B * Te / (e^2 * n_e * eta_anom)

    Args:
        ne: Electron density [m^-3].
        Te: Electron temperature [K].
        eta_anom: Anomalous resistivity [Ohm*m].
        Z_eff: Effective ion charge.

    Returns:
        Anomalous thermal conductivity [W/(m*K)].
    """
    from dpf.constants import e as e_charge
    # nu_eff = e^2 * n_e * eta_anom / m_e
    nu_eff = e_charge**2 * np.maximum(ne, 1e10) * np.maximum(eta_anom, 1e-20) / m_e
    # kappa_anom = n_e * k_B^2 * Te / (m_e * nu_eff)
    kappa = ne * k_B**2 * np.maximum(Te, 1.0) / (m_e * nu_eff)
    return kappa


def sweet_parker_diagnostic(
    B_field: np.ndarray,
    rho: np.ndarray,
    Te: np.ndarray,
    ne: np.ndarray,
    dx: float,
    L_system: float,
    Z_eff: float = 1.0,
    ion_mass: float = 3.34e-27,
) -> ReconnectionDiag:
    """Compute Sweet-Parker reconnection rate and plasmoid criterion.

    Sweet-Parker reconnection rate: v_in / v_A = S^{-1/2}
    where S = L * v_A / eta is the Lundquist number.

    Plasmoid instability (Loureiro 2007): at S > S_crit ~ 10^4,
    the Sweet-Parker sheet fragments into a chain of plasmoids.
    Number of plasmoids ~ S^{3/8}.

    Args:
        B_field: Magnetic field (3, nx, ny, nz) [T].
        rho: Mass density (nx, ny, nz) [kg/m^3].
        Te: Electron temperature (nx, ny, nz) [K].
        ne: Electron density (nx, ny, nz) [m^-3].
        dx: Grid spacing [m].
        L_system: System length scale [m].
        Z_eff: Effective charge.
        ion_mass: Ion mass [kg].

    Returns:
        ReconnectionDiag with all diagnostics.
    """
    from dpf.collision.spitzer import spitzer_resistivity

    # Volume-averaged quantities
    B_mag = np.sqrt(np.sum(B_field**2, axis=0))
    B_avg = float(np.mean(B_mag[B_mag > 0])) if np.any(B_mag > 0) else 0.0
    rho_avg = float(np.mean(rho))
    Te_avg = float(np.mean(Te))
    ne_avg = float(np.mean(ne))

    # Alfven speed
    v_A = B_avg / np.sqrt(mu_0 * max(rho_avg, 1e-20))

    # Spitzer resistivity (scalar via array API)
    eta = float(spitzer_resistivity(
        np.array([max(ne_avg, 1e10)]),
        np.array([max(Te_avg, 1.0)]),
        Z=Z_eff,
    )[0])

    # Lundquist number S = L * v_A * mu_0 / eta
    S = L_system * v_A * mu_0 / max(eta, 1e-20)

    # Sweet-Parker rate
    rate = 1.0 / max(np.sqrt(S), 1.0) if S > 0 else 0.0

    # Sheet thickness delta_SP = L / sqrt(S)
    delta_sp = L_system / max(np.sqrt(S), 1.0)

    # Plasmoid instability criterion (Loureiro 2007)
    S_crit = 1e4
    plasmoid_unstable = S_crit < S
    n_plasmoids = int(S**(3.0 / 8.0)) if plasmoid_unstable else 0

    # Regime classification
    if S < 100:
        regime = "collisionless"
    elif S_crit > S:
        regime = "sweet_parker"
    else:
        regime = "plasmoid"

    return ReconnectionDiag(
        S_lundquist=float(S),
        reconnection_rate=float(rate),
        delta_sp=float(delta_sp),
        L_sheet=L_system,
        plasmoid_unstable=plasmoid_unstable,
        S_crit=S_crit,
        n_plasmoids_est=n_plasmoids,
        regime=regime,
    )


def compute_energy_spectrum(
    velocity: np.ndarray,
    B_field: np.ndarray,
    rho: np.ndarray,
    dx: float,
) -> SpectrumResult:
    """Compute turbulent energy spectrum from MHD state via FFT.

    Computes the shell-averaged kinetic and magnetic energy spectra:
        E_kin(k) = 0.5 * rho * |v_hat(k)|^2
        E_mag(k) = |B_hat(k)|^2 / (2 * mu_0)

    Bins into spherical shells in k-space.

    MHD turbulence: expect k^{-5/3} (Kolmogorov) or k^{-3/2}
    (Iroshnikov-Kraichnan) in the inertial range.

    Args:
        velocity: Velocity field (3, nx, ny, nz) [m/s].
        B_field: Magnetic field (3, nx, ny, nz) [T].
        rho: Mass density (nx, ny, nz) [kg/m^3].
        dx: Grid spacing [m].

    Returns:
        SpectrumResult with wavenumber array and spectra.
    """
    nx, ny, nz = velocity.shape[1], velocity.shape[2], velocity.shape[3]
    rho_avg = float(np.mean(rho))

    # Kinetic energy density in Fourier space
    E_kin_total = np.zeros((nx, ny, nz))
    E_mag_total = np.zeros((nx, ny, nz))

    for comp in range(3):
        v_hat = np.fft.fftn(velocity[comp])
        E_kin_total += 0.5 * rho_avg * np.abs(v_hat)**2

        B_hat = np.fft.fftn(B_field[comp])
        E_mag_total += np.abs(B_hat)**2 / (2.0 * mu_0)

    # Wavenumber grid
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Shell-average: bin by |k|
    k_max = np.max(K)
    n_bins = min(nx // 2, 64)
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_kin_k = np.zeros(n_bins)
    E_mag_k = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_bins[i] <= K) & (k_bins[i + 1] > K)
        if np.any(mask):
            dk = k_bins[i + 1] - k_bins[i]
            E_kin_k[i] = float(np.sum(E_kin_total[mask])) / max(dk, 1e-30)
            E_mag_k[i] = float(np.sum(E_mag_total[mask])) / max(dk, 1e-30)

    E_total_k = E_kin_k + E_mag_k

    # Fit power law in inertial range (middle third of k)
    i_start = max(2, n_bins // 4)
    i_end = min(n_bins - 1, 3 * n_bins // 4)
    mask_fit = (E_total_k[i_start:i_end] > 0) & (k_centers[i_start:i_end] > 0)
    if np.sum(mask_fit) >= 3:
        log_k = np.log(k_centers[i_start:i_end][mask_fit])
        log_E = np.log(E_total_k[i_start:i_end][mask_fit])
        coeffs = np.polyfit(log_k, log_E, 1)
        spectral_index = float(coeffs[0])
    else:
        spectral_index = 0.0

    inertial_range = (float(k_centers[i_start]), float(k_centers[min(i_end, n_bins - 1)]))

    return SpectrumResult(
        k=k_centers,
        E_k=E_total_k,
        E_mag_k=E_mag_k,
        E_kin_k=E_kin_k,
        spectral_index=spectral_index,
        inertial_range=inertial_range,
    )
