"""Synthetic Thomson scattering diagnostic for DPF simulations.

Computes the full Salpeter spectral density function S(k, omega) via the
Faddeeva function (scipy.special.wofz), valid across all scattering regimes
(non-collective alpha << 1, transition alpha ~ 1, collective alpha >> 1).

The scattered power per unit solid angle, wavelength, and path length is:

    dP/(dOmega * dlambda * dL) = ne * r_e^2 * S(k, omega)

where r_e is the classical electron radius and S(k, omega) is the spectral
density function (Salpeter form factor).

References:
    Sheffield et al., "Plasma Scattering of EM Radiation", 2nd ed. (2011)
    Salpeter, Phys. Rev. 120:1528 (1960)
    Decker et al., Plasma Sources Sci. Technol. 5:112 (1996) — DPF Thomson
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import wofz

from dpf.constants import c, e, epsilon_0, k_B, m_e, pi

# Physical constants
eV = 1.602176634e-19  # J per eV
r_e = e**2 / (4.0 * pi * epsilon_0 * m_e * c**2)  # classical electron radius [m]
sigma_T = 8.0 * pi * r_e**2 / 3.0  # Thomson cross-section [m^2]
M_DEUTERIUM = 3.344e-27  # deuterium ion mass [kg]


def spectral_density_salpeter(
    omega: np.ndarray,
    k: float,
    ne: float,
    Te_eV: float,
    Ti_eV: float | None = None,
    m_i: float = M_DEUTERIUM,
    Z_ion: int = 1,
) -> np.ndarray:
    """Full Salpeter spectral density function S(k, omega) via Faddeeva.

    Valid at ALL alpha values (non-collective, transition, collective).
    No regime-switching needed. Uses the plasma dispersion function
    Z(zeta) = i * sqrt(pi) * wofz(zeta).

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency shift from laser [rad/s], shape (M,).
    k : float
        Scattering wavevector magnitude [m^-1].
    ne : float
        Electron density [m^-3].
    Te_eV : float
        Electron temperature [eV].
    Ti_eV : float or None
        Ion temperature [eV]. Defaults to Te_eV.
    m_i : float
        Ion mass [kg]. Default: deuterium (3.344e-27 kg).
    Z_ion : int
        Ion charge state.

    Returns
    -------
    np.ndarray
        S(k, omega), shape (M,), units [s/rad].
    """
    if Ti_eV is None:
        Ti_eV = Te_eV

    Te = Te_eV * eV / k_B
    Ti = Ti_eV * eV / k_B
    v_th_e = np.sqrt(2.0 * k_B * Te / m_e)
    v_th_i = np.sqrt(2.0 * k_B * Ti / m_i)

    lambda_De = np.sqrt(epsilon_0 * k_B * Te / (ne * e**2))
    lambda_Di = np.sqrt(epsilon_0 * k_B * Ti / (ne * Z_ion * e**2))
    alpha_e = 1.0 / (k * lambda_De)
    alpha_i = 1.0 / (k * lambda_Di)

    zeta_e = omega / (k * v_th_e)
    zeta_i = omega / (k * v_th_i)

    Z_e = 1j * np.sqrt(np.pi) * wofz(zeta_e)
    Z_i = 1j * np.sqrt(np.pi) * wofz(zeta_i)

    chi_e = -alpha_e**2 * (1.0 + zeta_e * Z_e)
    chi_i = -alpha_i**2 * (1.0 + zeta_i * Z_i)
    epsilon_d = 1.0 + chi_e + chi_i

    f_e = np.exp(-(zeta_e**2)) / (v_th_e * np.sqrt(np.pi))
    f_i = np.exp(-(zeta_i**2)) / (v_th_i * np.sqrt(np.pi))

    S_e = (2.0 * np.pi / k) * np.abs(1.0 - chi_e / epsilon_d) ** 2 * f_e
    S_i = (2.0 * np.pi / k) * np.abs(chi_e / epsilon_d) ** 2 * f_i * Z_ion

    return np.real(S_e + S_i)


def thomson_spectrum(
    ne: np.ndarray,
    Te_eV: np.ndarray,
    v_bulk: np.ndarray,
    wavelength_grid: np.ndarray,
    Ti_eV: np.ndarray | None = None,
    scattering_angle: float = np.pi / 2,
    laser_wavelength: float = 1064e-9,
    m_i: float = M_DEUTERIUM,
    Z_ion: int = 1,
) -> np.ndarray:
    """Thomson scattering spectral power density at each spatial point.

    Computes the full Salpeter form factor using scipy.special.wofz,
    working seamlessly across all alpha regimes.

    Parameters
    ----------
    ne : np.ndarray
        Electron density [m^-3], shape (N,).
    Te_eV : np.ndarray
        Electron temperature [eV], shape (N,).
    v_bulk : np.ndarray
        Bulk velocity projected onto scattering k-hat [m/s], shape (N,).
    wavelength_grid : np.ndarray
        Scattered wavelengths [m], shape (M,).
    Ti_eV : np.ndarray or None
        Ion temperature [eV], shape (N,). Defaults to Te_eV.
    scattering_angle : float
        Scattering angle theta [rad].
    laser_wavelength : float
        Probe laser wavelength [m].
    m_i : float
        Ion mass [kg].
    Z_ion : int
        Ion charge state.

    Returns
    -------
    np.ndarray
        Spectral power density [W/m^3/sr/m], shape (N, M).
        Multiply by ne * r_e^2 is already included.
    """
    if Ti_eV is None:
        Ti_eV = Te_eV

    N = len(ne)
    M = len(wavelength_grid)
    k_scatter = (4.0 * pi / laser_wavelength) * np.sin(scattering_angle / 2.0)

    result = np.zeros((N, M))
    for i in range(N):
        # Doppler shift: omega includes bulk flow
        delta_lambda = wavelength_grid - laser_wavelength
        v_doppler = v_bulk[i]
        delta_D = (laser_wavelength / c) * v_doppler * 2.0 * np.sin(scattering_angle / 2.0)
        delta_lambda_shifted = delta_lambda - delta_D

        omega = 2.0 * pi * c * delta_lambda_shifted / (laser_wavelength**2)

        S = spectral_density_salpeter(
            omega, k_scatter, ne[i], Te_eV[i], Ti_eV[i], m_i, Z_ion
        )

        # Convert S(k, omega) [s/rad] to S(k, lambda) [1/m]
        # domega/dlambda = 2*pi*c / lambda_0^2
        domega_dlambda = 2.0 * pi * c / (laser_wavelength**2)
        S_lambda = S * domega_dlambda

        # Scattered power: ne * r_e^2 * S_lambda
        result[i, :] = ne[i] * r_e**2 * S_lambda

    return result


def thomson_line_integrated(
    ne_2d: np.ndarray,
    Te_2d: np.ndarray,
    vz_2d: np.ndarray,
    r_cell: np.ndarray,
    chord_positions_z: np.ndarray,
    wavelength_grid: np.ndarray,
    Ti_2d: np.ndarray | None = None,
    scattering_angle: float = np.pi / 2,
    laser_wavelength: float = 1064e-9,
    m_i: float = M_DEUTERIUM,
    Z_ion: int = 1,
    z_cell: np.ndarray | None = None,
) -> np.ndarray:
    """Line-integrated Thomson spectrum along radial chords.

    Uses Abel transform from interferometry.py for line integration
    through the axisymmetric plasma.

    Parameters
    ----------
    ne_2d : np.ndarray
        Electron density [m^-3], shape (nr, nz).
    Te_2d : np.ndarray
        Electron temperature [eV], shape (nr, nz).
    vz_2d : np.ndarray
        Axial velocity [m/s], shape (nr, nz).
    r_cell : np.ndarray
        Radial cell centers [m], shape (nr,).
    chord_positions_z : np.ndarray
        Axial positions of measurement chords [m], shape (Nc,).
    wavelength_grid : np.ndarray
        Scattered wavelengths [m], shape (M,).
    Ti_2d : np.ndarray or None
        Ion temperature [eV], shape (nr, nz). Defaults to Te_2d.
    scattering_angle : float
        Scattering angle theta [rad].
    laser_wavelength : float
        Probe laser wavelength [m].
    m_i : float
        Ion mass [kg].
    Z_ion : int
        Ion charge state.
    z_cell : np.ndarray or None
        Axial cell centers [m], shape (nz,). When provided, the chord's
        z-index is found by nearest-neighbor lookup against this grid
        (correct units handling). When None, falls back to a legacy
        index mapping (chord_positions_z[ic] interpreted as a fractional
        position in [0, 1)) and emits RuntimeWarning. New callers SHOULD
        pass z_cell. [INFERRED — units bug fix per audit-diagnostics
        2026-04-27; original code path was a units mismatch versus the
        docstring "[m]"].

    Returns
    -------
    np.ndarray
        Line-integrated spectral power [W/m^2/sr/m], shape (Nc, M).
    """
    import warnings

    from dpf.diagnostics.interferometry import abel_transform

    if Ti_2d is None:
        Ti_2d = Te_2d

    nr, nz = ne_2d.shape
    Nc = len(chord_positions_z)
    M = len(wavelength_grid)

    if z_cell is None and nz > 1:
        warnings.warn(
            "thomson_line_integrated called without z_cell; "
            "chord_positions_z is being interpreted as a fractional "
            "index in [0, 1), NOT in metres as the docstring states. "
            "Pass z_cell to use the metric interpretation.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = np.zeros((Nc, M))
    for ic in range(Nc):
        z_target = chord_positions_z[ic]
        if z_cell is not None and nz > 1:
            # Correct: nearest neighbor in metres.
            iz = int(np.argmin(np.abs(z_cell - z_target)))
            iz = max(0, min(iz, nz - 1))
        elif nz > 1:
            # Legacy fallback: treat z_target as a fraction in [0, 1).
            iz = min(int(z_target * nz), nz - 1)
        else:
            iz = 0

        # Extract radial profiles at this z
        ne_r = ne_2d[:, iz]
        Te_r = Te_2d[:, iz]
        Ti_r = Ti_2d[:, iz]
        vz_r = vz_2d[:, iz]

        # Compute thomson spectrum at each radial point: shape (nr, M)
        emissivity = thomson_spectrum(
            ne_r, Te_r, vz_r, wavelength_grid, Ti_r,
            scattering_angle, laser_wavelength, m_i, Z_ion,
        )

        # Abel-transform each wavelength bin independently
        for m in range(M):
            result[ic, m] = np.sum(abel_transform(emissivity[:, m], r_cell))

    return result


def fit_te_ne_v(
    wavelength_grid: np.ndarray,
    spectrum: np.ndarray,
    scattering_angle: float = np.pi / 2,
    laser_wavelength: float = 1064e-9,
    m_i: float = M_DEUTERIUM,
    Z_ion: int = 1,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Extract Te, ne, Ti, v_flow from a Thomson spectrum via DE.

    Uses scipy.optimize.differential_evolution with the full Salpeter
    model as the forward model. Robust to initial conditions and
    multi-modal landscapes.

    Parameters
    ----------
    wavelength_grid : np.ndarray
        Scattered wavelengths [m], shape (M,).
    spectrum : np.ndarray
        Measured or synthetic spectrum, shape (M,).
    scattering_angle : float
        Scattering angle theta [rad].
    laser_wavelength : float
        Probe laser wavelength [m].
    m_i : float
        Ion mass [kg].
    Z_ion : int
        Ion charge state.
    bounds : dict or None
        Parameter bounds override. Keys: "log10_ne", "Te_eV", "Ti_eV",
        "v_flow". Values: (min, max) tuples.

    Returns
    -------
    dict
        {"Te_eV": float, "ne_m3": float, "Ti_eV": float,
         "v_flow_ms": float, "alpha": float, "chi2_dof": float,
         "converged": bool}
    """
    k_scatter = (4.0 * pi / laser_wavelength) * np.sin(scattering_angle / 2.0)

    # Normalize spectrum to avoid Jacobian conditioning issues
    S_peak = np.max(np.abs(spectrum))
    if S_peak == 0:
        return {
            "Te_eV": 0.0, "ne_m3": 0.0, "Ti_eV": 0.0,
            "v_flow_ms": 0.0, "alpha": 0.0, "chi2_dof": np.inf,
            "converged": False,
        }
    S_normed = spectrum / S_peak

    # Default bounds: [log10(ne), Te_eV, Ti_eV, v_flow]
    default_bounds = {
        "log10_ne": (20.0, 28.0),
        "Te_eV": (10.0, 5000.0),
        "Ti_eV": (10.0, 5000.0),
        "v_flow": (-1e6, 1e6),
    }
    if bounds is not None:
        default_bounds.update(bounds)

    de_bounds = [
        default_bounds["log10_ne"],
        default_bounds["Te_eV"],
        default_bounds["Ti_eV"],
        default_bounds["v_flow"],
    ]

    def cost(params: np.ndarray) -> float:
        log_ne, Te, Ti, v = params
        model = thomson_spectrum(
            np.array([10.0**log_ne]),
            np.array([Te]),
            np.array([v]),
            wavelength_grid,
            Ti_eV=np.array([Ti]),
            scattering_angle=scattering_angle,
            laser_wavelength=laser_wavelength,
            m_i=m_i,
            Z_ion=Z_ion,
        )
        model_normed = model[0] / S_peak
        return float(np.sum((model_normed - S_normed) ** 2))

    de_result = differential_evolution(
        cost, de_bounds, seed=42, maxiter=300, tol=1e-8, polish=True,
    )

    log_ne, Te, Ti, v = de_result.x
    ne_fit = 10.0**log_ne

    # Compute alpha for regime reporting
    lambda_De = np.sqrt(epsilon_0 * k_B * (Te * eV / k_B) / (ne_fit * e**2))
    alpha = 1.0 / (k_scatter * lambda_De)

    ndof = max(len(spectrum) - 4, 1)

    return {
        "Te_eV": float(Te),
        "ne_m3": float(ne_fit),
        "Ti_eV": float(Ti),
        "v_flow_ms": float(v),
        "alpha": float(alpha),
        "chi2_dof": float(de_result.fun / ndof),
        "converged": bool(de_result.success),
    }
