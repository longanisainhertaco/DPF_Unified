# Thomson Scattering Diagnostic — Prototype Code

## Module: `src/dpf/diagnostics/thomson_scattering.py`

```python
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

    Returns
    -------
    np.ndarray
        Line-integrated spectral power [W/m^2/sr/m], shape (Nc, M).
    """
    from dpf.diagnostics.interferometry import abel_transform

    if Ti_2d is None:
        Ti_2d = Te_2d

    nr, nz = ne_2d.shape
    Nc = len(chord_positions_z)
    M = len(wavelength_grid)

    # Build z-coordinate from grid (assume uniform spacing)
    # Caller should ensure chord_positions_z maps to valid z indices
    z_cell = np.linspace(0, 1, nz)  # placeholder; overridden by nearest-index lookup

    result = np.zeros((Nc, M))
    for ic in range(Nc):
        # Find nearest z-index for this chord
        z_target = chord_positions_z[ic]
        # If caller provides z_cell externally, use it; otherwise use column index
        iz = min(int(z_target * nz), nz - 1) if nz > 1 else 0

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
```

## Tests: `tests/test_thomson_scattering.py`

```python
"""Tests for Thomson scattering synthetic diagnostic.

Covers:
1. Non-collective Gaussian limit (alpha << 1)
2. Collective ion-acoustic peak (alpha >> 1, PF-1000 conditions)
3. Doppler shift from bulk flow
4. fit_te_ne_v roundtrip recovery
5. Line integration for uniform plasma
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.diagnostics.thomson_scattering import (
    M_DEUTERIUM,
    eV,
    r_e,
    spectral_density_salpeter,
    thomson_line_integrated,
    thomson_spectrum,
    fit_te_ne_v,
)
from dpf.constants import c, e, epsilon_0, k_B, m_e, pi


class TestNonCollectiveGaussian:
    """Test 1: alpha << 1 — spectrum should be a Gaussian."""

    def test_gaussian_shape(self):
        ne, Te_eV = 1e22, 200.0
        lambda0, theta = 532e-9, np.pi / 2
        wl = np.linspace(lambda0 - 60e-9, lambda0 + 60e-9, 1000)

        spec = thomson_spectrum(
            np.array([ne]), np.array([Te_eV]), np.array([0.0]),
            wl, scattering_angle=theta, laser_wavelength=lambda0,
        )

        # Verify alpha << 1
        k = (4.0 * pi / lambda0) * np.sin(theta / 2.0)
        Te_K = Te_eV * eV / k_B
        lambda_D = np.sqrt(epsilon_0 * k_B * Te_K / (ne * e**2))
        alpha = 1.0 / (k * lambda_D)
        assert alpha < 0.1, f"alpha = {alpha}, expected << 1"

        # Fit a Gaussian to extract width
        from scipy.optimize import curve_fit

        def gauss(x, A, mu, sig):
            return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

        popt, _ = curve_fit(
            gauss, wl, spec[0],
            p0=[spec[0].max(), lambda0, 10e-9],
        )
        sigma_fit = abs(popt[2])

        # Expected thermal width (1/e half-width)
        delta_th = lambda0 * np.sqrt(2.0 * k_B * Te_K / (m_e * c**2)) * np.sin(theta / 2.0)
        sigma_expected = delta_th / np.sqrt(2.0)

        assert abs(sigma_fit / sigma_expected - 1.0) < 0.02, (
            f"sigma_fit={sigma_fit*1e9:.3f} nm, expected={sigma_expected*1e9:.3f} nm"
        )

    def test_salpeter_sum_rule(self):
        """Integral of S(k, omega) d(omega) = 1/k (sum rule)."""
        ne, Te_eV = 1e22, 200.0
        lambda0, theta = 532e-9, np.pi / 2
        k = (4.0 * pi / lambda0) * np.sin(theta / 2.0)

        # Wide omega range to capture full spectrum
        omega = np.linspace(-5e15, 5e15, 10000)
        S = spectral_density_salpeter(omega, k, ne, Te_eV)
        integral = np.trapz(S, omega)

        assert abs(integral * k - 1.0) < 0.01, (
            f"Sum rule: integral*k = {integral*k:.4f}, expected 1.0"
        )


class TestCollectiveIonFeature:
    """Test 2: alpha >> 1 — ion acoustic feature dominates."""

    def test_ion_feature_dominance(self):
        ne, Te_eV, Ti_eV = 1e25, 300.0, 100.0
        lambda0, theta = 1064e-9, np.pi / 2
        wl = np.linspace(lambda0 - 50e-9, lambda0 + 50e-9, 2000)

        spec = thomson_spectrum(
            np.array([ne]), np.array([Te_eV]), np.array([0.0]),
            wl, Ti_eV=np.array([Ti_eV]),
            scattering_angle=theta, laser_wavelength=lambda0,
        )

        # Verify alpha > 1
        k = (4.0 * pi / lambda0) * np.sin(theta / 2.0)
        Te_K = Te_eV * eV / k_B
        lambda_D = np.sqrt(epsilon_0 * k_B * Te_K / (ne * e**2))
        alpha = 1.0 / (k * lambda_D)
        assert alpha > 2.0, f"alpha = {alpha:.2f}, expected > 2"

        # Ion feature near line center should dominate electron wings
        idx_1nm = np.argmin(np.abs(wl - (lambda0 + 1e-9)))
        idx_15nm = np.argmin(np.abs(wl - (lambda0 + 15e-9)))
        ratio = spec[0, idx_1nm] / spec[0, idx_15nm]
        assert ratio > 50, f"Ion/electron ratio = {ratio:.1f}, expected > 50"

    def test_alpha_value(self):
        """Verify computed alpha matches analytical prediction."""
        ne, Te_eV = 1e25, 300.0
        lambda0, theta = 1064e-9, np.pi / 2
        k = (4.0 * pi / lambda0) * np.sin(theta / 2.0)
        Te_K = Te_eV * eV / k_B
        lambda_D = np.sqrt(epsilon_0 * k_B * Te_K / (ne * e**2))
        alpha = 1.0 / (k * lambda_D)
        assert abs(alpha - 2.94) < 0.05, f"alpha = {alpha:.3f}, expected ~2.94"


class TestDopplerShift:
    """Test 3: bulk flow produces measurable spectral shift."""

    def test_peak_shift(self):
        ne, Te_eV, v_bulk = 1e22, 200.0, 2e5
        lambda0, theta = 532e-9, np.pi / 2
        wl = np.linspace(lambda0 - 60e-9, lambda0 + 60e-9, 2000)

        spec = thomson_spectrum(
            np.array([ne]), np.array([Te_eV]), np.array([v_bulk]),
            wl, scattering_angle=theta, laser_wavelength=lambda0,
        )

        peak_wl = wl[np.argmax(spec[0])]

        # Expected Doppler shift
        delta_D = (lambda0 / c) * v_bulk * 2.0 * np.sin(theta / 2.0)
        measured_shift = peak_wl - lambda0

        assert abs(measured_shift - delta_D) < 0.1e-9, (
            f"Shift = {measured_shift*1e9:.3f} nm, expected {delta_D*1e9:.3f} nm"
        )

    def test_width_unchanged_by_flow(self):
        """Bulk flow shifts the peak but doesn't broaden the spectrum."""
        ne, Te_eV = 1e22, 200.0
        lambda0, theta = 532e-9, np.pi / 2
        wl = np.linspace(lambda0 - 60e-9, lambda0 + 60e-9, 2000)

        from scipy.optimize import curve_fit

        def gauss(x, A, mu, sig):
            return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

        spec_0 = thomson_spectrum(
            np.array([ne]), np.array([Te_eV]), np.array([0.0]),
            wl, scattering_angle=theta, laser_wavelength=lambda0,
        )
        popt_0, _ = curve_fit(gauss, wl, spec_0[0], p0=[spec_0[0].max(), lambda0, 10e-9])

        spec_v = thomson_spectrum(
            np.array([ne]), np.array([Te_eV]), np.array([2e5]),
            wl, scattering_angle=theta, laser_wavelength=lambda0,
        )
        popt_v, _ = curve_fit(gauss, wl, spec_v[0], p0=[spec_v[0].max(), lambda0, 10e-9])

        assert abs(abs(popt_0[2]) / abs(popt_v[2]) - 1.0) < 0.01, (
            "Spectral width changed with bulk flow"
        )


@pytest.mark.slow
class TestFitRoundtrip:
    """Test 4: fit_te_ne_v recovers known parameters."""

    def test_recover_noncollective(self):
        """Recover Te, ne from non-collective spectrum (alpha << 1)."""
        ne_true, Te_true = 1e22, 200.0
        lambda0, theta = 532e-9, np.pi / 2
        wl = np.linspace(lambda0 - 60e-9, lambda0 + 60e-9, 500)

        spec = thomson_spectrum(
            np.array([ne_true]), np.array([Te_true]), np.array([0.0]),
            wl, scattering_angle=theta, laser_wavelength=lambda0,
        )

        result = fit_te_ne_v(
            wl, spec[0],
            scattering_angle=theta, laser_wavelength=lambda0,
            bounds={
                "log10_ne": (20.0, 24.0),
                "Te_eV": (50.0, 500.0),
                "Ti_eV": (50.0, 500.0),
                "v_flow": (-1e4, 1e4),
            },
        )

        assert result["converged"], "DE did not converge"
        assert abs(result["Te_eV"] / Te_true - 1.0) < 0.05, (
            f"Te recovery: {result['Te_eV']:.1f} eV, expected {Te_true:.1f} eV"
        )
        assert abs(np.log10(result["ne_m3"]) / np.log10(ne_true) - 1.0) < 0.02, (
            f"ne recovery: {result['ne_m3']:.2e}, expected {ne_true:.2e}"
        )

    def test_recover_collective(self):
        """Recover Te, ne, Ti from collective spectrum (PF-1000)."""
        ne_true, Te_true, Ti_true = 1e25, 300.0, 100.0
        lambda0, theta = 1064e-9, np.pi / 2
        wl = np.linspace(lambda0 - 50e-9, lambda0 + 50e-9, 500)

        spec = thomson_spectrum(
            np.array([ne_true]), np.array([Te_true]), np.array([0.0]),
            wl, Ti_eV=np.array([Ti_true]),
            scattering_angle=theta, laser_wavelength=lambda0,
        )

        result = fit_te_ne_v(
            wl, spec[0],
            scattering_angle=theta, laser_wavelength=lambda0,
            bounds={
                "log10_ne": (23.0, 27.0),
                "Te_eV": (50.0, 1000.0),
                "Ti_eV": (10.0, 500.0),
                "v_flow": (-1e4, 1e4),
            },
        )

        assert result["converged"], "DE did not converge"
        assert abs(result["Te_eV"] / Te_true - 1.0) < 0.10, (
            f"Te recovery: {result['Te_eV']:.1f} eV, expected {Te_true:.1f} eV"
        )
        assert abs(result["Ti_eV"] / Ti_true - 1.0) < 0.10, (
            f"Ti recovery: {result['Ti_eV']:.1f} eV, expected {Ti_true:.1f} eV"
        )


class TestLineIntegration:
    """Test 5: Abel-integrated spectrum for uniform plasma."""

    def test_uniform_plasma_scaling(self):
        """For uniform plasma, line-integrated power scales with path length."""
        nr, nz = 20, 10
        ne_val = 1e23
        Te_val = 100.0
        r_max = 0.01  # 1 cm

        r_cell = np.linspace(r_max / nr, r_max, nr)
        ne_2d = np.full((nr, nz), ne_val)
        Te_2d = np.full((nr, nz), Te_val)
        vz_2d = np.zeros((nr, nz))

        lambda0 = 1064e-9
        wl = np.linspace(lambda0 - 30e-9, lambda0 + 30e-9, 100)

        chord_z = np.array([0.5])  # single chord at midplane

        result = thomson_line_integrated(
            ne_2d, Te_2d, vz_2d, r_cell, chord_z, wl,
            laser_wavelength=lambda0,
        )

        assert result.shape == (1, 100)
        # For uniform plasma, the line-integrated signal should be nonzero
        assert np.max(result) > 0, "Line-integrated spectrum is zero"

        # The spectral shape should match the point spectrum (uniform plasma)
        point_spec = thomson_spectrum(
            np.array([ne_val]), np.array([Te_val]), np.array([0.0]),
            wl, laser_wavelength=lambda0,
        )
        # Normalize both and check shape similarity
        if np.max(result) > 0 and np.max(point_spec) > 0:
            r_normed = result[0] / np.max(result[0])
            p_normed = point_spec[0] / np.max(point_spec[0])
            shape_error = np.max(np.abs(r_normed - p_normed))
            assert shape_error < 0.05, (
                f"Spectral shape mismatch: max error = {shape_error:.3f}"
            )
```

## Integration Notes

### Registration in `__init__.py`

Add to `src/dpf/diagnostics/__init__.py`:

```python
"""Diagnostics module — HDF5 output, neutron yield, interferometry, X-ray, Thomson scattering, Pease-Braginskii."""
```

The module is imported on demand (no eager import needed). Users access it via:

```python
from dpf.diagnostics.thomson_scattering import thomson_spectrum, fit_te_ne_v
```

### Dependencies

All already available in the project:
- `numpy` (core)
- `scipy.special.wofz` (Faddeeva function)
- `scipy.optimize.differential_evolution` (robust fitting)
- `dpf.constants` (e, epsilon_0, m_e, k_B, c, pi)
- `dpf.diagnostics.interferometry.abel_transform` (line integration)

No new packages to install.

### Key Design Decisions

1. **Single code path via wofz**: No regime switching. The Faddeeva-based Salpeter function handles alpha from 0.01 to 100+ identically.

2. **Differential evolution for fitting** (not curve_fit): The investigation in `docs/investigations/thomson_fitting_robustness.md` showed curve_fit fails 100% of the time on raw Salpeter spectra due to Jacobian ill-conditioning (38 orders of magnitude between S ~ 1e-13 and ne ~ 1e25). DE with normalized spectrum is robust.

3. **Normalized fitting**: Spectrum normalized to peak=1, ne fit in log10 space. This eliminates the gradient-invisibility problem that kills TRF/L-BFGS-B.

4. **Abel transform reuse**: Each wavelength bin is Abel-transformed independently via the existing `interferometry.abel_transform()`. No new quadrature code.

5. **Units**: All SI. Wavelengths in meters, temperatures in eV (matching xray_imaging.py convention).
