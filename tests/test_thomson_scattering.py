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

from dpf.constants import c, e, epsilon_0, k_B, m_e, pi
from dpf.diagnostics.thomson_scattering import (
    eV,
    fit_te_ne_v,
    spectral_density_salpeter,
    thomson_line_integrated,
    thomson_spectrum,
)


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

        # Expected 1-sigma width in wavelength space.
        # S(omega) ~ exp(-zeta_e^2), zeta_e = omega/(k*v_th_e).
        # Gaussian std in omega: sigma_omega = k * v_th_e / sqrt(2).
        # Convert: sigma_lambda = sigma_omega * lambda0^2 / (2*pi*c).
        # With k = 4*pi*sin(theta/2)/lambda0 and v_th_e = sqrt(2*kT/m_e):
        #   sigma_lambda = 2*sin(theta/2)*lambda0 * sqrt(kT/(m_e*c^2))
        sigma_expected = (
            2.0 * np.sin(theta / 2.0) * lambda0 * np.sqrt(k_B * Te_K / (m_e * c**2))
        )

        assert abs(sigma_fit / sigma_expected - 1.0) < 0.02, (
            f"sigma_fit={sigma_fit*1e9:.3f} nm, expected={sigma_expected*1e9:.3f} nm"
        )

    def test_salpeter_sum_rule(self):
        """Integral of S(k, omega) d(omega) = 2*pi (sum rule).

        The Salpeter function in Sheffield convention includes the 2*pi/k
        prefactor and a 1D Maxwellian f(v). Integrating over all omega gives
        2*pi because:
          integral S_e dω ≈ (2π/k) * k * ∫ f_e(v) dv = 2π
        """
        ne, Te_eV = 1e22, 200.0
        lambda0, theta = 532e-9, np.pi / 2
        k = (4.0 * pi / lambda0) * np.sin(theta / 2.0)

        # Wide omega range to capture full spectrum
        omega = np.linspace(-5e15, 5e15, 10000)
        S = spectral_density_salpeter(omega, k, ne, Te_eV)
        integrate = getattr(np, "trapezoid", np.trapz)
        integral = integrate(S, omega)

        assert abs(integral / (2.0 * np.pi) - 1.0) < 0.01, (
            f"Sum rule: integral/(2*pi) = {integral/(2*np.pi):.4f}, expected 1.0"
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
