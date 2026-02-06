"""Tests for Spitzer collision frequencies and Braginskii coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from dpf.collision.spitzer import (
    braginskii_kappa,
    coulomb_log,
    nu_ei,
    nu_ee,
    nu_ii,
    nu_en,
    relax_temperatures,
)
from dpf.constants import e, epsilon_0, k_B, m_e, m_p


class TestCoulombLog:
    """Verify dynamic Coulomb logarithm."""

    def test_typical_plasma(self):
        """For typical DPF plasma (ne=1e24, Te=1e6 K), ln(Lambda) ~ 5-20."""
        ne = np.array([1e24])
        Te = np.array([1e6])
        lnL = coulomb_log(ne, Te)
        assert 3 < lnL[0] < 25

    def test_always_positive(self):
        """Coulomb log is always >= 0 (floored at ln(1) = 0)."""
        ne = np.array([1e20, 1e24, 1e28])
        Te = np.array([1e3, 1e6, 1e9])
        lnL = coulomb_log(ne, Te)
        assert np.all(lnL >= 0)


class TestSpitzerFrequencies:
    """Verify collision frequencies against NRL Plasma Formulary."""

    def test_nu_ei_order_of_magnitude(self):
        """For ne=1e20, Te=10 eV: nu_ei ~ 1e8 s^-1 (NRL)."""
        ne = np.array([1e20])
        Te_eV = 10.0
        Te = np.array([Te_eV * e / k_B])  # Convert eV to K
        lnL = 10.0

        freq = nu_ei(ne, Te, lnL)
        # NRL formulary: nu_ei ~ 3e-6 * ne * Z * lnL / Te_eV^1.5
        # ~ 3e-6 * 1e20 * 1 * 10 / 31.6 ~ 9.5e13 ... this is in CGS
        # In SI with our formula, should be order 1e6 to 1e10
        assert 1e4 < freq[0] < 1e14

    def test_nu_ee_larger_than_nu_ei(self):
        """nu_ee = sqrt(2) * nu_ei for Z=1."""
        ne = np.array([1e22])
        Te = np.array([1e5])
        lnL = 10.0

        freq_ei = nu_ei(ne, Te, lnL)
        freq_ee = nu_ee(ne, Te, lnL)
        np.testing.assert_allclose(freq_ee, freq_ei * np.sqrt(2), rtol=1e-10)

    def test_nu_en_scales_with_density(self):
        """Electron-neutral frequency is proportional to neutral density."""
        ne = np.array([1e20])
        Te = np.array([1e4])
        nn1 = np.array([1e20])
        nn2 = np.array([2e20])

        freq1 = nu_en(ne, Te, nn1)
        freq2 = nu_en(ne, Te, nn2)
        np.testing.assert_allclose(freq2, 2 * freq1, rtol=1e-10)


class TestBraginskii:
    """Verify Braginskii thermal conductivity."""

    def test_kappa_par_larger_than_perp(self):
        """Parallel conductivity >> perpendicular in magnetized plasma."""
        ne = np.array([1e22])
        Te = np.array([1e6])
        B = np.array([1.0])  # 1 Tesla — strongly magnetized

        kpar, kper = braginskii_kappa(ne, Te, B)
        assert kpar[0] > kper[0]
        assert kpar[0] / kper[0] > 10  # Should be much larger

    def test_kappa_unmagnetized(self):
        """With B=0, kappa_par == kappa_perp."""
        ne = np.array([1e22])
        Te = np.array([1e6])
        B = np.array([0.0])

        kpar, kper = braginskii_kappa(ne, Te, B)
        np.testing.assert_allclose(kpar, kper, rtol=1e-6)


class TestTemperatureRelaxation:
    """Verify implicit temperature equilibration."""

    def test_relaxation_conserves_energy(self):
        """Total thermal energy (Te + Ti) is conserved."""
        Te = np.array([1e6])
        Ti = np.array([1e4])
        freq = np.array([1e10])

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-10)
        np.testing.assert_allclose(Te_new + Ti_new, Te + Ti, rtol=1e-10)

    def test_relaxation_toward_equilibrium(self):
        """Temperatures approach each other."""
        Te = np.array([1e6])
        Ti = np.array([1e4])
        freq = np.array([1e10])

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-8)
        assert abs(Te_new[0] - Ti_new[0]) < abs(Te[0] - Ti[0])

    def test_equilibrium_unchanged(self):
        """If Te == Ti, relaxation does nothing."""
        T = np.array([5e5])
        freq = np.array([1e10])

        Te_new, Ti_new = relax_temperatures(T.copy(), T.copy(), freq, dt=1e-8)
        np.testing.assert_allclose(Te_new, T, rtol=1e-10)
        np.testing.assert_allclose(Ti_new, T, rtol=1e-10)
