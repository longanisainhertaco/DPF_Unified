"""Tests for the equation of state."""

from __future__ import annotations

import numpy as np

from dpf.constants import k_B, m_p
from dpf.fluid.eos import IdealEOS


class TestIdealEOS:
    """Verify two-temperature ideal gas EOS."""

    def setup_method(self):
        self.eos = IdealEOS(gamma=5.0 / 3.0)

    def test_ion_pressure(self):
        """p_i = n_i * k_B * T_i."""
        rho = np.array([1e-4])  # kg/m^3
        Ti = np.array([1e4])    # K
        n_i = rho / m_p
        expected = n_i * k_B * Ti
        result = self.eos.ion_pressure(rho, Ti)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_electron_pressure_different_from_ion(self):
        """Electron pressure uses Z * n_i, NOT the same as ion pressure."""
        rho = np.array([1e-4])
        T = np.array([1e4])

        p_i = self.eos.ion_pressure(rho, T)
        p_e = self.eos.electron_pressure(rho, T)

        # For Z=1, they should be equal
        np.testing.assert_allclose(p_i, p_e, rtol=1e-10)

        # For Z=2, electron pressure should be 2x ion
        eos_Z2 = IdealEOS(gamma=5.0 / 3.0, Z=2.0)
        p_e_Z2 = eos_Z2.electron_pressure(rho, T)
        np.testing.assert_allclose(p_e_Z2, 2.0 * p_i, rtol=1e-10)

    def test_total_pressure(self):
        """Total = ion + electron."""
        rho = np.array([1e-4])
        Ti = np.array([1e4])
        Te = np.array([2e4])

        p_total = self.eos.total_pressure(rho, Ti, Te)
        p_sum = self.eos.ion_pressure(rho, Ti) + self.eos.electron_pressure(rho, Te)
        np.testing.assert_allclose(p_total, p_sum, rtol=1e-10)

    def test_energy_positive(self):
        """Internal energy must be positive for positive T."""
        rho = np.array([1e-4])
        T = np.array([1e4])
        assert self.eos.ion_energy(rho, T) > 0
        assert self.eos.electron_energy(rho, T) > 0

    def test_sound_speed_positive(self):
        """Sound speed must be real and positive."""
        rho = np.array([1e-4])
        Ti = np.array([1e4])
        Te = np.array([1e4])
        cs = self.eos.sound_speed(rho, Ti, Te)
        assert cs > 0

    def test_temperature_recovery(self):
        """Temperature -> energy -> temperature roundtrip."""
        rho = np.array([1e-3])
        T_orig = np.array([5e4])
        e_int = self.eos.ion_energy(rho, T_orig)
        T_recovered = self.eos.temperature_from_energy(rho, e_int)
        np.testing.assert_allclose(T_recovered, T_orig, rtol=1e-6)
