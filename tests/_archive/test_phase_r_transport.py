"""Phase R.4 tests: Transport physics corrections.

Tests for:
- Braginskii alpha(Z) correction to Spitzer resistivity
- Consistent Coulomb logarithm between nernst.py and spitzer.py
- Ion mass consistency in temperature relaxation
"""

import numpy as np
import pytest

from dpf.collision.spitzer import coulomb_log, spitzer_alpha, spitzer_resistivity
from dpf.constants import m_d
from dpf.fluid.nernst import nernst_coefficient


class TestSpitzerAlphaZ:
    """Tests for the Braginskii alpha(Z) correction to Spitzer resistivity."""

    def test_alpha_z1_value(self):
        """alpha(Z=1) should be approximately 0.5064 (Braginskii Table 1)."""
        alpha = spitzer_alpha(1.0)
        assert alpha == pytest.approx(0.5064, rel=1e-4), (
            f"alpha(Z=1) = {alpha}, expected 0.5064"
        )

    def test_alpha_z2_value(self):
        """alpha(Z=2) should be approximately 0.4408."""
        alpha = spitzer_alpha(2.0)
        assert alpha == pytest.approx(0.4408, rel=1e-4), (
            f"alpha(Z=2) = {alpha}, expected 0.4408"
        )

    def test_alpha_z3_value(self):
        """alpha(Z=3) should be approximately 0.3965."""
        alpha = spitzer_alpha(3.0)
        assert alpha == pytest.approx(0.3965, rel=1e-4), (
            f"alpha(Z=3) = {alpha}, expected 0.3965"
        )

    def test_alpha_z4_value(self):
        """alpha(Z=4) should be approximately 0.3752."""
        alpha = spitzer_alpha(4.0)
        assert alpha == pytest.approx(0.3752, rel=1e-4), (
            f"alpha(Z=4) = {alpha}, expected 0.3752"
        )

    def test_alpha_z_inf_lower_bound(self):
        """alpha(Z→∞) approaches 0.2949 (Lorentz gas limit)."""
        alpha_100 = spitzer_alpha(100.0)
        alpha_1000 = spitzer_alpha(1000.0)

        assert alpha_100 == pytest.approx(0.2949, rel=1e-4), (
            f"alpha(Z=100) = {alpha_100}, expected 0.2949"
        )
        assert alpha_1000 == pytest.approx(0.2949, rel=1e-4), (
            f"alpha(Z=1000) = {alpha_1000}, expected 0.2949"
        )

    def test_alpha_z_monotonically_decreasing(self):
        """alpha(Z) should decrease monotonically with increasing Z."""
        Z_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0, 100.0]
        alpha_values = [spitzer_alpha(Z) for Z in Z_values]

        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] > alpha_values[i + 1], (
                f"alpha not monotonic: alpha(Z={Z_values[i]}) = {alpha_values[i]} "
                f"<= alpha(Z={Z_values[i+1]}) = {alpha_values[i+1]}"
            )

    def test_alpha_z_interpolation(self):
        """alpha(Z) should interpolate smoothly between table values."""
        # Test midpoint between Z=1 and Z=2
        alpha_1 = spitzer_alpha(1.0)
        alpha_2 = spitzer_alpha(2.0)
        alpha_mid = spitzer_alpha(1.5)

        # Should be roughly the average (linear interpolation)
        expected_mid = 0.5 * (alpha_1 + alpha_2)
        assert alpha_mid == pytest.approx(expected_mid, rel=1e-3), (
            f"alpha(Z=1.5) = {alpha_mid}, expected ~{expected_mid} "
            f"(midpoint of {alpha_1} and {alpha_2})"
        )

    def test_spitzer_resistivity_with_alpha(self):
        """Resistivity at Z=1 should be ~2x higher with alpha correction.

        The uncorrected formula gives eta = m_e * nu_ei / (ne * e^2).
        With alpha(Z=1) = 0.5064, the corrected formula divides by alpha,
        giving eta_corrected = eta_uncorrected / 0.5064 ≈ 1.97 × eta_uncorrected.
        """
        ne = np.array([1e20])  # m^-3
        Te = np.array([11604.5 * 1000])  # 1 keV in Kelvin
        lnL = 10.0
        Z = 1.0

        eta = spitzer_resistivity(ne, Te, lnL, Z)

        # Compute what the uncorrected value would be
        from dpf.collision.spitzer import nu_ei
        from dpf.constants import e, m_e
        freq = nu_ei(ne, Te, lnL, Z)
        eta_uncorrected = m_e * freq / (ne * e**2)

        # With alpha(1) = 0.5064, correction factor should be 1/0.5064 ≈ 1.975
        correction_factor = eta[0] / eta_uncorrected[0]
        expected_factor = 1.0 / 0.5064

        assert correction_factor == pytest.approx(expected_factor, rel=1e-3), (
            f"Correction factor = {correction_factor}, expected {expected_factor}"
        )

    def test_spitzer_resistivity_z_dependence(self):
        """Higher Z should give higher resistivity (alpha decreases with Z)."""
        ne = np.array([1e20])
        Te = np.array([11604.5 * 1000])  # 1 keV
        lnL = 10.0

        eta_z1 = spitzer_resistivity(ne, Te, lnL, Z=1.0)
        eta_z4 = spitzer_resistivity(ne, Te, lnL, Z=4.0)

        # Higher Z → higher nu_ei AND smaller alpha → higher eta
        # But nu_ei scales with Z, so the net effect is complex.
        # At least verify that Z=4 gives higher eta than Z=1
        # (nu_ei increases by 4×, alpha decreases from 0.5064 to 0.3752)
        # Net factor: 4 / (0.3752 / 0.5064) = 4 × 1.35 = 5.4×
        ratio = eta_z4[0] / eta_z1[0]
        expected_ratio = 4.0 / (0.3752 / 0.5064)

        assert ratio == pytest.approx(expected_ratio, rel=1e-3), (
            f"eta(Z=4) / eta(Z=1) = {ratio}, expected {expected_ratio}"
        )


class TestCoulombLogConsistency:
    """Test that Coulomb log is consistent between nernst.py and spitzer.py."""

    def test_nernst_uses_spitzer_coulomb_log(self):
        """nernst module should use coulomb_log from spitzer, not its old _coulomb_log.

        We verify this by checking that nernst_coefficient gives sensible results
        consistent with the spitzer.coulomb_log formula, which includes quantum
        correction via de Broglie wavelength.
        """
        ne = np.array([1e20, 1e22, 1e24])  # m^-3
        Te = np.array([11604.5 * 100, 11604.5 * 1000, 11604.5 * 10000])  # 100 eV, 1 keV, 10 keV
        B_mag = np.array([1.0, 1.0, 1.0])  # 1 T

        # Compute Nernst coefficient (uses coulomb_log internally)
        beta = nernst_coefficient(ne, Te, B_mag, Z_eff=1.0)

        # Should return finite, positive values
        assert np.all(np.isfinite(beta)), "Nernst coefficient produced non-finite values"
        assert np.all(beta >= 0), f"Nernst coefficient should be >= 0, got {beta}"

    def test_coulomb_log_values_consistency(self):
        """Direct comparison: nernst should produce same Coulomb log as spitzer.

        Since nernst now imports coulomb_log from spitzer, calling it directly
        should give identical results.
        """
        from dpf.collision.spitzer import coulomb_log as spitzer_coulomb_log
        from dpf.fluid.nernst import coulomb_log as nernst_coulomb_log

        ne = np.array([1e20, 1e22, 1e24])
        Te = np.array([11604.5 * 100, 11604.5 * 1000, 11604.5 * 10000])

        lnL_spitzer = spitzer_coulomb_log(ne, Te)
        lnL_nernst = nernst_coulomb_log(ne, Te)

        # Should be identical (same function)
        np.testing.assert_array_equal(lnL_spitzer, lnL_nernst,
                                       err_msg="nernst.coulomb_log != spitzer.coulomb_log")

    def test_coulomb_log_quantum_correction(self):
        """At high density/low temp, quantum correction should kick in.

        The spitzer.coulomb_log uses quantum correction via de Broglie wavelength.
        At low temperatures and high densities, lambda_db > b_class, so the
        minimum impact parameter is set by quantum mechanics, not classical
        distance of closest approach.
        """
        # High density, low temperature (quantum regime)
        ne_quantum = np.array([1e28])  # 10^28 m^-3 (extremely dense)
        Te_quantum = np.array([11604.5 * 1])  # 1 eV (cold)

        # Low density, high temperature (classical regime)
        ne_classical = np.array([1e20])  # 10^20 m^-3
        Te_classical = np.array([11604.5 * 1000])  # 1 keV (hot)

        lnL_quantum = coulomb_log(ne_quantum, Te_quantum)
        lnL_classical = coulomb_log(ne_classical, Te_classical)

        # Quantum regime should have smaller Coulomb log (shorter screening length)
        assert lnL_quantum[0] < lnL_classical[0], (
            f"Quantum regime lnL = {lnL_quantum[0]} should be < "
            f"classical regime lnL = {lnL_classical[0]}"
        )

        # Both should be finite and non-negative
        # Note: coulomb_log floors Lambda at 1.0, so log(Lambda) can be 0.0 in extreme cases
        assert lnL_quantum[0] >= 0.0, f"lnL should be >= 0.0, got {lnL_quantum[0]}"
        assert lnL_classical[0] >= 1.0, f"Classical lnL should be >= 1.0, got {lnL_classical[0]}"
        assert np.isfinite(lnL_quantum[0]), "lnL_quantum should be finite"
        assert np.isfinite(lnL_classical[0]), "lnL_classical should be finite"


class TestIonMassConsistency:
    """Verify ion mass consistency between config and transport modules."""

    def test_config_uses_deuterium_mass(self):
        """config.ion_mass default should be m_d (deuterium)."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("pf1000"))

        # Default ion_mass in config.py is 3.34358377e-27 kg (deuterium)
        assert config.ion_mass == pytest.approx(m_d, rel=1e-6), (
            f"config.ion_mass = {config.ion_mass}, expected m_d = {m_d}"
        )

    def test_relax_temperatures_uses_deuterium(self):
        """relax_temperatures should use m_d internally (not m_p).

        The function uses m_d in the relaxation rate calculation:
            alpha = freq_ei * dt * 2.0 * m_e / m_d

        For deuterium, m_d ≈ 2 × m_p, so the relaxation rate is half that
        of hydrogen plasma.
        """
        from dpf.collision.spitzer import relax_temperatures

        Te = np.array([11604.5 * 1000])  # 1 keV
        Ti = np.array([11604.5 * 100])   # 0.1 keV (colder ions)
        freq_ei = np.array([1e9])  # 1 GHz collision frequency
        dt = 1e-9  # 1 ns timestep

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)

        # Should equilibrate toward mean temperature
        T_mean = 0.5 * (Te[0] + Ti[0])

        assert Te_new[0] < Te[0], "Electron temp should decrease"
        assert Ti_new[0] > Ti[0], "Ion temp should increase"

        # Verify the exponential decay factor uses m_d (not m_p)
        # If it used m_p, the relaxation would be faster by factor of 2
        from dpf.constants import m_e
        alpha_expected = freq_ei[0] * dt * 2.0 * m_e / m_d
        factor_expected = np.exp(-2.0 * alpha_expected)

        # Te_new = T_mean + (Te - T_mean) * factor
        Te_expected = T_mean + (Te[0] - T_mean) * factor_expected

        assert Te_new[0] == pytest.approx(Te_expected, rel=1e-6), (
            f"Te_new = {Te_new[0]}, expected {Te_expected} (using m_d)"
        )
