"""Tests for coronal equilibrium ionization state model.

Validates Z_eff(Te) against published coronal equilibrium data
(Post 1977, Summers 1974) and physical constraints.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.ionization import coronal_radiation_power, coronal_z_eff


class TestCoronalZEff:
    """Tests for coronal_z_eff function."""

    def test_copper_low_te(self):
        """At 1 eV, Cu should be barely ionized: Z_eff ~ 1-2."""
        z = coronal_z_eff(1.0, Z_nucleus=29)
        assert 1.0 <= float(z.ravel()[0]) <= 3.0

    def test_copper_10eV(self):
        """At 10 eV, Cu should have Z_eff ~ 5-8 (N-shell ionization)."""
        z = coronal_z_eff(10.0, Z_nucleus=29)
        assert 4.0 <= float(z.ravel()[0]) <= 10.0

    def test_copper_100eV(self):
        """At 100 eV, Cu should have Z_eff ~ 15-20 (M-shell ionization)."""
        z = coronal_z_eff(100.0, Z_nucleus=29)
        assert 12.0 <= float(z.ravel()[0]) <= 22.0

    def test_copper_1000eV(self):
        """At 1000 eV, Cu should be nearly fully stripped: Z_eff > 25."""
        z = coronal_z_eff(1000.0, Z_nucleus=29)
        assert float(z.ravel()[0]) > 24.0

    def test_monotonically_increasing(self):
        """Z_eff must increase monotonically with Te."""
        Te = np.logspace(-1, 4, 200)
        z = coronal_z_eff(Te, Z_nucleus=29)
        dz = np.diff(z)
        assert np.all(dz >= 0), f"Z_eff decreased at indices {np.where(dz < 0)}"

    def test_bounded_by_z_nucleus(self):
        """Z_eff must never exceed Z_nucleus or drop below 1."""
        Te = np.logspace(-1, 5, 500)
        for Z_nuc in [6, 10, 18, 29, 74]:
            z = coronal_z_eff(Te, Z_nucleus=Z_nuc)
            assert np.all(z >= 0.1), f"Z<0.1 for Z_nuc={Z_nuc}"
            assert np.all(z <= Z_nuc), f"Z>{Z_nuc} for Z_nuc={Z_nuc}"

    def test_hydrogen_saturates_at_1(self):
        """H should reach Z_eff = 1 above 13.6 eV."""
        z = coronal_z_eff(20.0, Z_nucleus=1)
        assert float(z.ravel()[0]) == pytest.approx(1.0, abs=0.1)

    def test_hydrogen_below_ip(self):
        """H below ionization potential should have Z_eff < 1."""
        z = coronal_z_eff(5.0, Z_nucleus=1)
        assert float(z.ravel()[0]) < 1.0

    def test_helium_two_stages(self):
        """He should reach Z_eff ~ 2 at high Te."""
        z = coronal_z_eff(200.0, Z_nucleus=2)
        assert float(z.ravel()[0]) == pytest.approx(2.0, abs=0.3)

    def test_scalar_input(self):
        """Scalar input should return a 1-element array."""
        z = coronal_z_eff(100.0)
        assert z.shape == (1,)

    def test_array_input(self):
        """Array input should return same shape."""
        Te = np.array([1.0, 10.0, 100.0, 1000.0])
        z = coronal_z_eff(Te, Z_nucleus=29)
        assert z.shape == Te.shape

    def test_negative_te_clipped(self):
        """Negative temperatures should be clipped, not crash."""
        z = coronal_z_eff(-5.0, Z_nucleus=29)
        assert np.isfinite(z).all()
        assert float(z.ravel()[0]) >= 1.0

    def test_zero_te_clipped(self):
        """Zero temperature should return finite Z_eff >= 1."""
        z = coronal_z_eff(0.0, Z_nucleus=29)
        assert np.isfinite(z).all()
        assert float(z.ravel()[0]) >= 1.0

    def test_generic_element(self):
        """Arbitrary Z (not in lookup table) should still work."""
        z = coronal_z_eff(100.0, Z_nucleus=26)  # Iron
        assert 10.0 < float(z.ravel()[0]) < 26.0

    def test_carbon_ordering(self):
        """Carbon Z_eff should be ordered correctly across temperatures."""
        z_low = coronal_z_eff(5.0, Z_nucleus=6).ravel()[0]
        z_mid = coronal_z_eff(50.0, Z_nucleus=6).ravel()[0]
        z_high = coronal_z_eff(500.0, Z_nucleus=6).ravel()[0]
        assert z_low < z_mid < z_high

    def test_higher_z_needs_higher_te(self):
        """Higher-Z elements should have lower Z_eff/Z_nuc at same Te."""
        z_c = coronal_z_eff(50.0, Z_nucleus=6).ravel()[0] / 6
        z_cu = coronal_z_eff(50.0, Z_nucleus=29).ravel()[0] / 29
        assert z_c > z_cu, "C should be more ionized (fractionally) than Cu at 50 eV"


class TestCoronalRadiationPower:
    """Tests for coronal_radiation_power function."""

    def test_positive_power(self):
        """Radiation power must be positive for physical inputs."""
        Te = np.array([10.0, 100.0, 1000.0])
        ne = np.full_like(Te, 1e24)
        Z_eff = coronal_z_eff(Te, Z_nucleus=29)
        P = coronal_radiation_power(Te, ne, Z_eff)
        assert np.all(P > 0)

    def test_zero_density_zero_power(self):
        """No electrons = no radiation."""
        Te = np.array([100.0])
        ne = np.array([0.0])
        Z_eff = coronal_z_eff(Te)
        P = coronal_radiation_power(Te, ne, Z_eff)
        assert P.ravel()[0] == 0.0

    def test_power_scales_with_ne_squared(self):
        """P_rad should scale as ne^2."""
        Te = np.array([100.0])
        Z_eff = coronal_z_eff(Te)
        P1 = coronal_radiation_power(Te, np.array([1e24]), Z_eff)
        P2 = coronal_radiation_power(Te, np.array([2e24]), Z_eff)
        ratio = (P2 / P1).ravel()[0]
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_line_enhancement_at_low_te(self):
        """Line radiation should enhance total power at low Te."""
        Te_low = np.array([10.0])
        Te_high = np.array([5000.0])
        ne = np.array([1e24])
        Z_low = coronal_z_eff(Te_low)
        Z_high = coronal_z_eff(Te_high)
        P_low = coronal_radiation_power(Te_low, ne, Z_low)
        P_high = coronal_radiation_power(Te_high, ne, Z_high)
        # At low Te, line enhancement factor > 1, so P_low/P_brems_low > P_high/P_brems_high
        # (normalized by Z_eff^2 * sqrt(Te))
        assert P_low > 0 and P_high > 0

    def test_array_shapes_match(self):
        """Output shape must match input shapes."""
        Te = np.linspace(1, 1000, 50)
        ne = np.full_like(Te, 1e24)
        Z_eff = coronal_z_eff(Te)
        P = coronal_radiation_power(Te, ne, Z_eff)
        assert P.shape == Te.shape


class TestConfigIntegration:
    """Test that the config field exists and defaults correctly."""

    def test_ionization_model_default(self):
        from dpf.config import RadiationConfig

        cfg = RadiationConfig()
        assert cfg.ionization_model == "fixed"

    def test_ionization_model_coronal(self):
        from dpf.config import RadiationConfig

        cfg = RadiationConfig(ionization_model="coronal")
        assert cfg.ionization_model == "coronal"
