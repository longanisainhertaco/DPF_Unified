"""Tests for ionization module: Saha equation, Z_bar(Te, ne).

Test categories:
1. Saha parameter matches analytic formula
2. Z_bar -> 0 at low temperature (neutral gas)
3. Z_bar -> 1 at high temperature (fully ionized)
4. Z_bar transition region at correct temperature
5. Density dependence (higher ne -> lower Z_bar at same Te)
6. Array version matches scalar version
"""

from __future__ import annotations

import numpy as np

from dpf.constants import eV, k_B

# ═══════════════════════════════════════════════════════
# Saha Ionization Fraction Tests
# ═══════════════════════════════════════════════════════

class TestSahaIonizationFraction:
    """Tests for Saha ionization fraction."""

    def test_cold_gas_neutral(self):
        """At room temperature, hydrogen should be fully neutral (Z ~ 0)."""
        from dpf.atomic.ionization import saha_ionization_fraction

        Te = 300.0   # Room temperature [K]
        ne = 1e20    # Typical density

        Z = saha_ionization_fraction(Te, ne)
        assert Z < 1e-10, f"Z = {Z:.2e}, should be ~0 at room temperature"

    def test_hot_plasma_ionized(self):
        """At 10 eV, hydrogen should be fully ionized (Z ~ 1)."""
        from dpf.atomic.ionization import saha_ionization_fraction

        Te = 10.0 * eV / k_B   # 10 eV in Kelvin (~116,000 K)
        ne = 1e20

        Z = saha_ionization_fraction(Te, ne)
        assert Z > 0.99, f"Z = {Z:.4f}, should be ~1 at 10 eV"

    def test_very_hot_plasma(self):
        """At 1 keV, Z should be essentially 1."""
        from dpf.atomic.ionization import saha_ionization_fraction

        Te = 1000.0 * eV / k_B  # 1 keV in Kelvin
        ne = 1e24   # DPF pinch density

        Z = saha_ionization_fraction(Te, ne)
        assert Z > 0.999, f"Z = {Z:.6f}, should be ~1 at 1 keV"

    def test_transition_around_1eV(self):
        """The ionization transition for H should occur around 1-2 eV."""
        from dpf.atomic.ionization import saha_ionization_fraction

        ne = 1e20

        # At 0.5 eV: mostly neutral
        Te_low = 0.5 * eV / k_B
        Z_low = saha_ionization_fraction(Te_low, ne)

        # At 2 eV: mostly ionized
        Te_high = 2.0 * eV / k_B
        Z_high = saha_ionization_fraction(Te_high, ne)

        assert Z_low < 0.5, f"Z at 0.5 eV = {Z_low:.3f}, should be < 0.5"
        assert Z_high > 0.5, f"Z at 2 eV = {Z_high:.3f}, should be > 0.5"

    def test_monotonic_in_temperature(self):
        """Z_bar should monotonically increase with temperature."""
        from dpf.atomic.ionization import saha_ionization_fraction

        ne = 1e20
        temps = [100.0, 1000.0, 5000.0, 10000.0, 50000.0, 1e6]
        Z_values = [saha_ionization_fraction(T, ne) for T in temps]

        for i in range(len(Z_values) - 1):
            assert Z_values[i + 1] >= Z_values[i], (
                f"Z not monotonic: Z({temps[i]:.0f}K)={Z_values[i]:.4f} > "
                f"Z({temps[i+1]:.0f}K)={Z_values[i+1]:.4f}"
            )

    def test_higher_density_lower_ionization(self):
        """At fixed Te, higher ne should give lower Z_bar (recombination wins)."""
        from dpf.atomic.ionization import saha_ionization_fraction

        # Choose a temperature in the transition region
        Te = 1.0 * eV / k_B  # 1 eV

        Z_low_ne = saha_ionization_fraction(Te, 1e18)
        Z_high_ne = saha_ionization_fraction(Te, 1e24)

        assert Z_low_ne > Z_high_ne, (
            f"Higher density should reduce ionization: "
            f"Z(1e18)={Z_low_ne:.4f}, Z(1e24)={Z_high_ne:.4f}"
        )

    def test_bounded_zero_one(self):
        """Z_bar should always be in [0, 1]."""
        from dpf.atomic.ionization import saha_ionization_fraction

        test_cases = [
            (1.0, 1.0),          # Edge case: very low
            (300.0, 1e20),       # Room temp
            (1e4, 1e15),         # Low density, moderate temp
            (1e7, 1e25),         # DPF conditions
            (1e8, 1e30),         # Extreme
        ]
        for Te, ne in test_cases:
            Z = saha_ionization_fraction(Te, ne)
            assert 0.0 <= Z <= 1.0, f"Z({Te}, {ne}) = {Z} out of bounds"


# ═══════════════════════════════════════════════════════
# Saha Array Version Tests
# ═══════════════════════════════════════════════════════

class TestSahaArray:
    """Tests for vectorized Saha ionization."""

    def test_matches_scalar(self):
        """Array version should match scalar for each element."""
        from dpf.atomic.ionization import (
            saha_ionization_fraction,
            saha_ionization_fraction_array,
        )

        Te = np.array([300.0, 5000.0, 50000.0, 1e6])
        ne = np.array([1e20, 1e20, 1e20, 1e20])

        Z_array = saha_ionization_fraction_array(Te, ne)

        for i in range(len(Te)):
            Z_scalar = saha_ionization_fraction(Te[i], ne[i])
            np.testing.assert_allclose(
                Z_array[i], Z_scalar, rtol=1e-10,
                err_msg=f"Mismatch at Te={Te[i]}, ne={ne[i]}"
            )

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        from dpf.atomic.ionization import saha_ionization_fraction_array

        Te = np.full((4, 4, 4), 1e5)
        ne = np.full((4, 4, 4), 1e20)

        Z = saha_ionization_fraction_array(Te, ne)
        assert Z.shape == Te.shape
