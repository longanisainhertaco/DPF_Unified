"""Tests for p-B11 fusion reactivity and yield rate (Challenge 11).

Covers:
- pb11_reactivity: boundary behavior, positivity, peak location, monotonicity
- pb11_yield_rate: distinct-species factor, integral conservation, edge cases
- pb11_alpha_power_density: physical units and scaling
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import eV, k_B
from dpf.diagnostics.pb11_yield import (
    pb11_alpha_power_density,
    pb11_reactivity,
    pb11_reactivity_array,
    pb11_yield_rate,
)


# ---------------------------------------------------------------------------
# pb11_reactivity — scalar interface
# ---------------------------------------------------------------------------


class TestPb11ReactivityBoundary:
    """Tests for the low-temperature cutoff behavior."""

    def test_zero_at_zero_temperature(self):
        """Reactivity is exactly 0 at T=0 keV."""
        assert pb11_reactivity(0.0) == 0.0

    def test_zero_below_10_keV(self):
        """Reactivity returns 0 for T < 10 keV (Gamow-suppressed regime)."""
        assert pb11_reactivity(0.0) == 0.0
        assert pb11_reactivity(1.0) == 0.0
        assert pb11_reactivity(5.0) == 0.0
        assert pb11_reactivity(9.99) == 0.0

    def test_positive_above_100_keV(self):
        """Reactivity is strictly positive for T > 100 keV."""
        assert pb11_reactivity(100.0) > 0.0
        assert pb11_reactivity(300.0) > 0.0
        assert pb11_reactivity(600.0) > 0.0
        assert pb11_reactivity(1000.0) > 0.0


class TestPb11ReactivityPhysics:
    """Physics constraints on the p-B11 reactivity."""

    def test_peak_near_500_keV(self):
        """p-B11 reactivity peaks near 500 keV (resonance at 148 keV CM energy)."""
        temps = np.linspace(100.0, 1500.0, 300)
        svs = np.array([pb11_reactivity(T) for T in temps])
        peak_idx = int(np.argmax(svs))
        peak_T = float(temps[peak_idx])
        assert 300.0 <= peak_T <= 800.0, (
            f"p-B11 peak should be 300-800 keV, got {peak_T:.0f} keV"
        )

    def test_increases_from_50_to_300_keV(self):
        """Reactivity should increase from 50 keV to the peak (~300-600 keV)."""
        sv_50 = pb11_reactivity(50.0)
        sv_150 = pb11_reactivity(150.0)
        sv_300 = pb11_reactivity(300.0)
        assert sv_150 > sv_50, "Reactivity should increase from 50 to 150 keV"
        assert sv_300 > sv_150, "Reactivity should increase from 150 to 300 keV"

    def test_units_m3_per_second(self):
        """Reactivity at 300 keV is in physically expected range for p-B11.

        Nevins & Swain (2000) Table I: <sigma*v>(300 keV) ~ 6e-20 cm^3/s = 6e-26 m^3/s.
        Peak reactivity near 500 keV: ~8e-26 m^3/s.
        """
        sv = pb11_reactivity(300.0)
        assert 1.0e-27 < sv < 1.0e-18, (
            f"<sigma*v> at 300 keV = {sv:.3e} m^3/s, outside expected 1e-27 to 1e-18 range"
        )

    def test_much_smaller_than_dd_at_low_temperature(self):
        """p-B11 reactivity is negligible vs DD at DPF pinch temperatures (~10 keV).

        Cutoff is at T < 10 keV (strictly below).  At 5 keV p-B11 returns 0.
        DD at 10 keV: ~3e-25 m^3/s.  pB11 at 5 keV: 0 (Gamow-suppressed).
        """
        from dpf.diagnostics.neutron_yield import dd_reactivity

        sv_pB11_5keV = pb11_reactivity(5.0)
        sv_pB11_9keV = pb11_reactivity(9.9)
        sv_DD = dd_reactivity(10.0)
        assert sv_pB11_5keV == 0.0, "p-B11 should be zero at 5 keV"
        assert sv_pB11_9keV == 0.0, "p-B11 should be zero at 9.9 keV"
        assert sv_DD > 0.0, "DD should be nonzero at 10 keV"

    def test_monotone_increase_to_peak(self):
        """Reactivity increases monotonically from 10 keV to the ~500 keV peak."""
        # Check every 50 keV step from 10 to 500 keV is increasing
        temps = np.arange(10.0, 510.0, 50.0)
        svs = np.array([pb11_reactivity(T) for T in temps])
        for i in range(len(svs) - 1):
            assert svs[i + 1] >= svs[i], (
                f"Reactivity should increase: sv({temps[i]:.0f})={svs[i]:.3e} "
                f"> sv({temps[i+1]:.0f})={svs[i+1]:.3e}"
            )


class TestPb11ReactivityArray:
    """Tests for the vectorized pb11_reactivity_array interface."""

    def test_array_matches_scalar(self):
        """Vectorized output matches element-wise scalar calls."""
        temps = np.array([0.0, 5.0, 50.0, 200.0, 500.0, 1000.0])
        sv_arr = pb11_reactivity_array(temps)
        sv_scalar = np.array([pb11_reactivity(T) for T in temps])
        np.testing.assert_allclose(sv_arr, sv_scalar, rtol=1e-12)

    def test_array_shape_preserved(self):
        """Output shape matches input shape for multi-dimensional arrays."""
        temps = np.linspace(10.0, 600.0, 24).reshape(4, 6)
        sv = pb11_reactivity_array(temps)
        assert sv.shape == temps.shape

    def test_array_non_negative(self):
        """Reactivity array is non-negative everywhere."""
        temps = np.linspace(0.0, 1000.0, 100)
        sv = pb11_reactivity_array(temps)
        assert np.all(sv >= 0.0), "Reactivity must be non-negative"


# ---------------------------------------------------------------------------
# pb11_yield_rate — volumetric integration
# ---------------------------------------------------------------------------


class TestPb11YieldRate:
    """Tests for pb11_yield_rate volumetric integration."""

    def test_zero_yield_below_cutoff_temperature(self):
        """Yield rate is zero when ion temperature is below 10 keV everywhere."""
        shape = (4, 4)
        # Ti = 5 keV -> 5e3 * eV / k_B Kelvin
        Ti_K = (5.0e3 * eV / k_B) * np.ones(shape)
        n_p = 1.0e24 * np.ones(shape)
        n_B = 1.0e22 * np.ones(shape)
        cell_volumes = 1.0e-9 * np.ones(shape)

        rate_density, total_rate = pb11_yield_rate(n_p, n_B, Ti_K, cell_volumes)
        assert total_rate == pytest.approx(0.0, abs=1e-10)
        assert np.all(rate_density == 0.0)

    def test_positive_yield_at_high_temperature(self):
        """Yield rate is positive when Ti >> 100 keV."""
        shape = (3, 3)
        Ti_K = (300.0e3 * eV / k_B) * np.ones(shape)  # 300 keV
        n_p = 1.0e24 * np.ones(shape)
        n_B = 1.0e23 * np.ones(shape)
        cell_volumes = 1.0e-6 * np.ones(shape)

        rate_density, total_rate = pb11_yield_rate(n_p, n_B, Ti_K, cell_volumes)
        assert total_rate > 0.0
        assert np.all(rate_density >= 0.0)

    def test_integral_matches_analytical_uniform_field(self):
        """For uniform n_p, n_B, Ti over a grid, total_rate = n_p*n_B*sv*V_total.

        This tests that the summation correctly implements the integral.
        """
        nx, ny = 5, 6
        T_keV = 300.0
        Ti_K = (T_keV * 1.0e3 * eV / k_B) * np.ones((nx, ny))
        n_p = 2.0e24 * np.ones((nx, ny))
        n_B = 5.0e22 * np.ones((nx, ny))
        dV = 1.0e-7  # m^3 per cell
        cell_volumes = dV * np.ones((nx, ny))

        rate_density, total_rate = pb11_yield_rate(n_p, n_B, Ti_K, cell_volumes)

        sv = pb11_reactivity(T_keV)
        expected_density = n_p[0, 0] * n_B[0, 0] * sv
        expected_total = expected_density * dV * nx * ny

        np.testing.assert_allclose(
            rate_density,
            expected_density * np.ones((nx, ny)),
            rtol=1e-10,
        )
        assert total_rate == pytest.approx(expected_total, rel=1e-10)

    def test_no_half_factor_distinct_species(self):
        """p-B11 rate uses n_p * n_B (no 1/2), unlike identical-particle DD.

        Verified by comparing rate_density to n_p * n_B * sv directly.
        """
        Ti_K = (400.0e3 * eV / k_B) * np.ones((2, 2))
        n_p = np.array([[1.0e24, 2.0e24], [3.0e24, 4.0e24]])
        n_B = np.array([[1.0e22, 2.0e22], [3.0e22, 4.0e22]])
        cell_volumes = 1.0  # scalar

        rate_density, _ = pb11_yield_rate(n_p, n_B, Ti_K, cell_volumes)

        T_keV = 400.0
        sv = pb11_reactivity(T_keV)
        expected = n_p * n_B * sv

        np.testing.assert_allclose(rate_density, expected, rtol=1e-10)

    def test_scalar_cell_volumes(self):
        """Scalar cell_volumes broadcasts correctly across the grid."""
        shape = (3, 3, 3)
        Ti_K = (200.0e3 * eV / k_B) * np.ones(shape)
        n_p = 1.0e24 * np.ones(shape)
        n_B = 1.0e22 * np.ones(shape)
        dV_scalar = 1.0e-8

        rate_density, total_rate_scalar = pb11_yield_rate(n_p, n_B, Ti_K, dV_scalar)

        dV_array = dV_scalar * np.ones(shape)
        _, total_rate_array = pb11_yield_rate(n_p, n_B, Ti_K, dV_array)

        assert total_rate_scalar == pytest.approx(total_rate_array, rel=1e-10)

    def test_zero_density_gives_zero_yield(self):
        """Zero proton density yields zero total rate."""
        shape = (4, 4)
        Ti_K = (300.0e3 * eV / k_B) * np.ones(shape)
        n_p = np.zeros(shape)
        n_B = 1.0e23 * np.ones(shape)
        cell_volumes = 1.0e-6

        _, total_rate = pb11_yield_rate(n_p, n_B, Ti_K, cell_volumes)
        assert total_rate == pytest.approx(0.0, abs=1e-40)

    def test_rate_scales_linearly_with_density(self):
        """Yield rate is linear in both n_p and n_B (bilinear)."""
        shape = (2, 2)
        Ti_K = (400.0e3 * eV / k_B) * np.ones(shape)
        n_B = 1.0e22 * np.ones(shape)
        cell_volumes = 1.0e-7

        n_p_1 = 1.0e24 * np.ones(shape)
        n_p_2 = 2.0e24 * np.ones(shape)

        _, rate_1 = pb11_yield_rate(n_p_1, n_B, Ti_K, cell_volumes)
        _, rate_2 = pb11_yield_rate(n_p_2, n_B, Ti_K, cell_volumes)

        assert rate_2 == pytest.approx(2.0 * rate_1, rel=1e-10)


# ---------------------------------------------------------------------------
# pb11_alpha_power_density
# ---------------------------------------------------------------------------


class TestPb11AlphaPowerDensity:
    """Tests for the alpha-particle power density output."""

    def test_zero_below_cutoff(self):
        """Power density is zero where reactivity is zero (Ti < 10 keV)."""
        Ti_K = (5.0e3 * eV / k_B) * np.ones((3, 3))
        n_p = 1.0e24 * np.ones((3, 3))
        n_B = 1.0e22 * np.ones((3, 3))
        power = pb11_alpha_power_density(n_p, n_B, Ti_K)
        assert np.all(power == 0.0)

    def test_positive_at_peak_temperature(self):
        """Power density is positive at peak reactivity temperature."""
        Ti_K = (400.0e3 * eV / k_B) * np.ones((2, 2))
        n_p = 1.0e24 * np.ones((2, 2))
        n_B = 1.0e22 * np.ones((2, 2))
        power = pb11_alpha_power_density(n_p, n_B, Ti_K)
        assert np.all(power > 0.0)

    def test_q_value_scaling(self):
        """Power density = n_p * n_B * sv * Q with Q = 8.7 MeV."""
        Q_J = 8.7e6 * eV
        Ti_K = (300.0e3 * eV / k_B) * np.ones((2,))
        n_p = np.array([1.0e24, 2.0e24])
        n_B = np.array([1.0e22, 1.0e22])

        power = pb11_alpha_power_density(n_p, n_B, Ti_K)
        sv = pb11_reactivity(300.0)
        expected = n_p * n_B * sv * Q_J

        np.testing.assert_allclose(power, expected, rtol=1e-10)
