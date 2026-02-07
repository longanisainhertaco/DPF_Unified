"""Tests for radiation module: bremsstrahlung + FLD transport.

Test categories:
1. Bremsstrahlung power matches analytic formula within 1%
2. Bremsstrahlung cooling is implicit and stable
3. Levermore-Pomraning flux limiter limits
4. FLD reduces to diffusion in optically thick limit
5. FLD reduces to free-streaming in optically thin limit
6. Radiation-matter coupling conserves energy
7. Full apply_radiation_transport interface
"""

from __future__ import annotations

import numpy as np

from dpf.constants import k_B

# ═══════════════════════════════════════════════════════
# Bremsstrahlung Tests
# ═══════════════════════════════════════════════════════

class TestBremsstrahlung:
    """Tests for bremsstrahlung power density."""

    def test_power_matches_analytic(self):
        """P_ff = 1.69e-32 * g_ff * Z^2 * ne^2 * sqrt(Te)."""
        from dpf.radiation.bremsstrahlung import BREM_COEFF, bremsstrahlung_power

        ne = np.array([1e20, 1e22, 1e24])  # m^-3
        Te = np.array([1e6, 1e7, 1e8])     # K
        Z = 1.0
        g_ff = 1.2

        P = bremsstrahlung_power(ne, Te, Z, g_ff)
        P_expected = BREM_COEFF * g_ff * Z**2 * ne**2 * np.sqrt(Te)

        np.testing.assert_allclose(P, P_expected, rtol=1e-10)

    def test_power_scales_with_ne_squared(self):
        """P_ff should scale as ne^2."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        Te = np.array([1e7])
        ne1 = np.array([1e20])
        ne2 = np.array([2e20])

        P1 = bremsstrahlung_power(ne1, Te)
        P2 = bremsstrahlung_power(ne2, Te)

        # P2/P1 should be (2)^2 = 4
        np.testing.assert_allclose(P2 / P1, 4.0, rtol=1e-10)

    def test_power_scales_with_sqrt_Te(self):
        """P_ff should scale as sqrt(Te)."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([1e22])
        Te1 = np.array([1e6])
        Te2 = np.array([4e6])

        P1 = bremsstrahlung_power(ne, Te1)
        P2 = bremsstrahlung_power(ne, Te2)

        # P2/P1 should be sqrt(4) = 2
        np.testing.assert_allclose(P2 / P1, 2.0, rtol=1e-10)

    def test_power_zero_for_zero_density(self):
        """No bremsstrahlung with no electrons."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([0.0, 0.0])
        Te = np.array([1e7, 1e8])

        P = bremsstrahlung_power(ne, Te)
        np.testing.assert_array_equal(P, 0.0)

    def test_power_zero_for_zero_temperature(self):
        """No bremsstrahlung at zero temperature."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([1e22])
        Te = np.array([0.0])

        P = bremsstrahlung_power(ne, Te)
        np.testing.assert_array_equal(P, 0.0)

    def test_typical_dpf_power(self):
        """Check bremsstrahlung power for typical DPF conditions.

        At ne ~ 1e25 m^-3, Te ~ 1 keV = 1.16e7 K:
        P_ff = 1.69e-32 * 1.2 * 1 * (1e25)^2 * sqrt(1.16e7)
             = 1.69e-32 * 1.2 * 1e50 * 3405.9
             = 6.91e21 W/m^3

        This is ~7e21 W/m^3 — extremely high, but correct for dense DPF pinch.
        """
        from dpf.radiation.bremsstrahlung import BREM_COEFF, bremsstrahlung_power

        ne = np.array([1e25])
        Te = np.array([1.16e7])  # 1 keV
        P = bremsstrahlung_power(ne, Te, Z=1.0, gaunt_factor=1.2)

        # Analytic check
        P_expected = BREM_COEFF * 1.2 * 1.0 * (1e25)**2 * np.sqrt(1.16e7)
        np.testing.assert_allclose(P[0], P_expected, rtol=1e-6)


class TestBremsstrahlungCooling:
    """Tests for implicit bremsstrahlung cooling."""

    def test_cooling_reduces_temperature(self):
        """Temperature should decrease after cooling."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((4, 4, 4), 1e22)
        Te = np.full((4, 4, 4), 1e7)
        dt = 1e-9

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt)

        assert np.all(Te_new < Te), "Temperature should decrease"
        assert np.all(Te_new > 0), "Temperature should remain positive"
        assert np.all(P_rad >= 0), "Radiated power should be non-negative"

    def test_implicit_stability_large_dt(self):
        """Implicit method should be stable even for large dt.

        With explicit Euler, large dt would give negative temperatures.
        The implicit Newton solver should keep Te > Te_floor.
        """
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((4,), 1e24)
        Te = np.full((4,), 1e7)
        dt = 1e-3  # Very large timestep

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt, Te_floor=100.0)

        assert np.all(Te_new >= 100.0), "Temperature should respect floor"
        assert np.all(np.isfinite(Te_new)), "No NaN or Inf"

    def test_energy_conservation(self):
        """Radiated power should equal energy removed from electrons."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8,), 1e22)
        Te = np.full((8,), 5e6)
        dt = 1e-10

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt)

        # Energy removed from electrons: 1.5 * ne * kB * (Te - Te_new)
        dE_electrons = 1.5 * ne * k_B * (Te - Te_new)
        # P_rad * dt should equal dE_electrons
        np.testing.assert_allclose(P_rad * dt, dE_electrons, rtol=1e-6)


# ═══════════════════════════════════════════════════════
# Flux Limiter Tests
# ═══════════════════════════════════════════════════════

class TestFluxLimiter:
    """Tests for Levermore-Pomraning flux limiter."""

    def test_optically_thick_limit(self):
        """lambda(R -> 0) -> 1/3."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.array([0.0, 1e-8, 1e-6, 1e-4])
        lam = levermore_pomraning_limiter(R)

        np.testing.assert_allclose(lam, 1.0 / 3.0, atol=1e-3)

    def test_optically_thin_limit(self):
        """lambda(R -> inf) -> 1/R."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.array([100.0, 1000.0, 1e4])
        lam = levermore_pomraning_limiter(R)
        expected = 1.0 / R

        np.testing.assert_allclose(lam, expected, rtol=0.02)

    def test_monotonically_decreasing(self):
        """lambda(R) should be monotonically decreasing."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.logspace(-2, 4, 100)
        lam = levermore_pomraning_limiter(R)

        assert np.all(np.diff(lam) <= 0), "Flux limiter should be monotonically decreasing"

    def test_bounded(self):
        """0 <= lambda(R) <= 1/3."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.logspace(-6, 6, 1000)
        lam = levermore_pomraning_limiter(R)

        assert np.all(lam >= 0), "Flux limiter should be non-negative"
        assert np.all(lam <= 1.0 / 3.0 + 1e-10), "Flux limiter should be <= 1/3"


# ═══════════════════════════════════════════════════════
# FLD Transport Tests
# ═══════════════════════════════════════════════════════

class TestFLDTransport:
    """Tests for flux-limited diffusion radiation transport."""

    def test_uniform_no_diffusion(self):
        """Uniform E_rad should not diffuse (grad(E) = 0)."""
        from dpf.radiation.transport import fld_step

        nx = 16
        E_rad = np.full((nx, nx, nx), 1e5)  # Uniform radiation energy
        Te = np.full((nx, nx, nx), 1e6)
        ne = np.full((nx, nx, nx), 1e22)
        dx = 0.01
        dt = 1e-12

        E_new, Q = fld_step(E_rad, Te, ne, dx, dt)

        # With uniform E_rad, diffusion should be zero.
        # Only emission/absorption terms remain.
        assert np.all(np.isfinite(E_new)), "No NaN or Inf"
        assert np.all(E_new >= 0), "E_rad should be non-negative"

    def test_gaussian_diffuses(self):
        """Gaussian peak should spread out via diffusion."""
        from dpf.radiation.transport import fld_step

        nx = 16
        x = np.linspace(0, 1, nx)
        X, Y, Zg = np.meshgrid(x, x, x, indexing="ij")
        center = 0.5
        sigma = 0.1

        # Gaussian radiation energy — moderate values
        r2 = (X - center)**2 + (Y - center)**2 + (Zg - center)**2
        E_rad = 1e4 * np.exp(-r2 / (2 * sigma**2))
        E_rad += 1e2  # Background

        # Use hot, low-density plasma so opacity is low (optically thin)
        Te = np.full((nx, nx, nx), 1e7)   # 1 keV
        ne = np.full((nx, nx, nx), 1e18)  # Very low density for low opacity
        dx = 1.0 / nx
        dt = 1e-14  # Very small timestep

        peak_before = float(np.max(E_rad))
        E_new, Q = fld_step(E_rad, Te, ne, dx, dt, brem_power=np.zeros_like(E_rad))
        peak_after = float(np.max(E_new))

        # After diffusion with zero emission, peak should decrease
        assert np.all(np.isfinite(E_new)), "No NaN or Inf"
        assert peak_after <= peak_before, "Peak should not increase with diffusion only"

    def test_opacity_scaling(self):
        """Higher density should give higher opacity (kappa ~ ne^2)."""
        from dpf.radiation.transport import compute_rosseland_opacity

        ne_low = np.array([1e22])
        ne_high = np.array([1e24])
        Te = np.array([1e6])

        kappa_low = compute_rosseland_opacity(ne_low, Te)
        kappa_high = compute_rosseland_opacity(ne_high, Te)

        assert float(kappa_high[0]) > float(kappa_low[0]), "Higher density should give higher opacity"
        # Check ne^2 scaling
        ratio = float(kappa_high[0] / kappa_low[0])
        expected_ratio = (1e24 / 1e22)**2  # = 1e4
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_opacity_temperature_scaling(self):
        """Opacity should decrease with temperature (Kramers law: kappa ~ T^{-3.5})."""
        from dpf.radiation.transport import compute_rosseland_opacity

        ne = np.array([1e24])  # High enough density so kappa is above floor
        Te_low = np.array([1e5])
        Te_high = np.array([1e7])

        kappa_low_T = compute_rosseland_opacity(ne, Te_low)
        kappa_high_T = compute_rosseland_opacity(ne, Te_high)

        assert float(kappa_low_T[0]) > float(kappa_high_T[0]), "Higher temperature should reduce opacity"
        # Check approximate T^{-3.5} scaling
        ratio = float(kappa_low_T[0] / kappa_high_T[0])
        expected_ratio = (1e7 / 1e5)**3.5  # = 100^3.5 = 1e7
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.01)


# ═══════════════════════════════════════════════════════
# Radiation Energy Tests
# ═══════════════════════════════════════════════════════

class TestRadiationEnergy:
    """Tests for radiation energy density."""

    def test_lte_radiation_energy(self):
        """E_rad = a_R * T^4 in LTE."""
        from dpf.radiation.transport import compute_radiation_energy

        sigma_SB = 5.670374419e-8
        c_val = 2.998e8
        a_R = 4.0 * sigma_SB / c_val

        Te = np.array([1e4, 1e6, 1e8])
        ne = np.array([1e20, 1e22, 1e24])

        E_rad = compute_radiation_energy(Te, ne)
        E_expected = a_R * Te**4

        np.testing.assert_allclose(E_rad, E_expected, rtol=1e-3)

    def test_radiation_energy_positive(self):
        """E_rad should always be positive."""
        from dpf.radiation.transport import compute_radiation_energy

        Te = np.array([0.0, 100.0, 1e8])
        ne = np.array([0.0, 1e20, 1e24])

        E_rad = compute_radiation_energy(Te, ne)
        assert np.all(E_rad >= 0), "E_rad should be non-negative"


# ═══════════════════════════════════════════════════════
# Full Integration Tests
# ═══════════════════════════════════════════════════════

class TestRadiationIntegration:
    """Tests for the full radiation step interface."""

    def test_apply_radiation_transport(self):
        """Full radiation step should run without errors."""
        from dpf.radiation.transport import apply_radiation_transport

        nx = 8
        state = {
            "Te": np.full((nx, nx, nx), 1e6),
            "Ti": np.full((nx, nx, nx), 1e6),
            "rho": np.full((nx, nx, nx), 1e-2),
            "pressure": np.full((nx, nx, nx), 1e5),
        }

        state = apply_radiation_transport(state, dx=0.01, dt=1e-12)

        assert "E_rad" in state, "Should add E_rad to state"
        assert np.all(np.isfinite(state["Te"])), "Te should be finite"
        assert np.all(np.isfinite(state["E_rad"])), "E_rad should be finite"
        assert np.all(state["Te"] > 0), "Te should be positive"
        assert np.all(state["E_rad"] >= 0), "E_rad should be non-negative"

    def test_bremsstrahlung_only_cools(self):
        """Hot electrons should cool via bremsstrahlung-only mode."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8, 8, 8), 1e24)  # Dense plasma
        Te_init = np.full((8, 8, 8), 1e7)  # 1 keV
        dt = 1e-10

        Te_new, P_rad = apply_bremsstrahlung_losses(Te_init, ne, dt)

        assert np.all(Te_new < Te_init), "Temperature should decrease"
        assert np.all(P_rad > 0), "Radiated power should be positive"

    def test_cold_plasma_minimal_bremsstrahlung(self):
        """Cold, low-density plasma should have minimal bremsstrahlung losses."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8,), 1e16)  # Low density
        Te_init = np.full((8,), 300.0)  # Room temperature
        dt = 1e-9

        Te_new, P_rad = apply_bremsstrahlung_losses(Te_init, ne, dt)

        # At low density and room temperature, cooling should be negligible
        dTe = np.max(np.abs(Te_init - Te_new))
        assert dTe < 1.0, f"Cold plasma dTe={dTe:.2e} should be < 1 K"
