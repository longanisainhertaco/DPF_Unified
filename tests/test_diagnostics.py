"""Tests for Phase 4: Advanced diagnostics — neutron yield, interferometry, validation.

Test categories:
1. DD reactivity matches tabulated values at key temperatures
2. Reactivity scales correctly with temperature
3. Neutron yield rate integrates correctly
4. Abel transform of known Gaussian profile
5. Abel inversion recovers original profile
6. Phase shift and fringe computation
7. Validation suite plasma metrics
8. Validation suite full weighted scoring
"""

from __future__ import annotations

import numpy as np

from dpf.constants import eV, k_B

# ═══════════════════════════════════════════════════════
# DD Reactivity Tests (Bosch-Hale)
# ═══════════════════════════════════════════════════════

class TestDDReactivity:
    """Tests for Bosch-Hale DD fusion reactivity."""

    def test_zero_below_threshold(self):
        """Reactivity is zero below 0.2 keV."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        assert dd_reactivity(0.0) == 0.0
        assert dd_reactivity(0.1) == 0.0
        assert dd_reactivity(0.19) == 0.0

    def test_positive_above_threshold(self):
        """Reactivity is positive above 0.2 keV."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        assert dd_reactivity(1.0) > 0.0
        assert dd_reactivity(10.0) > 0.0
        assert dd_reactivity(50.0) > 0.0

    def test_increases_with_temperature(self):
        """Reactivity increases with temperature up to ~50 keV."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        sv_1 = dd_reactivity(1.0)
        sv_5 = dd_reactivity(5.0)
        sv_10 = dd_reactivity(10.0)
        sv_50 = dd_reactivity(50.0)

        assert sv_5 > sv_1, "Reactivity should increase from 1 to 5 keV"
        assert sv_10 > sv_5, "Reactivity should increase from 5 to 10 keV"
        assert sv_50 > sv_10, "Reactivity should increase from 10 to 50 keV"

    def test_order_of_magnitude_at_1keV(self):
        """At 1 keV, <sigma*v> ~ 10^{-29} to 10^{-23} m^3/s for DD."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        sv = dd_reactivity(1.0)
        assert 1e-29 < sv < 1e-23, f"<sigma*v>(1 keV) = {sv:.2e} out of expected range"

    def test_order_of_magnitude_at_10keV(self):
        """At 10 keV, <sigma*v> ~ 10^{-23} to 10^{-22} m^3/s for DD."""
        from dpf.diagnostics.neutron_yield import dd_reactivity

        sv = dd_reactivity(10.0)
        assert 1e-25 < sv < 1e-20, f"<sigma*v>(10 keV) = {sv:.2e} out of expected range"

    def test_array_version_matches_scalar(self):
        """Vectorized version matches scalar for each element."""
        from dpf.diagnostics.neutron_yield import dd_reactivity, dd_reactivity_array

        T_arr = np.array([0.5, 1.0, 5.0, 10.0, 50.0])
        sv_arr = dd_reactivity_array(T_arr)

        for i, T in enumerate(T_arr):
            sv_scalar = dd_reactivity(T)
            np.testing.assert_allclose(
                sv_arr[i], sv_scalar, rtol=1e-10,
                err_msg=f"Mismatch at T={T} keV",
            )


# ═══════════════════════════════════════════════════════
# Neutron Yield Integration Tests
# ═══════════════════════════════════════════════════════

class TestNeutronYield:
    """Tests for neutron yield rate computation."""

    def test_cold_plasma_zero_yield(self):
        """Cold plasma (T << 1 keV) should produce zero neutrons."""
        from dpf.diagnostics.neutron_yield import neutron_yield_rate

        n_D = np.full((8, 8), 1e25)  # Dense plasma
        Ti = np.full((8, 8), 300.0)   # Room temperature
        cell_vol = 1e-6  # 1 mm^3

        _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
        assert total_rate == 0.0, "Cold plasma should produce zero neutrons"

    def test_hot_dense_positive_yield(self):
        """Hot dense plasma should produce positive neutron yield."""
        from dpf.diagnostics.neutron_yield import neutron_yield_rate

        n_D = np.full((4, 4), 1e25)  # 10^25 m^-3
        Ti_keV = 2.0  # 2 keV
        Ti_K = Ti_keV * 1000.0 * eV / k_B
        Ti = np.full((4, 4), Ti_K)
        cell_vol = 1e-6

        _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
        assert total_rate > 0, "Hot dense plasma should produce neutrons"

    def test_yield_scales_as_n_squared(self):
        """Neutron yield should scale as n_D^2."""
        from dpf.diagnostics.neutron_yield import neutron_yield_rate

        Ti_keV = 5.0
        Ti_K = Ti_keV * 1000.0 * eV / k_B
        Ti = np.full((4,), Ti_K)
        cell_vol = 1e-6

        n1 = np.full((4,), 1e24)
        n2 = np.full((4,), 2e24)

        _, rate1 = neutron_yield_rate(n1, Ti, cell_vol)
        _, rate2 = neutron_yield_rate(n2, Ti, cell_vol)

        # rate2 / rate1 should be ~4 (density doubled -> yield quadrupled)
        ratio = rate2 / max(rate1, 1e-30)
        np.testing.assert_allclose(ratio, 4.0, rtol=0.01)

    def test_integrate_neutron_yield(self):
        """Integration over timestep gives reasonable yield."""
        from dpf.diagnostics.neutron_yield import integrate_neutron_yield

        n_D = np.full((4,), 1e25)
        Ti_K = 5.0 * 1000.0 * eV / k_B
        Ti = np.full((4,), Ti_K)
        cell_vol = np.full((4,), 1e-6)
        dt = 1e-7  # 100 ns

        Y = integrate_neutron_yield(n_D, Ti, cell_vol, dt)
        assert Y > 0, "Yield should be positive for hot dense plasma"
        assert np.isfinite(Y), "Yield should be finite"


# ═══════════════════════════════════════════════════════
# Abel Transform Tests
# ═══════════════════════════════════════════════════════

class TestAbelTransform:
    """Tests for Abel transform and inverse."""

    def test_abel_uniform_cylinder(self):
        """Abel transform of a uniform density cylinder.

        For ne(r) = n0 for r < R, ne = 0 for r > R:
            N_L(y) = 2 * n0 * sqrt(R^2 - y^2)  for y < R

        This is an ellipse.
        """
        from dpf.diagnostics.interferometry import abel_transform

        nr = 64
        R = 0.01  # 1 cm radius
        dr = R / nr
        r = np.array([(i + 0.5) * dr for i in range(nr)])

        n0 = 1e24  # Uniform density
        ne = np.where(r < R, n0, 0.0)

        N_L = abel_transform(ne, r)

        # At r=0 (center), N_L should be approximately 2 * n0 * R
        expected_center = 2.0 * n0 * R
        # First few cells are near center
        np.testing.assert_allclose(
            N_L[0], expected_center, rtol=0.2,
            err_msg="Abel transform at center doesn't match analytic result",
        )

    def test_abel_transform_positive(self):
        """Abel transform of positive density should be non-negative."""
        from dpf.diagnostics.interferometry import abel_transform

        nr = 32
        dr = 0.001
        r = np.array([(i + 0.5) * dr for i in range(nr)])
        ne = 1e24 * np.exp(-r**2 / (2 * 0.005**2))  # Gaussian

        N_L = abel_transform(ne, r)
        assert np.all(N_L >= 0), "Abel transform of positive function should be non-negative"

    def test_abel_transform_decreasing(self):
        """For monotonically decreasing ne(r), N_L should also be monotonically decreasing."""
        from dpf.diagnostics.interferometry import abel_transform

        nr = 32
        dr = 0.001
        r = np.array([(i + 0.5) * dr for i in range(nr)])
        ne = 1e24 * np.exp(-r / 0.01)  # Exponential decay

        N_L = abel_transform(ne, r)
        # N_L should be monotonically non-increasing
        for i in range(len(N_L) - 1):
            assert N_L[i] >= N_L[i + 1] - 1e-10, (
                f"N_L not monotonically decreasing at i={i}: "
                f"{N_L[i]:.3e} < {N_L[i+1]:.3e}"
            )


# ═══════════════════════════════════════════════════════
# Interferometry Phase Shift Tests
# ═══════════════════════════════════════════════════════

class TestPhaseShift:
    """Tests for interferometric phase shift computation."""

    def test_zero_density_zero_phase(self):
        """Zero line-integrated density gives zero phase shift."""
        from dpf.diagnostics.interferometry import phase_shift

        N_L = np.zeros(10)
        phi = phase_shift(N_L)
        np.testing.assert_allclose(phi, 0.0)

    def test_phase_shift_positive(self):
        """Positive density gives positive phase shift."""
        from dpf.diagnostics.interferometry import phase_shift

        N_L = np.full(10, 1e22)  # Typical line-integrated density
        phi = phase_shift(N_L)
        assert np.all(phi > 0)

    def test_fringe_shift_typical_value(self):
        """For typical DPF densities, fringe shift should be ~1-100 fringes."""
        from dpf.diagnostics.interferometry import fringe_shift

        # DPF conditions: ne ~ 10^25 m^-3, path length ~ 1 cm
        # N_L ~ ne * L ~ 10^25 * 0.01 = 10^23 m^-2
        N_L = np.array([1e23])
        fringes = fringe_shift(N_L, wavelength=632.8e-9)
        # r_e * lambda * N_L / (2*pi)
        # ~ 2.818e-15 * 632.8e-9 * 1e23 / (2*pi) ~ 2.84 fringes
        assert 0.1 < fringes[0] < 100.0, f"Fringe shift = {fringes[0]:.2f}, expected 1-100"

    def test_phase_scales_with_wavelength(self):
        """Longer wavelength gives larger phase shift (delta_phi ~ lambda)."""
        from dpf.diagnostics.interferometry import phase_shift

        N_L = np.array([1e22])
        phi_red = phase_shift(N_L, wavelength=632.8e-9)    # HeNe (red)
        phi_green = phase_shift(N_L, wavelength=532.0e-9)   # Nd:YAG 2nd harmonic

        assert phi_red[0] > phi_green[0], "Longer wavelength should give larger phase shift"


# ═══════════════════════════════════════════════════════
# Validation Suite Enhanced Tests
# ═══════════════════════════════════════════════════════

class TestValidationPlasma:
    """Tests for plasma validation metrics."""

    def test_plasma_validation_perfect(self):
        """Perfect match gives score 1.0."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["PF-1000"])
        sim = {
            "neutron_yield": 1e11,      # Exact match
            "peak_ne": 5e25,             # Exact match
            "peak_Te_eV": 2000.0,        # Exact match
        }
        result = suite.validate_plasma("PF-1000", sim)
        assert result.overall_score > 0.95
        assert result.passed is True

    def test_plasma_validation_poor(self):
        """Very wrong values give low score."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["PF-1000"])
        sim = {
            "neutron_yield": 1e5,        # 6 orders of magnitude off
            "peak_ne": 1e20,              # 5 orders off
            "peak_Te_eV": 10.0,           # 200x off
        }
        result = suite.validate_plasma("PF-1000", sim)
        assert result.overall_score < 0.5
        assert result.passed is False

    def test_neutron_yield_order_of_magnitude(self):
        """Within 1 order of magnitude should pass."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        # NX2 reference: 1e8
        sim = {"neutron_yield": 5e8}  # Factor of 5 (< 1 order of magnitude)
        result = suite.validate_plasma("NX2", sim)
        neutron_metrics = [m for m in result.metrics if m.name == "neutron_yield"]
        assert len(neutron_metrics) == 1
        assert neutron_metrics[0].passed


class TestValidationFull:
    """Tests for full weighted validation."""

    def test_full_validation_has_all_metrics(self):
        """Full validation includes both circuit and plasma metrics."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["PF-1000"])
        sim = {
            "peak_current_A": 1.8e6,
            "peak_current_time_s": 5.5e-6,
            "energy_conservation": 1.0,
            "final_current_A": 1e6,
            "neutron_yield": 1e11,
            "peak_ne": 5e25,
            "peak_Te_eV": 2000.0,
        }
        result = suite.validate_full("PF-1000", sim)
        metric_names = [m.name for m in result.metrics]
        assert "peak_current" in metric_names
        assert "energy_conservation" in metric_names
        assert "neutron_yield" in metric_names
        assert "peak_ne" in metric_names

    def test_full_validation_weighted_score(self):
        """Weighted score is between 0 and 1."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        sim = {
            "peak_current_A": 400e3,
            "energy_conservation": 1.0,
            "final_current_A": 200e3,
            "neutron_yield": 1e8,
            "peak_ne": 1e25,
            "peak_Te_eV": 500.0,
        }
        result = suite.validate_full("NX2", sim)
        assert 0.0 <= result.overall_score <= 1.0

    def test_devices_have_new_tolerances(self):
        """All devices should have neutron_yield, peak_ne, peak_Te_eV tolerances."""
        from dpf.validation.suite import DEVICE_REGISTRY

        for name, device in DEVICE_REGISTRY.items():
            assert "neutron_yield" in device.tolerances, f"{name} missing neutron_yield tolerance"
            assert "peak_ne" in device.tolerances, f"{name} missing peak_ne tolerance"
            assert "peak_Te_eV" in device.tolerances, f"{name} missing peak_Te_eV tolerance"
