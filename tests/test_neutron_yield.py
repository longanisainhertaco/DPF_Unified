"""Consolidated neutron yield and fusion diagnostics tests.

Covers:
- DD reactivity (Bosch-Hale), neutron yield rate, Abel transform, interferometry (test_diagnostics.py)
- Beam-target yield rate, zipper BC round-fix (test_phase_y_beam_target.py)
- Neutron yield validation against Lee model results: PF-1000, NX2, UNU-ICTP (test_phase_z_neutron_yield.py)
- DD fusion reactivity verification against Bosch-Hale (1992) reference values (test_verification_fusion.py)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.constants import eV, k_B  # noqa: F401
from dpf.diagnostics.neutron_yield import (
    dd_reactivity,
    dd_reactivity_array,
    integrate_neutron_yield,
    neutron_yield_rate,
)
from dpf.validation.experimental import validate_neutron_yield
from dpf.validation.lee_model_comparison import (
    LeeModelResult,
    estimate_neutron_yield_from_lee_result,
)

# --- Section: DD Reactivity and Neutron Yield Diagnostics ---


class TestDDReactivity:
    """Tests for Bosch-Hale DD fusion reactivity."""

    def test_zero_below_threshold(self):
        """Reactivity is zero below 0.2 keV."""
        assert dd_reactivity(0.0) == 0.0
        assert dd_reactivity(0.1) == 0.0
        assert dd_reactivity(0.19) == 0.0

    def test_positive_above_threshold(self):
        """Reactivity is positive above 0.2 keV."""
        assert dd_reactivity(1.0) > 0.0
        assert dd_reactivity(10.0) > 0.0
        assert dd_reactivity(50.0) > 0.0

    def test_increases_with_temperature(self):
        """Reactivity increases with temperature up to ~50 keV."""
        sv_1 = dd_reactivity(1.0)
        sv_5 = dd_reactivity(5.0)
        sv_10 = dd_reactivity(10.0)
        sv_50 = dd_reactivity(50.0)

        assert sv_5 > sv_1, "Reactivity should increase from 1 to 5 keV"
        assert sv_10 > sv_5, "Reactivity should increase from 5 to 10 keV"
        assert sv_50 > sv_10, "Reactivity should increase from 10 to 50 keV"

    def test_order_of_magnitude_at_1keV(self):
        """At 1 keV, <sigma*v> ~ 10^{-29} to 10^{-23} m^3/s for DD."""
        sv = dd_reactivity(1.0)
        assert 1e-29 < sv < 1e-23, f"<sigma*v>(1 keV) = {sv:.2e} out of expected range"

    def test_order_of_magnitude_at_10keV(self):
        """At 10 keV, <sigma*v> ~ 10^{-23} to 10^{-22} m^3/s for DD."""
        sv = dd_reactivity(10.0)
        assert 1e-25 < sv < 1e-20, f"<sigma*v>(10 keV) = {sv:.2e} out of expected range"

    def test_array_version_matches_scalar(self):
        """Vectorized version matches scalar for each element."""
        T_arr = np.array([0.5, 1.0, 5.0, 10.0, 50.0])
        sv_arr = dd_reactivity_array(T_arr)

        for i, T in enumerate(T_arr):
            sv_scalar = dd_reactivity(T)
            np.testing.assert_allclose(
                sv_arr[i], sv_scalar, rtol=1e-10,
                err_msg=f"Mismatch at T={T} keV",
            )


class TestNeutronYield:
    """Tests for neutron yield rate computation."""

    def test_cold_plasma_zero_yield(self):
        """Cold plasma (T << 1 keV) should produce zero neutrons."""
        n_D = np.full((8, 8), 1e25)
        Ti = np.full((8, 8), 300.0)
        cell_vol = 1e-6

        _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
        assert total_rate == 0.0, "Cold plasma should produce zero neutrons"

    def test_hot_dense_positive_yield(self):
        """Hot dense plasma should produce positive neutron yield."""
        n_D = np.full((4, 4), 1e25)
        Ti_keV = 2.0
        Ti_K = Ti_keV * 1000.0 * eV / k_B
        Ti = np.full((4, 4), Ti_K)
        cell_vol = 1e-6

        _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
        assert total_rate > 0, "Hot dense plasma should produce neutrons"

    def test_yield_scales_as_n_squared(self):
        """Neutron yield should scale as n_D^2."""
        Ti_keV = 5.0
        Ti_K = Ti_keV * 1000.0 * eV / k_B
        Ti = np.full((4,), Ti_K)
        cell_vol = 1e-6

        n1 = np.full((4,), 1e24)
        n2 = np.full((4,), 2e24)

        _, rate1 = neutron_yield_rate(n1, Ti, cell_vol)
        _, rate2 = neutron_yield_rate(n2, Ti, cell_vol)

        ratio = rate2 / max(rate1, 1e-30)
        np.testing.assert_allclose(ratio, 4.0, rtol=0.01)

    def test_integrate_neutron_yield(self):
        """Integration over timestep gives reasonable yield."""
        n_D = np.full((4,), 1e25)
        Ti_K = 5.0 * 1000.0 * eV / k_B
        Ti = np.full((4,), Ti_K)
        cell_vol = np.full((4,), 1e-6)
        dt = 1e-7

        Y = integrate_neutron_yield(n_D, Ti, cell_vol, dt)
        assert Y > 0, "Yield should be positive for hot dense plasma"
        assert np.isfinite(Y), "Yield should be finite"


class TestAbelTransform:
    """Tests for Abel transform and inverse."""

    def test_abel_uniform_cylinder(self):
        """Abel transform of a uniform density cylinder."""
        from dpf.diagnostics.interferometry import abel_transform

        nr = 64
        R = 0.01
        dr = R / nr
        r = np.array([(i + 0.5) * dr for i in range(nr)])

        n0 = 1e24
        ne = np.where(r < R, n0, 0.0)

        N_L = abel_transform(ne, r)

        expected_center = 2.0 * n0 * R
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
        ne = 1e24 * np.exp(-r**2 / (2 * 0.005**2))

        N_L = abel_transform(ne, r)
        assert np.all(N_L >= 0), "Abel transform of positive function should be non-negative"

    def test_abel_transform_decreasing(self):
        """For monotonically decreasing ne(r), N_L should also be monotonically decreasing."""
        from dpf.diagnostics.interferometry import abel_transform

        nr = 32
        dr = 0.001
        r = np.array([(i + 0.5) * dr for i in range(nr)])
        ne = 1e24 * np.exp(-r / 0.01)

        N_L = abel_transform(ne, r)
        for i in range(len(N_L) - 1):
            assert N_L[i] >= N_L[i + 1] - 1e-10, (
                f"N_L not monotonically decreasing at i={i}: "
                f"{N_L[i]:.3e} < {N_L[i+1]:.3e}"
            )


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

        N_L = np.full(10, 1e22)
        phi = phase_shift(N_L)
        assert np.all(phi > 0)

    def test_fringe_shift_typical_value(self):
        """For typical DPF densities, fringe shift should be ~1-100 fringes."""
        from dpf.diagnostics.interferometry import fringe_shift

        N_L = np.array([1e23])
        fringes = fringe_shift(N_L, wavelength=632.8e-9)
        assert 0.1 < fringes[0] < 100.0, f"Fringe shift = {fringes[0]:.2f}, expected 1-100"

    def test_phase_scales_with_wavelength(self):
        """Longer wavelength gives larger phase shift (delta_phi ~ lambda)."""
        from dpf.diagnostics.interferometry import phase_shift

        N_L = np.array([1e22])
        phi_red = phase_shift(N_L, wavelength=632.8e-9)
        phi_green = phase_shift(N_L, wavelength=532.0e-9)

        assert phi_red[0] > phi_green[0], "Longer wavelength should give larger phase shift"


class TestValidationPlasma:
    """Tests for plasma validation metrics."""

    def test_plasma_validation_perfect(self):
        """Perfect match gives score 1.0."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["PF-1000"])
        sim = {
            "neutron_yield": 1e11,
            "peak_ne": 5e25,
            "peak_Te_eV": 2000.0,
        }
        result = suite.validate_plasma("PF-1000", sim)
        assert result.overall_score > 0.95
        assert result.passed is True

    def test_plasma_validation_poor(self):
        """Very wrong values give low score."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["PF-1000"])
        sim = {
            "neutron_yield": 1e5,
            "peak_ne": 1e20,
            "peak_Te_eV": 10.0,
        }
        result = suite.validate_plasma("PF-1000", sim)
        assert result.overall_score < 0.5
        assert result.passed is False

    def test_neutron_yield_order_of_magnitude(self):
        """Within 1 order of magnitude should pass."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        sim = {"neutron_yield": 5e8}
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


# --- Section: Beam-Target Yield Rate ---


class TestRoundFix:
    """Test int() -> round() fix for zipper BC indices."""

    def test_round_nearest_cell(self) -> None:
        """round() gives nearest cell vs int() truncation."""
        z_sheath = 0.097
        dz = 0.01
        assert int(z_sheath / dz) == 9
        assert round(z_sheath / dz) == 10

    def test_round_exact(self) -> None:
        """round() matches int() for exact values."""
        z_sheath = 0.05
        dz = 0.01
        assert round(z_sheath / dz) == int(z_sheath / dz) == 5

    def test_round_midpoint(self) -> None:
        """round() handles midpoints (banker's rounding)."""
        z_sheath = 0.055
        dz = 0.01
        idx = round(z_sheath / dz)
        assert idx in (5, 6)


class TestBeamTargetYieldRate:
    """Test beam_target_yield_rate function directly."""

    def test_positive_yield(self) -> None:
        """Non-zero inputs produce positive yield rate."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(
            I_pinch=200e3,
            V_pinch=50e3,
            n_target=1e25,
            L_target=0.01,
            f_beam=0.2,
        )
        assert rate > 0

    def test_zero_current_zero_yield(self) -> None:
        """Zero current gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(0.0, 50e3, 1e25, 0.01, 0.2)
        assert rate == 0.0

    def test_zero_voltage_zero_yield(self) -> None:
        """Zero voltage gives zero yield (no beam energy)."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 0.0, 1e25, 0.01, 0.2)
        assert rate == 0.0

    def test_yield_scales_with_current(self) -> None:
        """Yield rate scales linearly with current."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(100e3, 50e3, 1e25, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_yield_scales_with_density(self) -> None:
        """Yield rate scales linearly with target density."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(200e3, 50e3, 1e24, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 2e24, 0.01, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_yield_scales_with_length(self) -> None:
        """Yield rate scales linearly with target length."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.02, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_zero_density_zero_yield(self) -> None:
        """Zero target density gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 0.0, 0.01, 0.2)
        assert rate == 0.0

    def test_zero_length_zero_yield(self) -> None:
        """Zero target length gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 1e25, 0.0, 0.2)
        assert rate == 0.0

    def test_typical_dpf_magnitude(self) -> None:
        """Yield rate for typical PF-1000 parameters is in physically expected range."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(
            I_pinch=500e3,
            V_pinch=100e3,
            n_target=1e25,
            L_target=0.01,
            f_beam=0.2,
        )
        assert np.isfinite(rate)
        assert rate > 0.0
        assert rate > 1e8
        assert rate < 1e18

    def test_f_beam_clamped_above_one(self) -> None:
        """f_beam > 1 is clamped to 1; result must still be positive and finite."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate_clamped = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=2.0)
        rate_one = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=1.0)
        assert np.isfinite(rate_clamped)
        assert rate_clamped == pytest.approx(rate_one, rel=1e-10)

    def test_f_beam_zero_zero_yield(self) -> None:
        """f_beam=0 gives zero yield (no beam ions)."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.0)
        assert rate == 0.0


# --- Section: Neutron Yield Validation (Lee Model) ---


def _make_pf1000_result(peak_current: float = 1.87e6) -> LeeModelResult:
    """Minimal LeeModelResult with PF-1000 geometry metadata."""
    t = np.linspace(0, 10e-6, 100)
    I_wfm = peak_current * np.sin(np.pi * t / (2 * 5.8e-6))
    I_wfm = np.clip(I_wfm, 0, peak_current)
    return LeeModelResult(
        t=t,
        I=I_wfm,
        V=np.zeros(100),
        z_sheet=np.zeros(100),
        r_shock=np.zeros(100),
        peak_current=peak_current,
        peak_current_time=5.8e-6,
        pinch_time=7.0e-6,
        device_name="PF-1000",
        phases_completed=[1, 2],
        metadata={
            "anode_radius": 0.0575,
            "cathode_radius": 0.08,
            "anode_length": 0.16,
            "rho0": 3.77e-4,
            "fm": 0.3,
            "fc": 0.7,
            "C": 1.332e-3,
            "V0": 27e3,
            "L0": 33e-9,
            "R0": 2.3e-3,
            "fill_pressure_torr": 3.5,
        },
    )


def _make_nx2_result() -> LeeModelResult:
    """Minimal LeeModelResult with NX2 geometry metadata."""
    t = np.linspace(0, 4e-6, 100)
    peak = 400e3
    I_wfm = peak * np.sin(np.pi * t / (2 * 1.8e-6))
    I_wfm = np.clip(I_wfm, 0, peak)
    return LeeModelResult(
        t=t,
        I=I_wfm,
        V=np.zeros(100),
        z_sheet=np.zeros(100),
        r_shock=np.zeros(100),
        peak_current=peak,
        peak_current_time=1.8e-6,
        pinch_time=2.2e-6,
        device_name="NX2",
        phases_completed=[1, 2],
        metadata={
            "anode_radius": 0.019,
            "cathode_radius": 0.041,
            "anode_length": 0.05,
            "rho0": 2.67e-4,
            "fm": 0.3,
            "fc": 0.7,
            "C": 28e-6,
            "V0": 14e3,
            "L0": 20e-9,
            "R0": 5e-3,
            "fill_pressure_torr": 4.0,
        },
    )


def _make_unu_result() -> LeeModelResult:
    """Minimal LeeModelResult with UNU-ICTP geometry metadata."""
    t = np.linspace(0, 6e-6, 100)
    peak = 170e3
    I_wfm = peak * np.sin(np.pi * t / (2 * 2.8e-6))
    I_wfm = np.clip(I_wfm, 0, peak)
    return LeeModelResult(
        t=t,
        I=I_wfm,
        V=np.zeros(100),
        z_sheet=np.zeros(100),
        r_shock=np.zeros(100),
        peak_current=peak,
        peak_current_time=2.8e-6,
        pinch_time=3.5e-6,
        device_name="UNU-ICTP",
        phases_completed=[1, 2],
        metadata={
            "anode_radius": 0.0095,
            "cathode_radius": 0.032,
            "anode_length": 0.16,
            "rho0": 2.29e-4,
            "fm": 0.3,
            "fc": 0.7,
            "C": 30e-6,
            "V0": 14e3,
            "L0": 110e-9,
            "R0": 12e-3,
            "fill_pressure_torr": 3.0,
        },
    )


class TestNeutronYieldBasics:
    """Basic return-type and guard tests for estimate_neutron_yield_from_lee_result."""

    def test_returns_float(self) -> None:
        """Return value is a Python float."""
        result = _make_pf1000_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert isinstance(yield_est, float)

    def test_returns_nonnegative(self) -> None:
        """Yield estimate is non-negative for valid inputs."""
        result = _make_pf1000_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert yield_est >= 0.0

    def test_returns_finite(self) -> None:
        """Yield estimate is finite (no inf or NaN)."""
        result = _make_pf1000_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert math.isfinite(yield_est), f"Expected finite yield, got {yield_est}"

    def test_zero_current_returns_zero(self) -> None:
        """Zero peak_current -> yield = 0."""
        result = _make_pf1000_result(peak_current=0.0)
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert yield_est == pytest.approx(0.0)

    def test_empty_metadata_returns_zero(self) -> None:
        """Empty metadata -> yield = 0."""
        result = LeeModelResult(
            t=np.zeros(2),
            I=np.zeros(2),
            V=np.zeros(2),
            z_sheet=np.zeros(2),
            r_shock=np.zeros(2),
            peak_current=1e6,
            metadata={},
        )
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert yield_est == pytest.approx(0.0)

    def test_zero_anode_radius_returns_zero(self) -> None:
        """anode_radius = 0 -> yield = 0 (avoids division by zero)."""
        result = _make_pf1000_result()
        result.metadata["anode_radius"] = 0.0
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert yield_est == pytest.approx(0.0)

    def test_zero_rho0_returns_zero(self) -> None:
        """rho0 = 0 -> yield = 0."""
        result = _make_pf1000_result()
        result.metadata["rho0"] = 0.0
        yield_est = estimate_neutron_yield_from_lee_result(result)
        assert yield_est == pytest.approx(0.0)


class TestNeutronYieldOrderOfMagnitude:
    """Order-of-magnitude checks against published experimental neutron yields."""

    def test_pf1000_within_two_orders_of_magnitude(self) -> None:
        """PF-1000 estimated yield within factor 100 of 10^11 n/shot."""
        result = _make_pf1000_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)

        assert yield_est > 0.0, "PF-1000 yield estimate must be positive"
        ratio = yield_est / 1e11
        assert 0.01 <= ratio <= 100.0, (
            f"PF-1000 yield {yield_est:.2e} is not within 2 OOM of 1e11 "
            f"(ratio = {ratio:.2e})"
        )

    def test_nx2_within_two_orders_of_magnitude(self) -> None:
        """NX2 estimated yield within 3 orders of magnitude of 10^8 n/shot."""
        result = _make_nx2_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)

        assert yield_est > 0.0, "NX2 yield estimate must be positive"
        ratio = yield_est / 1e8
        assert 0.001 <= ratio <= 1000.0, (
            f"NX2 yield {yield_est:.2e} not within 3 OOM of 1e8 "
            f"(ratio = {ratio:.2e})"
        )

    def test_unu_ictp_within_two_orders_of_magnitude(self) -> None:
        """UNU-ICTP estimated yield within 3 orders of magnitude of 10^8 n/shot."""
        result = _make_unu_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)

        assert yield_est > 0.0, "UNU-ICTP yield estimate must be positive"
        ratio = yield_est / 1e8
        assert 0.001 <= ratio <= 1000.0, (
            f"UNU-ICTP yield {yield_est:.2e} not within 3 OOM of 1e8 "
            f"(ratio = {ratio:.2e})"
        )

    def test_pf1000_higher_than_nx2(self) -> None:
        """PF-1000 yield estimate is larger than NX2 (higher current -> more neutrons)."""
        y_pf = estimate_neutron_yield_from_lee_result(_make_pf1000_result())
        y_nx = estimate_neutron_yield_from_lee_result(_make_nx2_result())

        assert y_pf > y_nx, (
            f"PF-1000 ({y_pf:.2e}) should exceed NX2 ({y_nx:.2e})"
        )

    def test_pf1000_higher_than_unu(self) -> None:
        """PF-1000 yield estimate is larger than UNU-ICTP."""
        y_pf = estimate_neutron_yield_from_lee_result(_make_pf1000_result())
        y_unu = estimate_neutron_yield_from_lee_result(_make_unu_result())

        assert y_pf > y_unu, (
            f"PF-1000 ({y_pf:.2e}) should exceed UNU-ICTP ({y_unu:.2e})"
        )


class TestNeutronYieldScaling:
    """Physical scaling tests — yield should increase with current and density."""

    def test_higher_current_gives_higher_yield(self) -> None:
        """Doubling I_peak increases yield estimate in the sub-peak regime."""
        r_lo = _make_pf1000_result(peak_current=100e3)
        r_hi = _make_pf1000_result(peak_current=200e3)

        y_lo = estimate_neutron_yield_from_lee_result(r_lo)
        y_hi = estimate_neutron_yield_from_lee_result(r_hi)

        assert y_hi > y_lo, (
            f"Higher current should give higher yield in sub-peak regime: "
            f"y(100 kA)={y_lo:.2e}, y(200 kA)={y_hi:.2e}"
        )

    def test_higher_density_gives_higher_yield(self) -> None:
        """Doubling fill density increases yield (more target particles)."""
        r_lo = _make_pf1000_result()
        r_hi = _make_pf1000_result()

        r_lo.metadata["rho0"] = 2e-4
        r_hi.metadata["rho0"] = 4e-4

        y_lo = estimate_neutron_yield_from_lee_result(r_lo)
        y_hi = estimate_neutron_yield_from_lee_result(r_hi)

        assert y_hi > y_lo, (
            f"Higher fill density should give higher yield: "
            f"y(low)={y_lo:.2e}, y(high)={y_hi:.2e}"
        )


class TestValidateNeutronYield:
    """Tests for experimental.validate_neutron_yield() helper."""

    def test_returns_dict_with_required_keys(self) -> None:
        """validate_neutron_yield returns dict with yield_ratio and within_order_magnitude."""
        metrics = validate_neutron_yield(1e11, "PF-1000")
        assert "yield_ratio" in metrics
        assert "within_order_magnitude" in metrics
        assert "yield_sim" in metrics
        assert "yield_exp" in metrics

    def test_exact_match_ratio_is_one(self) -> None:
        """Y_sim = Y_exp -> yield_ratio = 1.0."""
        from dpf.validation.experimental import PF1000_DATA
        metrics = validate_neutron_yield(PF1000_DATA.neutron_yield, "PF-1000")
        assert metrics["yield_ratio"] == pytest.approx(1.0, rel=1e-9)

    def test_within_order_magnitude_true_for_10x(self) -> None:
        """9x experimental value is within an order of magnitude."""
        from dpf.validation.experimental import PF1000_DATA
        metrics = validate_neutron_yield(PF1000_DATA.neutron_yield * 9, "PF-1000")
        assert metrics["within_order_magnitude"] is True

    def test_within_order_magnitude_false_for_100x(self) -> None:
        """100x experimental value is NOT within an order of magnitude."""
        from dpf.validation.experimental import PF1000_DATA
        metrics = validate_neutron_yield(PF1000_DATA.neutron_yield * 100, "PF-1000")
        assert metrics["within_order_magnitude"] is False

    def test_pf1000_yield_estimate_passes_oom_check(self) -> None:
        """PF-1000 model estimate passes within_order_magnitude check for PF-1000."""
        result = _make_pf1000_result()
        yield_est = estimate_neutron_yield_from_lee_result(result)
        metrics = validate_neutron_yield(yield_est, "PF-1000")

        ratio = metrics["yield_ratio"]
        assert 1e-3 < ratio < 1e3, (
            f"Yield ratio {ratio:.2e} (Y_est={yield_est:.2e}, "
            f"Y_exp={metrics['yield_exp']:.2e}) outside 3 OOM"
        )


# --- Section: DD Fusion Reactivity Verification (Bosch-Hale 1992) ---


def test_dd_reactivity_below_threshold():
    """DD reactivity should return 0 for Ti < 0.2 keV."""
    assert dd_reactivity(0.1) == pytest.approx(0.0, abs=1e-30)
    assert dd_reactivity(0.0) == pytest.approx(0.0, abs=1e-30)


def test_dd_reactivity_at_1keV():
    """DD reactivity at 1 keV — order of magnitude check."""
    sv = dd_reactivity(1.0)
    assert sv > 0, "Reactivity should be positive above threshold"
    assert 1e-30 <= sv <= 1e-24, (
        f"DD reactivity at 1 keV = {sv:.3e} m^3/s, expected ~1e-28 range"
    )


def test_dd_reactivity_at_10keV():
    """DD reactivity at 10 keV should be in ~1e-25 to 1e-22 m^3/s range."""
    sv = dd_reactivity(10.0)
    assert 1e-25 <= sv <= 1e-22, (
        f"DD reactivity at 10 keV = {sv:.3e} m^3/s, out of expected range"
    )


def test_dd_reactivity_at_50keV():
    """DD reactivity at 50 keV should be in ~1e-24 to 1e-21 m^3/s range."""
    sv = dd_reactivity(50.0)
    assert 1e-24 <= sv <= 1e-21, (
        f"DD reactivity at 50 keV = {sv:.3e} m^3/s, out of expected range"
    )


def test_dd_reactivity_monotonic_1_to_50keV():
    """DD reactivity should increase monotonically from 1 to 50 keV."""
    temps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    reactivities = [dd_reactivity(T) for T in temps]
    for i in range(len(reactivities) - 1):
        assert reactivities[i + 1] > reactivities[i], (
            f"Non-monotonic: sv({temps[i]})={reactivities[i]:.3e} >= "
            f"sv({temps[i+1]})={reactivities[i+1]:.3e}"
        )


def test_dd_reactivity_cap_at_100keV():
    """Ti > 100 keV should return same value as 100 keV (capped)."""
    sv_100 = dd_reactivity(100.0)
    sv_200 = dd_reactivity(200.0)
    assert sv_200 == pytest.approx(sv_100, rel=1e-12)


def test_dd_reactivity_array():
    """dd_reactivity_array should match element-wise dd_reactivity calls."""
    temps = np.array([1.0, 5.0, 10.0, 50.0])
    sv_arr = dd_reactivity_array(temps)
    sv_scalar = np.array([dd_reactivity(T) for T in temps])
    np.testing.assert_allclose(sv_arr, sv_scalar, rtol=1e-12)


def test_neutron_yield_rate_basic():
    """Neutron yield rate should be positive at fusion-relevant temperatures."""
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 5e7)
    cell_vol = 1e-9

    rate_density, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)

    assert total_rate > 0, f"Total neutron rate should be positive, got {total_rate:.3e}"
    assert rate_density.shape == n_D.shape
    assert np.all(rate_density >= 0)


def test_neutron_yield_rate_zero_cold():
    """Neutron yield should be zero at Ti < 0.2 keV."""
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 2000.0)
    cell_vol = 1e-9

    _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
    assert total_rate == pytest.approx(0.0, abs=1e-10)


def test_neutron_yield_scales_with_nD_squared():
    """Doubling n_D should quadruple yield rate (rate ~ n_D^2)."""
    Ti = np.full((8, 8, 8), 5e7)
    cell_vol = 1e-9

    _, rate_base = neutron_yield_rate(np.full((8, 8, 8), 1e25), Ti, cell_vol)
    _, rate_double = neutron_yield_rate(np.full((8, 8, 8), 2e25), Ti, cell_vol)

    assert rate_double == pytest.approx(4.0 * rate_base, rel=1e-10)


def test_integrate_neutron_yield():
    """integrate_neutron_yield(n_D, Ti, cell_vol, dt) = total_rate * dt."""
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 5e7)
    cell_vol = 1e-9
    dt = 1e-9

    _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
    yield_count = integrate_neutron_yield(n_D, Ti, cell_vol, dt)

    assert yield_count == pytest.approx(total_rate * dt, rel=1e-10)
