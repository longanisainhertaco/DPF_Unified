"""Phase AF tests: Post-Jensen Cu cooling, multi-event neutron decomposition,
and regime validity engine integration.

Tests for three research-informed implementations:
1. Tabulated Post-Jensen Cu cooling curves (Post et al. 1977, 21-point log-log)
2. Multi-event neutron decomposition (Goyon et al. 2025)
3. Regime validity diagnostic integration into engine step loop
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import eV, k_B

# ─────────────────────────────────────────────────────────────────────
# 1. Post-Jensen Cu Cooling Curves (Post et al., ADNDT 20:397, 1977)
# ─────────────────────────────────────────────────────────────────────


class TestPostJensenCuCooling:
    """Verify tabulated Cu (Z=29) cooling function against shell physics."""

    def test_peak_at_100eV(self):
        """Global M-shell peak should be near 100 eV."""
        from dpf.radiation.line_radiation import _cooling_copper

        # Scan 10-200 eV in 1 eV steps
        Te_scan = np.arange(10, 201, 1.0)
        vals = np.array([_cooling_copper(T) for T in Te_scan])
        peak_idx = np.argmax(vals)
        peak_Te = Te_scan[peak_idx]
        assert 80 <= peak_Te <= 120, f"M-shell peak at {peak_Te} eV, expected 80-120 eV"

    def test_peak_magnitude(self):
        """M-shell peak Lambda ~ 3e-30 W m^3 (within factor of 2)."""
        from dpf.radiation.line_radiation import _cooling_copper

        peak_val = _cooling_copper(100.0)
        assert 1e-30 < peak_val < 1e-29, f"Peak Lambda = {peak_val:.2e}, expected ~3e-30"

    def test_ar_like_trough(self):
        """Ar-like closed-shell trough at ~1000 eV (Cu18+ to Cu19+ gap)."""
        from dpf.radiation.line_radiation import _cooling_copper

        val_peak = _cooling_copper(100.0)
        val_trough = _cooling_copper(1000.0)
        # Trough should be at least 3x below M-shell peak
        assert val_trough < val_peak / 3.0, (
            f"Trough ({val_trough:.2e}) not deep enough vs peak ({val_peak:.2e})"
        )

    def test_l_shell_secondary_peak(self):
        """L-shell secondary peak at ~2-4 keV should be above trough."""
        from dpf.radiation.line_radiation import _cooling_copper

        val_trough = _cooling_copper(1000.0)
        val_l_peak = _cooling_copper(3000.0)
        assert val_l_peak > val_trough, (
            f"L-shell peak ({val_l_peak:.2e}) not above trough ({val_trough:.2e})"
        )

    def test_monotonic_rise_below_peak(self):
        """Cooling should monotonically increase from 1 eV to ~100 eV."""
        from dpf.radiation.line_radiation import _cooling_copper

        Te_vals = [1, 2, 5, 10, 20, 50, 80, 100]
        lambdas = [_cooling_copper(T) for T in Te_vals]
        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1], (
                f"Non-monotonic: Lambda({Te_vals[i]})={lambdas[i]:.2e} "
                f"< Lambda({Te_vals[i-1]})={lambdas[i-1]:.2e}"
            )

    def test_high_temperature_decline(self):
        """Above L-shell peak, cooling should decline toward 10 keV."""
        from dpf.radiation.line_radiation import _cooling_copper

        val_5keV = _cooling_copper(5000.0)
        val_10keV = _cooling_copper(10000.0)
        assert val_10keV < val_5keV, "Cooling should decline above 5 keV"

    def test_extrapolation_above_10keV(self):
        """Above 10 keV should extrapolate as power law, not crash."""
        from dpf.radiation.line_radiation import _cooling_copper

        val = _cooling_copper(50000.0)
        assert val > 0 and np.isfinite(val), f"Bad extrapolation: {val}"
        # Should be smaller than at 10 keV
        val_10keV = _cooling_copper(10000.0)
        assert val < val_10keV

    def test_below_1eV(self):
        """Below 1 eV should return a floor value, not zero or crash."""
        from dpf.radiation.line_radiation import _cooling_copper

        val = _cooling_copper(0.1)
        assert val > 0 and np.isfinite(val)
        assert val < 1e-33  # Should be negligibly small

    def test_cooling_function_dispatches_to_copper(self):
        """cooling_function(Te, Z=29) should use the new tabulated Cu data."""
        from dpf.radiation.line_radiation import _cooling_copper, cooling_function

        Te_K = 100.0 * eV / k_B  # 100 eV in Kelvin
        val_direct = _cooling_copper(100.0)
        val_dispatch = cooling_function(np.array([Te_K]), 29.0)[0]
        assert pytest.approx(val_direct, rel=1e-6) == val_dispatch

    def test_old_vs_new_much_higher(self):
        """New tabulated values should be much higher than old power-law at peak.

        The old crude fit peaked at ~5e-32 W m^3 (severely underestimating).
        The new tabulated values peak at ~3e-30 W m^3 (60x correction).
        """
        from dpf.radiation.line_radiation import _cooling_copper

        peak_val = _cooling_copper(100.0)
        old_peak_approx = 5.0e-32  # Old piecewise power-law peak
        # New value should be at least 10x the old
        assert peak_val > 10 * old_peak_approx, (
            f"New peak ({peak_val:.2e}) not significantly above old ({old_peak_approx:.2e})"
        )

    def test_interpolation_smoothness(self):
        """Cooling function should be reasonably smooth (no jumps > 3x in 10 eV)."""
        from dpf.radiation.line_radiation import _cooling_copper

        Te_fine = np.linspace(10, 5000, 500)
        vals = np.array([_cooling_copper(T) for T in Te_fine])
        # Check that adjacent values don't jump by more than 3x
        for i in range(1, len(vals)):
            ratio = vals[i] / max(vals[i - 1], 1e-40)
            assert 0.3 < ratio < 3.0, (
                f"Jump at Te={Te_fine[i]:.0f}: ratio={ratio:.2f}"
            )


# ─────────────────────────────────────────────────────────────────────
# 2. Multi-Event Neutron Decomposition (Goyon et al. 2025)
# ─────────────────────────────────────────────────────────────────────


class TestMultiEventNeutronDecomposition:
    """Test decompose_neutron_events from beam_target module."""

    def _make_single_event(self, t_peak_ns=500.0, fwhm_ns=20.0, n_points=1000):
        """Create synthetic single-event Gaussian neutron signal."""
        times = np.linspace(0, 1e-6, n_points)
        t_peak = t_peak_ns * 1e-9
        sigma = fwhm_ns * 1e-9 / 2.355
        rates = 1e15 * np.exp(-0.5 * ((times - t_peak) / sigma) ** 2)
        return times, rates

    def _make_double_event(self):
        """Create synthetic double-event signal (two Gaussian peaks)."""
        times = np.linspace(0, 1e-6, 2000)
        sigma1 = 15e-9 / 2.355
        sigma2 = 10e-9 / 2.355
        rates = (
            1e15 * np.exp(-0.5 * ((times - 400e-9) / sigma1) ** 2)
            + 0.5e15 * np.exp(-0.5 * ((times - 700e-9) / sigma2) ** 2)
        )
        return times, rates

    def test_single_event_detected(self):
        """Single Gaussian peak should produce exactly 1 event."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1

    def test_double_event_detected(self):
        """Two well-separated Gaussians should produce 2 events."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        assert result["n_events"] == 2

    def test_primary_fraction_single_event(self):
        """Single event should have primary_fraction > 0.7 (FWHM captures ~76% of Gaussian)."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["primary_fraction"] > 0.7

    def test_primary_fraction_double_event(self):
        """For double event with 2:1 ratio, primary fraction ~ 0.6-0.7."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        assert 0.5 < result["primary_fraction"] < 0.9

    def test_event_peak_times(self):
        """Peak times should match input Gaussian centers."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_double_event()
        result = decompose_neutron_events(times, rates, min_separation_ns=50.0)
        peak_times_ns = sorted([ev["peak_time"] * 1e9 for ev in result["events"]])
        assert len(peak_times_ns) == 2
        assert abs(peak_times_ns[0] - 400) < 10, f"First peak at {peak_times_ns[0]:.0f} ns"
        assert abs(peak_times_ns[1] - 700) < 10, f"Second peak at {peak_times_ns[1]:.0f} ns"

    def test_event_fwhm(self):
        """FWHM should roughly match input Gaussian width."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_single_event(fwhm_ns=20.0, n_points=2000)
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1
        fwhm = result["events"][0]["fwhm_ns"]
        assert 10 < fwhm < 40, f"FWHM={fwhm:.1f} ns, expected ~20 ns"

    def test_total_yield_positive(self):
        """Total yield should be positive for non-zero signal."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["total_yield"] > 0

    def test_empty_signal(self):
        """Zero signal should produce 0 events."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times = np.linspace(0, 1e-6, 100)
        rates = np.zeros_like(times)
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 0
        assert result["total_yield"] == 0.0

    def test_short_signal(self):
        """Signal with < 3 points should gracefully return 0 events."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        result = decompose_neutron_events(np.array([0, 1e-6]), np.array([1e10, 1e10]))
        assert result["n_events"] == 0

    def test_threshold_filtering(self):
        """High threshold should filter out small secondary events."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_double_event()  # primary 1e15, secondary 0.5e15
        # With 60% threshold, the secondary (50% of primary) should be filtered
        result = decompose_neutron_events(
            times, rates, threshold_fraction=0.6, min_separation_ns=50.0,
        )
        assert result["n_events"] == 1

    def test_close_peaks_merged(self):
        """Peaks closer than min_separation should merge into one event."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times = np.linspace(0, 1e-6, 2000)
        sigma = 5e-9 / 2.355
        # Two peaks 5 ns apart (will merge if min_separation_ns > 5)
        rates = (
            1e15 * np.exp(-0.5 * ((times - 500e-9) / sigma) ** 2)
            + 0.8e15 * np.exp(-0.5 * ((times - 505e-9) / sigma) ** 2)
        )
        result = decompose_neutron_events(times, rates, min_separation_ns=10.0)
        assert result["n_events"] == 1

    def test_event_dict_keys(self):
        """Each event dict should have all required keys."""
        from dpf.diagnostics.beam_target import decompose_neutron_events

        times, rates = self._make_single_event()
        result = decompose_neutron_events(times, rates)
        assert result["n_events"] == 1
        ev = result["events"][0]
        for key in ["peak_time", "peak_rate", "fwhm_ns", "yield_count",
                     "start_time", "end_time"]:
            assert key in ev, f"Missing key: {key}"
        assert ev["peak_rate"] > 0
        assert ev["yield_count"] > 0


# ─────────────────────────────────────────────────────────────────────
# 3. Regime Validity Engine Integration (Phase AE diagnostics → engine)
# ─────────────────────────────────────────────────────────────────────


class TestRegimeValidityEngineIntegration:
    """Test that the engine records regime validity diagnostics."""

    def test_engine_has_regime_result_attr(self):
        """SimulationEngine should have _last_regime_result after construction."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-9, dt_init=1e-11,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        assert hasattr(engine, "_last_regime_result")
        assert engine._last_regime_result is None  # Not yet computed

    @staticmethod
    def _get_engine_class():
        from dpf.engine import SimulationEngine
        return SimulationEngine

    def test_regime_validity_cold_fill(self):
        """Cold fill gas should be MHD-valid (directly test the diagnostic)."""
        from dpf.diagnostics.plasma_regime import regime_validity

        ne = np.ones((8, 8, 8)) * 1e23
        Te = np.ones((8, 8, 8)) * 300.0
        Ti = np.ones((8, 8, 8)) * 300.0
        v = np.ones((8, 8, 8)) * 1e3
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert result["fraction_valid"] > 0.9

    def test_regime_validity_hot_pinch_invalid(self):
        """Hot pinch plasma should partially violate MHD validity."""
        from dpf.diagnostics.plasma_regime import regime_validity

        ne = np.ones((8, 8, 8)) * 1e25
        Te_K = np.ones((8, 8, 8)) * (1e3 * eV / k_B)  # 1 keV
        Ti_K = Te_K.copy()
        v = np.ones((8, 8, 8)) * 1e5
        result = regime_validity(ne, Te_K, Ti_K, v, dx=1e-3)
        # At 1 keV, 1e25 m^-3: ND > 1 (collisionless) → MHD invalid
        assert result["fraction_valid"] < 1.0

    def test_regime_returns_nd_and_rm(self):
        """Result should contain ND, Rm, and fraction_valid."""
        from dpf.diagnostics.plasma_regime import regime_validity

        ne = np.ones(10) * 1e23
        Te = np.ones(10) * 1e4
        Ti = np.ones(10) * 1e4
        v = np.ones(10) * 1e4
        result = regime_validity(ne, Te, Ti, v, dx=1e-3)
        assert "ND" in result
        assert "Rm" in result
        assert "fraction_valid" in result
        assert 0.0 <= result["fraction_valid"] <= 1.0

    def test_engine_step_no_crash_on_regime_check(self):
        """Running 100+ steps should trigger regime validity check without crash."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7, dt_init=1e-11,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        # Run 105 steps to trigger the check at step 100
        for _ in range(105):
            result = engine.step()
            if result.finished:
                break
        # Should have computed regime validity by step 100
        assert engine._last_regime_result is not None or engine.step_count < 100
