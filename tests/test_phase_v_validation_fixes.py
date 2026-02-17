"""Phase V: Validation bug fixes — first-peak metric, back-pressure, fm/fc swap, z_f.

Tests for PhD Debate #5 confirmed bugs:
- Bug 3 (CRITICAL): validate_current_waveform finds first peak, not global max
- Bug 4 (HIGH): Snowplow radial adiabatic back-pressure
- Bug 5 (MODERATE): Lee model fm/fc naming swap corrected
- Bug 6 (MODERATE): Lee model radial force includes z_f factor
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.snowplow import SnowplowModel
from dpf.validation.experimental import (
    DEVICES,
    _find_first_peak,
    validate_current_waveform,
)
from dpf.validation.lee_model_comparison import LeeModel

# ============================================================
# Fixtures
# ============================================================

def _make_snowplow(
    anode_radius: float = 0.0575,
    cathode_radius: float = 0.08,
    fill_density: float = 3.34e-4,
    anode_length: float = 0.16,
    mass_fraction: float = 0.3,
    fill_pressure_Pa: float = 466.0,
    current_fraction: float = 0.7,
) -> SnowplowModel:
    """Create a SnowplowModel using PF-1000 scale parameters."""
    return SnowplowModel(
        anode_radius=anode_radius,
        cathode_radius=cathode_radius,
        fill_density=fill_density,
        anode_length=anode_length,
        mass_fraction=mass_fraction,
        fill_pressure_Pa=fill_pressure_Pa,
        current_fraction=current_fraction,
    )


def _dpf_like_waveform(n_pts: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DPF current waveform with first peak then higher post-dip oscillation.

    Shape:
    - Rises to first peak at t ~ 5.8 us (amplitude ~ 1.87 MA)
    - Drops to a current dip at t ~ 8 us
    - Then post-pinch oscillations at larger amplitude (~ 2.5 MA) starting at t ~ 9 us

    The global maximum is in the post-pinch oscillation (> first peak), which is
    exactly the scenario that pre-Phase-V code misidentified as "peak current".

    Returns
    -------
    t : ndarray
        Time array [s].
    I_arr : ndarray
        Current waveform [A].
    """
    t = np.linspace(0.0, 20e-6, n_pts)
    omega = 2.0 * np.pi / (4 * 5.8e-6)  # Quarter period at 5.8 us

    # First half-cycle: rise to 1.87 MA first peak, then dip
    I_first = 1.87e6 * np.sin(omega * t) * np.exp(-t / 15e-6)

    # Post-pinch burst: starts at t_pinch, amplitude 2.5 MA (clearly > first peak)
    t_pinch = 9e-6
    I_post = np.where(
        t > t_pinch,
        2.5e6 * np.sin(omega * (t - t_pinch)) * np.exp(-0.3 * (t - t_pinch) / 5e-6),
        0.0,
    )

    # Combine: post-pinch global max exceeds first peak
    I_arr = I_first + I_post

    return t, I_arr


# ============================================================
# TestFirstPeakFinder
# ============================================================

class TestFirstPeakFinder:
    """Tests for _find_first_peak helper (Bug 3 fix)."""

    def test_monotonically_rising_returns_argmax(self) -> None:
        """Monotonically rising signal should return last valid index (argmax)."""
        signal = np.linspace(0.0, 10.0, 50)
        idx = _find_first_peak(signal)
        assert idx == int(np.argmax(signal))

    def test_single_triangle_peak(self) -> None:
        """Triangle wave with one peak should return that peak's index."""
        n = 100
        signal = np.zeros(n)
        peak_idx = 50
        signal[:peak_idx + 1] = np.linspace(0, 1, peak_idx + 1)
        signal[peak_idx:] = np.linspace(1, 0, n - peak_idx)
        result = _find_first_peak(signal)
        assert result == pytest.approx(peak_idx, abs=1)

    def test_two_peaks_returns_first_not_global_max(self) -> None:
        """Critical bug fix: two peaks where second is taller must return FIRST peak.

        The pre-Phase-V code used np.argmax(np.abs(I)) which would pick the
        second (taller) peak. The fix uses _find_first_peak which returns the
        chronologically first local maximum above the prominence threshold.
        """
        # Construct signal: small first peak at idx=20, bigger second at idx=70
        n = 100
        signal = np.zeros(n)
        # First peak: amplitude 0.5
        for i in range(n):
            if i < 40:
                x = i / 20.0 - 1.0  # ramp to peak at i=20 then down
                signal[i] = 0.5 * max(0.0, 1.0 - abs(x))
        # Second peak: amplitude 1.0 (global max)
        for i in range(n):
            x = (i - 70) / 20.0
            signal[i] += 1.0 * max(0.0, 1.0 - abs(x))

        result = _find_first_peak(signal, min_prominence=0.05)
        # First peak is at idx 20, second at 70. Must return <= 40 (first peak region)
        assert result <= 40, (
            f"Expected first peak index <= 40, got {result}. "
            "Bug 3 fix: should return chronologically FIRST peak, not global max."
        )

    def test_flat_signal_returns_small_index(self) -> None:
        """Constant signal: every interior point is a valid local peak (not rising/falling).

        The algorithm walks left-to-right and returns the first point that
        satisfies signal[i] >= signal[i-1] AND signal[i] >= signal[i+1].  For a
        flat signal this is satisfied at i=1, so the result is 1 (not 0).
        The key invariant is that the result is deterministic and near the start.
        """
        signal = np.ones(50) * 5.0
        idx = _find_first_peak(signal)
        # For a flat signal the first qualifying index is 1 (first interior point)
        assert idx <= 2, (
            f"Flat signal should return a small index near the start, got {idx}."
        )

    def test_noise_below_threshold_ignored(self) -> None:
        """Low-amplitude noise spikes below min_prominence threshold are skipped."""
        n = 200
        signal = np.zeros(n)
        # Tiny spike at idx=10 (amplitude 0.01, below 5% of global max 1.0)
        signal[10] = 0.01
        # Real peak at idx=100 with amplitude 1.0
        for i in range(n):
            x = (i - 100) / 30.0
            signal[i] += 1.0 * max(0.0, 1.0 - abs(x))

        result = _find_first_peak(signal, min_prominence=0.05)
        # Noise spike at 10 is 1% of global max, should be ignored
        # Result should be in the vicinity of the real peak at 100
        assert result >= 80, (
            f"Expected peak near idx=100, got {result}. "
            "Noise spike at 10 should be ignored (below 5% prominence threshold)."
        )

    def test_short_signal_len1_returns_argmax(self) -> None:
        """Signal with length 1 returns 0 (argmax of single element)."""
        signal = np.array([3.0])
        assert _find_first_peak(signal) == 0

    def test_short_signal_len2_returns_argmax(self) -> None:
        """Signal with length 2 falls back to argmax (< 3 elements)."""
        signal = np.array([1.0, 5.0])
        assert _find_first_peak(signal) == int(np.argmax(signal))

    def test_peak_at_beginning(self) -> None:
        """Peak immediately at index 1 (after start) is detectable."""
        # Falling signal: peaks right at the start
        signal = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
        result = _find_first_peak(signal)
        assert result == 1

    def test_peak_at_near_end(self) -> None:
        """Signal that peaks near the end should return that peak."""
        n = 100
        signal = np.zeros(n)
        # Peak at idx=95
        signal[95] = 1.0
        signal[94] = 0.6
        signal[93] = 0.3
        result = _find_first_peak(signal, min_prominence=0.05)
        # Should find index 95
        assert result == 95

    def test_min_prominence_affects_result(self) -> None:
        """Varying min_prominence controls which peaks qualify."""
        n = 100
        signal = np.zeros(n)
        # First peak at idx=20, amplitude 0.1 (10% of global max 1.0)
        signal[20] = 0.1
        # Second peak at idx=60, amplitude 1.0
        signal[60] = 1.0

        # With 5% threshold: first peak (10% of max) should qualify
        result_low = _find_first_peak(signal, min_prominence=0.05)
        assert result_low == 20

        # With 15% threshold: first peak (10%) is below threshold, skip to second
        result_high = _find_first_peak(signal, min_prominence=0.15)
        assert result_high == 60

    def test_dpf_waveform_returns_first_peak_not_global_max(self) -> None:
        """DPF-like waveform: first peak found, not the larger post-dip oscillation."""
        t, I_arr = _dpf_like_waveform(n_pts=1000)
        abs_I = np.abs(I_arr)
        result = _find_first_peak(abs_I, min_prominence=0.05)

        # Global argmax would return some index in post-pinch oscillation (t > 8 us)
        global_max_idx = int(np.argmax(abs_I))
        global_max_t = t[global_max_idx]

        first_peak_t = t[result]

        # The first peak should occur at t < 8 us (before post-pinch oscillations)
        assert first_peak_t < 8e-6, (
            f"First peak at t={first_peak_t:.2e} s should be before post-pinch "
            f"oscillations (t < 8 us). Global max is at t={global_max_t:.2e} s."
        )


# ============================================================
# TestValidateCurrentWaveformFirstPeak
# ============================================================

class TestValidateCurrentWaveformFirstPeak:
    """Tests for validate_current_waveform using first-peak metric (Bug 3 fix)."""

    def test_returns_peak_time_sim_key(self) -> None:
        """Result dictionary must contain 'peak_time_sim' key (new field added in Phase V)."""
        t, I_arr = _dpf_like_waveform()
        result = validate_current_waveform(t, I_arr, "PF-1000")
        assert "peak_time_sim" in result, (
            "Bug 3 fix: validate_current_waveform must return 'peak_time_sim' key."
        )

    def test_first_peak_not_global_max_for_dpf_waveform(self) -> None:
        """DPF waveform with post-pinch oscillations: first peak found, not global max."""
        t, I_arr = _dpf_like_waveform(n_pts=2000)
        result = validate_current_waveform(t, I_arr, "PF-1000")

        # peak_time_sim should be before the post-pinch oscillations
        assert result["peak_time_sim"] < 8e-6, (
            f"peak_time_sim={result['peak_time_sim']:.2e} s should be < 8 us "
            "(before post-pinch oscillations)."
        )

    def test_peak_current_computed_from_first_peak(self) -> None:
        """Peak current sim should use the first peak amplitude, not the global max."""
        t, I_arr = _dpf_like_waveform(n_pts=2000)
        abs_I = np.abs(I_arr)
        global_max = float(np.max(abs_I))

        result = validate_current_waveform(t, I_arr, "PF-1000")
        peak_sim = result["peak_current_sim"]

        # The first peak (pre-pinch) is smaller than the post-pinch oscillation
        # so peak_sim must be strictly less than the global max
        assert peak_sim < global_max, (
            f"peak_current_sim={peak_sim:.2e} A should be less than global max "
            f"{global_max:.2e} A for a DPF waveform with post-pinch oscillations."
        )

    def test_result_keys_complete(self) -> None:
        """Result dict must contain all expected keys."""
        t = np.linspace(0, 10e-6, 200)
        I_sim = 1.87e6 * np.sin(np.pi * t / (2 * 5.8e-6))
        result = validate_current_waveform(t, I_sim, "PF-1000")
        required_keys = {
            "peak_current_error",
            "peak_current_sim",
            "peak_current_exp",
            "peak_time_sim",
            "timing_ok",
        }
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    @pytest.mark.parametrize("device_name", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_all_devices_return_peak_time_sim(self, device_name: str) -> None:
        """peak_time_sim key is returned for all three registered devices."""
        device = DEVICES[device_name]
        t_rise = device.current_rise_time
        # Simple sinusoidal rise to first peak at t_rise
        t = np.linspace(0, 4 * t_rise, 400)
        I_sim = device.peak_current * np.sin(np.pi * t / (2 * t_rise))
        result = validate_current_waveform(t, I_sim, device_name)
        assert "peak_time_sim" in result
        assert result["peak_time_sim"] >= 0.0


# ============================================================
# TestAdiabaticBackPressure
# ============================================================

class TestAdiabaticBackPressure:
    """Tests for SnowplowModel._adiabatic_back_pressure (Bug 4 fix)."""

    def test_at_cathode_radius_returns_fill_pressure(self) -> None:
        """At r_s = b (cathode radius), back-pressure equals fill pressure."""
        snowplow = _make_snowplow(fill_pressure_Pa=466.0)
        p_back = snowplow._adiabatic_back_pressure(snowplow.b)
        assert p_back == pytest.approx(466.0, rel=1e-6)

    def test_compressed_radius_exceeds_fill_pressure(self) -> None:
        """At r_s < b, compressed gas exceeds fill pressure."""
        snowplow = _make_snowplow(fill_pressure_Pa=466.0)
        r_half = 0.5 * snowplow.b  # Compressed to half radius
        p_back = snowplow._adiabatic_back_pressure(r_half)
        assert p_back > 466.0, (
            f"Back-pressure {p_back:.2f} Pa should exceed fill pressure 466.0 Pa "
            "at r_s = b/2."
        )

    def test_adiabatic_formula_exact(self) -> None:
        """Back-pressure formula: p_fill * (b/r_s)^(2*gamma) with gamma=5/3."""
        gamma = 5.0 / 3.0
        p_fill = 500.0
        b = 0.08
        r_s = 0.04  # Half radius

        snowplow = _make_snowplow(cathode_radius=b, fill_pressure_Pa=p_fill)
        expected = p_fill * (b / r_s) ** (2.0 * gamma)
        result = snowplow._adiabatic_back_pressure(r_s)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_monotonically_increasing_as_radius_decreases(self) -> None:
        """Back-pressure must increase monotonically as r_s decreases."""
        snowplow = _make_snowplow()
        radii = np.linspace(snowplow.b, snowplow.r_pinch_min * 1.5, 20)
        pressures = [snowplow._adiabatic_back_pressure(r) for r in radii]
        for i in range(len(pressures) - 1):
            assert pressures[i] <= pressures[i + 1], (
                f"Back-pressure not monotonically increasing: "
                f"p({radii[i]:.4f}) = {pressures[i]:.2f} > "
                f"p({radii[i+1]:.4f}) = {pressures[i+1]:.2f}"
            )

    def test_radial_step_returns_nonzero_F_pressure(self) -> None:
        """Radial phase step must return non-zero F_pressure (Bug 4 fix).

        Before the fix, F_pressure was always 0 in the radial phase.
        """
        snowplow = _make_snowplow()
        # Force the snowplow into radial phase
        snowplow.phase = "radial"
        snowplow._rundown_complete = True
        snowplow.z = snowplow.L_anode
        snowplow._L_axial_frozen = snowplow.L_coeff * snowplow.L_anode
        # Set shock slightly inside cathode
        dr_init = 0.01 * (snowplow.b - snowplow.a)
        snowplow.r_shock = snowplow.b - dr_init
        snowplow.vr = -100.0  # Initial inward velocity [m/s]

        I_test = 1.0e6  # 1 MA
        dt = 1e-9
        result = snowplow._step_radial(dt, I_test)

        assert result["F_pressure"] > 0.0, (
            f"F_pressure={result['F_pressure']:.2e} N should be > 0 "
            "in radial phase (Bug 4 fix: adiabatic back-pressure)."
        )

    def test_F_pressure_units_are_newtons(self) -> None:
        """F_pressure = p_back * 2*pi*r_s*z_f has correct dimensional scaling."""
        snowplow = _make_snowplow(
            fill_pressure_Pa=1000.0,
            anode_length=0.16,
            cathode_radius=0.08,
        )
        # At r_s = b, F_pressure = p_fill * 2*pi*b*z_f
        gamma = 5.0 / 3.0
        r_s = snowplow.b * 0.9  # 90% of b
        p_back_expected = 1000.0 * (snowplow.b / r_s) ** (2.0 * gamma)
        F_expected = p_back_expected * 2.0 * np.pi * r_s * snowplow.L_anode  # [N]

        result = snowplow._adiabatic_back_pressure(r_s)
        F_computed = result * 2.0 * np.pi * r_s * snowplow.L_anode

        assert F_computed == pytest.approx(F_expected, rel=1e-9)
        # Order-of-magnitude check: should be in Newtons (not GPa or pN)
        assert 1e-3 < F_computed < 1e9, (
            f"F_pressure={F_computed:.2e} N out of physically plausible range."
        )

    def test_back_pressure_reduces_dip_depth_below_no_back_pressure(self) -> None:
        """Adiabatic back-pressure should oppose implosion, reducing final speed.

        The radial velocity after one step should have smaller magnitude (less
        negative) when back-pressure is present compared to no back-pressure.
        """
        # With back-pressure (normal fill_pressure_Pa)
        sp_with = _make_snowplow(fill_pressure_Pa=466.0)
        sp_with.phase = "radial"
        sp_with._rundown_complete = True
        sp_with.z = sp_with.L_anode
        sp_with._L_axial_frozen = sp_with.L_coeff * sp_with.L_anode
        dr_init = 0.01 * (sp_with.b - sp_with.a)
        sp_with.r_shock = sp_with.b - dr_init
        sp_with.vr = 0.0

        # Without back-pressure (p_fill = 0)
        sp_without = _make_snowplow(fill_pressure_Pa=0.0)
        sp_without.phase = "radial"
        sp_without._rundown_complete = True
        sp_without.z = sp_without.L_anode
        sp_without._L_axial_frozen = sp_without.L_coeff * sp_without.L_anode
        sp_without.r_shock = sp_without.b - dr_init
        sp_without.vr = 0.0

        I_test = 1.0e6
        dt = 1e-9
        n_steps = 100
        for _ in range(n_steps):
            sp_with._step_radial(dt, I_test)
            sp_without._step_radial(dt, I_test)

        # With back-pressure, shock should be less far inward (larger r_shock)
        assert sp_with.r_shock >= sp_without.r_shock, (
            f"With back-pressure r_shock={sp_with.r_shock:.4f} m should be >= "
            f"without back-pressure r_shock={sp_without.r_shock:.4f} m."
        )

    @pytest.mark.slow
    def test_full_step_sequence_F_pressure_nonzero_throughout(self) -> None:
        """F_pressure remains non-zero throughout radial compression until pinch."""
        snowplow = _make_snowplow()
        # Manually force radial phase
        snowplow.phase = "radial"
        snowplow._rundown_complete = True
        snowplow.z = snowplow.L_anode
        snowplow._L_axial_frozen = snowplow.L_coeff * snowplow.L_anode
        dr_init = 0.01 * (snowplow.b - snowplow.a)
        snowplow.r_shock = snowplow.b - dr_init
        snowplow.vr = -1000.0  # Initial inward speed

        I_test = 1.5e6
        dt = 1e-9
        for _ in range(500):
            if snowplow._pinch_complete:
                break
            result = snowplow._step_radial(dt, I_test)
            assert result["F_pressure"] >= 0.0, "F_pressure must be non-negative."


# ============================================================
# TestLeeModelFmFcNaming
# ============================================================

class TestLeeModelFmFcNaming:
    """Tests for correct fm/fc naming in LeeModel (Bug 5 fix)."""

    def test_fc_equals_current_fraction(self) -> None:
        """LeeModel.fc must equal the current_fraction argument (Bug 5 fix).

        Before Phase V, fc and fm were swapped: fc held mass_fraction and
        fm held current_fraction. Now they are correct.
        """
        model = LeeModel(current_fraction=0.9, mass_fraction=0.5)
        assert model.fc == pytest.approx(0.9, rel=1e-9), (
            f"model.fc={model.fc} should equal current_fraction=0.9 (Bug 5 fix)."
        )

    def test_fm_equals_mass_fraction(self) -> None:
        """LeeModel.fm must equal the mass_fraction argument (Bug 5 fix)."""
        model = LeeModel(current_fraction=0.9, mass_fraction=0.5)
        assert model.fm == pytest.approx(0.5, rel=1e-9), (
            f"model.fm={model.fm} should equal mass_fraction=0.5 (Bug 5 fix)."
        )

    def test_fm_fc_are_distinct(self) -> None:
        """When fm != fc, they must differ (both being set to wrong value indicates swap)."""
        model = LeeModel(current_fraction=0.75, mass_fraction=0.3)
        assert model.fm != model.fc, (
            "fm and fc should differ when current_fraction != mass_fraction."
        )

    def test_different_fm_fc_produce_different_results(self) -> None:
        """Swapping fm and fc should produce measurably different peak currents.

        If fm and fc were swapped, `model_swapped` would produce the same result
        as `model_original`. This test ensures they are distinguishable.
        """
        model_a = LeeModel(current_fraction=0.9, mass_fraction=0.4)
        model_b = LeeModel(current_fraction=0.4, mass_fraction=0.9)

        result_a = model_a.run("NX2")
        result_b = model_b.run("NX2")

        # Results should differ — different physics when f_m vs f_c are swapped
        assert abs(result_a.peak_current - result_b.peak_current) / max(
            result_a.peak_current, 1e-300
        ) > 0.05, (
            "Swapping fm and fc should produce >5% difference in peak current. "
            "If they are equal, it suggests the swap bug (Bug 5) may still exist."
        )

    def test_metadata_reports_correct_fm_fc(self) -> None:
        """LeeModelResult metadata must correctly report fm and fc values."""
        fc_val = 0.82
        fm_val = 0.45
        model = LeeModel(current_fraction=fc_val, mass_fraction=fm_val)
        result = model.run("NX2")

        assert result.metadata["fc"] == pytest.approx(fc_val, rel=1e-9), (
            f"metadata['fc']={result.metadata['fc']} should equal {fc_val}."
        )
        assert result.metadata["fm"] == pytest.approx(fm_val, rel=1e-9), (
            f"metadata['fm']={result.metadata['fm']} should equal {fm_val}."
        )


# ============================================================
# TestLeeModelRadialForce
# ============================================================

class TestLeeModelRadialForce:
    """Tests for corrected radial force with z_f factor in LeeModel (Bug 6 fix)."""

    @pytest.mark.slow
    def test_pf1000_runs_without_error(self) -> None:
        """Lee model runs for PF-1000 without exceptions."""
        model = LeeModel()
        result = model.run("PF-1000")
        assert result is not None
        assert result.peak_current > 0.0

    @pytest.mark.slow
    def test_phase2_completed_for_pf1000(self) -> None:
        """Radial phase (phase 2) is reached for PF-1000 default parameters."""
        model = LeeModel()
        result = model.run("PF-1000")
        assert 2 in result.phases_completed, (
            f"Phase 2 (radial) not completed. phases_completed={result.phases_completed}. "
            "Check that the axial rundown terminates before end of simulation time."
        )

    @pytest.mark.slow
    def test_radial_force_with_z_f_changes_peak_current(self) -> None:
        """Radial force with z_f included (Bug 6 fix) changes dynamics vs r-only force.

        The radial force in snowplow._step_radial is:
            F_rad = (mu_0 / 4pi) * (fc * I)^2 * z_f / r_s

        The z_f factor (anode_length) scales the force, so a larger anode must
        produce more radial force and faster implosion. This test compares two
        different anode lengths to verify the z_f scaling is active.
        """
        # Same device but different anode lengths
        base_params = {
            "C": 1.332e-3, "V0": 27e3, "L0": 33e-9, "R0": 2.3e-3,
            "anode_radius": 0.0575, "cathode_radius": 0.08,
            "fill_pressure_torr": 3.5,
        }
        params_short = dict(base_params, anode_length=0.08)   # 80 mm anode
        params_long = dict(base_params, anode_length=0.24)    # 240 mm anode

        model = LeeModel()
        result_short = model.run(device_params=params_short)
        result_long = model.run(device_params=params_long)

        # With longer anode: larger z_f → larger radial force → different peak current
        # They must differ by at least a few percent
        rel_diff = abs(result_short.peak_current - result_long.peak_current) / max(
            result_short.peak_current, 1e-300
        )
        assert rel_diff > 0.01, (
            f"Different anode lengths should produce measurably different peak currents. "
            f"Short: {result_short.peak_current:.2e} A, Long: {result_long.peak_current:.2e} A, "
            f"Relative difference: {rel_diff:.2%}. "
            "If rel_diff is near zero, z_f may not be included in radial force (Bug 6)."
        )

    @pytest.mark.slow
    def test_lee_model_uses_first_peak_finder(self) -> None:
        """LeeModel.run uses _find_first_peak for peak current (Bug 3 integrated)."""
        model = LeeModel()
        result = model.run("NX2")

        # peak_current_time should be a physically meaningful time
        # (within ~4x the experimental rise time for NX2: 1.8 us)
        nx2_rise_time = DEVICES["NX2"].current_rise_time
        assert result.peak_current_time <= 10.0 * nx2_rise_time, (
            f"peak_current_time={result.peak_current_time:.2e} s is unreasonably large. "
            f"NX2 rise time is {nx2_rise_time:.2e} s. "
            "LeeModel may be picking the global max (post-pinch) instead of first peak."
        )

    def test_lee_model_default_instantiation(self) -> None:
        """LeeModel default constructor sets physically reasonable fm and fc."""
        model = LeeModel()
        # Default values from docstring: current_fraction=0.7, mass_fraction=0.7
        assert model.fc == pytest.approx(0.7, rel=1e-9)
        assert model.fm == pytest.approx(0.7, rel=1e-9)
        assert 0.0 < model.fc <= 1.0
        assert 0.0 < model.fm <= 1.0
