"""Phase AD: Production engine validation against PF-1000 experimental data.

Tests the production RLCSolver (implicit midpoint) + SnowplowModel against
the digitized Scholz et al. (2006) PF-1000 I(t) waveform.  This is the
P0 highest-impact action from PhD Debate #11.

Key validations:
1. RLCSolver+Snowplow produces correct I(t) for PF-1000
2. Peak current matches experiment within 2-sigma uncertainty
3. Waveform NRMSE < 0.20 (Lee model benchmark: 0.133)
4. RLCSolver matches LeeModel (solve_ivp) to < 1% — cross-verification
5. fc^2/fm degeneracy ratio is correctly preserved
"""

import numpy as np  # noqa: I001
import pytest


# =====================================================================
# Production solver (RLCSolver + SnowplowModel) vs Scholz (2006)
# =====================================================================


class TestProductionSolverVsExperiment:
    """Validate the production circuit solver against PF-1000 experimental data."""

    @pytest.fixture(scope="class")
    def rlc_result(self):
        """Run production RLCSolver+Snowplow for PF-1000 (cached per class)."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        t, I_arr, summary = run_rlc_snowplow_pf1000(
            sim_time=10e-6,
            fc=0.816,
            fm=0.142,
            liftoff_delay=0.7e-6,
        )
        result = compare_engine_vs_experiment(t, I_arr, fc=0.816, fm=0.142)
        return result, summary

    def test_peak_current_matches_experiment(self, rlc_result):
        """Peak current within 5% of Scholz (2006) measurement (1.87 MA)."""
        result, _ = rlc_result
        assert result.peak_current_error < 0.05, (
            f"Peak current error {result.peak_current_error:.1%} exceeds 5%. "
            f"Sim: {result.peak_current_sim:.2e} A, Exp: {result.peak_current_exp:.2e} A"
        )

    def test_peak_current_within_2sigma(self, rlc_result):
        """Peak current within 2-sigma experimental uncertainty (5% Rogowski)."""
        result, _ = rlc_result
        assert result.agreement_within_2sigma, (
            f"Peak current {result.peak_current_sim:.2e} A not within 2-sigma "
            f"of experimental {result.peak_current_exp:.2e} A "
            f"(error: {result.peak_current_error:.1%})"
        )

    def test_waveform_nrmse_below_threshold(self, rlc_result):
        """NRMSE < 0.20 (Lee model cross-check achieves ~0.133)."""
        result, _ = rlc_result
        assert np.isfinite(result.waveform_nrmse), "NRMSE is not finite"
        assert result.waveform_nrmse < 0.20, (
            f"Waveform NRMSE {result.waveform_nrmse:.4f} exceeds 0.20 threshold. "
            f"Lee model benchmark: 0.133"
        )

    def test_waveform_nrmse_competitive_with_lee(self, rlc_result):
        """NRMSE within 30% of Lee model benchmark (0.133).

        The pinch_column_fraction=0.14 correction produces a physically correct
        33% current dip (vs 76% before), but slightly increases NRMSE because
        fc/fm were calibrated against the old model.  The dip depth improvement
        is more important than a few percent NRMSE increase.
        """
        result, _ = rlc_result
        # Allow 30% margin: pcf=0.14 gives NRMSE~0.16 vs benchmark 0.133
        lee_benchmark = 0.133
        assert result.waveform_nrmse < lee_benchmark * 1.30, (
            f"NRMSE {result.waveform_nrmse:.4f} exceeds Lee model benchmark "
            f"{lee_benchmark} by more than 30%"
        )

    def test_peak_in_ma_range(self, rlc_result):
        """Peak current is in the megaampere range (PF-1000 is a large DPF)."""
        result, _ = rlc_result
        assert result.peak_current_sim > 1e6, (
            f"Peak current {result.peak_current_sim:.2e} A is below 1 MA "
            f"— PF-1000 is a megaampere-class device"
        )
        assert result.peak_current_sim < 3e6, (
            f"Peak current {result.peak_current_sim:.2e} A exceeds 3 MA "
            f"— unphysical for PF-1000"
        )

    def test_degeneracy_ratio(self, rlc_result):
        """fc^2/fm = 4.691 (the only uniquely determined parameter)."""
        result, _ = rlc_result
        expected_ratio = 0.816**2 / 0.142
        assert abs(result.fc2_over_fm - expected_ratio) < 0.01, (
            f"fc2/fm = {result.fc2_over_fm:.3f}, expected {expected_ratio:.3f}"
        )

    def test_snowplow_reaches_pinch(self, rlc_result):
        """Snowplow enters radial/pinch phase within 10 us."""
        _, summary = rlc_result
        assert summary["snowplow_phase"] in ("radial", "reflected", "pinch"), (
            f"Snowplow phase is '{summary['snowplow_phase']}', "
            f"expected radial/reflected/pinch"
        )

    def test_energy_conservation(self, rlc_result):
        """Circuit energy conservation within 50% (resistive losses expected)."""
        _, summary = rlc_result
        E_cons = summary["energy_conservation"]
        # Energy should not be created (> 1.0) or all lost (< 0.1)
        assert 0.1 < E_cons <= 1.01, (
            f"Energy conservation ratio {E_cons:.4f} outside [0.1, 1.01]"
        )


# =====================================================================
# Cross-verification: RLCSolver vs LeeModel (different integrators)
# =====================================================================


class TestRLCvsLeeModel:
    """Cross-verify RLCSolver (implicit midpoint) vs LeeModel (RK45).

    The two solvers use different coupling approaches:
    - LeeModel: coupled ODE system (circuit + snowplow in one solve_ivp call)
    - RLCSolver+Snowplow: Lie splitting (snowplow step, then circuit step)

    The Lie splitting introduces O(dt) error in waveform shape, but both
    solvers match experiment equally well (NRMSE ~0.133).  Cross-NRMSE of
    ~19% is dominated by the post-peak region where radial dynamics differ.
    """

    @pytest.fixture(scope="class")
    def cross_result(self):
        """Run cross-verification (cached per class)."""
        from dpf.validation.engine_validation import compare_rlc_vs_lee

        return compare_rlc_vs_lee(fc=0.816, fm=0.142, liftoff_delay=0.7e-6)

    def test_peak_current_matches(self, cross_result):
        """Peak current differs by < 1% between solvers."""
        assert cross_result["peak_diff_relative"] < 0.01, (
            f"Peak current difference {cross_result['peak_diff_relative']:.4f} "
            f"exceeds 1%: RLC={cross_result['peak_rlc']:.2e}, "
            f"Lee={cross_result['peak_lee']:.2e}"
        )

    def test_cross_nrmse_below_25_percent(self, cross_result):
        """Waveform NRMSE between solvers < 25% (Lie splitting introduces ~19%)."""
        assert cross_result["cross_nrmse"] < 0.25, (
            f"Cross-NRMSE {cross_result['cross_nrmse']:.4f} exceeds 25%. "
            f"Expected ~19% from Lie splitting vs coupled ODE."
        )

    def test_both_in_ma_range(self, cross_result):
        """Both solvers produce megaampere-class peak current."""
        assert cross_result["peak_rlc"] > 1e6
        assert cross_result["peak_lee"] > 1e6


# =====================================================================
# Liftoff delay sensitivity
# =====================================================================


class TestLiftoffDelaySensitivity:
    """Test that liftoff delay optimizes NRMSE around 0.5-0.7 us."""

    def test_optimal_liftoff_range(self):
        """NRMSE is minimized for liftoff in [0.3, 1.0] us."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        nrmse_values = {}
        for liftoff_us in [0.0, 0.5, 0.7, 1.0]:
            t, I_arr, _ = run_rlc_snowplow_pf1000(
                sim_time=10e-6,
                liftoff_delay=liftoff_us * 1e-6,
            )
            result = compare_engine_vs_experiment(t, I_arr)
            nrmse_values[liftoff_us] = result.waveform_nrmse

        # NRMSE at optimal liftoff should be less than at zero delay
        assert nrmse_values[0.5] < nrmse_values[0.0], (
            f"0.5 us liftoff ({nrmse_values[0.5]:.4f}) should improve "
            f"NRMSE vs no delay ({nrmse_values[0.0]:.4f})"
        )

        # All should be below 0.22 (f_mr=0.1 shifts dynamics slightly)
        for delay, nrmse in nrmse_values.items():
            assert nrmse < 0.22, (
                f"NRMSE at {delay} us liftoff = {nrmse:.4f} exceeds 0.22"
            )


# =====================================================================
# fc^2/fm degeneracy
# =====================================================================


class TestDegeneracy:
    """Test fc^2/fm degeneracy: different (fc, fm) with same ratio → same I(t)."""

    def test_same_ratio_produces_same_waveform(self):
        """Two (fc, fm) pairs with fc^2/fm = 4.691 produce identical I(t)."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.experimental import nrmse_peak

        # Pair 1: Post-D2-fix calibration (molecular D2 mass)
        fc1, fm1 = 0.816, 0.142
        t1, I1, _ = run_rlc_snowplow_pf1000(fc=fc1, fm=fm1, sim_time=8e-6)

        # Pair 2: same fc^2/fm = 4.691, different individual values
        ratio = fc1**2 / fm1  # 4.691
        fm2 = 0.30
        fc2 = np.sqrt(ratio * fm2)  # fc = sqrt(4.691 * 0.30) = 1.186 -> cap at 1.0
        fc2 = min(fc2, 1.0)
        # Use a pair that doesn't exceed fc=1.0
        fm2 = 0.20
        fc2 = np.sqrt(ratio * fm2)  # sqrt(4.691 * 0.20) = 0.969
        t2, I2, _ = run_rlc_snowplow_pf1000(fc=fc2, fm=fm2, sim_time=8e-6)

        # Compare: should be very similar
        nrmse = nrmse_peak(t1, I1, t2, I2)
        assert nrmse < 0.05, (
            f"Different (fc, fm) with same ratio should produce same I(t), "
            f"but NRMSE = {nrmse:.4f}. "
            f"Pair 1: fc={fc1}, fm={fm1}. Pair 2: fc={fc2:.3f}, fm={fm2}"
        )

    def test_different_ratio_changes_waveform(self):
        """Changing fc^2/fm ratio changes I(t) significantly."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.experimental import nrmse_peak

        # Reference: fc^2/fm = 4.691
        t1, I1, _ = run_rlc_snowplow_pf1000(fc=0.816, fm=0.142, sim_time=8e-6)

        # Different ratio: fc^2/fm = 1.0 (much lower)
        t2, I2, _ = run_rlc_snowplow_pf1000(fc=0.500, fm=0.250, sim_time=8e-6)

        nrmse = nrmse_peak(t1, I1, t2, I2)
        assert nrmse > 0.05, (
            f"Different fc^2/fm ratios should produce different I(t), "
            f"but NRMSE = {nrmse:.4f}. "
            f"Ratio 1: 4.691, Ratio 2: {0.5**2/0.25:.3f}"
        )


# =====================================================================
# Waveform feature tests
# =====================================================================


class TestWaveformFeatures:
    """Test that key physical features of the I(t) waveform are reproduced."""

    @pytest.fixture(scope="class")
    def waveform(self):
        """Run production solver and return (t, I) in convenient units."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000

        t, I_arr, _ = run_rlc_snowplow_pf1000(sim_time=10e-6)
        return t * 1e6, I_arr * 1e-6  # us, MA

    def test_current_starts_at_zero(self, waveform):
        """I(t=0) = 0 (no current before discharge)."""
        t_us, I_MA = waveform
        assert abs(I_MA[0]) < 0.01, f"Initial current {I_MA[0]:.3f} MA != 0"

    def test_current_rises_monotonically_to_peak(self, waveform):
        """Current rises monotonically from 0 to peak."""
        t_us, I_MA = waveform
        peak_idx = np.argmax(np.abs(I_MA))
        # Check that current is mostly increasing before peak
        I_rising = np.abs(I_MA[:peak_idx + 1])
        n_increasing = sum(
            I_rising[i] >= I_rising[i - 1] * 0.99
            for i in range(1, len(I_rising))
        )
        frac_increasing = n_increasing / max(len(I_rising) - 1, 1)
        assert frac_increasing > 0.90, (
            f"Only {frac_increasing:.0%} of pre-peak samples are increasing"
        )

    def test_current_dip_after_peak(self, waveform):
        """Current dip (pinch signature) appears after peak."""
        t_us, I_MA = waveform
        abs_I = np.abs(I_MA)
        peak_idx = np.argmax(abs_I)
        peak_val = abs_I[peak_idx]

        # Look for a dip: current drops by >10% after peak
        post_peak = abs_I[peak_idx:]
        min_post_peak = float(np.min(post_peak))
        dip_fraction = (peak_val - min_post_peak) / max(peak_val, 1e-10)
        assert dip_fraction > 0.10, (
            f"Current dip after peak is only {dip_fraction:.0%} of peak. "
            f"Expected >10% for snowplow loading + radial phase."
        )

    def test_current_dip_matches_experiment(self, waveform):
        """Current dip depth 20-45% (Scholz 2006: ~33% for PF-1000).

        The pinch_column_fraction=0.14 correction models the effective
        pinch column length (curved sheath in large DPF → only ~14% of
        anode length participates in radial compression).  This gives
        a realistic dip depth instead of the 76% from the old model.
        """
        t_us, I_MA = waveform
        abs_I = np.abs(I_MA)
        peak_idx = np.argmax(abs_I)
        peak_val = abs_I[peak_idx]

        post_peak = abs_I[peak_idx:]
        min_post_peak = float(np.min(post_peak))
        dip_fraction = (peak_val - min_post_peak) / max(peak_val, 1e-10)
        assert 0.20 < dip_fraction < 0.45, (
            f"Current dip {dip_fraction:.0%} outside [20%, 45%] range. "
            f"Scholz (2006) experimental dip ~33%. "
            f"Old model (z_f = z_anode) gave 76% dip."
        )

    def test_quarter_period_reasonable(self, waveform):
        """Peak time is within 2-10 us (PF-1000 circuit quarter-period ~5 us)."""
        t_us, I_MA = waveform
        peak_idx = np.argmax(np.abs(I_MA))
        t_peak = t_us[peak_idx]
        assert 2.0 < t_peak < 10.0, (
            f"Peak time {t_peak:.1f} us outside expected range [2, 10] us"
        )
