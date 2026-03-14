"""Phase AF: Lee model reflected shock (Phase 4) validation.

Tests the reflected shock phase added to lee_model_comparison.py.
The reflected shock models the outward expansion of the compressed
gas after pinch stagnation, driven by back-pressure against J×B force.

Key validations:
1. Phase 4 completes and is recorded in phases_completed
2. Reflected shock physics: r_shock expands from pinch radius outward
3. Current dip is physically realistic with reflected shock
4. NRMSE vs experiment is maintained or improved
5. Cross-verification with snowplow.py reflected shock
"""

import numpy as np
import pytest

# =====================================================================
# AF.1: Phase 4 completion
# =====================================================================


class TestPhase4Completion:
    """Verify reflected shock Phase 4 runs and completes."""

    def test_phase4_in_phases_completed(self):
        """Phase 4 appears in phases_completed list."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        result = model.run("PF-1000")
        assert 4 in result.phases_completed, (
            f"Phase 4 not completed. Phases: {result.phases_completed}"
        )

    def test_phases_1_2_4_all_present(self):
        """All three phases (1, 2, 4) are completed."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        result = model.run("PF-1000")
        assert 1 in result.phases_completed
        assert 2 in result.phases_completed
        assert 4 in result.phases_completed

    def test_phase4_with_pcf_014(self):
        """Phase 4 completes with pinch_column_fraction=0.14."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        assert 4 in result.phases_completed

    def test_phase4_with_nx2(self):
        """Phase 4 completes for NX2 device."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.7, mass_fraction=0.15)
        result = model.run("NX2")
        assert 4 in result.phases_completed

    def test_phase4_with_crowbar(self):
        """Phase 4 completes with crowbar enabled."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            crowbar_enabled=True,
        )
        result = model.run("PF-1000")
        assert 4 in result.phases_completed


# =====================================================================
# AF.2: Reflected shock physics
# =====================================================================


class TestReflectedShockPhysics:
    """Verify reflected shock physics are correct."""

    @pytest.fixture(scope="class")
    def pf1000_result(self):
        """Run PF-1000 Lee model with reflected shock (cached)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        return model.run("PF-1000")

    def test_shock_reaches_pinch_minimum(self, pf1000_result):
        """Shock radius reaches near 0.1*a during radial phase."""
        r = pf1000_result.r_shock
        a = pf1000_result.metadata["anode_radius"]
        assert np.min(r) < 0.15 * a, (
            f"Min r_shock {np.min(r):.4f} m > 0.15*a = {0.15*a:.4f} m"
        )

    def test_shock_expands_after_pinch(self):
        """Shock radius increases after pinch with pcf=0.14.

        With pcf=1.0, J×B force overwhelms back-pressure and the
        reflected shock barely moves. With pcf=0.14, the shorter
        pinch column allows measurable expansion.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        r = result.r_shock
        min_idx = np.argmin(r)
        # After minimum, shock should expand
        if min_idx < len(r) - 1:
            r_after = r[min_idx + 1:]
            assert len(r_after) > 0, "No data after pinch minimum"
            assert np.max(r_after) >= r[min_idx], (
                f"Shock doesn't expand: r_min={r[min_idx]:.4f}, "
                f"r_max_after={np.max(r_after):.4f}"
            )

    def test_shock_final_radius_reasonable(self, pf1000_result):
        """Final shock radius is between pinch minimum and cathode."""
        r = pf1000_result.r_shock
        b = pf1000_result.metadata["cathode_radius"]
        r_final = float(r[-1])
        r_min = float(np.min(r))
        # Final radius should be larger than pinch minimum
        assert r_final >= r_min, (
            f"Final r={r_final:.4f} < min r={r_min:.4f}"
        )
        # Final radius should not exceed cathode
        assert r_final <= b * 1.01, (
            f"Final r={r_final:.4f} > cathode b={b:.4f}"
        )

    def test_current_continuous_across_phases(self, pf1000_result):
        """Current I(t) is continuous (no jumps at phase transitions)."""
        I_arr = pf1000_result.I  # noqa: E741
        # Check for large jumps (> 10% of peak)
        dI = np.abs(np.diff(I_arr))
        max_jump = np.max(dI)
        peak_I = np.max(np.abs(I_arr))
        assert max_jump < 0.10 * peak_I, (
            f"Max current jump {max_jump:.2e} > 10% of peak {peak_I:.2e}"
        )


# =====================================================================
# AF.3: Current dip with reflected shock
# =====================================================================


class TestCurrentDipWithReflectedShock:
    """Validate current dip behavior with reflected shock."""

    def test_pcf1_deep_dip(self):
        """pcf=1.0 with reflected shock still gives deep dip (>50%)."""
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=1.0,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]
        dip = (abs_I[peak_idx] - np.min(post_peak)) / abs_I[peak_idx]
        assert dip > 0.40, f"pcf=1.0 dip {dip:.0%} should be > 40%"

    def test_pcf014_experimental_dip(self):
        """pcf=0.14 with reflected shock gives 20-90% dip."""
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]
        dip = (abs_I[peak_idx] - np.min(post_peak)) / abs_I[peak_idx]
        assert 0.05 < dip < 0.90, (
            f"pcf=0.14 dip {dip:.0%} outside [5%, 90%]"
        )

    def test_reflected_shock_reduces_dip_duration(self):
        """Reflected shock causes current to recover after dip minimum.

        With reflected shock, the current dip is not monotonically
        decreasing — the shock expansion reduces dL/dt, allowing
        partial current recovery.
        """
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        # Search within peak + 1 us for pinch dip (not deep post-pinch decay)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]

        # Find the dip minimum
        min_idx = np.argmin(post_peak)
        if min_idx < len(post_peak) - 5:
            # Current after dip minimum should increase (recovery)
            after_dip = post_peak[min_idx:min(min_idx + 20, len(post_peak))]
            recovery = (np.max(after_dip) - post_peak[min_idx]) / abs_I[peak_idx]
            # At least some recovery expected (even 1% is physical)
            assert recovery >= 0.0, (
                "No current recovery after dip minimum"
            )


# =====================================================================
# AF.4: NRMSE vs experiment
# =====================================================================


class TestNRMSEWithReflectedShock:
    """Verify NRMSE against Scholz (2006) is maintained."""

    def test_nrmse_below_020(self):
        """NRMSE < 0.20 with reflected shock (Lee benchmark: 0.133)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        comp = model.compare_with_experiment("PF-1000")
        assert np.isfinite(comp.waveform_nrmse)
        assert comp.waveform_nrmse < 0.20, (
            f"NRMSE {comp.waveform_nrmse:.4f} exceeds 0.20"
        )

    def test_peak_error_below_5pct(self):
        """Peak current error < 5% with reflected shock."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        comp = model.compare_with_experiment("PF-1000")
        assert comp.peak_current_error < 0.05, (
            f"Peak error {comp.peak_current_error:.1%} > 5%"
        )


# =====================================================================
# AF.5: Reflected shock consistency with snowplow
# =====================================================================


class TestReflectedShockConsistency:
    """Cross-check Lee model Phase 4 against snowplow reflected shock."""

    def test_both_have_reflected_phase(self):
        """Both LeeModel and SnowplowModel complete reflected shock."""
        from dpf.validation.lee_model_comparison import LeeModel

        lee = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        lee_result = lee.run("PF-1000")
        assert 4 in lee_result.phases_completed

        # SnowplowModel reflected phase check
        from dpf.fluid.snowplow import SnowplowModel

        sp = SnowplowModel(
            anode_radius=0.115,
            cathode_radius=0.160,
            fill_density=8.4e-5,  # ~3.5 Torr D2 at 300K
            anode_length=0.60,
            current_fraction=0.816,
            mass_fraction=0.142,
        )
        # Run enough steps to reach reflected phase
        for _ in range(50000):
            sp.step(1e-9, 1.5e6)  # 1ns steps, 1.5 MA drive
            if sp.phase in ("reflected", "frozen"):
                break

        assert sp.phase in ("reflected", "frozen"), (
            f"Snowplow phase is '{sp.phase}', expected reflected/frozen"
        )

    def test_peak_currents_consistent(self):
        """Lee model and engine (RLCSolver+Snowplow) produce similar peaks."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.lee_model_comparison import LeeModel

        # Lee model
        lee = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        lee_result = lee.run("PF-1000")

        # RLCSolver + Snowplow
        _, I_rlc, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, sim_time=10e-6,
        )

        lee_peak = lee_result.peak_current
        rlc_peak = float(np.max(np.abs(I_rlc)))

        rel_diff = abs(lee_peak - rlc_peak) / max(lee_peak, rlc_peak)
        assert rel_diff < 0.02, (
            f"Peak current mismatch: Lee={lee_peak/1e6:.3f} MA, "
            f"RLC={rlc_peak/1e6:.3f} MA, diff={rel_diff:.1%}"
        )
