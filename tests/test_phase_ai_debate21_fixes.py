"""Phase AI: Debate #21 verdict actions — measurement uncertainty, reflected shock density.

Implements three panel recommendations from PhD Debate #21 (6.3/10):
1. Tighten Phase AE test bounds to published fc/fm ranges (already in test_phase_ae)
2. State Scholz measurement uncertainty explicitly in experimental.py
3. Reflected shock density correction: 4*rho0 → 8*rho0 (double-shock R-H)

These address the panel's top concerns:
- Scholz measurement uncertainty was unstated across 21 debates
- Reflected shock density underestimate (single-shock vs double-shock R-H)
- Phase AE test bounds were too permissive (0.55-0.85 vs published 0.6-0.8)
"""

import numpy as np
import pytest

# =====================================================================
# Measurement uncertainty documentation
# =====================================================================


class TestMeasurementUncertainty:
    """Verify Scholz measurement uncertainty is explicitly stated."""

    def test_pf1000_has_digitization_uncertainty(self):
        """PF-1000 data includes digitization uncertainty."""
        from dpf.validation.experimental import PF1000_DATA

        assert PF1000_DATA.waveform_digitization_uncertainty > 0, (
            "PF-1000 waveform_digitization_uncertainty must be > 0"
        )
        assert PF1000_DATA.waveform_digitization_uncertainty == pytest.approx(0.03, abs=0.01)

    def test_pf1000_has_time_uncertainty(self):
        """PF-1000 data includes temporal digitization uncertainty."""
        from dpf.validation.experimental import PF1000_DATA

        assert PF1000_DATA.waveform_time_uncertainty > 0
        assert PF1000_DATA.waveform_time_uncertainty == pytest.approx(0.005, abs=0.005)

    def test_pf1000_has_measurement_notes(self):
        """PF-1000 data includes measurement provenance notes."""
        from dpf.validation.experimental import PF1000_DATA

        assert len(PF1000_DATA.measurement_notes) > 100
        assert "Scholz" in PF1000_DATA.measurement_notes
        assert "Rogowski" in PF1000_DATA.measurement_notes
        assert "digitization" in PF1000_DATA.measurement_notes.lower()
        assert "ASME V&V 20-2009" in PF1000_DATA.measurement_notes

    def test_pf1000_combined_uncertainty(self):
        """Combined uncertainty = sqrt(Rogowski^2 + digitization^2) ~ 5.8%."""
        from dpf.validation.experimental import PF1000_DATA

        u_rog = PF1000_DATA.peak_current_uncertainty  # 5%
        u_dig = PF1000_DATA.waveform_digitization_uncertainty  # 3%
        u_combined = np.sqrt(u_rog**2 + u_dig**2)
        assert u_combined == pytest.approx(0.058, abs=0.005)

    def test_validate_current_waveform_reports_uncertainty(self):
        """validate_current_waveform includes digitization in uncertainty budget."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.experimental import validate_current_waveform

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        result = validate_current_waveform(t, I_arr, "PF-1000")

        assert "digitization_1sigma" in result["uncertainty"]
        assert result["uncertainty"]["digitization_1sigma"] > 0
        assert "peak_current_total_exp_1sigma" in result["uncertainty"]
        # Total > Rogowski alone (quadrature sum includes digitization)
        assert result["uncertainty"]["peak_current_total_exp_1sigma"] > (
            result["uncertainty"]["peak_current_exp_1sigma"]
        )

    def test_validate_current_waveform_reports_notes(self):
        """validate_current_waveform includes measurement notes."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.experimental import validate_current_waveform

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        result = validate_current_waveform(t, I_arr, "PF-1000")

        assert "measurement_notes" in result
        assert len(result["measurement_notes"]) > 50

    def test_nx2_has_measurement_notes(self):
        """NX2 data includes measurement provenance notes."""
        from dpf.validation.experimental import NX2_DATA

        assert len(NX2_DATA.measurement_notes) > 20
        assert "Type B" in NX2_DATA.measurement_notes

    def test_unu_ictp_has_measurement_notes(self):
        """UNU-ICTP data includes measurement provenance notes."""
        from dpf.validation.experimental import UNU_ICTP_DATA

        assert len(UNU_ICTP_DATA.measurement_notes) > 20

    def test_effective_dof_documented(self):
        """Effective DOF ~5 is documented in measurement notes."""
        from dpf.validation.experimental import PF1000_DATA

        assert "independent data points" in PF1000_DATA.measurement_notes.lower() or (
            "effective" in PF1000_DATA.measurement_notes.lower()
        )


# =====================================================================
# Reflected shock density correction
# =====================================================================


class TestReflectedShockDensity:
    """Verify reflected shock uses double-shock R-H density (~8*rho0)."""

    def test_snowplow_reflected_shock_uses_8x(self):
        """Snowplow Phase 4 uses rho_post = 8*rho0 (double-shock estimate)."""
        import inspect

        from dpf.fluid.snowplow import SnowplowModel

        source = inspect.getsource(SnowplowModel)
        assert "8.0 * self.rho0" in source, (
            "Snowplow should use 8*rho0 for reflected shock (double-shock R-H)"
        )

    def test_lee_model_reflected_shock_uses_8x(self):
        """Lee model Phase 4 uses rho_post = 8*rho0 (double-shock estimate)."""
        import inspect

        from dpf.validation.lee_model_comparison import LeeModel

        source = inspect.getsource(LeeModel)
        assert "8.0 * rho0" in source, (
            "Lee model should use 8*rho0 for reflected shock"
        )

    def test_reflected_shock_density_comment_explains_physics(self):
        """Code comments explain the double-shock Rankine-Hugoniot reasoning."""
        import inspect

        from dpf.fluid.snowplow import SnowplowModel

        source = inspect.getsource(SnowplowModel)
        assert "double-shock" in source.lower() or "re-compress" in source.lower()

    def test_snowplow_pf1000_still_produces_valid_waveform(self):
        """PF-1000 snowplow with 8*rho0 still produces valid I(t)."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        peak = float(np.max(np.abs(I_arr)))
        assert 1.5e6 < peak < 2.5e6, f"Peak {peak/1e6:.2f} MA outside expected range"

    def test_lee_model_pf1000_still_produces_valid_waveform(self):
        """Lee model PF-1000 with 8*rho0 still produces valid I(t)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816, mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        assert 1.5e6 < result.peak_current < 2.5e6

    def test_nrmse_still_below_threshold(self):
        """NRMSE < 0.20 after reflected shock density correction."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        result = compare_engine_vs_experiment(t, I_arr, fc=0.816, fm=0.142)
        assert result.waveform_nrmse < 0.20, (
            f"NRMSE {result.waveform_nrmse:.4f} exceeds 0.20 after density correction"
        )


# =====================================================================
# Tightened published range bounds
# =====================================================================


class TestTightenedBounds:
    """Verify fc/fm bounds match published Lee & Saw ranges."""

    def test_published_fc_range_pf1000(self):
        """PF-1000 published fc range is [0.6, 0.8]."""
        from dpf.validation.calibration import _PUBLISHED_FC_FM_RANGES

        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fc"]
        assert fc_lo == pytest.approx(0.6)
        assert fc_hi == pytest.approx(0.8)

    def test_published_fm_range_pf1000(self):
        """PF-1000 published fm range is [0.05, 0.20]."""
        from dpf.validation.calibration import _PUBLISHED_FC_FM_RANGES

        fm_lo, fm_hi = _PUBLISHED_FC_FM_RANGES["PF-1000"]["fm"]
        assert fm_lo == pytest.approx(0.05)
        assert fm_hi == pytest.approx(0.20)

    def test_calibration_converges_in_published_bounds(self):
        """Calibration with published bounds converges.

        Known behavior: optimizer pins fc at upper bound (0.8) because the
        NRMSE minimum lies near fc~0.816 (outside [0.6, 0.8]).  The
        boundary-pinned result is a legitimate finding — it means the
        optimal fc slightly exceeds the published Lee & Saw (2014) range.
        """
        from dpf.validation.calibration import LeeModelCalibrator

        cal = LeeModelCalibrator("PF-1000", pinch_column_fraction=0.14)
        result = cal.calibrate(
            maxiter=50,
            fc_bounds=(0.6, 0.8),
            fm_bounds=(0.05, 0.20),
        )
        assert result.converged
        assert 0.6 <= result.best_fc <= 0.8
        assert 0.05 <= result.best_fm <= 0.20

    def test_calibration_nrmse_in_published_bounds(self):
        """Calibration NRMSE < 0.20 with published bounds."""
        from dpf.validation.calibration import LeeModelCalibrator

        cal = LeeModelCalibrator("PF-1000", pinch_column_fraction=0.14)
        result = cal.calibrate(
            maxiter=50,
            fc_bounds=(0.6, 0.8),
            fm_bounds=(0.05, 0.20),
        )
        assert result.objective_value < 0.20
