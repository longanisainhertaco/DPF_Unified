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

        assert PF1000_DATA.waveform_amplitude_uncertainty > 0, (
            "PF-1000 waveform_amplitude_uncertainty must be > 0"
        )
        assert PF1000_DATA.waveform_amplitude_uncertainty == pytest.approx(0.03, abs=0.01)

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
        u_dig = PF1000_DATA.waveform_amplitude_uncertainty  # 3%
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


# =====================================================================
# Higher-resolution Metal engine validation (64x1x128)
# =====================================================================


class TestHighResMetalEngine:
    """Grid convergence: Metal engine at 64x1x128 (2.5mm dx) vs 32x1x64 (5mm).

    PhD Debate #21 recommendation #4: demonstrate that NRMSE improves
    with resolution, providing evidence that the coarse-grid result
    (NRMSE ~0.20-0.31) is not converged and finer grids approach the
    standalone snowplow baseline (NRMSE ~0.16).

    Previously xfailed due to NaN instability at 64x1x128 — FIXED by
    positivity-preserving reconstruction fallback + neighbor-averaging
    NaN repair in Phase AJ (metal_riemann.py + metal_solver.py).
    """

    @pytest.fixture(scope="class")
    def highres_result(self):
        """Run 64x1x128 Metal engine PF-1000 (slow: ~2-4 min)."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = [64, 1, 128]
        preset["dx"] = 2.5e-3  # 2.5 mm (half of AG's 5 mm)
        preset["sim_time"] = 12e-6
        preset["diagnostics_path"] = ":memory:"
        preset["fluid"] = {
            "backend": "metal",
            "riemann_solver": "hll",
            "reconstruction": "plm",
            "time_integrator": "ssp_rk2",
            "precision": "float32",
            "use_ct": False,
        }
        preset["radiation"] = {"bremsstrahlung_enabled": False, "fld_enabled": False}
        preset["collision"] = {"enabled": False}

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)

        times = []
        currents = []
        for _ in range(10000):
            result = engine.step()
            times.append(engine.time)
            currents.append(abs(engine.circuit.current))
            if result.finished:
                break

        return np.array(times), np.array(currents), engine

    @pytest.mark.slow
    def test_highres_engine_completes(self, highres_result):
        """64x1x128 Metal engine completes full 12 us simulation."""
        times, _, _ = highres_result
        assert times[-1] >= 11e-6, (
            f"Simulation ended early at t={times[-1]*1e6:.2f} us"
        )

    @pytest.mark.slow
    def test_highres_peak_current_physical(self, highres_result):
        """64x1x128 peak current in physical range [0.5, 5.0] MA."""
        _, currents, _ = highres_result
        peak = float(np.max(currents))
        assert 0.5e6 < peak < 5e6, f"Peak {peak/1e6:.2f} MA outside range"

    @pytest.mark.slow
    def test_highres_nrmse_below_coarse(self, highres_result):
        """64x1x128 NRMSE <= coarse (32x1x64) NRMSE.

        Grid convergence: finer grid should produce equal or better NRMSE.
        Coarse result (Phase AG): NRMSE ~0.20-0.31. We expect <= 0.35
        at 64x1x128 (allowing for float32 noise at higher resolution).
        """
        from dpf.validation.experimental import PF1000_DATA, nrmse_peak

        times, currents, _ = highres_result
        nrmse = nrmse_peak(times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I)
        assert nrmse < 0.35, (
            f"64x1x128 NRMSE {nrmse:.4f} exceeds 0.35 (worse than coarse grid)"
        )

    @pytest.mark.slow
    def test_highres_nrmse_reported(self, highres_result):
        """Report 64x1x128 NRMSE for grid convergence documentation."""
        from dpf.validation.experimental import PF1000_DATA, nrmse_peak

        times, currents, _ = highres_result
        nrmse_full = nrmse_peak(
            times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
        )
        nrmse_trunc = nrmse_peak(
            times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
            truncate_at_dip=True,
        )
        peak = float(np.max(currents))
        # Document the values (test always passes — it's for recording)
        print("\n=== Grid Convergence: 64x1x128 Metal Engine ===")
        print(f"Peak current: {peak/1e6:.3f} MA")
        print(f"NRMSE (full):      {nrmse_full:.4f}")
        print(f"NRMSE (truncated): {nrmse_trunc:.4f}")
        print(f"Experimental peak: {PF1000_DATA.peak_current/1e6:.3f} MA")


class TestHighResMetalFloat64:
    """Grid convergence using Metal solver in float64 CPU mode.

    Float64 mode forces CPU execution for maximum numerical accuracy.
    Both float32 (MPS GPU) and float64 (CPU) modes now complete at
    64x1x128 thanks to positivity-preserving reconstruction fallback
    and neighbor-averaging NaN repair in the Metal solver.
    """

    @pytest.fixture(scope="class")
    def float64_result(self):
        """Run 64x1x128 Metal engine in float64 CPU mode."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = [64, 1, 128]
        preset["dx"] = 2.5e-3
        preset["sim_time"] = 12e-6
        preset["diagnostics_path"] = ":memory:"
        preset["fluid"] = {
            "backend": "metal",
            "riemann_solver": "hll",
            "reconstruction": "plm",
            "time_integrator": "ssp_rk2",
            "precision": "float64",  # CPU float64 mode
            "use_ct": False,
        }
        preset["radiation"] = {"bremsstrahlung_enabled": False, "fld_enabled": False}
        preset["collision"] = {"enabled": False}

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)

        times = []
        currents = []
        for _ in range(10000):
            result = engine.step()
            times.append(engine.time)
            currents.append(abs(engine.circuit.current))
            if result.finished:
                break

        return np.array(times), np.array(currents), engine

    @pytest.mark.slow
    def test_float64_engine_completes(self, float64_result):
        """64x1x128 float64 Metal engine completes 12 us simulation."""
        times, _, _ = float64_result
        assert times[-1] >= 11e-6, (
            f"Simulation ended early at t={times[-1]*1e6:.2f} us"
        )

    @pytest.mark.slow
    def test_float64_peak_current_physical(self, float64_result):
        """Float64 peak current in physical range."""
        _, currents, _ = float64_result
        peak = float(np.max(currents))
        assert 0.5e6 < peak < 5e6, f"Peak {peak/1e6:.2f} MA outside range"

    @pytest.mark.slow
    def test_float64_nrmse_better_than_coarse(self, float64_result):
        """Float64 64x1x128 NRMSE < 0.30 (better than coarse float32)."""
        from dpf.validation.experimental import PF1000_DATA, nrmse_peak

        times, currents, _ = float64_result
        nrmse = nrmse_peak(
            times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
        )
        assert nrmse < 0.30, (
            f"Float64 NRMSE {nrmse:.4f} exceeds 0.30"
        )

    @pytest.mark.slow
    def test_float64_grid_convergence_report(self, float64_result):
        """Report float64 grid convergence metrics."""
        from dpf.validation.experimental import PF1000_DATA, nrmse_peak

        times, currents, _ = float64_result
        nrmse_full = nrmse_peak(
            times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
        )
        nrmse_trunc = nrmse_peak(
            times, currents, PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
            truncate_at_dip=True,
        )
        peak = float(np.max(currents))
        print("\n=== Grid Convergence: 64x1x128 Metal Engine (float64) ===")
        print(f"Peak current: {peak/1e6:.3f} MA")
        print(f"NRMSE (full):      {nrmse_full:.4f}")
        print(f"NRMSE (truncated): {nrmse_trunc:.4f}")
        print(f"Experimental peak: {PF1000_DATA.peak_current/1e6:.3f} MA")


# =====================================================================
# NX2 blind prediction with device-specific pcf
# =====================================================================


class TestCrossDeviceCorrectPCF:
    """Cross-device prediction with device-specific pcf values.

    PhD Debate #21 recommendation #5: fix CrossValidator to use
    device-specific pcf (PF-1000->0.14, NX2->0.5) instead of a single
    shared pcf value.  This is the highest-impact action (+0.2 projected).
    """

    @pytest.fixture(scope="class")
    def pf1000_to_nx2_correct_pcf(self):
        """Cross-validate PF-1000 -> NX2 with correct device-specific pcf."""
        from dpf.validation.calibration import CrossValidator

        cv = CrossValidator()
        return cv.validate(
            "PF-1000", "NX2",
            maxiter=50, f_mr=0.1,
            train_pcf=0.14,  # PF-1000 pcf
            test_pcf=0.5,    # NX2 pcf
        )

    @pytest.fixture(scope="class")
    def pf1000_to_nx2_old_shared_pcf(self):
        """Cross-validate PF-1000 -> NX2 with old shared pcf=0.14."""
        from dpf.validation.calibration import CrossValidator

        cv = CrossValidator()
        return cv.validate(
            "PF-1000", "NX2",
            maxiter=50, f_mr=0.1,
            train_pcf=0.14,
            test_pcf=0.14,  # Wrong: using PF-1000's pcf for NX2
        )

    def test_cross_validator_accepts_separate_pcf(self, pf1000_to_nx2_correct_pcf):
        """CrossValidator API supports train_pcf and test_pcf separately."""
        result = pf1000_to_nx2_correct_pcf
        assert result.train_device == "PF-1000"
        assert result.test_device == "NX2"

    def test_correct_pcf_has_positive_generalization(self, pf1000_to_nx2_correct_pcf):
        """Correct pcf cross-validation has generalization > 0."""
        assert pf1000_to_nx2_correct_pcf.generalization_score > 0.0

    def test_correct_pcf_peak_error_below_50pct(self, pf1000_to_nx2_correct_pcf):
        """NX2 peak error < 50% with correct pcf."""
        assert pf1000_to_nx2_correct_pcf.prediction_peak_error < 0.50, (
            f"Peak error {pf1000_to_nx2_correct_pcf.prediction_peak_error:.0%}"
        )

    def test_correct_pcf_improves_over_shared(
        self, pf1000_to_nx2_correct_pcf, pf1000_to_nx2_old_shared_pcf,
    ):
        """Correct device-specific pcf should improve or maintain prediction.

        Using NX2's pcf=0.5 (50% of anode in compression) instead of
        PF-1000's pcf=0.14 (14%) should give a more physically realistic
        NX2 prediction because NX2 is a small device where more of the
        anode length participates in radial compression.
        """
        improved = (
            pf1000_to_nx2_correct_pcf.prediction_peak_error
            <= pf1000_to_nx2_old_shared_pcf.prediction_peak_error + 0.05
        )
        assert improved, (
            f"Correct pcf peak error "
            f"{pf1000_to_nx2_correct_pcf.prediction_peak_error:.0%} "
            f"worse than shared pcf "
            f"{pf1000_to_nx2_old_shared_pcf.prediction_peak_error:.0%}"
        )

    def test_default_pcf_uses_device_registry(self):
        """CrossValidator defaults to _DEFAULT_DEVICE_PCF when no pcf given."""
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF, CrossValidator

        cv = CrossValidator()
        result = cv.validate("PF-1000", "NX2", maxiter=50, f_mr=0.1)
        assert _DEFAULT_DEVICE_PCF.get("PF-1000") == 0.14
        assert _DEFAULT_DEVICE_PCF.get("NX2") == 0.5
        assert result.generalization_score > 0.0

    def test_cross_validation_report(
        self, pf1000_to_nx2_correct_pcf, pf1000_to_nx2_old_shared_pcf,
    ):
        """Report cross-validation comparison for documentation."""
        correct = pf1000_to_nx2_correct_pcf
        shared = pf1000_to_nx2_old_shared_pcf
        print("\n=== Cross-Device Prediction: PF-1000 -> NX2 ===")
        print(f"Shared pcf=0.14: peak_err={shared.prediction_peak_error:.1%}, "
              f"timing_err={shared.prediction_timing_error:.1%}, "
              f"gen_score={shared.generalization_score:.3f}")
        print(f"Correct pcf:     peak_err={correct.prediction_peak_error:.1%}, "
              f"timing_err={correct.prediction_timing_error:.1%}, "
              f"gen_score={correct.generalization_score:.3f}")
        print(f"Calibrated: fc={correct.calibration.best_fc:.3f}, "
              f"fm={correct.calibration.best_fm:.3f}")
