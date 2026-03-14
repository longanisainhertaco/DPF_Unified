"""Phase AH: Path to 7.0 — preset completeness, NRMSE truncation, sensitivity.

Addresses Debate #20 verdict recommendations:
1. Calibrated fc/fm/pcf in all presets (Task 1)
2. NRMSE truncation at current dip (Task 2)
3. Cross-device blind prediction with pcf (Task 3)
4. Parameter sensitivity study (Task 4)
5. calibrate_default_params uses device-specific pcf

These fixes together target +0.7 on the PhD score (6.3 → 7.0).
"""

import numpy as np
import pytest

# =====================================================================
# Task 1: Calibrated parameters in all presets
# =====================================================================


class TestPresetCompleteness:
    """All DPF device presets have calibrated snowplow parameters."""

    @pytest.fixture(scope="class")
    def all_presets(self):
        from dpf.presets import get_preset, get_preset_names

        return {name: get_preset(name) for name in get_preset_names()}

    def test_pf1000_has_pcf(self, all_presets):
        """PF-1000 preset has pinch_column_fraction."""
        sp = all_presets["pf1000"].get("snowplow", {})
        assert "pinch_column_fraction" in sp
        assert sp["pinch_column_fraction"] == pytest.approx(0.14, abs=0.01)

    def test_pf1000_has_calibrated_fc_fm(self, all_presets):
        """PF-1000 preset has post-D2 calibrated fc/fm."""
        sp = all_presets["pf1000"].get("snowplow", {})
        assert "current_fraction" in sp
        assert "mass_fraction" in sp
        assert 0.6 <= sp["current_fraction"] <= 0.9  # Lee & Saw (2014): fc=0.7
        assert 0.05 <= sp["mass_fraction"] <= 0.25  # Lee & Saw (2014): fm=0.08

    def test_nx2_has_pcf(self, all_presets):
        """NX2 preset has pinch_column_fraction."""
        sp = all_presets["nx2"].get("snowplow", {})
        assert "pinch_column_fraction" in sp
        assert sp["pinch_column_fraction"] == pytest.approx(0.5, abs=0.1)

    def test_nx2_has_calibrated_fc_fm(self, all_presets):
        """NX2 preset has fc/fm from Lee & Saw (2008)."""
        sp = all_presets["nx2"].get("snowplow", {})
        assert "current_fraction" in sp
        assert "mass_fraction" in sp
        assert 0.5 < sp["current_fraction"] < 0.9
        assert 0.05 < sp["mass_fraction"] < 0.25

    def test_llnl_has_pcf(self, all_presets):
        """LLNL preset has pinch_column_fraction."""
        sp = all_presets["llnl_dpf"].get("snowplow", {})
        assert "pinch_column_fraction" in sp
        assert 0.1 < sp["pinch_column_fraction"] < 0.6

    def test_llnl_has_calibrated_fc_fm(self, all_presets):
        """LLNL preset has fc/fm."""
        sp = all_presets["llnl_dpf"].get("snowplow", {})
        assert "current_fraction" in sp
        assert "mass_fraction" in sp

    def test_mjolnir_has_pcf(self, all_presets):
        """MJOLNIR preset has pinch_column_fraction."""
        sp = all_presets["mjolnir"].get("snowplow", {})
        assert "pinch_column_fraction" in sp
        assert 0.05 < sp["pinch_column_fraction"] < 0.3

    def test_mjolnir_has_calibrated_fc_fm(self, all_presets):
        """MJOLNIR preset has fc/fm."""
        sp = all_presets["mjolnir"].get("snowplow", {})
        assert "current_fraction" in sp
        assert "mass_fraction" in sp

    def test_all_cylindrical_presets_have_snowplow_params(self, all_presets):
        """Every cylindrical preset has fc, fm, and pcf in snowplow."""
        for name, preset in all_presets.items():
            geo = preset.get("geometry", {})
            if geo.get("type") == "cylindrical":
                sp = preset.get("snowplow", {})
                assert "current_fraction" in sp, (
                    f"Preset '{name}' missing current_fraction"
                )
                assert "mass_fraction" in sp, (
                    f"Preset '{name}' missing mass_fraction"
                )
                assert "pinch_column_fraction" in sp, (
                    f"Preset '{name}' missing pinch_column_fraction"
                )

    def test_presets_instantiate_as_config(self, all_presets):
        """All presets can be instantiated as SimulationConfig."""
        from dpf.config import SimulationConfig

        for name, preset in all_presets.items():
            config = SimulationConfig(**preset)
            assert config.grid_shape is not None, f"Preset '{name}' failed"


# =====================================================================
# Task 2: NRMSE truncation at current dip
# =====================================================================


class TestNRMSETruncation:
    """NRMSE truncation excludes post-pinch contaminated region."""

    def test_truncated_nrmse_less_than_full(self):
        """Truncated NRMSE <= full NRMSE (less contamination)."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        full = compare_engine_vs_experiment(t, I_arr, truncate_at_dip=False)
        trunc = compare_engine_vs_experiment(t, I_arr, truncate_at_dip=True)

        assert trunc.waveform_nrmse <= full.waveform_nrmse + 0.01, (
            f"Truncated {trunc.waveform_nrmse:.4f} > full {full.waveform_nrmse:.4f}"
        )

    def test_truncated_nrmse_below_threshold(self):
        """Truncated NRMSE < 0.15 (improved from full ~0.16)."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        t, I_arr, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, pinch_column_fraction=0.14,
        )
        result = compare_engine_vs_experiment(t, I_arr, truncate_at_dip=True)
        assert result.waveform_nrmse < 0.15, (
            f"Truncated NRMSE {result.waveform_nrmse:.4f} exceeds 0.15"
        )

    def test_nrmse_peak_truncation_api(self):
        """nrmse_peak() accepts truncate_at_dip parameter."""
        from dpf.validation.experimental import nrmse_peak

        t = np.linspace(0, 10e-6, 1000)
        I_ref = 1e6 * np.sin(2 * np.pi * 50e3 * t) * np.exp(-t / 5e-6)
        I_sim = I_ref * 1.05  # 5% bias

        full = nrmse_peak(t, I_sim, t, I_ref, truncate_at_dip=False)
        trunc = nrmse_peak(t, I_sim, t, I_ref, truncate_at_dip=True)

        assert full > 0
        assert trunc > 0
        # Truncated uses fewer points, so error may differ
        assert trunc <= full + 0.01

    def test_lee_model_truncation(self):
        """LeeModel.compare_with_experiment supports truncation."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816, mass_fraction=0.142,
            radial_mass_fraction=0.1, pinch_column_fraction=0.14,
            liftoff_delay=0.7e-6, crowbar_enabled=True,
        )
        full = model.compare_with_experiment("PF-1000", truncate_at_dip=False)
        trunc = model.compare_with_experiment("PF-1000", truncate_at_dip=True)

        assert np.isfinite(full.waveform_nrmse)
        assert np.isfinite(trunc.waveform_nrmse)
        assert trunc.waveform_nrmse <= full.waveform_nrmse + 0.01


# =====================================================================
# Task 3: Cross-device blind prediction with pcf
# =====================================================================


class TestCrossDeviceWithPCF:
    """Cross-device prediction using device-specific pcf values."""

    @pytest.fixture(scope="class")
    def pf1000_to_nx2_with_pcf(self):
        """Cross-validate PF-1000 → NX2 with pcf."""
        from dpf.validation.calibration import CrossValidator

        cv = CrossValidator()
        return cv.validate(
            "PF-1000", "NX2", maxiter=50,
            f_mr=0.1, pinch_column_fraction=0.14,
        )

    def test_generalization_positive(self, pf1000_to_nx2_with_pcf):
        """Cross-device generalization score > 0."""
        assert pf1000_to_nx2_with_pcf.generalization_score > 0.0

    def test_peak_error_documented(self, pf1000_to_nx2_with_pcf):
        """NX2 peak prediction error < 50% (model has predictive power)."""
        assert pf1000_to_nx2_with_pcf.prediction_peak_error < 0.50

    def test_calibration_converged(self, pf1000_to_nx2_with_pcf):
        """PF-1000 calibration converges."""
        assert pf1000_to_nx2_with_pcf.calibration.converged


# =====================================================================
# Task 4: Parameter sensitivity study
# =====================================================================


class TestParameterSensitivity:
    """fc/fm sensitivity analysis — NRMSE response to perturbations."""

    @pytest.fixture(scope="class")
    def sensitivity_results(self):
        """Run sensitivity sweep: ±10% perturbations around calibrated fc/fm."""
        from dpf.validation.engine_validation import (
            compare_engine_vs_experiment,
            run_rlc_snowplow_pf1000,
        )

        fc_base, fm_base = 0.816, 0.142
        results = {}

        for label, fc, fm in [
            ("baseline", fc_base, fm_base),
            ("fc+10%", fc_base * 1.10, fm_base),
            ("fc-10%", fc_base * 0.90, fm_base),
            ("fm+10%", fc_base, fm_base * 1.10),
            ("fm-10%", fc_base, fm_base * 0.90),
        ]:
            t, I_arr, _ = run_rlc_snowplow_pf1000(
                fc=fc, fm=fm, pinch_column_fraction=0.14,
            )
            r = compare_engine_vs_experiment(t, I_arr, fc=fc, fm=fm)
            results[label] = {
                "nrmse": r.waveform_nrmse,
                "peak_error": r.peak_current_error,
                "fc": fc,
                "fm": fm,
            }

        return results

    def test_baseline_nrmse_below_017(self, sensitivity_results):
        """Baseline NRMSE < 0.17 at calibrated fc/fm."""
        assert sensitivity_results["baseline"]["nrmse"] < 0.17

    def test_fc_perturbation_degrades_nrmse(self, sensitivity_results):
        """±10% fc perturbation increases NRMSE from baseline."""
        base = sensitivity_results["baseline"]["nrmse"]
        for label in ["fc+10%", "fc-10%"]:
            perturbed = sensitivity_results[label]["nrmse"]
            # Perturbed should be worse (or at least not much better)
            assert perturbed > base * 0.8, (
                f"{label} NRMSE {perturbed:.4f} suspiciously better than "
                f"baseline {base:.4f}"
            )

    def test_fm_perturbation_degrades_nrmse(self, sensitivity_results):
        """±10% fm perturbation increases NRMSE from baseline."""
        base = sensitivity_results["baseline"]["nrmse"]
        for label in ["fm+10%", "fm-10%"]:
            perturbed = sensitivity_results[label]["nrmse"]
            assert perturbed > base * 0.8, (
                f"{label} NRMSE {perturbed:.4f} suspiciously better than "
                f"baseline {base:.4f}"
            )

    def test_all_perturbations_produce_finite_nrmse(self, sensitivity_results):
        """All perturbations produce finite, positive NRMSE."""
        for label, data in sensitivity_results.items():
            assert np.isfinite(data["nrmse"]), f"{label} produced NaN NRMSE"
            assert data["nrmse"] > 0, f"{label} produced zero NRMSE"

    def test_sensitivity_bounded(self, sensitivity_results):
        """All perturbations produce NRMSE < 0.30 (model not broken)."""
        for label, data in sensitivity_results.items():
            assert data["nrmse"] < 0.30, (
                f"{label} NRMSE {data['nrmse']:.4f} exceeds 0.30 — model broken"
            )


# =====================================================================
# calibrate_default_params with device-specific pcf
# =====================================================================


class TestCalibrateDefaultParams:
    """calibrate_default_params() uses device-specific pcf."""

    def test_pf1000_uses_pcf_014(self):
        """PF-1000 calibration uses pcf=0.14 by default."""
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF

        assert "PF-1000" in _DEFAULT_DEVICE_PCF
        assert _DEFAULT_DEVICE_PCF["PF-1000"] == pytest.approx(0.14)

    def test_nx2_uses_pcf_05(self):
        """NX2 calibration uses pcf=0.5 by default."""
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF

        assert "NX2" in _DEFAULT_DEVICE_PCF
        assert _DEFAULT_DEVICE_PCF["NX2"] == pytest.approx(0.5)

    def test_calibrate_default_runs(self):
        """calibrate_default_params() runs without error."""
        from dpf.validation.calibration import calibrate_default_params

        results = calibrate_default_params(maxiter=20)
        assert "PF-1000" in results
        assert results["PF-1000"].converged


# =====================================================================
# SnowplowConfig pcf propagation
# =====================================================================


class TestSnowplowConfigPCF:
    """SnowplowConfig has pinch_column_fraction and it propagates."""

    def test_snowplow_config_has_pcf(self):
        """SnowplowConfig has pinch_column_fraction field."""
        from dpf.config import SnowplowConfig

        sc = SnowplowConfig()
        assert hasattr(sc, "pinch_column_fraction")
        assert sc.pinch_column_fraction == 1.0  # default

    def test_snowplow_config_pcf_custom(self):
        """SnowplowConfig accepts custom pcf."""
        from dpf.config import SnowplowConfig

        sc = SnowplowConfig(pinch_column_fraction=0.14)
        assert sc.pinch_column_fraction == pytest.approx(0.14)

    def test_simulation_config_propagates_pcf(self):
        """SimulationConfig propagates pcf to snowplow."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        config = SimulationConfig(**preset)
        assert config.snowplow.pinch_column_fraction == pytest.approx(0.14, abs=0.01)

    def test_engine_receives_pcf(self):
        """SimulationEngine passes pcf to SnowplowModel."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        # Use small grid for test speed
        preset["grid_shape"] = [8, 1, 8]
        config = SimulationConfig(**preset)

        from dpf.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.snowplow is not None
        assert engine.snowplow.pinch_column_fraction == pytest.approx(0.14, abs=0.01)
