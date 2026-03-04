"""Phase BO: Multi-condition validation tests.

Tests whether Lee model parameters (fc, fm, delay) transfer across operating
conditions of the same device.  This is the strongest form of model
validation: same hardware, different V0 and fill pressure.

Multi-condition pairs:
- PF-1000 (27 kV) → PF-1000-16kV (16 kV)
- PF-1000 (27 kV, Scholz) → PF-1000-Gribkov (27 kV, Gribkov)
"""

import numpy as np
import pytest

# ── Unit tests (non-slow): data integrity, imports, structure ─────────


class TestMultiConditionDataIntegrity:
    """Verify multi-condition device entries exist and are consistent."""

    def test_pf1000_16kv_registered(self):
        from dpf.validation.experimental import DEVICES
        assert "PF-1000-16kV" in DEVICES

    def test_pf1000_gribkov_registered(self):
        from dpf.validation.experimental import DEVICES
        assert "PF-1000-Gribkov" in DEVICES

    def test_pf1000_16kv_has_waveform(self):
        from dpf.validation.experimental import DEVICES
        dev = DEVICES["PF-1000-16kV"]
        assert dev.waveform_t is not None
        assert dev.waveform_I is not None
        assert len(dev.waveform_t) > 10
        assert len(dev.waveform_I) == len(dev.waveform_t)

    def test_pf1000_gribkov_has_waveform(self):
        from dpf.validation.experimental import DEVICES
        dev = DEVICES["PF-1000-Gribkov"]
        assert dev.waveform_t is not None
        assert dev.waveform_I is not None
        assert len(dev.waveform_t) > 50  # 94-point waveform

    def test_same_bank_parameters(self):
        """PF-1000-16kV and PF-1000 share the same capacitor bank."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        assert pf27.capacitance == pf16.capacitance
        assert pf27.inductance == pf16.inductance
        assert pf27.resistance == pf16.resistance
        assert pf27.anode_radius == pf16.anode_radius
        assert pf27.cathode_radius == pf16.cathode_radius
        assert pf27.anode_length == pf16.anode_length

    def test_different_operating_conditions(self):
        """PF-1000-16kV operates at different V0 and pressure."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        assert pf16.voltage < pf27.voltage  # 16 kV < 27 kV
        assert pf16.fill_pressure_torr < pf27.fill_pressure_torr  # 1.05 < 3.5

    def test_gribkov_same_conditions(self):
        """PF-1000-Gribkov: same device, same V0/pressure, different shot."""
        from dpf.validation.experimental import DEVICES
        pf_scholz = DEVICES["PF-1000"]
        pf_gribkov = DEVICES["PF-1000-Gribkov"]
        assert pf_scholz.voltage == pf_gribkov.voltage
        assert pf_scholz.fill_pressure_torr == pf_gribkov.fill_pressure_torr
        assert pf_scholz.capacitance == pf_gribkov.capacitance

    def test_gribkov_higher_resolution(self):
        """Gribkov waveform has higher point density than Scholz."""
        from dpf.validation.experimental import DEVICES
        pf_scholz = DEVICES["PF-1000"]
        pf_gribkov = DEVICES["PF-1000-Gribkov"]
        assert len(pf_gribkov.waveform_t) > len(pf_scholz.waveform_t)

    def test_16kv_lower_peak_current(self):
        """16 kV → lower stored energy → lower peak current."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        assert pf16.peak_current < pf27.peak_current

    def test_16kv_stored_energy_ratio(self):
        """E = 0.5 * C * V^2.  16 kV → 170.5 kJ, 27 kV → 486 kJ."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        E27 = 0.5 * pf27.capacitance * pf27.voltage**2
        E16 = 0.5 * pf16.capacitance * pf16.voltage**2
        # 16/27 kV → energy ratio ~ (16/27)^2 ≈ 0.35
        assert 0.30 < E16 / E27 < 0.40

    def test_crowbar_resistance_lookup_16kv(self):
        """PF-1000-16kV should use same crowbar as PF-1000."""
        from dpf.validation.calibration import _DEFAULT_CROWBAR_R
        assert "PF-1000-16kV" in _DEFAULT_CROWBAR_R
        assert _DEFAULT_CROWBAR_R["PF-1000-16kV"] == _DEFAULT_CROWBAR_R["PF-1000"]

    def test_pcf_lookup_16kv(self):
        """PF-1000-16kV should use same pcf as PF-1000."""
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF
        assert "PF-1000-16kV" in _DEFAULT_DEVICE_PCF
        assert _DEFAULT_DEVICE_PCF["PF-1000-16kV"] == _DEFAULT_DEVICE_PCF["PF-1000"]


class TestMultiConditionImports:
    """Verify the multi_condition_validation function is importable."""

    def test_import_function(self):
        from dpf.validation.calibration import multi_condition_validation
        assert callable(multi_condition_validation)

    def test_import_result_class(self):
        from dpf.validation.calibration import MultiConditionResult
        assert MultiConditionResult is not None

    def test_result_fields(self):
        from dpf.validation.calibration import MultiConditionResult
        r = MultiConditionResult(
            train_device="PF-1000",
            test_device="PF-1000-16kV",
            train_fc=0.8,
            train_fm=0.1,
            train_delay_us=0.5,
            train_nrmse=0.10,
            blind_nrmse=0.20,
            independent_nrmse=0.08,
            degradation=2.5,
        )
        assert r.train_device == "PF-1000"
        assert r.test_device == "PF-1000-16kV"
        assert r.degradation == pytest.approx(2.5)
        assert r.asme_blind is None  # Optional

    def test_invalid_device_raises(self):
        from dpf.validation.calibration import multi_condition_validation
        with pytest.raises(ValueError, match="not in DEVICES"):
            multi_condition_validation(
                train_device="NONEXISTENT",
                test_device="PF-1000-16kV",
                maxiter=1,
            )

    def test_no_waveform_raises(self):
        """PF-1000-20kV has no waveform → should raise."""
        from dpf.validation.calibration import multi_condition_validation
        with pytest.raises(ValueError, match="no digitized waveform"):
            multi_condition_validation(
                train_device="PF-1000",
                test_device="PF-1000-20kV",
                maxiter=1,
            )


class TestMultiConditionPhysics:
    """Physics consistency tests for multi-condition pairs."""

    def test_quarter_period_same(self):
        """T/4 depends only on bank (C0, L0), not V0.  Both PF-1000 entries
        share the same bank, so T/4 must be identical."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        T4_27 = np.pi / 2 * np.sqrt(pf27.inductance * pf27.capacitance)
        T4_16 = np.pi / 2 * np.sqrt(pf16.inductance * pf16.capacitance)
        assert pytest.approx(T4_16, rel=1e-10) == T4_27

    def test_peak_rlc_current_scales_with_v0(self):
        """I_peak_RLC ~ V0/sqrt(L0/C0).  Ratio should be 16/27."""
        from dpf.validation.experimental import DEVICES
        pf27 = DEVICES["PF-1000"]
        pf16 = DEVICES["PF-1000-16kV"]
        # Peak RLC (unloaded) current scales linearly with V0
        # since impedance Z0 = sqrt(L0/C0) is the same
        actual_ratio = pf16.peak_current / pf27.peak_current
        # Actual ratio won't be exact 16/27 because of plasma loading,
        # but should be in reasonable range
        assert 0.3 < actual_ratio < 0.8

    def test_scholz_gribkov_peak_within_shot_to_shot(self):
        """Scholz and Gribkov peaks should be within shot-to-shot variation."""
        from dpf.validation.experimental import DEVICES
        pf_scholz = DEVICES["PF-1000"]
        pf_gribkov = DEVICES["PF-1000-Gribkov"]
        ratio = pf_gribkov.peak_current / pf_scholz.peak_current
        # Same conditions → peaks within ~10% shot-to-shot variation
        assert 0.85 < ratio < 1.15

    def test_waveform_monotonic_rise(self):
        """All PF-1000 variant waveforms should have a monotonic rise phase."""
        from dpf.validation.experimental import DEVICES
        for name in ("PF-1000", "PF-1000-16kV", "PF-1000-Gribkov"):
            dev = DEVICES[name]
            waveform = dev.waveform_I
            # First half should generally increase (allowing small fluctuations)
            n_half = len(waveform) // 2
            # Net increase from start to midpoint
            assert waveform[n_half] > waveform[0], f"{name}: current should increase in first half"


class TestLeeModelMultiConditionPredict:
    """Test Lee model can run on multi-condition devices."""

    def test_lee_model_runs_pf1000_16kv(self):
        """Lee model runs on PF-1000-16kV without crashing."""
        from dpf.validation.lee_model_comparison import LeeModel
        model = LeeModel(
            current_fraction=0.8,
            mass_fraction=0.1,
            liftoff_delay=0.5e-6,
        )
        result = model.run("PF-1000-16kV")
        assert result is not None
        assert len(result.t) > 10
        assert np.max(result.I) > 0

    def test_lee_model_runs_gribkov(self):
        """Lee model runs on PF-1000-Gribkov without crashing."""
        from dpf.validation.lee_model_comparison import LeeModel
        model = LeeModel(
            current_fraction=0.8,
            mass_fraction=0.1,
            liftoff_delay=0.5e-6,
        )
        result = model.run("PF-1000-Gribkov")
        assert result is not None
        assert len(result.t) > 10
        assert np.max(result.I) > 0

    def test_nrmse_computable_16kv(self):
        """NRMSE can be computed for PF-1000-16kV."""
        from dpf.validation.experimental import DEVICES, nrmse_peak
        from dpf.validation.lee_model_comparison import LeeModel
        dev = DEVICES["PF-1000-16kV"]
        model = LeeModel(current_fraction=0.8, mass_fraction=0.1)
        result = model.run("PF-1000-16kV")
        nrmse = nrmse_peak(result.t, result.I, dev.waveform_t, dev.waveform_I)
        assert 0.0 < nrmse < 1.0

    def test_nrmse_computable_gribkov(self):
        """NRMSE can be computed for PF-1000-Gribkov."""
        from dpf.validation.experimental import DEVICES, nrmse_peak
        from dpf.validation.lee_model_comparison import LeeModel
        dev = DEVICES["PF-1000-Gribkov"]
        model = LeeModel(current_fraction=0.8, mass_fraction=0.1)
        result = model.run("PF-1000-Gribkov")
        nrmse = nrmse_peak(result.t, result.I, dev.waveform_t, dev.waveform_I)
        assert 0.0 < nrmse < 1.0


# ── Slow tests: actual calibration + prediction ─────────────────────


@pytest.mark.slow
class TestMultiCondition27to16kV:
    """Multi-condition: calibrate PF-1000 (27 kV), predict PF-1000-16kV."""

    def test_multi_condition_runs(self):
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-16kV",
            maxiter=10,
            run_asme=True,
        )
        assert result.train_device == "PF-1000"
        assert result.test_device == "PF-1000-16kV"
        assert result.blind_nrmse > 0
        assert result.independent_nrmse > 0
        assert result.degradation > 0

    def test_blind_nrmse_below_50pct(self):
        """Blind prediction should be < 50% NRMSE (reasonable transfer)."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-16kV",
            maxiter=10,
            run_asme=False,
        )
        assert result.blind_nrmse < 0.50

    def test_degradation_bounded(self):
        """Degradation should be < 10x (not catastrophic)."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-16kV",
            maxiter=10,
            run_asme=False,
        )
        assert result.degradation < 10.0

    def test_asme_assessment_runs(self):
        """ASME V&V 20 assessment produces valid result."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-16kV",
            maxiter=10,
            run_asme=True,
        )
        assert result.asme_blind is not None
        assert result.asme_blind.E > 0
        assert result.asme_blind.u_val > 0
        assert result.asme_blind.ratio > 0


@pytest.mark.slow
class TestMultiConditionScholzGribkov:
    """Cross-publication: calibrate on Scholz, predict Gribkov."""

    def test_multi_condition_runs(self):
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-Gribkov",
            maxiter=10,
            run_asme=True,
        )
        assert result.train_device == "PF-1000"
        assert result.test_device == "PF-1000-Gribkov"

    def test_cross_pub_low_degradation(self):
        """Same device + conditions → degradation should be < 3x."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-Gribkov",
            maxiter=10,
            run_asme=False,
        )
        # Same conditions, different shot → should transfer well
        assert result.degradation < 3.0

    def test_blind_nrmse_below_30pct(self):
        """Same conditions → blind NRMSE should be < 30%."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000",
            test_device="PF-1000-Gribkov",
            maxiter=10,
            run_asme=False,
        )
        assert result.blind_nrmse < 0.30


@pytest.mark.slow
class TestMultiConditionReverse:
    """Reverse direction: calibrate on 16 kV, predict 27 kV."""

    def test_reverse_runs(self):
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000-16kV",
            test_device="PF-1000",
            maxiter=10,
            run_asme=True,
        )
        assert result.train_device == "PF-1000-16kV"
        assert result.test_device == "PF-1000"

    def test_reverse_blind_nrmse_below_50pct(self):
        """Reverse direction should also transfer reasonably."""
        from dpf.validation.calibration import multi_condition_validation
        result = multi_condition_validation(
            train_device="PF-1000-16kV",
            test_device="PF-1000",
            maxiter=10,
            run_asme=False,
        )
        assert result.blind_nrmse < 0.50
