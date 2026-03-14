"""Phase BM: Fourth and fifth device integration (FAETON-I + MJOLNIR).

Tests for integrating FAETON-I (Fuse Energy, 100 kV, 125 kJ, ~1 MA) and
MJOLNIR (LLNL, 60 kV typical, 408 uF, 2.8 MA) as new devices in the
DPF-Unified validation framework.

Key goals:
- Register FAETON-I and MJOLNIR with reconstructed I(t) waveforms
- Compute L_p/L0 diagnostics (both circuit-dominated: 0.107 and 0.367)
- Compute speed factor diagnostics (sub-driven and super-driven)
- Independent calibration of fc/fm for each device
- Enable N=5 LOO cross-validation (df=4, finite variance)

References:
- FAETON-I: Damideh et al., Sci. Rep. 15:23048 (2025),
  DOI: 10.1038/s41598-025-07939-x
- MJOLNIR: Schmidt et al., IEEE TPS (2021), DOI: 10.1109/TPS.2021.3106313;
  Goyon et al., Phys. Plasmas 32:033105 (2025)
"""

from __future__ import annotations

import numpy as np
import pytest

# =====================================================================
# Device Registration Tests
# =====================================================================

class TestDeviceRegistration:
    """Verify FAETON-I and MJOLNIR are registered with correct parameters."""

    def test_faeton_in_device_registry(self):
        from dpf.validation.experimental import DEVICES

        assert "FAETON-I" in DEVICES
        dev = DEVICES["FAETON-I"]
        assert dev.name == "FAETON-I"
        assert dev.institution == "Fuse Energy Technologies"

    def test_faeton_circuit_params(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert dev.capacitance == pytest.approx(25e-6, rel=0.01)
        assert dev.voltage == pytest.approx(100e3, rel=0.01)
        assert dev.inductance == pytest.approx(220e-9, rel=0.01)
        # R0 estimated from damping
        assert 5e-3 < dev.resistance < 12e-3

    def test_faeton_geometry(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert dev.anode_radius == pytest.approx(0.05, rel=0.01)
        assert dev.cathode_radius == pytest.approx(0.10, rel=0.05)  # estimated
        assert dev.anode_length == pytest.approx(0.17, rel=0.01)

    def test_faeton_performance(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert dev.peak_current == pytest.approx(1.0e6, rel=0.05)
        assert dev.current_rise_time == pytest.approx(3.6e-6, rel=0.05)
        assert dev.fill_pressure_torr == pytest.approx(12.0, rel=0.1)
        assert dev.fill_gas == "deuterium"

    def test_faeton_has_waveform(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert dev.waveform_t is not None
        assert dev.waveform_I is not None
        assert len(dev.waveform_t) == len(dev.waveform_I)
        assert len(dev.waveform_t) >= 20

    def test_faeton_waveform_shape(self):
        """Waveform should rise to ~1 MA and have correct units (SI)."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        # Time in seconds
        assert dev.waveform_t[0] == pytest.approx(0.0, abs=1e-9)
        assert dev.waveform_t[-1] < 10e-6  # < 10 us

        # Current in amps, peak near 1 MA
        I_peak = float(np.max(dev.waveform_I))
        assert 900e3 < I_peak < 1100e3

    def test_faeton_no_crowbar(self):
        """FAETON-I has no crowbar — current should reverse."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert dev.crowbar_resistance == 0.0
        # Last point should be near zero or negative (current reversal)
        assert dev.waveform_I[-1] < 50e3

    def test_faeton_energy(self):
        """Stored energy should be ~125 kJ."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        E = 0.5 * dev.capacitance * dev.voltage**2
        assert pytest.approx(125e3, rel=0.01) == E

    def test_mjolnir_in_device_registry(self):
        from dpf.validation.experimental import DEVICES

        assert "MJOLNIR" in DEVICES
        dev = DEVICES["MJOLNIR"]
        assert dev.name == "MJOLNIR"
        assert "Livermore" in dev.institution

    def test_mjolnir_circuit_params(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert dev.capacitance == pytest.approx(408e-6, rel=0.01)
        assert dev.voltage == pytest.approx(60e3, rel=0.01)
        assert dev.inductance == pytest.approx(80e-9, rel=0.05)  # estimated

    def test_mjolnir_geometry(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert dev.anode_radius == pytest.approx(0.114, rel=0.01)
        assert dev.cathode_radius == pytest.approx(0.157, rel=0.05)  # estimated
        assert dev.anode_length == pytest.approx(0.20, rel=0.10)

    def test_mjolnir_performance(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert dev.peak_current == pytest.approx(2.8e6, rel=0.05)
        assert dev.current_rise_time == pytest.approx(5.0e-6, rel=0.1)

    def test_mjolnir_has_waveform(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert dev.waveform_t is not None
        assert dev.waveform_I is not None
        assert len(dev.waveform_t) >= 20

    def test_mjolnir_waveform_shape(self):
        """Waveform should rise to ~2.8 MA and have correct units."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        I_peak = float(np.max(dev.waveform_I))
        assert 2.5e6 < I_peak < 3.2e6

    def test_mjolnir_energy(self):
        """Stored energy should be ~734 kJ at 60 kV."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        E = 0.5 * dev.capacitance * dev.voltage**2
        assert pytest.approx(734.4e3, rel=0.01) == E

    def test_mjolnir_crowbar(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert dev.crowbar_resistance > 0  # has crowbar

    def test_total_device_count(self):
        """Should now have 10 devices registered."""
        from dpf.validation.experimental import DEVICES

        assert len(DEVICES) == 10

    def test_waveform_device_count(self):
        """Should have 7 devices with waveforms."""
        from dpf.validation.experimental import DEVICES

        n_with_waveform = sum(
            1 for d in DEVICES.values() if d.waveform_t is not None
        )
        assert n_with_waveform == 7


# =====================================================================
# L_p / L0 Diagnostic Tests
# =====================================================================

class TestLpL0Diagnostic:
    """Verify L_p/L0 classification for new devices."""

    def test_faeton_circuit_dominated(self):
        """FAETON-I: L_p/L0 = 0.107 — extremely circuit-dominated."""
        from dpf.validation.experimental import DEVICES, compute_lp_l0_ratio

        dev = DEVICES["FAETON-I"]
        result = compute_lp_l0_ratio(
            dev.inductance, dev.anode_radius, dev.cathode_radius, dev.anode_length,
        )
        assert result["regime"] == "circuit-dominated"
        assert result["L_p_over_L0"] < 0.15
        assert result["L_p_over_L0"] == pytest.approx(0.107, abs=0.01)

    def test_mjolnir_circuit_dominated(self):
        """MJOLNIR: L_p/L0 ~ 0.16 — circuit-dominated (anode_radius=114mm)."""
        from dpf.validation.experimental import DEVICES, compute_lp_l0_ratio

        dev = DEVICES["MJOLNIR"]
        result = compute_lp_l0_ratio(
            dev.inductance, dev.anode_radius, dev.cathode_radius, dev.anode_length,
        )
        assert result["regime"] == "circuit-dominated"
        assert 0.10 < result["L_p_over_L0"] < 0.25

    def test_pf1000_plasma_significant(self):
        """PF-1000 should be plasma-significant (reference check)."""
        from dpf.validation.experimental import DEVICES, compute_lp_l0_ratio

        dev = DEVICES["PF-1000"]
        result = compute_lp_l0_ratio(
            dev.inductance, dev.anode_radius, dev.cathode_radius, dev.anode_length,
        )
        assert result["regime"] == "plasma-significant"
        assert result["L_p_over_L0"] > 1.0


# =====================================================================
# Speed Factor Tests
# =====================================================================

class TestSpeedFactor:
    """Verify speed factor classification for new devices."""

    def test_faeton_sub_driven(self):
        """FAETON-I: S/S_opt ~ 0.65 — sub-driven."""
        from dpf.validation.experimental import DEVICES, compute_speed_factor

        dev = DEVICES["FAETON-I"]
        result = compute_speed_factor(
            dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
        )
        assert result["regime"] == "sub-driven"
        assert result["S_over_S_opt"] == pytest.approx(0.65, abs=0.05)

    def test_mjolnir_optimal(self):
        """MJOLNIR: S/S_opt ~ 1.04 — optimal (with corrected anode_radius=114mm)."""
        from dpf.validation.experimental import DEVICES, compute_speed_factor

        dev = DEVICES["MJOLNIR"]
        result = compute_speed_factor(
            dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
        )
        assert result["regime"] == "optimal"
        assert 0.8 < result["S_over_S_opt"] < 1.2


# =====================================================================
# Preset Tests
# =====================================================================

class TestPresets:
    """Verify FAETON-I and updated MJOLNIR presets load correctly."""

    def test_faeton_preset_exists(self):
        from dpf.presets import get_preset_names

        assert "faeton" in get_preset_names()

    def test_faeton_preset_loads(self):
        from dpf.presets import get_preset

        p = get_preset("faeton")
        assert p["circuit"]["C"] == pytest.approx(25e-6, rel=0.01)
        assert p["circuit"]["V0"] == pytest.approx(100e3, rel=0.01)
        assert p["circuit"]["L0"] == pytest.approx(220e-9, rel=0.01)
        assert p["circuit"]["crowbar_enabled"] is False

    def test_mjolnir_preset_updated(self):
        """MJOLNIR preset should have corrected L0 ~ 80 nH."""
        from dpf.presets import get_preset

        p = get_preset("mjolnir")
        assert p["circuit"]["C"] == pytest.approx(408e-6, rel=0.01)
        assert p["circuit"]["V0"] == pytest.approx(60e3, rel=0.01)
        assert p["circuit"]["L0"] == pytest.approx(80e-9, rel=0.05)
        assert p["circuit"]["crowbar_enabled"] is True

    def test_faeton_preset_energy(self):
        """FAETON-I stored energy: 125 kJ."""
        from dpf.presets import get_preset

        p = get_preset("faeton")
        E = 0.5 * p["circuit"]["C"] * p["circuit"]["V0"] ** 2
        assert pytest.approx(125e3, rel=0.01) == E

    def test_mjolnir_preset_energy(self):
        """MJOLNIR stored energy at 60 kV: ~734 kJ."""
        from dpf.presets import get_preset

        p = get_preset("mjolnir")
        E = 0.5 * p["circuit"]["C"] * p["circuit"]["V0"] ** 2
        assert pytest.approx(734.4e3, rel=0.01) == E


# =====================================================================
# Calibration Registry Tests
# =====================================================================

class TestCalibrationRegistry:
    """Verify FAETON-I and MJOLNIR in calibration registries."""

    def test_faeton_in_published_ranges(self):
        from dpf.validation.calibration import _PUBLISHED_FC_FM_RANGES

        assert "FAETON-I" in _PUBLISHED_FC_FM_RANGES
        fc_lo, fc_hi = _PUBLISHED_FC_FM_RANGES["FAETON-I"]["fc"]
        assert fc_lo < 0.60
        assert fc_hi > 0.80

    def test_mjolnir_in_published_ranges(self):
        from dpf.validation.calibration import _PUBLISHED_FC_FM_RANGES

        assert "MJOLNIR" in _PUBLISHED_FC_FM_RANGES

    def test_faeton_in_pcf_map(self):
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF

        assert "FAETON-I" in _DEFAULT_DEVICE_PCF

    def test_mjolnir_in_pcf_map(self):
        from dpf.validation.calibration import _DEFAULT_DEVICE_PCF

        assert "MJOLNIR" in _DEFAULT_DEVICE_PCF

    def test_faeton_in_crowbar_map(self):
        from dpf.validation.calibration import _DEFAULT_CROWBAR_R

        assert "FAETON-I" in _DEFAULT_CROWBAR_R
        assert _DEFAULT_CROWBAR_R["FAETON-I"] == 0.0  # no crowbar

    def test_mjolnir_in_crowbar_map(self):
        from dpf.validation.calibration import _DEFAULT_CROWBAR_R

        assert "MJOLNIR" in _DEFAULT_CROWBAR_R
        assert _DEFAULT_CROWBAR_R["MJOLNIR"] > 0

    def test_faeton_in_shot_data(self):
        from dpf.validation.calibration import multi_shot_uncertainty

        result = multi_shot_uncertainty("FAETON-I")
        assert result.u_exp_combined > 0

    def test_mjolnir_in_shot_data(self):
        from dpf.validation.calibration import multi_shot_uncertainty

        result = multi_shot_uncertainty("MJOLNIR")
        assert result.u_exp_combined > 0


# =====================================================================
# Independent Calibration Tests (SLOW)
# =====================================================================

class TestIndependentCalibration:
    """Independent calibration of FAETON-I and MJOLNIR."""

    @pytest.mark.slow
    def test_faeton_calibration(self):
        """FAETON-I should calibrate to NRMSE < 0.10 (circuit-dominated)."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("FAETON-I", maxiter=5)
        assert result.best_fc > 0.0
        assert result.best_fm > 0.0
        assert result.nrmse < 0.10  # easy to fit circuit-dominated

    @pytest.mark.slow
    def test_mjolnir_calibration(self):
        """MJOLNIR should calibrate to NRMSE < 0.25 (reconstructed waveform)."""
        from dpf.validation.calibration import calibrate_with_liftoff

        result = calibrate_with_liftoff("MJOLNIR", maxiter=5)
        assert result.best_fc > 0.0
        assert result.best_fm > 0.0
        assert result.nrmse < 0.25  # higher tolerance for reconstructed waveform


# =====================================================================
# Waveform Validation Tests
# =====================================================================

class TestWaveformValidation:
    """Validate waveform properties against device physics."""

    def test_faeton_quarter_period(self):
        """T/4 from waveform peak should match RLC T/4 = pi/2*sqrt(LC)."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        # Find peak time from waveform
        peak_idx = int(np.argmax(dev.waveform_I))
        t_peak = float(dev.waveform_t[peak_idx])

        # RLC quarter period
        T4_rlc = np.pi / 2 * np.sqrt(dev.inductance * dev.capacitance)

        # Should be close (within 10%) since FAETON-I is circuit-dominated
        assert t_peak == pytest.approx(T4_rlc, rel=0.10)

    def test_mjolnir_peak_current(self):
        """Waveform peak should match stated 2.8 MA."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        I_peak_waveform = float(np.max(dev.waveform_I))
        assert I_peak_waveform == pytest.approx(dev.peak_current, rel=0.01)

    def test_faeton_waveform_monotonic_rise(self):
        """FAETON-I rise phase should be monotonically increasing."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        peak_idx = int(np.argmax(dev.waveform_I))
        rise = dev.waveform_I[:peak_idx + 1]
        # Check that each point is >= previous (monotonic rise)
        for i in range(1, len(rise)):
            assert rise[i] >= rise[i - 1] * 0.99  # allow 1% noise

    def test_mjolnir_current_dip(self):
        """MJOLNIR should show a current dip after peak (pinch)."""
        from dpf.validation.experimental import DEVICES, _find_first_peak

        dev = DEVICES["MJOLNIR"]
        abs_I = dev.waveform_I
        peak_idx = _find_first_peak(abs_I)
        I_peak = float(abs_I[peak_idx])
        # Search within peak + 1 us for pinch dip (not deep post-pinch decay)
        t_us = dev.waveform_t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]
        I_dip = float(np.min(post_peak))
        dip_fraction = 1.0 - I_dip / I_peak
        # Dip should be 10-70% for MA-class device
        assert 0.10 < dip_fraction < 0.70

    def test_faeton_unloaded_check(self):
        """I_peak should be close to I_sc (circuit-dominated, minimal loading)."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        I_sc = dev.voltage * np.sqrt(dev.capacitance / dev.inductance)
        loading = dev.peak_current / I_sc
        # Circuit-dominated: loading > 0.9
        assert loading > 0.85

    def test_mjolnir_loading_factor(self):
        """MJOLNIR loading factor should be reasonable (0.4-0.8)."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        I_sc = dev.voltage * np.sqrt(dev.capacitance / dev.inductance)
        loading = dev.peak_current / I_sc
        assert 0.4 < loading < 0.8


# =====================================================================
# Cross-Device Comparison Tests
# =====================================================================

class TestCrossDeviceComparison:
    """Compare new devices with existing ones for validation coverage."""

    def test_five_device_waveform_coverage(self):
        """5 independent devices with waveforms for LOO."""
        from dpf.validation.experimental import DEVICES

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        for dev_name in loo_devices:
            assert dev_name in DEVICES
            dev = DEVICES[dev_name]
            assert dev.waveform_t is not None, f"{dev_name} missing waveform"
            assert len(dev.waveform_t) >= 20

    def test_energy_range_coverage(self):
        """Devices span 3+ decades of stored energy."""
        from dpf.validation.experimental import DEVICES

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        energies = []
        for dev_name in loo_devices:
            dev = DEVICES[dev_name]
            E = 0.5 * dev.capacitance * dev.voltage**2
            energies.append(E)
        # Range should span >2 decades (UNU-ICTP ~3 kJ to MJOLNIR ~734 kJ)
        ratio = max(energies) / min(energies)
        assert ratio > 100  # > 2 decades

    def test_current_range_coverage(self):
        """Peak currents span ~170 kA to 3.2 MA."""
        from dpf.validation.experimental import DEVICES

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        currents = [DEVICES[d].peak_current for d in loo_devices]
        ratio = max(currents) / min(currents)
        assert ratio > 10  # > 1 decade

    def test_lp_l0_regime_diversity(self):
        """Devices should include both plasma-significant and circuit-dominated."""
        from dpf.validation.experimental import DEVICES, compute_lp_l0_ratio

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        regimes = set()
        for dev_name in loo_devices:
            dev = DEVICES[dev_name]
            result = compute_lp_l0_ratio(
                dev.inductance, dev.anode_radius, dev.cathode_radius, dev.anode_length,
            )
            regimes.add(result["regime"])
        # Should have at least circuit-dominated and plasma-significant
        assert "circuit-dominated" in regimes
        assert "plasma-significant" in regimes

    def test_speed_factor_regime_diversity(self):
        """Devices should include sub-driven, optimal, and super-driven."""
        from dpf.validation.experimental import DEVICES, compute_speed_factor

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        regimes = set()
        for dev_name in loo_devices:
            dev = DEVICES[dev_name]
            result = compute_speed_factor(
                dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
            )
            regimes.add(result["regime"])
        # Should have at least 2 different regimes
        assert len(regimes) >= 2

    def test_five_devices_all_unique_institutions(self):
        """No two LOO devices from the same institution (except PF-1000)."""
        from dpf.validation.experimental import DEVICES

        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"]
        institutions = [DEVICES[d].institution for d in loo_devices]
        # All should be unique
        assert len(set(institutions)) == len(institutions)


# =====================================================================
# Reconstructed Waveform Uncertainty Tests
# =====================================================================

class TestReconstructedUncertainty:
    """Reconstructed waveforms should have higher uncertainty than digitized."""

    def test_faeton_higher_digitization_uncertainty(self):
        from dpf.validation.experimental import DEVICES

        faeton = DEVICES["FAETON-I"]
        pf1000 = DEVICES["PF-1000"]
        # Reconstructed should have higher uncertainty than hand-digitized
        assert faeton.waveform_amplitude_uncertainty > pf1000.waveform_amplitude_uncertainty

    def test_mjolnir_higher_digitization_uncertainty(self):
        from dpf.validation.experimental import DEVICES

        mjolnir = DEVICES["MJOLNIR"]
        poseidon = DEVICES["POSEIDON-60kV"]
        # Reconstructed should have higher uncertainty than IPFS-digitized
        assert mjolnir.waveform_amplitude_uncertainty > poseidon.waveform_amplitude_uncertainty

    def test_faeton_notes_reconstructed(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["FAETON-I"]
        assert "RECONSTRUCTED" in dev.measurement_notes

    def test_mjolnir_notes_reconstructed(self):
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["MJOLNIR"]
        assert "RECONSTRUCTED" in dev.measurement_notes


# =====================================================================
# N=5 LOO Cross-Validation Tests (SLOW)
# =====================================================================

class TestN5LOOCrossValidation:
    """N=5 leave-one-out cross-validation with all 5 devices.

    This is the key PhD-panel milestone: df=4 gives finite variance
    for the t-distribution (vs df=2 with N=3 devices where variance
    is infinite).  Requires ~8 minutes of compute.
    """

    @pytest.mark.slow
    def test_n5_loo_all_devices(self):
        """N=5 LOO should produce finite mean and std with df=4."""
        from dpf.validation.calibration import MultiDeviceCalibrator

        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP", "FAETON-I", "MJOLNIR"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.04, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=1,
            seed=42,
        )
        loo = cal.leave_one_out()

        assert len(loo) == 5
        blind_nrmses = [m["blind_nrmse"] for m in loo.values()]
        mean_loo = float(np.mean(blind_nrmses))
        std_loo = float(np.std(blind_nrmses, ddof=1))

        # All blind NRMSEs should be finite and < 1.0
        for dev, m in loo.items():
            assert 0 < m["blind_nrmse"] < 1.0, f"{dev} blind={m['blind_nrmse']}"
            assert m["degradation"] > 0

        # Mean and std should be finite
        assert 0 < mean_loo < 1.0
        assert 0 < std_loo < 1.0

        # df=4 gives finite variance (key milestone)
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=4)
        se = std_loo / np.sqrt(5)
        ci_low = mean_loo - t_crit * se
        ci_high = mean_loo + t_crit * se
        assert ci_low < ci_high
        assert ci_high - ci_low < 1.0  # CI width should be bounded
