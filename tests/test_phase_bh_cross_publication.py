"""Phase BH: Cross-publication validation and FIM nondimensionalization.

Resolves PhD Debate #43 highest-priority recommendation:
  Replace reconstructed 16kV waveform with genuine independent data.

Since Akel (2021) is paywalled, we use an alternative approach:
  Gribkov et al. (2007) provides a 90-point digitized PF-1000 I(t) waveform
  at 27 kV / 3.5 Torr D2 — same device, same conditions, but DIFFERENT shot
  and DIFFERENT digitization than the Scholz (2006) 26-point training data.

This enables cross-publication validation:
  - Calibrate on Scholz (2006): 26 points, peak 1.87 MA at 5.8 us
  - Predict on Gribkov (2007): 90 points, peak 1.846 MA at 6.39 us
  - The model must reproduce a waveform it has never seen, from a different
    measurement campaign of the same device at the same conditions.

Additionally:
  - Nondimensionalized FIM: normalize parameters by bound range before computing
    condition number, making it unit-independent (Debate #43 recommendation).
"""

import pytest  # noqa: I001


# --------------------------------------------------------------------------- #
#  Slow tests — cross-publication blind prediction
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestCrossPublicationPrediction:
    """Calibrate on Scholz (2006), predict on Gribkov (2007).

    Both are PF-1000 at 27 kV, 3.5 Torr D2, but different shots
    and different digitization sources. This tests whether calibrated
    parameters reproduce an unseen measurement of the same device.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from dpf.validation.calibration import blind_predict

        return blind_predict(
            train_device="PF-1000",          # Scholz (2006), 26 points
            test_device="PF-1000-Gribkov",   # Gribkov (2007), 90 points
            fc_bounds=(0.6, 0.80),
            fm_bounds=(0.10, 0.30),
            delay_bounds_us=(0.0, 2.0),
            pinch_column_fraction=0.14,
            crowbar_enabled=True,
            crowbar_resistance=1.5e-3,
            maxiter=200,
        )

    def test_train_nrmse_matches_baseline(self, result):
        """Training NRMSE should match Phase BF/BG baseline on Scholz."""
        assert result.train_nrmse < 0.12, (
            f"Train NRMSE {result.train_nrmse:.4f} exceeds 12%"
        )

    def test_blind_nrmse_below_20_percent(self, result):
        """Cross-publication NRMSE should be < 20%.

        Same device, same conditions — the model should reproduce the
        waveform shape. If NRMSE > 20%, it suggests the calibration is
        overfitting to digitization artifacts in the Scholz data.
        """
        assert result.test_nrmse < 0.20, (
            f"Cross-pub NRMSE {result.test_nrmse:.4f} exceeds 20%"
        )

    def test_blind_nrmse_close_to_training(self, result):
        """Cross-publication NRMSE should be within 2x of training NRMSE.

        Since it's the same device and conditions, the prediction error
        should not be dramatically worse than the training fit.
        Ratio ~1.75 observed: Gribkov's 90-point waveform captures finer
        structure than Scholz's 26-point, so some degradation is expected.
        """
        ratio = result.test_nrmse / result.train_nrmse
        assert ratio < 2.0, (
            f"Blind/train ratio {ratio:.2f} > 2.0 — model may overfit Scholz digitization"
        )

    def test_peak_current_error_below_10_percent(self, result):
        """Peak current error should be < 10%.

        Scholz: 1.87 MA, Gribkov: 1.846 MA (1.3% difference).
        The calibrated model should predict within 10% of the Gribkov peak.
        """
        assert result.peak_current_error < 0.10, (
            f"Peak current error {result.peak_current_error*100:.1f}% exceeds 10%"
        )

    def test_same_device_different_digitization(self, result):
        """Verify this IS cross-publication, not cross-condition."""
        assert result.train_device == "PF-1000"
        assert result.test_device == "PF-1000-Gribkov"

    def test_asme_ratio_reported(self, result):
        """ASME E/u_val should be finite and positive."""
        assert result.test_asme.ratio > 0
        assert result.test_asme.ratio < 100

    def test_gribkov_waveform_higher_resolution(self, result):
        """Gribkov has more data points than Scholz (validation > calibration)."""
        from dpf.validation.experimental import DEVICES

        scholz_pts = len(DEVICES["PF-1000"].waveform_t)
        gribkov_pts = len(DEVICES["PF-1000-Gribkov"].waveform_t)
        assert gribkov_pts > scholz_pts


# --------------------------------------------------------------------------- #
#  Slow tests — nondimensionalized FIM
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestNondimensionalizedFIM:
    """FIM with parameters normalized by bound range.

    Addresses Debate #43 finding: the original FIM mixes dimensionless
    (fc, fm) with microsecond (delay) parameters, making the condition
    number unit-dependent. Nondimensionalizing by [fc_range, fm_range,
    delay_range] gives a physically meaningful condition number.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from dpf.validation.calibration import fisher_information_matrix

        return fisher_information_matrix(
            device_name="PF-1000",
            fc=0.800,
            fm=0.100,
            delay_us=0.571,
            pinch_column_fraction=0.14,
            crowbar_enabled=True,
            crowbar_resistance=1.5e-3,
            step_size=0.01,
            nondimensionalize=True,
            param_ranges=(0.20, 0.20, 2.0),  # fc_range, fm_range, delay_range
        )

    def test_fim_shape(self, result):
        """FIM should be 3x3."""
        assert result.fim.shape == (3, 3)

    def test_fim_symmetric(self, result):
        """FIM should be symmetric."""
        import numpy as np

        assert np.allclose(result.fim, result.fim.T, atol=1e-10)

    def test_condition_number_different_from_raw(self, result):
        """Nondimensionalized cond number should differ from raw.

        The raw condition number was 4.82e3. After nondimensionalization,
        it should be different (and more physically meaningful).
        """
        raw_cond = 4.82e3
        assert result.condition_number != pytest.approx(raw_cond, rel=0.1)

    def test_condition_number_reported(self, result):
        """Condition number should be positive and finite."""
        import math

        assert result.condition_number > 0
        assert math.isfinite(result.condition_number)

    def test_condition_number_diagnostic(self, result):
        """Log nondimensionalized condition number for comparison."""
        print(f"\n  Nondimensionalized FIM condition number: {result.condition_number:.2e}")
        print(f"  Eigenvalues: {result.eigenvalues}")
        print(f"  Identifiable (cond < 1e4): {result.is_identifiable}")
        print("  Raw condition number was: 4.82e3")

        assert result.condition_number > 1.0  # Not degenerate


# --------------------------------------------------------------------------- #
#  Non-slow tests — validate Gribkov waveform and framework
# --------------------------------------------------------------------------- #


class TestGribkovWaveformAnalytical:
    """Non-slow tests verifying the Gribkov waveform data quality."""

    def test_gribkov_device_exists(self):
        """PF-1000-Gribkov is in the DEVICES dict."""
        from dpf.validation.experimental import DEVICES

        assert "PF-1000-Gribkov" in DEVICES

    def test_gribkov_waveform_available(self):
        """PF-1000-Gribkov has a digitized waveform."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["PF-1000-Gribkov"]
        assert dev.waveform_t is not None
        assert dev.waveform_I is not None
        assert len(dev.waveform_t) == len(dev.waveform_I)
        assert len(dev.waveform_t) >= 80  # 90 points expected

    def test_gribkov_peak_current(self):
        """Peak current should be ~1.846 MA."""
        from dpf.validation.experimental import DEVICES

        dev = DEVICES["PF-1000-Gribkov"]
        assert 1.7e6 <= dev.peak_current <= 1.95e6

    def test_gribkov_same_device_as_scholz(self):
        """Gribkov and Scholz are same device, same conditions."""
        from dpf.validation.experimental import DEVICES

        scholz = DEVICES["PF-1000"]
        gribkov = DEVICES["PF-1000-Gribkov"]
        assert scholz.voltage == gribkov.voltage  # Both 27 kV
        assert scholz.capacitance == gribkov.capacitance
        assert scholz.inductance == gribkov.inductance
        assert scholz.fill_pressure_torr == gribkov.fill_pressure_torr

    def test_gribkov_different_peak_from_scholz(self):
        """Different shot: Gribkov peak differs from Scholz peak."""
        from dpf.validation.experimental import DEVICES

        scholz = DEVICES["PF-1000"]
        gribkov = DEVICES["PF-1000-Gribkov"]
        # Different shots have different peak currents (shot-to-shot variation)
        assert scholz.peak_current != gribkov.peak_current
        # But within 5% of each other (same conditions)
        rel_diff = abs(scholz.peak_current - gribkov.peak_current) / scholz.peak_current
        assert rel_diff < 0.05

    def test_gribkov_higher_resolution(self):
        """Gribkov has more data points than Scholz."""
        from dpf.validation.experimental import DEVICES

        scholz = DEVICES["PF-1000"]
        gribkov = DEVICES["PF-1000-Gribkov"]
        assert len(gribkov.waveform_t) > len(scholz.waveform_t)

    def test_gribkov_waveform_monotonic_rise(self):
        """Current should rise monotonically for the first ~5 us."""
        import numpy as np  # noqa: I001

        from dpf.validation.experimental import DEVICES

        dev = DEVICES["PF-1000-Gribkov"]
        t_us = dev.waveform_t * 1e6  # s -> us
        I_kA = dev.waveform_I / 1e3  # A -> kA

        # Check monotonic rise from 0.5 to 3.0 us
        mask = (t_us >= 0.5) & (t_us <= 3.0)
        I_rise = I_kA[mask]
        dI = np.diff(I_rise)
        assert np.all(dI > 0), "Current should rise monotonically in 0.5-3.0 us"

    def test_gribkov_has_current_dip(self):
        """Waveform should show current dip (pinch signature)."""
        import numpy as np  # noqa: I001

        from dpf.validation.experimental import DEVICES

        dev = DEVICES["PF-1000-Gribkov"]
        I_MA = dev.waveform_I / 1e6  # MA

        # Find peak
        peak_idx = np.argmax(I_MA)
        I_peak = I_MA[peak_idx]

        # After peak, current should dip below 80% of peak
        post_peak = I_MA[peak_idx:]
        I_min = np.min(post_peak)
        dip_ratio = I_min / I_peak
        assert dip_ratio < 0.80, (
            f"Current dip ratio {dip_ratio:.3f} — expected < 0.80"
        )

    def test_gribkov_lower_digitization_uncertainty(self):
        """Gribkov uncertainty should be lower than Scholz (digital vs hand)."""
        from dpf.validation.experimental import DEVICES

        scholz = DEVICES["PF-1000"]
        gribkov = DEVICES["PF-1000-Gribkov"]
        assert gribkov.waveform_amplitude_uncertainty < scholz.waveform_amplitude_uncertainty

    def test_cross_publication_meaning(self):
        """Cross-publication validation tests measurement independence.

        Calibrate on Scholz (2006), predict on Gribkov (2007).
        NOT cross-condition (same V, same P), but tests whether
        calibration is robust to different shots and digitization.
        """
        import dataclasses  # noqa: I001

        from dpf.validation.calibration import BlindPredictionResult

        fields = {f.name for f in dataclasses.fields(BlindPredictionResult)}
        assert "train_device" in fields
        assert "test_device" in fields
        assert "test_nrmse" in fields
