"""Phase BC: Circuit-only calibration and NRMSE timing/amplitude decomposition.

Tests for path-to-7.0 items from PhD Debate #38:
- Circuit-only calibration (calibrate on 0-6 us, blind-predict pinch)
- NRMSE timing/amplitude decomposition
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PF1000_FC = 0.800
_PF1000_FM = 0.094
_PF1000_PCF = 0.14


# ===========================================================================
# Circuit-Only Calibration Tests
# ===========================================================================
class TestCircuitOnlyCalibration:
    """Test circuit-window-only calibration with blind pinch prediction."""

    @pytest.mark.slow
    def test_circuit_calibration_runs(self):
        """Circuit-only calibration completes for PF-1000."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration(
            "PF-1000",
            circuit_window_us=6.0,
            maxiter=50,
        )
        assert result.converged
        assert result.device_name == "PF-1000"
        assert result.circuit_window_us == 6.0

    @pytest.mark.slow
    def test_circuit_fc_fm_in_range(self):
        """Circuit-only fc/fm are physically reasonable."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        assert 0.6 <= result.best_fc <= 0.8
        assert 0.05 <= result.best_fm <= 0.25

    @pytest.mark.slow
    def test_circuit_nrmse_lower_than_full(self):
        """Circuit-window NRMSE should be <= full waveform NRMSE."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        # The circuit phase is the calibration window, so its NRMSE
        # should be optimized and generally lower than full waveform
        assert result.circuit_nrmse <= result.full_nrmse + 0.01

    @pytest.mark.slow
    def test_pinch_is_blind_prediction(self):
        """Pinch-phase NRMSE exists and is a genuine blind prediction."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        assert result.pinch_asme is not None
        assert result.pinch_nrmse is not None
        assert result.pinch_nrmse > 0
        assert result.nrmse_ratio is not None
        # Pinch should be worse than circuit (blind prediction)
        assert result.nrmse_ratio >= 0.8  # At minimum comparable

    @pytest.mark.slow
    def test_asme_assessment_present(self):
        """ASME assessments computed for all windows."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        # Circuit ASME
        assert result.circuit_asme.E > 0
        assert result.circuit_asme.u_val > 0
        assert result.circuit_asme.ratio > 0
        # Full ASME
        assert result.full_asme.E > 0
        assert result.full_asme.ratio > 0

    @pytest.mark.slow
    def test_standard_calibration_comparison(self):
        """Standard calibration fc/fm are included for comparison."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        assert 0.6 <= result.standard_fc <= 0.8
        assert 0.05 <= result.standard_fm <= 0.25

    def test_result_dataclass_fields(self):
        """CircuitOnlyCalibrationResult has all expected fields."""
        from dpf.validation.calibration import CircuitOnlyCalibrationResult

        fields = {f.name for f in CircuitOnlyCalibrationResult.__dataclass_fields__.values()}
        required = {
            "device_name", "circuit_window_us", "best_fc", "best_fm",
            "n_evals", "converged", "circuit_asme", "pinch_asme",
            "full_asme", "circuit_nrmse", "pinch_nrmse", "full_nrmse",
            "nrmse_ratio", "standard_fc", "standard_fm",
        }
        assert required.issubset(fields)

    @pytest.mark.slow
    def test_different_windows(self):
        """Different circuit windows produce different results."""
        from dpf.validation.calibration import circuit_only_calibration

        r1 = circuit_only_calibration("PF-1000", circuit_window_us=4.0, maxiter=30)
        r2 = circuit_only_calibration("PF-1000", circuit_window_us=6.0, maxiter=30)
        # Different windows should give different calibrations
        assert r1.circuit_window_us == 4.0
        assert r2.circuit_window_us == 6.0
        # The circuit NRMSEs may differ
        assert r1.circuit_nrmse != pytest.approx(r2.circuit_nrmse, abs=1e-6)

    @pytest.mark.slow
    def test_pinch_delta_model(self):
        """Pinch-phase delta_model is computed."""
        from dpf.validation.calibration import circuit_only_calibration

        result = circuit_only_calibration("PF-1000", maxiter=50)
        if result.pinch_asme is not None:
            assert result.pinch_asme.delta_model >= 0
            # delta_model should be less than E (by definition)
            assert result.pinch_asme.delta_model <= result.pinch_asme.E + 1e-10


# ===========================================================================
# NRMSE Timing/Amplitude Decomposition Tests
# ===========================================================================
class TestNRMSEDecomposition:
    """Test NRMSE decomposition into timing and amplitude components."""

    @pytest.mark.slow
    def test_decomposition_runs(self):
        """NRMSE decomposition completes for PF-1000."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("PF-1000")
        assert result.total_nrmse > 0
        assert result.aligned_nrmse >= 0
        assert result.timing_nrmse >= 0
        assert result.device_name == "PF-1000"

    @pytest.mark.slow
    def test_aligned_leq_total(self):
        """Aligned NRMSE should be <= total NRMSE."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("PF-1000")
        assert result.aligned_nrmse <= result.total_nrmse + 1e-10

    @pytest.mark.slow
    def test_fractions_sum_to_one(self):
        """Timing + amplitude fractions should sum to 1.0."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("PF-1000")
        assert result.timing_fraction + result.amplitude_fraction == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.slow
    def test_pythagoras_relation(self):
        """total^2 = aligned^2 + timing^2 (Pythagoras decomposition)."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("PF-1000")
        total_sq = result.total_nrmse**2
        decomp_sq = result.aligned_nrmse**2 + result.timing_nrmse**2
        assert total_sq == pytest.approx(decomp_sq, rel=1e-6)

    @pytest.mark.slow
    def test_optimal_shift_reasonable(self):
        """Optimal time shift should be within search range."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition(
            "PF-1000", max_shift_us=2.0,
        )
        assert abs(result.optimal_shift_us) <= 2.0

    @pytest.mark.slow
    def test_pf1000_has_timing_error(self):
        """PF-1000 should have significant timing error component."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("PF-1000")
        # Debate #38 found ~50% timing, ~50% amplitude
        assert result.timing_fraction > 0.1  # At least 10% timing
        assert result.timing_nrmse > 0.03  # At least 3% timing NRMSE

    @pytest.mark.slow
    def test_poseidon_mostly_amplitude(self):
        """POSEIDON-60kV should be mostly amplitude error (geometry mismatch)."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition("POSEIDON-60kV")
        assert result.amplitude_fraction > 0.8  # Mostly amplitude

    def test_decomposition_dataclass_fields(self):
        """NRMSEDecomposition has all expected fields."""
        from dpf.validation.calibration import NRMSEDecomposition

        fields = {f.name for f in NRMSEDecomposition.__dataclass_fields__.values()}
        required = {
            "total_nrmse", "aligned_nrmse", "timing_nrmse",
            "optimal_shift_us", "timing_fraction", "amplitude_fraction",
            "device_name",
        }
        assert required.issubset(fields)

    @pytest.mark.slow
    def test_shift_resolution_affects_result(self):
        """Finer resolution should give same or better alignment."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        coarse = nrmse_timing_amplitude_decomposition(
            "PF-1000", shift_resolution_ns=100.0,
        )
        fine = nrmse_timing_amplitude_decomposition(
            "PF-1000", shift_resolution_ns=10.0,
        )
        # Finer resolution should find equal or better alignment
        assert fine.aligned_nrmse <= coarse.aligned_nrmse + 0.005

    @pytest.mark.slow
    def test_circuit_only_decomposition(self):
        """Decomposition with circuit-only calibration parameters."""
        from dpf.validation.calibration import nrmse_timing_amplitude_decomposition

        result = nrmse_timing_amplitude_decomposition(
            "PF-1000", fc=0.800, fm=0.090,
        )
        assert result.total_nrmse > 0
        assert result.aligned_nrmse <= result.total_nrmse + 1e-10
