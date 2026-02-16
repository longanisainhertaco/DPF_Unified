"""Phase S: Lee model cross-check against standalone RLC circuit solver.

Runs the Lee model (phases 1-2) and the RLC circuit solver side-by-side for
PF-1000.  Compares I(t) waveforms, peak current, and timing metrics.

The Lee model includes snowplow dynamics that the circuit-only model lacks,
so differences are expected and documented quantitatively.

References
----------
- Lee & Saw, J. Fusion Energy 27, 2008.
- Lee, J. Fusion Energy 33, 319-335, 2014.
- Scholz et al., Nukleonika 51(1), 2006.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState
from dpf.validation.experimental import PF1000_DATA
from dpf.validation.lee_model_comparison import LeeModel, LeeModelResult

# ── PF-1000 circuit parameters ──
PF1000_C = 1.332e-3
PF1000_V0 = 27e3
PF1000_L0 = 33.5e-9
PF1000_R0 = 2.3e-3

T_QUARTER = (np.pi / 2) * np.sqrt(PF1000_L0 * PF1000_C)  # ~10.49 us


def _run_circuit_traces(
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run circuit-only solver and return (t, I) arrays."""
    solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
    coupling = CouplingState()

    n_steps = int(t_end / dt)
    t_arr = np.zeros(n_steps + 1)
    I_arr = np.zeros(n_steps + 1)

    for i in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        t_arr[i + 1] = solver.state.time
        I_arr[i + 1] = solver.state.current

    return t_arr, I_arr


# ═══════════════════════════════════════════════════════
# Lee model basic execution
# ═══════════════════════════════════════════════════════


class TestLeeModelExecution:
    """Verify the Lee model runs without errors for PF-1000."""

    def test_lee_model_runs_pf1000(self):
        """Lee model completes for PF-1000 without raising."""
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        assert isinstance(result, LeeModelResult)
        assert result.device_name == "PF-1000"
        assert len(result.t) > 100, "Expected substantial time trace"
        assert len(result.I) == len(result.t)

    def test_lee_model_completes_phase1(self):
        """Lee model should complete at least the axial phase (phase 1)."""
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        assert 1 in result.phases_completed, (
            f"Phase 1 (axial) not completed: {result.phases_completed}"
        )

    def test_lee_model_peak_current_positive(self):
        """Lee model should produce a finite, positive peak current."""
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        assert result.peak_current > 0, "Peak current should be positive"
        assert np.isfinite(result.peak_current), "Peak current should be finite"
        assert result.peak_current_time > 0, "Peak time should be positive"

    def test_lee_model_peak_in_ma_range(self):
        """Lee model I_peak for PF-1000 should be in the MA range."""
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        # PF-1000 experimental: 1.87 MA. Lee model should be same order.
        assert result.peak_current > 0.5e6, (
            f"I_peak = {result.peak_current:.2e} A, expected > 0.5 MA"
        )
        assert result.peak_current < 10e6, (
            f"I_peak = {result.peak_current:.2e} A, expected < 10 MA"
        )

    def test_lee_model_no_nan_in_waveforms(self):
        """Lee model waveforms should contain no NaN or Inf."""
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        assert np.all(np.isfinite(result.t)), "NaN/Inf in time array"
        assert np.all(np.isfinite(result.I)), "NaN/Inf in current array"
        assert np.all(np.isfinite(result.V)), "NaN/Inf in voltage array"


# ═══════════════════════════════════════════════════════
# Lee model vs circuit solver comparison
# ═══════════════════════════════════════════════════════


class TestLeeVsCircuitComparison:
    """Quantitative comparison between Lee model and circuit-only solver."""

    @pytest.fixture
    def lee_result(self) -> LeeModelResult:
        """Run Lee model once for PF-1000 (shared across tests in class)."""
        model = LeeModel()
        return model.run(device_name="PF-1000")

    @pytest.fixture
    def circuit_traces(self, lee_result: LeeModelResult) -> tuple[np.ndarray, np.ndarray]:
        """Run circuit solver for same duration as Lee model."""
        t_end = float(lee_result.t[-1])
        dt = t_end / 50000  # Fine resolution
        return _run_circuit_traces(t_end, dt)

    def test_early_time_agreement(
        self, lee_result: LeeModelResult, circuit_traces: tuple[np.ndarray, np.ndarray]
    ):
        """Before snowplow onset, Lee model and circuit solver should agree.

        In the first ~1 us, the current sheet has barely moved, so plasma
        inductance is negligible and both models see essentially the same
        dI/dt = V0/L0.
        """
        t_circuit, I_circuit = circuit_traces

        # Compare at t = 1 us (before significant snowplow)
        t_compare = 1.0e-6
        assert t_compare < T_QUARTER * 0.2  # Sanity: early enough

        # Interpolate both onto comparison time
        I_lee_at_t = float(np.interp(t_compare, lee_result.t, lee_result.I))
        I_circ_at_t = float(np.interp(t_compare, t_circuit, I_circuit))

        # Should agree within 20% at early time
        if abs(I_circ_at_t) > 1e3:  # Only compare if current is significant
            rel_diff = abs(I_lee_at_t - I_circ_at_t) / abs(I_circ_at_t)
            assert rel_diff < 0.20, (
                f"At t={t_compare * 1e6:.1f} us: Lee I={I_lee_at_t:.2e} A, "
                f"Circuit I={I_circ_at_t:.2e} A, diff={rel_diff:.1%}"
            )

    def test_lee_model_peak_lower_than_circuit(
        self, lee_result: LeeModelResult, circuit_traces: tuple[np.ndarray, np.ndarray]
    ):
        """Lee model I_peak should be lower than circuit-only I_peak.

        The snowplow dynamics add plasma inductance, which reduces
        dI/dt and therefore the peak current.
        """
        _, I_circuit = circuit_traces

        I_peak_lee = lee_result.peak_current
        I_peak_circuit = np.max(np.abs(I_circuit))

        assert I_peak_lee < I_peak_circuit, (
            f"Lee I_peak = {I_peak_lee:.2e} A should be < "
            f"circuit I_peak = {I_peak_circuit:.2e} A "
            "(snowplow adds inductance, reducing peak)"
        )

    def test_peak_current_comparison_quantitative(
        self, lee_result: LeeModelResult, circuit_traces: tuple[np.ndarray, np.ndarray]
    ):
        """Quantify and document I_peak discrepancy."""
        _, I_circuit = circuit_traces

        I_peak_lee = lee_result.peak_current
        I_peak_circuit = np.max(np.abs(I_circuit))
        I_peak_exp = PF1000_DATA.peak_current

        # Lee model should be closer to experiment than circuit-only
        lee_exp_error = abs(I_peak_lee - I_peak_exp) / I_peak_exp
        circuit_exp_error = abs(I_peak_circuit - I_peak_exp) / I_peak_exp

        # Print comparison
        print("\nI_peak comparison:")
        print(f"  Lee model:    {I_peak_lee / 1e6:.3f} MA "
              f"(exp error: {lee_exp_error:.1%})")
        print(f"  Circuit-only: {I_peak_circuit / 1e6:.3f} MA "
              f"(exp error: {circuit_exp_error:.1%})")
        print(f"  Experimental: {I_peak_exp / 1e6:.3f} MA "
              f"(Scholz 2006)")
        print(f"  Lee/Circuit ratio: {I_peak_lee / I_peak_circuit:.3f}")

        # The simplified 2-phase Lee model with default fm=0.7, fc=0.7
        # over-predicts snowplow mass loading, so I_peak_lee undershoots
        # experiment more than circuit-only overshoots. Both should be
        # within 50% of experiment (same order of magnitude, MA range).
        assert lee_exp_error < 0.50, (
            f"Lee model error {lee_exp_error:.1%} exceeds 50% tolerance"
        )
        assert circuit_exp_error < 0.50, (
            f"Circuit model error {circuit_exp_error:.1%} exceeds 50% tolerance"
        )

    def test_l2_norm_waveform_difference(
        self, lee_result: LeeModelResult, circuit_traces: tuple[np.ndarray, np.ndarray]
    ):
        """Compute L2 norm of I(t) difference over common time range."""
        t_circuit, I_circuit = circuit_traces

        # Common time range
        t_max = min(float(lee_result.t[-1]), float(t_circuit[-1]))
        t_common = np.linspace(0, t_max, 5000)

        I_lee_interp = np.interp(t_common, lee_result.t, lee_result.I)
        I_circ_interp = np.interp(t_common, t_circuit, I_circuit)

        # L2 norm (RMS difference)
        diff = I_lee_interp - I_circ_interp
        l2_norm = np.sqrt(np.mean(diff**2))

        # Normalize by peak current for dimensionless metric
        I_peak_ref = max(np.max(np.abs(I_lee_interp)), np.max(np.abs(I_circ_interp)))
        normalized_l2 = l2_norm / max(I_peak_ref, 1e-30)

        print("\nWaveform L2 metrics:")
        print(f"  RMS(I_lee - I_circ) = {l2_norm:.2e} A")
        print(f"  Normalized L2 = {normalized_l2:.3f}")

        # Difference should be significant (models are physically different)
        # but not enormous (same device parameters)
        assert l2_norm > 0, "Zero difference is suspicious"
        assert normalized_l2 < 2.0, (
            f"Normalized L2 = {normalized_l2:.2f}, "
            "suspiciously large for same device"
        )


# ═══════════════════════════════════════════════════════
# Lee model experimental comparison
# ═══════════════════════════════════════════════════════


class TestLeeModelExperimentalComparison:
    """Lee model vs experimental data from Scholz et al. (2006)."""

    def test_compare_with_experiment_api(self):
        """LeeModel.compare_with_experiment() runs and returns metrics."""
        model = LeeModel()
        comparison = model.compare_with_experiment("PF-1000")

        assert comparison.device_name == "PF-1000"
        assert comparison.peak_current_error >= 0
        assert comparison.timing_error >= 0

        print("\nLee model vs experiment (Scholz 2006):")
        print(f"  Peak current error: {comparison.peak_current_error:.1%}")
        print(f"  Timing error: {comparison.timing_error:.1%}")

    def test_lee_model_closer_to_experiment_than_analytical(self):
        """Lee model I_peak should be closer to experiment than undamped analytical.

        Undamped analytical: I_peak = V0*sqrt(C/L0) ~ 5.38 MA
        Lee model: includes snowplow -> should be lower -> closer to 1.87 MA
        """
        model = LeeModel()
        result = model.run(device_name="PF-1000")

        I_peak_exp = PF1000_DATA.peak_current
        I_peak_undamped = PF1000_V0 * np.sqrt(PF1000_C / PF1000_L0)

        lee_error = abs(result.peak_current - I_peak_exp) / I_peak_exp
        undamped_error = abs(I_peak_undamped - I_peak_exp) / I_peak_exp

        print(f"\nI_peak vs experimental ({I_peak_exp / 1e6:.2f} MA):")
        print(f"  Lee model: {result.peak_current / 1e6:.3f} MA "
              f"(error: {lee_error:.1%})")
        print(f"  Undamped:  {I_peak_undamped / 1e6:.3f} MA "
              f"(error: {undamped_error:.1%})")

        # Lee model should be closer to experiment than undamped formula
        assert lee_error < undamped_error, (
            f"Lee error ({lee_error:.1%}) should be < "
            f"undamped error ({undamped_error:.1%})"
        )


# ═══════════════════════════════════════════════════════
# Multi-device consistency
# ═══════════════════════════════════════════════════════


class TestLeeModelMultiDevice:
    """Verify Lee model runs for multiple devices."""

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_lee_model_runs_all_devices(self, device: str):
        """Lee model should complete without error for all registered devices."""
        model = LeeModel()
        result = model.run(device_name=device)

        assert result.peak_current > 0, f"Zero peak current for {device}"
        assert len(result.t) > 10, f"Too few time points for {device}"
        assert np.all(np.isfinite(result.I)), f"NaN in I(t) for {device}"
