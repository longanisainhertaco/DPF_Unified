"""Phase S: PF-1000 circuit-only validation against analytical and experimental data.

Tests the RLC circuit solver standalone (no MHD) for PF-1000 parameters.
Validates quarter-period, peak current, damping, and energy conservation
against analytical formulae and Scholz et al. (2006) experimental data.

References
----------
- Scholz et al., Nukleonika 51(1), 2006: PF-1000 at 27 kV, I_peak ~ 1.87 MA.
- Lee & Saw, J. Fusion Energy 27, 2008: Lee model reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState
from dpf.validation.experimental import PF1000_DATA

# ── PF-1000 circuit parameters (from presets.py) ──
PF1000_C = 1.332e-3       # Capacitance [F]
PF1000_V0 = 27e3          # Charging voltage [V]
PF1000_L0 = 33.5e-9       # External inductance [H]
PF1000_R0 = 2.3e-3        # External resistance [Ohm]

# Analytical reference values
T_QUARTER_ANALYTICAL = (np.pi / 2) * np.sqrt(PF1000_L0 * PF1000_C)  # ~10.49 us
I_PEAK_UNDAMPED = PF1000_V0 * np.sqrt(PF1000_C / PF1000_L0)         # ~5.38 MA
E_INITIAL = 0.5 * PF1000_C * PF1000_V0**2                            # ~485.6 kJ


def _run_circuit(
    C: float,
    V0: float,
    L0: float,
    R0: float,
    t_end: float,
    dt: float,
    Lp: float = 0.0,
    R_plasma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RLCSolver]:
    """Run standalone RLC circuit solver and return time traces.

    Returns
    -------
    t_arr : ndarray
        Time array [s].
    I_arr : ndarray
        Current waveform [A].
    V_arr : ndarray
        Voltage waveform [V].
    solver : RLCSolver
        Solver instance (for energy accounting inspection).
    """
    solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0)
    coupling = CouplingState(Lp=Lp, dL_dt=0.0, R_plasma=R_plasma)

    n_steps = int(t_end / dt)
    t_arr = np.zeros(n_steps + 1)
    I_arr = np.zeros(n_steps + 1)
    V_arr = np.zeros(n_steps + 1)

    t_arr[0] = 0.0
    I_arr[0] = solver.state.current
    V_arr[0] = solver.state.voltage

    for i in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        t_arr[i + 1] = solver.state.time
        I_arr[i + 1] = solver.state.current
        V_arr[i + 1] = solver.state.voltage

    return t_arr, I_arr, V_arr, solver


# ═══════════════════════════════════════════════════════
# T/4 validation
# ═══════════════════════════════════════════════════════


class TestPF1000CircuitQuarterPeriod:
    """Quarter-period validation: analytical vs numerical for PF-1000."""

    def test_undamped_quarter_period(self):
        """T/4 = (pi/2)*sqrt(L0*C) ~ 10.49 us for undamped PF-1000."""
        # Sanity check on the analytical value
        assert pytest.approx(10.49e-6, rel=0.01) == T_QUARTER_ANALYTICAL

        # Run undamped circuit (R0=0)
        t_end = 2.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000
        t_arr, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0,
            t_end=t_end, dt=dt,
        )

        # Find time of peak current
        peak_idx = np.argmax(np.abs(I_arr))
        T_quarter_num = t_arr[peak_idx]

        rel_error = abs(T_quarter_num - T_QUARTER_ANALYTICAL) / T_QUARTER_ANALYTICAL
        assert rel_error < 0.01, (
            f"T/4 numerical = {T_quarter_num * 1e6:.2f} us, "
            f"analytical = {T_QUARTER_ANALYTICAL * 1e6:.2f} us, "
            f"error = {rel_error:.2%}"
        )

    def test_damped_quarter_period(self):
        """Peak time for damped PF-1000 (R0=2.3 mOhm).

        For a damped RLC circuit: I(t) = (V0/(omega_d*L)) * e^{-alpha*t} * sin(omega_d*t)
        Peak time: t_peak = (1/omega_d) * arctan(omega_d / alpha)
        (reduces to pi/(2*omega_0) for alpha -> 0).
        """
        omega_0 = 1.0 / np.sqrt(PF1000_L0 * PF1000_C)
        alpha = PF1000_R0 / (2.0 * PF1000_L0)
        omega_d = np.sqrt(omega_0**2 - alpha**2)
        # Exact peak time for damped RLC
        T_peak_damped = (1.0 / omega_d) * np.arctan(omega_d / alpha)

        # Run damped circuit
        t_end = 2.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000
        t_arr, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        peak_idx = np.argmax(np.abs(I_arr))
        T_peak_num = t_arr[peak_idx]

        # Should be within 2% of analytical damped peak time
        rel_error = abs(T_peak_num - T_peak_damped) / T_peak_damped
        assert rel_error < 0.02, (
            f"Damped T_peak numerical = {T_peak_num * 1e6:.2f} us, "
            f"analytical = {T_peak_damped * 1e6:.2f} us, "
            f"error = {rel_error:.2%}"
        )


# ═══════════════════════════════════════════════════════
# Peak current validation
# ═══════════════════════════════════════════════════════


class TestPF1000CircuitPeakCurrent:
    """Peak current validation for PF-1000 circuit-only."""

    def test_undamped_peak_current(self):
        """I_peak = V0 * sqrt(C/L0) for undamped LC circuit."""
        t_end = 2.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000
        _t, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0,
            t_end=t_end, dt=dt,
        )

        I_peak_num = np.max(np.abs(I_arr))
        rel_error = abs(I_peak_num - I_PEAK_UNDAMPED) / I_PEAK_UNDAMPED
        assert rel_error < 0.01, (
            f"I_peak numerical = {I_peak_num:.3e} A, "
            f"analytical = {I_PEAK_UNDAMPED:.3e} A, "
            f"error = {rel_error:.2%}"
        )

    def test_damped_peak_lower_than_undamped(self):
        """Damped I_peak must be strictly lower than undamped I_peak."""
        t_end = 2.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000

        _, I_undamped, _, _ = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0,
            t_end=t_end, dt=dt,
        )
        _, I_damped, _, _ = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        I_peak_undamped = np.max(np.abs(I_undamped))
        I_peak_damped = np.max(np.abs(I_damped))

        assert I_peak_damped < I_peak_undamped, (
            f"Damped peak {I_peak_damped:.2e} A should be < "
            f"undamped peak {I_peak_undamped:.2e} A"
        )

    def test_circuit_only_exceeds_experimental(self):
        """Circuit-only I_peak (no plasma loading) should exceed experimental.

        Experimental: Scholz et al. (2006) I_peak ~ 1.87 MA with plasma loading.
        Circuit-only: No snowplow inertia or plasma resistance -> higher peak.
        This documents the expected discrepancy, NOT a bug.
        """
        t_end = 2.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000
        _, I_damped, _, _ = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        I_peak_circuit = np.max(np.abs(I_damped))
        I_peak_exp = PF1000_DATA.peak_current  # 1.87 MA

        # Circuit-only exceeds experimental (no plasma loading)
        assert I_peak_circuit > I_peak_exp, (
            f"Circuit-only I_peak = {I_peak_circuit:.2e} A should exceed "
            f"experimental I_peak = {I_peak_exp:.2e} A"
        )

        # Ratio should be > 1.5 (significant excess without plasma)
        ratio = I_peak_circuit / I_peak_exp
        assert ratio > 1.5, (
            f"Circuit-only / experimental ratio = {ratio:.2f}, "
            "expected > 1.5 without plasma loading"
        )


# ═══════════════════════════════════════════════════════
# Damping verification
# ═══════════════════════════════════════════════════════


class TestPF1000CircuitDamping:
    """Verify current decays between half-cycles due to resistance."""

    def test_current_decays_between_halfcycles(self):
        """With R0=2.3 mOhm, |I| at successive peaks should decrease."""
        t_end = 4.0 * T_QUARTER_ANALYTICAL  # Full period
        dt = T_QUARTER_ANALYTICAL / 5000

        _t, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        # Find peaks (local maxima of |I|)
        abs_I = np.abs(I_arr)
        peaks = []
        for i in range(1, len(abs_I) - 1):
            if abs_I[i] > abs_I[i - 1] and abs_I[i] > abs_I[i + 1]:
                peaks.append(abs_I[i])

        assert len(peaks) >= 2, f"Expected >= 2 peaks, found {len(peaks)}"

        # Each successive peak should be smaller (exponential decay)
        for i in range(1, len(peaks)):
            assert peaks[i] < peaks[i - 1], (
                f"Peak {i} ({peaks[i]:.2e} A) should be < "
                f"Peak {i - 1} ({peaks[i - 1]:.2e} A)"
            )

    def test_undamped_no_decay(self):
        """With R0=0, successive peaks should have equal magnitude."""
        t_end = 4.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 5000

        _t, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0,
            t_end=t_end, dt=dt,
        )

        abs_I = np.abs(I_arr)
        peaks = []
        for i in range(1, len(abs_I) - 1):
            if abs_I[i] > abs_I[i - 1] and abs_I[i] > abs_I[i + 1]:
                peaks.append(abs_I[i])

        assert len(peaks) >= 2, f"Expected >= 2 peaks, found {len(peaks)}"

        # All peaks should be within 1% of each other (no decay)
        for i in range(1, len(peaks)):
            rel_diff = abs(peaks[i] - peaks[0]) / peaks[0]
            assert rel_diff < 0.01, (
                f"Peak {i} ({peaks[i]:.2e} A) differs from "
                f"Peak 0 ({peaks[0]:.2e} A) by {rel_diff:.2%}"
            )


# ═══════════════════════════════════════════════════════
# Energy conservation
# ═══════════════════════════════════════════════════════


class TestPF1000CircuitEnergy:
    """Energy conservation and accounting for PF-1000 circuit."""

    def test_initial_energy_value(self):
        """E_initial = 0.5 * C * V0^2 ~ 485.6 kJ for PF-1000 at 27 kV."""
        assert pytest.approx(485.6e3, rel=0.01) == E_INITIAL, (
            f"E_initial = {E_INITIAL / 1e3:.1f} kJ, expected ~485.6 kJ"
        )

    def test_solver_initial_energy_matches(self):
        """RLCSolver initial energy_cap should equal 0.5*C*V0^2."""
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
        assert solver.state.energy_cap == pytest.approx(E_INITIAL, rel=1e-10)
        assert solver.total_energy() == pytest.approx(E_INITIAL, rel=1e-10)

    def test_undamped_energy_conservation(self):
        """For R=0: E_cap + E_ind should remain constant = E_initial."""
        t_end = 4.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 5000

        _t, _I, _V, solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0,
            t_end=t_end, dt=dt,
        )

        E_total_final = solver.total_energy()
        conservation = E_total_final / E_INITIAL

        # Implicit midpoint should conserve energy to < 0.1% for R=0
        assert conservation == pytest.approx(1.0, abs=0.001), (
            f"Undamped energy conservation = {conservation:.6f}"
        )

    def test_damped_energy_conservation(self):
        """For R>0: E_cap + E_ind + E_res should equal E_initial."""
        t_end = 4.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 5000

        _t, _I, _V, solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        E_total_final = solver.total_energy()
        conservation = E_total_final / E_INITIAL

        # Should conserve within 1% (2nd order implicit midpoint)
        assert conservation == pytest.approx(1.0, abs=0.01), (
            f"Damped energy conservation = {conservation:.6f}, "
            f"E_total = {E_total_final:.1f} J, E_init = {E_INITIAL:.1f} J"
        )

    def test_capacitor_depleted_at_quarter_period(self):
        """At T/4, capacitor energy should be nearly zero (transferred to inductor)."""
        dt = T_QUARTER_ANALYTICAL / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0)
        coupling = CouplingState()

        n_steps = int(T_QUARTER_ANALYTICAL / dt)
        for _ in range(n_steps):
            solver.step(coupling, back_emf=0.0, dt=dt)

        # At T/4, all energy should be in inductor (V ~ 0, I ~ max)
        E_cap_ratio = solver.state.energy_cap / E_INITIAL
        assert E_cap_ratio < 0.01, (
            f"After T/4, E_cap/E_initial = {E_cap_ratio:.4f}, "
            "expected < 1% (energy in inductor)"
        )


# ═══════════════════════════════════════════════════════
# Experimental comparison table
# ═══════════════════════════════════════════════════════


class TestPF1000ExperimentalComparison:
    """Document comparison between circuit-only and PF-1000 experimental data.

    Reference: Scholz et al., Nukleonika 51(1), 2006.
    PF-1000 at 27 kV: I_peak ~ 1.87 MA, T_rise ~ 5.8 us.
    """

    def test_experimental_data_consistency(self):
        """Verify PF1000_DATA matches expected values."""
        assert PF1000_DATA.peak_current == pytest.approx(1.87e6, rel=0.01)
        assert PF1000_DATA.current_rise_time == pytest.approx(5.8e-6, rel=0.01)
        assert PF1000_DATA.capacitance == pytest.approx(1.332e-3, rel=0.01)
        assert PF1000_DATA.voltage == pytest.approx(27e3, rel=0.01)

    def test_presets_experimental_parameter_agreement(self):
        """Presets and experimental data should have consistent PF-1000 params.

        Note: presets.py uses L0=33.5 nH, experimental.py uses L0=33 nH.
        This is a ~1.5% discrepancy — acceptable given measurement uncertainty.
        """
        assert PF1000_DATA.capacitance == pytest.approx(PF1000_C, rel=0.01)
        assert PF1000_DATA.voltage == pytest.approx(PF1000_V0, rel=0.01)
        assert PF1000_DATA.resistance == pytest.approx(PF1000_R0, rel=0.01)

        # Note: L0 has a small discrepancy (33 vs 33.5 nH)
        L0_exp = PF1000_DATA.inductance  # 33e-9
        L0_preset = PF1000_L0            # 33.5e-9
        rel_diff = abs(L0_exp - L0_preset) / L0_preset
        assert rel_diff < 0.02, (
            f"L0 discrepancy: experimental={L0_exp * 1e9:.1f} nH, "
            f"preset={L0_preset * 1e9:.1f} nH ({rel_diff:.1%})"
        )

    def test_comparison_table(self):
        """Generate quantitative comparison between circuit-only and experiment.

        Circuit-only LACKS plasma loading (snowplow + compression),
        so I_peak will be higher than experimental.
        """
        t_end = 3.0 * T_QUARTER_ANALYTICAL
        dt = T_QUARTER_ANALYTICAL / 10000

        t_arr, I_arr, _V, _solver = _run_circuit(
            C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
            t_end=t_end, dt=dt,
        )

        peak_idx = np.argmax(np.abs(I_arr))
        I_peak_sim = float(np.abs(I_arr[peak_idx]))
        T_peak_sim = float(t_arr[peak_idx])

        I_peak_exp = PF1000_DATA.peak_current
        T_rise_exp = PF1000_DATA.current_rise_time

        # Print comparison table
        table = "\n".join([
            "=" * 65,
            "PF-1000 Circuit-Only vs Experimental Comparison",
            "=" * 65,
            f"{'Quantity':<30} {'Circuit-Only':<18} {'Experimental':<18}",
            "-" * 65,
            f"{'I_peak [MA]':<30} {I_peak_sim / 1e6:<18.3f} {I_peak_exp / 1e6:<18.3f}",
            f"{'T_peak [us]':<30} {T_peak_sim * 1e6:<18.2f} {T_rise_exp * 1e6:<18.2f}",
            f"{'I_undamped [MA]':<30} {I_PEAK_UNDAMPED / 1e6:<18.3f} {'N/A':<18}",
            f"{'T/4 analytical [us]':<30} {T_QUARTER_ANALYTICAL * 1e6:<18.2f} {'N/A':<18}",
            f"{'E_initial [kJ]':<30} {E_INITIAL / 1e3:<18.1f} {'N/A':<18}",
            "-" * 65,
            f"I_peak ratio (sim/exp): {I_peak_sim / I_peak_exp:.2f}",
            f"T_peak ratio (sim/exp): {T_peak_sim / T_rise_exp:.2f}",
            "Note: Circuit-only lacks plasma loading (snowplow + compression).",
            "Ref: Scholz et al., Nukleonika 51(1), 2006.",
            "=" * 65,
        ])
        print(table)

        # Assertions: simulation ran and produced physical results
        assert I_peak_sim > 0
        assert T_peak_sim > 0

        # Circuit-only I_peak should be 2-5x experimental
        ratio = I_peak_sim / I_peak_exp
        assert 1.5 < ratio < 5.0, (
            f"I_peak ratio = {ratio:.2f}, expected 1.5-5.0"
        )
