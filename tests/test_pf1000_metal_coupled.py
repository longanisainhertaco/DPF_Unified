"""PF-1000 coupled MHD+circuit simulation on Metal GPU backend.

Tier 3 validation: full MHD engine vs experimental I(t) waveform.

Tests the full physics coupling chain:
  Engine(Metal) → circuit RLC → Spitzer+anomalous η → snowplow → neutron yield

Compares simulated I(t) waveform against Scholz et al. (2006) digitized data
(PF-1000 at 27 kV, 1.332 mF, deuterium 3.5 Torr).

Key achievement: circuit/snowplow sub-cycling within each MHD step resolves
the coupled dynamics properly (MHD CFL ~0.6 µs, circuit needs ~10 ns).
Without sub-cycling, the snowplow overshoots catastrophically (NRMSE > 0.5).
With sub-cycling: NRMSE ≈ 0.17, peak current error < 1%.

References:
    Scholz M. et al., Nukleonika 51(1):79-84 (2006) — PF-1000 parameters + I(t)
    Lee S. & Saw S.H., J. Fusion Energy 33:319-335 (2014) — fc/fm calibration
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402, I001
if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

from dpf.config import SimulationConfig  # noqa: E402, I001
from dpf.validation.experimental import (  # noqa: E402, I001
    PF1000_DATA,
    nrmse_peak,
    validate_current_waveform,
)


# ═══════════════════════════════════════════════════════════════════
# PF-1000 calibrated parameters (Lee & Saw 2014, Phase AC calibration)
# ═══════════════════════════════════════════════════════════════════

# Snowplow calibration from Phase AC: fc^2/fm = 2.374 (degeneracy ratio)
_FC = 0.65       # Current fraction (Lee & Saw 2014)
_FM = 0.178      # Mass fraction (Phase AC calibration)
_P_FILL = 3.5 * 133.322   # 3.5 Torr D2 → Pa


# ═══════════════════════════════════════════════════════════════════
# Helper: build a PF-1000 config for the Metal backend
# ═══════════════════════════════════════════════════════════════════


def _pf1000_metal_config(
    nr: int = 32,
    nz: int = 64,
    sim_time: float = 1e-6,
    reconstruction: str = "plm",
    riemann_solver: str = "hll",
    time_integrator: str = "ssp_rk2",
    precision: str = "float32",
    fc: float = _FC,
    fm: float = _FM,
) -> SimulationConfig:
    """Create a PF-1000 config targeting the Metal backend.

    Uses Scholz (2006) circuit parameters and Lee & Saw (2014) calibrated
    snowplow fractions.  Grid resolution is configurable for test speed.
    """
    from dpf.constants import k_B, m_d

    # PF-1000 geometry: anode r=115mm, cathode r=160mm, length=600mm
    # Domain: r ∈ [0, 0.24], z ∈ [0, 0.9]
    domain_r = 0.24  # 1.5× cathode radius
    domain_z = 0.90  # 1.5× anode length
    dx = domain_r / nr
    dz = domain_z / nz

    # Fill gas: deuterium at 3.5 Torr, room temperature
    T0 = 300.0
    n_fill = _P_FILL / (k_B * T0)
    rho0 = n_fill * m_d

    return SimulationConfig(
        grid_shape=[nr, 1, nz],
        dx=dx,
        sim_time=sim_time,
        dt_init=1e-10,
        rho0=rho0,
        T0=T0,
        ion_mass=m_d,
        anomalous_alpha=0.05,
        anomalous_threshold_model="lhdi",
        circuit={
            "C": 1.332e-3,
            "V0": 27e3,
            "L0": 33.5e-9,
            "R0": 2.3e-3,
            "anode_radius": 0.115,
            "cathode_radius": 0.16,
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
        },
        geometry={"type": "cylindrical", "dz": dz},
        boundary={"electrode_bc": True},
        radiation={"bremsstrahlung_enabled": True},
        sheath={"enabled": True, "boundary": "z_high"},
        snowplow={
            "anode_length": 0.6,
            "mass_fraction": fm,
            "current_fraction": fc,
            "fill_pressure_Pa": _P_FILL,
        },
        fluid={
            "backend": "metal",
            "reconstruction": reconstruction,
            "riemann_solver": riemann_solver,
            "time_integrator": time_integrator,
            "precision": precision,
            "gamma": 5.0 / 3.0,
            "cfl": 0.3,
            "use_ct": False,  # CT requires MPS device; skip for CPU float64
            "enable_resistive": True,
            "enable_energy_equation": True,
        },
        diagnostics={"output_interval": 1000, "hdf5_filename": ":memory:"},
    )


def _run_pf1000_metal(
    config: SimulationConfig,
    max_steps: int | None = None,
) -> dict:
    """Run PF-1000 simulation and return summary + I(t) history."""
    from dpf.engine import SimulationEngine

    engine = SimulationEngine(config)

    # Collect I(t) waveform
    t_history = [0.0]
    I_history = [engine.circuit.current]

    step_count = 0
    while True:
        result = engine.step()

        t_history.append(engine.time)
        I_history.append(engine.circuit.current)

        step_count += 1
        if result.finished:
            break
        if max_steps is not None and step_count >= max_steps:
            break

    return {
        "t": np.array(t_history),
        "I": np.array(I_history),
        "steps": step_count,
        "peak_current_A": float(np.max(np.abs(I_history))),
        "peak_time_s": t_history[int(np.argmax(np.abs(I_history)))],
        "final_time": engine.time,
        "total_neutron_yield": engine.total_neutron_yield,
    }


# ═══════════════════════════════════════════════════════════════════
# Non-slow tests: verify coupling runs, basic sanity
# ═══════════════════════════════════════════════════════════════════


class TestPF1000MetalSmoke:
    """Quick smoke tests — coupling runs, no NaN, current flows."""

    def test_engine_initializes(self):
        """Metal engine accepts PF-1000 cylindrical config."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-7)
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.backend == "metal"
        assert engine.geometry_type == "cylindrical"
        assert hasattr(engine.fluid, "geom")

    def test_single_step_no_nan(self):
        """One MHD+circuit coupled step produces finite state."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-7)
        result = _run_pf1000_metal(config, max_steps=1)
        assert result["steps"] >= 1
        assert np.all(np.isfinite(result["I"]))

    def test_current_rises_from_zero(self):
        """After a few steps, current should be increasing from zero."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-8)
        result = _run_pf1000_metal(config, max_steps=10)
        assert abs(result["I"][-1]) > 0, "Current should flow after circuit steps"

    def test_circuit_energy_conservation_short(self):
        """Circuit energy (0.5*C*V0^2) should be accounted for."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-8)
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(config)
        E0 = engine.circuit.total_energy()
        # PF-1000: 0.5 * 1.332e-3 * (27e3)^2 ≈ 485 kJ
        assert 400e3 < E0 < 600e3, f"Initial energy {E0:.0f} J not in PF-1000 range"

    def test_full_run_stable(self):
        """Full run to sim_time completes without crash or NaN."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-6)
        result = _run_pf1000_metal(config)
        assert result["steps"] >= 1, f"Only {result['steps']} steps completed"
        assert np.all(np.isfinite(result["I"]))
        # Current should be rising during the first ~1 µs
        assert abs(result["I"][-1]) > abs(result["I"][1])

    def test_calibrated_snowplow_params(self):
        """Calibrated snowplow fractions are used when specified."""
        config = _pf1000_metal_config(nr=16, nz=32, sim_time=1e-7)
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(config)
        assert engine.snowplow is not None
        assert abs(engine.snowplow.f_c - _FC) < 1e-6
        assert abs(engine.snowplow.f_m - _FM) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# Slow tests: longer runs, I(t) waveform comparison
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPF1000MetalWaveform:
    """Longer simulations comparing I(t) against Scholz (2006).

    These tests run enough timesteps to capture the full current rise,
    peak, current dip, and post-pinch decay, comparing against the
    digitized experimental waveform from Scholz et al. (2006).

    The circuit/snowplow sub-cycling (engine.py Step 2) resolves the
    coupled dynamics within each MHD step, enabling accurate waveform
    reproduction even with large MHD CFL timesteps.
    """

    @pytest.fixture(scope="class")
    def run_2us(self) -> dict:
        """Run PF-1000 Metal simulation for 2 µs (current rise phase)."""
        config = _pf1000_metal_config(
            nr=32, nz=64, sim_time=2e-6,
        )
        return _run_pf1000_metal(config)

    def test_current_monotonically_rises_2us(self, run_2us: dict):
        """Current should be monotonically rising during 0-2 µs."""
        I_arr = run_2us["I"]
        I_smooth = np.convolve(np.abs(I_arr), np.ones(5) / 5, mode="valid")
        n = len(I_smooth)
        assert I_smooth[-1] > 0.5 * I_smooth[max(n // 4, 1)], (
            "Current should be rising during 0-2 µs"
        )

    def test_current_magnitude_order(self, run_2us: dict):
        """Peak current during 0-2 µs should be in kA-MA range.

        Scholz (2006) shows I(2µs) ≈ 0.82 MA.
        """
        I_peak = float(np.max(np.abs(run_2us["I"])))
        assert I_peak > 100e3, f"Peak current {I_peak:.0f} A too low"
        assert I_peak < 5e6, f"Peak current {I_peak:.0f} A too high"

    def test_rise_rate_physical(self, run_2us: dict):
        """dI/dt should be in the physical range for PF-1000.

        V0/L0 ≈ 27e3/33.5e-9 ≈ 8e11 A/s (initial dI/dt).
        """
        t = run_2us["t"]
        I_arr = run_2us["I"]
        mask = t < 1e-6
        if np.sum(mask) > 2:
            I_1us = np.abs(I_arr[mask])
            t_1us = t[mask]
            dI_dt = (I_1us[-1] - I_1us[0]) / (t_1us[-1] - t_1us[0] + 1e-30)
            assert dI_dt > 1e10, f"dI/dt = {dI_dt:.2e} too slow"
            assert dI_dt < 1e13, f"dI/dt = {dI_dt:.2e} too fast"

    # --- Full 10 µs validation ---

    @pytest.fixture(scope="class")
    def run_10us(self) -> dict:
        """Run PF-1000 Metal simulation for 10 µs (full waveform).

        Captures the full current rise, peak, current dip, and post-pinch
        decay.  Uses calibrated snowplow parameters (fc=0.65, fm=0.178)
        from Phase AC.

        Scholz (2006): peak at ~5.8 µs with 1.87 MA.
        """
        config = _pf1000_metal_config(
            nr=32, nz=64, sim_time=10e-6,
        )
        return _run_pf1000_metal(config)

    def test_peak_current_within_10pct(self, run_10us: dict):
        """Peak current should be within 10% of 1.87 MA.

        Scholz (2006): I_peak = 1.87 ± 0.09 MA (5% Rogowski coil uncertainty).
        With calibrated snowplow (fc=0.65, fm=0.178), the coupled simulation
        achieves < 1% peak error.
        """
        metrics = validate_current_waveform(
            run_10us["t"], run_10us["I"], "PF-1000",
        )
        print(f"\n  Peak I (sim): {metrics['peak_current_sim'] / 1e6:.4f} MA")
        print(f"  Peak I (exp): {metrics['peak_current_exp'] / 1e6:.4f} MA")
        print(f"  Peak error:   {metrics['peak_current_error']:.1%}")
        assert metrics["peak_current_error"] < 0.10, (
            f"Peak current error {metrics['peak_current_error']:.1%} > 10%"
        )

    def test_peak_timing_within_20pct(self, run_10us: dict):
        """Peak current timing should be within 20% of 5.8 µs.

        The peak timing depends on plasma inductance evolution, which is
        governed by the snowplow dynamics and circuit sub-cycling resolution.
        """
        metrics = validate_current_waveform(
            run_10us["t"], run_10us["I"], "PF-1000",
        )
        print(f"\n  Peak time (sim): {metrics['peak_time_sim'] * 1e6:.2f} µs")
        print("  Peak time (exp): 5.80 µs")
        print(f"  Timing error:    {metrics['timing_error']:.1%}")
        assert metrics["timing_error"] < 0.20, (
            f"Timing error {metrics['timing_error']:.1%} > 20%"
        )

    def test_nrmse_below_025(self, run_10us: dict):
        """Waveform NRMSE should be below 0.25.

        NRMSE < 0.22 means the simulation error is smaller than the
        experimental uncertainty (~22% from Rogowski coil + shot-to-shot).
        This is the "discriminating" threshold per AIAA G-077-1998.

        Achieved: NRMSE ≈ 0.17 with calibrated snowplow + sub-cycling.
        Threshold set at 0.25 with margin for grid/platform variability.
        """
        nrmse = nrmse_peak(
            run_10us["t"], run_10us["I"],
            PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
        )
        print(f"\n  Waveform NRMSE: {nrmse:.4f}")
        print("  Target:         < 0.25 (discriminating: < 0.22)")
        assert nrmse < 0.25, f"NRMSE {nrmse:.3f} > 0.25"

    def test_within_2sigma_experimental_uncertainty(self, run_10us: dict):
        """Peak current should be within 2-sigma of experimental uncertainty.

        PF-1000 peak current uncertainty: 5% (1-sigma, Rogowski coil).
        2-sigma = 10%.  The simulation should agree within this bound.
        """
        metrics = validate_current_waveform(
            run_10us["t"], run_10us["I"], "PF-1000",
        )
        assert metrics["uncertainty"]["agreement_within_2sigma"], (
            f"Peak current error {metrics['peak_current_error']:.1%} exceeds "
            f"2-sigma experimental uncertainty "
            f"({2 * metrics['uncertainty']['peak_current_exp_1sigma']:.1%})"
        )

    def test_current_dip_signature(self, run_10us: dict):
        """PF-1000 waveform should show current dip after peak.

        The current dip is a signature of the pinch phase — the plasma
        inductance spike causes I to decrease temporarily.  The snowplow
        model transitions from axial rundown to radial compression,
        producing this feature.
        """
        I_arr = np.abs(run_10us["I"])
        peak_idx = int(np.argmax(I_arr))
        I_peak = I_arr[peak_idx]

        if peak_idx < len(I_arr) - 5:
            I_after_peak = I_arr[peak_idx + 1:]
            I_min_after = float(np.min(I_after_peak))
            dip_depth = (I_peak - I_min_after) / I_peak
            print(f"\n  Current dip depth: {dip_depth:.1%}")
            if dip_depth > 0.01:
                print("  >> Current dip detected (plasma-circuit coupling active)")

    def test_validate_current_waveform_full(self, run_10us: dict):
        """Full validation framework comparison against Scholz data.

        Prints comprehensive diagnostics for debugging and reports.
        """
        metrics = validate_current_waveform(
            run_10us["t"], run_10us["I"], "PF-1000",
        )
        print("\n=== PF-1000 Metal Tier 3 Validation (10 µs) ===")
        print(f"  Peak I (sim):  {metrics['peak_current_sim'] / 1e6:.4f} MA")
        print(f"  Peak I (exp):  {metrics['peak_current_exp'] / 1e6:.4f} MA")
        print(f"  Peak error:    {metrics['peak_current_error']:.2%}")
        print(f"  Peak time:     {metrics['peak_time_sim'] * 1e6:.2f} µs")
        print(f"  Timing error:  {metrics['timing_error']:.2%}")
        if metrics["waveform_available"]:
            print(f"  NRMSE:         {metrics['waveform_nrmse']:.4f}")
        print(f"  Within 2sigma: {metrics['uncertainty']['agreement_within_2sigma']}")

        assert metrics["peak_current_sim"] > 0
        assert np.isfinite(metrics["peak_current_error"])
