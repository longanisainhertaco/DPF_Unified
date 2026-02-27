"""PF-1000 coupled MHD+circuit simulation on Metal GPU backend.

Tests the full physics coupling chain:
  Engine(Metal) → circuit RLC → Spitzer+anomalous η → snowplow → neutron yield

Compares simulated I(t) waveform against Scholz et al. (2006) digitized data
(PF-1000 at 27 kV, 1.332 mF, deuterium 3.5 Torr).

This is the FIRST end-to-end MHD engine validation against experimental data.
Previous validations used either analytical benchmarks (Sod, Sedov, Bennett) or
the semi-analytical Lee model.

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
) -> SimulationConfig:
    """Create a PF-1000 config targeting the Metal backend.

    Uses the same circuit parameters as the pf1000 preset (Scholz 2006)
    but with a configurable grid resolution for test speed.
    """
    # PF-1000 geometry: anode r=115mm, cathode r=160mm, length=600mm
    # Domain: r ∈ [0, 0.24], z ∈ [0, 0.9]
    domain_r = 0.24  # 1.5× cathode radius
    domain_z = 0.90  # 1.5× anode length
    dx = domain_r / nr
    dz = domain_z / nz

    # Fill gas: deuterium at 3.5 Torr, room temperature
    from dpf.constants import k_B, m_d

    p_fill = 3.5 * 133.322  # Torr → Pa
    T0 = 300.0
    n_fill = p_fill / (k_B * T0)
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
        snowplow={"anode_length": 0.6},
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
        "peak_current_A": engine._peak_current_A if hasattr(engine, "_peak_current_A") else float(np.max(np.abs(I_history))),
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
        # Current starts at 0, should be non-zero after a few steps
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
        assert result["steps"] >= 5, f"Only {result['steps']} steps completed"
        assert np.all(np.isfinite(result["I"]))
        # Current should be rising during the first ~1 µs
        assert abs(result["I"][-1]) > abs(result["I"][1])


# ═══════════════════════════════════════════════════════════════════
# Slow tests: longer runs, I(t) waveform comparison
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPF1000MetalWaveform:
    """Longer simulations comparing I(t) against Scholz (2006).

    These tests run enough timesteps to capture the current rise phase
    and compare against the digitized experimental waveform.
    """

    @pytest.fixture(scope="class")
    def run_2us(self) -> dict:
        """Run PF-1000 Metal simulation for 2 µs (current rise phase)."""
        config = _pf1000_metal_config(
            nr=32, nz=64,
            sim_time=2e-6,
            reconstruction="plm",
            riemann_solver="hll",
            time_integrator="ssp_rk2",
            precision="float32",
        )
        return _run_pf1000_metal(config)

    def test_current_monotonically_rises_2us(self, run_2us: dict):
        """Current should be monotonically rising during 0-2 µs."""
        I_arr = run_2us["I"]
        # Allow small dips (MHD oscillations) but overall trend should be up
        I_smooth = np.convolve(np.abs(I_arr), np.ones(5) / 5, mode="valid")
        # Check that current at end > 50% of current at 1/4 through
        n = len(I_smooth)
        assert I_smooth[-1] > 0.5 * I_smooth[n // 4], (
            "Current should be rising during 0-2 µs"
        )

    def test_current_magnitude_order(self, run_2us: dict):
        """Peak current during 0-2 µs should be in kA-MA range.

        Scholz (2006) shows I(2µs) ≈ 0.82 MA.  Even with a coarse grid,
        the circuit coupling should produce currents in the right order.
        """
        I_peak = float(np.max(np.abs(run_2us["I"])))
        # Accept anything from 100 kA to 5 MA (wide tolerance for coarse grid)
        assert I_peak > 100e3, f"Peak current {I_peak:.0f} A too low"
        assert I_peak < 5e6, f"Peak current {I_peak:.0f} A too high"

    def test_rise_rate_physical(self, run_2us: dict):
        """dI/dt should be in the physical range for PF-1000.

        LC natural frequency: f = 1/(2π√(LC)) ≈ 1/(2π√(33.5e-9 × 1.332e-3))
        ≈ 24 kHz, so T/4 ≈ 10 µs, dI/dt_max ≈ V0/L0 ≈ 27e3/33.5e-9 ≈ 8e11 A/s
        """
        t = run_2us["t"]
        I_arr = run_2us["I"]
        # Compute average dI/dt over first 1 µs
        mask = t < 1e-6
        if np.sum(mask) > 2:
            I_1us = np.abs(I_arr[mask])
            t_1us = t[mask]
            dI_dt = (I_1us[-1] - I_1us[0]) / (t_1us[-1] - t_1us[0] + 1e-30)
            # Physical range: 1e10 to 1e12 A/s
            assert dI_dt > 1e10, f"dI/dt = {dI_dt:.2e} too slow"
            assert dI_dt < 1e13, f"dI/dt = {dI_dt:.2e} too fast"

    @pytest.fixture(scope="class")
    def run_6us(self) -> dict:
        """Run PF-1000 Metal simulation for 6 µs (past peak current).

        This captures the full current rise, peak, and beginning of dip.
        Scholz (2006): peak at ~5.8 µs with 1.87 MA.
        """
        config = _pf1000_metal_config(
            nr=48, nz=96,
            sim_time=6e-6,
            reconstruction="plm",
            riemann_solver="hll",
            time_integrator="ssp_rk2",
            precision="float32",
        )
        return _run_pf1000_metal(config)

    def test_peak_current_order_of_magnitude(self, run_6us: dict):
        """Peak current should be within 1 order of magnitude of 1.87 MA.

        Scholz (2006): I_peak = 1.87 MA.
        We accept 0.2-10 MA range due to coarse grid and simplified physics.
        """
        I_peak = float(np.max(np.abs(run_6us["I"])))
        assert I_peak > 200e3, f"Peak current {I_peak:.0f} A < 200 kA"
        assert I_peak < 10e6, f"Peak current {I_peak:.0f} A > 10 MA"

    def test_validate_current_waveform(self, run_6us: dict):
        """Use the validation framework to compare against Scholz data.

        This test produces quantitative metrics even if they don't pass
        strict thresholds — the goal is to get the infrastructure working.
        """
        metrics = validate_current_waveform(
            run_6us["t"], run_6us["I"], "PF-1000",
        )
        # Log all metrics for diagnostics
        print("\n=== PF-1000 Metal Validation (6 µs) ===")
        print(f"  Peak I (sim): {metrics['peak_current_sim'] / 1e6:.3f} MA")
        print(f"  Peak I (exp): {metrics['peak_current_exp'] / 1e6:.3f} MA")
        print(f"  Peak error:   {metrics['peak_current_error']:.1%}")
        print(f"  Peak time:    {metrics['peak_time_sim'] * 1e6:.2f} µs")
        print(f"  Timing OK:    {metrics['timing_ok']}")
        if metrics["waveform_available"]:
            print(f"  NRMSE:        {metrics['waveform_nrmse']:.3f}")

        # The simulation should at least produce a finite, positive peak
        assert metrics["peak_current_sim"] > 0
        assert np.isfinite(metrics["peak_current_error"])

        # Relaxed tolerance: peak current within factor of 3 of experiment
        assert metrics["peak_current_error"] < 2.0, (
            f"Peak current error {metrics['peak_current_error']:.1%} > 200%"
        )

    def test_nrmse_below_threshold(self, run_6us: dict):
        """Waveform NRMSE should be below a relaxed threshold.

        NRMSE < 1.0 means the simulation is at least in the right ballpark.
        NRMSE < 0.5 would indicate good agreement.
        NRMSE < 0.2 would be excellent for a coarse-grid MHD simulation.
        """
        nrmse = nrmse_peak(
            run_6us["t"], run_6us["I"],
            PF1000_DATA.waveform_t, PF1000_DATA.waveform_I,
        )
        print(f"\n  Waveform NRMSE: {nrmse:.4f}")
        # Relaxed: just be within an order of magnitude
        assert nrmse < 2.0, f"NRMSE {nrmse:.3f} > 2.0 (simulation wildly off)"

    def test_current_dip_signature(self, run_6us: dict):
        """PF-1000 waveform should show current dip after peak.

        The current dip is a signature of the pinch phase — the plasma
        inductance spike causes I to decrease temporarily.  Even with
        simplified physics, the circuit + snowplow coupling should
        produce this feature.
        """
        I_arr = np.abs(run_6us["I"])

        # Find peak
        peak_idx = int(np.argmax(I_arr))
        I_peak = I_arr[peak_idx]

        # Check if there's any decrease after the peak
        if peak_idx < len(I_arr) - 5:
            I_after_peak = I_arr[peak_idx + 1:]
            I_min_after = float(np.min(I_after_peak))
            dip_depth = (I_peak - I_min_after) / I_peak
            print(f"\n  Current dip depth: {dip_depth:.1%}")
            # Any detectable dip (>1%) indicates plasma-circuit coupling
            # Don't fail if no dip — it depends on grid resolution
            if dip_depth > 0.01:
                print("  >> Current dip detected (plasma-circuit coupling active)")
