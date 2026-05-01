"""CI validation tests for DPF-Unified.

Runs production circuit solver against experimental waveforms for all devices
with digitized data. Fast enough for pre-push hooks (<30s total).

Usage::

    pytest -m validation           # run only validation tests
    pytest -m "not validation"     # skip validation tests
    pytest tests/test_validation_ci.py -v  # verbose validation run
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CI_THRESHOLDS, device_tol
from dpf.validation.engine_validation import (
    compare_engine_vs_experiment,
    run_rlc_snowplow_pf1000,
)
from dpf.validation.experimental_comparison import nrmse_peak
from dpf.validation.experimental_devices import DEVICES

# ---------------------------------------------------------------------------
# Thresholds (centralized in conftest.py)
# ---------------------------------------------------------------------------
NRMSE_WARN = CI_THRESHOLDS["nrmse_warn"]
NRMSE_FAIL = CI_THRESHOLDS["nrmse_fail"]
IPEAK_ERR_FAIL = CI_THRESHOLDS["ipeak_fail"]

# Devices with digitized waveforms (name -> expected approx I_peak for sanity)
WAVEFORM_DEVICES = {
    "PF-1000": 1.87e6,
    "PF-1000-Gribkov": 1.846e6,
    "UNU-ICTP": 182e3,  # Updated: 182 kA per KR p.152 [Lee & Saw 2014] (was 169 kA from IPFS 13.5 kV file)
    "PF-1000-16kV": 1.2e6,
    "POSEIDON-60kV": 3.19e6,
    "FAETON-I": 1.0e6,
    "MJOLNIR": 2.8e6,
}


def _run_circuit_for_device(device_name: str):
    """Run RLC+snowplow for a device and return (t, I, summary)."""
    dev = DEVICES[device_name]

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.constants import k_B, m_D2
    from dpf.core.bases import CouplingState
    from dpf.fluid.snowplow import SnowplowModel

    p_Pa = dev.fill_pressure_torr * 133.322
    rho0 = (p_Pa / (k_B * 300.0)) * m_D2

    circuit = RLCSolver(
        C=dev.capacitance, V0=dev.voltage, L0=dev.inductance, R0=dev.resistance,
        anode_radius=dev.anode_radius, cathode_radius=dev.cathode_radius,
        crowbar_enabled=True, crowbar_mode="voltage_zero",
    )

    fc = dev.lee_fc or 0.7
    fm = dev.lee_fm or 0.13
    fmr = dev.lee_fmr or 0.1

    snowplow = SnowplowModel(
        anode_radius=dev.anode_radius,
        cathode_radius=dev.cathode_radius,
        fill_density=rho0,
        anode_length=dev.anode_length,
        mass_fraction=fm,
        current_fraction=fc,
        radial_mass_fraction=fmr,
        fill_pressure_Pa=p_Pa,
    )

    # Determine sim_time from device rise time (2.5x to capture post-peak)
    sim_time = max(2.5 * (dev.current_rise_time or 6e-6), 8e-6)
    dt = min(1e-9, sim_time / 10000)
    n_steps = int(sim_time / dt)

    coupling = CouplingState()
    t_list = [0.0]
    I_list = [0.0]
    t = 0.0

    for _ in range(n_steps):
        sp_result = snowplow.step(dt, coupling.current)
        coupling.Lp = sp_result["L_plasma"]
        coupling.dL_dt = sp_result["dL_dt"]
        coupling.R_plasma = sp_result.get("R_plasma", 0.0)
        coupling = circuit.step(coupling, 0.0, dt)
        t += dt
        t_list.append(t)
        I_list.append(coupling.current)

    return np.array(t_list), np.array(I_list)


# ---------------------------------------------------------------------------
# PF-1000 reference validation (most important single test)
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPF1000Validation:
    """PF-1000 circuit validation against Scholz et al. 2006."""

    def test_pf1000_peak_current(self):
        """Peak current within device tolerance of 1.87 MA."""
        tol = device_tol("PF-1000")
        t, I, summary = run_rlc_snowplow_pf1000()
        I_peak = np.max(np.abs(I))
        error = abs(I_peak - 1.87e6) / 1.87e6
        assert error < tol["I_peak"], f"I_peak={I_peak/1e6:.3f} MA, error={error:.1%}"

    def test_pf1000_waveform_nrmse(self):
        """Waveform NRMSE below device threshold against Scholz 26-point data."""
        tol = device_tol("PF-1000")
        result = compare_engine_vs_experiment(*run_rlc_snowplow_pf1000()[:2])
        assert result.waveform_nrmse < tol["nrmse"], (
            f"NRMSE={result.waveform_nrmse:.3f} exceeds device threshold {tol['nrmse']}"
        )

    def test_pf1000_gribkov_nrmse(self):
        """Waveform NRMSE below Gribkov device threshold (94-point data).

        Higher threshold than Scholz because Gribkov has 94 points covering
        the full discharge including post-peak where the snowplow model
        diverges. Scholz (26 pts) stops earlier, hiding the post-peak gap.
        """
        tol = device_tol("PF-1000-Gribkov")
        result = compare_engine_vs_experiment(
            *run_rlc_snowplow_pf1000()[:2],
            device_name="PF-1000-Gribkov",
        )
        assert result.waveform_nrmse < tol["nrmse"], (
            f"NRMSE={result.waveform_nrmse:.3f} vs Gribkov exceeds {tol['nrmse']}"
        )

    def test_pf1000_energy_conservation(self):
        """Energy conservation ratio < 1.05 (no energy creation)."""
        _, _, summary = run_rlc_snowplow_pf1000()
        e_ratio = summary.get("energy_conservation", 1.0)
        assert e_ratio <= 1.05, f"Energy ratio {e_ratio:.3f} — energy not conserved"

    def test_pf1000_dual_reference(self):
        """Compare against both Scholz (26-pt) and Gribkov (94-pt).

        Scholz 2006: 26 hand-digitized points, t_peak ~5.8 us
        Gribkov 2007: 94 digital oscilloscope points, flat-top 5.2-6.6 us

        Both measure the same device (PF-1000 at 27 kV, 3.5 Torr D2).
        Gribkov's t_peak ambiguity (1.5% variation across flat-top) means
        timing error vs Gribkov is more informative than vs Scholz.
        """
        t, I, _ = run_rlc_snowplow_pf1000()
        scholz = compare_engine_vs_experiment(t, I, device_name="PF-1000")
        gribkov = compare_engine_vs_experiment(t, I, device_name="PF-1000-Gribkov")

        # Scholz threshold: 10% reflects Akel 2021 device params + Lee snowplow
        # accuracy on PF-1000 at 27 kV. Scholz hand-digitization noise + finite
        # mass-loading window keep error in 8-10% band; tighter threshold held
        # only when params were uncalibrated. Per 2026-04-24 bisect.
        assert scholz.peak_current_error < 0.10, (
            f"Scholz I_peak error {scholz.peak_current_error:.1%}"
        )
        # Gribkov threshold: 10% — same Akel 2021 calibration band as Scholz.
        # Post zipper-BC fix (commit 5b54f0a, 2026-04-27), Python backend
        # behaves correctly and Gribkov error sits at ~8% (was 5.3% on the
        # broken-zipper baseline that artificially suppressed peak current).
        # Published params are inputs, not knobs — the 8% floor is real.
        assert gribkov.peak_current_error < 0.10, (
            f"Gribkov I_peak error {gribkov.peak_current_error:.1%}"
        )

        # Scholz NRMSE should be lower (fewer post-peak points)
        assert scholz.waveform_nrmse < gribkov.waveform_nrmse, (
            f"Expected Scholz NRMSE < Gribkov, got "
            f"{scholz.waveform_nrmse:.3f} vs {gribkov.waveform_nrmse:.3f}"
        )


# ---------------------------------------------------------------------------
# Multi-device validation (all devices with waveforms)
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestMultiDeviceValidation:
    """Circuit validation across all devices with digitized waveforms."""

    @pytest.mark.parametrize("device_name", sorted(WAVEFORM_DEVICES.keys()))
    def test_device_no_nan(self, device_name):
        """Device simulation completes without NaN."""
        t, I = _run_circuit_for_device(device_name)
        assert not np.any(np.isnan(I)), f"{device_name} produced NaN"
        assert np.max(np.abs(I)) > 0, f"{device_name} produced zero current"

    @pytest.mark.parametrize("device_name", sorted(WAVEFORM_DEVICES.keys()))
    def test_device_peak_current_order(self, device_name):
        """Peak current within 50% of expected (order-of-magnitude sanity)."""
        expected = WAVEFORM_DEVICES[device_name]
        t, I = _run_circuit_for_device(device_name)
        I_peak = np.max(np.abs(I))
        ratio = I_peak / expected
        assert 0.5 < ratio < 2.0, (
            f"{device_name}: I_peak={I_peak:.0f} A, "
            f"expected ~{expected:.0f} A, ratio={ratio:.2f}"
        )

    # Devices where NRMSE > 0.30 is expected due to reconstructed waveforms
    # or extreme operating conditions not well-captured by the snowplow model
    # Devices where NRMSE > 0.30 is expected: reconstructed waveforms,
    # 94-point full-discharge data that exposes post-peak model limitations,
    # or extreme operating conditions
    XFAIL_NRMSE = {"MJOLNIR", "PF-1000-Gribkov"}

    @pytest.mark.parametrize(
        "device_name",
        [d for d in sorted(WAVEFORM_DEVICES.keys())
         if DEVICES.get(d) and DEVICES[d].waveform_t is not None],
    )
    def test_device_waveform_nrmse(self, device_name):
        """Waveform NRMSE below hard-fail threshold."""
        dev = DEVICES[device_name]
        t, I = _run_circuit_for_device(device_name)
        nrmse = nrmse_peak(t, I, dev.waveform_t, dev.waveform_I)
        if device_name in self.XFAIL_NRMSE:
            pytest.xfail(f"{device_name}: reconstructed waveform, NRMSE={nrmse:.3f}")
        assert nrmse < NRMSE_FAIL, (
            f"{device_name}: NRMSE={nrmse:.3f} exceeds fail threshold {NRMSE_FAIL}"
        )


# ---------------------------------------------------------------------------
# Preset smoke tests (all 16 presets, no waveform comparison)
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPresetSmoke:
    """Verify all presets can instantiate and run without crashing."""

    def test_all_presets_load(self):
        """All presets load without error."""
        from dpf.presets import get_preset, get_preset_names
        names = get_preset_names()
        assert len(names) >= 14, f"Only {len(names)} presets found"
        for name in names:
            preset = get_preset(name)
            assert isinstance(preset, dict), f"Preset {name} is not a dict"


# ---------------------------------------------------------------------------
# MLX MHD validation (requires Apple Silicon + MLX)
# ---------------------------------------------------------------------------

try:
    import mlx.core  # noqa: F401 — used for availability check
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLXValidation:
    """MLX MHD solver validation against PF-1000."""

    def test_mlx_pf1000_100steps(self):
        """MLX solver runs 100 steps without NaN on PF-1000 grid."""
        from dpf.metal.mlx_solver import MLXMHDSolver

        nr, nz = 16, 32
        dr = (0.16 - 0.01) / nr
        dz = 0.60 / nz
        solver = MLXMHDSolver(
            grid_shape=(nr, 1, nz),
            dx=dr,
            dz=dz,
            riemann_solver="hll",
            reconstruction="plm",
            cfl=0.3,
        )

        shape = (nr, 1, nz)
        state = {
            "rho": np.full(shape, 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full(shape, 1e3),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full(shape, 1e4),
            "Ti": np.full(shape, 1e4),
            "psi": np.zeros(shape),
        }

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        rho = np.asarray(state["rho"])
        assert not np.any(np.isnan(rho)), "MLX solver produced NaN in density"
        assert np.all(rho > 0), "MLX solver produced negative density"
