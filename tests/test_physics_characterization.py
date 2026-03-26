"""S6 Physics characterization: A/B comparisons for optional physics modules.

Runs controlled comparisons on PF-1000 at coarse resolution (16x1x32, 100 steps)
to characterize the impact of:
  - HLLS vs HLL Riemann solver
  - Braginskii viscosity ON vs OFF
  - CIV anomalous resistivity ON vs OFF

Each comparison reports: max density ratio, max pressure ratio, max B_theta ratio,
and NaN detection across the two runs.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not installed")  # noqa: E402, I001

from dpf.config import SimulationConfig  # noqa: E402
from dpf.engine import SimulationEngine  # noqa: E402
from dpf.metal.mlx_device import HAS_MLX  # noqa: E402
from dpf.presets import get_preset  # noqa: E402

logger = logging.getLogger(__name__)

_METAL_GPU_AVAILABLE = HAS_MLX and (mlx.default_device().type == mlx.gpu)

requires_metal = pytest.mark.skipif(
    not _METAL_GPU_AVAILABLE,
    reason="Metal GPU not available",
)

N_STEPS = 100
GRID_SHAPE = [16, 1, 32]


@dataclass
class RunResult:
    """Collected metrics from a characterization run."""

    label: str
    n_steps: int
    I_peak_A: float
    rho_max: float
    p_max: float
    Bt_max: float
    has_nan: bool
    final_time: float


def _base_preset() -> dict[str, Any]:
    """PF-1000 preset with coarse grid and MLX HLL+PLM baseline."""
    preset = get_preset("pf1000")
    preset["grid_shape"] = list(GRID_SHAPE)
    preset["sim_time"] = 12e-6
    preset["fluid"] = {
        "backend": "mlx",
        "riemann_solver": "hll",
        "reconstruction": "plm",
        "time_integrator": "ssp_rk2",
        "precision": "float32",
        "use_ct": False,
        "enable_viscosity": False,
        "enable_hall": False,
    }
    preset["radiation"] = {"bremsstrahlung_enabled": False, "fld_enabled": False}
    preset["collision"] = {"enabled": False}
    preset["anomalous_civ_enabled"] = False
    return preset


def _run_engine(preset: dict[str, Any], label: str) -> RunResult:
    """Run the engine for N_STEPS and collect metrics."""
    config = SimulationConfig(**preset)
    engine = SimulationEngine(config)

    I_peak = 0.0
    rho_max = 0.0
    p_max = 0.0
    Bt_max = 0.0
    has_nan = False

    for _ in range(N_STEPS):
        result = engine.step()

        I_abs = abs(engine.circuit.current)
        if I_abs > I_peak:
            I_peak = I_abs

        state = engine.state
        rho = np.asarray(state["rho"], dtype=np.float64)
        pressure = np.asarray(state["pressure"], dtype=np.float64)
        B = np.asarray(state["B"], dtype=np.float64)

        if np.any(np.isnan(rho)) or np.any(np.isnan(pressure)):
            has_nan = True

        rho_max = max(rho_max, float(np.nanmax(rho)))
        p_max = max(p_max, float(np.nanmax(pressure)))

        # B_theta is component index 2 (Br, Bz, Bt)
        if B.ndim >= 1 and B.shape[0] >= 3:
            Bt_max = max(Bt_max, float(np.nanmax(np.abs(B[2]))))

        if result.finished:
            break

    return RunResult(
        label=label,
        n_steps=N_STEPS,
        I_peak_A=I_peak,
        rho_max=rho_max,
        p_max=p_max,
        Bt_max=Bt_max,
        has_nan=has_nan,
        final_time=engine.time,
    )


def _print_comparison(baseline: RunResult, variant: RunResult) -> dict[str, float]:
    """Print and return delta ratios between baseline and variant."""
    rho_ratio = variant.rho_max / baseline.rho_max if baseline.rho_max > 0 else float("inf")
    p_ratio = variant.p_max / baseline.p_max if baseline.p_max > 0 else float("inf")
    Bt_ratio = variant.Bt_max / baseline.Bt_max if baseline.Bt_max > 0 else float("inf")
    I_ratio = variant.I_peak_A / baseline.I_peak_A if baseline.I_peak_A > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"  {baseline.label} vs {variant.label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>14} {'Variant':>14} {'Ratio':>10}")
    print(f"  {'-'*58}")
    print(f"  {'I_peak [kA]':<20} {baseline.I_peak_A/1e3:>14.2f} {variant.I_peak_A/1e3:>14.2f} {I_ratio:>10.4f}")
    print(f"  {'rho_max [kg/m3]':<20} {baseline.rho_max:>14.6e} {variant.rho_max:>14.6e} {rho_ratio:>10.4f}")
    print(f"  {'p_max [Pa]':<20} {baseline.p_max:>14.6e} {variant.p_max:>14.6e} {p_ratio:>10.4f}")
    print(f"  {'Bt_max [T]':<20} {baseline.Bt_max:>14.6e} {variant.Bt_max:>14.6e} {Bt_ratio:>10.4f}")
    print(f"  {'NaN baseline':<20} {str(baseline.has_nan):>14}")
    print(f"  {'NaN variant':<20} {str(variant.has_nan):>14}")
    print(f"  {'final_time [us]':<20} {baseline.final_time*1e6:>14.4f} {variant.final_time*1e6:>14.4f}")

    return {
        "rho_ratio": rho_ratio,
        "p_ratio": p_ratio,
        "Bt_ratio": Bt_ratio,
        "I_ratio": I_ratio,
    }


@requires_metal
@pytest.mark.slow
@pytest.mark.validation
class TestPhysicsCharacterization:
    """A/B comparisons of optional physics modules on PF-1000."""

    def test_baseline_hll_plm_completes(self):
        """Baseline: HLL+PLM, no Hall, no viscosity, no CIV runs without NaN."""
        preset = _base_preset()
        result = _run_engine(preset, "HLL+PLM baseline")
        assert not result.has_nan, "Baseline produced NaN"
        assert result.I_peak_A > 0, "No current detected"
        assert result.rho_max > 0, "No density detected"

    def test_hlls_vs_hll(self):
        """HLLS entropy-based Riemann solver vs HLL: compare peak metrics."""
        baseline_preset = _base_preset()
        hlls_preset = copy.deepcopy(baseline_preset)
        hlls_preset["fluid"]["riemann_solver"] = "hlls"

        baseline = _run_engine(baseline_preset, "HLL")
        variant = _run_engine(hlls_preset, "HLLS")

        deltas = _print_comparison(baseline, variant)

        assert not baseline.has_nan, "HLL baseline produced NaN"
        assert not variant.has_nan, "HLLS variant produced NaN"

        # HLLS should produce broadly similar results (within 2x)
        assert 0.5 < deltas["rho_ratio"] < 2.0, (
            f"HLLS rho_max deviated {deltas['rho_ratio']:.2f}x from HLL"
        )
        assert 0.5 < deltas["p_ratio"] < 2.0, (
            f"HLLS p_max deviated {deltas['p_ratio']:.2f}x from HLL"
        )

    def test_viscosity_on_vs_off(self):
        """Braginskii viscosity ON vs OFF: viscosity should damp extremes."""
        baseline_preset = _base_preset()
        visc_preset = copy.deepcopy(baseline_preset)
        visc_preset["fluid"]["enable_viscosity"] = True

        baseline = _run_engine(baseline_preset, "viscosity OFF")
        variant = _run_engine(visc_preset, "viscosity ON")

        deltas = _print_comparison(baseline, variant)

        assert not baseline.has_nan, "Baseline produced NaN"
        assert not variant.has_nan, "Viscosity variant produced NaN"

        # Viscosity should not amplify density or pressure (ratio <= 1.5)
        assert deltas["rho_ratio"] < 1.5, (
            f"Viscosity amplified rho_max by {deltas['rho_ratio']:.2f}x"
        )

    def test_civ_on_vs_off(self):
        """CIV anomalous resistivity ON vs OFF."""
        baseline_preset = _base_preset()
        civ_preset = copy.deepcopy(baseline_preset)
        civ_preset["anomalous_civ_enabled"] = True
        civ_preset["anomalous_civ_alpha"] = 0.05

        baseline = _run_engine(baseline_preset, "CIV OFF")
        variant = _run_engine(civ_preset, "CIV ON")

        deltas = _print_comparison(baseline, variant)

        assert not baseline.has_nan, "Baseline produced NaN"
        assert not variant.has_nan, "CIV variant produced NaN"

        # CIV adds resistivity at sheath — should not blow up
        assert deltas["rho_ratio"] < 2.0, (
            f"CIV amplified rho_max by {deltas['rho_ratio']:.2f}x"
        )

    def test_summary_table(self, capsys):
        """Run all comparisons and print a summary table."""
        baseline_preset = _base_preset()
        baseline = _run_engine(baseline_preset, "HLL+PLM baseline")

        comparisons: list[tuple[str, RunResult]] = []

        # HLLS
        hlls_preset = copy.deepcopy(baseline_preset)
        hlls_preset["fluid"]["riemann_solver"] = "hlls"
        comparisons.append(("HLLS", _run_engine(hlls_preset, "HLLS")))

        # Viscosity
        visc_preset = copy.deepcopy(baseline_preset)
        visc_preset["fluid"]["enable_viscosity"] = True
        comparisons.append(("Viscosity", _run_engine(visc_preset, "Viscosity ON")))

        # CIV
        civ_preset = copy.deepcopy(baseline_preset)
        civ_preset["anomalous_civ_enabled"] = True
        comparisons.append(("CIV", _run_engine(civ_preset, "CIV ON")))

        print(f"\n{'='*72}")
        print("  S6 PHYSICS CHARACTERIZATION SUMMARY (PF-1000, 16x1x32, 100 steps)")
        print(f"{'='*72}")
        print(f"  {'Module':<12} {'rho ratio':>12} {'p ratio':>12} {'Bt ratio':>12} {'NaN?':>8}")
        print(f"  {'-'*56}")

        for name, variant in comparisons:
            rho_r = variant.rho_max / baseline.rho_max if baseline.rho_max > 0 else float("inf")
            p_r = variant.p_max / baseline.p_max if baseline.p_max > 0 else float("inf")
            Bt_r = variant.Bt_max / baseline.Bt_max if baseline.Bt_max > 0 else float("inf")
            nan_str = "YES" if variant.has_nan else "no"
            print(f"  {name:<12} {rho_r:>12.4f} {p_r:>12.4f} {Bt_r:>12.4f} {nan_str:>8}")

        print(f"{'='*72}")

        # All should be NaN-free
        for name, variant in comparisons:
            assert not variant.has_nan, f"{name} produced NaN"
