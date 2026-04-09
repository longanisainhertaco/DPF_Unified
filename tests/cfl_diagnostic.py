#!/usr/bin/env python3
"""CFL diagnostic: detect timestep collapse from vacuum Alfven speed spikes.

Runs a short MHD discharge (100 steps) and reports dt_min, dt_max, and the
cell that determines the CFL limit. If dt_min < 1e-12, the solver has a
vacuum v_Alfven problem that must be fixed before debugging L_p coupling.

From v3.4 architecture: "If dt_min < 1e-12 due to v_Alfven spike in vacuum,
fix the vacuum treatment. Do NOT build orchestration around a physics bug
masquerading as a wait time."
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpf.metal.mlx_device import HAS_MLX

if not HAS_MLX:
    print("ERROR: MLX not available", file=sys.stderr)
    sys.exit(1)

import mlx.core as mx
from dpf.metal.mlx_engine import run_mlx_discharge


def main() -> None:
    print("=== CFL Diagnostic ===")
    print("Running 100-step PF-1000 MHD discharge...")
    t0 = time.time()

    result = run_mlx_discharge(
        "pf1000",
        mode="mhd",
        max_steps=100,  # short — just enough to measure dt
        grid_shape=(32, 1, 64),
    )

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print()

    # Extract dt info from result — format depends on run_mlx_discharge output
    # The function returns time-series arrays: t_us, I_MA, Lp_nH, etc.
    t_arr = np.asarray(result.get("t_us", []))
    if len(t_arr) < 2:
        print("WARNING: Not enough timesteps returned.")
        print(f"Result keys: {list(result.keys())}")
        print(f"I_peak = {result.get('I_peak_MA', '?')} MA")
        print(f"n_steps = {result.get('n_steps', '?')}")
        print(f"elapsed = {result.get('elapsed_s', '?')} s")
        return

    # Compute dt from time series (t is in microseconds)
    t_s = t_arr * 1e-6  # convert to seconds
    dts = np.diff(t_s)
    dts = dts[dts > 0]  # filter zero/negative
    if len(dts) == 0:
        print("No valid dt values computed from time series.")
        return

    dt_min = min(dts)
    dt_max = max(dts)
    dt_mean = np.mean(dts)
    n_steps = len(dts)

    print(f"Steps:  {n_steps}")
    print(f"dt_min: {dt_min:.3e} s")
    print(f"dt_max: {dt_max:.3e} s")
    print(f"dt_avg: {dt_mean:.3e} s")
    print()

    if dt_min < 1e-12:
        print("CRITICAL: dt collapsed below 1e-12 s.")
        print("This indicates a vacuum v_Alfven spike.")
        print("FIX: Mask vacuum cells from CFL computation (rho < 1e-4 * rho_max).")
        print("Do NOT proceed with L_p debugging until this is fixed.")
    elif dt_min < 1e-10:
        print("WARNING: dt is very small (< 1e-10 s).")
        print("May indicate high v_Alfven in low-density regions.")
        print("Check vacuum treatment before proceeding.")
    else:
        print("OK: CFL timestep is reasonable.")
        print("Proceed to L_p coupling diagnosis.")

    # Estimate total steps for full 8 us discharge
    t_total = 8e-6
    est_steps = int(t_total / dt_mean) if dt_mean > 0 else 0
    est_time = est_steps * elapsed / max(n_steps, 1)
    print(f"\nEstimated full discharge: ~{est_steps} steps, ~{est_time/60:.0f} min wall time")


if __name__ == "__main__":
    main()
