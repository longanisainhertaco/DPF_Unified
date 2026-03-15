#!/usr/bin/env python3
"""Validate neutron yield scaling Y_n ~ I^4 across DPF device presets.

Runs the Lee model for all available presets with published data, computes
I_peak and Y_n, and checks against the I^4 scaling law. Reports which
devices follow the scaling and which deviate (Challenge 13: MJ scaling).

Usage:
    python3 scripts/validate_scaling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app_engine import run_simulation_core
from app_validation import PRESET_TO_DEVICE, validate_against_published


def main() -> None:
    results = []
    print(f"{'Device':<16} {'I_peak [MA]':>12} {'Yn':>12} {'I^4 pred':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 76)

    for preset_name in PRESET_TO_DEVICE:
        try:
            data = run_simulation_core(preset_name=preset_name, sim_time_us=40.0)
        except Exception as exc:
            print(f"{preset_name:<16} {'ERROR':>12} {str(exc)[:40]}")
            continue

        I_peak = data.get("I_pre_dip", data["I_peak"])
        yn_data = data.get("neutron_yield")
        Yn = yn_data["Y_neutron"] if yn_data else 0.0

        results.append({"preset": preset_name, "I_peak": I_peak, "Yn": Yn})

    if not results:
        print("No results.")
        return

    # Fit Y_n = C * I^alpha using log-log regression
    valid = [(r["I_peak"], r["Yn"]) for r in results if r["Yn"] > 0 and r["I_peak"] > 0]
    if len(valid) >= 2:
        log_I = np.log10([v[0] for v in valid])
        log_Y = np.log10([v[1] for v in valid])
        alpha, log_C = np.polyfit(log_I, log_Y, 1)
        C = 10**log_C
        print(f"\nScaling fit: Y_n = {C:.2e} * I_peak^{alpha:.2f}")
        print(f"Expected: alpha ~ 4.0 (Lee & Saw 2014)")
        print(f"Deviation from I^4: {abs(alpha - 4.0):.2f}")
    else:
        alpha, C = 4.0, 1e9
        print("\nInsufficient data for scaling fit")

    # Print results with I^4 prediction
    print(f"\n{'Device':<16} {'I_peak [MA]':>12} {'Yn (sim)':>12} {'Yn (I^4)':>12} {'Ratio':>8}")
    print("-" * 68)
    for r in results:
        I_pred_Yn = C * r["I_peak"] ** alpha if r["I_peak"] > 0 else 0
        ratio = r["Yn"] / I_pred_Yn if I_pred_Yn > 0 else 0
        status = "OK" if 0.3 < ratio < 3.0 else "OUTLIER"
        print(
            f"{r['preset']:<16} {r['I_peak']:>12.3f} {r['Yn']:>12.2e} "
            f"{I_pred_Yn:>12.2e} {ratio:>8.2f}  {status}"
        )


if __name__ == "__main__":
    main()
