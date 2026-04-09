#!/usr/bin/env python3
"""Extract scalar diagnostics from MHD discharge output for PIRT analysis.

Usage: python3 tests/extract_scalars.py output.npz [--json scalars.json]

Extracts: I_peak, t_peak, radial_phase_duration, waveform_L2, dI_dt_rise,
           L_p(t), r_eff(t), z_sheath(t), dt_min, energy_conservation_error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def extract(data_path: str) -> dict:
    """Extract scalar diagnostics from simulation output."""
    d = np.load(data_path, allow_pickle=True)

    scalars: dict = {}

    # Current waveform
    t = d.get("time", d.get("t", np.array([])))
    I = d.get("current", d.get("I", np.array([])))

    if len(I) > 0 and len(t) > 0:
        idx_peak = int(np.argmax(np.abs(I)))
        scalars["I_peak_A"] = float(np.abs(I[idx_peak]))
        scalars["t_peak_s"] = float(t[idx_peak])

        # dI/dt at current rise (first 50% of rise)
        idx_half = np.searchsorted(np.abs(I[:idx_peak]), 0.5 * scalars["I_peak_A"])
        if idx_half > 0 and idx_half < len(t) - 1:
            dt_rise = t[idx_half] - t[max(0, idx_half - 1)]
            dI_rise = abs(I[idx_half]) - abs(I[max(0, idx_half - 1)])
            scalars["dI_dt_rise_As"] = float(dI_rise / max(dt_rise, 1e-30))

        # Waveform L2 norm (for comparison with RADPF)
        scalars["waveform_L2"] = float(np.sqrt(np.mean(I**2)))

    # Plasma inductance
    Lp = d.get("Lp", d.get("L_p", np.array([])))
    if len(Lp) > 0:
        scalars["Lp_max_H"] = float(np.max(Lp))

    # Timestep
    dt_arr = d.get("dt", np.array([]))
    if len(dt_arr) > 0:
        scalars["dt_min_s"] = float(np.min(dt_arr[dt_arr > 0])) if np.any(dt_arr > 0) else 0.0
        scalars["dt_max_s"] = float(np.max(dt_arr))
        scalars["n_steps"] = int(len(dt_arr))

    # Energy conservation
    E_total = d.get("E_total", np.array([]))
    if len(E_total) > 1:
        scalars["energy_conservation_error"] = float(abs(E_total[-1] - E_total[0]) / max(abs(E_total[0]), 1e-30))

    return scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MHD discharge scalars")
    parser.add_argument("data_path", help="Path to .npz output file")
    parser.add_argument("--json", default="scalars.json", help="Output JSON path")
    args = parser.parse_args()

    scalars = extract(args.data_path)

    with open(args.json, "w") as f:
        json.dump(scalars, f, indent=2)

    print(f"Extracted {len(scalars)} scalars to {args.json}")
    for k, v in sorted(scalars.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
