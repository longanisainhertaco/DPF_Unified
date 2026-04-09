#!/usr/bin/env python3
"""Run Lee model (RADPF-equivalent) for all parameter combinations and save outputs.

Sweeps over: fc, fm, V0, pressure. Saves I(t), Lp(t), scalars for each run.
Uses the same Lee model equations as RADPF v5.16 (Lee & Saw 2014).

Usage:
    python3 scripts/radpf_parameter_sweep.py [--device pf1000] [--output-dir output/sweep]
    python3 scripts/radpf_parameter_sweep.py --fc 0.6,0.7,0.8 --fm 0.08,0.13,0.19
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpf.metal.mlx_engine import run_mlx_discharge


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="RADPF parameter sweep")
    parser.add_argument("--device", default="pf1000", help="Device preset")
    parser.add_argument("--output-dir", default="output/sweep", help="Output directory")
    parser.add_argument("--fc", default="0.6,0.7,0.8", help="Comma-separated fc values")
    parser.add_argument("--fm", default="0.08,0.13,0.19", help="Comma-separated fm values")
    parser.add_argument("--V0-kV", default="27", help="Comma-separated V0 values (kV)")
    parser.add_argument("--pressure-torr", default="3.5", help="Comma-separated pressure values (Torr)")
    parser.add_argument("--mode", default="lee", choices=["lee", "mhd"], help="Simulation mode")
    parser.add_argument("--max-steps", type=int, default=20000, help="Max timesteps per run")
    parser.add_argument("--grid", default="32,1,64", help="Grid shape for MHD mode (nr,ny,nz)")
    args = parser.parse_args()

    fc_vals = parse_float_list(args.fc)
    fm_vals = parse_float_list(args.fm)
    V0_vals = parse_float_list(args.V0_kV)
    p_vals = parse_float_list(args.pressure_torr)

    grid_shape = tuple(int(x) for x in args.grid.split(","))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(fc_vals, fm_vals, V0_vals, p_vals))
    n_total = len(combos)
    print(f"=== RADPF Parameter Sweep ===")
    print(f"Device: {args.device}, Mode: {args.mode}")
    print(f"fc: {fc_vals}")
    print(f"fm: {fm_vals}")
    print(f"V0: {V0_vals} kV")
    print(f"P:  {p_vals} Torr")
    print(f"Total combinations: {n_total}")
    print()

    summary = []
    t_start = time.time()

    for i, (fc, fm, V0, p) in enumerate(combos):
        tag = f"fc{fc:.2f}_fm{fm:.2f}_V{V0:.0f}kV_p{p:.1f}T"
        print(f"[{i+1}/{n_total}] {tag} ...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = run_mlx_discharge(
                args.device,
                mode=args.mode,
                max_steps=args.max_steps,
                fc=fc,
                fm=fm,
                V0_kV=V0,
                pressure_torr=p,
                grid_shape=grid_shape,
            )
            elapsed = time.time() - t0

            I_peak = result.get("I_peak_MA", 0)
            t_peak = result.get("t_peak_us", 0)
            n_steps = result.get("n_steps", 0)

            print(f"I_peak={I_peak:.3f} MA, t_peak={t_peak:.2f} us, {elapsed:.1f}s")

            # Save time series
            run_data = {
                "parameters": {"fc": fc, "fm": fm, "V0_kV": V0, "pressure_Torr": p},
                "scalars": {
                    "I_peak_MA": float(I_peak),
                    "t_peak_us": float(t_peak),
                    "n_steps": n_steps,
                    "elapsed_s": elapsed,
                    "Lp_max_nH": float(np.max(result.get("Lp_nH", [0]))),
                },
                "time_series": {
                    "t_us": [float(x) for x in np.asarray(result.get("t_us", []))],
                    "I_MA": [float(x) for x in np.asarray(result.get("I_MA", []))],
                    "Lp_nH": [float(x) for x in np.asarray(result.get("Lp_nH", []))],
                },
            }

            with open(out_dir / f"{tag}.json", "w") as f:
                json.dump(run_data, f)

            summary.append({
                "tag": tag, "fc": fc, "fm": fm, "V0_kV": V0, "p_Torr": p,
                "I_peak_MA": float(I_peak), "t_peak_us": float(t_peak),
                "Lp_max_nH": run_data["scalars"]["Lp_max_nH"],
                "elapsed_s": elapsed, "status": "OK",
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED: {e} ({elapsed:.1f}s)")
            summary.append({
                "tag": tag, "fc": fc, "fm": fm, "V0_kV": V0, "p_Torr": p,
                "status": "FAILED", "error": str(e),
            })

    total_time = time.time() - t_start
    print(f"\n=== Summary ===")
    print(f"Completed: {sum(1 for s in summary if s['status']=='OK')}/{n_total}")
    print(f"Total time: {total_time:.1f}s ({total_time/max(n_total,1):.1f}s/run)")

    # Save summary
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {out_dir}/sweep_summary.json")

    # Print table
    print(f"\n{'fc':>5s} {'fm':>5s} {'V0':>5s} {'P':>5s} {'I_peak':>8s} {'t_peak':>7s} {'Lp_max':>7s}")
    for s in summary:
        if s["status"] == "OK":
            print(f"{s['fc']:5.2f} {s['fm']:5.2f} {s['V0_kV']:5.0f} {s['p_Torr']:5.1f} "
                  f"{s['I_peak_MA']:8.3f} {s['t_peak_us']:7.2f} {s['Lp_max_nH']:7.1f}")


if __name__ == "__main__":
    main()
