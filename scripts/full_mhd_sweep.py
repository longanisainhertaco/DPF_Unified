#!/usr/bin/env python3
"""Full MHD sweep: 10 shots per device through MLX solver with all physics.

32x64 cylindrical grid, HLLS + PLM + SSP-RK2, full physics stack.
Logs timing and key metrics for each shot.
"""
from __future__ import annotations

import json
import csv
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))


DEVICES = {
    "pf1000":       {"V0": (25, 30),  "P": (3, 6),   "fc": (0.65, 0.85), "fm": (0.06, 0.15), "st": 10e-6},
    "pf1000_akel":  {"V0": (15, 17),  "P": (3, 7),   "fc": (0.65, 0.85), "fm": (0.06, 0.15), "st": 10e-6},
    "pf1000_20kv":  {"V0": (19, 22),  "P": (3, 6),   "fc": (0.65, 0.85), "fm": (0.06, 0.15), "st": 10e-6},
    "nx2":          {"V0": (11, 14),  "P": (2, 5),   "fc": (0.60, 0.80), "fm": (0.05, 0.15), "st": 5e-6},
    "unu_ictp":     {"V0": (13, 15),  "P": (2, 5),   "fc": (0.55, 0.75), "fm": (0.05, 0.20), "st": 5e-6},
    "llnl_dpf":     {"V0": (20, 24),  "P": (2, 5),   "fc": (0.60, 0.80), "fm": (0.05, 0.20), "st": 2e-6},
    "mjolnir":      {"V0": (55, 65),  "P": (5, 9),   "fc": (0.60, 0.80), "fm": (0.70, 1.10), "st": 10e-6},
    "faeton":       {"V0": (90, 105), "P": (4, 10),  "fc": (0.45, 0.70), "fm": (0.20, 0.60), "st": 10e-6},
    "poseidon":     {"V0": (38, 42),  "P": (3, 5),   "fc": (0.55, 0.75), "fm": (0.15, 0.35), "st": 10e-6},
    "poseidon_60kv":{"V0": (55, 65),  "P": (3, 5),   "fc": (0.35, 0.55), "fm": (0.30, 0.50), "st": 6e-6},
    "aecs_pf2":     {"V0": (14, 17),  "P": (2, 5),   "fc": (0.60, 0.80), "fm": (0.05, 0.20), "st": 8e-6},
    "pf400j":       {"V0": (22, 28),  "P": (6, 12),  "fc": (0.60, 0.80), "fm": (0.05, 0.20), "st": 4e-6},
}

N_SHOTS = 10


def run_mhd_shot(preset_name: str, V0_kV: float, P_torr: float,
                 fc: float, fm: float, sim_time: float) -> dict:
    """Run one full MHD shot through pure-MLX engine."""
    from dpf.metal.mlx_engine import run_mlx_discharge

    t0 = time.perf_counter()
    result = run_mlx_discharge(
        preset_name=preset_name,
        mode="mhd",
        max_steps=50000,
        fc=fc, fm=fm,
        V0_kV=V0_kV,
        pressure_torr=P_torr,
        grid_shape=(32, 1, 64),
    )
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": round(elapsed, 2),
        "n_steps": result["n_steps"],
        "I_peak_MA": round(result["I_peak_MA"], 4),
        "t_peak_us": round(result["t_peak_us"], 2),
        "wall_time_s": round(elapsed, 2),
        "nan": False,
    }


def run_mhd_shot_safe(preset_name: str, V0_kV: float, P_torr: float,
                      fc: float, fm: float, sim_time: float) -> dict:
    """Wrapper with error handling."""
    try:
        return run_mhd_shot(preset_name, V0_kV, P_torr, fc, fm, sim_time)
    except Exception as e:
        return {"elapsed_s": 0, "n_steps": 0, "error": str(e)[:120], "nan": True}


def main():
    out_dir = Path("training/mhd_sweep_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    all_summaries = []
    total_t0 = time.time()

    for device, params in DEVICES.items():
        print(f"\n{'='*70}")
        print(f"MHD SWEEP: {device} ({N_SHOTS} shots, 32x64 grid, full physics)")
        print(f"{'='*70}")

        shots = []
        for i in range(N_SHOTS):
            V0 = float(rng.uniform(*params["V0"]))
            P = float(rng.uniform(*params["P"]))
            fc = float(rng.uniform(*params["fc"]))
            fm = float(rng.uniform(*params["fm"]))
            st = params["st"]

            t0 = time.time()
            result = run_mhd_shot_safe(device, V0, P, fc, fm, st)
            wall = time.time() - t0

            shot = {
                "shot_id": i, "device": device,
                "V0_kV": round(V0, 2), "pressure_torr": round(P, 2),
                "fc": round(fc, 3), "fm": round(fm, 3),
                **result,
                "wall_s": round(wall, 2),
            }
            shots.append(shot)

            status = "OK" if not result.get("nan") and "error" not in result else "FAIL"
            print(f"  [{i+1:2d}/{N_SHOTS}] {status} {result['elapsed_s']:6.1f}s "
                  f"V0={V0:.0f}kV P={P:.1f}T fc={fc:.2f} fm={fm:.2f} "
                  f"I={result.get('I_peak_MA', 0):.3f}MA")

        # Save raw
        with open(out_dir / f"{device}_{N_SHOTS}shots_mhd.json", "w") as f:
            json.dump(shots, f, indent=2)

        ok_shots = [s for s in shots if not s.get("nan") and "error" not in s]
        times = [s["wall_s"] for s in ok_shots]
        summary = {
            "device": device,
            "n_shots": N_SHOTS, "n_ok": len(ok_shots), "n_fail": N_SHOTS - len(ok_shots),
            "avg_s": round(np.mean(times), 1) if times else 0,
            "std_s": round(np.std(times), 1) if times else 0,
            "min_s": round(min(times), 1) if times else 0,
            "max_s": round(max(times), 1) if times else 0,
        }
        all_summaries.append(summary)
        print(f"  => {len(ok_shots)}/{N_SHOTS} ok, avg={summary['avg_s']:.1f}s/shot")

    total_elapsed = time.time() - total_t0

    with open(out_dir / "mhd_sweep_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "n_devices": len(DEVICES), "n_shots_per_device": N_SHOTS,
            "total_shots": len(DEVICES) * N_SHOTS,
            "grid": "32x1x64", "solver": "HLLS+PLM+SSP-RK2",
            "total_wall_time_s": round(total_elapsed, 1),
            "summaries": all_summaries,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(DEVICES)} devices x {N_SHOTS} shots = {len(DEVICES)*N_SHOTS} MHD runs")
    print(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Results in {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
