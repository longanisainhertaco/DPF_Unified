#!/usr/bin/env python3
"""Full parameter sweep: 1000 shots per device on Apple Silicon.

Randomizes V0, pressure, fc, fm within physical ranges for each device.
Logs every shot's timing and key metrics for data analytics.

Output: training/sweep_results/{device}_1000shots.json + summary CSV.
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

from app_engine import run_simulation_core
from dpf.presets import get_preset


# Real DPF machines with their parameter ranges
DEVICES = {
    "pf1000":       {"V0": (20, 35),  "P": (2, 8),   "fc": (0.55, 0.85), "fm": (0.05, 0.30), "st": 10},
    "pf1000_akel":  {"V0": (14, 18),  "P": (3, 8),   "fc": (0.55, 0.85), "fm": (0.05, 0.25), "st": 10},
    "pf1000_20kv":  {"V0": (18, 24),  "P": (3, 7),   "fc": (0.55, 0.85), "fm": (0.05, 0.25), "st": 12},
    "nx2":          {"V0": (10, 15),  "P": (2, 6),   "fc": (0.55, 0.80), "fm": (0.05, 0.20), "st": 5},
    "unu_ictp":     {"V0": (12, 16),  "P": (2, 6),   "fc": (0.50, 0.80), "fm": (0.05, 0.30), "st": 5},
    "llnl_dpf":     {"V0": (18, 25),  "P": (2, 6),   "fc": (0.55, 0.80), "fm": (0.05, 0.25), "st": 2},
    "mjolnir":      {"V0": (50, 70),  "P": (4, 10),  "fc": (0.55, 0.85), "fm": (0.50, 1.20), "st": 14},
    "faeton":       {"V0": (80, 110), "P": (3, 12),  "fc": (0.40, 0.80), "fm": (0.10, 0.70), "st": 12},
    "poseidon":     {"V0": (35, 45),  "P": (2, 6),   "fc": (0.50, 0.80), "fm": (0.10, 0.40), "st": 12},
    "poseidon_60kv":{"V0": (50, 70),  "P": (2, 6),   "fc": (0.30, 0.60), "fm": (0.20, 0.60), "st": 6},
    "aecs_pf2":     {"V0": (12, 18),  "P": (2, 6),   "fc": (0.55, 0.80), "fm": (0.05, 0.25), "st": 10},
    "pf400j":       {"V0": (20, 30),  "P": (5, 15),  "fc": (0.55, 0.85), "fm": (0.05, 0.30), "st": 5},
}

N_SHOTS = 1000


def run_sweep(device: str, params: dict, rng: np.random.Generator) -> list[dict]:
    """Run N_SHOTS randomized simulations for one device."""
    V0_lo, V0_hi = params["V0"]
    P_lo, P_hi = params["P"]
    fc_lo, fc_hi = params["fc"]
    fm_lo, fm_hi = params["fm"]
    st = params["st"]

    shots = []
    t_start = time.time()

    for i in range(N_SHOTS):
        V0 = float(rng.uniform(V0_lo, V0_hi))
        P = float(rng.uniform(P_lo, P_hi))
        fc = float(rng.uniform(fc_lo, fc_hi))
        fm = float(rng.uniform(fm_lo, fm_hi))

        t0 = time.perf_counter()
        try:
            data = run_simulation_core(
                preset_name=device, sim_time_us=st,
                V0_kV=V0, pressure_torr=P, fc=fc, fm=fm,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            shot = {
                "shot_id": i,
                "V0_kV": V0, "pressure_torr": P, "fc": fc, "fm": fm,
                "I_peak_MA": float(data["I_peak"]),
                "t_peak_us": float(data["t_peak"]),
                "dip_pct": float(data.get("dip_pct", 0)),
                "n_steps": data["n_steps"],
                "elapsed_ms": round(elapsed_ms, 2),
                "status": "ok",
            }

            ny = data.get("neutron_yield")
            if ny:
                shot["Yn"] = float(ny.get("Y_neutron", 0))
                shot["T_keV"] = float(ny.get("T_eff_keV", 0))

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            shot = {
                "shot_id": i,
                "V0_kV": V0, "pressure_torr": P, "fc": fc, "fm": fm,
                "elapsed_ms": round(elapsed_ms, 2),
                "status": f"error: {str(e)[:80]}",
            }

        shots.append(shot)

        if (i + 1) % 100 == 0:
            avg_ms = sum(s["elapsed_ms"] for s in shots) / len(shots)
            ok_count = sum(1 for s in shots if s["status"] == "ok")
            elapsed_total = time.time() - t_start
            print(f"  {device}: {i+1}/{N_SHOTS}  avg={avg_ms:.1f}ms/shot  "
                  f"ok={ok_count}/{i+1}  elapsed={elapsed_total:.0f}s")

    return shots


def main():
    out_dir = Path("training/sweep_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    all_summaries = []

    total_t0 = time.time()

    for device, params in DEVICES.items():
        print(f"\n{'='*60}")
        print(f"SWEEPING: {device} ({N_SHOTS} shots)")
        print(f"{'='*60}")

        shots = run_sweep(device, params, rng)

        # Save raw shots
        with open(out_dir / f"{device}_1000shots.json", "w") as f:
            json.dump(shots, f)

        # Summary stats
        ok_shots = [s for s in shots if s["status"] == "ok"]
        times_ms = [s["elapsed_ms"] for s in ok_shots]
        I_peaks = [s["I_peak_MA"] for s in ok_shots]

        summary = {
            "device": device,
            "n_shots": N_SHOTS,
            "n_ok": len(ok_shots),
            "n_fail": N_SHOTS - len(ok_shots),
            "avg_ms": round(np.mean(times_ms), 2) if times_ms else 0,
            "std_ms": round(np.std(times_ms), 2) if times_ms else 0,
            "min_ms": round(np.min(times_ms), 2) if times_ms else 0,
            "max_ms": round(np.max(times_ms), 2) if times_ms else 0,
            "p50_ms": round(np.percentile(times_ms, 50), 2) if times_ms else 0,
            "p95_ms": round(np.percentile(times_ms, 95), 2) if times_ms else 0,
            "p99_ms": round(np.percentile(times_ms, 99), 2) if times_ms else 0,
            "I_peak_mean_MA": round(np.mean(I_peaks), 4) if I_peaks else 0,
            "I_peak_std_MA": round(np.std(I_peaks), 4) if I_peaks else 0,
        }
        all_summaries.append(summary)

        print(f"  DONE: {len(ok_shots)}/{N_SHOTS} ok, "
              f"avg={summary['avg_ms']:.1f}ms, p95={summary['p95_ms']:.1f}ms, "
              f"I_peak={summary['I_peak_mean_MA']:.3f}+-{summary['I_peak_std_MA']:.3f} MA")

    # Write summary CSV
    with open(out_dir / "sweep_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(DEVICES)} devices x {N_SHOTS} shots = {len(DEVICES)*N_SHOTS} total")
    print(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Results in {out_dir}/")
    print(f"{'='*60}")

    # Save combined manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "n_devices": len(DEVICES),
            "n_shots_per_device": N_SHOTS,
            "total_shots": len(DEVICES) * N_SHOTS,
            "total_wall_time_s": round(total_elapsed, 1),
            "summaries": all_summaries,
        }, f, indent=2)


if __name__ == "__main__":
    main()
