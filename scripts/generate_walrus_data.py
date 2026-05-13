#!/usr/bin/env python3
"""Generate exploratory WALRUS candidate trajectories from Lee model simulations.

Produces JSON trajectory summaries that may be converted later by a reviewed
export pipeline. These outputs are not Well HDF5 files and are not validation
evidence.

Usage:
    python3 scripts/generate_walrus_data.py [--n-trajectories 100] [--output-dir training/walrus]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("walrus_datagen")


def generate_trajectory(
    preset_name: str,
    V0_kV: float,
    pressure_torr: float,
    fc: float,
    fm: float,
    sim_time_us: float = 15.0,
) -> dict | None:
    """Run a Lee model simulation and return trajectory data."""
    from app_engine import run_simulation_core

    try:
        data = run_simulation_core(
            preset_name=preset_name,
            sim_time_us=sim_time_us,
            V0_kV=V0_kV,
            pressure_torr=pressure_torr,
            fc=fc,
            fm=fm,
        )
        if data.get("nan_detected"):
            return None
        return data
    except Exception as e:
        logger.warning("Failed %s V0=%.0fkV P=%.1fTorr: %s", preset_name, V0_kV, pressure_torr, e)
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="training/walrus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Parameter ranges for diversity
    presets = ["pf1000", "unu_ictp", "faeton", "poseidon_60kv"]
    V0_ranges = {
        "pf1000": (20, 35),
        "unu_ictp": (12, 16),
        "faeton": (80, 110),
        "poseidon_60kv": (50, 70),
    }
    pressure_ranges = {
        "pf1000": (2, 8),
        "unu_ictp": (2, 6),
        "faeton": (3, 12),
        "poseidon_60kv": (2, 6),
    }

    results = []
    n_success = 0
    n_target = args.n_trajectories

    for i in range(n_target * 2):  # oversample for failures
        if n_success >= n_target:
            break

        preset = rng.choice(presets)
        V0_lo, V0_hi = V0_ranges[preset]
        P_lo, P_hi = pressure_ranges[preset]

        V0 = rng.uniform(V0_lo, V0_hi)
        P = rng.uniform(P_lo, P_hi)
        fc = rng.uniform(0.55, 0.85)
        fm = rng.uniform(0.05, 0.30)

        data = generate_trajectory(preset, V0, P, fc, fm)
        if data is None:
            continue

        # Save as JSON (lightweight, convertible to Well HDF5 later)
        traj = {
            "preset": preset,
            "V0_kV": float(V0),
            "pressure_torr": float(P),
            "fc": float(fc),
            "fm": float(fm),
            "I_peak_MA": float(data["I_peak"]),
            "t_peak_us": float(data["t_peak"]),
            "n_steps": data["n_steps"],
            "t_us": [float(x) for x in data["t_us"]],
            "I_MA": [float(x) for x in data["I_MA"]],
            "V_kV": [float(x) for x in data["V_kV"]],
        }
        fname = output_dir / f"traj_{n_success:04d}.json"
        with open(fname, "w") as f:
            json.dump(traj, f)

        n_success += 1
        if n_success % 10 == 0:
            logger.info("Generated %d/%d trajectories", n_success, n_target)

    # Summary
    logger.info("Done: %d/%d trajectories in %s", n_success, n_target, output_dir)

    # Write manifest
    manifest = {
        "n_trajectories": n_success,
        "generated_file_format": "json",
        "artifact_classification": "exploratory_training_candidate",
        "validation_status": "not_validation_evidence",
        "source_status": "not_source_backed",
        "presets": presets,
        "parameter_ranges": {
            "V0_kV": V0_ranges,
            "pressure_torr": pressure_ranges,
            "fc": [0.55, 0.85],
            "fm": [0.05, 0.30],
        },
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
