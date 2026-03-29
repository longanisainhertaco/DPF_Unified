#!/usr/bin/env python3
"""Multi-device calibration runner.

Runs the 4-phase calibration pipeline for each device that needs it.
Results are saved to docs/calibration_results/ and .last-validate-result is updated.

Usage:
    python3 scripts/calibrate_all_devices.py [--device DEVICE] [--trials N]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpf.validation.mlx_calibration import run_calibration_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_all")

# Devices needing calibration, ordered by priority
DEVICES = [
    "pf1000",       # NRMSE=0.165, need < 0.15
    "mjolnir",      # NRMSE=0.162, need < 0.15
    "poseidon",     # t_peak 36.6% off
    "unu_ictp",     # NRMSE=0.092, already good but can improve
    "faeton",       # NRMSE=0.025, already excellent
    "poseidon_60kv",  # NRMSE=0.115, already good
]


def calibrate_device(preset: str, n_trials: int = 50) -> dict:
    logger.info("=" * 70)
    logger.info("CALIBRATING: %s (%d Optuna trials)", preset, n_trials)
    logger.info("=" * 70)

    t0 = time.time()
    try:
        result = run_calibration_pipeline(
            preset_name=preset,
            n_optuna_trials=n_trials,
            skip_phase3=False,
            skip_phase4=True,  # skip fine grid for speed
        )
        elapsed = time.time() - t0
        best = result.best
        logger.info(
            "DONE %s: fc=%.3f fm=%.3f I_err=%.1f%% t_err=%.1f%% J=%.4f (%d evals, %.0fs)",
            preset, best.best_fc, best.best_fm,
            best.peak_current_error * 100, best.timing_error * 100,
            best.objective_value, best.n_evals, elapsed,
        )
        return {
            "preset": preset,
            "fc": best.best_fc,
            "fm": best.best_fm,
            "I_peak_error_pct": best.peak_current_error * 100,
            "timing_error_pct": best.timing_error * 100,
            "objective": best.objective_value,
            "n_evals": best.n_evals,
            "converged": best.converged,
            "wall_time_s": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("FAILED %s: %s (%.0fs)", preset, e, elapsed)
        return {
            "preset": preset,
            "error": str(e),
            "wall_time_s": elapsed,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="Single device to calibrate")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per device")
    args = parser.parse_args()

    devices = [args.device] if args.device else DEVICES
    results = []

    for preset in devices:
        result = calibrate_device(preset, n_trials=args.trials)
        results.append(result)

        # Save intermediate results
        out_dir = Path("docs/calibration_results")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / f"{preset}_calibration.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 70)
    for r in results:
        if "error" in r:
            logger.info("  %s: FAILED — %s", r["preset"], r["error"])
        else:
            logger.info(
                "  %s: fc=%.3f fm=%.3f I_err=%.1f%% converged=%s",
                r["preset"], r["fc"], r["fm"],
                r["I_peak_error_pct"], r["converged"],
            )

    # Save full results
    with open("docs/calibration_results/all_devices.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to docs/calibration_results/")


if __name__ == "__main__":
    main()
