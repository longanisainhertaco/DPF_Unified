#!/usr/bin/env python3
"""Multi-device MLX calibration sweep.

Runs Optuna TPE calibration on multiple DPF devices sequentially.
Uses warm-start from published Lee params, narrowed bounds, and
parallel workers for GPU utilization.

Usage:
    python3 scripts/calibrate_multi_device.py [--devices pf1000,unu_ictp] [--trials 30]
    python3 scripts/calibrate_multi_device.py --devices faeton --trials 15 --workers 2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dpf.validation.mlx_calibration import (
    MLXTrialResult,
    parallel_optuna_optimize,
    run_mlx_forward_model,
)

logger = logging.getLogger(__name__)

# Published Lee model params as warm-start centers
# Source: experimental_devices.py lee_fc/lee_fm fields
DEVICE_SEEDS: dict[str, tuple[float, float]] = {
    "pf1000":        (0.70, 0.08),
    "unu_ictp":      (0.70, 0.08),
    "poseidon_60kv": (0.60, 0.275),
    "faeton":        (0.70, 0.70),
}

# Narrowed search bounds: +/-0.15 around Lee fc, +/-0.10 around Lee fm
# Clipped to physical limits [0.3, 0.95] for fc and [0.01, 0.95] for fm
DEVICE_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "pf1000":        {"fc": (0.50, 0.85), "fm": (0.03, 0.20)},
    "unu_ictp":      {"fc": (0.55, 0.85), "fm": (0.03, 0.20)},
    "poseidon_60kv": {"fc": (0.45, 0.75), "fm": (0.15, 0.40)},
    "faeton":        {"fc": (0.55, 0.85), "fm": (0.40, 0.90)},
}

# Device-specific pass/fail tolerances from conftest.py DEVICE_TOLERANCES
PASS_CRITERIA: dict[str, dict[str, float]] = {
    "pf1000":        {"I_peak": 0.05, "t_peak": 0.15, "nrmse": 0.20},  # t_peak 15%: Gribkov 94-pt waveform has 5-7% digitization uncertainty
    "unu_ictp":      {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.15},
    "poseidon_60kv": {"I_peak": 0.05, "t_peak": 0.05, "nrmse": 0.15},
    "faeton":        {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.10},
}


@dataclass
class DeviceCalibrationResult:
    preset: str
    device_name: str
    best_fc: float
    best_fm: float
    lee_fc: float
    lee_fm: float
    I_peak_error: float
    t_peak_error: float
    nrmse: float
    objective: float
    converged: bool
    n_evals: int
    wall_time_min: float
    passes_tolerance: bool


def _best_nrmse(trials: list[MLXTrialResult]) -> float:
    """Extract nrmse from the best successful trial.

    CalibrationResult has no nrmse field; MLXTrialResult does.
    Find the trial with minimum objective among successful runs.
    """
    successful = [t for t in trials if t.success]
    if not successful:
        return 10.0
    best = min(successful, key=lambda t: t.objective)
    return best.nrmse


def calibrate_device(
    preset: str,
    n_trials: int = 30,
    n_workers: int = 3,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
) -> DeviceCalibrationResult:
    """Run calibration pipeline for a single device.

    Args:
        preset: Device preset name (e.g., "pf1000").
        n_trials: Total Optuna evaluations (Phase 1+2).
        n_workers: Parallel worker processes for Optuna.
        grid_shape: Grid resolution (nr, ny, nz).

    Returns:
        DeviceCalibrationResult with best parameters and metrics.
    """
    seed_fc, seed_fm = DEVICE_SEEDS[preset]
    bounds = DEVICE_BOUNDS[preset]
    tol = PASS_CRITERIA[preset]

    t0 = time.monotonic()

    # Phase 0: Evaluate at published Lee params (baseline, no Optuna budget)
    baseline = run_mlx_forward_model(
        fc=seed_fc, fm=seed_fm, preset_name=preset, grid_shape=grid_shape,
    )
    logger.info(
        "%s baseline (Lee): fc=%.3f fm=%.3f I_err=%.1f%% NRMSE=%.3f",
        preset, seed_fc, seed_fm, baseline.peak_error * 100, baseline.nrmse,
    )

    # Phase 1+2: Optuna TPE with parallel workers
    cal_result, trials = parallel_optuna_optimize(
        fc_bounds=bounds["fc"],
        fm_bounds=bounds["fm"],
        n_trials=n_trials,
        n_workers=n_workers,
        preset_name=preset,
        grid_shape=grid_shape,
    )

    elapsed_min = (time.monotonic() - t0) / 60.0
    nrmse = _best_nrmse(trials)

    # Check if Optuna-best OR baseline Lee params pass tolerance.
    # Composite objective can find points that minimize NRMSE but worsen
    # individual I_peak/t_peak metrics vs the Lee baseline.
    optuna_passes = (
        cal_result.peak_current_error <= tol["I_peak"]
        and cal_result.timing_error <= tol["t_peak"]
    )
    baseline_passes = (
        baseline.peak_error <= tol["I_peak"]
        and baseline.timing_error <= tol["t_peak"]
    )
    passes = optuna_passes or baseline_passes

    # Use whichever result actually passes (prefer Optuna if both pass)
    if optuna_passes:
        best_fc = cal_result.best_fc
        best_fm = cal_result.best_fm
        best_I_err = cal_result.peak_current_error
        best_t_err = cal_result.timing_error
        best_obj = cal_result.objective_value
    elif baseline_passes:
        best_fc = seed_fc
        best_fm = seed_fm
        best_I_err = baseline.peak_error
        best_t_err = baseline.timing_error
        best_obj = baseline.objective
    else:
        best_fc = cal_result.best_fc
        best_fm = cal_result.best_fm
        best_I_err = cal_result.peak_current_error
        best_t_err = cal_result.timing_error
        best_obj = cal_result.objective_value

    return DeviceCalibrationResult(
        preset=preset,
        device_name=cal_result.device_name,
        best_fc=best_fc,
        best_fm=best_fm,
        lee_fc=seed_fc,
        lee_fm=seed_fm,
        I_peak_error=best_I_err,
        t_peak_error=best_t_err,
        nrmse=nrmse,
        objective=best_obj,
        converged=cal_result.converged,
        n_evals=cal_result.n_evals,
        wall_time_min=elapsed_min,
        passes_tolerance=passes,
    )


def run_sweep(
    devices: list[str] | None = None,
    n_trials: int = 30,
    n_workers: int = 3,
    output_path: Path | None = None,
) -> list[DeviceCalibrationResult]:
    """Run calibration on all devices sequentially, print comparison table, save JSON.

    Args:
        devices: List of preset names. Default: all four (pf1000, unu_ictp,
            poseidon_60kv, faeton).
        n_trials: Optuna trial budget per device.
        n_workers: Parallel workers per device (Optuna parallel).
        output_path: JSON output path. Default: results/multi_device_calibration.json.

    Returns:
        List of DeviceCalibrationResult, one per device.
    """
    if devices is None:
        devices = list(DEVICE_SEEDS.keys())

    results: list[DeviceCalibrationResult] = []
    t_total = time.monotonic()

    for i, preset in enumerate(devices, 1):
        logger.info("=" * 60)
        logger.info("DEVICE %d/%d: %s", i, len(devices), preset)
        logger.info("=" * 60)
        result = calibrate_device(preset, n_trials=n_trials, n_workers=n_workers)
        results.append(result)
        logger.info(
            "%s done: fc=%.3f fm=%.3f I_err=%.1f%% t_err=%.1f%% NRMSE=%.3f [%s] %.1f min",
            result.preset, result.best_fc, result.best_fm,
            result.I_peak_error * 100, result.t_peak_error * 100,
            result.nrmse,
            "PASS" if result.passes_tolerance else "FAIL",
            result.wall_time_min,
        )

    total_min = (time.monotonic() - t_total) / 60.0

    # Print comparison table
    print("\n" + "=" * 95)
    print(f"MULTI-DEVICE CALIBRATION SUMMARY ({total_min:.1f} min total, {n_trials} trials each)")
    print("=" * 95)
    print(
        f"{'Device':<16} {'fc_opt':>6} {'fm_opt':>6} {'fc_Lee':>6} {'fm_Lee':>6} "
        f"{'I_err%':>7} {'t_err%':>7} {'NRMSE':>6} {'Obj':>6} {'Pass':>5}"
    )
    print("-" * 95)
    for r in results:
        status = "PASS" if r.passes_tolerance else "FAIL"
        print(
            f"{r.preset:<16} {r.best_fc:>6.3f} {r.best_fm:>6.3f} "
            f"{r.lee_fc:>6.2f} {r.lee_fm:>6.3f} "
            f"{r.I_peak_error * 100:>6.1f}% {r.t_peak_error * 100:>6.1f}% "
            f"{r.nrmse:>6.3f} {r.objective:>6.3f} {status:>5}"
        )
    print("=" * 95)

    n_pass = sum(r.passes_tolerance for r in results)
    print(f"\n{n_pass}/{len(results)} devices pass tolerance criteria.")

    # Save JSON results
    if output_path is None:
        output_path = Path("results/multi_device_calibration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Multi-device MLX calibration sweep.")
    parser.add_argument(
        "--devices", type=str, default=None,
        help="Comma-separated preset names (default: all four)",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Optuna trial budget per device (default: 30)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel Optuna workers per device (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output path (default: results/multi_device_calibration.json)",
    )
    args = parser.parse_args()
    devices = args.devices.split(",") if args.devices else None
    output = Path(args.output) if args.output else None
    run_sweep(devices=devices, n_trials=args.trials, n_workers=args.workers, output_path=output)
