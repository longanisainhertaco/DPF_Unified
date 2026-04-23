"""MLX MHD solver calibration of fc/fm against experimental waveforms.

Runs the full SimulationEngine with backend='mlx' for each (fc, fm) trial,
extracts the I(t) waveform, and computes a weighted objective:
    J = w_peak * |I_peak_err| + w_timing * |t_peak_err| + w_nrmse * NRMSE

Supports:
- Coarse grid scan (Phase 1) for landscape mapping
- Optuna TPE optimization (Phase 2) for efficient search
- Multi-fidelity verification (Phase 3-4) at higher resolution

Reference: FC_FM_CALIBRATION_DMAIC.md
"""

from __future__ import annotations

import logging
import time as wall_time
from dataclasses import dataclass, field

import numpy as np

from dpf.validation._calibration_data import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class MLXTrialResult:
    """Result of a single MLX forward-model evaluation."""

    fc: float
    fm: float
    I_peak_A: float
    t_peak_s: float
    nrmse: float
    peak_error: float
    timing_error: float
    objective: float
    wall_time_s: float
    steps: int
    success: bool
    grid_shape: tuple[int, int, int] = (32, 1, 64)


@dataclass
class MLXCalibrationResult:
    """Full calibration pipeline result."""

    best: CalibrationResult
    trials: list[MLXTrialResult] = field(default_factory=list)
    phases_completed: int = 0
    total_wall_time_s: float = 0.0


def run_mlx_forward_model(
    fc: float,
    fm: float,
    preset_name: str = "pf1000",
    grid_shape: tuple[int, int, int] | None = None,
    sim_time: float | None = None,
    peak_weight: float = 0.4,
    timing_weight: float = 0.3,
    waveform_weight: float = 0.3,
    handoff_mode: str = "lee_only",
) -> MLXTrialResult:
    """Run MLX solver with given (fc, fm) and return waveform metrics.

    Args:
        fc: Current fraction (0.5-0.85).
        fm: Mass fraction (0.03-0.30).
        preset_name: Device preset name.
        grid_shape: Override grid (nr, ny, nz). Default: preset value.
        sim_time: Override simulation time [s]. Default: preset value.
        peak_weight: Weight for peak current error.
        timing_weight: Weight for timing error.
        waveform_weight: Weight for waveform NRMSE.
        handoff_mode: Snowplow Lp handoff mode ("lee_only" or "full_mhd").

    Returns:
        MLXTrialResult with metrics and diagnostics.
    """
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine
    from dpf.presets import get_preset
    from dpf.validation.experimental import DEVICES
    from dpf.validation.experimental_comparison import nrmse_peak

    t_start = wall_time.monotonic()

    # Load preset and override fc/fm
    preset = get_preset(preset_name)
    preset["snowplow"]["current_fraction"] = fc
    preset["snowplow"]["mass_fraction"] = fm
    preset["snowplow"]["handoff_mode"] = handoff_mode

    # Force MLX backend with stable settings
    preset["fluid"] = preset.get("fluid", {})
    preset["fluid"]["backend"] = "mlx"
    preset["fluid"]["riemann_solver"] = "hlls"  # GPU-native, zero CPU round-trips
    preset["fluid"]["reconstruction"] = "plm"
    preset["fluid"]["time_integrator"] = "ssp_rk2"

    if grid_shape is not None:
        preset["grid_shape"] = list(grid_shape)
        # Scale dx to maintain physical domain size
        orig_shape = get_preset(preset_name)["grid_shape"]
        orig_dx = get_preset(preset_name)["dx"]
        preset["dx"] = orig_dx * orig_shape[0] / grid_shape[0]

    if sim_time is not None:
        preset["sim_time"] = sim_time

    # Disable HDF5 diagnostics for speed
    if "diagnostics" not in preset:
        preset["diagnostics"] = {}
    preset["diagnostics"]["hdf5_enabled"] = False

    # Build config and engine
    config = SimulationConfig(**preset)
    engine = SimulationEngine(config)

    # Run and collect I(t)
    times: list[float] = []
    currents: list[float] = []
    step_count = 0

    try:
        while engine.time < config.sim_time:
            result = engine.step()
            times.append(engine.time)
            currents.append(abs(engine.circuit.current))
            step_count += 1
            if result.finished:
                break
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        logger.warning("MLX trial fc=%.3f fm=%.3f failed: %s", fc, fm, exc)
        return MLXTrialResult(
            fc=fc, fm=fm, I_peak_A=0.0, t_peak_s=0.0,
            nrmse=10.0, peak_error=1.0, timing_error=1.0,
            objective=10.0, wall_time_s=wall_time.monotonic() - t_start,
            steps=step_count, success=False,
            grid_shape=tuple(config.grid_shape),
        )

    t_sim = np.array(times)
    I_sim = np.array(currents)

    if len(I_sim) < 10:
        return MLXTrialResult(
            fc=fc, fm=fm, I_peak_A=0.0, t_peak_s=0.0,
            nrmse=10.0, peak_error=1.0, timing_error=1.0,
            objective=10.0, wall_time_s=wall_time.monotonic() - t_start,
            steps=step_count, success=False,
            grid_shape=tuple(config.grid_shape),
        )

    # Extract metrics
    I_peak_sim = float(np.max(I_sim))
    t_peak_sim = float(t_sim[np.argmax(I_sim)])

    # Get experimental device
    _PRESET_TO_DEVICE = {
        "pf1000": "PF-1000", "pf1000_akel": "PF-1000-16kV",
        "pf1000_20kv": "PF-1000-20kV", "nx2": "NX2",
        "unu_ictp": "UNU-ICTP", "poseidon": "POSEIDON",
        "poseidon_60kv": "POSEIDON-60kV", "mjolnir": "MJOLNIR",
        "faeton": "FAETON-I",
    }
    device_name = _PRESET_TO_DEVICE.get(preset_name, preset_name)
    dev = DEVICES.get(device_name)
    if dev is None:
        raise ValueError(f"No experimental data for {device_name}")

    I_peak_exp = dev.peak_current
    t_peak_exp = dev.current_rise_time

    peak_error = abs(I_peak_sim - I_peak_exp) / max(I_peak_exp, 1e-10)
    timing_error = abs(t_peak_sim - t_peak_exp) / max(t_peak_exp, 1e-10)

    # Waveform NRMSE
    nrmse = 10.0
    if dev.waveform_t is not None and dev.waveform_I is not None:
        try:
            nrmse = float(nrmse_peak(t_sim, I_sim, dev.waveform_t, dev.waveform_I))
        except Exception:
            nrmse = 10.0

    # Composite objective
    objective = (
        peak_weight * peak_error
        + timing_weight * timing_error
        + waveform_weight * min(nrmse, 2.0)
    )

    elapsed = wall_time.monotonic() - t_start

    return MLXTrialResult(
        fc=fc, fm=fm,
        I_peak_A=I_peak_sim, t_peak_s=t_peak_sim,
        nrmse=nrmse, peak_error=peak_error, timing_error=timing_error,
        objective=objective, wall_time_s=elapsed,
        steps=step_count, success=True,
        grid_shape=tuple(config.grid_shape),
    )


def coarse_grid_scan(
    fc_values: list[float] | None = None,
    fm_values: list[float] | None = None,
    preset_name: str = "pf1000",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
) -> list[MLXTrialResult]:
    """Phase 1: Coarse grid scan to map the objective landscape.

    Args:
        fc_values: fc grid points. Default: 5 points in [0.60, 0.80].
        fm_values: fm grid points. Default: 5 points in [0.03, 0.30].
        preset_name: Device preset.
        grid_shape: Coarse grid for fast evaluation.

    Returns:
        List of MLXTrialResult for all (fc, fm) combinations.

    Notes:
        fc lower bound aligned to 0.60 to match the paper-consistent
        range used by ``_calibration_advanced.py`` (``fc_bounds=(0.6, 0.8)``).
        Paper-attested fc values: fc=0.70 for PF1000 (Malek et al. 2025,
        PPT 12(1):1, Section 2) and KSU PF (Lee 2014, JFE 33:319, Fig. 7).
        The previous lower bound 0.50 was unattested in the Lee literature.
    """
    if fc_values is None:
        fc_values = [0.60, 0.65, 0.70, 0.75, 0.80]
    if fm_values is None:
        fm_values = [0.03, 0.10, 0.17, 0.24, 0.30]

    results: list[MLXTrialResult] = []
    total = len(fc_values) * len(fm_values)

    for i, fc in enumerate(fc_values):
        for j, fm in enumerate(fm_values):
            idx = i * len(fm_values) + j + 1
            logger.info(
                "Phase 1 [%d/%d]: fc=%.3f fm=%.3f grid=%s",
                idx, total, fc, fm, grid_shape,
            )
            trial = run_mlx_forward_model(
                fc=fc, fm=fm,
                preset_name=preset_name,
                grid_shape=grid_shape,
            )
            results.append(trial)
            logger.info(
                "  -> I_peak=%.3f MA, t_peak=%.2f us, NRMSE=%.3f, J=%.4f (%.1fs)",
                trial.I_peak_A / 1e6, trial.t_peak_s * 1e6,
                trial.nrmse, trial.objective, trial.wall_time_s,
            )

    return results


def optuna_optimize(
    fc_bounds: tuple[float, float] = (0.50, 0.85),
    fm_bounds: tuple[float, float] = (0.03, 0.30),
    n_trials: int = 40,
    preset_name: str = "pf1000",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    seed: int = 42,
) -> tuple[CalibrationResult, list[MLXTrialResult]]:
    """Phase 2: Optuna TPE optimization within narrowed bounds.

    Args:
        fc_bounds: Search bounds for fc.
        fm_bounds: Search bounds for fm.
        n_trials: Number of Optuna trials.
        preset_name: Device preset.
        grid_shape: Grid resolution for trials.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (CalibrationResult, list of all trial results).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials: list[MLXTrialResult] = []

    def objective(trial: optuna.Trial) -> float:
        fc = trial.suggest_float("fc", fc_bounds[0], fc_bounds[1])
        fm = trial.suggest_float("fm", fm_bounds[0], fm_bounds[1])

        result = run_mlx_forward_model(
            fc=fc, fm=fm,
            preset_name=preset_name,
            grid_shape=grid_shape,
        )
        trials.append(result)

        logger.info(
            "Optuna [%d/%d]: fc=%.3f fm=%.3f -> J=%.4f (I_peak=%.3f MA, NRMSE=%.3f)",
            len(trials), n_trials, fc, fm,
            result.objective, result.I_peak_A / 1e6, result.nrmse,
        )

        if not result.success:
            return 10.0
        return result.objective

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_fc = best.params["fc"]
    best_fm = best.params["fm"]

    # Find the matching trial result
    best_trial = min(
        [t for t in trials if t.success],
        key=lambda t: t.objective,
        default=None,
    )

    cal_result = CalibrationResult(
        best_fc=best_fc,
        best_fm=best_fm,
        peak_current_error=best_trial.peak_error if best_trial else 1.0,
        timing_error=best_trial.timing_error if best_trial else 1.0,
        objective_value=best.value,
        n_evals=len(trials),
        converged=best.value < 0.5,
        device_name=preset_name,
    )

    return cal_result, trials


def _worker_eval(args: tuple) -> MLXTrialResult:
    """Worker function for parallel Optuna — runs in a separate process."""
    fc, fm, preset_name, grid_shape, handoff_mode = args
    return run_mlx_forward_model(
        fc=fc, fm=fm,
        preset_name=preset_name,
        grid_shape=grid_shape,
        handoff_mode=handoff_mode,
    )


def parallel_optuna_optimize(
    fc_bounds: tuple[float, float] = (0.50, 0.85),
    fm_bounds: tuple[float, float] = (0.03, 0.30),
    n_trials: int = 40,
    n_workers: int = 3,
    preset_name: str = "pf1000",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    seed: int = 42,
    handoff_mode: str = "lee_only",
) -> tuple[CalibrationResult, list[MLXTrialResult]]:
    """Phase 2 parallel: Optuna TPE with ask/tell and multiprocessing.

    Each worker runs in a separate process with its own MLX context
    (MLX is single-dispatch per process). Uses constant_liar sampler
    to encourage exploration across parallel trials.

    Args:
        fc_bounds: Search bounds for fc.
        fm_bounds: Search bounds for fm.
        n_trials: Total number of evaluations.
        n_workers: Number of parallel worker processes.
        preset_name: Device preset.
        grid_shape: Grid resolution for trials.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (CalibrationResult, list of all trial results).
    """
    from concurrent.futures import ProcessPoolExecutor

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        constant_liar=True,  # prevents duplicate suggestions for parallel trials
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trials: list[MLXTrialResult] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        while completed < n_trials:
            # Ask for a batch of trials
            batch_size = min(n_workers, n_trials - completed)
            optuna_trials = [study.ask() for _ in range(batch_size)]

            # Build worker args
            worker_args = []
            for ot in optuna_trials:
                fc = ot.suggest_float("fc", fc_bounds[0], fc_bounds[1])
                fm = ot.suggest_float("fm", fm_bounds[0], fm_bounds[1])
                worker_args.append((fc, fm, preset_name, grid_shape, handoff_mode))

            # Run batch in parallel
            futures = list(pool.map(_worker_eval, worker_args))

            # Report results back to Optuna
            for ot, result in zip(optuna_trials, futures, strict=True):
                trials.append(result)
                value = result.objective if result.success else 10.0
                study.tell(ot, value)
                completed += 1

                logger.info(
                    "Parallel Optuna [%d/%d]: fc=%.3f fm=%.3f -> J=%.4f "
                    "(I_peak=%.3f MA, %.1fs)",
                    completed, n_trials,
                    result.fc, result.fm, result.objective,
                    result.I_peak_A / 1e6, result.wall_time_s,
                )

    best = study.best_trial
    best_fc = best.params["fc"]
    best_fm = best.params["fm"]

    best_trial = min(
        [t for t in trials if t.success],
        key=lambda t: t.objective,
        default=None,
    )

    cal_result = CalibrationResult(
        best_fc=best_fc,
        best_fm=best_fm,
        peak_current_error=best_trial.peak_error if best_trial else 1.0,
        timing_error=best_trial.timing_error if best_trial else 1.0,
        objective_value=best.value,
        n_evals=len(trials),
        converged=best.value < 0.5,
        device_name=preset_name,
    )

    return cal_result, trials


def fd_gradient_calibrate(
    preset_name: str = "pf1000",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    x0: tuple[float, float] = (0.682, 0.061),
    eps: float = 0.01,
    maxfun: int = 15,
    handoff_mode: str = "lee_only",
) -> tuple[float, float, float]:
    """Gradient-based calibration using finite-difference L-BFGS-B.

    Local refinement around a warm-start point (e.g., from Optuna coarse
    scan). Uses central finite differences for gradient computation.
    Typically converges in 10-15 evaluations vs 40-70 for Optuna TPE.

    Args:
        preset_name: Device preset.
        grid_shape: Grid resolution.
        x0: Initial (fc, fm) guess (warm start from prior calibration).
        eps: Finite difference step size.
        maxfun: Maximum function evaluations.
        handoff_mode: Snowplow Lp handoff mode.

    Returns:
        (best_fc, best_fm, best_objective).
    """
    from scipy.optimize import minimize

    eval_count = 0

    def obj(x: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        r = run_mlx_forward_model(
            x[0], x[1],
            preset_name=preset_name,
            grid_shape=grid_shape,
            handoff_mode=handoff_mode,
        )
        logger.info(
            "FD [%d/%d]: fc=%.3f fm=%.3f -> J=%.4f",
            eval_count, maxfun, x[0], x[1], r.objective,
        )
        return r.objective if r.success else 10.0

    result = minimize(
        obj,
        np.array(x0),
        method="L-BFGS-B",
        bounds=[(0.50, 0.85), (0.03, 0.30)],
        options={"maxfun": maxfun, "eps": eps},
    )

    return float(result.x[0]), float(result.x[1]), float(result.fun)


def run_calibration_pipeline(
    preset_name: str = "pf1000",
    n_optuna_trials: int = 40,
    skip_phase3: bool = False,
    skip_phase4: bool = False,
) -> MLXCalibrationResult:
    """Run the full 4-phase MLX calibration pipeline.

    Phase 1: 5x5 coarse grid scan (32x1x64)
    Phase 2: Optuna TPE optimization (32x1x64)
    Phase 3: Medium grid verification (48x1x96) on top 3
    Phase 4: Fine grid validation (64x1x128) on winner

    Args:
        preset_name: Device preset to calibrate.
        n_optuna_trials: Number of Phase 2 Optuna trials.
        skip_phase3: Skip medium-grid verification.
        skip_phase4: Skip fine-grid validation.

    Returns:
        MLXCalibrationResult with best parameters and all trials.
    """
    t_start = wall_time.monotonic()
    all_trials: list[MLXTrialResult] = []

    # === Phase 1: Coarse grid scan ===
    logger.info("=" * 60)
    logger.info("PHASE 1: Coarse Grid Scan (5x5, 32x1x64)")
    logger.info("=" * 60)

    phase1 = coarse_grid_scan(
        preset_name=preset_name,
        grid_shape=(32, 1, 64),
    )
    all_trials.extend(phase1)

    # Find best region from Phase 1
    successful = [t for t in phase1 if t.success]
    if not successful:
        logger.error("Phase 1: no successful trials")
        return MLXCalibrationResult(
            best=CalibrationResult(
                best_fc=0.7, best_fm=0.08,
                peak_current_error=1.0, timing_error=1.0,
                objective_value=10.0, n_evals=len(all_trials),
                converged=False, device_name=preset_name,
            ),
            trials=all_trials,
            phases_completed=1,
            total_wall_time_s=wall_time.monotonic() - t_start,
        )

    # Sort by objective, narrow bounds around top 5
    successful.sort(key=lambda t: t.objective)
    top5 = successful[:5]
    fc_narrow = (
        max(0.40, min(t.fc for t in top5) - 0.05),
        min(0.90, max(t.fc for t in top5) + 0.05),
    )
    fm_narrow = (
        max(0.02, min(t.fm for t in top5) - 0.03),
        min(0.40, max(t.fm for t in top5) + 0.03),
    )

    logger.info(
        "Phase 1 best: fc=%.3f fm=%.3f J=%.4f. Narrowed: fc=%s fm=%s",
        top5[0].fc, top5[0].fm, top5[0].objective,
        fc_narrow, fm_narrow,
    )

    # === Phase 2: Optuna TPE ===
    logger.info("=" * 60)
    logger.info("PHASE 2: Optuna TPE (%d trials, 32x1x64)", n_optuna_trials)
    logger.info("=" * 60)

    cal_result, phase2 = optuna_optimize(
        fc_bounds=fc_narrow,
        fm_bounds=fm_narrow,
        n_trials=n_optuna_trials,
        preset_name=preset_name,
        grid_shape=(32, 1, 64),
    )
    all_trials.extend(phase2)

    logger.info(
        "Phase 2 best: fc=%.3f fm=%.3f J=%.4f (I_err=%.1f%%, t_err=%.1f%%)",
        cal_result.best_fc, cal_result.best_fm, cal_result.objective_value,
        cal_result.peak_current_error * 100, cal_result.timing_error * 100,
    )

    best_result = cal_result
    phases = 2

    # === Phase 3: Medium grid verification ===
    if not skip_phase3:
        logger.info("=" * 60)
        logger.info("PHASE 3: Medium Grid Verification (48x1x96)")
        logger.info("=" * 60)

        # Top 3 candidates from Phase 2
        phase2_ok = [t for t in phase2 if t.success]
        phase2_ok.sort(key=lambda t: t.objective)
        candidates = phase2_ok[:3]

        phase3: list[MLXTrialResult] = []
        for c in candidates:
            trial = run_mlx_forward_model(
                fc=c.fc, fm=c.fm,
                preset_name=preset_name,
                grid_shape=(48, 1, 96),
            )
            phase3.append(trial)
            all_trials.append(trial)
            logger.info(
                "  fc=%.3f fm=%.3f -> J=%.4f (I_err=%.1f%%, NRMSE=%.3f)",
                trial.fc, trial.fm, trial.objective,
                trial.peak_error * 100, trial.nrmse,
            )

        phase3_ok = [t for t in phase3 if t.success]
        if phase3_ok:
            winner = min(phase3_ok, key=lambda t: t.objective)
            best_result = CalibrationResult(
                best_fc=winner.fc,
                best_fm=winner.fm,
                peak_current_error=winner.peak_error,
                timing_error=winner.timing_error,
                objective_value=winner.objective,
                n_evals=len(all_trials),
                converged=winner.objective < 0.5,
                device_name=preset_name,
            )
        phases = 3

    # === Phase 4: Fine grid validation ===
    if not skip_phase4:
        logger.info("=" * 60)
        logger.info("PHASE 4: Fine Grid Validation (64x1x128)")
        logger.info("=" * 60)

        fine_trial = run_mlx_forward_model(
            fc=best_result.best_fc,
            fm=best_result.best_fm,
            preset_name=preset_name,
            grid_shape=(64, 1, 128),
        )
        all_trials.append(fine_trial)

        if fine_trial.success:
            best_result = CalibrationResult(
                best_fc=fine_trial.fc,
                best_fm=fine_trial.fm,
                peak_current_error=fine_trial.peak_error,
                timing_error=fine_trial.timing_error,
                objective_value=fine_trial.objective,
                n_evals=len(all_trials),
                converged=fine_trial.objective < 0.5,
                device_name=preset_name,
            )
            logger.info(
                "Phase 4 FINAL: fc=%.3f fm=%.3f J=%.4f "
                "(I_peak=%.3f MA, I_err=%.1f%%, t_err=%.1f%%, NRMSE=%.3f)",
                fine_trial.fc, fine_trial.fm, fine_trial.objective,
                fine_trial.I_peak_A / 1e6,
                fine_trial.peak_error * 100,
                fine_trial.timing_error * 100,
                fine_trial.nrmse,
            )
        phases = 4

    total_time = wall_time.monotonic() - t_start

    logger.info("=" * 60)
    logger.info(
        "CALIBRATION COMPLETE: fc=%.3f fm=%.3f J=%.4f (%d evals, %.1f min)",
        best_result.best_fc, best_result.best_fm,
        best_result.objective_value, len(all_trials), total_time / 60,
    )
    logger.info("=" * 60)

    return MLXCalibrationResult(
        best=best_result,
        trials=all_trials,
        phases_completed=phases,
        total_wall_time_s=total_time,
    )
