#!/usr/bin/env python3
"""Gradient-based fc/fm calibration using finite-difference sensitivities.

Complements the Optuna TPE optimizer (calibrate_multi_device.py) with a local
gradient descent refinement. Typically 5-15 evaluations vs Optuna's 65+.

Strategy:
  1. Start from published Lee params or Optuna best
  2. Compute finite-difference gradients: d(loss)/d(fc), d(loss)/d(fm)
  3. Gradient descent with Armijo line search
  4. Converge when |grad| < tol or loss < target

Usage:
    python3 scripts/calibrate_gradient.py --device pf1000
    python3 scripts/calibrate_gradient.py --device pf1000 --fc0 0.797 --fm0 0.084
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradientResult:
    fc: float
    fm: float
    loss: float
    peak_error: float
    timing_error: float
    n_evals: int
    wall_time_s: float
    converged: bool
    history: list[dict]


def _loss_fn(
    fc: float,
    fm: float,
    preset_name: str,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    sim_time: float | None = None,
) -> tuple[float, float, float]:
    """Evaluate loss = weighted error for given (fc, fm).

    Returns (objective, peak_error, timing_error).
    """
    from dpf.validation.mlx_calibration import run_mlx_forward_model

    result = run_mlx_forward_model(
        fc=fc, fm=fm,
        preset_name=preset_name,
        grid_shape=grid_shape,
        sim_time=sim_time,
        peak_weight=0.4,
        timing_weight=0.3,
        waveform_weight=0.3,
    )
    return result.objective, result.peak_error, result.timing_error


def _fd_gradient(
    fc: float,
    fm: float,
    loss_center: float,
    preset_name: str,
    h_fc: float = 0.005,
    h_fm: float = 0.002,
    **kwargs,
) -> tuple[float, float, int]:
    """Central finite-difference gradient of loss w.r.t. (fc, fm).

    Returns (dloss_dfc, dloss_dfm, n_evals).
    """
    loss_fc_p, _, _ = _loss_fn(fc + h_fc, fm, preset_name, **kwargs)
    loss_fc_m, _, _ = _loss_fn(fc - h_fc, fm, preset_name, **kwargs)
    dloss_dfc = (loss_fc_p - loss_fc_m) / (2.0 * h_fc)

    loss_fm_p, _, _ = _loss_fn(fc, fm + h_fm, preset_name, **kwargs)
    loss_fm_m, _, _ = _loss_fn(fc, fm - h_fm, preset_name, **kwargs)
    dloss_dfm = (loss_fm_p - loss_fm_m) / (2.0 * h_fm)

    return dloss_dfc, dloss_dfm, 4  # 4 additional evals


def calibrate_gradient(
    preset_name: str = "pf1000",
    fc0: float = 0.70,
    fm0: float = 0.08,
    lr: float = 0.5,
    max_iters: int = 20,
    tol: float = 1e-4,
    target_loss: float = 0.05,
    fc_bounds: tuple[float, float] = (0.50, 0.95),
    fm_bounds: tuple[float, float] = (0.03, 0.30),
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    sim_time: float | None = None,
) -> GradientResult:
    """Run gradient descent calibration for (fc, fm).

    Parameters
    ----------
    preset_name : str
        Device preset.
    fc0, fm0 : float
        Initial guess for current/mass fractions.
    lr : float
        Learning rate (step size multiplier).
    max_iters : int
        Maximum gradient descent iterations.
    tol : float
        Convergence tolerance on |grad|.
    target_loss : float
        Stop if loss falls below this.
    fc_bounds, fm_bounds : tuple
        Parameter bounds.
    grid_shape : tuple
        Solver grid resolution.
    sim_time : float or None
        Override simulation time.
    """
    t0 = time.monotonic()
    fc, fm = fc0, fm0
    n_evals = 0
    history = []
    converged = False

    # Initial evaluation
    loss, I_err, t_err = _loss_fn(fc, fm, preset_name, grid_shape, sim_time)
    n_evals += 1
    logger.info(f"[iter 0] fc={fc:.4f} fm={fm:.4f} loss={loss:.4f} "
                f"I_err={I_err:.3f} t_err={t_err:.3f}")
    history.append({"iter": 0, "fc": fc, "fm": fm, "loss": loss,
                     "peak_error": I_err, "timing_error": t_err})

    for it in range(1, max_iters + 1):
        if loss < target_loss:
            converged = True
            logger.info(f"Converged: loss {loss:.4f} < target {target_loss}")
            break

        # Finite-difference gradient
        dfc, dfm, fd_evals = _fd_gradient(
            fc, fm, loss, preset_name, grid_shape=grid_shape, sim_time=sim_time,
        )
        n_evals += fd_evals

        grad_norm = np.sqrt(dfc**2 + dfm**2)
        if grad_norm < tol:
            converged = True
            logger.info(f"Converged: |grad| {grad_norm:.6f} < tol {tol}")
            break

        # Gradient step with projection to bounds
        fc_new = np.clip(fc - lr * dfc, *fc_bounds)
        fm_new = np.clip(fm - lr * dfm, *fm_bounds)

        # Evaluate new point
        loss_new, I_err_new, t_err_new = _loss_fn(
            fc_new, fm_new, preset_name, grid_shape, sim_time,
        )
        n_evals += 1

        # Simple Armijo: accept if loss decreased, else halve step
        if loss_new < loss:
            fc, fm = fc_new, fm_new
            loss, I_err, t_err = loss_new, I_err_new, t_err_new
        else:
            # Backtrack: try half step
            fc_half = np.clip(fc - 0.5 * lr * dfc, *fc_bounds)
            fm_half = np.clip(fm - 0.5 * lr * dfm, *fm_bounds)
            loss_half, I_err_half, t_err_half = _loss_fn(
                fc_half, fm_half, preset_name, grid_shape, sim_time,
            )
            n_evals += 1
            if loss_half < loss:
                fc, fm = fc_half, fm_half
                loss, I_err, t_err = loss_half, I_err_half, t_err_half
                lr *= 0.5  # shrink step for future
            else:
                logger.warning(f"[iter {it}] No improvement, reducing lr")
                lr *= 0.25

        logger.info(f"[iter {it}] fc={fc:.4f} fm={fm:.4f} loss={loss:.4f} "
                     f"|grad|={grad_norm:.4f} lr={lr:.4f} evals={n_evals}")
        history.append({"iter": it, "fc": fc, "fm": fm, "loss": loss,
                         "peak_error": I_err, "timing_error": t_err,
                         "grad_norm": grad_norm})

    wall_time = time.monotonic() - t0
    return GradientResult(
        fc=fc, fm=fm, loss=loss,
        peak_error=I_err, timing_error=t_err,
        n_evals=n_evals, wall_time_s=wall_time,
        converged=converged, history=history,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Gradient-based fc/fm calibration")
    parser.add_argument("--device", default="pf1000")
    parser.add_argument("--fc0", type=float, default=None)
    parser.add_argument("--fm0", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--grid", default="32,1,64")
    args = parser.parse_args()

    # Default starting points from Phase Q calibration
    defaults = {
        "pf1000": (0.797, 0.084),
        "unu_ictp": (0.70, 0.08),
        "poseidon_60kv": (0.60, 0.275),
        "faeton": (0.70, 0.70),
    }
    fc0, fm0 = defaults.get(args.device, (0.70, 0.08))
    if args.fc0 is not None:
        fc0 = args.fc0
    if args.fm0 is not None:
        fm0 = args.fm0

    grid = tuple(int(x) for x in args.grid.split(","))

    result = calibrate_gradient(
        preset_name=args.device,
        fc0=fc0, fm0=fm0,
        lr=args.lr,
        max_iters=args.max_iters,
        grid_shape=grid,
    )

    print(f"\n{'='*60}")
    print(f"Device: {args.device}")
    print(f"Best: fc={result.fc:.4f}, fm={result.fm:.4f}")
    print(f"Loss: {result.loss:.4f}")
    print(f"I_peak error: {result.peak_error:.1%}")
    print(f"t_peak error: {result.timing_error:.1%}")
    print(f"Evaluations: {result.n_evals}")
    print(f"Wall time: {result.wall_time_s:.1f}s")
    print(f"Converged: {result.converged}")

    # Save results
    out_path = Path(f"results/gradient_calibration_{args.device}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "fc": result.fc, "fm": result.fm,
            "loss": result.loss,
            "peak_error": result.peak_error,
            "timing_error": result.timing_error,
            "n_evals": result.n_evals,
            "wall_time_s": result.wall_time_s,
            "converged": result.converged,
            "history": result.history,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
