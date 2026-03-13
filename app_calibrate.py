"""Auto-calibration of Lee model parameters against published experimental data.

Optimizes f_m (mass fraction) and f_c (current fraction) to minimize deviation
between simulated and published I_peak and t_peak. Uses differential evolution
for global optimization over the (f_m, f_c) parameter space.

This is a key differentiator vs existing DPF tools: RADPF requires manual trial-
and-error fitting, taking hours of expert effort per device. Auto-calibration
achieves comparable or better fits in seconds.

Usage:
    result = auto_calibrate("pf1000")
    print(result["best_fm"], result["best_fc"], result["I_peak_dev_pct"])
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app_engine import run_simulation_core


def _get_reference(preset_name: str) -> dict[str, float] | None:
    """Get published reference data for a preset."""
    try:
        from app_validation import _get_device
        dev = _get_device(preset_name)
        if dev is None:
            return None
        result: dict[str, float] = {
            "I_peak_MA": dev.peak_current / 1e6,
            "t_peak_us": dev.current_rise_time * 1e6,
        }
        if dev.neutron_yield > 0:
            result["Yn"] = dev.neutron_yield
        if dev.waveform_t is not None and dev.waveform_I is not None:
            result["has_waveform"] = 1.0
        return result
    except ImportError:
        return None


def _objective(
    params: np.ndarray,
    preset_name: str,
    ref: dict[str, float],
    sim_time_us: float,
    w_I: float,
    w_t: float,
    w_nrmse: float,
) -> float:
    """Objective function for optimization: weighted deviation from published data."""
    fm, fc = params
    try:
        data = run_simulation_core(
            preset_name, sim_time_us, fm=fm, fc=fc,
        )
    except Exception:
        return 1e6

    sim_I = data.get("I_pre_dip", data.get("I_peak", 0.0))
    sim_t = data.get("t_pre_dip", data.get("t_peak", 0.0))

    ref_I = ref["I_peak_MA"]
    ref_t = ref["t_peak_us"]

    if ref_I <= 0 or sim_I <= 0:
        return 1e6

    dI = abs(sim_I - ref_I) / ref_I
    dt_err = abs(sim_t - ref_t) / ref_t

    cost = w_I * dI + w_t * dt_err

    if ref.get("has_waveform") and w_nrmse > 0:
        try:
            from app_validation import _get_device
            from dpf.validation.experimental import nrmse_peak
            dev = _get_device(preset_name)
            if dev and dev.waveform_t is not None:
                t_sim_s = np.array(data["t_us"]) * 1e-6
                I_sim_A = np.array(data["I_MA"]) * 1e6
                nrmse = nrmse_peak(t_sim_s, I_sim_A, dev.waveform_t, dev.waveform_I)
                cost += w_nrmse * nrmse
        except Exception:
            pass

    return float(cost)


def auto_calibrate(
    preset_name: str,
    sim_time_us: float = 20.0,
    fm_bounds: tuple[float, float] = (0.02, 0.6),
    fc_bounds: tuple[float, float] = (0.4, 0.95),
    w_I: float = 1.0,
    w_t: float = 0.5,
    w_nrmse: float = 0.3,
    maxiter: int = 30,
    popsize: int = 10,
    tol: float = 1e-3,
) -> dict[str, Any]:
    """Auto-calibrate f_m and f_c for a preset against published data.

    Uses scipy.optimize.differential_evolution for global optimization.

    Args:
        preset_name: Preset to calibrate.
        sim_time_us: Simulation time [us].
        fm_bounds: Search range for mass fraction.
        fc_bounds: Search range for current fraction.
        w_I: Weight for I_peak deviation (default 1.0).
        w_t: Weight for t_peak deviation (default 0.5).
        w_nrmse: Weight for waveform NRMSE (default 0.3).
        maxiter: Maximum DE iterations.
        popsize: DE population size multiplier.
        tol: Convergence tolerance.

    Returns:
        Dictionary with best_fm, best_fc, deviations, and simulation results.
    """
    ref = _get_reference(preset_name)
    if ref is None:
        return {"error": f"No published data for preset '{preset_name}'"}

    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return _grid_calibrate(preset_name, ref, sim_time_us, fm_bounds, fc_bounds)

    result = differential_evolution(
        _objective,
        bounds=[fm_bounds, fc_bounds],
        args=(preset_name, ref, sim_time_us, w_I, w_t, w_nrmse),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=42,
        workers=1,
        updating="deferred",
    )

    best_fm, best_fc = result.x
    data = run_simulation_core(preset_name, sim_time_us, fm=best_fm, fc=best_fc)
    sim_I = data.get("I_pre_dip", data.get("I_peak", 0.0))
    sim_t = data.get("t_pre_dip", data.get("t_peak", 0.0))

    cal_result: dict[str, Any] = {
        "preset": preset_name,
        "best_fm": round(best_fm, 4),
        "best_fc": round(best_fc, 4),
        "I_peak_sim_MA": sim_I,
        "I_peak_ref_MA": ref["I_peak_MA"],
        "I_peak_dev_pct": abs(sim_I - ref["I_peak_MA"]) / ref["I_peak_MA"] * 100,
        "t_peak_sim_us": sim_t,
        "t_peak_ref_us": ref["t_peak_us"],
        "t_peak_dev_pct": abs(sim_t - ref["t_peak_us"]) / ref["t_peak_us"] * 100,
        "cost": result.fun,
        "converged": result.success,
        "iterations": result.nit,
    }

    return cal_result


def _grid_calibrate(
    preset_name: str,
    ref: dict[str, float],
    sim_time_us: float,
    fm_bounds: tuple[float, float],
    fc_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Fallback grid search when scipy is not available."""
    fm_vals = np.linspace(fm_bounds[0], fm_bounds[1], 15)
    fc_vals = np.linspace(fc_bounds[0], fc_bounds[1], 10)

    best_cost = 1e6
    best_fm, best_fc = 0.15, 0.70

    for fm in fm_vals:
        for fc in fc_vals:
            cost = _objective(
                np.array([fm, fc]), preset_name, ref, sim_time_us, 1.0, 0.5, 0.0,
            )
            if cost < best_cost:
                best_cost = cost
                best_fm, best_fc = fm, fc

    data = run_simulation_core(preset_name, sim_time_us, fm=best_fm, fc=best_fc)
    sim_I = data.get("I_pre_dip", data.get("I_peak", 0.0))
    sim_t = data.get("t_pre_dip", data.get("t_peak", 0.0))

    return {
        "preset": preset_name,
        "best_fm": round(best_fm, 4),
        "best_fc": round(best_fc, 4),
        "I_peak_sim_MA": sim_I,
        "I_peak_ref_MA": ref["I_peak_MA"],
        "I_peak_dev_pct": abs(sim_I - ref["I_peak_MA"]) / ref["I_peak_MA"] * 100,
        "t_peak_sim_us": sim_t,
        "t_peak_ref_us": ref["t_peak_us"],
        "t_peak_dev_pct": abs(sim_t - ref["t_peak_us"]) / ref["t_peak_us"] * 100,
        "cost": best_cost,
        "converged": True,
        "iterations": len(fm_vals) * len(fc_vals),
        "method": "grid_search",
    }


def calibrate_all_presets(
    sim_time_us: float = 20.0,
) -> list[dict[str, Any]]:
    """Run auto-calibration on all presets with published data."""
    from app_validation import PRESET_TO_DEVICE
    results = []
    for preset_name in PRESET_TO_DEVICE:
        cal = auto_calibrate(preset_name, sim_time_us=sim_time_us)
        results.append(cal)
    return results


def format_calibration_markdown(cal: dict[str, Any]) -> str:
    """Format calibration result as markdown."""
    if "error" in cal:
        return f"No published data for `{cal.get('preset', '?')}`."

    dI = cal["I_peak_dev_pct"]
    dt = cal["t_peak_dev_pct"]
    grade = "PASS" if dI <= 5 else "FAIR" if dI <= 15 else "POOR" if dI <= 30 else "FAIL"

    return (
        f"**Auto-calibrated** `{cal['preset']}`: "
        f"f_m={cal['best_fm']:.3f}, f_c={cal['best_fc']:.3f}\n\n"
        f"| Quantity | Simulation | Published | Deviation |\n"
        f"|----------|-----------|-----------|----------|\n"
        f"| I_peak | {cal['I_peak_sim_MA']:.3f} MA | "
        f"{cal['I_peak_ref_MA']:.3f} MA | {dI:.1f}% ({grade}) |\n"
        f"| t_peak | {cal['t_peak_sim_us']:.2f} us | "
        f"{cal['t_peak_ref_us']:.1f} us | {dt:.1f}% |\n"
    )
