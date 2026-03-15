"""Auto-calibration of Lee model parameters via LeeModelCalibrator.

Delegates optimization to dpf.validation.calibration.LeeModelCalibrator
(Nelder-Mead + LeeModel.run()) instead of running its own optimizer.
"""
from __future__ import annotations

from typing import Any


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


def auto_calibrate(
    preset_name: str,
    sim_time_us: float | None = None,
    optimize_radial: bool = False,
) -> dict[str, Any]:
    """Auto-calibrate fc/fm via LeeModelCalibrator."""
    from app_validation import PRESET_TO_DEVICE
    from dpf.validation.calibration import (
        _DEFAULT_CROWBAR_R,
        _DEFAULT_DEVICE_PCF,
        LeeModelCalibrator,
    )
    from dpf.validation.experimental import DEVICES

    device_name = PRESET_TO_DEVICE.get(preset_name)
    if device_name is None:
        return {"error": f"Unknown preset: {preset_name}"}

    device = DEVICES.get(device_name)
    if device is None:
        return {"error": f"No experimental data for {device_name}"}

    pcf = _DEFAULT_DEVICE_PCF.get(device_name, 0.14)
    crowbar_r = _DEFAULT_CROWBAR_R.get(device_name, 0.0)

    try:
        cal = LeeModelCalibrator(
            device_name=device_name,
            pinch_column_fraction=pcf,
            crowbar_enabled=crowbar_r > 0,
            crowbar_resistance=crowbar_r,
        )
        result = cal.calibrate(
            fc_bounds=(0.4, 0.95),
            fm_bounds=(0.02, 0.6),
            maxiter=100,
        )
    except Exception as e:
        return {"error": f"Calibration failed: {e}"}

    out: dict[str, Any] = {
        "best_fc": result.best_fc,
        "best_fm": result.best_fm,
        "I_peak_error": result.peak_current_error,
        "t_peak_error": result.timing_error,
        "n_evals": result.n_evals,
        "converged": result.converged,
        "device_name": device_name,
        "preset": preset_name,
    }

    # Add published Lee params for comparison
    try:
        benchmark = cal.benchmark_against_published(result)
        out["published_fc"] = benchmark.get("fc_published_range")
        out["published_fm"] = benchmark.get("fm_published_range")
        out["fc_in_range"] = benchmark.get("fc_in_range")
        out["fm_in_range"] = benchmark.get("fm_in_range")
        out["published_reference"] = benchmark.get("reference")
    except (KeyError, Exception):
        pass

    if device is not None:
        out["lee_fc"] = device.lee_fc
        out["lee_fm"] = device.lee_fm
        out["lee_fmr"] = device.lee_fmr
        out["lee_fcr"] = device.lee_fcr
        out["lee_reference"] = device.lee_reference

    return out


def get_published_params(preset_name: str) -> tuple[float | None, float | None]:
    """Get published Lee model fc/fm for a device preset."""
    from app_validation import PRESET_TO_DEVICE
    from dpf.validation.experimental import DEVICES

    device_name = PRESET_TO_DEVICE.get(preset_name)
    if device_name is None:
        return None, None
    device = DEVICES.get(device_name)
    if device is None:
        return None, None
    return device.lee_fc, device.lee_fm


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
        return f"Calibration error: {cal['error']}"

    fc = cal["best_fc"]
    fm = cal["best_fm"]
    I_err_pct = cal.get("I_peak_error", 0.0) * 100
    t_err_pct = cal.get("t_peak_error", 0.0) * 100 if cal.get("t_peak_error") is not None else None

    grade = "PASS" if I_err_pct <= 5 else "FAIR" if I_err_pct <= 10 else "POOR" if I_err_pct <= 20 else "FAIL"

    lines = [
        f"**Calibrated Parameters**: fc={fc:.3f}, fm={fm:.3f}",
        "",
    ]

    pub_fc = cal.get("published_fc")
    pub_fm = cal.get("published_fm")
    pub_ref = cal.get("published_reference", "")
    if pub_fc and pub_fm:
        ref_label = f" ({pub_ref})" if pub_ref else ""
        lines.append(f"**Published Lee Params**: fc={pub_fc[0]:.2f}–{pub_fc[1]:.2f}, fm={pub_fm[0]:.3f}–{pub_fm[1]:.3f}{ref_label}")

        fc_ok = cal.get("fc_in_range")
        fm_ok = cal.get("fm_in_range")
        if fc_ok is not None and fm_ok is not None:
            fc_str = "Yes" if fc_ok else "No"
            fm_str = "Yes" if fm_ok else "No"
            lines.append(f"**Within Published Range?**: fc={fc_str}, fm={fm_str}")
        lines.append("")
    elif cal.get("lee_fc") or cal.get("lee_fm"):
        lee_fc = cal.get("lee_fc", 0.0)
        lee_fm = cal.get("lee_fm", 0.0)
        lee_ref = cal.get("lee_reference", "")
        ref_label = f" ({lee_ref})" if lee_ref else ""
        if lee_fc or lee_fm:
            lines.append(f"**Published Lee Params**: fc={lee_fc:.3f}, fm={lee_fm:.3f}{ref_label}")
            lines.append("")

    lines.append(f"**I_peak Error**: {I_err_pct:.1f}% ({grade})")

    if t_err_pct is not None:
        lines.append(f"**t_peak Error**: {t_err_pct:.1f}%")

    Yn_exp = cal.get("Yn_exp")
    Yn_sim = cal.get("Yn_sim")
    if Yn_exp and Yn_sim:
        import math
        log_dev = abs(math.log10(Yn_sim) - math.log10(Yn_exp)) if Yn_sim > 0 and Yn_exp > 0 else None
        yn_grade = "PASS" if log_dev is not None and log_dev < 1.0 else "FAIL"
        log_str = f" ({log_dev:.2f} dec {yn_grade})" if log_dev is not None else ""
        lines.append(f"**Neutron Yield**: sim={Yn_sim:.2e}, exp={Yn_exp:.2e}{log_str}")

    n_evals = cal.get("n_evals")
    if n_evals:
        lines.append(f"**Evaluations**: {n_evals}")

    return "\n".join(lines) + "\n"
