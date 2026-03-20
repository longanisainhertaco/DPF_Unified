"""Full waveform NRMSE + dI/dt validation for all devices with digitized I(t).

Sprint Item #1: Honest waveform shape comparison beyond scalar I_peak / t_peak.

For each device with a digitized experimental waveform:
  1. Run simulation with published Lee model parameters
  2. Compute full waveform NRMSE (peak-normalized)
  3. Compute dI/dt via np.gradient for both sim and experiment
  4. Report dI/dt zero-crossing timing comparison (= time of peak current)
  5. Report current dip depth comparison where applicable
  6. Report rise-phase-only NRMSE (truncated at experimental peak time)

Devices with waveforms: PF-1000, PF-1000-Gribkov, PF-1000-16kV,
                         UNU-ICTP, POSEIDON-60kV, FAETON-I, MJOLNIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_engine import run_simulation_core
from dpf.validation.experimental import (
    DEVICES,
    ExperimentalDevice,
    nrmse_peak,
)

# Map device name -> preset name for run_simulation_core
DEVICE_TO_PRESET: dict[str, str] = {
    "PF-1000": "pf1000",
    "PF-1000-Gribkov": "pf1000",
    "PF-1000-16kV": "pf1000",
    "PF-1000-20kV": "pf1000_20kv",
    "UNU-ICTP": "unu_ictp",
    "POSEIDON-60kV": "poseidon_60kv",
    "FAETON-I": "faeton",
    "MJOLNIR": "mjolnir",
}


def run_device(name: str, dev: ExperimentalDevice) -> dict:
    """Run simulation for a device and return results dict."""
    preset = DEVICE_TO_PRESET.get(name)
    if preset is None:
        return {"error": f"No preset mapping for {name}"}

    # Determine sim_time: 2x experimental rise time or 2x waveform duration
    if dev.waveform_t is not None:
        t_max_exp = float(dev.waveform_t[-1])
        sim_time_us = max(t_max_exp * 1e6 * 1.2, dev.current_rise_time * 1e6 * 2.5)
    else:
        sim_time_us = dev.current_rise_time * 1e6 * 2.5

    kwargs: dict = {
        "preset_name": preset,
        "sim_time_us": sim_time_us,
        "V0_kV": dev.voltage / 1e3,
        "C_uF": dev.capacitance * 1e6,
        "L0_nH": dev.inductance * 1e9,
        "R0_mOhm": dev.resistance * 1e3,
        "anode_r_mm": dev.anode_radius * 1e3,
        "cathode_r_mm": dev.cathode_radius * 1e3,
        "anode_len_mm": dev.anode_length * 1e3,
        "pressure_torr": dev.fill_pressure_torr,
        "fc": dev.lee_fc if dev.lee_fc > 0 else None,
        "fm": dev.lee_fm if dev.lee_fm > 0 else None,
    }
    if dev.lee_fmr > 0:
        kwargs["fmr"] = dev.lee_fmr
    if dev.lee_fcr > 0:
        kwargs["fcr"] = dev.lee_fcr

    # PF-1000 variants need R0 correction (from validate_24shot.py)
    if name in ("PF-1000", "PF-1000-Gribkov"):
        kwargs["R0_mOhm"] = dev.resistance * 1e3 + 6.43
    elif name == "PF-1000-16kV":
        kwargs["R0_mOhm"] = dev.resistance * 1e3 + 6.43

    return run_simulation_core(**kwargs)


def find_dip(t: np.ndarray, I: np.ndarray) -> tuple[float, float, float]:
    """Find current dip after peak. Returns (t_dip, I_dip, dip_depth_pct)."""
    abs_I = np.abs(I)
    peak_idx = int(np.argmax(abs_I))
    I_peak = float(abs_I[peak_idx])

    # Search for local minimum in post-peak region (within 2x peak time)
    t_peak = float(t[peak_idx])
    search_end = np.searchsorted(t, 2.0 * t_peak)
    search_end = min(search_end, len(abs_I))

    post_peak = abs_I[peak_idx:search_end]
    if len(post_peak) < 3:
        return float(t[peak_idx]), I_peak, 0.0

    dip_offset = int(np.argmin(post_peak))
    if dip_offset <= 1:
        return float(t[peak_idx]), I_peak, 0.0

    dip_idx = peak_idx + dip_offset
    I_dip = float(abs_I[dip_idx])
    t_dip = float(t[dip_idx])
    dip_depth = (1.0 - I_dip / I_peak) * 100.0 if I_peak > 0 else 0.0
    return t_dip, I_dip, dip_depth


def safe_gradient(I: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute dI/dt handling duplicate time points (e.g. Gribkov waveform)."""
    # Remove duplicate time points
    dt = np.diff(t)
    mask = np.ones(len(t), dtype=bool)
    for i in range(len(dt)):
        if dt[i] <= 0:
            mask[i + 1] = False
    if not np.all(mask):
        t_clean = t[mask]
        I_clean = I[mask]
        dIdt_clean = np.gradient(I_clean, t_clean)
        # Interpolate back to original grid
        return np.interp(t, t_clean, dIdt_clean)
    return np.gradient(I, t)


def didt_zero_crossing(t: np.ndarray, I: np.ndarray) -> float:
    """Find time where dI/dt crosses zero (= peak current time)."""
    dIdt = safe_gradient(I, t)
    # Find first zero crossing after significant current (>10% of peak)
    abs_I = np.abs(I)
    threshold = 0.1 * np.max(abs_I)
    for i in range(1, len(dIdt)):
        if abs_I[i] > threshold and dIdt[i - 1] > 0 and dIdt[i] <= 0:
            # Linear interpolation for precise crossing
            if dIdt[i - 1] != dIdt[i]:
                frac = dIdt[i - 1] / (dIdt[i - 1] - dIdt[i])
                return float(t[i - 1] + frac * (t[i] - t[i - 1]))
            return float(t[i])
    return float(t[int(np.argmax(abs_I))])


def analyze_device(name: str, dev: ExperimentalDevice) -> dict:
    """Run simulation and compute all waveform metrics."""
    result = run_device(name, dev)
    if "error" in result:
        return {"device": name, "error": result["error"]}

    t_sim_s = np.array(result["t_us"]) * 1e-6
    I_sim_A = np.array(result["I_MA"]) * 1e6

    t_exp = dev.waveform_t
    I_exp = dev.waveform_I

    metrics: dict = {
        "device": name,
        "provenance": dev.waveform_provenance,
        "reliability": dev.reliability,
        "n_exp_points": len(t_exp),
    }

    # Scalar metrics
    I_peak_sim = float(np.max(np.abs(I_sim_A)))
    I_peak_exp = dev.peak_current
    metrics["I_peak_sim_MA"] = I_peak_sim / 1e6
    metrics["I_peak_exp_MA"] = I_peak_exp / 1e6
    metrics["I_peak_err_pct"] = abs(I_peak_sim - I_peak_exp) / I_peak_exp * 100

    # 1. Full waveform NRMSE
    metrics["nrmse_full"] = nrmse_peak(t_sim_s, I_sim_A, t_exp, I_exp)

    # 2. Rise-phase NRMSE (truncated at experimental peak time)
    t_peak_exp = dev.current_rise_time
    metrics["nrmse_rise"] = nrmse_peak(
        t_sim_s, I_sim_A, t_exp, I_exp, max_time=t_peak_exp
    )

    # 3. NRMSE truncated at dip
    metrics["nrmse_to_dip"] = nrmse_peak(
        t_sim_s, I_sim_A, t_exp, I_exp, truncate_at_dip=True
    )

    # 4. dI/dt zero-crossing comparison
    t_zero_sim = didt_zero_crossing(t_sim_s, I_sim_A)
    t_zero_exp = didt_zero_crossing(t_exp, I_exp)
    metrics["dIdt_zero_sim_us"] = t_zero_sim * 1e6
    metrics["dIdt_zero_exp_us"] = t_zero_exp * 1e6
    metrics["dIdt_zero_err_pct"] = (
        abs(t_zero_sim - t_zero_exp) / max(t_zero_exp, 1e-15) * 100
    )

    # 5. dI/dt peak magnitude comparison (maximum rate of current rise)
    dIdt_sim = safe_gradient(I_sim_A, t_sim_s)
    dIdt_exp = safe_gradient(I_exp, t_exp)
    max_dIdt_sim = float(np.max(dIdt_sim))
    max_dIdt_exp = float(np.max(dIdt_exp))
    metrics["max_dIdt_sim_GA_s"] = max_dIdt_sim / 1e9
    metrics["max_dIdt_exp_GA_s"] = max_dIdt_exp / 1e9
    if max_dIdt_exp > 0:
        metrics["max_dIdt_err_pct"] = abs(max_dIdt_sim - max_dIdt_exp) / max_dIdt_exp * 100
    else:
        metrics["max_dIdt_err_pct"] = float("nan")

    # 6. dI/dt NRMSE (resample sim dI/dt onto exp time grid)
    dIdt_sim_resampled = np.interp(t_exp, t_sim_s, dIdt_sim)
    dIdt_rmse = float(np.sqrt(np.mean((dIdt_sim_resampled - dIdt_exp) ** 2)))
    dIdt_peak_exp = float(np.max(np.abs(dIdt_exp)))
    metrics["dIdt_nrmse"] = dIdt_rmse / max(dIdt_peak_exp, 1e-300)

    # 7. Current dip comparison
    t_dip_sim, I_dip_sim, dip_sim = find_dip(t_sim_s, I_sim_A)
    t_dip_exp, I_dip_exp, dip_exp = find_dip(t_exp, I_exp)
    metrics["dip_depth_sim_pct"] = dip_sim
    metrics["dip_depth_exp_pct"] = dip_exp
    metrics["dip_time_sim_us"] = t_dip_sim * 1e6
    metrics["dip_time_exp_us"] = t_dip_exp * 1e6

    return metrics


def main() -> None:
    # Collect devices that have waveform data
    waveform_devices = {
        name: dev for name, dev in DEVICES.items()
        if dev.waveform_t is not None and dev.waveform_I is not None
    }

    print("=" * 90)
    print("FULL WAVEFORM NRMSE + dI/dt VALIDATION")
    print("=" * 90)
    print(f"Devices with digitized waveforms: {len(waveform_devices)}")
    print(f"Devices: {', '.join(waveform_devices.keys())}")
    print()

    all_results = []
    for name, dev in waveform_devices.items():
        if name not in DEVICE_TO_PRESET:
            print(f"  SKIP {name}: no preset mapping")
            continue

        print(f"Running {name}...", end=" ", flush=True)
        try:
            m = analyze_device(name, dev)
            all_results.append(m)
            if "error" in m:
                print(f"ERROR: {m['error']}")
            else:
                print(f"done (NRMSE={m['nrmse_full']:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({"device": name, "error": str(e)})

    # Summary table
    print()
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    # Header
    hdr = (
        f"{'Device':<18} {'Prov':<6} {'I_pk err%':>9} {'NRMSE':>7} "
        f"{'NRMSE rise':>10} {'NRMSE dip':>9} {'dIdt NRMSE':>10} "
        f"{'dIdt t0 err%':>12} {'Dip sim%':>8} {'Dip exp%':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for m in all_results:
        if "error" in m:
            print(f"{m['device']:<18} ERROR: {m['error']}")
            continue

        prov = m.get("provenance", "?")[:6]
        print(
            f"{m['device']:<18} {prov:<6} "
            f"{m['I_peak_err_pct']:>8.1f}% "
            f"{m['nrmse_full']:>7.3f} "
            f"{m['nrmse_rise']:>10.3f} "
            f"{m['nrmse_to_dip']:>9.3f} "
            f"{m['dIdt_nrmse']:>10.3f} "
            f"{m['dIdt_zero_err_pct']:>11.1f}% "
            f"{m['dip_depth_sim_pct']:>7.1f}% "
            f"{m['dip_depth_exp_pct']:>7.1f}%"
        )

    # Detailed per-device output
    print()
    print("=" * 90)
    print("DETAILED PER-DEVICE RESULTS")
    print("=" * 90)

    for m in all_results:
        if "error" in m:
            continue
        print(f"\n--- {m['device']} ({m['provenance']}, {m['reliability']}) ---")
        print(f"  Exp waveform points: {m['n_exp_points']}")
        print(f"  I_peak: sim={m['I_peak_sim_MA']:.3f} MA, exp={m['I_peak_exp_MA']:.3f} MA, err={m['I_peak_err_pct']:.1f}%")
        print(f"  Full waveform NRMSE:  {m['nrmse_full']:.4f}  ({m['nrmse_full']*100:.1f}%)")
        print(f"  Rise-phase NRMSE:     {m['nrmse_rise']:.4f}  ({m['nrmse_rise']*100:.1f}%)")
        print(f"  To-dip NRMSE:         {m['nrmse_to_dip']:.4f}  ({m['nrmse_to_dip']*100:.1f}%)")
        print(f"  dI/dt NRMSE:          {m['dIdt_nrmse']:.4f}  ({m['dIdt_nrmse']*100:.1f}%)")
        print(f"  dI/dt zero crossing:  sim={m['dIdt_zero_sim_us']:.2f} us, exp={m['dIdt_zero_exp_us']:.2f} us, err={m['dIdt_zero_err_pct']:.1f}%")
        print(f"  Max dI/dt:            sim={m['max_dIdt_sim_GA_s']:.2f} GA/s, exp={m['max_dIdt_exp_GA_s']:.2f} GA/s, err={m['max_dIdt_err_pct']:.1f}%")
        print(f"  Current dip:          sim={m['dip_depth_sim_pct']:.1f}% at {m['dip_time_sim_us']:.2f} us, exp={m['dip_depth_exp_pct']:.1f}% at {m['dip_time_exp_us']:.2f} us")

    # Statistical summary
    measured_results = [m for m in all_results if "error" not in m and m.get("provenance") == "measured"]
    all_valid = [m for m in all_results if "error" not in m]

    if measured_results:
        print()
        print("=" * 90)
        print("STATISTICAL SUMMARY (measured waveforms only)")
        print("=" * 90)
        nrmse_vals = [m["nrmse_full"] for m in measured_results]
        nrmse_rise_vals = [m["nrmse_rise"] for m in measured_results]
        ipk_errs = [m["I_peak_err_pct"] for m in measured_results]
        didt_nrmse = [m["dIdt_nrmse"] for m in measured_results if np.isfinite(m["dIdt_nrmse"])]
        print(f"  N devices (measured): {len(measured_results)}")
        print(f"  Full NRMSE:  mean={np.mean(nrmse_vals):.3f}, max={np.max(nrmse_vals):.3f}, min={np.min(nrmse_vals):.3f}")
        print(f"  Rise NRMSE:  mean={np.mean(nrmse_rise_vals):.3f}, max={np.max(nrmse_rise_vals):.3f}")
        print(f"  I_peak err:  mean={np.mean(ipk_errs):.1f}%, max={np.max(ipk_errs):.1f}%")
        if didt_nrmse:
            print(f"  dI/dt NRMSE: mean={np.mean(didt_nrmse):.3f}, max={np.max(didt_nrmse):.3f}")

    if all_valid:
        print()
        print("=" * 90)
        print("STATISTICAL SUMMARY (all devices with waveforms)")
        print("=" * 90)
        nrmse_vals = [m["nrmse_full"] for m in all_valid]
        nrmse_rise_vals = [m["nrmse_rise"] for m in all_valid]
        ipk_errs = [m["I_peak_err_pct"] for m in all_valid]
        didt_nrmse = [m["dIdt_nrmse"] for m in all_valid if np.isfinite(m["dIdt_nrmse"])]
        print(f"  N devices (all): {len(all_valid)}")
        print(f"  Full NRMSE:  mean={np.mean(nrmse_vals):.3f}, max={np.max(nrmse_vals):.3f}, min={np.min(nrmse_vals):.3f}")
        print(f"  Rise NRMSE:  mean={np.mean(nrmse_rise_vals):.3f}, max={np.max(nrmse_rise_vals):.3f}")
        print(f"  I_peak err:  mean={np.mean(ipk_errs):.1f}%, max={np.max(ipk_errs):.1f}%")
        if didt_nrmse:
            print(f"  dI/dt NRMSE: mean={np.mean(didt_nrmse):.3f}, max={np.max(didt_nrmse):.3f}")

    # Verdict
    print()
    print("=" * 90)
    if measured_results:
        mean_nrmse = np.mean([m["nrmse_full"] for m in measured_results])
        max_nrmse = np.max([m["nrmse_full"] for m in measured_results])
        if mean_nrmse < 0.10:
            print(f"VERDICT: PASS — Mean waveform NRMSE {mean_nrmse:.3f} < 10% (measured devices)")
        elif mean_nrmse < 0.20:
            print(f"VERDICT: MARGINAL — Mean waveform NRMSE {mean_nrmse:.3f} < 20% but > 10%")
        else:
            print(f"VERDICT: FAIL — Mean waveform NRMSE {mean_nrmse:.3f} >= 20%")
        print(f"  Worst case: {max_nrmse:.3f} ({max([m for m in measured_results], key=lambda x: x['nrmse_full'])['device']})")
    print("=" * 90)


if __name__ == "__main__":
    main()
