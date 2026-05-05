"""Experimental validation bridge for DPF web UI.

Wraps dpf.validation.experimental to compare simulation results against
published measurements. Maps preset names to device records.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PRESET_TO_DEVICE: dict[str, str] = {
    "pf1000": "PF-1000",
    "pf1000_akel": "PF-1000-16kV",
    "pf1000_20kv": "PF-1000-20kV",
    "nx2": "NX2",
    "unu_ictp": "UNU-ICTP",
    # "poseidon" (40 kV) is _REFERENCE_ONLY — Herold 1989 not in KnowledgeReference/. Excluded.
    "poseidon_60kv": "POSEIDON-60kV",
    "mjolnir": "MJOLNIR",
    "faeton": "FAETON-I",
}


def _get_device(preset_name: str):
    """Look up ExperimentalDevice for a preset, or None."""
    device_key = PRESET_TO_DEVICE.get(preset_name)
    if device_key is None:
        return None
    try:
        from dpf.validation.experimental import DEVICES
        return DEVICES.get(device_key)
    except ImportError:
        return None


def validate_against_published(
    data: dict[str, Any],
    preset_name: str,
) -> dict[str, Any] | None:
    """Compare simulation results to published experimental data.

    Returns dict with deviations and pass/fail status, or None if no
    published data exists for the preset.
    """
    dev = _get_device(preset_name)
    if dev is None:
        return None

    sim_I_peak = data.get("I_pre_dip") or data.get("I_peak", 0.0)
    # t_peak_dev_pct must use the global current maximum time, not t_pre_dip.
    # t_pre_dip is the local peak inside the 0.5 us radial-entry window used for
    # dip-depth baseline (FAETON fix, Wave-5 O3). For non-FAETON devices that reach
    # global I_peak before radial onset (e.g. POSEIDON-60kV), t_pre_dip is a sub-peak
    # well before t_peak, causing a spurious +70.5% t_peak error (Wave-6 S13 regression).
    # t_peak is always the global argmax and is correct for all devices.
    sim_t_peak = data.get("t_peak", 0.0)

    ref_I = dev.peak_current / 1e6
    ref_t = dev.current_rise_time * 1e6

    dI_pct = abs(sim_I_peak - ref_I) / ref_I * 100 if ref_I > 0 else float("inf")
    dt_pct = abs(sim_t_peak - ref_t) / ref_t * 100 if ref_t > 0 else float("inf")

    result: dict[str, Any] = {
        "preset": preset_name,
        "device": dev.name,
        "source": dev.reference,
        "reliability": getattr(dev, "reliability", "measured"),
        "I_peak_sim_MA": sim_I_peak,
        "I_peak_ref_MA": ref_I,
        "I_peak_dev_pct": dI_pct,
        "I_peak_uncertainty": dev.peak_current_uncertainty,
        "t_peak_sim_us": sim_t_peak,
        "t_peak_ref_us": ref_t,
        "t_peak_dev_pct": dt_pct,
    }

    # Waveform NRMSE if digitized data exists
    if dev.waveform_t is not None and dev.waveform_I is not None:
        try:
            from dpf.validation.experimental import nrmse_peak
            t_sim_s = np.array(data["t_us"]) * 1e-6
            I_sim_A = np.array(data["I_MA"]) * 1e6
            nrmse = nrmse_peak(t_sim_s, I_sim_A, dev.waveform_t, dev.waveform_I)
            result["waveform_nrmse"] = nrmse
        except Exception:
            pass

    # Current dip depth comparison (CTQ-3)
    if dev.waveform_t is not None and dev.waveform_I is not None:
        exp_I = np.abs(dev.waveform_I) / 1e6  # MA
        exp_peak_idx = int(np.argmax(exp_I))
        exp_post = exp_I[exp_peak_idx:]
        if len(exp_post) > 2:
            exp_dip_idx = exp_peak_idx + int(np.argmin(exp_post[:max(len(exp_post) // 2, 2)]))
            exp_dip_pct = (exp_I[exp_peak_idx] - exp_I[exp_dip_idx]) / max(exp_I[exp_peak_idx], 1e-30) * 100

            sim_dip_pct = data.get("dip_pct", 0.0)
            result["dip_sim_pct"] = sim_dip_pct
            result["dip_exp_pct"] = float(exp_dip_pct)
            result["dip_error_pct"] = abs(sim_dip_pct - exp_dip_pct)

    # Neutron yield comparison
    ny = data.get("neutron_yield")
    if ny and dev.neutron_yield > 0:
        sim_yn = ny.get("Y_neutron", 0)
        if sim_yn > 0:
            log_ratio = abs(np.log10(sim_yn / dev.neutron_yield))
            result["Yn_sim"] = sim_yn
            result["Yn_ref"] = dev.neutron_yield
            result["Yn_log_ratio"] = log_ratio
            result["bt_fraction"] = ny.get("bt_fraction", 0)
            result["V_pinch_kV"] = ny.get("V_pinch_kV", 0)

    return result


def format_validation_markdown(val: dict[str, Any] | None) -> str:
    """Format validation result as markdown for display in the UI."""
    if val is None:
        return ""

    dI = val["I_peak_dev_pct"]
    dt = val["t_peak_dev_pct"]
    u_I = val.get("I_peak_uncertainty", 0)

    def grade(pct: float) -> str:
        if pct <= 5:
            return "PASS"
        if pct <= 15:
            return "FAIR"
        if pct <= 30:
            return "POOR"
        return "FAIL"

    reliability = val.get("reliability", "measured")
    reliability_badge = ""
    if reliability == "reference_only":
        reliability_badge = " [REFERENCE ONLY — not validated]"

    lines = [
        "---",
        f"**Validation vs. Published Data**{reliability_badge}",
        f"*{val['source']}*",
        "",
        "| Quantity | Simulation | Published | Deviation |",
        "|----------|-----------|-----------|-----------|",
    ]

    I_ref_str = f"{val['I_peak_ref_MA']:.3f} MA"
    if u_I > 0:
        I_ref_str += f" (1s: {u_I*100:.0f}%)"
    lines.append(
        f"| I_peak | {val['I_peak_sim_MA']:.3f} MA | "
        f"{I_ref_str} | "
        f"{dI:.1f}% ({grade(dI)}) |"
    )
    lines.append(
        f"| t_peak | {val['t_peak_sim_us']:.1f} us | "
        f"{val['t_peak_ref_us']:.1f} us | "
        f"{dt:.1f}% ({grade(dt)}) |"
    )

    if "waveform_nrmse" in val:
        nrmse = val["waveform_nrmse"]
        wg = grade(nrmse * 100)
        lines.append(f"| I(t) NRMSE | {nrmse:.3f} | — | {nrmse*100:.1f}% ({wg}) |")

    if "Yn_sim" in val:
        bt_pct = val.get("bt_fraction", 0) * 100
        V_kV = val.get("V_pinch_kV", 0)
        yn_extra = f" ({bt_pct:.0f}% BT"
        if V_kV > 1:
            yn_extra += f", V={V_kV:.0f}kV"
        yn_extra += ")"
        lines.append(
            f"| Yn (D-D) | {val['Yn_sim']:.2e}{yn_extra} | "
            f"{val['Yn_ref']:.2e} | {val['Yn_log_ratio']:.1f} decades |"
        )

    # Explanation of WHY deviations exist
    lines.append("")
    lines.append("**Why do simulation and experiment differ?**")

    explanations = []
    if dI > 15:
        explanations.append(
            "**Current**: The Lee model uses two empirical fitting parameters "
            "(fc = current fraction, fm = mass fraction) that are calibrated to "
            "specific operating conditions. At different voltages or pressures, "
            "the optimal fc/fm change. Large I_peak deviations usually mean "
            "the mass loading (fm) is wrong for this condition."
        )
    elif dI > 5:
        explanations.append(
            "**Current**: Minor deviation is typical. The 0D Lee model assumes "
            "a uniform current sheath, but real DPF sheaths have non-uniform "
            "mass loading and filamentary structure."
        )

    if dt > 15:
        explanations.append(
            "**Timing**: The rise time depends strongly on external inductance (L0) "
            "and total circuit resistance. Published L0 values often exclude "
            "bus bar and connection inductance (typically 5-20 nH extra)."
        )

    if "Yn_log_ratio" in val and val["Yn_log_ratio"] > 0.5:
        explanations.append(
            "**Yield**: Neutron yield is extremely sensitive to pinch conditions "
            "(T ~ I^4 dependence). A 10% current error produces ~46% yield error. "
            "The beam-target mechanism (70-90% of yield) depends on pinch voltage "
            "V_pinch = (dL/dt)*I, which is hard to predict from 0D models."
        )

    if val.get("reliability") == "reference_only":
        explanations.append(
            "**Reference data**: This comparison uses model output (e.g. RADPF), "
            "not direct experimental measurement. Deviations reflect model "
            "differences, not physics errors."
        )

    if not explanations:
        explanations.append(
            "Good agreement. The Lee model captures the main circuit-plasma "
            "coupling physics. Remaining differences are from 3D effects "
            "(filaments, instabilities) that the 0D model cannot resolve."
        )

    for exp in explanations:
        lines.append(f"\n> {exp}")

    return "\n".join(lines)


def run_convergence_study(
    preset_name: str,
    backend: str = "metal_plm",
    sim_time_us: float = 15.0,
) -> dict[str, Any]:
    """Run the same simulation at 3 grid resolutions and compare I_peak.

    This checks whether the solution is grid-converged by measuring
    how much I_peak changes between coarse, medium, and fine grids.

    Usage::

        from app_validation import run_convergence_study
        results = run_convergence_study("pf1000", backend="metal_plm")
        print(f"Converged: {results['converged']} ({results['convergence_pct']:.1f}%)")

    Args:
        preset_name: Device preset (e.g. "pf1000", "nx2").
        backend: MHD backend to use.
        sim_time_us: Simulation duration in microseconds.

    Returns:
        Dict with per-grid results, convergence_pct, and converged flag.
        convergence_pct < 5% indicates grid convergence.
    """
    from app_mhd import MHD_GRID_PRESETS, run_mhd_simulation

    results: dict[str, Any] = {}
    for grid_name in ["coarse", "medium", "fine"]:
        r = run_mhd_simulation(
            backend=backend,
            grid_preset=grid_name,
            preset_name=preset_name,
            sim_time_us=sim_time_us,
        )
        results[grid_name] = {
            "grid": MHD_GRID_PRESETS[grid_name],
            "I_peak_MA": r["I_peak"],
            "n_steps": r["n_steps"],
        }

    I_coarse = results["coarse"]["I_peak_MA"]
    I_fine = results["fine"]["I_peak_MA"]
    convergence_pct = abs(I_coarse - I_fine) / max(abs(I_fine), 1e-10) * 100
    results["convergence_pct"] = convergence_pct
    results["converged"] = convergence_pct < 5.0

    return results
