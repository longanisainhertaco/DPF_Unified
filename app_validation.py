"""Experimental validation bridge for DPF web UI.

Wraps dpf.validation.experimental to compare simulation results against
published measurements. Maps preset names to device records.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PRESET_TO_DEVICE: dict[str, str] = {
    "pf1000": "PF-1000",
    "nx2": "NX2",
    "unu_ictp": "UNU-ICTP",
    "poseidon": "POSEIDON",
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

    sim_I_peak = data.get("I_pre_dip", data.get("I_peak", 0.0))
    sim_t_peak = data.get("t_pre_dip", data.get("t_peak", 0.0))

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

    return "\n".join(lines)
