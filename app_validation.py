"""Experimental validation data for DPF device presets.

Published reference values for I_peak, t_peak, and neutron yield from the
literature cited in each preset. Used to compute % deviation of simulation
results from experimental measurements.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PUBLISHED_DATA: dict[str, dict[str, Any]] = {
    "pf1000": {
        "I_peak_MA": 1.87,
        "t_peak_us": 5.5,
        "source": "Scholz et al., Nukleonika 51(1):79-84 (2006), 27 kV D2",
        "notes": "I_peak from short-circuit-calibrated Lee model fit at 27 kV",
        "neutron_yield": None,
        "extra": {
            "16kV_I_peak_MA": 1.24,
            "16kV_source": "Akel et al., Rad. Phys. Chem. 188:109633 (2021), avg of 24 shots",
            "16kV_neutron_yield_range": (1.7e8, 1.11e10),
        },
    },
    "nx2": {
        "I_peak_MA": 0.40,
        "t_peak_us": 0.38,
        "source": "Lee & Saw, J. Fusion Energy 27:292 (2008), 11.5 kV D2",
        "neutron_yield": 1e6,
        "neutron_yield_range": (5e5, 5e6),
        "notes": "Compact 1.85 kJ device, Yn from Lee et al. J. Appl. Phys. 106 (2009)",
    },
    "unu_ictp": {
        "I_peak_MA": 0.170,
        "t_peak_us": 2.8,
        "source": "Lee et al., Am. J. Phys. 56:62 (1988), 14 kV D2",
        "neutron_yield": 1e8,
        "neutron_yield_range": (5e7, 3e8),
        "notes": "3 kJ PFF, Yn from Lee (2014) Review at 3 Torr D2",
    },
    "llnl_dpf": {
        "I_peak_MA": 0.25,
        "t_peak_us": 0.8,
        "source": "Deutsch & Kies, Plasma Phys. Control. Fusion 30:263 (1988)",
        "neutron_yield": None,
        "notes": "4 kJ compact diagnostic device",
    },
    "poseidon": {
        "I_peak_MA": 1.5,
        "t_peak_us": 3.5,
        "source": "Herold et al., Nucl. Fusion 29:33 (1989), 40 kV D2",
        "neutron_yield": 1e11,
        "neutron_yield_range": (5e10, 3e11),
        "notes": "480 kJ MA-class, Yn at 3.5 Torr D2",
    },
    "poseidon_60kv": {
        "I_peak_MA": 2.0,
        "t_peak_us": 2.5,
        "source": "IPFS (plasmafocus.net), 60 kV D2, Lee model digitized I(t)",
        "neutron_yield": None,
        "notes": "280.8 kJ at 60 kV, IPFS fitted parameters",
    },
    "mjolnir": {
        "I_peak_MA": 2.8,
        "t_peak_us": 4.0,
        "source": "Schmidt et al., IEEE TPS (2021); Goyon et al., Phys. Plasmas 32:033105 (2025)",
        "neutron_yield": None,
        "notes": "2 MJ at 60 kV, 84-cable transmission line, I_peak from Goyon Fig. 3",
    },
    "faeton": {
        "I_peak_MA": 0.93,
        "t_peak_us": 3.0,
        "source": "Damideh et al., Sci. Rep. 15:23048 (2025), 100 kV D2",
        "neutron_yield": 1e10,
        "neutron_yield_range": (5e9, 5e10),
        "notes": "125 kJ, Yn from Damideh Fig. 5 at 12 Torr D2",
    },
}


def validate_against_published(
    data: dict[str, Any],
    preset_name: str,
) -> dict[str, Any] | None:
    """Compare simulation results to published experimental data.

    Returns dict with deviations and pass/fail status, or None if no
    published data exists for the preset.
    """
    ref = PUBLISHED_DATA.get(preset_name)
    if ref is None:
        return None

    sim_I_peak = data.get("I_peak", 0.0)
    sim_t_peak = data.get("t_peak", 0.0)

    ref_I = ref["I_peak_MA"]
    ref_t = ref["t_peak_us"]

    dI_pct = abs(sim_I_peak - ref_I) / ref_I * 100 if ref_I > 0 else float("inf")
    dt_pct = abs(sim_t_peak - ref_t) / ref_t * 100 if ref_t > 0 else float("inf")

    result: dict[str, Any] = {
        "preset": preset_name,
        "source": ref["source"],
        "notes": ref.get("notes", ""),
        "I_peak_sim_MA": sim_I_peak,
        "I_peak_ref_MA": ref_I,
        "I_peak_dev_pct": dI_pct,
        "t_peak_sim_us": sim_t_peak,
        "t_peak_ref_us": ref_t,
        "t_peak_dev_pct": dt_pct,
    }

    ny = data.get("neutron_yield")
    if ny and ref.get("neutron_yield") is not None:
        sim_yn = ny.get("Y_neutron", 0)
        ref_yn = ref["neutron_yield"]
        if ref_yn > 0 and sim_yn > 0:
            log_ratio = abs(np.log10(sim_yn / ref_yn))
            result["Yn_sim"] = sim_yn
            result["Yn_ref"] = ref_yn
            result["Yn_log_ratio"] = log_ratio
            yn_range = ref.get("neutron_yield_range")
            if yn_range:
                result["Yn_in_range"] = yn_range[0] <= sim_yn <= yn_range[1]

    return result


def format_validation_markdown(val: dict[str, Any] | None) -> str:
    """Format validation result as markdown for display in the UI."""
    if val is None:
        return ""

    dI = val["I_peak_dev_pct"]
    dt = val["t_peak_dev_pct"]

    def grade(pct: float) -> str:
        if pct <= 5:
            return "PASS"
        if pct <= 15:
            return "FAIR"
        if pct <= 30:
            return "POOR"
        return "FAIL"

    lines = [
        "---",
        "**Validation vs. Published Data**",
        f"*{val['source']}*",
        "",
        "| Quantity | Simulation | Published | Deviation |",
        "|----------|-----------|-----------|-----------|",
        (
            f"| I_peak | {val['I_peak_sim_MA']:.3f} MA | "
            f"{val['I_peak_ref_MA']:.3f} MA | "
            f"{dI:.1f}% ({grade(dI)}) |"
        ),
        (
            f"| t_peak | {val['t_peak_sim_us']:.1f} us | "
            f"{val['t_peak_ref_us']:.1f} us | "
            f"{dt:.1f}% ({grade(dt)}) |"
        ),
    ]

    if "Yn_sim" in val:
        yn_s = val["Yn_sim"]
        yn_r = val["Yn_ref"]
        lr = val["Yn_log_ratio"]
        in_range = val.get("Yn_in_range")
        range_str = " (in range)" if in_range else " (out of range)" if in_range is not None else ""
        lines.append(
            f"| Yn (D-D) | {yn_s:.2e} | {yn_r:.2e} | "
            f"{lr:.1f} decades{range_str} |"
        )

    if val.get("notes"):
        lines.append(f"\n*{val['notes']}*")

    return "\n".join(lines)
