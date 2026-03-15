"""Comparison mode and save/load configuration for DPF web UI."""
from __future__ import annotations

import json
import tempfile
from typing import Any

import gradio as gr
import numpy as np

from app_plots import create_comparison_fig

MAX_COMPARISON_RUNS = 8


def _next_seq(runs: list[dict]) -> int:
    if not runs:
        return 1
    return max(r.get("_seq", 0) for r in runs) + 1


def add_to_comparison(
    runs: list[dict], data: dict[str, Any], backend: str,
) -> list[dict]:
    runs = runs or []
    seq = _next_seq(runs)
    device = data.get("device", "?")
    snowplow = data.get("snowplow_params", {})
    fc = snowplow.get("current_fraction", data.get("circuit", {}).get("fc"))
    fm = snowplow.get("mass_fraction", data.get("circuit", {}).get("fm"))
    V0_kV = data.get("circuit", {}).get("V0", 0) / 1e3
    label = f"{device} {backend} #{seq}"
    run_entry = {
        "t_us": data["t_us"],
        "I_MA": data["I_MA"],
        "V_kV": data["V_kV"],
        "_label": label,
        "_seq": seq,
        "_backend": backend,
        "_V0_kV": V0_kV,
        "_fc": fc,
        "_fm": fm,
    }
    runs = runs[-MAX_COMPARISON_RUNS + 1:] + [run_entry]
    return runs


def comparison_summary(runs: list[dict]) -> str:
    if len(runs) < 2:
        return "*Run at least 2 simulations to see comparison metrics.*"
    lines = [
        "| # | Run | Backend | V0 [kV] | fc | fm | I_peak [MA] | t_peak [us] |",
        "|---|-----|---------|---------|----|----|------------|------------|",
    ]
    for i, r in enumerate(runs):
        I_arr = np.array(r["I_MA"])
        pk = int(np.argmax(np.abs(I_arr)))
        fc = f"{r['_fc']:.3f}" if r.get("_fc") is not None else "-"
        fm = f"{r['_fm']:.3f}" if r.get("_fm") is not None else "-"
        V0 = f"{r.get('_V0_kV', 0):.0f}"
        lines.append(
            f"| {i} | {r['_label']} | {r.get('_backend', '?')} | {V0} "
            f"| {fc} | {fm} | {abs(I_arr[pk]):.3f} | {r['t_us'][pk]:.1f} |"
        )
    lines.append("")
    lines.append("*Enter the **#** index above to remove a specific run.*")
    return "\n".join(lines)


def remove_from_comparison(runs: list[dict], index: int) -> tuple[list[dict], Any, str]:
    if not runs or index < 0 or index >= len(runs):
        return runs, create_comparison_fig(runs), comparison_summary(runs)
    label = runs[index].get("_label", f"Run {index + 1}")
    runs = runs[:index] + runs[index + 1:]
    fig = create_comparison_fig(runs)
    md = comparison_summary(runs) if runs else f"*Removed {label}. No runs remaining.*"
    return runs, fig, md


def clear_comparison():
    fig = create_comparison_fig([])
    return [], fig, "*Comparison cleared.*"


def save_config(
    backend, grid_preset, preset_name, sim_time_us, gas_key,
    V0_kV, C_uF, L0_nH, R0_mOhm,
    anode_r, cathode_r, anode_len,
    fc, fm, crowbar_on, crowbar_R, pressure,
) -> str:
    cfg = {
        "backend": backend, "grid_preset": grid_preset,
        "preset_name": preset_name, "sim_time_us": sim_time_us,
        "gas_key": gas_key,
        "V0_kV": V0_kV, "C_uF": C_uF, "L0_nH": L0_nH, "R0_mOhm": R0_mOhm,
        "anode_r_mm": anode_r, "cathode_r_mm": cathode_r,
        "anode_len_mm": anode_len,
        "fc": fc, "fm": fm, "crowbar_on": crowbar_on,
        "crowbar_R_mOhm": crowbar_R, "pressure_torr": pressure,
    }
    import os
    temp_dir = os.environ.get("DPF_TEMP_DIR", tempfile.gettempdir())
    path = os.path.join(temp_dir, f"dpf_config_{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def load_config(file_obj):
    if file_obj is None:
        raise gr.Error("No file selected.")
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    with open(path) as f:
        cfg = json.load(f)
    return [
        cfg.get("backend", "lee"),
        cfg.get("grid_preset", "medium"),
        cfg.get("preset_name", "pf1000"),
        cfg.get("sim_time_us", 40),
        cfg.get("gas_key", "D2"),
        cfg.get("V0_kV", 27),
        cfg.get("C_uF", 1332),
        cfg.get("L0_nH", 33.5),
        cfg.get("R0_mOhm", 2.3),
        cfg.get("anode_r_mm", 115),
        cfg.get("cathode_r_mm", 160),
        cfg.get("anode_len_mm", 600),
        cfg.get("fc", 0.8),
        cfg.get("fm", 0.094),
        cfg.get("crowbar_on", True),
        cfg.get("crowbar_R_mOhm", 1.5),
        cfg.get("pressure_torr"),
    ]
