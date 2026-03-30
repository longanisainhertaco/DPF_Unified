#!/usr/bin/env python3
"""Analyze MHD sweep results: device comparison, scaling laws, sensitivity.

Reads training/mhd_sweep_results/*_mhd.json and produces a comprehensive
analytics report at training/mhd_sweep_results/analysis_report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

OUT_DIR = _root / "training" / "mhd_sweep_results"


def load_all_shots() -> dict[str, list[dict]]:
    """Load all device shot data."""
    devices = {}
    for f in sorted(OUT_DIR.glob("*_mhd.json")):
        if f.name in ("manifest.json", "mhd_sweep_summary.csv"):
            continue
        data = json.loads(f.read_text())
        ok = [s for s in data if s.get("wall_s", 0) > 1.0 and s.get("I_peak_MA", 0) > 0.001]
        if ok:
            dev = f.stem.replace("_10shots_mhd", "")
            devices[dev] = ok
    return devices


def device_summary(devices: dict) -> str:
    """Per-device statistics table."""
    lines = [
        "## 1. Device Summary (Full MHD, MLX Metal GPU)\n",
        "| Device | Shots | Avg s | I_peak (MA) | Std | Min | Max | ms/step |",
        "|--------|-------|-------|-------------|-----|-----|-----|---------|",
    ]
    for dev, shots in devices.items():
        t = [s["wall_s"] for s in shots]
        I = [s["I_peak_MA"] for s in shots]
        steps = [s.get("n_steps", 1) for s in shots]
        ms = np.mean([ti / max(ni, 1) * 1000 for ti, ni in zip(t, steps)])
        lines.append(
            f"| {dev} | {len(shots)} | {np.mean(t):.1f} | "
            f"{np.mean(I):.3f} | {np.std(I):.3f} | {np.min(I):.3f} | {np.max(I):.3f} | {ms:.1f} |"
        )
    return "\n".join(lines) + "\n"


def scaling_analysis(devices: dict) -> str:
    """I_peak vs V0 scaling, energy scaling."""
    lines = [
        "\n## 2. Scaling Analysis\n",
        "### I_peak vs V0 (per device)\n",
        "| Device | V0 range (kV) | I_peak range (MA) | Correlation r | Scaling exponent |",
        "|--------|--------------|-------------------|---------------|-----------------|",
    ]
    for dev, shots in devices.items():
        V0 = np.array([s["V0_kV"] for s in shots])
        I = np.array([s["I_peak_MA"] for s in shots])
        if len(V0) < 3 or np.std(V0) < 0.1:
            continue
        r = np.corrcoef(V0, I)[0, 1] if np.std(I) > 0 else 0
        # log-log fit for scaling exponent: I ~ V0^alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            logV = np.log(V0)
            logI = np.log(np.maximum(I, 1e-10))
            if np.std(logV) > 0:
                alpha = np.polyfit(logV, logI, 1)[0]
            else:
                alpha = 0
        lines.append(
            f"| {dev} | {V0.min():.0f}-{V0.max():.0f} | "
            f"{I.min():.3f}-{I.max():.3f} | {r:.3f} | {alpha:.2f} |"
        )

    lines.extend([
        "\n### Energy Stored vs I_peak (cross-device)\n",
        "| Device | E_stored (kJ) | I_peak (MA) | I/sqrt(E) |",
        "|--------|--------------|-------------|-----------|",
    ])
    for dev, shots in devices.items():
        from dpf.presets import get_preset
        try:
            p = get_preset(dev)
            C = p["circuit"]["C"]
            V0_nom = p["circuit"]["V0"]
            E_kJ = 0.5 * C * V0_nom**2 / 1e3
            I_mean = np.mean([s["I_peak_MA"] for s in shots])
            ratio = I_mean / np.sqrt(E_kJ) if E_kJ > 0 else 0
            lines.append(f"| {dev} | {E_kJ:.1f} | {I_mean:.3f} | {ratio:.3f} |")
        except Exception:
            pass

    return "\n".join(lines) + "\n"


def sensitivity_analysis(devices: dict) -> str:
    """Parameter sensitivity: which input affects I_peak most."""
    lines = [
        "\n## 3. Parameter Sensitivity (|correlation| with I_peak)\n",
        "| Device | V0 | Pressure | fc | fm | Dominant |",
        "|--------|-----|----------|-----|-----|----------|",
    ]
    for dev, shots in devices.items():
        if len(shots) < 5:
            continue
        I = np.array([s["I_peak_MA"] for s in shots])
        V0 = np.array([s["V0_kV"] for s in shots])
        P = np.array([s["pressure_torr"] for s in shots])
        fc = np.array([s["fc"] for s in shots])
        fm = np.array([s["fm"] for s in shots])

        corrs = {}
        for name, x in [("V0", V0), ("P", P), ("fc", fc), ("fm", fm)]:
            if np.std(x) > 0 and np.std(I) > 0:
                corrs[name] = abs(np.corrcoef(x, I)[0, 1])
            else:
                corrs[name] = 0.0

        dominant = max(corrs, key=corrs.get)
        lines.append(
            f"| {dev} | {corrs['V0']:.2f} | {corrs['P']:.2f} | "
            f"{corrs['fc']:.2f} | {corrs['fm']:.2f} | **{dominant}** |"
        )

    return "\n".join(lines) + "\n"


def performance_analysis(devices: dict) -> str:
    """Compute performance breakdown."""
    lines = [
        "\n## 4. Compute Performance (M3 Pro, Metal GPU)\n",
        "| Device | Steps | Wall (s) | ms/step | Throughput (zone-cycles/s) |",
        "|--------|-------|----------|---------|--------------------------|",
    ]
    grid_cells = 32 * 1 * 64  # 2048 cells

    for dev, shots in devices.items():
        steps_arr = [s.get("n_steps", 0) for s in shots]
        wall_arr = [s["wall_s"] for s in shots]
        avg_steps = np.mean(steps_arr)
        avg_wall = np.mean(wall_arr)
        ms_step = avg_wall / max(avg_steps, 1) * 1000
        zc_per_s = grid_cells * avg_steps / max(avg_wall, 0.01)
        lines.append(
            f"| {dev} | {avg_steps:.0f} | {avg_wall:.1f} | "
            f"{ms_step:.1f} | {zc_per_s:.0e} |"
        )

    return "\n".join(lines) + "\n"


def mhd_vs_lee(devices: dict) -> str:
    """Compare MHD results to Lee model baseline."""
    lines = [
        "\n## 5. MHD vs Lee Model Comparison\n",
        "The MHD solver resolves spatial structure (sheath, B-field, pressure)\n"
        "that the Lee model lumps into empirical parameters. Systematic offsets\n"
        "quantify the value of spatial resolution.\n",
        "| Device | I_peak MHD (MA) | I_peak Lee (MA) | Offset | Interpretation |",
        "|--------|----------------|----------------|--------|----------------|",
    ]

    # Lee model reference from last validation
    lee_ref = {
        "pf1000": 1.794, "pf1000_akel": 1.558, "pf1000_20kv": 1.330,
        "nx2": 0.346, "unu_ictp": 0.159, "llnl_dpf": 0.168,
        "mjolnir": 2.687, "faeton": 0.964, "poseidon": 2.610,
        "poseidon_60kv": 3.175, "aecs_pf2": 0.142, "pf400j": 0.124,
    }

    for dev, shots in devices.items():
        I_mhd = np.mean([s["I_peak_MA"] for s in shots])
        I_lee = lee_ref.get(dev, 0)
        if I_lee > 0:
            offset = (I_mhd - I_lee) / I_lee * 100
            interp = "MHD higher" if offset > 5 else ("MHD lower" if offset < -5 else "~agree")
            lines.append(
                f"| {dev} | {I_mhd:.3f} | {I_lee:.3f} | {offset:+.1f}% | {interp} |"
            )

    return "\n".join(lines) + "\n"


def variability_analysis(devices: dict) -> str:
    """Shot-to-shot variability from randomized parameters."""
    lines = [
        "\n## 6. Shot-to-Shot Variability\n",
        "Standard deviation from 10 randomized shots quantifies parameter sensitivity.\n"
        "Devices with high CoV are sensitive to operating conditions.\n",
        "| Device | I_peak Mean | Std | CoV (%) | Interpretation |",
        "|--------|------------|-----|---------|----------------|",
    ]
    for dev, shots in devices.items():
        I = np.array([s["I_peak_MA"] for s in shots])
        mean = np.mean(I)
        std = np.std(I)
        cov = std / mean * 100 if mean > 0 else 0
        interp = "stable" if cov < 5 else ("moderate" if cov < 10 else "sensitive")
        lines.append(f"| {dev} | {mean:.3f} | {std:.3f} | {cov:.1f} | {interp} |")

    return "\n".join(lines) + "\n"


def main():
    devices = load_all_shots()
    total = sum(len(v) for v in devices.values())
    print(f"Loaded {total} shots from {len(devices)} devices")

    report = [
        "# DPF-Unified MHD Sweep Analysis Report\n",
        f"**Date**: 2026-03-29\n"
        f"**Platform**: M3 Pro MacBook Pro, 36GB, Metal GPU\n"
        f"**Solver**: MLX HLLS + PLM + SSP-RK2, 32x64 cylindrical\n"
        f"**Physics**: Spitzer resistivity, anomalous R, Braginskii transport, bremsstrahlung\n"
        f"**Shots**: {total} ({len(devices)} devices x 10 shots, randomized V0/P/fc/fm)\n",
    ]

    report.append(device_summary(devices))
    report.append(scaling_analysis(devices))
    report.append(sensitivity_analysis(devices))
    report.append(performance_analysis(devices))
    report.append(mhd_vs_lee(devices))
    report.append(variability_analysis(devices))

    report_text = "\n".join(report)
    out_path = OUT_DIR / "analysis_report.md"
    out_path.write_text(report_text)
    print(f"Report written to {out_path}")
    print(report_text)


if __name__ == "__main__":
    main()
