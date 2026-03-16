#!/usr/bin/env python3
"""Run all DPF presets through the Lee model and report accuracy.

Generates a validation matrix showing I_peak, t_peak, and Yn for
each device, compared against published experimental data.

Usage:
    python3 scripts/validate_all_presets.py
    python3 scripts/validate_all_presets.py --preset pf1000
    python3 scripts/validate_all_presets.py --csv results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Published experimental data for validation
PUBLISHED_DATA: dict[str, dict[str, float]] = {
    "pf1000": {
        "I_peak_kA": 1870.0,  # Scholz 2006, 27 kV, 3.5 Torr D2
        "t_peak_us": 5.8,
        "Yn": 1e8,  # Order of magnitude
        "source": "Scholz et al., Nukleonika 51(1):79-84 (2006)",
    },
    "unu_ictp": {
        "I_peak_kA": 170.0,  # Lee et al. 1988
        "t_peak_us": 2.0,
        "Yn": 1e6,
        "source": "Lee et al., Am. J. Phys. 56:62 (1988)",
    },
    "faeton": {
        "I_peak_kA": 900.0,  # Damideh 2025
        "t_peak_us": 4.5,
        "Yn": 2.5e9,
        "source": "Damideh et al. (2025)",
    },
    "poseidon": {
        "I_peak_kA": 1500.0,  # Herold et al.
        "t_peak_us": 4.0,
        "Yn": 1e9,
        "source": "Herold et al., IPPLM",
    },
    "mjolnir": {
        "I_peak_kA": 3000.0,  # Goyon/Offermann
        "t_peak_us": 8.0,
        "Yn": 1e10,
        "source": "Goyon et al. (2025), Offermann et al. (2021)",
    },
    "pf400j": {
        "I_peak_kA": 123.0,  # Soto et al. 2009, base RLC amplitude
        "t_peak_us": 0.291,
        "Yn": 1e4,
        "source": "Soto et al., PSST 18:015007 (2009)",
    },
}


def run_preset(preset_name: str) -> dict:
    """Run a single preset through the Lee model."""
    from dpf.presets import get_preset
    from dpf.validation.lee_model_comparison import LeeModel

    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sc = preset.get("snowplow", {})

    model = LeeModel(
        fill_gas_mass=preset.get("ion_mass", 6.69e-27),
        current_fraction=sc.get("current_fraction", 0.7),
        mass_fraction=sc.get("mass_fraction", 0.15),
        radial_mass_fraction=sc.get("radial_mass_fraction"),
        pinch_column_fraction=sc.get("pinch_column_fraction", 1.0),
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    p_pa = sc.get("fill_pressure_Pa", 400.0)
    p_torr = p_pa / 133.322

    t0 = time.perf_counter()
    try:
        result = model.run(device_params={
            "C": cc["C"],
            "V0": cc["V0"],
            "L0": cc["L0"],
            "R0": cc.get("R0", 0.0),
            "anode_radius": cc["anode_radius"],
            "cathode_radius": cc["cathode_radius"],
            "anode_length": sc.get("anode_length", 0.16),
            "fill_pressure_torr": p_torr,
        })
        elapsed = time.perf_counter() - t0

        I_peak_kA = float(np.max(np.abs(result.I))) / 1e3
        t_peak_us = float(result.t[np.argmax(np.abs(result.I))]) * 1e6

        return {
            "preset": preset_name,
            "I_peak_kA": I_peak_kA,
            "t_peak_us": t_peak_us,
            "pinch_time_us": result.pinch_time * 1e6 if result.pinch_time else None,
            "phases": result.phases_completed,
            "elapsed_ms": elapsed * 1e3,
            "Yn": result.metadata.get("neutron_yield", 0.0),
            "error": None,
        }
    except Exception as exc:
        return {
            "preset": preset_name,
            "error": str(exc),
            "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        }


def format_table(results: list[dict]) -> str:
    """Format results as a markdown table."""
    lines = [
        "# DPF-Unified Validation Matrix",
        "",
        "| Device | I_peak [kA] | Published [kA] | Error | t_peak [us] | Published [us] | Phases | Time [ms] |",
        "|--------|-------------|----------------|-------|-------------|----------------|--------|-----------|",
    ]

    for r in results:
        if r.get("error"):
            lines.append(f"| {r['preset']} | ERROR | — | — | — | — | — | {r['elapsed_ms']:.0f} |")
            continue

        pub = PUBLISHED_DATA.get(r["preset"], {})
        pub_I = pub.get("I_peak_kA", 0)
        pub_t = pub.get("t_peak_us", 0)

        if pub_I > 0:
            I_err = abs(r["I_peak_kA"] - pub_I) / pub_I * 100
            I_err_str = f"{I_err:.1f}%"
        else:
            I_err_str = "N/A"

        phases_str = ",".join(str(p) for p in r.get("phases", []))

        lines.append(
            f"| {r['preset']} | {r['I_peak_kA']:.0f} | "
            f"{pub_I:.0f} | {I_err_str} | "
            f"{r['t_peak_us']:.3f} | {pub_t:.3f} | "
            f"{phases_str} | {r['elapsed_ms']:.0f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate all DPF presets")
    parser.add_argument("--preset", help="Run only this preset")
    parser.add_argument("--csv", help="Output CSV file")
    args = parser.parse_args()

    from dpf.presets import list_presets

    if args.preset:
        presets_to_run = [args.preset]
    else:
        # Run all real-device presets (skip tutorial, custom, cartesian, phase_p)
        skip = {"tutorial", "custom", "cartesian_demo", "phase_p_fidelity", "pf1000_akel"}
        all_presets = [p["name"] for p in list_presets()]
        presets_to_run = [p for p in all_presets if p not in skip]

    results = []
    for name in presets_to_run:
        print(f"Running {name}...", end=" ", flush=True)
        r = run_preset(name)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            print(f"I_peak={r['I_peak_kA']:.0f} kA, t_peak={r['t_peak_us']:.3f} us")
        results.append(r)

    print()
    print(format_table(results))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "preset", "I_peak_kA", "t_peak_us", "pinch_time_us",
                "phases", "elapsed_ms", "Yn", "error",
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
