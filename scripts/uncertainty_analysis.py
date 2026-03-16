"""Sobol sensitivity analysis on the Lee model (PF-1000 preset).

Outputs:
    docs/research-reference/uncertainty_analysis.json
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np

# Resolve project root so script runs from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from SALib.sample.sobol import sample as sobol_sample  # SALib >= 1.5
except ImportError:
    from SALib.sample.saltelli import sample as sobol_sample  # SALib < 1.5
from SALib.analyze import sobol

from app_engine import run_simulation_core

PROBLEM: dict = {
    "num_vars": 4,
    "names": ["fc", "fm", "pressure_torr", "V0_kV"],
    "bounds": [[0.5, 0.9], [0.05, 0.25], [1.0, 8.0], [20.0, 35.0]],
}

N_SALTELLI = 256  # total runs = N*(2*num_vars+2) = 256*10 = 2560
SIM_TIME_US = 40.0
PRESET = "pf1000"
OUTPUT_JSON = ROOT / "docs/research-reference/uncertainty_analysis.json"


def run_batch(param_values: np.ndarray) -> dict[str, np.ndarray]:
    """Run all parameter combinations; return dict of output arrays."""
    n = len(param_values)
    Y_ipeak = np.zeros(n)
    Y_dip = np.zeros(n)
    Y_yn = np.zeros(n)

    t0 = time.perf_counter()
    failed = 0

    for i, params in enumerate(param_values):
        fc, fm, pressure, V0 = params
        try:
            result = run_simulation_core(
                preset_name=PRESET,
                sim_time_us=SIM_TIME_US,
                fc=fc,
                fm=fm,
                pressure_torr=pressure,
                V0_kV=V0,
            )
            Y_ipeak[i] = result["I_peak"]
            Y_dip[i] = result["dip_pct"]
            yn_dict = result.get("neutron_yield")
            Y_yn[i] = yn_dict["Y_neutron"] if yn_dict else 0.0
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Run {i} failed ({params}): {exc}")
            Y_ipeak[i] = np.nan
            Y_dip[i] = np.nan
            Y_yn[i] = np.nan
            failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  failures={failed}")

    total = time.perf_counter() - t0
    print(f"Completed {n} runs in {total:.1f}s ({total/n:.3f}s/run), failures={failed}")
    return {"I_peak": Y_ipeak, "dip_pct": Y_dip, "Y_neutron": Y_yn}


def analyze_output(
    Y: np.ndarray, label: str
) -> dict:
    """Run Sobol analysis on a single output array (NaN-safe via masking)."""
    mask = np.isfinite(Y)
    n_valid = mask.sum()
    n_total = len(Y)

    if n_valid < 0.9 * n_total:
        warnings.warn(f"{label}: only {n_valid}/{n_total} valid runs — results may be unreliable")

    # SALib requires a full 2D sample; replace NaN with median for analysis
    Y_clean = Y.copy()
    if not np.all(mask):
        Y_clean[~mask] = np.nanmedian(Y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol.analyze(PROBLEM, Y_clean, calc_second_order=False, print_to_console=False)

    return {
        "S1": Si["S1"].tolist(),
        "S1_conf": Si["S1_conf"].tolist(),
        "ST": Si["ST"].tolist(),
        "ST_conf": Si["ST_conf"].tolist(),
        "n_valid": int(n_valid),
        "n_total": int(n_total),
        "mean": float(np.nanmean(Y)),
        "std": float(np.nanstd(Y)),
        "min": float(np.nanmin(Y)),
        "max": float(np.nanmax(Y)),
    }


def rank_parameters(Si_dict: dict) -> list[str]:
    """Return parameter names sorted by total-order index (descending)."""
    names = PROBLEM["names"]
    return [names[i] for i in np.argsort(Si_dict["ST"])[::-1]]


def main() -> None:
    print(f"Sobol UQ: N={N_SALTELLI}, total_runs={N_SALTELLI * (2 * PROBLEM['num_vars'] + 2)}")
    print(f"Preset: {PRESET}, sim_time={SIM_TIME_US} µs")

    param_values = sobol_sample(PROBLEM, N_SALTELLI, calc_second_order=False)
    print(f"Generated {len(param_values)} parameter samples")

    outputs = run_batch(param_values)

    results: dict = {
        "problem": PROBLEM,
        "N_saltelli": N_SALTELLI,
        "preset": PRESET,
        "sim_time_us": SIM_TIME_US,
        "outputs": {},
    }

    print("\n=== Sobol Sensitivity Indices ===")
    for label, Y in outputs.items():
        Si = analyze_output(Y, label)
        results["outputs"][label] = Si

        print(f"\n{label}  (mean={Si['mean']:.4g}, std={Si['std']:.4g})")
        print(f"  {'Parameter':<20} {'S1':>8} {'±':>6} {'ST':>8} {'±':>6}")
        for k, name in enumerate(PROBLEM["names"]):
            print(
                f"  {name:<20} {Si['S1'][k]:>8.4f} {Si['S1_conf'][k]:>6.4f}"
                f" {Si['ST'][k]:>8.4f} {Si['ST_conf'][k]:>6.4f}"
            )

        ranked = rank_parameters(Si)
        print(f"  Ranked by total-order: {ranked}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
