"""ShinkaEvolve evaluator for DPF MHD solver.

Called by ShinkaEvolve as:
  python evaluate.py --program_path <path/to/main.py> --results_dir <path>
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/Users/anthonyzamora/dpf-unified/src")

import numpy as np

from shinka.core.wrap_eval import run_shinka_eval

REF_PATH = Path("/Users/anthonyzamora/dpf-unified/tests/reference_data/radpf_pf1000_27kv.json")
with open(REF_PATH) as f:
    _REF = json.load(f)


def validate_result(result: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(result, dict):
        return False, f"Expected dict, got {type(result)}"
    if "I_peak_MA" not in result:
        return False, f"Missing I_peak_MA. Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
    ipeak = result["I_peak_MA"]
    if not (0 < ipeak < 100):
        return False, f"I_peak_MA={ipeak} unphysical"
    # Reject truncated simulations — discharge must complete to sim_time.
    # If t_peak equals the last timestep, the current was still rising when
    # the simulation ended. This prevents gaming via truncation.
    t_peak = result.get("t_peak_us", 0)
    sim_time_us = 12.0  # MHD mode uses 12 us (20% margin for late t_peak)
    if t_peak >= sim_time_us * 0.99:
        return False, f"Simulation truncated: t_peak={t_peak:.2f} us = sim_time (current never peaked)"
    return True, None


def aggregate(results: List[Any]) -> Dict[str, Any]:
    if not results or not isinstance(results[0], dict):
        return {"fitness": 0.0}
    result = results[0]

    I_mhd = abs(result.get("I_peak_MA", 0))
    I_ref = _REF["scalars"]["I_peak_MA"]
    t_mhd = result.get("t_peak_us", 0)
    t_ref = _REF["scalars"]["t_peak_us"]
    Lp_mhd = float(np.max(result.get("Lp_mhd_nH", [0])))
    Lp_ref = _REF["scalars"]["Lp_max_nH"]

    I_arr = np.asarray(result.get("I_MA", []))
    I_ref_arr = np.asarray(_REF["time_series"]["I_kA"]) * 1e-3
    l2_mhd = float(np.sqrt(np.mean(I_arr**2))) if len(I_arr) > 0 else 0
    l2_ref = float(np.sqrt(np.mean(I_ref_arr**2)))

    err_ipeak = abs(I_mhd - I_ref) / max(I_ref, 1e-30)
    err_tpeak = abs(t_mhd - t_ref) / max(t_ref, 1e-30)
    err_l2 = abs(l2_mhd - l2_ref) / max(l2_ref, 1e-30)
    err_lp = abs(Lp_mhd - Lp_ref) / max(Lp_ref, 1e-30)

    # Equal weighting — no paper justifies preferring one metric over another.
    # Lp_mhd excluded until voltage-flux coupling is implemented.
    # Three circuit-observable metrics: I_peak, t_peak, waveform RMS.
    fitness = max(0.0, 1.0 - (err_ipeak + err_tpeak + err_l2) / 3.0)

    return {
        "combined_score": fitness,
        "fitness": fitness,
        "err_ipeak": err_ipeak,
        "err_tpeak": err_tpeak,
        "err_waveform_l2": err_l2,
        "err_lp_max": err_lp,
        "I_peak_MA": I_mhd,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    run_shinka_eval(
        program_path=args.program_path,
        results_dir=args.results_dir,
        experiment_fn_name="run_mhd_discharge",
        num_runs=1,
        validate_fn=validate_result,
        aggregate_metrics_fn=aggregate,
    )
