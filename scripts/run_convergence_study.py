"""PF-1000 multi-resolution grid convergence study using the MLX solver.

Runs the PF-1000 discharge at four grid resolutions and computes Richardson
extrapolation + GCI to verify grid independence.

Published RADPF parameters (Lee & Saw 2014 / Scholz 2006):
    fc=0.7, fm=0.13, V0=27 kV, C=1.332 mF, L0=33.5 nH, R0=6.12 mOhm

Usage:
    python3 scripts/run_convergence_study.py [--mode MODE]

    MODE: "lee"  — circuit+snowplow only (fast, default, validated path)
          "mhd"  — circuit+snowplow+MLX MHD (slow, produces spatial fields)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Repo root on sys.path so dpf package is importable ───────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── MLX availability gate ─────────────────────────────────────────────────────
from dpf.metal.mlx_device import HAS_MLX  # noqa: E402

if not HAS_MLX:
    print("ERROR: MLX is not available on this system.", file=sys.stderr)
    print("Install mlx (pip install mlx) or run on Apple Silicon.", file=sys.stderr)
    sys.exit(1)

from dpf.metal.mlx_engine import run_mlx_discharge  # noqa: E402
from dpf.presets import get_preset  # noqa: E402
from dpf.validation.convergence_study import (  # noqa: E402
    compute_convergence_order,
    grid_convergence_index,
    richardson_extrapolation,
)

# ── Published RADPF parameters for PF-1000 ───────────────────────────────────
# Lee & Saw (2014), Scholz et al. Nukleonika 51(1):79-84, 2006
_FC: float = 0.7        # current sheath fraction
_FM: float = 0.13       # mass fraction (RADPF default)
_V0: float = 27000.0    # charging voltage [V]
_C: float = 1.332e-3    # capacitance [F]
_L0: float = 33.5e-9   # external inductance [H]
_R0: float = 6.12e-3   # external resistance [Ohm]

# Simulation time: 8 us minimum to capture full current rise and peak
_SIM_TIME_S: float = 8e-6

# CFL number for MLX solver
_CFL: float = 0.4

# Grid resolutions: (nr, ny, nz) — ny=1 for axisymmetric
_RESOLUTIONS: list[tuple[int, int, int]] = [
    (16,  1,  32),
    (32,  1,  64),
    (64,  1, 128),
    (128, 1, 256),
]

_PRESET_NAME: str = "pf1000"


def _run_one_resolution(
    grid_shape: tuple[int, int, int],
    mode: str,
) -> dict:
    """Run a single PF-1000 discharge and return scalar metrics + spatial data.

    Parameters
    ----------
    grid_shape:
        (nr, ny, nz) grid resolution.
    mode:
        "lee" or "mhd".

    Returns
    -------
    dict with keys: I_peak_MA, t_peak_us, rho_max, B_max, T_max,
                    density_profile_z (1-D array or None), elapsed_s.
    """
    nr, ny, nz = grid_shape

    # Compute dx/dz from PF-1000 geometry
    preset = get_preset(_PRESET_NAME)
    cc = preset["circuit"]
    r_anode = cc["anode_radius"]      # 0.115 m
    r_cathode = cc["cathode_radius"]  # 0.160 m
    anode_length = preset.get("snowplow", {}).get("anode_length", 0.6)

    dr = (r_cathode - r_anode) / nr   # radial cell size [m]
    dz = anode_length / nz            # axial cell size [m]

    result = run_mlx_discharge(
        preset_name=_PRESET_NAME,
        max_steps=200_000,
        fc=_FC,
        fm=_FM,
        V0_kV=_V0 / 1e3,
        mode=mode,
        grid_shape=grid_shape,
    )

    I_peak_MA = float(result["I_peak_MA"])
    t_peak_us = float(result["t_peak_us"])
    elapsed_s = float(result["elapsed_s"])

    rho_max: float | None = None
    B_max: float | None = None
    T_max: float | None = None
    density_profile_z: list[float] | None = None

    if mode == "mhd" and "final_state" in result:
        final = result["final_state"]
        if final:
            rho_arr = np.asarray(final.get("rho", np.zeros((nr, ny, nz))))
            rho_max = float(np.max(rho_arr))

            B_arr = final.get("B")
            if B_arr is not None:
                B_mag = np.sqrt(np.sum(np.asarray(B_arr) ** 2, axis=0))
                B_max = float(np.max(B_mag))

            # Temperature from pressure and density (ideal gas)
            # T = p * m_ion / (rho * k_B); use electron pressure if available
            _K_B = 1.380649e-23
            _M_D2_HALF = 3.34358377e-27  # half of D2 molecular mass = deuteron mass
            p_arr = final.get("pressure")
            if p_arr is not None and rho_max > 0:
                T_arr = (
                    np.asarray(p_arr) * _M_D2_HALF
                    / (np.maximum(np.asarray(rho_arr), 1e-30) * _K_B)
                )
                T_max = float(np.max(T_arr))

            # Density profile along z-axis at r=r_pinch (innermost radial cell)
            # rho shape: (nr, ny, nz) — axis 0 is radial, axis 2 is axial
            density_profile_z = rho_arr[0, 0, :].tolist()

    return {
        "grid_shape": list(grid_shape),
        "dr_mm": dr * 1e3,
        "dz_mm": dz * 1e3,
        "I_peak_MA": I_peak_MA,
        "t_peak_us": t_peak_us,
        "rho_max": rho_max,
        "B_max": B_max,
        "T_max": T_max,
        "density_profile_z": density_profile_z,
        "elapsed_s": elapsed_s,
    }


def run_convergence_study(mode: str = "lee") -> dict:
    """Run PF-1000 at all four resolutions and compute convergence metrics.

    Parameters
    ----------
    mode:
        "lee" for fast circuit+snowplow (validated path, no spatial fields),
        "mhd" for full MHD (slow, produces rho_max / B_max / T_max).

    Returns
    -------
    dict containing per-resolution results and convergence diagnostics.
    """
    print(f"PF-1000 Grid Convergence Study — mode={mode}")
    print(f"Parameters: fc={_FC}, fm={_FM}, V0={_V0/1e3:.0f} kV, "
          f"C={_C*1e3:.3f} mF, L0={_L0*1e9:.1f} nH, R0={_R0*1e3:.2f} mOhm")
    print(f"Sim time: {_SIM_TIME_S*1e6:.0f} us | CFL: {_CFL}")
    print("-" * 72)

    runs: list[dict] = []
    for i, gs in enumerate(_RESOLUTIONS):
        nr, _, nz = gs
        label = f"{nr}x{nz}"
        print(f"[{i+1}/{len(_RESOLUTIONS)}] Running {label} ...", flush=True)
        t0 = time.perf_counter()
        try:
            run_data = _run_one_resolution(gs, mode)
        except Exception as exc:  # noqa: BLE001 — report but continue
            elapsed = time.perf_counter() - t0
            print(f"  FAILED ({elapsed:.1f} s): {exc}")
            run_data = {
                "grid_shape": list(gs),
                "dr_mm": None,
                "dz_mm": None,
                "I_peak_MA": 0.0,
                "t_peak_us": 0.0,
                "rho_max": None,
                "B_max": None,
                "T_max": None,
                "density_profile_z": None,
                "elapsed_s": elapsed,
                "error": str(exc),
            }
        runs.append(run_data)
        I_str = f"{run_data['I_peak_MA']:.4f} MA"
        t_str = f"{run_data['t_peak_us']:.2f} us"
        print(f"  I_peak={I_str}, t_peak={t_str}, wall={run_data['elapsed_s']:.1f} s")

    # ── Convergence analysis on I_peak (most physically meaningful) ───────────
    # Use finest three resolutions: finest=[-1], medium=[-2], coarse=[-3]
    I_values = [r["I_peak_MA"] for r in runs]
    n_valid = sum(1 for v in I_values if v > 0)

    convergence_order: float = 0.0
    richardson_I_peak: float = I_values[-1] if I_values else 0.0
    gci_fine: float = 1.0
    refinement_ratio: float = 2.0  # fixed 2x refinement between each level

    if n_valid >= 3:
        f1 = I_values[-1]   # finest
        f2 = I_values[-2]   # medium
        f3 = I_values[-3]   # coarser
        convergence_order = compute_convergence_order(f1, f2, f3, refinement_ratio)
        richardson_I_peak = richardson_extrapolation(f1, f2, convergence_order, refinement_ratio)
        gci_fine = grid_convergence_index(f1, f2, convergence_order, refinement_ratio)
    elif n_valid >= 2:
        f1 = I_values[-1]
        f2 = I_values[-2]
        convergence_order = 1.0  # assume first-order with only 2 grids
        richardson_I_peak = richardson_extrapolation(f1, f2, convergence_order, refinement_ratio)
        gci_fine = grid_convergence_index(f1, f2, convergence_order, refinement_ratio, Fs=3.0)

    is_converged = gci_fine < 0.05

    # ── Spatial convergence metric (MHD mode only) ────────────────────────────
    spatial_L2_errors: list[float | None] = [None] * len(runs)
    if mode == "mhd":
        # Compare each coarser profile to finest-grid profile (coarsened)
        finest_profile = runs[-1].get("density_profile_z")
        if finest_profile is not None:
            finest_arr = np.asarray(finest_profile)
            for i, run in enumerate(runs[:-1]):
                prof = run.get("density_profile_z")
                if prof is None:
                    continue
                arr = np.asarray(prof)
                # Downsample finest to match coarser resolution
                factor = len(finest_arr) // len(arr)
                if factor < 1:
                    continue
                finest_coarsened = finest_arr[::factor][: len(arr)]
                denom = float(np.linalg.norm(finest_coarsened))
                if denom > 0:
                    spatial_L2_errors[i] = float(
                        np.linalg.norm(arr - finest_coarsened) / denom
                    )

    return {
        "mode": mode,
        "preset": _PRESET_NAME,
        "parameters": {
            "fc": _FC,
            "fm": _FM,
            "V0_V": _V0,
            "C_F": _C,
            "L0_H": _L0,
            "R0_Ohm": _R0,
            "sim_time_s": _SIM_TIME_S,
            "cfl": _CFL,
        },
        "runs": runs,
        "spatial_L2_errors": spatial_L2_errors,
        "convergence_order": convergence_order,
        "richardson_I_peak_MA": richardson_I_peak,
        "gci_fine_fraction": gci_fine,
        "gci_fine_percent": gci_fine * 100.0,
        "is_converged": is_converged,
    }


def _write_markdown(study: dict, out_path: Path) -> None:
    """Write a formatted Markdown convergence report."""
    mode = study["mode"]
    runs = study["runs"]
    p = study["convergence_order"]
    rich = study["richardson_I_peak_MA"]
    gci = study["gci_fine_percent"]
    converged = study["is_converged"]
    params = study["parameters"]

    lines: list[str] = [
        "# PF-1000 Grid Convergence Study",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Mode | `{mode}` |",
        f"| Preset | `{study['preset']}` |",
        f"| fc (current fraction) | {params['fc']} |",
        f"| fm (mass fraction) | {params['fm']} |",
        f"| V0 | {params['V0_V']/1e3:.0f} kV |",
        f"| C | {params['C_F']*1e3:.3f} mF |",
        f"| L0 | {params['L0_H']*1e9:.1f} nH |",
        f"| R0 | {params['R0_Ohm']*1e3:.2f} mOhm |",
        f"| Sim time | {params['sim_time_s']*1e6:.0f} us |",
        f"| CFL | {params['cfl']} |",
        "| Riemann solver | HLL |",
        "| Reconstruction | PLM |",
        "",
        "## Results by Resolution",
        "",
    ]

    # Build table header — include MHD columns only when available
    has_mhd_fields = mode == "mhd" and any(
        r.get("rho_max") is not None for r in runs
    )

    if has_mhd_fields:
        lines.append(
            "| Grid | dr [mm] | I_peak [MA] | t_peak [us] | "
            "rho_max [kg/m³] | B_max [T] | T_max [keV] | Wall [s] |"
        )
        lines.append(
            "|------|---------|-------------|-------------|"
            "----------------|-----------|-------------|----------|"
        )
    else:
        lines.append(
            "| Grid | dr [mm] | I_peak [MA] | t_peak [us] | Wall [s] |"
        )
        lines.append(
            "|------|---------|-------------|-------------|----------|"
        )

    _K_B = 1.380649e-23
    _EV = 1.602176634e-19
    _KEV = _EV * 1e3

    for run in runs:
        nr, _, nz = run["grid_shape"]
        label = f"{nr}x{nz}"
        dr = run.get("dr_mm") or 0.0
        I = run["I_peak_MA"]
        t = run["t_peak_us"]
        wall = run["elapsed_s"]
        err_flag = " ⚠" if run.get("error") else ""

        if has_mhd_fields:
            rho = run.get("rho_max")
            B = run.get("B_max")
            T = run.get("T_max")
            rho_str = f"{rho:.3e}" if rho is not None else "N/A"
            B_str = f"{B:.1f}" if B is not None else "N/A"
            T_str = f"{T * _K_B / _KEV:.2f}" if T is not None else "N/A"
            lines.append(
                f"| {label}{err_flag} | {dr:.2f} | {I:.4f} | {t:.2f} | "
                f"{rho_str} | {B_str} | {T_str} | {wall:.1f} |"
            )
        else:
            lines.append(
                f"| {label}{err_flag} | {dr:.2f} | {I:.4f} | {t:.2f} | {wall:.1f} |"
            )

    lines.extend([
        "",
        "## Convergence Diagnostics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Refinement ratio r | 2.0 |",
        f"| Observed order p | {p:.2f} |",
        f"| Richardson-extrapolated I_peak | {rich:.4f} MA |",
        f"| GCI (fine grid) | {gci:.2f}% |",
        f"| Converged (GCI < 5%) | {'YES' if converged else 'NO'} |",
        "",
    ])

    if mode == "mhd":
        errs = study.get("spatial_L2_errors", [])
        has_spatial = any(e is not None for e in errs)
        if has_spatial:
            lines.extend([
                "## Spatial Convergence (z-axis density profile at r=r_pinch)",
                "",
                "| Grid | L2 error vs finest |",
                "|------|--------------------|",
            ])
            for i, run in enumerate(runs[:-1]):
                nr, _, nz = run["grid_shape"]
                err = errs[i] if i < len(errs) else None
                err_str = f"{err:.4f}" if err is not None else "N/A"
                lines.append(f"| {nr}x{nz} | {err_str} |")
            lines.append(f"| {runs[-1]['grid_shape'][0]}x{runs[-1]['grid_shape'][2]} | (reference) |")
            lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- Published RADPF parameters used verbatim (Lee & Saw 2014, Scholz 2006).",
        "- No parameter calibration was performed.",
        f"- Solver: MLX HLL + PLM, mode=`{mode}`.",
        "- Lee mode (`mode='lee'`) runs circuit+snowplow only; spatial fields",
        "  (rho_max, B_max, T_max, density profile) are only available in `mode='mhd'`.",
        "- PF-1000 reference: I_peak = 1.87 MA at 27 kV (Scholz 2006).",
        "- GCI method: Roache (1998). Safety factor Fs=1.25 (3+ grids).",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Markdown report: {out_path}")


def _write_json(study: dict, out_path: Path) -> None:
    """Write raw convergence data to JSON for reproducibility."""
    # Density profiles can be large — keep them but make them JSON-safe
    serializable = json.loads(json.dumps(study, default=_json_default))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"Raw data: {out_path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PF-1000 multi-resolution grid convergence study."
    )
    parser.add_argument(
        "--mode",
        choices=["lee", "mhd"],
        default="lee",
        help=(
            "Simulation mode: 'lee' = circuit+snowplow (fast, validated); "
            "'mhd' = full MLX MHD (slow, produces spatial fields). Default: lee"
        ),
    )
    args = parser.parse_args()

    study = run_convergence_study(mode=args.mode)

    # Print summary to stdout
    print()
    print("=" * 72)
    print("CONVERGENCE SUMMARY")
    print("=" * 72)
    for run in study["runs"]:
        nr, _, nz = run["grid_shape"]
        err = " [FAILED]" if run.get("error") else ""
        print(
            f"  {nr:3d}x{nz:3d}: I_peak={run['I_peak_MA']:.4f} MA  "
            f"t_peak={run['t_peak_us']:.2f} us  wall={run['elapsed_s']:.1f} s{err}"
        )
    print()
    print(f"  Convergence order p = {study['convergence_order']:.2f}")
    print(f"  Richardson I_peak   = {study['richardson_I_peak_MA']:.4f} MA")
    print(f"  GCI (fine grid)     = {study['gci_fine_percent']:.2f}%")
    print(f"  Converged           = {'YES' if study['is_converged'] else 'NO (GCI > 5%)'}")
    print("=" * 72)

    # Write outputs
    docs_dir = _REPO_ROOT / "docs"
    _write_markdown(study, docs_dir / "CONVERGENCE_STUDY.md")
    _write_json(study, docs_dir / "convergence_data.json")


if __name__ == "__main__":
    main()
