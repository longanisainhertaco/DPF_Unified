#!/usr/bin/env python3
"""PIRT (Phenomena Identification and Ranking Table) traversal for MHD diagnosis.

Given extracted scalars from a simulation, walks the causal chain of DPF physics
to identify the FIRST point of failure. Outputs a diagnosis with recommended fix.

Usage: python3 tests/pirt_traverse.py scalars.json [--radpf reference.json]

PIRT Chain (causal order):
  CFL dt → sheath detection (z_sheath) → r_eff → L_p → dL/dt → back-EMF → I_peak → t_peak
"""
from __future__ import annotations

import argparse
import json
import sys


def diagnose(scalars: dict, radpf: dict | None = None) -> list[str]:
    """Walk the PIRT causal chain and report first failure."""
    findings: list[str] = []

    # 1. CFL timestep
    dt_min = scalars.get("dt_min_s", 1.0)
    if dt_min < 1e-12:
        findings.append(
            "CRITICAL: dt_min = {:.2e} s — CFL collapsed. Vacuum v_Alfven spike. "
            "FIX: Mask vacuum cells (rho < 1e-4 * rho_max) from CFL.".format(dt_min)
        )
        return findings  # stop here — nothing downstream is meaningful
    elif dt_min < 1e-10:
        findings.append(f"WARNING: dt_min = {dt_min:.2e} s — very small. Check vacuum treatment.")

    # 2. Energy conservation
    energy_err = scalars.get("energy_conservation_error", 0)
    if energy_err > 0.1:
        findings.append(
            f"WARNING: Energy conservation error = {energy_err:.1%}. "
            "Source terms or BCs leaking energy. Check operator splitting."
        )

    # 3. Compare to RADPF if available
    if radpf:
        # I_peak
        I_sim = scalars.get("I_peak_A", 0)
        I_ref = radpf.get("I_peak_A", 0)
        if I_ref > 0 and I_sim > 0:
            err = abs(I_sim - I_ref) / I_ref
            if err > 0.10:
                findings.append(
                    f"FAIL: I_peak = {I_sim/1e6:.3f} MA vs RADPF {I_ref/1e6:.3f} MA ({err:.1%} error). "
                    "Cause: L_p too high (r_eff too small) or too low (r_eff too large)."
                )
            else:
                findings.append(f"OK: I_peak within {err:.1%} of RADPF.")

        # t_peak
        t_sim = scalars.get("t_peak_s", 0)
        t_ref = radpf.get("t_peak_s", 0)
        if t_ref > 0 and t_sim > 0:
            err = abs(t_sim - t_ref) / t_ref
            if err > 0.15:
                findings.append(
                    f"FAIL: t_peak = {t_sim*1e6:.2f} us vs RADPF {t_ref*1e6:.2f} us ({err:.1%} error). "
                    "Cause: sheath velocity too slow (fm too low) or axial extent wrong."
                )
            else:
                findings.append(f"OK: t_peak within {err:.1%} of RADPF.")

    if not findings:
        findings.append("No issues detected. Proceed to multi-angle acceptance test.")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="PIRT diagnosis of MHD discharge")
    parser.add_argument("scalars_path", help="Path to scalars.json")
    parser.add_argument("--radpf", default=None, help="Path to RADPF reference scalars.json")
    args = parser.parse_args()

    with open(args.scalars_path) as f:
        scalars = json.load(f)

    radpf = None
    if args.radpf:
        with open(args.radpf) as f:
            radpf = json.load(f)

    findings = diagnose(scalars, radpf)
    print("=== PIRT Diagnosis ===")
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    main()
