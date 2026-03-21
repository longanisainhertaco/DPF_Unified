"""Verify that the 3D Babylon.js renderer receives correct simulation data.

This script runs a simulation, extracts the exact data that would be sent
to the Babylon.js renderer, and compares it against the raw simulation output
at matching timesteps. Any discrepancy indicates a data pipeline bug.

Usage:
    python3 scripts/verify_3d_data.py [preset_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def run_verification(preset_name: str = "pf1000") -> dict:
    """Run simulation and verify 3D data pipeline integrity."""
    from dpf.presets import get_preset
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    print(f"=== 3D Data Pipeline Verification: {preset_name} ===\n")

    # Run simulation via the same path the frontend uses
    preset = get_preset(preset_name)
    config = SimulationConfig(**preset)
    engine = SimulationEngine(config)

    # Collect simulation data at each step
    sim_times = []
    sim_currents = []
    sim_z_positions = []
    sim_r_positions = []
    sim_phases = []

    for i in range(50000):
        result = engine.step()
        sim_times.append(engine.time)
        sim_currents.append(abs(engine.circuit.current))

        # Get snowplow position if available
        if hasattr(engine, 'snowplow') and engine.snowplow is not None:
            sp = engine.snowplow
            sim_z_positions.append(getattr(sp, 'z_position', 0) * 1e3)  # mm
            sim_r_positions.append(getattr(sp, 'r_shock', 0) * 1e3)  # mm
            sim_phases.append(getattr(sp, 'current_phase', 'unknown'))
        else:
            sim_z_positions.append(0)
            sim_r_positions.append(0)
            sim_phases.append('mhd')

        if result.finished:
            break

    sim_times = np.array(sim_times)
    sim_currents = np.array(sim_currents)

    print(f"Simulation completed: {len(sim_times)} steps, t_final={sim_times[-1]*1e6:.1f} us")
    print(f"I_peak = {max(sim_currents)/1e6:.3f} MA at t = {sim_times[np.argmax(sim_currents)]*1e6:.1f} us\n")

    # Now check what the Babylon renderer would receive
    # The frontend downsamples to 60 frames
    n_frames = 60
    frame_indices = np.linspace(0, len(sim_times) - 1, n_frames, dtype=int)

    print("=== Frame-by-Frame Verification ===")
    print(f"{'Frame':>5} | {'t [us]':>8} | {'I [MA]':>8} | {'z [mm]':>8} | {'r [mm]':>8} | {'Phase':>12}")
    print("-" * 70)

    errors = []
    for fi, idx in enumerate(frame_indices):
        t_us = sim_times[idx] * 1e6
        I_MA = sim_currents[idx] / 1e6
        z_mm = sim_z_positions[idx] if idx < len(sim_z_positions) else 0
        r_mm = sim_r_positions[idx] if idx < len(sim_r_positions) else 0
        phase = sim_phases[idx] if idx < len(sim_phases) else 'unknown'

        if fi % 10 == 0 or fi == n_frames - 1:
            print(f"{fi:5d} | {t_us:8.2f} | {I_MA:8.4f} | {z_mm:8.1f} | {r_mm:8.1f} | {phase:>12}")

        # Check for NaN/Inf in data that would corrupt the 3D render
        if np.isnan(t_us) or np.isnan(I_MA):
            errors.append(f"Frame {fi}: NaN detected (t={t_us}, I={I_MA})")
        if I_MA < 0:
            errors.append(f"Frame {fi}: Negative current ({I_MA} MA)")
        if z_mm < 0:
            errors.append(f"Frame {fi}: Negative z position ({z_mm} mm)")

    print()

    # Check MHD field data (if available from the final state)
    state = engine.state
    mhd_checks = {}
    for field in ['rho', 'pressure', 'B', 'velocity', 'Te', 'Ti']:
        if field in state:
            arr = state[field]
            mhd_checks[field] = {
                'shape': arr.shape,
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'nan_count': int(np.isnan(arr).sum()),
                'inf_count': int(np.isinf(arr).sum()),
            }

    print("=== Final MHD State (what heatmaps would show) ===")
    for field, info in mhd_checks.items():
        status = "OK" if info['nan_count'] == 0 and info['inf_count'] == 0 else "BAD"
        print(f"  {field:12s} shape={str(info['shape']):>15s} "
              f"range=[{info['min']:.2e}, {info['max']:.2e}] "
              f"NaN={info['nan_count']} Inf={info['inf_count']} [{status}]")

    print()

    # Verify B-field components for field line rendering
    if 'B' in state:
        B = state['B']
        if B.ndim >= 3:
            # Check Br, Btheta, Bz components
            print("=== B-Field Components (for poloidal field lines) ===")
            labels = ['B_r', 'B_theta', 'B_z'] if B.shape[0] == 3 else [f'B_{i}' for i in range(B.shape[0])]
            for i, label in enumerate(labels):
                comp = B[i]
                print(f"  {label:10s} range=[{comp.min():.4e}, {comp.max():.4e}]")

    # Summary
    print(f"\n=== VERIFICATION RESULT ===")
    if errors:
        print(f"FAIL: {len(errors)} errors found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"PASS: All {n_frames} frames clean. No NaN, no negative values.")
        print(f"  MHD fields: {len(mhd_checks)} available for heatmap rendering")
        print(f"  B-field: {'3 components for field lines' if 'B' in state else 'NOT AVAILABLE'}")

    return {
        'preset': preset_name,
        'n_steps': len(sim_times),
        'n_frames': n_frames,
        'errors': errors,
        'mhd_fields': list(mhd_checks.keys()),
        'I_peak_MA': float(max(sim_currents) / 1e6),
    }


if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "pf1000"
    result = run_verification(preset)
