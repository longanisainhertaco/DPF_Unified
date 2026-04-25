"""DPF MHD Engine Configuration — ShinkaEvolve initial program.

The EVOLVE-BLOCK contains ONLY numerical parameters (grid resolution).
Physics parameters (fc, fm, V0, pressure) are FIXED at published RADPF
values and are OUTSIDE the EVOLVE-BLOCK. ShinkaEvolve cannot modify them.

If no grid resolution produces correct I_peak with published parameters,
the code has a physics bug that ShinkaEvolve cannot fix.

Published parameters (Lee & Saw 2014, Scholz 2006, RADPF defaults):
    fc = 0.7, fm = 0.13, V0 = 27 kV, P = 3.5 Torr
"""
import sys
sys.path.insert(0, "/Users/anthonyzamora/dpf-unified/src")

import numpy as np

# Published RADPF parameters — INPUTS, NOT KNOBS.
# Lee & Saw, J. Fusion Energy 33:319 (2014).
# These are immutable. ShinkaEvolve cannot touch them.
FC = 0.7
FM = 0.13
V0_KV = 27.0
PRESSURE_TORR = 3.5


def run_mhd_discharge(seed: int = 0, **kwargs):
    """Run PF-1000 MHD discharge and return result dict."""
    from dpf.metal.mlx_engine import run_mlx_discharge

    # EVOLVE-BLOCK-START
    # Numerical parameters only. No physics constants.
    # Grid resolution: higher = better compression resolution but slower.
    # Valid range: 32-128.
    GRID_NR = 64       # radial cells (axis to cathode)
    GRID_NZ = 64       # axial cells (insulator to anode end)
    # EVOLVE-BLOCK-END

    result = run_mlx_discharge(
        "pf1000",
        mode="mhd",
        max_steps=50000,
        grid_shape=(GRID_NR, 1, GRID_NZ),
        fc=FC,
        fm=FM,
        pressure_torr=PRESSURE_TORR,
        V0_kV=V0_KV,
    )

    return result


if __name__ == "__main__":
    result = run_mhd_discharge()
    print(f"I_peak = {result['I_peak_MA']:.4f} MA")
    print(f"t_peak = {result['t_peak_us']:.2f} us")
