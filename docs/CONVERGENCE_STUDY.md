# PF-1000 Grid Convergence Study

## Parameters

| Parameter | Value |
|-----------|-------|
| Mode | `lee` |
| Preset | `pf1000` |
| fc (current fraction) | 0.7 |
| fm (mass fraction) | 0.13 |
| V0 | 27 kV |
| C | 1.332 mF |
| L0 | 33.5 nH |
| R0 | 6.12 mOhm |
| Sim time | 8 us |
| CFL | 0.4 |
| Riemann solver | HLL |
| Reconstruction | PLM |

## Results by Resolution

| Grid | dr [mm] | I_peak [MA] | t_peak [us] | Wall [s] |
|------|---------|-------------|-------------|----------|
| 16x32 | 2.81 | 1.8180 | 6.32 | 0.0 |
| 32x64 | 1.41 | 1.8180 | 6.32 | 0.0 |
| 64x128 | 0.70 | 1.8180 | 6.32 | 0.0 |
| 128x256 | 0.35 | 1.8180 | 6.32 | 0.0 |

## Convergence Diagnostics

| Metric | Value |
|--------|-------|
| Refinement ratio r | 2.0 |
| Observed order p | 0.00 |
| Richardson-extrapolated I_peak | 1.8180 MA |
| GCI (fine grid) | 100.00% |
| Converged (GCI < 5%) | NO |

## Interpretation

**I_peak is grid-independent by design in Lee mode.** The circuit + snowplow ODE
system drives the current waveform. The MHD grid resolution has no effect because
the Lee model computes sheath position, velocity, and inductance analytically —
not from the MHD state. This confirms the README's known limitation #1.

This result is **correct and expected**: the Lee model convergence test validates
that the circuit solver produces reproducible results independent of grid
resolution. MHD spatial convergence (density profiles, B-field structure, pinch
radius) requires `--mode mhd`, which runs the full MLX MHD solver and takes
significantly longer (~28+ minutes across 4 resolutions).

## Notes

- Published RADPF parameters used verbatim (Lee & Saw 2014, Scholz 2006).
- No parameter calibration was performed.
- Solver: MLX HLL + PLM, mode=`lee`.
- Lee mode (`mode='lee'`) runs circuit+snowplow only; spatial fields
  (rho_max, B_max, T_max, density profile) are only available in `mode='mhd'`.
- PF-1000 reference: I_peak = 1.87 MA at 27 kV (Scholz 2006).
- GCI method: Roache (1998). Safety factor Fs=1.25 (3+ grids).
- MHD spatial convergence study: run `python3 scripts/run_convergence_study.py --mode mhd`
  (recommended as overnight job; captures rho_max, B_max, T_max, z-axis density profile).
