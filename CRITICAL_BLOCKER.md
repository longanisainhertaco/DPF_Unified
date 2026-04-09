# CRITICAL BLOCKER: MHD Solver Cannot Drive DPF Discharge

## Status: ACTIVE

## The Problem
test_mlx_pf1000.py criterion M2 is XFAIL: the MHD solver produces wrong plasma
resistance. When MHD drives the circuit (without Lee snowplow), I_peak diverges
from experiment. The 2.8% accuracy belongs to the Lee model, not the MHD solver.

## Root Cause (Partially Fixed)
Sheath detection in `_update_coupling()` used `argmax(column_density)` which
picked fill gas boundaries instead of the compressed sheath due to cylindrical
volume element bias. Fixed 2026-04-08: now uses on-axis density profile.
L_p error on Bennett pinch: 27% -> 4.5%.

## What Remains
1. Run full MHD discharge with the sheath detection fix and compare I_peak to RADPF
2. If still wrong: diagnose with extract_scalars.py + pirt_traverse.py
3. CFL diagnostic: verify dt is not collapsing due to vacuum v_Alfven spike
4. Replace all silent floors with telemetry.apply_floor()
5. Multi-angle acceptance test: I_peak, t_peak, radial duration, waveform L2, dI/dt

## Success Criterion
test_mhd_acceptance.py passes all 5 angles against RADPF reference data:
- I_peak within 10% of RADPF
- t_peak within 15% of RADPF
- Radial phase duration within 20%
- I(t) waveform L2 norm within 20%
- dI/dt at current rise within 25%

## Task DAG
```
Task 1: CFL diagnostic on current solver [TODO]
Task 2: Run RADPF, save reference data [HUMAN - Anthony]
Task 3: Write test_mhd_acceptance.py with 5-angle test [BLOCKED BY: 2]
Task 4: Run full MHD discharge with sheath fix [BLOCKED BY: 1]
Task 5: Compare MHD output to RADPF reference [BLOCKED BY: 2, 4]
Task 6: Diagnose divergence point if test fails [BLOCKED BY: 5]
Task 7: All other work [BLOCKED BY: 5 passes]
```

## Files That Matter
- `src/dpf/metal/mlx_coupling.py` — L_p computation (FIXED: sheath detection)
- `src/dpf/metal/mlx_solver.py` — MHD solver step()
- `src/dpf/metal/mlx_engine.py` — discharge simulation driver
- `tests/test_mlx_pf1000.py` — M2 criterion (currently XFAIL)
- `tests/reference_data/` — RADPF truth traces (Anthony generates)
