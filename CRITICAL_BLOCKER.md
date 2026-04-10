# CRITICAL BLOCKER: MHD t_peak 66% Late

## Status: ACTIVE

## What Works
- I_peak = 2.6% error at 80x56 grid with published params (fc=0.7, fm=0.13)
- ShinkaEvolve confirmed: grid tuning alone cannot fix t_peak (419 evaluations)
- Spitzer resistivity confirmed negligible at DPF conditions (Rm >> 100, Johnson 2024)
- Voltage-flux coupling (Sun 2025 Eq. 15-17) verified to reproduce Lee formula

## What Doesn't Work
- t_peak = 66% late (10.5 us vs target 6.32 us) — invariant across ALL grids
- Root cause: B_theta fills entire z-domain via Alfven wave from cathode BC in ~1 us
- L_p is effectively constant from the start (no progressive growth)
- The snowplow captures L_p growth because z_f grows analytically; MHD doesn't
  because the cathode BC lacks z-dependent current flow physics

## What's Needed
The cathode BC needs to model the fact that radial current only flows between
the electrodes WHERE THE SHEATH HAS PASSED. This requires either:
1. Full electrode geometry with anode as internal conductor (ALEGRA approach)
2. A current-sheet tracking algorithm that determines WHERE radial current flows
3. The snowplow z_f as a BC oracle (works but is not self-consistent)

## Success Criterion
test_mhd_acceptance.py passes all 5 angles:
- I_peak within 10% (currently 2.6% at 80x56 — PASSES)
- t_peak within 15% (currently 66% — FAILS)
- Waveform L2 within 20%
- dI/dt rise within 25%
- Lp_max within 30%

## Task DAG
```
Task 1: CFL diagnostic [DONE - clean]
Task 2: RADPF reference data [DONE]
Task 3: 5-angle acceptance test [DONE - 1/5 passes at 64x64]
Task 4: Grid search via ShinkaEvolve [DONE - 419 evals, score ceiling 0.740]
Task 5: Spitzer resistivity [DONE - negligible effect, Rm >> 100]
Task 6: Fix t_peak — requires electrode geometry change [TODO]
Task 7: All other work [BLOCKED BY: 6]
```
