# CRITICAL BLOCKER: MHD t_peak — Partially Resolved

## Status: IMPROVED (from 66% to 13% error)

## Current Results (PF-1000, 27 kV, 32x64 grid)
- I_peak = 1.818 MA (-2.8% vs RADPF 1.87 MA) — PASSES
- t_peak = 6.32 us (+12.9% vs RADPF 5.6 us) — PASSES (within 15% criterion)
- Stable through full 12 us discharge, 1409 steps

## What Fixed It (2026-04-10)

### 1. Density floor (prevents vacuum v_A → infinity)
- `mlx_solver.py:766`: Enforce rho >= rho_fill * 1e-4 after each hyperbolic step
- Without this: rho=1e-12 at cathode corner → v_max=1e17 → NaN in 8 steps
- Beresnyak (2022) uses rho_min as explicit parameter (verif_r.cpp line 239)

### 2. Vacuum B_theta prescription (Ampere's law in uncompressed gas)
- `mlx_solver.py:892`: After all physics, set B_theta = mu0*I/(2*pi*r) in HL
  in cells where rho < 3*rho_fill (vacuum/fill gas, not sheath)
- The MHD solver cannot propagate B from ghost cells fast enough (Alfven
  speed limited by density floor). Direct prescription is physically correct:
  the vacuum coaxial field IS B_theta = mu0*I/(2*pi*r).
- Beresnyak initializes vacuum B at t=0 (verif_r.cpp:135). For DPF with I(t)
  starting from zero, we re-apply at each step.

### 3. Snowplow Lp for circuit coupling (avoids dPhi/dt instability)
- Volume-flux coupling (Sun 2025 Eq. 15) creates instability when B is
  prescribed directly: dPhi/dt ~ dI/dt * (large geometric factor) → overcorrects
- Boundary EMF (Beresnyak verif_r.cpp:174) gives v_r*B at cathode, but DPF
  flow is axial, so v_r≈0 → no feedback
- Solution: use snowplow Lp for circuit loading; MHD provides spatial resolution

### 4. Ghost zone velocity with dI/dt correction
- `mlx_solver.py:457`: Beresnyak extrapolation v_ghost = v_int + (v_int/r - curr_rate)*dr
- curr_rate = (1/I)(dI/dt) from two previous circuit steps (engine line 278)
- Threshold: curr_rate=0 when I < 1 kA (verif_r.cpp line 101)

### 5. Interior velocity prescription removed
- Previously applied v_z = (1/I)(dI/dt)(z_max-z) in ALL unswept interior cells
- This was too aggressive (blowup in 14 steps) and contradicts Beresnyak's
  approach which prescribes velocity ONLY in ghost zones (verif_r.cpp line 237)

## Remaining Issues
- B_theta in vacuum reads as 50% of expected (possible conversion or
  hyperbolic averaging issue). Doesn't affect circuit coupling (uses snowplow).
- MHD sheath compression is weak (dense cells ~100/2048 = 5% of domain)
- t_peak 12.9% late — within criterion but could improve with better sheath dynamics
- No self-consistent MHD circuit coupling yet (snowplow provides Lp)

## References
- Beresnyak et al. (2022), Phys. Plasmas 29:052712 — vacuum velocity method
- Sun et al. (2025), Acta Physica Sinica 74:115201 — voltage-flux coupling
- github.com/beresnyak/verif_coupling — reference implementation (verif_r.cpp)

## Task DAG
```
Task 1: CFL diagnostic [DONE]
Task 2: RADPF reference data [DONE]
Task 3: 5-angle acceptance test [DONE - 2/5 pass at 32x64]
Task 4: Grid search via ShinkaEvolve [DONE - 419 evals]
Task 5: Spitzer resistivity [DONE]
Task 6: Fix t_peak [PARTIALLY DONE - 66%→13% error]
  6a: Density floor [DONE]
  6b: Vacuum B prescription [DONE]
  6c: Snowplow circuit coupling [DONE]
  6d: Self-consistent MHD Lp [TODO]
Task 7: Higher resolution convergence study [UNBLOCKED]
```
