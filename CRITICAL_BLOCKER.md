# CRITICAL BLOCKER: MHD t_peak — REGRESSED

## Status: REGRESSED from 2.8% baseline (as of 2026-04-20)

## Current Results (PF-1000, 27 kV, 32x64 grid) — Apr 20 telemetry
Source: `~/.claude/dpf-validation/latest.json` (timestamp 2026-04-20T12:13:32)

- I_peak = 1.655 MA (+11.5% deviation vs RADPF 1.87 MA) — **REGRESSED** from 2.8% baseline (2026-04-10)
- t_peak = 6.02 us (+3.8% vs RADPF 5.80 us) — PASSES
- waveform_nrmse = 0.181
- V_pinch = 29.2 kV

Bisect pending on candidates bd9bd9c, 93b536f, 934eefd, 2daca88 (all on `fix/beta-lee-calibration`).
See `observations.md [2026-04-24 11:24] — claude-code-degradation-postmortem` for context on regression discovery.

## What Was Believed to Fix It (2026-04-10, PRE-REGRESSION)

### 1. Density floor (prevents vacuum v_A → infinity)
- `mlx_solver.py:766`: Enforce rho >= rho_fill * 1e-4 after each hyperbolic step
- Without this: rho=1e-12 at cathode corner → v_max=1e17 → NaN in 8 steps
- Beresnyak (2022) uses rho_min as explicit parameter (verif_r.cpp line 239)

### 2. Vacuum B_theta prescription (Ampere's law in uncompressed gas)
- `mlx_solver.py:892`: After all physics, set B_theta = mu0*I/(2*pi*r) in HL
  in cells where rho < 3*rho_fill (vacuum/fill gas, not sheath)
- **EMPIRICAL departure from Beresnyak 2022.** Beresnyak Sec. VII lists three
  remedies for vacuum-region mismatch:
    (a) set velocity of fluid in vacuum region so E-field matches the true
        vacuum solution,
    (b) minimize initial volume of vacuum so fictitious waves dissipate quickly
        (Laplace solve at IC), or
    (c) set vacuum density low enough that the Alfvénic timescale is much
        shorter than the plasma evolution time.
  Per-step interior B re-prescription is **NOT** in that list. Our approach is
  empirically motivated by density-floor-limited vacuum Alfvén speed preventing
  ghost-to-interior propagation within a step. Marked `# EMPIRICAL` pending
  validation or replacement with a Beresnyak-compliant approach.

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
- I_peak error is now 11.5% (regressed from 2.8%) — bisect pending.
- No self-consistent MHD circuit coupling yet (snowplow provides Lp)

## References
- Beresnyak et al. (2022), Phys. Plasmas 29:052712 — vacuum velocity method;
  Sec. VI & VII enumerate the three legitimate remedies above.
- Sun et al. (2025), Acta Physica Sinica 74:115201 — voltage-flux coupling
  (paper not on disk as of 2026-04-24; citation marked UNVERIFIED wherever used
  in source; see CHANGE 4 in fix/degradation-cleanup-apr24 branch).
- github.com/beresnyak/verif_coupling — reference implementation (verif_r.cpp)

## Task DAG
```
Task 1: CFL diagnostic [DONE]
Task 2: RADPF reference data [DONE]
Task 3: 5-angle acceptance test [DONE - 2/5 pass at 32x64]
Task 4: Grid search via ShinkaEvolve [DONE - 419 evals]
Task 5: Spitzer resistivity [DONE]
Task 6: Fix t_peak [REGRESSED - I_peak 2.8%→11.5% as of 2026-04-20]
  6a: Density floor [DONE]
  6b: Vacuum B prescription [DONE, but EMPIRICAL — see Section 2 above]
  6c: Snowplow circuit coupling [DONE]
  6d: Self-consistent MHD Lp [TODO]
  6e: Bisect regression on bd9bd9c/93b536f/934eefd/2daca88 [PENDING]
Task 7: Higher resolution convergence study [BLOCKED - resolve Task 6e first]
```
