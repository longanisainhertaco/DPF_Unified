# CRITICAL BLOCKER: MHD t_peak — RE-ANCHORED TO KR-CANONICAL INPUTS

## Status: IMPROVED (66% → ~13% error) — RE-ANCHORED TO KR-CANONICAL INPUTS

> **Apr-10 baseline (2.8% I_peak) was against an uncalibrated parameter set
> (R0=2.3 mΩ with EMPIRICAL R0_CORRECTION=6.43 mΩ knob; stale Lee fits).
> That result was a calibration artifact, not a model accuracy number.
> The KR-canonical re-anchor (Apr-23+) is the new truth.**
>
> The Apr-20 "regression to 11.5%" reported in the prior version of this
> document was the SYMPTOM of the EMPIRICAL knob being correctly removed;
> the reported error widened because the old number was self-fitted, not
> because the model got worse. See `feedback/akel-1pct-claim-was-self-fit.md`
> and `feedback/papers-are-truth.md`.

## Current Results (PF-1000, 27 kV — HEAD `5746c81`, KR-canonical inputs)
- I_peak = 2.013 MA (+7.6% vs Scholz 2006 1.87 MA) — PASSES (< 15% criterion)
- t_peak: +11.5% vs RADPF 5.6 µs — PASSES (within 15% criterion)
- waveform_nrmse = 0.235 (Wave-2/3 BREM_COEFF + /mu_0 effect)
- Basis: Malek 2025 PPT 12(1):9 Lee fits (fc=0.7, fm=0.13, fmr=0.35, fcr=0.65)
         + Akel 2021 device geometry (L0=25 nH, R0=2.3 mΩ bare-bank, z0=480 mm)
- The 7.6% I_peak deficit is the agreed accuracy budget for paper-fidelity.
  Calibrating R0 or Lee fits to close this gap is forbidden by
  `feedback/papers-are-truth.md` and `published-parameters-are-inputs-not-knobs`.

## Superseded Apr-10 Results (pre-KR-canonical — DO NOT USE FOR VALIDATION)
- I_peak = 1.818 MA (-2.8% vs RADPF 1.87 MA)
- t_peak = 6.32 µs (+12.9% vs RADPF 5.6 µs)
- These numbers used R0_CORRECTION=6.43 mΩ EMPIRICAL knob (calibration-as-bug,
  documented in `feedback/akel-1pct-claim-was-self-fit.md`).
- Stable through full 12 µs discharge, 1409 steps

## Re-anchored to KR-canonical Inputs (2026-04-23+)

Five commits rewrote the PF-1000 parameter chain to verbatim published values:

| SHA | Change |
|-----|--------|
| `e219ebb` | PF-1000 R0=6.1 mΩ per Akel 2021 p.2; two test classes xfail'd pending threshold recalibration |
| `695a309` | engine_validation.py aligned to Akel 2021 (L0=25 nH, R0=6.1 mΩ, z0=480 mm) |
| `b08c615` | Lee fits replaced with Malek 2025 KR-canonical: fm=0.13, fmr=0.35, fcr=0.65 |
| `0958947` | DEVICE_TOLERANCES I_peak threshold bumped to 0.12 (regression fence, not a published gate) |
| `5746c81` | Wave-9/10 unified canonical preset: R0=2.3 mΩ bare-bank (plasma R via sheath), dropped EMPIRICAL R0_CORRECTION=6.43 mΩ knob; final accuracy 2.013 MA = +7.6% |

Key rationale from `5746c81` body: *"Akel 1.27% claim was a self-fit (R0_CORRECTION=6.43 + per-shot fm, papers-are-truth violation)"*. The prior EMPIRICAL knob was the bug; removing it is correct even though it widens the apparent error.

---

## What Was Believed to Fix It (2026-04-10, PRE-KR-CANONICAL)

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
- t_peak +11.5% late — within criterion but could improve with better sheath dynamics
- No self-consistent MHD circuit coupling yet (snowplow provides Lp)
- I_peak +7.6% vs Scholz 1.87 MA — within 15% criterion; closing gap requires
  better sheath physics, NOT parameter adjustment

## References
- Beresnyak et al. (2022), Phys. Plasmas 29:052712 — vacuum velocity method;
  Sec. VI & VII enumerate the three legitimate remedies above.
- Sun et al. (2025), Acta Physica Sinica 74:115201 — voltage-flux coupling
  [KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md §2.4]
- Malek et al. (2025), Plasma Phys. Tech. 12(1):9 — Lee fits fc=0.7, fm=0.13,
  fmr=0.35, fcr=0.65 [KR: plasma-physics-and-technology-1211-9-2025.md §3 ll.177-180]
- Akel et al. (2021), Radiat. Phys. Chem. 188:109633 — PF-1000 device parameters
  [KR: radiation-physics-and-chemistry-188-2021-109633.md]
- github.com/beresnyak/verif_coupling — reference implementation (verif_r.cpp)

## Task DAG
```
Task 1: CFL diagnostic [DONE]
Task 2: RADPF reference data [DONE; pending Anthony's Malek 2025 fcr=0.65 regen
        per docs/RADPF_REGENERATION_PLAYBOOK.md]
Task 3: 5-angle acceptance test [DONE - 2/5 pass at 32x64]
Task 4: Grid search via ShinkaEvolve [DONE - 419 evals]
Task 5: Spitzer resistivity [DONE]
Task 6: Fix t_peak [RE-ANCHORED - 66%→~13% error; KR-canonical inputs; PASSES 15% criterion]
  NOTE: Apr-10 2.8% I_peak was calibration artifact (EMPIRICAL R0_CORRECTION knob).
        HEAD 5746c81: +7.6% I_peak / +11.5% t_peak against KR-canonical Akel/Malek inputs.
        Paper-fidelity deficit is INTENTIONAL; closing it requires sheath physics, not tuning.
  6a: Density floor [DONE]
  6b: Vacuum B prescription [DONE, but EMPIRICAL — see Section 2 above]
  6c: Snowplow circuit coupling [DONE]
  6d: Self-consistent MHD Lp [TODO — post PR-B follow-up]
Task 7: Higher resolution convergence study [UNBLOCKED — sheath physics improvements]
```
