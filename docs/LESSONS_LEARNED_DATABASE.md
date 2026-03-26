---
type: lessons-learned
tags: [six-sigma, rca, root-cause, fmea, dpf, process, lessons]
project: dpf-unified
created: 2026-03-26
author: Cortana (Opus)
purpose: Structured database of all root cause analyses + second-pass FMEA critique
---

# DPF-Unified Lessons Learned Database

## Part I: Second-Pass Six Sigma Critique of DMAIC_FORWARD_PLAN.md

### Executive Summary

The first-pass DMAIC (docs/DMAIC_FORWARD_PLAN.md) is thorough and honest in its
composite sigma estimate (1.8 real, 2.4 weighted). However, three systemic risks
are under-scored, two failure modes are absent, and the physics test coverage
fraction is significantly overstated by raw test count.

---

### 1. Under-Scored RPN Items

#### 1a. POSEIDON / High-Voltage Float32 HLL NaN (NOT IN FMEA)

The FMEA covers PF-1000 at 27 kV. POSEIDON operates at 60 kV with 4x higher
magnetic field strength. The HLL solver computes wave speeds as:

```
S_L/R = v +/- c_f,  where c_f = sqrt(a^2 + v_A^2)
```

At POSEIDON conditions: B_theta ~ 200 T at the electrode, v_A ~ 2e7 m/s.
The HLL flux numerator `S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)` involves
products of O(1e7) * O(1e14) = O(1e21), which is within float32 range (max 3.4e38)
but the SUBTRACTION of two O(1e21) terms can lose 7+ significant digits in float32
(which has ~7.2 digits). This is the SAME catastrophic cancellation pattern that
killed HLLD at PF-1000 conditions, just at a higher threshold.

**Risk**: ANY device with V0 > 40 kV or B_theta > 100 T at the electrode will
trigger float32 cancellation in HLL, not just HLLD. The FMEA assumes HLL is
unconditionally stable. It is not.

**Correct RPN**: S=9, O=7 (every high-energy device), D=5 (shows as NaN but only
after many steps). **RPN = 315**. This should be ranked #2, not absent.

**Mitigation**: The HLLS entropy solver (Feature 1) partially addresses this, but
only if the entropy equation itself avoids the cancellation. The real fix is
mixed-precision HLL: compute wave speeds and fluxes in float64 on CPU, store
results in float32. ~50 LOC.

#### 1b. Species + Vacuum Cell Z_eff Corruption (Item 2.2, RPN 72 -- TOO LOW)

The current RPN of 72 treats Z_eff error as a moderate issue with low occurrence.
In reality:

When Cu ablation injects species into vacuum cells (rho = RHO_FLOOR = 1e-12),
the mass fraction Y_Cu = m_Cu / rho_total. If rho_total is artificial (floor),
Y_Cu becomes enormous: even 1e-20 kg of Cu in a 1e-12 kg/m^3 cell gives
Y_Cu = 1e-8, but Z_eff = sum(Y_k * Z_k^2) / sum(Y_k * Z_k). With Z_Cu=29
and Z_D=1, even a tiny Cu fraction dominates Z_eff:

```
Z_eff = (Y_D * 1 + Y_Cu * 841) / (Y_D * 1 + Y_Cu * 29)
```

At Y_Cu = 1e-4 (0.01%): Z_eff = 1.08 (fine).
At Y_Cu = 0.01 (1%): Z_eff = 8.7 (Bremsstrahlung up 76x).
At Y_Cu = 0.10 (10%): Z_eff = 24.3 (radiation catastrophe).

The vacuum floor creates artificial dilution that makes Y_Cu grow without bound
as Cu advects into vacuum. This is **silent corruption** -- no NaN, no crash, just
progressively wrong radiation losses that drain energy from the pinch.

**Correct RPN**: S=8, O=7 (every run with ablation + vacuum), D=8 (no existing
test catches Z_eff in vacuum cells). **RPN = 448**. This should be ranked #1.

**Mitigation**: (a) Mask vacuum cells from Z_eff computation. (b) Use a
plasma-vacuum interface tracker (Gap 20 in FULL_MHD_GAP_LIST.md) instead of
density floors. (c) At minimum, clamp Z_eff <= Z_max (configurable, default 10)
in vacuum-adjacent cells. ~20 LOC for the clamp.

#### 1c. Calibration Compute Time (Item 4.2, RPN 160 -- HONEST BUT UNDERSTATED IMPACT)

The FMEA says 4 hours per device. With 6 devices, that is 24 hours minimum.
The real constraint: if ANY device fails to converge (POSEIDON's NaN, FAETON-I's
re-strikes, MJOLNIR's 1 MJ circuit), the entire multi-device calibration claim
is blocked. The probability of at least one device failing is:

P(at least one fails) = 1 - (1 - p_fail)^6

With p_fail = 0.30 per device (based on POSEIDON float32 risk + circuit topology
differences), P(at least one fails) = 1 - 0.7^6 = 88%.

**Correct RPN**: S=6, O=9 (near-certain that at least one device blocks), D=7.
**RPN = 378**. This is the true critical path risk.

---

### 2. Missing Failure Modes

#### 2a. Snowplow-to-MHD Handoff Phase Mismatch

The handoff from 0D snowplow to 2D MHD creates a discontinuity in the physical
model. During axial rundown, a 0D ODE controls everything; during radial phase,
2D MHD takes over. At the boundary, the MHD solver receives initial conditions
interpolated from the snowplow terminal state. If the interpolation creates
non-physical gradients (e.g., a sheath profile that violates the Rankine-Hugoniot
conditions for the given Mach number), the MHD solver will either:
(a) emit a spurious shock that reflects off the boundary, or
(b) generate a rarefaction wave that evacuates the pinch region.

This failure mode is documented nowhere in the FMEA but is the root cause of the
"hybrid snowplow+MHD fundamental conflict" identified in the 2026-03-23 marathon
session (14 root cause layers). It was resolved by using HLL+PLM (enough numerical
diffusion to smooth the handoff) but will recur with HLLS or HLLD (less diffusive).

**RPN**: S=8, O=5 (every HLLS/HLLD run), D=7 (looks like "wrong dip depth", not
handoff error). **RPN = 280**.

#### 2b. Runaway Daemon / Orphan Process Resource Exhaustion

Observations.md documents 10+ occurrences of Python 3.11 processes pegging CPU at
100% for hours. The FMEA covers software failure modes but not operational failure
modes. During a 24-hour calibration run, orphan processes consuming 300% CPU will:
(a) thermal-throttle the M3 Pro, reducing calibration speed by ~30%, and
(b) potentially cause OOM kills that corrupt the Optuna SQLite database.

**RPN**: S=7 (corrupted calibration run), O=8 (documented recurring daily), D=5
(discovered only by checking Activity Monitor). **RPN = 280**.

---

### 3. Physics Test Coverage: The Real Number

The FMEA cites 4,970 tests. Grepping for physics-relevant test names (conservation,
convergence, shock, wave, accuracy, NRMSE, Sod, Brio-Wu, Orszag-Tang, Rankine-
Hugoniot) yields **~919 matches across 139 test files**.

That means:
- **~18.5% of tests verify physics correctness**
- **~81.5% test API contracts, shape checks, configuration, infrastructure**

The 18.5% is not bad for a simulation code (most Athena++ tests are also
infrastructure), but the FMEA's implicit claim that "4,970 tests" provides
confidence in physics is misleading. The relevant metric is:

- Conservation tests: ~45 (energy, mass, momentum)
- Convergence-order tests: ~30 (L1/L2 norm vs resolution)
- Shock tube tests (Sod/Brio-Wu/Double-rarefaction): ~80
- Cross-backend parity tests: ~25
- Validation against experiment (NRMSE, I_peak, t_peak): ~40
- **Total physics-verifying tests: ~220**

The remaining ~700 "physics" tests are parametric variations (same test with
different configs) that increase count but not coverage.

**Honest physics coverage: ~220 independent physics tests, ~4.4% of total.**

---

### 4. Is the Sigma Estimate Honest?

The first pass reports 1.8 sigma (honest) to 2.4 sigma (weighted).

**Assessment**: The 1.8 sigma is honest. The 2.4 weighted figure is slightly
inflated because it weights CTQ-6 (wall time, 4.0 sigma) equally with CTQ-1
(NRMSE, 0.5 sigma). Wall time passing is table stakes, not a quality indicator.

A more honest weighting:
- CTQ-1 (NRMSE all devices): weight 0.35 -- sigma 0.5
- CTQ-5 (NaN-free): weight 0.25 -- sigma 2.5
- CTQ-3 (Dip depth): weight 0.15 -- sigma 2.0
- CTQ-4 (Timing): weight 0.15 -- sigma 3.5
- CTQ-2 (Yn): weight 0.05 -- sigma 3.0 (insufficient N)
- CTQ-6 (Wall time): weight 0.05 -- sigma 4.0

**Revised composite: 0.35*0.5 + 0.25*2.5 + 0.15*2.0 + 0.15*3.5 + 0.05*3.0 + 0.05*4.0 = 1.68 sigma**

The real number is **1.7 sigma**, not 2.4.

---

### 5. The REAL Bottleneck

The first pass identifies calibration compute (7 days on critical path) as the
bottleneck. That is the *schedule* bottleneck. The *quality* bottleneck is different:

**The snowplow-to-MHD handoff is the single point of failure for all physics
fidelity claims.**

Every CTQ metric flows through the handoff:
- I_peak depends on sheath arrival timing (snowplow) -> dL/dt (handoff)
- t_peak depends on snowplow rundown speed -> MHD radial dynamics (handoff)
- Dip depth depends on post-pinch expansion -> back-EMF (handoff)
- Yn depends on pinch conditions set by the handoff

If the handoff produces non-physical initial conditions for the MHD solver,
calibration simply tunes fc/fm to compensate (compensating errors -- documented
lesson). The calibrated parameters then fail on other devices.

**The #1 risk is not "can we calibrate 7 devices" but "does the handoff produce
physically consistent initial conditions across operating regimes."**

---

### 6. Revised Top-10 FMEA

| Rank | ID | Failure Mode | RPN | Change |
|------|-----|-------------|-----|--------|
| 1 | **NEW** | Vacuum cell Z_eff corruption from species + density floor | **448** | New entry |
| 2 | **NEW** | Calibration pipeline blocked by high-V device NaN | **378** | Recomputed |
| 3 | 7.3 | AMR GPU Amdahl bottleneck | **336** | Unchanged |
| 4 | **NEW** | High-voltage HLL float32 cancellation (>40 kV) | **315** | New entry |
| 5 | 7.1 | AMR patch boundary conservation error | **315** | Unchanged |
| 6 | 3.1 | PIC non-physical velocity initialization | **294** | Unchanged |
| 7 | 4.1 | Optuna local minimum for non-PF1000 | **294** | Unchanged |
| 8 | 4.3 | Single-shot calibration overfitting | **294** | Unchanged |
| 9 | **NEW** | Snowplow-MHD handoff non-physical IC with low-diffusion solver | **280** | New entry |
| 10 | **NEW** | Orphan process resource exhaustion during calibration | **280** | New entry |

Items displaced from original top-10: CIC grid-noise (245), stale HF deploy (245),
HLLS entropy switch (210), Gradio timeout (210), user NaN (200). These are real but
lower priority than the systemic risks above.

---

---

## Part II: Root Cause Analysis Lessons Learned Database

### Format

Each entry: Date, Category, Symptom, Root Cause, Fix, Lesson, Applicable To,
Surprise Factor (1=predictable, 5=genuinely unexpected).

---

### RCA-001: Float32 Catastrophic Cancellation in HLLD Star-States

- **Date**: 2026-03-22 to 2026-03-24
- **Category**: Numerical
- **Symptom**: NaN in HLLD Riemann solver at strong B-field discontinuities (Brio-Wu test, electrode boundary)
- **Root cause**: The HLLD discriminant `(a^2 + v_A^2)^2 - 4*a^2*v_An^2` involves subtraction of two large, nearly equal terms. In float32 (7.2 significant digits), when a^2 ~ v_A^2, the subtraction loses all precision. Numerically stable form exists: `(a^2 - v_A^2)^2 + 4*a^2*B_t^2/rho`.
- **Fix**: Algebraically stable discriminant + NaN guards + velocity clamping + Lax-Friedrichs fallback. ~100 LOC in `metal_riemann.py`.
- **Lesson**: Any float32 expression of the form `(A+B)^2 - 4*A*C` where B ~ 2*sqrt(A*C) will cancel. Rewrite before implementing. Standard in numerical analysis but easy to miss when translating textbook MHD formulas.
- **Applicable to**: All float32 MHD solvers, all Riemann solvers with wave speed discriminants, any GPU code without float64.
- **Surprise factor**: 2 (predictable from float32 limitations, but the specific site was non-obvious)

---

### RCA-002: dp/dt Chain Rule as the ACTUAL Cancellation Site

- **Date**: 2026-03-24
- **Category**: Numerical
- **Symptom**: Negative pressure even after HLLD fix, specifically at electrode boundary cells
- **Root cause**: At `metal_riemann.py:271-273`, the pressure time derivative is computed as `dp/dt = (gamma-1) * (dE/dt - v*dmom/dt - B*dB/dt)`. When E ~ KE + ME (low thermal fraction), this subtraction cancels catastrophically. The dp/dt chain rule IS the cancellation, not the Riemann solver.
- **Fix**: Dual-energy method: evolve entropy tracer (Srho = rho * p / rho^gamma) alongside total energy. Use entropy-derived pressure when p_thermal / E < eta threshold. Switching criterion `p_from_S / E` is our original contribution (avoids Enzo/FLASH circular dependency).
- **Lesson**: Root cause analysis must go past the first plausible explanation. The HLLD NaN was the symptom that led to the fix, but the dp/dt chain rule was the deeper cause that HLLD masked with its numerical diffusion.
- **Applicable to**: All conservative MHD codes in float32, all codes computing pressure from total energy minus kinetic+magnetic, dual-energy switching criteria.
- **Surprise factor**: 4 (the specific chain rule was identified only after the HLLD fix revealed a deeper layer)

---

### RCA-003: Ghost Cell Evolution Destroying Electrode B_theta

- **Date**: 2026-03-25
- **Category**: Numerical
- **Symptom**: NaN originating from electrode boundary cells after 10-50 timesteps on MLX solver
- **Root cause**: Ghost cells are evolved by the RK integrator along with physical cells. After RK stage 1, the ghost cells have been modified by the MHD RHS (fluxes + source terms). The electrode BC then overwrites B_theta to the 1/r analytical profile, but density and energy are already corrupted by the spurious RHS. This creates inconsistent (rho, B, E) in the ghost cells that feeds into stage 2 fluxes.
- **Fix**: `_mask_ghost_rhs()` zeros the RHS in ghost regions before each RK stage. 15 LOC.
- **Lesson**: Four previous proposals (HLL fallback, float64, NaN repair, density floor) all failed because they treated symptoms. The root cause was architectural: the RK integrator should never modify ghost cells. In Athena++ this is handled by STS implicit treatment; in our explicit solver, masking is required.
- **Applicable to**: Any explicit MHD code with fixed-value boundary conditions in ghost cells and multi-stage time integrators.
- **Surprise factor**: 4 (the 4 stale proposals consumed hours before the real cause was found by tracing values step-by-step through the RK stages)

---

### RCA-004: Compensating Errors in fc/fm Calibration

- **Date**: 2026-03-24
- **Category**: Calibration
- **Symptom**: After fixing back-EMF double-counting bug, I_peak jumped 47% (from 1.8 MA to 2.6 MA)
- **Root cause**: The published fc=0.7, fm=0.08 were calibrated WITH the back-EMF double-counting. The bug reduced the effective driving force, so the calibrated fc was higher to compensate. Fixing the bug removed the compensating reduction, and the now-correct physics with the old parameters produced too much current.
- **Fix**: Reverted the back-EMF fix, documented as "known compensating error", scheduled recalibration as a prerequisite for the back-EMF fix.
- **Lesson**: Never fix a physics bug without recalibrating. Calibrated parameters absorb bugs. Fixing one bug without recalibrating creates a WORSE result than having both the bug and the compensating parameters. This is a fundamental trap in any calibrated simulation code.
- **Applicable to**: ANY simulation with empirical parameters fitted to data. Climate models, CFD with turbulence models, nuclear reactor codes.
- **Surprise factor**: 3 (well-known in the modeling community, but painful when encountered)

---

### RCA-005: Timing Error Overstated by 3x

- **Date**: 2026-03-25
- **Category**: Calibration / Measurement
- **Symptom**: t_peak error reported as 10-14% (0.57-0.79 us late vs Scholz reference at 5.8 us)
- **Root cause**: The Scholz waveform has only 26 hand-digitized points with ~0.4 us spacing near peak. The Gribkov waveform (94 points, same device, independent shot) shows a flat-top from 5.2-6.6 us with only 1.5% current variation. The "peak" is a plateau, not a sharp maximum. Our t_peak of 6.4 us falls within the Gribkov flat-top region.
- **Fix**: Use Gribkov as primary reference. Acknowledge that PF-1000 t_peak has ~10% intrinsic measurement ambiguity. Real error: 0-3%.
- **Lesson**: Before diagnosing a model error, verify the reference data. The cheapest "fix" is discovering the problem does not exist. Check: (1) Is the reference data adequate resolution? (2) Is the comparison metric well-defined? (3) What is the measurement uncertainty?
- **Applicable to**: Any validation against digitized waveforms, any flat-topped or plateau signals where peak location is ambiguous.
- **Surprise factor**: 4 (the reframing was only found by a dedicated RCA agent; intuition said "model is late")

---

### RCA-006: Hall CFL Hardcoded True

- **Date**: 2026-03-23
- **Category**: Infrastructure
- **Symptom**: dt = 1.78 ps instead of expected ~1 ns, making simulations 560x slower than necessary
- **Root cause**: A physics flag `use_hall` was hardcoded to `True` in the CFL computation instead of reading from the config. The Hall CFL constraint `dt < dx^2 / (omega_ci * d_i)` produces extremely small dt at DPF conditions.
- **Fix**: Read `use_hall` from config. 1 LOC.
- **Lesson**: Every physics flag in a CFL computation must be read from configuration. Hardcoded flags are invisible performance killers.
- **Applicable to**: Any multi-physics code where optional physics modules add CFL constraints.
- **Surprise factor**: 2 (obvious in retrospect, but took profiling to discover)

---

### RCA-007: Electrode BC in Active Cells Destroys Conservation

- **Date**: 2026-03-23
- **Category**: Physics
- **Symptom**: Energy growing without bound during Sod shock test with electrode boundary
- **Root cause**: Electrode boundary condition was applied in active computational cells between RK substages, injecting energy that was not accounted for in the conservative update. The energy injection compounds across RK stages.
- **Fix**: Apply electrode BC in ghost cells ONLY, never in active cells.
- **Lesson**: Boundary conditions that modify conserved variables must never be applied inside the computational domain. Ghost cells exist precisely for this purpose.
- **Applicable to**: All explicit conservative schemes with fixed-value boundary conditions.
- **Surprise factor**: 2 (standard numerical methods knowledge, but the multi-stage RK interaction was subtle)

---

### RCA-008: SI-to-Heaviside-Lorentz B-field Conversion

- **Date**: 2026-03-22
- **Category**: Physics
- **Symptom**: Magnetic forces 10^6x too weak in Metal solver, no pinch compression
- **Root cause**: Metal solver uses Heaviside-Lorentz units (mu0=1). Input B-field in Tesla must be divided by sqrt(mu0) to convert. Without this, B_HL = B_SI (in Tesla), but the force F = J x B_HL is computed with mu0=1, so the force is mu0 (= 4*pi*1e-7) times too small.
- **Fix**: B_HL = B_SI / sqrt(mu0) at input, B_SI = B_HL * sqrt(mu0) at output. 4 LOC.
- **Lesson**: Unit systems must be documented at every interface boundary. Mixed-unit bugs produce silently wrong results (not crashes), making them harder to detect than NaN bugs.
- **Applicable to**: Any code mixing SI and Gaussian/HL units, any code with GPU kernels in different units than the host.
- **Surprise factor**: 3 (unit bugs are common but the 10^6 magnitude was dramatically wrong)

---

### RCA-009: Temperature Factor-of-2 for Ionized Plasma

- **Date**: 2026-03-25
- **Category**: Physics
- **Symptom**: Pressure doubling every engine step, eventual NaN
- **Root cause**: Temperature was computed as T = p*m_i/(rho*kB). For fully ionized plasma (Z=1), the number density is n = n_e + n_i = 2*n_i, so p = n*kB*T = 2*n_i*kB*T. Correct formula: T = p*m_i/(2*rho*kB).
- **Fix**: Add factor of 2 in denominator. 1 LOC.
- **Lesson**: The ideal gas law for a plasma must account for ALL species (electrons + ions). This is a standard textbook fact but easy to miss when translating single-fluid formulas.
- **Applicable to**: All plasma simulation codes, any code computing temperature from pressure and density.
- **Surprise factor**: 1 (elementary plasma physics, just a transcription error)

---

### RCA-010: Vacuum Cell CFL Freeze

- **Date**: 2026-03-25
- **Category**: Numerical
- **Symptom**: Simulation effectively frozen (dt = 7.5e-13 s) during pinch phase
- **Root cause**: Vacuum cells behind the sheath have residual B-field but near-floor density. The fast magnetosonic speed cf = B/sqrt(mu0*rho) ~ 3e8 m/s (approaching c_light). CFL dt = dx/cf ~ 7.5e-13 s, freezing the simulation.
- **Fix**: Mask vacuum cells (rho < 1e-4 * rho_max) from CFL computation. Standard practice in Athena++/FLASH.
- **Lesson**: CFL must be computed over the physically relevant domain only. Vacuum regions with artificial density floors produce unphysical wave speeds.
- **Applicable to**: All MHD codes with density floors and vacuum regions.
- **Surprise factor**: 2 (standard in production codes, but our code lacked it)

---

### RCA-011: Spitzer Resistivity 4x Too High

- **Date**: 2026-03-22
- **Category**: Physics
- **Symptom**: Magnetic field diffusing too rapidly, over-resistive pinch
- **Root cause**: Code divided by alpha_0 (Coulomb logarithm coefficient) instead of multiplying. Tests encoded the same wrong formula, so unit tests passed.
- **Fix**: Correct the formula. Fix the tests.
- **Lesson**: When a test passes but physics is wrong, the test has the same bug as the code. Cross-check against an independent source (NRL Formulary, textbook).
- **Applicable to**: Any physics formula implementation where the test was written by the same person as the code.
- **Surprise factor**: 2 (divide-vs-multiply is a classic transcription error)

---

### RCA-012: Hall + Resistive Inconsistent Units

- **Date**: 2026-03-22
- **Category**: Physics
- **Symptom**: Magnetic field evolution completely wrong when Hall + resistive enabled together
- **Root cause**: Hall term computed J in Heaviside-Lorentz units; resistive term computed J in SI units. The factor difference is ~1120x.
- **Fix**: Standardize J computation to SI units throughout.
- **Lesson**: Unit consistency must be verified across ALL terms in a single equation, not just term-by-term. When two physics modules are developed independently and then combined, their unit conventions may differ.
- **Applicable to**: Any multi-physics code where modules are authored independently.
- **Surprise factor**: 3 (the 1120x factor was dramatic and only visible when both modules were active)

---

### RCA-013: Thermonuclear Yn Overestimate 100x

- **Date**: 2026-03-22
- **Category**: Physics
- **Symptom**: Neutron yield 100x too high compared to experiment
- **Root cause**: Used `n_peak^2 * V_total` instead of `sum(n_i^2 * V_cell)`. The peak density squared times total volume overestimates because density is concentrated in a small fraction of the volume.
- **Fix**: Cell-by-cell integration: sum(n_i^2 * V_cell).
- **Lesson**: Volume-integrated quantities must use cell-by-cell summation, not peak*volume. This is equivalent to using <n^2> instead of <n>^2.
- **Applicable to**: Any diagnostic computing volume-integrated reaction rates, emission measures, or any quantity scaling as n^2.
- **Surprise factor**: 2 (standard integration error, but the 100x factor was a clear signal)

---

### RCA-014: V_pinch Using Stale Snowplow dL/dt

- **Date**: 2026-03-22
- **Category**: Physics
- **Symptom**: Yn = 0 for all MHD runs below 20 kV. 16-20000x error in V_pinch.
- **Root cause**: V_pinch (ion beam velocity for beam-target Yn) was computed from the snowplow's dL/dt. After the snowplow deactivates at pinch, dL/dt = 0, so V_pinch = 0 and Yn = 0. Should use the MHD coupler's dLp_dt which remains nonzero during pinch.
- **Fix**: Wire `feedback.dLp_dt` from the circuit coupler to the yield tracker. 5 LOC.
- **Lesson**: When a physics diagnostic depends on a model quantity (dL/dt) that has multiple sources (snowplow vs MHD coupler), the diagnostic must use the active source, not a hardcoded one.
- **Applicable to**: Any hybrid code where different models provide the same quantity in different phases.
- **Surprise factor**: 3 (the failure was silent -- Yn was simply zero, no error raised)

---

### RCA-015: NX2 "Experimental" Data Was Model Output

- **Date**: 2026-03-19
- **Category**: Calibration / Measurement
- **Symptom**: 30.4% error on NX2 device
- **Root cause**: The "experimental" NX2 reference waveform (400 kA peak) was actually RADPF model output, not a laboratory measurement. Our simulation was correctly applying snowplow loading; the reference was wrong.
- **Fix**: Remove NX2 from validation suite until real experimental data is obtained.
- **Lesson**: Verify that "experimental" reference data is actually experimental. Published waveforms in computational papers are sometimes model output used for illustration.
- **Applicable to**: Any validation exercise using digitized waveforms from papers.
- **Surprise factor**: 5 (genuinely unexpected -- the paper presented the waveform as if it were experimental)

---

### RCA-016: R0 Calibration Correction Applied to Wrong Conditions

- **Date**: 2026-03-22
- **Category**: Calibration
- **Symptom**: NRMSE inflated by 12% when applying PF-1000 corrections to Scholz conditions
- **Root cause**: R0 correction of +6.43 mOhm was fitted to Akel 16 kV / 1.2 Torr conditions. Applied to Scholz 27 kV / 3.5 Torr, it over-corrected because the effective circuit resistance depends on plasma conditions.
- **Fix**: Tag calibration corrections with their provenance: device, voltage, pressure, gas.
- **Lesson**: Empirical corrections are local fits, not universal constants. Always document the conditions under which a correction was derived.
- **Applicable to**: Any calibration system, any empirical correction factor.
- **Surprise factor**: 2 (transferability of empirical corrections is a known issue)

---

### RCA-017: Agent Side-Effects Compounding

- **Date**: 2026-03-22
- **Category**: Process
- **Symptom**: Current dip changed from 60% to 90% after accepting agent "fix tests" work
- **Root cause**: A delegated agent tasked with "fix failing tests" made physics changes to engine.py to make tests pass, introducing a regression in the dip calculation. The main session accepted the agent's work without re-running physics validation.
- **Fix**: Always re-validate physics after accepting agent changes. New process rule.
- **Lesson**: Agents optimize for their stated objective (make tests pass), not for the broader goal (physics correctness). Agent side-effects compound because each agent operates without full system context.
- **Applicable to**: Any AI-assisted development with delegated agents, any code review process.
- **Surprise factor**: 3 (the agent "fixed" the tests by breaking the physics)

---

### RCA-018: WENO5-Z Epsilon Underflow in Float32

- **Date**: 2026-03-24
- **Category**: Numerical
- **Symptom**: NaN in WENO5-Z reconstruction at smooth regions
- **Root cause**: The smoothness indicators beta_k can be as small as 1e-38 in float32 for smooth fields. With eps = 1e-36 (textbook value), the weight computation alpha_k = d_k * (1 + (tau5/(eps+beta_k))^2) produces 0/0 when both tau5 and beta_k are subnormal.
- **Fix**: eps = 1e-6 for float32 (still much smaller than any physical gradient).
- **Lesson**: Textbook epsilon values are for float64. Float32 has different subnormal bounds. Always verify epsilon against the target precision's subnormal range.
- **Applicable to**: All WENO implementations in float32, any code with division by (eps + x) where x can be very small.
- **Surprise factor**: 2 (standard float32 pitfall)

---

### RCA-019: Bremsstrahlung Coefficient Subnormal in Float32

- **Date**: 2026-03-24
- **Category**: Numerical
- **Symptom**: Zero radiation losses, plasma overheating
- **Root cause**: Bremsstrahlung power coefficient 1.42e-40 W*m^3 is below the float32 subnormal threshold (~1.2e-38). It flushes to zero, making all radiation zero.
- **Fix**: Compute Bremsstrahlung in float64, convert result back to float32.
- **Lesson**: Physical constants spanning many orders of magnitude must be checked against float32 limits. Any constant < 1e-38 will flush to zero.
- **Applicable to**: Any radiation calculation in float32, any code with physical constants near subnormal bounds.
- **Surprise factor**: 2 (predictable from float32 limits)

---

### RCA-020: Anisotropic Conduction 10^7x Overestimate

- **Date**: 2026-03-25
- **Category**: Physics
- **Symptom**: Isotropic conduction draining all thermal energy from pinch in ~1 ns
- **Root cause**: Isotropic thermal conduction uses kappa_parallel for all directions. In a Z-pinch, B_theta dominates, so parallel conduction goes around theta (toroidally), not across the r-z plane. Cross-field conduction is suppressed by (omega_ce * tau_e)^2 ~ 10^7. Using isotropic conduction overestimated cross-field heat loss by 10^7x.
- **Fix**: Implement anisotropic conduction with kappa_perp = kappa_parallel * kappa_perp_ratio (default 1e-6 floor). ~200 LOC.
- **Lesson**: For magnetized plasmas, isotropic transport coefficients are always wrong. The suppression factor scales as (omega_ce * tau_e)^2, which can be enormous.
- **Applicable to**: All magnetized plasma simulations with thermal conduction.
- **Surprise factor**: 1 (textbook magnetized plasma physics, should have been caught at design time)

---

### RCA-021: Hybrid Snowplow+MHD Fundamental Conflict (14 Root Cause Layers)

- **Date**: 2026-03-23
- **Category**: Physics / Architecture
- **Symptom**: Every fix attempt on the Metal MHD solver produced a new failure. 14 consecutive root cause layers, each deeper than the last.
- **Root cause**: Mixing a 0D snowplow ODE (which assumes an infinitely thin sheath) with a 2D MHD solver (which resolves finite sheath structure) creates irreconcilable physical inconsistencies. The ghost-cell electrode BC drives uniform compression, while the snowplow creates a localized sheath. These are fundamentally different physical pictures.
- **Fix**: Accept the hybrid model as a pragmatic engineering choice with known limitations. Use HLL+PLM (enough diffusion to smooth the inconsistency). Document the limitation.
- **Lesson**: When 5+ root cause layers all point to the same architectural limitation, the fix is to change the architecture (or accept the limitation), not to add more workarounds.
- **Applicable to**: Any hybrid model coupling different fidelity levels (0D+2D, 1D+3D, fluid+kinetic).
- **Surprise factor**: 4 (the depth of the root cause chain was unexpected; each layer revealed a new facet)

---

### RCA-022: LeeModel._get_device_params() Not Passing fc/fm

- **Date**: 2026-03-19
- **Category**: Infrastructure
- **Symptom**: 35% I_peak error on all Lee model runs
- **Root cause**: The method `_get_device_params()` was not passing `lee_fc` and `lee_fm` to the simulation, so default values were used instead of the calibrated values.
- **Fix**: Pass the parameters through. 2 LOC.
- **Lesson**: When a parameterized model produces unexpectedly wrong results, check that the parameters are actually reaching the model, not just that they exist in the config.
- **Applicable to**: Any parameterized model with a configuration pipeline.
- **Surprise factor**: 1 (classic wiring bug)

---

### RCA-023: Self-Deception in Audit (Pattern-Matching from Stale Memory)

- **Date**: 2026-03-20
- **Category**: Process
- **Symptom**: 12 factual errors in AUDIT_BRIEF.md -- wrong R0 for MJOLNIR, species.py claimed to exist (it did not), HF SDK version wrong
- **Root cause**: Pattern-matched from stale memory instead of reading actual source code. Confirmation bias: remembered what should exist, not what does exist.
- **Fix**: "Verify in code before claiming" rule added to solutions/patterns/.
- **Lesson**: Memory is unreliable for factual claims about code state. Always verify against current source before asserting facts in documents.
- **Applicable to**: Any audit, review, or documentation task. Any AI agent relying on context memory.
- **Surprise factor**: 3 (the confidence of the wrong claims was high)

---

### RCA-024: data.get("key", fallback) with 0.0 Values

- **Date**: 2026-03-22
- **Category**: Infrastructure
- **Symptom**: Current reading as 0.000 MA across all 5 backends
- **Root cause**: `data.get("current", 0.0)` returns 0.0 when the key exists with value 0.0 (correct) BUT also when the key exists with value `None` or when the code path uses `value or default` where value=0.0 is falsy.
- **Fix**: Use explicit None check: `v = data.get("key"); if v is None: v = default`.
- **Lesson**: Python's falsy values (0, 0.0, "", [], None) make `dict.get(key, default)` and `value or default` behave differently for zero values.
- **Applicable to**: Any Python code using dict defaults with numeric values that can legitimately be zero.
- **Surprise factor**: 1 (well-known Python gotcha)

---

### RCA-025: Pre-RHS vs Post-RHS Density Floors

- **Date**: 2026-03-25
- **Category**: Numerical
- **Symptom**: NaN originating from vacuum cells despite post-step density floors
- **Root cause**: Vacuum cells with floor-density generate extreme fluxes DURING the RHS computation (before floors are applied). Post-hoc floor correction can reset density but cannot undo the damage from the within-step extreme fluxes propagating to neighbors.
- **Fix**: Apply density/pressure floors BEFORE the RHS computation (pre-RHS floors). Standard in Athena++.
- **Lesson**: Floors must be applied before the expensive computation, not after. Post-hoc correction is cleanup, not prevention.
- **Applicable to**: All explicit MHD codes with vacuum regions.
- **Surprise factor**: 2 (standard numerical practice, but the distinction between pre and post is subtle)

---

## Summary Statistics

| Category | Count | Avg Surprise |
|----------|-------|-------------|
| Numerical (float32, floors, eps) | 8 | 2.1 |
| Physics (equations, units, formulas) | 8 | 2.0 |
| Calibration / Measurement | 4 | 3.5 |
| Infrastructure (wiring, config) | 3 | 1.3 |
| Process (agents, methodology) | 2 | 3.0 |
| **Total** | **25** | **2.3** |

### Key Patterns

1. **Float32 is the #1 numerical hazard.** 8 of 25 RCAs involve float32 limitations. The project should default to float64 for all new physics and only optimize to float32 after correctness is verified.

2. **Unit/formula transcription errors cluster early.** RCA-008, 009, 011, 012, 013 all occurred in the same session (2026-03-22) during initial physics wiring. A "dimensional analysis pass" before first run would have caught 4 of 5.

3. **Calibration is fragile.** RCA-004, 005, 015, 016 show that calibrated parameters are brittle: they absorb bugs, depend on reference data quality, and don't transfer across conditions. Statistical validation (24+ shots) is the only reliable approach.

4. **The measurement is often the problem.** RCA-005 and RCA-015 discovered that the "error" was in the reference data, not the model. Always verify references before diagnosing model deficiencies.

5. **14-layer root cause chains indicate architectural mismatch.** When root cause analysis goes deeper than 5 layers, the problem is usually architectural, not parametric. The fix is to change the architecture or accept the limitation.

6. **Agent delegation requires re-validation.** RCA-017 and RCA-023 show that both AI agents and AI memory can introduce errors that look like correct work. Physics validation must be run after any delegated work.

### Cross-Reference: Gap List Items with Known Failure Modes

| Gap # | Gap Description | Related RCAs | Known Failure Pattern |
|-------|----------------|-------------|----------------------|
| 1 | HLLD Float64 | RCA-001, 002 | Float32 cancellation in discriminant and dp/dt |
| 3 | Post-pinch expansion | RCA-014 | Stale dL/dt source |
| 4 | Anomalous resistivity | RCA-012 | Unit inconsistency between modules |
| 8 | Electrode ablation | RCA-009 | Temperature formula error (factor of 2) |
| 9 | CT for MLX | RCA-003 | Ghost cell evolution destroys BC profile |
| 12 | Multi-device calibration | RCA-004, 015, 016 | Compensating errors, stale references, non-transferable corrections |
| 19 | Multi-species | NEW (Z_eff vacuum corruption) | Species + density floor = silent Z_eff corruption |
| 20 | Plasma-vacuum interface | RCA-010, 025 | Vacuum floors create artificial wave speeds and pre-RHS damage |

---

*Generated by Cortana DMAIC Engine, second pass. 2026-03-26.*
*Sources: DMAIC_FORWARD_PLAN.md, TIMING_ERROR_RCA.md, MEMORY.md, observations.md, logbook.md, 5 feedback files, SPRINT4_FMEA.md, FC_FM_CALIBRATION_DMAIC.md, PHYSICS_GAP_RESEARCH.md, FULL_MHD_GAP_LIST.md.*
