# Sprint 4 Validation Review: MLX Solver

**Date**: 2026-03-24
**Author**: dpf-validation-engineer (Cortana)
**Status**: Pre-execution review -- go/no-go assessment
**Method**: Cross-reference of GAP_ANALYSIS, FMEA, RISK_ANALYSIS, METAL_V2_DOD, PHASE_B_RESEARCH against current solver state

---

## 1. Current State Assessment

| Metric | Value | Source |
|--------|-------|--------|
| MLX tests passing | 361 | Session report |
| FMEA bugs fixed | 6 | Session report |
| PF-1000 I_peak (engine run) | 0.958 MA | Session report (20% below 1.2 MA target) |
| Simulation stall time | t = 1.37 us | CFL issue under investigation |
| Temperature factor-2 bug | Fixed | Session report |
| MLX vs PyTorch speedup | 2.68x on 32x64 | Session report |

**Assessment**: The solver runs and produces physically reasonable output (0.958 MA is in the right order of magnitude, not garbage). The 20% I_peak deficit and CFL stall are the two blockers for Sprint 4 completion.

---

## 2. Blocker Analysis: CFL Stall at t = 1.37 us

### Root Cause Candidates (ranked by FMEA RPN)

| # | Candidate | RPN | Evidence |
|---|-----------|-----|----------|
| 1 | FM-2.1: CFL returns oversized dt on first step (vacuum + electrode BC) | 120 | Stall at 1.37 us suggests crash occurs during rundown, not initialization. Partially rules this out unless dt oscillates. |
| 2 | FM-3.1: Duplicate mhd_rhs implementations | 392 | Highest RPN. If the active mhd_rhs uses zero-padded boundary fluxes (timestepper version), the electrode cells stagnate, creating increasingly extreme gradients that eventually shrink dt to zero. |
| 3 | FM-3.2: WENO5-Z boundary cell stagnation | 252 | 2-3 cells near electrode get zero flux divergence. Frozen boundary layer grows progressively more unphysical, eventually demanding dt -> 0. |
| 4 | FM-5.4: HLLD/HLL tangential index mismatch | 168 | If HLL fallback triggers in axial direction, swapped tangential components create wrong fluxes that destabilize CFL. |
| 5 | N4 (RISK_ANALYSIS): Dual-energy switching changes wave speed mid-RK-stage | Medium | Pressure switching between stages could cause effective CFL > 1 on some stages. |

### Most Likely Root Cause

FM-3.1 (RPN 392) + FM-3.2 (RPN 252) compound: the active `mhd_rhs` in `mlx_timestepper.py` uses zero-padded boundary fluxes, causing the 2-3 cells near the electrode to receive no flux update. Over ~1000 steps to t=1.37 us, these cells develop extreme gradients. The fast magnetosonic speed at the boundary-interior interface spikes, driving CFL dt toward zero. The simulation doesn't crash (no NaN) -- it stalls because dt becomes vanishingly small.

### Recommended Fix Sequence

1. **Unify mhd_rhs** (FM-3.1): Delete the timestepper's version, import from mlx_riemann.py. This is the highest-RPN item. ~20 LOC, ~1 hour.
2. **PLM fallback at boundaries** (FM-3.2): For the 2-3 cells where WENO5-Z lacks stencil, compute PLM fluxes and merge with WENO5-Z interior. ~30 LOC, ~2 hours.
3. **Fix HLL tangential indices** (FM-5.4): Align dim=1 tangential mapping between HLLD and HLL. ~5 LOC, ~30 min.
4. **Update Srho in bremsstrahlung** (FM-4.2, RPN 336): Add entropy update to radiation source. ~5 LOC, ~30 min.
5. **Conservative CFL using max(c_f(p_E), c_f(p_S))** (N4): Prevent mid-stage CFL violation. ~10 LOC, ~1 hour.

**Total pre-sprint fix effort**: ~70 LOC, ~5 hours.

---

## 3. DoD Criterion Assessment (M1-M8)

### M1: No Negative Pressure

| Aspect | Assessment |
|--------|-----------|
| Testable now? | YES -- entropy formulation + dual-energy implemented, P_FLOOR enforced |
| Blocker | None for unit-test-level validation. Full discharge test blocked by CFL stall. |
| Fix effort | 0 LOC (mechanism exists) |
| Risk of failure | LOW. Dual-energy switching + P_FLOOR is defense-in-depth. Electrode conditions (beta=7e-7) produce positive p_S by construction. |
| Go/No-Go | **GO** |

### M2: PF-1000 I_peak Within 10% of 1.2 MA

| Aspect | Assessment |
|--------|-----------|
| Testable now? | NO -- simulation stalls at 1.37 us (needs to reach ~5 us for I_peak) |
| Blocker | CFL stall. Also: GAP_ANALYSIS P1 (.geom not attached) and FM-1.3 (coupling_interface stub). |
| Fix effort | CFL fix ~5h. P1 .geom attachment ~5 LOC. Coupling interface: engine coupler handles Lp independently -- verify, not fix. |
| Risk of failure | HIGH. Current I_peak = 0.958 MA (20% low). After CFL fix, I_peak may improve (boundary cells contribute to correct B-field profile) or may need circuit parameter tuning. The 20% deficit could be from missing back-EMF wiring (FM-1.3), incorrect geometric sources (FM-3.3), or incorrect Lp computation. |
| Go/No-Go | **CONDITIONAL GO** -- proceed after CFL fix, with understanding that I_peak tuning may require 1-2 additional debug iterations |

### M3: Mass Conservation < 5%

| Aspect | Assessment |
|--------|-----------|
| Testable now? | YES for short runs; NO for full discharge (CFL stall) |
| Blocker | CFL stall for full discharge |
| Fix effort | 0 LOC (conservative formulation should handle this). Write test: ~30 LOC. |
| Risk of failure | LOW. The solver uses conservative variables throughout the Riemann solve. Outflow BCs allow some mass loss, which the 5% threshold accommodates. |
| Go/No-Go | **GO** |

### M4: Energy Conservation < 10%

| Aspect | Assessment |
|--------|-----------|
| Testable now? | PARTIALLY -- short-run tests exist, full discharge blocked |
| Blocker | CFL stall. Also: FM-4.2 (Srho not updated by bremsstrahlung, RPN 336). N6 (blending zone drift, ~1% over full discharge). |
| Fix effort | FM-4.2 fix: ~5 LOC. Test: ~50 LOC. |
| Risk of failure | MEDIUM. Two systematic energy leaks identified: (1) entropy tracer not tracking radiation losses, (2) blending zone drift. Combined, these could reach 5-8% over a full discharge. The 10% threshold provides margin but not much. |
| Go/No-Go | **CONDITIONAL GO** -- fix FM-4.2 first, then validate |

### M5: No NaN Propagation

| Aspect | Assessment |
|--------|-----------|
| Testable now? | YES for short runs (tested up to 50 steps); NO for full discharge |
| Blocker | CFL stall. FM-2.1 (CFL oversized first step) is a NaN risk. N2 (HLLD float32 overflow at electrode) is a NaN risk. |
| Fix effort | Existing NaN guards (HLL fallback, velocity clamping) address most paths. Full discharge NaN test: ~10 LOC. |
| Risk of failure | MEDIUM. Short runs pass. Long integration through pinch (beta~0.01 to beta~1e-4) is the danger zone -- HLLD intermediate states at beta transitions haven't been exercised. |
| Go/No-Go | **CONDITIONAL GO** -- depends on CFL fix enabling full discharge |

### M6: Completes 5 Phases (t > 2 * t_peak)

| Aspect | Assessment |
|--------|-----------|
| Testable now? | NO -- stalls at 1.37 us, need ~12 us |
| Blocker | CFL stall is the direct blocker |
| Fix effort | Entirely dependent on CFL fix. If fix works, test is ~30 LOC. |
| Risk of failure | HIGH. Even after CFL fix, the pinch phase (t~4-5 us) introduces extreme conditions. Post-pinch expansion may trigger new instabilities. Multiple failure modes between t=1.37 us and t=12 us. |
| Go/No-Go | **BLOCKED** until CFL fix verified |

### M7: Float32 on Metal GPU

| Aspect | Assessment |
|--------|-----------|
| Testable now? | YES |
| Blocker | None |
| Fix effort | 0 LOC |
| Risk of failure | LOW (already satisfied -- MLX uses float32 throughout) |
| Go/No-Go | **GO** (already passing) |

### M8: div(B) = 0 to Relative 1e-6

| Aspect | Assessment |
|--------|-----------|
| Testable now? | PARTIALLY -- CT module exists but is NOT wired into the timestepper (GAP_ANALYSIS 4.6) |
| Blocker | CT not called during mhd_rhs. Need to wire it in OR demonstrate empirically that the Riemann solver approach maintains div(B) without CT. |
| Fix effort | Wire CT: ~20-40 LOC. Alternatively, measure div(B) on existing runs: ~30 LOC test. |
| Risk of failure | MEDIUM. In axisymmetric (r,z) geometry, B_theta is cell-centered (no CT needed), and B_r/B_z are updated through Riemann solver fluxes which preserve div(B) for directionally-split sweeps. The question is whether unsplit 2D updates maintain this. Empirical measurement needed. |
| Go/No-Go | **CONDITIONAL GO** -- measure first, wire CT if needed |

### M-Criteria Summary

| Status | Criteria |
|--------|----------|
| GO | M1, M3, M7 |
| CONDITIONAL GO | M2, M4, M5, M8 |
| BLOCKED | M6 |

**All conditionals gate on CFL stall resolution.**

---

## 4. DoD Criterion Assessment (S1-S9)

| ID | Testable? | Blocker | Fix Effort | Risk | Go/No-Go |
|----|-----------|---------|------------|------|----------|
| S1 | NO | CFL stall + M2 | 40 LOC test | HIGH (depends on I_peak accuracy) | BLOCKED |
| S2 | NO | CFL stall + M6 | 20 LOC test | HIGH (requires correct Lp/back-EMF) | BLOCKED |
| S3 | NO | CFL stall + circuit coupling | 10 LOC test | HIGH (voltage spike requires correct dLp/dt) | BLOCKED |
| S4 | NO | CFL stall + M6 | 150 LOC test | MEDIUM (3 devices, any could fail) | BLOCKED |
| S5 | YES | None | 0 LOC (existing test passes) | LOW | **GO** (already satisfied) |
| S6 | YES | None | 0 LOC (existing test passes) | LOW | **GO** (already satisfied) |
| S7 | PARTIALLY | Need N=256 test | 40 LOC test | LOW (Sod is well-exercised) | **GO** |
| S8 | NO | Need convergence study | 80 LOC test | MEDIUM (diffusion convergence untested) | **GO** (unblocked, just needs test) |
| S9 | NO | Need benchmark script | 200 LOC | MEDIUM (MLX vs C++ is uncertain) | **GO** (unblocked) |

---

## 5. Revised Sprint 4 Execution Sequence

### Critical Path Reorder

The original plan assumed CFL stall doesn't exist. It does. The revised sequence separates CFL-blocked work from unblocked work.

### Phase A: Pre-Sprint Fixes (Days 1-2, CRITICAL PATH)

| Priority | Action | File | LOC | Blocks |
|----------|--------|------|-----|--------|
| P0 | Unify mhd_rhs (FM-3.1, RPN 392) | mlx_timestepper.py, mlx_riemann.py | 20 | CFL fix |
| P0 | PLM fallback at WENO5-Z boundaries (FM-3.2, RPN 252) | mlx_riemann.py | 30 | CFL fix |
| P1 | Fix HLL tangential indices (FM-5.4, RPN 168) | mlx_timestepper.py | 5 | Correctness |
| P1 | Update Srho in bremsstrahlung (FM-4.2, RPN 336) | mlx_sources.py | 5 | M4 |
| P1 | Attach .geom to MLX solver in engine.py (GAP P1) | engine.py | 5 | M2, M3, M4, M6 |
| P2 | Conservative CFL (max of both pressure estimates) | mlx_timestepper.py | 10 | M5 stability |
| -- | **Validation gate**: re-run PF-1000 engine test | -- | -- | Must reach t > 5 us |

**Estimated effort**: 75 LOC, 6-8 hours. If the PF-1000 run does not reach t > 5 us after P0 fixes, STOP and diagnose before proceeding. Do not attempt WU-4.1.

### Phase B: Parallel Unblocked Work (Days 2-4, while CFL fix is verified)

These items have zero dependency on CFL stall resolution:

| WU | Deliverable | DoD | LOC | Hours |
|----|-------------|-----|-----|-------|
| WU-4.3 | mlx_benchmark.py | S9 | 200 | 4 |
| WU-4.4 | mx.compile() tuning | S9 | 30 | 2 |
| S7 test | Sod at N=256, L1 < 1e-2 | S7 | 40 | 1 |
| S8 test | Diffusion convergence (4 resolutions) | S8 | 80 | 3 |
| M8 measure | div(B) measurement on existing runs | M8 | 30 | 1 |
| CT wire | Wire mlx_ct.py if M8 fails | M8 | 40 | 2 |

**Total**: ~420 LOC, ~13 hours. These can all run in parallel with CFL diagnosis.

### Phase C: PF-1000 Validation (Days 5-7, after CFL fix confirmed)

| WU | Deliverable | DoD | LOC | Hours |
|----|-------------|-----|-----|-------|
| WU-4.1a | test_mlx_pf1000.py scaffold | M2, M5, M6 | 150 | 4 |
| WU-4.1b | Full discharge debug | M5, M6 | 50 | 8 |
| WU-4.1c | M1-M8 assertions | M1-M8 | 100 | 3 |
| WU-4.1d | Fix any failing M-criteria | varies | 100 | 8 |

### Phase D: Multi-Device + Acceptance (Days 8-10)

| WU | Deliverable | DoD | LOC | Hours |
|----|-------------|-----|-----|-------|
| WU-4.2 | test_mlx_multidevice.py | S4 | 150 | 4 |
| WU-4.1e | S1-S3 assertions | S1, S2, S3 | 70 | 3 |
| WU-4.5 | test_mlx_acceptance.py | ALL | 200 | 5 |
| Final | Fix remaining failures | varies | 100 | 8 |

---

## 6. Realistic Timeline Estimate

| Phase | Calendar Days | Confidence |
|-------|--------------|------------|
| Phase A (pre-sprint fixes) | 2 days | 80% -- root cause analysis may take longer if FM-3.1 isn't the CFL culprit |
| Phase B (unblocked parallel work) | 3 days (overlaps Phase A) | 95% -- these are straightforward tests |
| Phase C (PF-1000 validation) | 3-5 days | 50% -- pinch phase may surface new failure modes |
| Phase D (multi-device + acceptance) | 3-4 days | 60% -- multi-device tests may reveal device-specific tuning |
| **Total** | **9-14 days** | **40-50% on schedule** |

**Why 40-50% confidence**: Phase C is the wild card. History with Metal v1 shows that getting a DPF simulation through the pinch phase in float32 is the hardest part. The CFL stall is likely just the first of several failure modes. Each new failure (NaN at pinch, pressure corruption at beta transition, circuit oscillation post-pinch) requires ~4-8 hours of diagnosis and fix. Budget for 2-3 such cycles.

---

## 7. Risk Items Not Addressed by Current Plan

| Risk | Source | Impact | Mitigation Gap |
|------|--------|--------|---------------|
| FM-3.3: Geometric source double-counting | FMEA RPN 147 | Wrong pinch timing | Not in any work unit. Needs call-chain audit before Phase C. |
| N3: Entropy tracer drift at shocks | RISK_ANALYSIS | 5-10% pressure error post-shock | sync_threshold tuning not scheduled. |
| N5: Cylindrical axis singularity | RISK_ANALYSIS | Axis oscillations | Not tested. Need uniform-state stability test at axis. |
| R9: CT insufficient at production resolution | PHASE_B_RESEARCH | div(B) grows | Deferred to "measure then fix" but no timeline if measurement fails. |
| I1: backend="mlx" vs "metal" naming | RISK_ANALYSIS | User confusion | No action planned. Should add log message. |

---

## 8. Recommendation

### Ship Decision: Ship with Known Limitations

**Rationale**: Blocking until all M1-M8 pass could take 3-4 weeks given the uncertainty around pinch-phase stability. The solver already provides value at its current state:

- 361 tests passing (numerical correctness established for short runs)
- 2.68x faster than PyTorch MPS
- Sod/Brio-Wu standard benchmarks pass (S5, S6)
- Float32 entropy-based dual-energy is working (M1 mechanism verified)

### Recommended Ship Criteria (Tiered)

**Tier 1 -- Minimum viable ship (target: 7 days)**:
- CFL stall fixed (simulation advances past 5 us)
- M1, M3, M5, M7 pass (no negative pressure, mass conservation, no NaN, float32)
- S5, S6, S7, S8 pass (cross-backend parity, shock tubes, convergence)
- WU-4.3 benchmark script exists

**Tier 2 -- Full Sprint 4 (target: 14 days)**:
- M2, M4, M6, M8 pass (I_peak, energy conservation, full discharge, div(B))
- S1, S2, S3, S4 pass (waveform, current dip, voltage spike, multi-device)
- WU-4.5 acceptance test green

**Tier 3 -- Publication quality (target: 21+ days)**:
- S9 (faster than Athena++) validated with full grid scaling
- Phase 2 DoD criteria (P2-1 through P2-6) researched and documented
- Cross-backend comparison with Python cylindrical + Athena++ at matched resolution

### Specific Action Items (Ordered)

1. Fix FM-3.1 (unify mhd_rhs) -- highest RPN, most likely CFL root cause
2. Fix FM-3.2 (PLM boundary fallback) -- second highest CFL contributor
3. Verify CFL fix: PF-1000 must reach t > 5 us
4. In parallel: write S7, S8 tests + benchmark script + measure div(B)
5. After CFL fix confirmed: write test_mlx_pf1000.py, iterate on M-criteria
6. Multi-device tests last (they depend on single-device working)

### What Not to Do

- Do not attempt WU-4.4 (mx.compile tuning) before WU-4.3 (benchmark) provides a measurement baseline. Optimization without measurement is noise.
- Do not write WU-4.5 (acceptance test) before WU-4.1 (PF-1000 test) is green. The acceptance test is a superset; building it before the subset passes wastes time on failing assertions.
- Do not attempt multi-device validation (WU-4.2) before single-device (WU-4.1) passes. Device-specific failures should not be conflated with solver-general failures.
- Do not invest in CT wiring (M8) before measuring div(B) empirically. The axisymmetric geometry may not need explicit CT.

---

## 9. Outstanding FMEA Items Requiring Resolution Before Sprint 4

| FM | RPN | Status | Action |
|----|-----|--------|--------|
| FM-3.1 (dual mhd_rhs) | 392 | OPEN | Fix P0 |
| FM-4.2 (Srho not in bremsstrahlung) | 336 | OPEN | Fix P1 |
| FM-3.2 (WENO5 boundary stagnation) | 252 | OPEN | Fix P0 |
| FM-5.4 (HLL tangential indices) | 168 | OPEN | Fix P1 |
| FM-3.3 (geometric source double-count) | 147 | OPEN | Audit call chain, fix if active |
| FM-5.2 (HLLD D_L degeneration) | 140 | DEFERRED | Test gate in Sprint 4 (monitor LF fallback rate) |
| FM-1.3 (coupling stub) | 120 | DEFERRED | Engine coupler handles Lp independently |
| FM-2.1 (CFL oversized first step) | 120 | OPEN | Add dt_max safety cap |

**Items with RPN >= 200 must be fixed before Sprint 4 starts. Items with RPN 120-200 should be fixed or test-gated.**
