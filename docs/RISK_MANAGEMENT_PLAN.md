# DPF-Unified Risk Management Plan

**Date**: 2026-03-26 | **Scope**: All research scaffolds + current system state | **Fidelity**: 7.5/10 -> 9.0/10 target

---

## 1. Risk Registry

| ID | Category | Risk | P | I | Score | Owner | Mitigation | Contingency | Status |
|----|----------|------|---|---|-------|-------|------------|-------------|--------|
| R01 | Physics | AMR cylindrical refluxing wrong r-weighting — no reference impl exists | 4 | 5 | 20 | Dev | Standalone reflux test with analytical solution before integration | Accept 0.3-3% mass drift; defer refluxing to Phase C | OPEN |
| R02 | Physics | PIC ghost-cell NaN propagation kills simulation in 1 step | 5 | 5 | 25 | Dev | NaN guard in interpolation (8 LOC); mask ghost cells before PIC reads | Disable PIC coupling at boundaries | OPEN |
| R03 | Physics | Hall MHD unit bug (mu_0 missing) — Hall has NEVER been physically active | 5 | 4 | 20 | Dev | Fix in H1 (30 LOC); write quantitative dispersion test first (TDD) | Hall term is optional; disable until fixed | OPEN |
| R04 | Physics | PIC Boris pusher non-relativistic — superluminal particles in ~6000 steps at DPF E-fields | 4 | 5 | 20 | Dev | Add relativistic gamma factor (15 LOC) + velocity cap at 0.99c (6 LOC) | Limit PIC to low-field regions only | OPEN |
| R05 | Physics | PIC Esirkepov + reflecting BC incompatibility — charge conservation broken at boundaries | 4 | 4 | 16 | Dev | Two-segment deposition for reflected particles (40 LOC) or switch to absorbing BC | Use CIC deposition near boundaries | OPEN |
| R06 | Physics | PIC E-field missing Hall term — trajectories wrong by up to 100% at pinch | 4 | 4 | 16 | Dev | Measure E_Hall/E_conv in V3; add Hall E if >10% (~30 LOC). Requires Hall fix R03. | Document as known limitation; bound error | OPEN |
| R07 | Physics | AMR conservation error without refluxing: 0.3-3% cumulative, NOT 1e-6/step as scaffold claims | 3 | 4 | 12 | Dev | Move refluxing to Phase A (add ~150 LOC) or limit to smooth-flow tests | Accept error with explicit monitoring; flag in results | OPEN |
| R08 | Performance | HLL/HLLS CPU round-trip: 45-55% of step time wasted on GPU sync | 5 | 3 | 15 | Dev | Port to pure MLX (OPT-1/2, ~240 LOC). HLLS has zero float32 risk. | Keep CPU path; accept 2x slower | OPEN |
| R09 | Performance | Whistler CFL 1000x mismatch in early rundown kills Hall MHD performance | 4 | 3 | 12 | Dev | Cap sub-cycles at 100 (not 20); consider implicit Hall (backward Euler, +50 LOC) | Disable Hall in pre-pinch phase | OPEN |
| R10 | Architecture | mx.compile() recompiles on AMR regrid if block count changes shape | 3 | 3 | 9 | Dev | Pre-allocate MAX_BLOCKS=16 with masking; verify no recompile via compile stats | Use sequential block processing (slower but simpler) | OPEN |
| R11 | Architecture | AMR block size 32x64 exceeds L1 cache (80KB > 48KB on M3); no profiling data | 3 | 3 | 9 | Dev | Profile 16x32 vs 32x64 before committing; 16x32 fits L1 at 20KB | Use uniform grid until profiling complete | OPEN |
| R12 | Integration | PIC V3 depends on Hall MHD fix — cross-scaffold dependency | 4 | 3 | 12 | Dev | Implement Hall H1 before PIC V3; they're sequential anyway | Run V3 with reduced-accuracy E-field; document limitation | OPEN |
| R13 | Integration | AMR + PIC on refined grid not addressed by any scaffold | 2 | 4 | 8 | Dev | Flag for Phase D+; requires particle position mapping across levels | Keep PIC on uniform grid; interpolate AMR fields to uniform | OPEN |
| R14 | Physics | Thomson fit_te_ne_v fails in collective regime — curve_fit never converges (0/4 in testing) | 4 | 3 | 12 | Dev | Use differential_evolution instead of curve_fit (+30 LOC, 0.6s runtime) | Provide moment-based estimates without fitting | OPEN |
| R15 | Physics | PIC Esirkepov multi-cell crossing silently zeroes charge — beam ions at 100 keV safe at dt=1ns but fail at dt=10ns | 3 | 4 | 12 | Dev | Runtime CFL check: max(v)*dt < dx; sub-cycle if violated | Cap particle velocity; warn user | OPEN |
| R16 | Physics | Hall CFL formula bug IN THE SCAFFOLD — uses max(ne) labeled as ne_min | 5 | 3 | 15 | Dev | Fix formula: use min(ne) excluding vacuum cells | Wrong CFL = wrong sub-cycle count = inaccurate Hall | OPEN |
| R17 | Resource | PIC 10K particles insufficient for converged Yn — Schmidt used 10^6 | 3 | 3 | 9 | Dev | Convergence study: 10K/50K/100K; document convergence threshold | Accept statistical noise; report error bars | OPEN |
| R18 | Physics | Nernst advection of B comparable to Hall at DPF conditions — not analyzed | 2 | 3 | 6 | Dev | Order-of-magnitude check; document as known limitation | Accept; Nernst is out of scope for current fidelity target | OPEN |
| R19 | Performance | Fused PLM+HLL Metal kernel (OPT-4) has highest FMEA RPN=160 — complex MSL, hard to debug | 2 | 4 | 8 | Dev | Defer to Phase 2; validate simpler MLX port delivers adequate speedup first | Skip entirely if Phase 1 delivers 2x+ | OPEN |
| R20 | Architecture | Parthenon-VIBE arXiv:2509.19701 may not exist (Sept 2025 ID) — AMR Amdahl analysis loses grounding | 3 | 2 | 6 | Dev | Verify paper existence before relying on its conclusions | Use conservative regrid overhead estimates | OPEN |
| R21 | Integration | PIC dt mismatch bug (line 1561) — Esirkepov uses self.dt, not push dt | 5 | 3 | 15 | Dev | Pass dt explicitly through deposit() (5 LOC) | Benign until sub-cycling is added, then catastrophic | OPEN |
| R22 | Physics | PIC Nanbu self-collision bias — same array for both species | 4 | 2 | 8 | Dev | Copy velocity array before Nanbu call (10 LOC) | Under-estimates isotropization by 10-30% | OPEN |
| R23 | Schedule | AMR LOC underestimated 40-60% per Six Sigma review — 1,800-2,600 vs 1,150-1,550 claimed | 4 | 3 | 12 | Dev | Use revised estimates for planning; cylindrical refluxing alone is 500-700 LOC | Descope Phase C/D; stop at 2-level no-reflux | OPEN |
| R24 | Physics | Existing 4,900+ tests — no scaffold addresses keeping them green during implementation | 3 | 4 | 12 | Dev | Regression gate: full suite after every phase; use_amr=False default | Revert and bisect on failure | OPEN |

---

## 2. Risk Heat Map

### RED (Score > 15) — Must mitigate before implementation

| ID | Score | Risk Summary |
|----|-------|-------------|
| R02 | 25 | PIC ghost-cell NaN: instant sim death |
| R01 | 20 | AMR cylindrical refluxing: no reference, conservation at stake |
| R03 | 20 | Hall mu_0 bug: Hall has never worked correctly |
| R04 | 20 | PIC Boris non-relativistic: superluminal particles |
| R05 | 16 | PIC Esirkepov + reflecting BC: charge conservation broken |
| R06 | 16 | PIC E-field missing Hall: wrong trajectories at pinch |
| R16 | 15 | Hall CFL formula bug in scaffold design |
| R21 | 15 | PIC dt mismatch: ticking time bomb |
| R08 | 15 | HLL/HLLS CPU round-trip: 45-55% perf waste |

### YELLOW (Score 8-15) — Mitigate during implementation

| ID | Score | Risk Summary |
|----|-------|-------------|
| R07 | 12 | AMR conservation 100x worse than claimed |
| R09 | 12 | Whistler CFL kills Hall performance |
| R12 | 12 | PIC-Hall cross-scaffold dependency |
| R14 | 12 | Thomson fitting fails in collective regime |
| R15 | 12 | PIC multi-cell crossing zeroes charge |
| R23 | 12 | AMR LOC underestimated 40-60% |
| R24 | 12 | Regression test maintenance during implementation |
| R10 | 9 | mx.compile shape constraints for AMR |
| R11 | 9 | AMR block size vs L1 cache |
| R17 | 9 | PIC particle count convergence |

### GREEN (Score < 8) — Accept and monitor

| ID | Score | Risk Summary |
|----|-------|-------------|
| R13 | 8 | AMR + PIC integration (future) |
| R19 | 8 | Fused Metal kernel complexity |
| R22 | 8 | Nanbu self-collision bias |
| R18 | 6 | Nernst advection not analyzed |
| R20 | 6 | Parthenon-VIBE reference validity |

---

## 3. Cross-Scaffold Dependencies

### Dependency Chain
```
Hall MHD H1 (mu_0 fix, 30 LOC)
    |
    v
PIC V3 (E-field accuracy needs Hall)         MLX Opt (HLL GPU port)
    |                                              |
    v                                              v
PIC V4 (full discharge + absorbing BC)        AMR Phase A (needs fast solver)
                                                   |
                                                   v
                                              AMR Phase B-D
Thomson (INDEPENDENT — no upstream deps)
```

### Failure Cascade Analysis

**If AMR Phase A fails**: PIC is unaffected. Thomson is unaffected. MLX optimization is unaffected. Fidelity path loses the 9.5/10 AMR contribution but 9.0/10 is still achievable via Hall + PIC + diagnostics. AMR is the only path to resolution beyond 128x256 without 8x cost.

**If HLL GPU port has subtle float32 bug**: Blast radius is the entire MLX solver — every simulation using HLL/HLLS would have wrong fluxes. HLLS is inherently float32-safe (entropy pressure, no E-KE-ME cancellation), so the risk is confined to HLL. Mitigation: A/B test against CPU float64 reference for 1 week before removing fallback. Keep `"hll_cpu"` alias permanently.

**If PIC produces wrong Yn**: Does NOT invalidate the MHD calibration — fc/fm are calibrated to I(t) waveform, not Yn. PIC is diagnostic (it reads MHD fields), not constitutive (it doesn't change the calibration). Wrong Yn means the PIC module needs debugging, not recalibration of the MHD solver.

**If Thomson shows Te is wrong**: Recalibrate the two-temperature model's electron-ion equilibration rate. This does NOT affect I(t) or pressure (single-fluid quantities). It does affect Yn (beam-target cross-section depends on Ti, not Te directly). The recalibration scope is limited to `electron_coupling_coefficient` in the 2T engine, not fc/fm.

---

## 4. Mitigation Strategies (RED Risks)

| ID | Mitigation | LOC | Time | Pre-condition |
|----|-----------|-----|------|---------------|
| R02 | NaN guard in `interpolate_field_to_particles`: return 0.0 for NaN/Inf cells; add `mx.where(isfinite, field, 0)` before interpolation | 8 | 30 min | None |
| R01 | Standalone cylindrical reflux test: annular Sod problem with known analytical conservation. Build test BEFORE integrating into AMR. | 80 test + 150 impl | 2 sessions | None |
| R03 | Fix mu_0 factor in `mlx_sources.py:apply_hall_mhd`. Write whistler dispersion test FIRST (TDD red-green). | 30 fix + 60 test | 1 session | None |
| R04 | Add relativistic gamma to Boris: `qdt_over_2m = charge*dt/(2*mass*gamma)` where `gamma = sqrt(1 + |v|^2/c^2)`. Add velocity cap at 0.99c. | 21 | 1 hour | None |
| R05 | Replace reflecting BC with absorbing at electrodes (anode/cathode). Particles hitting boundary are removed. | 60 | 2 hours | R02 fixed |
| R06 | Add Hall E-field to `_step_pic`: `E_hall = (J x B)/(n_e * e)`. Compute J from curl(B). | 30 | 2 hours | R03 fixed |
| R16 | Fix `ne_min` in scaffold to `min(ne)` excluding vacuum: `ne_min = float(mx.min(mx.where(rho > 1e-10, ne, 1e30)).item())` | 3 | 15 min | None |
| R21 | Pass dt through deposit(): `deposit_current_esirkepov(..., dt=dt)` instead of `self.dt` | 5 | 15 min | None |
| R08 | Port HLLS to pure MLX first (zero float32 risk, ~130 LOC). Then HLL with entropy pressure (~120 LOC). | 250 | 1-2 days | None |

---

## 5. Decision Gates

### Hall MHD
- **Start H2 (CFL + sub-cycling) when**: H1 unit fix passes whistler dispersion test within 10% of analytical phase speed.
- **Accept Hall as production-ready when**: Cross-backend parity L1(dB) < 1e-3 AND energy conservation < 1e-6 over 100 steps.

### PIC Validation
- **Start V3 (MHD-coupled) when**: V1 unit tests all green AND V2 gyration energy drift < 1e-8/gyroperiod AND R02/R04/R21 fixes merged.
- **Start V4 (full discharge) when**: V3 energy conservation < 5% AND absorbing BC implemented (R05) AND sub-cycling working.
- **Accept Yn results when**: Convergence study (10K/50K/100K particles) shows Yn stable within factor-of-2 AND beam-target > thermonuclear by >10x.

### AMR
- **Start Phase B (auto-refinement) when**: Phase A passes smooth advection test with `|delta_mass/mass| < 1e-8` per step on uniform-equivalent grid.
- **Start Phase C (refluxing) when**: Phase B auto-refinement triggers correctly on Sod shock AND conservation error without refluxing is measured and documented.
- **Accept AMR for production when**: 2-level AMR on PF-1000 is faster than uniform 128x256 AND conservation < 1e-4 cumulative AND I_peak matches within 5% of uniform-grid result.

### MLX Optimization
- **Start Phase 2 (fused kernels) when**: Phase 1 (pure MLX HLL/HLLS) delivers >= 1.5x speedup AND all 471 MLX tests pass.
- **Accept optimization as complete when**: PF-1000 64x128 full discharge < 4 min (2x speedup from 8 min baseline).

### Thomson Scattering
- **Accept module when**: Gaussian limit test within 1%, Doppler shift within 0.05 nm, Salpeter sum rule within 0.1%, fit roundtrip within 10% on Te/ne.

### Overall Fidelity
- **Claim 8.5/10 when**: Hall MHD bug fixed + MLX optimization Phase 1 complete.
- **Claim 9.0/10 when**: PIC V4 produces physically plausible Yn + AMR Phase A operational + Thomson diagnostic validated.

---

## 6. Resource Allocation

### Agent-Parallelizable Work
| Task | Agent Model | Parallelizable? | Reason |
|------|------------|-----------------|--------|
| PIC V1 unit test files (7 files) | Sonnet | YES — 7 independent files | Each file tests one function in isolation |
| Thomson module implementation | Sonnet | YES — independent of all scaffolds | Pure diagnostic, reads state dict |
| MLX HLL/HLLS parity tests | Haiku | YES — template-based test files | Standard A/B comparison pattern |
| AMR block data structure | Sonnet | YES — new module, no existing edits | `amr.py` is a new file |
| Hall MHD H5 test suite | Sonnet | YES — after H1-H4 merged | Independent test file |

### Requires Human Judgment
| Task | Why |
|------|-----|
| Hall MHD H1 unit fix | Physics correctness — HL unit algebra must be verified by hand |
| PIC Boris relativistic upgrade | Subtle momentum conservation implications |
| AMR cylindrical refluxing | No reference implementation — research-grade problem |
| fc/fm recalibration after any physics change | Compensating errors — previous calibration assumes current bugs |
| Decision gates (go/no-go) | Risk tolerance is a human judgment call |

### Critical Path
```
Week 1:  Hall H1-H3 (1 day) + MLX OPT-1/2 HLLS port (1 day)
         || Thomson implementation (agent, 1 day)
         || PIC V1 unit tests (agents, 1 day)
Week 2:  Hall H4-H5 (0.5 day) + MLX OPT-1/2 HLL port (1 day)
         + PIC bug fixes R02/R04/R21 (0.5 day)
         + PIC V2 integration tests (1 day)
Week 3:  PIC V3 MHD-coupled tests (2 days)
         || AMR Phase A data structures (2 days)
Week 4:  PIC V4 end-to-end (2 days)
         || AMR Phase A ghost exchange + solver wiring (2 days)
Week 5:  AMR Phase B auto-refinement (2 days)
         + MLX optimization Phase 2 if needed (2 days)
Week 6:  AMR Phase C refluxing (3 days) — research-grade
```

**Critical path**: Hall H1 -> PIC bug fixes -> PIC V3 -> PIC V4 (constrains fidelity 9.0).
**Parallel path**: MLX optimization + Thomson + AMR Phase A (independent of PIC).

---

## 7. Schedule Risk

### Timeline Estimates

| Milestone | Optimistic | Expected | Pessimistic | Key Risk Factor |
|-----------|-----------|----------|-------------|-----------------|
| Hall MHD complete (H1-H5) | 1 day | 2 days | 4 days | Whistler sub-cycling tuning |
| MLX Opt Phase 1 (HLL/HLLS GPU) | 2 days | 3 days | 5 days | float32 edge cases in HLL |
| Thomson diagnostic complete | 1 day | 2 days | 3 days | Abel transform for spectral use |
| PIC V1-V2 complete | 2 days | 4 days | 7 days | Numba JIT overhead; 8 bug fixes |
| PIC V3-V4 complete | 3 days | 6 days | 12 days | Ghost-cell NaN chain; sub-cycling |
| AMR Phase A-B complete | 4 days | 8 days | 16 days | LOC underestimate; block size tuning |
| AMR Phase C (refluxing) | 3 days | 6 days | 12 days | No reference impl; research-grade |
| **Fidelity 9.0/10** | **12 days** | **24 days** | **45 days** | PIC + AMR compound delays |

### #1 Schedule Risk: AMR Cylindrical Refluxing

AMR Phase C (cylindrical refluxing) is the longest pole. No reference implementation exists in any Python GPU code. The Six Sigma review estimates 500-700 LOC vs the scaffold's 300-400. This is research-grade work where the first attempt is unlikely to be correct.

If AMR refluxing slips, fidelity 9.0/10 is still achievable without it — Hall + PIC + Thomson + MLX optimization deliver the physics upgrades. AMR delivers resolution, which primarily matters for fidelity 9.5/10.

### #2 Schedule Risk: PIC Compound Bugs

The PIC module has 8 known bugs (3 blocking) and 14 untested functions. The investigation found compound bug interactions (reflecting BC + Esirkepov, ghost-cell NaN chain, superluminal Boris) that will manifest during V3-V4. Each compound bug requires understanding two interacting systems simultaneously. The scaffold estimates 7-11 days; the Six Sigma review says 12-18 days. Realistic: 2-3 weeks for V1-V4 with all bug fixes.

### Schedule Mitigation Strategy

1. **Ship Hall + Thomson + MLX Opt first** (Week 1-2). These are small, well-understood, high-value. They raise fidelity from 7.5 to ~8.5 with low risk.
2. **PIC in parallel with AMR Phase A** (Week 3-4). Both are large but independent.
3. **AMR refluxing is the last item** (Week 5-6). If it slips, everything else is already delivered.
4. **Descope AMR to 2-level no-reflux** if calendar exceeds 6 weeks. The 2.7-3.5x speedup from AMR without refluxing is still valuable for parameter sweeps, even with 0.3% mass drift.
