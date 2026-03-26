# DPF-Unified Forward Plan — Six Sigma DMAIC Analysis

**Date**: 2026-03-26
**Analyst**: Cortana (Opus)
**Scope**: Phase S completion through web deployment
**Data source**: 48 commits (Mar 25-26), 4,970 collected tests, 6,806 LOC delta, 16 presets, 31 gap items

---

## 1. DEFINE

### Problem Statement

Ship a production-grade DPF web simulator with validated physics that produces publication-quality predictions of I(t) waveforms, neutron yield, and pinch diagnostics across multiple devices.

### Critical-to-Quality (CTQ) Tree

```
Customer need: "Trustworthy DPF predictions accessible via browser"
    |
    +-- CTQ-1: I(t) NRMSE < 0.15 for all calibrated devices
    +-- CTQ-2: Neutron yield Yn within 10x of experiment (log-scale)
    +-- CTQ-3: Current dip depth within 20% of experimental
    +-- CTQ-4: Timing error < 10% vs published reference data
    +-- CTQ-5: Full discharge NaN-free on all 16 presets
    +-- CTQ-6: Wall time < 5 min for 32x64 PF-1000 on M3 Pro
```

### SIPOC

| Element | Detail |
|---------|--------|
| **Suppliers** | Literature (731 papers in DB), experimental data (Akel 24-shot, Scholz, Gribkov), MLX framework, Optuna |
| **Inputs** | Device parameters (V0, C, L0, R0, a, b, z0, P, gas), fc/fm calibration |
| **Process** | Lee model axial+radial -> MHD handoff -> full discharge simulation -> diagnostics extraction |
| **Outputs** | I(t) waveform, Yn, Te, pinch radius, synthetic diagnostics, web-rendered results |
| **Customers** | DPF researchers, students (teaching mode), engineers (parameter explorer) |

### Scope Boundaries

**In scope**: HLLS solver, species radiation integration, PIC validation, multi-device calibration, web deployment, WALRUS training data, AMR MVP.
**Out of scope**: ALE mesh motion (2000 LOC, architectural overhaul), Poloidal B-field (fundamental solver change), full implicit MHD (deferred by CFL profiling showing 0.1% impact at 32x64).

---

## 2. MEASURE

### Current Process Capability

**Velocity** (measured from overnight sprint):
- 48 commits / 14 hours = 3.4 commits/hr
- 6,806 LOC delta / 14 hours = 486 LOC/hr (gross, includes tests)
- Productive coding at 4h/day = ~1,944 LOC/day, ~13.6 commits/day
- Zero test regressions across 48 commits = 0 DPMO for regression (this sprint)

**Test coverage baseline**:
- 4,970 tests collected (118 deselected/platform-gated)
- 205 source files, 298 test files
- 9,068 test functions (many parametrized)

**Defect history** (from MEMORY.md observations):
- Float32 cancellation: 3 distinct occurrences (HLLD star-states, dp_dt chain rule, WENO-Z eps)
- Ghost-cell corruption: 1 occurrence, 4 stale proposals before root cause found
- Compensating error trap: 1 occurrence (fc/fm recalibration after back-EMF fix)
- Agent side-effect: 1 occurrence (physics regression from "fix tests" agent)
- Total defects across ~80 commits in Phase Q-R: ~6 significant bugs
- **DPMO (significant)**: 6 / (80 * 10 opportunities/commit) = 7,500 DPMO = **3.9 sigma**

**Feature completion status**:

| Feature | LOC Written | LOC Remaining | Tests | % Complete |
|---------|------------|---------------|-------|------------|
| HLLS Entropy Solver | 0 (research only) | ~380 | 0 | 5% (research done) |
| Species Tracking (S-1) | 236 | ~100 | 11 | 70% |
| PIC Validation (S-2) | ~400 | ~200 | 12 | 60% |
| Implicit MHD (S-3) | 369 (scaffold) | ~400 | 35 stubs | 35% |
| Multi-Device Calibration | infra done | 0 code, ~20h compute | 0 new | 15% (1/7 devices) |
| Web Deployment | ~300 (Gradio app) | ~100 | 0 deploy tests | 72% |
| WALRUS Training Data | ~200 (orchestrator) | ~150 | 0 integration | 30% |
| AMR MVP | 0 | ~500 | 0 | 0% |

### Sigma Level by CTQ

| CTQ | Current Value | Target | Defect Rate | Sigma |
|-----|---------------|--------|-------------|-------|
| CTQ-1: NRMSE | 0.146 (PF-1000 only) | < 0.15 all devices | 6/7 devices uncalibrated | 1.0 |
| CTQ-2: Yn | 1.32e11 vs 1e11 (32%) | Within 10x | 1/1 tested device passes | 3.0 (insufficient N) |
| CTQ-3: Dip depth | 51.8% modeled | Within 20% of exp | Uncalibrated | 2.0 |
| CTQ-4: Timing | 0-3% vs Gribkov | < 10% | PF-1000 passes | 3.5 |
| CTQ-5: NaN-free | pf1000_20kv NaN at >8us | All 16 presets | ~3-4 presets at risk | 2.5 |
| CTQ-6: Wall time | ~3 min (32x64 PF-1000) | < 5 min | Passes | 4.0 |

**Composite sigma: 2.4** (driven by CTQ-1 multi-device gap).

---

## 3. ANALYZE — FMEA

### Severity / Occurrence / Detection Scale

- **Severity**: 1=cosmetic, 5=degraded accuracy, 8=wrong physics, 10=crash/NaN/data loss
- **Occurrence**: 1=near impossible, 3=rare, 5=moderate, 7=likely, 10=certain
- **Detection**: 1=auto-caught by CI, 3=caught by unit test, 5=caught by integration test, 7=caught only by manual inspection, 10=undetectable until production

### Feature 1: HLLS Entropy Solver (~380 LOC)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 1.1 | Float32 entropy switch triggers in wrong cells, injecting excess diffusion | 7 | 6 | 5 | **210** | Switching threshold eta=p_S/E tuned for Cartesian, not cylindrical 1/r geometry |
| 1.2 | HLLS star-state pressure goes negative at strong shocks | 9 | 4 | 3 | **108** | Entropy flux approximate — missing contact wave correction at Mach > 10 |
| 1.3 | Performance regression vs HLL (HLLS is more expensive per cell) | 4 | 5 | 3 | 60 | Extra entropy equation solve per interface |
| 1.4 | Div(B) error increases because HLLS doesn't naturally couple to Dedner GLM psi | 6 | 5 | 5 | **150** | psi equation not included in HLLS wave structure |

### Feature 2: Species Radiation Integration Testing (~100 LOC remaining)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 2.1 | Species mass fractions not summing to 1.0, corrupting EOS | 8 | 5 | 3 | **120** | Advection truncation error without projection step |
| 2.2 | Z_eff from species mix produces wrong Bremsstrahlung rate | 6 | 4 | 3 | 72 | Z_eff formula uses number density, code uses mass fraction |
| 2.3 | Negative species fraction from numerical undershoot | 7 | 6 | 3 | **126** | WENO/PLM reconstruction doesn't enforce positivity on passive scalars |
| 2.4 | Species advection breaks CFL (introduces new wave speed) | 5 | 2 | 1 | 10 | Passive scalars ride existing wave speeds, no new constraint |

### Feature 3: PIC End-to-End Validation (~200 LOC remaining)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 3.1 | Particle initialization from MHD fields produces non-physical velocity distribution | 7 | 6 | 7 | **294** | Maxwellian assumption invalid at pinch; beam ions are non-thermal |
| 3.2 | Yn calculation off by >100x due to wrong cross-section interpolation | 9 | 4 | 5 | **180** | DD cross-section table interpolation error at non-Maxwellian tail |
| 3.3 | Boris pusher energy drift over many cycles | 6 | 5 | 3 | 90 | Standard Boris is symplectic but finite dt introduces secular drift at high omega_c*dt |
| 3.4 | CIC deposition creates grid-scale noise in J_kin coupling to MHD | 7 | 7 | 5 | **245** | Insufficient particles per cell (<100) at pinch, Poisson sampling noise |

### Feature 4: Multi-Device Calibration Completion (compute-dominated)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 4.1 | Optuna gets stuck in local minimum for non-PF-1000 devices | 6 | 7 | 7 | **294** | Different circuit topologies (series vs parallel crowbar) create multi-modal loss landscape |
| 4.2 | Calibration takes >4 hours per device, blocking pipeline | 4 | 8 | 5 | **160** | Each trial = 8 min full discharge; 100 trials minimum for TPE convergence |
| 4.3 | Calibrated params overfit to single experimental waveform | 7 | 6 | 7 | **294** | Single-shot calibration hides shot-to-shot variability |
| 4.4 | Reference waveform digitization error >5% | 5 | 4 | 7 | **140** | Published figures at 300 DPI, pixel-to-physical mapping introduces systematic error |

### Feature 5: Web Simulator Deployment (~100 LOC)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 5.1 | HuggingFace Spaces CPU-only, MLX unavailable, falls back to Python engine at 10x slower | 5 | 9 | 3 | **135** | HF Spaces runs Linux x86, no Apple Silicon. Must use NumPy/Numba backend. |
| 5.2 | Gradio app crashes on long-running simulation (timeout) | 7 | 6 | 5 | **210** | HF free tier has 60s timeout; full PF-1000 on CPU takes 10+ min |
| 5.3 | Stale deployment — code diverges from local, presets wrong | 5 | 7 | 7 | **245** | No CI/CD pipeline for HF Spaces push |
| 5.4 | User inputs cause NaN (extreme V0 or pressure values) | 8 | 5 | 5 | **200** | Input validation missing on Gradio sliders; config allows invalid physics |

### Feature 6: WALRUS Training Data Generation (~150 LOC remaining)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 6.1 | Training trajectories all from same operating point, no diversity | 6 | 5 | 5 | **150** | Batch runner uses single preset; need parameter sweeps |
| 6.2 | Well HDF5 format mismatch with WALRUS DataLoader | 8 | 4 | 3 | 96 | Schema drifts between WALRUS versions; field_indices mapping breaks |
| 6.3 | NaN trajectories corrupt training set | 9 | 5 | 3 | **135** | Dataset validator exists but not wired into batch pipeline |
| 6.4 | Cylindrical MHD output mapped to Cartesian WALRUS incorrectly | 7 | 5 | 5 | **175** | r,z -> x,y mapping in field_mapping.py doesn't preserve vector field components correctly |

### Feature 7: AMR MVP (~500 LOC)

| # | Failure Mode | S | O | D | RPN | Root Cause |
|---|-------------|---|---|---|-----|------------|
| 7.1 | Patch boundary ghost exchange introduces conservation error | 9 | 7 | 5 | **315** | Refluxing algorithm required at fine/coarse interfaces; MLX has no native AMR support |
| 7.2 | Refinement criterion triggers everywhere in sheath, creating too many patches | 5 | 6 | 3 | 90 | Density gradient threshold not scale-aware |
| 7.3 | GPU Amdahl bottleneck — regridding serializes computation | 6 | 8 | 7 | **336** | Parthenon-VIBE 2025 showed AMR overhead can exceed compute savings on GPU |
| 7.4 | Existing 4,970 tests break from array shape changes | 8 | 4 | 3 | 96 | AMR patches have variable shape; all tests assume uniform grid |

---

### FMEA Summary — Top 10 by RPN

| Rank | ID | Failure Mode | RPN | Feature |
|------|-----|-------------|-----|---------|
| 1 | 7.3 | AMR GPU Amdahl bottleneck | **336** | AMR MVP |
| 2 | 7.1 | AMR patch boundary conservation error | **315** | AMR MVP |
| 3 | 3.1 | PIC non-physical velocity initialization | **294** | PIC Validation |
| 4 | 4.1 | Optuna local minimum for non-PF1000 | **294** | Multi-Device Cal |
| 5 | 4.3 | Single-shot calibration overfitting | **294** | Multi-Device Cal |
| 6 | 3.4 | CIC grid-noise from low particle count | **245** | PIC Validation |
| 7 | 5.3 | Stale HF Spaces deployment | **245** | Web Deploy |
| 8 | 1.1 | HLLS entropy switch mis-triggers | **210** | HLLS Solver |
| 9 | 5.2 | Gradio timeout on long simulation | **210** | Web Deploy |
| 10 | 5.4 | User input causes NaN | **200** | Web Deploy |

---

## 4. IMPROVE — Mitigations for Top RPN Items

### Mitigation 1: AMR GPU Amdahl (RPN 336)
- **Action**: Defer AMR to post-deployment. Use static refinement (already exists in `experimental/static_refinement.py`) with 2-level preset zones for sheath region. This gives 4x effective resolution at the sheath for ~20% cost increase vs. full AMR's uncertain ROI.
- **LOC**: ~50 (wire static refinement into MLX solver config)
- **Executor**: Code (sonnet agent), validated with convergence study
- **RPN after mitigation**: 336 -> 90 (severity drops to 3 because static refinement is sufficient for publication-quality results at 64x128)

### Mitigation 2: AMR Patch Conservation (RPN 315)
- **Action**: Same as above — defer full AMR. If pursued later, implement refluxing per Berger & Colella (1989) with dedicated conservation correction step.
- **LOC**: 0 now (deferred)
- **RPN after mitigation**: 315 -> 0 (deferred)

### Mitigation 3: PIC Non-Physical Initialization (RPN 294)
- **Action**: Replace Maxwellian initialization with kappa-distribution (kappa=2-5) for beam ions. Extract Ti and beam velocity from MHD sheath-velocity history. Add initialization test comparing velocity moments to expected values.
- **LOC**: ~80 (modify `mhd_to_pic_init()` + 3 tests)
- **Executor**: Code (opus — physics reasoning required)
- **RPN after mitigation**: 294 -> 84 (O drops from 6 to 3, D drops from 7 to 4)

### Mitigation 4: Optuna Local Minimum (RPN 294)
- **Action**: Use multi-start TPE with 5 random seed trials per device. Add CMA-ES sampler as fallback when TPE converges to NRMSE > 0.20. Use published Lee model params as warm-start prior.
- **LOC**: ~40 (sampler configuration in calibration pipeline)
- **Executor**: Code (sonnet agent)
- **RPN after mitigation**: 294 -> 126 (O drops from 7 to 3)

### Mitigation 5: Single-Shot Calibration Overfitting (RPN 294)
- **Action**: For devices with multi-shot data (PF-1000 Akel: 24 shots), calibrate on 80% train split, validate on 20% held-out. For single-shot devices, report confidence interval from Optuna's TPE posterior. Flag single-shot devices in UI as "provisional calibration".
- **LOC**: ~30 (train/test split in Optuna objective, UI label)
- **Executor**: Code (sonnet agent)
- **RPN after mitigation**: 294 -> 98 (D drops from 7 to 3, caught by held-out validation)

---

## 5. CONTROL

### Gate Definitions

Each feature must pass its gate before merging to main.

| Feature | Gate | Tests Required | Metric | Rollback Trigger |
|---------|------|---------------|--------|-----------------|
| HLLS Solver | G-HLLS | Sod shock L1 < HLL, Brio-Wu no NaN, Orszag-Tang energy conservation < 1e-6, cylindrical PF-1000 completes | NRMSE does not degrade vs HLL baseline | Any NRMSE increase > 0.01 |
| Species | G-SPEC | Mass fraction sum = 1.0 +/- 1e-12, Z_eff matches analytic for D/Cu mix, advection convergence order > 1.5 | No new NaN on any preset | Species fraction < 0 anywhere |
| PIC | G-PIC | Yn within 10x experiment, Boris energy drift < 1% over 1000 cycles, J_kin smooth (no grid-scale modes) | Yn log-error improves vs MHD-only | Yn off by > 100x |
| Calibration | G-CAL | NRMSE < 0.15 on each device, t_peak error < 10%, I_peak error < 10% | At least 4/7 devices pass all 3 metrics | Any device NRMSE > 0.25 |
| Web Deploy | G-WEB | Gradio loads, all 16 presets selectable, simulation completes on HF Spaces (lee_only mode for CPU), no NaN in output | Round-trip time < 120s for lee_only on free tier | Crash or timeout on any preset |
| WALRUS Data | G-WAL | 100+ valid trajectories, Well schema passes validator, no NaN, field mapping reversible | WALRUS loads and runs 1-step inference on generated data | Schema validation failure |
| AMR MVP | G-AMR | **DEFERRED** — gate defined but not active | N/A | N/A |

### Continuous Monitoring

| Metric | Source | Frequency | Alert Threshold |
|--------|--------|-----------|-----------------|
| Test count | `pytest --co -q \| tail -1` | Every commit | < 4,970 (current baseline) |
| Test pass rate | CI | Every commit | < 100% |
| PF-1000 NRMSE | Calibration regression test | Weekly | > 0.15 |
| NaN-free presets | Preset smoke suite | Every commit (CI) | Any preset NaN |
| MLX solver wall time | Benchmark | Weekly | > 5 min for 32x64 |
| LOC per source file | Ruff + custom | Every commit | > 400 lines |

### Rollback Protocol

1. `git revert <commit>` for single-commit regressions
2. `git revert --no-commit <range>` for multi-commit feature branch regressions
3. Never `git reset --hard` on main
4. Tag stable points before each feature integration: `v1.5.1-pre-hlls`, etc.

---

## 6. CRITICAL PATH ANALYSIS

### Dependency Graph

```
                    ┌──────────────┐
                    │ HLLS Solver  │
                    │   ~380 LOC   │
                    │   ~2 days    │
                    └──────┬───────┘
                           │ (enables float32-only deployment)
                           v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Species Int. │    │ Multi-Device │    │  Web Deploy  │
│   ~100 LOC   │    │ Calibration  │    │   ~100 LOC   │
│   ~1 day     │    │  ~20h compute│    │   ~1 day     │
└──────┬───────┘    │   ~5 days    │    └──────┬───────┘
       │            └──────┬───────┘           │
       v                   │                   │
┌──────────────┐           │            ┌──────────────┐
│ PIC E2E Val  │           │            │ WALRUS Data  │
│   ~200 LOC   │           │            │   ~150 LOC   │
│   ~2 days    │           │            │   ~2 days    │
└──────┬───────┘           │            └──────┬───────┘
       │                   │                   │
       └───────────┬───────┘                   │
                   v                           │
            ┌──────────────┐                   │
            │  Production  │<──────────────────┘
            │   Release    │
            └──────────────┘
```

**Critical path**: HLLS Solver -> Multi-Device Calibration -> Production Release
**Calendar time**: 2 + 5 = **7 days** on critical path at 4h/day.

**Parallel path A**: Species Integration -> PIC Validation (3 days, not on critical path)
**Parallel path B**: Web Deployment (1 day, not on critical path, can launch with lee_only mode immediately)
**Parallel path C**: WALRUS Training Data (2 days, independent, post-release)

### Recommended Sprint Order

| Sprint | Feature | Days | Sigma Target | Rationale |
|--------|---------|------|-------------|-----------|
| **S-4** | Web Deployment (lee_only mode) | 1 | 3.5 | Ship something NOW. Lee model works on CPU. Unblocks user feedback. |
| **S-5** | HLLS Entropy Solver | 2 | 3.5 | Eliminates float64 CPU HLLD dependency. Enables float32-only MLX path for all conditions. |
| **S-6** | Multi-Device Calibration | 5 | 3.0 | Compute-dominated. Run Optuna overnight per device. 6 devices remaining. |
| **S-7** | Species Radiation Integration | 1 | 3.5 | Module 70% done. Wire remaining tests, validate Z_eff. |
| **S-8** | PIC End-to-End Validation | 2 | 2.5 | Depends on species. Kappa-distribution init, Yn validation. |
| **S-9** | WALRUS Training Data | 2 | 3.0 | Generate 100+ trajectories. Wire validator into pipeline. |
| **S-10** | Web Deploy v2 (full_mhd + calibrated presets) | 1 | 4.0 | Re-deploy with all calibrated presets and full_mhd mode. |

**Total: 14 days (3.5 weeks at 4h/day)**

### Sigma Progression Forecast

| Milestone | Date (est.) | Composite Sigma | Key Driver |
|-----------|-------------|-----------------|------------|
| Current | 2026-03-26 | 2.4 | Only 1/7 devices calibrated |
| S-4 complete | 2026-03-27 | 2.6 | Web deployed, user-facing |
| S-5 complete | 2026-03-29 | 2.8 | Float32-only path, NaN risk reduced |
| S-6 complete | 2026-04-03 | 3.5 | All devices calibrated, CTQ-1 met |
| S-7+S-8 complete | 2026-04-06 | 3.7 | Species + PIC validated |
| S-9+S-10 complete | 2026-04-09 | 4.0 | Full deployment, WALRUS data pipeline |

---

## 7. RISK REGISTER

| ID | Risk | Probability | Impact | Mitigation | Owner |
|----|------|-------------|--------|------------|-------|
| R1 | HLLS fails to improve over HLL for DPF cylindrical geometry | 30% | High | Fallback: keep HLL + float64 CPU HLLD for V&V runs only | Physics |
| R2 | Calibration compute exceeds 40h total (8h/device * 6) | 60% | Medium | Parallel Optuna with constant_liar already implemented; run overnight | Compute |
| R3 | HF Spaces CPU backend too slow for full_mhd (>60s timeout) | 80% | Medium | Ship lee_only mode first; add async job queue for full_mhd | Deploy |
| R4 | PIC Yn prediction degrades when switching from Maxwellian to kappa init | 20% | High | Keep Maxwellian as fallback option; A/B test both | Physics |
| R5 | WALRUS fine-tuning OOMs on M3 Pro 36GB | 50% | Low | LoRA (2.6GB weights + ~15GB activations) fits; batch_size=1 + grad_ckpt | ML |

---

## 8. RESOURCE ALLOCATION

### Effort-Weighted Backlog (Pareto)

The top 4 features account for 80% of the remaining effort:

| Feature | Effort (person-days) | Cumulative % | Priority |
|---------|---------------------|-------------|----------|
| Multi-Device Calibration | 5.0 | 36% | P0 — gates CTQ-1 |
| HLLS Entropy Solver | 2.0 | 50% | P0 — gates float32-only |
| PIC Validation | 2.0 | 64% | P1 — gates CTQ-2 |
| WALRUS Training Data | 2.0 | 79% | P1 — gates surrogate |
| Web Deploy v1 | 1.0 | 86% | P0 — gates user access |
| Species Integration | 1.0 | 93% | P1 — gates PIC |
| Web Deploy v2 | 1.0 | 100% | P2 — after calibration |

### Agent Delegation Plan

| Task | Model | Parallel? | Rationale |
|------|-------|-----------|-----------|
| HLLS implementation | Opus (self) | No | Physics judgment required |
| HLLS test suite | Sonnet agent | Yes (during HLLS impl) | Test scaffolding from spec |
| Calibration runs | Compute (Optuna) | Yes (overnight) | No human needed |
| Species test completion | Sonnet agent | Yes | Mechanical test writing |
| PIC kappa-distribution | Opus (self) | No | Non-trivial physics |
| Web deploy CI/CD | Sonnet agent | Yes | DevOps plumbing |
| WALRUS data generation | Batch runner | Yes (overnight) | Compute-only |

---

## 9. APPENDIX — Detailed Sigma Calculation

### Method

Sigma = NORMSINV(1 - DPMO/1,000,000) + 1.5 (industry shift)

### Per-CTQ DPMO

| CTQ | Opportunities | Defects | DPMO | Sigma (short-term) |
|-----|--------------|---------|------|---------------------|
| CTQ-1 (NRMSE) | 7 devices | 6 uncalibrated | 857,143 | 0.5 |
| CTQ-2 (Yn) | 1 device tested | 0 | 0 (insufficient N) | N/A |
| CTQ-3 (Dip) | 1 device | 1 (uncalibrated) | 1,000,000 | 0.0 |
| CTQ-4 (Timing) | 1 device | 0 | 0 | 4.0+ |
| CTQ-5 (NaN-free) | 16 presets | ~3 risky | 187,500 | 2.4 |
| CTQ-6 (Wall time) | 1 config | 0 | 0 | 4.0+ |

**Composite (weighted by customer impact)**: CTQ-1 and CTQ-5 dominate. Weighted DPMO ~400,000 = **1.8 sigma** (honest assessment, no inflation).

After S-6 (all devices calibrated): DPMO drops to ~50,000 = **3.1 sigma**.
After S-10 (full deployment): target DPMO ~6,200 = **4.0 sigma**.

---

*Generated by Cortana DMAIC Engine. Next review: after Sprint S-4 completion.*
