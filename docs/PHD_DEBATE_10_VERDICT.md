# PhD Debate #10 — Post Phase AC Experimental Validation Assessment

**Date**: 2026-02-26
**Scope**: Phase AC (PF-1000 calibration, I(t) waveform comparison vs Scholz et al. 2006, liftoff delay), Phase AC.1 (liftoff delay feature)
**Previous Score**: 6.1/10 (Debate #9)

## VERDICT: CONSENSUS (3-0) — 6.2/10

The panel agrees on a +0.1 improvement from 6.1 to 6.2 (computed 6.23, rounded down). Phase AC represents the first quantitative comparison of any DPF-Unified component against published experimental data — a genuine milestone. However, this comparison validates the standalone Lee model (2-phase, scipy solve_ivp), NOT the production MHD engine (RLCSolver + SnowplowModel + MHDSolver). The modest score increase reflects the tension between real infrastructure progress and the fundamental gap that the primary simulation tool remains experimentally unvalidated.

### Panel Positions
- **Dr. PP (Pulsed Power)**: AGREE 6.2/10 — "Phase AC adds genuine comparison infrastructure. fm=0.178 in published range confirms D1 fix impact. But LeeModel validation ≠ RLCSolver validation. Voltage reversal corrected to 47.7% (not 61%)."
- **Dr. DPF (Plasma Physics)**: AGREE 6.2/10 — "Retract D3 (R0 discrepancy) — external R0 is correct, R_plasma added dynamically. Retract fm 'artifact' claim — circular reasoning + fc^2/fm degeneracy. The comparison infrastructure is real but does not constitute AIAA-standard validation."
- **Dr. EE (Electrical Engineering)**: AGREE 6.2/10 — "Concede 753 kHz was a unit error (mF vs uF); actual f0=23.8 kHz, skin effect negligible. Phase AC waveform NRMSE against Scholz (2006) is genuine new work deserving credit. GUM critique softened to 'naming issue, computationally conservative.'"

### Scoring Breakdown

| Category | Debate #9 | Debate #10 | Change | Weight | Notes |
|----------|-----------|------------|--------|--------|-------|
| MHD Numerics | 7.5 | 7.5 | — | 0.18 | Unchanged; WENO-Z + HLLD + SSP-RK3 |
| Transport | 7.6 | 7.6 | — | 0.12 | Unchanged; GMS Coulomb log in place |
| Circuit | 5.5 | 5.8 | +0.3 | 0.12 | D3 retracted (R0 architecture correct). D1 confirmed via fm range. BDF2 dL/dt undocumented strength. No RLCSolver-to-experiment validation. |
| DPF-Specific | 5.5 | 5.7 | +0.2 | 0.22 | fm=0.178 in [0.05,0.20] confirms D1 fix. Reflected shock 4×rho0 (R-H). Liftoff delay. fc^2/fm degeneracy limits individual parameter uniqueness. |
| Validation | 3.5 | 3.9 | +0.4 | 0.18 | First experimental comparison (NRMSE=0.192 vs Scholz 2006). 26-pt digitized waveform. Peak region <0.10. But Lee-vs-experiment, not MHD-vs-experiment. |
| AI/ML | 3.5 | 3.5 | — | 0.08 | Unchanged |
| Software | 7.2 | 7.2 | — | 0.10 | 2451 non-slow tests; 40 Phase AC tests |

**Composite**: 0.18×7.5 + 0.12×7.6 + 0.12×5.8 + 0.22×5.7 + 0.18×3.9 + 0.08×3.5 + 0.10×7.2 = 1.35 + 0.912 + 0.696 + 1.254 + 0.702 + 0.28 + 0.72 = **5.914**

**Adjusted**: 6.1 + (5.914 - 5.762) = 6.1 + 0.15 = **6.25 ≈ 6.2/10**

(The 5.762 is the Debate #9 composite from the same weight formula; the +0.15 delta reflects genuine Phase AC contributions.)

## Key Findings from Cross-Examination

### Retracted Claims (Phase 2-3)

1. **D3 bug (R0=2.3 mOhm vs RESF r0=6.1 mOhm) — RETRACTED by Dr. DPF**
   - R0=2.3 mOhm is external circuit resistance (measured by short-circuit discharge)
   - `rlc_solver.py` line 251: `R_eff = self.R_total + coupling.R_plasma` — plasma resistance added dynamically
   - Lee's RESF normalizes *total* effective resistance (external + plasma); comparing to external-only R0 is apples-to-oranges
   - All three panelists agree: NOT a bug

2. **fm=0.178 "artifact of parameter compensation" — RETRACTED by Dr. DPF**
   - Original argument was circular: R0 wrong → fm compensates → fm wrong → R0 wrong
   - fc^2/fm degeneracy (confirmed 3-0) means individual parameters are non-unique
   - Without independent measurement of R0 or fm, cannot use one to invalidate the other

3. **753 kHz skin depth concern — RETRACTED by Dr. EE**
   - Root cause: unit error (C=1.332e-6 F used instead of 1.332e-3 F)
   - Actual f0 = 23.8 kHz; skin depth = 423 μm; electrode AC resistance = 0.033 mΩ (1.4% of R0)
   - Skin effect negligible for PF-1000 primary discharge frequency

4. **Voltage reversal 61% — CORRECTED by Dr. PP to 47.7%**
   - Correct calculation: exp(-αT_half) = exp(-0.740) = 0.477
   - Crowbar fires at V=0 crossing regardless, making the exact percentage operationally moot

5. **"Effective DOF is 1-3" — CORRECTED by Dr. PP to 4-7**
   - Waveform has 7 inflection points, ~4.3 independent features by autocorrelation
   - Original claim conflated model parameters (2) with data information content (4-7)

### Confirmed Findings (3-0 Consensus)

1. **fc^2/fm degeneracy is exact in the Lee model**
   - F_mag ∝ (fc·I)², M_swept ∝ fm·ρ₀·A·z → acceleration ∝ fc²/fm
   - Any (fc, fm) pair with fc²/fm = 2.374 produces identical I(t) waveforms
   - The ratio fc²/fm is the only independently determined physical parameter
   - Individual fc and fm values are non-unique; should not be treated as physically meaningful
   - **Recommendation**: Report fc²/fm ratio alongside individual values in all calibration output

2. **LeeModel validation ≠ RLCSolver validation**
   - `lee_model_comparison.py` uses `scipy.integrate.solve_ivp` (RK45 adaptive)
   - `rlc_solver.py` uses implicit midpoint method (fixed dt)
   - Phase AC tests exercise LeeModel only; no test feeds MHD engine output into `validate_current_waveform()`
   - The RLCSolver has never been compared to experimental data
   - **Critical gap**: Production solver is structurally verified (D1 test) but NOT experimentally validated

3. **Phase AC is "comparison" not "validation" per AIAA G-077-1998**
   - Validation requires: (a) production solver vs experiment, (b) uncertainty propagation, (c) grid convergence, (d) stated confidence intervals
   - Phase AC achieves: (a) cross-check solver vs experiment, (b) partial uncertainty budget, (c) N/A for 0D, (d) 2-sigma check
   - Correct classification: "preliminary experimental comparison with partial uncertainty quantification"

4. **"Code better than documentation" — confirmed by Dr. EE with 5 examples**
   - BDF2 dL/dt in RLCSolver (undocumented)
   - Crowbar physics model (documented as config flag, not as physics)
   - Reflected shock phase in Lee model (present but not described)
   - Waveform resampling with interpolation in NRMSE computation
   - Beam-target neutron yield model (~115 LOC, not in docs)

5. **GUM budget construction: conservative, not wrong**
   - `u_combined = sqrt(u_exp² + discrepancy²)` is conceptually questionable (discrepancy ≠ uncertainty)
   - But `agreement_within_2sigma` uses only `u_exp`, not `u_combined` — the pass/fail criterion is correct and conservative
   - The issue is misleading variable naming, not computational error

### Phase AC Achievements (Confirmed)

| Achievement | Status | Evidence |
|------------|--------|---------|
| PF-1000 calibration: fm=0.178, fc=0.650 | In published range | fm ∈ [0.05, 0.20], fc ∈ [0.65, 0.80] per Lee & Saw (2014) |
| Peak current match: 1.87 MA | Exact (within optimizer tolerance) | Scholz et al. (2006) reports 1.87 MA |
| Full waveform NRMSE: 0.192 | First quantitative comparison | 26 digitized points from Scholz (2006), Fig. 2 |
| Peak region [4,7] μs NRMSE: <0.10 | Excellent agreement | Model captures sinusoidal LC + snowplow loading |
| Current dip signature captured | Qualitatively correct | Dip at 6.8 μs, model shows 12.1% error |
| Liftoff delay: 0.7 μs → NRMSE 0.138 | Implemented and tested | 28% NRMSE improvement; 4 new tests |
| D1 fix experimentally confirmed | fm dropped from 0.95 to 0.178 | 5x reduction, now in published range |

### Phase AC Limitations (Confirmed)

| Limitation | Severity | Resolution Path |
|-----------|----------|----------------|
| Lee model (2-phase), not 5-phase | Medium | Implement phases 3-5 (crowbar, slow compression) |
| 0D model compared, not MHD engine | **High** | Run engine.py for PF-1000, compare I(t) |
| fc at optimizer boundary (0.650) | Medium | Widen fc_bounds to (0.50, 0.90), re-run |
| fc^2/fm degeneracy | Medium | Report ratio; add radial-phase observables to break degeneracy |
| Single device (PF-1000 only) | Medium | Digitize NX2 waveform, run cross-device validation |
| No RLCSolver cross-verification | Medium | Compare LeeModel and RLCSolver for R_plasma=0 circuit |
| GUM variable naming misleading | Low | Rename u_combined → u_comparison_combined |
| L0 inconsistency (33 vs 33.5 nH) | Low | Unify to 33.5 nH from Lee & Saw (2014) |

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** — fc^2/fm degeneracy derived from snowplow EOM
- [x] **Dimensional analysis verified** — RESF = R0·√(C/L) confirmed dimensionless [Ω·(1/Ω) = 1]
- [x] **3+ peer-reviewed citations** — Scholz et al. (2006), Lee & Saw (2014), AIAA G-077-1998, GUM (JCGM 100:2008)
- [x] **Experimental evidence cited** — PF-1000 I(t) from Scholz et al. Nukleonika 51(1):79-84
- [x] **All assumptions explicitly listed** — fc^2/fm degeneracy, 2-phase Lee model, planar R-H for reflected shock
- [x] **Uncertainty budget** — 5% peak current (Rogowski), 10% timing, 20% shot-to-shot, 2% digitization
- [x] **All cross-examination criticisms addressed** — D3 retracted, 753 kHz corrected, voltage reversal corrected, DOF corrected
- [x] **No unresolved logical fallacies** — fm circular argument retracted
- [x] **Explicit agreement from each panelist** — 3-0 consensus at 6.2/10

## Score Progression

| Debate | Score | Delta | Key Driver |
|--------|-------|-------|-----------|
| #2 | 5.0 | — | Baseline (Phase R) |
| #3 | 4.5 | -0.5 | Full audit exposed identity gap |
| #4 | 5.2 | +0.7 | Phase S snowplow + DPF physics |
| #5 | 5.6 | +0.4 | Phase U Metal cylindrical |
| #6 | 6.1 | +0.5 | Phase W Lee model fixes |
| #7 | 6.5 | +0.4 | Phase X LHDI + calibration |
| #8 | 5.8 | -0.7 | D1 double circuit step discovery |
| #9 | 6.1 | +0.3 | D1/D2 fix, accuracy team, GMS Coulomb log |
| **#10** | **6.2** | **+0.1** | **Phase AC: first experimental comparison** |

## Recommendations for Next Score Increase

### P0 (Critical Path to 7.0)

1. **Run MHD engine for PF-1000, compare I(t) vs Scholz** — The single highest-impact action. Would validate the production solver, not just the cross-check model. If NRMSE < 0.20, Validation jumps from 3.9 to ~5.5, composite from 6.2 to ~6.5.

2. **Cross-verify LeeModel vs RLCSolver** — Run both solvers with R_plasma=0, L_plasma=0 (unloaded circuit). Compare I(t). They should produce identical damped sinusoids. This closes the verification gap between the two circuit implementations.

3. **Widen fc_bounds and re-calibrate** — Run with fc_bounds=(0.50, 0.90) to determine if fc=0.650 was a boundary artifact. Report fc^2/fm ratio.

### P1 (Important)

4. **Add crowbar to Lee model comparison** — Improve post-pinch I(t) match (currently NRMSE dominated by 7-10 μs region).

5. **Digitize NX2 waveform** — Second device enables cross-device validation (calibrate on PF-1000, predict NX2).

6. **Implement Lee phases 3-5** — Slow compression, radiative phase, expanded column. Reduces systematic model error.

### P2 (Desirable)

7. **Formal AIAA V&V hierarchy** — Grid convergence study on MHD solver, chi-squared goodness-of-fit, separation of amplitude and timing errors.

8. **Parameter sensitivity analysis** — Compute Hessian at calibration optimum. Report parameter uncertainties and correlation matrix.

9. **Strang splitting** — Upgrade circuit-MHD coupling from Lie O(dt) to Strang O(dt²).

## Dissenting Opinions

None. All three panelists converged to 6.2/10 through the rebuttal process. The debate was productive: 5 claims were retracted or corrected (D3 bug, fm artifact, 753 kHz, voltage reversal, effective DOF), and all category scores converged to within ±0.2 across panelists.

## Debate Quality Assessment

This was the most rigorous debate in the series. Phase 2 cross-examination produced 5 retracted claims and 3 arithmetic corrections. Phase 3 rebuttals demonstrated genuine intellectual honesty — all three panelists conceded errors when presented with evidence. The fc^2/fm degeneracy discovery is a novel analytical contribution that should be documented in the codebase.

### Errors Caught by This Debate

| Error | Source | Caught By | Impact |
|-------|--------|-----------|--------|
| D3 "bug" (R0 vs RESF total R) | Dr. DPF Phase 1 | Dr. EE Phase 2 | Would have triggered unnecessary R0 change |
| 753 kHz frequency | Dr. EE Phase 1 | Dr. PP + Dr. DPF Phase 2 | Eliminated spurious skin effect concern |
| Voltage reversal 61% | Dr. PP Phase 1 | Dr. EE Phase 2 | Corrected to 47.7% |
| fm "artifact" claim | Dr. DPF Phase 1 | Dr. PP Phase 2 | Prevented circular parameter "fix" |
| "Effective DOF 1-3" | Dr. PP Phase 1 | Dr. DPF Phase 2 | Corrected to 4-7 |
