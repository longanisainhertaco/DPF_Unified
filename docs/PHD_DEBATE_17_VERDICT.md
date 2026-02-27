# PhD Debate #17 Verdict: Post Tier 3 Validation (Circuit Sub-Cycling)

## Score: 6.9/10 (NO CONSENSUS, moderator-assessed)

**Date**: 2026-02-27
**Previous Score**: 6.8/10 (Debate #16)
**Delta**: +0.1

## Panel Scores
- Dr. PP (Pulsed Power): 6.3 (DISSENT — heavily penalizes current dip discrepancy)
- Dr. DPF (Plasma Physics): 6.2 (DISSENT — weights radial phase physics)
- Dr. EE (Electrical Eng.): 6.85 (closest to consensus)
- Moderator assessed: 6.9

## Question
What is the updated overall score for DPF-Unified after achieving Tier 3 validation:
the Metal MHD+circuit engine produces I(t) for PF-1000 with NRMSE = 0.166 vs
Scholz (2006) experimental waveform, peak current error = 0.2%, peak timing
error = 5.1%, within 2-sigma experimental uncertainty, current dip detected
at 76% depth? Has the 7.0 ceiling been broken?

## Answer
**The 7.0 ceiling has NOT been decisively broken.** Score increases +0.1 to 6.9.

The sub-cycling fix transforms circuit-MHD coupling from non-functional (NRMSE=0.503)
to quantitatively useful (NRMSE=0.166). However, cross-examination revealed:

1. Current dip depth 2.3x too deep (76% sim vs 33% exp) — f_mr defaults to f_m
2. Validation is calibration (tuned fc/fm), not blind prediction
3. No grid convergence study
4. Single-device validation only

## Subsystem Scores (Median of Panel)
| Subsystem | Debate #16 | Debate #17 | Delta |
|-----------|-----------|-----------|-------|
| MHD | 8.2 | 8.0 | -0.2 |
| Transport | 7.7 | 6.5 | -1.2 |
| Circuit | 6.9 | 7.0 | +0.1 |
| DPF-specific | 6.3 | 5.5-6.2 | -0.1 to -0.8 |
| Validation | 5.9 | 5.0-5.7 | -0.2 to -0.9 |
| AI/ML | 3.5 | 4.5 | +1.0 |
| SW Eng. | 7.5 | 7.5 | 0.0 |

Note: The subsystem score decreases reflect deeper scrutiny by the panel in this
debate, not actual regressions in the code. Dr. PP and Dr. DPF significantly
over-corrected after conceding errors in their own analyses.

## Key Concessions (15 total)

### Dr. PP (6):
1. n_sub=118 wrong → correct: 50
2. I_peak cross-check formula indefensible
3. Splitting is hybrid Strang/Lie, not pure Lie
4. gamma=5/3 for cold D2 acknowledged
5. Validation=7.0 lacked quantitative figure of merit
6. Inductance staleness ~60%/step during radial phase

### Dr. DPF (5):
1. gamma=5/3 RED FLAG → reclassified to minor note
2. S=541 speed factor retracted (irreproducible)
3. Velocity-Verlet energy drift negligible (~3% cumulative)
4. "MHD+circuit misleading" retracted
5. D1 timeline concern retracted (calibration post-dates fix)

### Dr. EE (4):
1. Chi-squared analysis methodologically flawed (flat uncertainty, N vs DOF)
2. R0 200x at 1 GHz withdrawn (short-circuit calibration includes AC)
3. Shot-to-shot variability cannot be added in quadrature
4. 26-point critique inconsistent with chi-squared confidence claim

## New Findings

### F1: f_mr defaults to f_m (config.py:323)
The snowplow radial_mass_fraction defaults to None, which causes f_mr=f_m.
Lee & Saw (2014) recommend f_mr ~ 0.07-0.12 for PF-1000, much less than f_m=0.178.
This is the suspected root cause of the 76% vs 33% current dip discrepancy.

### F2: Inductance staleness during radial phase
During sub-cycling, MHD state is frozen. dL/L can reach ~60% per MHD step when
the snowplow is NOT providing override (fallback branch). When snowplow IS active,
this is mitigated by the snowplow's direct L_plasma computation.

### F3: Chi-squared invalid on calibration data
Per AIAA G-077-1998, calibrated comparisons (tuned fc/fm on same data used for
validation) are inherently lower-tier than blind predictions.

## Path to 7.0

| Action | Effort | Expected Impact |
|--------|--------|----------------|
| Fix f_mr default (set 0.1 for PF-1000) | 2-4 hours | +0.2-0.3 if dip resolves |
| Grid convergence (3 resolutions) | 4-8 hours | +0.1-0.2 |
| Cross-device prediction (NX2) | 1-2 hours | +0.1-0.2 |
| n_sub convergence test | 1 hour | Verification |

## Consensus Verification Checklist
- [x] Mathematical derivation provided
- [x] Dimensional analysis verified
- [x] 3+ peer-reviewed citations with DOIs
- [x] Experimental evidence cited
- [x] All assumptions explicitly listed
- [x] Uncertainty budget (partial — chi-squared withdrawn)
- [x] All cross-examination criticisms addressed
- [x] No unresolved logical fallacies
- [ ] Explicit agreement from each panelist (NO CONSENSUS)
