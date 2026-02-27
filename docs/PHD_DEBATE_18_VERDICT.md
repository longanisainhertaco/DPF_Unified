# PhD Debate #18 — Post f_mr Fix + Phase AD Engine Validation

**Date**: 2026-02-27
**Question**: Has fixing the f_mr pipeline (5/5 call sites) and completing Phase AD engine validation moved the project past 7.0/10?

## VERDICT: MAJORITY (2-1) — Score 6.6/10

### 7.0 NOT Broken (Unanimous)

All three panelists agree the project has NOT broken 7.0/10.

### Panel Final Scores
- **Dr. PP (Pulsed Power)**: 6.5 — f_mr correctly plumbed but zero physics impact on peak current
- **Dr. DPF (Dense Plasma Focus)**: 6.85 — Phase AD structure excellent, but no radial-phase validation
- **Dr. EE (Electrical Engineering)**: 6.4 — 21% timing error is a critical blocker; f_mr has 0.00% peak current change

### Moderator Assessment: 6.6/10

Weighted average of panel scores. Down from 6.9 (Debate #17) because:
1. f_mr fix expected to improve current dip but showed ZERO physics impact (+0% peak current, +4% NRMSE)
2. Cross-examination revealed calibration overfitting (fc/fm tuned on same data)
3. 21% timing error confirmed as model-form error, not correctable by parameter tuning

### Key Findings

#### 1. f_mr Has Zero Physics Impact
- Peak current change: **0.00%** (confirmed by Dr. EE's measurement)
- NRMSE change: **+4%** (slightly worse, not better)
- Reason: Peak current is determined during axial phase (I_peak occurs at t ~ 5 μs); f_mr only affects radial phase dynamics (t > 5 μs)
- The f_mr fix was CORRECT engineering (consistent pipeline) but provided NO physics improvement

#### 2. 21% Timing Error is Model-Form
- Present in BOTH RLCSolver and LeeModel (different integrators, same timing error)
- NOT a Lie splitting artifact (cross-verification NRMSE = 19% is dominated by post-peak dynamics)
- Root cause: snowplow model lacks insulator flashover phase, uses crude liftoff delay model
- Cannot be fixed by parameter tuning alone

#### 3. Current Dip Discrepancy: 76% sim vs 33% exp (2.3x)
- Hardcoded compression ratio 10:1 vs Scholz (2006) X-ray data showing 19-58:1
- Inductance dominance: ΔL/L ≈ 81% at r_pinch = 0.1a
- J×B force overpowers back-pressure in the model
- Requires reflected shock Phase 4 (Lee & Saw 2014 Section III.D) or configurable CR

#### 4. Chi-Squared Withdrawn
- Dr. EE computed autocorrelation of the smooth waveform: n_eff ≈ 2 independent DOF
- Chi-squared test with 2 DOF is meaningless for waveform comparison
- NRMSE = 0.133 is the honest metric (not a p-value)

#### 5. No MHD-vs-Experiment Comparison
- All Phase AD tests validate 0D snowplow+circuit against experiment
- The MHD solver has ZERO experimental validation
- This is the single largest gap blocking 7.0

### Phase 2 Cross-Examination Highlights

**Dr. PP found remaining f_mr bug** (2/5 call sites not passing f_mr in _objective and calibrate final comparison). Fixed during the debate.

**Dr. DPF challenged**: No reflected shock, no thermal conduction, no radiation loss — radial phase physics is placeholder-level.

**Dr. EE challenged**: 26-point digitized waveform has n_eff ≈ 2. All "waveform matching" claims are effectively 2-parameter fits to 2 observables.

### Phase 3 Concessions
- **Dr. PP**: Conceded 5 points (Lie splitting, fc²/fm, I_peak formula, GUM, timing error)
- **Dr. DPF**: Conceded 5 points, lowered to 6.3 (m=0 not in code, hardcoded CR, no parasitic L, unfalsifiable m=0)
- **Dr. EE**: Conceded 3 points, lowered to 6.4 (chi² withdrawn, grid convergence misapplied, radial phase blocker)

### Consensus Verification Checklist
- [x] Mathematical derivation provided — fc²/fm degeneracy, inductance scaling
- [x] Dimensional analysis verified — snowplow EOM, Pease-Braginskii
- [x] 3+ peer-reviewed citations — Scholz (2006), Lee & Saw (2014), Miyoshi & Kusano (2005)
- [x] Experimental evidence cited — Scholz PF-1000 I(t) waveform
- [x] All assumptions explicitly listed — thin-sheath, ideal gas, no radiation
- [x] Uncertainty budget — 14% (k=1) peak current, 21% timing
- [x] All cross-examination criticisms addressed — all challenges answered
- [ ] No unresolved logical fallacies — calibration vs prediction circular reasoning remains
- [x] Explicit agreement/dissent — PP 6.5, DPF 6.85, EE 6.4

### Roadmap to 7.0

| Action | Expected Impact | Confidence | Effort |
|--------|----------------|------------|--------|
| Reflected shock Phase 4 in snowplow | Fix 76%→30-40% dip | MEDIUM-HIGH | 4-8 hours |
| Blind cross-device prediction (PF-1000→NX2) | +0.1-0.2 | HIGH | 1-2 hours |
| Configurable compression ratio per device | +0.05-0.1 | HIGH | 1 hour |
| GUM-compliant uncertainty propagation | +0.05-0.1 | MEDIUM | 2-4 hours |
| MHD engine vs experiment | +0.3-0.5 (breaks 7.0) | MEDIUM | 8-16 hours |

### Score Progression
| Debate | Score | Delta |
|--------|-------|-------|
| #16 | 6.8 | — |
| #17 | 6.9 | +0.1 |
| **#18** | **6.6** | **-0.3** |

Score decreased because f_mr fix was expected to be high-impact but delivered zero physics improvement. The cross-examination process correctly identified this as a non-improvement.
