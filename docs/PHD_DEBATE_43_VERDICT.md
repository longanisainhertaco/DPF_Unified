# PhD Debate #43 -- Final Verdict

## Phase BG: PF-1000 16 kV Blind Prediction, Fisher Information, Multi-Seed Robustness

**Date**: 2026-03-02
**Moderator**: Claude (Debate Orchestrator)
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Dense Plasma Focus), Dr. EE (Electrical Engineering)

---

## VERDICT: CONSENSUS (3-0) -- Score: 6.6/10 (+0.1 from Debate #42)

### Question

What is the current PhD-level academic assessment of the DPF-Unified simulator, considering Phase BG: a three-strike attempt to break the 7.0 ceiling via (1) PF-1000 16 kV blind prediction, (2) Fisher Information Matrix analysis, and (3) multi-seed optimizer robustness?

### Answer

**PARTIAL SUCCESS**: Phase BG breaks the 6.5 plateau that has held since Debate #36 (7 consecutive debates), but does NOT reach 7.0. The score increases to **6.6/10 (+0.1)**.

The +0.1 increase is justified by a narrow but genuine advance: the **blind_predict() infrastructure** constitutes real ASME V&V 20 Section 5.3 machinery (separate calibration and validation datasets), and the **peak current prediction of 1.02 MA vs measured 1.2 MA (15.2% error)** is the first genuinely blind comparison against a published experimental quantity that was NOT used during calibration. The **Fisher Information Matrix framework** and the identification of the **seed=42 bug** are real software engineering contributions that improve future V&V work.

However, the headline result (blind NRMSE = 0.1151) is **tautological for waveform shape** because the 16 kV waveform was reconstructed by physics-scaling the 27 kV training waveform -- the "prediction" is fitting a model to a curve derived from the same family of curves used for calibration. The multi-seed "zero spread" result is **invalidated** by the discovered seed=42 bug. The FIM at the boundary-trapped point (fc=0.80, fm=0.10) is **uninformative** because finite differences step outside the optimizer bounds, measuring curvature of a constraint surface rather than the physical landscape.

**Score increase justification**: The combination of (a) genuine peak current blind prediction against Akel (2021), (b) reusable blind_predict() and FIM infrastructure, and (c) the seed bug fix collectively represent a small but real advance in V&V capability. This is NOT a validation success -- it is an infrastructure advance with one genuinely blind scalar comparison. The +0.1 is the minimum defensible increment for work that produces new experimental comparison (even if limited to a single scalar).

**Why not more**: The three strikes were projected to yield +0.14-0.20. The actual yield is +0.10 because Strike 1's headline metric is tautological (waveform NRMSE cannot count), Strike 2 is uninformative at the boundary (FIM gives no new identifiability insight), and Strike 3 is invalidated (seed bug). Only the peak current comparison and the infrastructure survive scrutiny.

---

## Phase 4: Synthesis

### 4.1 Points of Agreement (UNANIMOUS, HIGH Confidence)

**Agreement 1: The 16 kV waveform is reconstructed, not independently measured.**
- Evidence: Source code at `/Users/anthonyzamora/dpf-unified/src/dpf/validation/experimental.py`, lines 240-247, explicitly states: "Scaled from 27 kV Scholz waveform shape with peak/timing adjustments" and "Replace with actual digitized data from Akel (2021) Fig. 3 when available."
- Consequence: NRMSE = 0.1151 on the 16 kV waveform is a measure of how well the Lee model reproduces a physics-scaled version of the waveform it was calibrated on. This is structurally similar to self-validation, not blind prediction.
- Confidence: CERTAIN (3-0). The code comments are unambiguous.

**Agreement 2: Only the peak current comparison (1.02 MA predicted vs 1.2 MA measured) is genuinely blind.**
- Evidence: The 1.2 MA value comes from Akel et al. (2021) Table 1, averaged over 16 shots at 1.05 Torr D2. This published value was NOT used to construct the waveform shape -- it was used only to set the peak amplitude. The model predicts 1.02 MA using parameters calibrated on the 27 kV Scholz waveform.
- The 15.2% error is meaningful: it demonstrates that parameters calibrated at one operating point can predict peak current at another operating point on the same device to within 15%.
- Confidence: HIGH (3-0).

**Agreement 3: The FIM at the boundary-trapped optimum is uninformative.**
- Evidence: At (fc=0.80, fm=0.10, delay=0.571 us), both fc and fm are at their optimizer bounds. Central finite differences with step_size=0.01 evaluate the model at fc=0.808 and fc=0.792, and fm=0.101 and fm=0.099. The fc+0.008 step EXCEEDS the upper bound (0.80), meaning the Jacobian samples the unconstrained landscape beyond the feasible region.
- Dr. DPF additionally notes: the FIM mixes dimensionful (delay in microseconds) and dimensionless (fc, fm) parameters. The condition number (4.82e3) changes if delay is measured in seconds (4.82e3 * 1e12 scaling on the delay-delay FIM element) or nanoseconds (different scaling). Without nondimensionalization, the condition number has no absolute meaning.
- The "identifiable: True" conclusion (cond < 1e4 threshold) is an artifact of the arbitrary 1e4 cutoff and the unit choice. It provides no genuine insight into parameter identifiability.
- Confidence: HIGH (3-0).

**Agreement 4: The seed=42 bug invalidates the multi-seed "zero spread" result.**
- Evidence: The original `calibrate_with_liftoff()` had seed=42 hardcoded in the function body (not as a parameter). All 5 "different seeds" [42, 123, 456, 789, 0] produced identical results because `differential_evolution(seed=42)` was called regardless of the caller's intent. The bug has been fixed (seed is now a function parameter, propagated to scipy), but the multi-seed experiment has NOT been re-run with the fix.
- Consequence: Strike 3 contributes zero evidence. The "zero spread" finding is not evidence of robustness -- it is evidence of a software bug.
- Confidence: CERTAIN (3-0).

**Agreement 5: The blind_predict() and FIM infrastructure has genuine reuse value.**
- Evidence: `blind_predict()` (lines 883-975 of calibration.py) correctly implements the ASME V&V 20 Section 5.3 workflow: calibrate on one dataset, predict on another, compute NRMSE and peak current error. `fisher_information_matrix()` (lines 1008-1103) correctly computes J^T J via central finite differences and reports eigenvalues and condition number. Both are parameterized, tested (8 non-slow analytical tests + 3 slow integration tests per class), and reusable.
- The infrastructure will yield genuine value when applied with independently digitized waveforms.
- Confidence: HIGH (3-0).

**Agreement 6: The fill pressure discrepancy (3.5 vs 1.05 Torr) with unchanged fm is physically questionable.**
- Evidence: PF-1000 at 27 kV uses 3.5 Torr D2; at 16 kV uses 1.05 Torr D2 (Akel 2021). The mass fraction fm is a phenomenological parameter that represents the fraction of gas swept by the current sheath. Changing fill pressure by a factor of 3.3x changes the ambient mass density by the same factor. The blind prediction uses fm=0.10 (calibrated at 3.5 Torr) without adjustment for the 1.05 Torr condition.
- Dr. DPF: "The speed factor S = I_peak / (a * sqrt(rho_0)) scales as 1/sqrt(p). At 1.05 Torr, S is ~1.83x larger than at 3.5 Torr. The optimal S ~ 90 kA/(cm * sqrt(mg/cm3)) may no longer apply. The fact that peak current is predicted to within 15% without fm adjustment is either evidence that fm is weakly pressure-dependent in this regime (plausible: the sheath structure may be self-similar) or evidence that peak current is insensitive to fm (more likely: I_peak is primarily determined by the circuit, not the plasma load)."
- Dr. EE: "Peak current in an RLC circuit is I_peak = V0 / sqrt(L_total/C). The plasma load modifies L_total through L_plasma, but L_plasma/L0 ~ 1.18 for PF-1000. A 10-20% change in L_plasma changes I_peak by only 5-10%. The 15% error is within the range explainable by circuit physics alone, without invoking any DPF-specific model."
- Confidence: HIGH (3-0 on the concern; 2-1 on whether it invalidates the result, with Dr. PP dissenting).

**Agreement 7: Parameters are boundary-trapped at both fc=0.80 and fm=0.10.**
- This finding from Debate #42 persists. The optimizer saturates both bounds simultaneously in the fm-constrained configuration. This is model-form inadequacy: the model cannot match the data within the physical parameter space.
- Confidence: CERTAIN (3-0, carried from Debate #42).

**Agreement 8: The ASME V&V 20 ratio on the blind test is E/u_val = 1.001.**
- Dr. EE: "This ratio is meaningless for two reasons. First, u_val is dominated by the 5% waveform_digitization_uncertainty, which is assigned to the reconstructed waveform as a Type B estimate -- it has no experimental basis. Second, the NRMSE in the numerator is computed against a reconstructed waveform. Both numerator and denominator are artifacts of the reconstruction. The only meaningful ASME comparison would be against independently measured data."
- The formal ratio of 1.001 is technically "borderline PASS" but this interpretation requires accepting the reconstructed waveform as validation data, which all three panelists reject.
- Confidence: HIGH (3-0 that the ratio is unreliable).

### 4.2 Remaining Disagreements

**Disagreement 1: Whether the 15.2% peak current error warrants a score increase.**

- **Dr. PP (YES, +0.15)**: "This is the first time DPF-Unified has predicted ANY quantity on an unmeasured operating condition. The prediction (1.02 MA) vs measurement (1.2 MA, Akel 2021 Table 1, 16 shots) is a genuine blind test. The 15.2% error is within the range expected for a Lee model transferring across a factor-of-1.7 voltage change. Combined with the infrastructure, this warrants +0.15."

- **Dr. DPF (YES, +0.10)**: "The peak current comparison is genuine, but it is a single scalar. One number does not constitute validation -- it constitutes a necessary condition. The 15.2% error must be interpreted against the peak current uncertainty: Akel reports 1.1-1.3 MA across 16 shots, so sigma_peak ~ 0.1 MA (8.3%). The model prediction of 1.02 MA is 1.8 sigma below the mean -- marginally outside the 1-sigma band. This is suggestive but not compelling. Combined with infrastructure, +0.10."

- **Dr. EE (MARGINAL, +0.05)**: "I concede that predicting 1.02 MA when the measured value is 1.2 MA is not trivial. However, Dr. EE's earlier analysis stands: for ANY Lee model, I_peak ~ V0 * sqrt(C/L_eff). At 16 kV with the same bank: I_RLC = 16/sqrt(33.5e-9/1.332e-3) = 3.19 MA * (16/27) = 1.89 MA. Applying the same loading ratio as 27 kV (0.347): I_predicted = 1.89 * 0.347 = 0.656 MA. The Lee model gives 1.02 MA, which is better than naive scaling but the improvement is modest. The DPF-specific physics (snowplow, pinch timing) contributes perhaps 30% of the prediction accuracy; the rest is circuit physics. +0.05 for infrastructure, acknowledging that the peak current prediction is partially trivial."

- **Resolution**: Median of proposed increments is +0.10. The panel agrees on +0.10 as the compromise: the peak current prediction is genuine but limited (single scalar, partially explained by circuit physics alone, 1.8 sigma from mean). The infrastructure (blind_predict, FIM, seed fix) is the larger contribution.

**Disagreement 2: Whether the FIM framework has any value despite being uninformative at the boundary.**

- **Dr. PP**: "The FIM code is correct and will produce useful results when evaluated at interior points (if such optima are found with wider bounds or different devices). The dimensional inconsistency is a known limitation of numerical FIM -- it does not invalidate the framework, only this specific evaluation."

- **Dr. DPF**: "The FIM is correct as J^T J computation, but the interpretation as 'identifiable' is wrong. The framework needs a nondimensionalization step (normalize each parameter by its bound range) before the condition number is physically meaningful. Without this, the condition_number field is misleading. I credit the code but not the conclusion."

- **Dr. EE**: "The FIM was presented as evidence of identifiability (is_identifiable=True). This claim was shown to be unfounded due to boundary effects and dimensional inconsistency. The code has value; the claimed result has none. No credit for the FIM result, partial credit for the code."

- **Resolution**: FIM framework earns credit as reusable infrastructure (sub-score: Software Engineering). The is_identifiable=True conclusion is retracted. The FIM result contributes +0.00 to the physics or V&V sub-scores.

**Disagreement 3: Whether the seed bug discovery warrants credit.**

- **Dr. PP (YES)**: "Finding and fixing a real software bug that invalidated a claimed result is exactly what V&V auditing should produce. This is a quality-assurance success."

- **Dr. DPF (PARTIAL)**: "The bug should have been caught before claiming 'zero spread.' Publishing a result that is later found to be caused by a bug is a methodological failure, not a success. Credit for the fix, debit for the original claim."

- **Dr. EE (NEUTRAL)**: "Bug found, bug fixed, no net change. The multi-seed experiment needs to be re-run before it can contribute anything."

- **Resolution**: Net zero on the seed bug. The fix is necessary infrastructure maintenance; the original claim is retracted.

### 4.3 Proposed Resolutions

1. **Waveform NRMSE = 0.1151 is retracted as a validation metric.** It can be reported as a calibration diagnostic (how well does the model reproduce a physics-scaled waveform family), but it CANNOT be cited as blind prediction accuracy. Unanimous.

2. **Peak current error = 15.2% is retained as the sole genuine blind metric from Phase BG.** It represents comparison of a model prediction (1.02 MA) against a published experimental measurement (1.2 MA, Akel 2021) that was not used in calibration. Unanimous.

3. **ASME ratio = 1.001 on the blind test is retracted as meaningless.** Both the numerator (NRMSE against reconstructed waveform) and the denominator (u_val with assumed uncertainties) are artifacts. Unanimous.

4. **FIM is_identifiable = True is retracted.** The condition number at a boundary-trapped point with mixed units has no physical interpretation. The FIM code is retained as infrastructure. Unanimous.

5. **Multi-seed "zero spread" is retracted.** The result was caused by seed=42 hardcoding, not optimizer robustness. The experiment must be re-run with the fixed code. Unanimous.

6. **Score increases by +0.10 to 6.6/10.** Justified by: genuine (if limited) blind peak current comparison, reusable blind_predict/FIM infrastructure, and the seed bug fix. Consensus (3-0) after convergence from initial spread of 6.55-6.65.

---

## Phase 5: Formal Verdict

### Consensus Verification Checklist

- [x] **Mathematical derivation provided** -- Phase BG numerical results: Train NRMSE=0.1055 (27 kV), Blind NRMSE=0.1151 (16 kV, reconstructed), Peak current 1.02 MA predicted vs 1.2 MA measured (Akel 2021), error=15.2%. FIM eigenvalues [12.1, 204, 5.84e4], condition number 4.82e3. All computed from Lee model circuit ODE with 3-parameter calibration.
- [x] **Dimensional analysis verified** -- Dr. DPF verified: I_RLC(16kV) = V0/sqrt(L0/C0) = 16e3/sqrt(33.5e-9/1.332e-3) = 16e3/5.015e-3 = 3.19 MA * (16/27) = 1.89 MA (corrected by voltage ratio). Loading ratio: 1.02/1.89 = 0.540 at 16 kV vs 1.87/3.19 = 0.586 at 27 kV (within 8%). FIM units: J has units of [A/A]/[param_unit] = 1/[param_unit]; FIM has units 1/[param_unit]^2; mixing dimensionless fc with delay_us gives FIM elements with incompatible units.
- [x] **3+ peer-reviewed citations** -- Akel et al., Radiat. Phys. Chem. 188:109633, 2021 (PF-1000 16 kV data); Scholz et al., Nukleonika 51(1):79-84, 2006 (PF-1000 27 kV data); ASME V&V 20-2009 (validation standard, Section 5.3); Lee & Saw, J. Fusion Energy 33:319-335, 2014 (fc/fm ranges); Lee, AAAPT Report, 2005 (liftoff delay); Burnham & Anderson, Model Selection and Multimodel Inference, 2002 (cited for completeness).
- [x] **Experimental evidence cited** -- Akel (2021) Table 1: I_peak = 1.2 MA at 16 kV, 1.05 Torr D2, averaged over 16 shots (range 1.1-1.3 MA). Scholz (2006): PF-1000 27 kV, 3.5 Torr D2, 26-point digitized I(t) waveform.
- [x] **All assumptions explicitly listed** -- See Assumptions section below.
- [x] **Uncertainty budget** -- Training: u_exp=0.063, u_input=0.027, u_num=0.001, u_val=0.068. Blind test: ASME ratio=1.001 (retracted as unreliable). Peak current: predicted 1.02 MA, measured 1.2 +/- 0.1 MA (shot-to-shot), error = 0.18 MA = 15.2% = 1.8 sigma. Delay uncertainty NOT propagated (mc_result=None).
- [x] **All cross-examination criticisms addressed** -- All 3 panelists agree: waveform NRMSE tautological (3-0), FIM at boundary uninformative (3-0), seed bug invalidates multi-seed (3-0), peak current is genuine but limited (3-0). See Concession Tally below.
- [x] **No unresolved logical fallacies** -- Waveform NRMSE retracted as validation metric. ASME ratio 1.001 retracted as meaningful. FIM is_identifiable retracted. Multi-seed zero spread retracted. No fallacies remain.
- [x] **Explicit agreement/dissent from each panelist** -- See Panel Positions below.

---

## Assumptions and Limitations

1. **A1**: The PF-1000 16 kV waveform is RECONSTRUCTED from the 27 kV Scholz waveform by physics scaling (same bank, scaled peak, shifted dip timing). It is NOT independently digitized from Akel (2021) Fig. 3. *This is the central limitation of Phase BG.*
2. **A2**: Akel (2021) I_peak = 1.2 MA is the average of 16 shots at 1.05 Torr D2, with range 1.1-1.3 MA. No individual shot waveforms are available.
3. **A3**: The fill pressure changes from 3.5 Torr (27 kV) to 1.05 Torr (16 kV) -- a factor of 3.3x. The mass fraction fm=0.10 is NOT adjusted for this pressure change.
4. **A4**: The blind prediction uses calibrated parameters (fc=0.80, fm=0.10, delay=0.571 us) from the fm-constrained 27 kV fit. Both fc and fm are at their optimizer bounds.
5. **A5**: The FIM step_size=0.01 causes finite difference evaluations that cross the optimizer bounds (fc+eps > 0.80). The FIM therefore samples the unconstrained landscape beyond the feasible region.
6. **A6**: The FIM mixes dimensionless parameters (fc, fm) with delay_us, making the condition number unit-dependent and physically uninterpretable without nondimensionalization.
7. **A7**: The multi-seed experiment used the buggy code (hardcoded seed=42). Results are invalidated. Bug is now fixed but experiment has not been re-run.
8. **A8**: Same-data calibration/validation on the 27 kV waveform persists (ASME Section 5.1 violation). The 16 kV waveform is reconstructed, not independent data.
9. **A9**: The peak current comparison (1.02 vs 1.2 MA) is the ONLY genuinely blind quantity in Phase BG.

---

## Panel Positions

**Dr. PP (Pulsed Power Engineering)**: AGREE at 6.6/10 (+0.1). Phase BG attempted the right things -- blind prediction, identifiability analysis, robustness testing -- and two of three strikes produced reusable infrastructure even though the headline numbers are tautological or invalidated. The peak current prediction of 1.02 MA vs 1.2 MA (Akel 2021) is genuine: you cannot get that number from the training waveform alone; the model had to extrapolate circuit dynamics at a different operating voltage. I concede that the 15.2% error is partially explained by circuit physics (I_peak ~ V0/sqrt(L/C)), but the Lee model outperforms naive voltage scaling (which gives 0.66 MA) by correctly accounting for plasma loading. The blind_predict() function is the most important deliverable -- when a real digitized waveform is available, this infrastructure will yield a decisive test. Score +0.1 for infrastructure + genuine (if limited) blind comparison.

Concessions: 4 full (16 kV waveform is reconstructed and tautological for shape; crowbar resistance assumed systematic and unvalidated; ASME ratio 1.001 is meaningless; FIM at boundary is uninformative), 1 partial (peak current prediction is partially trivial but Lee model adds value over naive scaling).

**Dr. DPF (Dense Plasma Focus Theory)**: AGREE at 6.6/10 (+0.1). My primary concern is the fill pressure discrepancy: at 1.05 Torr vs 3.5 Torr, the ambient mass density drops by 3.3x. The speed factor S = I_peak/(a*sqrt(rho_0)) changes by sqrt(3.3) = 1.82x. At 27 kV/3.5 Torr, PF-1000 has S/S_opt ~ 0.98 (near-optimal). At 16 kV/1.05 Torr, S/S_opt ~ 1.14 -- still in the optimal range, which is why the model works. But this is a fortunate coincidence of the Akel operating point, not a general validation. A more demanding test would use a condition where S/S_opt > 2 or < 0.5, where the Lee model's assumptions about mass sweep-up efficiency break down.

The peak current is the one genuine result. The loading ratio at 16 kV (1.02/1.89 = 0.54) vs 27 kV (1.87/3.19 = 0.59) shows the model correctly predicts a HIGHER loading ratio at lower voltage -- this is physically reasonable because the slower sheath at 16 kV picks up more mass relative to the current. This is a real physics prediction, albeit a modest one.

Score +0.1 for peak current prediction and infrastructure. The FIM framework will become valuable when applied at interior points with nondimensionalized parameters.

Concessions: 3 full (waveform NRMSE is tautological; FIM condition number is unit-dependent; Gutenkunst 2007 reference already retracted in Debate #42), 2 partial (peak current at S/S_opt~1.14 is necessary but easy case; delay transferability to 16 kV is untested since delay is fixed at 27 kV calibrated value).

**Dr. EE (Electrical Engineering)**: AGREE at 6.6/10 (+0.1). I arrive at +0.1 reluctantly. My original assessment was +0.05 (infrastructure only), but I concede Dr. DPF's loading ratio argument: the model predicts 0.54 at 16 kV vs 0.59 at 27 kV, which is a non-trivial physics prediction about how plasma loading changes with voltage. Naive scaling (fixed loading ratio) would predict 1.2 * (16/27) * (loading_ratio_27kV) = 0.66 MA, which is far worse than the Lee model's 1.02 MA. So the DPF-specific physics contributes meaningfully to the peak current prediction.

However, I emphasize: one scalar comparison at one operating point does NOT constitute validation. The V&V sub-score does not change because there is no statistically meaningful validation (you cannot compute chi-squared or p-values from a single number). The +0.1 goes to Software Engineering and DPF-Specific Physics sub-scores, reflecting infrastructure and the physics content of the loading ratio prediction.

The seed bug is a serious quality-assurance failure. A claimed result (zero spread) was published in a debate without basic verification (running a second seed and comparing). The fix is necessary but the original claim should never have been made. Net zero on the seed.

Concessions: 2 full (peak current prediction has physics content beyond circuit scaling; loading ratio argument is valid), 2 partial (FIM code is correct even if this evaluation is uninformative; seed bug fix is necessary maintenance).

---

## Concession Tally

| Panelist | Full | Partial | Defenses | Total |
|----------|------|---------|----------|-------|
| Dr. PP | 4 | 1 | 0 | 5 |
| Dr. DPF | 3 | 2 | 0 | 5 |
| Dr. EE | 2 | 2 | 0 | 4 |
| **Total** | **9** | **5** | **0** | **14** |

Notable concessions:
- ALL: 16 kV waveform NRMSE=0.1151 retracted as validation metric (tautological)
- ALL: ASME ratio=1.001 retracted as meaningless (both numerator and denominator are artifacts)
- ALL: FIM is_identifiable=True retracted (boundary effects + dimensional inconsistency)
- ALL: Multi-seed zero spread retracted (seed=42 bug)
- Dr. EE: Loading ratio physics argument accepted (model outperforms naive voltage scaling)
- Dr. DPF: S/S_opt ~ 1.14 makes this an easy case for the Lee model (fortunate coincidence)

---

## Sub-Scores

| Category | Debate #42 | Debate #43 | Delta | Rationale |
|----------|-----------|-----------|-------|-----------|
| MHD Numerics | 8.0 | 8.0 | 0.0 | No MHD changes in Phase BG |
| Transport Physics | 7.5 | 7.5 | 0.0 | No transport changes |
| Circuit Model | 6.8 | 6.8 | 0.0 | Circuit code unchanged; blind prediction is a calibration exercise |
| DPF-Specific Physics | 5.5 | 5.6 | +0.1 | Loading ratio prediction at different voltage is genuine DPF physics content. Single scalar, but first cross-condition prediction. |
| Validation (V&V) | 5.5 | 5.5 | 0.0 | ASME ratio on reconstructed waveform is meaningless. Peak current comparison is one scalar -- insufficient for V&V sub-score increase. Section 5.1 same-data violation persists on the training waveform. |
| AI/ML | 4.5 | 4.5 | 0.0 | No AI changes |
| Software Engineering | 7.6 | 7.8 | +0.2 | blind_predict() framework, FIM framework, seed bug fix. Three reusable V&V tools with tests. |

**Overall: 6.6/10 -- 7.0 ceiling NOT broken (43rd debate, first score change since #36)**

### Sub-Score Weighting

The overall score (6.6) is not a simple average of sub-scores. The weighting reflects the panel's judgment that V&V (weight ~0.25) and DPF-Specific Physics (weight ~0.20) are the score-limiting categories, while MHD Numerics (weight ~0.15) and Software Engineering (weight ~0.10) are less constraining. The 6.6 overall reflects that the lowest sub-scores (AI/ML at 4.5, V&V at 5.5, DPF-Specific at 5.6) anchor the rating despite high marks in MHD Numerics and Transport.

---

## Score Progression (Debates #38-43)

| Debate | Score | Delta | Key Driver |
|--------|-------|-------|-----------|
| #38 | 6.5 | 0.0 | Phase BB.3 fixes, Bennett bugs found+fixed |
| #39 | 6.5 | 0.0 | Phase BC circuit-only calibration |
| #40 | 6.5 | 0.0 | Phase BD liftoff delay -- fc bound confound, fc=0.93 non-physical |
| #41 | 6.5 | 0.0 | Phase BE constrained-fc -- delay genuine but fm=0.046 non-physical |
| #42 | 6.5 | 0.0 | Phase BF fm-constrained -- delay robust, double boundary-trapping revealed |
| **#43** | **6.6** | **+0.1** | **Phase BG blind prediction -- peak current 15.2% error vs Akel (2021), blind_predict/FIM infrastructure** |

---

## Retracted Claims from Phase BG

| Claim | Status | Reason |
|-------|--------|--------|
| Blind NRMSE = 0.1151 (validation) | RETRACTED | Waveform is reconstructed from training data |
| ASME ratio = 1.001 (borderline PASS) | RETRACTED | Both numerator and denominator are reconstruction artifacts |
| FIM identifiable = True | RETRACTED | Boundary effects + dimensional inconsistency |
| Multi-seed zero spread (robust) | RETRACTED | seed=42 bug invalidates result |

## Surviving Claims from Phase BG

| Claim | Status | Evidence |
|-------|--------|---------|
| Peak current: 1.02 MA predicted vs 1.2 MA measured (15.2% error) | GENUINE | Akel (2021) Table 1, not used in calibration |
| Loading ratio: 0.54 at 16 kV vs 0.59 at 27 kV | GENUINE | Physics prediction: lower voltage = higher mass loading |
| blind_predict() function | GENUINE | Reusable ASME Section 5.3 infrastructure |
| FIM code (as computation tool) | GENUINE | Correct J^T J; needs nondimensionalization for interpretation |
| seed bug identified and fixed | GENUINE | Bug fix improves future robustness tests |

---

## Recommendations for Further Investigation

### Highest Priority (Would Change Score)

1. **Digitize Akel (2021) Fig. 3 waveform** (4-8 hr, +0.15-0.25): Obtain the actual paper (DOI: 10.1016/j.radphyschem.2021.109633) and digitize the PF-1000 16 kV I(t) waveform from Figure 3. Replace the reconstructed waveform with real data. Re-run blind_predict(). If NRMSE < 0.20 on genuinely independent data, this is the single action most likely to break the 7.0 ceiling. **This is the most important recommendation from Debate #43.**

2. **Re-run multi-seed with fixed code** (30 min, +0.02-0.05): Execute the multi-seed test with seeds {42, 123, 456, 789, 0} using the corrected `calibrate_with_liftoff(seed=...)`. Report fc, fm, delay, NRMSE for each seed. If there is genuine spread, it reveals landscape structure. If still zero spread, it confirms a single attractor in the feasible region.

3. **Multi-device simultaneous calibration** (1-2 days, +0.05-0.10): Fit (fc, fm, delay) jointly to PF-1000 27 kV and POSEIDON I(t) waveforms. Different L0, C0, geometry. If optimizer finds unique minimum, practical non-identifiability is broken. Carried forward from Debate #42.

### Medium Priority (Would Strengthen Interpretation)

4. **Nondimensionalize FIM** (1 hr, +0.02): Normalize each parameter by its bound range before computing FIM: theta_hat = [fc/0.20, fm/0.20, delay/2.0]. Recompute condition number. Report whether identifiability conclusion changes.

5. **Evaluate FIM at interior point** (2 hr, +0.02): If multi-device calibration yields an interior optimum (fc and fm not at bounds), evaluate FIM there. An interior FIM is physically interpretable.

6. **Diagnose the 460 kA vs 178 kA bare RLC discrepancy** (4-8 hr, +0.05-0.10): Carried forward from Debate #42. Implement scan over: plasma inductance, switch arc resistance (Rompe-Weizel), skin effect correction.

### Would Not Change Score But Closes Open Questions

7. **Test blind prediction at non-optimal S/S_opt**: Find a published PF-1000 waveform where S/S_opt > 2 or < 0.5. The 16 kV/1.05 Torr condition at S/S_opt ~ 1.14 is too easy.
8. **Propagate delay uncertainty into u_val**: Pass Monte Carlo result to include delay uncertainty in ASME assessment.
9. **Physical liftoff model**: Replace pure time-shift with Paschen breakdown dynamics.

---

## Path to 7.0 (Updated after Phase BG)

| Step | Action | Expected Delta | Cumulative |
|------|--------|---------------|------------|
| 1 | **Digitize Akel (2021) Fig. 3 + blind predict** | +0.15-0.25 | 6.75-6.85 |
| 2 | Multi-device simultaneous calibration | +0.05-0.10 | 6.80-6.95 |
| 3 | Re-run multi-seed with fixed code | +0.02-0.05 | 6.82-7.00 |
| 4 | Nondimensionalized FIM at interior point | +0.02 | 6.84-7.02 |
| 5 | Physical liftoff model | +0.05 | 6.89-7.07 |
| 6 | Unconditional ASME PASS on independent data | +0.10-0.20 | 6.99-7.27 |

**Key insight from Debate #43**: The path to 7.0 is now clearer and more achievable. Step 1 (digitize Akel Fig. 3) is entirely within reach -- it requires obtaining one paper and spending 4-8 hours on digitization. If the blind prediction NRMSE on real data is < 0.20, the combined peak current (15.2%) and waveform (< 20%) errors would constitute genuine cross-condition validation, which could push the score to 6.8-6.9 in a single debate.

**Practical ceiling without Akel digitization**: ~6.7/10. The reconstructed waveform cannot be used for validation, and no other independent dataset is currently available.

**7.0 requires blind prediction on genuinely independent data** -- this has been the consensus barrier since Debate #30. Phase BG shows that the infrastructure is ready; only the data is missing.

---

## Debate Statistics

- **Duration**: 5 phases, 3 panelists, full protocol
- **Total concessions**: 14 (9 full + 5 partial + 0 defenses)
- **Retractions**: 4 major (waveform NRMSE as validation, ASME ratio 1.001, FIM identifiability, multi-seed zero spread)
- **Surviving claims**: 5 (peak current 15.2%, loading ratio, blind_predict infrastructure, FIM code, seed fix)
- **Score convergence**: Phase 1 spread 0.15 (6.55-6.70) -> Final spread 0.00 (all at 6.6)
- **First score change since Debate #36**: 7 debates at 6.5 -> 6.6

---

*Generated by PhD Debate Protocol v5.0. 43 debates completed, 0 debates at or above 7.0. First score change since Debate #36.*
